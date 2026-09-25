use std::collections::VecDeque;

use super::ub_block::PARTIAL_WRITE_TICKS;
use super::{C220UbBlock, C220UbBlockProgress, C220UbRequest, C220UbRequestError};

const QUEUE_CAPACITY: usize = 2;
const REQUEST_TICKS: u64 = 1;
const TRANSPORT_TICKS: u64 = 1;

#[cfg(test)]
mod tests;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum C220UbServicePort {
    ScalarWrite,
    ScalarRead,
    MteWrite0,
    MteWrite1,
    MteRead,
}

impl C220UbServicePort {
    pub const ALL: [Self; 5] = [
        Self::ScalarWrite,
        Self::ScalarRead,
        Self::MteWrite0,
        Self::MteWrite1,
        Self::MteRead,
    ];

    const fn response_ticks(self) -> u64 {
        match self {
            Self::ScalarWrite => 0,
            Self::ScalarRead | Self::MteWrite0 | Self::MteWrite1 => 3,
            Self::MteRead => 5,
        }
    }

    const fn is_write(self) -> bool {
        matches!(self, Self::ScalarWrite | Self::MteWrite0 | Self::MteWrite1)
    }
}

/// Vector activity before Scalar/MTE arbitration in the same UB cycle.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct C220UbVectorActivity {
    pub bank_mask: u64,
    pub triggered: bool,
    pub write_pending: bool,
    pub read_pending: bool,
}

/// A contiguous, unmasked access at the UB memory-service boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220UbServiceRequest {
    pub id: u64,
    pub address: u64,
    pub bytes: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220UbServiceBlock {
    pub block: C220UbBlock,
    pub partial_write: bool,
    pub progress: C220UbBlockProgress,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220UbServiceEntry {
    pub request: C220UbServiceRequest,
    pub received_tick: u64,
    pub ready_tick: u64,
    pub blocks: Vec<C220UbServiceBlock>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220UbServiceResponse {
    pub request: C220UbServiceRequest,
    pub completion_tick: u64,
    pub ready_tick: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220UbServiceDecision {
    pub port: C220UbServicePort,
    pub request_id: u64,
    pub block_index: usize,
    pub bank: u8,
    pub granted: bool,
    pub second_grant: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220UbServiceCycle {
    pub tick: u64,
    pub higher_priority_banks: u64,
    pub bank_mask: u64,
    pub scalar_write_blocked: bool,
    pub scalar_read_blocked: bool,
    pub decisions: Vec<C220UbServiceDecision>,
    pub completed: Vec<(C220UbServicePort, C220UbServiceResponse)>,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum C220UbServiceError {
    #[error(transparent)]
    Access(#[from] C220UbRequestError),
    #[error("UB service time reversed from {previous} to {requested}")]
    TimeReversed { previous: u64, requested: u64 },
    #[error("UB service {phase} already ran at tick {tick}")]
    RepeatedCallback { phase: &'static str, tick: u64 },
    #[error("UB service time overflowed")]
    TimeOverflow,
    #[error("UB service request {0} is already pending on this port")]
    DuplicateRequest(u64),
}

/// Scalar and MTE ports of one UB, serviced after Vector grants.
/// Input and response queue readiness triggers a whole callback: once invoked,
/// the callback visits every nonempty port in priority order.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct C220UbService {
    scalar_uses_vector_ports: bool,
    inputs: [VecDeque<C220UbServiceEntry>; 5],
    responses: [VecDeque<C220UbServiceResponse>; 5],
    transport: [VecDeque<C220UbServiceResponse>; 5],
    observed_tick: Option<u64>,
    arbitration_tick: Option<u64>,
    send_tick: Option<u64>,
    receive_ticks: [Option<u64>; 5],
    response_ticks: [Option<u64>; 5],
}

impl C220UbService {
    pub const fn scalar_uses_vector_ports(&self) -> bool {
        self.scalar_uses_vector_ports
    }

    pub(crate) fn set_scalar_uses_vector_ports(&mut self, shared: bool) {
        self.scalar_uses_vector_ports = shared;
    }

    pub fn new(scalar_uses_vector_ports: bool) -> Self {
        Self {
            scalar_uses_vector_ports,
            ..Self::default()
        }
    }

    pub fn is_idle(&self) -> bool {
        self.inputs.iter().all(VecDeque::is_empty)
            && self.responses.iter().all(VecDeque::is_empty)
            && self.transport.iter().all(VecDeque::is_empty)
    }

    pub fn inputs(&self, port: C220UbServicePort) -> &VecDeque<C220UbServiceEntry> {
        &self.inputs[port as usize]
    }
    pub fn responses(&self, port: C220UbServicePort) -> &VecDeque<C220UbServiceResponse> {
        &self.responses[port as usize]
    }
    pub fn transport(&self, port: C220UbServicePort) -> &VecDeque<C220UbServiceResponse> {
        &self.transport[port as usize]
    }
    pub fn can_receive(&self, tick: u64, port: C220UbServicePort) -> bool {
        self.receive_ticks[port as usize] != Some(tick)
            && self.inputs[port as usize].len() < QUEUE_CAPACITY
    }

    pub fn receive(
        &mut self,
        tick: u64,
        port: C220UbServicePort,
        request: C220UbServiceRequest,
    ) -> Result<bool, C220UbServiceError> {
        self.check_time(tick)?;
        let index = port as usize;
        if self.inputs[index]
            .iter()
            .any(|entry| entry.request.id == request.id)
            || self.responses[index]
                .iter()
                .chain(&self.transport[index])
                .any(|entry| entry.request.id == request.id)
        {
            return Err(C220UbServiceError::DuplicateRequest(request.id));
        }
        if !self.can_receive(tick, port) {
            return Ok(false);
        }
        let ready_tick = tick
            .checked_add(REQUEST_TICKS)
            .ok_or(C220UbServiceError::TimeOverflow)?;
        let access = if request.address == u64::MAX {
            C220UbRequest::default()
        } else {
            C220UbRequest::from_accesses(&[(request.address, request.bytes as usize)])?
        };
        let blocks = access
            .blocks()
            .iter()
            .copied()
            .map(|block| C220UbServiceBlock {
                partial_write: port.is_write() && (block.bytes != 32 || block.address % 32 != 0),
                block,
                progress: C220UbBlockProgress::Waiting,
            })
            .collect();
        self.inputs[index].push_back(C220UbServiceEntry {
            request,
            received_tick: tick,
            ready_tick,
            blocks,
        });
        self.receive_ticks[index] = Some(tick);
        self.observed_tick = Some(tick);
        Ok(true)
    }

    /// Banks are shared across masters; Vector bank-group masks do not
    /// constrain Scalar/MTE grants. Pending Vector requests can block Scalar
    /// grants when Scalar is configured to share Vector ports.
    pub fn arbitrate(
        &mut self,
        tick: u64,
        vector: C220UbVectorActivity,
    ) -> Result<C220UbServiceCycle, C220UbServiceError> {
        self.check_callback(tick, self.arbitration_tick, "arbitrate")?;
        let mut cycle = C220UbServiceCycle {
            tick,
            higher_priority_banks: vector.bank_mask,
            bank_mask: vector.bank_mask,
            scalar_write_blocked: self.scalar_uses_vector_ports && vector.write_pending,
            scalar_read_blocked: self.scalar_uses_vector_ports && vector.read_pending,
            decisions: Vec::new(),
            completed: Vec::new(),
        };
        if vector.triggered
            || self
                .inputs
                .iter()
                .any(|queue| queue.front().is_some_and(|entry| entry.ready_tick <= tick))
        {
            for port in C220UbServicePort::ALL {
                if let Some(entry) = self.inputs[port as usize].front() {
                    tick.checked_add(port.response_ticks())
                        .ok_or(C220UbServiceError::TimeOverflow)?;
                    if entry.blocks.iter().any(|block| {
                        block.partial_write && block.progress == C220UbBlockProgress::Waiting
                    }) {
                        tick.checked_add(PARTIAL_WRITE_TICKS)
                            .ok_or(C220UbServiceError::TimeOverflow)?;
                    }
                }
            }
            for port in C220UbServicePort::ALL {
                let Some(entry) = self.inputs[port as usize].front_mut() else {
                    continue;
                };
                if (port == C220UbServicePort::ScalarWrite && cycle.scalar_write_blocked)
                    || (port == C220UbServicePort::ScalarRead && cycle.scalar_read_blocked)
                {
                    continue;
                }
                for (block_index, block) in entry.blocks.iter_mut().enumerate() {
                    let Some(second_grant) = block.progress.pending_grant() else {
                        continue;
                    };
                    let bit = 1_u64 << block.block.bank.id;
                    let granted = cycle.bank_mask & bit == 0;
                    cycle.decisions.push(C220UbServiceDecision {
                        port,
                        request_id: entry.request.id,
                        block_index,
                        bank: block.block.bank.id,
                        granted,
                        second_grant,
                    });
                    if granted {
                        cycle.bank_mask |= bit;
                        block.progress.grant(tick, block.partial_write);
                    }
                }
            }
            // Completion checks follow all grants, so a partial write whose
            // delay ends now can retry no earlier than the next arbitration.
            for port in C220UbServicePort::ALL {
                let index = port as usize;
                let Some(entry) = self.inputs[index].front_mut() else {
                    continue;
                };
                for block in &mut entry.blocks {
                    block.progress.finish_tick(tick);
                }
                if entry
                    .blocks
                    .iter()
                    .all(|block| matches!(block.progress, C220UbBlockProgress::Complete { .. }))
                {
                    let response = C220UbServiceResponse {
                        request: entry.request,
                        completion_tick: tick,
                        ready_tick: tick + port.response_ticks(),
                    };
                    self.inputs[index].pop_front();
                    self.responses[index].push_back(response);
                    cycle.completed.push((port, response));
                }
            }
        }
        self.arbitration_tick = Some(tick);
        self.observed_tick = Some(tick);
        Ok(cycle)
    }

    pub fn send_responses(
        &mut self,
        tick: u64,
    ) -> Result<Vec<(C220UbServicePort, C220UbServiceResponse)>, C220UbServiceError> {
        self.check_callback(tick, self.send_tick, "send")?;
        let mut sent = Vec::new();
        if self
            .responses
            .iter()
            .any(|queue| queue.front().is_some_and(|head| head.ready_tick <= tick))
        {
            let ready_tick = tick
                .checked_add(TRANSPORT_TICKS)
                .ok_or(C220UbServiceError::TimeOverflow)?;
            for port in C220UbServicePort::ALL {
                let index = port as usize;
                if self.responses[index].is_empty() {
                    continue;
                }
                if self.transport[index].len() == QUEUE_CAPACITY {
                    break;
                }
                let mut response = self.responses[index].pop_front().expect("response head");
                response.ready_tick = ready_tick;
                self.transport[index].push_back(response);
                sent.push((port, response));
            }
        }
        self.send_tick = Some(tick);
        self.observed_tick = Some(tick);
        Ok(sent)
    }

    pub fn take_response(
        &mut self,
        tick: u64,
        port: C220UbServicePort,
    ) -> Result<Option<C220UbServiceResponse>, C220UbServiceError> {
        self.check_time(tick)?;
        let index = port as usize;
        if self.response_ticks[index] == Some(tick) {
            return Ok(None);
        }
        let response = self.transport[index].pop_front_if(|head| head.ready_tick <= tick);
        if response.is_some() {
            self.response_ticks[index] = Some(tick);
        }
        self.observed_tick = Some(tick);
        Ok(response)
    }

    fn check_time(&self, tick: u64) -> Result<(), C220UbServiceError> {
        if let Some(previous) = self.observed_tick
            && tick < previous
        {
            return Err(C220UbServiceError::TimeReversed {
                previous,
                requested: tick,
            });
        }
        Ok(())
    }

    fn check_callback(
        &self,
        tick: u64,
        last: Option<u64>,
        phase: &'static str,
    ) -> Result<(), C220UbServiceError> {
        self.check_time(tick)?;
        if last == Some(tick) {
            return Err(C220UbServiceError::RepeatedCallback { phase, tick });
        }
        Ok(())
    }
}
