use std::collections::{BTreeMap, VecDeque};
use std::num::NonZeroU32;

use super::{
    C220MteL1Output, C220MteL1OutputCredits, C220MteL1OutputCycle, C220MteL1OutputDestination,
    C220MteL1OutputError, C220MteL1OutputQueues, C220MteL1OutputSend, C220MteL1OutputTransfer,
    C220MteL1ReadArbiter, C220MteL1ReadDecision, C220MteL1ReadDestination, C220MteL1ReadHead,
    C220MteL1ReadPort, C220MteOutputPlan,
};
use crate::sim::c220::memory::l1::{C220L1Access, C220L1Request};

mod events;
pub use events::{C220MteL1Callback, C220MteL1EventOutcome, C220MteL1Events};

const INPUT_TICKS: u64 = 4;
const INPUT_CAPACITY: usize = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MteL1ReadOperation<T> {
    pub instruction_id: u64,
    pub access: C220L1Access,
    pub destination: C220MteL1OutputDestination,
    pub output_address: u64,
    pub output_bytes: u32,
    pub output_bandwidth: NonZeroU32,
    pub completes_logical_uop: bool,
    pub last_in_instruction: bool,
    pub payload: T,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MteL1ReadRequest<T> {
    pub id: u64,
    pub operation: C220MteL1ReadOperation<T>,
}

impl<T> C220MteL1ReadOperation<T> {
    pub fn map_payload<U>(self, map: impl FnOnce(T) -> U) -> C220MteL1ReadOperation<U> {
        C220MteL1ReadOperation {
            instruction_id: self.instruction_id,
            access: self.access,
            destination: self.destination,
            output_address: self.output_address,
            output_bytes: self.output_bytes,
            output_bandwidth: self.output_bandwidth,
            completes_logical_uop: self.completes_logical_uop,
            last_in_instruction: self.last_in_instruction,
            payload: map(self.payload),
        }
    }
}

impl<T> C220MteL1ReadRequest<T> {
    pub const fn l1_request(&self) -> C220L1Request {
        C220L1Request {
            id: self.id,
            access: self.operation.access,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct C220MteL1CycleInputs {
    pub request_ready: bool,
    pub response: Option<u64>,
    pub output_credits: C220MteL1OutputCredits,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MteL1Queues {
    pub inputs: [usize; 3],
    pub awaiting_response: usize,
    pub output: C220MteL1OutputQueues,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220MteL1Cycle<T> {
    pub tick: u64,
    pub decision: C220MteL1ReadDecision,
    pub sent: Option<C220MteL1ReadRequest<T>>,
    pub received: Option<C220MteL1ReadRequest<T>>,
    pub output: C220MteL1OutputCycle<C220MteL1ReadRequest<T>>,
    pub queues: C220MteL1Queues,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220MteL1ReadSend<T> {
    pub tick: u64,
    pub decision: C220MteL1ReadDecision,
    pub sent: Option<C220MteL1ReadRequest<T>>,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum C220MteL1Error {
    #[error("L1 interface time reversed from {previous} to {requested}")]
    TimeReversed { previous: u64, requested: u64 },
    #[error("L1 interface {phase} callback already ran at tick {tick}")]
    RepeatedCallback { phase: &'static str, tick: u64 },
    #[error("L1 interface time or request ID overflowed")]
    Overflow,
    #[error("L1 response {0} does not identify an outstanding request")]
    UnknownResponse(u64),
    #[error(transparent)]
    Output(#[from] C220MteL1OutputError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Entry<T> {
    ready_tick: u64,
    request: C220MteL1ReadRequest<T>,
}

/// Shared by all producers feeding the MTE L1 read interface. Input queues,
/// request IDs, in-flight responses, and output bandwidth have one owner.
/// Memory-bank arbitration and the request/response transport remain external.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220MteL1Interface<T> {
    inputs: [VecDeque<Entry<T>>; 3],
    arbiter: C220MteL1ReadArbiter,
    in_flight: BTreeMap<u64, C220MteL1ReadRequest<T>>,
    output: C220MteL1Output<C220MteL1ReadRequest<T>>,
    next_request_id: u64,
    observed_tick: Option<u64>,
    send_tick: Option<u64>,
    receive_tick: Option<u64>,
}

impl<T> Default for C220MteL1Interface<T> {
    fn default() -> Self {
        Self {
            inputs: std::array::from_fn(|_| VecDeque::new()),
            arbiter: C220MteL1ReadArbiter::default(),
            in_flight: BTreeMap::new(),
            output: C220MteL1Output::default(),
            next_request_id: 0,
            observed_tick: None,
            send_tick: None,
            receive_tick: None,
        }
    }
}

impl<T: Copy> C220MteL1Interface<T> {
    pub fn is_idle(&self) -> bool {
        self.inputs.iter().all(VecDeque::is_empty)
            && self.in_flight.is_empty()
            && self.output.is_idle()
    }

    pub fn can_push(&self, port: C220MteL1ReadPort) -> bool {
        self.inputs[port as usize].len() < INPUT_CAPACITY
    }

    pub fn queue_state(&self) -> C220MteL1Queues {
        C220MteL1Queues {
            inputs: std::array::from_fn(|index| self.inputs[index].len()),
            awaiting_response: self.in_flight.len(),
            output: self.output.queue_state(),
        }
    }

    pub fn outstanding_requests(&self) -> impl Iterator<Item = &C220MteL1ReadRequest<T>> {
        self.in_flight.values()
    }

    pub fn read_decision(&self, tick: u64) -> C220MteL1ReadDecision {
        self.arbiter.preview(
            tick,
            self.heads(),
            self.output.queue_state().output_fragments,
        )
    }

    pub(in crate::sim::c220::mte) fn validate_push(
        &self,
        tick: u64,
        port: C220MteL1ReadPort,
    ) -> Result<(), C220MteL1Error> {
        self.check_time(tick)?;
        if self.can_push(port) {
            tick.checked_add(INPUT_TICKS)
                .ok_or(C220MteL1Error::Overflow)?;
            self.next_request_id
                .checked_add(1)
                .ok_or(C220MteL1Error::Overflow)?;
        }
        Ok(())
    }

    /// Returns None on a full port without consuming the operation or an ID.
    /// Push before or after step to specify the producer's callback phase.
    pub fn push(
        &mut self,
        tick: u64,
        port: C220MteL1ReadPort,
        operation: C220MteL1ReadOperation<T>,
    ) -> Result<Option<C220MteL1ReadRequest<T>>, C220MteL1Error> {
        self.validate_push(tick, port)?;
        if !self.can_push(port) {
            return Ok(None);
        }
        let request = C220MteL1ReadRequest {
            id: self.next_request_id,
            operation,
        };
        self.inputs[port as usize].push_back(Entry {
            ready_tick: tick + INPUT_TICKS,
            request,
        });
        self.next_request_id += 1;
        self.observed_tick = Some(tick);
        Ok(Some(request))
    }

    pub fn step(
        &mut self,
        tick: u64,
        inputs: C220MteL1CycleInputs,
    ) -> Result<C220MteL1Cycle<T>, C220MteL1Error> {
        self.check_callback(tick, self.send_tick, "send request")?;
        self.validate_response(tick, inputs.response)?;
        self.output.validate_step(tick, inputs.output_credits)?;
        // Read eligibility uses occupancy at callback entry, before output drains.
        let request = self.send_request(tick, inputs.request_ready)?;
        let output = self.output.step(tick, inputs.output_credits)?;
        let received = self.receive_response(tick, inputs.response)?;
        Ok(C220MteL1Cycle {
            tick,
            decision: request.decision,
            sent: request.sent,
            received,
            output,
            queues: self.queue_state(),
        })
    }

    pub fn send_request(
        &mut self,
        tick: u64,
        request_ready: bool,
    ) -> Result<C220MteL1ReadSend<T>, C220MteL1Error> {
        self.check_callback(tick, self.send_tick, "send request")?;
        let decision = self.arbiter.arbitrate(
            tick,
            self.heads(),
            self.output.queue_state().output_fragments,
        );
        let sent = decision.selected.filter(|_| request_ready).map(|port| {
            let request = self.inputs[port as usize]
                .pop_front()
                .expect("selected input")
                .request;
            self.in_flight.insert(request.id, request);
            request
        });
        self.observed_tick = Some(tick);
        self.send_tick = Some(tick);
        Ok(C220MteL1ReadSend {
            tick,
            decision,
            sent,
        })
    }

    fn validate_response(
        &self,
        tick: u64,
        response: Option<u64>,
    ) -> Result<Option<C220MteL1ReadRequest<T>>, C220MteL1Error> {
        self.check_callback(tick, self.receive_tick, "receive response")?;
        let received = response
            .map(|id| {
                self.in_flight
                    .get(&id)
                    .copied()
                    .ok_or(C220MteL1Error::UnknownResponse(id))
            })
            .transpose()?;
        if received.is_some_and(|request| request.operation.completes_logical_uop) {
            self.output.validate_receive(tick)?;
        }
        Ok(received)
    }

    pub fn receive_response(
        &mut self,
        tick: u64,
        response: Option<u64>,
    ) -> Result<Option<C220MteL1ReadRequest<T>>, C220MteL1Error> {
        let received = self.validate_response(tick, response)?;
        if let Some(request) = received {
            self.in_flight.remove(&request.id);
            let operation = request.operation;
            if operation.completes_logical_uop {
                self.output.receive(
                    tick,
                    operation.destination,
                    C220MteOutputPlan::new(
                        operation.instruction_id,
                        request.id,
                        operation.output_address,
                        operation.output_bytes,
                        operation.last_in_instruction,
                        operation.output_bandwidth,
                    ),
                    request,
                )?;
            }
        }
        self.observed_tick = Some(tick);
        self.receive_tick = Some(tick);
        Ok(received)
    }

    pub fn send_output(
        &mut self,
        tick: u64,
        credits: C220MteL1OutputCredits,
    ) -> Result<C220MteL1OutputSend<C220MteL1ReadRequest<T>>, C220MteL1Error> {
        self.check_time(tick)?;
        let output = self.output.send(tick, credits)?;
        self.observed_tick = Some(tick);
        Ok(output)
    }

    pub fn retire(
        &mut self,
        tick: u64,
    ) -> Result<Option<C220MteL1OutputTransfer<C220MteL1ReadRequest<T>>>, C220MteL1Error> {
        self.check_time(tick)?;
        let retired = self.output.retire(tick)?;
        self.observed_tick = Some(tick);
        Ok(retired)
    }

    pub fn input_ready_tick(&self, port: C220MteL1ReadPort) -> Option<u64> {
        self.inputs[port as usize]
            .front()
            .map(|head| head.ready_tick)
    }

    pub fn acknowledgment_ready_tick(&self) -> Option<u64> {
        self.output.acknowledgment_ready_tick()
    }

    pub fn retirement_ready_tick(&self) -> Option<u64> {
        self.output.retirement_ready_tick()
    }

    fn heads(&self) -> [Option<C220MteL1ReadHead>; 3] {
        std::array::from_fn(|index| {
            self.inputs[index].front().map(|entry| C220MteL1ReadHead {
                ready_tick: entry.ready_tick,
                destination: match entry.request.operation.destination {
                    C220MteL1OutputDestination::SparseIndex => C220MteL1ReadDestination::Sp,
                    C220MteL1OutputDestination::Bt => C220MteL1ReadDestination::Bt,
                    C220MteL1OutputDestination::Fb => C220MteL1ReadDestination::Fb,
                    C220MteL1OutputDestination::Smask => C220MteL1ReadDestination::Smask,
                    C220MteL1OutputDestination::L0a(_) => C220MteL1ReadDestination::L0a,
                    C220MteL1OutputDestination::L0b(_) => C220MteL1ReadDestination::L0b,
                },
            })
        })
    }

    fn check_time(&self, tick: u64) -> Result<(), C220MteL1Error> {
        if let Some(previous) = self.observed_tick
            && tick < previous
        {
            return Err(C220MteL1Error::TimeReversed {
                previous,
                requested: tick,
            });
        }
        Ok(())
    }

    fn check_callback(
        &self,
        tick: u64,
        previous: Option<u64>,
        phase: &'static str,
    ) -> Result<(), C220MteL1Error> {
        self.check_time(tick)?;
        if previous == Some(tick) {
            return Err(C220MteL1Error::RepeatedCallback { phase, tick });
        }
        Ok(())
    }
}
