use crate::sim::c220::mte::interface::biu_read::C220BiuReadError;
use std::collections::{BTreeMap, VecDeque};
mod biu;

use super::{
    C220Nd2NzReadPlan, C220Nd2NzReadRequest, C220Nd2NzReadRoute, C220Nd2NzResponse,
    C220Nd2NzStaging, C220Nd2NzStagingConfig, C220Nd2NzStagingError, C220Nd2NzWritePlan,
    C220Nd2NzWriteRequest,
};
use crate::isa::c220::mte::nd2nz::C220Nd2NzTransfer;
use crate::sim::c220::mte::{
    interface::{
        C220MteL1WriteError, C220MteL1WriteInterface, C220MteL1WritePort, C220MteOutputFragment,
    },
    uop::C220DmaUopMode,
};

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum C220Nd2NzEngineError {
    #[error(transparent)]
    Biu(#[from] C220BiuReadError),
    #[error(transparent)]
    Staging(#[from] C220Nd2NzStagingError),
    #[error(transparent)]
    L1(#[from] C220MteL1WriteError),
    #[error("ND2NZ instruction {0} is already active")]
    DuplicateInstruction(u64),
    #[error("ND2NZ response {request_id} does not belong to instruction {instruction_id}")]
    UnknownResponse {
        instruction_id: u64,
        request_id: u64,
    },
    #[error("ND2NZ time reversed from {previous} to {requested}")]
    TimeReversed { previous: u64, requested: u64 },
    #[error("ND2NZ {callback} callback repeated at tick {tick}")]
    RepeatedCallback { callback: &'static str, tick: u64 },
    #[error("ND2NZ time or request ID overflow")]
    Overflow,
    #[error("ND2NZ empty instructions must complete before engine admission")]
    EmptyInstruction,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220Nd2NzIssuedRead {
    pub sid: u8,
    pub instruction_id: u64,
    pub request_id: u64,
    pub ready_tick: u64,
    pub mode: C220DmaUopMode,
    pub request: C220Nd2NzReadRequest,
}

#[derive(Debug, Clone)]
struct Source {
    sid: u8,
    instruction_id: u64,
    ready_tick: u64,
    mode: C220DmaUopMode,
    plan: C220Nd2NzReadPlan,
    head: C220Nd2NzReadRequest,
    next_id: u64,
}

#[derive(Debug, Clone)]
struct Command {
    instruction_id: u64,
    ready_tick: u64,
    plan: C220Nd2NzWritePlan,
    head: C220Nd2NzWriteRequest,
    next_id: u64,
}

/// Dedicated ND2NZ scheduling and response ownership. The owner schedules the
/// independent callbacks and transports issued reads; L1 writes use Port1.
#[derive(Debug, Clone)]
pub struct C220Nd2NzEngine {
    staging: C220Nd2NzStaging,
    source: Option<Source>,
    generated: VecDeque<C220Nd2NzIssuedRead>,
    commands: VecDeque<Command>,
    responses: BTreeMap<(u64, u64), C220Nd2NzResponse>,
    observed_tick: Option<u64>,
    generate_tick: Option<u64>,
    read_tick: Option<u64>,
    response_ticks: [Option<u64>; 2],
    write_tick: Option<u64>,
}

impl C220Nd2NzEngine {
    pub fn new(config: C220Nd2NzStagingConfig) -> Result<Self, C220Nd2NzEngineError> {
        Ok(Self {
            staging: C220Nd2NzStaging::new(config)?,
            source: None,
            generated: VecDeque::new(),
            commands: VecDeque::new(),
            responses: BTreeMap::new(),
            observed_tick: None,
            generate_tick: None,
            read_tick: None,
            response_ticks: [None; 2],
            write_tick: None,
        })
    }

    pub fn staging(&self) -> &C220Nd2NzStaging {
        &self.staging
    }

    pub fn generated(&self) -> &VecDeque<C220Nd2NzIssuedRead> {
        &self.generated
    }

    pub fn active_instruction(&self) -> Option<u64> {
        self.commands.front().map(|command| command.instruction_id)
    }

    pub fn pending_instructions(&self) -> usize {
        self.commands.len()
    }

    pub fn outstanding_responses(&self) -> usize {
        self.responses.len()
    }

    pub fn can_submit(&self) -> bool {
        self.source.is_none() && self.generated.len() < 7 && self.commands.len() <= 1
    }

    pub fn is_idle(&self) -> bool {
        self.source.is_none() && self.generated.is_empty() && self.commands.is_empty()
    }

    pub fn is_drained(&self) -> bool {
        self.is_idle() && self.responses.is_empty() && self.staging.is_idle()
    }

    pub fn submit(
        &mut self,
        tick: u64,
        instruction_id: u64,
        transfer: C220Nd2NzTransfer,
        route: C220Nd2NzReadRoute,
        mode: C220DmaUopMode,
    ) -> Result<bool, C220Nd2NzEngineError> {
        self.observe(tick)?;
        if self
            .commands
            .iter()
            .any(|c| c.instruction_id == instruction_id)
            || self.responses.keys().any(|&(id, _)| id == instruction_id)
            || self
                .generated
                .iter()
                .any(|r| r.instruction_id == instruction_id)
        {
            return Err(C220Nd2NzEngineError::DuplicateInstruction(instruction_id));
        }
        if !self.can_submit() {
            return Ok(false);
        }
        let ready_tick = tick.checked_add(1).ok_or(C220Nd2NzEngineError::Overflow)?;
        let rows = self.staging.config().rows;
        let mut reads = C220Nd2NzReadPlan::new(transfer, route, mode, rows);
        let mut writes = C220Nd2NzWritePlan::new(transfer, rows);
        let read = reads.next().ok_or(C220Nd2NzEngineError::EmptyInstruction)?;
        let write = writes
            .next()
            .ok_or(C220Nd2NzEngineError::EmptyInstruction)?;
        self.source = Some(Source {
            sid: transfer.sid(),
            instruction_id,
            ready_tick,
            mode,
            plan: reads,
            head: read,
            next_id: 0,
        });
        self.commands.push_back(Command {
            instruction_id,
            ready_tick,
            plan: writes,
            head: write,
            next_id: 0,
        });
        Ok(true)
    }

    /// Move one read from the instruction queue into the seven-entry queue.
    pub fn generate(&mut self, tick: u64) -> Result<bool, C220Nd2NzEngineError> {
        self.observe(tick)?;
        Self::callback(&mut self.generate_tick, tick, "generate")?;
        let Some(source) = self.source.as_mut() else {
            return Ok(false);
        };
        if source.ready_tick > tick || self.generated.len() >= 7 {
            return Ok(false);
        }
        let ready_tick = tick.checked_add(6).ok_or(C220Nd2NzEngineError::Overflow)?;
        let next_id = source
            .next_id
            .checked_add(1)
            .ok_or(C220Nd2NzEngineError::Overflow)?;
        self.generated.push_back(C220Nd2NzIssuedRead {
            sid: source.sid,
            instruction_id: source.instruction_id,
            request_id: source.next_id,
            ready_tick,
            mode: source.mode,
            request: source.head.clone(),
        });
        source.next_id = next_id;
        if let Some(next) = source.plan.next() {
            source.head = next;
        } else {
            self.source = None;
        }
        Ok(true)
    }

    /// `dispatch_ready` includes downstream credit and any hardware-sync gate
    /// resolved by the owning MTE dispatcher.
    pub fn take_read(
        &mut self,
        tick: u64,
        dispatch_ready: bool,
    ) -> Result<Option<C220Nd2NzIssuedRead>, C220Nd2NzEngineError> {
        self.observe(tick)?;
        Self::callback(&mut self.read_tick, tick, "read send")?;
        if !dispatch_ready || self.generated.front().is_none_or(|r| r.ready_tick > tick) {
            return Ok(None);
        }
        let head = self.generated.front().expect("eligible generated read");
        let response = C220Nd2NzResponse::new(&head.request)?;
        self.responses
            .insert((head.instruction_id, head.request_id), response);
        Ok(self.generated.pop_front())
    }

    /// Retry the same response until it is fully consumed. A later instruction
    /// cannot enter the alignment buffers before the active command's last send.
    pub fn receive(
        &mut self,
        tick: u64,
        instruction_id: u64,
        request_id: u64,
    ) -> Result<u32, C220Nd2NzEngineError> {
        self.observe(tick)?;
        let key = (instruction_id, request_id);
        if !self.responses.contains_key(&key) {
            return Err(C220Nd2NzEngineError::UnknownResponse {
                instruction_id,
                request_id,
            });
        }
        if self.active_instruction() != Some(instruction_id) {
            return Ok(0);
        }
        let response = self.responses.get_mut(&key).expect("validated response");
        let route = match response.route() {
            C220Nd2NzReadRoute::PerRow => 0,
            C220Nd2NzReadRoute::ContiguousRows => 1,
        };
        Self::callback(&mut self.response_ticks[route], tick, "response")?;
        let accepted = self.staging.receive(tick, response)?;
        if response.is_complete() {
            self.responses.remove(&key);
        }
        Ok(accepted)
    }

    pub fn response_remaining(&self, instruction_id: u64, request_id: u64) -> Option<u32> {
        self.responses
            .get(&(instruction_id, request_id))
            .map(C220Nd2NzResponse::remaining_bytes)
    }

    pub fn stage_lane(
        &mut self,
        tick: u64,
        lane: u32,
    ) -> Result<Option<u32>, C220Nd2NzEngineError> {
        self.observe(tick)?;
        Ok(self.staging.stage_lane(tick, lane)?)
    }

    pub fn stage_small(&mut self, tick: u64) -> Result<u32, C220Nd2NzEngineError> {
        self.observe(tick)?;
        Ok(self.staging.stage_small(tick)?)
    }

    pub fn send_l1(
        &mut self,
        tick: u64,
        output: &mut C220MteL1WriteInterface,
    ) -> Result<Option<C220MteOutputFragment>, C220Nd2NzEngineError> {
        self.observe(tick)?;
        Self::callback(&mut self.write_tick, tick, "write send")?;
        let Some(command) = self.commands.front() else {
            return Ok(None);
        };
        if command.ready_tick > tick || !self.staging.write_ready(tick, command.head.rows)? {
            return Ok(None);
        }
        let next_id = command
            .next_id
            .checked_add(1)
            .ok_or(C220Nd2NzEngineError::Overflow)?;
        let fragment = C220MteOutputFragment {
            instruction_id: command.instruction_id,
            request_id: command.next_id,
            destination_address: command.head.destination_address,
            bytes: command.head.bytes,
            last_in_uop: true,
            last_in_instruction: command.head.last_in_instruction,
        };
        if !output.push(tick, C220MteL1WritePort::Port1, fragment)? {
            return Ok(None);
        }
        let command = self.commands.front_mut().expect("pending command");
        let debited = self
            .staging
            .take_write_credit(tick, command.head.rows, true)?;
        debug_assert!(debited);
        if let Some(next) = command.plan.next() {
            command.head = next;
            command.next_id = next_id;
        } else {
            self.commands.pop_front();
        }
        Ok(Some(fragment))
    }

    fn observe(&mut self, tick: u64) -> Result<(), C220Nd2NzEngineError> {
        if let Some(previous) = self.observed_tick
            && tick < previous
        {
            return Err(C220Nd2NzEngineError::TimeReversed {
                previous,
                requested: tick,
            });
        }
        self.observed_tick = Some(tick);
        Ok(())
    }

    fn callback(
        last: &mut Option<u64>,
        tick: u64,
        callback: &'static str,
    ) -> Result<(), C220Nd2NzEngineError> {
        if *last == Some(tick) {
            return Err(C220Nd2NzEngineError::RepeatedCallback { callback, tick });
        }
        *last = Some(tick);
        Ok(())
    }
}
