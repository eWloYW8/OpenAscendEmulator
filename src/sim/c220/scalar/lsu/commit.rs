use std::collections::{BTreeMap, VecDeque};

mod data;
pub use data::C220LoadResponse;
#[cfg(test)]
mod tests;

use super::C220LsuRequestId;
use super::scheduler::{C220LsuLoadPath, C220LsuLoadValue, C220LsuStoreValue};
use super::store_buffer::C220LsuPairPart;
use super::write_queue::C220LsuWriteId;
use crate::architecture::Architecture;
use crate::sim::c220::scalar::C220LoadOperands;
use crate::sim::common::scalar::{ScalarMachine, ScalarMachineError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220LoadCommitMode {
    DataBypass,
    Retirement,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct C220LoadId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220LoadRetirement {
    pub instruction: C220LoadId,
    /// Combined values. Split loads use the first request/address and the final
    /// response's tick/path; `responses` retains each individual transaction.
    pub data: C220LsuLoadValue,
    pub issue_tick: u64,
    pub admission_tick: u64,
    pub writeback_tick: Option<u64>,
    pub retire_tick: u64,
    pub suppressed: bool,
    pub register_value: u64,
    pub second_register_value: Option<u64>,
    /// Individual request responses, in operand order for a split load.
    pub responses: [Option<C220LoadResponse>; 2],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PendingLoad {
    operands: C220LoadOperands,
    issue_tick: u64,
    admission: Option<(u64, C220LsuRequestId)>,
    second_request: Option<C220LsuRequestId>,
    responses: [Option<C220LsuLoadValue>; 2],
    suppressed: bool,
    writeback_tick: Option<u64>,
    data: Option<C220LsuLoadValue>,
}

#[derive(Debug, thiserror::Error)]
pub enum C220LsuCommitError {
    #[error("LSU commit clock moved backwards")]
    TimeReversal,
    #[error("LSU retirement clock overflow")]
    Overflow,
    #[error("load commit requires a C220 register file and valid captured operands")]
    InvalidOperands,
    #[error("LSU request was already issued or its completion does not match")]
    InvalidRequest,
    #[error("LSU retirement transport is full")]
    RetirementFull,
    #[error(transparent)]
    Register(#[from] ScalarMachineError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220LsuRetirement {
    Load(C220LoadRetirement),
    Store {
        data: C220LsuStoreValue,
        retire_tick: u64,
    },
    DirectStore {
        request: C220LsuRequestId,
        write: C220LsuWriteId,
        response_tick: u64,
        retire_tick: u64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RetirementToken {
    Load(C220LoadId),
    Store(C220LsuStoreValue),
    DirectStore {
        request: C220LsuRequestId,
        write: C220LsuWriteId,
        response_tick: u64,
    },
}

/// Load register ownership and the shared scalar LSU retirement transport.
/// The dispatcher must call `supersede` only after accepting a replacing writer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220LsuCommitLane {
    mode: C220LoadCommitMode,
    tick: u64,
    pending: BTreeMap<C220LoadId, PendingLoad>,
    requests: BTreeMap<C220LsuRequestId, C220LoadId>,
    owners: BTreeMap<u8, C220LoadId>,
    retirements: VecDeque<(u64, RetirementToken)>,
}

impl C220LsuCommitLane {
    pub fn new(mode: C220LoadCommitMode) -> Self {
        Self {
            mode,
            tick: 0,
            pending: BTreeMap::new(),
            requests: BTreeMap::new(),
            owners: BTreeMap::new(),
            retirements: VecDeque::new(),
        }
    }

    pub fn pending_destination(&self, register: u8) -> Option<C220LoadId> {
        self.owners.get(&register).copied()
    }

    pub fn pending_count(&self) -> usize {
        self.pending.len()
            + self
                .retirements
                .iter()
                .filter(|(_, token)| !matches!(token, RetirementToken::Load(_)))
                .count()
    }

    pub fn retirement_occupancy(&self) -> usize {
        self.retirements.len()
    }

    pub fn check_retirement_send(&self, tick: u64) -> Result<(), C220LsuCommitError> {
        self.check_tick(tick)?;
        tick.checked_add(1).ok_or(C220LsuCommitError::Overflow)?;
        if self.retirements.len() == 64 {
            return Err(C220LsuCommitError::RetirementFull);
        }
        Ok(())
    }

    pub fn complete_store_at(
        &mut self,
        tick: u64,
        data: C220LsuStoreValue,
    ) -> Result<(), C220LsuCommitError> {
        self.check_retirement_send(tick)?;
        if data.tick > tick {
            return Err(C220LsuCommitError::InvalidRequest);
        }
        self.check_store_request(data.request)?;
        self.retirements
            .push_back((tick + 1, RetirementToken::Store(data)));
        self.tick = tick;
        Ok(())
    }

    pub fn complete_direct_store_at(
        &mut self,
        tick: u64,
        request: C220LsuRequestId,
        write: C220LsuWriteId,
    ) -> Result<(), C220LsuCommitError> {
        self.check_retirement_send(tick)?;
        self.check_store_request(request)?;
        self.retirements.push_back((
            tick + 1,
            RetirementToken::DirectStore {
                request,
                write,
                response_tick: tick,
            },
        ));
        self.tick = tick;
        Ok(())
    }

    fn check_store_request(&self, request: C220LsuRequestId) -> Result<(), C220LsuCommitError> {
        if self.requests.contains_key(&request)
            || self.retirements.iter().any(|(_, token)| match token {
                RetirementToken::Store(data) => data.request == request,
                RetirementToken::DirectStore {
                    request: pending, ..
                } => *pending == request,
                RetirementToken::Load(_) => false,
            })
        {
            return Err(C220LsuCommitError::InvalidRequest);
        }
        Ok(())
    }

    pub fn next_retirement_tick(&self) -> Option<u64> {
        self.retirements.front().map(|entry| entry.0)
    }

    /// Suppress the producing instruction, but release only this register's
    /// dependency. Other destination dependencies are not implicitly cancelled.
    pub fn supersede(&mut self, register: u8) {
        if let Some(request) = self.owners.remove(&register)
            && let Some(pending) = self.pending.get_mut(&request)
        {
            pending.suppressed = true;
        }
    }

    /// Called at accepted CCU issue, before the LSU evaluates cache stages.
    pub fn issue(
        &mut self,
        tick: u64,
        instruction: C220LoadId,
        operands: C220LoadOperands,
        machine: &mut ScalarMachine,
    ) -> Result<(), C220LsuCommitError> {
        self.check_tick(tick)?;
        Self::check_machine(machine, operands)?;
        if self.pending.contains_key(&instruction)
            || self
                .retirements
                .iter()
                .any(|(_, token)| matches!(token, RetirementToken::Load(id) if *id == instruction))
        {
            return Err(C220LsuCommitError::InvalidRequest);
        }
        if let Some(base) = operands.updated_base {
            machine.set_xreg(operands.base_register, base)?;
        }
        for register in operands.destinations() {
            self.supersede(register);
        }
        self.pending.insert(
            instruction,
            PendingLoad {
                operands,
                issue_tick: tick,
                admission: None,
                second_request: None,
                responses: [None; 2],
                suppressed: false,
                writeback_tick: None,
                data: None,
            },
        );
        for register in operands.destinations() {
            self.owners.insert(register, instruction);
        }
        self.tick = tick;
        Ok(())
    }

    /// Bind a cache request only when the LSU accepts the dispatched instruction.
    pub fn admit(
        &mut self,
        tick: u64,
        instruction: C220LoadId,
        request: C220LsuRequestId,
        second_request: Option<C220LsuRequestId>,
    ) -> Result<(), C220LsuCommitError> {
        self.check_tick(tick)?;
        let pending = self
            .pending
            .get_mut(&instruction)
            .ok_or(C220LsuCommitError::InvalidRequest)?;
        if pending.admission.is_some()
            || self.requests.contains_key(&request)
            || second_request.is_some_and(|second| {
                second == request
                    || self.requests.contains_key(&second)
                    || pending.operands.second_destination.is_none()
            })
        {
            return Err(C220LsuCommitError::InvalidRequest);
        }
        pending.admission = Some((tick, request));
        pending.second_request = second_request;
        self.requests.insert(request, instruction);
        if let Some(second) = second_request {
            self.requests.insert(second, instruction);
        }
        self.tick = tick;
        Ok(())
    }

    /// Consume one ready notification. The event scheduler controls invocation order.
    pub fn retire_next_at(
        &mut self,
        tick: u64,
        machine: &mut ScalarMachine,
    ) -> Result<Option<C220LsuRetirement>, C220LsuCommitError> {
        self.check_tick(tick)?;
        let Some((_, token)) = self
            .retirements
            .front()
            .copied()
            .filter(|entry| entry.0 <= tick)
        else {
            self.tick = tick;
            return Ok(None);
        };
        let request = match token {
            RetirementToken::Load(request) => request,
            RetirementToken::Store(data) => {
                self.retirements.pop_front();
                self.tick = tick;
                return Ok(Some(C220LsuRetirement::Store {
                    data,
                    retire_tick: tick,
                }));
            }
            RetirementToken::DirectStore {
                request,
                write,
                response_tick,
            } => {
                self.retirements.pop_front();
                self.tick = tick;
                return Ok(Some(C220LsuRetirement::DirectStore {
                    request,
                    write,
                    response_tick,
                    retire_tick: tick,
                }));
            }
        };
        let Some(pending) = self
            .pending
            .get(&request)
            .copied()
            .filter(|pending| pending.data.is_some())
        else {
            self.retirements.pop_front();
            self.tick = tick;
            return Ok(None);
        };
        Self::check_machine(machine, pending.operands)?;
        let data = pending.data.expect("queued load data");
        let mut writeback_tick = pending.writeback_tick;
        if self.mode == C220LoadCommitMode::Retirement && !pending.suppressed {
            self.write_result(data, machine)?;
            writeback_tick = Some(tick);
        }
        self.retirements.pop_front();
        self.pending.remove(&request);
        self.requests.remove(&data.request);
        if let Some(second) = pending.second_request {
            self.requests.remove(&second);
        }
        self.tick = tick;
        Ok(Some(C220LsuRetirement::Load(C220LoadRetirement {
            instruction: request,
            data,
            issue_tick: pending.issue_tick,
            admission_tick: pending.admission.expect("completed request").0,
            writeback_tick,
            retire_tick: tick,
            suppressed: pending.suppressed,
            responses: pending
                .responses
                .map(|response| response.map(C220LoadResponse::from)),
            register_value: machine.xregs()[usize::from(pending.operands.destination_register)],
            second_register_value: pending
                .operands
                .second_destination
                .map(|(register, _)| machine.xregs()[usize::from(register)]),
        })))
    }

    fn write_result(
        &mut self,
        data: C220LsuLoadValue,
        machine: &mut ScalarMachine,
    ) -> Result<(), C220LsuCommitError> {
        if let Some((register, _)) = data.operands.second_destination {
            machine.set_xreg(register, data.second_value.expect("validated pair data"))?;
        }
        machine.set_xreg(data.operands.destination_register, data.value)?;
        for register in data.operands.destinations() {
            self.owners.remove(&register);
        }
        Ok(())
    }

    fn check_tick(&self, tick: u64) -> Result<(), C220LsuCommitError> {
        if tick < self.tick {
            Err(C220LsuCommitError::TimeReversal)
        } else {
            Ok(())
        }
    }

    fn check_machine(
        machine: &ScalarMachine,
        operands: C220LoadOperands,
    ) -> Result<(), C220LsuCommitError> {
        if machine.architecture() != Architecture::Dav2201
            || usize::from(operands.destination_register) >= machine.xregs().len()
            || usize::from(operands.base_register) >= machine.xregs().len()
            || operands
                .second_destination
                .is_some_and(|(register, _)| usize::from(register) >= machine.xregs().len())
        {
            Err(C220LsuCommitError::InvalidOperands)
        } else {
            Ok(())
        }
    }
}
