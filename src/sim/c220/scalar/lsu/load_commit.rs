use std::collections::{BTreeMap, VecDeque};

use super::C220LsuRequestId;
use super::scheduler::C220LsuLoadValue;
use crate::architecture::Architecture;
use crate::sim::c220::scalar::C220LoadOperands;
use crate::sim::common::scalar::{ScalarMachine, ScalarMachineError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220LoadCommitMode {
    DataBypass,
    Retirement,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220LoadRetirement {
    pub data: C220LsuLoadValue,
    pub issue_tick: u64,
    pub writeback_tick: Option<u64>,
    pub retire_tick: u64,
    pub suppressed: bool,
    pub register_value: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PendingLoad {
    operands: C220LoadOperands,
    issue_tick: u64,
    suppressed: bool,
    writeback_tick: Option<u64>,
    data: Option<C220LsuLoadValue>,
}

#[derive(Debug, thiserror::Error)]
pub enum C220LoadCommitError {
    #[error("load commit clock moved backwards")]
    TimeReversal,
    #[error("load retirement clock overflow")]
    Overflow,
    #[error("load commit requires a C220 register file and valid captured operands")]
    InvalidOperands,
    #[error("load request was already issued or its completion does not match")]
    InvalidRequest,
    #[error("load retirement transport is full")]
    RetirementFull,
    #[error(transparent)]
    Register(#[from] ScalarMachineError),
}

/// Register ownership and retirement transport for single-register loads.
/// The dispatcher must call `supersede` only after accepting a replacing writer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220LoadCommitLane {
    mode: C220LoadCommitMode,
    tick: u64,
    pending: BTreeMap<C220LsuRequestId, PendingLoad>,
    owners: BTreeMap<u8, C220LsuRequestId>,
    retirements: VecDeque<(u64, C220LsuRequestId)>,
}

impl C220LoadCommitLane {
    pub fn new(mode: C220LoadCommitMode) -> Self {
        Self {
            mode,
            tick: 0,
            pending: BTreeMap::new(),
            owners: BTreeMap::new(),
            retirements: VecDeque::new(),
        }
    }

    pub fn pending_destination(&self, register: u8) -> Option<C220LsuRequestId> {
        self.owners.get(&register).copied()
    }

    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    pub fn next_retirement_tick(&self) -> Option<u64> {
        self.retirements.front().map(|entry| entry.0)
    }

    pub fn supersede(&mut self, register: u8) {
        if let Some(request) = self.owners.remove(&register) {
            self.pending
                .get_mut(&request)
                .expect("live load owner")
                .suppressed = true;
        }
    }

    /// Called at accepted CCU issue, before the LSU evaluates cache stages.
    pub fn issue(
        &mut self,
        tick: u64,
        request: C220LsuRequestId,
        operands: C220LoadOperands,
        machine: &mut ScalarMachine,
    ) -> Result<(), C220LoadCommitError> {
        self.check_tick(tick)?;
        Self::check_machine(machine, operands)?;
        if self.pending.contains_key(&request) {
            return Err(C220LoadCommitError::InvalidRequest);
        }
        if let Some(base) = operands.updated_base {
            machine.set_xreg(operands.base_register, base)?;
        }
        self.supersede(operands.destination_register);
        self.pending.insert(
            request,
            PendingLoad {
                operands,
                issue_tick: tick,
                suppressed: false,
                writeback_tick: None,
                data: None,
            },
        );
        self.owners.insert(operands.destination_register, request);
        self.tick = tick;
        Ok(())
    }

    /// Data becomes available now; architectural retirement remains separate.
    /// A rejected completion remains owned by the caller, without register changes.
    pub fn complete_data_at(
        &mut self,
        tick: u64,
        data: C220LsuLoadValue,
        machine: &mut ScalarMachine,
    ) -> Result<(), C220LoadCommitError> {
        self.check_tick(tick)?;
        Self::check_machine(machine, data.operands)?;
        let ready = tick.checked_add(1).ok_or(C220LoadCommitError::Overflow)?;
        let pending = self
            .pending
            .get(&data.request)
            .ok_or(C220LoadCommitError::InvalidRequest)?;
        if pending.operands != data.operands
            || pending.data.is_some()
            || data.tick > tick
            || data.tick < pending.issue_tick
        {
            return Err(C220LoadCommitError::InvalidRequest);
        }
        if self.retirements.len() == 64 {
            return Err(C220LoadCommitError::RetirementFull);
        }
        if self.mode == C220LoadCommitMode::DataBypass && !pending.suppressed {
            machine.set_xreg(data.operands.destination_register, data.value)?;
            self.owners.remove(&data.operands.destination_register);
            self.pending
                .get_mut(&data.request)
                .expect("checked load")
                .writeback_tick = Some(tick);
        }
        self.pending
            .get_mut(&data.request)
            .expect("checked load")
            .data = Some(data);
        self.retirements.push_back((ready, data.request));
        self.tick = tick;
        Ok(())
    }

    /// Consume one ready notification. The event scheduler controls invocation order.
    pub fn retire_next_at(
        &mut self,
        tick: u64,
        machine: &mut ScalarMachine,
    ) -> Result<Option<C220LoadRetirement>, C220LoadCommitError> {
        self.check_tick(tick)?;
        let Some((_, request)) = self
            .retirements
            .front()
            .copied()
            .filter(|entry| entry.0 <= tick)
        else {
            self.tick = tick;
            return Ok(None);
        };
        let pending = self.pending[&request];
        Self::check_machine(machine, pending.operands)?;
        let data = pending.data.expect("queued load data");
        let mut writeback_tick = pending.writeback_tick;
        if self.mode == C220LoadCommitMode::Retirement && !pending.suppressed {
            machine.set_xreg(pending.operands.destination_register, data.value)?;
            self.owners.remove(&pending.operands.destination_register);
            writeback_tick = Some(tick);
        }
        self.retirements.pop_front();
        self.pending.remove(&request);
        self.tick = tick;
        Ok(Some(C220LoadRetirement {
            data,
            issue_tick: pending.issue_tick,
            writeback_tick,
            retire_tick: tick,
            suppressed: pending.suppressed,
            register_value: machine.xregs()[usize::from(pending.operands.destination_register)],
        }))
    }

    fn check_tick(&self, tick: u64) -> Result<(), C220LoadCommitError> {
        if tick < self.tick {
            Err(C220LoadCommitError::TimeReversal)
        } else {
            Ok(())
        }
    }

    fn check_machine(
        machine: &ScalarMachine,
        operands: C220LoadOperands,
    ) -> Result<(), C220LoadCommitError> {
        if machine.architecture() != Architecture::Dav2201
            || usize::from(operands.destination_register) >= machine.xregs().len()
            || usize::from(operands.base_register) >= machine.xregs().len()
        {
            Err(C220LoadCommitError::InvalidOperands)
        } else {
            Ok(())
        }
    }
}
