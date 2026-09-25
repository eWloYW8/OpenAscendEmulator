use super::*;
use crate::sim::c220::scalar::lsu::scheduler::C220LsuAtomicCompletion;
use crate::sim::c220::scalar::{C220AtomicStoreOperands, C220AtomicStoreResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220CoreAtomicIssue {
    pub instruction_id: u64,
    pub tick: u64,
    pub operands: C220AtomicStoreOperands,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220CoreAtomicCompletion {
    pub issue: C220CoreAtomicIssue,
    pub result: C220AtomicStoreResult,
    pub data: C220LsuAtomicCompletion,
    pub retire_tick: u64,
}

impl C220Core {
    pub fn take_atomic_store_completions(&mut self) -> Vec<C220CoreAtomicCompletion> {
        self.lsu
            .as_mut()
            .map(|lsu| std::mem::take(&mut lsu.atomic_completions))
            .unwrap_or_default()
    }

    pub fn pending_atomic_store_instructions(&self) -> impl Iterator<Item = &C220CoreAtomicIssue> {
        self.lsu.iter().flat_map(|lsu| {
            lsu.atomics
                .values()
                .map(|entry| &entry.0)
                .chain(lsu.ingress.iter().filter_map(|entry| match entry {
                    DispatchedLsu::AtomicStore(issue) => Some(issue),
                    _ => None,
                }))
        })
    }

    pub(in crate::sim::c220::core) fn step_atomic_store_at(
        &mut self,
        tick: u64,
        word: u32,
    ) -> Result<C220CoreStep, C220CoreError> {
        let pc = self.state.scalar().pc();
        if self.state.scalar().is_halted() {
            return Err(crate::sim::c220::state::C220ExecutionError::ProgramEnded { pc }.into());
        }
        let next = tick.checked_add(1).ok_or(C220CoreError::TimeOverflow)?;
        if let Some(resume_tick) = [3, 67, 90]
            .into_iter()
            .filter_map(|spr| self.scalar_timing.pending_spr_retirement(spr))
            .filter(|ready| *ready > tick)
            .max()
        {
            return Ok(C220CoreStep::Stalled(C220Stall {
                tick,
                pc,
                resume_tick,
                cause: C220StallCause::ScalarDependency,
            }));
        }
        let operands = C220AtomicStoreOperands::capture(self.state.scalar().machine(), pc, word)
            .map_err(crate::sim::common::scalar::ScalarInstructionError::from)?;
        let lsu = self.lsu.as_mut().ok_or(C220CoreError::LsuUnconfigured)?;
        if lsu.cache.is_none() {
            return Err(C220CoreError::LsuUnconfigured);
        }
        if lsu.config.stores.line_bytes != 64
            || (operands.effective_address & 63) + operands.bytes().len() as u64 > 64
        {
            return Err(C220CoreError::UnsupportedTimedLsuAccess);
        }
        if lsu.ingress.len() == 2 {
            return Ok(C220CoreStep::Stalled(C220Stall {
                tick,
                pc,
                resume_tick: next,
                cause: C220StallCause::LsuDependency,
            }));
        }
        if !operands.is_external() {
            return Err(
                crate::sim::c220::scalar::C220AtomicStoreError::LocalAddress(
                    operands.effective_address,
                )
                .into(),
            );
        }
        if let Some(base) = operands.updated_base {
            self.state
                .scalar_mut()
                .machine_mut()
                .set_xreg(operands.instruction().base_register, base)
                .map_err(crate::sim::common::scalar::ScalarInstructionError::from)?;
        }
        let issue = C220CoreAtomicIssue {
            instruction_id: self.next_instruction_id,
            tick,
            operands,
        };
        lsu.ingress.push_back(DispatchedLsu::AtomicStore(issue));
        lsu.next_tick = Some(lsu.next_tick.map_or(next, |prior| prior.min(next)));
        self.state.commit_c220_sequential_issue();
        Ok(C220CoreStep::Executed {
            tick,
            instruction: C220CoreInstruction::AtomicStore(issue),
        })
    }
}
