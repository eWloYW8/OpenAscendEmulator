use super::*;
use crate::sim::c220::scalar::C220StoreOperands;
use crate::sim::c220::scalar::lsu::commit::C220StoreResponse;
use crate::sim::c220::scalar::lsu::scheduler::C220LsuStoreValue;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220CoreStoreIssue {
    pub instruction_id: u64,
    pub tick: u64,
    pub operands: C220StoreOperands,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220CoreStoreCompletion {
    pub issue: C220CoreStoreIssue,
    pub data: C220LsuStoreValue,
    pub retire_tick: u64,
    /// A later transport notification for an already retired store instruction.
    /// It does not represent another memory write.
    pub repeated_notification: bool,
    /// Cache completions in operand order, including an earlier half whose
    /// notification was consumed before the whole instruction could retire.
    pub responses: [Option<C220StoreResponse>; 2],
}

impl C220Core {
    pub fn pending_store_instructions(&self) -> impl Iterator<Item = &C220CoreStoreIssue> {
        self.lsu.iter().flat_map(|lsu| {
            lsu.stores
                .values()
                .chain(lsu.ingress.iter().filter_map(|entry| match entry {
                    DispatchedLsu::Store(issue) => Some(issue),
                    _ => None,
                }))
        })
    }

    pub fn take_store_completions(&mut self) -> Vec<C220CoreStoreCompletion> {
        self.lsu
            .as_mut()
            .map(|lsu| std::mem::take(&mut lsu.store_completions))
            .unwrap_or_default()
    }

    pub(in crate::sim::c220::core) fn step_store_at(
        &mut self,
        tick: u64,
        word: u32,
    ) -> Result<C220CoreStep, C220CoreError> {
        let pc = self.state.scalar().pc();
        if self.state.scalar().is_halted() {
            return Err(crate::sim::c220::state::C220ExecutionError::ProgramEnded { pc }.into());
        }
        let next_tick = tick.checked_add(1).ok_or(C220CoreError::TimeOverflow)?;
        let operands = C220StoreOperands::capture(self.state.scalar().machine(), pc, word)
            .map_err(crate::sim::common::scalar::ScalarInstructionError::from)?;
        if operands.second_source_operand.is_some()
            && (operands.effective_address & 63) + operands.bytes().len() as u64 > 64
            && !operands
                .effective_address
                .is_multiple_of(u64::from(operands.width_bytes))
        {
            return Err(C220CoreError::UnsupportedTimedLsuAccess);
        }
        if let Some(resume_tick) = [67, 68]
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
        let lsu = self.lsu.as_mut().ok_or(C220CoreError::LsuUnconfigured)?;
        if lsu.cache.is_none() {
            return Err(C220CoreError::LsuUnconfigured);
        }
        if lsu.ingress.len() == 2 {
            return Ok(C220CoreStep::Stalled(C220Stall {
                tick,
                pc,
                resume_tick: next_tick,
                cause: C220StallCause::LsuDependency,
            }));
        }
        if let Some(base) = operands.updated_base {
            self.state
                .scalar_mut()
                .machine_mut()
                .write_existing_xreg(operands.base_register, base);
        }
        let issue = C220CoreStoreIssue {
            instruction_id: self.next_instruction_id,
            tick,
            operands,
        };
        lsu.ingress.push_back(DispatchedLsu::Store(issue));
        lsu.next_tick = Some(
            lsu.next_tick
                .map_or(next_tick, |prior| prior.min(next_tick)),
        );
        self.state.commit_c220_sequential_issue();
        Ok(C220CoreStep::Executed {
            tick,
            instruction: C220CoreInstruction::Store(issue),
        })
    }
}
