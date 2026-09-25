use super::*;
use crate::sim::c220::scalar::C220PreloadOperands;
use crate::sim::c220::scalar::lsu::scheduler::C220LsuPreloadCompletion;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220CorePreloadIssue {
    pub instruction_id: u64,
    pub tick: u64,
    pub operands: C220PreloadOperands,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220CorePreloadCompletion {
    pub issue: C220CorePreloadIssue,
    pub data: C220LsuPreloadCompletion,
    pub retire_tick: u64,
}

impl C220Core {
    pub fn take_preload_completions(&mut self) -> Vec<C220CorePreloadCompletion> {
        self.lsu
            .as_mut()
            .map(|lsu| std::mem::take(&mut lsu.preload_completions))
            .unwrap_or_default()
    }

    pub fn pending_preload_instructions(&self) -> impl Iterator<Item = &C220CorePreloadIssue> {
        self.lsu.iter().flat_map(|lsu| {
            lsu.preloads
                .values()
                .map(|entry| &entry.0)
                .chain(lsu.ingress.iter().filter_map(|entry| match entry {
                    DispatchedLsu::Preload(issue) => Some(issue),
                    _ => None,
                }))
        })
    }

    pub(in crate::sim::c220::core) fn step_preload_at(
        &mut self,
        tick: u64,
        word: u32,
    ) -> Result<C220CoreStep, C220CoreError> {
        let pc = self.state.scalar().pc();
        if self.state.scalar().is_halted() {
            return Err(crate::sim::c220::state::C220ExecutionError::ProgramEnded { pc }.into());
        }
        let next = tick.checked_add(1).ok_or(C220CoreError::TimeOverflow)?;
        let operands = C220PreloadOperands::capture(self.state.scalar().machine(), pc, word)
            .map_err(crate::sim::common::scalar::ScalarInstructionError::from)?;
        let lsu = self.lsu.as_mut().ok_or(C220CoreError::LsuUnconfigured)?;
        if lsu.cache.is_none() {
            return Err(C220CoreError::LsuUnconfigured);
        }
        if lsu.config.misses.line_bytes != 64 {
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
        let issue = C220CorePreloadIssue {
            instruction_id: self.next_instruction_id,
            tick,
            operands,
        };
        lsu.ingress.push_back(DispatchedLsu::Preload(issue));
        lsu.next_tick = Some(lsu.next_tick.map_or(next, |prior| prior.min(next)));
        self.state.commit_c220_sequential_issue();
        Ok(C220CoreStep::Executed {
            tick,
            instruction: C220CoreInstruction::Preload(issue),
        })
    }
}
