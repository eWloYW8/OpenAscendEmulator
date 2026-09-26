use super::{C220Core, C220CoreError, C220CoreInstruction, C220CoreStep};
use crate::isa::flow::PipelineBarrierStep;
use crate::sim::c220::schedule::{C220Stall, C220StallCause};

/// A local ordering boundary, attached to the last accepted FIX instruction.
/// It consumes neither a command slot nor an outstanding-command credit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FixpBarrier {
    pub instruction_id: u64,
    pub step: PipelineBarrierStep,
    pub predecessor: Option<u64>,
    pub issued_tick: u64,
    pub requires_idle: bool,
}

impl C220Core {
    pub fn pending_fixp_barriers(&self) -> impl ExactSizeIterator<Item = &C220FixpBarrier> {
        self.fixp_frontend.barriers.iter()
    }

    fn fixp_instruction_pending(&self, id: u64) -> bool {
        self.fixp_frontend.contains_instruction(id)
            || self
                .fixp_engine()
                .is_some_and(|engine| engine.command_retirement_queue().contains(&id))
    }

    pub(super) fn step_fixp_barrier_at(
        &mut self,
        tick: u64,
        step: PipelineBarrierStep,
    ) -> Result<C220CoreStep, C220CoreError> {
        let config = self
            .fixp_frontend
            .config
            .ok_or(C220CoreError::FixpUnconfigured)?;
        if self.fixp_issue_queue_len() >= config.issue_queue_depth.get() as usize {
            return Ok(C220CoreStep::Stalled(C220Stall {
                tick,
                pc: step.pc,
                resume_tick: tick.checked_add(1).ok_or(C220CoreError::TimeOverflow)?,
                cause: C220StallCause::FixpIssueQueueFull,
            }));
        }
        let barrier = C220FixpBarrier {
            instruction_id: self.next_instruction_id,
            step,
            predecessor: self.fixp_frontend.last_issued,
            issued_tick: tick,
            requires_idle: self
                .fixp_frontend
                .issued
                .back()
                .is_some_and(|(_, command)| {
                    matches!(
                        command.operation,
                        super::fixp_frontend::CapturedFixpOperation::Flag(_)
                    )
                }),
        };
        let pending = barrier
            .predecessor
            .is_some_and(|id| self.fixp_instruction_pending(id));
        if pending {
            self.fixp_frontend.barriers.push_back(barrier);
        }
        self.state.commit_c220_sequential_issue();
        Ok(C220CoreStep::Executed {
            tick,
            instruction: C220CoreInstruction::FixpBarrier {
                barrier,
                completed_tick: (!pending).then_some(tick),
            },
        })
    }

    pub(super) fn release_fixp_barriers_at(&mut self, tick: u64) {
        while let Some(&barrier) = self.fixp_frontend.barriers.front() {
            if barrier
                .predecessor
                .is_some_and(|id| self.fixp_instruction_pending(id))
                || (barrier.requires_idle && self.outstanding_fixp_commands() != 0)
            {
                break;
            }
            self.fixp_frontend.barriers.pop_front();
            self.fixp_frontend.outcomes.push(C220CoreStep::Executed {
                tick,
                instruction: C220CoreInstruction::FixpBarrier {
                    barrier,
                    completed_tick: Some(tick),
                },
            });
        }
    }
}
