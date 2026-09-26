use super::{C220Core, C220CoreError, C220CoreInstruction, C220CoreStep, C220Mte3Operation};
use crate::isa::flow::PipelineBarrierStep;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Mte3Barrier {
    pub instruction_id: u64,
    pub step: PipelineBarrierStep,
    pub predecessor: Option<u64>,
    pub issued_tick: u64,
    pub requires_idle: bool,
}

impl C220Core {
    pub fn pending_mte3_barriers(&self) -> impl ExactSizeIterator<Item = &C220Mte3Barrier> {
        self.mte3_issue_queue.barriers.iter()
    }

    fn mte3_instruction_pending(&self, id: u64) -> bool {
        self.queued_mte3_instructions()
            .any(|c| c.instruction_id == id)
            || self.mte3.native_commands.contains_key(&id)
    }

    pub(super) fn step_mte3_barrier_at(
        &mut self,
        tick: u64,
        step: PipelineBarrierStep,
    ) -> Result<C220CoreStep, C220CoreError> {
        if !self.mte3.physical {
            return Err(C220CoreError::Mte3FrontendRequired);
        }
        if let Some(cause) = self.mte3_accept_blocker()? {
            return self.mte3_stall(tick, step.pc, cause);
        }
        let barrier = C220Mte3Barrier {
            instruction_id: self.next_instruction_id,
            step,
            predecessor: self.mte3_issue_queue.last_accepted,
            issued_tick: tick,
            requires_idle: self
                .queued_mte3_instructions()
                .last()
                .is_some_and(|c| matches!(c.operation, C220Mte3Operation::Flag(_))),
        };
        let pending = barrier
            .predecessor
            .is_some_and(|id| self.mte3_instruction_pending(id));
        if pending {
            self.mte3_issue_queue.barriers.push_back(barrier);
        }
        self.state.commit_c220_sequential_issue();
        Ok(C220CoreStep::Executed {
            tick,
            instruction: C220CoreInstruction::Mte3Barrier {
                barrier,
                completed_tick: (!pending).then_some(tick),
            },
        })
    }

    pub(super) fn release_mte3_barriers_at(&mut self, tick: u64) {
        while let Some(&barrier) = self.mte3_issue_queue.barriers.front() {
            if barrier
                .predecessor
                .is_some_and(|id| self.mte3_instruction_pending(id))
                || (barrier.requires_idle && !self.mte3.native_commands.is_empty())
            {
                break;
            }
            self.mte3_issue_queue.barriers.pop_front();
            self.mte3_issue_queue.outcomes.push(C220CoreStep::Executed {
                tick,
                instruction: C220CoreInstruction::Mte3Barrier {
                    barrier,
                    completed_tick: Some(tick),
                },
            });
        }
    }
}
