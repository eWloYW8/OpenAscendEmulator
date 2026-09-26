use super::{C220Core, C220CoreError, C220CoreInstruction, C220CoreStep, C220Mte2Operation};
use crate::isa::flow::PipelineBarrierStep;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Mte2Barrier {
    pub instruction_id: u64,
    pub step: PipelineBarrierStep,
    pub predecessor: Option<u64>,
    pub issued_tick: u64,
    pub requires_idle: bool,
}

impl C220Core {
    pub fn pending_mte2_barriers(&self) -> impl ExactSizeIterator<Item = &C220Mte2Barrier> {
        self.mte2_frontend.barriers.iter()
    }

    fn mte2_instruction_pending(&self, id: u64) -> bool {
        self.queued_mte2_instructions()
            .any(|c| c.instruction_id == id)
            || self.queued_mte2_commands().any(|c| c.instruction_id == id)
            || self.mte2.pending_commands().any(|c| c.instruction_id == id)
    }

    pub(super) fn step_mte2_barrier_at(
        &mut self,
        tick: u64,
        step: PipelineBarrierStep,
    ) -> Result<C220CoreStep, C220CoreError> {
        if self.mte_pipeline.is_none() {
            return Err(C220CoreError::MteUnconfigured);
        }
        if let Some(cause) = self.mte2_accept_blocker() {
            return self.mte2_stall(tick, step.pc, cause);
        }
        let barrier = C220Mte2Barrier {
            instruction_id: self.next_instruction_id,
            step,
            predecessor: self.mte2_frontend.last_accepted,
            issued_tick: tick,
            requires_idle: self
                .queued_mte2_instructions()
                .last()
                .is_some_and(|c| matches!(c.operation, C220Mte2Operation::Flag(_))),
        };
        let pending = barrier
            .predecessor
            .is_some_and(|id| self.mte2_instruction_pending(id));
        if pending {
            self.mte2_frontend.barriers.push_back(barrier);
        }
        self.state.commit_c220_sequential_issue();
        Ok(C220CoreStep::Executed {
            tick,
            instruction: C220CoreInstruction::Mte2Barrier {
                barrier,
                completed_tick: (!pending).then_some(tick),
            },
        })
    }

    pub(super) fn release_mte2_barriers_at(&mut self, tick: u64) {
        while let Some(&barrier) = self.mte2_frontend.barriers.front() {
            if barrier
                .predecessor
                .is_some_and(|id| self.mte2_instruction_pending(id))
                || (barrier.requires_idle && self.outstanding_mte2_commands() != 0)
            {
                break;
            }
            self.mte2_frontend.barriers.pop_front();
            self.mte2_frontend.outcomes.push(C220CoreStep::Executed {
                tick,
                instruction: C220CoreInstruction::Mte2Barrier {
                    barrier,
                    completed_tick: Some(tick),
                },
            });
        }
    }
}
