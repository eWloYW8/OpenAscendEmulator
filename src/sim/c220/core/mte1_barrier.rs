use super::{C220Core, C220CoreError, C220CoreInstruction, C220CoreStep, C220Mte1Operation};
use crate::isa::flow::PipelineBarrierStep;
use crate::sim::c220::schedule::C220Stall;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Mte1Barrier {
    pub instruction_id: u64,
    pub step: PipelineBarrierStep,
    pub predecessor: Option<u64>,
    pub issued_tick: u64,
    pub requires_idle: bool,
}

impl C220Core {
    pub fn pending_mte1_barriers(&self) -> impl ExactSizeIterator<Item = &C220Mte1Barrier> {
        self.mte1_frontend.barriers.iter()
    }

    fn mte1_instruction_pending(&self, id: u64) -> bool {
        self.mte1_frontend
            .issued
            .iter()
            .any(|c| c.instruction_id == id)
            || self
                .mte1_frontend
                .commands
                .iter()
                .any(|c| c.instruction_id == id)
            || self.mte1.pending_commands().any(|c| c.instruction_id == id)
    }

    pub(super) fn step_mte1_barrier_at(
        &mut self,
        tick: u64,
        step: PipelineBarrierStep,
    ) -> Result<C220CoreStep, C220CoreError> {
        if self.mte_pipeline.is_none() {
            return Err(C220CoreError::MteUnconfigured);
        }
        if let Some(cause) = self.mte1_accept_blocker() {
            return Ok(C220CoreStep::Stalled(C220Stall {
                tick,
                pc: step.pc,
                resume_tick: tick.checked_add(1).ok_or(C220CoreError::TimeOverflow)?,
                cause,
            }));
        }
        let barrier = C220Mte1Barrier {
            instruction_id: self.next_instruction_id,
            step,
            predecessor: self.mte1_frontend.last_accepted,
            issued_tick: tick,
            requires_idle: self
                .mte1_frontend
                .issued
                .back()
                .is_some_and(|c| matches!(c.operation, C220Mte1Operation::Flag(_))),
        };
        let pending = barrier
            .predecessor
            .is_some_and(|id| self.mte1_instruction_pending(id));
        if pending {
            self.mte1_frontend.barriers.push_back(barrier);
        }
        self.state.commit_c220_sequential_issue();
        Ok(C220CoreStep::Executed {
            tick,
            instruction: C220CoreInstruction::Mte1Barrier {
                barrier,
                completed_tick: (!pending).then_some(tick),
            },
        })
    }

    pub(super) fn release_mte1_barriers_at(&mut self, tick: u64) {
        while let Some(&barrier) = self.mte1_frontend.barriers.front() {
            if barrier
                .predecessor
                .is_some_and(|id| self.mte1_instruction_pending(id))
                || (barrier.requires_idle && self.outstanding_mte1_commands() != 0)
            {
                break;
            }
            self.mte1_frontend.barriers.pop_front();
            self.mte1_frontend.outcomes.push(C220CoreStep::Executed {
                tick,
                instruction: C220CoreInstruction::Mte1Barrier {
                    barrier,
                    completed_tick: Some(tick),
                },
            });
        }
    }
}
