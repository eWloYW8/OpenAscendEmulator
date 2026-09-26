use super::{C220Core, C220CoreError, C220CoreInstruction, C220CoreStep};
use crate::isa::flow::PipelineBarrierStep;
use crate::sim::c220::cube::frontend::{C220CubeBarrier, C220CubeCommand};
use crate::sim::c220::schedule::{C220Stall, C220StallCause};

impl C220Core {
    pub fn pending_cube_barriers(&self) -> impl ExactSizeIterator<Item = &C220CubeBarrier> {
        self.cube_frontend.barriers.iter()
    }

    fn cube_instruction_pending(&self, id: u64) -> bool {
        self.cube.pipeline.has_pending_instruction(id)
            || self
                .cube_frontend
                .active
                .is_some_and(|command| command.instruction_id == id)
            || self
                .cube_frontend
                .commands
                .iter()
                .any(|command| command.instruction_id == id)
    }

    pub(super) fn step_cube_barrier_at(
        &mut self,
        tick: u64,
        step: PipelineBarrierStep,
    ) -> Result<C220CoreStep, C220CoreError> {
        if self.cube_frontend.commands.len() >= self.cube_frontend.config.queue_depth.get() as usize
        {
            return Ok(C220CoreStep::Stalled(C220Stall {
                tick,
                pc: step.pc,
                resume_tick: tick.checked_add(1).ok_or(C220CoreError::TimeOverflow)?,
                cause: C220StallCause::CubeIssueQueueFull,
            }));
        }
        let barrier = C220CubeBarrier {
            instruction_id: self.next_instruction_id,
            step,
            predecessor: self.cube_frontend.last_accepted,
            issued_tick: tick,
            requires_idle: self
                .cube_frontend
                .commands
                .back()
                .is_some_and(|command| matches!(command.command, C220CubeCommand::Flag(_))),
        };
        let pending = barrier
            .predecessor
            .is_some_and(|id| self.cube_instruction_pending(id));
        if pending {
            self.cube_frontend.barriers.push_back(barrier);
        }
        self.state.commit_c220_sequential_issue();
        Ok(C220CoreStep::Executed {
            tick,
            instruction: C220CoreInstruction::CubeBarrier {
                barrier,
                completed_tick: (!pending).then_some(tick),
            },
        })
    }

    pub(super) fn release_cube_barriers_at(&mut self, tick: u64) {
        while let Some(&barrier) = self.cube_frontend.barriers.front() {
            if barrier
                .predecessor
                .is_some_and(|id| self.cube_instruction_pending(id))
                || (barrier.requires_idle && self.outstanding_cube_commands() != 0)
            {
                break;
            }
            self.cube_frontend.barriers.pop_front();
            self.cube_frontend.outcomes.push(C220CoreStep::Executed {
                tick,
                instruction: C220CoreInstruction::CubeBarrier {
                    barrier,
                    completed_tick: Some(tick),
                },
            });
        }
    }
}
