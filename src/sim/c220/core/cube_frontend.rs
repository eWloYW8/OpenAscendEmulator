use std::collections::VecDeque;

use super::{C220Core, C220CoreError, C220CoreInstruction, C220CoreStep};
use crate::sim::c220::cube::C220CubeIssue;
use crate::sim::c220::cube::frontend::{
    C220CubeBarrier, C220CubeCommand, C220CubeFrontendConfig, C220CubeQueuedCommand,
};
use crate::sim::c220::schedule::{C220Stall, C220StallCause};

pub(super) struct CubeFrontend {
    pub(super) config: C220CubeFrontendConfig,
    pub(super) commands: VecDeque<C220CubeQueuedCommand>,
    pub(super) active: Option<C220CubeQueuedCommand>,
    pub(super) last_accepted: Option<u64>,
    pub(super) barriers: VecDeque<C220CubeBarrier>,
    pub next_tick: Option<u64>,
    pub outcomes: Vec<C220CoreStep>,
}

impl CubeFrontend {
    pub(super) fn new(config: C220CubeFrontendConfig) -> Self {
        Self {
            config,
            commands: VecDeque::new(),
            active: None,
            last_accepted: None,
            barriers: VecDeque::new(),
            next_tick: None,
            outcomes: Vec::new(),
        }
    }
}

impl C220Core {
    pub fn queued_cube_commands(&self) -> impl Iterator<Item = C220CubeQueuedCommand> + '_ {
        self.cube_frontend.commands.iter().copied()
    }

    pub fn active_cube_control(&self) -> Option<C220CubeQueuedCommand> {
        self.cube_frontend.active
    }

    pub fn cube_frontend_outcomes(&self) -> &[C220CoreStep] {
        &self.cube_frontend.outcomes
    }

    pub fn outstanding_cube_commands(&self) -> usize {
        self.cube.pipeline.pending_retirement_count()
            + usize::from(self.cube_frontend.active.is_some())
    }

    pub(super) fn enqueue_cube_at(
        &mut self,
        tick: u64,
        pc: u64,
        word: u32,
        command: C220CubeCommand,
    ) -> Result<C220CoreStep, C220CoreError> {
        let ready_tick = tick.checked_add(1).ok_or(C220CoreError::TimeOverflow)?;
        if self.cube_frontend.commands.len() >= self.cube_frontend.config.queue_depth.get() as usize
        {
            return Ok(C220CoreStep::Stalled(C220Stall {
                tick,
                pc,
                resume_tick: ready_tick,
                cause: C220StallCause::CubeIssueQueueFull,
            }));
        }
        let queued = C220CubeQueuedCommand {
            instruction_id: self.next_instruction_id,
            pc,
            word,
            accepted_tick: tick,
            ready_tick,
            command,
        };
        self.cube_frontend.commands.push_back(queued);
        self.cube_frontend.last_accepted = Some(queued.instruction_id);
        self.cube_frontend.next_tick.get_or_insert(ready_tick);
        self.state.commit_c220_sequential_issue();
        Ok(C220CoreStep::Executed {
            tick,
            instruction: C220CoreInstruction::CubeQueued(queued),
        })
    }

    pub(super) fn dispatch_cube_head_at(&mut self, tick: u64) -> Result<(), C220CoreError> {
        if self.cube_frontend.next_tick.is_none_or(|next| next > tick) {
            return Ok(());
        }
        let retry = tick.checked_add(1).ok_or(C220CoreError::TimeOverflow)?;
        if self.cube_frontend.active.is_none() {
            let queued = *self
                .cube_frontend
                .commands
                .front()
                .expect("armed Cube receive");
            let pending = self.cube.pipeline.pending_retirement_count();
            let mut cause = if tick < self.cube.pipeline.next_accept_tick() {
                Some(C220StallCause::CubeDependency)
            } else if pending >= self.cube_frontend.config.outstanding_limit.get() as usize {
                Some(C220StallCause::CubeOutstandingLimit)
            } else if self.cube_frontend.barriers.iter().any(|barrier| {
                barrier
                    .predecessor
                    .is_some_and(|id| id < queued.instruction_id)
            }) {
                Some(C220StallCause::CubeBarrier)
            } else if pending != 0
                && matches!(
                    queued.command,
                    C220CubeCommand::WriteSpr(_) | C220CubeCommand::CrossCore { .. }
                )
            {
                Some(C220StallCause::CubeDependency)
            } else {
                None
            };
            if cause.is_none()
                && let C220CubeCommand::WaitMte1(step) = queued.command
                && !self.mte1.wait_event(step.flag_id)
            {
                cause = Some(C220StallCause::PipelineEventDependency);
            }
            if let Some(cause) = cause {
                self.cube_frontend
                    .outcomes
                    .push(C220CoreStep::Stalled(C220Stall {
                        tick,
                        pc: queued.pc,
                        resume_tick: retry,
                        cause,
                    }));
                self.cube_frontend.next_tick = Some(retry);
                return Ok(());
            }
            self.cube_frontend.active = self.cube_frontend.commands.pop_front();
        }
        let queued = self.cube_frontend.active.expect("received Cube command");
        let instruction = match queued.command {
            C220CubeCommand::WaitMte1(step) => C220CoreInstruction::CubeFlag(step),
            C220CubeCommand::Mmad {
                instruction,
                registers,
                execution_control,
                timing_control,
            } => {
                let parameters = instruction.parameters(registers);
                let ticket = self.cube.pipeline.preview_issue(
                    tick,
                    instruction,
                    parameters,
                    timing_control,
                )?;
                let issue = C220CubeIssue {
                    instruction_id: queued.instruction_id,
                    pc: queued.pc,
                    word: queued.word,
                    instruction,
                    registers,
                    parameters,
                    execution_control,
                    ticket,
                };
                self.cube.issue(issue, &mut self.local_memory)?;
                C220CoreInstruction::Cube(issue)
            }
            C220CubeCommand::WriteSpr(mut step) => {
                let machine = self.state.scalar_mut().machine_mut();
                step.prior_destination_value = machine.spr_value(step.destination_spr);
                machine
                    .set_spr_value(step.destination_spr, step.value)
                    .map_err(crate::sim::c220::cube::C220CubeRuntimeError::from)?;
                C220CoreInstruction::CubeSpr {
                    instruction_id: queued.instruction_id,
                    step,
                }
            }
            C220CubeCommand::HardwareFlag(step) => {
                match self.dispatch_cube_hardware_flag_at(tick, queued.instruction_id, step)? {
                    C220CoreStep::Executed { instruction, .. } => instruction,
                    stalled @ C220CoreStep::Stalled(_) => {
                        self.cube_frontend.outcomes.push(stalled);
                        self.cube_frontend.next_tick = Some(retry);
                        return Ok(());
                    }
                }
            }
            C220CubeCommand::CrossCore {
                instruction,
                payload,
            } => C220CoreInstruction::CrossCore(crate::sim::c220::sync::C220CrossCoreReception {
                instruction_id: queued.instruction_id,
                pc: queued.pc,
                tick,
                instruction,
                payload,
            }),
        };
        self.cube_frontend.active = None;
        self.cube_frontend
            .outcomes
            .push(C220CoreStep::Executed { tick, instruction });
        self.cube_frontend.next_tick = self
            .cube_frontend
            .commands
            .front()
            .map(|next| next.ready_tick.max(retry));
        self.cube.advance_event(
            tick,
            &mut self.local_memory,
            &mut self.hardware_flags,
            self.state.scalar_mut().machine_mut(),
        )?;
        self.release_cube_barriers_at(tick);
        Ok(())
    }
}
