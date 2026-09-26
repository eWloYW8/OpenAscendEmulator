use super::{C220Core, C220CoreError, C220CoreInstruction, C220CoreStep};
use crate::sim::c220::mte::mte1::{C220_MTE1_OUTSTANDING_LIMIT, C220Mte1Command};
use crate::sim::c220::schedule::{C220Stall, C220StallCause};
use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Mte1QueuedCommand {
    pub instruction_id: u64,
    pub pc: u64,
    pub word: u32,
    pub accepted_tick: u64,
    pub ready_tick: u64,
    pub command: C220Mte1Command,
}

#[derive(Default)]
pub(super) struct Mte1Frontend {
    pub commands: VecDeque<C220Mte1QueuedCommand>,
    pub outcomes: Vec<C220CoreStep>,
}

impl C220Core {
    pub fn queued_mte1_commands(&self) -> impl Iterator<Item = C220Mte1QueuedCommand> + '_ {
        self.mte1_frontend.commands.iter().copied()
    }

    pub fn mte1_frontend_outcomes(&self) -> &[C220CoreStep] {
        &self.mte1_frontend.outcomes
    }

    pub fn outstanding_mte1_commands(&self) -> usize {
        self.mte1_frontend.commands.len() + self.mte1.pending_commands().count()
    }

    pub(super) fn pending_mte1_tick(&self) -> Option<u64> {
        self.mte1.next_event_tick().or_else(|| {
            self.mte1_frontend
                .commands
                .front()
                .map(|_| self.mte1.tick().saturating_add(1))
        })
    }

    pub(super) fn mte1_accept_blocker(&self) -> Option<C220StallCause> {
        if self.mte1_frontend.commands.len() == 3 {
            Some(C220StallCause::Mte1CommandQueueFull)
        } else if self.outstanding_mte1_commands() >= C220_MTE1_OUTSTANDING_LIMIT {
            Some(C220StallCause::Mte1OutstandingLimit)
        } else {
            None
        }
    }

    pub(super) fn enqueue_mte1_at(
        &mut self,
        tick: u64,
        pc: u64,
        word: u32,
        command: C220Mte1Command,
    ) -> Result<C220CoreStep, C220CoreError> {
        let ready_tick = tick.checked_add(3).ok_or(C220CoreError::TimeOverflow)?;
        let queued = C220Mte1QueuedCommand {
            instruction_id: self.next_instruction_id,
            pc,
            word,
            accepted_tick: tick,
            ready_tick,
            command,
        };
        if let C220Mte1Command::WriteSpr(step) = command {
            self.state
                .scalar_mut()
                .machine_mut()
                .set_spr_value(step.destination_spr, step.value)
                .map_err(C220CoreError::MteSpr)?;
        }
        self.mte1_frontend.commands.push_back(queued);
        self.update_mte1_command_head();
        self.state.commit_c220_sequential_issue();
        Ok(C220CoreStep::Executed {
            tick,
            instruction: C220CoreInstruction::Mte1Queued(queued),
        })
    }

    fn update_mte1_command_head(&mut self) {
        if let Some(pipeline) = &mut self.mte_pipeline {
            pipeline
                .set_mte1_command_head(self.mte1_frontend.commands.front().map(|c| c.ready_tick));
        }
    }

    pub(super) fn dispatch_mte1_head_at(&mut self, tick: u64) -> Result<(), C220CoreError> {
        let queued = *self
            .mte1_frontend
            .commands
            .front()
            .expect("armed MTE1 dispatch");
        let pipeline = self.mte_pipeline.as_ref().expect("armed MTE1 dispatch");
        let result = if !self.mte1.can_issue(pipeline, queued.command) {
            C220CoreStep::Stalled(C220Stall {
                tick,
                pc: queued.pc,
                resume_tick: tick.checked_add(1).ok_or(C220CoreError::TimeOverflow)?,
                cause: C220StallCause::Mte1IssueRate,
            })
        } else {
            let pipeline = self.mte_pipeline.as_mut().expect("armed MTE1 dispatch");
            if !self.mte1.prepare_flags(
                pipeline,
                queued.instruction_id,
                queued.command,
                &mut self.hardware_flags,
            )? {
                self.mte1_frontend
                    .outcomes
                    .push(C220CoreStep::Stalled(C220Stall {
                        tick,
                        pc: queued.pc,
                        resume_tick: tick.checked_add(1).ok_or(C220CoreError::TimeOverflow)?,
                        cause: C220StallCause::HardwareFlagDependency,
                    }));
                pipeline.finish_mte1_dispatch();
                return Ok(());
            }
            let control = if let C220Mte1Command::HardwareFlag(step) = queued.command {
                Some(self.dispatch_hardware_flag_at(tick, queued.instruction_id, step)?)
            } else {
                None
            };
            if let Some(stalled @ C220CoreStep::Stalled(_)) = control {
                stalled
            } else {
                let issue = self.mte1.issue(
                    self.mte_pipeline.as_mut().expect("armed MTE1 dispatch"),
                    queued.instruction_id,
                    queued.pc,
                    queued.command,
                    &mut self.hardware_flags,
                )?;
                control.unwrap_or(C220CoreStep::Executed {
                    tick,
                    instruction: C220CoreInstruction::Mte1 {
                        instruction_id: queued.instruction_id,
                        pc: queued.pc,
                        command: queued.command,
                        issue,
                    },
                })
            }
        };
        if matches!(result, C220CoreStep::Executed { .. }) {
            self.mte1_frontend.commands.pop_front();
        }
        self.mte1_frontend.outcomes.push(result);
        self.update_mte1_command_head();
        self.mte_pipeline
            .as_mut()
            .expect("armed MTE1 dispatch")
            .finish_mte1_dispatch();
        Ok(())
    }
}
