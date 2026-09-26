use std::{collections::VecDeque, num::NonZeroU32};

use super::{C220Core, C220CoreError, C220CoreInstruction, C220CoreStep, C220Mte2Barrier};
use crate::isa::flow::{FlagOperation, FlagStep};
use crate::sim::c220::device::C220CoreKind;
use crate::sim::c220::mte::mte2::C220Mte2Command;
use crate::sim::c220::schedule::C220StallCause;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Mte2FrontendConfig {
    pub cube_issue_queue_depth: NonZeroU32,
    pub vector_issue_queue_depth: NonZeroU32,
    pub outstanding_limit: NonZeroU32,
}

impl Default for C220Mte2FrontendConfig {
    fn default() -> Self {
        Self {
            cube_issue_queue_depth: NonZeroU32::new(32).unwrap(),
            vector_issue_queue_depth: NonZeroU32::new(16).unwrap(),
            outstanding_limit: NonZeroU32::new(31).unwrap(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220Mte2Operation {
    Command(C220Mte2Command),
    Flag(FlagStep),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Mte2IssuedInstruction {
    pub instruction_id: u64,
    pub pc: u64,
    pub word: u32,
    pub accepted_tick: u64,
    /// Earliest reception by the command scheduler.
    pub ready_tick: u64,
    pub operation: C220Mte2Operation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Mte2QueuedCommand {
    pub instruction_id: u64,
    pub pc: u64,
    pub word: u32,
    pub accepted_tick: u64,
    /// Earliest dispatch to the selected command generator.
    pub ready_tick: u64,
    pub command: C220Mte2Command,
}

#[derive(Default)]
pub(super) struct Mte2Frontend {
    pub config: C220Mte2FrontendConfig,
    issued: VecDeque<C220Mte2IssuedInstruction>,
    commands: VecDeque<C220Mte2QueuedCommand>,
    last_received: Option<u64>,
    pub last_accepted: Option<u64>,
    pub barriers: VecDeque<C220Mte2Barrier>,
    pub outcomes: Vec<C220CoreStep>,
}

impl Mte2Frontend {
    pub(super) fn new(config: C220Mte2FrontendConfig) -> Self {
        Self {
            config,
            ..Self::default()
        }
    }
}

impl C220Core {
    pub fn queued_mte2_instructions(
        &self,
    ) -> impl ExactSizeIterator<Item = &C220Mte2IssuedInstruction> {
        self.mte2_frontend.issued.iter()
    }

    pub fn queued_mte2_commands(&self) -> impl ExactSizeIterator<Item = &C220Mte2QueuedCommand> {
        self.mte2_frontend.commands.iter()
    }

    pub fn mte2_frontend_outcomes(&self) -> &[C220CoreStep] {
        &self.mte2_frontend.outcomes
    }

    pub fn outstanding_mte2_commands(&self) -> usize {
        self.mte2_frontend.commands.len() + self.mte2.pending_commands().count()
    }

    pub fn mte2_is_busy(&self) -> bool {
        !self.mte2_frontend.issued.is_empty() || self.outstanding_mte2_commands() != 0
    }

    pub(super) fn mte2_accept_blocker(&self) -> Option<C220StallCause> {
        let depth = if self.mte_pipeline.as_ref()?.core_kind() == C220CoreKind::Cube {
            self.mte2_frontend.config.cube_issue_queue_depth
        } else {
            self.mte2_frontend.config.vector_issue_queue_depth
        };
        (self.mte2_frontend.issued.len() >= depth.get() as usize)
            .then_some(C220StallCause::Mte2IssueQueueFull)
    }

    pub(super) fn enqueue_mte2_issue_at(
        &mut self,
        tick: u64,
        pc: u64,
        word: u32,
        operation: C220Mte2Operation,
    ) -> Result<C220CoreStep, C220CoreError> {
        let queued = C220Mte2IssuedInstruction {
            instruction_id: self.next_instruction_id,
            pc,
            word,
            accepted_tick: tick,
            ready_tick: tick.checked_add(1).ok_or(C220CoreError::TimeOverflow)?,
            operation,
        };
        self.mte2_frontend.issued.push_back(queued);
        self.mte2_frontend.last_accepted = Some(queued.instruction_id);
        self.update_mte2_heads();
        self.state.commit_c220_sequential_issue();
        Ok(C220CoreStep::Executed {
            tick,
            instruction: C220CoreInstruction::Mte2Queued(queued),
        })
    }

    fn update_mte2_heads(&mut self) {
        let pipeline = self
            .mte_pipeline
            .as_mut()
            .expect("configured MTE2 frontend");
        pipeline.set_mte2_issue_head(self.mte2_frontend.issued.front().map(|c| c.ready_tick));
        pipeline.set_mte2_command_head(self.mte2_frontend.commands.front().map(|c| c.ready_tick));
    }

    pub(super) fn transfer_mte2_issue_at(&mut self, tick: u64) -> Result<(), C220CoreError> {
        let queued = *self
            .mte2_frontend
            .issued
            .front()
            .expect("armed MTE2 reception");
        let mut cause = if self.mte2_frontend.commands.len() == 3 {
            Some(C220StallCause::Mte2CommandQueueFull)
        } else if self.outstanding_mte2_commands()
            >= self.mte2_frontend.config.outstanding_limit.get() as usize
        {
            Some(C220StallCause::Mte2OutstandingLimit)
        } else if self.mte2_frontend.barriers.iter().any(|barrier| {
            barrier
                .predecessor
                .is_some_and(|id| id < queued.instruction_id)
        }) {
            Some(C220StallCause::Mte2Barrier)
        } else if matches!(
            queued.operation,
            C220Mte2Operation::Command(C220Mte2Command::CrossCore { .. })
        ) && self.outstanding_mte2_commands() != 0
        {
            Some(C220StallCause::Mte2Dependency)
        } else {
            None
        };
        if cause.is_none()
            && let C220Mte2Operation::Flag(step) = queued.operation
            && step.instruction.operation == FlagOperation::Wait
            && self
                .pipeline_events
                .consume(queued.instruction_id, step, tick)
                .is_none()
        {
            cause = Some(C220StallCause::PipelineEventDependency);
        }
        let outcome = if let Some(cause) = cause {
            self.mte2_stall(tick, queued.pc, cause)?
        } else {
            let instruction = match queued.operation {
                C220Mte2Operation::Flag(step) => {
                    if step.instruction.operation == FlagOperation::Set {
                        let predecessor = (self.outstanding_mte2_commands() != 0).then(|| {
                            self.mte2_frontend
                                .last_received
                                .expect("outstanding MTE2 command")
                        });
                        self.pipeline_events
                            .set(queued.instruction_id, step, predecessor, tick);
                    }
                    C220CoreInstruction::Mte2Flag(step)
                }
                C220Mte2Operation::Command(command) => {
                    let command = C220Mte2QueuedCommand {
                        instruction_id: queued.instruction_id,
                        pc: queued.pc,
                        word: queued.word,
                        accepted_tick: tick,
                        ready_tick: tick.checked_add(3).ok_or(C220CoreError::TimeOverflow)?,
                        command,
                    };
                    self.mte2_frontend.commands.push_back(command);
                    self.mte2_frontend.last_received = Some(queued.instruction_id);
                    C220CoreInstruction::Mte2Scheduled(command)
                }
            };
            self.mte2_frontend.issued.pop_front();
            C220CoreStep::Executed { tick, instruction }
        };
        self.mte2_frontend.outcomes.push(outcome);
        self.release_mte2_barriers_at(tick);
        self.update_mte2_heads();
        self.mte_pipeline
            .as_mut()
            .expect("armed MTE2 reception")
            .finish_mte2_issue();
        Ok(())
    }

    pub(super) fn dispatch_mte2_head_at(&mut self, tick: u64) -> Result<(), C220CoreError> {
        let queued = *self
            .mte2_frontend
            .commands
            .front()
            .expect("armed MTE2 dispatch");
        let outcome = if self.can_dispatch_mte2(queued.command, tick)? {
            let issue =
                self.dispatch_mte2_command(queued.instruction_id, queued.pc, queued.command)?;
            self.mte2_frontend.commands.pop_front();
            C220CoreStep::Executed {
                tick,
                instruction: C220CoreInstruction::Mte2(issue),
            }
        } else {
            self.mte2_stall(tick, queued.pc, C220StallCause::Mte2IssueRate)?
        };
        self.mte2_frontend.outcomes.push(outcome);
        self.update_mte2_heads();
        self.mte_pipeline
            .as_mut()
            .expect("armed MTE2 dispatch")
            .finish_mte2_dispatch();
        Ok(())
    }
}
