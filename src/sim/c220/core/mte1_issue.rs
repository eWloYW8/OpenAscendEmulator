use std::num::NonZeroU32;

use super::{C220Core, C220CoreError, C220CoreInstruction, C220CoreStep, C220Mte1QueuedCommand};
use crate::isa::flow::{FlagOperation, FlagStep};
use crate::sim::c220::mte::mte1::C220Mte1Command;
use crate::sim::c220::schedule::{C220Stall, C220StallCause};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Mte1FrontendConfig {
    pub issue_queue_depth: NonZeroU32,
    pub outstanding_limit: NonZeroU32,
}

impl Default for C220Mte1FrontendConfig {
    fn default() -> Self {
        Self {
            issue_queue_depth: NonZeroU32::new(32).unwrap(),
            outstanding_limit: NonZeroU32::new(31).unwrap(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220Mte1Operation {
    Command(C220Mte1Command),
    Flag(FlagStep),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Mte1IssuedInstruction {
    pub instruction_id: u64,
    pub pc: u64,
    pub word: u32,
    pub accepted_tick: u64,
    /// Earliest reception by the MTE command scheduler.
    pub ready_tick: u64,
    pub operation: C220Mte1Operation,
}

impl C220Core {
    pub fn queued_mte1_instructions(&self) -> impl Iterator<Item = C220Mte1IssuedInstruction> + '_ {
        self.mte1_frontend.issued.iter().copied()
    }

    pub(super) fn enqueue_mte1_issue_at(
        &mut self,
        tick: u64,
        pc: u64,
        word: u32,
        operation: C220Mte1Operation,
    ) -> Result<C220CoreStep, C220CoreError> {
        let queued = C220Mte1IssuedInstruction {
            instruction_id: self.next_instruction_id,
            pc,
            word,
            accepted_tick: tick,
            ready_tick: tick.checked_add(1).ok_or(C220CoreError::TimeOverflow)?,
            operation,
        };
        let mut mte2 = None;
        if let C220Mte1Operation::Command(C220Mte1Command::WriteSpr(step)) = operation {
            self.state
                .scalar_mut()
                .machine_mut()
                .set_spr_value(step.destination_spr, step.value)
                .map_err(C220CoreError::MteSpr)?;
            if matches!(step.destination_spr, 13 | 15) {
                mte2 = Some(self.append_mte2_issue_at(
                    tick,
                    pc,
                    word,
                    super::C220Mte2Operation::Command(
                        crate::sim::c220::mte::mte2::C220Mte2Command::WriteSpr(step),
                    ),
                )?);
            }
        }
        self.mte1_frontend.issued.push_back(queued);
        self.mte1_frontend.last_accepted = Some(queued.instruction_id);
        self.update_mte1_command_head();
        self.state.commit_c220_sequential_issue();
        Ok(C220CoreStep::Executed {
            tick,
            instruction: match mte2 {
                Some(mte2) => C220CoreInstruction::MteSprQueued { mte1: queued, mte2 },
                None => C220CoreInstruction::Mte1Queued(queued),
            },
        })
    }

    pub(super) fn transfer_mte1_issue_at(&mut self, tick: u64) -> Result<(), C220CoreError> {
        let queued = *self
            .mte1_frontend
            .issued
            .front()
            .expect("armed MTE1 reception");
        let mut cause = if self.mte1_frontend.commands.len() == 3 {
            Some(C220StallCause::Mte1CommandQueueFull)
        } else if self.outstanding_mte1_commands()
            >= self.mte1_frontend.config.outstanding_limit.get() as usize
        {
            Some(C220StallCause::Mte1OutstandingLimit)
        } else if self.mte1_frontend.barriers.iter().any(|barrier| {
            barrier
                .predecessor
                .is_some_and(|id| id < queued.instruction_id)
        }) {
            Some(C220StallCause::Mte1Barrier)
        } else if matches!(
            queued.operation,
            C220Mte1Operation::Command(C220Mte1Command::CrossCore { .. })
        ) && self.outstanding_mte1_commands() != 0
        {
            Some(C220StallCause::Mte1Dependency)
        } else {
            None
        };
        if cause.is_none()
            && let C220Mte1Operation::Flag(step) = queued.operation
            && step.instruction.operation == FlagOperation::Wait
            && self
                .pipeline_events
                .consume(queued.instruction_id, step, tick)
                .is_none()
        {
            cause = Some(C220StallCause::PipelineEventDependency);
        }
        let result = if let Some(cause) = cause {
            C220CoreStep::Stalled(C220Stall {
                tick,
                pc: queued.pc,
                resume_tick: tick.checked_add(1).ok_or(C220CoreError::TimeOverflow)?,
                cause,
            })
        } else {
            let instruction = match queued.operation {
                C220Mte1Operation::Flag(step) => {
                    if step.instruction.operation == FlagOperation::Set {
                        let predecessor = self
                            .mte1_frontend
                            .commands
                            .back()
                            .map(|command| command.instruction_id)
                            .or_else(|| {
                                self.mte1
                                    .pending_commands()
                                    .last()
                                    .map(|command| command.instruction_id)
                            });
                        self.pipeline_events
                            .set(queued.instruction_id, step, predecessor, tick);
                    }
                    C220CoreInstruction::Mte1Flag(step)
                }
                C220Mte1Operation::Command(command) => {
                    let command = C220Mte1QueuedCommand {
                        instruction_id: queued.instruction_id,
                        pc: queued.pc,
                        word: queued.word,
                        accepted_tick: tick,
                        ready_tick: tick.checked_add(3).ok_or(C220CoreError::TimeOverflow)?,
                        command,
                    };
                    self.mte1_frontend.commands.push_back(command);
                    C220CoreInstruction::Mte1Scheduled(command)
                }
            };
            self.mte1_frontend.issued.pop_front();
            C220CoreStep::Executed { tick, instruction }
        };
        self.mte1_frontend.outcomes.push(result);
        self.release_mte1_barriers_at(tick);
        self.update_mte1_command_head();
        self.mte_pipeline
            .as_mut()
            .expect("armed MTE1 reception")
            .finish_mte1_issue();
        Ok(())
    }
}
