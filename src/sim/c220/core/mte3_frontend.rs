use std::{collections::VecDeque, num::NonZeroU32};

use super::{C220Core, C220CoreError, C220CoreInstruction, C220CoreStep, C220Mte3Barrier};
use crate::isa::flow::{FlagOperation, FlagStep};
use crate::sim::c220::device::C220CoreKind;
use crate::sim::c220::mte::mte3::C220Mte3Step;
use crate::sim::c220::mte::mte3::frontend::C220Mte3Command;
use crate::sim::c220::schedule::{C220Stall, C220StallCause};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Mte3IssueQueueConfig {
    pub cube_depth: NonZeroU32,
    pub vector_depth: NonZeroU32,
}

impl Default for C220Mte3IssueQueueConfig {
    fn default() -> Self {
        Self {
            cube_depth: NonZeroU32::new(32).unwrap(),
            vector_depth: NonZeroU32::new(16).unwrap(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220Mte3Operation {
    Command(C220Mte3Command),
    Flag(FlagStep),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Mte3IssuedInstruction {
    pub instruction_id: u64,
    pub pc: u64,
    pub word: u32,
    pub accepted_tick: u64,
    pub ready_tick: u64,
    pub operation: C220Mte3Operation,
}

pub(super) struct Mte3IssueQueue {
    pub config: C220Mte3IssueQueueConfig,
    issued: VecDeque<C220Mte3IssuedInstruction>,
    last_received: Option<u64>,
    pub last_accepted: Option<u64>,
    pub barriers: VecDeque<C220Mte3Barrier>,
    pub outcomes: Vec<C220CoreStep>,
}

impl Mte3IssueQueue {
    pub(super) fn is_idle(&self) -> bool {
        self.issued.is_empty() && self.barriers.is_empty()
    }

    pub(super) fn new(config: C220Mte3IssueQueueConfig) -> Self {
        Self {
            config,
            issued: VecDeque::new(),
            last_received: None,
            last_accepted: None,
            barriers: VecDeque::new(),
            outcomes: Vec::new(),
        }
    }
}

impl C220Core {
    pub fn queued_mte3_instructions(
        &self,
    ) -> impl ExactSizeIterator<Item = &C220Mte3IssuedInstruction> {
        self.mte3_issue_queue.issued.iter()
    }

    pub fn mte3_frontend_outcomes(&self) -> &[C220CoreStep] {
        &self.mte3_issue_queue.outcomes
    }

    pub fn mte3_is_busy(&self) -> bool {
        !self.mte3_issue_queue.issued.is_empty()
            || !self.mte3.native_commands.is_empty()
            || self.mte3.pending_commands().next().is_some()
    }

    pub(super) fn mte3_stall(
        &self,
        tick: u64,
        pc: u64,
        cause: C220StallCause,
    ) -> Result<C220CoreStep, C220CoreError> {
        Ok(C220CoreStep::Stalled(C220Stall {
            tick,
            pc,
            cause,
            resume_tick: tick.checked_add(1).ok_or(C220CoreError::TimeOverflow)?,
        }))
    }

    pub(super) fn mte3_accept_blocker(&self) -> Result<Option<C220StallCause>, C220CoreError> {
        let pipeline = self
            .mte_pipeline
            .as_ref()
            .ok_or(C220CoreError::MteUnconfigured)?;
        let depth = if pipeline.core_kind() == C220CoreKind::Cube {
            self.mte3_issue_queue.config.cube_depth
        } else {
            self.mte3_issue_queue.config.vector_depth
        };
        Ok((self.mte3_issue_queue.issued.len() >= depth.get() as usize)
            .then_some(C220StallCause::Mte3IssueQueueFull))
    }

    pub(super) fn enqueue_mte3_at(
        &mut self,
        tick: u64,
        pc: u64,
        word: u32,
        operation: C220Mte3Operation,
    ) -> Result<C220CoreStep, C220CoreError> {
        if let Some(cause) = self.mte3_accept_blocker()? {
            return self.mte3_stall(tick, pc, cause);
        }
        let queued = C220Mte3IssuedInstruction {
            instruction_id: self.next_instruction_id,
            pc,
            word,
            accepted_tick: tick,
            ready_tick: tick.checked_add(1).ok_or(C220CoreError::TimeOverflow)?,
            operation,
        };
        self.mte3_issue_queue.issued.push_back(queued);
        self.mte3_issue_queue.last_accepted = Some(queued.instruction_id);
        self.update_mte3_head();
        self.state.commit_c220_sequential_issue();
        Ok(C220CoreStep::Executed {
            tick,
            instruction: C220CoreInstruction::Mte3Queued(queued),
        })
    }

    fn update_mte3_head(&mut self) {
        self.mte_pipeline
            .as_mut()
            .expect("configured MTE3 issue queue")
            .set_mte3_issue_head(self.mte3_issue_queue.issued.front().map(|c| c.ready_tick));
    }

    pub(super) fn transfer_mte3_issue_at(&mut self, tick: u64) -> Result<(), C220CoreError> {
        let queued = *self
            .mte3_issue_queue
            .issued
            .front()
            .expect("armed MTE3 reception");
        let frontend = self
            .mte_pipeline
            .as_ref()
            .expect("configured MTE3 pipeline")
            .mte3_frontend();
        let mut cause = if frontend.queued_commands() == 3 {
            Some(C220StallCause::Mte3QueueFull)
        } else if !frontend.can_issue() {
            Some(C220StallCause::Mte3OutstandingLimit)
        } else if self.mte3_issue_queue.barriers.iter().any(|barrier| {
            barrier
                .predecessor
                .is_some_and(|id| id < queued.instruction_id)
        }) {
            Some(C220StallCause::Mte3Barrier)
        } else if matches!(
            queued.operation,
            C220Mte3Operation::Command(C220Mte3Command::CrossCore { .. })
        ) && !self.mte3.native_commands.is_empty()
        {
            Some(C220StallCause::Mte3Dependency)
        } else {
            None
        };
        if cause.is_none()
            && let C220Mte3Operation::Flag(step) = queued.operation
            && step.instruction.operation == FlagOperation::Wait
            && self
                .pipeline_events
                .consume(queued.instruction_id, step, tick)
                .is_none()
        {
            cause = Some(C220StallCause::PipelineEventDependency);
        }
        let outcome = if let Some(cause) = cause {
            self.mte3_stall(tick, queued.pc, cause)?
        } else {
            let instruction = match queued.operation {
                C220Mte3Operation::Flag(step) => {
                    if step.instruction.operation == FlagOperation::Set {
                        let predecessor = (!self.mte3.native_commands.is_empty()).then(|| {
                            self.mte3_issue_queue
                                .last_received
                                .expect("outstanding MTE3 command")
                        });
                        self.pipeline_events
                            .set(queued.instruction_id, step, predecessor, tick);
                    }
                    C220CoreInstruction::Mte3Flag(step)
                }
                C220Mte3Operation::Command(command) => {
                    let pipeline = self
                        .mte_pipeline
                        .as_mut()
                        .expect("configured MTE3 pipeline");
                    let instruction = match command {
                        C220Mte3Command::L1Output(command) => C220CoreInstruction::Mte3L1Output(
                            pipeline.issue_mte3_l1_output(queued.instruction_id, command)?,
                        ),
                        C220Mte3Command::MovPad(command) => C220CoreInstruction::Mte3MovPad(
                            pipeline.issue_mte3_mov_pad(queued.instruction_id, command)?,
                        ),
                        C220Mte3Command::Dma(plan) => {
                            let record = pipeline.issue_mte3_dma(queued.instruction_id, plan)?;
                            C220CoreInstruction::Mte3Dma {
                                step: C220Mte3Step {
                                    pc: queued.pc,
                                    word: queued.word,
                                    next_pc: queued.pc.wrapping_add(4),
                                    transfer: plan,
                                },
                                record,
                            }
                        }
                        C220Mte3Command::CrossCore {
                            instruction,
                            payload,
                        } => C220CoreInstruction::Mte3CrossCore(pipeline.issue_mte3_cross_core(
                            queued.instruction_id,
                            instruction,
                            payload,
                        )?),
                    };
                    self.mte3
                        .native_commands
                        .insert(queued.instruction_id, (queued.pc, queued.word));
                    self.mte3_issue_queue.last_received = Some(queued.instruction_id);
                    instruction
                }
            };
            self.mte3_issue_queue.issued.pop_front();
            C220CoreStep::Executed { tick, instruction }
        };
        self.mte3_issue_queue.outcomes.push(outcome);
        self.release_mte3_barriers_at(tick);
        self.update_mte3_head();
        self.mte_pipeline
            .as_mut()
            .expect("armed MTE3 reception")
            .finish_mte3_issue();
        Ok(())
    }
}
