use super::{C220Core, C220CoreError, C220CoreInstruction, C220CoreStep};
use crate::isa::c220::hflag::{C220HardwareFlagInstruction, C220HardwareFlagStep};
use crate::isa::c220::mte::factor::{C220FactorLoad, C220FactorLoadInstruction};
use crate::isa::c220::mte::fixp::{C220FixpDestination, C220FixpInstruction};
use crate::sim::c220::mte::fixp::{
    C220FixpCommand, C220FixpExecutionError, C220FixpExternalCommand,
};
use crate::sim::c220::schedule::{C220Stall, C220StallCause};
use std::collections::VecDeque;
use std::num::NonZeroU32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FixpFrontendConfig {
    /// Instructions accepted by the core but not yet transferred to MTE.
    pub issue_queue_depth: NonZeroU32,
    /// Commands transferred to MTE retain this credit until retirement.
    pub outstanding_limit: NonZeroU32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct FixpIssue {
    pub instruction_id: u64,
    pub pc: u64,
    pub word: u32,
}

/// Register operands belong to the issued instruction, not to the later
/// dispatch attempt. Memory contents remain live execution inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CapturedFixpOperation {
    L1(C220FixpCommand),
    Transport {
        destination: C220FixpDestination,
        command: C220FixpExternalCommand,
    },
    Factor(C220FactorLoad),
    HardwareFlag(C220HardwareFlagStep),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CapturedFixpCommand {
    pub issue: FixpIssue,
    pub operation: CapturedFixpOperation,
}

#[derive(Default)]
pub(super) struct FixpFrontend {
    pub config: Option<C220FixpFrontendConfig>,
    issued: VecDeque<(u64, CapturedFixpCommand)>,
    commands: VecDeque<(u64, CapturedFixpCommand)>,
    pub last_issued: Option<u64>,
    pub barriers: VecDeque<super::C220FixpBarrier>,
    pub outcomes: Vec<C220CoreStep>,
}

impl FixpFrontend {
    pub(super) fn contains_instruction(&self, id: u64) -> bool {
        self.issued
            .iter()
            .chain(&self.commands)
            .any(|(_, command)| command.issue.instruction_id == id)
    }
}

impl C220Core {
    pub(super) fn capture_fixp_command(
        &self,
        pc: u64,
        word: u32,
    ) -> Result<CapturedFixpCommand, C220CoreError> {
        let machine = self.state.scalar().machine();
        let operation = if let Some(instruction) = C220FactorLoadInstruction::decode(word) {
            CapturedFixpOperation::Factor(instruction.capture(machine.xregs()))
        } else if let Some(instruction) = C220HardwareFlagInstruction::decode(word) {
            CapturedFixpOperation::HardwareFlag(instruction.resolve(pc, machine.xregs())?)
        } else {
            let destination = C220FixpInstruction::decode(word)
                .ok_or(C220FixpExecutionError::Instruction(word))?
                .destination;
            let control = machine
                .spr_value(3)
                .ok_or(C220FixpExecutionError::MissingSpr(3))?;
            let local = if destination == C220FixpDestination::L1 {
                Some(C220FixpCommand::capture_l1(
                    word,
                    control,
                    |register| machine.xregs().get(usize::from(register)).copied(),
                    |register| machine.spr_value(u16::from(register)),
                )?)
            } else {
                None
            };
            match local {
                Some(command) if !command.descriptor.nz_to_nd() || self.external_fixp.is_none() => {
                    CapturedFixpOperation::L1(command)
                }
                _ => CapturedFixpOperation::Transport {
                    destination,
                    command: C220FixpExternalCommand::capture_destination(
                        word,
                        destination,
                        control,
                        self.state.isa_instance_index,
                        |register| machine.xregs().get(usize::from(register)).copied(),
                        |register| machine.spr_value(u16::from(register)),
                    )?,
                },
            }
        };
        Ok(CapturedFixpCommand {
            issue: FixpIssue {
                instruction_id: self.next_instruction_id,
                pc,
                word,
            },
            operation,
        })
    }

    pub(super) fn step_fixp_at(
        &mut self,
        tick: u64,
        pc: u64,
        word: u32,
    ) -> Result<C220CoreStep, C220CoreError> {
        let config = self
            .fixp_frontend
            .config
            .ok_or(C220CoreError::FixpUnconfigured)?;
        if self.fixp_frontend.issued.len() >= config.issue_queue_depth.get() as usize {
            return Ok(C220CoreStep::Stalled(C220Stall {
                tick,
                pc,
                resume_tick: tick.checked_add(1).ok_or(C220CoreError::TimeOverflow)?,
                cause: C220StallCause::FixpIssueQueueFull,
            }));
        }
        if self.mte_pipeline.is_none() {
            return Err(C220CoreError::MteUnconfigured);
        }
        if self.fixp_engine().is_none() {
            return Err(C220CoreError::FixpUnconfigured);
        }
        let command = self.capture_fixp_command(pc, word)?;
        let ready_tick = tick.checked_add(1).ok_or(C220CoreError::TimeOverflow)?;
        self.fixp_frontend.issued.push_back((ready_tick, command));
        self.fixp_frontend.last_issued = Some(command.issue.instruction_id);
        self.update_fixp_command_head();
        self.state.commit_c220_sequential_issue();
        Ok(C220CoreStep::Executed {
            tick,
            instruction: C220CoreInstruction::FixpQueued {
                instruction_id: command.issue.instruction_id,
                pc,
                word,
                ready_tick,
            },
        })
    }

    /// Transfer, dispatch and backpressure observed during the last advance.
    pub fn fixp_frontend_outcomes(&self) -> &[C220CoreStep] {
        &self.fixp_frontend.outcomes
    }

    pub fn queued_fixp_commands(&self) -> usize {
        self.fixp_frontend.issued.len() + self.fixp_frontend.commands.len()
    }

    pub fn fixp_issue_queue_len(&self) -> usize {
        self.fixp_frontend.issued.len()
    }

    pub fn fixp_command_queue_len(&self) -> usize {
        self.fixp_frontend.commands.len()
    }

    pub fn outstanding_fixp_commands(&self) -> usize {
        self.fixp_frontend.commands.len()
            + self
                .fixp_engine()
                .map_or(0, |engine| engine.command_retirement_queue().len())
    }

    pub(super) fn transfer_fixp_issue_at(&mut self, tick: u64) -> Result<(), C220CoreError> {
        self.release_fixp_barriers_at(tick);
        let config = self
            .fixp_frontend
            .config
            .ok_or(C220CoreError::FixpUnconfigured)?;
        let blocked = if self.fixp_frontend.commands.len() == 3 {
            Some(C220StallCause::FixpCommandQueueFull)
        } else if self.outstanding_fixp_commands() >= config.outstanding_limit.get() as usize {
            Some(C220StallCause::FixpOutstandingLimit)
        } else if self.fixp_frontend.barriers.front().is_some_and(|barrier| {
            self.fixp_frontend
                .issued
                .front()
                .is_some_and(|(_, command)| {
                    barrier
                        .predecessor
                        .is_some_and(|id| id < command.issue.instruction_id)
                })
        }) {
            Some(C220StallCause::FixpBarrier)
        } else {
            None
        };
        if let Some(cause) = blocked {
            let command = self
                .fixp_frontend
                .issued
                .front()
                .expect("armed FIX issue")
                .1;
            self.fixp_frontend
                .outcomes
                .push(C220CoreStep::Stalled(C220Stall {
                    tick,
                    pc: command.issue.pc,
                    resume_tick: tick.checked_add(1).ok_or(C220CoreError::TimeOverflow)?,
                    cause,
                }));
        } else {
            let ready_tick = tick.checked_add(3).ok_or(C220CoreError::TimeOverflow)?;
            let (_, command) = self
                .fixp_frontend
                .issued
                .pop_front()
                .expect("armed FIX issue");
            self.fixp_frontend.commands.push_back((ready_tick, command));
            self.fixp_frontend.outcomes.push(C220CoreStep::Executed {
                tick,
                instruction: C220CoreInstruction::FixpScheduled {
                    instruction_id: command.issue.instruction_id,
                    ready_tick,
                },
            });
        }
        self.update_fixp_command_head();
        self.mte_pipeline
            .as_mut()
            .expect("armed FIX issue")
            .finish_fixp_issue();
        Ok(())
    }

    fn update_fixp_command_head(&mut self) {
        let head = self.fixp_frontend.commands.front();
        if let Some(pipeline) = &mut self.mte_pipeline {
            pipeline
                .set_fixp_issue_head(self.fixp_frontend.issued.front().map(|(ready, _)| *ready));
            pipeline.set_fixp_command_head(
                head.map(|(ready, _)| *ready),
                head.is_some_and(|(_, command)| {
                    matches!(
                        command.operation,
                        CapturedFixpOperation::L1(_) | CapturedFixpOperation::Transport { .. }
                    )
                }),
            );
        }
    }

    pub(super) fn dispatch_fixp_head_at(&mut self, tick: u64) -> Result<(), C220CoreError> {
        let (_, command) = *self
            .fixp_frontend
            .commands
            .front()
            .expect("armed FIX dispatch");
        let result = self.dispatch_captured_fixp_at(tick, command)?;
        if matches!(result, C220CoreStep::Executed { .. }) {
            self.fixp_frontend.commands.pop_front();
        }
        self.fixp_frontend.outcomes.push(result);
        self.update_fixp_command_head();
        self.mte_pipeline
            .as_mut()
            .expect("armed FIX dispatch")
            .finish_fixp_dispatch();
        Ok(())
    }

    /// Dispatch does not advance the scalar PC or allocate another identity.
    pub(super) fn dispatch_captured_fixp_at(
        &mut self,
        tick: u64,
        command: CapturedFixpCommand,
    ) -> Result<C220CoreStep, C220CoreError> {
        let issue = command.issue;
        match command.operation {
            CapturedFixpOperation::L1(command) => self.dispatch_fixp_l1_at(tick, issue, command),
            CapturedFixpOperation::Transport {
                destination,
                command,
            } => self.dispatch_external_fixp_at(tick, issue, destination, command),
            CapturedFixpOperation::Factor(load) => self.dispatch_factor_at(tick, issue, load),
            CapturedFixpOperation::HardwareFlag(step) => {
                let result = self.dispatch_hardware_flag_at(tick, issue.instruction_id, step)?;
                if matches!(result, C220CoreStep::Executed { .. }) {
                    self.fixp_engine_mut()
                        .ok_or(C220CoreError::FixpUnconfigured)?
                        .admit_control(tick, issue.instruction_id)?;
                }
                Ok(result)
            }
        }
    }
}
