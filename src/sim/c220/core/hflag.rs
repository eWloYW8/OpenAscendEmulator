use crate::isa::c220::hflag::{
    C220HardwareFlagOperation, C220HardwareFlagSourcePipe, C220MatrixMemory,
};
use crate::sim::c220::sync::{C220HardwareFlagEvent, C220HardwareFlagTimingError};

use crate::sim::c220::schedule::{C220Stall, C220StallCause};

use super::{C220Core, C220CoreError, C220CoreInstruction, C220CoreStep};

impl C220Core {
    pub(super) fn step_cube_hardware_flag_at(
        &mut self,
        tick: u64,
        pc: u64,
        instruction: crate::isa::c220::hflag::C220HardwareFlagInstruction,
    ) -> Result<C220CoreStep, C220CoreError> {
        let step = instruction.resolve(pc, self.state.scalar().machine().xregs())?;
        self.enqueue_cube_at(
            tick,
            pc,
            instruction.word,
            crate::sim::c220::cube::frontend::C220CubeCommand::HardwareFlag(step),
        )
    }

    pub(super) fn dispatch_cube_hardware_flag_at(
        &mut self,
        tick: u64,
        instruction_id: u64,
        step: crate::isa::c220::hflag::C220HardwareFlagStep,
    ) -> Result<C220CoreStep, C220CoreError> {
        let instruction = step.instruction;
        let pc = step.pc;
        let result = if instruction.operation == C220HardwareFlagOperation::Set {
            if self.hardware_flags.has_pending_cube_flag(step) {
                return Ok(C220CoreStep::Stalled(C220Stall {
                    tick,
                    pc,
                    resume_tick: tick.checked_add(1).ok_or(C220CoreError::TimeOverflow)?,
                    cause: C220StallCause::HardwareFlagDependency,
                }));
            }
            let ready = if !instruction.trigger {
                self.hardware_flags.enqueue_cube_set(instruction_id, step)?;
                None
            } else {
                match self.hardware_flags.schedule_set(step, tick) {
                    Ok(ready) => Some(ready),
                    Err(C220HardwareFlagTimingError::AlmostFull { .. }) => {
                        return Ok(C220CoreStep::Stalled(C220Stall {
                            tick,
                            pc,
                            resume_tick: tick.checked_add(1).ok_or(C220CoreError::TimeOverflow)?,
                            cause: C220StallCause::HardwareFlagDependency,
                        }));
                    }
                    Err(error) => return Err(error.into()),
                }
            };
            C220CoreStep::Executed {
                tick,
                instruction: C220CoreInstruction::HardwareFlag {
                    instruction_id,
                    step,
                    token_ready_tick: ready,
                },
            }
        } else {
            self.dispatch_cube_hardware_wait_at(tick, instruction_id, step)?
        };
        Ok(result)
    }

    fn dispatch_cube_hardware_wait_at(
        &mut self,
        tick: u64,
        instruction_id: u64,
        step: crate::isa::c220::hflag::C220HardwareFlagStep,
    ) -> Result<C220CoreStep, C220CoreError> {
        self.hardware_flags.advance_to(tick)?;
        let retry = tick.checked_add(1).ok_or(C220CoreError::TimeOverflow)?;
        let blocked = if self.hardware_flags.has_pending_cube_flag(step) {
            Some(retry)
        } else if step.instruction.trigger {
            match self.hardware_flags.wait_ready_tick(step)? {
                Some(ready) if ready <= tick => None,
                Some(ready) => Some(ready),
                None => Some(retry),
            }
        } else {
            None
        };
        if let Some(resume_tick) = blocked {
            return Ok(C220CoreStep::Stalled(C220Stall {
                tick,
                pc: step.pc,
                resume_tick,
                cause: C220StallCause::HardwareFlagDependency,
            }));
        }
        if step.instruction.trigger {
            self.hardware_flags.consume_wait(step)?;
        } else {
            self.hardware_flags
                .enqueue_cube_wait(instruction_id, step)?;
        }
        Ok(C220CoreStep::Executed {
            tick,
            instruction: C220CoreInstruction::HardwareFlag {
                instruction_id,
                step,
                token_ready_tick: None,
            },
        })
    }

    pub(super) fn dispatch_hardware_flag_at(
        &mut self,
        tick: u64,
        instruction_id: u64,
        step: crate::isa::c220::hflag::C220HardwareFlagStep,
    ) -> Result<C220CoreStep, C220CoreError> {
        let instruction = step.instruction;
        let pc = step.pc;
        if instruction.execution_pipe_code() == 2 {
            return self.dispatch_cube_hardware_flag_at(tick, instruction_id, step);
        }
        let trigger_blocked = match instruction.execution_pipe_code() {
            3 => self
                .mte_pipeline
                .as_ref()
                .is_some_and(|p| !p.selected_generator_idle()),
            10 => self
                .fixp_engine()
                .is_some_and(|engine| !engine.hardware_flag_trigger_ready()),
            _ => unreachable!("MTE hardware flag dispatch requires MTE1 or FIX"),
        };
        if instruction.trigger && trigger_blocked {
            return Ok(C220CoreStep::Stalled(C220Stall {
                tick,
                pc,
                resume_tick: tick.checked_add(1).ok_or(C220CoreError::TimeOverflow)?,
                cause: C220StallCause::HardwareFlagDependency,
            }));
        }
        let token_ready_tick = match instruction.operation {
            C220HardwareFlagOperation::Set => {
                if !instruction.trigger {
                    match (instruction.source_pipe, instruction.memory) {
                        (
                            C220HardwareFlagSourcePipe::Mte1,
                            C220MatrixMemory::L0a
                            | C220MatrixMemory::L0b
                            | C220MatrixMemory::BiasTable,
                        )
                        | (C220HardwareFlagSourcePipe::Fix, C220MatrixMemory::L0c) => {
                            self.hardware_flags
                                .enqueue_mte_flag(instruction_id, step, tick)?;
                        }
                        (source_pipe, memory) => {
                            return Err(C220CoreError::UnsupportedHardwareFlagCheckpoint {
                                source_pipe,
                                memory,
                            });
                        }
                    }
                    None
                } else {
                    match self
                        .hardware_flags
                        .schedule_event(C220HardwareFlagEvent::capture_mte(step, tick), tick)
                    {
                        Ok(ready_tick) => Some(ready_tick),
                        Err(C220HardwareFlagTimingError::AlmostFull { .. }) => {
                            return Ok(C220CoreStep::Stalled(C220Stall {
                                tick,
                                pc,
                                resume_tick: tick
                                    .checked_add(1)
                                    .ok_or(C220HardwareFlagTimingError::TimeOverflow)?,
                                cause: C220StallCause::HardwareFlagDependency,
                            }));
                        }
                        Err(error) => return Err(error.into()),
                    }
                }
            }
            C220HardwareFlagOperation::Wait => {
                if !instruction.trigger {
                    if !matches!(instruction.destination_pipe_code, 3 | 10) {
                        return Err(C220CoreError::UnsupportedHardwareFlagCheckpoint {
                            source_pipe: instruction.source_pipe,
                            memory: instruction.memory,
                        });
                    }
                    self.hardware_flags
                        .enqueue_mte_flag(instruction_id, step, tick)?;
                } else {
                    match self.hardware_flags.consume_wait(step) {
                        Ok(()) => {}
                        Err(C220HardwareFlagTimingError::MissingToken { .. }) => {
                            return Ok(C220CoreStep::Stalled(C220Stall {
                                tick,
                                pc,
                                resume_tick: tick
                                    .checked_add(1)
                                    .ok_or(C220CoreError::TimeOverflow)?,
                                cause: C220StallCause::HardwareFlagDependency,
                            }));
                        }
                        Err(error) => return Err(error.into()),
                    }
                }
                None
            }
        };
        Ok(C220CoreStep::Executed {
            tick,
            instruction: C220CoreInstruction::HardwareFlag {
                instruction_id,
                step,
                token_ready_tick,
            },
        })
    }
}
