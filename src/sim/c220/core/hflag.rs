use crate::isa::c220::hflag::{
    C220HardwareFlagOperation, C220HardwareFlagSourcePipe, C220MatrixMemory,
};
use crate::sim::c220::sync::{C220HardwareFlagEvent, C220HardwareFlagTimingError};

use crate::sim::c220::schedule::{C220Stall, C220StallCause};

use super::{C220Core, C220CoreError, C220CoreInstruction, C220CoreStep};

impl C220Core {
    pub(super) fn dispatch_hardware_flag_at(
        &mut self,
        tick: u64,
        instruction_id: u64,
        step: crate::isa::c220::hflag::C220HardwareFlagStep,
    ) -> Result<C220CoreStep, C220CoreError> {
        let instruction = step.instruction;
        let pc = step.pc;
        let trigger_blocked = match instruction.source_pipe {
            C220HardwareFlagSourcePipe::Mte1 => self
                .mte_pipeline
                .as_ref()
                .is_some_and(|p| !p.selected_generator_idle()),
            C220HardwareFlagSourcePipe::Fix => self
                .fixp_engine()
                .is_some_and(|engine| !engine.hardware_flag_trigger_ready()),
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
                if instruction.trigger {
                    let ready_tick = self.hardware_flags.wait_ready_tick(step)?.map_or_else(
                        || {
                            tick.checked_add(1)
                                .ok_or(C220HardwareFlagTimingError::TimeOverflow)
                        },
                        Ok,
                    )?;
                    if tick < ready_tick {
                        return Ok(C220CoreStep::Stalled(C220Stall {
                            tick,
                            pc,
                            resume_tick: ready_tick,
                            cause: C220StallCause::HardwareFlagDependency,
                        }));
                    }
                    self.hardware_flags.consume_wait(step)?;
                } else if instruction.source_pipe == C220HardwareFlagSourcePipe::Fix {
                    if instruction.memory != C220MatrixMemory::L0c {
                        return Err(C220CoreError::UnsupportedHardwareFlagCheckpoint {
                            source_pipe: instruction.source_pipe,
                            memory: instruction.memory,
                        });
                    }
                    self.hardware_flags
                        .enqueue_mte_flag(instruction_id, step, tick)?;
                } else {
                    self.hardware_flags
                        .enqueue_cube_wait(instruction_id, step)?;
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
