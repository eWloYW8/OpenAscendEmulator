use crate::isa::c220::hflag::{
    C220HardwareFlagInstruction, C220HardwareFlagOperation, C220HardwareFlagSourcePipe,
    C220MatrixMemory,
};
use crate::sim::c220::sync::{C220HardwareFlagEvent, C220HardwareFlagTimingError};

use crate::sim::c220::schedule::{C220Stall, C220StallCause};

use super::{C220Core, C220CoreError, C220CoreInstruction, C220CoreStep};

impl C220Core {
    pub(super) fn step_hardware_flag_at(
        &mut self,
        tick: u64,
        pc: u64,
        instruction: C220HardwareFlagInstruction,
    ) -> Result<C220CoreStep, C220CoreError> {
        let step = instruction.resolve(pc, self.state.scalar().machine().xregs())?;
        if instruction.trigger
            && instruction.source_pipe == C220HardwareFlagSourcePipe::Mte1
            && self
                .mte_pipeline
                .as_ref()
                .is_some_and(|p| !p.selected_generator_idle())
        {
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
                        ) => {
                            self.hardware_flags.enqueue_mte_set(
                                self.next_instruction_id,
                                step,
                                tick,
                            )?;
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
                } else {
                    self.hardware_flags
                        .enqueue_cube_wait(self.next_instruction_id, step)?;
                }
                None
            }
        };
        self.state.commit_c220_sequential_issue();
        Ok(C220CoreStep::Executed {
            tick,
            instruction: C220CoreInstruction::HardwareFlag {
                instruction_id: self.next_instruction_id,
                step,
                token_ready_tick,
            },
        })
    }
}
