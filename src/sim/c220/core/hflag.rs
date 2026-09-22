use crate::isa::c220::hflag::{
    C220HardwareFlagInstruction, C220HardwareFlagOperation, C220HardwareFlagSourcePipe,
    C220MatrixMemory,
};
use crate::isa::c220::mte::load2d::C220Load2dDestination;
use crate::sim::c220::sync::C220HardwareFlagTimingError;

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
        let token_ready_tick = match instruction.operation {
            C220HardwareFlagOperation::Set => {
                let checkpoint_tick = if instruction.trigger {
                    tick
                } else {
                    match (instruction.source_pipe, instruction.memory) {
                        (C220HardwareFlagSourcePipe::Mte1, C220MatrixMemory::L0a) => self
                            .mte1
                            .timing
                            .pending_data_ready_tick(C220Load2dDestination::L0a)
                            .unwrap_or(tick),
                        (C220HardwareFlagSourcePipe::Mte1, C220MatrixMemory::L0b) => self
                            .mte1
                            .timing
                            .pending_data_ready_tick(C220Load2dDestination::L0b)
                            .unwrap_or(tick),
                        (source_pipe, memory) => {
                            return Err(C220CoreError::UnsupportedHardwareFlagCheckpoint {
                                source_pipe,
                                memory,
                            });
                        }
                    }
                };
                Some(
                    self.hardware_flags
                        .schedule_set(step, checkpoint_tick.max(tick))?,
                )
            }
            C220HardwareFlagOperation::Wait => {
                if instruction.trigger {
                    let ready_tick = self.hardware_flags.wait_ready_tick(step)?.ok_or(
                        C220HardwareFlagTimingError::MissingToken {
                            event_id: step.event_id,
                        },
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
