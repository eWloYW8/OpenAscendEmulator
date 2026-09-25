use crate::architecture::Architecture;
use crate::isa::c220::mte::bias::C220MovL1ToBtInstruction;
use crate::isa::c220::mte::load2d::C220Load2dInstruction;
use crate::isa::c220::mte::load2d_sparse::C220Load2dSparseInstruction;
use crate::isa::c220::mte::load2d_transpose::C220Load2dTransposeInstruction;
use crate::isa::c220::mte::set2d::C220Set2dInstruction;
use crate::isa::flow::{FlagInstruction, FlagOperation};
use crate::sim::c220::mte::mte1::C220Mte1Command;
use crate::sim::c220::mte::mte1::frontend::C220Mte1ReadTransfer;
use crate::sim::c220::mte::mte1::load2d::C220Load2dTransferError;

use crate::sim::c220::schedule::{C220Stall, C220StallCause};

use super::{C220Core, C220CoreError, C220CoreInstruction, C220CoreStep};

impl C220Core {
    /// Inspect LOAD3Dv2 inputs without issuing or advancing the core clock.
    pub fn capture_load3d_v2(
        &self,
        word: u32,
    ) -> Result<
        Option<crate::sim::c220::mte::load3d::C220Load3dV2Command>,
        crate::sim::c220::mte::load3d::C220Load3dCaptureError,
    > {
        let machine = self.state.scalar().machine();
        crate::isa::c220::mte::load3d::C220Load3dV2Instruction::decode(word)
            .map(|instruction| {
                crate::sim::c220::mte::load3d::C220Load3dV2Command::capture(
                    instruction,
                    machine.xregs(),
                    |register| machine.spr_value(register),
                )
            })
            .transpose()
    }

    pub(super) fn step_mte1_at(
        &mut self,
        tick: u64,
        pc: u64,
        word: u32,
    ) -> Result<C220CoreStep, C220CoreError> {
        let registers = self.state.scalar().machine().xregs();
        let command = if let Some(instruction) =
            crate::isa::c220::control::C220SetCrossCoreInstruction::decode(word)
        {
            if let Some(resume_tick) = self.mte1.next_event_tick() {
                return Ok(C220CoreStep::Stalled(C220Stall {
                    tick,
                    pc,
                    resume_tick: resume_tick
                        .max(tick.checked_add(1).ok_or(C220CoreError::TimeOverflow)?),
                    cause: C220StallCause::Mte1Dependency,
                }));
            }
            Some(C220Mte1Command::CrossCore {
                instruction,
                payload: crate::sim::c220::sync::C220DeviceSync::from_value(
                    registers[usize::from(instruction.source_register)],
                ),
            })
        } else if let Some(command) = self.capture_load3d_v2(word)? {
            Some(C220Mte1Command::Read(C220Mte1ReadTransfer::Load3dv2(
                command,
            )))
        } else if let Some(decoded) = C220Set2dInstruction::decode(word) {
            let pattern = self
                .state
                .scalar()
                .machine()
                .spr_value(15)
                .ok_or(C220CoreError::MissingSet2dPatternSpr)?;
            Some(C220Mte1Command::Set2d(decoded.capture(registers, pattern)))
        } else if let Some(decoded) =
            C220Load2dInstruction::decode(word).filter(|instruction| instruction.is_mte1())
        {
            Some(C220Mte1Command::Read(C220Mte1ReadTransfer::Load2d(
                decoded
                    .capture(registers)
                    .map_err(C220Load2dTransferError::from)?,
            )))
        } else if let Some(decoded) = C220Load2dSparseInstruction::decode(word) {
            Some(C220Mte1Command::Read(C220Mte1ReadTransfer::Load2dSparse(
                decoded.capture(registers),
            )))
        } else if let Some(decoded) = C220Load2dTransposeInstruction::decode(word) {
            Some(C220Mte1Command::Read(
                C220Mte1ReadTransfer::Load2dTranspose(decoded.capture(registers)),
            ))
        } else {
            C220MovL1ToBtInstruction::decode(word).map(|decoded| {
                C220Mte1Command::Read(C220Mte1ReadTransfer::Bt(decoded.capture(registers)))
            })
        };
        let instruction = if let Some(command) = command {
            let pipeline = self
                .mte_pipeline
                .as_mut()
                .ok_or(C220CoreError::MteUnconfigured)?;
            if !self.mte1.can_issue(pipeline, command) {
                return Ok(C220CoreStep::Stalled(C220Stall {
                    tick,
                    pc,
                    resume_tick: tick.checked_add(1).ok_or(C220CoreError::TimeOverflow)?,
                    cause: C220StallCause::Mte1IssueRate,
                }));
            }
            let issue = self.mte1.issue(
                pipeline,
                self.next_instruction_id,
                pc,
                command,
                &mut self.hardware_flags,
            )?;
            self.state.commit_c220_sequential_issue();
            C220CoreInstruction::Mte1 {
                instruction_id: self.next_instruction_id,
                pc,
                command,
                issue,
            }
        } else {
            let flag = FlagInstruction::decode(Architecture::Dav2201, word)
                .expect("matched C220 MTE1 flag")
                .resolve(pc, self.state.scalar().machine().xregs());
            match flag.instruction.operation {
                FlagOperation::Set => self.mte1.set_event(flag.flag_id),
                FlagOperation::Wait => {
                    if !self.mte1.wait_event(flag.flag_id) {
                        return Ok(C220CoreStep::Stalled(C220Stall {
                            tick,
                            pc,
                            resume_tick: tick.checked_add(1).ok_or(C220CoreError::TimeOverflow)?,
                            cause: C220StallCause::Mte1Dependency,
                        }));
                    }
                }
            }
            self.state.commit_c220_sequential_issue();
            C220CoreInstruction::Mte1Flag(flag)
        };
        Ok(C220CoreStep::Executed { tick, instruction })
    }
}
