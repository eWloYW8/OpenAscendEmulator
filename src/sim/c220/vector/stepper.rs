use super::*;
use crate::architecture::Architecture;
use crate::isa::c220::gather::{C220GatherInstruction, C220GatherKind};
use crate::isa::c220::reduce::C220ReductionInstruction;
use crate::isa::c220::ternary::C220TernaryInstruction;
use crate::isa::c220::vector::{
    C220BroadcastInstruction, C220CopyInstruction, C220MovevInstruction, C220ShiftInstruction,
    C220TransposeInstruction, C220VecArithmeticHint, C220VecArithmeticOperation,
};
use crate::isa::c220::vector_scalar::C220VectorScalarInstruction;
use crate::isa::flow::FlagInstruction;
use crate::memory::sparse::MemoryByteState;
use crate::sim::c220::fp16::C220Fp16Mode;
use crate::sim::c220::mte::state::C220OutputToken;
use crate::sim::c220::vector::broadcast::{C220BroadcastIssue, plan_c220_broadcast_issue};
use crate::sim::c220::vector::copy::{C220CopyIssue, plan_c220_copy_issue};
use crate::sim::c220::vector::gather::{C220GatherIssue, plan_c220_gather_issue};
use crate::sim::c220::vector::reduce::{C220ReductionIssue, plan_c220_reduction_issue};
use crate::sim::c220::vector::scalar::{
    C220VectorScalarIssue, C220VectorScalarOperand, plan_c220_vector_scalar_issue,
};
use crate::sim::c220::vector::shift::{C220ShiftIssue, plan_c220_shift_issue};
use crate::sim::c220::vector::ternary::{C220TernaryIssue, plan_c220_ternary_issue};
use crate::sim::c220::vector::transpose::{C220TransposeIssue, plan_c220_transpose_issue};
use crate::sim::c220::vector::{
    C220_VECTOR_TILE_BYTES, C220Fp32Step, C220MovevStep, C220VectorAddresses,
    C220VectorArithmeticIssue, C220VectorArithmeticModes, C220VectorControl, C220VectorStore,
    decode_c220_fp32_control, decode_c220_movev_control, decode_c220_repeat_masks,
    decode_c220_vector_unary_control, execute_c220_fp32_to_ub, plan_c220_movev_to_ub,
    plan_c220_vector_arithmetic_issue, write_c220_movev_to_ub,
};
use crate::sim::mte_stepper::{MteCoreStepper, MteStepperError};

struct ResolvedC220VectorArithmetic {
    pc: u64,
    control: C220VectorControl,
    addresses: C220VectorAddresses,
    iteration_masks: Vec<[u64; 4]>,
}

struct ResolvedC220UnaryVector {
    pc: u64,
    mask_control: u64,
    control: C220VectorControl,
    addresses: C220VectorAddresses,
    iteration_masks: Vec<[u64; 4]>,
}

impl MteCoreStepper {
    pub(crate) fn preview_c220_gather_word(
        &self,
        word: u32,
    ) -> Result<C220GatherIssue, MteStepperError> {
        let pc = self.scalar.pc();
        if self.scalar.is_halted() {
            return Err(MteStepperError::ProgramEnded { pc });
        }
        if self.scalar.machine().architecture() != Architecture::Dav2201 {
            return Err(MteStepperError::UnsupportedWord { pc, word });
        }
        if self.c220.output_buffer_busy() {
            return Err(MteStepperError::OutputDependencyOutstanding);
        }
        let instruction = C220GatherInstruction::decode(word)
            .ok_or(MteStepperError::UnsupportedWord { pc, word })?;
        let machine = self.scalar.machine();
        let registers = machine.xregs();
        let control_value = registers[usize::from(instruction.control_register)];
        let iteration_masks = match instruction.kind {
            C220GatherKind::Elements(width) => decode_c220_repeat_masks(
                machine
                    .spr_value(3)
                    .ok_or(C220VectorError::MissingMaskState)?,
                machine
                    .spr_value(100)
                    .ok_or(C220VectorError::MissingMaskState)?,
                machine
                    .spr_value(101)
                    .ok_or(C220VectorError::MissingMaskState)?,
                width.lane_count(),
                (control_value >> 56) as u8,
            )?,
            C220GatherKind::Blocks => Vec::new(),
        };
        Ok(plan_c220_gather_issue(
            pc,
            word,
            control_value,
            registers[usize::from(instruction.destination_register)],
            registers[usize::from(instruction.index_register)],
            iteration_masks,
            &self.ub,
        )?)
    }

    pub(crate) fn preview_c220_ternary_word(
        &self,
        word: u32,
    ) -> Result<C220TernaryIssue, MteStepperError> {
        let pc = self.scalar.pc();
        if self.scalar.is_halted() {
            return Err(MteStepperError::ProgramEnded { pc });
        }
        if self.scalar.machine().architecture() != Architecture::Dav2201 {
            return Err(MteStepperError::UnsupportedWord { pc, word });
        }
        if self.c220.output_buffer_busy() {
            return Err(MteStepperError::OutputDependencyOutstanding);
        }
        let instruction = C220TernaryInstruction::decode(word)
            .ok_or(MteStepperError::UnsupportedWord { pc, word })?;
        let machine = self.scalar.machine();
        let registers = machine.xregs();
        let control =
            decode_c220_fp32_control(registers[usize::from(instruction.control_register)])?;
        let mask_control = machine
            .spr_value(3)
            .ok_or(C220VectorError::MissingMaskState)?;
        let iteration_masks = decode_c220_repeat_masks(
            mask_control,
            machine
                .spr_value(100)
                .ok_or(C220VectorError::MissingMaskState)?,
            machine
                .spr_value(101)
                .ok_or(C220VectorError::MissingMaskState)?,
            instruction.width.lane_count(),
            control.encoded_repeat_count,
        )?;
        Ok(plan_c220_ternary_issue(
            pc,
            word,
            control,
            C220VectorAddresses {
                source_0: registers[usize::from(instruction.source_0_register)],
                source_1: registers[usize::from(instruction.source_1_register)],
                destination: registers[usize::from(instruction.destination_register)],
            },
            &iteration_masks,
            C220Fp16Mode::from_control_spr(mask_control),
            &self.ub,
        )?)
    }

    pub(crate) fn preview_c220_reduction_word(
        &self,
        word: u32,
    ) -> Result<C220ReductionIssue, MteStepperError> {
        let instruction =
            C220ReductionInstruction::decode(word).ok_or(MteStepperError::UnsupportedWord {
                pc: self.scalar.pc(),
                word,
            })?;
        let resolved = self.resolve_c220_unary_vector(
            word,
            instruction.destination_register,
            instruction.source_register,
            instruction.control_register,
            instruction.width.element_bytes(),
            true,
        )?;
        Ok(plan_c220_reduction_issue(
            resolved.pc,
            word,
            resolved.control,
            resolved.addresses,
            &resolved.iteration_masks,
            C220Fp16Mode::from_control_spr(resolved.mask_control),
            &self.ub,
        )?)
    }

    pub(crate) fn preview_c220_transpose_word(
        &self,
        word: u32,
    ) -> Result<C220TransposeIssue, MteStepperError> {
        let pc = self.scalar.pc();
        if self.scalar.is_halted() {
            return Err(MteStepperError::ProgramEnded { pc });
        }
        if self.scalar.machine().architecture() != Architecture::Dav2201 {
            return Err(MteStepperError::UnsupportedWord { pc, word });
        }
        let instruction = C220TransposeInstruction::decode(word)
            .ok_or(MteStepperError::UnsupportedWord { pc, word })?;
        if self.c220.output_buffer_busy() {
            return Err(MteStepperError::OutputDependencyOutstanding);
        }
        let xregs = self.scalar.machine().xregs();
        Ok(plan_c220_transpose_issue(
            pc,
            word,
            xregs[usize::from(instruction.source_register)],
            xregs[usize::from(instruction.destination_register)],
            &self.ub,
        )?)
    }

    pub(crate) fn preview_c220_broadcast_word(
        &self,
        word: u32,
    ) -> Result<C220BroadcastIssue, MteStepperError> {
        let pc = self.scalar.pc();
        if self.scalar.is_halted() {
            return Err(MteStepperError::ProgramEnded { pc });
        }
        if self.scalar.machine().architecture() != Architecture::Dav2201 {
            return Err(MteStepperError::UnsupportedWord { pc, word });
        }
        let instruction = C220BroadcastInstruction::decode(word)
            .ok_or(MteStepperError::UnsupportedWord { pc, word })?;
        if self.c220.output_buffer_busy() {
            return Err(MteStepperError::OutputDependencyOutstanding);
        }
        let xregs = self.scalar.machine().xregs();
        Ok(plan_c220_broadcast_issue(
            pc,
            word,
            xregs[usize::from(instruction.control_register)],
            xregs[usize::from(instruction.source_register)],
            xregs[usize::from(instruction.destination_register)],
            &self.ub,
        )?)
    }

    pub fn step_c220_movev_word(&mut self, word: u32) -> Result<C220MovevStep, MteStepperError> {
        let step = self.preview_c220_movev_word(word)?;
        write_c220_movev_to_ub(&step, &mut self.ub)?;
        self.commit_c220_vector_issue(None);
        Ok(step)
    }

    pub(crate) fn preview_c220_movev_word(
        &self,
        word: u32,
    ) -> Result<C220MovevStep, MteStepperError> {
        let pc = self.scalar.pc();
        if self.scalar.is_halted() {
            return Err(MteStepperError::ProgramEnded { pc });
        }
        if self.scalar.machine().architecture() != Architecture::Dav2201 {
            return Err(MteStepperError::UnsupportedWord { pc, word });
        }
        let instruction = C220MovevInstruction::decode(word)
            .ok_or(MteStepperError::UnsupportedWord { pc, word })?;
        let element_bytes = instruction
            .supported_element_bytes()
            .ok_or(MteStepperError::UnsupportedWord { pc, word })?;
        if self.c220.output_buffer_busy() {
            return Err(MteStepperError::OutputDependencyOutstanding);
        }
        let machine = self.scalar.machine();
        let xregs = machine.xregs();
        let control = xregs[usize::from(instruction.control_register)];
        let control = decode_c220_movev_control(control)?;
        let mask_control = machine
            .spr_value(3)
            .ok_or(C220VectorError::MissingMaskState)?;
        let mask0 = machine
            .spr_value(100)
            .ok_or(C220VectorError::MissingMaskState)?;
        let mask1 = machine
            .spr_value(101)
            .ok_or(C220VectorError::MissingMaskState)?;
        let lane_count = C220_VECTOR_TILE_BYTES / usize::from(element_bytes);
        let iteration_masks = decode_c220_repeat_masks(
            mask_control,
            mask0,
            mask1,
            lane_count,
            control.encoded_repeat_count,
        )?;
        let destination_address = xregs[usize::from(instruction.destination_register)];
        let scalar_word = xregs[usize::from(instruction.source_register)] as u32;
        Ok(plan_c220_movev_to_ub(
            pc,
            word,
            control,
            destination_address,
            scalar_word,
            &iteration_masks,
            &self.ub,
        )?)
    }

    pub(crate) fn resolve_c220_vector_flag_id(
        &self,
        pc: u64,
        word: u32,
    ) -> Result<u8, MteStepperError> {
        let instruction = FlagInstruction::decode(Architecture::Dav2201, word)
            .filter(|instruction| {
                instruction.source_pipe_code == 4 && instruction.trigger_pipe_code == 1
            })
            .ok_or(MteStepperError::UnsupportedWord { pc, word })?;
        let flag_id = instruction
            .resolve(pc, self.scalar.machine().xregs())
            .flag_id;
        u8::try_from(flag_id)
            .ok()
            .filter(|id| *id < 2)
            .ok_or(MteStepperError::UnsupportedVectorFlagId { flag_id })
    }

    pub fn step_c220_vadd_word(&mut self, word: u32) -> Result<C220Fp32Step, MteStepperError> {
        self.step_c220_fp32_word(word, Some(C220VecArithmeticOperation::Add))
    }

    pub fn step_c220_vsub_word(&mut self, word: u32) -> Result<C220Fp32Step, MteStepperError> {
        self.step_c220_fp32_word(word, Some(C220VecArithmeticOperation::Subtract))
    }

    pub fn step_c220_vmul_word(&mut self, word: u32) -> Result<C220Fp32Step, MteStepperError> {
        self.step_c220_fp32_word(word, Some(C220VecArithmeticOperation::Multiply))
    }

    pub fn step_c220_vdiv_word(&mut self, word: u32) -> Result<C220Fp32Step, MteStepperError> {
        self.step_c220_fp32_word(word, Some(C220VecArithmeticOperation::Divide))
    }

    pub fn step_c220_vmax_word(&mut self, word: u32) -> Result<C220Fp32Step, MteStepperError> {
        self.step_c220_fp32_word(word, Some(C220VecArithmeticOperation::Maximum))
    }

    pub fn step_c220_vmin_word(&mut self, word: u32) -> Result<C220Fp32Step, MteStepperError> {
        self.step_c220_fp32_word(word, Some(C220VecArithmeticOperation::Minimum))
    }

    pub fn step_c220_vabs_word(&mut self, word: u32) -> Result<C220Fp32Step, MteStepperError> {
        self.step_c220_fp32_word(word, Some(C220VecArithmeticOperation::Absolute))
    }

    pub fn step_c220_vrelu_word(&mut self, word: u32) -> Result<C220Fp32Step, MteStepperError> {
        self.step_c220_fp32_word(word, Some(C220VecArithmeticOperation::Rectify))
    }

    pub fn step_c220_vector_word(&mut self, word: u32) -> Result<C220Fp32Step, MteStepperError> {
        self.step_c220_fp32_word(word, None)
    }

    pub(crate) fn preview_c220_vector_word(
        &self,
        word: u32,
    ) -> Result<C220VectorArithmeticIssue, MteStepperError> {
        let resolved = self.resolve_c220_vector_arithmetic(word, None, true)?;
        let control_spr = self
            .scalar
            .machine()
            .spr_value(3)
            .ok_or(C220VectorError::MissingMaskState)?;
        let issue = plan_c220_vector_arithmetic_issue(
            resolved.pc,
            word,
            resolved.control,
            resolved.addresses,
            &resolved.iteration_masks,
            C220VectorArithmeticModes::from_control_spr(control_spr),
            &self.ub,
        )?;
        Ok(issue)
    }

    pub(crate) fn preview_c220_vector_scalar_word(
        &self,
        word: u32,
    ) -> Result<C220VectorScalarIssue, MteStepperError> {
        let instruction =
            C220VectorScalarInstruction::decode(word).ok_or(MteStepperError::UnsupportedWord {
                pc: self.scalar.pc(),
                word,
            })?;
        let resolved = self.resolve_c220_unary_vector(
            word,
            instruction.destination_register,
            instruction.source_register,
            instruction.control_register,
            instruction.dtype.element_bytes(),
            true,
        )?;
        let issue = plan_c220_vector_scalar_issue(
            resolved.pc,
            word,
            C220VectorScalarOperand {
                bits: self.scalar.machine().xregs()[usize::from(instruction.scalar_register)]
                    as u32,
                fp16_mode: C220Fp16Mode::from_control_spr(resolved.mask_control),
                integer_saturating: resolved.mask_control & (1 << 53) != 0,
            },
            resolved.control,
            resolved.addresses,
            &resolved.iteration_masks,
            &self.ub,
        )?;
        Ok(issue)
    }

    pub(crate) fn preview_c220_shift_word(
        &self,
        word: u32,
    ) -> Result<C220ShiftIssue, MteStepperError> {
        let instruction =
            C220ShiftInstruction::decode(word).ok_or(MteStepperError::UnsupportedWord {
                pc: self.scalar.pc(),
                word,
            })?;
        let resolved = self.resolve_c220_unary_vector(
            word,
            instruction.destination_register,
            instruction.source_register,
            instruction.control_register,
            instruction.element_bytes,
            true,
        )?;
        Ok(plan_c220_shift_issue(
            resolved.pc,
            word,
            self.scalar.machine().xregs()[usize::from(instruction.shift_register)] as u32,
            resolved.control,
            resolved.addresses,
            &resolved.iteration_masks,
            &self.ub,
        )?)
    }

    pub(crate) fn preview_c220_copy_word(
        &self,
        word: u32,
    ) -> Result<C220CopyIssue, MteStepperError> {
        let instruction =
            C220CopyInstruction::decode(word).ok_or(MteStepperError::UnsupportedWord {
                pc: self.scalar.pc(),
                word,
            })?;
        let resolved = self.resolve_c220_unary_vector(
            word,
            instruction.destination_register,
            instruction.source_register,
            instruction.control_register,
            instruction.element_bytes,
            false,
        )?;
        Ok(plan_c220_copy_issue(
            resolved.pc,
            word,
            resolved.control,
            resolved.addresses,
            &resolved.iteration_masks,
            &self.ub,
        )?)
    }

    fn resolve_c220_unary_vector(
        &self,
        word: u32,
        destination_register: u8,
        source_register: u8,
        control_register: u8,
        element_bytes: u8,
        supports_count_mask: bool,
    ) -> Result<ResolvedC220UnaryVector, MteStepperError> {
        let pc = self.scalar.pc();
        if self.scalar.is_halted() {
            return Err(MteStepperError::ProgramEnded { pc });
        }
        let machine = self.scalar.machine();
        if machine.architecture() != Architecture::Dav2201 {
            return Err(MteStepperError::UnsupportedWord { pc, word });
        }
        if self.c220.output_buffer_busy() {
            return Err(MteStepperError::OutputDependencyOutstanding);
        }
        let xregs = machine.xregs();
        let mask_control = machine
            .spr_value(3)
            .ok_or(C220VectorError::MissingMaskState)?;
        if !supports_count_mask && mask_control & (1 << 56) != 0 {
            return Err(C220VectorError::UnsupportedMaskControl {
                control: mask_control,
            }
            .into());
        }
        let control = decode_c220_vector_unary_control(xregs[usize::from(control_register)]);
        let iteration_masks = decode_c220_repeat_masks(
            mask_control,
            machine
                .spr_value(100)
                .ok_or(C220VectorError::MissingMaskState)?,
            machine
                .spr_value(101)
                .ok_or(C220VectorError::MissingMaskState)?,
            C220_VECTOR_TILE_BYTES / usize::from(element_bytes),
            control.encoded_repeat_count,
        )?;
        Ok(ResolvedC220UnaryVector {
            pc,
            mask_control,
            control,
            addresses: C220VectorAddresses {
                source_0: xregs[usize::from(source_register)],
                source_1: 0,
                destination: xregs[usize::from(destination_register)],
            },
            iteration_masks,
        })
    }

    pub(crate) fn commit_c220_vector_issue(&mut self, destination_address: Option<u64>) {
        if let Some(source_address) = destination_address {
            self.c220.unsignaled_output = Some(C220OutputToken { source_address });
        }
        self.scalar.advance_sequential();
    }

    pub(crate) fn commit_c220_vector_stores(
        &mut self,
        stores: &[C220VectorStore],
    ) -> Result<(), MteStepperError> {
        let segments = stores
            .iter()
            .map(|store| {
                (
                    store.address,
                    store.data[..usize::from(store.width_bytes)]
                        .iter()
                        .copied()
                        .map(MemoryByteState::Known)
                        .collect(),
                )
            })
            .collect::<Vec<_>>();
        self.ub.write_segments(&segments)?;
        Ok(())
    }

    fn step_c220_fp32_word(
        &mut self,
        word: u32,
        expected_operation: Option<C220VecArithmeticOperation>,
    ) -> Result<C220Fp32Step, MteStepperError> {
        let resolved = self.resolve_c220_vector_arithmetic(word, expected_operation, false)?;
        let step = execute_c220_fp32_to_ub(
            resolved.pc,
            word,
            resolved.control,
            resolved.addresses,
            &resolved.iteration_masks,
            &mut self.ub,
        )?;
        self.c220.unsignaled_output = Some(C220OutputToken {
            source_address: step.destination_address,
        });
        self.scalar.advance_sequential();
        Ok(step)
    }

    fn resolve_c220_vector_arithmetic(
        &self,
        word: u32,
        expected_operation: Option<C220VecArithmeticOperation>,
        allow_non_fp32: bool,
    ) -> Result<ResolvedC220VectorArithmetic, MteStepperError> {
        let pc = self.scalar.pc();
        if self.scalar.is_halted() {
            return Err(MteStepperError::ProgramEnded { pc });
        }
        if self.scalar.machine().architecture() != Architecture::Dav2201 {
            return Err(MteStepperError::UnsupportedWord { pc, word });
        }
        if self.c220.output_buffer_busy() {
            return Err(MteStepperError::OutputDependencyOutstanding);
        }
        let hint = C220VecArithmeticHint::from_word(word)
            .filter(|hint| {
                hint.has_fp32_value_path()
                    || (allow_non_fp32
                        && (hint.has_s32_value_path()
                            || hint.has_s16_value_path()
                            || hint.has_f16_value_path()
                            || hint.has_bitwise_b16_value_path()))
            })
            .filter(|hint| expected_operation.is_none_or(|operation| hint.operation == operation))
            .ok_or(MteStepperError::UnsupportedWord { pc, word })?;
        let machine = self.scalar.machine();
        let xregs = machine.xregs();
        let control = xregs[usize::from(hint.control_register)];
        let control = if matches!(
            hint.operation,
            C220VecArithmeticOperation::Absolute
                | C220VecArithmeticOperation::Rectify
                | C220VecArithmeticOperation::Not
        ) {
            decode_c220_vector_unary_control(control)
        } else {
            decode_c220_fp32_control(control)?
        };
        let ctrl = machine
            .spr_value(3)
            .ok_or(C220VectorError::MissingMaskState)?;
        let modes = C220VectorArithmeticModes::from_control_spr(ctrl);
        let result_element_bytes = modes
            .result_element_bytes(hint)
            .ok_or(MteStepperError::UnsupportedWord { pc, word })?;
        let mask_control = if modes.widen_s16 {
            ctrl & !(1 << 52)
        } else {
            ctrl
        };
        let mask0 = machine.spr_value(100);
        let mask1 = machine.spr_value(101);
        let iteration_masks = decode_c220_repeat_masks(
            mask_control,
            mask0.ok_or(C220VectorError::MissingMaskState)?,
            mask1.ok_or(C220VectorError::MissingMaskState)?,
            C220_VECTOR_TILE_BYTES / usize::from(result_element_bytes),
            control.encoded_repeat_count,
        )?;
        Ok(ResolvedC220VectorArithmetic {
            pc,
            control,
            addresses: C220VectorAddresses {
                source_0: xregs[usize::from(hint.source_0_register)],
                source_1: hint
                    .source_1_register
                    .map_or(0, |register| xregs[usize::from(register)]),
                destination: xregs[usize::from(hint.destination_register)],
            },
            iteration_masks,
        })
    }
}
