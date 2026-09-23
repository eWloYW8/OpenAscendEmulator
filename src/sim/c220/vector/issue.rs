use crate::architecture::Architecture;
use crate::isa::c220::vector::C220MovevControl;
use crate::isa::c220::vector::axpy::C220AxpyInstruction;
use crate::isa::c220::vector::conversion::C220ConversionInstruction;
use crate::isa::c220::vector::fused::C220FusedInstruction;
use crate::isa::c220::vector::gather::{C220GatherInstruction, C220GatherKind};
use crate::isa::c220::vector::merge::C220MergeInstruction;
use crate::isa::c220::vector::reduce::C220ReductionInstruction;
use crate::isa::c220::vector::scalar::C220VectorScalarInstruction;
use crate::isa::c220::vector::sort::C220SortInstruction;
use crate::isa::c220::vector::special::C220SpecialUnaryInstruction;
use crate::isa::c220::vector::ternary::C220TernaryInstruction;
use crate::isa::c220::vector::{
    C220BroadcastInstruction, C220CopyInstruction, C220MovevInstruction, C220ShiftInstruction,
    C220TransposeInstruction, C220VecArithmeticHint, C220VecArithmeticOperation,
};
use crate::sim::c220::numeric::fp16::C220Fp16Mode;
use crate::sim::c220::state::{C220ExecutionError, C220State};
use crate::sim::c220::vector::ops::axpy::{
    C220AxpyIssue, C220AxpyIssueInputs, plan_c220_axpy_issue,
};
use crate::sim::c220::vector::ops::broadcast::{C220BroadcastIssue, plan_c220_broadcast_issue};
use crate::sim::c220::vector::ops::conversion::{
    C220ConversionIssue, C220ConversionIssueInputs, plan_c220_conversion_issue,
};
use crate::sim::c220::vector::ops::copy::{C220CopyIssue, plan_c220_copy_issue};
use crate::sim::c220::vector::ops::fused::{
    C220FusedIssue, C220FusedIssueInputs, plan_c220_fused_issue,
};
use crate::sim::c220::vector::ops::gather::{C220GatherIssue, plan_c220_gather_issue};
use crate::sim::c220::vector::ops::merge::{C220MergeIssue, plan_c220_merge_issue};
use crate::sim::c220::vector::ops::reduce::{C220ReductionIssue, plan_c220_reduction_issue};
use crate::sim::c220::vector::ops::scalar::{
    C220VectorScalarIssue, C220VectorScalarOperand, plan_c220_vector_scalar_issue,
};
use crate::sim::c220::vector::ops::shift::{C220ShiftIssue, plan_c220_shift_issue};
use crate::sim::c220::vector::ops::sort::{C220SortIssue, plan_c220_sort_issue};
use crate::sim::c220::vector::ops::special::{
    C220SpecialUnaryIssue, plan_c220_special_unary_issue,
};
use crate::sim::c220::vector::ops::ternary::{C220TernaryIssue, plan_c220_ternary_issue};
use crate::sim::c220::vector::ops::transpose::{C220TransposeIssue, plan_c220_transpose_issue};
use crate::sim::c220::vector::{
    C220_VECTOR_TILE_BYTES, C220MovevStep, C220VectorAddresses, C220VectorArithmeticIssue,
    C220VectorArithmeticModes, C220VectorControl, C220VectorError, C220VectorStore,
    decode_c220_repeat_masks, plan_c220_movev_to_ub, plan_c220_vector_arithmetic_issue,
};

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

impl C220State {
    fn vector_issue_pc(&self, word: u32) -> Result<u64, C220ExecutionError> {
        let pc = self.scalar.pc();
        if self.scalar.is_halted() {
            return Err(C220ExecutionError::ProgramEnded { pc });
        }
        if self.scalar.machine().architecture() != Architecture::Dav2201 {
            return Err(C220ExecutionError::UnsupportedWord { pc, word });
        }
        Ok(pc)
    }

    pub(crate) fn preview_c220_merge_word(
        &self,
        word: u32,
    ) -> Result<C220MergeIssue, C220ExecutionError> {
        let pc = self.vector_issue_pc(word)?;
        let machine = self.scalar.machine();
        C220MergeInstruction::decode(word)
            .ok_or(C220ExecutionError::UnsupportedWord { pc, word })?;
        Ok(plan_c220_merge_issue(pc, word, machine.xregs(), &self.ub)?)
    }

    pub(crate) fn preview_c220_sort_word(
        &self,
        word: u32,
    ) -> Result<C220SortIssue, C220ExecutionError> {
        let pc = self.vector_issue_pc(word)?;
        let machine = self.scalar.machine();
        let instruction = C220SortInstruction::decode(word)
            .ok_or(C220ExecutionError::UnsupportedWord { pc, word })?;
        let registers = machine.xregs();
        Ok(plan_c220_sort_issue(
            pc,
            word,
            registers[usize::from(instruction.control_register)],
            C220VectorAddresses {
                destination: registers[usize::from(instruction.destination_register)],
                source_0: registers[usize::from(instruction.value_register)],
                source_1: registers[usize::from(instruction.index_register)],
            },
            &self.ub,
        )?)
    }

    pub(crate) fn preview_c220_fused_word(
        &self,
        word: u32,
    ) -> Result<C220FusedIssue, C220ExecutionError> {
        let pc = self.vector_issue_pc(word)?;
        let machine = self.scalar.machine();
        let instruction = C220FusedInstruction::decode(word)
            .ok_or(C220ExecutionError::UnsupportedWord { pc, word })?;
        let xregs = machine.xregs();
        let mask_control = machine
            .spr_value(3)
            .ok_or(C220VectorError::MissingMaskState)?;
        let deq_scale = machine.spr_value(12).unwrap_or(0);
        let control =
            C220VectorControl::decode_binary(xregs[usize::from(instruction.control_register)]);
        let iteration_masks = decode_c220_repeat_masks(
            mask_control,
            machine
                .spr_value(100)
                .ok_or(C220VectorError::MissingMaskState)?,
            machine
                .spr_value(101)
                .ok_or(C220VectorError::MissingMaskState)?,
            instruction.lane_count(),
            control.encoded_repeat_count,
        )?;
        Ok(plan_c220_fused_issue(
            C220FusedIssueInputs {
                pc,
                word,
                control,
                addresses: C220VectorAddresses {
                    source_0: xregs[usize::from(instruction.source_0_register)],
                    source_1: xregs[usize::from(instruction.source_1_register)],
                    destination: xregs[usize::from(instruction.destination_register)],
                },
                iteration_masks: &iteration_masks,
                fp16_mode: C220Fp16Mode::from_control_spr(mask_control),
                arithmetic_saturating: mask_control & (1 << 53) != 0,
                integer_saturating: mask_control & (1 << 59) == 0,
                descriptor_address: 32 * (deq_scale & 0x3fff),
                deq_scale: deq_scale as u16,
            },
            &self.ub,
        )?)
    }

    pub(crate) fn preview_c220_conversion_word(
        &self,
        word: u32,
    ) -> Result<C220ConversionIssue, C220ExecutionError> {
        let pc = self.vector_issue_pc(word)?;
        let machine = self.scalar.machine();
        let instruction = C220ConversionInstruction::decode(word)
            .ok_or(C220ExecutionError::UnsupportedWord { pc, word })?;
        let xregs = machine.xregs();
        let mask_control = machine
            .spr_value(3)
            .ok_or(C220VectorError::MissingMaskState)?;
        let control =
            C220VectorControl::decode_unary(xregs[usize::from(instruction.control_register)]);
        let deq_scale = machine.spr_value(12).unwrap_or(0);
        let iteration_masks = decode_c220_repeat_masks(
            mask_control,
            machine
                .spr_value(100)
                .ok_or(C220VectorError::MissingMaskState)?,
            machine
                .spr_value(101)
                .ok_or(C220VectorError::MissingMaskState)?,
            instruction.lane_count(),
            control.encoded_repeat_count,
        )?;
        Ok(plan_c220_conversion_issue(
            C220ConversionIssueInputs {
                pc,
                word,
                control,
                addresses: C220VectorAddresses {
                    source_0: xregs[usize::from(instruction.source_register)],
                    source_1: 32 * (deq_scale & 0x3fff),
                    destination: xregs[usize::from(instruction.destination_register)],
                },
                iteration_masks: &iteration_masks,
                fp16_mode: C220Fp16Mode::from_control_spr(mask_control),
                integer_saturating: mask_control & (1 << 59) == 0,
                deq_scale,
            },
            &self.ub,
        )?)
    }

    pub(crate) fn preview_c220_special_unary_word(
        &self,
        word: u32,
    ) -> Result<C220SpecialUnaryIssue, C220ExecutionError> {
        let instruction = C220SpecialUnaryInstruction::decode(word).ok_or(
            C220ExecutionError::UnsupportedWord {
                pc: self.scalar.pc(),
                word,
            },
        )?;
        let resolved = self.resolve_c220_unary_vector(
            word,
            instruction.destination_register,
            instruction.source_register,
            instruction.control_register,
            instruction.width.element_bytes(),
            true,
        )?;
        Ok(plan_c220_special_unary_issue(
            resolved.pc,
            word,
            resolved.control,
            resolved.addresses,
            &resolved.iteration_masks,
            C220Fp16Mode::from_control_spr(resolved.mask_control),
            &self.ub,
        )?)
    }

    pub(crate) fn preview_c220_axpy_word(
        &self,
        word: u32,
    ) -> Result<C220AxpyIssue, C220ExecutionError> {
        let instruction =
            C220AxpyInstruction::decode(word).ok_or(C220ExecutionError::UnsupportedWord {
                pc: self.scalar.pc(),
                word,
            })?;
        let resolved = self.resolve_c220_unary_vector(
            word,
            instruction.destination_register,
            instruction.source_register,
            instruction.control_register,
            instruction.width.destination_element_bytes(),
            true,
        )?;
        Ok(plan_c220_axpy_issue(
            C220AxpyIssueInputs {
                pc: resolved.pc,
                word,
                scalar_bits: self.scalar.machine().xregs()[usize::from(instruction.scalar_register)]
                    as u32,
                control: resolved.control,
                addresses: resolved.addresses,
                iteration_masks: &resolved.iteration_masks,
                fp16_mode: C220Fp16Mode::from_control_spr(resolved.mask_control),
            },
            &self.ub,
        )?)
    }

    pub(crate) fn preview_c220_gather_word(
        &self,
        word: u32,
    ) -> Result<C220GatherIssue, C220ExecutionError> {
        let pc = self.vector_issue_pc(word)?;
        let instruction = C220GatherInstruction::decode(word)
            .ok_or(C220ExecutionError::UnsupportedWord { pc, word })?;
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
    ) -> Result<C220TernaryIssue, C220ExecutionError> {
        let pc = self.vector_issue_pc(word)?;
        let instruction = C220TernaryInstruction::decode(word)
            .ok_or(C220ExecutionError::UnsupportedWord { pc, word })?;
        let machine = self.scalar.machine();
        let registers = machine.xregs();
        let control =
            C220VectorControl::decode_binary(registers[usize::from(instruction.control_register)]);
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
    ) -> Result<C220ReductionIssue, C220ExecutionError> {
        let instruction =
            C220ReductionInstruction::decode(word).ok_or(C220ExecutionError::UnsupportedWord {
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
    ) -> Result<C220TransposeIssue, C220ExecutionError> {
        let pc = self.vector_issue_pc(word)?;
        let instruction = C220TransposeInstruction::decode(word)
            .ok_or(C220ExecutionError::UnsupportedWord { pc, word })?;
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
    ) -> Result<C220BroadcastIssue, C220ExecutionError> {
        let pc = self.vector_issue_pc(word)?;
        let instruction = C220BroadcastInstruction::decode(word)
            .ok_or(C220ExecutionError::UnsupportedWord { pc, word })?;
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

    pub(crate) fn preview_c220_movev_word(
        &self,
        word: u32,
    ) -> Result<C220MovevStep, C220ExecutionError> {
        let pc = self.vector_issue_pc(word)?;
        let instruction = C220MovevInstruction::decode(word)
            .ok_or(C220ExecutionError::UnsupportedWord { pc, word })?;
        let element_bytes = instruction
            .supported_element_bytes()
            .ok_or(C220ExecutionError::UnsupportedWord { pc, word })?;
        let machine = self.scalar.machine();
        let xregs = machine.xregs();
        let control = xregs[usize::from(instruction.control_register)];
        let control = C220MovevControl::decode(control);
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

    pub(crate) fn preview_c220_vector_word(
        &self,
        word: u32,
    ) -> Result<C220VectorArithmeticIssue, C220ExecutionError> {
        let resolved = self.resolve_c220_vector_arithmetic(word)?;
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
    ) -> Result<C220VectorScalarIssue, C220ExecutionError> {
        let instruction = C220VectorScalarInstruction::decode(word).ok_or(
            C220ExecutionError::UnsupportedWord {
                pc: self.scalar.pc(),
                word,
            },
        )?;
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
    ) -> Result<C220ShiftIssue, C220ExecutionError> {
        let instruction =
            C220ShiftInstruction::decode(word).ok_or(C220ExecutionError::UnsupportedWord {
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
    ) -> Result<C220CopyIssue, C220ExecutionError> {
        let instruction =
            C220CopyInstruction::decode(word).ok_or(C220ExecutionError::UnsupportedWord {
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
    ) -> Result<ResolvedC220UnaryVector, C220ExecutionError> {
        let pc = self.vector_issue_pc(word)?;
        let machine = self.scalar.machine();
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
        let control = C220VectorControl::decode_unary(xregs[usize::from(control_register)]);
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
            self.output.publish_vector_output(source_address);
        }
        self.commit_c220_sequential_issue();
    }

    pub(crate) fn commit_c220_vector_stores(
        &mut self,
        stores: &[C220VectorStore],
    ) -> Result<(), C220ExecutionError> {
        super::access::commit_vector_stores(&mut self.ub, stores)?;
        Ok(())
    }

    fn resolve_c220_vector_arithmetic(
        &self,
        word: u32,
    ) -> Result<ResolvedC220VectorArithmetic, C220ExecutionError> {
        let pc = self.vector_issue_pc(word)?;
        let hint = C220VecArithmeticHint::from_word(word)
            .filter(|hint| {
                hint.has_fp32_value_path()
                    || hint.has_s32_value_path()
                    || hint.has_s16_value_path()
                    || hint.has_f16_value_path()
                    || hint.has_bitwise_b16_value_path()
            })
            .ok_or(C220ExecutionError::UnsupportedWord { pc, word })?;
        let machine = self.scalar.machine();
        let xregs = machine.xregs();
        let control = xregs[usize::from(hint.control_register)];
        let control = if matches!(
            hint.operation,
            C220VecArithmeticOperation::Absolute
                | C220VecArithmeticOperation::Rectify
                | C220VecArithmeticOperation::Not
        ) {
            C220VectorControl::decode_unary(control)
        } else {
            C220VectorControl::decode_binary(control)
        };
        let ctrl = machine
            .spr_value(3)
            .ok_or(C220VectorError::MissingMaskState)?;
        let modes = C220VectorArithmeticModes::from_control_spr(ctrl);
        let result_element_bytes = modes
            .result_element_bytes(hint)
            .ok_or(C220ExecutionError::UnsupportedWord { pc, word })?;
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
