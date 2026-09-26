use crate::isa::c220::control::C220VectorControlInstruction;
use crate::isa::c220::vector::axpy::C220AxpyInstruction;
use crate::isa::c220::vector::compare::{
    C220CompareMaskInstruction, C220MoveMaskDirection, C220MoveMaskInstruction,
    C220PackedCompareInstruction,
};
use crate::isa::c220::vector::conversion::C220ConversionInstruction;
use crate::isa::c220::vector::fused::C220FusedInstruction;
use crate::isa::c220::vector::gather::C220GatherInstruction;
use crate::isa::c220::vector::load_va::C220LoadVaInstruction;
use crate::isa::c220::vector::merge::C220MergeInstruction;
use crate::isa::c220::vector::no_effect::C220NoEffectVectorInstruction;
use crate::isa::c220::vector::reduce::C220ReductionInstruction;
use crate::isa::c220::vector::scalar::C220VectorScalarInstruction;
use crate::isa::c220::vector::select::C220SelectInstruction;
use crate::isa::c220::vector::sort::C220SortInstruction;
use crate::isa::c220::vector::special::C220SpecialUnaryInstruction;
use crate::isa::c220::vector::spr::C220VectorSprWrite;
use crate::isa::c220::vector::ternary::C220TernaryInstruction;
use crate::isa::c220::vector::{
    C220BroadcastInstruction, C220CopyInstruction, C220MoveVaInstruction, C220MovemaskHint,
    C220MovevInstruction, C220NchwInstruction, C220ShiftInstruction, C220TransposeInstruction,
    C220VecArithmeticHint,
};
use crate::sim::c220::schedule::{C220Stall, C220StallCause};
use crate::sim::c220::vector::C220VectorInstruction;
use crate::sim::c220::vector::ops::compare::{
    C220CompareMask, plan_c220_compare_mask_issue, plan_c220_move_mask_issue,
    plan_c220_packed_compare_issue,
};
use crate::sim::c220::vector::ops::nchw::plan_c220_nchw_issue;
use crate::sim::c220::vector::ops::select::{C220SelectMode, plan_c220_select_issue};
use crate::sim::c220::vector::pipeline::C220VectorQueueClass;
use crate::sim::c220::vector::va::plan_c220_load_va_issue;
use crate::sim::c220::vector::{C220VectorError, C220VectorMaskState, decode_c220_repeat_masks};

use super::runtime::{C220VectorRuntimeError, VectorEngine};
use crate::sim::c220::state::{C220ExecutionError, C220State};

pub(in crate::sim::c220) enum VectorStep {
    Issued(C220VectorInstruction),
    Stalled(C220Stall),
}

impl VectorEngine {
    pub(in crate::sim::c220) fn dispatch_at(
        &mut self,
        tick: u64,
        instruction_id: u64,
        request: &super::C220VectorRequest,
        state: &mut C220State,
    ) -> Result<VectorStep, C220VectorRuntimeError> {
        if let Some(previous) = self.last_dispatched_id
            && instruction_id <= previous
        {
            return Err(C220VectorRuntimeError::InstructionOrder {
                previous,
                requested: instruction_id,
            });
        }
        let pc = request.pc;
        let word = request.word;
        let inputs = request.context(state.scalar().machine(), state.ub());
        if let Some(resume_tick) = self.pipeline.instruction_buffer_ready_tick()
            && tick < resume_tick
        {
            return Ok(VectorStep::Stalled(C220Stall {
                tick,
                pc,
                resume_tick,
                cause: C220StallCause::VectorDependency,
            }));
        }
        {
            let queue_class = C220VectorQueueClass::from_word(word);
            let mut resume_tick = self.pipeline.pending_queue_hazard_tick(queue_class);
            if queue_class == C220VectorQueueClass::Vms4 {
                resume_tick = resume_tick
                    .into_iter()
                    .chain(self.vmsu.pending_drain_tick())
                    .max();
            }
            if let Some(resume_tick) = resume_tick
                && tick < resume_tick
            {
                return Ok(VectorStep::Stalled(C220Stall {
                    tick,
                    pc,
                    resume_tick,
                    cause: C220StallCause::VectorDependency,
                }));
            }
        }

        if (C220MoveVaInstruction::decode(word).is_some()
            || C220VectorSprWrite::decode(word).is_some())
            && let Some(resume_tick) = self
                .pipeline
                .pending_register_write_blocker_tick()
                .into_iter()
                .chain(self.vmsu.pending_drain_tick())
                .max()
            && tick < resume_tick
        {
            return Ok(VectorStep::Stalled(C220Stall {
                tick,
                pc,
                resume_tick,
                cause: C220StallCause::VectorDependency,
            }));
        }
        let instruction = match word {
            _ if C220VectorSprWrite::decode(word).is_some() => {
                let decoded = C220VectorSprWrite::decode(word).expect("matched decode");
                self.pipeline.issue_register_write_at(tick)?;
                let step = super::spr::execute_write(
                    state.scalar_mut().machine_mut(),
                    pc,
                    word,
                    decoded,
                    request.registers.xregs()[usize::from(decoded.source_register)],
                )?;

                C220VectorInstruction::WriteSpr(step)
            }
            _ if C220MoveVaInstruction::decode(word).is_some() => {
                let decoded = C220MoveVaInstruction::decode(word).expect("matched decode");
                let instruction = C220VectorInstruction::MoveAddress {
                    pc,
                    word,
                    instruction: decoded,
                };
                self.issue_at(tick, &instruction)?;
                self.va.write_pair(decoded, inputs.machine.xregs());

                instruction
            }
            _ if C220LoadVaInstruction::decode(word).is_some() => {
                let decoded = C220LoadVaInstruction::decode(word).expect("matched decode");
                if decoded.high_half
                    && let Some(resume_tick) = self.pipeline.pending_load_va_drain_tick()
                    && tick < resume_tick
                {
                    return Ok(VectorStep::Stalled(C220Stall {
                        tick,
                        pc,
                        resume_tick,
                        cause: C220StallCause::VectorDependency,
                    }));
                }
                let source_address = inputs.machine.xregs()[usize::from(decoded.source_register)];
                let step = plan_c220_load_va_issue(pc, word, source_address, state.ub())?;
                let instruction = C220VectorInstruction::LoadAddress(step);
                self.issue_at(tick, &instruction)?;

                instruction
            }
            _ if C220MovemaskHint::from_word(word).is_some() => {
                let hint = C220MovemaskHint::from_word(word).expect("matched decode");
                let step = crate::sim::c220::scalar::C220MovemaskStep {
                    pc,
                    word,
                    source_register: hint.source_register,
                    source_value: request.registers.xregs()[usize::from(hint.source_register)],
                    destination_spr: hint.destination_spr,
                    prior_destination_value: state
                        .scalar()
                        .machine()
                        .spr_value(hint.destination_spr),
                };
                let instruction = C220VectorInstruction::Movemask(step);
                self.issue_at(tick, &instruction)?;
                state
                    .scalar_mut()
                    .machine_mut()
                    .set_spr_value(step.destination_spr, step.source_value)?;

                instruction
            }
            _ if C220VectorControlInstruction::decode(word).is_some() => {
                let decoded = C220VectorControlInstruction::decode(word).expect("matched decode");
                let instruction = C220VectorInstruction::Control {
                    pc,
                    word,
                    instruction: decoded,
                };
                self.issue_at(tick, &instruction)?;

                instruction
            }
            _ if C220NoEffectVectorInstruction::decode(word).is_some() => {
                let decoded = C220NoEffectVectorInstruction::decode(word).expect("matched decode");
                let machine = inputs.machine;
                let control = machine.xregs()[usize::from(decoded.control_register)];
                let repeat_count = decode_c220_repeat_masks(
                    machine
                        .spr_value(3)
                        .ok_or(C220VectorError::MissingMaskState)?,
                    machine
                        .spr_value(100)
                        .ok_or(C220VectorError::MissingMaskState)?,
                    machine
                        .spr_value(101)
                        .ok_or(C220VectorError::MissingMaskState)?,
                    decoded.lane_count(),
                    (control >> 56) as u8,
                )?
                .len();
                let instruction = C220VectorInstruction::NoEffect {
                    pc,
                    word,
                    instruction: decoded,
                    repeat_count,
                    lane_groups: decoded.lane_groups(),
                };
                self.issue_at(tick, &instruction)?;

                instruction
            }
            _ if C220MovevInstruction::decode(word).is_some() => {
                let step = inputs.preview_c220_movev_word(word)?;
                let instruction = C220VectorInstruction::Move(step);
                self.issue_at(tick, &instruction)?;

                instruction
            }
            _ if C220NchwInstruction::decode(word).is_some() => {
                if let Some(resume_tick) = self.pipeline.pending_load_va_drain_tick()
                    && tick < resume_tick
                {
                    return Ok(VectorStep::Stalled(C220Stall {
                        tick,
                        pc,
                        resume_tick,
                        cause: C220StallCause::VectorDependency,
                    }));
                }
                let decoded = C220NchwInstruction::decode(word).expect("matched decode");
                let control = inputs.machine.xregs()[usize::from(decoded.control_register)];
                let step = plan_c220_nchw_issue(pc, word, control, &self.va, state.ub())?;
                let instruction = C220VectorInstruction::Nchw(step);
                self.issue_at(tick, &instruction)?;

                instruction
            }
            _ if C220MoveMaskInstruction::decode(word).is_some() => {
                let decoded = C220MoveMaskInstruction::decode(word).expect("matched decode");
                if matches!(decoded.direction, C220MoveMaskDirection::FromMemory)
                    && let Some(resume_tick) = self.pending_drain_tick()
                    && tick < resume_tick
                {
                    return Ok(VectorStep::Stalled(C220Stall {
                        tick,
                        pc,
                        resume_tick,
                        cause: C220StallCause::VectorDependency,
                    }));
                }
                let step = plan_c220_move_mask_issue(pc, word, inputs.machine.xregs(), state.ub())?;
                let instruction = C220VectorInstruction::MoveMask(step);
                self.issue_at(tick, &instruction)?;

                instruction
            }
            _ if C220CompareMaskInstruction::decode(word).is_some() => {
                let decoded = C220CompareMaskInstruction::decode(word).expect("matched decode");
                let machine = inputs.machine;
                let registers = machine.xregs();
                let step = plan_c220_compare_mask_issue(
                    pc,
                    word,
                    registers[usize::from(decoded.control_register)],
                    C220VectorMaskState {
                        control: machine
                            .spr_value(3)
                            .ok_or(C220VectorError::MissingMaskState)?,
                        low: machine
                            .spr_value(100)
                            .ok_or(C220VectorError::MissingMaskState)?,
                        high: machine
                            .spr_value(101)
                            .ok_or(C220VectorError::MissingMaskState)?,
                    },
                    C220CompareMask::from_bits([
                        machine
                            .spr_value(104)
                            .ok_or(C220VectorError::MissingMaskState)?,
                        machine
                            .spr_value(105)
                            .ok_or(C220VectorError::MissingMaskState)?,
                    ]),
                    registers,
                    state.ub(),
                )?;
                let instruction = C220VectorInstruction::CompareMask(step);
                self.issue_at(tick, &instruction)?;

                instruction
            }
            _ if C220SelectInstruction::decode(word).is_some() => {
                let decoded = C220SelectInstruction::decode(word).expect("matched decode");
                let machine = inputs.machine;
                let registers = machine.xregs();
                let control_value = registers[usize::from(decoded.control_register)];
                let mode = C220SelectMode::decode(control_value).ok_or(
                    C220VectorError::UnsupportedSelectMode(((control_value >> 48) & 3) as u8),
                )?;
                if matches!(mode, C220SelectMode::TensorTensor)
                    && self.pipeline.has_pending_compare_mask_write()
                {
                    return Ok(VectorStep::Stalled(C220Stall {
                        tick,
                        pc,
                        resume_tick: self.pipeline.pending_drain_tick().unwrap_or(tick),
                        cause: C220StallCause::VectorDependency,
                    }));
                }
                let step = plan_c220_select_issue(
                    pc,
                    word,
                    control_value,
                    C220VectorMaskState {
                        control: machine
                            .spr_value(3)
                            .ok_or(C220VectorError::MissingMaskState)?,
                        low: machine
                            .spr_value(100)
                            .ok_or(C220VectorError::MissingMaskState)?,
                        high: machine
                            .spr_value(101)
                            .ok_or(C220VectorError::MissingMaskState)?,
                    },
                    self.pipeline.compare_mask(),
                    registers,
                    state.ub(),
                )?;
                let instruction = C220VectorInstruction::Select(step);
                self.issue_at(tick, &instruction)?;

                instruction
            }
            _ if C220PackedCompareInstruction::decode(word).is_some() => {
                let decoded = C220PackedCompareInstruction::decode(word).expect("matched decode");
                let registers = inputs.machine.xregs();
                let control = registers[usize::from(decoded.control_register)];
                let step =
                    plan_c220_packed_compare_issue(pc, word, control, registers, state.ub())?;
                let instruction = C220VectorInstruction::PackedCompare(step);
                self.issue_at(tick, &instruction)?;

                instruction
            }
            _ if C220ReductionInstruction::decode(word).is_some() => {
                let step = inputs.preview_c220_reduction_word(word)?;
                let instruction = C220VectorInstruction::Reduction(step);
                self.issue_at(tick, &instruction)?;

                instruction
            }
            _ if C220SortInstruction::decode(word).is_some() => {
                let step = inputs.preview_c220_sort_word(word)?;
                let instruction = C220VectorInstruction::Sort(step);
                self.issue_at(tick, &instruction)?;

                instruction
            }
            _ if C220MergeInstruction::decode(word).is_some() => {
                if let Some(resume_tick) = self.pending_drain_tick()
                    && tick < resume_tick
                {
                    return Ok(VectorStep::Stalled(C220Stall {
                        tick,
                        pc,
                        resume_tick,
                        cause: C220StallCause::VectorDependency,
                    }));
                }
                let step = inputs.preview_c220_merge_word(word)?;
                self.vmsu.issue_at(tick, step.clone(), state.ub())?;

                C220VectorInstruction::Merge(step)
            }
            _ if C220TernaryInstruction::decode(word).is_some() => {
                let step = inputs.preview_c220_ternary_word(word)?;
                let instruction = C220VectorInstruction::Ternary(step);
                self.issue_at(tick, &instruction)?;

                instruction
            }
            _ if C220AxpyInstruction::decode(word).is_some() => {
                let step = inputs.preview_c220_axpy_word(word)?;
                let instruction = C220VectorInstruction::Axpy(step);
                self.issue_at(tick, &instruction)?;

                instruction
            }
            _ if C220SpecialUnaryInstruction::decode(word).is_some() => {
                let step = inputs.preview_c220_special_unary_word(word)?;
                let instruction = C220VectorInstruction::SpecialUnary(step);
                self.issue_at(tick, &instruction)?;

                instruction
            }
            _ if C220FusedInstruction::decode(word).is_some() => {
                let step = inputs.preview_c220_fused_word(word)?;
                let instruction = C220VectorInstruction::Fused(step);
                self.issue_at(tick, &instruction)?;

                instruction
            }
            _ if C220ConversionInstruction::decode(word).is_some() => {
                let step = inputs.preview_c220_conversion_word(word)?;
                let instruction = C220VectorInstruction::Conversion(step);
                self.issue_at(tick, &instruction)?;

                instruction
            }
            _ if C220GatherInstruction::decode(word).is_some() => {
                let step = inputs.preview_c220_gather_word(word)?;
                let instruction = C220VectorInstruction::Gather(step);
                self.issue_at(tick, &instruction)?;

                instruction
            }
            _ if C220VecArithmeticHint::from_word(word).is_some() => {
                let step = inputs.preview_c220_vector_word(word)?;
                let instruction = C220VectorInstruction::Arithmetic(step);
                self.issue_at(tick, &instruction)?;

                instruction
            }
            _ if C220VectorScalarInstruction::decode(word).is_some() => {
                let step = inputs.preview_c220_vector_scalar_word(word)?;
                let instruction = C220VectorInstruction::Scalar(step);
                self.issue_at(tick, &instruction)?;

                instruction
            }
            _ if C220ShiftInstruction::decode(word).is_some() => {
                let step = inputs.preview_c220_shift_word(word)?;
                let instruction = C220VectorInstruction::Shift(step);
                self.issue_at(tick, &instruction)?;

                instruction
            }
            _ if C220CopyInstruction::decode(word).is_some() => {
                let step = inputs.preview_c220_copy_word(word)?;
                let instruction = C220VectorInstruction::Copy(step);
                self.issue_at(tick, &instruction)?;

                instruction
            }
            _ if C220BroadcastInstruction::decode(word).is_some() => {
                let step = inputs.preview_c220_broadcast_word(word)?;
                let instruction = C220VectorInstruction::Broadcast(step);
                self.issue_at(tick, &instruction)?;

                instruction
            }
            _ if C220TransposeInstruction::decode(word).is_some() => {
                let step = inputs.preview_c220_transpose_word(word)?;
                let instruction = C220VectorInstruction::Transpose(step);
                self.issue_at(tick, &instruction)?;

                instruction
            }

            _ => return Err(C220ExecutionError::UnsupportedWord { pc, word }.into()),
        };
        self.record_dispatch(tick, instruction_id, pc, word);
        Ok(VectorStep::Issued(instruction))
    }
}

pub(in crate::sim::c220) fn is_vector_word(word: u32) -> bool {
    C220MoveVaInstruction::decode(word).is_some()
        || C220VectorSprWrite::decode(word).is_some()
        || C220LoadVaInstruction::decode(word).is_some()
        || C220MovemaskHint::from_word(word).is_some()
        || C220VectorControlInstruction::decode(word).is_some()
        || C220NoEffectVectorInstruction::decode(word).is_some()
        || C220MovevInstruction::decode(word).is_some()
        || C220NchwInstruction::decode(word).is_some()
        || C220MoveMaskInstruction::decode(word).is_some()
        || C220CompareMaskInstruction::decode(word).is_some()
        || C220SelectInstruction::decode(word).is_some()
        || C220PackedCompareInstruction::decode(word).is_some()
        || C220ReductionInstruction::decode(word).is_some()
        || C220SortInstruction::decode(word).is_some()
        || C220MergeInstruction::decode(word).is_some()
        || C220TernaryInstruction::decode(word).is_some()
        || C220AxpyInstruction::decode(word).is_some()
        || C220SpecialUnaryInstruction::decode(word).is_some()
        || C220FusedInstruction::decode(word).is_some()
        || C220ConversionInstruction::decode(word).is_some()
        || C220GatherInstruction::decode(word).is_some()
        || C220VecArithmeticHint::from_word(word).is_some()
        || C220VectorScalarInstruction::decode(word).is_some()
        || C220ShiftInstruction::decode(word).is_some()
        || C220CopyInstruction::decode(word).is_some()
        || C220BroadcastInstruction::decode(word).is_some()
        || C220TransposeInstruction::decode(word).is_some()
}
