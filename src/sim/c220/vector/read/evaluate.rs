use super::{
    C220VectorLaneOutcome, C220VectorReadOperation, C220VectorReadSample, PendingVectorRead,
};
use crate::isa::c220::vector::compare::C220MoveMaskDirection;
use crate::memory::ub::UbMemory;
use crate::sim::c220::vector::lanes::f16::{
    C220F16ValueInputs, evaluate_c220_f16_repeat_from_bytes,
};
use crate::sim::c220::vector::lanes::s16::{
    C220S16ValueInputs, C220S16WidenInputs, evaluate_c220_s16_repeat_from_bytes,
    evaluate_c220_s16_widen_repeat_from_bytes,
};
use crate::sim::c220::vector::lanes::s32::{
    C220S32ValueInputs, evaluate_c220_s32_repeat_from_bytes,
};
use crate::sim::c220::vector::ops::axpy::evaluate_c220_axpy_repeat;
use crate::sim::c220::vector::ops::broadcast::evaluate_c220_broadcast_repeat;
use crate::sim::c220::vector::ops::compare::{
    C220CompareMask, C220CompareMaskUpdate, C220PackedCompareValueInputs,
    evaluate_c220_compare_mask_uop, evaluate_c220_packed_compare_uop,
};
use crate::sim::c220::vector::ops::conversion::{
    C220ConversionValueInputs, evaluate_c220_conversion_repeat,
};
use crate::sim::c220::vector::ops::copy::{C220CopyValueInputs, evaluate_c220_copy_repeat};
use crate::sim::c220::vector::ops::fused::{C220FusedValueInputs, evaluate_c220_fused_repeat};
use crate::sim::c220::vector::ops::gather::evaluate_c220_gather_data_uop;
use crate::sim::c220::vector::ops::nchw::evaluate_c220_nchw_repeat;
use crate::sim::c220::vector::ops::reduce::evaluate_c220_reduction_repeat;
use crate::sim::c220::vector::ops::scalar::{
    C220VectorScalarValueInputs, evaluate_c220_vector_scalar_repeat,
};
use crate::sim::c220::vector::ops::select::{C220SelectMode, evaluate_c220_select_uop};
use crate::sim::c220::vector::ops::shift::{C220ShiftValueInputs, evaluate_c220_shift_repeat};
use crate::sim::c220::vector::ops::sort::evaluate_c220_sort_repeat;
use crate::sim::c220::vector::ops::special::evaluate_c220_special_unary_repeat;
use crate::sim::c220::vector::ops::ternary::{
    C220TernaryValueInputs, evaluate_c220_ternary_repeat,
};
use crate::sim::c220::vector::ops::transpose::evaluate_c220_transpose;
use crate::sim::c220::vector::va::evaluate_c220_load_va;
use crate::sim::c220::vector::{
    C220VectorError, C220VectorStore, evaluate_c220_fp32_repeat_from_bytes,
};

impl PendingVectorRead {
    pub(in crate::sim::c220::vector) fn sample(
        &self,
        ub: &UbMemory,
        compare_mask: C220CompareMask,
    ) -> Result<(C220VectorReadSample, Vec<C220VectorStore>), C220VectorError> {
        let lane_group = self.lane_group.unwrap_or_default();
        let lane_active = |index: usize| {
            self.lane_slice.map_or_else(
                || index / 64 == usize::from(lane_group),
                |(first, count)| index >= first && index < first + count,
            ) && self.mask[index / 64] & (1_u64 << (index % 64)) != 0
        };
        let mut reduction_update = None;
        let mut conversion_lanes = None;
        let mut fused_lanes = None;
        let mut sort_lanes = None;
        let mut va_update = None;
        let (lanes, stores) = match self.operation.clone() {
            C220VectorReadOperation::LoadVa { issue } => {
                va_update = Some(evaluate_c220_load_va(issue, &self.source_0_bytes));
                (Vec::new(), Vec::new())
            }
            C220VectorReadOperation::Broadcast {
                instruction,
                control,
            } => {
                let (values, stores) = evaluate_c220_broadcast_repeat(
                    instruction,
                    control,
                    self.addresses.destination,
                    self.repeat_index,
                    &self.source_0_bytes,
                )?;
                (
                    values
                        .into_iter()
                        .map(|bits| C220VectorLaneOutcome {
                            active: true,
                            bits,
                            fp16_status: None,
                            fp32_status: None,
                        })
                        .collect(),
                    stores,
                )
            }
            C220VectorReadOperation::Transpose => {
                let (values, stores) =
                    evaluate_c220_transpose(self.addresses.destination, &self.source_0_bytes)?;
                (
                    values
                        .into_iter()
                        .map(|bits| C220VectorLaneOutcome {
                            active: true,
                            bits,
                            fp16_status: None,
                            fp32_status: None,
                        })
                        .collect(),
                    stores,
                )
            }
            C220VectorReadOperation::CompareMask { issue } => {
                let values = evaluate_c220_compare_mask_uop(
                    &issue,
                    self.repeat_index,
                    self.lane_slice,
                    &self.source_0_bytes,
                    &self.source_1_bytes,
                )?;
                (
                    values
                        .into_iter()
                        .map(|(active, value)| C220VectorLaneOutcome {
                            active,
                            bits: u32::from(value),
                            fp16_status: None,
                            fp32_status: None,
                        })
                        .collect(),
                    Vec::new(),
                )
            }
            C220VectorReadOperation::MoveMask { issue } => (Vec::new(), issue.stores(compare_mask)),
            C220VectorReadOperation::SelectMaskLoad => (Vec::new(), Vec::new()),
            C220VectorReadOperation::Select { issue } => {
                let mask_bytes = if issue.mode == C220SelectMode::TensorScalar {
                    &self.source_1_bytes
                } else {
                    &self.destination_bytes
                };
                let selection_mask = C220CompareMask::from_bits([
                    u64::from_le_bytes(mask_bytes[..8].try_into().expect("low selection mask")),
                    u64::from_le_bytes(mask_bytes[8..16].try_into().expect("high selection mask")),
                ]);
                let (first, count) = self
                    .lane_slice
                    .unwrap_or((usize::from(lane_group) * 64, 64));
                let mut values = Vec::with_capacity(count);
                let mut stores = Vec::with_capacity(count);
                for group in first / 64..(first + count).div_ceil(64) {
                    let (group_values, group_stores) = evaluate_c220_select_uop(
                        &issue,
                        self.repeat_index,
                        group as u8,
                        compare_mask,
                        selection_mask,
                        &self.source_0_bytes,
                        &self.source_1_bytes,
                    )?;
                    values.extend(group_values);
                    stores.extend(group_stores);
                }
                (
                    values
                        .into_iter()
                        .enumerate()
                        .map(|(index, bits)| C220VectorLaneOutcome {
                            active: lane_active(first + index),
                            bits,
                            fp16_status: None,
                            fp32_status: None,
                        })
                        .collect(),
                    stores,
                )
            }
            C220VectorReadOperation::Nchw { instruction, rows } => {
                let (values, stores) = evaluate_c220_nchw_repeat(
                    instruction,
                    self.repeat_index / 2,
                    *rows,
                    &self.source_0_bytes,
                )?;
                (
                    values
                        .into_iter()
                        .map(|bits| C220VectorLaneOutcome {
                            active: true,
                            bits,
                            fp16_status: None,
                            fp32_status: None,
                        })
                        .collect(),
                    stores,
                )
            }
            C220VectorReadOperation::PackedCompare {
                instruction,
                scalar_bits,
            } => {
                let (values, stores) = evaluate_c220_packed_compare_uop(
                    C220PackedCompareValueInputs {
                        instruction,
                        control: self.control,
                        addresses: self.addresses,
                        scalar_bits,
                        repeat_index: self.repeat_index,
                    },
                    &self.source_0_bytes,
                    &self.source_1_bytes,
                )?;
                (
                    values
                        .into_iter()
                        .map(|bits| C220VectorLaneOutcome {
                            active: true,
                            bits,
                            fp16_status: None,
                            fp32_status: None,
                        })
                        .collect(),
                    stores,
                )
            }
            C220VectorReadOperation::Reduction { issue } => {
                let outcome = evaluate_c220_reduction_repeat(
                    &issue,
                    self.repeat_index,
                    &self.source_0_bytes,
                )?;
                reduction_update = outcome.state_update;
                (
                    outcome
                        .lanes
                        .into_iter()
                        .map(|lane| C220VectorLaneOutcome {
                            active: lane.active,
                            bits: lane.bits,
                            fp16_status: lane.fp16_status,
                            fp32_status: lane.fp32_status,
                        })
                        .collect(),
                    outcome.stores,
                )
            }
            C220VectorReadOperation::Sort { issue } => {
                let (values, stores) = evaluate_c220_sort_repeat(
                    &issue,
                    self.repeat_index,
                    &self.source_0_bytes,
                    &self.source_1_bytes,
                )?;
                sort_lanes = Some(values.clone());
                (
                    values
                        .into_iter()
                        .map(|lane| C220VectorLaneOutcome {
                            active: true,
                            bits: lane.value_bits,
                            fp16_status: None,
                            fp32_status: None,
                        })
                        .collect(),
                    stores,
                )
            }
            C220VectorReadOperation::Ternary { issue } => {
                let (values, stores) = evaluate_c220_ternary_repeat(
                    C220TernaryValueInputs {
                        issue: &issue,
                        repeat_index: self.repeat_index,
                        lane_group,
                        lane_slice: self.lane_slice,
                    },
                    &self.source_0_bytes,
                    &self.source_1_bytes,
                    &self.destination_bytes,
                    ub,
                )?;
                (
                    values
                        .into_iter()
                        .map(|lane| C220VectorLaneOutcome {
                            active: lane.active,
                            bits: lane.bits,
                            fp16_status: lane.fp16_status,
                            fp32_status: lane.fp32_status,
                        })
                        .collect(),
                    stores,
                )
            }
            C220VectorReadOperation::Axpy { issue } => {
                let (values, stores) = evaluate_c220_axpy_repeat(
                    &issue,
                    self.repeat_index,
                    lane_group,
                    self.lane_slice,
                    &self.source_0_bytes,
                    &self.destination_bytes,
                    ub,
                )?;
                (
                    values
                        .into_iter()
                        .map(|lane| C220VectorLaneOutcome {
                            active: lane.active,
                            bits: lane.bits,
                            fp16_status: lane.fp16_status,
                            fp32_status: lane.fp32_status,
                        })
                        .collect(),
                    stores,
                )
            }
            C220VectorReadOperation::SpecialUnary { issue } => {
                let (values, stores) = evaluate_c220_special_unary_repeat(
                    &issue,
                    self.repeat_index,
                    lane_group,
                    self.lane_slice,
                    &self.source_0_bytes,
                    ub,
                )?;
                (
                    values
                        .into_iter()
                        .map(|lane| C220VectorLaneOutcome {
                            active: lane.active,
                            bits: lane.bits,
                            fp16_status: lane.fp16_status,
                            fp32_status: lane.fp32_status,
                        })
                        .collect(),
                    stores,
                )
            }
            C220VectorReadOperation::Conversion { issue } => {
                let (values, stores) = evaluate_c220_conversion_repeat(
                    C220ConversionValueInputs {
                        issue: &issue,
                        repeat_index: self.repeat_index,
                        lane_group,
                        lane_slice: self.lane_slice,
                    },
                    &self.source_0_bytes,
                    &self.source_1_bytes,
                    ub,
                )?;
                conversion_lanes = Some(values.clone());
                (
                    values
                        .into_iter()
                        .map(|lane| C220VectorLaneOutcome {
                            active: lane.active,
                            bits: lane.result_bits as u32,
                            fp16_status: None,
                            fp32_status: None,
                        })
                        .collect(),
                    stores,
                )
            }
            C220VectorReadOperation::Fused { issue } => {
                let (values, stores) = evaluate_c220_fused_repeat(
                    C220FusedValueInputs {
                        issue: &issue,
                        repeat_index: self.repeat_index,
                        lane_group,
                        lane_slice: self.lane_slice,
                    },
                    &self.source_0_bytes,
                    &self.source_1_bytes,
                    &self.destination_bytes,
                    ub,
                )?;
                fused_lanes = Some(values.clone());
                (
                    values
                        .into_iter()
                        .map(|lane| C220VectorLaneOutcome {
                            active: lane.active,
                            bits: u32::from(lane.result_bits),
                            fp16_status: None,
                            fp32_status: None,
                        })
                        .collect(),
                    stores,
                )
            }
            C220VectorReadOperation::GatherIndex => (Vec::new(), Vec::new()),
            C220VectorReadOperation::GatherData { issue, group } => {
                let (values, stores) = evaluate_c220_gather_data_uop(
                    &issue,
                    self.repeat_index,
                    group,
                    &self.source_0_bytes,
                    &self.source_1_bytes,
                )?;
                (
                    values
                        .into_iter()
                        .map(|lane| C220VectorLaneOutcome {
                            active: lane.active,
                            bits: lane.bits,
                            fp16_status: None,
                            fp32_status: None,
                        })
                        .collect(),
                    stores,
                )
            }
            C220VectorReadOperation::Arithmetic { hint, modes, .. } if modes.widens(hint) => {
                let (values, stores) = evaluate_c220_s16_widen_repeat_from_bytes(
                    C220S16WidenInputs {
                        hint,
                        control: self.control,
                        addresses: self.addresses,
                        repeat_index: self.repeat_index,
                        mask: self.mask,
                    },
                    &self.source_0_bytes,
                    &self.source_1_bytes,
                    ub,
                )?;
                (
                    values
                        .into_iter()
                        .map(|lane| C220VectorLaneOutcome {
                            active: lane.active,
                            bits: lane.bits,
                            fp16_status: None,
                            fp32_status: None,
                        })
                        .collect(),
                    stores,
                )
            }
            C220VectorReadOperation::Arithmetic { hint, modes, .. }
                if hint.has_f16_value_path() =>
            {
                let (values, stores) = evaluate_c220_f16_repeat_from_bytes(
                    C220F16ValueInputs {
                        hint,
                        control: self.control,
                        addresses: self.addresses,
                        repeat_index: self.repeat_index,
                        lane_group,
                        lane_slice: self.lane_slice,
                        mask: self.mask,
                        mode: modes.fp16_mode,
                    },
                    &self.source_0_bytes,
                    &self.source_1_bytes,
                    ub,
                )?;
                (
                    values
                        .into_iter()
                        .map(|lane| C220VectorLaneOutcome {
                            active: lane.active,
                            bits: lane.bits,
                            fp16_status: lane.status,
                            fp32_status: None,
                        })
                        .collect(),
                    stores,
                )
            }
            C220VectorReadOperation::Arithmetic { hint, modes, .. }
                if hint.has_s16_value_path() || hint.has_bitwise_b16_value_path() =>
            {
                let (values, stores) = evaluate_c220_s16_repeat_from_bytes(
                    C220S16ValueInputs {
                        hint,
                        control: self.control,
                        addresses: self.addresses,
                        repeat_index: self.repeat_index,
                        lane_group,
                        lane_slice: self.lane_slice,
                        mask: self.mask,
                        saturating: modes.integer_saturating,
                    },
                    &self.source_0_bytes,
                    &self.source_1_bytes,
                    ub,
                )?;
                (
                    values
                        .into_iter()
                        .map(|lane| C220VectorLaneOutcome {
                            active: lane.active,
                            bits: lane.bits,
                            fp16_status: None,
                            fp32_status: None,
                        })
                        .collect(),
                    stores,
                )
            }
            C220VectorReadOperation::Arithmetic { hint, modes, .. }
                if hint.has_s32_value_path() =>
            {
                let (values, stores) = evaluate_c220_s32_repeat_from_bytes(
                    C220S32ValueInputs {
                        hint,
                        control: self.control,
                        addresses: self.addresses,
                        repeat_index: self.repeat_index,
                        mask: self.mask,
                        saturating: modes.integer_saturating,
                    },
                    &self.source_0_bytes,
                    &self.source_1_bytes,
                    ub,
                )?;
                (
                    values
                        .into_iter()
                        .map(|lane| C220VectorLaneOutcome {
                            active: lane.active,
                            bits: lane.bits,
                            fp16_status: None,
                            fp32_status: None,
                        })
                        .collect(),
                    stores,
                )
            }
            C220VectorReadOperation::Arithmetic { hint, .. } => {
                let result = evaluate_c220_fp32_repeat_from_bytes(
                    hint,
                    self.control,
                    self.addresses,
                    self.repeat_index,
                    &self.mask,
                    self.source_0_bytes.clone(),
                    self.source_1_bytes.clone(),
                    ub,
                )?;
                (
                    result
                        .lanes
                        .into_iter()
                        .map(|lane| C220VectorLaneOutcome {
                            active: lane.active,
                            bits: lane.bits,
                            fp16_status: None,
                            fp32_status: lane.status,
                        })
                        .collect(),
                    result.stores,
                )
            }
            C220VectorReadOperation::VectorScalar {
                instruction,
                scalar,
            } => {
                let (values, stores) = evaluate_c220_vector_scalar_repeat(
                    C220VectorScalarValueInputs {
                        instruction,
                        scalar,
                        control: self.control,
                        addresses: self.addresses,
                        mask: self.mask,
                        repeat_index: self.repeat_index,
                        lane_group,
                        lane_slice: self.lane_slice,
                    },
                    &self.source_0_bytes,
                )?;
                (
                    values
                        .into_iter()
                        .enumerate()
                        .map(|(index, value)| C220VectorLaneOutcome {
                            active: lane_active(index),
                            bits: value.bits,
                            fp16_status: value.fp16_status,
                            fp32_status: value.fp32_status,
                        })
                        .collect(),
                    stores,
                )
            }
            C220VectorReadOperation::Shift { instruction, shift } => {
                let (values, stores) = evaluate_c220_shift_repeat(
                    C220ShiftValueInputs {
                        instruction,
                        shift,
                        control: self.control,
                        addresses: self.addresses,
                        mask: self.mask,
                        repeat_index: self.repeat_index,
                        lane_group,
                        lane_slice: self.lane_slice,
                    },
                    &self.source_0_bytes,
                )?;
                (
                    values
                        .into_iter()
                        .enumerate()
                        .map(|(index, bits)| C220VectorLaneOutcome {
                            active: lane_active(index),
                            bits,
                            fp16_status: None,
                            fp32_status: None,
                        })
                        .collect(),
                    stores,
                )
            }
            C220VectorReadOperation::Copy { instruction } => {
                let (values, stores) = evaluate_c220_copy_repeat(
                    C220CopyValueInputs {
                        instruction,
                        control: self.control,
                        addresses: self.addresses,
                        mask: self.mask,
                        repeat_index: self.repeat_index,
                        lane_group,
                        lane_slice: self.lane_slice,
                    },
                    &self.source_0_bytes,
                )?;
                (
                    values
                        .into_iter()
                        .enumerate()
                        .map(|(index, bits)| C220VectorLaneOutcome {
                            active: lane_active(index),
                            bits,
                            fp16_status: None,
                            fp32_status: None,
                        })
                        .collect(),
                    stores,
                )
            }
        };
        let compare_update = match &self.operation {
            C220VectorReadOperation::CompareMask { issue } => Some({
                let mut write_mask = [0_u64; 2];
                let mut values = [0_u64; 2];
                let first_lane = self.lane_slice.map_or_else(
                    || usize::from(self.lane_group.unwrap_or_default()) * 64,
                    |(first_lane, _)| first_lane,
                );
                for (index, lane) in lanes.iter().enumerate() {
                    if !lane.active {
                        continue;
                    }
                    let bit = first_lane + index;
                    write_mask[bit / 64] |= 1_u64 << (bit % 64);
                    if lane.bits != 0 {
                        values[bit / 64] |= 1_u64 << (bit % 64);
                    }
                }
                for word in 0..issue.instruction.width.lane_count() / 64 {
                    values[word] |= issue.initial_compare_mask.bits()[word] & !write_mask[word];
                    write_mask[word] = u64::MAX;
                }
                C220CompareMaskUpdate { write_mask, values }
            }),
            C220VectorReadOperation::MoveMask { issue }
                if matches!(
                    issue.instruction.direction,
                    C220MoveMaskDirection::FromMemory
                ) =>
            {
                Some(C220CompareMaskUpdate {
                    write_mask: [u64::MAX; 2],
                    values: [
                        u64::from_le_bytes(
                            self.source_0_bytes[..8].try_into().expect("low mask word"),
                        ),
                        u64::from_le_bytes(
                            self.source_0_bytes[8..16]
                                .try_into()
                                .expect("high mask word"),
                        ),
                    ],
                })
            }
            _ => None,
        };
        Ok((
            C220VectorReadSample {
                pc: self.pc,
                word: self.word,
                repeat_index: self.repeat_index,
                lane_group: self.lane_group,
                tick: self.ready_tick.expect("read sample has a ready tick"),
                accesses: self.accesses.clone(),
                read0_grants: self.port0.request.grants().to_vec(),
                read1_grants: self.port1.request.grants().to_vec(),
                source_0_bytes: self.source_0_bytes.clone(),
                source_1_bytes: self.source_1_bytes.clone(),
                destination_bytes: self.destination_bytes.clone(),
                lanes,
                conversion_lanes,
                fused_lanes,
                sort_lanes,
                compare_update,
                reduction_update,
                va_update,
            },
            stores,
        ))
    }
}
