use crate::architecture::c220::C220UbBank;
use crate::isa::c220::reduce::{
    C220ExtremumOperation, C220ExtremumOutput, C220ReductionInstruction, C220ReductionKind,
    C220ReductionWidth,
};
use crate::isa::c220::vector_scalar::C220VectorScalarOperation;
use crate::memory::ub::UbMemory;
use crate::numeric::fp32::{Fp32ValueStatus, Fp32VectorOperation, evaluate_fp32_value};
use crate::sim::c220::fp16::{C220Fp16Mode, C220Fp16Status, evaluate_c220_fp16};
use crate::sim::c220::vector::{
    C220_VECTOR_BLOCK_BYTES, C220_VECTOR_TILE_BYTES, C220VectorAddresses, C220VectorControl,
    C220VectorError, C220VectorReadAccess, C220VectorStore, check_repeat_limit,
    plan_c220_vector_read_accesses, vector_destination_address_for_width,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220ReductionIssue {
    pub pc: u64,
    pub word: u32,
    pub instruction: C220ReductionInstruction,
    pub control: C220VectorControl,
    pub addresses: C220VectorAddresses,
    pub iteration_masks: Vec<[u64; 4]>,
    pub fp16_mode: C220Fp16Mode,
    pub(crate) write_targets: Vec<C220VectorStore>,
}

impl C220ReductionIssue {
    pub fn read_accesses_for_repeat(
        &self,
        repeat_index: usize,
    ) -> Result<Vec<C220VectorReadAccess>, C220VectorError> {
        let mask = self
            .iteration_masks
            .get(repeat_index)
            .ok_or(C220VectorError::MissingMaskState)?;
        plan_c220_vector_read_accesses(
            self.control,
            self.addresses,
            repeat_index,
            mask,
            1,
            self.instruction.width.element_bytes(),
            None,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum C220ReductionValue {
    F16 { bits: u16, status: C220Fp16Status },
    F32 { bits: u32, status: Fp32ValueStatus },
}

impl C220ReductionValue {
    pub(crate) const fn zero(width: C220ReductionWidth) -> Self {
        match width {
            C220ReductionWidth::F16 => Self::F16 {
                bits: 0,
                status: C220Fp16Status {
                    nan_operand: false,
                    infinity_operand: false,
                    invalid: false,
                    overflow: false,
                    underflow: false,
                },
            },
            C220ReductionWidth::F32 => Self::F32 {
                bits: 0,
                status: Fp32ValueStatus {
                    overflow: false,
                    underflow: false,
                    nan_operand: false,
                    infinity_operand: false,
                    opposite_infinities: false,
                    zero_times_infinity: false,
                    division_by_zero: false,
                    indeterminate_division: false,
                },
            },
        }
    }

    pub(crate) fn add(self, other: Self, fp16_mode: C220Fp16Mode) -> Option<Self> {
        match (self, other) {
            (
                Self::F16 {
                    bits: first,
                    status: first_status,
                },
                Self::F16 {
                    bits: second,
                    status: second_status,
                },
            ) => {
                let result =
                    evaluate_c220_fp16(C220VectorScalarOperation::Add, first, second, fp16_mode);
                Some(Self::F16 {
                    bits: result.bits,
                    status: merge_fp16_status(first_status, second_status, result.status),
                })
            }
            (
                Self::F32 {
                    bits: first,
                    status: first_status,
                },
                Self::F32 {
                    bits: second,
                    status: second_status,
                },
            ) => {
                let result = evaluate_fp32_value(Fp32VectorOperation::Add, first, second);
                Some(Self::F32 {
                    bits: result.bits,
                    status: merge_fp32_status(first_status, second_status, result.status),
                })
            }
            _ => None,
        }
    }

    fn extremum(
        self,
        other: Self,
        operation: C220ExtremumOperation,
        fp16_mode: C220Fp16Mode,
    ) -> Option<Self> {
        let scalar_operation = match operation {
            C220ExtremumOperation::Maximum => C220VectorScalarOperation::Maximum,
            C220ExtremumOperation::Minimum => C220VectorScalarOperation::Minimum,
        };
        let fp32_operation = match operation {
            C220ExtremumOperation::Maximum => Fp32VectorOperation::Maximum,
            C220ExtremumOperation::Minimum => Fp32VectorOperation::Minimum,
        };
        match (self, other) {
            (
                Self::F16 {
                    bits: first,
                    status: first_status,
                },
                Self::F16 {
                    bits: second,
                    status: second_status,
                },
            ) => {
                let result = evaluate_c220_fp16(scalar_operation, first, second, fp16_mode);
                Some(Self::F16 {
                    bits: result.bits,
                    status: merge_fp16_status(first_status, second_status, result.status),
                })
            }
            (
                Self::F32 {
                    bits: first,
                    status: first_status,
                },
                Self::F32 {
                    bits: second,
                    status: second_status,
                },
            ) => {
                let result = evaluate_fp32_value(fp32_operation, first, second);
                Some(Self::F32 {
                    bits: result.bits,
                    status: merge_fp32_status(first_status, second_status, result.status),
                })
            }
            _ => None,
        }
    }

    pub(crate) const fn bits(self) -> u32 {
        match self {
            Self::F16 { bits, .. } => bits as u32,
            Self::F32 { bits, .. } => bits,
        }
    }

    const fn fp16_status(self) -> Option<C220Fp16Status> {
        match self {
            Self::F16 { status, .. } => Some(status),
            Self::F32 { .. } => None,
        }
    }

    const fn fp32_status(self) -> Option<Fp32ValueStatus> {
        match self {
            Self::F16 { .. } => None,
            Self::F32 { status, .. } => Some(status),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct C220IndexedReductionValue {
    value: C220ReductionValue,
    index: u32,
}

impl C220IndexedReductionValue {
    fn extremum(
        self,
        candidate: Self,
        operation: C220ExtremumOperation,
        fp16_mode: C220Fp16Mode,
    ) -> Option<Self> {
        let value = self.value.extremum(candidate.value, operation, fp16_mode)?;
        Some(Self {
            value,
            index: if value.bits() == candidate.value.bits() {
                candidate.index
            } else {
                self.index
            },
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum C220ReductionStateUpdate {
    Add(C220ReductionValue),
    Extremum {
        value: C220ReductionValue,
        index: u32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum C220ReductionState {
    Add {
        value: C220ReductionValue,
        fp16_mode: C220Fp16Mode,
    },
    Extremum {
        winner: Option<C220IndexedReductionValue>,
        operation: C220ExtremumOperation,
        fp16_mode: C220Fp16Mode,
    },
}

impl C220ReductionState {
    pub(crate) fn for_instruction(
        instruction: C220ReductionInstruction,
        fp16_mode: C220Fp16Mode,
    ) -> Option<Self> {
        match instruction.kind {
            C220ReductionKind::WholeAdd {
                write_accumulator: true,
            } => Some(Self::Add {
                value: C220ReductionValue::zero(instruction.width),
                fp16_mode,
            }),
            C220ReductionKind::WholeExtremum { operation, .. } => Some(Self::Extremum {
                winner: None,
                operation,
                fp16_mode,
            }),
            _ => None,
        }
    }

    pub(crate) fn apply(&mut self, update: C220ReductionStateUpdate) {
        match (self, update) {
            (Self::Add { value, fp16_mode }, C220ReductionStateUpdate::Add(candidate)) => {
                *value = value
                    .add(candidate, *fp16_mode)
                    .expect("reduction widths match");
            }
            (
                Self::Extremum {
                    winner,
                    operation,
                    fp16_mode,
                },
                C220ReductionStateUpdate::Extremum { value, index },
            ) => {
                let candidate = C220IndexedReductionValue { value, index };
                *winner = Some(match *winner {
                    Some(current) => current
                        .extremum(candidate, *operation, *fp16_mode)
                        .expect("reduction widths match"),
                    None => candidate,
                });
            }
            _ => unreachable!("reduction update matches its instruction state"),
        }
    }

    pub(crate) fn result(self) -> Option<(u16, u64)> {
        match self {
            Self::Add { value, .. } => Some((87, u64::from(value.bits()))),
            Self::Extremum {
                winner: Some(winner),
                ..
            } => Some((
                63,
                u64::from(winner.value.bits()) | (u64::from(winner.index) << 32),
            )),
            Self::Extremum { winner: None, .. } => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220ReductionLaneOutcome {
    pub active: bool,
    pub bits: u32,
    pub fp16_status: Option<C220Fp16Status>,
    pub fp32_status: Option<Fp32ValueStatus>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct C220ReductionOutcome {
    pub lanes: Vec<C220ReductionLaneOutcome>,
    pub stores: Vec<C220VectorStore>,
    pub state_update: Option<C220ReductionStateUpdate>,
}

pub fn plan_c220_reduction_issue(
    pc: u64,
    word: u32,
    control: C220VectorControl,
    addresses: C220VectorAddresses,
    iteration_masks: &[[u64; 4]],
    fp16_mode: C220Fp16Mode,
    ub: &UbMemory,
) -> Result<C220ReductionIssue, C220VectorError> {
    check_repeat_limit(iteration_masks.len())?;
    let instruction = C220ReductionInstruction::decode(word)
        .ok_or(C220VectorError::UnsupportedWord { pc, word })?;
    let element_bytes = instruction.width.element_bytes();
    let lanes = instruction.width.lanes();
    let mut write_targets = Vec::new();
    for (repeat_index, mask) in iteration_masks.iter().enumerate() {
        for access in plan_c220_vector_read_accesses(
            control,
            addresses,
            repeat_index,
            mask,
            1,
            element_bytes,
            None,
        )? {
            ub.check_range(access.address, C220_VECTOR_BLOCK_BYTES)?;
        }
        match instruction.kind {
            C220ReductionKind::WholeAdd {
                write_accumulator: true,
            } => {}
            C220ReductionKind::WholeAdd {
                write_accumulator: false,
            } => push_target(
                &mut write_targets,
                control,
                addresses,
                repeat_index,
                0,
                element_bytes,
                ub,
            )?,
            C220ReductionKind::WholeExtremum { output, .. } => match output {
                C220ExtremumOutput::ValueIndex | C220ExtremumOutput::IndexValue => {
                    push_target(
                        &mut write_targets,
                        control,
                        addresses,
                        repeat_index,
                        0,
                        element_bytes,
                        ub,
                    )?;
                    push_target(
                        &mut write_targets,
                        control,
                        addresses,
                        repeat_index,
                        1,
                        element_bytes,
                        ub,
                    )?;
                }
                C220ExtremumOutput::Value => push_target(
                    &mut write_targets,
                    control,
                    addresses,
                    repeat_index,
                    0,
                    element_bytes,
                    ub,
                )?,
                C220ExtremumOutput::Index => push_target(
                    &mut write_targets,
                    control,
                    addresses,
                    repeat_index,
                    0,
                    4,
                    ub,
                )?,
            },
            C220ReductionKind::GroupAdd | C220ReductionKind::GroupExtremum { .. } => {
                let group_lanes = C220_VECTOR_BLOCK_BYTES / usize::from(element_bytes);
                for group in 0..8 {
                    if mask_range_active(mask, group * group_lanes, group_lanes) {
                        push_target(
                            &mut write_targets,
                            control,
                            addresses,
                            repeat_index,
                            group,
                            element_bytes,
                            ub,
                        )?;
                    }
                }
            }
            C220ReductionKind::PairAdd => {
                for pair in 0..lanes / 2 {
                    if mask_range_active(mask, pair * 2, 2) {
                        push_target(
                            &mut write_targets,
                            control,
                            addresses,
                            repeat_index,
                            pair,
                            element_bytes,
                            ub,
                        )?;
                    }
                }
            }
        }
    }
    Ok(C220ReductionIssue {
        pc,
        word,
        instruction,
        control,
        addresses,
        iteration_masks: iteration_masks.to_vec(),
        fp16_mode,
        write_targets,
    })
}

pub(crate) fn evaluate_c220_reduction_repeat(
    issue: &C220ReductionIssue,
    repeat_index: usize,
    source_bytes: &[u8],
) -> Result<C220ReductionOutcome, C220VectorError> {
    if source_bytes.len() != C220_VECTOR_TILE_BYTES {
        return Err(C220VectorError::InvalidSourceTile {
            actual: source_bytes.len(),
            expected: C220_VECTOR_TILE_BYTES,
        });
    }
    let mask = issue
        .iteration_masks
        .get(repeat_index)
        .ok_or(C220VectorError::InvalidRepeatIndex(repeat_index))?;
    let mut lanes = Vec::new();
    let mut stores = Vec::new();
    let state_update = match issue.instruction.kind {
        C220ReductionKind::WholeAdd { write_accumulator } => {
            let values = decode_add_values(issue.instruction.width, source_bytes, mask);
            let value = balanced_add(&values, issue.fp16_mode);
            lanes.push(value_lane(true, value));
            if write_accumulator {
                Some(C220ReductionStateUpdate::Add(value))
            } else {
                stores.push(value_store(issue, repeat_index, 0, value)?);
                None
            }
        }
        C220ReductionKind::GroupAdd => {
            let values = decode_add_values(issue.instruction.width, source_bytes, mask);
            let group_lanes =
                C220_VECTOR_BLOCK_BYTES / usize::from(issue.instruction.width.element_bytes());
            for (index, group) in values.chunks_exact(group_lanes).enumerate() {
                let active = mask_range_active(mask, index * group_lanes, group_lanes);
                let value = balanced_add(group, issue.fp16_mode);
                lanes.push(value_lane(active, value));
                if active {
                    stores.push(value_store(issue, repeat_index, index, value)?);
                }
            }
            None
        }
        C220ReductionKind::PairAdd => {
            let values = decode_add_values(issue.instruction.width, source_bytes, mask);
            for (index, pair) in values.chunks_exact(2).enumerate() {
                let active = mask_range_active(mask, index * 2, 2);
                let value = pair[0]
                    .add(pair[1], issue.fp16_mode)
                    .expect("pair widths match");
                lanes.push(value_lane(active, value));
                if active {
                    stores.push(value_store(issue, repeat_index, index, value)?);
                }
            }
            None
        }
        C220ReductionKind::WholeExtremum { operation, output } => {
            let winner = extremum_range(
                issue.instruction.width,
                source_bytes,
                mask,
                0,
                issue.instruction.width.lanes(),
                operation,
                issue.fp16_mode,
            )
            .unwrap_or(C220IndexedReductionValue {
                value: C220ReductionValue::zero(issue.instruction.width),
                index: 0,
            });
            append_extremum_output(issue, repeat_index, output, winner, &mut lanes, &mut stores)?;
            let global_index = repeat_index
                .checked_mul(issue.instruction.width.lanes())
                .and_then(|base| base.checked_add(winner.index as usize))
                .expect("C220 repeat limits fit a 32-bit lane index");
            Some(C220ReductionStateUpdate::Extremum {
                value: winner.value,
                index: global_index as u32,
            })
        }
        C220ReductionKind::GroupExtremum { operation } => {
            let group_lanes =
                C220_VECTOR_BLOCK_BYTES / usize::from(issue.instruction.width.element_bytes());
            for group in 0..8 {
                let start = group * group_lanes;
                let winner = extremum_range(
                    issue.instruction.width,
                    source_bytes,
                    mask,
                    start,
                    group_lanes,
                    operation,
                    issue.fp16_mode,
                );
                let value = winner.map_or_else(
                    || C220ReductionValue::zero(issue.instruction.width),
                    |winner| winner.value,
                );
                lanes.push(value_lane(winner.is_some(), value));
                if winner.is_some() {
                    stores.push(value_store(issue, repeat_index, group, value)?);
                }
            }
            None
        }
    };
    Ok(C220ReductionOutcome {
        lanes,
        stores,
        state_update,
    })
}

fn decode_add_values(
    width: C220ReductionWidth,
    source_bytes: &[u8],
    mask: &[u64; 4],
) -> Vec<C220ReductionValue> {
    let element_bytes = usize::from(width.element_bytes());
    source_bytes
        .chunks_exact(element_bytes)
        .enumerate()
        .map(|(lane, bytes)| {
            if !mask_bit(mask, lane) {
                return C220ReductionValue::zero(width);
            }
            match width {
                C220ReductionWidth::F16 => C220ReductionValue::F16 {
                    bits: u16::from_le_bytes(bytes.try_into().expect("two-byte lane")),
                    status: C220Fp16Status::default(),
                },
                C220ReductionWidth::F32 => C220ReductionValue::F32 {
                    bits: u32::from_le_bytes(bytes.try_into().expect("four-byte lane")),
                    status: Fp32ValueStatus::default(),
                },
            }
        })
        .collect()
}

fn decode_value(width: C220ReductionWidth, source_bytes: &[u8], lane: usize) -> C220ReductionValue {
    let element_bytes = usize::from(width.element_bytes());
    let start = lane * element_bytes;
    let bytes = &source_bytes[start..start + element_bytes];
    match width {
        C220ReductionWidth::F16 => C220ReductionValue::F16 {
            bits: u16::from_le_bytes(bytes.try_into().expect("two-byte lane")),
            status: C220Fp16Status::default(),
        },
        C220ReductionWidth::F32 => C220ReductionValue::F32 {
            bits: u32::from_le_bytes(bytes.try_into().expect("four-byte lane")),
            status: Fp32ValueStatus::default(),
        },
    }
}

fn extremum_range(
    width: C220ReductionWidth,
    source_bytes: &[u8],
    mask: &[u64; 4],
    start: usize,
    count: usize,
    operation: C220ExtremumOperation,
    fp16_mode: C220Fp16Mode,
) -> Option<C220IndexedReductionValue> {
    let mut winner: Option<C220IndexedReductionValue> = None;
    for lane in start..start + count {
        if !mask_bit(mask, lane) {
            continue;
        }
        let candidate = C220IndexedReductionValue {
            value: decode_value(width, source_bytes, lane),
            index: lane as u32,
        };
        winner = Some(match winner {
            Some(current) => current
                .extremum(candidate, operation, fp16_mode)
                .expect("reduction widths match"),
            None => candidate,
        });
    }
    winner
}

const fn value_lane(active: bool, value: C220ReductionValue) -> C220ReductionLaneOutcome {
    C220ReductionLaneOutcome {
        active,
        bits: value.bits(),
        fp16_status: value.fp16_status(),
        fp32_status: value.fp32_status(),
    }
}

const fn index_lane(index: u32) -> C220ReductionLaneOutcome {
    C220ReductionLaneOutcome {
        active: true,
        bits: index,
        fp16_status: None,
        fp32_status: None,
    }
}

fn value_store(
    issue: &C220ReductionIssue,
    repeat_index: usize,
    lane_index: usize,
    value: C220ReductionValue,
) -> Result<C220VectorStore, C220VectorError> {
    reduction_store(
        issue,
        repeat_index,
        lane_index,
        issue.instruction.width.element_bytes(),
        value.bits(),
    )
}

fn index_store(
    issue: &C220ReductionIssue,
    repeat_index: usize,
    lane_index: usize,
    width_bytes: u8,
    index: u32,
) -> Result<C220VectorStore, C220VectorError> {
    reduction_store(issue, repeat_index, lane_index, width_bytes, index)
}

fn reduction_store(
    issue: &C220ReductionIssue,
    repeat_index: usize,
    lane_index: usize,
    width_bytes: u8,
    bits: u32,
) -> Result<C220VectorStore, C220VectorError> {
    let address = vector_destination_address_for_width(
        issue.control,
        issue.addresses,
        repeat_index,
        lane_index,
        width_bytes,
    )?;
    Ok(C220VectorStore {
        repeat_index,
        lane_index,
        address,
        bank: C220UbBank::from_address(address),
        width_bytes,
        data: bits.to_le_bytes(),
    })
}

fn append_extremum_output(
    issue: &C220ReductionIssue,
    repeat_index: usize,
    output: C220ExtremumOutput,
    winner: C220IndexedReductionValue,
    lanes: &mut Vec<C220ReductionLaneOutcome>,
    stores: &mut Vec<C220VectorStore>,
) -> Result<(), C220VectorError> {
    let element_bytes = issue.instruction.width.element_bytes();
    match output {
        C220ExtremumOutput::ValueIndex => {
            lanes.push(value_lane(true, winner.value));
            lanes.push(index_lane(winner.index));
            stores.push(value_store(issue, repeat_index, 0, winner.value)?);
            stores.push(index_store(
                issue,
                repeat_index,
                1,
                element_bytes,
                winner.index,
            )?);
        }
        C220ExtremumOutput::IndexValue => {
            lanes.push(index_lane(winner.index));
            lanes.push(value_lane(true, winner.value));
            stores.push(index_store(
                issue,
                repeat_index,
                0,
                element_bytes,
                winner.index,
            )?);
            stores.push(value_store(issue, repeat_index, 1, winner.value)?);
        }
        C220ExtremumOutput::Value => {
            lanes.push(value_lane(true, winner.value));
            stores.push(value_store(issue, repeat_index, 0, winner.value)?);
        }
        C220ExtremumOutput::Index => {
            lanes.push(index_lane(winner.index));
            stores.push(index_store(issue, repeat_index, 0, 4, winner.index)?);
        }
    }
    Ok(())
}

fn balanced_add(values: &[C220ReductionValue], fp16_mode: C220Fp16Mode) -> C220ReductionValue {
    debug_assert!(values.len().is_power_of_two());
    let mut level = values.to_vec();
    while level.len() > 1 {
        level = level
            .chunks_exact(2)
            .map(|pair| pair[0].add(pair[1], fp16_mode).expect("tree widths match"))
            .collect();
    }
    level[0]
}

fn push_target(
    targets: &mut Vec<C220VectorStore>,
    control: C220VectorControl,
    addresses: C220VectorAddresses,
    repeat_index: usize,
    lane_index: usize,
    element_bytes: u8,
    ub: &UbMemory,
) -> Result<(), C220VectorError> {
    let address = vector_destination_address_for_width(
        control,
        addresses,
        repeat_index,
        lane_index,
        element_bytes,
    )?;
    ub.check_range(address, usize::from(element_bytes))?;
    targets.push(C220VectorStore {
        repeat_index,
        lane_index,
        address,
        bank: C220UbBank::from_address(address),
        width_bytes: element_bytes,
        data: [0; 4],
    });
    Ok(())
}

fn mask_range_active(mask: &[u64; 4], start: usize, count: usize) -> bool {
    (start..start + count).any(|lane| mask_bit(mask, lane))
}

fn mask_bit(mask: &[u64; 4], lane: usize) -> bool {
    mask[lane / 64] & (1_u64 << (lane % 64)) != 0
}

const fn merge_fp16_status(
    first: C220Fp16Status,
    second: C220Fp16Status,
    current: C220Fp16Status,
) -> C220Fp16Status {
    C220Fp16Status {
        nan_operand: first.nan_operand || second.nan_operand || current.nan_operand,
        infinity_operand: first.infinity_operand
            || second.infinity_operand
            || current.infinity_operand,
        invalid: first.invalid || second.invalid || current.invalid,
        overflow: first.overflow || second.overflow || current.overflow,
        underflow: first.underflow || second.underflow || current.underflow,
    }
}

const fn merge_fp32_status(
    first: Fp32ValueStatus,
    second: Fp32ValueStatus,
    current: Fp32ValueStatus,
) -> Fp32ValueStatus {
    Fp32ValueStatus {
        overflow: first.overflow || second.overflow || current.overflow,
        underflow: first.underflow || second.underflow || current.underflow,
        nan_operand: first.nan_operand || second.nan_operand || current.nan_operand,
        infinity_operand: first.infinity_operand
            || second.infinity_operand
            || current.infinity_operand,
        opposite_infinities: first.opposite_infinities
            || second.opposite_infinities
            || current.opposite_infinities,
        zero_times_infinity: first.zero_times_infinity
            || second.zero_times_infinity
            || current.zero_times_infinity,
        division_by_zero: first.division_by_zero
            || second.division_by_zero
            || current.division_by_zero,
        indeterminate_division: first.indeterminate_division
            || second.indeterminate_division
            || current.indeterminate_division,
    }
}
