use crate::architecture::c220::C220UbBank;
use crate::isa::c220::conversion::{C220ConversionRound, C220ConversionType};
use crate::isa::c220::fused::{C220FusedFormat, C220FusedInstruction, C220FusedOperation};
use crate::isa::c220::vector_scalar::C220VectorScalarOperation;
use crate::memory::ub::UbMemory;
use crate::numeric::fp32::{Fp32VectorOperation, evaluate_fp32_value};
use crate::sim::c220::fp16::{
    C220Fp16Mode, C220Fp16Status, evaluate_c220_fp16, evaluate_c220_fp16_relu,
};

use super::conversion::{C220ConversionStatus, convert_ordinary, deq_s16_to_b8, deq_s32_to_f16};
use super::{
    C220_VECTOR_BLOCK_BYTES, C220_VECTOR_BLOCK_COUNT, C220VectorAddresses, C220VectorControl,
    C220VectorError, C220VectorReadAccess, C220VectorStore, check_repeat_limit,
    plan_c220_vector_read_accesses, store_data, vector_destination_address_for_width,
};

const F32_SIGN: u32 = 0x8000_0000;
const F32_INFINITY: u32 = 0x7f80_0000;
const F32_MAX_FINITE: u32 = 0x7f7f_ffff;
const F32_CANONICAL_NAN: u32 = 0x7fff_ffff;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FusedLaneOutcome {
    pub active: bool,
    pub source_0_bits: u32,
    pub source_1_bits: u32,
    pub result_bits: u16,
    pub status: C220ConversionStatus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220FusedIssue {
    pub pc: u64,
    pub word: u32,
    pub instruction: C220FusedInstruction,
    pub control: C220VectorControl,
    pub addresses: C220VectorAddresses,
    pub iteration_masks: Vec<[u64; 4]>,
    pub fp16_mode: C220Fp16Mode,
    pub arithmetic_saturating: bool,
    pub integer_saturating: bool,
    pub descriptor_address: u64,
    pub deq_scale: u16,
    pub(crate) write_targets: Vec<C220VectorStore>,
}

#[derive(Debug, Clone, Copy)]
pub struct C220FusedIssueInputs<'a> {
    pub pc: u64,
    pub word: u32,
    pub control: C220VectorControl,
    pub addresses: C220VectorAddresses,
    pub iteration_masks: &'a [[u64; 4]],
    pub fp16_mode: C220Fp16Mode,
    pub arithmetic_saturating: bool,
    pub integer_saturating: bool,
    pub descriptor_address: u64,
    pub deq_scale: u16,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct C220FusedValueInputs<'a> {
    pub issue: &'a C220FusedIssue,
    pub repeat_index: usize,
    pub lane_group: u8,
    pub lane_slice: Option<(usize, usize)>,
}

impl C220FusedIssue {
    pub fn read_accesses_for_repeat(
        &self,
        repeat_index: usize,
        lane_group: u8,
    ) -> Result<Vec<C220VectorReadAccess>, C220VectorError> {
        if lane_group >= self.instruction.lane_groups() {
            return Err(C220VectorError::InvalidLaneGroup(lane_group));
        }
        let mask = self
            .iteration_masks
            .get(repeat_index)
            .ok_or(C220VectorError::MissingMaskState)?;
        if self.instruction.format != C220FusedFormat::VectorDeqS16ToB8 {
            return plan_c220_vector_read_accesses(
                self.control,
                self.addresses,
                repeat_index,
                mask,
                2,
                self.instruction.source_element_bytes(),
                Some(lane_group),
            );
        }
        if mask[0] == 0 {
            return Ok(Vec::new());
        }
        let mut accesses = Vec::with_capacity(C220_VECTOR_BLOCK_COUNT * 2 + 4);
        for (source_index, base, block_stride, repeat_stride) in [
            (
                0,
                self.addresses.source_0,
                self.control.source_0_block_stride,
                self.control.source_0_repeat_stride,
            ),
            (
                1,
                self.addresses.source_1,
                self.control.source_1_block_stride,
                self.control.source_1_repeat_stride,
            ),
        ] {
            for block in 0..C220_VECTOR_BLOCK_COUNT {
                let address = source_block_address(
                    base,
                    repeat_index,
                    repeat_stride,
                    block,
                    block_stride,
                    source_index,
                )?;
                accesses.push(C220VectorReadAccess {
                    source_index,
                    block_index: block as u8,
                    buffer_offset: (block * C220_VECTOR_BLOCK_BYTES) as u16,
                    bytes: C220_VECTOR_BLOCK_BYTES as u16,
                    address,
                    active_lane_mask: u16::MAX,
                });
            }
        }
        for block in 0..4 {
            let address = self
                .descriptor_address
                .checked_add((block * C220_VECTOR_BLOCK_BYTES) as u64)
                .ok_or(C220VectorError::SourceAddressOverflow {
                    source_index: 2,
                    base: self.descriptor_address,
                    block,
                })?;
            accesses.push(C220VectorReadAccess {
                source_index: 2,
                block_index: block as u8,
                buffer_offset: (block * C220_VECTOR_BLOCK_BYTES) as u16,
                bytes: C220_VECTOR_BLOCK_BYTES as u16,
                address,
                active_lane_mask: u16::MAX,
            });
        }
        Ok(accesses)
    }

    pub fn logical_lane_for_store(&self, store: &C220VectorStore) -> Option<usize> {
        match self.instruction.format {
            C220FusedFormat::VectorDeqS16ToB8 => {
                let block = store.lane_index / C220_VECTOR_BLOCK_BYTES;
                let within_block = store.lane_index % C220_VECTOR_BLOCK_BYTES;
                let half_offset = usize::from(self.instruction.destination_high) * 16;
                (half_offset..half_offset + 16)
                    .contains(&within_block)
                    .then_some(block * 16 + within_block - half_offset)
            }
            C220FusedFormat::F16ToS8 | C220FusedFormat::F16ToU8
                if self.instruction.operation == C220FusedOperation::Multiply =>
            {
                store
                    .lane_index
                    .checked_sub(usize::from(self.instruction.destination_high) * 128)
            }
            _ => Some(store.lane_index),
        }
    }

    pub fn stores_for_lane_group(
        &self,
        repeat_index: usize,
        lane_group: u8,
    ) -> Vec<C220VectorStore> {
        self.write_targets
            .iter()
            .copied()
            .filter(|store| store.repeat_index == repeat_index)
            .filter(|store| {
                self.logical_lane_for_store(store)
                    .is_some_and(|lane| lane / 64 == usize::from(lane_group))
            })
            .collect()
    }
}

pub fn plan_c220_fused_issue(
    inputs: C220FusedIssueInputs<'_>,
    ub: &UbMemory,
) -> Result<C220FusedIssue, C220VectorError> {
    check_repeat_limit(inputs.iteration_masks.len())?;
    let instruction =
        C220FusedInstruction::decode(inputs.word).ok_or(C220VectorError::UnsupportedWord {
            pc: inputs.pc,
            word: inputs.word,
        })?;
    let mut iteration_masks = inputs.iteration_masks.to_vec();
    for mask in &mut iteration_masks {
        mask[instruction.lane_count().div_ceil(64)..].fill(0);
    }
    let mut issue = C220FusedIssue {
        pc: inputs.pc,
        word: inputs.word,
        instruction,
        control: inputs.control,
        addresses: inputs.addresses,
        iteration_masks,
        fp16_mode: inputs.fp16_mode,
        arithmetic_saturating: inputs.arithmetic_saturating,
        integer_saturating: inputs.integer_saturating,
        descriptor_address: inputs.descriptor_address,
        deq_scale: inputs.deq_scale,
        write_targets: Vec::new(),
    };
    issue.write_targets = plan_write_targets(&issue, ub)?;
    Ok(issue)
}

fn plan_write_targets(
    issue: &C220FusedIssue,
    ub: &UbMemory,
) -> Result<Vec<C220VectorStore>, C220VectorError> {
    let mut targets =
        Vec::with_capacity(issue.instruction.lane_count() * issue.iteration_masks.len());
    for (repeat_index, mask) in issue.iteration_masks.iter().enumerate() {
        for group in 0..issue.instruction.lane_groups() {
            for access in issue.read_accesses_for_repeat(repeat_index, group)? {
                ub.check_range(access.address, usize::from(access.bytes))?;
            }
        }
        for logical_lane in 0..issue.instruction.lane_count() {
            if mask[logical_lane / 64] & (1_u64 << (logical_lane % 64)) == 0 {
                continue;
            }
            let lane_index = destination_lane_index(issue.instruction, logical_lane);
            let width = issue.instruction.destination_element_bytes();
            let address = vector_destination_address_for_width(
                issue.control,
                issue.addresses,
                repeat_index,
                lane_index,
                width,
            )?;
            ub.check_range(address, usize::from(width))?;
            targets.push(C220VectorStore {
                repeat_index,
                lane_index,
                address,
                bank: C220UbBank::from_address(address),
                width_bytes: width,
                data: [0; 8],
            });
        }
    }
    Ok(targets)
}

pub(crate) fn evaluate_c220_fused_repeat(
    inputs: C220FusedValueInputs<'_>,
    source_0_bytes: &[u8],
    source_1_bytes: &[u8],
    descriptor_bytes: &[u8],
    ub: &UbMemory,
) -> Result<(Vec<C220FusedLaneOutcome>, Vec<C220VectorStore>), C220VectorError> {
    let issue = inputs.issue;
    if inputs.lane_group >= issue.instruction.lane_groups() {
        return Err(C220VectorError::InvalidLaneGroup(inputs.lane_group));
    }
    let required_source_bytes =
        issue.instruction.lane_count() * usize::from(issue.instruction.source_element_bytes());
    for bytes in [source_0_bytes, source_1_bytes] {
        if bytes.len() < required_source_bytes {
            return Err(C220VectorError::InvalidSourceTile {
                actual: bytes.len(),
                expected: required_source_bytes,
            });
        }
    }
    if issue.instruction.format == C220FusedFormat::VectorDeqS16ToB8 && descriptor_bytes.len() < 128
    {
        return Err(C220VectorError::InvalidSourceTile {
            actual: descriptor_bytes.len(),
            expected: 128,
        });
    }
    let mask = issue
        .iteration_masks
        .get(inputs.repeat_index)
        .ok_or(C220VectorError::MissingMaskState)?;
    let mut lanes = Vec::with_capacity(issue.instruction.lane_count());
    let mut stores = Vec::with_capacity(64);
    for lane_index in 0..issue.instruction.lane_count() {
        let source_0 = read_lane(
            source_0_bytes,
            lane_index,
            issue.instruction.source_element_bytes(),
        );
        let source_1 = read_lane(
            source_1_bytes,
            lane_index,
            issue.instruction.source_element_bytes(),
        );
        let in_uop = inputs.lane_slice.map_or_else(
            || lane_index / 64 == usize::from(inputs.lane_group),
            |(first_lane, lane_count)| {
                lane_index >= first_lane && lane_index < first_lane + lane_count
            },
        );
        let active = in_uop && mask[lane_index / 64] & (1_u64 << (lane_index % 64)) != 0;
        if !active {
            lanes.push(C220FusedLaneOutcome {
                active,
                source_0_bits: source_0,
                source_1_bits: source_1,
                result_bits: 0,
                status: Default::default(),
            });
            continue;
        }
        let (result, status) =
            evaluate_lane(issue, lane_index, source_0, source_1, descriptor_bytes);
        let lane_index = destination_lane_index(issue.instruction, lane_index);
        let width = issue.instruction.destination_element_bytes();
        let address = vector_destination_address_for_width(
            issue.control,
            issue.addresses,
            inputs.repeat_index,
            lane_index,
            width,
        )?;
        ub.check_range(address, usize::from(width))?;
        stores.push(C220VectorStore {
            repeat_index: inputs.repeat_index,
            lane_index,
            address,
            bank: C220UbBank::from_address(address),
            width_bytes: width,
            data: store_data(result.to_le_bytes()),
        });
        lanes.push(C220FusedLaneOutcome {
            active,
            source_0_bits: source_0,
            source_1_bits: source_1,
            result_bits: result,
            status,
        });
    }
    Ok((lanes, stores))
}

fn evaluate_lane(
    issue: &C220FusedIssue,
    lane_index: usize,
    first: u32,
    second: u32,
    descriptors: &[u8],
) -> (u16, C220ConversionStatus) {
    match issue.instruction.format {
        C220FusedFormat::S16ToS8 => {
            let (relu, overflow) =
                integer_add_relu(issue, first as u16 as i16, second as u16 as i16);
            let result = if issue.integer_saturating {
                relu.min(i16::from(i8::MAX)) as u8
            } else {
                relu as u8
            };
            (
                u16::from(result),
                C220ConversionStatus {
                    overflow: overflow || issue.integer_saturating && relu > i16::from(i8::MAX),
                    ..Default::default()
                },
            )
        }
        C220FusedFormat::F16ToS8 | C220FusedFormat::F16ToU8 => {
            let operation = match issue.instruction.operation {
                C220FusedOperation::Multiply => C220VectorScalarOperation::Multiply,
                C220FusedOperation::AddRelu | C220FusedOperation::SubtractRelu => {
                    C220VectorScalarOperation::Add
                }
                C220FusedOperation::AddDeqRelu => {
                    unreachable!("VADDDEQRELU does not use fp16 arithmetic")
                }
            };
            let second = if issue.instruction.operation == C220FusedOperation::SubtractRelu {
                second as u16 ^ 0x8000
            } else {
                second as u16
            };
            let arithmetic = evaluate_c220_fp16(operation, first as u16, second, issue.fp16_mode);
            let (bits, fp_status) = if issue.instruction.operation == C220FusedOperation::Multiply {
                (arithmetic.bits, arithmetic.status)
            } else {
                let relu = evaluate_c220_fp16_relu(arithmetic.bits, C220Fp16Mode::NonSaturating);
                (relu.bits, merge_fp16_status(arithmetic.status, relu.status))
            };
            let destination = if issue.instruction.format == C220FusedFormat::F16ToS8 {
                C220ConversionType::S8
            } else {
                C220ConversionType::U8
            };
            let (result, conversion_status) = convert_ordinary(
                C220ConversionType::F16,
                destination,
                C220ConversionRound::NearestEven,
                u64::from(bits),
                issue.fp16_mode,
                issue.integer_saturating,
            );
            (
                result as u16,
                merge_conversion_status(conversion_status, fp_status),
            )
        }
        C220FusedFormat::F32ToF16 => {
            let operation = match issue.instruction.operation {
                C220FusedOperation::AddRelu => Fp32VectorOperation::Add,
                C220FusedOperation::SubtractRelu => Fp32VectorOperation::Subtract,
                C220FusedOperation::Multiply => unreachable!("F32 VMULCONV is not encoded"),
                C220FusedOperation::AddDeqRelu => {
                    unreachable!("VADDDEQRELU uses s32 sources")
                }
            };
            let arithmetic = evaluate_fp32_value(operation, first, second);
            let bits = fp32_relu(arithmetic.bits, issue.fp16_mode);
            let (result, mut status) = convert_ordinary(
                C220ConversionType::F32,
                C220ConversionType::F16,
                C220ConversionRound::NearestEven,
                u64::from(bits),
                issue.fp16_mode,
                issue.integer_saturating,
            );
            status.nan_operand |= arithmetic.status.nan_operand;
            status.infinity_operand |= arithmetic.status.infinity_operand;
            status.invalid |= arithmetic.status.invalid || arithmetic.status.opposite_infinities;
            status.overflow |= arithmetic.status.overflow;
            status.underflow |= arithmetic.status.underflow;
            (result as u16, status)
        }
        C220FusedFormat::S32ToF16 => {
            let first = first as i32;
            let second = second as i32;
            let exact = i64::from(first) + i64::from(second);
            let add_overflow = exact < i64::from(i32::MIN) || exact > i64::from(i32::MAX);
            let sum = if add_overflow && issue.arithmetic_saturating {
                exact.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
            } else {
                exact as i32
            };
            let (bits, mut status) = deq_s32_to_f16(sum, issue.deq_scale, issue.fp16_mode);
            let scale_magnitude = issue.deq_scale & 0x7fff;
            if scale_magnitude < 0x7c00 {
                status.overflow |= add_overflow;
            }
            let relu = evaluate_c220_fp16_relu(bits as u16, C220Fp16Mode::NonSaturating);
            (relu.bits, status)
        }
        C220FusedFormat::VectorDeqS16ToB8 => {
            let (relu, overflow) =
                integer_add_relu(issue, first as u16 as i16, second as u16 as i16);
            let offset = lane_index % 16 * 8;
            let descriptor = u64::from_le_bytes(
                descriptors[offset..offset + 8]
                    .try_into()
                    .expect("VDEQ descriptor"),
            );
            let (result, mut status) = deq_s16_to_b8(relu, descriptor);
            status.overflow |= overflow;
            (result as u16, status)
        }
    }
}

fn integer_add_relu(issue: &C220FusedIssue, first: i16, second: i16) -> (i16, bool) {
    let exact = match issue.instruction.operation {
        C220FusedOperation::AddRelu => i32::from(first) + i32::from(second),
        C220FusedOperation::SubtractRelu => i32::from(first) - i32::from(second),
        C220FusedOperation::Multiply => unreachable!("integer fused multiply is not encoded"),
        C220FusedOperation::AddDeqRelu => {
            unreachable!("VADDDEQRELU uses s32 sources")
        }
    };
    let overflow = exact < i32::from(i16::MIN) || exact > i32::from(i16::MAX);
    let value = if overflow && issue.arithmetic_saturating {
        exact.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
    } else {
        exact as i16
    };
    (value.max(0), overflow)
}

fn fp32_relu(bits: u32, mode: C220Fp16Mode) -> u32 {
    let magnitude = bits & !F32_SIGN;
    if magnitude > F32_INFINITY {
        if mode == C220Fp16Mode::Saturating {
            0
        } else {
            F32_CANONICAL_NAN
        }
    } else if bits & F32_SIGN != 0 {
        0
    } else if magnitude == F32_INFINITY && mode == C220Fp16Mode::Saturating {
        F32_MAX_FINITE
    } else {
        bits
    }
}

fn destination_lane_index(instruction: C220FusedInstruction, logical_lane: usize) -> usize {
    match instruction.format {
        C220FusedFormat::VectorDeqS16ToB8 => {
            logical_lane / 16 * C220_VECTOR_BLOCK_BYTES
                + usize::from(instruction.destination_high) * 16
                + logical_lane % 16
        }
        C220FusedFormat::F16ToS8 | C220FusedFormat::F16ToU8
            if instruction.operation == C220FusedOperation::Multiply =>
        {
            logical_lane + usize::from(instruction.destination_high) * 128
        }
        _ => logical_lane,
    }
}

fn source_block_address(
    base: u64,
    repeat_index: usize,
    repeat_stride: u16,
    block: usize,
    block_stride: u16,
    source_index: u8,
) -> Result<u64, C220VectorError> {
    let offset = repeat_index as u64 * u64::from(repeat_stride) * C220_VECTOR_BLOCK_BYTES as u64
        + block as u64 * u64::from(block_stride) * C220_VECTOR_BLOCK_BYTES as u64;
    base.checked_add(offset)
        .ok_or(C220VectorError::SourceAddressOverflow {
            source_index,
            base,
            block,
        })
}

fn read_lane(bytes: &[u8], lane_index: usize, width: u8) -> u32 {
    let at = lane_index * usize::from(width);
    match width {
        2 => u32::from(u16::from_le_bytes([bytes[at], bytes[at + 1]])),
        4 => u32::from_le_bytes(bytes[at..at + 4].try_into().expect("u32 lane")),
        _ => unreachable!("fused source width"),
    }
}

const fn merge_fp16_status(first: C220Fp16Status, second: C220Fp16Status) -> C220Fp16Status {
    C220Fp16Status {
        nan_operand: first.nan_operand || second.nan_operand,
        infinity_operand: first.infinity_operand || second.infinity_operand,
        invalid: first.invalid || second.invalid,
        overflow: first.overflow || second.overflow,
        underflow: first.underflow || second.underflow,
    }
}

const fn merge_conversion_status(
    mut conversion: C220ConversionStatus,
    arithmetic: C220Fp16Status,
) -> C220ConversionStatus {
    conversion.nan_operand |= arithmetic.nan_operand;
    conversion.infinity_operand |= arithmetic.infinity_operand;
    conversion.invalid |= arithmetic.invalid;
    conversion.overflow |= arithmetic.overflow;
    conversion.underflow |= arithmetic.underflow;
    conversion
}
