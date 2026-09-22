use crate::architecture::c220::C220UbBank;
use crate::isa::c220::conversion::{
    C220ConversionInstruction, C220ConversionKind, C220ConversionRound, C220ConversionType,
};
use crate::memory::ub::UbMemory;
use crate::sim::c220::fp16::{C220Fp16Mode, round_finite_to_f16, to_f64};

use super::{
    C220_VECTOR_BLOCK_BYTES, C220VectorAddresses, C220VectorControl, C220VectorError,
    C220VectorReadAccess, C220VectorStore, check_repeat_limit, store_data,
    vector_destination_address_for_width,
};

const F16_SIGN: u16 = 0x8000;
const F16_INFINITY: u16 = 0x7c00;
const F16_FRACTION: u16 = 0x03ff;
const F16_MAX_FINITE: u16 = 0x7bff;
const F16_CANONICAL_NAN: u16 = 0x7fff;
const BF16_SIGN: u16 = 0x8000;
const BF16_INFINITY: u16 = 0x7f80;
const BF16_FRACTION: u16 = 0x007f;
const BF16_MAX_FINITE: u16 = 0x7f7f;
const BF16_CANONICAL_NAN: u16 = 0x7fff;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct C220ConversionStatus {
    pub nan_operand: bool,
    pub infinity_operand: bool,
    pub invalid: bool,
    pub overflow: bool,
    pub underflow: bool,
    pub inexact: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220ConversionLaneOutcome {
    pub active: bool,
    pub source_bits: u64,
    pub result_bits: u64,
    pub status: C220ConversionStatus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220ConversionIssue {
    pub pc: u64,
    pub word: u32,
    pub instruction: C220ConversionInstruction,
    pub control: C220VectorControl,
    pub addresses: C220VectorAddresses,
    pub iteration_masks: Vec<[u64; 4]>,
    pub fp16_mode: C220Fp16Mode,
    pub integer_saturating: bool,
    pub deq_scale: u64,
    pub(crate) write_targets: Vec<C220VectorStore>,
}

#[derive(Debug, Clone, Copy)]
pub struct C220ConversionIssueInputs<'a> {
    pub pc: u64,
    pub word: u32,
    pub control: C220VectorControl,
    pub addresses: C220VectorAddresses,
    pub iteration_masks: &'a [[u64; 4]],
    pub fp16_mode: C220Fp16Mode,
    pub integer_saturating: bool,
    pub deq_scale: u64,
}

impl C220ConversionIssue {
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
        let source_bits = usize::from(self.instruction.kind.source_type().element_bits());
        let source_bytes = self.instruction.source_bytes_per_repeat();
        let lanes_per_block = C220_VECTOR_BLOCK_BYTES * 8 / source_bits;
        let mut accesses = Vec::with_capacity(
            source_bytes.div_ceil(C220_VECTOR_BLOCK_BYTES)
                + usize::from(matches!(
                    self.instruction.kind,
                    C220ConversionKind::VectorDeqS16ToS8 { .. }
                )) * 4,
        );
        for block in 0..source_bytes.div_ceil(C220_VECTOR_BLOCK_BYTES) {
            let first_lane = block * lanes_per_block;
            let last_lane = (first_lane + lanes_per_block).min(self.instruction.lane_count());
            if first_lane / 64 != usize::from(lane_group)
                && last_lane.saturating_sub(1) / 64 != usize::from(lane_group)
            {
                continue;
            }
            let active = (first_lane..last_lane).any(|lane| {
                lane / 64 == usize::from(lane_group)
                    && mask[lane / 64] & (1_u64 << (lane % 64)) != 0
            });
            if !active {
                continue;
            }
            let offset = repeat_index as u64
                * u64::from(self.control.source_0_repeat_stride)
                * C220_VECTOR_BLOCK_BYTES as u64
                + block as u64
                    * u64::from(self.control.source_0_block_stride.max(1))
                    * C220_VECTOR_BLOCK_BYTES as u64;
            let address = self.addresses.source_0.checked_add(offset).ok_or(
                C220VectorError::SourceAddressOverflow {
                    source_index: 0,
                    base: self.addresses.source_0,
                    block,
                },
            )?;
            accesses.push(C220VectorReadAccess {
                source_index: 0,
                block_index: block as u8,
                buffer_offset: (block * C220_VECTOR_BLOCK_BYTES) as u16,
                bytes: C220_VECTOR_BLOCK_BYTES as u16,
                address,
                active_lane_mask: u16::MAX,
            });
        }
        let first_lane = usize::from(lane_group) * 64;
        let last_lane = (first_lane + 64).min(self.instruction.lane_count());
        let group_active =
            (first_lane..last_lane).any(|lane| mask[lane / 64] & (1_u64 << (lane % 64)) != 0);
        if group_active
            && matches!(
                self.instruction.kind,
                C220ConversionKind::VectorDeqS16ToS8 { .. }
            )
        {
            for block in 0..4 {
                let offset = (block * C220_VECTOR_BLOCK_BYTES) as u64;
                let address = self.addresses.source_1.checked_add(offset).ok_or(
                    C220VectorError::SourceAddressOverflow {
                        source_index: 1,
                        base: self.addresses.source_1,
                        block,
                    },
                )?;
                accesses.push(C220VectorReadAccess {
                    source_index: 1,
                    block_index: block as u8,
                    buffer_offset: (block * C220_VECTOR_BLOCK_BYTES) as u16,
                    bytes: C220_VECTOR_BLOCK_BYTES as u16,
                    address,
                    active_lane_mask: u16::MAX,
                });
            }
        }
        Ok(accesses)
    }

    pub fn logical_lane_for_store(&self, store: &C220VectorStore) -> Option<usize> {
        match self.instruction.kind {
            C220ConversionKind::Ordinary {
                destination: C220ConversionType::S4,
                ..
            } => Some(store.lane_index * 2),
            C220ConversionKind::VectorDeqS16ToS8 { high_half }
            | C220ConversionKind::ScalarDeqS16ToS8 { high_half } => {
                let block = store.lane_index / C220_VECTOR_BLOCK_BYTES;
                let within_block = store.lane_index % C220_VECTOR_BLOCK_BYTES;
                let half_offset = usize::from(high_half) * 16;
                (half_offset..half_offset + 16)
                    .contains(&within_block)
                    .then_some(block * 16 + within_block - half_offset)
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

pub fn plan_c220_conversion_issue(
    inputs: C220ConversionIssueInputs<'_>,
    ub: &UbMemory,
) -> Result<C220ConversionIssue, C220VectorError> {
    check_repeat_limit(inputs.iteration_masks.len())?;
    let instruction =
        C220ConversionInstruction::decode(inputs.word).ok_or(C220VectorError::UnsupportedWord {
            pc: inputs.pc,
            word: inputs.word,
        })?;
    let lane_count = instruction.lane_count();
    let mut effective_masks = inputs.iteration_masks.to_vec();
    for mask in &mut effective_masks {
        mask[lane_count.div_ceil(64)..].fill(0);
        if !lane_count.is_multiple_of(64) {
            mask[lane_count / 64] &= (1_u64 << (lane_count % 64)) - 1;
        }
    }
    let mut issue = C220ConversionIssue {
        pc: inputs.pc,
        word: inputs.word,
        instruction,
        control: inputs.control,
        addresses: inputs.addresses,
        iteration_masks: effective_masks,
        fp16_mode: inputs.fp16_mode,
        integer_saturating: inputs.integer_saturating,
        deq_scale: inputs.deq_scale,
        write_targets: Vec::new(),
    };
    issue.write_targets = plan_write_targets(&issue, ub)?;
    Ok(issue)
}

fn plan_write_targets(
    issue: &C220ConversionIssue,
    ub: &UbMemory,
) -> Result<Vec<C220VectorStore>, C220VectorError> {
    let destination = issue.instruction.kind.destination_type();
    let lane_count = issue.instruction.lane_count();
    let mut targets = Vec::with_capacity(lane_count * issue.iteration_masks.len());
    for (repeat_index, mask) in issue.iteration_masks.iter().enumerate() {
        for group in 0..issue.instruction.lane_groups() {
            for access in issue.read_accesses_for_repeat(repeat_index, group)? {
                ub.check_range(access.address, usize::from(access.bytes))?;
            }
        }
        if destination == C220ConversionType::S4 {
            for byte_index in 0..lane_count.div_ceil(2) {
                let first_lane = byte_index * 2;
                let active = (first_lane..(first_lane + 2).min(lane_count))
                    .any(|lane| mask[lane / 64] & (1_u64 << (lane % 64)) != 0);
                if !active {
                    continue;
                }
                let address = packed_destination_address(issue, repeat_index, byte_index)?;
                ub.check_range(address, 1)?;
                targets.push(C220VectorStore {
                    repeat_index,
                    lane_index: byte_index,
                    address,
                    bank: C220UbBank::from_address(address),
                    width_bytes: 1,
                    data: [0; 8],
                });
            }
        } else {
            let width = destination
                .element_bytes()
                .ok_or(C220VectorError::UnsupportedElementWidth(0))?;
            for logical_lane in 0..lane_count {
                if mask[logical_lane / 64] & (1_u64 << (logical_lane % 64)) == 0 {
                    continue;
                }
                let lane_index = destination_lane_index(issue.instruction.kind, logical_lane);
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
    }
    Ok(targets)
}

fn packed_destination_address(
    issue: &C220ConversionIssue,
    repeat_index: usize,
    byte_index: usize,
) -> Result<u64, C220VectorError> {
    let block = byte_index / C220_VECTOR_BLOCK_BYTES;
    let offset = repeat_index as u64
        * u64::from(issue.control.destination_repeat_stride)
        * C220_VECTOR_BLOCK_BYTES as u64
        + block as u64
            * u64::from(issue.control.destination_block_stride.max(1))
            * C220_VECTOR_BLOCK_BYTES as u64
        + (byte_index % C220_VECTOR_BLOCK_BYTES) as u64;
    issue
        .addresses
        .destination
        .checked_add(offset)
        .ok_or(C220VectorError::AddressOverflow {
            base: issue.addresses.destination,
            lane: byte_index,
        })
}

pub fn evaluate_c220_conversion_repeat(
    issue: &C220ConversionIssue,
    repeat_index: usize,
    lane_group: u8,
    source_bytes: &[u8],
    deq_bytes: &[u8],
    ub: &UbMemory,
) -> Result<(Vec<C220ConversionLaneOutcome>, Vec<C220VectorStore>), C220VectorError> {
    if lane_group >= issue.instruction.lane_groups() {
        return Err(C220VectorError::InvalidLaneGroup(lane_group));
    }
    if source_bytes.len() < issue.instruction.source_bytes_per_repeat() {
        return Err(C220VectorError::InvalidSourceTile {
            actual: source_bytes.len(),
            expected: issue.instruction.source_bytes_per_repeat(),
        });
    }
    if matches!(
        issue.instruction.kind,
        C220ConversionKind::VectorDeqS16ToS8 { .. }
    ) && deq_bytes.len() < 128
    {
        return Err(C220VectorError::InvalidSourceTile {
            actual: deq_bytes.len(),
            expected: 128,
        });
    }
    let mask = issue
        .iteration_masks
        .get(repeat_index)
        .ok_or(C220VectorError::MissingMaskState)?;
    let lane_count = issue.instruction.lane_count();
    let mut lanes = Vec::with_capacity(lane_count);
    for lane_index in 0..lane_count {
        let source_bits = read_source_bits(
            issue.instruction.kind.source_type(),
            source_bytes,
            lane_index,
        );
        let active = lane_index / 64 == usize::from(lane_group)
            && mask[lane_index / 64] & (1_u64 << (lane_index % 64)) != 0;
        if active {
            let (result_bits, status) = convert_lane(issue, lane_index, source_bits, deq_bytes);
            lanes.push(C220ConversionLaneOutcome {
                active,
                source_bits,
                result_bits,
                status,
            });
        } else {
            lanes.push(C220ConversionLaneOutcome {
                active,
                source_bits,
                result_bits: 0,
                status: C220ConversionStatus::default(),
            });
        }
    }
    let stores = if issue.instruction.kind.destination_type() == C220ConversionType::S4 {
        packed_stores(issue, repeat_index, lane_group, mask, &lanes, ub)?
    } else {
        ordinary_stores(issue, repeat_index, &lanes, ub)?
    };
    Ok((lanes, stores))
}

fn ordinary_stores(
    issue: &C220ConversionIssue,
    repeat_index: usize,
    lanes: &[C220ConversionLaneOutcome],
    ub: &UbMemory,
) -> Result<Vec<C220VectorStore>, C220VectorError> {
    let width = issue
        .instruction
        .kind
        .destination_type()
        .element_bytes()
        .ok_or(C220VectorError::UnsupportedElementWidth(0))?;
    let mut stores = Vec::new();
    for (logical_lane, lane) in lanes.iter().enumerate() {
        if !lane.active {
            continue;
        }
        let lane_index = destination_lane_index(issue.instruction.kind, logical_lane);
        let address = vector_destination_address_for_width(
            issue.control,
            issue.addresses,
            repeat_index,
            lane_index,
            width,
        )?;
        ub.check_range(address, usize::from(width))?;
        stores.push(C220VectorStore {
            repeat_index,
            lane_index,
            address,
            bank: C220UbBank::from_address(address),
            width_bytes: width,
            data: store_data(lane.result_bits.to_le_bytes()),
        });
    }
    Ok(stores)
}

fn packed_stores(
    issue: &C220ConversionIssue,
    repeat_index: usize,
    lane_group: u8,
    mask: &[u64; 4],
    lanes: &[C220ConversionLaneOutcome],
    ub: &UbMemory,
) -> Result<Vec<C220VectorStore>, C220VectorError> {
    let first_byte = usize::from(lane_group) * 32;
    let last_byte = (first_byte + 32).min(issue.instruction.lane_count().div_ceil(2));
    let mut stores = Vec::new();
    for byte_index in first_byte..last_byte {
        let first_lane = byte_index * 2;
        let low_active = mask[first_lane / 64] & (1_u64 << (first_lane % 64)) != 0;
        let high_active = first_lane + 1 < lanes.len()
            && mask[(first_lane + 1) / 64] & (1_u64 << ((first_lane + 1) % 64)) != 0;
        if !low_active && !high_active {
            continue;
        }
        let address = packed_destination_address(issue, repeat_index, byte_index)?;
        let mut byte = if low_active && high_active {
            0
        } else {
            ub.read_known(address, 1)?[0]
        };
        if low_active {
            byte = byte & 0xf0 | lanes[first_lane].result_bits as u8 & 0x0f;
        }
        if high_active {
            byte = byte & 0x0f | (lanes[first_lane + 1].result_bits as u8 & 0x0f) << 4;
        }
        stores.push(C220VectorStore {
            repeat_index,
            lane_index: byte_index,
            address,
            bank: C220UbBank::from_address(address),
            width_bytes: 1,
            data: store_data([byte]),
        });
    }
    Ok(stores)
}

fn read_source_bits(dtype: C220ConversionType, bytes: &[u8], lane: usize) -> u64 {
    match dtype {
        C220ConversionType::S4 => {
            let byte = bytes[lane / 2];
            u64::from(if lane.is_multiple_of(2) {
                byte & 0x0f
            } else {
                byte >> 4
            })
        }
        C220ConversionType::U8 | C220ConversionType::S8 => u64::from(bytes[lane]),
        C220ConversionType::S16 | C220ConversionType::F16 | C220ConversionType::Bf16 => {
            let at = lane * 2;
            u64::from(u16::from_le_bytes([bytes[at], bytes[at + 1]]))
        }
        C220ConversionType::S32 | C220ConversionType::F32 => {
            let at = lane * 4;
            u64::from(u32::from_le_bytes(
                bytes[at..at + 4].try_into().expect("u32 lane"),
            ))
        }
        C220ConversionType::S64 => {
            let at = lane * 8;
            u64::from_le_bytes(bytes[at..at + 8].try_into().expect("u64 lane"))
        }
    }
}

fn destination_lane_index(kind: C220ConversionKind, logical_lane: usize) -> usize {
    match kind {
        C220ConversionKind::VectorDeqS16ToS8 { high_half }
        | C220ConversionKind::ScalarDeqS16ToS8 { high_half } => {
            logical_lane / 16 * C220_VECTOR_BLOCK_BYTES
                + usize::from(high_half) * 16
                + logical_lane % 16
        }
        _ => logical_lane,
    }
}

fn convert_lane(
    issue: &C220ConversionIssue,
    lane_index: usize,
    source_bits: u64,
    deq_bytes: &[u8],
) -> (u64, C220ConversionStatus) {
    match issue.instruction.kind {
        C220ConversionKind::S4ToF16 => {
            let value = sign_extend(source_bits, 4) as f64;
            (u64::from(round_finite_to_f16(value)), Default::default())
        }
        C220ConversionKind::Ordinary {
            source,
            destination,
            round,
        } => convert_ordinary(
            source,
            destination,
            round,
            source_bits,
            issue.fp16_mode,
            issue.integer_saturating,
        ),
        C220ConversionKind::VectorDeqS16ToS8 { .. } => {
            let descriptor_offset = lane_index % 16 * 8;
            let descriptor = u64::from_le_bytes(
                deq_bytes[descriptor_offset..descriptor_offset + 8]
                    .try_into()
                    .expect("VDEQ descriptor"),
            );
            deq_s16_to_b8(source_bits as u16 as i16, descriptor)
        }
        C220ConversionKind::ScalarDeqS16ToS8 { .. } => {
            deq_s16_to_b8(source_bits as u16 as i16, issue.deq_scale)
        }
        C220ConversionKind::ScalarDeqS32ToF16 => deq_s32_to_f16(
            source_bits as u32 as i32,
            issue.deq_scale as u16,
            issue.fp16_mode,
        ),
    }
}

pub(crate) fn deq_s16_to_b8(source: i16, descriptor: u64) -> (u64, C220ConversionStatus) {
    let scale_bits = descriptor as u32 & 0xffff_e000;
    let scale_abs = scale_bits & 0x7fff_ffff;
    let scale_nan = scale_abs & 0x7f80_0000 == 0x7f80_0000 && scale_abs & 0x007f_e000 != 0;
    let scale_infinite = scale_abs == 0x7f80_0000;
    let scale = if scale_nan {
        0.0
    } else {
        f32::from_bits(scale_bits)
    };
    let product = f64::from(source) * f64::from(scale);
    let bounded = product.clamp(-(f32::MAX as f64), f32::MAX as f64) as f32;
    let quantized = f64::from(bounded).round_ties_even().clamp(-256.0, 255.0) as i32;
    let bias = sign_extend((descriptor >> 37) & 0x1ff, 9) as i32;
    let biased = quantized + bias;
    let signed = descriptor & (1_u64 << 46) != 0;
    let (result, invalid) = if signed {
        (biased.clamp(-128, 127) as i8 as u8, false)
    } else if biased < 0 {
        (0, true)
    } else {
        (biased.min(255) as u8, false)
    };
    (
        u64::from(result),
        C220ConversionStatus {
            nan_operand: scale_nan,
            infinity_operand: scale_infinite,
            invalid,
            ..Default::default()
        },
    )
}

pub(crate) fn deq_s32_to_f16(
    source: i32,
    scale_bits: u16,
    mode: C220Fp16Mode,
) -> (u64, C220ConversionStatus) {
    let scale_abs = scale_bits & !F16_SIGN;
    let scale_nan = scale_abs & F16_INFINITY == F16_INFINITY && scale_abs & F16_FRACTION != 0;
    let scale_infinite = scale_abs == F16_INFINITY;
    let mut status = C220ConversionStatus {
        nan_operand: scale_nan,
        infinity_operand: scale_infinite,
        ..Default::default()
    };
    if scale_nan || source == 0 && scale_infinite {
        return (
            u64::from(if mode == C220Fp16Mode::Saturating {
                0
            } else {
                F16_CANONICAL_NAN
            }),
            status,
        );
    }
    if scale_infinite {
        let sign = u16::from(source.is_negative()) << 15 ^ (scale_bits & F16_SIGN);
        return (
            u64::from(
                sign | if mode == C220Fp16Mode::Saturating {
                    F16_MAX_FINITE
                } else {
                    F16_INFINITY
                },
            ),
            status,
        );
    }

    let narrowed_source = round_finite_to_f16(f64::from(source) * 2_f64.powi(-17));
    let exact = to_f64(narrowed_source) * to_f64(scale_bits) * 131_072.0;
    let magnitude = exact.abs();
    status.overflow = magnitude >= 65_520.0;
    status.underflow = magnitude > 0.0 && magnitude <= 2_f64.powi(-25);
    let mut result = round_finite_to_f16(exact);
    if status.overflow {
        let sign = if exact.is_sign_negative() {
            F16_SIGN
        } else {
            0
        };
        result = sign
            | if mode == C220Fp16Mode::Saturating {
                F16_MAX_FINITE
            } else {
                F16_INFINITY
            };
    }
    (u64::from(result), status)
}

pub(crate) fn convert_ordinary(
    source: C220ConversionType,
    destination: C220ConversionType,
    round: C220ConversionRound,
    source_bits: u64,
    fp16_mode: C220Fp16Mode,
    integer_saturating: bool,
) -> (u64, C220ConversionStatus) {
    let value = source_value(source, source_bits);
    match destination {
        C220ConversionType::F32 => convert_to_f32(value, round),
        C220ConversionType::F16 => convert_to_f16(value, round, fp16_mode),
        C220ConversionType::Bf16 => convert_to_bf16(value, round, fp16_mode),
        C220ConversionType::S64 => convert_to_signed(value, round, 64, integer_saturating),
        C220ConversionType::S32 => convert_to_signed(value, round, 32, integer_saturating),
        C220ConversionType::S16 => convert_to_signed(value, round, 16, integer_saturating),
        C220ConversionType::S8 => convert_to_signed(value, round, 8, integer_saturating),
        C220ConversionType::U8 => convert_to_unsigned(value, round, 8, integer_saturating),
        C220ConversionType::S4 => convert_to_signed(value, round, 4, integer_saturating),
    }
}

#[derive(Debug, Clone, Copy)]
enum SourceValue {
    Float {
        value: f64,
        nan: bool,
        infinite: bool,
    },
    Signed(i64),
    Unsigned(u64),
}

fn source_value(dtype: C220ConversionType, bits: u64) -> SourceValue {
    match dtype {
        C220ConversionType::F32 => {
            let bits = bits as u32;
            SourceValue::Float {
                value: f64::from(f32::from_bits(bits)),
                nan: f32::from_bits(bits).is_nan(),
                infinite: bits & 0x7fff_ffff == 0x7f80_0000,
            }
        }
        C220ConversionType::F16 => {
            let bits = bits as u16;
            SourceValue::Float {
                value: to_f64(bits),
                nan: bits & F16_INFINITY == F16_INFINITY && bits & F16_FRACTION != 0,
                infinite: bits & !F16_SIGN == F16_INFINITY,
            }
        }
        C220ConversionType::Bf16 => {
            let bits = bits as u16;
            SourceValue::Float {
                value: f64::from(f32::from_bits(u32::from(bits) << 16)),
                nan: bits & BF16_INFINITY == BF16_INFINITY && bits & BF16_FRACTION != 0,
                infinite: bits & !BF16_SIGN == BF16_INFINITY,
            }
        }
        C220ConversionType::S64 => SourceValue::Signed(bits as i64),
        C220ConversionType::S32 => SourceValue::Signed(bits as u32 as i32 as i64),
        C220ConversionType::S16 => SourceValue::Signed(bits as u16 as i16 as i64),
        C220ConversionType::S8 => SourceValue::Signed(bits as u8 as i8 as i64),
        C220ConversionType::U8 => SourceValue::Unsigned(bits as u8 as u64),
        C220ConversionType::S4 => SourceValue::Signed(sign_extend(bits, 4)),
    }
}

fn sign_extend(bits: u64, width: u32) -> i64 {
    ((bits << (64 - width)) as i64) >> (64 - width)
}

fn value_as_f64(value: SourceValue) -> f64 {
    match value {
        SourceValue::Float { value, .. } => value,
        SourceValue::Signed(value) => value as f64,
        SourceValue::Unsigned(value) => value as f64,
    }
}

fn base_status(value: SourceValue) -> C220ConversionStatus {
    match value {
        SourceValue::Float { nan, infinite, .. } => C220ConversionStatus {
            nan_operand: nan,
            infinity_operand: infinite,
            invalid: nan,
            ..Default::default()
        },
        _ => Default::default(),
    }
}

fn convert_to_signed(
    value: SourceValue,
    round: C220ConversionRound,
    width: u32,
    saturating: bool,
) -> (u64, C220ConversionStatus) {
    let mut status = base_status(value);
    if status.nan_operand {
        return (0, status);
    }
    let number = value_as_f64(value);
    let rounded = round_integral(number, round);
    status.inexact = rounded != number;
    let min = -(1_i128 << (width - 1));
    let max = (1_i128 << (width - 1)) - 1;
    if !rounded.is_finite() || rounded < min as f64 || rounded > max as f64 {
        status.overflow = true;
        status.invalid = true;
    }
    let integer = if status.overflow {
        if saturating {
            if rounded.is_sign_negative() { min } else { max }
        } else {
            wrapped_signed_from_f64(rounded, width)
        }
    } else {
        rounded as i128
    };
    let mask = if width == 64 {
        u64::MAX
    } else {
        (1_u64 << width) - 1
    };
    (integer as u64 & mask, status)
}

fn convert_to_unsigned(
    value: SourceValue,
    round: C220ConversionRound,
    width: u32,
    saturating: bool,
) -> (u64, C220ConversionStatus) {
    let mut status = base_status(value);
    if status.nan_operand {
        return (0, status);
    }
    let number = value_as_f64(value);
    let rounded = round_integral(number, round);
    status.inexact = rounded != number;
    let max = (1_u128 << width) - 1;
    if !rounded.is_finite() || rounded < 0.0 || rounded > max as f64 {
        status.overflow = true;
        status.invalid = true;
    }
    let integer = if status.overflow {
        if saturating {
            if rounded.is_sign_negative() { 0 } else { max }
        } else {
            wrapped_unsigned_from_f64(rounded, width)
        }
    } else {
        rounded as u128
    };
    (integer as u64, status)
}

fn wrapped_signed_from_f64(value: f64, width: u32) -> i128 {
    wrapped_unsigned_from_f64(value, width) as i128
}

fn wrapped_unsigned_from_f64(value: f64, width: u32) -> u128 {
    if !value.is_finite() {
        return 0;
    }
    let modulus = 2_f64.powi(width as i32);
    value.rem_euclid(modulus) as u128
}

fn round_integral(value: f64, round: C220ConversionRound) -> f64 {
    match round {
        C220ConversionRound::NearestEven => value.round_ties_even(),
        C220ConversionRound::NearestAway => value.round(),
        C220ConversionRound::Floor => value.floor(),
        C220ConversionRound::Ceil => value.ceil(),
        C220ConversionRound::TowardZero => value.trunc(),
        C220ConversionRound::ToOdd => value.trunc(),
    }
}

fn convert_to_f32(value: SourceValue, round: C220ConversionRound) -> (u64, C220ConversionStatus) {
    let mut status = base_status(value);
    if status.nan_operand {
        return (u64::from(0x7fff_ffff_u32), status);
    }
    let exact = match value {
        SourceValue::Float { value, .. } => round_integral(value, round),
        _ => value_as_f64(value),
    };
    let mut nearest = exact as f32;
    if nearest.is_infinite() && exact.is_finite() {
        status.overflow = true;
    }
    if f64::from(nearest) != exact && exact.is_finite() {
        status.inexact = true;
        nearest = directed_f32(exact, nearest, round);
    }
    (u64::from(nearest.to_bits()), status)
}

fn directed_f32(exact: f64, nearest: f32, round: C220ConversionRound) -> f32 {
    let nearest_value = f64::from(nearest);
    let lower = if nearest_value > exact {
        next_down_f32(nearest)
    } else {
        nearest
    };
    let upper = if nearest_value < exact {
        next_up_f32(nearest)
    } else {
        nearest
    };
    match round {
        C220ConversionRound::NearestEven => nearest,
        C220ConversionRound::NearestAway => {
            let down_distance = exact - f64::from(lower);
            let up_distance = f64::from(upper) - exact;
            if down_distance == up_distance {
                if exact.is_sign_negative() {
                    lower
                } else {
                    upper
                }
            } else {
                nearest
            }
        }
        C220ConversionRound::Floor => lower,
        C220ConversionRound::Ceil => upper,
        C220ConversionRound::TowardZero => {
            if exact.is_sign_negative() {
                upper
            } else {
                lower
            }
        }
        C220ConversionRound::ToOdd => {
            let toward_zero = if exact.is_sign_negative() {
                upper
            } else {
                lower
            };
            if toward_zero.to_bits() & 1 != 0 {
                toward_zero
            } else if exact.is_sign_negative() {
                next_down_f32(toward_zero)
            } else {
                next_up_f32(toward_zero)
            }
        }
    }
}

fn next_up_f32(value: f32) -> f32 {
    if value.is_nan() || value == f32::INFINITY {
        value
    } else if value == 0.0 {
        f32::from_bits(1)
    } else if value.is_sign_negative() {
        f32::from_bits(value.to_bits() - 1)
    } else {
        f32::from_bits(value.to_bits() + 1)
    }
}

fn next_down_f32(value: f32) -> f32 {
    if value.is_nan() || value == f32::NEG_INFINITY {
        value
    } else if value == 0.0 {
        f32::from_bits(0x8000_0001)
    } else if value.is_sign_negative() {
        f32::from_bits(value.to_bits() + 1)
    } else {
        f32::from_bits(value.to_bits() - 1)
    }
}

fn convert_to_f16(
    value: SourceValue,
    round: C220ConversionRound,
    mode: C220Fp16Mode,
) -> (u64, C220ConversionStatus) {
    let mut status = base_status(value);
    let exact = value_as_f64(value);
    if status.nan_operand {
        return (
            u64::from(if mode == C220Fp16Mode::Saturating {
                0
            } else {
                F16_CANONICAL_NAN
            }),
            status,
        );
    }
    if status.infinity_operand {
        let sign = if exact.is_sign_negative() {
            F16_SIGN
        } else {
            0
        };
        return (
            u64::from(
                sign | if mode == C220Fp16Mode::Saturating {
                    F16_MAX_FINITE
                } else {
                    F16_INFINITY
                },
            ),
            status,
        );
    }
    let nearest = round_finite_to_f16(exact);
    let mut bits = directed_f16(exact, nearest, round);
    let represented = f16_value(bits);
    status.inexact = represented != exact;
    if bits & !F16_SIGN == F16_INFINITY || exact.abs() > 65504.0 {
        status.overflow = true;
        if mode == C220Fp16Mode::Saturating {
            bits = bits & F16_SIGN | F16_MAX_FINITE;
        }
    }
    status.underflow = exact != 0.0 && bits & !F16_SIGN == 0;
    (u64::from(bits), status)
}

fn directed_f16(exact: f64, nearest: u16, round: C220ConversionRound) -> u16 {
    let nearest_value = f16_value(nearest);
    if nearest_value == exact {
        return nearest;
    }
    let lower = if nearest_value > exact {
        next_down_f16(nearest)
    } else {
        nearest
    };
    let upper = if nearest_value < exact {
        next_up_f16(nearest)
    } else {
        nearest
    };
    match round {
        C220ConversionRound::NearestEven => nearest,
        C220ConversionRound::NearestAway => {
            if exact - f16_value(lower) == f16_value(upper) - exact {
                if exact.is_sign_negative() {
                    lower
                } else {
                    upper
                }
            } else {
                nearest
            }
        }
        C220ConversionRound::Floor => lower,
        C220ConversionRound::Ceil => upper,
        C220ConversionRound::TowardZero => {
            if exact.is_sign_negative() {
                upper
            } else {
                lower
            }
        }
        C220ConversionRound::ToOdd => {
            let toward_zero = if exact.is_sign_negative() {
                upper
            } else {
                lower
            };
            if toward_zero & 1 != 0 {
                toward_zero
            } else if exact.is_sign_negative() {
                next_down_f16(toward_zero)
            } else {
                next_up_f16(toward_zero)
            }
        }
    }
}

fn f16_value(bits: u16) -> f64 {
    if bits & F16_INFINITY == F16_INFINITY && bits & F16_FRACTION != 0 {
        f64::NAN
    } else {
        to_f64(bits)
    }
}

fn next_up_f16(bits: u16) -> u16 {
    if bits & F16_INFINITY == F16_INFINITY && bits & F16_FRACTION != 0 || bits == F16_INFINITY {
        bits
    } else if bits & !F16_SIGN == 0 {
        1
    } else if bits & F16_SIGN != 0 {
        bits - 1
    } else {
        bits + 1
    }
}

fn next_down_f16(bits: u16) -> u16 {
    if bits & F16_INFINITY == F16_INFINITY && bits & F16_FRACTION != 0
        || bits == F16_SIGN | F16_INFINITY
    {
        bits
    } else if bits & !F16_SIGN == 0 {
        F16_SIGN | 1
    } else if bits & F16_SIGN != 0 {
        bits + 1
    } else {
        bits - 1
    }
}

fn convert_to_bf16(
    value: SourceValue,
    round: C220ConversionRound,
    mode: C220Fp16Mode,
) -> (u64, C220ConversionStatus) {
    let mut status = base_status(value);
    let exact = value_as_f64(value);
    if status.nan_operand {
        return (
            u64::from(if mode == C220Fp16Mode::Saturating {
                0
            } else {
                BF16_CANONICAL_NAN
            }),
            status,
        );
    }
    if status.infinity_operand {
        let sign = if exact.is_sign_negative() {
            BF16_SIGN
        } else {
            0
        };
        return (
            u64::from(
                sign | if mode == C220Fp16Mode::Saturating {
                    BF16_MAX_FINITE
                } else {
                    BF16_INFINITY
                },
            ),
            status,
        );
    }
    let fp32_bits = (exact as f32).to_bits();
    let discarded = fp32_bits & 0xffff;
    let mut bits = (fp32_bits >> 16) as u16;
    if discarded != 0 {
        status.inexact = true;
        let increment = match round {
            C220ConversionRound::NearestEven => {
                discarded > 0x8000 || discarded == 0x8000 && bits & 1 != 0
            }
            C220ConversionRound::NearestAway => discarded >= 0x8000,
            C220ConversionRound::Floor => fp32_bits >> 31 != 0,
            C220ConversionRound::Ceil => fp32_bits >> 31 == 0,
            C220ConversionRound::TowardZero => false,
            C220ConversionRound::ToOdd => bits & 1 == 0,
        };
        bits = bits.wrapping_add(u16::from(increment));
    }
    if bits & !BF16_SIGN == BF16_INFINITY {
        status.overflow = true;
        if mode == C220Fp16Mode::Saturating {
            bits = bits & BF16_SIGN | BF16_MAX_FINITE;
        }
    }
    status.underflow = exact != 0.0 && bits & !BF16_SIGN == 0;
    (u64::from(bits), status)
}
