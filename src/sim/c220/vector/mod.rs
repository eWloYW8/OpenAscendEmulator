use crate::architecture::c220::C220UbBank;
use crate::isa::c220::vector::{C220VecArithmeticHint, C220VecArithmeticOperation};
use crate::memory::sparse::MemoryByteState;
use crate::memory::ub::{UbMemory, UbMemoryError};
use crate::numeric::fp32::{
    Fp32LaneOutcome, Fp32MaskLayout, Fp32VectorError, Fp32VectorOperation, evaluate_fp32_value,
    evaluate_masked_fp32_lanes,
};
use crate::sim::c220::fp16::C220Fp16Mode;
use thiserror::Error;

pub mod axpy;
pub mod broadcast;
pub mod compare;
pub mod conversion;
pub mod copy;
mod f16;
mod fma;
pub mod fused;
pub mod gather;
pub mod merge;
mod movev;
pub mod nchw;
pub mod pipeline;
pub mod read;
pub mod reduce;
mod s16;
mod s32;
pub mod scalar;
pub mod select;
pub mod shift;
pub mod sort;
pub mod special;
mod special_tables;
mod stepper;
pub mod ternary;
pub mod timing;
pub mod transpose;
pub mod vmsu;

pub use movev::{
    C220MovevControl, C220MovevStep, decode_c220_movev_control, execute_c220_movev_to_ub,
};
pub(crate) use movev::{plan_c220_movev_to_ub, write_c220_movev_to_ub};

pub(crate) const C220_VECTOR_BLOCK_BYTES: usize = 32;
pub(crate) const C220_VECTOR_BLOCK_COUNT: usize = 8;
pub(crate) const C220_VECTOR_TILE_BYTES: usize = C220_VECTOR_BLOCK_BYTES * C220_VECTOR_BLOCK_COUNT;
const C220_VECTOR32_LANES: usize = C220_VECTOR_TILE_BYTES / 4;
const MAX_VECTOR_REPEATS: usize = 4096;

#[cfg(test)]
mod test_words {
    pub const C220_CAPTURED_MOVEV_WORD: u32 = 0x82a0_6014;
    pub const C220_CAPTURED_MOVEV_CONTROL: u64 = 0x0100_0008_0001_0001;
    pub const C220_CAPTURED_VADD_WORD: u32 = 0x85e0_d720;
    pub const C220_CAPTURED_VADD_CONTROL: u64 = 0x0100_0808_0801_0101;
    pub const C220_CAPTURED_VSUB_WORD: u32 = 0x85dc_b619;
    pub const C220_CAPTURED_VMUL_WORD: u32 = 0x89dc_b618;
    pub const C220_CAPTURED_VMUL_CONTROL: u64 = C220_CAPTURED_VADD_CONTROL;
}

#[cfg(test)]
pub use test_words::*;
const C220_COUNT_MASK_CONTROL: u64 = 1 << 56;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220VectorStore {
    pub repeat_index: usize,
    pub lane_index: usize,
    pub address: u64,
    pub bank: C220UbBank,
    pub width_bytes: u8,
    pub data: [u8; 8],
}

pub(crate) fn store_data<const N: usize>(bytes: [u8; N]) -> [u8; 8] {
    assert!(N <= 8, "C220 vector store payload exceeds one lane");
    let mut data = [0; 8];
    data[..N].copy_from_slice(&bytes);
    data
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220VectorReadAccess {
    pub source_index: u8,
    pub block_index: u8,
    pub buffer_offset: u16,
    pub bytes: u16,
    pub address: u64,
    pub active_lane_mask: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220Fp32Step {
    pub pc: u64,
    pub word: u32,
    pub hint: C220VecArithmeticHint,
    pub control: C220VectorControl,
    pub source_0_address: u64,
    pub source_1_address: u64,
    pub destination_address: u64,
    pub source_0_bytes: Vec<u8>,
    pub source_1_bytes: Vec<u8>,
    pub iteration_masks: Vec<[u64; 4]>,
    pub lanes: Vec<Fp32LaneOutcome>,
    pub stores: Vec<C220VectorStore>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220VectorArithmeticModes {
    pub integer_saturating: bool,
    pub fp16_mode: C220Fp16Mode,
    pub widen_s16: bool,
}

impl C220VectorArithmeticModes {
    pub const fn from_control_spr(value: u64) -> Self {
        Self {
            integer_saturating: value & (1 << 53) != 0,
            fp16_mode: C220Fp16Mode::from_control_spr(value),
            widen_s16: value & (1 << 52) != 0,
        }
    }

    pub const fn result_element_bytes(self, hint: C220VecArithmeticHint) -> Option<u8> {
        if self.widens(hint) {
            Some(4)
        } else {
            hint.modeled_element_bytes()
        }
    }

    pub const fn widens(self, hint: C220VecArithmeticHint) -> bool {
        self.widen_s16
            && hint.has_s16_value_path()
            && matches!(
                hint.operation,
                C220VecArithmeticOperation::Add
                    | C220VecArithmeticOperation::Subtract
                    | C220VecArithmeticOperation::Multiply
            )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220VectorArithmeticIssue {
    pub pc: u64,
    pub word: u32,
    pub hint: C220VecArithmeticHint,
    pub control: C220VectorControl,
    pub addresses: C220VectorAddresses,
    pub iteration_masks: Vec<[u64; 4]>,
    pub source_element_bytes: u8,
    pub result_element_bytes: u8,
    pub modes: C220VectorArithmeticModes,
    pub(crate) write_targets: Vec<C220VectorStore>,
}

impl C220VectorArithmeticIssue {
    pub fn read_accesses_for_repeat(
        &self,
        repeat_index: usize,
        lane_group: u8,
    ) -> Result<Vec<C220VectorReadAccess>, C220VectorError> {
        let mask = self
            .iteration_masks
            .get(repeat_index)
            .ok_or(C220VectorError::MissingMaskState)?;
        plan_c220_vector_arithmetic_read_accesses(
            self.hint,
            self.control,
            self.addresses,
            repeat_index,
            mask,
            (self.result_element_bytes == 2).then_some(lane_group),
        )
    }
}

pub(crate) struct C220Fp32RepeatOutcome {
    pub source_0_bytes: Vec<u8>,
    pub source_1_bytes: Vec<u8>,
    pub lanes: Vec<Fp32LaneOutcome>,
    pub stores: Vec<C220VectorStore>,
}

#[derive(Debug, Error)]
pub enum C220VectorError {
    #[error("unsupported C220 vector word {word:#010x} at PC {pc:#x}")]
    UnsupportedWord { pc: u64, word: u32 },
    #[error("instruction is not a supported C220 signed-32 vector operation")]
    UnsupportedS32Operation,
    #[error("instruction is not a supported C220 signed-16 vector operation")]
    UnsupportedS16Operation,
    #[error("instruction is not a supported C220 float-16 vector operation")]
    UnsupportedF16Operation,
    #[error("C220 vector destination at {base:#x} overflows at lane {lane}")]
    AddressOverflow { base: u64, lane: usize },
    #[error("C220 vector source {source_index} at {base:#x} overflows at block {block}")]
    SourceAddressOverflow {
        source_index: u8,
        base: u64,
        block: usize,
    },
    #[error("cannot reserve {lanes} C220 vector store records")]
    HostAllocationFailed { lanes: usize },
    #[error("C220 vector mask control {control:#x} is unsupported")]
    UnsupportedMaskControl { control: u64 },
    #[error("C220 vector repeat count {count} exceeds host limit {limit}")]
    RepeatLimitExceeded { count: u64, limit: usize },
    #[error("C220 vector mask state is incomplete")]
    MissingMaskState,
    #[error("C220 VSEL mode {0} is unsupported")]
    UnsupportedSelectMode(u8),
    #[error("C220 VSEL tensor mask is not available")]
    MissingSelectionMask,
    #[error("C220 VA{register}[{index}] has not been initialized")]
    MissingVaEntry { register: u8, index: u8 },
    #[error("C220 vector repeat index {0} is out of range")]
    InvalidRepeatIndex(usize),
    #[error("C220 vector source tile has {actual} bytes, expected {expected}")]
    InvalidSourceTile { actual: usize, expected: usize },
    #[error("unsupported C220 vector element width {0}")]
    UnsupportedElementWidth(u8),
    #[error("unsupported C220 vector arithmetic type selector {0}")]
    UnsupportedArithmeticType(u8),
    #[error("C220 conversion kind is not executable yet")]
    UnsupportedConversionKind,
    #[error("C220 vector lane group {0} is out of range")]
    InvalidLaneGroup(u8),
    #[error("C220 count mask requires zero high word, got {high:#x}")]
    UnsupportedCountMaskHigh { high: u64 },
    #[error("C220 count mask {count} exceeds the vector tile lane count")]
    CountMaskExceedsTile { count: u64 },
    #[error(transparent)]
    Ub(#[from] UbMemoryError),
    #[error(transparent)]
    Fp32(#[from] Fp32VectorError),
}

pub fn decode_c220_fp32_mask(
    control: u64,
    mask0: u64,
    mask1: u64,
) -> Result<[u64; 4], C220VectorError> {
    decode_c220_tile_mask(control, mask0, mask1, C220_VECTOR32_LANES)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220VectorControl {
    pub encoded_repeat_count: u8,
    pub destination_block_stride: u16,
    pub source_0_block_stride: u16,
    pub source_1_block_stride: u16,
    pub destination_repeat_stride: u16,
    pub source_0_repeat_stride: u16,
    pub source_1_repeat_stride: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220VectorMaskState {
    pub control: u64,
    pub low: u64,
    pub high: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220VectorAddresses {
    pub source_0: u64,
    pub source_1: u64,
    pub destination: u64,
}

pub fn decode_c220_fp32_control(control: u64) -> Result<C220VectorControl, C220VectorError> {
    Ok(C220VectorControl {
        encoded_repeat_count: (control >> 56) as u8,
        destination_block_stride: (control & 0xff) as u16,
        source_0_block_stride: ((control >> 8) & 0xff) as u16,
        source_1_block_stride: ((control >> 16) & 0xff) as u16,
        destination_repeat_stride: ((control >> 24) & 0xff) as u16,
        source_0_repeat_stride: ((control >> 32) & 0xff) as u16,
        source_1_repeat_stride: ((control >> 40) & 0xff) as u16,
    })
}

pub fn decode_c220_vector_unary_control(control: u64) -> C220VectorControl {
    C220VectorControl {
        encoded_repeat_count: (control >> 56) as u8,
        destination_block_stride: (control & 0xffff) as u16,
        source_0_block_stride: ((control >> 16) & 0xffff) as u16,
        source_1_block_stride: 0,
        destination_repeat_stride: (((control >> 32) & 0xff) | (((control >> 52) & 0xf) << 8))
            as u16,
        source_0_repeat_stride: ((control >> 40) & 0xfff) as u16,
        source_1_repeat_stride: 0,
    }
}

pub(crate) fn decode_c220_repeat_masks(
    mask_control: u64,
    mask0: u64,
    mask1: u64,
    lane_count: usize,
    encoded_repeat_count: u8,
) -> Result<Vec<[u64; 4]>, C220VectorError> {
    let mask_mode = mask_control & !((1 << 48) | (1 << 53) | (1 << 59));
    let repeat_count = match mask_mode {
        0 => u64::from(encoded_repeat_count),
        C220_COUNT_MASK_CONTROL => {
            if mask1 != 0 {
                return Err(C220VectorError::UnsupportedCountMaskHigh { high: mask1 });
            }
            mask0.div_ceil(lane_count as u64)
        }
        _ => {
            return Err(C220VectorError::UnsupportedMaskControl {
                control: mask_control,
            });
        }
    };
    if repeat_count > MAX_VECTOR_REPEATS as u64 {
        return Err(C220VectorError::RepeatLimitExceeded {
            count: repeat_count,
            limit: MAX_VECTOR_REPEATS,
        });
    }
    let repeats = repeat_count as usize;
    let mut masks = Vec::new();
    masks
        .try_reserve_exact(repeats)
        .map_err(|_| C220VectorError::HostAllocationFailed { lanes: repeats })?;
    for repeat in 0..repeats {
        let mask = if mask_mode == 0 {
            [mask0, mask1, 0, 0]
        } else {
            let remaining = mask0.saturating_sub((repeat * lane_count) as u64);
            decode_c220_tile_mask(
                C220_COUNT_MASK_CONTROL,
                remaining.min(lane_count as u64),
                0,
                lane_count,
            )?
        };
        masks.push(mask);
    }
    Ok(masks)
}

pub(crate) fn decode_c220_tile_mask(
    control: u64,
    mask0: u64,
    mask1: u64,
    lane_count: usize,
) -> Result<[u64; 4], C220VectorError> {
    match control & !((1 << 48) | (1 << 53) | (1 << 59)) {
        0 => Ok([mask0, mask1, 0, 0]),
        C220_COUNT_MASK_CONTROL => {
            if mask1 != 0 {
                return Err(C220VectorError::UnsupportedCountMaskHigh { high: mask1 });
            }
            if mask0 > lane_count as u64 {
                return Err(C220VectorError::CountMaskExceedsTile { count: mask0 });
            }
            let low = if mask0 >= 64 {
                u64::MAX
            } else {
                (1_u64 << mask0) - 1
            };
            let high_count = mask0.saturating_sub(64);
            let high = if high_count == 64 {
                u64::MAX
            } else {
                (1_u64 << high_count) - 1
            };
            Ok([low, high, 0, 0])
        }
        _ => Err(C220VectorError::UnsupportedMaskControl { control }),
    }
}

pub(crate) fn plan_c220_unary_write_targets(
    control: C220VectorControl,
    addresses: C220VectorAddresses,
    iteration_masks: &[[u64; 4]],
    element_bytes: u8,
    ub: &UbMemory,
) -> Result<Vec<C220VectorStore>, C220VectorError> {
    let lane_count = C220_VECTOR_TILE_BYTES / usize::from(element_bytes);
    check_repeat_limit(iteration_masks.len())?;
    let max_lanes = lane_count * iteration_masks.len();
    let mut write_targets = Vec::new();
    write_targets
        .try_reserve_exact(max_lanes)
        .map_err(|_| C220VectorError::HostAllocationFailed { lanes: max_lanes })?;
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
        for lane_index in 0..lane_count {
            if mask[lane_index / 64] & (1_u64 << (lane_index % 64)) == 0 {
                continue;
            }
            let address = vector_destination_address_for_width(
                control,
                addresses,
                repeat_index,
                lane_index,
                element_bytes,
            )?;
            ub.check_range(address, usize::from(element_bytes))?;
            write_targets.push(C220VectorStore {
                repeat_index,
                lane_index,
                address,
                bank: C220UbBank::from_address(address),
                width_bytes: element_bytes,
                data: [0; 8],
            });
        }
    }
    Ok(write_targets)
}

pub fn execute_c220_fp32_to_ub(
    pc: u64,
    word: u32,
    control: C220VectorControl,
    addresses: C220VectorAddresses,
    iteration_masks: &[[u64; 4]],
    ub: &mut UbMemory,
) -> Result<C220Fp32Step, C220VectorError> {
    check_repeat_limit(iteration_masks.len())?;
    let hint = C220VecArithmeticHint::from_word(word)
        .filter(|hint| hint.has_fp32_value_path())
        .ok_or(C220VectorError::UnsupportedWord { pc, word })?;
    let mut staged = (iteration_masks.len() > 1).then(|| ub.clone());
    let memory = if let Some(staged) = &mut staged {
        staged
    } else {
        &mut *ub
    };
    let mut source_0_bytes = Vec::new();
    let mut source_1_bytes = Vec::new();
    let mut lanes = Vec::new();
    let mut stores = Vec::new();
    let max_lanes = C220_VECTOR32_LANES * iteration_masks.len();
    stores
        .try_reserve_exact(max_lanes)
        .map_err(|_| C220VectorError::HostAllocationFailed { lanes: max_lanes })?;
    for (repeat_index, active_mask) in iteration_masks.iter().enumerate() {
        let repeat =
            evaluate_c220_fp32_repeat(hint, control, addresses, repeat_index, active_mask, memory)?;
        let writes = repeat
            .stores
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
        memory.write_segments(&writes)?;
        source_0_bytes.extend(repeat.source_0_bytes);
        source_1_bytes.extend(repeat.source_1_bytes);
        lanes.extend(repeat.lanes);
        stores.extend(repeat.stores);
    }
    if let Some(staged) = staged {
        *ub = staged;
    }
    Ok(C220Fp32Step {
        pc,
        word,
        hint,
        control,
        source_0_address: addresses.source_0,
        source_1_address: addresses.source_1,
        destination_address: addresses.destination,
        source_0_bytes,
        source_1_bytes,
        iteration_masks: iteration_masks.to_vec(),
        lanes,
        stores,
    })
}

pub(crate) fn evaluate_c220_fp32_repeat(
    hint: C220VecArithmeticHint,
    control: C220VectorControl,
    addresses: C220VectorAddresses,
    repeat_index: usize,
    active_mask: &[u64; 4],
    ub: &UbMemory,
) -> Result<C220Fp32RepeatOutcome, C220VectorError> {
    let accesses = plan_c220_vector_arithmetic_read_accesses(
        hint,
        control,
        addresses,
        repeat_index,
        active_mask,
        None,
    )?;
    evaluate_c220_fp32_repeat_with_accesses(
        hint,
        control,
        addresses,
        repeat_index,
        active_mask,
        &accesses,
        ub,
    )
}

pub(crate) fn evaluate_c220_fp32_repeat_with_accesses(
    hint: C220VecArithmeticHint,
    control: C220VectorControl,
    addresses: C220VectorAddresses,
    repeat_index: usize,
    active_mask: &[u64; 4],
    accesses: &[C220VectorReadAccess],
    ub: &UbMemory,
) -> Result<C220Fp32RepeatOutcome, C220VectorError> {
    let mut source_0_bytes = vec![0; C220_VECTOR_TILE_BYTES];
    let mut source_1_bytes = vec![0; C220_VECTOR_TILE_BYTES];
    for access in accesses {
        let destination = if access.source_index == 0 {
            &mut source_0_bytes
        } else {
            &mut source_1_bytes
        };
        let offset = usize::from(access.buffer_offset);
        let bytes = usize::from(access.bytes);
        destination[offset..offset + bytes].copy_from_slice(&ub.read_known(access.address, bytes)?);
    }
    evaluate_c220_fp32_repeat_from_bytes(
        hint,
        control,
        addresses,
        repeat_index,
        active_mask,
        source_0_bytes,
        source_1_bytes,
        ub,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_c220_fp32_repeat_from_bytes(
    hint: C220VecArithmeticHint,
    control: C220VectorControl,
    addresses: C220VectorAddresses,
    repeat_index: usize,
    active_mask: &[u64; 4],
    source_0_bytes: Vec<u8>,
    source_1_bytes: Vec<u8>,
    ub: &UbMemory,
) -> Result<C220Fp32RepeatOutcome, C220VectorError> {
    let first = source_0_bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("four bytes")))
        .collect::<Vec<_>>();
    let second = source_1_bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("four bytes")))
        .collect::<Vec<_>>();
    let lanes = hint.evaluate_fp32_lanes(&first, &second, active_mask)?;
    let mut stores = Vec::new();
    for (lane_index, lane) in lanes.iter().enumerate() {
        if !lane.active {
            continue;
        }
        let address = vector_destination_address(control, addresses, repeat_index, lane_index)?;
        ub.check_range(address, 4)?;
        stores.push(C220VectorStore {
            repeat_index,
            lane_index,
            address,
            bank: C220UbBank::from_address(address),
            width_bytes: 4,
            data: store_data(lane.bits.to_le_bytes()),
        });
    }
    Ok(C220Fp32RepeatOutcome {
        source_0_bytes,
        source_1_bytes,
        lanes,
        stores,
    })
}

pub fn plan_c220_vector_arithmetic_issue(
    pc: u64,
    word: u32,
    control: C220VectorControl,
    addresses: C220VectorAddresses,
    iteration_masks: &[[u64; 4]],
    modes: C220VectorArithmeticModes,
    ub: &UbMemory,
) -> Result<C220VectorArithmeticIssue, C220VectorError> {
    check_repeat_limit(iteration_masks.len())?;
    let hint = C220VecArithmeticHint::from_word(word)
        .filter(|hint| modes.result_element_bytes(*hint).is_some())
        .ok_or(C220VectorError::UnsupportedWord { pc, word })?;
    let source_element_bytes = hint
        .modeled_element_bytes()
        .expect("checked supported type");
    let result_element_bytes = modes
        .result_element_bytes(hint)
        .expect("checked supported type");
    let lane_count = C220_VECTOR_TILE_BYTES / usize::from(result_element_bytes);
    let mut effective_masks = iteration_masks.to_vec();
    if modes.widens(hint) {
        for mask in &mut effective_masks {
            mask[1..].fill(0);
        }
    }
    let mut write_targets = Vec::new();
    let max_lanes = lane_count * iteration_masks.len();
    write_targets
        .try_reserve_exact(max_lanes)
        .map_err(|_| C220VectorError::HostAllocationFailed { lanes: max_lanes })?;
    for (repeat_index, mask) in effective_masks.iter().enumerate() {
        for access in plan_c220_vector_arithmetic_read_accesses(
            hint,
            control,
            addresses,
            repeat_index,
            mask,
            None,
        )? {
            ub.check_range(access.address, C220_VECTOR_BLOCK_BYTES)?;
        }
        for lane_index in 0..lane_count {
            if mask[lane_index / 64] & (1_u64 << (lane_index % 64)) == 0 {
                continue;
            }
            let address = vector_destination_address_for_width(
                control,
                addresses,
                repeat_index,
                lane_index,
                result_element_bytes,
            )?;
            ub.check_range(address, usize::from(result_element_bytes))?;
            write_targets.push(C220VectorStore {
                repeat_index,
                lane_index,
                address,
                bank: C220UbBank::from_address(address),
                width_bytes: result_element_bytes,
                data: [0; 8],
            });
        }
    }
    Ok(C220VectorArithmeticIssue {
        pc,
        word,
        hint,
        control,
        addresses,
        iteration_masks: effective_masks,
        source_element_bytes,
        result_element_bytes,
        modes,
        write_targets,
    })
}

pub(crate) fn vector_destination_address(
    control: C220VectorControl,
    addresses: C220VectorAddresses,
    repeat_index: usize,
    lane_index: usize,
) -> Result<u64, C220VectorError> {
    vector_destination_address_for_width(control, addresses, repeat_index, lane_index, 4)
}

pub(crate) fn vector_destination_address_for_width(
    control: C220VectorControl,
    addresses: C220VectorAddresses,
    repeat_index: usize,
    lane_index: usize,
    element_bytes: u8,
) -> Result<u64, C220VectorError> {
    if !matches!(element_bytes, 1 | 2 | 4 | 8) {
        return Err(C220VectorError::UnsupportedElementWidth(element_bytes));
    }
    let lanes_per_block = C220_VECTOR_BLOCK_BYTES / usize::from(element_bytes);
    let block = lane_index / lanes_per_block;
    let offset = repeat_index as u64
        * u64::from(control.destination_repeat_stride)
        * C220_VECTOR_BLOCK_BYTES as u64
        + block as u64
            * u64::from(control.destination_block_stride.max(1))
            * C220_VECTOR_BLOCK_BYTES as u64
        + ((lane_index % lanes_per_block) * usize::from(element_bytes)) as u64;
    addresses
        .destination
        .checked_add(offset)
        .ok_or(C220VectorError::AddressOverflow {
            base: addresses.destination,
            lane: repeat_index * (C220_VECTOR_TILE_BYTES / usize::from(element_bytes)) + lane_index,
        })
}

pub(crate) fn check_repeat_limit(repeats: usize) -> Result<(), C220VectorError> {
    if repeats > MAX_VECTOR_REPEATS {
        return Err(C220VectorError::RepeatLimitExceeded {
            count: repeats as u64,
            limit: MAX_VECTOR_REPEATS,
        });
    }
    Ok(())
}

pub fn plan_c220_vector_arithmetic_read_accesses(
    hint: C220VecArithmeticHint,
    control: C220VectorControl,
    addresses: C220VectorAddresses,
    repeat_index: usize,
    active_mask: &[u64; 4],
    lane_group: Option<u8>,
) -> Result<Vec<C220VectorReadAccess>, C220VectorError> {
    let element_bytes =
        hint.modeled_element_bytes()
            .ok_or(C220VectorError::UnsupportedArithmeticType(
                hint.dtype_selector,
            ))?;
    plan_c220_vector_read_accesses(
        control,
        addresses,
        repeat_index,
        active_mask,
        1 + usize::from(hint.source_1_register.is_some()),
        element_bytes,
        lane_group,
    )
}

pub(crate) fn plan_c220_vector_read_accesses(
    control: C220VectorControl,
    addresses: C220VectorAddresses,
    repeat_index: usize,
    active_mask: &[u64; 4],
    source_count: usize,
    element_bytes: u8,
    lane_group: Option<u8>,
) -> Result<Vec<C220VectorReadAccess>, C220VectorError> {
    if !matches!(element_bytes, 1 | 2 | 4 | 8) {
        return Err(C220VectorError::UnsupportedElementWidth(element_bytes));
    }
    let lanes_per_block = C220_VECTOR_BLOCK_BYTES / usize::from(element_bytes);
    let mut accesses = Vec::new();
    accesses
        .try_reserve_exact(C220_VECTOR_BLOCK_COUNT * source_count)
        .map_err(|_| UbMemoryError::HostAllocationFailed {
            requested: C220_VECTOR_BLOCK_COUNT * source_count,
        })?;
    let sources = [
        (
            0,
            addresses.source_0,
            control.source_0_block_stride,
            control.source_0_repeat_stride,
        ),
        (
            1,
            addresses.source_1,
            control.source_1_block_stride,
            control.source_1_repeat_stride,
        ),
    ];
    for (source_index, base, block_stride, repeat_stride) in sources.into_iter().take(source_count)
    {
        for block in 0..C220_VECTOR_BLOCK_COUNT {
            let first_lane = block * lanes_per_block;
            if lane_group.is_some_and(|group| first_lane / 64 != usize::from(group)) {
                continue;
            }
            let active_lane_mask = ((active_mask[first_lane / 64] >> (first_lane % 64))
                & ((1_u64 << lanes_per_block) - 1)) as u16;
            if active_lane_mask == 0 {
                continue;
            }
            let offset =
                repeat_index as u64 * C220_VECTOR_BLOCK_BYTES as u64 * u64::from(repeat_stride)
                    + block as u64 * C220_VECTOR_BLOCK_BYTES as u64 * u64::from(block_stride);
            let address =
                base.checked_add(offset)
                    .ok_or(C220VectorError::SourceAddressOverflow {
                        source_index,
                        base,
                        block: repeat_index * C220_VECTOR_BLOCK_COUNT + block,
                    })?;
            address
                .checked_add(C220_VECTOR_BLOCK_BYTES as u64)
                .ok_or(UbMemoryError::RangeOverflow)?;
            accesses.push(C220VectorReadAccess {
                source_index,
                block_index: block as u8,
                buffer_offset: (block * C220_VECTOR_BLOCK_BYTES) as u16,
                bytes: C220_VECTOR_BLOCK_BYTES as u16,
                address,
                active_lane_mask,
            });
        }
    }
    Ok(accesses)
}

pub(crate) fn plan_c220_destination_read_accesses(
    control: C220VectorControl,
    addresses: C220VectorAddresses,
    repeat_index: usize,
    active_mask: &[u64; 4],
    element_bytes: u8,
    lane_group: Option<u8>,
) -> Result<Vec<C220VectorReadAccess>, C220VectorError> {
    if !matches!(element_bytes, 1 | 2 | 4 | 8) {
        return Err(C220VectorError::UnsupportedElementWidth(element_bytes));
    }
    let lanes_per_block = C220_VECTOR_BLOCK_BYTES / usize::from(element_bytes);
    let lane_count = C220_VECTOR_TILE_BYTES / usize::from(element_bytes);
    let mut accesses = Vec::with_capacity(C220_VECTOR_BLOCK_COUNT);
    for block_index in 0..C220_VECTOR_BLOCK_COUNT {
        let first_lane = block_index * lanes_per_block;
        if first_lane >= lane_count
            || lane_group.is_some_and(|group| first_lane / 64 != usize::from(group))
        {
            continue;
        }
        let active_lane_mask = ((active_mask[first_lane / 64] >> (first_lane % 64))
            & ((1_u64 << lanes_per_block) - 1)) as u16;
        if active_lane_mask == 0 {
            continue;
        }
        accesses.push(C220VectorReadAccess {
            source_index: 2,
            block_index: block_index as u8,
            buffer_offset: (block_index * C220_VECTOR_BLOCK_BYTES) as u16,
            bytes: C220_VECTOR_BLOCK_BYTES as u16,
            address: vector_destination_address_for_width(
                control,
                addresses,
                repeat_index,
                first_lane,
                element_bytes,
            )?,
            active_lane_mask,
        });
    }
    Ok(accesses)
}

impl C220VecArithmeticHint {
    pub fn evaluate_fp32_lanes(
        self,
        first: &[u32],
        second: &[u32],
        iteration_mask: &[u64; 4],
    ) -> Result<Vec<Fp32LaneOutcome>, Fp32VectorError> {
        if !self.has_fp32_value_path() {
            return Err(Fp32VectorError::UnsupportedInstruction);
        }
        let rectify_result = matches!(
            self.operation,
            C220VecArithmeticOperation::AddRectify | C220VecArithmeticOperation::SubtractRectify
        );
        let operation = match self.operation {
            C220VecArithmeticOperation::Absolute => Fp32VectorOperation::Absolute,
            C220VecArithmeticOperation::Rectify => Fp32VectorOperation::Rectify,
            C220VecArithmeticOperation::Add | C220VecArithmeticOperation::AddRectify => {
                Fp32VectorOperation::Add
            }
            C220VecArithmeticOperation::Subtract | C220VecArithmeticOperation::SubtractRectify => {
                Fp32VectorOperation::Subtract
            }
            C220VecArithmeticOperation::Multiply => Fp32VectorOperation::Multiply,
            C220VecArithmeticOperation::Divide => Fp32VectorOperation::Divide,
            C220VecArithmeticOperation::Maximum => Fp32VectorOperation::Maximum,
            C220VecArithmeticOperation::Minimum => Fp32VectorOperation::Minimum,
            C220VecArithmeticOperation::Or
            | C220VecArithmeticOperation::And
            | C220VecArithmeticOperation::Not => {
                return Err(Fp32VectorError::UnsupportedInstruction);
            }
        };
        let mut lanes = evaluate_masked_fp32_lanes(
            operation,
            Fp32MaskLayout::C220Lane,
            first,
            second,
            iteration_mask,
        )?;
        if rectify_result {
            for lane in &mut lanes {
                if lane.active {
                    lane.bits =
                        evaluate_fp32_value(Fp32VectorOperation::Rectify, lane.bits, 0).bits;
                }
            }
        }
        Ok(lanes)
    }
}

#[cfg(test)]
mod tests;
