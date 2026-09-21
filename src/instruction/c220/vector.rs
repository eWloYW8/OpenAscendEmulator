use crate::instruction::fp32_vector::{
    Fp32LaneOutcome, Fp32MaskLayout, Fp32VectorError, Fp32VectorOperation,
    evaluate_masked_fp32_lanes,
};
use crate::memory::sparse::MemoryByteState;
use crate::memory::ub::{UbMemory, UbMemoryError};
use crate::memory::ub_bank_c220::C220UbBank;
use thiserror::Error;

pub(crate) const C220_VECTOR_BLOCK_BYTES: usize = 32;
pub(crate) const C220_VECTOR_BLOCK_COUNT: usize = 8;
pub(crate) const C220_VECTOR_TILE_BYTES: usize = C220_VECTOR_BLOCK_BYTES * C220_VECTOR_BLOCK_COUNT;
const C220_FP32_LANES: usize = C220_VECTOR_TILE_BYTES / 4;
const C220_FP32_LANES_PER_BLOCK: usize = C220_VECTOR_BLOCK_BYTES / 4;
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
pub struct C220MovevInstruction {
    pub word: u32,
    pub dtype_selector: u8,
    pub destination_register: u8,
    pub source_register: u8,
    pub control_register: u8,
}

impl C220MovevInstruction {
    pub const fn decode(word: u32) -> Option<Self> {
        if (word >> 29) != 4
            || ((word >> 25) & 0xf) != 1
            || ((word >> 7) & 0x1f) != 0
            || word & 3 != 0
        {
            return None;
        }
        let dtype_selector = ((word >> 22) & 7) as u8;
        Some(Self {
            word,
            dtype_selector,
            destination_register: ((word >> 17) & 0x1f) as u8,
            source_register: ((word >> 12) & 0x1f) as u8,
            control_register: ((word >> 2) & 0x1f) as u8,
        })
    }

    pub const fn supported_element_bytes(self) -> Option<u8> {
        match self.dtype_selector {
            1 => Some(2),
            2 => Some(4),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220VectorStore {
    pub repeat_index: usize,
    pub lane_index: usize,
    pub address: u64,
    pub bank: C220UbBank,
    pub width_bytes: u8,
    pub data: [u8; 4],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Fp32ReadAccess {
    pub source_index: u8,
    pub block_index: u8,
    pub address: u64,
    pub active_lane_mask: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220MovevStep {
    pub pc: u64,
    pub word: u32,
    pub instruction: C220MovevInstruction,
    pub control: C220MovevControl,
    pub destination_address: u64,
    pub scalar_word: u32,
    pub iteration_masks: Vec<[u64; 4]>,
    pub stores: Vec<C220VectorStore>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220Fp32Step {
    pub pc: u64,
    pub word: u32,
    pub hint: C220VecArithmeticHint,
    pub control: C220Fp32Control,
    pub source_0_address: u64,
    pub source_1_address: u64,
    pub destination_address: u64,
    pub source_0_bytes: Vec<u8>,
    pub source_1_bytes: Vec<u8>,
    pub iteration_masks: Vec<[u64; 4]>,
    pub lanes: Vec<Fp32LaneOutcome>,
    pub stores: Vec<C220VectorStore>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220Fp32Issue {
    pub pc: u64,
    pub word: u32,
    pub hint: C220VecArithmeticHint,
    pub control: C220Fp32Control,
    pub addresses: C220Fp32Addresses,
    pub iteration_masks: Vec<[u64; 4]>,
    pub(crate) write_targets: Vec<C220VectorStore>,
}

impl C220Fp32Issue {
    pub fn read_accesses_for_repeat(
        &self,
        repeat_index: usize,
    ) -> Result<Vec<C220Fp32ReadAccess>, C220VectorError> {
        let mask = self
            .iteration_masks
            .get(repeat_index)
            .ok_or(C220VectorError::MissingMaskState)?;
        plan_c220_fp32_read_accesses(self.hint, self.control, self.addresses, repeat_index, mask)
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
    decode_c220_tile_mask(control, mask0, mask1, C220_FP32_LANES)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MovevControl {
    pub encoded_repeat_count: u8,
    pub destination_block_stride: u16,
    pub destination_repeat_stride: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Fp32Control {
    pub encoded_repeat_count: u8,
    pub destination_block_stride: u16,
    pub source_0_block_stride: u16,
    pub source_1_block_stride: u16,
    pub destination_repeat_stride: u16,
    pub source_0_repeat_stride: u16,
    pub source_1_repeat_stride: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Fp32Addresses {
    pub source_0: u64,
    pub source_1: u64,
    pub destination: u64,
}

pub fn decode_c220_movev_control(control: u64) -> Result<C220MovevControl, C220VectorError> {
    Ok(C220MovevControl {
        encoded_repeat_count: (control >> 56) as u8,
        destination_block_stride: (control & 0xffff) as u16,
        destination_repeat_stride: (((control >> 32) & 0xff) | (((control >> 52) & 0xf) << 8))
            as u16,
    })
}

pub fn decode_c220_fp32_control(control: u64) -> Result<C220Fp32Control, C220VectorError> {
    Ok(C220Fp32Control {
        encoded_repeat_count: (control >> 56) as u8,
        destination_block_stride: (control & 0xff) as u16,
        source_0_block_stride: ((control >> 8) & 0xff) as u16,
        source_1_block_stride: ((control >> 16) & 0xff) as u16,
        destination_repeat_stride: ((control >> 24) & 0xff) as u16,
        source_0_repeat_stride: ((control >> 32) & 0xff) as u16,
        source_1_repeat_stride: ((control >> 40) & 0xff) as u16,
    })
}

pub fn decode_c220_fp32_unary_control(control: u64) -> C220Fp32Control {
    C220Fp32Control {
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
    let repeat_count = match mask_control {
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
        let mask = if mask_control == 0 {
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
    match control {
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

pub fn execute_c220_movev_to_ub(
    pc: u64,
    word: u32,
    control: C220MovevControl,
    destination_address: u64,
    scalar_word: u32,
    iteration_masks: &[[u64; 4]],
    ub: &mut UbMemory,
) -> Result<C220MovevStep, C220VectorError> {
    check_repeat_limit(iteration_masks.len())?;
    let instruction =
        C220MovevInstruction::decode(word).ok_or(C220VectorError::UnsupportedWord { pc, word })?;
    let element_bytes = usize::from(
        instruction
            .supported_element_bytes()
            .ok_or(C220VectorError::UnsupportedWord { pc, word })?,
    );
    let lane_count = C220_VECTOR_TILE_BYTES / element_bytes;
    let scalar_bytes = scalar_word.to_le_bytes();
    let mut stores = Vec::new();
    let max_lanes = lane_count * iteration_masks.len();
    stores
        .try_reserve_exact(max_lanes)
        .map_err(|_| C220VectorError::HostAllocationFailed { lanes: max_lanes })?;
    for (repeat_index, active_mask) in iteration_masks.iter().enumerate() {
        for lane_index in 0..lane_count {
            if active_mask[lane_index / 64] & (1_u64 << (lane_index % 64)) == 0 {
                continue;
            }
            let lanes_per_block = C220_VECTOR_BLOCK_BYTES / element_bytes;
            let block = lane_index / lanes_per_block;
            let offset = repeat_index as u64
                * u64::from(control.destination_repeat_stride)
                * C220_VECTOR_BLOCK_BYTES as u64
                + block as u64
                    * u64::from(control.destination_block_stride.max(1))
                    * C220_VECTOR_BLOCK_BYTES as u64
                + ((lane_index % lanes_per_block) * element_bytes) as u64;
            let address = destination_address.checked_add(offset).ok_or(
                C220VectorError::AddressOverflow {
                    base: destination_address,
                    lane: repeat_index * lane_count + lane_index,
                },
            )?;
            ub.check_range(address, element_bytes)?;
            stores.push(C220VectorStore {
                repeat_index,
                lane_index,
                address,
                bank: C220UbBank::from_address(address),
                width_bytes: element_bytes as u8,
                data: scalar_bytes,
            });
        }
    }
    let writes = stores
        .iter()
        .map(|store| {
            (
                store.address,
                scalar_bytes[..element_bytes]
                    .iter()
                    .copied()
                    .map(MemoryByteState::Known)
                    .collect(),
            )
        })
        .collect::<Vec<_>>();
    ub.write_segments(&writes)?;
    Ok(C220MovevStep {
        pc,
        word,
        instruction,
        control,
        destination_address,
        scalar_word,
        iteration_masks: iteration_masks.to_vec(),
        stores,
    })
}

pub fn execute_c220_fp32_to_ub(
    pc: u64,
    word: u32,
    control: C220Fp32Control,
    addresses: C220Fp32Addresses,
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
    let max_lanes = C220_FP32_LANES * iteration_masks.len();
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
                    store.data.map(MemoryByteState::Known).to_vec(),
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
    control: C220Fp32Control,
    addresses: C220Fp32Addresses,
    repeat_index: usize,
    active_mask: &[u64; 4],
    ub: &UbMemory,
) -> Result<C220Fp32RepeatOutcome, C220VectorError> {
    let accesses =
        plan_c220_fp32_read_accesses(hint, control, addresses, repeat_index, active_mask)?;
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
    control: C220Fp32Control,
    addresses: C220Fp32Addresses,
    repeat_index: usize,
    active_mask: &[u64; 4],
    accesses: &[C220Fp32ReadAccess],
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
        let offset = usize::from(access.block_index) * C220_VECTOR_BLOCK_BYTES;
        destination[offset..offset + C220_VECTOR_BLOCK_BYTES]
            .copy_from_slice(&ub.read_known(access.address, C220_VECTOR_BLOCK_BYTES)?);
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
    control: C220Fp32Control,
    addresses: C220Fp32Addresses,
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
        let address = fp32_destination_address(control, addresses, repeat_index, lane_index)?;
        ub.check_range(address, 4)?;
        stores.push(C220VectorStore {
            repeat_index,
            lane_index,
            address,
            bank: C220UbBank::from_address(address),
            width_bytes: 4,
            data: lane.bits.to_le_bytes(),
        });
    }
    Ok(C220Fp32RepeatOutcome {
        source_0_bytes,
        source_1_bytes,
        lanes,
        stores,
    })
}

pub fn plan_c220_fp32_issue(
    pc: u64,
    word: u32,
    control: C220Fp32Control,
    addresses: C220Fp32Addresses,
    iteration_masks: &[[u64; 4]],
    ub: &UbMemory,
) -> Result<C220Fp32Issue, C220VectorError> {
    check_repeat_limit(iteration_masks.len())?;
    let hint = C220VecArithmeticHint::from_word(word)
        .filter(|hint| hint.has_fp32_value_path())
        .ok_or(C220VectorError::UnsupportedWord { pc, word })?;
    let mut write_targets = Vec::new();
    let max_lanes = C220_FP32_LANES * iteration_masks.len();
    write_targets
        .try_reserve_exact(max_lanes)
        .map_err(|_| C220VectorError::HostAllocationFailed { lanes: max_lanes })?;
    for (repeat_index, mask) in iteration_masks.iter().enumerate() {
        for access in plan_c220_fp32_read_accesses(hint, control, addresses, repeat_index, mask)? {
            ub.check_range(access.address, C220_VECTOR_BLOCK_BYTES)?;
        }
        for lane_index in 0..C220_FP32_LANES {
            if mask[lane_index / 64] & (1_u64 << (lane_index % 64)) == 0 {
                continue;
            }
            let address = fp32_destination_address(control, addresses, repeat_index, lane_index)?;
            ub.check_range(address, 4)?;
            write_targets.push(C220VectorStore {
                repeat_index,
                lane_index,
                address,
                bank: C220UbBank::from_address(address),
                width_bytes: 4,
                data: [0; 4],
            });
        }
    }
    Ok(C220Fp32Issue {
        pc,
        word,
        hint,
        control,
        addresses,
        iteration_masks: iteration_masks.to_vec(),
        write_targets,
    })
}

fn fp32_destination_address(
    control: C220Fp32Control,
    addresses: C220Fp32Addresses,
    repeat_index: usize,
    lane_index: usize,
) -> Result<u64, C220VectorError> {
    let lanes_per_block = C220_VECTOR_BLOCK_BYTES / 4;
    let block = lane_index / lanes_per_block;
    let offset = repeat_index as u64
        * u64::from(control.destination_repeat_stride)
        * C220_VECTOR_BLOCK_BYTES as u64
        + block as u64
            * u64::from(control.destination_block_stride.max(1))
            * C220_VECTOR_BLOCK_BYTES as u64
        + ((lane_index % lanes_per_block) * 4) as u64;
    addresses
        .destination
        .checked_add(offset)
        .ok_or(C220VectorError::AddressOverflow {
            base: addresses.destination,
            lane: repeat_index * C220_FP32_LANES + lane_index,
        })
}

fn check_repeat_limit(repeats: usize) -> Result<(), C220VectorError> {
    if repeats > MAX_VECTOR_REPEATS {
        return Err(C220VectorError::RepeatLimitExceeded {
            count: repeats as u64,
            limit: MAX_VECTOR_REPEATS,
        });
    }
    Ok(())
}

pub fn plan_c220_fp32_read_accesses(
    hint: C220VecArithmeticHint,
    control: C220Fp32Control,
    addresses: C220Fp32Addresses,
    repeat_index: usize,
    active_mask: &[u64; 4],
) -> Result<Vec<C220Fp32ReadAccess>, C220VectorError> {
    let mut accesses = Vec::new();
    let source_count = 1 + usize::from(hint.source_1_register.is_some());
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
            let first_lane = block * C220_FP32_LANES_PER_BLOCK;
            let active_lane_mask = ((active_mask[first_lane / 64] >> (first_lane % 64))
                & ((1 << C220_FP32_LANES_PER_BLOCK) - 1)) as u8;
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
            accesses.push(C220Fp32ReadAccess {
                source_index,
                block_index: block as u8,
                address,
                active_lane_mask,
            });
        }
    }
    Ok(accesses)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MovemaskHint {
    pub source_register: u8,
    pub destination_spr: u16,
}

impl C220MovemaskHint {
    pub const fn from_word(word: u32) -> Option<Self> {
        if word & 0xffc0_0000 != 0x8040_0000 {
            return None;
        }
        Some(Self {
            source_register: ((word >> 2) & 0x1f) as u8,
            destination_spr: 100 + ((word >> 7) & 1) as u16,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220VecArithmeticOperation {
    Absolute,
    Add,
    Subtract,
    Multiply,
    Maximum,
    Minimum,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220VecArithmeticHint {
    pub operation: C220VecArithmeticOperation,
    pub destination_register: u8,
    pub source_0_register: u8,
    pub source_1_register: Option<u8>,
    pub control_register: u8,
    pub dtype_selector: u8,
}

impl C220VecArithmeticHint {
    pub const fn from_word(word: u32) -> Option<Self> {
        if (word >> 29) != 4 {
            return None;
        }
        let operation = match (((word >> 25) & 0xf), word & 3) {
            (1, 0) if ((word >> 7) & 0x1f) == 6 && word & (1 << 24) != 0 => {
                C220VecArithmeticOperation::Absolute
            }
            (2, 0) => C220VecArithmeticOperation::Add,
            (2, 1) => C220VecArithmeticOperation::Subtract,
            (3, 0) => C220VecArithmeticOperation::Maximum,
            (3, 1) => C220VecArithmeticOperation::Minimum,
            (4, 0) => C220VecArithmeticOperation::Multiply,
            _ => return None,
        };
        let dtype_selector = ((word >> 22) & 3) as u8;
        Some(Self {
            operation,
            destination_register: ((word >> 17) & 0x1f) as u8,
            source_0_register: ((word >> 12) & 0x1f) as u8,
            source_1_register: match operation {
                C220VecArithmeticOperation::Absolute => None,
                _ => Some(((word >> 7) & 0x1f) as u8),
            },
            control_register: ((word >> 2) & 0x1f) as u8,
            dtype_selector,
        })
    }

    pub const fn has_fp32_value_path(self) -> bool {
        self.dtype_selector == 3
    }

    pub fn evaluate_fp32_lanes(
        self,
        first: &[u32],
        second: &[u32],
        iteration_mask: &[u64; 4],
    ) -> Result<Vec<Fp32LaneOutcome>, Fp32VectorError> {
        if !self.has_fp32_value_path() {
            return Err(Fp32VectorError::UnsupportedInstruction);
        }
        let operation = match self.operation {
            C220VecArithmeticOperation::Absolute => Fp32VectorOperation::Absolute,
            C220VecArithmeticOperation::Add => Fp32VectorOperation::Add,
            C220VecArithmeticOperation::Subtract => Fp32VectorOperation::Subtract,
            C220VecArithmeticOperation::Multiply => Fp32VectorOperation::Multiply,
            C220VecArithmeticOperation::Maximum => Fp32VectorOperation::Maximum,
            C220VecArithmeticOperation::Minimum => Fp32VectorOperation::Minimum,
        };
        evaluate_masked_fp32_lanes(
            operation,
            Fp32MaskLayout::C220Lane,
            first,
            second,
            iteration_mask,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn movev_control() -> C220MovevControl {
        decode_c220_movev_control(C220_CAPTURED_MOVEV_CONTROL).unwrap()
    }

    fn fp32_control() -> C220Fp32Control {
        decode_c220_fp32_control(C220_CAPTURED_VADD_CONTROL).unwrap()
    }

    fn addresses(source_0: u64, source_1: u64, destination: u64) -> C220Fp32Addresses {
        C220Fp32Addresses {
            source_0,
            source_1,
            destination,
        }
    }

    #[test]
    fn vector_controls_decode_block_and_repeat_strides() {
        assert_eq!(movev_control().destination_block_stride, 1);
        assert_eq!(movev_control().destination_repeat_stride, 8);
        assert_eq!(fp32_control().source_0_block_stride, 1);
        assert_eq!(fp32_control().destination_repeat_stride, 8);
        assert_eq!(
            decode_c220_movev_control((2_u64 << 56) | (3 << 52) | (5 << 32) | (9 << 16) | 7)
                .unwrap(),
            C220MovevControl {
                encoded_repeat_count: 2,
                destination_block_stride: 7,
                destination_repeat_stride: (3 << 8) | 5,
            }
        );
        assert_eq!(
            decode_c220_fp32_control(
                (2_u64 << 56) | (6 << 40) | (5 << 32) | (4 << 24) | (3 << 16) | (2 << 8) | 1
            )
            .unwrap(),
            C220Fp32Control {
                encoded_repeat_count: 2,
                destination_block_stride: 1,
                source_0_block_stride: 2,
                source_1_block_stride: 3,
                destination_repeat_stride: 4,
                source_0_repeat_stride: 5,
                source_1_repeat_stride: 6,
            }
        );
        assert_eq!(
            decode_c220_movev_control(0x0200_0008_0001_0001)
                .unwrap()
                .encoded_repeat_count,
            2
        );
        assert_eq!(
            decode_c220_fp32_control(0x0200_0808_0801_0101)
                .unwrap()
                .encoded_repeat_count,
            2
        );
    }

    #[test]
    fn vabs_uses_one_source_and_preserves_masked_destination_lanes() {
        let word = 0x83c0_0300 | (3 << 17) | (4 << 12) | (5 << 2);
        let hint = C220VecArithmeticHint::from_word(word).unwrap();
        assert_eq!(C220VecArithmeticHint::from_word(word & !(1 << 24)), None);
        assert_eq!(hint.operation, C220VecArithmeticOperation::Absolute);
        assert_eq!(hint.destination_register, 3);
        assert_eq!(hint.source_0_register, 4);
        assert_eq!(hint.source_1_register, None);
        assert_eq!(hint.control_register, 5);
        assert!(hint.has_fp32_value_path());

        let wide = decode_c220_fp32_unary_control(
            (2_u64 << 56) | (1 << 52) | (9 << 40) | (7 << 32) | (3 << 16) | 2,
        );
        assert_eq!(wide.destination_block_stride, 2);
        assert_eq!(wide.source_0_block_stride, 3);
        assert_eq!(wide.destination_repeat_stride, 0x107);
        assert_eq!(wide.source_0_repeat_stride, 9);
        assert_eq!(wide.encoded_repeat_count, 2);

        let control =
            decode_c220_fp32_unary_control((1_u64 << 56) | (8 << 40) | (8 << 32) | (1 << 16) | 1);
        let addresses = addresses(0, u64::MAX, 0x200);
        let mask = [0b111, 0, 0, 0];
        let mut ub = UbMemory::new(1024, 256);
        let mut source = [0; C220_VECTOR_BLOCK_BYTES];
        for (lane, bits) in [
            (-3.5_f32).to_bits(),
            (-f32::INFINITY).to_bits(),
            0xff80_0001,
        ]
        .into_iter()
        .enumerate()
        {
            source[lane * 4..lane * 4 + 4].copy_from_slice(&bits.to_le_bytes());
        }
        ub.write_states(0, &source.map(MemoryByteState::Known))
            .unwrap();
        ub.write_states(
            0x20c,
            &0xdead_beef_u32.to_le_bytes().map(MemoryByteState::Known),
        )
        .unwrap();

        let accesses = plan_c220_fp32_read_accesses(hint, control, addresses, 0, &mask).unwrap();
        assert_eq!(accesses.len(), 1);
        assert_eq!(accesses[0].source_index, 0);
        let step =
            execute_c220_fp32_to_ub(0x4000, word, control, addresses, &[mask], &mut ub).unwrap();
        assert_eq!(step.stores.len(), 3);
        assert_eq!(step.lanes[0].bits, 3.5_f32.to_bits());
        assert_eq!(step.lanes[1].bits, f32::INFINITY.to_bits());
        assert_eq!(step.lanes[2].bits, 0x7fff_ffff);
        assert!(step.lanes[2].status.unwrap().nan_operand);
        assert_eq!(step.source_1_bytes, vec![0; C220_VECTOR_TILE_BYTES]);
        assert_eq!(
            ub.read_known(0x20c, 4).unwrap(),
            0xdead_beef_u32.to_le_bytes()
        );
    }

    #[test]
    fn movemask_selects_source_and_mask_register() {
        let first = C220MovemaskHint::from_word(0x8040_0000).unwrap();
        assert_eq!(first.source_register, 0);
        assert_eq!(first.destination_spr, 100);

        let second = C220MovemaskHint::from_word(0x8040_008c).unwrap();
        assert_eq!(second.source_register, 3);
        assert_eq!(second.destination_spr, 101);
        for word in [0x8040_0000 ^ (1 << 22), 0x8240_0000, 0x0040_0000] {
            assert_eq!(C220MovemaskHint::from_word(word), None);
        }
    }

    #[test]
    fn captured_fp32_mask_expands_count_mode_and_preserves_bitset_mode() {
        assert_eq!(
            decode_c220_fp32_mask(0, 0x5555_5555, 0).unwrap(),
            [0x5555_5555, 0, 0, 0]
        );
        assert_eq!(
            decode_c220_fp32_mask(C220_COUNT_MASK_CONTROL, 32, 0).unwrap(),
            [0xffff_ffff, 0, 0, 0]
        );
        assert_eq!(
            decode_c220_fp32_mask(C220_COUNT_MASK_CONTROL, 0, 0).unwrap(),
            [0; 4]
        );
        assert_eq!(
            decode_c220_fp32_mask(C220_COUNT_MASK_CONTROL, 17, 0).unwrap(),
            [0x1ffff, 0, 0, 0]
        );
        assert_eq!(
            decode_c220_fp32_mask(C220_COUNT_MASK_CONTROL, 64, 0).unwrap(),
            [u64::MAX, 0, 0, 0]
        );
        assert!(matches!(
            decode_c220_fp32_mask(1, 32, 0),
            Err(C220VectorError::UnsupportedMaskControl { .. })
        ));
        assert!(matches!(
            decode_c220_fp32_mask(C220_COUNT_MASK_CONTROL, 32, 1),
            Err(C220VectorError::UnsupportedCountMaskHigh { .. })
        ));
        assert!(matches!(
            decode_c220_fp32_mask(C220_COUNT_MASK_CONTROL, 65, 0),
            Err(C220VectorError::CountMaskExceedsTile { .. })
        ));
    }

    #[test]
    fn vendor_add_and_sub_words_select_distinct_vec_handlers() {
        let add = C220VecArithmeticHint::from_word(0x85dc_b618).unwrap();
        let subtract = C220VecArithmeticHint::from_word(0x85dc_b619).unwrap();
        assert_eq!(add.operation, C220VecArithmeticOperation::Add);
        assert_eq!(subtract.operation, C220VecArithmeticOperation::Subtract);
        assert_eq!(add.destination_register, 14);
        assert_eq!(add.source_0_register, 11);
        assert_eq!(add.source_1_register, Some(12));
        assert_eq!(add.control_register, 6);
        assert_eq!(add.dtype_selector, 3);
        assert!(add.has_fp32_value_path());
        assert!(subtract.has_fp32_value_path());
    }

    #[test]
    fn multiply_word_selects_fp32_lane_path() {
        let hint = C220VecArithmeticHint::from_word(C220_CAPTURED_VMUL_WORD).unwrap();
        assert_eq!(hint.operation, C220VecArithmeticOperation::Multiply);
        assert_eq!(hint.destination_register, 14);
        assert_eq!(hint.source_0_register, 11);
        assert_eq!(hint.source_1_register, Some(12));
        assert_eq!(hint.control_register, 6);
        assert!(hint.has_fp32_value_path());
        let first = [2.0_f32.to_bits(), 0];
        let second = [3.0_f32.to_bits(), f32::INFINITY.to_bits()];
        let lanes = hint
            .evaluate_fp32_lanes(&first, &second, &[3, 0, 0, 0])
            .unwrap();
        assert_eq!(lanes[0].bits, 6.0_f32.to_bits());
        assert_eq!(lanes[1].bits, 0x7fff_ffff);
        for wrong in [
            C220_CAPTURED_VMUL_WORD ^ (1 << 25),
            C220_CAPTURED_VMUL_WORD | 1,
        ] {
            assert_eq!(C220VecArithmeticHint::from_word(wrong), None);
        }
    }

    #[test]
    fn captured_vadd_registers_map_destination_sources_and_control() {
        let hint = C220VecArithmeticHint::from_word(C220_CAPTURED_VADD_WORD).unwrap();
        assert_eq!(hint.destination_register, 16);
        assert_eq!(hint.source_0_register, 13);
        assert_eq!(hint.source_1_register, Some(14));
        assert_eq!(hint.control_register, 8);
        assert!(hint.has_fp32_value_path());
    }

    #[test]
    fn rejects_other_route_or_leaf() {
        let add = 0x85dc_b618;
        for wrong in [add ^ (1 << 29), add ^ (3 << 25), add | 2, add | 3] {
            assert_eq!(C220VecArithmeticHint::from_word(wrong), None);
        }
    }

    #[test]
    fn maximum_and_minimum_select_their_own_opcode_family() {
        let maximum = C220VecArithmeticHint::from_word(0x87dc_b618).unwrap();
        let minimum = C220VecArithmeticHint::from_word(0x87dc_b619).unwrap();
        assert_eq!(maximum.operation, C220VecArithmeticOperation::Maximum);
        assert_eq!(minimum.operation, C220VecArithmeticOperation::Minimum);
        assert!(maximum.has_fp32_value_path());
        assert!(minimum.has_fp32_value_path());
        let first = [(-0.0_f32).to_bits(), 0x7fc0_1234, f32::INFINITY.to_bits()];
        let second = [
            0.0_f32.to_bits(),
            1.0_f32.to_bits(),
            f32::NEG_INFINITY.to_bits(),
        ];
        let mask = [0b111, 0, 0, 0];
        let max_lanes = maximum.evaluate_fp32_lanes(&first, &second, &mask).unwrap();
        let min_lanes = minimum.evaluate_fp32_lanes(&first, &second, &mask).unwrap();
        assert_eq!(
            max_lanes.iter().map(|lane| lane.bits).collect::<Vec<_>>(),
            [0, 0x7fff_ffff, f32::INFINITY.to_bits()]
        );
        assert_eq!(
            min_lanes.iter().map(|lane| lane.bits).collect::<Vec<_>>(),
            [0x8000_0000, 0x7fff_ffff, f32::NEG_INFINITY.to_bits()]
        );
        assert!(max_lanes[1].status.unwrap().nan_operand);
        assert!(min_lanes[2].status.unwrap().infinity_operand);
    }

    #[test]
    fn dtype_selector_only_enables_the_supported_fp32_value_path() {
        let base = 0x85dc_b618 & !(3 << 22);
        for selector in 0..4 {
            let hint = C220VecArithmeticHint::from_word(base | (selector << 22)).unwrap();
            assert_eq!(hint.dtype_selector, selector as u8);
            assert_eq!(hint.has_fp32_value_path(), selector == 3);
        }
    }

    #[test]
    fn captured_fp32_words_reach_the_masked_value_stage() {
        let first = [1.0_f32.to_bits(), 2.0_f32.to_bits()];
        let second = [3.0_f32.to_bits(), 4.0_f32.to_bits()];
        let mask = [1, 0, 0, 0];
        let add = C220VecArithmeticHint::from_word(0x85dc_b618).unwrap();
        let sub = C220VecArithmeticHint::from_word(0x85dc_b619).unwrap();
        let added = add.evaluate_fp32_lanes(&first, &second, &mask).unwrap();
        let subtracted = sub.evaluate_fp32_lanes(&first, &second, &mask).unwrap();
        assert_eq!(added[0].bits, 4.0_f32.to_bits());
        assert_eq!(subtracted[0].bits, (-2.0_f32).to_bits());
        assert_eq!(added[1].bits, 0);
        assert_eq!(subtracted[1].bits, 0);
    }

    #[test]
    fn captured_movev_add_and_sub_share_live_ub_state() {
        let x = (0..32_u32)
            .flat_map(|lane| (lane as f32).to_le_bytes())
            .collect::<Vec<_>>();
        let y = (0..32)
            .flat_map(|_| 0.5_f32.to_le_bytes())
            .collect::<Vec<_>>();
        let mut ub = UbMemory::new(512, 256);
        ub.write_states(
            0,
            &x.iter()
                .copied()
                .map(MemoryByteState::Known)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        ub.write_states(
            0x80,
            &y.iter()
                .copied()
                .map(MemoryByteState::Known)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let scalar_word = (-123.0_f32).to_bits();
        let fill = execute_c220_movev_to_ub(
            0x1131_2648,
            C220_CAPTURED_MOVEV_WORD,
            movev_control(),
            0x100,
            scalar_word,
            &[[u64::MAX; 4]],
            &mut ub,
        )
        .unwrap();
        assert_eq!(fill.stores.len(), 64);
        assert_eq!(fill.stores[0].data, scalar_word.to_le_bytes());
        let add = execute_c220_fp32_to_ub(
            0x1131_2660,
            C220_CAPTURED_VADD_WORD,
            fp32_control(),
            addresses(0, 0x80, 0x100),
            &[[0x5555_5555, 0, 0, 0]],
            &mut ub,
        )
        .unwrap();
        assert_eq!(add.source_0_bytes[..128], x);
        assert_eq!(add.source_0_bytes[128..], [0; 128]);
        assert_eq!(add.source_1_bytes[..128], y);
        assert_eq!(add.source_1_bytes[128..], [0; 128]);
        assert_eq!(add.stores.len(), 16);
        let prior_sub = ub.read_known(0x100, 128).unwrap();
        for lane in 0_usize..32 {
            let at = lane * 4;
            if lane.is_multiple_of(2) {
                assert_eq!(prior_sub[at..at + 4], (lane as f32 + 0.5).to_le_bytes());
            } else {
                assert_eq!(prior_sub[at..at + 4], scalar_word.to_le_bytes());
            }
        }

        let sub = execute_c220_fp32_to_ub(
            0x1131_2660,
            C220_CAPTURED_VSUB_WORD,
            fp32_control(),
            addresses(0, 0x80, 0x100),
            &[[0xffff_ffff, 0, 0, 0]],
            &mut ub,
        )
        .unwrap();
        assert_eq!(sub.source_1_bytes[128..], [0; 128]);
        assert_eq!(sub.stores.len(), 32);
        let output = ub.read_known(0x100, 128).unwrap();
        for lane in 0..32 {
            let at = lane * 4;
            assert_eq!(output[at..at + 4], (lane as f32 - 0.5).to_le_bytes());
        }

        let before = ub.clone();
        assert!(matches!(
            execute_c220_fp32_to_ub(
                0,
                C220_CAPTURED_VADD_WORD | 2,
                fp32_control(),
                addresses(0, 0x80, 0x100),
                &[[u64::MAX, 0, 0, 0]],
                &mut ub,
            ),
            Err(C220VectorError::UnsupportedWord { .. })
        ));
        assert_eq!(ub, before);
        assert!(matches!(
            execute_c220_fp32_to_ub(
                0,
                C220_CAPTURED_VADD_WORD,
                fp32_control(),
                addresses(0, 0x80, u64::MAX - 1),
                &[[u64::MAX, 0, 0, 0]],
                &mut ub,
            ),
            Err(C220VectorError::Ub(UbMemoryError::RangeOverflow))
        ));
        assert_eq!(ub, before);

        ub.write_states(0x17f, &[MemoryByteState::Unknown]).unwrap();
        assert_eq!(
            ub.read_known(0x17f, 1),
            Err(UbMemoryError::UnknownByte { address: 0x17f })
        );
        let accesses = plan_c220_fp32_read_accesses(
            C220VecArithmeticHint::from_word(C220_CAPTURED_VADD_WORD).unwrap(),
            fp32_control(),
            addresses(0, 0x80, 0x100),
            0,
            &[0xffff_ffff, 0, 0, 0],
        )
        .unwrap();
        assert_eq!(accesses.len(), 8);
        assert!(accesses.iter().all(|access| access.block_index < 4));
        execute_c220_fp32_to_ub(
            0,
            C220_CAPTURED_VSUB_WORD,
            fp32_control(),
            addresses(0, 0x80, 0x100),
            &[[0xffff_ffff, 0, 0, 0]],
            &mut ub,
        )
        .unwrap();

        ub.write_states(0xff, &[MemoryByteState::Unknown]).unwrap();
        let before = ub.clone();
        assert!(matches!(
            execute_c220_fp32_to_ub(
                0,
                C220_CAPTURED_VSUB_WORD,
                fp32_control(),
                addresses(0, 0x80, 0x100),
                &[[0xffff_ffff, 0, 0, 0]],
                &mut ub,
            ),
            Err(C220VectorError::Ub(UbMemoryError::UnknownByte {
                address: 0xff
            }))
        ));
        assert_eq!(ub, before);
    }

    #[test]
    fn full_c220_tile_reaches_last_fp32_lane() {
        let mut ub = UbMemory::new(768, 256);
        let ones = 1.0_f32.to_le_bytes().repeat(64);
        let zeros = [MemoryByteState::Known(0); 256];
        ub.write_states(
            0,
            &ones
                .iter()
                .copied()
                .map(MemoryByteState::Known)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        ub.write_states(0x100, &zeros).unwrap();
        let mask = [1_u64 << 63, 0, 0, 0];
        let fill = execute_c220_movev_to_ub(
            0,
            C220_CAPTURED_MOVEV_WORD,
            movev_control(),
            0x100,
            2.0_f32.to_bits(),
            &[mask],
            &mut ub,
        )
        .unwrap();
        assert_eq!(fill.stores[0].address, 0x1fc);
        let add = execute_c220_fp32_to_ub(
            4,
            C220_CAPTURED_VADD_WORD,
            fp32_control(),
            addresses(0, 0x100, 0x200),
            &[mask],
            &mut ub,
        )
        .unwrap();
        assert_eq!(add.lanes.len(), 64);
        assert_eq!(add.stores[0].address, 0x2fc);
        assert_eq!(ub.read_known(0x2fc, 4).unwrap(), 3.0_f32.to_le_bytes());
    }

    #[test]
    fn single_repeat_block_strides_change_vector_addresses() {
        let mut ub = UbMemory::new(1024, 256);
        let ones = 1.0_f32.to_le_bytes().repeat(8);
        let twos = 2.0_f32.to_le_bytes().repeat(8);
        for block in 0..8 {
            for (address, bytes) in [(block * 64, &ones), (0x400 + block * 96, &twos)] {
                ub.write_states(
                    address,
                    &bytes
                        .iter()
                        .copied()
                        .map(MemoryByteState::Known)
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            }
        }
        let control = C220Fp32Control {
            destination_block_stride: 2,
            source_0_block_stride: 2,
            source_1_block_stride: 3,
            ..fp32_control()
        };
        let mask = [1 | (1 << 8) | (1 << 63), 0, 0, 0];
        let result = execute_c220_fp32_to_ub(
            0,
            C220_CAPTURED_VADD_WORD,
            control,
            addresses(0, 0x400, 0x800),
            &[mask],
            &mut ub,
        )
        .unwrap();
        assert_eq!(result.stores.len(), 3);
        for store in &result.stores {
            let block = store.lane_index / 8;
            let lane = store.lane_index % 8;
            assert_eq!(store.address, 0x800 + block as u64 * 64 + lane as u64 * 4);
            assert_eq!(
                ub.read_known(store.address, 4).unwrap(),
                3.0_f32.to_le_bytes()
            );
        }
        assert_eq!(
            ub.read_states(0x820, 4).unwrap(),
            [MemoryByteState::Unknown; 4]
        );

        let movev = execute_c220_movev_to_ub(
            4,
            C220_CAPTURED_MOVEV_WORD,
            C220MovevControl {
                destination_block_stride: 2,
                ..movev_control()
            },
            0xc00,
            7,
            &[[1 | (1 << 8), 0, 0, 0]],
            &mut ub,
        )
        .unwrap();
        assert_eq!(movev.stores[0].address, 0xc00);
        assert_eq!(movev.stores[1].address, 0xc40);
    }

    #[test]
    fn count_mask_spans_repeats_and_applies_only_the_final_tail() {
        let raw = (2_u64 << 56) | (8 << 32) | 1;
        let control = decode_c220_movev_control(raw).unwrap();
        let masks = decode_c220_repeat_masks(C220_COUNT_MASK_CONTROL, 65, 0, 64, 2).unwrap();
        assert_eq!(masks, vec![[u64::MAX, 0, 0, 0], [1, 0, 0, 0]]);

        let mut ub = UbMemory::new(512, 256);
        let step =
            execute_c220_movev_to_ub(0, C220_CAPTURED_MOVEV_WORD, control, 0, 7, &masks, &mut ub)
                .unwrap();
        assert_eq!(step.stores.len(), 65);
        assert_eq!(step.stores[64].repeat_index, 1);
        assert_eq!(step.stores[64].address, 0x100);
        assert_eq!(ub.read_known(0x100, 4).unwrap(), 7_u32.to_le_bytes());
        assert_eq!(
            ub.read_states(0x104, 4).unwrap(),
            [MemoryByteState::Unknown; 4]
        );
    }

    #[test]
    fn fp32_repeats_read_previous_repeat_writes_on_aliasing_ub() {
        let control = decode_c220_fp32_control((2_u64 << 56) | (1 << 16) | (1 << 8) | 1).unwrap();
        let mut ub = UbMemory::new(512, 256);
        let ones = 1.0_f32.to_le_bytes().repeat(64);
        let twos = 2.0_f32.to_le_bytes().repeat(64);
        ub.write_states(
            0,
            &ones
                .into_iter()
                .map(MemoryByteState::Known)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        ub.write_states(
            0x100,
            &twos
                .into_iter()
                .map(MemoryByteState::Known)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let step = execute_c220_fp32_to_ub(
            0,
            C220_CAPTURED_VADD_WORD,
            control,
            addresses(0, 0x100, 0),
            &[[1, 0, 0, 0]; 2],
            &mut ub,
        )
        .unwrap();
        assert_eq!(step.stores.len(), 2);
        assert_eq!(step.stores[1].repeat_index, 1);
        assert_eq!(step.source_0_bytes.len(), 512);
        assert_eq!(step.source_0_bytes[256..260], 3.0_f32.to_le_bytes());
        assert_eq!(ub.read_known(0, 4).unwrap(), 5.0_f32.to_le_bytes());
    }

    #[test]
    fn fp32_repeat_strides_advance_each_operand_independently() {
        let raw = (2_u64 << 56) | (16 << 40) | (8 << 32) | (8 << 24) | (1 << 16) | (1 << 8) | 1;
        let control = decode_c220_fp32_control(raw).unwrap();
        let masks = decode_c220_repeat_masks(C220_COUNT_MASK_CONTROL, 65, 0, 64, 2).unwrap();
        let mut ub = UbMemory::new(2048, 256);
        for (address, value) in [(0, 1.0_f32), (0x100, 3.0), (0x400, 2.0), (0x600, 4.0)] {
            let states = value
                .to_le_bytes()
                .repeat(64)
                .into_iter()
                .map(MemoryByteState::Known)
                .collect::<Vec<_>>();
            ub.write_states(address, &states).unwrap();
        }
        let step = execute_c220_fp32_to_ub(
            0,
            C220_CAPTURED_VADD_WORD,
            control,
            addresses(0, 0x400, 0x800),
            &masks,
            &mut ub,
        )
        .unwrap();
        assert_eq!(step.stores.len(), 65);
        assert_eq!(step.stores[64].address, 0x900);
        assert_eq!(step.stores[64].repeat_index, 1);
        assert_eq!(step.source_0_bytes[256..260], 3.0_f32.to_le_bytes());
        assert_eq!(step.source_1_bytes[256..260], 4.0_f32.to_le_bytes());
        assert_eq!(ub.read_known(0x800, 4).unwrap(), 3.0_f32.to_le_bytes());
        assert_eq!(ub.read_known(0x900, 4).unwrap(), 7.0_f32.to_le_bytes());
        assert_eq!(
            ub.read_states(0x904, 4).unwrap(),
            [MemoryByteState::Unknown; 4]
        );
    }
}
