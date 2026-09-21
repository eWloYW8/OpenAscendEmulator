use crate::instruction::fp32_vector::{
    Fp32LaneOutcome, Fp32MaskLayout, Fp32VectorError, Fp32VectorOperation,
    evaluate_masked_fp32_lanes,
};
use crate::memory::sparse::MemoryByteState;
use crate::memory::ub::{UbMemory, UbMemoryError};
use crate::memory::ub_bank_c220::C220UbBank;
use serde::Serialize;
use thiserror::Error;

pub(crate) const C220_VECTOR_BLOCK_BYTES: usize = 32;
pub(crate) const C220_VECTOR_BLOCK_COUNT: usize = 8;
pub(crate) const C220_VECTOR_TILE_BYTES: usize = C220_VECTOR_BLOCK_BYTES * C220_VECTOR_BLOCK_COUNT;
const C220_FP32_LANES: usize = C220_VECTOR_TILE_BYTES / 4;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct C220VectorStore {
    pub lane_index: usize,
    pub address: u64,
    pub bank: C220UbBank,
    pub width_bytes: u8,
    pub data: [u8; 4],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct C220MovevStep {
    pub pc: u64,
    pub word: u32,
    pub instruction: C220MovevInstruction,
    pub control: C220MovevControl,
    pub destination_address: u64,
    pub scalar_word: u32,
    pub active_mask: [u64; 4],
    pub stores: Vec<C220VectorStore>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
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
    #[error("C220 MOVEV control {control:#x} requires an unsupported repeat count")]
    UnsupportedMovevControl { control: u64 },
    #[error("C220 FP32 vector control {control:#x} requires an unsupported repeat count")]
    UnsupportedFp32Control { control: u64 },
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct C220MovevControl {
    pub destination_block_stride: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct C220Fp32Control {
    pub destination_block_stride: u8,
    pub source_0_block_stride: u8,
    pub source_1_block_stride: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct C220Fp32Addresses {
    pub source_0: u64,
    pub source_1: u64,
    pub destination: u64,
}

pub fn decode_c220_movev_control(control: u64) -> Result<C220MovevControl, C220VectorError> {
    let repeat_count = (control >> 56) as u8;
    let destination_block_stride = ((control >> 16) & 0xffff) as u16;
    if repeat_count != 1 {
        return Err(C220VectorError::UnsupportedMovevControl { control });
    }
    Ok(C220MovevControl {
        destination_block_stride,
    })
}

pub fn decode_c220_fp32_control(control: u64) -> Result<C220Fp32Control, C220VectorError> {
    let repeat_count = (control >> 56) as u8;
    let destination_block_stride = ((control >> 8) & 0xff) as u8;
    let source_0_block_stride = ((control >> 16) & 0xff) as u8;
    let source_1_block_stride = (control & 0xff) as u8;
    if repeat_count != 1 {
        return Err(C220VectorError::UnsupportedFp32Control { control });
    }
    Ok(C220Fp32Control {
        destination_block_stride,
        source_0_block_stride,
        source_1_block_stride,
    })
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
    active_mask: &[u64; 4],
    ub: &mut UbMemory,
) -> Result<C220MovevStep, C220VectorError> {
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
    stores
        .try_reserve_exact(lane_count)
        .map_err(|_| C220VectorError::HostAllocationFailed { lanes: lane_count })?;
    for lane_index in 0..lane_count {
        if active_mask[lane_index / 64] & (1_u64 << (lane_index % 64)) == 0 {
            continue;
        }
        let lanes_per_block = C220_VECTOR_BLOCK_BYTES / element_bytes;
        let block = lane_index / lanes_per_block;
        let offset =
            block * C220_VECTOR_BLOCK_BYTES * usize::from(control.destination_block_stride)
                + (lane_index % lanes_per_block) * element_bytes;
        let address = destination_address.checked_add(offset as u64).ok_or(
            C220VectorError::AddressOverflow {
                base: destination_address,
                lane: lane_index,
            },
        )?;
        ub.check_range(address, element_bytes)?;
        stores.push(C220VectorStore {
            lane_index,
            address,
            bank: C220UbBank::from_address(address),
            width_bytes: element_bytes as u8,
            data: scalar_bytes,
        });
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
        active_mask: *active_mask,
        stores,
    })
}

pub fn execute_c220_fp32_to_ub(
    pc: u64,
    word: u32,
    control: C220Fp32Control,
    addresses: C220Fp32Addresses,
    active_mask: &[u64; 4],
    ub: &mut UbMemory,
) -> Result<C220Fp32Step, C220VectorError> {
    let hint = C220VecArithmeticHint::from_word(word)
        .filter(|hint| hint.has_fp32_value_path())
        .ok_or(C220VectorError::UnsupportedWord { pc, word })?;
    let source_0_bytes =
        read_c220_strided_tile(ub, addresses.source_0, control.source_0_block_stride, 0)?;
    let source_1_bytes =
        read_c220_strided_tile(ub, addresses.source_1, control.source_1_block_stride, 1)?;
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
    stores
        .try_reserve_exact(lanes.len())
        .map_err(|_| C220VectorError::HostAllocationFailed { lanes: lanes.len() })?;
    for (lane_index, lane) in lanes.iter().enumerate() {
        if !lane.active {
            continue;
        }
        let lanes_per_block = C220_VECTOR_BLOCK_BYTES / 4;
        let block = lane_index / lanes_per_block;
        let offset =
            block * C220_VECTOR_BLOCK_BYTES * usize::from(control.destination_block_stride)
                + (lane_index % lanes_per_block) * 4;
        let address = addresses.destination.checked_add(offset as u64).ok_or(
            C220VectorError::AddressOverflow {
                base: addresses.destination,
                lane: lane_index,
            },
        )?;
        ub.check_range(address, 4)?;
        let data = lane.bits.to_le_bytes();
        stores.push(C220VectorStore {
            lane_index,
            address,
            bank: C220UbBank::from_address(address),
            width_bytes: 4,
            data,
        });
    }
    let writes = stores
        .iter()
        .map(|store| {
            (
                store.address,
                store
                    .data
                    .iter()
                    .copied()
                    .map(MemoryByteState::Known)
                    .collect(),
            )
        })
        .collect::<Vec<_>>();
    ub.write_segments(&writes)?;
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
        lanes,
        stores,
    })
}

fn read_c220_strided_tile(
    ub: &UbMemory,
    base: u64,
    block_stride: u8,
    source: u8,
) -> Result<Vec<u8>, C220VectorError> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(C220_VECTOR_TILE_BYTES)
        .map_err(|_| UbMemoryError::HostAllocationFailed {
            requested: C220_VECTOR_TILE_BYTES,
        })?;
    for block in 0..C220_VECTOR_BLOCK_COUNT {
        let address = base
            .checked_add(block as u64 * C220_VECTOR_BLOCK_BYTES as u64 * u64::from(block_stride))
            .ok_or(C220VectorError::SourceAddressOverflow {
                source_index: source,
                base,
                block,
            })?;
        bytes.extend(ub.read_known(address, C220_VECTOR_BLOCK_BYTES)?);
    }
    Ok(bytes)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum C220VecArithmeticOperation {
    Add,
    Subtract,
    Multiply,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct C220VecArithmeticHint {
    pub operation: C220VecArithmeticOperation,
    pub x_register_index_0: u8,
    pub x_register_index_4: u8,
    pub x_register_index_6: u8,
    pub x_register_index_8: u8,
    pub dtype_selector: u8,
}

impl C220VecArithmeticHint {
    pub const fn from_word(word: u32) -> Option<Self> {
        if (word >> 29) != 4 {
            return None;
        }
        let operation = match (((word >> 25) & 0xf), word & 3) {
            (2, 0) => C220VecArithmeticOperation::Add,
            (2, 1) => C220VecArithmeticOperation::Subtract,
            (4, 0) => C220VecArithmeticOperation::Multiply,
            _ => return None,
        };
        let dtype_selector = ((word >> 22) & 3) as u8;
        Some(Self {
            operation,
            x_register_index_0: ((word >> 17) & 0x1f) as u8,
            x_register_index_4: ((word >> 12) & 0x1f) as u8,
            x_register_index_6: ((word >> 7) & 0x1f) as u8,
            x_register_index_8: ((word >> 2) & 0x1f) as u8,
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
            C220VecArithmeticOperation::Add => Fp32VectorOperation::Add,
            C220VecArithmeticOperation::Subtract => Fp32VectorOperation::Subtract,
            C220VecArithmeticOperation::Multiply => Fp32VectorOperation::Multiply,
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
    fn single_repeat_controls_do_not_depend_on_unused_repeat_strides() {
        assert_eq!(movev_control().destination_block_stride, 1);
        assert_eq!(fp32_control().source_0_block_stride, 1);
        assert_eq!(
            decode_c220_movev_control(0x01ff_ffff_0002_ffff)
                .unwrap()
                .destination_block_stride,
            2
        );
        assert_eq!(
            decode_c220_fp32_control(0x01ff_ffff_ff02_0304).unwrap(),
            C220Fp32Control {
                destination_block_stride: 3,
                source_0_block_stride: 2,
                source_1_block_stride: 4,
            }
        );
        assert!(matches!(
            decode_c220_movev_control(0x0200_0008_0001_0001),
            Err(C220VectorError::UnsupportedMovevControl { .. })
        ));
        assert!(matches!(
            decode_c220_fp32_control(0x0200_0808_0801_0101),
            Err(C220VectorError::UnsupportedFp32Control { .. })
        ));
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
        assert_eq!(add.x_register_index_0, 14);
        assert_eq!(add.x_register_index_4, 11);
        assert_eq!(add.x_register_index_6, 12);
        assert_eq!(add.x_register_index_8, 6);
        assert_eq!(add.dtype_selector, 3);
        assert!(add.has_fp32_value_path());
        assert!(subtract.has_fp32_value_path());
    }

    #[test]
    fn multiply_word_selects_fp32_lane_path() {
        let hint = C220VecArithmeticHint::from_word(C220_CAPTURED_VMUL_WORD).unwrap();
        assert_eq!(hint.operation, C220VecArithmeticOperation::Multiply);
        assert_eq!(hint.x_register_index_0, 14);
        assert_eq!(hint.x_register_index_4, 11);
        assert_eq!(hint.x_register_index_6, 12);
        assert_eq!(hint.x_register_index_8, 6);
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
        assert_eq!(hint.x_register_index_0, 16);
        assert_eq!(hint.x_register_index_4, 13);
        assert_eq!(hint.x_register_index_6, 14);
        assert_eq!(hint.x_register_index_8, 8);
        assert!(hint.has_fp32_value_path());
    }

    #[test]
    fn rejects_other_route_or_leaf() {
        let add = 0x85dc_b618;
        for wrong in [add ^ (1 << 29), add ^ (1 << 25), add | 2, add | 3] {
            assert_eq!(C220VecArithmeticHint::from_word(wrong), None);
        }
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
            &[u64::MAX; 4],
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
            &[0x5555_5555, 0, 0, 0],
            &mut ub,
        )
        .unwrap();
        assert_eq!(add.source_0_bytes[..128], x);
        assert_eq!(add.source_0_bytes[128..], y);
        assert_eq!(add.source_1_bytes[..128], y);
        assert_eq!(
            add.source_1_bytes[128..],
            scalar_word.to_le_bytes().repeat(32)
        );
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
            &[0xffff_ffff, 0, 0, 0],
            &mut ub,
        )
        .unwrap();
        assert_eq!(sub.source_1_bytes[128..], prior_sub);
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
                &[u64::MAX, 0, 0, 0],
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
                &[u64::MAX, 0, 0, 0],
                &mut ub,
            ),
            Err(C220VectorError::Ub(UbMemoryError::RangeOverflow))
        ));
        assert_eq!(ub, before);

        ub.write_states(0x17f, &[MemoryByteState::Unknown]).unwrap();
        let before = ub.clone();
        assert_eq!(
            ub.read_known(0x17f, 1),
            Err(UbMemoryError::UnknownByte { address: 0x17f })
        );
        assert!(matches!(
            execute_c220_fp32_to_ub(
                0,
                C220_CAPTURED_VSUB_WORD,
                fp32_control(),
                addresses(0, 0x80, 0x100),
                &[0xffff_ffff, 0, 0, 0],
                &mut ub,
            ),
            Err(C220VectorError::Ub(UbMemoryError::UnknownByte {
                address: 0x17f
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
            &mask,
            &mut ub,
        )
        .unwrap();
        assert_eq!(fill.stores[0].address, 0x1fc);
        let add = execute_c220_fp32_to_ub(
            4,
            C220_CAPTURED_VADD_WORD,
            fp32_control(),
            addresses(0, 0x100, 0x200),
            &mask,
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
        };
        let mask = [1 | (1 << 8) | (1 << 63), 0, 0, 0];
        let result = execute_c220_fp32_to_ub(
            0,
            C220_CAPTURED_VADD_WORD,
            control,
            addresses(0, 0x400, 0x800),
            &mask,
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
            },
            0xc00,
            7,
            &[1 | (1 << 8), 0, 0, 0],
            &mut ub,
        )
        .unwrap();
        assert_eq!(movev.stores[0].address, 0xc00);
        assert_eq!(movev.stores[1].address, 0xc40);
    }
}
