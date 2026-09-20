use crate::fp32_vector::{
    Fp32LaneOutcome, Fp32MaskLayout, Fp32VectorError, Fp32VectorOperation,
    evaluate_masked_fp32_lanes,
};
use crate::replay_memory::MemoryByteState;
use crate::ub_replay::{UbReplayError, UbReplayMemory};
use serde::Serialize;
use thiserror::Error;

const CAPTURED_FP32_TILE_BYTES: usize = 128;
const CAPTURED_SOURCE_READ_BYTES: usize = 256;
pub const C220_CAPTURED_MOVEV_WORD: u32 = 0x82a0_6014;
pub const C220_CAPTURED_MOVEV_CONTROL: u64 = 0x0100_0008_0001_0001;
pub const C220_CAPTURED_VADD_WORD: u32 = 0x85e0_d720;
pub const C220_CAPTURED_VADD_CONTROL: u64 = 0x0100_0808_0801_0101;
pub const C220_CAPTURED_VSUB_WORD: u32 = 0x85dc_b619;
pub const C220_CAPTURED_VMUL_WORD: u32 = 0x89dc_b618;
pub const C220_CAPTURED_VMUL_CONTROL: u64 = C220_CAPTURED_VADD_CONTROL;
const C220_COUNT_MASK_CONTROL: u64 = 1 << 56;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct C220CapturedVectorStore {
    pub lane_index: usize,
    pub address: u64,
    pub data: [u8; 4],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct C220CapturedMovevStep {
    pub pc: u64,
    pub word: u32,
    pub destination_address: u64,
    pub scalar_word: u32,
    pub stores: Vec<C220CapturedVectorStore>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct C220CapturedFp32Step {
    pub pc: u64,
    pub word: u32,
    pub hint: C220VecArithmeticHint,
    pub source_0_address: u64,
    pub source_1_address: u64,
    pub destination_address: u64,
    pub source_0_bytes: Vec<u8>,
    pub source_1_bytes: Vec<u8>,
    pub lanes: Vec<Fp32LaneOutcome>,
    pub stores: Vec<C220CapturedVectorStore>,
}

#[derive(Debug, Error)]
pub enum C220CapturedVectorError {
    #[error("C220 vector word {word:#010x} at PC {pc:#x} is outside the captured path")]
    UnsupportedWord { pc: u64, word: u32 },
    #[error("C220 vector destination at {base:#x} overflows at lane {lane}")]
    AddressOverflow { base: u64, lane: usize },
    #[error("cannot reserve {lanes} C220 vector store records")]
    HostAllocationFailed { lanes: usize },
    #[error("C220 captured vector mask control {control:#x} is unsupported")]
    UnsupportedMaskControl { control: u64 },
    #[error("C220 MOVEV control {control:#x} is outside the supported tile layout")]
    UnsupportedMovevControl { control: u64 },
    #[error("C220 MOVEV requires a full 32-lane mask, got mask0={mask0:?}, mask1={mask1:?}")]
    UnsupportedMovevMask {
        mask0: Option<u64>,
        mask1: Option<u64>,
    },
    #[error("C220 VADD control {control:#x} is outside the supported FP32 tile layout")]
    UnsupportedVaddControl { control: u64 },
    #[error("C220 VMUL control {control:#x} is outside the supported FP32 tile layout")]
    UnsupportedVmulControl { control: u64 },
    #[error(
        "C220 VADD requires the observed alternating mask, got CTRL={ctrl:?}, mask0={mask0:?}, mask1={mask1:?}"
    )]
    UnsupportedVaddMask {
        ctrl: Option<u64>,
        mask0: Option<u64>,
        mask1: Option<u64>,
    },
    #[error(
        "C220 VSUB requires the observed count mask, got CTRL={ctrl:?}, mask0={mask0:?}, mask1={mask1:?}"
    )]
    UnsupportedVsubMask {
        ctrl: Option<u64>,
        mask0: Option<u64>,
        mask1: Option<u64>,
    },
    #[error(
        "C220 VMUL requires the observed count mask, got CTRL={ctrl:?}, mask0={mask0:?}, mask1={mask1:?}"
    )]
    UnsupportedVmulMask {
        ctrl: Option<u64>,
        mask0: Option<u64>,
        mask1: Option<u64>,
    },
    #[error("C220 captured count mask requires zero high word, got {high:#x}")]
    UnsupportedCountMaskHigh { high: u64 },
    #[error("C220 captured count mask {count} exceeds the FP32 tile lane count")]
    CountMaskExceedsTile { count: u64 },
    #[error(transparent)]
    Ub(#[from] UbReplayError),
    #[error(transparent)]
    Fp32(#[from] Fp32VectorError),
}

pub fn decode_captured_c220_fp32_mask(
    control: u64,
    mask0: u64,
    mask1: u64,
) -> Result<[u64; 4], C220CapturedVectorError> {
    match control {
        0 => Ok([mask0, mask1, 0, 0]),
        C220_COUNT_MASK_CONTROL => {
            if mask1 != 0 {
                return Err(C220CapturedVectorError::UnsupportedCountMaskHigh { high: mask1 });
            }
            if mask0 > (CAPTURED_FP32_TILE_BYTES / 4) as u64 {
                return Err(C220CapturedVectorError::CountMaskExceedsTile { count: mask0 });
            }
            Ok([(1_u64 << mask0) - 1, 0, 0, 0])
        }
        _ => Err(C220CapturedVectorError::UnsupportedMaskControl { control }),
    }
}

pub fn execute_captured_c220_movev_to_ub(
    pc: u64,
    word: u32,
    destination_address: u64,
    scalar_word: u32,
    ub: &mut UbReplayMemory,
) -> Result<C220CapturedMovevStep, C220CapturedVectorError> {
    if word != C220_CAPTURED_MOVEV_WORD {
        return Err(C220CapturedVectorError::UnsupportedWord { pc, word });
    }
    let scalar_bytes = scalar_word.to_le_bytes();
    let states: [MemoryByteState; CAPTURED_FP32_TILE_BYTES] =
        std::array::from_fn(|index| MemoryByteState::Known(scalar_bytes[index % 4]));
    ub.write_states(destination_address, &states)?;
    let stores = (0..CAPTURED_FP32_TILE_BYTES / 4)
        .map(|lane_index| C220CapturedVectorStore {
            lane_index,
            address: destination_address + (lane_index * 4) as u64,
            data: scalar_bytes,
        })
        .collect();
    Ok(C220CapturedMovevStep {
        pc,
        word,
        destination_address,
        scalar_word,
        stores,
    })
}

pub fn execute_captured_c220_fp32_to_ub(
    pc: u64,
    word: u32,
    source_0_address: u64,
    source_1_address: u64,
    destination_address: u64,
    active_mask: &[u64; 4],
    ub: &mut UbReplayMemory,
) -> Result<C220CapturedFp32Step, C220CapturedVectorError> {
    if !matches!(
        word,
        C220_CAPTURED_VADD_WORD | C220_CAPTURED_VSUB_WORD | C220_CAPTURED_VMUL_WORD
    ) {
        return Err(C220CapturedVectorError::UnsupportedWord { pc, word });
    }
    let hint = C220VecArithmeticHint::from_word(word)
        .filter(|hint| hint.has_fp32_value_path())
        .ok_or(C220CapturedVectorError::UnsupportedWord { pc, word })?;
    let source_0_bytes = ub.read_known(source_0_address, CAPTURED_SOURCE_READ_BYTES)?;
    let source_1_bytes = ub.read_known(source_1_address, CAPTURED_SOURCE_READ_BYTES)?;
    let first = source_0_bytes[..CAPTURED_FP32_TILE_BYTES]
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("four bytes")))
        .collect::<Vec<_>>();
    let second = source_1_bytes[..CAPTURED_FP32_TILE_BYTES]
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("four bytes")))
        .collect::<Vec<_>>();
    let lanes = hint.evaluate_fp32_lanes(&first, &second, active_mask)?;
    let mut stores = Vec::new();
    stores
        .try_reserve_exact(lanes.len())
        .map_err(|_| C220CapturedVectorError::HostAllocationFailed { lanes: lanes.len() })?;
    let mut staged = ub.clone();
    for (lane_index, lane) in lanes.iter().enumerate() {
        if !lane.active {
            continue;
        }
        let address = destination_address
            .checked_add((lane_index * 4) as u64)
            .ok_or(C220CapturedVectorError::AddressOverflow {
                base: destination_address,
                lane: lane_index,
            })?;
        let data = lane.bits.to_le_bytes();
        staged.write_states(address, &data.map(MemoryByteState::Known))?;
        stores.push(C220CapturedVectorStore {
            lane_index,
            address,
            data,
        });
    }
    *ub = staged;
    Ok(C220CapturedFp32Step {
        pc,
        word,
        hint,
        source_0_address,
        source_1_address,
        destination_address,
        source_0_bytes,
        source_1_bytes,
        lanes,
        stores,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct C220MovemaskHint {
    pub source_register: u8,
    pub destination_spr: u16,
    pub vendor_isa_name: u16,
}

impl C220MovemaskHint {
    pub const fn from_word(word: u32) -> Option<Self> {
        if word & 0xffc0_0000 != 0x8040_0000 {
            return None;
        }
        Some(Self {
            source_register: ((word >> 2) & 0x1f) as u8,
            destination_spr: 100 + ((word >> 7) & 1) as u16,
            vendor_isa_name: 155,
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
    pub vendor_isa_name: u16,
    pub x_register_index_0: u8,
    pub x_register_index_4: u8,
    pub x_register_index_6: u8,
    pub x_register_index_8: u8,
    pub dtype_selector: u8,
    pub vendor_dtype_code: u8,
}

impl C220VecArithmeticHint {
    pub const fn from_word(word: u32) -> Option<Self> {
        if (word >> 29) != 4 {
            return None;
        }
        let (operation, vendor_isa_name) = match (((word >> 25) & 0xf), word & 3) {
            (2, 0) => (C220VecArithmeticOperation::Add, 189),
            (2, 1) => (C220VecArithmeticOperation::Subtract, 190),
            (4, 0) => (C220VecArithmeticOperation::Multiply, 194),
            _ => return None,
        };
        let dtype_selector = ((word >> 22) & 3) as u8;
        let vendor_dtype_code = match dtype_selector {
            0 => 10,
            1 => 8,
            2 => 6,
            _ => 14,
        };
        Some(Self {
            operation,
            vendor_isa_name,
            x_register_index_0: ((word >> 17) & 0x1f) as u8,
            x_register_index_4: ((word >> 12) & 0x1f) as u8,
            x_register_index_6: ((word >> 7) & 0x1f) as u8,
            x_register_index_8: ((word >> 2) & 0x1f) as u8,
            dtype_selector,
            vendor_dtype_code,
        })
    }

    pub const fn has_fp32_value_path(self) -> bool {
        self.vendor_dtype_code == 14
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

    #[test]
    fn movemask_selects_source_and_mask_register() {
        let first = C220MovemaskHint::from_word(0x8040_0000).unwrap();
        assert_eq!(first.source_register, 0);
        assert_eq!(first.destination_spr, 100);
        assert_eq!(first.vendor_isa_name, 155);

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
            decode_captured_c220_fp32_mask(0, 0x5555_5555, 0).unwrap(),
            [0x5555_5555, 0, 0, 0]
        );
        assert_eq!(
            decode_captured_c220_fp32_mask(C220_COUNT_MASK_CONTROL, 32, 0).unwrap(),
            [0xffff_ffff, 0, 0, 0]
        );
        assert_eq!(
            decode_captured_c220_fp32_mask(C220_COUNT_MASK_CONTROL, 0, 0).unwrap(),
            [0; 4]
        );
        assert_eq!(
            decode_captured_c220_fp32_mask(C220_COUNT_MASK_CONTROL, 17, 0).unwrap(),
            [0x1ffff, 0, 0, 0]
        );
        assert!(matches!(
            decode_captured_c220_fp32_mask(1, 32, 0),
            Err(C220CapturedVectorError::UnsupportedMaskControl { .. })
        ));
        assert!(matches!(
            decode_captured_c220_fp32_mask(C220_COUNT_MASK_CONTROL, 32, 1),
            Err(C220CapturedVectorError::UnsupportedCountMaskHigh { .. })
        ));
        assert!(matches!(
            decode_captured_c220_fp32_mask(C220_COUNT_MASK_CONTROL, 33, 0),
            Err(C220CapturedVectorError::CountMaskExceedsTile { .. })
        ));
    }

    #[test]
    fn vendor_add_and_sub_words_select_distinct_vec_handlers() {
        let add = C220VecArithmeticHint::from_word(0x85dc_b618).unwrap();
        let subtract = C220VecArithmeticHint::from_word(0x85dc_b619).unwrap();
        assert_eq!(add.operation, C220VecArithmeticOperation::Add);
        assert_eq!(subtract.operation, C220VecArithmeticOperation::Subtract);
        assert_eq!(add.vendor_isa_name, 189);
        assert_eq!(subtract.vendor_isa_name, 190);
        assert_eq!(add.x_register_index_0, 14);
        assert_eq!(add.x_register_index_4, 11);
        assert_eq!(add.x_register_index_6, 12);
        assert_eq!(add.x_register_index_8, 6);
        assert_eq!(add.dtype_selector, 3);
        assert_eq!(add.vendor_dtype_code, 14);
        assert!(add.has_fp32_value_path());
        assert!(subtract.has_fp32_value_path());
    }

    #[test]
    fn multiply_word_selects_fp32_lane_path() {
        let hint = C220VecArithmeticHint::from_word(C220_CAPTURED_VMUL_WORD).unwrap();
        assert_eq!(hint.operation, C220VecArithmeticOperation::Multiply);
        assert_eq!(hint.vendor_isa_name, 194);
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
    fn dtype_table_matches_vendor_constants() {
        let base = 0x85dc_b618 & !(3 << 22);
        for (selector, code) in [(0, 10), (1, 8), (2, 6), (3, 14)] {
            let hint = C220VecArithmeticHint::from_word(base | (selector << 22)).unwrap();
            assert_eq!(hint.dtype_selector, selector as u8);
            assert_eq!(hint.vendor_dtype_code, code);
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
        let mut ub = UbReplayMemory::new(384, 256);
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
        let fill = execute_captured_c220_movev_to_ub(
            0x1131_2648,
            C220_CAPTURED_MOVEV_WORD,
            0x100,
            scalar_word,
            &mut ub,
        )
        .unwrap();
        assert_eq!(fill.stores.len(), 32);
        assert_eq!(fill.stores[0].data, scalar_word.to_le_bytes());
        let add = execute_captured_c220_fp32_to_ub(
            0x1131_2660,
            C220_CAPTURED_VADD_WORD,
            0,
            0x80,
            0x100,
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

        let sub = execute_captured_c220_fp32_to_ub(
            0x1131_2660,
            C220_CAPTURED_VSUB_WORD,
            0,
            0x80,
            0x100,
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
            execute_captured_c220_fp32_to_ub(
                0,
                C220_CAPTURED_VADD_WORD ^ 1,
                0,
                0x80,
                0x100,
                &[u64::MAX, 0, 0, 0],
                &mut ub,
            ),
            Err(C220CapturedVectorError::UnsupportedWord { .. })
        ));
        assert_eq!(ub, before);
        assert!(matches!(
            execute_captured_c220_fp32_to_ub(
                0,
                C220_CAPTURED_VADD_WORD,
                0,
                0x80,
                u64::MAX - 1,
                &[u64::MAX, 0, 0, 0],
                &mut ub,
            ),
            Err(C220CapturedVectorError::Ub(UbReplayError::RangeOverflow))
        ));
        assert_eq!(ub, before);

        ub.write_states(0x17f, &[MemoryByteState::Unknown]).unwrap();
        let before = ub.clone();
        assert_eq!(
            ub.read_known(0x17f, 1),
            Err(UbReplayError::UnknownByte { address: 0x17f })
        );
        assert!(matches!(
            execute_captured_c220_fp32_to_ub(
                0,
                C220_CAPTURED_VSUB_WORD,
                0,
                0x80,
                0x100,
                &[0xffff_ffff, 0, 0, 0],
                &mut ub,
            ),
            Err(C220CapturedVectorError::Ub(UbReplayError::UnknownByte {
                address: 0x17f
            }))
        ));
        assert_eq!(ub, before);
    }
}
