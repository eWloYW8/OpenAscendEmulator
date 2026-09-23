pub mod bias;
mod burst;
pub mod factor;
pub mod fixp;
pub mod load2d;
pub mod load2d_sparse;
pub mod load2d_transpose;
pub mod out_to_l1;
pub mod set2d;

use burst::BurstLayout;

use thiserror::Error;

#[cfg(test)]
mod test_words {
    pub const CAPTURED_C220_MOV_UB_TO_OUT_WORD: u32 = 0x7094_e190;
    pub const CAPTURED_C220_SUB_MOV_UB_TO_OUT_WORD: u32 = 0x7090_c210;
    pub const CAPTURED_C220_MOV_OUT_TO_UB_X_WORD: u32 = 0x711f_3188;
    pub const CAPTURED_C220_MOV_OUT_TO_UB_Y_WORD: u32 = 0x7124_f188;
    pub const CAPTURED_C220_SUB_MOV_OUT_TO_UB_X_WORD: u32 = 0x711b_1208;
    pub const CAPTURED_C220_SUB_MOV_OUT_TO_UB_Y_WORD: u32 = 0x7120_d208;
    pub const CAPTURED_C220_TILING_MOV_OUT_TO_UB_WORD: u32 = 0x7100_1108;
    pub const CAPTURED_C220_SUB_TILING_MOV_OUT_TO_UB_WORD: u32 = 0x7100_1188;
}

#[cfg(test)]
pub use test_words::*;
pub const C220_MOV_UB_TO_OUT_UNIT_BYTES: u64 = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220MovDirection {
    HbmToUb,
    UbToHbm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MovInstruction {
    pub word: u32,
    pub direction: C220MovDirection,
    pub destination_register: u8,
    pub source_register: u8,
    pub descriptor_register: u8,
}

impl C220MovInstruction {
    pub const fn decode(word: u32) -> Option<Self> {
        if (word >> 29) != 3 || ((word >> 27) & 3) != 2 {
            return None;
        }
        let direction = match (((word >> 23) & 0xf), word & 0x7f) {
            (2, 0x08) => C220MovDirection::HbmToUb,
            (1, 0x10) => C220MovDirection::UbToHbm,
            _ => return None,
        };
        Some(Self {
            word,
            direction,
            destination_register: ((word >> 17) & 0x1f) as u8,
            source_register: ((word >> 12) & 0x1f) as u8,
            descriptor_register: ((word >> 7) & 0x1f) as u8,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220DmaMovDescriptor {
    pub instruction_word: u32,
    pub xm: u64,
    pub burst_count: u16,
    pub burst_length: u16,
    pub source_gap: u16,
    pub destination_gap: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220DmaMovSegment {
    pub burst_index: u16,
    pub unit_index: u16,
    pub source_local: u64,
    pub destination_hbm: u64,
    pub bytes: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MovOutToUbDescriptor {
    pub instruction_word: u32,
    pub xm: u64,
    pub burst_count: u16,
    pub burst_length: u16,
    pub source_gap: u16,
    pub destination_gap: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MovOutToUbSegment {
    pub burst_index: u16,
    pub unit_index: u16,
    pub source_hbm: u64,
    pub destination_local: u64,
    pub bytes: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum C220MovOutToUbError {
    #[error("unsupported C220 MOV_OUT_TO_UB word {word:#010x}")]
    UnsupportedWord { word: u32 },
    #[error("unsupported C220 MOV_OUT_TO_UB XM descriptor {xm:#x}")]
    UnsupportedXm { xm: u64 },
    #[error("cannot allocate C220 MOV_OUT_TO_UB segment records")]
    AllocationFailed,
    #[error("C220 MOV_OUT_TO_UB descriptor fields disagree with XM")]
    InconsistentDescriptor,
    #[error("C220 MOV_OUT_TO_UB source or destination address overflows")]
    AddressOverflow,
}

impl C220MovOutToUbDescriptor {
    pub const fn is_word(word: u32) -> bool {
        matches!(
            C220MovInstruction::decode(word),
            Some(C220MovInstruction {
                direction: C220MovDirection::HbmToUb,
                ..
            })
        )
    }

    pub fn decode(instruction_word: u32, xm: u64) -> Result<Self, C220MovOutToUbError> {
        if !Self::is_word(instruction_word) {
            return Err(C220MovOutToUbError::UnsupportedWord {
                word: instruction_word,
            });
        }
        if xm & 0xf != 0 {
            return Err(C220MovOutToUbError::UnsupportedXm { xm });
        }
        let layout = BurstLayout::decode(xm);
        Ok(Self {
            instruction_word,
            xm,
            burst_count: layout.count,
            burst_length: layout.length,
            source_gap: layout.source_gap,
            destination_gap: layout.destination_gap,
        })
    }

    pub fn segments(
        self,
        source_hbm: u64,
        destination_local: u64,
    ) -> Result<Vec<C220MovOutToUbSegment>, C220MovOutToUbError> {
        let segments = self.segment_iter(source_hbm, destination_local)?;
        let mut records = Vec::new();
        records
            .try_reserve_exact(segments.len())
            .map_err(|_| C220MovOutToUbError::AllocationFailed)?;
        records.extend(segments);
        Ok(records)
    }

    pub const fn is_disabled(self) -> bool {
        self.burst_count == 0 || self.burst_length == 0
    }

    pub fn segment_iter(
        self,
        source_hbm: u64,
        destination_local: u64,
    ) -> Result<impl ExactSizeIterator<Item = C220MovOutToUbSegment> + Clone, C220MovOutToUbError>
    {
        let decoded = Self::decode(self.instruction_word, self.xm)?;
        if decoded.burst_count != self.burst_count
            || decoded.burst_length != self.burst_length
            || decoded.source_gap != self.source_gap
            || decoded.destination_gap != self.destination_gap
        {
            return Err(C220MovOutToUbError::InconsistentDescriptor);
        }
        let layout = BurstLayout {
            count: self.burst_count,
            length: self.burst_length,
            source_gap: self.source_gap,
            destination_gap: self.destination_gap,
        };
        Ok(layout
            .segments(source_hbm, destination_local)
            .ok_or(C220MovOutToUbError::AddressOverflow)?
            .map(|segment| C220MovOutToUbSegment {
                burst_index: segment.burst_index,
                unit_index: segment.unit_index,
                source_hbm: segment.source,
                destination_local: segment.destination,
                bytes: segment.bytes,
            }))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum C220DmaMovError {
    #[error("unsupported C220 MOV_UB_TO_OUT word {word:#010x}")]
    UnsupportedWord { word: u32 },
    #[error("unsupported C220 MOV_UB_TO_OUT low mode bits {mode:#x}")]
    UnsupportedMode { mode: u8 },
    #[error("cannot allocate C220 MOV_UB_TO_OUT segment records")]
    AllocationFailed,
    #[error("C220 MOV_UB_TO_OUT descriptor fields disagree with XM")]
    InconsistentDescriptor,
    #[error("C220 MOV_UB_TO_OUT source or destination address overflows")]
    AddressOverflow,
}

impl C220DmaMovDescriptor {
    pub const fn is_word(word: u32) -> bool {
        matches!(
            C220MovInstruction::decode(word),
            Some(C220MovInstruction {
                direction: C220MovDirection::UbToHbm,
                ..
            })
        )
    }

    pub fn decode(instruction_word: u32, xm: u64) -> Result<Self, C220DmaMovError> {
        if !Self::is_word(instruction_word) {
            return Err(C220DmaMovError::UnsupportedWord {
                word: instruction_word,
            });
        }
        let mode = (xm & 0xf) as u8;
        if mode != 0 {
            return Err(C220DmaMovError::UnsupportedMode { mode });
        }
        let layout = BurstLayout::decode(xm);
        Ok(Self {
            instruction_word,
            xm,
            burst_count: layout.count,
            burst_length: layout.length,
            source_gap: layout.source_gap,
            destination_gap: layout.destination_gap,
        })
    }

    pub fn segments(
        self,
        source_local: u64,
        destination_hbm: u64,
    ) -> Result<Vec<C220DmaMovSegment>, C220DmaMovError> {
        let segments = self.segment_iter(source_local, destination_hbm)?;
        let mut records = Vec::new();
        records
            .try_reserve_exact(segments.len())
            .map_err(|_| C220DmaMovError::AllocationFailed)?;
        records.extend(segments);
        Ok(records)
    }

    pub const fn is_disabled(self) -> bool {
        self.burst_count == 0 || self.burst_length == 0
    }

    pub fn segment_iter(
        self,
        source_local: u64,
        destination_hbm: u64,
    ) -> Result<impl ExactSizeIterator<Item = C220DmaMovSegment> + Clone, C220DmaMovError> {
        let decoded = Self::decode(self.instruction_word, self.xm)?;
        if self.burst_count != decoded.burst_count
            || self.burst_length != decoded.burst_length
            || self.source_gap != decoded.source_gap
            || self.destination_gap != decoded.destination_gap
        {
            return Err(C220DmaMovError::InconsistentDescriptor);
        }
        let layout = BurstLayout {
            count: self.burst_count,
            length: self.burst_length,
            source_gap: self.source_gap,
            destination_gap: self.destination_gap,
        };
        Ok(layout
            .segments(source_local, destination_hbm)
            .ok_or(C220DmaMovError::AddressOverflow)?
            .map(|segment| C220DmaMovSegment {
                burst_index: segment.burst_index,
                unit_index: segment.unit_index,
                source_local: segment.source,
                destination_hbm: segment.destination,
                bytes: segment.bytes,
            }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captured_descriptor_predicts_four_live_coordinates() {
        for word in [
            CAPTURED_C220_MOV_UB_TO_OUT_WORD,
            CAPTURED_C220_SUB_MOV_UB_TO_OUT_WORD,
        ] {
            let descriptor = C220DmaMovDescriptor::decode(word, 0x40010).unwrap();
            assert_eq!(descriptor.burst_count, 1);
            assert_eq!(descriptor.burst_length, 4);
            let segments = descriptor.segments(0x100, 0x1151_4000).unwrap();
            assert_eq!(segments.len(), 4);
            for (index, segment) in segments.iter().enumerate() {
                assert_eq!(segment.source_local, 0x100 + index as u64 * 32);
                assert_eq!(segment.destination_hbm, 0x1151_4000 + index as u64 * 32);
                assert_eq!(segment.bytes, 32);
            }
        }
    }

    #[test]
    fn zero_gap_plan_does_not_embed_fixture_addresses() {
        let descriptor =
            C220DmaMovDescriptor::decode(CAPTURED_C220_MOV_UB_TO_OUT_WORD, 0x20020).unwrap();
        let segments = descriptor.segments(0x40, 0x2000).unwrap();
        assert_eq!(segments.len(), 4);
        assert_eq!(segments[0].destination_hbm, 0x2000);
        assert_eq!(segments[2].destination_hbm, 0x2040);
        assert_eq!(segments[3].source_local, 0xa0);
        assert_eq!(segments[3].burst_index, 1);
    }

    #[test]
    fn descriptor_modes_empty_commands_and_full_size_range() {
        let word = CAPTURED_C220_MOV_UB_TO_OUT_WORD;
        assert!(matches!(
            C220DmaMovDescriptor::decode(word ^ 1, 0x40010),
            Err(C220DmaMovError::UnsupportedWord { .. })
        ));
        assert!(matches!(
            C220DmaMovDescriptor::decode(word, 0x40011),
            Err(C220DmaMovError::UnsupportedMode { .. })
        ));
        for xm in [0, 0x10, 0x10000] {
            let disabled = C220DmaMovDescriptor::decode(word, xm).unwrap();
            assert!(disabled.is_disabled());
            assert_eq!(disabled.segment_iter(u64::MAX, u64::MAX).unwrap().len(), 0);
        }
        let large = C220DmaMovDescriptor::decode(word, 0xffff_fff0).unwrap();
        let mut segments = large.segment_iter(0, 0).unwrap();
        let units = 4095 * 65535;
        assert_eq!(segments.len(), units);
        assert_eq!(
            segments.nth(units - 1).unwrap().source_local,
            u64::from(4094_u32.wrapping_mul(65535 * 32)) + 65534 * 32
        );
        assert_eq!(segments.len(), 0);
        let descriptor = C220DmaMovDescriptor::decode(word, 0x40010).unwrap();
        assert_eq!(
            descriptor.segments(u64::MAX - 16, 0x1000),
            Err(C220DmaMovError::AddressOverflow)
        );
    }

    #[test]
    fn captured_mte2_input_words_predict_four_live_coordinates() {
        for (word, local_base) in [
            (CAPTURED_C220_MOV_OUT_TO_UB_X_WORD, 0),
            (CAPTURED_C220_MOV_OUT_TO_UB_Y_WORD, 0x80),
            (CAPTURED_C220_SUB_MOV_OUT_TO_UB_X_WORD, 0),
            (CAPTURED_C220_SUB_MOV_OUT_TO_UB_Y_WORD, 0x80),
        ] {
            let descriptor = C220MovOutToUbDescriptor::decode(word, 0x40010).unwrap();
            let segments = descriptor.segments(0x1151_1c00, local_base).unwrap();
            for (index, segment) in segments.iter().enumerate() {
                assert_eq!(segment.source_hbm, 0x1151_1c00 + index as u64 * 32);
                assert_eq!(segment.destination_local, local_base + index as u64 * 32);
                assert_eq!(segment.bytes, 32);
            }
        }
    }

    #[test]
    fn tiling_word_moves_one_32_byte_unit() {
        for word in [
            CAPTURED_C220_TILING_MOV_OUT_TO_UB_WORD,
            CAPTURED_C220_SUB_TILING_MOV_OUT_TO_UB_WORD,
        ] {
            let descriptor = C220MovOutToUbDescriptor::decode(word, 0x10010).unwrap();
            assert_eq!(
                descriptor.segments(0x1251_5400, 0).unwrap(),
                [C220MovOutToUbSegment {
                    burst_index: 0,
                    unit_index: 0,
                    source_hbm: 0x1251_5400,
                    destination_local: 0,
                    bytes: 32,
                }]
            );
            assert_eq!(
                C220MovOutToUbDescriptor::decode(word, 0x40010)
                    .unwrap()
                    .segments(0x1251_5400, 0)
                    .unwrap()
                    .len(),
                4
            );
        }
    }

    #[test]
    fn source_and_destination_gaps_advance_each_burst_independently() {
        let xm = (3_u64 << 48) | (1 << 32) | (2 << 16) | (2 << 4);
        let input =
            C220MovOutToUbDescriptor::decode(CAPTURED_C220_MOV_OUT_TO_UB_X_WORD, xm).unwrap();
        let input_segments = input.segments(0x1000, 0x80).unwrap();
        assert_eq!(input_segments.len(), 4);
        assert_eq!((input.source_gap, input.destination_gap), (1, 3));
        assert_eq!(
            (
                input_segments[2].source_hbm,
                input_segments[2].destination_local
            ),
            (0x1060, 0x120)
        );
        assert_eq!(
            (
                input_segments[3].source_hbm,
                input_segments[3].destination_local
            ),
            (0x1080, 0x140)
        );

        let output = C220DmaMovDescriptor::decode(CAPTURED_C220_MOV_UB_TO_OUT_WORD, xm).unwrap();
        let output_segments = output.segments(0x80, 0x2000).unwrap();
        assert_eq!((output.source_gap, output.destination_gap), (1, 3));
        assert_eq!(
            (
                output_segments[2].source_local,
                output_segments[2].destination_hbm
            ),
            (0xe0, 0x20a0)
        );
        assert_eq!(
            (
                output_segments[3].source_local,
                output_segments[3].destination_hbm
            ),
            (0x100, 0x20c0)
        );
    }

    #[test]
    fn captured_mte2_plan_rejects_other_modes_and_overflow() {
        assert!(matches!(
            C220MovOutToUbDescriptor::decode(0, 0x40010),
            Err(C220MovOutToUbError::UnsupportedWord { .. })
        ));
        assert!(matches!(
            C220MovOutToUbDescriptor::decode(CAPTURED_C220_MOV_OUT_TO_UB_X_WORD, 0x40011),
            Err(C220MovOutToUbError::UnsupportedXm { .. })
        ));
        let descriptor =
            C220MovOutToUbDescriptor::decode(CAPTURED_C220_MOV_OUT_TO_UB_X_WORD, 0x40010).unwrap();
        assert_eq!(
            descriptor.segments(u64::MAX - 32, 0),
            Err(C220MovOutToUbError::AddressOverflow)
        );
        assert_eq!(
            descriptor.segments(0, u64::MAX - 32),
            Err(C220MovOutToUbError::AddressOverflow)
        );
        assert_eq!(
            C220MovOutToUbDescriptor {
                xm: 0x10010,
                ..descriptor
            }
            .segments(0, 0),
            Err(C220MovOutToUbError::InconsistentDescriptor)
        );
    }

    #[test]
    fn output_descriptor_revalidates_public_fields_before_planning() {
        let descriptor =
            C220DmaMovDescriptor::decode(CAPTURED_C220_MOV_UB_TO_OUT_WORD, 0x40010).unwrap();
        assert_eq!(
            C220DmaMovDescriptor {
                burst_count: 4096,
                ..descriptor
            }
            .segments(0, 0),
            Err(C220DmaMovError::InconsistentDescriptor)
        );
        assert_eq!(
            C220DmaMovDescriptor {
                xm: 0x40011,
                ..descriptor
            }
            .segments(0, 0),
            Err(C220DmaMovError::UnsupportedMode { mode: 1 })
        );
    }
}
