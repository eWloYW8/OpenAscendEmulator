use serde::Serialize;
use thiserror::Error;

pub const CAPTURED_C220_MOV_UB_TO_OUT_WORD: u32 = 0x7094_e190;
pub const CAPTURED_C220_SUB_MOV_UB_TO_OUT_WORD: u32 = 0x7090_c210;
pub const CAPTURED_C220_MOV_OUT_TO_UB_X_WORD: u32 = 0x711f_3188;
pub const CAPTURED_C220_MOV_OUT_TO_UB_Y_WORD: u32 = 0x7124_f188;
pub const CAPTURED_C220_SUB_MOV_OUT_TO_UB_X_WORD: u32 = 0x711b_1208;
pub const CAPTURED_C220_SUB_MOV_OUT_TO_UB_Y_WORD: u32 = 0x7120_d208;
pub const C220_MOV_UB_TO_OUT_UNIT_BYTES: u64 = 32;
pub const MAX_C220_DMAMOV_SEGMENTS: u64 = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct C220DmaMovDescriptor {
    pub instruction_word: u32,
    pub xm: u64,
    pub burst_count: u16,
    pub burst_length: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct C220DmaMovSegment {
    pub burst_index: u16,
    pub unit_index: u16,
    pub source_local: u64,
    pub destination_hbm: u64,
    pub bytes: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct C220MovOutToUbDescriptor {
    pub instruction_word: u32,
    pub xm: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct C220MovOutToUbSegment {
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
    #[error("C220 MOV_OUT_TO_UB source or destination address overflows")]
    AddressOverflow,
}

impl C220MovOutToUbDescriptor {
    pub fn decode(instruction_word: u32, xm: u64) -> Result<Self, C220MovOutToUbError> {
        if !matches!(
            instruction_word,
            CAPTURED_C220_MOV_OUT_TO_UB_X_WORD
                | CAPTURED_C220_MOV_OUT_TO_UB_Y_WORD
                | CAPTURED_C220_SUB_MOV_OUT_TO_UB_X_WORD
                | CAPTURED_C220_SUB_MOV_OUT_TO_UB_Y_WORD
        ) {
            return Err(C220MovOutToUbError::UnsupportedWord {
                word: instruction_word,
            });
        }
        if xm != 0x40010 {
            return Err(C220MovOutToUbError::UnsupportedXm { xm });
        }
        Ok(Self {
            instruction_word,
            xm,
        })
    }

    pub fn segments(
        self,
        source_hbm: u64,
        destination_local: u64,
    ) -> Result<[C220MovOutToUbSegment; 4], C220MovOutToUbError> {
        let mut segments = [C220MovOutToUbSegment {
            source_hbm: 0,
            destination_local: 0,
            bytes: 32,
        }; 4];
        for (index, segment) in segments.iter_mut().enumerate() {
            let offset = (index as u64) * 32;
            segment.source_hbm = source_hbm
                .checked_add(offset)
                .and_then(|value| value.checked_add(32).map(|_| value))
                .ok_or(C220MovOutToUbError::AddressOverflow)?;
            segment.destination_local = destination_local
                .checked_add(offset)
                .and_then(|value| value.checked_add(32).map(|_| value))
                .ok_or(C220MovOutToUbError::AddressOverflow)?;
        }
        Ok(segments)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum C220DmaMovError {
    #[error("unsupported C220 MOV_UB_TO_OUT word {word:#010x}")]
    UnsupportedWord { word: u32 },
    #[error("unsupported C220 MOV_UB_TO_OUT low mode bits {mode:#x}")]
    UnsupportedMode { mode: u8 },
    #[error(
        "C220 MOV_UB_TO_OUT nonzero gaps are not modeled: source={source_gap}, destination={destination_gap}"
    )]
    UnsupportedGaps {
        source_gap: u16,
        destination_gap: u16,
    },
    #[error("C220 MOV_UB_TO_OUT burst count or length is zero")]
    EmptyDescriptor,
    #[error("C220 MOV_UB_TO_OUT requests {requested} segments, limit {limit}")]
    TooManySegments { requested: u64, limit: u64 },
    #[error("C220 MOV_UB_TO_OUT source or destination address overflows")]
    AddressOverflow,
}

impl C220DmaMovDescriptor {
    pub fn decode(instruction_word: u32, xm: u64) -> Result<Self, C220DmaMovError> {
        if !matches!(
            instruction_word,
            CAPTURED_C220_MOV_UB_TO_OUT_WORD | CAPTURED_C220_SUB_MOV_UB_TO_OUT_WORD
        ) {
            return Err(C220DmaMovError::UnsupportedWord {
                word: instruction_word,
            });
        }
        let mode = (xm & 0xf) as u8;
        if mode != 0 {
            return Err(C220DmaMovError::UnsupportedMode { mode });
        }
        let source_gap = ((xm >> 32) & 0xffff) as u16;
        let destination_gap = ((xm >> 48) & 0xffff) as u16;
        if source_gap != 0 || destination_gap != 0 {
            return Err(C220DmaMovError::UnsupportedGaps {
                source_gap,
                destination_gap,
            });
        }
        let burst_count = ((xm >> 4) & 0xfff) as u16;
        let burst_length = ((xm >> 16) & 0xffff) as u16;
        if burst_count == 0 || burst_length == 0 {
            return Err(C220DmaMovError::EmptyDescriptor);
        }
        let requested = u64::from(burst_count) * u64::from(burst_length);
        if requested > MAX_C220_DMAMOV_SEGMENTS {
            return Err(C220DmaMovError::TooManySegments {
                requested,
                limit: MAX_C220_DMAMOV_SEGMENTS,
            });
        }
        Ok(Self {
            instruction_word,
            xm,
            burst_count,
            burst_length,
        })
    }

    pub fn segments(
        self,
        source_local: u64,
        destination_hbm: u64,
    ) -> Result<Vec<C220DmaMovSegment>, C220DmaMovError> {
        let capacity = usize::from(self.burst_count) * usize::from(self.burst_length);
        let mut segments = Vec::with_capacity(capacity);
        for burst_index in 0..self.burst_count {
            for unit_index in 0..self.burst_length {
                let linear_index =
                    u64::from(burst_index) * u64::from(self.burst_length) + u64::from(unit_index);
                let offset = linear_index * C220_MOV_UB_TO_OUT_UNIT_BYTES;
                let source = source_local
                    .checked_add(offset)
                    .ok_or(C220DmaMovError::AddressOverflow)?;
                let destination = destination_hbm
                    .checked_add(offset)
                    .ok_or(C220DmaMovError::AddressOverflow)?;
                source
                    .checked_add(C220_MOV_UB_TO_OUT_UNIT_BYTES)
                    .ok_or(C220DmaMovError::AddressOverflow)?;
                destination
                    .checked_add(C220_MOV_UB_TO_OUT_UNIT_BYTES)
                    .ok_or(C220DmaMovError::AddressOverflow)?;
                segments.push(C220DmaMovSegment {
                    burst_index,
                    unit_index,
                    source_local: source,
                    destination_hbm: destination,
                    bytes: C220_MOV_UB_TO_OUT_UNIT_BYTES as u32,
                });
            }
        }
        Ok(segments)
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
    fn unsupported_modes_and_invalid_sizes_fail_closed() {
        let word = CAPTURED_C220_MOV_UB_TO_OUT_WORD;
        assert!(matches!(
            C220DmaMovDescriptor::decode(word ^ 1, 0x40010),
            Err(C220DmaMovError::UnsupportedWord { .. })
        ));
        assert!(matches!(
            C220DmaMovDescriptor::decode(word, 0x40011),
            Err(C220DmaMovError::UnsupportedMode { .. })
        ));
        assert!(matches!(
            C220DmaMovDescriptor::decode(word, 0x1_0000_0000 | 0x40010),
            Err(C220DmaMovError::UnsupportedGaps { .. })
        ));
        assert_eq!(
            C220DmaMovDescriptor::decode(word, 0),
            Err(C220DmaMovError::EmptyDescriptor)
        );
        assert!(matches!(
            C220DmaMovDescriptor::decode(word, 0xffff_0ff0),
            Err(C220DmaMovError::TooManySegments { .. })
        ));
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
    }
}
