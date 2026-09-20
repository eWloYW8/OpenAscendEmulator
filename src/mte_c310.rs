use thiserror::Error;

use crate::architecture::Architecture;
use crate::isa::AicDecoderHint;

pub const MAX_C310_MOV_ALIGN_COORDINATES: usize = 4096;
pub const C310_TILING_MOV_ALIGN_WORD: u32 = 0x748e_121c;
pub const C310_SUB_TILING_MOV_ALIGN_WORD: u32 = 0x748e_219c;
pub const C310_ADD_MOV_ALIGN_X_WORD: u32 = 0x74b2_1022;
pub const C310_ADD_MOV_ALIGN_Y_WORD: u32 = 0x7484_00a2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct C310MovAlignRegisterSelectors {
    pub destination: u8,
    pub source: u8,
    pub shape: u8,
    pub stride: u8,
}

impl C310MovAlignRegisterSelectors {
    pub fn from_captured_word(word: u32) -> Result<Self, C310CapturedMovAlignError> {
        if !matches!(
            word,
            C310_ADD_MOV_ALIGN_X_WORD
                | C310_ADD_MOV_ALIGN_Y_WORD
                | C310_TILING_MOV_ALIGN_WORD
                | C310_SUB_TILING_MOV_ALIGN_WORD
                | 0x74ad_8bae
                | 0x74b3_6bae
                | 0x74e1_192c
                | 0x74c4_16a0
        ) {
            return Err(C310CapturedMovAlignError::UnsupportedWord);
        }
        Ok(Self {
            destination: ((word >> 17) & 0x1f) as u8,
            source: ((word >> 12) & 0x1f) as u8,
            shape: ((word >> 7) & 0x1f) as u8,
            stride: ((word >> 2) & 0x1f) as u8,
        })
    }

    pub fn capture(
        self,
        xregs: &[u64; 32],
        loop_spr: u64,
        inner_stride_spr: u64,
        outer_stride_spr: u64,
    ) -> C310CapturedMovAlignRegisters {
        C310CapturedMovAlignRegisters {
            destination: xregs[self.destination as usize],
            source: xregs[self.source as usize],
            shape: xregs[self.shape as usize],
            stride: xregs[self.stride as usize],
            loop_spr,
            inner_stride_spr,
            outer_stride_spr,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct C310MovAlignParameters {
    pub source_base: u64,
    pub destination_base: u64,
    pub burst_count: u32,
    pub source_burst_stride: u64,
    pub destination_burst_stride: u64,
    pub inner_count: u64,
    pub source_inner_stride: u64,
    pub destination_inner_stride: u64,
    pub outer_count: u64,
    pub source_outer_stride: u64,
    pub destination_outer_stride: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct C310MovAlignCoordinate {
    pub source_address: u64,
    pub destination_address: u64,
    pub burst_index: u32,
    pub inner_index: u64,
    pub outer_index: u64,
}

#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum C310MovAlignCoordinateError {
    #[error("MOV_ALIGN_V2 coordinate dimension is zero")]
    ZeroDimension,
    #[error("MOV_ALIGN_V2 coordinate count exceeds the bounded limit")]
    TooManyCoordinates,
    #[error("MOV_ALIGN_V2 coordinate address overflows u64")]
    AddressOverflow,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct C310CapturedMovAlignRegisters {
    pub destination: u64,
    pub source: u64,
    pub shape: u64,
    pub stride: u64,
    pub loop_spr: u64,
    pub inner_stride_spr: u64,
    pub outer_stride_spr: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct C310TilingMovAlignRegisters {
    pub source_xreg1: u64,
    pub shape_xreg4: u64,
    pub destination_and_stride_xreg7: u64,
    pub loop_spr105: u64,
    pub inner_stride_spr106: u64,
    pub outer_stride_spr107: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct C310CapturedMovAlignDecode {
    pub parameters: C310MovAlignParameters,
    pub burst_bytes: u32,
    pub source_memory_class: u8,
    pub destination_memory_class: u8,
    pub spr_indices: [u16; 3],
}

#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum C310CapturedMovAlignError {
    #[error("MOV_ALIGN_V2 word is outside the captured C310 Sub/Add fixtures")]
    UnsupportedWord,
    #[error("MOV_ALIGN_V2 register shape or SPR mode is outside the captured fixture")]
    UnsupportedRegisterMode,
}

impl C310CapturedMovAlignRegisters {
    pub fn decode_tiling_word(
        self,
        word: u32,
    ) -> Result<C310CapturedMovAlignDecode, C310CapturedMovAlignError> {
        if !matches!(
            word,
            C310_TILING_MOV_ALIGN_WORD | C310_SUB_TILING_MOV_ALIGN_WORD
        ) {
            return Err(C310CapturedMovAlignError::UnsupportedWord);
        }
        let Some(AicDecoderHint::C310MovAlignV2 {
            source_memory_class: 10,
            destination_memory_class: 9,
            dtype_field: 0,
            ..
        }) = AicDecoderHint::from_word(Architecture::Dav3510, word)
        else {
            return Err(C310CapturedMovAlignError::UnsupportedWord);
        };
        if self.shape != 0x4000_0010
            || self.stride != 0
            || self.destination != 0
            || self.loop_spr != 0x20_0001
            || self.inner_stride_spr != 0
            || self.outer_stride_spr != 0
        {
            return Err(C310CapturedMovAlignError::UnsupportedRegisterMode);
        }
        Ok(C310CapturedMovAlignDecode {
            parameters: C310MovAlignParameters {
                source_base: self.source,
                destination_base: self.destination,
                burst_count: 1,
                source_burst_stride: 0,
                destination_burst_stride: 0,
                inner_count: 1,
                source_inner_stride: 0,
                destination_inner_stride: 0,
                outer_count: 1,
                source_outer_stride: 0,
                destination_outer_stride: 0,
            },
            burst_bytes: 32,
            source_memory_class: 10,
            destination_memory_class: 9,
            spr_indices: [105, 106, 107],
        })
    }

    pub fn decode_hbm_to_ub_word(
        self,
        word: u32,
    ) -> Result<C310CapturedMovAlignDecode, C310CapturedMovAlignError> {
        if !matches!(
            word,
            C310_ADD_MOV_ALIGN_X_WORD | C310_ADD_MOV_ALIGN_Y_WORD | 0x74ad_8bae | 0x74b3_6bae
        ) {
            return Err(C310CapturedMovAlignError::UnsupportedWord);
        }
        self.decode_captured_word(word)
    }

    pub fn decode_captured_word(
        self,
        word: u32,
    ) -> Result<C310CapturedMovAlignDecode, C310CapturedMovAlignError> {
        let (spr_indices, expected_shape) = match word {
            C310_ADD_MOV_ALIGN_X_WORD | C310_ADD_MOV_ALIGN_Y_WORD | 0x74ad_8bae | 0x74b3_6bae => {
                ([105, 106, 107], 0x0400_0001_0000_0010)
            }
            0x74e1_192c | 0x74c4_16a0 => ([112, 113, 114], 0x0000_0001_0000_0010),
            _ => return Err(C310CapturedMovAlignError::UnsupportedWord),
        };
        let Some(AicDecoderHint::C310MovAlignV2 {
            source_memory_class,
            destination_memory_class,
            ..
        }) = AicDecoderHint::from_word(Architecture::Dav3510, word)
        else {
            return Err(C310CapturedMovAlignError::UnsupportedWord);
        };
        if self.shape != expected_shape
            || self.stride != 0x0000_8000_0000_0080
            || self.loop_spr != 0x20_0001
            || self.inner_stride_spr != 0
            || self.outer_stride_spr != 0
        {
            return Err(C310CapturedMovAlignError::UnsupportedRegisterMode);
        }
        let low_40_mask = (1u64 << 40) - 1;
        let high_21_mask = (1u64 << 21) - 1;
        let stride = |value: u64, class: u8| {
            if class == 10 {
                value & low_40_mask
            } else {
                (value >> 40) & high_21_mask
            }
        };
        let parameters = C310MovAlignParameters {
            source_base: self.source,
            destination_base: self.destination,
            burst_count: ((self.shape >> 4) & high_21_mask) as u32,
            source_burst_stride: stride(self.stride, source_memory_class),
            destination_burst_stride: stride(self.stride, destination_memory_class),
            inner_count: self.loop_spr & high_21_mask,
            source_inner_stride: stride(self.inner_stride_spr, source_memory_class),
            destination_inner_stride: stride(self.inner_stride_spr, destination_memory_class),
            outer_count: (self.loop_spr >> 21) & ((1u64 << 22) - 1),
            source_outer_stride: stride(self.outer_stride_spr, source_memory_class),
            destination_outer_stride: stride(self.outer_stride_spr, destination_memory_class),
        };
        Ok(C310CapturedMovAlignDecode {
            parameters,
            burst_bytes: ((self.shape >> 25) & high_21_mask) as u32,
            source_memory_class,
            destination_memory_class,
            spr_indices,
        })
    }
}

impl C310TilingMovAlignRegisters {
    pub fn decode(
        self,
        word: u32,
    ) -> Result<C310CapturedMovAlignDecode, C310CapturedMovAlignError> {
        if word != C310_TILING_MOV_ALIGN_WORD {
            return Err(C310CapturedMovAlignError::UnsupportedWord);
        }
        C310CapturedMovAlignRegisters {
            destination: self.destination_and_stride_xreg7,
            source: self.source_xreg1,
            shape: self.shape_xreg4,
            stride: self.destination_and_stride_xreg7,
            loop_spr: self.loop_spr105,
            inner_stride_spr: self.inner_stride_spr106,
            outer_stride_spr: self.outer_stride_spr107,
        }
        .decode_tiling_word(word)
    }
}

impl C310MovAlignParameters {
    pub fn coordinates(self) -> Result<Vec<C310MovAlignCoordinate>, C310MovAlignCoordinateError> {
        if self.burst_count == 0 || self.inner_count == 0 || self.outer_count == 0 {
            return Err(C310MovAlignCoordinateError::ZeroDimension);
        }
        let count = (self.burst_count as u64)
            .checked_mul(self.inner_count)
            .and_then(|value| value.checked_mul(self.outer_count))
            .ok_or(C310MovAlignCoordinateError::TooManyCoordinates)?;
        if count > MAX_C310_MOV_ALIGN_COORDINATES as u64 {
            return Err(C310MovAlignCoordinateError::TooManyCoordinates);
        }
        let mut coordinates = Vec::with_capacity(count as usize);
        for outer_index in 0..self.outer_count {
            for inner_index in 0..self.inner_count {
                for burst_index in 0..self.burst_count {
                    let source_address = address(
                        self.source_base,
                        outer_index,
                        self.source_outer_stride,
                        inner_index,
                        self.source_inner_stride,
                        burst_index,
                        self.source_burst_stride,
                    )?;
                    let destination_address = address(
                        self.destination_base,
                        outer_index,
                        self.destination_outer_stride,
                        inner_index,
                        self.destination_inner_stride,
                        burst_index,
                        self.destination_burst_stride,
                    )?;
                    coordinates.push(C310MovAlignCoordinate {
                        source_address,
                        destination_address,
                        burst_index,
                        inner_index,
                        outer_index,
                    });
                }
            }
        }
        Ok(coordinates)
    }
}

#[allow(clippy::too_many_arguments)]
fn address(
    base: u64,
    outer: u64,
    outer_stride: u64,
    inner: u64,
    inner_stride: u64,
    burst: u32,
    burst_stride: u64,
) -> Result<u64, C310MovAlignCoordinateError> {
    outer
        .checked_mul(outer_stride)
        .and_then(|value| base.checked_add(value))
        .and_then(|value| {
            inner
                .checked_mul(inner_stride)
                .and_then(|offset| value.checked_add(offset))
        })
        .and_then(|value| {
            u64::from(burst)
                .checked_mul(burst_stride)
                .and_then(|offset| value.checked_add(offset))
        })
        .ok_or(C310MovAlignCoordinateError::AddressOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn captured(source: u64, target: u64) -> C310MovAlignParameters {
        C310MovAlignParameters {
            source_base: source,
            destination_base: target,
            burst_count: 1,
            source_burst_stride: 128,
            destination_burst_stride: 128,
            inner_count: 1,
            source_inner_stride: 0,
            destination_inner_stride: 0,
            outer_count: 1,
            source_outer_stride: 0,
            destination_outer_stride: 0,
        }
    }

    fn captured_registers(word: u32, source: u64, target: u64) -> C310CapturedMovAlignRegisters {
        C310CapturedMovAlignRegisters {
            destination: target,
            source,
            shape: if matches!(word, 0x74e1_192c | 0x74c4_16a0) {
                0x0000_0001_0000_0010
            } else {
                0x0400_0001_0000_0010
            },
            stride: 0x0000_8000_0000_0080,
            loop_spr: 0x20_0001,
            inner_stride_spr: 0,
            outer_stride_spr: 0,
        }
    }

    #[test]
    fn captured_registers_predict_sub_and_add_coordinate_parameters() {
        for (word, source, target, classes, sprs) in [
            (
                C310_ADD_MOV_ALIGN_X_WORD,
                0x1b92_da00,
                0,
                (10, 9),
                [105, 106, 107],
            ),
            (
                C310_ADD_MOV_ALIGN_Y_WORD,
                0x1b92_ec00,
                0x80,
                (10, 9),
                [105, 106, 107],
            ),
            (0x74ad_8bae, 0x1b92_e200, 0, (10, 9), [105, 106, 107]),
            (0x74b3_6bae, 0x1b92_f400, 0x80, (10, 9), [105, 106, 107]),
            (0x74e1_192c, 0x100, 0x1b93_0600, (9, 10), [112, 113, 114]),
            (0x74c4_16a0, 0x100, 0x1b93_0600, (9, 10), [112, 113, 114]),
        ] {
            let decoded = captured_registers(word, source, target)
                .decode_captured_word(word)
                .unwrap();
            assert_eq!(decoded.parameters, captured(source, target));
            assert_eq!(decoded.burst_bytes, 128);
            assert_eq!(
                (
                    decoded.source_memory_class,
                    decoded.destination_memory_class
                ),
                classes
            );
            assert_eq!(decoded.spr_indices, sprs);
        }
    }

    #[test]
    fn first_tiling_word_uses_one_32_byte_hbm_to_ub_coordinate() {
        let registers = C310TilingMovAlignRegisters {
            source_xreg1: 0x1c93_1200,
            shape_xreg4: 0x4000_0010,
            destination_and_stride_xreg7: 0,
            loop_spr105: 0x20_0001,
            inner_stride_spr106: 0,
            outer_stride_spr107: 0,
        };
        let decoded = registers.decode(C310_TILING_MOV_ALIGN_WORD).unwrap();
        assert_eq!(
            (
                decoded.source_memory_class,
                decoded.destination_memory_class
            ),
            (10, 9)
        );
        assert_eq!(decoded.spr_indices, [105, 106, 107]);
        assert_eq!(decoded.burst_bytes, 32);
        assert_eq!(
            decoded.parameters.coordinates().unwrap(),
            [C310MovAlignCoordinate {
                source_address: 0x1c93_1200,
                destination_address: 0,
                burst_index: 0,
                inner_index: 0,
                outer_index: 0,
            }]
        );
        assert_eq!(
            registers.decode(C310_TILING_MOV_ALIGN_WORD ^ 1),
            Err(C310CapturedMovAlignError::UnsupportedWord)
        );
        assert_eq!(
            C310TilingMovAlignRegisters {
                loop_spr105: 0,
                ..registers
            }
            .decode(C310_TILING_MOV_ALIGN_WORD),
            Err(C310CapturedMovAlignError::UnsupportedRegisterMode)
        );
        assert_eq!(
            C310TilingMovAlignRegisters {
                shape_xreg4: 0x4000_0030,
                ..registers
            }
            .decode(C310_TILING_MOV_ALIGN_WORD),
            Err(C310CapturedMovAlignError::UnsupportedRegisterMode)
        );
    }

    #[test]
    fn tiling_variants_select_distinct_source_and_shape_registers() {
        let mut xregs = [0_u64; 32];
        xregs[1] = 0x3000;
        xregs[2] = 0x4000;
        xregs[3] = 0x4000_0010;
        xregs[4] = 0x4000_0010;
        for (word, source_register, shape_register, source_address) in [
            (C310_TILING_MOV_ALIGN_WORD, 1, 4, 0x3000),
            (C310_SUB_TILING_MOV_ALIGN_WORD, 2, 3, 0x4000),
        ] {
            let selectors = C310MovAlignRegisterSelectors::from_captured_word(word).unwrap();
            assert_eq!(selectors.destination, 7);
            assert_eq!(selectors.source, source_register);
            assert_eq!(selectors.shape, shape_register);
            assert_eq!(selectors.stride, 7);
            let decoded = selectors
                .capture(&xregs, 0x20_0001, 0, 0)
                .decode_tiling_word(word)
                .unwrap();
            assert_eq!(decoded.burst_bytes, 32);
            assert_eq!(decoded.parameters.source_base, source_address);
            assert_eq!(decoded.parameters.destination_base, 0);
            assert_eq!(decoded.parameters.coordinates().unwrap().len(), 1);
        }
        xregs[3] = 0x4000_0030;
        assert_eq!(
            C310MovAlignRegisterSelectors::from_captured_word(C310_SUB_TILING_MOV_ALIGN_WORD)
                .unwrap()
                .capture(&xregs, 0x20_0001, 0, 0)
                .decode_tiling_word(C310_SUB_TILING_MOV_ALIGN_WORD),
            Err(C310CapturedMovAlignError::UnsupportedRegisterMode)
        );
    }

    #[test]
    fn rejects_other_words_and_register_modes() {
        let registers = captured_registers(0x74ad_8bae, 0, 0);
        assert_eq!(
            registers.decode_captured_word(0x74ad_8baf),
            Err(C310CapturedMovAlignError::UnsupportedWord)
        );
        let mut modified = registers;
        modified.shape ^= 1;
        assert_eq!(
            modified.decode_captured_word(0x74ad_8bae),
            Err(C310CapturedMovAlignError::UnsupportedRegisterMode)
        );
        assert_eq!(
            captured_registers(0x74c4_16a0, 0x100, 0x1b93_0600).decode_hbm_to_ub_word(0x74c4_16a0),
            Err(C310CapturedMovAlignError::UnsupportedWord)
        );
    }

    #[test]
    fn captured_words_select_their_own_physical_x_registers() {
        for (word, expected) in [
            (C310_ADD_MOV_ALIGN_X_WORD, [25, 1, 0, 8]),
            (C310_ADD_MOV_ALIGN_Y_WORD, [2, 0, 1, 8]),
            (0x74ad_8bae, [22, 24, 23, 11]),
            (0x74b3_6bae, [25, 22, 23, 11]),
            (0x74e1_192c, [16, 17, 18, 11]),
            (0x74c4_16a0, [2, 1, 13, 8]),
        ] {
            let selectors = C310MovAlignRegisterSelectors::from_captured_word(word).unwrap();
            assert_eq!(
                [
                    selectors.destination,
                    selectors.source,
                    selectors.shape,
                    selectors.stride,
                ],
                expected
            );
        }
        assert_eq!(
            C310MovAlignRegisterSelectors::from_captured_word(0x74ad_8baf),
            Err(C310CapturedMovAlignError::UnsupportedWord)
        );
    }

    #[test]
    fn captured_sub_x_y_and_z_addresses() {
        for (source, target) in [(0x1b92_e200, 0), (0x1b92_f400, 0x80), (0x100, 0x1b93_0600)] {
            assert_eq!(
                captured(source, target).coordinates().unwrap(),
                [C310MovAlignCoordinate {
                    source_address: source,
                    destination_address: target,
                    burst_index: 0,
                    inner_index: 0,
                    outer_index: 0,
                }]
            );
        }
    }

    #[test]
    fn nested_loop_order_and_strides() {
        let parameters = C310MovAlignParameters {
            source_base: 1000,
            destination_base: 2000,
            burst_count: 2,
            source_burst_stride: 4,
            destination_burst_stride: 8,
            inner_count: 2,
            source_inner_stride: 16,
            destination_inner_stride: 32,
            outer_count: 2,
            source_outer_stride: 64,
            destination_outer_stride: 128,
        };
        let coordinates = parameters.coordinates().unwrap();
        assert_eq!(coordinates.len(), 8);
        assert_eq!(coordinates[0].source_address, 1000);
        assert_eq!(coordinates[1].source_address, 1004);
        assert_eq!(coordinates[2].source_address, 1016);
        assert_eq!(coordinates[4].source_address, 1064);
        assert_eq!(coordinates[7].destination_address, 2168);
        assert_eq!(
            (
                coordinates[7].outer_index,
                coordinates[7].inner_index,
                coordinates[7].burst_index
            ),
            (1, 1, 1)
        );
    }

    #[test]
    fn rejects_unbounded_or_overflowing_coordinates() {
        let mut parameters = captured(0, 0);
        parameters.burst_count = 0;
        assert_eq!(
            parameters.coordinates(),
            Err(C310MovAlignCoordinateError::ZeroDimension)
        );
        parameters.burst_count = 4097;
        assert_eq!(
            parameters.coordinates(),
            Err(C310MovAlignCoordinateError::TooManyCoordinates)
        );
        parameters.burst_count = 2;
        parameters.source_base = u64::MAX;
        assert_eq!(
            parameters.coordinates(),
            Err(C310MovAlignCoordinateError::AddressOverflow)
        );
    }
}
