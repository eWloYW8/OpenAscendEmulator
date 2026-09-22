use thiserror::Error;

pub const MAX_C310_MOV_ALIGN_COORDINATES: usize = 4096;

#[cfg(test)]
mod test_words {
    pub const C310_TILING_MOV_ALIGN_WORD: u32 = 0x748e_121c;
    pub const C310_SUB_TILING_MOV_ALIGN_WORD: u32 = 0x748e_219c;
    pub const C310_ADD_MOV_ALIGN_X_WORD: u32 = 0x74b2_1022;
    pub const C310_ADD_MOV_ALIGN_Y_WORD: u32 = 0x7484_00a2;
}

#[cfg(test)]
pub use test_words::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct C310MovAlignRegisterSelectors {
    pub destination: u8,
    pub source: u8,
    pub shape: u8,
    pub stride: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct C310MovAlignInstruction {
    pub word: u32,
    pub direction_field: u8,
    pub dtype_field: u8,
    pub source_memory_class: u8,
    pub destination_memory_class: u8,
}

impl C310MovAlignInstruction {
    pub const fn from_word(word: u32) -> Option<Self> {
        if word >> 29 != 3 || ((word >> 27) & 3) != 2 || ((word >> 24) & 7) != 4 {
            return None;
        }
        let direction_field = ((word >> 22) & 3) as u8;
        let (source_memory_class, destination_memory_class) = match direction_field {
            0 => (10, 8),
            1 => (8, 10),
            2 => (10, 9),
            _ => (9, 10),
        };
        Some(Self {
            word,
            direction_field,
            dtype_field: (word & 3) as u8,
            source_memory_class,
            destination_memory_class,
        })
    }
}

impl C310MovAlignRegisterSelectors {
    pub fn from_hbm_to_ub_word(word: u32) -> Result<Self, C310MovAlignError> {
        if !matches!(
            C310MovAlignInstruction::from_word(word),
            Some(C310MovAlignInstruction {
                source_memory_class: 10,
                destination_memory_class: 9,
                dtype_field: 0 | 2,
                ..
            })
        ) {
            return Err(C310MovAlignError::UnsupportedWord);
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
    ) -> C310MovAlignRegisters {
        C310MovAlignRegisters {
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
pub struct C310MovAlignRegisters {
    pub destination: u64,
    pub source: u64,
    pub shape: u64,
    pub stride: u64,
    pub loop_spr: u64,
    pub inner_stride_spr: u64,
    pub outer_stride_spr: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct C310MovAlignDecode {
    pub parameters: C310MovAlignParameters,
    pub burst_bytes: u32,
    pub source_memory_class: u8,
    pub destination_memory_class: u8,
}

#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum C310MovAlignError {
    #[error("unsupported MOV_ALIGN_V2 instruction word")]
    UnsupportedWord,
    #[error("unsupported MOV_ALIGN_V2 register shape or SPR mode")]
    UnsupportedRegisterMode,
}

impl C310MovAlignRegisters {
    pub fn decode_hbm_to_ub(self, word: u32) -> Result<C310MovAlignDecode, C310MovAlignError> {
        C310MovAlignRegisterSelectors::from_hbm_to_ub_word(word)?;
        const FIELD_MASK: u64 = (1 << 21) - 1;
        let burst_count = ((self.shape >> 4) & FIELD_MASK) as u32;
        let burst_bytes = ((self.shape >> 25) & FIELD_MASK) as u32;
        let inner_count = self.loop_spr & FIELD_MASK;
        let outer_count = (self.loop_spr >> 21) & ((1 << 22) - 1);
        if burst_count == 0
            || burst_bytes == 0
            || !burst_bytes.is_multiple_of(8)
            || inner_count == 0
            || outer_count == 0
        {
            return Err(C310MovAlignError::UnsupportedRegisterMode);
        }
        let source_stride = |value: u64| value & ((1_u64 << 40) - 1);
        let destination_stride = |value: u64| (value >> 40) & FIELD_MASK;
        Ok(C310MovAlignDecode {
            parameters: C310MovAlignParameters {
                source_base: self.source,
                destination_base: self.destination,
                burst_count,
                source_burst_stride: source_stride(self.stride),
                destination_burst_stride: destination_stride(self.stride),
                inner_count,
                source_inner_stride: source_stride(self.inner_stride_spr),
                destination_inner_stride: destination_stride(self.inner_stride_spr),
                outer_count,
                source_outer_stride: source_stride(self.outer_stride_spr),
                destination_outer_stride: destination_stride(self.outer_stride_spr),
            },
            burst_bytes,
            source_memory_class: 10,
            destination_memory_class: 9,
        })
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
