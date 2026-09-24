use crate::numeric::fp32::Fp32ValueStatus;
use crate::sim::c220::numeric::fp16::C220Fp16Status;

use super::C220FixpFp16Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220FixpSourceFormat {
    Fp32,
    Int32,
    Fp16,
}

impl C220FixpSourceFormat {
    pub const fn lane_bytes(self) -> u32 {
        match self {
            Self::Fp32 | Self::Int32 => 4,
            Self::Fp16 => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220FixpOutputFormat {
    Int4,
    /// Each lane's factor selects signed or unsigned byte interpretation.
    Bits8,
    Int32,
    Int16,
    Fp32,
    Fp16,
    Bf16,
}

impl C220FixpOutputFormat {
    pub const fn from_conversion_mode(source: C220FixpSourceFormat, mode: u8) -> Option<Self> {
        match (source, mode) {
            (_, 6) => Some(Self::Fp16),
            (C220FixpSourceFormat::Fp16, _) => None,
            (C220FixpSourceFormat::Int32, 0) => Some(Self::Int32),
            (C220FixpSourceFormat::Fp32, 0) => Some(Self::Fp32),
            // Nonzero conversion modes select the interpretation of 32-bit
            // source lanes, independently of the instruction's source tag.
            (_, 8 | 9 | 23 | 24) => Some(Self::Bits8),
            (_, 21 | 22 | 25 | 26) => Some(Self::Int4),
            (_, 12 | 13) => Some(Self::Int16),
            (_, 1 | 10 | 11) => Some(Self::Fp16),
            (_, 16) => Some(Self::Bf16),
            _ => None,
        }
    }

    /// Byte extent of complete elements; an unpaired Int4 lane is not stored.
    pub const fn storage_bytes(self, lanes: u32) -> u32 {
        match self {
            Self::Int4 => lanes / 2,
            Self::Bits8 => lanes,
            Self::Fp32 | Self::Int32 => lanes.wrapping_mul(4),
            Self::Fp16 | Self::Bf16 | Self::Int16 => lanes.wrapping_mul(2),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220FixpLaneStatus {
    /// Output is cleared without applying a numerical conversion.
    Cleared,
    Requant(crate::sim::c220::numeric::requant::C220FixpRequantOutcome),
    Integer,
    Int16(crate::sim::c220::numeric::fixp::C220FixpInt16Status),
    Fp32(Fp32ValueStatus),
    Fp16(C220Fp16Status),
    DequantFp16(crate::sim::c220::numeric::fixp::C220FixpDequantFp16Outcome),
    Bf16(Fp32ValueStatus),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220FixpConversionResult {
    pub format: C220FixpOutputFormat,
    pub bytes: Vec<u8>,
    /// Diagnostic status, without an implicit architectural register write.
    pub lane_status: Vec<C220FixpLaneStatus>,
}

impl From<C220FixpFp16Result> for C220FixpConversionResult {
    fn from(result: C220FixpFp16Result) -> Self {
        Self {
            format: C220FixpOutputFormat::Fp16,
            bytes: result.bytes,
            lane_status: result
                .lane_status
                .into_iter()
                .map(C220FixpLaneStatus::Fp16)
                .collect(),
        }
    }
}
