use crate::numeric::fp32::Fp32ValueStatus;
use crate::sim::c220::numeric::fp16::C220Fp16Status;

use super::C220FixpFp16Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220FixpSourceFormat {
    Fp32,
    Int32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220FixpOutputFormat {
    Int32,
    Fp32,
    Fp16,
    Bf16,
}

impl C220FixpOutputFormat {
    pub const fn from_conversion_mode(source: C220FixpSourceFormat, mode: u8) -> Option<Self> {
        match (source, mode) {
            (C220FixpSourceFormat::Int32, 0) => Some(Self::Int32),
            (C220FixpSourceFormat::Fp32, 0) => Some(Self::Fp32),
            (C220FixpSourceFormat::Fp32, 1) => Some(Self::Fp16),
            (C220FixpSourceFormat::Fp32, 16) => Some(Self::Bf16),
            _ => None,
        }
    }

    pub const fn lane_bytes(self) -> u32 {
        match self {
            Self::Fp32 | Self::Int32 => 4,
            Self::Fp16 | Self::Bf16 => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220FixpLaneStatus {
    Integer,
    Fp32(Fp32ValueStatus),
    Fp16(C220Fp16Status),
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
