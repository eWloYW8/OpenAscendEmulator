use super::timing::C220VectorWritePlanError;
use crate::memory::ub::UbMemoryError;
use crate::numeric::fp32::Fp32VectorError;
use thiserror::Error;

#[derive(Debug, thiserror::Error)]
pub enum C220VectorUopError {
    #[error("vector repeat {repeat_index} has no writeback schedule")]
    MissingVectorWriteback { repeat_index: usize },
    #[error(transparent)]
    WritePlan(#[from] C220VectorWritePlanError),
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
    #[error("C220 VA register {0} is out of range")]
    InvalidVaRegister(u8),
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
