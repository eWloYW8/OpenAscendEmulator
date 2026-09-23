use crate::isa::c220::mte::C220MovOutToUbError;
use crate::memory::ub::UbMemoryError;
use crate::sim::c220::mte::C220TransferError;
use crate::sim::c220::vector::C220VectorError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum C220ExecutionError {
    #[error("program ended before MTE word at PC {pc:#x}")]
    ProgramEnded { pc: u64 },
    #[error("word {word:#010x} at PC {pc:#x} is not an implemented MTE2 or flag operation")]
    UnsupportedWord { pc: u64, word: u32 },
    #[error("SPR {index} is unavailable for MTE word at PC {pc:#x}")]
    MissingSpr { pc: u64, index: u16 },
    #[error("MTE2 transfer byte count overflows usize")]
    TransferSizeOverflow,
    #[error(transparent)]
    C220(#[from] C220MovOutToUbError),
    #[error(transparent)]
    Ub(#[from] UbMemoryError),
    #[error(transparent)]
    C220Transfer(#[from] C220TransferError),
    #[error(transparent)]
    Vector(#[from] C220VectorError),
}
