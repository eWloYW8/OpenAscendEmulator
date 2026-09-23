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
    #[error("C220 output was not produced before its flag")]
    OutputNotProduced,
    #[error("C220 output is still waiting for a flag")]
    UnsignaledOutput,
    #[error("C220 output flag {flag_id} is already set")]
    OutputFlagAlreadySet { flag_id: u8 },
    #[error("C220 output flag {flag_id} was not set before wait")]
    OutputWaitWithoutFlag { flag_id: u8 },
    #[error("C220 output flag ID {flag_id} is outside the modeled range")]
    UnsupportedOutputFlagId { flag_id: u32 },
    #[error("C220 output transfer has no matching completed flag")]
    OutputNotReady,
    #[error("C220 output dependency is already waiting for transfer")]
    OutputDependencyOutstanding,
    #[error("C220 MTE2 reuse flag {flag_id} is already set")]
    ReuseFlagAlreadySet { flag_id: u8 },
    #[error("C220 MTE2 reuse flag {flag_id} was not set before wait")]
    ReuseWaitWithoutFlag { flag_id: u8 },
    #[error("C220 MTE3 completion flag {flag_id} is already set")]
    CompletionFlagAlreadySet { flag_id: u8 },
    #[error("C220 MTE3 completion flag {flag_id} was not set before wait")]
    CompletionWaitWithoutFlag { flag_id: u8 },
    #[error("C220 output was not copied before MTE3 completion flag")]
    OutputNotCopied,
    #[error(transparent)]
    C220(#[from] C220MovOutToUbError),
    #[error(transparent)]
    Ub(#[from] UbMemoryError),
    #[error(transparent)]
    C220Transfer(#[from] C220TransferError),
    #[error(transparent)]
    Vector(#[from] C220VectorError),
}
