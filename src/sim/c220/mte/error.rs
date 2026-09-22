use crate::isa::c220::mte::{C220DmaMovError, C220MovOutToUbError};
use crate::memory::mapped::MappedMemoryError;
use crate::memory::ub::UbMemoryError;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum C220TransferError {
    #[error(transparent)]
    InputDescriptor(#[from] C220MovOutToUbError),
    #[error(transparent)]
    OutputDescriptor(#[from] C220DmaMovError),
    #[error(transparent)]
    Ub(#[from] UbMemoryError),
    #[error(transparent)]
    Destination(#[from] MappedMemoryError),
}
