use crate::isa::c220::mte::{
    C220DmaMovDescriptor, C220DmaMovError, C220MovOutToUbDescriptor, C220MovOutToUbError,
    C220MovOutToUbSegment,
};
use crate::memory::mapped::{MappedMemory, MappedMemoryError};
use crate::memory::sparse::MemoryByteState;
use crate::memory::ub::{UbMemory, UbMemoryError, UbTransferResult};
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

pub struct C220PreparedOutput {
    pub(crate) writes: Vec<(u64, Vec<MemoryByteState>)>,
    pub result: UbTransferResult,
}

pub fn copy_c220_mov_out_to_ub(
    ub: &mut UbMemory,
    source: &MappedMemory,
    descriptor: C220MovOutToUbDescriptor,
    source_address: u64,
    destination_address: u64,
) -> Result<UbTransferResult, C220TransferError> {
    let segments = descriptor.segments(source_address, destination_address)?;
    Ok(ub.copy_segments(
        source,
        segments.into_iter().map(|segment| {
            (
                segment.source_hbm,
                segment.destination_local,
                segment.bytes as usize,
            )
        }),
    )?)
}

pub fn copy_c220_mov_ub_to_hbm(
    ub: &UbMemory,
    destination: &mut MappedMemory,
    descriptor: C220DmaMovDescriptor,
    source_address: u64,
    destination_address: u64,
) -> Result<UbTransferResult, C220TransferError> {
    let prepared = prepare_c220_mov_ub_to_hbm(ub, descriptor, source_address, destination_address)?;
    destination.write_segments_at(&prepared.writes)?;
    Ok(prepared.result)
}

pub fn prepare_c220_mov_ub_to_hbm(
    ub: &UbMemory,
    descriptor: C220DmaMovDescriptor,
    source_address: u64,
    destination_address: u64,
) -> Result<C220PreparedOutput, C220TransferError> {
    let segments = descriptor.segments(source_address, destination_address)?;
    let bytes = segments
        .len()
        .checked_mul(32)
        .ok_or(UbMemoryError::ResultSizeOverflow)?;
    let mut writes = Vec::new();
    writes
        .try_reserve_exact(segments.len())
        .map_err(|_| UbMemoryError::HostAllocationFailed {
            requested: segments.len(),
        })?;
    let mut known_bytes = 0;
    for segment in &segments {
        let states = ub.read_states(segment.source_local, segment.bytes as usize)?;
        known_bytes += states
            .iter()
            .filter(|state| matches!(state, MemoryByteState::Known(_)))
            .count();
        writes.push((segment.destination_hbm, states));
    }
    Ok(C220PreparedOutput {
        writes,
        result: UbTransferResult {
            segment_count: segments.len(),
            bytes,
            known_bytes,
            unknown_bytes: bytes - known_bytes,
        },
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Mte2TransferPlan {
    pub descriptor: C220MovOutToUbDescriptor,
    pub source_address: u64,
    pub destination_address: u64,
    pub bytes: usize,
    pub dma_mode_word: u64,
}

impl C220Mte2TransferPlan {
    pub fn descriptor_segments(self) -> Result<Vec<C220MovOutToUbSegment>, C220MovOutToUbError> {
        self.descriptor
            .segments(self.source_address, self.destination_address)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Mte3TransferPlan {
    pub descriptor: C220DmaMovDescriptor,
    pub source_address: u64,
    pub destination_address: u64,
    pub bytes: usize,
    pub dma_mode_word: u64,
}
