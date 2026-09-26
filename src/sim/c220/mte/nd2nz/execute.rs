use crate::isa::c220::mte::nd2nz::C220Nd2NzTransfer;
use crate::memory::mapped::{MappedMemory, MappedMemoryError};
use crate::memory::sparse::MemoryByteState;
use crate::sim::c220::memory::{C220LocalBuffer, C220LocalBufferError};

#[derive(Debug, thiserror::Error)]
pub enum C220Nd2NzError {
    #[error(transparent)]
    Source(#[from] MappedMemoryError),
    #[error(transparent)]
    Destination(#[from] C220LocalBufferError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct C220Nd2NzResult {
    pub segments: u64,
    pub source_bytes: u64,
    pub destination_bytes: u64,
    pub padding_bytes: u64,
    pub unknown_destination_bytes: u64,
    pub discarded_destination_bytes: u64,
}

/// Functional execution in matrix, row, block order. The timing owner decides
/// when to call this function; it does not predict completion cycles.
pub fn execute_c220_nd2nz(
    transfer: C220Nd2NzTransfer,
    source: &MappedMemory,
    destination: &mut C220LocalBuffer,
) -> Result<C220Nd2NzResult, C220Nd2NzError> {
    let mut result = C220Nd2NzResult::default();
    for segment in transfer.segments() {
        let input = source.read_states_at(segment.source_address, segment.input_bytes as usize)?;
        let mut output = [MemoryByteState::Known(0); 64];
        output[..input.len()].copy_from_slice(&input);
        let output = &output[..segment.output_bytes as usize];
        result.segments += 1;
        result.source_bytes += u64::from(segment.input_bytes);
        if segment
            .destination_address
            .checked_add(u64::from(segment.output_bytes - 1))
            .is_none()
        {
            result.discarded_destination_bytes += u64::from(segment.output_bytes);
            continue;
        }
        destination.write_states_linear(segment.destination_address, output)?;
        result.destination_bytes += u64::from(segment.output_bytes);
        result.padding_bytes += u64::from(segment.output_bytes - segment.input_bytes);
        result.unknown_destination_bytes += input
            .iter()
            .filter(|b| **b == MemoryByteState::Unknown)
            .count() as u64;
    }
    Ok(result)
}
