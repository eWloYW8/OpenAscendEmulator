use crate::isa::c220::mte::l1_to_out::C220MovL1ToOutTransfer;
use crate::memory::mapped::{MappedMemory, MappedMemoryError};
use crate::memory::sparse::MemoryByteState;
use crate::memory::ub::UbTransferResult;
use crate::sim::c220::memory::{C220LocalBuffer, C220LocalBufferError};

use super::atomic::{C220AtomicConfig, commit_output};

mod engine;
mod read_plan;
pub use engine::{
    C220L1OutputCommand, C220L1OutputCommandState, C220L1OutputEngine, C220L1OutputEngineConfig,
    C220L1OutputEngineError,
};
pub use read_plan::{C220L1OutputRead, C220L1OutputReadPlan, C220L1OutputRoute};

#[derive(Debug, thiserror::Error)]
pub enum C220L1OutputError {
    #[error("L1 output address range overflows")]
    AddressOverflow,
    #[error(transparent)]
    Source(#[from] C220LocalBufferError),
    #[error(transparent)]
    Destination(#[from] MappedMemoryError),
}

/// Executes memory effects in transfer order; the caller controls retirement.
pub fn execute_c220_mov_l1_to_out(
    source: &C220LocalBuffer,
    destination: &mut MappedMemory,
    transfer: C220MovL1ToOutTransfer,
    control: u64,
    atomic: C220AtomicConfig,
) -> Result<UbTransferResult, C220L1OutputError> {
    let segments = transfer
        .segments()
        .ok_or(C220L1OutputError::AddressOverflow)?;
    let mut result = UbTransferResult {
        segment_count: 0,
        bytes: 0,
        known_bytes: 0,
        unknown_bytes: 0,
    };
    for segment in segments {
        let states = source.read_initialized_states_linear(segment.source_address, 32)?;
        let known_bytes = states
            .iter()
            .filter(|state| matches!(state, MemoryByteState::Known(_)))
            .count();
        let written = commit_output(
            &[(segment.destination_address, states)],
            destination,
            control,
            atomic,
            UbTransferResult {
                segment_count: 1,
                bytes: 32,
                known_bytes,
                unknown_bytes: 32 - known_bytes,
            },
        )?;
        result.segment_count += written.segment_count;
        result.bytes += written.bytes;
        result.known_bytes += written.known_bytes;
        result.unknown_bytes += written.unknown_bytes;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::c220::mte::{l1_to_out::C220MovL1ToOutInstruction, read_register_mask};
    use crate::memory::{region::MemoryRegion, sparse::SparseMemory};

    #[test]
    fn strided_output_preserves_unknowns_and_applies_atomics() {
        let word = (3 << 29) | (2 << 27) | (4 << 23) | (1 << 17) | (2 << 12) | (3 << 7) | (2 << 3);
        let mut registers = [0; 32];
        registers[1] = 0x1000;
        registers[2] = 64;
        registers[3] = (2 << 48) | (1 << 32) | (2 << 16) | (2 << 4) | 5;
        for ignored in [0, 7, 1 << 22, (1 << 22) | 7] {
            let instruction = C220MovL1ToOutInstruction::decode(word | ignored).unwrap();
            assert_eq!(read_register_mask(word | ignored), Some(14));
            let transfer = instruction.capture(&registers);
            assert_eq!(transfer.sid(), 5);
            let segments: Vec<_> = transfer.segments().unwrap().collect();
            assert_eq!(segments.len(), 4);
            assert_eq!(segments[2].source_address, 160);
            assert_eq!(segments[2].destination_address, 0x1080);
            let mut source = C220LocalBuffer::new(16);
            for segment in &segments {
                source
                    .write_states_linear(segment.source_address, &[MemoryByteState::Known(1); 32])
                    .unwrap();
            }
            source
                .write_states_linear(64, &[MemoryByteState::Unknown])
                .unwrap();
            let mut destination = MappedMemory::bind(
                SparseMemory::new(
                    vec![MemoryRegion::new(256, vec![2; 256]).unwrap()],
                    256,
                    256,
                ),
                &[0x1000],
            )
            .unwrap();
            let result = execute_c220_mov_l1_to_out(
                &source,
                &mut destination,
                transfer,
                5 << 6,
                C220AtomicConfig {
                    enabled: true,
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(
                (result.segment_count, result.bytes, result.unknown_bytes),
                (4, 128, 1)
            );
            assert_eq!(
                destination.read_states_at(0x1000, 1).unwrap(),
                [MemoryByteState::Unknown]
            );
            assert_eq!(destination.read_known_at(0x1001, 63).unwrap(), vec![3; 63]);
            assert_eq!(destination.read_known_at(0x1040, 64).unwrap(), vec![2; 64]);
            assert_eq!(destination.read_known_at(0x1080, 64).unwrap(), vec![3; 64]);
            let mut disabled = transfer;
            disabled.xm = 0;
            assert_eq!(disabled.segments().unwrap().len(), 0);
        }
    }
}
