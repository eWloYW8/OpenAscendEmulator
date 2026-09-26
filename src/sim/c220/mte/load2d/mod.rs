mod external;
mod transpose;
pub use external::{
    C220Load2dExternalRequest, C220Load2dExternalRequests, prepare_c220_external_load2d,
};
mod uop;
pub use transpose::prepare_c220_load2d_transpose;
pub use uop::{C220Load2dReadUop, C220Load2dRequestPlan};

use thiserror::Error;

use crate::isa::c220::mte::load2d::{C220Load2dDestination, C220Load2dError, C220Load2dTransfer};
use crate::memory::sparse::MemoryByteState;
use crate::sim::c220::memory::{C220LocalBufferError, C220LocalMemory};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Load2dTransferResult {
    pub segment_count: usize,
    pub bytes: usize,
    pub known_bytes: usize,
    pub unknown_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220PreparedLoad2d {
    destination: C220Load2dDestination,
    writes: Vec<(u64, Vec<MemoryByteState>)>,
    pub result: C220Load2dTransferResult,
}

impl C220PreparedLoad2d {
    pub const fn destination(&self) -> C220Load2dDestination {
        self.destination
    }

    pub fn commit(self, memory: &mut C220LocalMemory) -> Result<(), C220Load2dTransferError> {
        for (address, states) in self.writes {
            match self.destination {
                C220Load2dDestination::L0a => {
                    memory.l0a_mut().write_states_linear(address, &states)?;
                }
                C220Load2dDestination::L0b => {
                    memory.l0b_mut().write_states_linear(address, &states)?;
                }
                C220Load2dDestination::L1 => {
                    memory.l1_mut().write_states_linear(address, &states)?;
                }
                destination => {
                    return Err(C220Load2dTransferError::UnsupportedDestination(destination));
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum C220Load2dTransferError {
    #[error(transparent)]
    Decode(#[from] C220Load2dError),
    #[error(transparent)]
    LocalBuffer(#[from] C220LocalBufferError),
    #[error(transparent)]
    ExternalMemory(#[from] crate::memory::mapped::MappedMemoryError),
    #[error("unsupported LOAD_2D destination {0:?}")]
    UnsupportedDestination(C220Load2dDestination),
    #[error("cannot reserve {requested} LOAD_2D write records")]
    AllocationFailed { requested: usize },
}

pub fn prepare_c220_load2d(
    memory: &C220LocalMemory,
    transfer: C220Load2dTransfer,
) -> Result<C220PreparedLoad2d, C220Load2dTransferError> {
    if !transfer.instruction.is_mte1() {
        return Err(C220Load2dError::UnsupportedRoute {
            source_buffer: transfer.instruction.source,
            destination_buffer: transfer.instruction.destination,
        }
        .into());
    }
    prepare_blocks(transfer, |address, bytes| {
        Ok(memory.l1().read_initialized_states_linear(address, bytes)?)
    })
}

fn prepare_blocks(
    transfer: C220Load2dTransfer,
    mut read: impl FnMut(u64, usize) -> Result<Vec<MemoryByteState>, C220Load2dTransferError>,
) -> Result<C220PreparedLoad2d, C220Load2dTransferError> {
    if !matches!(
        transfer.instruction.destination,
        C220Load2dDestination::L0a | C220Load2dDestination::L0b | C220Load2dDestination::L1
    ) {
        return Err(C220Load2dTransferError::UnsupportedDestination(
            transfer.instruction.destination,
        ));
    }
    let count = usize::from(transfer.descriptor.repeat_count);
    let mut writes = Vec::new();
    writes
        .try_reserve_exact(count)
        .map_err(|_| C220Load2dTransferError::AllocationFailed { requested: count })?;
    let mut known_bytes = 0;
    for segment in transfer.segments() {
        let mut states = read(segment.source_address, segment.bytes as usize)?;
        if transfer.instruction.transpose {
            states = transpose_halfwords(states);
        }
        known_bytes += states
            .iter()
            .filter(|state| matches!(state, MemoryByteState::Known(_)))
            .count();
        writes.push((segment.destination_address, states));
    }
    let bytes = count * 512;
    Ok(C220PreparedLoad2d {
        destination: transfer.instruction.destination,
        writes,
        result: C220Load2dTransferResult {
            segment_count: count,
            bytes,
            known_bytes,
            unknown_bytes: bytes - known_bytes,
        },
    })
}

fn transpose_halfwords(states: Vec<MemoryByteState>) -> Vec<MemoryByteState> {
    let mut transposed = vec![MemoryByteState::Unknown; states.len()];
    for (source_index, halfword) in states.chunks_exact(2).enumerate() {
        let destination_offset = (source_index / 16 + (source_index % 16) * 16) * 2;
        transposed[destination_offset..destination_offset + 2].copy_from_slice(halfword);
    }
    transposed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::c220::mte::load2d::C220Load2dInstruction;
    use crate::sim::c220::memory::C220LocalMemoryConfig;

    #[test]
    fn load2d_preserves_linear_addresses_and_unknown_bytes() {
        for destination in 0..=1 {
            for format_bits in [0, 8, 4, 6, 12, 14] {
                let transpose = format_bits & 4 != 0;
                let mut memory = C220LocalMemory::new(C220LocalMemoryConfig {
                    l1_bytes: 768,
                    l0a_bytes: 768,
                    l0b_bytes: 768,
                    ..Default::default()
                })
                .unwrap();
                memory.l1_mut().write_known(0, &[99; 512]).unwrap();
                let mut source = (0..=u8::MAX)
                    .map(MemoryByteState::Known)
                    .collect::<Vec<_>>();
                source[200] = MemoryByteState::Unknown;
                memory.l1_mut().write_states_linear(512, &source).unwrap();
                source.resize(512, MemoryByteState::Known(0));
                let mut registers = [0; 32];
                registers[0] = 512;
                registers[2] = 512;
                registers[3] = (1 << 16) | (1 << 24);
                let word = 0x6000_2180 | destination | format_bits;
                let transfer = C220Load2dInstruction::decode(word)
                    .unwrap()
                    .capture(&registers)
                    .unwrap();
                let prepared = prepare_c220_load2d(&memory, transfer).unwrap();
                assert_eq!(prepared.result.known_bytes, 511);
                assert_eq!(prepared.result.unknown_bytes, 1);
                prepared.commit(&mut memory).unwrap();
                let target = if destination == 0 {
                    memory.l0a()
                } else {
                    memory.l0b()
                };
                let mut expected = vec![MemoryByteState::Known(0); 512];
                for (index, state) in source.into_iter().enumerate() {
                    let destination_index = if transpose {
                        let row = index / 32;
                        let column = (index % 32) / 2;
                        column * 32 + row * 2 + index % 2
                    } else {
                        index
                    };
                    expected[destination_index] = state;
                }
                assert_eq!(target.read_states_linear(512, 512).unwrap(), expected);
                assert_eq!(
                    target.read_states(0, 512).unwrap(),
                    vec![MemoryByteState::Unknown; 512]
                );
                assert_eq!(target.tracked_bytes(), 512);
                assert_eq!(memory.l1().tracked_bytes(), 768);
                assert_eq!(
                    memory.l1().read_states_linear(768, 256).unwrap(),
                    vec![MemoryByteState::Unknown; 256]
                );
            }
        }
    }
}
