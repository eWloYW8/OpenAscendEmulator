mod uop;
pub use uop::{C220Load2dReadUop, C220Load2dRequestPlan};

use thiserror::Error;

use crate::isa::c220::mte::load2d::{
    C220Load2dDestination, C220Load2dElementFormat, C220Load2dError, C220Load2dTransfer,
};
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
    #[error("LOAD_2D destination {0:?} is not implemented by the MTE1 data path")]
    UnsupportedDestination(C220Load2dDestination),
    #[error("cannot reserve {requested} LOAD_2D write records")]
    AllocationFailed { requested: usize },
}

pub fn prepare_c220_load2d(
    memory: &C220LocalMemory,
    transfer: C220Load2dTransfer,
) -> Result<C220PreparedLoad2d, C220Load2dTransferError> {
    if !matches!(
        transfer.instruction.destination,
        C220Load2dDestination::L0a | C220Load2dDestination::L0b
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
        let mut states = memory
            .l1()
            .read_states_linear(segment.source_address, segment.bytes as usize)?;
        if transfer.instruction.transpose {
            states = transpose_block(states, transfer.instruction.element_format);
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

fn transpose_block(
    states: Vec<MemoryByteState>,
    element_format: C220Load2dElementFormat,
) -> Vec<MemoryByteState> {
    match element_format {
        C220Load2dElementFormat::B4 => transpose_b4(states),
        C220Load2dElementFormat::B8 => transpose_bytes(states, 1),
        C220Load2dElementFormat::B16 => transpose_bytes(states, 2),
        C220Load2dElementFormat::B32 => transpose_bytes(states, 4),
    }
}

fn transpose_bytes(states: Vec<MemoryByteState>, element_bytes: usize) -> Vec<MemoryByteState> {
    let element_count = states.len() / element_bytes;
    let destination_row_length = element_count / 16;
    let mut transposed = vec![MemoryByteState::Unknown; states.len()];
    for source_index in 0..element_count {
        let destination_index = source_index / 16 + (source_index % 16) * destination_row_length;
        let source_offset = source_index * element_bytes;
        let destination_offset = destination_index * element_bytes;
        transposed[destination_offset..destination_offset + element_bytes]
            .copy_from_slice(&states[source_offset..source_offset + element_bytes]);
    }
    transposed
}

fn transpose_b4(states: Vec<MemoryByteState>) -> Vec<MemoryByteState> {
    let nibble_count = states.len() * 2;
    let destination_row_length = nibble_count / 16;
    let mut transposed = vec![None; nibble_count];
    for source_index in 0..nibble_count {
        let source = states[source_index / 2];
        let nibble = match source {
            MemoryByteState::Known(byte) if source_index & 1 == 0 => Some(byte & 0xf),
            MemoryByteState::Known(byte) => Some(byte >> 4),
            MemoryByteState::Unknown => None,
        };
        let destination_index = source_index / 16 + (source_index % 16) * destination_row_length;
        transposed[destination_index] = nibble;
    }
    transposed
        .chunks_exact(2)
        .map(|pair| match (pair[0], pair[1]) {
            (Some(low), Some(high)) => MemoryByteState::Known(low | (high << 4)),
            _ => MemoryByteState::Unknown,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::c220::mte::load2d::C220Load2dInstruction;
    use crate::sim::c220::memory::C220LocalMemoryConfig;

    #[test]
    fn load2d_preserves_linear_addresses_and_unknown_bytes() {
        for destination in 0..=1 {
            for transpose in [false, true] {
                let mut memory = C220LocalMemory::new(C220LocalMemoryConfig {
                    l1_bytes: 768,
                    l0a_bytes: 768,
                    l0b_bytes: 768,
                    ..Default::default()
                })
                .unwrap();
                memory.l1_mut().write_known(0, &[99; 512]).unwrap();
                let mut source = vec![MemoryByteState::Known(7); 512];
                source[300] = MemoryByteState::Unknown;
                memory.l1_mut().write_states_linear(512, &source).unwrap();
                let mut registers = [0; 32];
                registers[0] = 512;
                registers[2] = 512;
                registers[3] = (1 << 16) | (1 << 24);
                let word = 0x6000_2180 | destination | (u32::from(transpose) << 2);
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
                let expected = if transpose {
                    transpose_block(source, C220Load2dElementFormat::B8)
                } else {
                    source
                };
                assert_eq!(target.read_states_linear(512, 512).unwrap(), expected);
                assert_eq!(
                    target.read_states(0, 512).unwrap(),
                    vec![MemoryByteState::Unknown; 512]
                );
                assert_eq!(target.tracked_bytes(), 512);
            }
        }
    }
}
