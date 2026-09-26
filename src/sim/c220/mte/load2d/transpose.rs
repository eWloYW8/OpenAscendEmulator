use super::{C220Load2dTransferError, C220Load2dTransferResult, C220PreparedLoad2d};
use crate::isa::c220::mte::load2d::{C220Load2dDestination, C220Load2dElementFormat};
use crate::isa::c220::mte::load2d_transpose::C220Load2dTransposeTransfer;
use crate::memory::sparse::MemoryByteState;
use crate::sim::c220::memory::C220LocalMemory;

pub fn prepare_c220_load2d_transpose(
    memory: &C220LocalMemory,
    transfer: C220Load2dTransposeTransfer,
) -> Result<C220PreparedLoad2d, C220Load2dTransferError> {
    let destination = transfer.instruction.destination;
    if !matches!(
        destination,
        C220Load2dDestination::L0a | C220Load2dDestination::L0b
    ) {
        return Err(C220Load2dTransferError::UnsupportedDestination(destination));
    }
    let mut segments = transfer.segments();
    let count = segments.len();
    let mut writes = Vec::new();
    writes
        .try_reserve_exact(count)
        .map_err(|_| C220Load2dTransferError::AllocationFailed { requested: count })?;
    let mut known_bytes = 0;
    let group = usize::from(transfer.fractals_per_repeat());
    while let Some(first) = segments.next() {
        let input = memory
            .l1()
            .read_initialized_states_linear(first.source_address, group * 512)?;
        let output = transpose_group(&input, transfer.instruction.element_format);
        let mut address = first.destination_address;
        for (index, block) in output.chunks_exact(512).enumerate() {
            if index != 0 {
                address = segments
                    .next()
                    .expect("complete fractal group")
                    .destination_address;
            }
            known_bytes += block
                .iter()
                .filter(|state| matches!(state, MemoryByteState::Known(_)))
                .count();
            writes.push((address, block.to_vec()));
        }
    }
    let bytes = count * 512;
    Ok(C220PreparedLoad2d {
        destination,
        writes,
        result: C220Load2dTransferResult {
            segment_count: count,
            bytes,
            known_bytes,
            unknown_bytes: bytes - known_bytes,
        },
    })
}

fn transpose_group(
    input: &[MemoryByteState],
    format: C220Load2dElementFormat,
) -> Vec<MemoryByteState> {
    let mut output = vec![MemoryByteState::Unknown; input.len()];
    match format {
        C220Load2dElementFormat::B4 => {
            for (index, byte) in output.iter_mut().enumerate() {
                let source_nibble = |destination: usize| {
                    let source = (destination % 64) * 64 + destination / 64;
                    match input[source / 2] {
                        MemoryByteState::Known(value) => Some((value >> ((source % 2) * 4)) & 15),
                        MemoryByteState::Unknown => None,
                    }
                };
                *byte = match (source_nibble(index * 2), source_nibble(index * 2 + 1)) {
                    (Some(low), Some(high)) => MemoryByteState::Known(low | (high << 4)),
                    _ => MemoryByteState::Unknown,
                };
            }
        }
        C220Load2dElementFormat::B8 => transpose_square(input, &mut output, 32, 1),
        C220Load2dElementFormat::B16 => transpose_square(input, &mut output, 16, 2),
        C220Load2dElementFormat::B32 => {
            for (source, destination) in [0, 2, 1, 3].into_iter().enumerate() {
                transpose_square(
                    &input[source * 256..(source + 1) * 256],
                    &mut output[destination * 256..(destination + 1) * 256],
                    8,
                    4,
                );
            }
        }
    }
    output
}

fn transpose_square(
    input: &[MemoryByteState],
    output: &mut [MemoryByteState],
    side: usize,
    element_bytes: usize,
) {
    for (source, element) in input.chunks_exact(element_bytes).enumerate() {
        let destination = (source % side * side + source / side) * element_bytes;
        output[destination..destination + element_bytes].copy_from_slice(element);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::c220::mte::load2d_transpose::C220Load2dTransposeInstruction;

    #[test]
    fn grouped_transpose_scatters_all_formats_without_l0_aliasing() {
        for (format_bits, side, element_bytes) in [(0, 32, 1), (1, 16, 2), (2, 64, 0), (3, 8, 4)] {
            for destination in 0..=1 {
                let word = (3 << 29)
                    | (1 << 27)
                    | (5 << 24)
                    | (format_bits << 22)
                    | (2 << 12)
                    | (3 << 7)
                    | (4 << 2)
                    | destination;
                let mut registers = [0; 32];
                registers[0] = 70000;
                registers[2] = 500000;
                registers[3] = 1 << 16;
                registers[4] = 2;
                let transfer = C220Load2dTransposeInstruction::decode(word)
                    .unwrap()
                    .capture(&registers);
                let len = usize::from(transfer.fractals_per_repeat()) * 512;
                let mut input = (0..len)
                    .map(|index| MemoryByteState::Known((index * 37 + index / 251) as u8))
                    .collect::<Vec<_>>();
                input[31] = MemoryByteState::Unknown;
                let mut memory = C220LocalMemory::new(Default::default()).unwrap();
                memory
                    .l1_mut()
                    .write_states_linear(transfer.source_base, &input)
                    .unwrap();
                let prepared = prepare_c220_load2d_transpose(&memory, transfer).unwrap();
                assert_eq!(prepared.result.bytes, len);
                assert_eq!(
                    prepared.result.unknown_bytes,
                    if element_bytes == 0 { 2 } else { 1 }
                );
                prepared.commit(&mut memory).unwrap();
                let target = if destination == 0 {
                    memory.l0a()
                } else {
                    memory.l0b()
                };
                let mut output = Vec::new();
                for segment in transfer.segments() {
                    output.extend(
                        target
                            .read_states_linear(segment.destination_address, 512)
                            .unwrap(),
                    );
                }
                for (destination_byte, actual) in output.iter().enumerate() {
                    let expected = if element_bytes == 0 {
                        let read = |n: usize| {
                            let source = (n % side) * side + n / side;
                            match input[source / 2] {
                                MemoryByteState::Known(value) => {
                                    Some((value >> (source % 2 * 4)) & 15)
                                }
                                MemoryByteState::Unknown => None,
                            }
                        };
                        match (read(destination_byte * 2), read(destination_byte * 2 + 1)) {
                            (Some(lo), Some(hi)) => MemoryByteState::Known(lo | hi << 4),
                            _ => MemoryByteState::Unknown,
                        }
                    } else {
                        let tile = if element_bytes == 4 {
                            destination_byte / 256
                        } else {
                            0
                        };
                        let byte = if element_bytes == 4 {
                            destination_byte % 256
                        } else {
                            destination_byte
                        };
                        let element = byte / element_bytes;
                        let source_tile = (tile % 2) * 2 + tile / 2;
                        input[source_tile * 256
                            + (element % side * side + element / side) * element_bytes
                            + byte % element_bytes]
                    };
                    assert_eq!(*actual, expected);
                }
                assert_eq!(target.tracked_bytes(), len);
                assert!(
                    target
                        .read_states(0, 512)
                        .unwrap()
                        .iter()
                        .all(|s| *s == MemoryByteState::Unknown)
                );
            }
        }
    }
}
