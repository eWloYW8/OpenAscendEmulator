use crate::isa::c220::mte::out_to_l1::{C220L1DmaDescriptor, C220L1DmaLayout};
use crate::memory::mapped::{MappedMemory, MappedMemoryError};
use crate::memory::sparse::MemoryByteState;
use crate::sim::c220::memory::{C220LocalBuffer, C220LocalBufferError};

#[derive(Debug, thiserror::Error)]
pub enum C220L1DmaError {
    #[error(transparent)]
    Source(#[from] MappedMemoryError),
    #[error(transparent)]
    Destination(#[from] C220LocalBufferError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct C220L1DmaResult {
    pub segments: usize,
    pub source_bytes: usize,
    pub destination_bytes: usize,
    pub discarded_destination_bytes: usize,
    pub unknown_destination_bytes: usize,
}

/// Executes functional memory effects. Timing admission and retirement are
/// controlled by the caller. Padding is the captured 16-bit fill value.
pub fn execute_c220_mov_out_to_l1(
    source: &MappedMemory,
    destination: &mut C220LocalBuffer,
    descriptor: C220L1DmaDescriptor,
    source_address: u64,
    destination_address: u64,
    padding: u16,
) -> Result<C220L1DmaResult, C220L1DmaError> {
    let mut result = C220L1DmaResult::default();
    let fill = padding.to_le_bytes();
    for segment in descriptor.segments(source_address, destination_address) {
        let input = source.read_states_at(segment.source_address, segment.source_bytes as usize)?;
        let mut output = [MemoryByteState::Known(0); 32];
        for (index, byte) in output.iter_mut().enumerate() {
            *byte = MemoryByteState::Known(
                fill[if descriptor.layout == C220L1DmaLayout::Pad1 {
                    0
                } else {
                    index % 2
                }],
            );
        }
        let copied = input.len().min(segment.destination_bytes as usize);
        output[..copied].copy_from_slice(&input[..copied]);
        let output = &output[..segment.destination_bytes as usize];
        result.segments += 1;
        result.source_bytes += input.len();
        if segment
            .destination_address
            .checked_add(u64::from(segment.destination_bytes - 1))
            .is_none()
        {
            result.discarded_destination_bytes += output.len();
            continue;
        }
        destination.write_states_linear(segment.destination_address, output)?;
        result.destination_bytes += output.len();
        result.unknown_destination_bytes += output
            .iter()
            .filter(|byte| **byte == MemoryByteState::Unknown)
            .count();
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::c220::mte::out_to_l1::C220MovOutToL1Instruction;
    use crate::memory::{region::MemoryRegion, sparse::SparseMemory};

    #[test]
    fn all_layouts_preserve_strides_padding_and_unknown_bytes() {
        let data: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let mut source = MappedMemory::bind(
            SparseMemory::new(vec![MemoryRegion::new(256, data).unwrap()], 256, 256),
            &[0x1000],
        )
        .unwrap();
        source.write_unknown_at(0x1000, 1).unwrap();
        for (mode, (read, written)) in [
            (32, 32),
            (1, 32),
            (2, 32),
            (4, 32),
            (8, 32),
            (16, 32),
            (32, 4),
            (32, 8),
            (32, 16),
        ]
        .into_iter()
        .enumerate()
        {
            let word = 0x7100_0020
                | (mode as u32 & 7)
                | ((mode as u32 >> 3) << 22)
                | (3 << 17)
                | (4 << 12)
                | (5 << 7);
            let instruction = C220MovOutToL1Instruction::decode(word).unwrap();
            assert_eq!(
                (
                    instruction.destination_register,
                    instruction.source_register,
                    instruction.descriptor_register
                ),
                (3, 4, 5)
            );
            let descriptor = C220L1DmaDescriptor {
                xm: (2 << 48) | (1 << 32) | (2 << 16) | (2 << 4),
                layout: instruction.layout,
            };
            let mut destination = C220LocalBuffer::new(16);
            let result = execute_c220_mov_out_to_l1(
                &source,
                &mut destination,
                descriptor,
                0x1000,
                12,
                0xbbaa,
            )
            .unwrap();
            assert_eq!(result.source_bytes, read * 4);
            assert_eq!(result.destination_bytes, written * 4);
            assert_eq!(result.unknown_destination_bytes, 1);
            for burst in 0..2 {
                for unit in 0..2 {
                    let source_offset = (burst * 3 + unit) * read;
                    let destination_offset = 12 + (burst * 4 + unit) * written;
                    let expected: Vec<_> = (0..written)
                        .map(|i| {
                            if i < read {
                                if source_offset + i == 0 {
                                    MemoryByteState::Unknown
                                } else {
                                    MemoryByteState::Known((source_offset + i) as u8)
                                }
                            } else {
                                MemoryByteState::Known(if mode == 1 || i % 2 == 0 {
                                    0xaa
                                } else {
                                    0xbb
                                })
                            }
                        })
                        .collect();
                    assert_eq!(
                        destination
                            .read_states_linear(destination_offset as u64, written)
                            .unwrap(),
                        expected
                    );
                }
            }
            assert_eq!(
                destination.read_states(0, 12).unwrap(),
                vec![MemoryByteState::Unknown; 12]
            );
            assert!(destination.read_states(12, 32).is_err());
            assert!(
                C220L1DmaDescriptor {
                    xm: 0,
                    ..descriptor
                }
                .segments(0, 0)
                .next()
                .is_none()
            );
        }
        assert!(C220MovOutToL1Instruction::decode(0x7140_0021).is_none());
        assert!(C220MovOutToL1Instruction::decode(0x7100_0008).is_none());
        let mut destination = C220LocalBuffer::new(16);
        let result = execute_c220_mov_out_to_l1(
            &source,
            &mut destination,
            C220L1DmaDescriptor {
                xm: (1 << 16) | (1 << 4),
                layout: C220L1DmaLayout::Copy32,
            },
            0x1000,
            u64::MAX - 15,
            0,
        )
        .unwrap();
        assert_eq!(result.discarded_destination_bytes, 32);
        assert_eq!(result.destination_bytes, 0);
        assert_eq!(destination.tracked_bytes(), 0);
    }
}
