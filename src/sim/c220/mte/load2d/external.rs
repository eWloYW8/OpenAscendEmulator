use super::{C220Load2dTransferError, C220PreparedLoad2d, prepare_blocks};
use crate::isa::c220::mte::load2d::{
    C220Load2dDestination, C220Load2dError, C220Load2dSegment, C220Load2dSegments,
    C220Load2dTransfer,
};
use crate::memory::mapped::MappedMemory;
use crate::sim::c220::mte::uop::C220DmaUopMode;

pub fn prepare_c220_external_load2d(
    memory: &MappedMemory,
    transfer: C220Load2dTransfer,
) -> Result<C220PreparedLoad2d, C220Load2dTransferError> {
    check_external(transfer)?;
    prepare_blocks(transfer, |address, bytes| {
        Ok(memory.read_states_at(address, bytes)?)
    })
}

fn check_external(transfer: C220Load2dTransfer) -> Result<(), C220Load2dError> {
    if !transfer.instruction.is_external() {
        return Err(C220Load2dError::UnsupportedRoute {
            source_buffer: transfer.instruction.source,
            destination_buffer: transfer.instruction.destination,
        });
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Load2dExternalRequest {
    pub repeat_index: u8,
    pub source_address: u64,
    pub destination_address: u64,
    pub bytes: u32,
    pub destination: C220Load2dDestination,
    pub aligned_source: bool,
    pub last_in_block: bool,
    pub last_in_instruction: bool,
}

/// External reads retain 512-byte logical blocks. An unaligned source base
/// sends whole blocks; an aligned base uses the selected DMA split mode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220Load2dExternalRequests {
    sid: u8,
    segments: C220Load2dSegments,
    current: Option<C220Load2dSegment>,
    offset: u32,
    aligned_source: bool,
    mode: C220DmaUopMode,
    destination: C220Load2dDestination,
    layout: crate::sim::c220::mte::uop::C220DmaDestinationLayout,
}

impl C220Load2dExternalRequests {
    pub fn new(
        transfer: C220Load2dTransfer,
        mode: C220DmaUopMode,
    ) -> Result<Self, C220Load2dError> {
        check_external(transfer)?;
        Ok(Self {
            sid: transfer.descriptor.sid,
            segments: transfer.segments(),
            current: None,
            offset: 0,
            aligned_source: transfer.source_base.is_multiple_of(512),
            mode,
            destination: transfer.instruction.destination,
            layout: crate::sim::c220::mte::uop::C220DmaDestinationLayout {
                base: transfer.destination_base,
                burst_bytes: 512,
                burst_stride: u64::from(transfer.descriptor.destination_stride_blocks()) * 512,
            },
        })
    }

    pub(crate) fn dma_metadata(
        &self,
    ) -> (
        crate::sim::c220::mte::uop::C220DmaDestinationLayout,
        C220DmaUopMode,
        bool,
    ) {
        (self.layout, self.mode, self.aligned_source)
    }

    pub const fn sid(&self) -> u8 {
        self.sid
    }
}

impl Iterator for C220Load2dExternalRequests {
    type Item = C220Load2dExternalRequest;

    fn next(&mut self) -> Option<Self::Item> {
        let segment = self.current.or_else(|| self.segments.next())?;
        let source_address = segment.source_address.wrapping_add(u64::from(self.offset));
        let bytes = if self.aligned_source {
            self.mode
                .split_bytes(source_address, segment.bytes - self.offset)
        } else {
            segment.bytes
        };
        let last_in_block = self.offset + bytes == segment.bytes;
        let request = C220Load2dExternalRequest {
            repeat_index: segment.repeat_index,
            source_address,
            destination_address: segment
                .destination_address
                .wrapping_add(u64::from(self.offset)),
            bytes,
            destination: self.destination,
            aligned_source: self.aligned_source,
            last_in_block,
            last_in_instruction: last_in_block && self.segments.len() == 0,
        };
        self.current = (!last_in_block).then_some(segment);
        self.offset = if last_in_block {
            0
        } else {
            self.offset + bytes
        };
        Some(request)
    }
}

impl std::iter::FusedIterator for C220Load2dExternalRequests {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::c220::mte::load2d::C220Load2dInstruction;
    use crate::memory::{region::MemoryRegion, sparse::SparseMemory};
    use crate::sim::c220::memory::C220LocalMemory;

    #[test]
    fn external_blocks_preserve_routes_strides_transpose_and_alignment() {
        let base = 1_u64 << 40;
        let input: Vec<u8> = (0..4096).map(|i| (i % 251) as u8).collect();
        let memory = MappedMemory::bind(
            SparseMemory::new(
                vec![MemoryRegion::new(4096, input.clone()).unwrap()],
                4096,
                4096,
            ),
            &[base],
        )
        .unwrap();
        for destination in 0..=2 {
            for transpose in [false, true] {
                if destination == 2 && transpose {
                    continue;
                }
                for unaligned in [0, 7] {
                    let instruction = C220Load2dInstruction::decode(
                        (3 << 29)
                            | (1 << 17)
                            | (2 << 12)
                            | (3 << 7)
                            | 16
                            | 32
                            | destination
                            | (u32::from(transpose) << 2),
                    )
                    .unwrap();
                    let mut registers = [0; 32];
                    registers[1] = 1024;
                    registers[2] = base + unaligned;
                    registers[3] = 2 | (2 << 16) | (1 << 24) | (1 << 44);
                    let transfer = instruction.capture(&registers).unwrap();
                    for (mode, parts) in [
                        (C220DmaUopMode::Wide512, 1),
                        (C220DmaUopMode::Wide256, 2),
                        (C220DmaUopMode::Fixed128, 4),
                        (C220DmaUopMode::Unbounded, 1),
                    ] {
                        let requests: Vec<_> = C220Load2dExternalRequests::new(transfer, mode)
                            .unwrap()
                            .collect();
                        assert_eq!(requests.len(), 2 * if unaligned == 0 { parts } else { 1 });
                        assert_eq!(requests.iter().map(|r| r.bytes).sum::<u32>(), 1024);
                        assert_eq!(requests.iter().filter(|r| r.last_in_block).count(), 2);
                        assert_eq!(requests.iter().filter(|r| r.last_in_instruction).count(), 1);
                        assert!(requests.last().unwrap().last_in_instruction);
                        for request in &requests {
                            let block = transfer
                                .segments()
                                .nth(request.repeat_index as usize)
                                .unwrap();
                            let offset = request.source_address - block.source_address;
                            assert_eq!(
                                request.destination_address,
                                block.destination_address + offset
                            );
                            assert_eq!(request.aligned_source, unaligned == 0);
                        }
                    }
                    let mut local = C220LocalMemory::new(Default::default()).unwrap();
                    let prepared = prepare_c220_external_load2d(&memory, transfer).unwrap();
                    assert_eq!(prepared.result.bytes, 1024);
                    prepared.commit(&mut local).unwrap();
                    let target = match destination {
                        0 => local.l0a(),
                        1 => local.l0b(),
                        _ => local.l1(),
                    };
                    for block in transfer.segments() {
                        let start = (block.source_address - base) as usize;
                        let actual = target
                            .read_initialized_linear(block.destination_address, 512)
                            .unwrap();
                        for (index, byte) in input[start..start + 512].iter().enumerate() {
                            let output = if transpose {
                                (index / 32 + ((index / 2) % 16) * 16) * 2 + index % 2
                            } else {
                                index
                            };
                            assert_eq!(actual[output], *byte);
                        }
                    }
                    registers[3] = 0;
                    registers[2] = u64::MAX;
                    let empty = instruction.capture(&registers).unwrap();
                    assert_eq!(
                        C220Load2dExternalRequests::new(empty, C220DmaUopMode::Fixed128)
                            .unwrap()
                            .next(),
                        None
                    );
                    assert_eq!(
                        prepare_c220_external_load2d(&memory, empty)
                            .unwrap()
                            .result
                            .bytes,
                        0
                    );
                }
            }
        }
    }
}
