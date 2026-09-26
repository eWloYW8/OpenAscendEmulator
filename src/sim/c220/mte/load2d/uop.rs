use crate::isa::c220::mte::load2d_transpose::C220Load2dTransposeTransfer;
use std::iter::FusedIterator;
use std::num::NonZeroU32;

use crate::sim::c220::mte::interface::{C220L0WritePort, C220MteOutputPlan};

use crate::isa::c220::mte::load2d::{
    C220_LOAD_2D_BLOCK_BYTES, C220Load2dDestination, C220Load2dError, C220Load2dSegment,
    C220Load2dSegments, C220Load2dTransfer,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Load2dReadUop {
    pub logical: C220Load2dSegment,
    pub fractal_index: Option<u8>,
    pub destination: C220Load2dDestination,
    pub source_address: u64,
    pub input_bytes: u32,
    pub input_offset: u32,
    pub completes_logical_uop: bool,
    pub last_in_instruction: bool,
}

impl C220Load2dReadUop {
    pub const L0_OUTPUT_PORT: C220L0WritePort = C220L0WritePort::Port0;

    /// Only the completing L1 response produces output for its logical block.
    pub fn outputs(
        self,
        instruction_id: u64,
        request_id: u64,
        bytes_per_fragment: NonZeroU32,
    ) -> C220MteOutputPlan {
        C220MteOutputPlan::new(
            instruction_id,
            request_id,
            self.logical.destination_address,
            if self.completes_logical_uop {
                self.logical.bytes
            } else {
                0
            },
            self.last_in_instruction,
            bytes_per_fragment,
        )
    }
}

/// Physical L1 reads keep the full logical output block and destination.
/// Splits use byte counts, not alignment boundaries; only the final read of
/// each block enables its downstream output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220Load2dRequestPlan {
    segments: Segments,
    destination: C220Load2dDestination,
    current: Option<(C220Load2dSegment, Option<u8>)>,
    input_offset: u32,
    access_width: NonZeroU32,
    remaining: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Segments {
    Ordinary(C220Load2dSegments),
    Transpose {
        transfer: C220Load2dTransposeTransfer,
        index: usize,
    },
}

impl Segments {
    fn next(&mut self) -> Option<(C220Load2dSegment, Option<u8>)> {
        match self {
            Self::Ordinary(segments) => segments.next().map(|segment| (segment, None)),
            Self::Transpose { transfer, index } => {
                let segment = transfer.segments().nth(*index)?;
                *index += 1;
                Some((
                    C220Load2dSegment {
                        repeat_index: segment.repeat_index,
                        source_address: segment.source_address,
                        destination_address: segment.destination_address,
                        bytes: segment.bytes,
                    },
                    Some(segment.fractal_index),
                ))
            }
        }
    }
}

impl C220Load2dRequestPlan {
    pub fn new(
        transfer: C220Load2dTransfer,
        l1_dmac_access_width: NonZeroU32,
    ) -> Result<Self, C220Load2dError> {
        if !transfer.instruction.is_mte1() {
            return Err(C220Load2dError::UnsupportedRoute {
                source_buffer: transfer.instruction.source,
                destination_buffer: transfer.instruction.destination,
            });
        }
        Ok(Self {
            segments: Segments::Ordinary(transfer.segments()),
            destination: transfer.instruction.destination,
            current: None,
            input_offset: 0,
            access_width: l1_dmac_access_width,
            remaining: usize::from(transfer.descriptor.repeat_count)
                * C220_LOAD_2D_BLOCK_BYTES.div_ceil(l1_dmac_access_width.get()) as usize,
        })
    }

    pub fn new_transpose(
        transfer: C220Load2dTransposeTransfer,
        access_width: NonZeroU32,
    ) -> Result<Self, C220Load2dError> {
        let destination = transfer.instruction.destination;
        if !matches!(
            destination,
            C220Load2dDestination::L0a | C220Load2dDestination::L0b
        ) {
            return Err(C220Load2dError::UnsupportedRoute {
                source_buffer: crate::isa::c220::mte::load2d::C220Load2dSource::L1,
                destination_buffer: destination,
            });
        }
        Ok(Self {
            segments: Segments::Transpose { transfer, index: 0 },
            destination,
            current: None,
            input_offset: 0,
            access_width,
            remaining: transfer.segments().len()
                * C220_LOAD_2D_BLOCK_BYTES.div_ceil(access_width.get()) as usize,
        })
    }
}

impl Iterator for C220Load2dRequestPlan {
    type Item = C220Load2dReadUop;

    fn next(&mut self) -> Option<Self::Item> {
        let (logical, fractal_index) = match self.current {
            Some(logical) => logical,
            None => self.segments.next()?,
        };
        let input_bytes = (logical.bytes - self.input_offset).min(self.access_width.get());
        let completes_logical_uop = self.input_offset + input_bytes == logical.bytes;
        let uop = C220Load2dReadUop {
            logical,
            fractal_index,
            destination: self.destination,
            source_address: logical
                .source_address
                .wrapping_add(u64::from(self.input_offset)),
            input_bytes,
            input_offset: self.input_offset,
            completes_logical_uop,
            last_in_instruction: self.remaining == 1,
        };
        self.remaining -= 1;
        self.current = (!completes_logical_uop).then_some((logical, fractal_index));
        self.input_offset = if completes_logical_uop {
            0
        } else {
            self.input_offset + input_bytes
        };
        Some(uop)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for C220Load2dRequestPlan {}
impl FusedIterator for C220Load2dRequestPlan {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::c220::mte::load2d::C220Load2dInstruction;

    #[test]
    fn physical_splits_preserve_logical_blocks_and_completion_markers() {
        for word in [0x6000_2180, 0x6000_2181, 0x6000_218d, 0x6000_21a0] {
            let instruction = C220Load2dInstruction::decode(word).unwrap();
            let mut registers = [0; 32];
            registers[0] = 0x800;
            registers[2] = u64::MAX - 30;
            registers[3] = (2 << 16) | (3 << 24) | (1 << 44);
            let transfer = instruction.capture(&registers).unwrap();
            let logical = transfer.segments().collect::<Vec<_>>();
            for width in [1, 32, 96, 512, 1024] {
                let mut plan =
                    C220Load2dRequestPlan::new(transfer, NonZeroU32::new(width).unwrap()).unwrap();
                let count = plan.len();
                let requests = plan.by_ref().collect::<Vec<_>>();
                let mut alternate_registers = registers;
                alternate_registers[3] |= 1 << 60;
                let alternate = instruction.capture(&alternate_registers).unwrap();
                assert_eq!(
                    C220Load2dRequestPlan::new(alternate, NonZeroU32::new(width).unwrap())
                        .unwrap()
                        .collect::<Vec<_>>(),
                    requests
                );
                assert_eq!(requests.len(), count);
                assert_eq!(count, 2 * 512_u32.div_ceil(width) as usize);
                assert_eq!(requests.iter().map(|r| r.input_bytes).sum::<u32>(), 1024);
                assert_eq!(
                    requests.iter().filter(|r| r.completes_logical_uop).count(),
                    2
                );
                assert_eq!(requests.iter().filter(|r| r.last_in_instruction).count(), 1);
                for (index, block) in logical.iter().enumerate() {
                    let mut offset = 0;
                    for request in requests
                        .iter()
                        .filter(|r| r.logical.repeat_index as usize == index)
                    {
                        assert_eq!(request.logical, *block);
                        assert_eq!(
                            request.source_address,
                            block.source_address.wrapping_add(u64::from(offset))
                        );
                        assert_eq!(request.input_offset, offset);
                        offset += request.input_bytes;
                        assert_eq!(request.completes_logical_uop, offset == 512);
                    }
                    assert_eq!(offset, 512);
                }
                assert!(requests.last().unwrap().last_in_instruction);
                assert_eq!(plan.len(), 0);
                assert_eq!(plan.next(), None);
            }
        }
    }
}
