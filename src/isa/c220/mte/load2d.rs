use thiserror::Error;

pub const C220_LOAD_2D_BLOCK_BYTES: u32 = 512;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220Load2dSource {
    L1,
    Out,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220Load2dDestination {
    L0a,
    L0b,
    L1,
    Reserved,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220Load2dAddressMode {
    Increment,
    Decrement,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220Load2dElementFormat {
    B4,
    B8,
    B16,
    B32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Load2dInstruction {
    pub word: u32,
    pub source: C220Load2dSource,
    pub destination: C220Load2dDestination,
    pub transpose: bool,
    pub element_format: C220Load2dElementFormat,
    pub address_mode: C220Load2dAddressMode,
    pub reserved_bit_6: bool,
    pub destination_register: u8,
    pub source_register: u8,
    pub descriptor_register: u8,
}

impl C220Load2dInstruction {
    pub const fn decode(word: u32) -> Option<Self> {
        if word >> 29 != 3 || (word >> 27) & 3 != 0 || (word >> 22) & 0x1f != 0 {
            return None;
        }
        let transpose = word & (1 << 2) != 0;
        let destination_bits = if transpose { word & 1 } else { word & 3 };
        let destination = match destination_bits {
            0 => C220Load2dDestination::L0a,
            1 => C220Load2dDestination::L0b,
            2 => C220Load2dDestination::L1,
            _ => C220Load2dDestination::Reserved,
        };
        let element_format = if transpose {
            match ((word >> 3) & 1) | (((word >> 1) & 1) << 1) {
                0 => C220Load2dElementFormat::B8,
                1 => C220Load2dElementFormat::B16,
                2 => C220Load2dElementFormat::B4,
                _ => C220Load2dElementFormat::B32,
            }
        } else if word & (1 << 3) == 0 {
            C220Load2dElementFormat::B8
        } else {
            C220Load2dElementFormat::B16
        };
        Some(Self {
            word,
            source: if word & (1 << 4) == 0 {
                C220Load2dSource::L1
            } else {
                C220Load2dSource::Out
            },
            destination,
            transpose,
            element_format,
            address_mode: if word & (1 << 5) == 0 {
                C220Load2dAddressMode::Increment
            } else {
                C220Load2dAddressMode::Decrement
            },
            reserved_bit_6: word & (1 << 6) != 0,
            destination_register: ((word >> 17) & 0x1f) as u8,
            source_register: ((word >> 12) & 0x1f) as u8,
            descriptor_register: ((word >> 7) & 0x1f) as u8,
        })
    }

    pub const fn is_mte1(self) -> bool {
        matches!(self.source, C220Load2dSource::L1)
            && matches!(
                self.destination,
                C220Load2dDestination::L0a | C220Load2dDestination::L0b
            )
    }

    pub fn capture(self, xregs: &[u64; 32]) -> Result<C220Load2dTransfer, C220Load2dError> {
        if !self.is_mte1() && !self.is_external() {
            return Err(C220Load2dError::UnsupportedRoute {
                source_buffer: self.source,
                destination_buffer: self.destination,
            });
        }
        if self.reserved_bit_6 {
            return Err(C220Load2dError::UnsupportedInstructionBit6);
        }
        let descriptor = C220Load2dDescriptor::decode(
            xregs[usize::from(self.descriptor_register)],
            self.address_mode,
        )?;
        Ok(C220Load2dTransfer {
            instruction: self,
            descriptor,
            source_base: xregs[usize::from(self.source_register)],
            destination_base: xregs[usize::from(self.destination_register)],
        })
    }

    pub const fn is_external(self) -> bool {
        matches!(self.source, C220Load2dSource::Out)
            && !matches!(self.destination, C220Load2dDestination::Reserved)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Load2dDescriptor {
    pub raw: u64,
    pub start_index: u16,
    pub repeat_count: u8,
    pub source_stride_blocks: u16,
    pub sid: u8,
    pub destination_gap_blocks: u16,
    pub address_mode: C220Load2dAddressMode,
}

impl C220Load2dDescriptor {
    pub const fn decode(
        raw: u64,
        address_mode: C220Load2dAddressMode,
    ) -> Result<Self, C220Load2dError> {
        let repeat_count = ((raw >> 16) & 0xff) as u8;
        if repeat_count != 0 && raw >> 60 != 0 {
            return Err(C220Load2dError::UnsupportedDescriptorHighBits {
                bits: (raw >> 60) as u8,
            });
        }
        Ok(Self {
            raw,
            start_index: raw as u16,
            repeat_count,
            source_stride_blocks: ((raw >> 24) & 0xffff) as u16,
            sid: ((raw >> 40) & 0xf) as u8,
            destination_gap_blocks: ((raw >> 44) & 0xffff) as u16,
            address_mode,
        })
    }

    pub const fn destination_stride_blocks(self) -> u32 {
        self.destination_gap_blocks as u32 + 1
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Load2dTransfer {
    pub instruction: C220Load2dInstruction,
    pub descriptor: C220Load2dDescriptor,
    pub source_base: u64,
    pub destination_base: u64,
}

impl C220Load2dTransfer {
    pub fn segments(self) -> C220Load2dSegments {
        C220Load2dSegments {
            transfer: self,
            repeats: 0..self.descriptor.repeat_count,
        }
    }

    pub const fn byte_count(self) -> u64 {
        self.descriptor.repeat_count as u64 * C220_LOAD_2D_BLOCK_BYTES as u64
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220Load2dSegments {
    transfer: C220Load2dTransfer,
    repeats: std::ops::Range<u8>,
}

impl Iterator for C220Load2dSegments {
    type Item = C220Load2dSegment;

    fn next(&mut self) -> Option<Self::Item> {
        let repeat_index = self.repeats.next()?;
        let transfer = self.transfer;
        let start_bytes =
            u32::from(transfer.descriptor.start_index).wrapping_mul(C220_LOAD_2D_BLOCK_BYTES);
        let stride_bytes = u32::from(transfer.descriptor.source_stride_blocks)
            .wrapping_mul(C220_LOAD_2D_BLOCK_BYTES);
        let repeat_bytes = u32::from(repeat_index).wrapping_mul(stride_bytes);
        let source_offset = match transfer.descriptor.address_mode {
            C220Load2dAddressMode::Increment => start_bytes.wrapping_add(repeat_bytes),
            C220Load2dAddressMode::Decrement => start_bytes.wrapping_sub(repeat_bytes),
        };
        let destination_offset = u64::from(repeat_index)
            .wrapping_mul(u64::from(transfer.descriptor.destination_stride_blocks()))
            .wrapping_mul(u64::from(C220_LOAD_2D_BLOCK_BYTES));
        Some(C220Load2dSegment {
            repeat_index,
            source_address: transfer.source_base.wrapping_add(u64::from(source_offset)),
            destination_address: transfer.destination_base.wrapping_add(destination_offset),
            bytes: C220_LOAD_2D_BLOCK_BYTES,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.repeats.size_hint()
    }
}

impl ExactSizeIterator for C220Load2dSegments {}
impl std::iter::FusedIterator for C220Load2dSegments {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Load2dSegment {
    pub repeat_index: u8,
    pub source_address: u64,
    pub destination_address: u64,
    pub bytes: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum C220Load2dError {
    #[error("unsupported LOAD_2D route {source_buffer:?} -> {destination_buffer:?}")]
    UnsupportedRoute {
        source_buffer: C220Load2dSource,
        destination_buffer: C220Load2dDestination,
    },
    #[error("LOAD_2D instruction bit 6 is unsupported")]
    UnsupportedInstructionBit6,
    #[error("LOAD_2D descriptor high bits 60:63 are unsupported: {bits:#x}")]
    UnsupportedDescriptorHighBits { bits: u8 },
}
