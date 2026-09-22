use std::iter::FusedIterator;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220Set2dDestination {
    L0a,
    L0b,
    L1,
}

impl C220Set2dDestination {
    pub const fn block_bytes(self) -> u32 {
        match self {
            Self::L0a | Self::L0b => 512,
            Self::L1 => 32,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220Set2dElementFormat {
    B16,
    B32,
}

impl C220Set2dElementFormat {
    pub const fn bytes(self) -> usize {
        match self {
            Self::B16 => 2,
            Self::B32 => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Set2dInstruction {
    pub word: u32,
    pub destination: C220Set2dDestination,
    pub element_format: C220Set2dElementFormat,
    pub destination_register: u8,
    pub descriptor_register: u8,
}

impl C220Set2dInstruction {
    pub const fn decode(word: u32) -> Option<Self> {
        if word >> 29 != 3 || (word >> 27) & 3 != 0 || (word >> 22) & 31 != 1 {
            return None;
        }
        let destination = match word & 3 {
            0 => C220Set2dDestination::L0a,
            1 => C220Set2dDestination::L0b,
            2 => C220Set2dDestination::L1,
            _ => return None,
        };
        Some(Self {
            word,
            destination,
            element_format: if word & 4 == 0 {
                C220Set2dElementFormat::B16
            } else {
                C220Set2dElementFormat::B32
            },
            destination_register: ((word >> 17) & 31) as u8,
            descriptor_register: ((word >> 7) & 31) as u8,
        })
    }

    pub const fn capture(self, xregs: &[u64; 32], spr15: u64) -> C220Set2dFill {
        C220Set2dFill {
            instruction: self,
            descriptor: C220Set2dDescriptor::decode(xregs[self.descriptor_register as usize]),
            destination_base: xregs[self.destination_register as usize],
            pattern_register: spr15,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Set2dDescriptor {
    pub raw: u64,
    pub repeat_count: u16,
    pub burst_blocks: u16,
    pub destination_gap_blocks: u16,
}

impl C220Set2dDescriptor {
    pub const fn decode(raw: u64) -> Self {
        Self {
            raw,
            repeat_count: (raw & 0x7fff) as u16,
            burst_blocks: ((raw >> 16) & 0x7fff) as u16,
            destination_gap_blocks: ((raw >> 32) & 0x7fff) as u16,
        }
    }

    pub const fn is_disabled(self) -> bool {
        self.repeat_count == 0
    }

    pub const fn is_empty(self) -> bool {
        self.repeat_count == 0 || self.burst_blocks == 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Set2dFill {
    pub instruction: C220Set2dInstruction,
    pub descriptor: C220Set2dDescriptor,
    pub destination_base: u64,
    pub pattern_register: u64,
}

impl C220Set2dFill {
    pub const fn burst_bytes(self) -> u32 {
        self.descriptor.burst_blocks as u32 * self.instruction.destination.block_bytes()
    }

    pub const fn destination_stride(self) -> u64 {
        (self.descriptor.burst_blocks as u64 + self.descriptor.destination_gap_blocks as u64)
            * self.instruction.destination.block_bytes() as u64
    }

    pub const fn byte_count(self) -> u64 {
        self.burst_bytes() as u64 * self.descriptor.repeat_count as u64
    }

    pub fn segments(self) -> C220Set2dSegments {
        C220Set2dSegments {
            fill: self,
            repeats: 0..self.descriptor.repeat_count,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Set2dSegment {
    pub repeat_index: u16,
    pub destination_address: u64,
    pub bytes: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220Set2dSegments {
    fill: C220Set2dFill,
    repeats: std::ops::Range<u16>,
}

impl Iterator for C220Set2dSegments {
    type Item = C220Set2dSegment;

    fn next(&mut self) -> Option<Self::Item> {
        let repeat_index = self.repeats.next()?;
        Some(C220Set2dSegment {
            repeat_index,
            destination_address: self
                .fill
                .destination_base
                .wrapping_add(u64::from(repeat_index).wrapping_mul(self.fill.destination_stride())),
            bytes: self.fill.burst_bytes(),
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.repeats.size_hint()
    }
}

impl ExactSizeIterator for C220Set2dSegments {}
impl FusedIterator for C220Set2dSegments {}
