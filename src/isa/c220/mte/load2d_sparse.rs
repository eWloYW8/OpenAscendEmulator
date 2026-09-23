pub const C220_SPARSE_WEIGHT_BYTES: u32 = 512;
pub const C220_SPARSE_INDEX_BYTES: u32 = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Load2dSparseInstruction {
    pub word: u32,
    pub destination_register: u8,
    pub source_register: u8,
    pub descriptor_register: u8,
}

impl C220Load2dSparseInstruction {
    pub const fn decode(word: u32) -> Option<Self> {
        if word >> 29 != 3 || (word >> 27) & 3 != 1 || (word >> 22) & 31 != 24 {
            return None;
        }
        Some(Self {
            word,
            destination_register: ((word >> 17) & 31) as u8,
            source_register: ((word >> 12) & 31) as u8,
            descriptor_register: ((word >> 7) & 31) as u8,
        })
    }

    pub fn capture(self, registers: &[u64; 32]) -> C220Load2dSparseTransfer {
        C220Load2dSparseTransfer {
            instruction: self,
            destination_base: registers[usize::from(self.destination_register)],
            packed_source: registers[usize::from(self.source_register)],
            descriptor: registers[usize::from(self.descriptor_register)],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Load2dSparseTransfer {
    pub instruction: C220Load2dSparseInstruction,
    pub destination_base: u64,
    pub packed_source: u64,
    pub descriptor: u64,
}

impl C220Load2dSparseTransfer {
    pub const fn repeat_count(self) -> u8 {
        (self.descriptor >> 16) as u8
    }

    pub const fn start_index(self) -> u16 {
        self.descriptor as u16
    }

    pub const fn weight_source_base(self) -> u64 {
        self.packed_source as u32 as u64
    }

    pub const fn index_source_base(self) -> u64 {
        self.packed_source >> 32
    }

    pub fn segments(self) -> impl ExactSizeIterator<Item = C220Load2dSparseSegment> + Clone {
        (0..self.repeat_count()).map(move |repeat_index| {
            let repeat = u64::from(repeat_index);
            let source_index = u64::from(self.start_index()) + repeat;
            C220Load2dSparseSegment {
                repeat_index,
                weight_source_address: self.weight_source_base() + source_index * 512,
                index_source_address: self.index_source_base() + source_index * 128,
                weight_destination_address: self.destination_base.wrapping_add(repeat * 512),
                index_destination_address: (self.destination_base >> 2) + repeat * 128,
            }
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Load2dSparseSegment {
    pub repeat_index: u8,
    pub weight_source_address: u64,
    pub index_source_address: u64,
    pub weight_destination_address: u64,
    pub index_destination_address: u64,
}
