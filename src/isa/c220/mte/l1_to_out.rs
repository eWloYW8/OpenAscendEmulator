use super::burst::BurstLayout;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MovL1ToOutInstruction {
    pub word: u32,
    pub destination_register: u8,
    pub source_register: u8,
    pub descriptor_register: u8,
}

impl C220MovL1ToOutInstruction {
    pub const fn decode(word: u32) -> Option<Self> {
        if word >> 29 != 3
            || (word >> 27) & 3 != 2
            || (word >> 23) & 15 != 4
            || (word >> 3) & 15 != 2
        {
            return None;
        }
        Some(Self {
            word,
            destination_register: ((word >> 17) & 31) as u8,
            source_register: ((word >> 12) & 31) as u8,
            descriptor_register: ((word >> 7) & 31) as u8,
        })
    }

    pub fn capture(self, registers: &[u64; 32]) -> C220MovL1ToOutTransfer {
        C220MovL1ToOutTransfer {
            instruction: self,
            source_address: registers[usize::from(self.source_register)],
            destination_address: registers[usize::from(self.destination_register)],
            xm: registers[usize::from(self.descriptor_register)],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MovL1ToOutTransfer {
    pub instruction: C220MovL1ToOutInstruction,
    pub source_address: u64,
    pub destination_address: u64,
    pub xm: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MovL1ToOutSegment {
    pub burst_index: u16,
    pub unit_index: u16,
    pub source_address: u64,
    pub destination_address: u64,
    pub bytes: u32,
}

impl C220MovL1ToOutTransfer {
    pub const fn is_disabled(self) -> bool {
        (self.xm >> 4) & 0xfff == 0 || (self.xm >> 16) & 0xffff == 0
    }

    pub const fn sid(self) -> u8 {
        (self.xm & 15) as u8
    }

    pub fn segments(self) -> Option<impl ExactSizeIterator<Item = C220MovL1ToOutSegment> + Clone> {
        Some(
            BurstLayout::decode(self.xm)
                .segments(self.source_address, self.destination_address)?
                .map(|segment| C220MovL1ToOutSegment {
                    burst_index: segment.burst_index,
                    unit_index: segment.unit_index,
                    source_address: segment.source,
                    destination_address: segment.destination,
                    bytes: segment.bytes,
                }),
        )
    }
}
