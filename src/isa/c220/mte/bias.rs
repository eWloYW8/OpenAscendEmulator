pub const C220_BT_INPUT_BLOCK_BYTES: u32 = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MovL1ToBtInstruction {
    pub word: u32,
    pub destination_register: u8,
    pub source_register: u8,
    pub descriptor_register: u8,
}

impl C220MovL1ToBtInstruction {
    pub const fn decode(word: u32) -> Option<Self> {
        if word >> 29 != 3
            || (word >> 27) & 3 != 2
            || (word >> 23) & 15 != 4
            || (word >> 3) & 15 != 5
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

    pub const fn capture(self, xregs: &[u64; 32]) -> C220BtTransfer {
        C220BtTransfer {
            instruction: self,
            descriptor: C220BtDescriptor::decode(xregs[self.descriptor_register as usize]),
            source_base: xregs[self.source_register as usize],
            destination_base: xregs[self.destination_register as usize],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220BtDescriptor {
    pub raw: u64,
    pub sid: u8,
    pub convert_f16_to_f32: bool,
    pub burst_count: u16,
    pub burst_blocks: u16,
    pub source_gap: u16,
    pub destination_gap: u16,
}

impl C220BtDescriptor {
    pub const fn decode(raw: u64) -> Self {
        Self {
            raw,
            sid: (raw & 15) as u8,
            convert_f16_to_f32: raw & 8 != 0,
            burst_count: ((raw >> 4) & 0xfff) as u16,
            burst_blocks: (raw >> 16) as u16,
            source_gap: (raw >> 32) as u16,
            destination_gap: (raw >> 48) as u16,
        }
    }

    pub const fn is_empty(self) -> bool {
        self.burst_count == 0 || self.burst_blocks == 0
    }

    pub const fn output_block_bytes(self) -> u32 {
        if self.convert_f16_to_f32 { 128 } else { 64 }
    }

    pub const fn source_stride(self) -> u32 {
        self.burst_blocks as u32 * C220_BT_INPUT_BLOCK_BYTES + self.source_gap as u32 * 32
    }

    pub const fn destination_stride(self) -> u64 {
        self.burst_blocks as u64 * self.output_block_bytes() as u64
            + self.destination_gap as u64 * 64
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220BtTransfer {
    pub instruction: C220MovL1ToBtInstruction,
    pub descriptor: C220BtDescriptor,
    pub source_base: u64,
    pub destination_base: u64,
}

impl C220BtTransfer {
    pub const fn block_count(self) -> u32 {
        self.descriptor.burst_count as u32 * self.descriptor.burst_blocks as u32
    }

    pub const fn input_bytes(self) -> u64 {
        self.block_count() as u64 * C220_BT_INPUT_BLOCK_BYTES as u64
    }

    pub const fn output_bytes(self) -> u64 {
        self.block_count() as u64 * self.descriptor.output_block_bytes() as u64
    }

    pub fn segments(self) -> impl ExactSizeIterator<Item = C220BtSegment> {
        (0..self.block_count()).map(move |index| {
            let burst_index = index / u32::from(self.descriptor.burst_blocks);
            let block_index = index % u32::from(self.descriptor.burst_blocks);
            let source_offset = burst_index.wrapping_mul(self.descriptor.source_stride());
            C220BtSegment {
                burst_index: burst_index as u16,
                block_index: block_index as u16,
                source_address: self
                    .source_base
                    .wrapping_add(u64::from(source_offset))
                    .wrapping_add(u64::from(block_index * C220_BT_INPUT_BLOCK_BYTES)),
                destination_address: self
                    .destination_base
                    .wrapping_add(u64::from(burst_index) * self.descriptor.destination_stride())
                    .wrapping_add(u64::from(
                        block_index * self.descriptor.output_block_bytes(),
                    )),
                input_bytes: C220_BT_INPUT_BLOCK_BYTES,
                output_bytes: self.descriptor.output_block_bytes(),
            }
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220BtSegment {
    pub burst_index: u16,
    pub block_index: u16,
    pub source_address: u64,
    pub destination_address: u64,
    pub input_bytes: u32,
    pub output_bytes: u32,
}
