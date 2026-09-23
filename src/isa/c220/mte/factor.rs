#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220FactorSource {
    L1,
    Ub,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FactorLoadInstruction {
    pub source: C220FactorSource,
    pub destination_register: u8,
    pub source_register: u8,
    pub descriptor_register: u8,
}

impl C220FactorLoadInstruction {
    pub const fn decode(word: u32) -> Option<Self> {
        if word >> 29 != 6 || (word >> 24) & 31 != 0 {
            return None;
        }
        Some(Self {
            source: if word & (1 << 22) == 0 {
                C220FactorSource::L1
            } else {
                C220FactorSource::Ub
            },
            destination_register: ((word >> 17) & 31) as u8,
            source_register: ((word >> 12) & 31) as u8,
            descriptor_register: ((word >> 7) & 31) as u8,
        })
    }

    pub const fn capture(self, xregs: &[u64; 32]) -> C220FactorLoad {
        let destination = xregs[self.destination_register as usize];
        C220FactorLoad {
            source: self.source,
            source_address: xregs[self.source_register as usize],
            destination_address: (destination & 65535)
                + if (destination >> 16) & 65535 != 0 {
                    2048
                } else {
                    0
                },
            descriptor: C220FactorDescriptor(xregs[self.descriptor_register as usize]),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FactorDescriptor(pub u64);

impl C220FactorDescriptor {
    pub const fn convert(self) -> bool {
        self.0 & 8 != 0
    }
    pub const fn stream_id(self) -> u8 {
        (self.0 & 15) as u8
    }
    pub const fn burst_count(self) -> u32 {
        ((self.0 >> 4) & 4095) as u32
    }
    pub const fn burst_blocks(self) -> u32 {
        ((self.0 >> 16) & 65535) as u32
    }
    pub const fn output_block_bytes(self) -> u32 {
        if self.convert() { 256 } else { 128 }
    }
    pub const fn source_stride(self) -> u32 {
        self.burst_blocks() * 128 + ((self.0 >> 32) & 65535) as u32 * 32
    }
    pub const fn destination_stride(self) -> u64 {
        (self.burst_blocks() as u64 + (self.0 >> 48)) * self.output_block_bytes() as u64
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FactorLoad {
    pub source: C220FactorSource,
    pub source_address: u64,
    pub destination_address: u64,
    pub descriptor: C220FactorDescriptor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FactorBlock {
    pub burst: u32,
    pub block: u32,
    pub source_address: u64,
    pub destination_address: u64,
    pub output_bytes: u32,
}

impl C220FactorLoad {
    pub fn blocks(self) -> impl ExactSizeIterator<Item = C220FactorBlock> {
        let descriptor = self.descriptor;
        (0..descriptor.burst_count() * descriptor.burst_blocks()).map(move |index| {
            let burst = index / descriptor.burst_blocks();
            let block = index % descriptor.burst_blocks();
            C220FactorBlock {
                burst,
                block,
                source_address: self
                    .source_address
                    .wrapping_add(u64::from(burst.wrapping_mul(descriptor.source_stride())))
                    .wrapping_add(u64::from(block) * 128),
                destination_address: self
                    .destination_address
                    .wrapping_add(u64::from(burst) * descriptor.destination_stride())
                    .wrapping_add(u64::from(block * descriptor.output_block_bytes())),
                output_bytes: descriptor.output_block_bytes(),
            }
        })
    }
}
