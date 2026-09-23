#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MovOutToL1Instruction {
    pub destination_register: u8,
    pub source_register: u8,
    pub descriptor_register: u8,
    pub layout: C220L1DmaLayout,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum C220L1DmaLayout {
    Copy32,
    Pad1,
    Pad2,
    Pad4,
    Pad8,
    Pad16,
    Take4,
    Take8,
    Take16,
}

impl C220L1DmaLayout {
    pub const fn source_bytes(self) -> u32 {
        match self {
            Self::Pad1 => 1,
            Self::Pad2 => 2,
            Self::Pad4 => 4,
            Self::Pad8 => 8,
            Self::Pad16 => 16,
            _ => 32,
        }
    }

    pub const fn destination_bytes(self) -> u32 {
        match self {
            Self::Take4 => 4,
            Self::Take8 => 8,
            Self::Take16 => 16,
            _ => 32,
        }
    }
}

impl C220MovOutToL1Instruction {
    pub const fn decode(word: u32) -> Option<Self> {
        if word >> 29 != 3
            || (word >> 27) & 3 != 2
            || (word >> 23) & 15 != 2
            || (word >> 3) & 15 != 4
        {
            return None;
        }
        let layout = match (word & 7) | ((word >> 19) & 8) {
            0 => C220L1DmaLayout::Copy32,
            1 => C220L1DmaLayout::Pad1,
            2 => C220L1DmaLayout::Pad2,
            3 => C220L1DmaLayout::Pad4,
            4 => C220L1DmaLayout::Pad8,
            5 => C220L1DmaLayout::Pad16,
            6 => C220L1DmaLayout::Take4,
            7 => C220L1DmaLayout::Take8,
            8 => C220L1DmaLayout::Take16,
            _ => return None,
        };
        Some(Self {
            destination_register: ((word >> 17) & 31) as u8,
            source_register: ((word >> 12) & 31) as u8,
            descriptor_register: ((word >> 7) & 31) as u8,
            layout,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220L1DmaSegment {
    pub burst_index: u16,
    pub unit_index: u16,
    pub source_address: u64,
    pub destination_address: u64,
    pub source_bytes: u32,
    pub destination_bytes: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220L1DmaDescriptor {
    pub xm: u64,
    pub layout: C220L1DmaLayout,
}

impl C220L1DmaDescriptor {
    pub const fn burst_count(self) -> u16 {
        ((self.xm >> 4) & 0xfff) as u16
    }

    pub const fn burst_length(self) -> u16 {
        (self.xm >> 16) as u16
    }

    pub const fn source_gap(self) -> u16 {
        (self.xm >> 32) as u16
    }

    pub const fn destination_gap(self) -> u16 {
        (self.xm >> 48) as u16
    }

    pub const fn is_disabled(self) -> bool {
        self.burst_count() == 0 || self.burst_length() == 0
    }

    pub fn segments(
        self,
        source: u64,
        destination: u64,
    ) -> impl ExactSizeIterator<Item = C220L1DmaSegment> + Clone {
        let burst = super::burst::BurstLayout::decode(self.xm);
        let source_bytes = self.layout.source_bytes();
        let destination_bytes = self.layout.destination_bytes();
        let source_stride = (u32::from(burst.length) + u32::from(burst.source_gap)) * source_bytes;
        let destination_stride = (u64::from(burst.length) + u64::from(burst.destination_gap))
            * u64::from(destination_bytes);
        (0..usize::from(burst.count) * usize::from(burst.length)).map(move |index| {
            let burst_index = (index / usize::from(burst.length)) as u16;
            let unit_index = (index % usize::from(burst.length)) as u16;
            C220L1DmaSegment {
                burst_index,
                unit_index,
                source_bytes,
                destination_bytes,
                source_address: source
                    .wrapping_add(u64::from(
                        u32::from(burst_index).wrapping_mul(source_stride),
                    ))
                    .wrapping_add(u64::from(unit_index) * u64::from(source_bytes)),
                destination_address: destination
                    .wrapping_add(u64::from(burst_index) * destination_stride)
                    .wrapping_add(u64::from(unit_index) * u64::from(destination_bytes)),
            }
        })
    }
}
