#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220GatherWidth {
    B16,
    B32,
}

impl C220GatherWidth {
    pub const fn element_bytes(self) -> u8 {
        match self {
            Self::B16 => 2,
            Self::B32 => 4,
        }
    }

    pub const fn lane_count(self) -> usize {
        256 / self.element_bytes() as usize
    }

    pub const fn groups_per_repeat(self) -> u8 {
        (self.lane_count() / 16) as u8
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220GatherKind {
    Elements(C220GatherWidth),
    Blocks,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220GatherInstruction {
    pub kind: C220GatherKind,
    pub destination_register: u8,
    pub index_register: u8,
    pub control_register: u8,
}

impl C220GatherInstruction {
    pub const fn decode(word: u32) -> Option<Self> {
        let kind = match word & 0xffc0_007f {
            0x8000_0042 => C220GatherKind::Elements(C220GatherWidth::B16),
            0x8000_004a => C220GatherKind::Elements(C220GatherWidth::B32),
            0x8000_0043 => C220GatherKind::Blocks,
            _ => return None,
        };
        Some(Self {
            kind,
            destination_register: ((word >> 17) & 0x1f) as u8,
            index_register: ((word >> 12) & 0x1f) as u8,
            control_register: ((word >> 7) & 0x1f) as u8,
        })
    }
}
