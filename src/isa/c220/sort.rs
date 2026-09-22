#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220SortWidth {
    F16,
    F32,
}

impl C220SortWidth {
    pub const fn element_bytes(self) -> u8 {
        match self {
            Self::F16 => 2,
            Self::F32 => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220SortInstruction {
    pub width: C220SortWidth,
    pub destination_register: u8,
    pub value_register: u8,
    pub index_register: u8,
    pub control_register: u8,
}

impl C220SortInstruction {
    pub const fn decode(word: u32) -> Option<Self> {
        const REGISTERS: u32 = 0x003e_0000 | 0x0001_f000 | 0x0000_0f80 | 0x0000_007c;
        let width = match word & !REGISTERS {
            0x8540_0002 => C220SortWidth::F16,
            0x85c0_0002 => C220SortWidth::F32,
            _ => return None,
        };
        Some(Self {
            width,
            destination_register: ((word >> 17) & 0x1f) as u8,
            value_register: ((word >> 12) & 0x1f) as u8,
            index_register: ((word >> 7) & 0x1f) as u8,
            control_register: ((word >> 2) & 0x1f) as u8,
        })
    }
}
