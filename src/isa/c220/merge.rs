#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220MergeWidth {
    F16,
    F32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MergeInstruction {
    pub width: C220MergeWidth,
    pub destination_register: u8,
    pub sources_register: u8,
    pub lengths_register: u8,
    pub control_register: u8,
}

impl C220MergeInstruction {
    pub const fn decode(word: u32) -> Option<Self> {
        const REGISTERS: u32 = 0x003e_0000 | 0x0001_f000 | 0x0000_0f80 | 0x0000_007c;
        let width = match word & !REGISTERS {
            0x8540_0003 => C220MergeWidth::F16,
            0x85c0_0003 => C220MergeWidth::F32,
            _ => return None,
        };
        Some(Self {
            width,
            destination_register: ((word >> 17) & 0x1f) as u8,
            sources_register: ((word >> 12) & 0x1f) as u8,
            lengths_register: ((word >> 7) & 0x1f) as u8,
            control_register: ((word >> 2) & 0x1f) as u8,
        })
    }
}
