use crate::isa::c220::compare::C220CompareWidth;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220SelectInstruction {
    pub width: C220CompareWidth,
    pub destination_register: u8,
    pub source_0_register: u8,
    pub source_1_register: u8,
    pub control_register: u8,
}

impl C220SelectInstruction {
    pub const fn decode(word: u32) -> Option<Self> {
        let width = match word & 0xffc0_0003 {
            0x9d40_0000 => C220CompareWidth::F16,
            0x9dc0_0000 => C220CompareWidth::F32,
            _ => return None,
        };
        Some(Self {
            width,
            destination_register: ((word >> 17) & 0x1f) as u8,
            source_0_register: ((word >> 12) & 0x1f) as u8,
            source_1_register: ((word >> 7) & 0x1f) as u8,
            control_register: ((word >> 2) & 0x1f) as u8,
        })
    }
}
