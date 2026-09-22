#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220LoadVaInstruction {
    pub destination_va: u8,
    pub source_register: u8,
    pub high_half: bool,
}

impl C220LoadVaInstruction {
    pub const fn decode(word: u32) -> Option<Self> {
        if word & 0xffc0_0ffd != 0x8080_0000 {
            return None;
        }
        Some(Self {
            destination_va: ((word >> 17) & 0x1f) as u8,
            source_register: ((word >> 12) & 0x1f) as u8,
            high_half: word & 2 != 0,
        })
    }
}
