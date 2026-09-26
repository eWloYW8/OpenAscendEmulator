#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220CubeSprWrite {
    pub destination_spr: u16,
    pub source_register: u8,
}

impl C220CubeSprWrite {
    pub const fn decode(word: u32) -> Option<Self> {
        if word >> 24 != 2 || (word >> 7) & 31 != 18 || (word >> 17) & 127 != 52 {
            return None;
        }
        Some(Self {
            destination_spr: 52,
            source_register: ((word >> 12) & 31) as u8,
        })
    }
}
