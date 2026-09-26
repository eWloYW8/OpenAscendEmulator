#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Mte1SprWrite {
    pub destination_spr: u16,
    pub source_register: u8,
}

impl C220Mte1SprWrite {
    pub const fn decode(word: u32) -> Option<Self> {
        if word >> 24 != 2 || (word >> 7) & 31 != 18 {
            return None;
        }
        let destination_spr = ((word >> 17) & 127) as u16;
        if !matches!(destination_spr, 10 | 13 | 15 | 22 | 53 | 54 | 58 | 92) {
            return None;
        }
        Some(Self {
            destination_spr,
            source_register: ((word >> 12) & 31) as u8,
        })
    }
}
