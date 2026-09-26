#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220VectorSprWrite {
    pub destination_spr: u16,
    pub source_register: u8,
}

impl C220VectorSprWrite {
    pub const fn decode(word: u32) -> Option<Self> {
        if word >> 24 != 2 || (word >> 7) & 31 != 18 {
            return None;
        }
        let destination_spr = ((word >> 17) & 127) as u16;
        if !matches!(destination_spr, 12 | 17 | 19 | 48..=51 | 55..=57 | 60 | 63 | 69) {
            return None;
        }
        Some(Self {
            destination_spr,
            source_register: ((word >> 12) & 31) as u8,
        })
    }
}
