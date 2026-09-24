#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220ScalarSprImmediate {
    pub destination_spr: u16,
    pub immediate: u16,
}

impl C220ScalarSprImmediate {
    pub const fn decode(word: u32) -> Option<Self> {
        if word >> 29 != 0 || ((word >> 24) & 0x1f) != 18 {
            return None;
        }
        Some(Self {
            destination_spr: ((word >> 17) & 0x7f) as u16,
            immediate: word as u16,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220ScalarConversion {
    F32ToS32Truncate,
    S32ToF32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220ScalarConversionHint {
    pub conversion: C220ScalarConversion,
    pub destination_register: u8,
    pub source_register: u8,
}

impl C220ScalarConversionHint {
    pub const fn from_word(word: u32) -> Option<Self> {
        if word >> 29 != 0 || ((word >> 24) & 0x1f) != 2 || ((word >> 7) & 0x1f) != 11 {
            return None;
        }
        let conversion = match word & 0x1f {
            3 => C220ScalarConversion::F32ToS32Truncate,
            5 => C220ScalarConversion::S32ToF32,
            _ => return None,
        };
        Some(Self {
            conversion,
            destination_register: (((word >> 17) & 0x1f) | ((word >> 1) & 0x20)) as u8,
            source_register: (((word >> 12) & 0x1f) | (word & 0x20)) as u8,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_conversion_words_decode() {
        let truncate = C220ScalarConversionHint::from_word(0x0210_8583).unwrap();
        assert_eq!(truncate.conversion, C220ScalarConversion::F32ToS32Truncate);
        assert_eq!(
            (truncate.destination_register, truncate.source_register),
            (8, 8)
        );
        assert_eq!(
            C220ScalarConversionHint::from_word(0x0210_8585)
                .unwrap()
                .conversion,
            C220ScalarConversion::S32ToF32
        );
        assert!(C220ScalarConversionHint::from_word(0x0210_8589).is_none());
    }
}
