#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220PreloadOffset {
    Immediate(u16),
    Register(u8),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220ScalarPreload {
    pub base_register: u8,
    pub offset: C220PreloadOffset,
    pub post_index: bool,
}

impl C220ScalarPreload {
    pub const fn decode(word: u32) -> Option<Self> {
        if word >> 29 != 0 {
            return None;
        }
        match (word >> 24) & 31 {
            1 if (word >> 4) & 7 == 5 => Some(Self {
                base_register: (((word >> 12) & 31) | ((word & 2) << 4)) as u8,
                offset: C220PreloadOffset::Register((((word >> 7) & 31) | ((word & 1) << 5)) as u8),
                post_index: word & 8 != 0,
            }),
            8 if (word >> 22) & 3 == 3 => Some(Self {
                base_register: ((word >> 12) & 63) as u8,
                offset: C220PreloadOffset::Immediate((word & 0xfff) as u16),
                post_index: false,
            }),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220AtomicStoreOffset {
    Immediate(i16),
    Register(u8),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220ScalarAtomicStore {
    pub source_register: u8,
    pub base_register: u8,
    pub width_bytes: u8,
    pub offset: C220AtomicStoreOffset,
    pub post_index: bool,
}

impl C220ScalarAtomicStore {
    pub const fn decode(word: u32) -> Option<Self> {
        if word >> 29 != 0 {
            return None;
        }
        let (offset, post_index) = match (word >> 24) & 31 {
            1 if (word >> 4) & 7 == 7 => (
                C220AtomicStoreOffset::Register(((word >> 7) & 31) as u8),
                word & 8 != 0,
            ),
            22 | 23 => (
                C220AtomicStoreOffset::Immediate(((word as i16) << 4) >> 4),
                word & (1 << 24) != 0,
            ),
            _ => return None,
        };
        Some(Self {
            source_register: ((word >> 17) & 31) as u8,
            base_register: ((word >> 12) & 31) as u8,
            width_bytes: 1 << ((word >> 22) & 3),
            offset,
            post_index,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220ScalarDeviceLoad {
    pub destination_register: u8,
    pub base_register: u8,
    pub width_bytes: u8,
    pub offset_bytes: i16,
}

impl C220ScalarDeviceLoad {
    pub const fn decode(word: u32) -> Option<Self> {
        if word >> 29 != 0 || (word >> 25) & 0xf != 13 {
            return None;
        }
        Some(Self {
            destination_register: ((word >> 17) & 31) as u8,
            base_register: ((word >> 12) & 31) as u8,
            width_bytes: 1 << ((word >> 22) & 3),
            offset_bytes: ((word as i16) << 4) >> 4,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220ScalarDirectStore {
    pub source_register: u8,
    pub base_register: u8,
    pub width_bytes: u8,
    pub offset_bytes: i16,
}

impl C220ScalarDirectStore {
    pub const fn decode(word: u32) -> Option<Self> {
        if word >> 29 != 0 || (word >> 25) & 0xf != 12 {
            return None;
        }
        Some(Self {
            source_register: ((word >> 17) & 31) as u8,
            base_register: ((word >> 12) & 31) as u8,
            width_bytes: 1 << ((word >> 22) & 3),
            offset_bytes: ((word as i16) << 4) >> 4,
        })
    }

    pub const fn effective_address(self, base: u64) -> u64 {
        base.wrapping_add_signed(self.offset_bytes as i64)
    }
}

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
    F32ToS32NearestAway,
    F32ToS32Floor,
    F32ToS32Ceil,
    F32ToS32Truncate,
    F32ToS32NearestEven,
    S32ToF32,
    F32ToF16,
    F16ToF32,
    F32ToF16Odd,
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
            0 => C220ScalarConversion::F32ToS32NearestAway,
            1 => C220ScalarConversion::F32ToS32Floor,
            2 => C220ScalarConversion::F32ToS32Ceil,
            3 => C220ScalarConversion::F32ToS32Truncate,
            4 => C220ScalarConversion::F32ToS32NearestEven,
            5 => C220ScalarConversion::S32ToF32,
            6 => C220ScalarConversion::F32ToF16,
            7 => C220ScalarConversion::F16ToF32,
            8 => C220ScalarConversion::F32ToF16Odd,
            _ => return None,
        };
        Some(Self {
            conversion,
            destination_register: ((word >> 17) & 0x1f) as u8,
            source_register: ((word >> 12) & 0x1f) as u8,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_conversion_words_decode() {
        for opcode in [0x1800_0000, 0x1900_0000] {
            for size in 0..4 {
                let word = opcode | (size << 22) | (31 << 17) | (17 << 12) | 0xffd;
                let store = C220ScalarDirectStore::decode(word).unwrap();
                assert_eq!((store.source_register, store.base_register), (31, 17));
                assert_eq!(store.width_bytes, 1 << size);
                assert_eq!(store.offset_bytes, -3);
                assert_eq!(store.effective_address(2), u64::MAX);
            }
        }
        assert!(C220ScalarDirectStore::decode(0x1a00_0000).is_none());
        assert!(C220ScalarDirectStore::decode(0x3800_0000).is_none());
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
