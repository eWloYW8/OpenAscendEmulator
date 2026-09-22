#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220ConversionType {
    S4,
    U8,
    S8,
    S16,
    F16,
    Bf16,
    S32,
    F32,
    S64,
}

impl C220ConversionType {
    pub const fn element_bits(self) -> u8 {
        match self {
            Self::S4 => 4,
            Self::U8 | Self::S8 => 8,
            Self::S16 | Self::F16 | Self::Bf16 => 16,
            Self::S32 | Self::F32 => 32,
            Self::S64 => 64,
        }
    }

    pub const fn element_bytes(self) -> Option<u8> {
        let bits = self.element_bits();
        if bits < 8 { None } else { Some(bits / 8) }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220ConversionRound {
    NearestEven,
    NearestAway,
    Floor,
    Ceil,
    TowardZero,
    ToOdd,
}

impl C220ConversionRound {
    const fn decode(selector: u8) -> Option<Self> {
        match selector {
            0 => Some(Self::NearestEven),
            1 => Some(Self::NearestAway),
            2 => Some(Self::Floor),
            3 => Some(Self::Ceil),
            4 => Some(Self::TowardZero),
            5 => Some(Self::ToOdd),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220ConversionKind {
    Ordinary {
        source: C220ConversionType,
        destination: C220ConversionType,
        round: C220ConversionRound,
    },
    S4ToF16,
    VectorDeqS16ToS8 {
        high_half: bool,
    },
    ScalarDeqS16ToS8 {
        high_half: bool,
    },
    ScalarDeqS32ToF16,
}

impl C220ConversionKind {
    pub const fn source_type(self) -> C220ConversionType {
        match self {
            Self::Ordinary { source, .. } => source,
            Self::S4ToF16 => C220ConversionType::S4,
            Self::VectorDeqS16ToS8 { .. } | Self::ScalarDeqS16ToS8 { .. } => {
                C220ConversionType::S16
            }
            Self::ScalarDeqS32ToF16 => C220ConversionType::S32,
        }
    }

    pub const fn destination_type(self) -> C220ConversionType {
        match self {
            Self::Ordinary { destination, .. } => destination,
            Self::S4ToF16 | Self::ScalarDeqS32ToF16 => C220ConversionType::F16,
            Self::VectorDeqS16ToS8 { .. } | Self::ScalarDeqS16ToS8 { .. } => C220ConversionType::S8,
        }
    }

    pub const fn conversion_id(self) -> u16 {
        match self {
            Self::S4ToF16 => 1018,
            Self::VectorDeqS16ToS8 { high_half: false } => 1019,
            Self::VectorDeqS16ToS8 { high_half: true } => 1020,
            Self::ScalarDeqS16ToS8 { high_half: false } => 1021,
            Self::ScalarDeqS16ToS8 { high_half: true } => 1022,
            Self::ScalarDeqS32ToF16 => 1023,
            Self::Ordinary {
                source,
                destination,
                round,
            } => {
                let source = source_selector(source);
                let destination = destination_selector(destination);
                let round = round_selector(round);
                destination as u16 | ((source as u16) << 4) | ((round as u16) << 7)
            }
        }
    }

    pub const fn execute_ticks(self) -> u8 {
        match self {
            Self::VectorDeqS16ToS8 { .. } | Self::ScalarDeqS16ToS8 { .. } => 11,
            Self::ScalarDeqS32ToF16 => 9,
            _ => 5,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220ConversionInstruction {
    pub kind: C220ConversionKind,
    pub destination_register: u8,
    pub source_register: u8,
    pub control_register: u8,
}

impl C220ConversionInstruction {
    pub fn decode(word: u32) -> Option<Self> {
        if word & 0xfe00_0000 != 0x8c00_0000 {
            return None;
        }
        let round_selector = (word & 7) as u8;
        let destination_selector = ((word >> 3) & 0xf) as u8;
        let source_selector = ((word >> 22) & 7) as u8;
        let conversion_id = destination_selector as u16
            | ((source_selector as u16) << 4)
            | ((round_selector as u16) << 7);
        let kind = match conversion_id {
            1018 => C220ConversionKind::S4ToF16,
            1019 => C220ConversionKind::VectorDeqS16ToS8 { high_half: false },
            1020 => C220ConversionKind::VectorDeqS16ToS8 { high_half: true },
            1021 => C220ConversionKind::ScalarDeqS16ToS8 { high_half: false },
            1022 => C220ConversionKind::ScalarDeqS16ToS8 { high_half: true },
            1023 => C220ConversionKind::ScalarDeqS32ToF16,
            _ => {
                let source = decode_source_type(source_selector)?;
                let destination = decode_destination_type(destination_selector)?;
                let round = C220ConversionRound::decode(round_selector)?;
                if !ordinary_form_is_supported(source, destination, round) {
                    return None;
                }
                C220ConversionKind::Ordinary {
                    source,
                    destination,
                    round,
                }
            }
        };
        Some(Self {
            kind,
            destination_register: ((word >> 17) & 0x1f) as u8,
            source_register: ((word >> 12) & 0x1f) as u8,
            control_register: ((word >> 7) & 0x1f) as u8,
        })
    }

    pub const fn lane_count(self) -> usize {
        let source_bits = self.kind.source_type().element_bits() as usize;
        let destination_bits = self.kind.destination_type().element_bits() as usize;
        2048 / if source_bits > destination_bits {
            source_bits
        } else {
            destination_bits
        }
    }

    pub const fn lane_groups(self) -> u8 {
        self.lane_count().div_ceil(64) as u8
    }

    pub const fn source_bytes_per_repeat(self) -> usize {
        (self.lane_count() * self.kind.source_type().element_bits() as usize).div_ceil(8)
    }

    pub const fn destination_bytes_per_repeat(self) -> usize {
        (self.lane_count() * self.kind.destination_type().element_bits() as usize).div_ceil(8)
    }
}

const fn decode_source_type(selector: u8) -> Option<C220ConversionType> {
    match selector {
        0 => Some(C220ConversionType::F32),
        1 => Some(C220ConversionType::F16),
        2 => Some(C220ConversionType::Bf16),
        3 => Some(C220ConversionType::S64),
        4 => Some(C220ConversionType::S32),
        5 => Some(C220ConversionType::S16),
        6 => Some(C220ConversionType::S8),
        7 => Some(C220ConversionType::U8),
        _ => None,
    }
}

const fn decode_destination_type(selector: u8) -> Option<C220ConversionType> {
    match selector {
        0 => Some(C220ConversionType::F32),
        1 => Some(C220ConversionType::F16),
        2 => Some(C220ConversionType::Bf16),
        3 => Some(C220ConversionType::S64),
        4 => Some(C220ConversionType::S32),
        5 => Some(C220ConversionType::S16),
        6 => Some(C220ConversionType::S8),
        7 => Some(C220ConversionType::U8),
        8 => Some(C220ConversionType::S4),
        _ => None,
    }
}

const fn source_selector(source: C220ConversionType) -> u8 {
    match source {
        C220ConversionType::F32 => 0,
        C220ConversionType::F16 => 1,
        C220ConversionType::Bf16 => 2,
        C220ConversionType::S64 => 3,
        C220ConversionType::S32 => 4,
        C220ConversionType::S16 => 5,
        C220ConversionType::S8 => 6,
        C220ConversionType::U8 => 7,
        C220ConversionType::S4 => 7,
    }
}

const fn destination_selector(destination: C220ConversionType) -> u8 {
    match destination {
        C220ConversionType::F32 => 0,
        C220ConversionType::F16 => 1,
        C220ConversionType::Bf16 => 2,
        C220ConversionType::S64 => 3,
        C220ConversionType::S32 => 4,
        C220ConversionType::S16 => 5,
        C220ConversionType::S8 => 6,
        C220ConversionType::U8 => 7,
        C220ConversionType::S4 => 8,
    }
}

const fn round_selector(round: C220ConversionRound) -> u8 {
    match round {
        C220ConversionRound::NearestEven => 0,
        C220ConversionRound::NearestAway => 1,
        C220ConversionRound::Floor => 2,
        C220ConversionRound::Ceil => 3,
        C220ConversionRound::TowardZero => 4,
        C220ConversionRound::ToOdd => 5,
    }
}

const fn ordinary_form_is_supported(
    source: C220ConversionType,
    destination: C220ConversionType,
    round: C220ConversionRound,
) -> bool {
    use C220ConversionRound::*;
    use C220ConversionType::*;
    let any_standard_round = !matches!(round, ToOdd);
    match (source, destination) {
        (F32, F16) => true,
        (F32, F32 | Bf16 | S64 | S32 | S16) => any_standard_round,
        (F16, S32 | S16 | S8 | U8 | S4) => any_standard_round,
        (Bf16, S32) | (S64, F32) | (S32, F32) | (S16, F16) => any_standard_round,
        (F16, F32)
        | (Bf16, F32)
        | (S64, S32)
        | (S32, S64)
        | (S32, S16)
        | (S16, F32)
        | (S8, F16)
        | (U8, F16) => matches!(round, NearestEven),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_captured_conversion_boundaries() {
        for (word, kind, lanes, ticks) in [
            (
                0x8c00_110d,
                C220ConversionKind::Ordinary {
                    source: C220ConversionType::F32,
                    destination: C220ConversionType::F16,
                    round: C220ConversionRound::ToOdd,
                },
                64,
                5,
            ),
            (0x8dc0_1157, C220ConversionKind::S4ToF16, 128, 5),
            (
                0x8dc0_115f,
                C220ConversionKind::VectorDeqS16ToS8 { high_half: false },
                128,
                11,
            ),
            (0x8dc0_117f, C220ConversionKind::ScalarDeqS32ToF16, 64, 9),
        ] {
            let instruction = C220ConversionInstruction::decode(word).unwrap();
            assert_eq!(instruction.kind, kind);
            assert_eq!(instruction.lane_count(), lanes);
            assert_eq!(instruction.kind.execute_ticks(), ticks);
        }
        assert!(C220ConversionInstruction::decode(0x8c80_1128).is_none());
    }
}
