#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220CompareWidth {
    F16,
    S32,
    F32,
}

impl C220CompareWidth {
    pub const fn element_bytes(self) -> u8 {
        match self {
            Self::F16 => 2,
            Self::S32 | Self::F32 => 4,
        }
    }

    pub const fn groups_per_repeat(self) -> usize {
        match self {
            Self::F16 => 2,
            Self::S32 | Self::F32 => 1,
        }
    }

    pub const fn lane_count(self) -> usize {
        256 / self.element_bytes() as usize
    }

    pub const fn packed_bytes_per_repeat(self) -> usize {
        match self {
            Self::F16 => 16,
            Self::S32 | Self::F32 => 8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220CompareCondition {
    Equal,
    NotEqual,
    Less,
    Greater,
    GreaterEqual,
    LessEqual,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220PackedCompareOperand {
    VectorRegister(u8),
    ScalarRegister(u8),
    ScalarMemoryRegister(u8),
}

impl C220PackedCompareOperand {
    pub const fn register(self) -> u8 {
        match self {
            Self::VectorRegister(register)
            | Self::ScalarRegister(register)
            | Self::ScalarMemoryRegister(register) => register,
        }
    }

    pub const fn is_vector(self) -> bool {
        matches!(self, Self::VectorRegister(_))
    }
}

impl C220CompareCondition {
    pub const fn decode(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Equal),
            1 => Some(Self::NotEqual),
            2 => Some(Self::Less),
            3 => Some(Self::Greater),
            4 => Some(Self::GreaterEqual),
            5 => Some(Self::LessEqual),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220PackedCompareInstruction {
    pub width: C220CompareWidth,
    pub condition: C220CompareCondition,
    pub destination_register: u8,
    pub source_0_register: u8,
    pub operand: C220PackedCompareOperand,
    pub control_register: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220CompareMaskInstruction {
    pub width: C220CompareWidth,
    pub condition: C220CompareCondition,
    pub source_0_register: u8,
    pub source_1_register: u8,
    pub control_register: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220MoveMaskDirection {
    ToMemory,
    FromMemory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MoveMaskInstruction {
    pub direction: C220MoveMaskDirection,
    pub address_register: u8,
}

impl C220MoveMaskInstruction {
    pub const fn decode(word: u32) -> Option<Self> {
        let direction = match word & 0xffc1_ffff {
            0x9e00_0002 => C220MoveMaskDirection::ToMemory,
            0x9e00_0003 => C220MoveMaskDirection::FromMemory,
            _ => return None,
        };
        Some(Self {
            direction,
            address_register: ((word >> 17) & 0x1f) as u8,
        })
    }
}

impl C220CompareMaskInstruction {
    pub const fn decode(word: u32) -> Option<Self> {
        let width = match word & 0xfff0_0003 {
            0x9940_0000 => C220CompareWidth::F16,
            0x99c0_0000 => C220CompareWidth::F32,
            _ => return None,
        };
        let condition = match C220CompareCondition::decode(((word >> 17) & 7) as u8) {
            Some(condition) => condition,
            None => return None,
        };
        Some(Self {
            width,
            condition,
            source_0_register: ((word >> 12) & 0x1f) as u8,
            source_1_register: ((word >> 7) & 0x1f) as u8,
            control_register: ((word >> 2) & 0x1f) as u8,
        })
    }
}

impl C220PackedCompareInstruction {
    pub const fn decode(word: u32) -> Option<Self> {
        let decoded_condition = match C220CompareCondition::decode(((word >> 22) & 7) as u8) {
            Some(condition) => condition,
            None => return None,
        };
        let (width, condition, operand) = match word & 0xfe00_0003 {
            0x9800_0001 => (
                C220CompareWidth::F16,
                decoded_condition,
                C220PackedCompareOperand::VectorRegister(((word >> 7) & 0x1f) as u8),
            ),
            0x9800_0002 if word & 0x01c0_0000 == 0 => (
                C220CompareWidth::S32,
                C220CompareCondition::Equal,
                C220PackedCompareOperand::VectorRegister(((word >> 7) & 0x1f) as u8),
            ),
            0x9800_0003 => (
                C220CompareWidth::F32,
                decoded_condition,
                C220PackedCompareOperand::VectorRegister(((word >> 7) & 0x1f) as u8),
            ),
            0x9a00_0002 => (
                C220CompareWidth::F16,
                decoded_condition,
                C220PackedCompareOperand::ScalarRegister(((word >> 7) & 0x1f) as u8),
            ),
            0x9a00_0003 => (
                C220CompareWidth::F32,
                decoded_condition,
                C220PackedCompareOperand::ScalarRegister(((word >> 7) & 0x1f) as u8),
            ),
            0x9800_0002 if word & 0x01c0_0000 == 0x0100_0000 => (
                C220CompareWidth::S32,
                C220CompareCondition::Equal,
                C220PackedCompareOperand::ScalarMemoryRegister(((word >> 7) & 0x1f) as u8),
            ),
            _ => return None,
        };
        Some(Self {
            width,
            condition,
            destination_register: ((word >> 17) & 0x1f) as u8,
            source_0_register: ((word >> 12) & 0x1f) as u8,
            operand,
            control_register: ((word >> 2) & 0x1f) as u8,
        })
    }
}
