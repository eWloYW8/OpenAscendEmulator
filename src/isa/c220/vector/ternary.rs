#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220TernaryOperation {
    MultiplyAccumulate,
    MultiplyAdd,
    MultiplyAddRelu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220TernaryWidth {
    F16,
    F16ToF32,
    F32,
}

impl C220TernaryWidth {
    pub const fn source_element_bytes(self) -> u8 {
        match self {
            Self::F16 | Self::F16ToF32 => 2,
            Self::F32 => 4,
        }
    }

    pub const fn destination_element_bytes(self) -> u8 {
        match self {
            Self::F16 => 2,
            Self::F16ToF32 | Self::F32 => 4,
        }
    }

    pub const fn lane_count(self) -> usize {
        256 / self.destination_element_bytes() as usize
    }

    pub const fn lane_groups(self) -> u8 {
        match self {
            Self::F16 => 2,
            Self::F16ToF32 | Self::F32 => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220TernaryInstruction {
    pub operation: C220TernaryOperation,
    pub width: C220TernaryWidth,
    pub destination_register: u8,
    pub source_0_register: u8,
    pub source_1_register: u8,
    pub control_register: u8,
}

impl C220TernaryInstruction {
    pub const fn decode(word: u32) -> Option<Self> {
        let (operation, width) = match word & 0xffc0_0003 {
            0x8940_0001 => (
                C220TernaryOperation::MultiplyAccumulate,
                C220TernaryWidth::F16,
            ),
            0x8980_0001 => (
                C220TernaryOperation::MultiplyAccumulate,
                C220TernaryWidth::F16ToF32,
            ),
            0x89c0_0001 => (
                C220TernaryOperation::MultiplyAccumulate,
                C220TernaryWidth::F32,
            ),
            0x9340_0000 => (C220TernaryOperation::MultiplyAdd, C220TernaryWidth::F16),
            0x93c0_0000 => (C220TernaryOperation::MultiplyAdd, C220TernaryWidth::F32),
            0x9340_0001 => (C220TernaryOperation::MultiplyAddRelu, C220TernaryWidth::F16),
            0x93c0_0001 => (C220TernaryOperation::MultiplyAddRelu, C220TernaryWidth::F32),
            _ => return None,
        };
        Some(Self {
            operation,
            width,
            destination_register: ((word >> 17) & 0x1f) as u8,
            source_0_register: ((word >> 12) & 0x1f) as u8,
            source_1_register: ((word >> 7) & 0x1f) as u8,
            control_register: ((word >> 2) & 0x1f) as u8,
        })
    }
}
