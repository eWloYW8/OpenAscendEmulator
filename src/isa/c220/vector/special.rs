#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220SpecialUnaryOperation {
    Exp,
    Ln,
    Reciprocal,
    ReciprocalSqrt,
    Sqrt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220SpecialUnaryWidth {
    F16,
    F32,
}

impl C220SpecialUnaryWidth {
    pub const fn element_bytes(self) -> u8 {
        match self {
            Self::F16 => 2,
            Self::F32 => 4,
        }
    }

    pub const fn lane_groups(self) -> u8 {
        4 / self.element_bytes()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220SpecialUnaryInstruction {
    pub operation: C220SpecialUnaryOperation,
    pub width: C220SpecialUnaryWidth,
    pub destination_register: u8,
    pub source_register: u8,
    pub control_register: u8,
}

impl C220SpecialUnaryInstruction {
    pub const fn decode(word: u32) -> Option<Self> {
        let (operation, width) = match word & 0xffc0_0f83 {
            0x8340_0080 => (C220SpecialUnaryOperation::Exp, C220SpecialUnaryWidth::F16),
            0x83c0_0080 => (C220SpecialUnaryOperation::Exp, C220SpecialUnaryWidth::F32),
            0x8340_0280 => (C220SpecialUnaryOperation::Ln, C220SpecialUnaryWidth::F16),
            0x83c0_0280 => (C220SpecialUnaryOperation::Ln, C220SpecialUnaryWidth::F32),
            0x8340_0200 => (
                C220SpecialUnaryOperation::Reciprocal,
                C220SpecialUnaryWidth::F16,
            ),
            0x83c0_0200 => (
                C220SpecialUnaryOperation::Reciprocal,
                C220SpecialUnaryWidth::F32,
            ),
            0x8340_0100 => (
                C220SpecialUnaryOperation::ReciprocalSqrt,
                C220SpecialUnaryWidth::F16,
            ),
            0x83c0_0100 => (
                C220SpecialUnaryOperation::ReciprocalSqrt,
                C220SpecialUnaryWidth::F32,
            ),
            0x8340_0c80 => (C220SpecialUnaryOperation::Sqrt, C220SpecialUnaryWidth::F16),
            0x83c0_0c80 => (C220SpecialUnaryOperation::Sqrt, C220SpecialUnaryWidth::F32),
            _ => return None,
        };
        Some(Self {
            operation,
            width,
            destination_register: ((word >> 17) & 0x1f) as u8,
            source_register: ((word >> 12) & 0x1f) as u8,
            control_register: ((word >> 2) & 0x1f) as u8,
        })
    }
}
