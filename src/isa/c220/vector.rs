#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MovevInstruction {
    pub word: u32,
    pub dtype_selector: u8,
    pub destination_register: u8,
    pub source_register: u8,
    pub control_register: u8,
}

impl C220MovevInstruction {
    pub const fn decode(word: u32) -> Option<Self> {
        if (word >> 29) != 4
            || ((word >> 25) & 0xf) != 1
            || ((word >> 7) & 0x1f) != 0
            || word & 3 != 0
        {
            return None;
        }
        let dtype_selector = ((word >> 22) & 7) as u8;
        Some(Self {
            word,
            dtype_selector,
            destination_register: ((word >> 17) & 0x1f) as u8,
            source_register: ((word >> 12) & 0x1f) as u8,
            control_register: ((word >> 2) & 0x1f) as u8,
        })
    }

    pub const fn supported_element_bytes(self) -> Option<u8> {
        match self.dtype_selector {
            1 => Some(2),
            2 => Some(4),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MovemaskHint {
    pub source_register: u8,
    pub destination_spr: u16,
}

impl C220MovemaskHint {
    pub const fn from_word(word: u32) -> Option<Self> {
        if word & 0xffc0_0000 != 0x8040_0000 {
            return None;
        }
        Some(Self {
            source_register: ((word >> 2) & 0x1f) as u8,
            destination_spr: 100 + ((word >> 7) & 1) as u16,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220ShiftOperation {
    Left,
    RightUnsigned,
    RightSigned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220ShiftInstruction {
    pub operation: C220ShiftOperation,
    pub element_bytes: u8,
    pub round: bool,
    pub destination_register: u8,
    pub source_register: u8,
    pub shift_register: u8,
    pub control_register: u8,
}

impl C220ShiftInstruction {
    pub const fn decode(word: u32) -> Option<Self> {
        let (operation, element_bytes) = match word & 0xffc0_0003 {
            0x9c80_0003 => (C220ShiftOperation::Left, 2),
            0x9cc0_0003 => (C220ShiftOperation::Left, 4),
            0x9b00_0000 | 0x9b00_0001 => (C220ShiftOperation::RightUnsigned, 2),
            0x9b40_0000 | 0x9b40_0001 => (C220ShiftOperation::RightSigned, 2),
            0x9b80_0000 | 0x9b80_0001 => (C220ShiftOperation::RightUnsigned, 4),
            0x9bc0_0000 | 0x9bc0_0001 => (C220ShiftOperation::RightSigned, 4),
            _ => return None,
        };
        Some(Self {
            operation,
            element_bytes,
            round: matches!(operation, C220ShiftOperation::RightSigned) && word & 1 != 0,
            destination_register: ((word >> 17) & 0x1f) as u8,
            source_register: ((word >> 12) & 0x1f) as u8,
            shift_register: ((word >> 7) & 0x1f) as u8,
            control_register: ((word >> 2) & 0x1f) as u8,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220VecArithmeticOperation {
    Absolute,
    Rectify,
    Not,
    Add,
    Subtract,
    Multiply,
    Divide,
    Maximum,
    Minimum,
    Or,
    And,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220VecArithmeticHint {
    pub operation: C220VecArithmeticOperation,
    pub destination_register: u8,
    pub source_0_register: u8,
    pub source_1_register: Option<u8>,
    pub control_register: u8,
    pub dtype_selector: u8,
}

impl C220VecArithmeticHint {
    pub const fn from_word(word: u32) -> Option<Self> {
        if (word >> 29) != 4 {
            return None;
        }
        let operation = match word & 0xffc0_0003 {
            0x9a40_0000 => C220VecArithmeticOperation::Or,
            0x9a40_0001 => C220VecArithmeticOperation::And,
            _ if word & 0xffc0_0f83 == 0x8240_0800 => C220VecArithmeticOperation::Not,
            _ => match (((word >> 25) & 0xf), word & 3) {
                (1, 0) if ((word >> 7) & 0x1f) == 3 && word & (1 << 24) != 0 => {
                    C220VecArithmeticOperation::Rectify
                }
                (1, 0) if ((word >> 7) & 0x1f) == 6 && word & (1 << 24) != 0 => {
                    C220VecArithmeticOperation::Absolute
                }
                (2, 0) => C220VecArithmeticOperation::Add,
                (2, 1) => C220VecArithmeticOperation::Subtract,
                (3, 0) => C220VecArithmeticOperation::Maximum,
                (3, 1) => C220VecArithmeticOperation::Minimum,
                (4, 0) => C220VecArithmeticOperation::Multiply,
                (4, 1) => C220VecArithmeticOperation::Divide,
                _ => return None,
            },
        };
        let dtype_selector = match operation {
            C220VecArithmeticOperation::Divide | C220VecArithmeticOperation::Rectify => {
                ((word >> 22) & 7) as u8
            }
            _ => ((word >> 22) & 3) as u8,
        };
        Some(Self {
            operation,
            destination_register: ((word >> 17) & 0x1f) as u8,
            source_0_register: ((word >> 12) & 0x1f) as u8,
            source_1_register: match operation {
                C220VecArithmeticOperation::Absolute
                | C220VecArithmeticOperation::Rectify
                | C220VecArithmeticOperation::Not => None,
                _ => Some(((word >> 7) & 0x1f) as u8),
            },
            control_register: ((word >> 2) & 0x1f) as u8,
            dtype_selector,
        })
    }

    pub const fn has_fp32_value_path(self) -> bool {
        match self.operation {
            C220VecArithmeticOperation::Divide | C220VecArithmeticOperation::Rectify => {
                self.dtype_selector == 7
            }
            _ => self.dtype_selector == 3,
        }
    }

    pub const fn has_s32_value_path(self) -> bool {
        self.dtype_selector == 0
            && matches!(
                self.operation,
                C220VecArithmeticOperation::Add
                    | C220VecArithmeticOperation::Subtract
                    | C220VecArithmeticOperation::Multiply
                    | C220VecArithmeticOperation::Maximum
                    | C220VecArithmeticOperation::Minimum
            )
    }

    pub const fn has_s16_value_path(self) -> bool {
        self.dtype_selector == 2
            && matches!(
                self.operation,
                C220VecArithmeticOperation::Add
                    | C220VecArithmeticOperation::Subtract
                    | C220VecArithmeticOperation::Multiply
                    | C220VecArithmeticOperation::Maximum
                    | C220VecArithmeticOperation::Minimum
            )
    }

    pub const fn has_f16_value_path(self) -> bool {
        self.dtype_selector == 1
            && matches!(
                self.operation,
                C220VecArithmeticOperation::Add
                    | C220VecArithmeticOperation::Subtract
                    | C220VecArithmeticOperation::Multiply
                    | C220VecArithmeticOperation::Maximum
                    | C220VecArithmeticOperation::Minimum
            )
    }

    pub const fn has_bitwise_b16_value_path(self) -> bool {
        self.dtype_selector == 1
            && matches!(
                self.operation,
                C220VecArithmeticOperation::Or
                    | C220VecArithmeticOperation::And
                    | C220VecArithmeticOperation::Not
            )
    }

    pub const fn modeled_element_bytes(self) -> Option<u8> {
        if self.has_s16_value_path()
            || self.has_f16_value_path()
            || self.has_bitwise_b16_value_path()
        {
            Some(2)
        } else if self.has_s32_value_path() || self.has_fp32_value_path() {
            Some(4)
        } else {
            None
        }
    }
}
