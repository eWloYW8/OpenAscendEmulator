#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220VectorScalarType {
    F16,
    S16,
    S32,
    F32,
}

impl C220VectorScalarType {
    pub const fn element_bytes(self) -> u8 {
        match self {
            Self::F16 | Self::S16 => 2,
            Self::S32 | Self::F32 => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220VectorScalarOperation {
    Add,
    Multiply,
    Maximum,
    Minimum,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220VectorScalarInstruction {
    pub operation: C220VectorScalarOperation,
    pub dtype: C220VectorScalarType,
    pub destination_register: u8,
    pub source_register: u8,
    pub scalar_register: u8,
    pub control_register: u8,
}

impl C220VectorScalarInstruction {
    pub const fn decode(word: u32) -> Option<Self> {
        let (operation, dtype) = match word & 0xffc0_0003 {
            0x9640_0000 => (
                C220VectorScalarOperation::Maximum,
                C220VectorScalarType::F16,
            ),
            0x9640_0001 => (
                C220VectorScalarOperation::Minimum,
                C220VectorScalarType::F16,
            ),
            0x9600_0000 => (
                C220VectorScalarOperation::Maximum,
                C220VectorScalarType::S32,
            ),
            0x9600_0001 => (
                C220VectorScalarOperation::Minimum,
                C220VectorScalarType::S32,
            ),
            0x9680_0000 => (
                C220VectorScalarOperation::Maximum,
                C220VectorScalarType::S16,
            ),
            0x9680_0001 => (
                C220VectorScalarOperation::Minimum,
                C220VectorScalarType::S16,
            ),
            0x96c0_0000 => (
                C220VectorScalarOperation::Maximum,
                C220VectorScalarType::F32,
            ),
            0x96c0_0001 => (
                C220VectorScalarOperation::Minimum,
                C220VectorScalarType::F32,
            ),
            0x9700_0000 => (C220VectorScalarOperation::Add, C220VectorScalarType::S32),
            0x9700_0001 => (
                C220VectorScalarOperation::Multiply,
                C220VectorScalarType::S32,
            ),
            0x9740_0000 => (C220VectorScalarOperation::Add, C220VectorScalarType::F16),
            0x9740_0001 => (
                C220VectorScalarOperation::Multiply,
                C220VectorScalarType::F16,
            ),
            0x9780_0000 => (C220VectorScalarOperation::Add, C220VectorScalarType::S16),
            0x9780_0001 => (
                C220VectorScalarOperation::Multiply,
                C220VectorScalarType::S16,
            ),
            0x97c0_0000 => (C220VectorScalarOperation::Add, C220VectorScalarType::F32),
            0x97c0_0001 => (
                C220VectorScalarOperation::Multiply,
                C220VectorScalarType::F32,
            ),
            _ => return None,
        };
        Some(Self {
            operation,
            dtype,
            destination_register: ((word >> 17) & 0x1f) as u8,
            source_register: ((word >> 12) & 0x1f) as u8,
            scalar_register: ((word >> 7) & 0x1f) as u8,
            control_register: ((word >> 2) & 0x1f) as u8,
        })
    }
}
