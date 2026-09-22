use crate::isa::c220::vector::{
    C220MovevInstruction, C220VecArithmeticHint, C220VecArithmeticOperation,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220NoEffectVectorOperation {
    Movev,
    Vci,
    Vexp,
    Vrsqrt,
    Vrelu,
    Vrec,
    Vln,
    Vabs,
    Vrpac,
    Vbs16,
    Vms4,
    Vextract,
    Vconcat,
    Vmergech,
    RpnCorDiag,
    Vsqrt,
    RpnCor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220NoEffectVectorInstruction {
    pub operation: C220NoEffectVectorOperation,
    pub dtype_selector: u8,
    pub control_register: u8,
}

impl C220NoEffectVectorInstruction {
    pub const fn decode(word: u32) -> Option<Self> {
        if let Some(instruction) = C220MovevInstruction::decode(word)
            && instruction.supported_element_bytes().is_none()
        {
            return Some(Self {
                operation: C220NoEffectVectorOperation::Movev,
                dtype_selector: instruction.dtype_selector,
                control_register: instruction.control_register,
            });
        }
        if let Some(hint) = C220VecArithmeticHint::from_word(word) {
            let operation = match (hint.operation, hint.dtype_selector) {
                (C220VecArithmeticOperation::Rectify, 6) => C220NoEffectVectorOperation::Vrelu,
                (C220VecArithmeticOperation::Absolute, 4) => C220NoEffectVectorOperation::Vabs,
                _ => return None,
            };
            return Some(Self {
                operation,
                dtype_selector: hint.dtype_selector,
                control_register: hint.control_register,
            });
        }
        let dtype_selector = ((word >> 22) & 7) as u8;
        let unsupported_special = match word & 0xfe00_0f83 {
            0x8200_0080 => Some(C220NoEffectVectorOperation::Vexp),
            0x8200_0100 => Some(C220NoEffectVectorOperation::Vrsqrt),
            0x8200_0200 => Some(C220NoEffectVectorOperation::Vrec),
            0x8200_0280 => Some(C220NoEffectVectorOperation::Vln),
            0x8200_0c80 => Some(C220NoEffectVectorOperation::Vsqrt),
            _ => None,
        };
        if let Some(operation) = unsupported_special
            && !matches!(dtype_selector, 5 | 7)
        {
            return Some(Self {
                operation,
                dtype_selector,
                control_register: ((word >> 2) & 0x1f) as u8,
            });
        }
        let operation = match word & 0xfe00_0f83 {
            0x8200_0001 => C220NoEffectVectorOperation::Vci,
            0x8200_0a80 => C220NoEffectVectorOperation::Vconcat,
            0x8200_0b01 => C220NoEffectVectorOperation::Vmergech,
            0x8200_0c01 => C220NoEffectVectorOperation::RpnCor,
            _ => match word & 0xfe00_0f82 {
                0x8200_0880 => C220NoEffectVectorOperation::Vrpac,
                0x8200_0900 => C220NoEffectVectorOperation::Vbs16,
                0x8200_0980 => C220NoEffectVectorOperation::Vms4,
                0x8200_0a00 => C220NoEffectVectorOperation::Vextract,
                0x8200_0b80 => C220NoEffectVectorOperation::RpnCorDiag,
                _ => return None,
            },
        };
        Some(Self {
            operation,
            dtype_selector,
            control_register: ((word >> 2) & 0x1f) as u8,
        })
    }

    pub const fn lane_count(self) -> usize {
        match self.dtype_selector {
            0 | 1 => 256,
            2 | 5 | 6 => 128,
            3 | 4 | 7 => 64,
            _ => unreachable!(),
        }
    }

    pub const fn lane_groups(self) -> u8 {
        (self.lane_count() / 64) as u8
    }
}
