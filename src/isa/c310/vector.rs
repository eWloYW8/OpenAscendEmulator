pub const C310_CAPTURED_VLD_V0_WORD: u32 = 0x0018_0008;
pub const C310_CAPTURED_VLD_V1_WORD: u32 = 0x0220_0008;
pub const C310_CAPTURED_VLDI_V0_WORD: u32 = 0x0008_0018;
pub const C310_CAPTURED_VLDI_V1_WORD: u32 = 0x0210_0018;
pub const C310_CAPTURED_PSET_WORD: u32 = 0x8204_0155;
pub const C310_CAPTURED_VDUPS_WORD: u32 = 0x801a_2550;
pub const C310_CAPTURED_VST_WORD: u32 = 0x4020_0108;
pub const C310_CAPTURED_SUB_VST_WORD: u32 = 0x4028_0108;
pub const C310_CAPTURED_PLT32_WORD: u32 = 0xa22c_0150;
pub const C310_CAPTURED_SMOVI32_WORD: u32 = 0xc200_410d;
pub const C310_CAPTURED_VECTOR_LOAD_BYTES: usize = 256;
pub const C310_MASK0_SPR_INDEX: u16 = 152;
pub const C310_MASK1_SPR_INDEX: u16 = 153;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C310CapturedVectorLoadHint {
    pub destination_v_register: u8,
    pub source_s_register: u8,
    pub source_a_register: Option<u8>,
    pub byte_count: usize,
}

impl C310CapturedVectorLoadHint {
    pub const fn from_word(word: u32) -> Option<Self> {
        let (destination_v_register, source_a_register) = match word {
            C310_CAPTURED_VLD_V0_WORD => (0, Some(0)),
            C310_CAPTURED_VLD_V1_WORD => (1, Some(0)),
            C310_CAPTURED_VLDI_V0_WORD => (0, None),
            C310_CAPTURED_VLDI_V1_WORD => (1, None),
            _ => return None,
        };
        Some(Self {
            destination_v_register,
            source_s_register: ((word >> 18) & 0x0f) as u8,
            source_a_register,
            byte_count: C310_CAPTURED_VECTOR_LOAD_BYTES,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C310ObservedMovemaskHint {
    pub source_x_register: u8,
    pub destination_spr: u16,
}

impl C310ObservedMovemaskHint {
    pub const fn from_word(word: u32) -> Option<Self> {
        if word & !0x001f_0020 != 0x15c0_0013 {
            return None;
        }
        let source_x_register = ((word >> 16) & 0x1f) as u8;
        let destination_spr = if word & 0x20 == 0 {
            C310_MASK0_SPR_INDEX
        } else {
            C310_MASK1_SPR_INDEX
        };
        Some(Self {
            source_x_register,
            destination_spr,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C310RvecMovpHint {
    pub destination_p_register: u8,
    pub btype: u8,
}

impl C310RvecMovpHint {
    pub const fn from_word(word: u32) -> Option<Self> {
        if word >> 30 != 2
            || (word >> 20) & 0x1f != 0
            || (word >> 9) & 0x7ff != 0x200
            || word & 0x7f != 0x56
        {
            return None;
        }
        Some(Self {
            destination_p_register: ((word >> 25) & 0x1f) as u8,
            btype: ((word >> 7) & 3) as u8,
        })
    }

    pub const fn has_u32_value_path(self) -> bool {
        self.btype == 2
    }

    pub const fn u32_source_spr_index(self) -> Option<u16> {
        if self.has_u32_value_path() {
            Some(C310_MASK0_SPR_INDEX)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C310RvecVstiHint {
    pub source_v_register: u8,
    pub scalar_register: u8,
    pub offset: u8,
    pub predicate_register: u8,
    pub p: bool,
    pub distance: u8,
}

impl C310RvecVstiHint {
    pub const fn from_word(word: u32) -> Option<Self> {
        if word >> 30 != 1 || (word >> 6) & 1 != 0 || word & 3 != 2 {
            return None;
        }
        Some(Self {
            source_v_register: ((word >> 25) & 0x1f) as u8,
            scalar_register: ((word >> 19) & 0x3f) as u8,
            offset: ((word >> 11) & 0xff) as u8,
            predicate_register: ((word >> 8) & 7) as u8,
            p: (word >> 7) & 1 != 0,
            distance: ((word >> 2) & 0xf) as u8,
        })
    }

    pub const fn dtype_code(self) -> u8 {
        self.distance + 24
    }

    pub const fn has_normal_u32_store_path(self) -> bool {
        self.dtype_code() == 26
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C310RvecArithmeticOperation {
    Add,
    Subtract,
    Multiply,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C310RvecArithmeticHint {
    pub operation: C310RvecArithmeticOperation,
    pub destination_v_register: u8,
    pub first_source_v_register: u8,
    pub second_source_v_register: u8,
    pub predicate_register: u8,
    pub dtype_selector: u8,
}

impl C310RvecArithmeticHint {
    pub const fn from_word(word: u32) -> Option<Self> {
        if (word >> 30) != 2 {
            return None;
        }
        let operation = match (((word >> 19) & 1), ((word >> 6) & 1), word & 0x3f) {
            (1, 0, 0) => C310RvecArithmeticOperation::Add,
            (1, 0, 1) => C310RvecArithmeticOperation::Subtract,
            (0, 1, 0) => C310RvecArithmeticOperation::Multiply,
            _ => return None,
        };
        Some(Self {
            operation,
            destination_v_register: ((word >> 25) & 0x1f) as u8,
            first_source_v_register: ((word >> 20) & 0x1f) as u8,
            second_source_v_register: ((word >> 13) & 0x1f) as u8,
            predicate_register: ((word >> 10) & 7) as u8,
            dtype_selector: ((((word >> 18) & 1) << 3) | ((word >> 7) & 7)) as u8,
        })
    }

    pub const fn has_fp32_value_path(self) -> bool {
        self.dtype_selector == 7
    }
}
