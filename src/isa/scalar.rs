use crate::architecture::Architecture;
use crate::isa::class::AicClass;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarKey8Operation {
    AddImmediate,
    MultiplyImmediate,
    SubtractImmediate,
    DcPreload,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarKey0Operation {
    Add,
    Subtract,
    Multiply,
    MultiplyAdd,
    Divide,
    Remainder,
    Minimum,
    Maximum,
    And,
    Or,
    Xor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarKey7Operation {
    MoveImmediate,
    MoveKeep,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarLoadStoreOperation {
    Load,
    Store,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarStoreImmediateValue {
    Zero,
    One,
    Ones,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScalarAddressEffect {
    pub effective_address: u64,
    pub updated_base: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScalarDirectBiuRoute {
    pub effective_address: u64,
    pub bit_24_set: bool,
    pub outside_local_window: bool,
}

impl ScalarAddressEffect {
    pub const fn direct_biu_route(self, local_window_base: u64) -> Option<ScalarDirectBiuRoute> {
        const WINDOW_MASK: u64 = 0x1fffffe000000;
        let bit_24_set = self.effective_address & 0x1000000 != 0;
        let outside_local_window =
            (self.effective_address & WINDOW_MASK) != (local_window_base & WINDOW_MASK);
        if bit_24_set || outside_local_window {
            Some(ScalarDirectBiuRoute {
                effective_address: self.effective_address,
                bit_24_set,
                outside_local_window,
            })
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZeroExtendWidth {
    U8,
    U16,
    U32,
}

impl ZeroExtendWidth {
    pub const fn mask(self) -> u64 {
        match self {
            Self::U8 => 0xff,
            Self::U16 => 0xffff,
            Self::U32 => 0xffff_ffff,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarInstruction {
    ScalarIndexedLoad {
        width_bytes: u8,
        destination_register: u8,
        base_register: u8,
        offset_register: u8,
    },
    ScalarIndexedImmediateStore {
        width_bytes: u8,
        base_register: u8,
        offset_register: u8,
        value: ScalarStoreImmediateValue,
    },
    ScalarPairLoad {
        dtype_field: u8,
        width_bytes: u8,
        first_destination_register: u8,
        second_destination_register: u8,
        base_register: u8,
        signed_offset: i8,
        sign_extend: bool,
    },
    ScalarPairStore {
        dtype_field: u8,
        width_bytes: u8,
        first_source_register: u8,
        second_source_register: u8,
        base_register: u8,
        signed_offset: i8,
    },
    ScalarLoadStoreImmediate {
        operation: ScalarLoadStoreOperation,
        dtype_field: u8,
        width_bytes: u8,
        data_register: u8,
        base_register: u8,
        signed_offset: i16,
        post_index: bool,
        sign_extend: Option<bool>,
    },
    ScalarStoreImmediate {
        width_bytes: u8,
        base_register: u8,
        signed_offset: i16,
        post_index: bool,
        value: ScalarStoreImmediateValue,
    },
    ScalarKey0 {
        operation: ScalarKey0Operation,
        dtype_field: u8,
        destination_register: u8,
        first_source_register: u8,
        second_source_register: u8,
    },
    ScalarCompare {
        dtype_field: u8,
        condition_field: u8,
        first_source_register: u8,
        second_source_register: u8,
    },
    ScalarCompareRegister {
        dtype_field: u8,
        condition_field: u8,
        destination_register: u8,
        first_source_register: u8,
        second_source_register: u8,
    },
    ScalarCompareImmediate {
        condition_field: u8,
        source_register: u8,
        encoded_immediate: u16,
    },
    ScalarSelect {
        dtype_field: u8,
        destination_register: u8,
        first_source_register: u8,
        second_source_register: u8,
    },
    ScalarMoveX8Immediate {
        destination_register: u8,
        encoded_immediate: u16,
    },
    ScalarKey2MoveRegister {
        dtype_field: u8,
        destination_register: u8,
        source_register: u8,
    },
    ScalarKey2Negate {
        dtype_field: u8,
        destination_register: u8,
        source_register: u8,
    },
    ScalarKey2Absolute {
        dtype_field: u8,
        destination_register: u8,
        source_register: u8,
    },
    ScalarKey2IntegerSqrt {
        dtype_field: u8,
        destination_register: u8,
        source_register: u8,
    },
    ScalarKey2BitwiseNot {
        dtype_field: u8,
        destination_register: u8,
        source_register: u8,
    },
    ScalarKey2MoveFromSpr {
        destination_register: u8,
        encoded_source_spr: u16,
    },
    ScalarKey2MoveToSpr {
        encoded_destination_spr: u16,
        source_register: u8,
    },
    ScalarKey2ZeroExtend {
        width: ZeroExtendWidth,
        destination_register: u8,
        source_register: u8,
    },
    ScalarKey2SignExtend {
        width_bits: u8,
        destination_register: u8,
        source_register: u8,
    },
    ScalarKey2ShiftLeft {
        dtype_field: u8,
        destination_register: u8,
        count_register: Option<u8>,
        encoded_immediate: u8,
    },
    ScalarKey2ShiftRight {
        dtype_field: u8,
        destination_register: u8,
        count_register: Option<u8>,
        encoded_immediate: u8,
    },
    ScalarKey2FindFirst {
        destination_register: u8,
        source_register: u8,
        find_set: bool,
    },
    ScalarKey2Insert {
        destination_register: u8,
        source_register: u8,
        least_significant_bit: u8,
        width_bits: u8,
    },
    ScalarKey2InsertImmediate {
        destination_register: u8,
        position: u8,
        immediate: u8,
        extended: bool,
    },
    ScalarKey2BitSet {
        destination_register: u8,
        source_register: u8,
        set_bit: bool,
    },
    ScalarKey7 {
        operation: ScalarKey7Operation,
        destination_register: u8,
        encoded_immediate: u16,
        halfword_lane: Option<u8>,
    },
    ScalarKey8 {
        operation: ScalarKey8Operation,
        destination_register: Option<u8>,
        source_register: u8,
        encoded_immediate: u16,
    },
}

impl ScalarInstruction {
    pub const fn scalar_address_effect(self, base: u64) -> Option<ScalarAddressEffect> {
        let Self::ScalarLoadStoreImmediate {
            signed_offset,
            post_index,
            ..
        } = self
        else {
            return None;
        };
        let adjusted = base.wrapping_add(signed_offset as i64 as u64);
        Some(if post_index {
            ScalarAddressEffect {
                effective_address: base,
                updated_base: Some(adjusted),
            }
        } else {
            ScalarAddressEffect {
                effective_address: adjusted,
                updated_base: None,
            }
        })
    }

    pub const fn from_word(architecture: Architecture, word: u32) -> Option<Self> {
        match AicClass::from_word(word) {
            AicClass::Scalar
                if matches!(architecture, Architecture::Dav3510)
                    && ((word >> 24) & 0x1f) == 1
                    && word & 0x7f == 0 =>
            {
                Some(Self::ScalarIndexedLoad {
                    width_bytes: 1 << ((word >> 22) & 3),
                    destination_register: ((word >> 17) & 0x1f) as u8,
                    base_register: ((word >> 12) & 0x1f) as u8,
                    offset_register: ((word >> 7) & 0x1f) as u8,
                })
            }
            AicClass::Scalar
                if matches!(
                    (architecture, word),
                    (Architecture::Dav2201, 0x011c_b600)
                        | (Architecture::Dav2201, 0x0118_9500)
                        | (Architecture::Dav2201, 0x0120_8780)
                        | (Architecture::Dav2201, 0x0122_d700)
                        | (Architecture::Dav2201, 0x0124_a880)
                        | (Architecture::Dav2201, 0x0126_f800)
                ) =>
            {
                Some(Self::ScalarIndexedLoad {
                    width_bytes: 1,
                    destination_register: ((word >> 17) & 0x1f) as u8,
                    base_register: ((word >> 12) & 0x1f) as u8,
                    offset_register: ((word >> 7) & 0x1f) as u8,
                })
            }
            AicClass::Scalar
                if matches!(architecture, Architecture::Dav3510)
                    && ((word >> 24) & 0x1f) == 14
                    && word & 0x7c == 0
                    && word & 3 != 3 =>
            {
                Some(Self::ScalarIndexedImmediateStore {
                    width_bytes: 1 << ((word >> 22) & 3),
                    base_register: ((word >> 12) & 0x1f) as u8,
                    offset_register: ((word >> 7) & 0x1f) as u8,
                    value: match word & 3 {
                        0 => ScalarStoreImmediateValue::Zero,
                        1 => ScalarStoreImmediateValue::One,
                        _ => ScalarStoreImmediateValue::Ones,
                    },
                })
            }
            AicClass::Scalar
                if matches!(
                    (architecture, word),
                    (Architecture::Dav2201, 0x0e00_b601)
                        | (Architecture::Dav2201, 0x0e00_9501)
                        | (Architecture::Dav2201, 0x0e00_8781)
                        | (Architecture::Dav2201, 0x0e00_d701)
                        | (Architecture::Dav2201, 0x0e00_a881)
                        | (Architecture::Dav2201, 0x0e00_f801)
                ) =>
            {
                Some(Self::ScalarIndexedImmediateStore {
                    width_bytes: 1,
                    base_register: ((word >> 12) & 0x1f) as u8,
                    offset_register: ((word >> 7) & 0x1f) as u8,
                    value: ScalarStoreImmediateValue::One,
                })
            }
            AicClass::Scalar
                if (matches!(architecture, Architecture::Dav2201)
                    && ((word >> 24) & 0x1f) == 9
                    || matches!(architecture, Architecture::Dav3510)
                        && matches!((word >> 24) & 0x1f, 9 | 12 | 13)) =>
            {
                let raw_offset = ((word >> 1) & 0x3f) as i8;
                let signed_offset = if raw_offset & 0x20 != 0 {
                    raw_offset - 64
                } else {
                    raw_offset
                };
                let dtype_field = ((word >> 22) & 3) as u8;
                let width_bytes = 1 << ((word >> 22) & 3);
                let first_register = ((word >> 17) & 0x1f) as u8;
                let second_register = ((word >> 7) & 0x1f) as u8;
                let base_register = ((word >> 12) & 0x1f) as u8;
                if word & 1 == 0 {
                    Some(Self::ScalarPairLoad {
                        dtype_field,
                        width_bytes,
                        first_destination_register: first_register,
                        second_destination_register: second_register,
                        base_register,
                        signed_offset,
                        sign_extend: matches!(architecture, Architecture::Dav3510)
                            && word & (1 << 24) != 0,
                    })
                } else {
                    Some(Self::ScalarPairStore {
                        dtype_field,
                        width_bytes,
                        first_source_register: first_register,
                        second_source_register: second_register,
                        base_register,
                        signed_offset,
                    })
                }
            }
            AicClass::Scalar
                if matches!(architecture, Architecture::Dav2201)
                    && matches!((word >> 24) & 0x1f, 3 | 19 | 4 | 20) =>
            {
                let operation = if matches!((word >> 24) & 0x1f, 3 | 19) {
                    ScalarLoadStoreOperation::Load
                } else {
                    ScalarLoadStoreOperation::Store
                };
                Some(scalar_load_store_hint(
                    word,
                    operation,
                    word & (1 << 28) != 0,
                    None,
                ))
            }
            AicClass::Scalar if ((word >> 24) & 0x1f) == 15 => {
                let value = match word & 3 {
                    0 => ScalarStoreImmediateValue::Zero,
                    1 => ScalarStoreImmediateValue::One,
                    2 => ScalarStoreImmediateValue::Ones,
                    _ => return None,
                };
                let raw_offset = (((word >> 10) & 0xf80) | ((word >> 5) & 0x7f)) as i16;
                Some(Self::ScalarStoreImmediate {
                    width_bytes: 1 << ((word >> 22) & 3),
                    base_register: ((word >> 12) & 0x1f) as u8,
                    signed_offset: if raw_offset & 0x800 != 0 {
                        raw_offset - 4096
                    } else {
                        raw_offset
                    },
                    post_index: word & 4 != 0,
                    value,
                })
            }
            AicClass::Scalar
                if matches!(architecture, Architecture::Dav3510)
                    && matches!((word >> 24) & 0x1f, 28 | 31 | 3 | 19) =>
            {
                let operation = if matches!((word >> 24) & 0x1f, 28 | 31) {
                    ScalarLoadStoreOperation::Load
                } else {
                    ScalarLoadStoreOperation::Store
                };
                Some(scalar_load_store_hint(
                    word,
                    operation,
                    if matches!(operation, ScalarLoadStoreOperation::Load) {
                        word & (1 << 25) != 0
                    } else {
                        word & (1 << 28) != 0
                    },
                    if matches!(operation, ScalarLoadStoreOperation::Load) {
                        Some(word & (1 << 24) != 0)
                    } else {
                        None
                    },
                ))
            }
            AicClass::Scalar if ((word >> 24) & 0x1f) == 16 => Some(Self::ScalarMoveX8Immediate {
                destination_register: ((word >> 17) & 0x1f) as u8,
                encoded_immediate: word as u16,
            }),
            AicClass::Scalar if ((word >> 24) & 0x1f) == 10 => Some(Self::ScalarCompareImmediate {
                condition_field: ((word >> 22) & 7) as u8,
                source_register: ((word >> 12) & 0x1f) as u8,
                encoded_immediate: (word & 0xfff) as u16,
            }),
            AicClass::Scalar if ((word >> 24) & 0x1f) == 0 && (word & 0xf) == 9 => {
                let mut destination_register = ((word >> 17) & 0x1f) as u8;
                let mut first_source_register = ((word >> 12) & 0x1f) as u8;
                let mut second_source_register = ((word >> 7) & 0x1f) as u8;
                if matches!(architecture, Architecture::Dav2201) {
                    destination_register |= ((word >> 1) & 0x20) as u8;
                    first_source_register |= (word & 0x20) as u8;
                    second_source_register |= ((word << 1) & 0x20) as u8;
                }
                Some(Self::ScalarSelect {
                    dtype_field: ((word >> 22) & 3) as u8,
                    destination_register,
                    first_source_register,
                    second_source_register,
                })
            }
            AicClass::Scalar if ((word >> 24) & 0x1f) == 0 && (word & 0xf) == 14 => {
                Some(Self::ScalarCompare {
                    dtype_field: ((word >> 22) & 3) as u8,
                    condition_field: ((word >> 4) & 7) as u8,
                    first_source_register: ((word >> 12) & 0x1f) as u8,
                    second_source_register: ((word >> 7) & 0x1f) as u8,
                })
            }
            AicClass::Scalar if ((word >> 24) & 0x1f) == 0 && (word & 0xf) == 15 => {
                Some(Self::ScalarCompareRegister {
                    dtype_field: ((word >> 22) & 3) as u8,
                    condition_field: ((word >> 4) & 7) as u8,
                    destination_register: ((word >> 17) & 0x1f) as u8,
                    first_source_register: ((word >> 12) & 0x1f) as u8,
                    second_source_register: ((word >> 7) & 0x1f) as u8,
                })
            }
            AicClass::Scalar if ((word >> 24) & 0x1f) == 0 => {
                let operation = match word & 0xf {
                    1 => ScalarKey0Operation::Add,
                    2 => ScalarKey0Operation::Subtract,
                    3 => ScalarKey0Operation::Multiply,
                    4 => ScalarKey0Operation::MultiplyAdd,
                    5 => ScalarKey0Operation::Divide,
                    6 => ScalarKey0Operation::Remainder,
                    7 => ScalarKey0Operation::Maximum,
                    8 => ScalarKey0Operation::Minimum,
                    10 => ScalarKey0Operation::And,
                    11 => ScalarKey0Operation::Or,
                    12 => ScalarKey0Operation::Xor,
                    _ => return None,
                };
                let mut destination_register = ((word >> 17) & 0x1f) as u8;
                let mut first_source_register = ((word >> 12) & 0x1f) as u8;
                let mut second_source_register = ((word >> 7) & 0x1f) as u8;
                if matches!(architecture, Architecture::Dav2201) {
                    destination_register |= ((word >> 1) & 0x20) as u8;
                    first_source_register |= (word & 0x20) as u8;
                    second_source_register |= ((word << 1) & 0x20) as u8;
                }
                Some(Self::ScalarKey0 {
                    operation,
                    dtype_field: ((word >> 22) & 3) as u8,
                    destination_register,
                    first_source_register,
                    second_source_register,
                })
            }
            AicClass::Scalar if ((word >> 24) & 0x1f) == 2 && ((word >> 7) & 0x1f) == 0 => {
                let (destination_register, source_register) =
                    scalar_key2_registers(architecture, word);
                Some(Self::ScalarKey2IntegerSqrt {
                    dtype_field: ((word >> 22) & 3) as u8,
                    destination_register,
                    source_register,
                })
            }
            AicClass::Scalar if ((word >> 24) & 0x1f) == 2 && ((word >> 7) & 0x1f) == 1 => {
                let (destination_register, source_register) =
                    scalar_key2_registers(architecture, word);
                Some(Self::ScalarKey2Negate {
                    dtype_field: ((word >> 22) & 3) as u8,
                    destination_register,
                    source_register,
                })
            }
            AicClass::Scalar if ((word >> 24) & 0x1f) == 2 && ((word >> 7) & 0x1f) == 2 => {
                let (destination_register, source_register) =
                    scalar_key2_registers(architecture, word);
                Some(Self::ScalarKey2Absolute {
                    dtype_field: ((word >> 22) & 3) as u8,
                    destination_register,
                    source_register,
                })
            }
            AicClass::Scalar if ((word >> 24) & 0x1f) == 2 && ((word >> 7) & 0x1f) == 3 => {
                let (destination_register, source_register) =
                    scalar_key2_registers(architecture, word);
                Some(Self::ScalarKey2BitwiseNot {
                    dtype_field: ((word >> 22) & 3) as u8,
                    destination_register,
                    source_register,
                })
            }
            AicClass::Scalar if ((word >> 24) & 0x1f) == 2 && ((word >> 7) & 0x1f) == 4 => {
                Some(Self::ScalarKey2ShiftLeft {
                    dtype_field: ((word >> 22) & 3) as u8,
                    destination_register: ((word >> 17) & 0x1f) as u8,
                    count_register: if word & 0x40 != 0 {
                        Some(((word >> 12) & 0x1f) as u8)
                    } else {
                        None
                    },
                    encoded_immediate: (word & 0x3f) as u8,
                })
            }
            AicClass::Scalar if ((word >> 24) & 0x1f) == 2 && ((word >> 7) & 0x1f) == 5 => {
                Some(Self::ScalarKey2ShiftRight {
                    dtype_field: ((word >> 22) & 3) as u8,
                    destination_register: ((word >> 17) & 0x1f) as u8,
                    count_register: if word & 0x40 != 0 {
                        Some(((word >> 12) & 0x1f) as u8)
                    } else {
                        None
                    },
                    encoded_immediate: (word & 0x3f) as u8,
                })
            }
            AicClass::Scalar
                if (matches!(architecture, Architecture::Dav3510)
                    || matches!(
                        (architecture, word),
                        (Architecture::Dav2201, 0x02de_d380)
                            | (Architecture::Dav2201, 0x02d6_9380)
                            | (Architecture::Dav2201, 0x02d8_b380)
                            | (Architecture::Dav2201, 0x02da_9380)
                            | (Architecture::Dav2201, 0x02d6_8380)
                            | (Architecture::Dav2201, 0x02de_b380)
                            | (Architecture::Dav2201, 0x02da_a380)
                            | (Architecture::Dav2201, 0x02da_b380)
                            | (Architecture::Dav2201, 0x02dc_d380)
                            | (Architecture::Dav2201, 0x02e0_f380)
                    ))
                    && ((word >> 24) & 0x1f) == 2
                    && ((word >> 7) & 0x1f) == 7 =>
            {
                Some(Self::ScalarKey2FindFirst {
                    destination_register: ((word >> 17) & 0x1f) as u8,
                    source_register: ((word >> 12) & 0x1f) as u8,
                    find_set: word & 0x40 != 0,
                })
            }
            AicClass::Scalar if ((word >> 24) & 0x1f) == 2 && ((word >> 7) & 0x1f) == 8 => {
                Some(Self::ScalarKey2BitSet {
                    destination_register: ((word >> 17) & 0x1f) as u8,
                    source_register: ((word >> 12) & 0x1f) as u8,
                    set_bit: word & 0x40 != 0,
                })
            }
            AicClass::Scalar if ((word >> 24) & 0x1f) == 2 && ((word >> 7) & 0x1f) == 16 => {
                let (destination_register, source_register) =
                    scalar_key2_registers(architecture, word);
                Some(Self::ScalarKey2MoveRegister {
                    dtype_field: ((word >> 22) & 3) as u8,
                    destination_register,
                    source_register,
                })
            }
            AicClass::Scalar if ((word >> 24) & 0x1f) == 2 && ((word >> 7) & 0x1f) == 17 => {
                let high_bit = if matches!(architecture, Architecture::Dav3510) {
                    (word & 1) << 7
                } else {
                    0
                };
                Some(Self::ScalarKey2MoveFromSpr {
                    destination_register: ((word >> 17) & 0x1f) as u8,
                    encoded_source_spr: (high_bit | ((word >> 17) & 0x60) | ((word >> 12) & 0x1f))
                        as u16,
                })
            }
            AicClass::Scalar if ((word >> 24) & 0x1f) == 2 && ((word >> 7) & 0x1f) == 18 => {
                let encoded_destination_spr = if matches!(architecture, Architecture::Dav2201) {
                    ((word >> 17) & 0x7f) as u16
                } else {
                    (((word & 1) << 7) | ((word >> 17) & 0x7f)) as u16
                };
                let mut source_register = ((word >> 12) & 0x1f) as u8;
                if matches!(architecture, Architecture::Dav2201) {
                    source_register |= (word & 0x20) as u8;
                }
                Some(Self::ScalarKey2MoveToSpr {
                    encoded_destination_spr,
                    source_register,
                })
            }
            AicClass::Scalar if ((word >> 24) & 0x1f) == 2 && ((word >> 7) & 0x1f) == 19 => {
                let width_bits = match (word >> 22) & 3 {
                    0 => 8,
                    1 => 16,
                    2 => 32,
                    _ => return None,
                };
                let (destination_register, source_register) =
                    scalar_key2_registers(architecture, word);
                Some(Self::ScalarKey2SignExtend {
                    width_bits,
                    destination_register,
                    source_register,
                })
            }
            AicClass::Scalar if ((word >> 24) & 0x1f) == 2 && ((word >> 7) & 0x1f) == 20 => {
                let width = match (word >> 22) & 3 {
                    0 => ZeroExtendWidth::U8,
                    1 => ZeroExtendWidth::U16,
                    2 => ZeroExtendWidth::U32,
                    _ => return None,
                };
                let (destination_register, source_register) =
                    scalar_key2_registers(architecture, word);
                Some(Self::ScalarKey2ZeroExtend {
                    width,
                    destination_register,
                    source_register,
                })
            }
            AicClass::Scalar
                if ((word >> 24) & 0x1f) == 2 && matches!((word >> 7) & 0x1f, 22 | 23) =>
            {
                Some(Self::ScalarKey2InsertImmediate {
                    destination_register: ((word >> 17) & 0x1f) as u8,
                    position: (((word >> 17) & 0x20) | ((word >> 12) & 0x1f)) as u8,
                    immediate: word as u8,
                    extended: word & 0x80_0000 != 0,
                })
            }
            AicClass::Scalar
                if ((word >> 24) & 0x1f) == 2 && matches!((word >> 7) & 0x1f, 24..=27) =>
            {
                Some(Self::ScalarKey2Insert {
                    destination_register: ((word >> 17) & 0x1f) as u8,
                    source_register: ((word >> 12) & 0x1f) as u8,
                    least_significant_bit: (((word >> 18) & 0x30) | ((word >> 5) & 0xf)) as u8,
                    width_bits: ((word & 0x1f) + 1) as u8,
                })
            }
            AicClass::Scalar if ((word >> 24) & 0x1f) == 7 => {
                let move_keep = word & (1 << 16) != 0;
                Some(Self::ScalarKey7 {
                    operation: if move_keep {
                        ScalarKey7Operation::MoveKeep
                    } else {
                        ScalarKey7Operation::MoveImmediate
                    },
                    destination_register: ((word >> 17) & 0x1f) as u8,
                    encoded_immediate: (word & 0xffff) as u16,
                    halfword_lane: if move_keep {
                        Some(((word >> 22) & 3) as u8)
                    } else {
                        None
                    },
                })
            }
            AicClass::Scalar if ((word >> 24) & 0x1f) == 8 => {
                let variant = ((word >> 22) & 3) as u8;
                let operation = match variant {
                    0 => ScalarKey8Operation::AddImmediate,
                    1 => ScalarKey8Operation::MultiplyImmediate,
                    2 => ScalarKey8Operation::SubtractImmediate,
                    _ => ScalarKey8Operation::DcPreload,
                };
                let destination_register = if variant == 3 {
                    None
                } else {
                    Some(((word >> 17) & 0x1f) as u8)
                };
                let source_mask = if variant == 3 { 0x3f } else { 0x1f };
                Some(Self::ScalarKey8 {
                    operation,
                    destination_register,
                    source_register: ((word >> 12) & source_mask) as u8,
                    encoded_immediate: (word & 0xfff) as u16,
                })
            }
            _ => None,
        }
    }
}

const fn scalar_load_store_hint(
    word: u32,
    operation: ScalarLoadStoreOperation,
    post_index: bool,
    sign_extend: Option<bool>,
) -> ScalarInstruction {
    let raw_offset = (word & 0xfff) as i16;
    ScalarInstruction::ScalarLoadStoreImmediate {
        operation,
        dtype_field: ((word >> 22) & 3) as u8,
        width_bytes: 1 << ((word >> 22) & 3),
        data_register: ((word >> 17) & 0x1f) as u8,
        base_register: ((word >> 12) & 0x1f) as u8,
        signed_offset: if raw_offset & 0x800 != 0 {
            raw_offset - 4096
        } else {
            raw_offset
        },
        post_index,
        sign_extend,
    }
}

const fn scalar_key2_registers(architecture: Architecture, word: u32) -> (u8, u8) {
    let mut destination_register = ((word >> 17) & 0x1f) as u8;
    let mut source_register = ((word >> 12) & 0x1f) as u8;
    if matches!(architecture, Architecture::Dav2201) {
        destination_register |= ((word >> 1) & 0x20) as u8;
        source_register |= (word & 0x20) as u8;
    }
    (destination_register, source_register)
}

#[cfg(test)]
mod tests;
