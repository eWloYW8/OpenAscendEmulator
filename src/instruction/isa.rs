use crate::device::architecture::Architecture;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AicClass {
    Scalar,
    FlowControl,
    MemoryTransfer,
    Vector,
    Fixp,
    Cube,
    Unregistered(u8),
}

impl AicClass {
    pub const fn from_word(word: u32) -> Self {
        match (word >> 29) as u8 {
            0 => Self::Scalar,
            2 => Self::FlowControl,
            3 => Self::MemoryTransfer,
            4 => Self::Vector,
            6 => Self::Fixp,
            7 => Self::Cube,
            other => Self::Unregistered(other),
        }
    }
}

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
pub enum AicDecoderHint {
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
    C310MovAlignV2 {
        direction_field: u8,
        dtype_field: u8,
        source_memory_class: u8,
        destination_memory_class: u8,
    },
}

impl AicDecoderHint {
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
            AicClass::MemoryTransfer
                if matches!(architecture, Architecture::Dav3510)
                    && ((word >> 27) & 3) == 2
                    && ((word >> 24) & 7) == 4 =>
            {
                let direction_field = ((word >> 22) & 3) as u8;
                let (source_memory_class, destination_memory_class) = match direction_field {
                    0 => (10, 8),
                    1 => (8, 10),
                    2 => (10, 9),
                    _ => (9, 10),
                };
                Some(Self::C310MovAlignV2 {
                    direction_field,
                    dtype_field: (word & 3) as u8,
                    source_memory_class,
                    destination_memory_class,
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
) -> AicDecoderHint {
    let raw_offset = (word & 0xfff) as i16;
    AicDecoderHint::ScalarLoadStoreImmediate {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AicInstructionWord {
    Ordinary {
        pc: u64,
        word: u32,
        class: AicClass,
    },
    C310PushPb {
        pc: u64,
        word: u32,
    },
    C310PushqShort {
        pc: u64,
        word: u32,
        subcode: u8,
    },
    C310PushqLong {
        pc: u64,
        first_word: u32,
        second_word: u32,
        subcode: u8,
    },
}

impl AicInstructionWord {
    pub const fn raw_bits(self) -> u64 {
        match self {
            Self::Ordinary { word, .. }
            | Self::C310PushPb { word, .. }
            | Self::C310PushqShort { word, .. } => word as u64,
            Self::C310PushqLong {
                first_word,
                second_word,
                ..
            } => (first_word as u64) | ((second_word as u64) << 32),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PendingLongword {
    pc: u64,
    first_word: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AicWordFramer {
    architecture: Architecture,
    pending: Option<PendingLongword>,
}

impl AicWordFramer {
    pub const fn new(architecture: Architecture) -> Self {
        Self {
            architecture,
            pending: None,
        }
    }

    pub fn push(
        &mut self,
        pc: u64,
        word: u32,
    ) -> Result<Option<AicInstructionWord>, AicFramingError> {
        if let Some(pending) = self.pending {
            let expected_pc = pending
                .pc
                .checked_add(4)
                .ok_or(AicFramingError::ProgramCounterOverflow(pending.pc))?;
            if pc != expected_pc {
                return Err(AicFramingError::NonAdjacentLongword {
                    expected_pc,
                    actual_pc: pc,
                });
            }
            let expected_high_byte = (pending.first_word >> 24) as u8;
            let actual_high_byte = (word >> 24) as u8;
            if actual_high_byte != expected_high_byte {
                return Err(AicFramingError::MismatchedLongwordHighByte {
                    expected: expected_high_byte,
                    actual: actual_high_byte,
                });
            }
            self.pending = None;
            return Ok(Some(AicInstructionWord::C310PushqLong {
                pc: pending.pc,
                first_word: pending.first_word,
                second_word: word,
                subcode: (word & 0x1f) as u8,
            }));
        }

        let class = AicClass::from_word(word);
        if self.architecture == Architecture::Dav3510 {
            let high_subcode = ((word >> 24) & 0x1f) as u8;
            if class == AicClass::Scalar && high_subcode == 21 {
                if word & (1 << 23) == 0 {
                    self.pending = Some(PendingLongword {
                        pc,
                        first_word: word,
                    });
                    return Ok(None);
                }
                return Ok(Some(AicInstructionWord::C310PushqShort {
                    pc,
                    word,
                    subcode: (word & 0x1f) as u8,
                }));
            }
            if class == AicClass::FlowControl && high_subcode == 3 {
                return Ok(Some(AicInstructionWord::C310PushPb { pc, word }));
            }
        }

        Ok(Some(AicInstructionWord::Ordinary { pc, word, class }))
    }

    pub fn push_le_bytes(
        &mut self,
        pc: u64,
        bytes: [u8; 4],
    ) -> Result<Option<AicInstructionWord>, AicFramingError> {
        self.push(pc, u32::from_le_bytes(bytes))
    }

    pub fn finish(&self) -> Result<(), AicFramingError> {
        match self.pending {
            Some(pending) => Err(AicFramingError::IncompleteLongword { pc: pending.pc }),
            None => Ok(()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum AicFramingError {
    #[error("C310 longword at PC {0:#x} has no representable next PC")]
    ProgramCounterOverflow(u64),
    #[error("C310 longword suffix PC {actual_pc:#x} does not follow {expected_pc:#x}")]
    NonAdjacentLongword { expected_pc: u64, actual_pc: u64 },
    #[error("C310 longword suffix high byte {actual:#x} does not match {expected:#x}")]
    MismatchedLongwordHighByte { expected: u8, actual: u8 },
    #[error("C310 longword beginning at PC {pc:#x} has no suffix")]
    IncompleteLongword { pc: u64 },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn movx8_immediate_fields_match_both_architectures() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            assert_eq!(
                AicDecoderHint::from_word(architecture, 0x1004_2002),
                Some(AicDecoderHint::ScalarMoveX8Immediate {
                    destination_register: 2,
                    encoded_immediate: 0x2002,
                })
            );
            assert!(!matches!(
                AicDecoderHint::from_word(architecture, 0x1204_2002),
                Some(AicDecoderHint::ScalarMoveX8Immediate { .. })
            ));
        }
    }

    #[test]
    fn ordinary_dispatch_classes_match_both_pem_implementations() {
        let expected = [
            AicClass::Scalar,
            AicClass::Unregistered(1),
            AicClass::FlowControl,
            AicClass::MemoryTransfer,
            AicClass::Vector,
            AicClass::Unregistered(5),
            AicClass::Fixp,
            AicClass::Cube,
        ];
        for (key, class) in expected.into_iter().enumerate() {
            assert_eq!(AicClass::from_word((key as u32) << 29), class);
        }
    }

    #[test]
    fn scalar_immediate_memory_routes_and_fields_are_architecture_specific() {
        use ScalarLoadStoreOperation::{Load, Store};

        assert_eq!(
            AicDecoderHint::from_word(Architecture::Dav2201, 0x03ce_5db8),
            Some(AicDecoderHint::ScalarLoadStoreImmediate {
                operation: Load,
                dtype_field: 3,
                width_bytes: 8,
                data_register: 7,
                base_register: 5,
                signed_offset: -584,
                post_index: false,
                sign_extend: None,
            })
        );
        assert_eq!(
            AicDecoderHint::from_word(Architecture::Dav2201, 0x13ce_5008),
            Some(AicDecoderHint::ScalarLoadStoreImmediate {
                operation: Load,
                dtype_field: 3,
                width_bytes: 8,
                data_register: 7,
                base_register: 5,
                signed_offset: 8,
                post_index: true,
                sign_extend: None,
            })
        );
        assert_eq!(
            AicDecoderHint::from_word(Architecture::Dav2201, 0x04c6_6000),
            Some(AicDecoderHint::ScalarLoadStoreImmediate {
                operation: Store,
                dtype_field: 3,
                width_bytes: 8,
                data_register: 3,
                base_register: 6,
                signed_offset: 0,
                post_index: false,
                sign_extend: None,
            })
        );
        assert_eq!(
            AicDecoderHint::from_word(Architecture::Dav3510, 0x1cce_5db8),
            Some(AicDecoderHint::ScalarLoadStoreImmediate {
                operation: Load,
                dtype_field: 3,
                width_bytes: 8,
                data_register: 7,
                base_register: 5,
                signed_offset: -584,
                post_index: false,
                sign_extend: Some(false),
            })
        );
        assert_eq!(
            AicDecoderHint::from_word(Architecture::Dav3510, 0x1fce_5db8),
            Some(AicDecoderHint::ScalarLoadStoreImmediate {
                operation: Load,
                dtype_field: 3,
                width_bytes: 8,
                data_register: 7,
                base_register: 5,
                signed_offset: -584,
                post_index: true,
                sign_extend: Some(true),
            })
        );
        assert_eq!(
            AicDecoderHint::from_word(Architecture::Dav3510, 0x03ce_5db8),
            Some(AicDecoderHint::ScalarLoadStoreImmediate {
                operation: Store,
                dtype_field: 3,
                width_bytes: 8,
                data_register: 7,
                base_register: 5,
                signed_offset: -584,
                post_index: false,
                sign_extend: None,
            })
        );
        assert_eq!(
            AicDecoderHint::from_word(Architecture::Dav3510, 0x13ce_5db8),
            Some(AicDecoderHint::ScalarLoadStoreImmediate {
                operation: Store,
                dtype_field: 3,
                width_bytes: 8,
                data_register: 7,
                base_register: 5,
                signed_offset: -584,
                post_index: true,
                sign_extend: None,
            })
        );
        assert_eq!(
            AicDecoderHint::from_word(Architecture::Dav3510, 0x04c6_6000),
            None
        );
        assert_eq!(
            AicDecoderHint::from_word(Architecture::Dav2201, 0x1cce_5db8),
            None
        );
    }

    #[test]
    fn captured_indexed_scalar_loads_decode_on_supported_architectures() {
        for (architecture, other, word, destination, base, offset) in [
            (
                Architecture::Dav2201,
                Architecture::Dav3510,
                0x011c_b600,
                14,
                11,
                12,
            ),
            (
                Architecture::Dav3510,
                Architecture::Dav2201,
                0x0103_2980,
                1,
                18,
                19,
            ),
            (
                Architecture::Dav2201,
                Architecture::Dav3510,
                0x0118_9500,
                12,
                9,
                10,
            ),
            (
                Architecture::Dav2201,
                Architecture::Dav3510,
                0x0120_8780,
                16,
                8,
                15,
            ),
            (
                Architecture::Dav2201,
                Architecture::Dav3510,
                0x0122_d700,
                17,
                13,
                14,
            ),
            (
                Architecture::Dav2201,
                Architecture::Dav3510,
                0x0124_a880,
                18,
                10,
                17,
            ),
            (
                Architecture::Dav2201,
                Architecture::Dav3510,
                0x0126_f800,
                19,
                15,
                16,
            ),
        ] {
            assert_eq!(
                AicDecoderHint::from_word(architecture, word),
                Some(AicDecoderHint::ScalarIndexedLoad {
                    width_bytes: 1,
                    destination_register: destination,
                    base_register: base,
                    offset_register: offset,
                })
            );
            assert_eq!(
                AicDecoderHint::from_word(other, word),
                (other == Architecture::Dav3510).then_some(AicDecoderHint::ScalarIndexedLoad {
                    width_bytes: 1,
                    destination_register: destination,
                    base_register: base,
                    offset_register: offset,
                })
            );
            assert_eq!(AicDecoderHint::from_word(architecture, word | 1), None);
        }
    }

    #[test]
    fn c310_indexed_loads_decode_supported_widths_and_reject_control_bits() {
        let word = 0x0103_4b00;
        for (dtype, width_bytes) in [(0, 1), (1, 2), (2, 4), (3, 8)] {
            let encoded = word | (dtype << 22);
            assert_eq!(
                AicDecoderHint::from_word(Architecture::Dav3510, encoded),
                Some(AicDecoderHint::ScalarIndexedLoad {
                    width_bytes,
                    destination_register: 1,
                    base_register: 20,
                    offset_register: 22,
                })
            );
            assert_eq!(
                AicDecoderHint::from_word(Architecture::Dav2201, encoded),
                None
            );
        }
        for changed in [word | 1, word | 2, word | 4, word | 8, word | 0x10] {
            assert_eq!(
                AicDecoderHint::from_word(Architecture::Dav3510, changed),
                None
            );
        }
    }

    #[test]
    fn captured_indexed_immediate_stores_decode_on_supported_architectures() {
        for (architecture, other, word, base, offset) in [
            (
                Architecture::Dav2201,
                Architecture::Dav3510,
                0x0e00_d701,
                13,
                14,
            ),
            (
                Architecture::Dav2201,
                Architecture::Dav3510,
                0x0e00_8781,
                8,
                15,
            ),
            (
                Architecture::Dav2201,
                Architecture::Dav3510,
                0x0e00_9501,
                9,
                10,
            ),
            (
                Architecture::Dav2201,
                Architecture::Dav3510,
                0x0e00_b601,
                11,
                12,
            ),
            (
                Architecture::Dav2201,
                Architecture::Dav3510,
                0x0e00_a881,
                10,
                17,
            ),
            (
                Architecture::Dav2201,
                Architecture::Dav3510,
                0x0e00_f801,
                15,
                16,
            ),
            (
                Architecture::Dav3510,
                Architecture::Dav2201,
                0x0e01_2981,
                18,
                19,
            ),
        ] {
            assert_eq!(
                AicDecoderHint::from_word(architecture, word),
                Some(AicDecoderHint::ScalarIndexedImmediateStore {
                    width_bytes: 1,
                    base_register: base,
                    offset_register: offset,
                    value: ScalarStoreImmediateValue::One,
                })
            );
            assert_eq!(
                AicDecoderHint::from_word(other, word),
                (other == Architecture::Dav3510).then_some(
                    AicDecoderHint::ScalarIndexedImmediateStore {
                        width_bytes: 1,
                        base_register: base,
                        offset_register: offset,
                        value: ScalarStoreImmediateValue::One,
                    }
                )
            );
            assert_eq!(
                AicDecoderHint::from_word(architecture, word ^ 1),
                (architecture == Architecture::Dav3510).then_some(
                    AicDecoderHint::ScalarIndexedImmediateStore {
                        width_bytes: 1,
                        base_register: base,
                        offset_register: offset,
                        value: ScalarStoreImmediateValue::Zero,
                    }
                )
            );
        }
    }

    #[test]
    fn c310_indexed_immediate_stores_decode_width_and_value() {
        let word = 0x0e01_4b00;
        for (dtype, width_bytes) in [(0, 1), (1, 2), (2, 4), (3, 8)] {
            for (value_bits, value) in [
                (0, ScalarStoreImmediateValue::Zero),
                (1, ScalarStoreImmediateValue::One),
                (2, ScalarStoreImmediateValue::Ones),
            ] {
                let encoded = word | (dtype << 22) | value_bits;
                assert_eq!(
                    AicDecoderHint::from_word(Architecture::Dav3510, encoded),
                    Some(AicDecoderHint::ScalarIndexedImmediateStore {
                        width_bytes,
                        base_register: 20,
                        offset_register: 22,
                        value,
                    })
                );
                assert_eq!(
                    AicDecoderHint::from_word(Architecture::Dav2201, encoded),
                    None
                );
            }
        }
        for changed in [word | 3, word | 4, word | 8, word | 0x10] {
            assert_eq!(
                AicDecoderHint::from_word(Architecture::Dav3510, changed),
                None
            );
        }
    }

    #[test]
    fn pair_load_fields_and_opcode_groups_are_architecture_specific() {
        let c310_word = 0x0cca_0190;
        assert_eq!(
            AicDecoderHint::from_word(Architecture::Dav3510, c310_word),
            Some(AicDecoderHint::ScalarPairLoad {
                dtype_field: 3,
                width_bytes: 8,
                first_destination_register: 5,
                second_destination_register: 3,
                base_register: 0,
                signed_offset: 8,
                sign_extend: false,
            })
        );
        assert!(!matches!(
            AicDecoderHint::from_word(Architecture::Dav2201, c310_word),
            Some(AicDecoderHint::ScalarPairLoad { .. })
        ));
        let c220_word = (c310_word & !(0x1f << 24)) | (9 << 24);
        assert!(matches!(
            AicDecoderHint::from_word(Architecture::Dav2201, c220_word),
            Some(AicDecoderHint::ScalarPairLoad {
                signed_offset: 8,
                sign_extend: false,
                ..
            })
        ));
        assert!(matches!(
            AicDecoderHint::from_word(Architecture::Dav2201, 0x09c4_00b0),
            Some(AicDecoderHint::ScalarPairLoad {
                first_destination_register: 2,
                second_destination_register: 1,
                base_register: 0,
                signed_offset: 24,
                ..
            })
        ));
        assert!(matches!(
            AicDecoderHint::from_word(Architecture::Dav2201, 0x09c8_0190),
            Some(AicDecoderHint::ScalarPairLoad {
                first_destination_register: 4,
                second_destination_register: 3,
                base_register: 0,
                signed_offset: 8,
                ..
            })
        ));
        assert!(matches!(
            AicDecoderHint::from_word(Architecture::Dav3510, c310_word | (1 << 24)),
            Some(AicDecoderHint::ScalarPairLoad {
                sign_extend: true,
                ..
            })
        ));
        assert_eq!(
            AicDecoderHint::from_word(Architecture::Dav3510, c310_word | 1),
            Some(AicDecoderHint::ScalarPairStore {
                dtype_field: 3,
                width_bytes: 8,
                first_source_register: 5,
                second_source_register: 3,
                base_register: 0,
                signed_offset: 8,
            })
        );
    }

    #[test]
    fn scalar_immediate_address_effect_matches_pem_handler_branches() {
        let cases = [
            (Architecture::Dav2201, 0x03ce_5db8, 0x1c7f60, 0x1c7d18, None),
            (
                Architecture::Dav2201,
                0x13ce_5008,
                0x1c7f68,
                0x1c7f68,
                Some(0x1c7f70),
            ),
            (Architecture::Dav3510, 0x1cce_5db8, 0x1c7f60, 0x1c7d18, None),
            (
                Architecture::Dav3510,
                0x1fce_5db8,
                0x1c7f60,
                0x1c7f60,
                Some(0x1c7d18),
            ),
            (Architecture::Dav3510, 0x03ce_5db8, 0x1c7f60, 0x1c7d18, None),
            (
                Architecture::Dav3510,
                0x13ce_5db8,
                0x1c7f60,
                0x1c7f60,
                Some(0x1c7d18),
            ),
            (Architecture::Dav2201, 0x03ce_5fff, 0, u64::MAX, None),
        ];
        for (architecture, word, base, effective_address, updated_base) in cases {
            assert_eq!(
                AicDecoderHint::from_word(architecture, word)
                    .unwrap()
                    .scalar_address_effect(base),
                Some(ScalarAddressEffect {
                    effective_address,
                    updated_base,
                })
            );
        }
        assert_eq!(
            AicDecoderHint::from_word(Architecture::Dav2201, 0x073a_7f80)
                .unwrap()
                .scalar_address_effect(0x1000),
            None
        );
    }

    #[test]
    fn scalar_direct_biu_selection_depends_on_runtime_window_base() {
        for (architecture, word) in [
            (Architecture::Dav2201, 0x03ce_5db8),
            (Architecture::Dav3510, 0x1cce_5db8),
        ] {
            let effect = AicDecoderHint::from_word(architecture, word)
                .unwrap()
                .scalar_address_effect(0x1000_0248)
                .unwrap();
            assert_eq!(effect.effective_address, 0x1000_0000);
            assert_eq!(
                effect.direct_biu_route(0),
                Some(ScalarDirectBiuRoute {
                    effective_address: 0x1000_0000,
                    bit_24_set: false,
                    outside_local_window: true,
                })
            );
            assert_eq!(effect.direct_biu_route(0x1000_0000), None);
        }
        let bit24 = ScalarAddressEffect {
            effective_address: 0x0100_0000,
            updated_base: None,
        };
        assert_eq!(
            bit24.direct_biu_route(0x0100_0000),
            Some(ScalarDirectBiuRoute {
                effective_address: 0x0100_0000,
                bit_24_set: true,
                outside_local_window: false,
            })
        );
    }

    #[test]
    fn installed_c310_clear_l2_object_contains_three_matching_load_words() {
        for (word, data_register, base_register, signed_offset) in [
            (0x1cc2_0008, 1, 0, 8),
            (0x1cc2_1000, 1, 1, 0),
            (0x1cc0_0000, 0, 0, 0),
        ] {
            assert_eq!(
                AicDecoderHint::from_word(Architecture::Dav3510, word),
                Some(AicDecoderHint::ScalarLoadStoreImmediate {
                    operation: ScalarLoadStoreOperation::Load,
                    dtype_field: 3,
                    width_bytes: 8,
                    data_register,
                    base_register,
                    signed_offset,
                    post_index: false,
                    sign_extend: Some(false),
                })
            );
        }
    }

    #[test]
    fn c220_does_not_apply_c310_pushq_rules() {
        let mut framer = AicWordFramer::new(Architecture::Dav2201);
        let word = (21_u32 << 24) | 7;
        assert_eq!(
            framer.push(0x100, word),
            Ok(Some(AicInstructionWord::Ordinary {
                pc: 0x100,
                word,
                class: AicClass::Scalar,
            }))
        );
        assert_eq!(framer.finish(), Ok(()));
    }

    #[test]
    fn c310_short_pushq_and_push_pb_are_distinct() {
        let mut framer = AicWordFramer::new(Architecture::Dav3510);
        let short = (21_u32 << 24) | (1 << 23) | 7;
        let decoded = framer.push_le_bytes(0x100, short.to_le_bytes()).unwrap();
        assert_eq!(
            decoded,
            Some(AicInstructionWord::C310PushqShort {
                pc: 0x100,
                word: short,
                subcode: 7,
            })
        );
        assert_eq!(decoded.unwrap().raw_bits(), short as u64);

        let push_pb = (2_u32 << 29) | (3 << 24);
        assert_eq!(
            framer.push(0x104, push_pb),
            Ok(Some(AicInstructionWord::C310PushPb {
                pc: 0x104,
                word: push_pb,
            }))
        );
        assert_eq!(framer.finish(), Ok(()));
    }

    #[test]
    fn c310_longword_uses_next_word_as_high_half() {
        let mut framer = AicWordFramer::new(Architecture::Dav3510);
        let first = (21_u32 << 24) | 2;
        let second = (21_u32 << 24) | 5;
        assert_eq!(framer.push(0x200, first), Ok(None));
        assert_eq!(
            framer.finish(),
            Err(AicFramingError::IncompleteLongword { pc: 0x200 })
        );
        let decoded = framer.push(0x204, second).unwrap().unwrap();
        assert_eq!(
            decoded,
            AicInstructionWord::C310PushqLong {
                pc: 0x200,
                first_word: first,
                second_word: second,
                subcode: 5,
            }
        );
        assert_eq!(decoded.raw_bits(), (first as u64) | ((second as u64) << 32));
        assert_eq!(framer.finish(), Ok(()));
    }

    #[test]
    fn invalid_longword_suffix_keeps_first_word_pending() {
        let mut framer = AicWordFramer::new(Architecture::Dav3510);
        let first = 21_u32 << 24;
        assert_eq!(framer.push(0x300, first), Ok(None));
        assert_eq!(
            framer.push(0x308, first),
            Err(AicFramingError::NonAdjacentLongword {
                expected_pc: 0x304,
                actual_pc: 0x308,
            })
        );
        assert_eq!(
            framer.push(0x304, 22_u32 << 24),
            Err(AicFramingError::MismatchedLongwordHighByte {
                expected: 21,
                actual: 22,
            })
        );
        assert_eq!(
            framer.push(0x304, first),
            Ok(Some(AicInstructionWord::C310PushqLong {
                pc: 0x300,
                first_word: first,
                second_word: first,
                subcode: 0,
            }))
        );
    }

    #[test]
    fn scalar_key8_decoder_fields_match_both_pem_binaries() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let variants = [
                ScalarKey8Operation::AddImmediate,
                ScalarKey8Operation::MultiplyImmediate,
                ScalarKey8Operation::SubtractImmediate,
                ScalarKey8Operation::DcPreload,
            ];
            for (variant, operation) in variants.into_iter().enumerate() {
                let encoded_xd = if variant == 3 { 31 } else { 30 };
                let word = (8_u32 << 24)
                    | ((variant as u32) << 22)
                    | (encoded_xd << 17)
                    | (29 << 12)
                    | 0x788;
                let hint = AicDecoderHint::from_word(architecture, word);
                assert_eq!(
                    hint,
                    Some(AicDecoderHint::ScalarKey8 {
                        operation,
                        destination_register: (variant != 3).then_some(30),
                        source_register: if variant == 3 { 61 } else { 29 },
                        encoded_immediate: 0x788,
                    })
                );
            }
            assert_eq!(
                AicDecoderHint::from_word(architecture, 0x083d_d788),
                Some(AicDecoderHint::ScalarKey8 {
                    operation: ScalarKey8Operation::AddImmediate,
                    destination_register: Some(30),
                    source_register: 29,
                    encoded_immediate: 0x788,
                })
            );
            assert_eq!(
                AicDecoderHint::from_word(architecture, 0x08c0_0000),
                Some(AicDecoderHint::ScalarKey8 {
                    operation: ScalarKey8Operation::DcPreload,
                    destination_register: None,
                    source_register: 0,
                    encoded_immediate: 0,
                })
            );
        }
    }

    #[test]
    fn scalar_key0_register_fields_decode_on_both_architectures() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            for (word, operation, dtype, xd, xn, xm) in [
                (0x003b_d781, ScalarKey0Operation::Add, 0, 29, 29, 15),
                (0x0002_1102, ScalarKey0Operation::Subtract, 0, 1, 1, 2),
                (0x000e_8383, ScalarKey0Operation::Multiply, 0, 7, 8, 7),
                (0x003a_f884, ScalarKey0Operation::MultiplyAdd, 0, 29, 15, 17),
                (0x00de_f80a, ScalarKey0Operation::And, 3, 15, 15, 16),
                (0x00c0_100b, ScalarKey0Operation::Or, 3, 0, 1, 0),
            ] {
                assert_eq!(
                    AicDecoderHint::from_word(architecture, word),
                    Some(AicDecoderHint::ScalarKey0 {
                        operation,
                        dtype_field: dtype,
                        destination_register: xd,
                        first_source_register: xn,
                        second_source_register: xm,
                    })
                );
            }
            assert_eq!(AicDecoderHint::from_word(architecture, 0x003b_d780), None);
        }
        assert_eq!(
            AicDecoderHint::from_word(Architecture::Dav2201, 0x003b_d7f1),
            Some(AicDecoderHint::ScalarKey0 {
                operation: ScalarKey0Operation::Add,
                dtype_field: 0,
                destination_register: 61,
                first_source_register: 61,
                second_source_register: 47,
            })
        );
        assert_eq!(
            AicDecoderHint::from_word(Architecture::Dav3510, 0x003b_d7f1),
            Some(AicDecoderHint::ScalarKey0 {
                operation: ScalarKey0Operation::Add,
                dtype_field: 0,
                destination_register: 29,
                first_source_register: 29,
                second_source_register: 15,
            })
        );
    }

    #[test]
    fn scalar_key2_shift_left_source_selector_matches_both_pem_binaries() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            assert_eq!(
                AicDecoderHint::from_word(architecture, 0x02c8_020f),
                Some(AicDecoderHint::ScalarKey2ShiftLeft {
                    dtype_field: 3,
                    destination_register: 4,
                    count_register: None,
                    encoded_immediate: 15,
                })
            );
            assert_eq!(
                AicDecoderHint::from_word(architecture, 0x02d5_6240),
                Some(AicDecoderHint::ScalarKey2ShiftLeft {
                    dtype_field: 3,
                    destination_register: 10,
                    count_register: Some(22),
                    encoded_immediate: 0,
                })
            );
        }
    }

    #[test]
    fn scalar_negate_register_fields_match_both_architectures() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            assert_eq!(
                AicDecoderHint::from_word(architecture, 0x0202_2080),
                Some(AicDecoderHint::ScalarKey2Negate {
                    dtype_field: 0,
                    destination_register: 1,
                    source_register: 2,
                })
            );
        }
        assert_eq!(
            AicDecoderHint::from_word(Architecture::Dav2201, 0x0202_20e0),
            Some(AicDecoderHint::ScalarKey2Negate {
                dtype_field: 0,
                destination_register: 33,
                source_register: 34,
            })
        );
        assert_eq!(
            AicDecoderHint::from_word(Architecture::Dav3510, 0x0202_20e0),
            Some(AicDecoderHint::ScalarKey2Negate {
                dtype_field: 0,
                destination_register: 1,
                source_register: 2,
            })
        );
    }

    #[test]
    fn scalar_shift_right_decodes_immediate_and_register_counts() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            assert_eq!(
                AicDecoderHint::from_word(architecture, 0x0252_028f),
                Some(AicDecoderHint::ScalarKey2ShiftRight {
                    dtype_field: 1,
                    destination_register: 9,
                    count_register: None,
                    encoded_immediate: 15,
                })
            );
            assert_eq!(
                AicDecoderHint::from_word(architecture, 0x0212_62c0),
                Some(AicDecoderHint::ScalarKey2ShiftRight {
                    dtype_field: 0,
                    destination_register: 9,
                    count_register: Some(6),
                    encoded_immediate: 0,
                })
            );
        }
    }

    #[test]
    fn scalar_key2_mov_spr_xn_fields_match_both_pem_decoders() {
        for (architecture, word, spr, source) in [
            (Architecture::Dav2201, 0x0206_1900, 3, 1),
            (Architecture::Dav2201, 0x0207_3900, 3, 19),
            (Architecture::Dav2201, 0x0206_1920, 3, 33),
            (Architecture::Dav3510, 0x0206_0900, 3, 0),
            (Architecture::Dav3510, 0x02b4_0900, 90, 0),
            (Architecture::Dav3510, 0x02d2_0900, 105, 0),
            (Architecture::Dav3510, 0x02e0_0900, 112, 0),
            (Architecture::Dav3510, 0x0230_0901, 152, 0),
        ] {
            assert_eq!(
                AicDecoderHint::from_word(architecture, word),
                Some(AicDecoderHint::ScalarKey2MoveToSpr {
                    encoded_destination_spr: spr,
                    source_register: source,
                })
            );
        }
        assert_eq!(
            AicDecoderHint::from_word(Architecture::Dav2201, 0x0230_0901),
            Some(AicDecoderHint::ScalarKey2MoveToSpr {
                encoded_destination_spr: 24,
                source_register: 0,
            })
        );
    }

    #[test]
    fn scalar_key2_mov_xd_spr_uses_both_high_spr_encoding_fields() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            for (word, destination_register, encoded_source_spr) in [
                (0x029e_3880, 15, 67),
                (0x021f_0880, 15, 16),
                (0x0200_4880, 0, 4),
                (0x0280_8880, 0, 72),
            ] {
                assert_eq!(
                    AicDecoderHint::from_word(architecture, word),
                    Some(AicDecoderHint::ScalarKey2MoveFromSpr {
                        destination_register,
                        encoded_source_spr,
                    })
                );
            }
        }
        assert_eq!(
            AicDecoderHint::from_word(Architecture::Dav2201, 0x029e_3881),
            AicDecoderHint::from_word(Architecture::Dav2201, 0x029e_3880)
        );
        assert_eq!(
            AicDecoderHint::from_word(Architecture::Dav3510, 0x029e_3881),
            Some(AicDecoderHint::ScalarKey2MoveFromSpr {
                destination_register: 15,
                encoded_source_spr: 195,
            })
        );
    }

    #[test]
    fn zero_extend_fields_follow_architecture_specific_register_decoders() {
        for (dtype, width) in [
            (0_u32, ZeroExtendWidth::U8),
            (1, ZeroExtendWidth::U16),
            (2, ZeroExtendWidth::U32),
        ] {
            let word = 0x0208_3a00 | (dtype << 22);
            for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
                assert_eq!(
                    AicDecoderHint::from_word(architecture, word),
                    Some(AicDecoderHint::ScalarKey2ZeroExtend {
                        width,
                        destination_register: 4,
                        source_register: 3,
                    })
                );
            }
        }
        assert_eq!(
            AicDecoderHint::from_word(Architecture::Dav2201, 0x0208_3a60),
            Some(AicDecoderHint::ScalarKey2ZeroExtend {
                width: ZeroExtendWidth::U8,
                destination_register: 36,
                source_register: 35,
            })
        );
        assert_eq!(
            AicDecoderHint::from_word(Architecture::Dav3510, 0x0208_3a60),
            Some(AicDecoderHint::ScalarKey2ZeroExtend {
                width: ZeroExtendWidth::U8,
                destination_register: 4,
                source_register: 3,
            })
        );
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            assert_eq!(AicDecoderHint::from_word(architecture, 0x02c8_3a00), None);
            assert_eq!(
                AicDecoderHint::from_word(architecture, 0x0288_3980),
                Some(AicDecoderHint::ScalarKey2SignExtend {
                    width_bits: 32,
                    destination_register: 4,
                    source_register: 3,
                })
            );
        }
    }

    #[test]
    fn sign_extend_decodes_all_supported_source_widths_on_both_architectures() {
        for (dtype, width_bits) in [(0_u32, 8_u8), (1, 16), (2, 32)] {
            let word = 0x021c_f980 | (dtype << 22);
            for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
                assert_eq!(
                    AicDecoderHint::from_word(architecture, word),
                    Some(AicDecoderHint::ScalarKey2SignExtend {
                        width_bits,
                        destination_register: 14,
                        source_register: 15,
                    })
                );
            }
        }
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            assert_eq!(AicDecoderHint::from_word(architecture, 0x02dc_f980), None);
        }
    }

    #[test]
    fn register_move_fields_follow_architecture_specific_register_decoders() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            assert_eq!(
                AicDecoderHint::from_word(architecture, 0x0209_e800),
                Some(AicDecoderHint::ScalarKey2MoveRegister {
                    dtype_field: 0,
                    destination_register: 4,
                    source_register: 30,
                })
            );
        }
        assert_eq!(
            AicDecoderHint::from_word(Architecture::Dav2201, 0x0209_e860),
            Some(AicDecoderHint::ScalarKey2MoveRegister {
                dtype_field: 0,
                destination_register: 36,
                source_register: 62,
            })
        );
        assert_eq!(
            AicDecoderHint::from_word(Architecture::Dav3510, 0x0209_e860),
            Some(AicDecoderHint::ScalarKey2MoveRegister {
                dtype_field: 0,
                destination_register: 4,
                source_register: 30,
            })
        );
    }

    #[test]
    fn scalar_key7_decoder_fields_match_both_pem_binaries_and_vendor_words() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            assert_eq!(
                AicDecoderHint::from_word(architecture, 0x073a_7f80),
                Some(AicDecoderHint::ScalarKey7 {
                    operation: ScalarKey7Operation::MoveImmediate,
                    destination_register: 29,
                    encoded_immediate: 0x7f80,
                    halfword_lane: None,
                })
            );
            assert_eq!(
                AicDecoderHint::from_word(architecture, 0x077b_0010),
                Some(AicDecoderHint::ScalarKey7 {
                    operation: ScalarKey7Operation::MoveKeep,
                    destination_register: 29,
                    encoded_immediate: 0x10,
                    halfword_lane: Some(1),
                })
            );
            assert_eq!(
                AicDecoderHint::from_word(architecture, 0x07cd_ffff),
                Some(AicDecoderHint::ScalarKey7 {
                    operation: ScalarKey7Operation::MoveKeep,
                    destination_register: 6,
                    encoded_immediate: 0xffff,
                    halfword_lane: Some(3),
                })
            );
        }
        assert_eq!(
            AicDecoderHint::from_word(Architecture::Dav3510, 0x073a_ff20),
            Some(AicDecoderHint::ScalarKey7 {
                operation: ScalarKey7Operation::MoveImmediate,
                destination_register: 29,
                encoded_immediate: 0xff20,
                halfword_lane: None,
            })
        );
        assert_eq!(
            AicDecoderHint::from_word(Architecture::Dav3510, 0x077b_000f),
            Some(AicDecoderHint::ScalarKey7 {
                operation: ScalarKey7Operation::MoveKeep,
                destination_register: 29,
                encoded_immediate: 15,
                halfword_lane: Some(1),
            })
        );
    }

    #[test]
    fn mov_align_v2_hint_is_c310_mte_type2_only() {
        for word in [0x7402_7204, 0x7402_7284] {
            assert_eq!(
                AicDecoderHint::from_word(Architecture::Dav3510, word),
                Some(AicDecoderHint::C310MovAlignV2 {
                    direction_field: 0,
                    dtype_field: 0,
                    source_memory_class: 10,
                    destination_memory_class: 8,
                })
            );
            assert_eq!(AicDecoderHint::from_word(Architecture::Dav2201, word), None);
        }
        assert_eq!(
            AicDecoderHint::from_word(Architecture::Dav3510, 0x7002_7204),
            None
        );
        assert_eq!(
            AicDecoderHint::from_word(Architecture::Dav3510, 0x7302_7204),
            None
        );
        for (word, direction_field, dtype_field, source_memory_class, destination_memory_class) in [
            (0x74ad_8bae, 2, 2, 10, 9),
            (0x74b3_6bae, 2, 2, 10, 9),
            (0x74e1_192c, 3, 0, 9, 10),
        ] {
            assert_eq!(
                AicDecoderHint::from_word(Architecture::Dav3510, word),
                Some(AicDecoderHint::C310MovAlignV2 {
                    direction_field,
                    dtype_field,
                    source_memory_class,
                    destination_memory_class,
                })
            );
        }
    }

    #[test]
    fn insert_fields_follow_the_scalar_key_two_decoder_on_both_architectures() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            assert_eq!(
                AicDecoderHint::from_word(architecture, 0x0202_7cc2),
                Some(AicDecoderHint::ScalarKey2Insert {
                    destination_register: 1,
                    source_register: 7,
                    least_significant_bit: 6,
                    width_bits: 3,
                })
            );
            assert_eq!(
                AicDecoderHint::from_word(architecture, 0x0243_8b00),
                Some(AicDecoderHint::ScalarKey2InsertImmediate {
                    destination_register: 1,
                    position: 56,
                    immediate: 0,
                    extended: false,
                })
            );
            assert_eq!(
                AicDecoderHint::from_word(architecture, 0x02c0_1400),
                Some(AicDecoderHint::ScalarKey2BitSet {
                    destination_register: 0,
                    source_register: 1,
                    set_bit: false,
                })
            );
        }
    }

    #[test]
    fn scalar_immediate_store_decodes_width_value_and_signed_offset_on_both_architectures() {
        let word = 0x0f35_e580;
        assert_eq!(
            AicDecoderHint::from_word(Architecture::Dav2201, word),
            Some(AicDecoderHint::ScalarStoreImmediate {
                width_bytes: 1,
                base_register: 30,
                signed_offset: -724,
                post_index: false,
                value: ScalarStoreImmediateValue::Zero,
            })
        );
        assert_eq!(
            AicDecoderHint::from_word(Architecture::Dav3510, word),
            AicDecoderHint::from_word(Architecture::Dav2201, word)
        );
        assert_eq!(
            AicDecoderHint::from_word(Architecture::Dav3510, 0x0f35_e900),
            Some(AicDecoderHint::ScalarStoreImmediate {
                width_bytes: 1,
                base_register: 30,
                signed_offset: -696,
                post_index: false,
                value: ScalarStoreImmediateValue::Zero,
            })
        );

        for (value_bits, value) in [
            (0, ScalarStoreImmediateValue::Zero),
            (1, ScalarStoreImmediateValue::One),
            (2, ScalarStoreImmediateValue::Ones),
        ] {
            for (dtype, width_bytes) in [(0, 1), (1, 2), (2, 4), (3, 8)] {
                let variant = (word & !(3 << 22)) | (dtype << 22) | value_bits;
                for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
                    assert!(matches!(
                        AicDecoderHint::from_word(architecture, variant),
                        Some(AicDecoderHint::ScalarStoreImmediate {
                            width_bytes: decoded_width,
                            value: decoded_value,
                            ..
                        }) if decoded_width == width_bytes && decoded_value == value
                    ));
                }
            }
        }
        assert_eq!(
            AicDecoderHint::from_word(Architecture::Dav2201, word | 3),
            None
        );
        assert!(matches!(
            AicDecoderHint::from_word(Architecture::Dav2201, word | 4),
            Some(AicDecoderHint::ScalarStoreImmediate {
                post_index: true,
                ..
            })
        ));
    }

    #[test]
    fn scalar_compare_uses_two_sources_and_a_three_bit_condition() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            assert_eq!(
                AicDecoderHint::from_word(architecture, 0x0000_011e),
                Some(AicDecoderHint::ScalarCompare {
                    dtype_field: 0,
                    condition_field: 1,
                    first_source_register: 0,
                    second_source_register: 2,
                })
            );
            assert_eq!(
                AicDecoderHint::from_word(architecture, 0x0040_011e),
                Some(AicDecoderHint::ScalarCompare {
                    dtype_field: 1,
                    condition_field: 1,
                    first_source_register: 0,
                    second_source_register: 2,
                })
            );
            assert_eq!(
                AicDecoderHint::from_word(architecture, 0x0000_039f),
                Some(AicDecoderHint::ScalarCompareRegister {
                    dtype_field: 0,
                    condition_field: 1,
                    destination_register: 0,
                    first_source_register: 0,
                    second_source_register: 7,
                })
            );
        }
        assert_eq!(
            AicDecoderHint::from_word(Architecture::Dav2201, 0x0040_222e),
            Some(AicDecoderHint::ScalarCompare {
                dtype_field: 1,
                condition_field: 2,
                first_source_register: 2,
                second_source_register: 4,
            })
        );
    }

    #[test]
    fn captured_find_first_decodes_registers_and_match_bit() {
        assert_eq!(
            AicDecoderHint::from_word(Architecture::Dav2201, 0x02d6_9380),
            Some(AicDecoderHint::ScalarKey2FindFirst {
                destination_register: 11,
                source_register: 9,
                find_set: false,
            })
        );
        assert_eq!(
            AicDecoderHint::from_word(Architecture::Dav2201, 0x02d8_b380),
            Some(AicDecoderHint::ScalarKey2FindFirst {
                destination_register: 12,
                source_register: 11,
                find_set: false,
            })
        );
        assert_eq!(
            AicDecoderHint::from_word(Architecture::Dav2201, 0x02da_9380),
            Some(AicDecoderHint::ScalarKey2FindFirst {
                destination_register: 13,
                source_register: 9,
                find_set: false,
            })
        );
        assert_eq!(
            AicDecoderHint::from_word(Architecture::Dav2201, 0x02d6_8380),
            Some(AicDecoderHint::ScalarKey2FindFirst {
                destination_register: 11,
                source_register: 8,
                find_set: false,
            })
        );
        assert_eq!(
            AicDecoderHint::from_word(Architecture::Dav2201, 0x02de_d380),
            Some(AicDecoderHint::ScalarKey2FindFirst {
                destination_register: 15,
                source_register: 13,
                find_set: false,
            })
        );
        assert_eq!(
            AicDecoderHint::from_word(Architecture::Dav2201, 0x02da_b380),
            Some(AicDecoderHint::ScalarKey2FindFirst {
                destination_register: 13,
                source_register: 11,
                find_set: false,
            })
        );
        assert_eq!(
            AicDecoderHint::from_word(Architecture::Dav2201, 0x02dc_d380),
            Some(AicDecoderHint::ScalarKey2FindFirst {
                destination_register: 14,
                source_register: 13,
                find_set: false,
            })
        );
        assert_eq!(
            AicDecoderHint::from_word(Architecture::Dav2201, 0x02e0_f380),
            Some(AicDecoderHint::ScalarKey2FindFirst {
                destination_register: 16,
                source_register: 15,
                find_set: false,
            })
        );
        assert_eq!(
            AicDecoderHint::from_word(Architecture::Dav2201, 0x02de_d3c0),
            None
        );
        assert_eq!(
            AicDecoderHint::from_word(Architecture::Dav3510, 0x02d6_0380),
            Some(AicDecoderHint::ScalarKey2FindFirst {
                destination_register: 11,
                source_register: 0,
                find_set: false,
            })
        );
        assert_eq!(
            AicDecoderHint::from_word(Architecture::Dav3510, 0x02d6_03c0),
            Some(AicDecoderHint::ScalarKey2FindFirst {
                destination_register: 11,
                source_register: 0,
                find_set: true,
            })
        );
        assert_eq!(
            AicDecoderHint::from_word(Architecture::Dav2201, 0x02d6_0380),
            None
        );
    }

    #[test]
    fn scalar_compare_immediate_decodes_signed_twelve_bit_operand() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            assert_eq!(
                AicDecoderHint::from_word(architecture, 0x0a00_9000),
                Some(AicDecoderHint::ScalarCompareImmediate {
                    condition_field: 0,
                    source_register: 9,
                    encoded_immediate: 0,
                })
            );
            assert_eq!(
                AicDecoderHint::from_word(architecture, 0x0a80_9fff),
                Some(AicDecoderHint::ScalarCompareImmediate {
                    condition_field: 2,
                    source_register: 9,
                    encoded_immediate: 0xfff,
                })
            );
        }
    }

    #[test]
    fn scalar_select_decodes_the_three_x_registers() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            assert_eq!(
                AicDecoderHint::from_word(architecture, 0x00d4_a289),
                Some(AicDecoderHint::ScalarSelect {
                    dtype_field: 3,
                    destination_register: 10,
                    first_source_register: 10,
                    second_source_register: 5,
                })
            );
        }
    }
}
