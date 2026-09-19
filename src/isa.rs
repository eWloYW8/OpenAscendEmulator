
use crate::architecture::Architecture;
use serde::Serialize;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ScalarKey8Operation {
    AddImmediate,
    MultiplyImmediate,
    SubtractImmediate,
    DcPreload,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ScalarKey0Operation {
    Add,
    Multiply,
    And,
    Or,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ScalarKey7Operation {
    MoveImmediate,
    MoveKeep,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ScalarLoadStoreOperation {
    Load,
    Store,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ScalarAddressEffect {
    pub effective_address: u64,
    pub updated_base: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum AicDecoderHint {
    ScalarLoadStoreImmediate {
        operation: ScalarLoadStoreOperation,
        vendor_isa_name: u16,
        dtype_field: u8,
        width_bytes: u8,
        data_register: u8,
        base_register: u8,
        signed_offset: i16,
        post_index: bool,
        sign_extend: Option<bool>,
    },
    ScalarKey0 {
        operation: ScalarKey0Operation,
        vendor_isa_name: u16,
        dtype_field: u8,
        destination_register: u8,
        first_source_register: u8,
        second_source_register: u8,
    },
    ScalarKey2MoveRegister {
        vendor_isa_name: u16,
        dtype_field: u8,
        destination_register: u8,
        source_register: u8,
    },
    ScalarKey2MoveToSpr {
        vendor_isa_name: u16,
        encoded_destination_spr: u16,
        source_register: u8,
    },
    ScalarKey2ZeroExtend {
        vendor_isa_name: u16,
        width: ZeroExtendWidth,
        destination_register: u8,
        source_register: u8,
    },
    ScalarKey2ShiftLeft {
        vendor_isa_name: u16,
        dtype_field: u8,
        destination_register: u8,
        count_register: Option<u8>,
        encoded_immediate: u8,
    },
    ScalarKey7 {
        operation: ScalarKey7Operation,
        vendor_isa_name: u16,
        destination_register: u8,
        encoded_immediate: u16,
        halfword_lane: Option<u8>,
    },
    ScalarKey8 {
        operation: ScalarKey8Operation,
        vendor_isa_name: u16,
        destination_register: Option<u8>,
        source_register: u8,
        encoded_immediate: u16,
    },
    C310MovAlignV2 {
        vendor_isa_name: u16,
        direction_field: u8,
        dtype_field: u8,
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
            AicClass::Scalar if ((word >> 24) & 0x1f) == 0 => {
                let (operation, vendor_isa_name) = match word & 0xf {
                    1 => (ScalarKey0Operation::Add, 0),
                    3 => (ScalarKey0Operation::Multiply, 2),
                    10 => (ScalarKey0Operation::And, 9),
                    11 => (ScalarKey0Operation::Or, 10),
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
                    vendor_isa_name,
                    dtype_field: ((word >> 22) & 3) as u8,
                    destination_register,
                    first_source_register,
                    second_source_register,
                })
            }
            AicClass::Scalar if ((word >> 24) & 0x1f) == 2 && ((word >> 7) & 0x1f) == 4 => {
                Some(Self::ScalarKey2ShiftLeft {
                    vendor_isa_name: 25,
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
            AicClass::Scalar if ((word >> 24) & 0x1f) == 2 && ((word >> 7) & 0x1f) == 16 => {
                let (destination_register, source_register) =
                    scalar_key2_registers(architecture, word);
                Some(Self::ScalarKey2MoveRegister {
                    vendor_isa_name: 33,
                    dtype_field: ((word >> 22) & 3) as u8,
                    destination_register,
                    source_register,
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
                    vendor_isa_name: 35,
                    encoded_destination_spr,
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
                    vendor_isa_name: 37,
                    width,
                    destination_register,
                    source_register,
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
                    vendor_isa_name: if move_keep { 45 } else { 44 },
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
                    vendor_isa_name: 46 + variant as u16,
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
                Some(Self::C310MovAlignV2 {
                    vendor_isa_name: 141,
                    direction_field: ((word >> 22) & 3) as u8,
                    dtype_field: (word & 3) as u8,
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
        vendor_isa_name: if matches!(operation, ScalarLoadStoreOperation::Load) {
            40
        } else {
            41
        },
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
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
                vendor_isa_name: 40,
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
                vendor_isa_name: 40,
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
                vendor_isa_name: 41,
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
                vendor_isa_name: 40,
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
                vendor_isa_name: 40,
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
                vendor_isa_name: 41,
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
                vendor_isa_name: 41,
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
                    vendor_isa_name: 40,
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
                        vendor_isa_name: 46 + variant as u16,
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
                    vendor_isa_name: 46,
                    destination_register: Some(30),
                    source_register: 29,
                    encoded_immediate: 0x788,
                })
            );
            assert_eq!(
                AicDecoderHint::from_word(architecture, 0x08c0_0000),
                Some(AicDecoderHint::ScalarKey8 {
                    operation: ScalarKey8Operation::DcPreload,
                    vendor_isa_name: 49,
                    destination_register: None,
                    source_register: 0,
                    encoded_immediate: 0,
                })
            );
        }
    }

    #[test]
    fn scalar_key0_register_fields_and_name_routes_match_both_pem_binaries() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            for (word, operation, name, dtype, xd, xn, xm) in [
                (0x003b_d781, ScalarKey0Operation::Add, 0, 0, 29, 29, 15),
                (0x000e_8383, ScalarKey0Operation::Multiply, 2, 0, 7, 8, 7),
                (0x00de_f80a, ScalarKey0Operation::And, 9, 3, 15, 15, 16),
                (0x00c0_100b, ScalarKey0Operation::Or, 10, 3, 0, 1, 0),
            ] {
                assert_eq!(
                    AicDecoderHint::from_word(architecture, word),
                    Some(AicDecoderHint::ScalarKey0 {
                        operation,
                        vendor_isa_name: name,
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
                vendor_isa_name: 0,
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
                vendor_isa_name: 0,
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
                    vendor_isa_name: 25,
                    dtype_field: 3,
                    destination_register: 4,
                    count_register: None,
                    encoded_immediate: 15,
                })
            );
            assert_eq!(
                AicDecoderHint::from_word(architecture, 0x02d5_6240),
                Some(AicDecoderHint::ScalarKey2ShiftLeft {
                    vendor_isa_name: 25,
                    dtype_field: 3,
                    destination_register: 10,
                    count_register: Some(22),
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
                    vendor_isa_name: 35,
                    encoded_destination_spr: spr,
                    source_register: source,
                })
            );
        }
        assert_eq!(
            AicDecoderHint::from_word(Architecture::Dav2201, 0x0230_0901),
            Some(AicDecoderHint::ScalarKey2MoveToSpr {
                vendor_isa_name: 35,
                encoded_destination_spr: 24,
                source_register: 0,
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
                        vendor_isa_name: 37,
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
                vendor_isa_name: 37,
                width: ZeroExtendWidth::U8,
                destination_register: 36,
                source_register: 35,
            })
        );
        assert_eq!(
            AicDecoderHint::from_word(Architecture::Dav3510, 0x0208_3a60),
            Some(AicDecoderHint::ScalarKey2ZeroExtend {
                vendor_isa_name: 37,
                width: ZeroExtendWidth::U8,
                destination_register: 4,
                source_register: 3,
            })
        );
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            assert_eq!(AicDecoderHint::from_word(architecture, 0x02c8_3a00), None);
            assert_eq!(AicDecoderHint::from_word(architecture, 0x0288_3980), None);
        }
    }

    #[test]
    fn register_move_fields_follow_architecture_specific_register_decoders() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            assert_eq!(
                AicDecoderHint::from_word(architecture, 0x0209_e800),
                Some(AicDecoderHint::ScalarKey2MoveRegister {
                    vendor_isa_name: 33,
                    dtype_field: 0,
                    destination_register: 4,
                    source_register: 30,
                })
            );
        }
        assert_eq!(
            AicDecoderHint::from_word(Architecture::Dav2201, 0x0209_e860),
            Some(AicDecoderHint::ScalarKey2MoveRegister {
                vendor_isa_name: 33,
                dtype_field: 0,
                destination_register: 36,
                source_register: 62,
            })
        );
        assert_eq!(
            AicDecoderHint::from_word(Architecture::Dav3510, 0x0209_e860),
            Some(AicDecoderHint::ScalarKey2MoveRegister {
                vendor_isa_name: 33,
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
                    vendor_isa_name: 44,
                    destination_register: 29,
                    encoded_immediate: 0x7f80,
                    halfword_lane: None,
                })
            );
            assert_eq!(
                AicDecoderHint::from_word(architecture, 0x077b_0010),
                Some(AicDecoderHint::ScalarKey7 {
                    operation: ScalarKey7Operation::MoveKeep,
                    vendor_isa_name: 45,
                    destination_register: 29,
                    encoded_immediate: 0x10,
                    halfword_lane: Some(1),
                })
            );
            assert_eq!(
                AicDecoderHint::from_word(architecture, 0x07cd_ffff),
                Some(AicDecoderHint::ScalarKey7 {
                    operation: ScalarKey7Operation::MoveKeep,
                    vendor_isa_name: 45,
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
                vendor_isa_name: 44,
                destination_register: 29,
                encoded_immediate: 0xff20,
                halfword_lane: None,
            })
        );
        assert_eq!(
            AicDecoderHint::from_word(Architecture::Dav3510, 0x077b_000f),
            Some(AicDecoderHint::ScalarKey7 {
                operation: ScalarKey7Operation::MoveKeep,
                vendor_isa_name: 45,
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
                    vendor_isa_name: 141,
                    direction_field: 0,
                    dtype_field: 0,
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
    }
}
