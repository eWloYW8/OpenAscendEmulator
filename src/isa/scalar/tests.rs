use super::*;

#[test]
fn movx8_immediate_fields_match_both_architectures() {
    for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
        assert_eq!(
            ScalarInstruction::from_word(architecture, 0x1004_2002),
            Some(ScalarInstruction::ScalarMoveX8Immediate {
                destination_register: 2,
                encoded_immediate: 0x2002,
            })
        );
        assert!(!matches!(
            ScalarInstruction::from_word(architecture, 0x1204_2002),
            Some(ScalarInstruction::ScalarMoveX8Immediate { .. })
        ));
    }
}

#[test]
fn scalar_immediate_memory_routes_and_fields_are_architecture_specific() {
    use ScalarLoadStoreOperation::{Load, Store};

    assert_eq!(
        ScalarInstruction::from_word(Architecture::Dav2201, 0x03ce_5db8),
        Some(ScalarInstruction::ScalarLoadStoreImmediate {
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
        ScalarInstruction::from_word(Architecture::Dav2201, 0x13ce_5008),
        Some(ScalarInstruction::ScalarLoadStoreImmediate {
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
        ScalarInstruction::from_word(Architecture::Dav2201, 0x04c6_6000),
        Some(ScalarInstruction::ScalarLoadStoreImmediate {
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
        ScalarInstruction::from_word(Architecture::Dav3510, 0x1cce_5db8),
        Some(ScalarInstruction::ScalarLoadStoreImmediate {
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
        ScalarInstruction::from_word(Architecture::Dav3510, 0x1fce_5db8),
        Some(ScalarInstruction::ScalarLoadStoreImmediate {
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
        ScalarInstruction::from_word(Architecture::Dav3510, 0x03ce_5db8),
        Some(ScalarInstruction::ScalarLoadStoreImmediate {
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
        ScalarInstruction::from_word(Architecture::Dav3510, 0x13ce_5db8),
        Some(ScalarInstruction::ScalarLoadStoreImmediate {
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
        ScalarInstruction::from_word(Architecture::Dav3510, 0x04c6_6000),
        None
    );
    assert_eq!(
        ScalarInstruction::from_word(Architecture::Dav2201, 0x1cce_5db8),
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
            ScalarInstruction::from_word(architecture, word),
            Some(ScalarInstruction::ScalarIndexedLoad {
                width_bytes: 1,
                destination_register: destination,
                base_register: base,
                offset_register: offset,
            })
        );
        assert_eq!(
            ScalarInstruction::from_word(other, word),
            (other == Architecture::Dav3510).then_some(ScalarInstruction::ScalarIndexedLoad {
                width_bytes: 1,
                destination_register: destination,
                base_register: base,
                offset_register: offset,
            })
        );
        assert_eq!(ScalarInstruction::from_word(architecture, word | 1), None);
    }
}

#[test]
fn c310_indexed_loads_decode_supported_widths_and_reject_control_bits() {
    let word = 0x0103_4b00;
    for (dtype, width_bytes) in [(0, 1), (1, 2), (2, 4), (3, 8)] {
        let encoded = word | (dtype << 22);
        assert_eq!(
            ScalarInstruction::from_word(Architecture::Dav3510, encoded),
            Some(ScalarInstruction::ScalarIndexedLoad {
                width_bytes,
                destination_register: 1,
                base_register: 20,
                offset_register: 22,
            })
        );
        assert_eq!(
            ScalarInstruction::from_word(Architecture::Dav2201, encoded),
            None
        );
    }
    for changed in [word | 1, word | 2, word | 4, word | 8, word | 0x10] {
        assert_eq!(
            ScalarInstruction::from_word(Architecture::Dav3510, changed),
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
            ScalarInstruction::from_word(architecture, word),
            Some(ScalarInstruction::ScalarIndexedImmediateStore {
                width_bytes: 1,
                base_register: base,
                offset_register: offset,
                value: ScalarStoreImmediateValue::One,
            })
        );
        assert_eq!(
            ScalarInstruction::from_word(other, word),
            (other == Architecture::Dav3510).then_some(
                ScalarInstruction::ScalarIndexedImmediateStore {
                    width_bytes: 1,
                    base_register: base,
                    offset_register: offset,
                    value: ScalarStoreImmediateValue::One,
                }
            )
        );
        assert_eq!(
            ScalarInstruction::from_word(architecture, word ^ 1),
            (architecture == Architecture::Dav3510).then_some(
                ScalarInstruction::ScalarIndexedImmediateStore {
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
                ScalarInstruction::from_word(Architecture::Dav3510, encoded),
                Some(ScalarInstruction::ScalarIndexedImmediateStore {
                    width_bytes,
                    base_register: 20,
                    offset_register: 22,
                    value,
                })
            );
            assert_eq!(
                ScalarInstruction::from_word(Architecture::Dav2201, encoded),
                None
            );
        }
    }
    for changed in [word | 3, word | 4, word | 8, word | 0x10] {
        assert_eq!(
            ScalarInstruction::from_word(Architecture::Dav3510, changed),
            None
        );
    }
}

#[test]
fn pair_load_fields_and_opcode_groups_are_architecture_specific() {
    let c310_word = 0x0cca_0190;
    assert_eq!(
        ScalarInstruction::from_word(Architecture::Dav3510, c310_word),
        Some(ScalarInstruction::ScalarPairLoad {
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
        ScalarInstruction::from_word(Architecture::Dav2201, c310_word),
        Some(ScalarInstruction::ScalarPairLoad { .. })
    ));
    let c220_word = (c310_word & !(0x1f << 24)) | (9 << 24);
    assert!(matches!(
        ScalarInstruction::from_word(Architecture::Dav2201, c220_word),
        Some(ScalarInstruction::ScalarPairLoad {
            signed_offset: 8,
            sign_extend: false,
            ..
        })
    ));
    assert!(matches!(
        ScalarInstruction::from_word(Architecture::Dav2201, 0x09c4_00b0),
        Some(ScalarInstruction::ScalarPairLoad {
            first_destination_register: 2,
            second_destination_register: 1,
            base_register: 0,
            signed_offset: 24,
            ..
        })
    ));
    assert!(matches!(
        ScalarInstruction::from_word(Architecture::Dav2201, 0x09c8_0190),
        Some(ScalarInstruction::ScalarPairLoad {
            first_destination_register: 4,
            second_destination_register: 3,
            base_register: 0,
            signed_offset: 8,
            ..
        })
    ));
    assert!(matches!(
        ScalarInstruction::from_word(Architecture::Dav3510, c310_word | (1 << 24)),
        Some(ScalarInstruction::ScalarPairLoad {
            sign_extend: true,
            ..
        })
    ));
    assert_eq!(
        ScalarInstruction::from_word(Architecture::Dav3510, c310_word | 1),
        Some(ScalarInstruction::ScalarPairStore {
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
            ScalarInstruction::from_word(architecture, word)
                .unwrap()
                .scalar_address_effect(base),
            Some(ScalarAddressEffect {
                effective_address,
                updated_base,
            })
        );
    }
    assert_eq!(
        ScalarInstruction::from_word(Architecture::Dav2201, 0x073a_7f80)
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
        let effect = ScalarInstruction::from_word(architecture, word)
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
            ScalarInstruction::from_word(Architecture::Dav3510, word),
            Some(ScalarInstruction::ScalarLoadStoreImmediate {
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
            let word =
                (8_u32 << 24) | ((variant as u32) << 22) | (encoded_xd << 17) | (29 << 12) | 0x788;
            let hint = ScalarInstruction::from_word(architecture, word);
            assert_eq!(
                hint,
                Some(ScalarInstruction::ScalarKey8 {
                    operation,
                    destination_register: (variant != 3).then_some(30),
                    source_register: if variant == 3 { 61 } else { 29 },
                    encoded_immediate: 0x788,
                })
            );
        }
        assert_eq!(
            ScalarInstruction::from_word(architecture, 0x083d_d788),
            Some(ScalarInstruction::ScalarKey8 {
                operation: ScalarKey8Operation::AddImmediate,
                destination_register: Some(30),
                source_register: 29,
                encoded_immediate: 0x788,
            })
        );
        assert_eq!(
            ScalarInstruction::from_word(architecture, 0x08c0_0000),
            Some(ScalarInstruction::ScalarKey8 {
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
                ScalarInstruction::from_word(architecture, word),
                Some(ScalarInstruction::ScalarKey0 {
                    operation,
                    dtype_field: dtype,
                    destination_register: xd,
                    first_source_register: xn,
                    second_source_register: xm,
                })
            );
        }
        assert_eq!(
            ScalarInstruction::from_word(architecture, 0x003b_d780),
            None
        );
    }
    assert_eq!(
        ScalarInstruction::from_word(Architecture::Dav2201, 0x003b_d7f1),
        Some(ScalarInstruction::ScalarKey0 {
            operation: ScalarKey0Operation::Add,
            dtype_field: 0,
            destination_register: 61,
            first_source_register: 61,
            second_source_register: 47,
        })
    );
    assert_eq!(
        ScalarInstruction::from_word(Architecture::Dav3510, 0x003b_d7f1),
        Some(ScalarInstruction::ScalarKey0 {
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
            ScalarInstruction::from_word(architecture, 0x02c8_020f),
            Some(ScalarInstruction::ScalarKey2ShiftLeft {
                dtype_field: 3,
                destination_register: 4,
                count_register: None,
                encoded_immediate: 15,
            })
        );
        assert_eq!(
            ScalarInstruction::from_word(architecture, 0x02d5_6240),
            Some(ScalarInstruction::ScalarKey2ShiftLeft {
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
            ScalarInstruction::from_word(architecture, 0x0202_2080),
            Some(ScalarInstruction::ScalarKey2Negate {
                dtype_field: 0,
                destination_register: 1,
                source_register: 2,
            })
        );
    }
    assert_eq!(
        ScalarInstruction::from_word(Architecture::Dav2201, 0x0202_20e0),
        Some(ScalarInstruction::ScalarKey2Negate {
            dtype_field: 0,
            destination_register: 33,
            source_register: 34,
        })
    );
    assert_eq!(
        ScalarInstruction::from_word(Architecture::Dav3510, 0x0202_20e0),
        Some(ScalarInstruction::ScalarKey2Negate {
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
            ScalarInstruction::from_word(architecture, 0x0252_028f),
            Some(ScalarInstruction::ScalarKey2ShiftRight {
                dtype_field: 1,
                destination_register: 9,
                count_register: None,
                encoded_immediate: 15,
            })
        );
        assert_eq!(
            ScalarInstruction::from_word(architecture, 0x0212_62c0),
            Some(ScalarInstruction::ScalarKey2ShiftRight {
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
            ScalarInstruction::from_word(architecture, word),
            Some(ScalarInstruction::ScalarKey2MoveToSpr {
                encoded_destination_spr: spr,
                source_register: source,
            })
        );
    }
    assert_eq!(
        ScalarInstruction::from_word(Architecture::Dav2201, 0x0230_0901),
        Some(ScalarInstruction::ScalarKey2MoveToSpr {
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
                ScalarInstruction::from_word(architecture, word),
                Some(ScalarInstruction::ScalarKey2MoveFromSpr {
                    destination_register,
                    encoded_source_spr,
                })
            );
        }
    }
    assert_eq!(
        ScalarInstruction::from_word(Architecture::Dav2201, 0x029e_3881),
        ScalarInstruction::from_word(Architecture::Dav2201, 0x029e_3880)
    );
    assert_eq!(
        ScalarInstruction::from_word(Architecture::Dav3510, 0x029e_3881),
        Some(ScalarInstruction::ScalarKey2MoveFromSpr {
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
                ScalarInstruction::from_word(architecture, word),
                Some(ScalarInstruction::ScalarKey2ZeroExtend {
                    width,
                    destination_register: 4,
                    source_register: 3,
                })
            );
        }
    }
    assert_eq!(
        ScalarInstruction::from_word(Architecture::Dav2201, 0x0208_3a60),
        Some(ScalarInstruction::ScalarKey2ZeroExtend {
            width: ZeroExtendWidth::U8,
            destination_register: 36,
            source_register: 35,
        })
    );
    assert_eq!(
        ScalarInstruction::from_word(Architecture::Dav3510, 0x0208_3a60),
        Some(ScalarInstruction::ScalarKey2ZeroExtend {
            width: ZeroExtendWidth::U8,
            destination_register: 4,
            source_register: 3,
        })
    );
    for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
        assert_eq!(
            ScalarInstruction::from_word(architecture, 0x02c8_3a00),
            None
        );
        assert_eq!(
            ScalarInstruction::from_word(architecture, 0x0288_3980),
            Some(ScalarInstruction::ScalarKey2SignExtend {
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
                ScalarInstruction::from_word(architecture, word),
                Some(ScalarInstruction::ScalarKey2SignExtend {
                    width_bits,
                    destination_register: 14,
                    source_register: 15,
                })
            );
        }
    }
    for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
        assert_eq!(
            ScalarInstruction::from_word(architecture, 0x02dc_f980),
            None
        );
    }
}

#[test]
fn register_move_fields_follow_architecture_specific_register_decoders() {
    for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
        assert_eq!(
            ScalarInstruction::from_word(architecture, 0x0209_e800),
            Some(ScalarInstruction::ScalarKey2MoveRegister {
                dtype_field: 0,
                destination_register: 4,
                source_register: 30,
            })
        );
    }
    assert_eq!(
        ScalarInstruction::from_word(Architecture::Dav2201, 0x0209_e860),
        Some(ScalarInstruction::ScalarKey2MoveRegister {
            dtype_field: 0,
            destination_register: 36,
            source_register: 62,
        })
    );
    assert_eq!(
        ScalarInstruction::from_word(Architecture::Dav3510, 0x0209_e860),
        Some(ScalarInstruction::ScalarKey2MoveRegister {
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
            ScalarInstruction::from_word(architecture, 0x073a_7f80),
            Some(ScalarInstruction::ScalarKey7 {
                operation: ScalarKey7Operation::MoveImmediate,
                destination_register: 29,
                encoded_immediate: 0x7f80,
                halfword_lane: None,
            })
        );
        assert_eq!(
            ScalarInstruction::from_word(architecture, 0x077b_0010),
            Some(ScalarInstruction::ScalarKey7 {
                operation: ScalarKey7Operation::MoveKeep,
                destination_register: 29,
                encoded_immediate: 0x10,
                halfword_lane: Some(1),
            })
        );
        assert_eq!(
            ScalarInstruction::from_word(architecture, 0x07cd_ffff),
            Some(ScalarInstruction::ScalarKey7 {
                operation: ScalarKey7Operation::MoveKeep,
                destination_register: 6,
                encoded_immediate: 0xffff,
                halfword_lane: Some(3),
            })
        );
    }
    assert_eq!(
        ScalarInstruction::from_word(Architecture::Dav3510, 0x073a_ff20),
        Some(ScalarInstruction::ScalarKey7 {
            operation: ScalarKey7Operation::MoveImmediate,
            destination_register: 29,
            encoded_immediate: 0xff20,
            halfword_lane: None,
        })
    );
    assert_eq!(
        ScalarInstruction::from_word(Architecture::Dav3510, 0x077b_000f),
        Some(ScalarInstruction::ScalarKey7 {
            operation: ScalarKey7Operation::MoveKeep,
            destination_register: 29,
            encoded_immediate: 15,
            halfword_lane: Some(1),
        })
    );
}

#[test]
fn insert_fields_follow_the_scalar_key_two_decoder_on_both_architectures() {
    for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
        assert_eq!(
            ScalarInstruction::from_word(architecture, 0x0202_7cc2),
            Some(ScalarInstruction::ScalarKey2Insert {
                destination_register: 1,
                source_register: 7,
                least_significant_bit: 6,
                width_bits: 3,
            })
        );
        assert_eq!(
            ScalarInstruction::from_word(architecture, 0x0243_8b00),
            Some(ScalarInstruction::ScalarKey2InsertImmediate {
                destination_register: 1,
                position: 56,
                immediate: 0,
                extended: false,
            })
        );
        assert_eq!(
            ScalarInstruction::from_word(architecture, 0x02c0_1400),
            Some(ScalarInstruction::ScalarKey2BitSet {
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
        ScalarInstruction::from_word(Architecture::Dav2201, word),
        Some(ScalarInstruction::ScalarStoreImmediate {
            width_bytes: 1,
            base_register: 30,
            signed_offset: -724,
            post_index: false,
            value: ScalarStoreImmediateValue::Zero,
        })
    );
    assert_eq!(
        ScalarInstruction::from_word(Architecture::Dav3510, word),
        ScalarInstruction::from_word(Architecture::Dav2201, word)
    );
    assert_eq!(
        ScalarInstruction::from_word(Architecture::Dav3510, 0x0f35_e900),
        Some(ScalarInstruction::ScalarStoreImmediate {
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
                    ScalarInstruction::from_word(architecture, variant),
                    Some(ScalarInstruction::ScalarStoreImmediate {
                        width_bytes: decoded_width,
                        value: decoded_value,
                        ..
                    }) if decoded_width == width_bytes && decoded_value == value
                ));
            }
        }
    }
    assert_eq!(
        ScalarInstruction::from_word(Architecture::Dav2201, word | 3),
        None
    );
    assert!(matches!(
        ScalarInstruction::from_word(Architecture::Dav2201, word | 4),
        Some(ScalarInstruction::ScalarStoreImmediate {
            post_index: true,
            ..
        })
    ));
}

#[test]
fn scalar_compare_uses_two_sources_and_a_three_bit_condition() {
    for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
        assert_eq!(
            ScalarInstruction::from_word(architecture, 0x0000_011e),
            Some(ScalarInstruction::ScalarCompare {
                dtype_field: 0,
                condition_field: 1,
                first_source_register: 0,
                second_source_register: 2,
            })
        );
        assert_eq!(
            ScalarInstruction::from_word(architecture, 0x0040_011e),
            Some(ScalarInstruction::ScalarCompare {
                dtype_field: 1,
                condition_field: 1,
                first_source_register: 0,
                second_source_register: 2,
            })
        );
        assert_eq!(
            ScalarInstruction::from_word(architecture, 0x0000_039f),
            Some(ScalarInstruction::ScalarCompareRegister {
                dtype_field: 0,
                condition_field: 1,
                destination_register: 0,
                first_source_register: 0,
                second_source_register: 7,
            })
        );
    }
    assert_eq!(
        ScalarInstruction::from_word(Architecture::Dav2201, 0x0040_222e),
        Some(ScalarInstruction::ScalarCompare {
            dtype_field: 1,
            condition_field: 2,
            first_source_register: 2,
            second_source_register: 4,
        })
    );
}

#[test]
fn find_first_decodes_registers_and_match_bit() {
    assert_eq!(
        ScalarInstruction::from_word(Architecture::Dav2201, 0x02d6_9380),
        Some(ScalarInstruction::ScalarKey2FindFirst {
            destination_register: 11,
            source_register: 9,
            find_set: false,
        })
    );
    assert_eq!(
        ScalarInstruction::from_word(Architecture::Dav2201, 0x02d8_b380),
        Some(ScalarInstruction::ScalarKey2FindFirst {
            destination_register: 12,
            source_register: 11,
            find_set: false,
        })
    );
    assert_eq!(
        ScalarInstruction::from_word(Architecture::Dav2201, 0x02da_9380),
        Some(ScalarInstruction::ScalarKey2FindFirst {
            destination_register: 13,
            source_register: 9,
            find_set: false,
        })
    );
    assert_eq!(
        ScalarInstruction::from_word(Architecture::Dav2201, 0x02d6_8380),
        Some(ScalarInstruction::ScalarKey2FindFirst {
            destination_register: 11,
            source_register: 8,
            find_set: false,
        })
    );
    assert_eq!(
        ScalarInstruction::from_word(Architecture::Dav2201, 0x02de_d380),
        Some(ScalarInstruction::ScalarKey2FindFirst {
            destination_register: 15,
            source_register: 13,
            find_set: false,
        })
    );
    assert_eq!(
        ScalarInstruction::from_word(Architecture::Dav2201, 0x02da_b380),
        Some(ScalarInstruction::ScalarKey2FindFirst {
            destination_register: 13,
            source_register: 11,
            find_set: false,
        })
    );
    assert_eq!(
        ScalarInstruction::from_word(Architecture::Dav2201, 0x02dc_d380),
        Some(ScalarInstruction::ScalarKey2FindFirst {
            destination_register: 14,
            source_register: 13,
            find_set: false,
        })
    );
    assert_eq!(
        ScalarInstruction::from_word(Architecture::Dav2201, 0x02e0_f380),
        Some(ScalarInstruction::ScalarKey2FindFirst {
            destination_register: 16,
            source_register: 15,
            find_set: false,
        })
    );
    assert_eq!(
        ScalarInstruction::from_word(Architecture::Dav2201, 0x02de_d3c0),
        Some(ScalarInstruction::ScalarKey2FindFirst {
            destination_register: 15,
            source_register: 13,
            find_set: true,
        })
    );
    assert_eq!(
        ScalarInstruction::from_word(Architecture::Dav3510, 0x02d6_0380),
        Some(ScalarInstruction::ScalarKey2FindFirst {
            destination_register: 11,
            source_register: 0,
            find_set: false,
        })
    );
    assert_eq!(
        ScalarInstruction::from_word(Architecture::Dav3510, 0x02d6_03c0),
        Some(ScalarInstruction::ScalarKey2FindFirst {
            destination_register: 11,
            source_register: 0,
            find_set: true,
        })
    );
    assert_eq!(
        ScalarInstruction::from_word(Architecture::Dav2201, 0x02d6_0380),
        Some(ScalarInstruction::ScalarKey2FindFirst {
            destination_register: 11,
            source_register: 0,
            find_set: false,
        })
    );
    assert_eq!(
        ScalarInstruction::from_word(Architecture::Dav2201, 0x02d6_0390),
        None
    );
}

#[test]
fn scalar_compare_immediate_decodes_signed_twelve_bit_operand() {
    for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
        assert_eq!(
            ScalarInstruction::from_word(architecture, 0x0a00_9000),
            Some(ScalarInstruction::ScalarCompareImmediate {
                condition_field: 0,
                source_register: 9,
                encoded_immediate: 0,
            })
        );
        assert_eq!(
            ScalarInstruction::from_word(architecture, 0x0a80_9fff),
            Some(ScalarInstruction::ScalarCompareImmediate {
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
            ScalarInstruction::from_word(architecture, 0x00d4_a289),
            Some(ScalarInstruction::ScalarSelect {
                dtype_field: 3,
                destination_register: 10,
                first_source_register: 10,
                second_source_register: 5,
            })
        );
    }
}
