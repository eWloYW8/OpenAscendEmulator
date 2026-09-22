use super::*;
use crate::isa::c220::vector::C220MovemaskHint;

fn movev_control() -> C220MovevControl {
    decode_c220_movev_control(C220_CAPTURED_MOVEV_CONTROL).unwrap()
}

fn fp32_control() -> C220VectorControl {
    decode_c220_fp32_control(C220_CAPTURED_VADD_CONTROL).unwrap()
}

fn addresses(source_0: u64, source_1: u64, destination: u64) -> C220VectorAddresses {
    C220VectorAddresses {
        source_0,
        source_1,
        destination,
    }
}

#[test]
fn vector_controls_decode_block_and_repeat_strides() {
    assert_eq!(movev_control().destination_block_stride, 1);
    assert_eq!(movev_control().destination_repeat_stride, 8);
    assert_eq!(fp32_control().source_0_block_stride, 1);
    assert_eq!(fp32_control().destination_repeat_stride, 8);
    assert_eq!(
        decode_c220_movev_control((2_u64 << 56) | (3 << 52) | (5 << 32) | (9 << 16) | 7).unwrap(),
        C220MovevControl {
            encoded_repeat_count: 2,
            destination_block_stride: 7,
            destination_repeat_stride: (3 << 8) | 5,
        }
    );
    assert_eq!(
        decode_c220_fp32_control(
            (2_u64 << 56) | (6 << 40) | (5 << 32) | (4 << 24) | (3 << 16) | (2 << 8) | 1
        )
        .unwrap(),
        C220VectorControl {
            encoded_repeat_count: 2,
            destination_block_stride: 1,
            source_0_block_stride: 2,
            source_1_block_stride: 3,
            destination_repeat_stride: 4,
            source_0_repeat_stride: 5,
            source_1_repeat_stride: 6,
        }
    );
    assert_eq!(
        decode_c220_movev_control(0x0200_0008_0001_0001)
            .unwrap()
            .encoded_repeat_count,
        2
    );
    assert_eq!(
        decode_c220_fp32_control(0x0200_0808_0801_0101)
            .unwrap()
            .encoded_repeat_count,
        2
    );
}

#[test]
fn vabs_uses_one_source_and_preserves_masked_destination_lanes() {
    let word = 0x83c0_0300 | (3 << 17) | (4 << 12) | (5 << 2);
    let hint = C220VecArithmeticHint::from_word(word).unwrap();
    assert_eq!(C220VecArithmeticHint::from_word(word & !(1 << 24)), None);
    assert_eq!(hint.operation, C220VecArithmeticOperation::Absolute);
    assert_eq!(hint.destination_register, 3);
    assert_eq!(hint.source_0_register, 4);
    assert_eq!(hint.source_1_register, None);
    assert_eq!(hint.control_register, 5);
    assert!(hint.has_fp32_value_path());

    let wide = decode_c220_vector_unary_control(
        (2_u64 << 56) | (1 << 52) | (9 << 40) | (7 << 32) | (3 << 16) | 2,
    );
    assert_eq!(wide.destination_block_stride, 2);
    assert_eq!(wide.source_0_block_stride, 3);
    assert_eq!(wide.destination_repeat_stride, 0x107);
    assert_eq!(wide.source_0_repeat_stride, 9);
    assert_eq!(wide.encoded_repeat_count, 2);

    let control =
        decode_c220_vector_unary_control((1_u64 << 56) | (8 << 40) | (8 << 32) | (1 << 16) | 1);
    let addresses = addresses(0, u64::MAX, 0x200);
    let mask = [0b111, 0, 0, 0];
    let mut ub = UbMemory::new(1024, 256);
    let mut source = [0; C220_VECTOR_BLOCK_BYTES];
    for (lane, bits) in [
        (-3.5_f32).to_bits(),
        (-f32::INFINITY).to_bits(),
        0xff80_0001,
    ]
    .into_iter()
    .enumerate()
    {
        source[lane * 4..lane * 4 + 4].copy_from_slice(&bits.to_le_bytes());
    }
    ub.write_states(0, &source.map(MemoryByteState::Known))
        .unwrap();
    ub.write_states(
        0x20c,
        &0xdead_beef_u32.to_le_bytes().map(MemoryByteState::Known),
    )
    .unwrap();

    let accesses =
        plan_c220_vector_arithmetic_read_accesses(hint, control, addresses, 0, &mask, None)
            .unwrap();
    assert_eq!(accesses.len(), 1);
    assert_eq!(accesses[0].source_index, 0);
    let step = execute_c220_fp32_to_ub(0x4000, word, control, addresses, &[mask], &mut ub).unwrap();
    assert_eq!(step.stores.len(), 3);
    assert_eq!(step.lanes[0].bits, 3.5_f32.to_bits());
    assert_eq!(step.lanes[1].bits, f32::INFINITY.to_bits());
    assert_eq!(step.lanes[2].bits, 0x7fff_ffff);
    assert!(step.lanes[2].status.unwrap().nan_operand);
    assert_eq!(step.source_1_bytes, vec![0; C220_VECTOR_TILE_BYTES]);
    assert_eq!(
        ub.read_known(0x20c, 4).unwrap(),
        0xdead_beef_u32.to_le_bytes()
    );
}

#[test]
fn movemask_selects_source_and_mask_register() {
    let first = C220MovemaskHint::from_word(0x8040_0000).unwrap();
    assert_eq!(first.source_register, 0);
    assert_eq!(first.destination_spr, 100);

    let second = C220MovemaskHint::from_word(0x8040_008c).unwrap();
    assert_eq!(second.source_register, 3);
    assert_eq!(second.destination_spr, 101);
    for word in [0x8040_0000 ^ (1 << 22), 0x8240_0000, 0x0040_0000] {
        assert_eq!(C220MovemaskHint::from_word(word), None);
    }
}

#[test]
fn captured_fp32_mask_expands_count_mode_and_preserves_bitset_mode() {
    assert_eq!(
        decode_c220_fp32_mask(0, 0x5555_5555, 0).unwrap(),
        [0x5555_5555, 0, 0, 0]
    );
    assert_eq!(
        decode_c220_fp32_mask(C220_COUNT_MASK_CONTROL, 32, 0).unwrap(),
        [0xffff_ffff, 0, 0, 0]
    );
    assert_eq!(
        decode_c220_fp32_mask(C220_COUNT_MASK_CONTROL, 0, 0).unwrap(),
        [0; 4]
    );
    assert_eq!(
        decode_c220_fp32_mask(C220_COUNT_MASK_CONTROL, 17, 0).unwrap(),
        [0x1ffff, 0, 0, 0]
    );
    assert_eq!(
        decode_c220_fp32_mask(C220_COUNT_MASK_CONTROL, 64, 0).unwrap(),
        [u64::MAX, 0, 0, 0]
    );
    assert!(matches!(
        decode_c220_fp32_mask(1, 32, 0),
        Err(C220VectorError::UnsupportedMaskControl { .. })
    ));
    assert!(matches!(
        decode_c220_fp32_mask(1 << 52, 1, 0),
        Err(C220VectorError::UnsupportedMaskControl { .. })
    ));
    assert!(matches!(
        decode_c220_fp32_mask(C220_COUNT_MASK_CONTROL, 32, 1),
        Err(C220VectorError::UnsupportedCountMaskHigh { .. })
    ));
    assert!(matches!(
        decode_c220_fp32_mask(C220_COUNT_MASK_CONTROL, 65, 0),
        Err(C220VectorError::CountMaskExceedsTile { .. })
    ));
}

#[test]
fn vendor_add_and_sub_words_select_distinct_vec_handlers() {
    let add = C220VecArithmeticHint::from_word(0x85dc_b618).unwrap();
    let subtract = C220VecArithmeticHint::from_word(0x85dc_b619).unwrap();
    assert_eq!(add.operation, C220VecArithmeticOperation::Add);
    assert_eq!(subtract.operation, C220VecArithmeticOperation::Subtract);
    assert_eq!(add.destination_register, 14);
    assert_eq!(add.source_0_register, 11);
    assert_eq!(add.source_1_register, Some(12));
    assert_eq!(add.control_register, 6);
    assert_eq!(add.dtype_selector, 3);
    assert!(add.has_fp32_value_path());
    assert!(subtract.has_fp32_value_path());
}

#[test]
fn multiply_word_selects_fp32_lane_path() {
    let hint = C220VecArithmeticHint::from_word(C220_CAPTURED_VMUL_WORD).unwrap();
    assert_eq!(hint.operation, C220VecArithmeticOperation::Multiply);
    assert_eq!(hint.destination_register, 14);
    assert_eq!(hint.source_0_register, 11);
    assert_eq!(hint.source_1_register, Some(12));
    assert_eq!(hint.control_register, 6);
    assert!(hint.has_fp32_value_path());
    let first = [2.0_f32.to_bits(), 0];
    let second = [3.0_f32.to_bits(), f32::INFINITY.to_bits()];
    let lanes = hint
        .evaluate_fp32_lanes(&first, &second, &[3, 0, 0, 0])
        .unwrap();
    assert_eq!(lanes[0].bits, 6.0_f32.to_bits());
    assert_eq!(lanes[1].bits, 0x7fff_ffff);
    assert_eq!(
        C220VecArithmeticHint::from_word(C220_CAPTURED_VMUL_WORD ^ (1 << 25)),
        None
    );
}

#[test]
fn divide_word_uses_its_own_dtype_selector_and_two_sources() {
    let word = 0x89dc_b619;
    let hint = C220VecArithmeticHint::from_word(word).unwrap();
    assert_eq!(hint.operation, C220VecArithmeticOperation::Divide);
    assert_eq!(hint.dtype_selector, 7);
    assert!(hint.has_fp32_value_path());
    assert_eq!(hint.source_1_register, Some(12));
    assert_eq!(
        hint.evaluate_fp32_lanes(&[3.0_f32.to_bits()], &[2.0_f32.to_bits()], &[1, 0, 0, 0],)
            .unwrap()[0]
            .bits,
        1.5_f32.to_bits()
    );
    let wrong_dtype = C220VecArithmeticHint::from_word(word & !(1 << 24)).unwrap();
    assert!(!wrong_dtype.has_fp32_value_path());
}

#[test]
fn captured_vadd_registers_map_destination_sources_and_control() {
    let hint = C220VecArithmeticHint::from_word(C220_CAPTURED_VADD_WORD).unwrap();
    assert_eq!(hint.destination_register, 16);
    assert_eq!(hint.source_0_register, 13);
    assert_eq!(hint.source_1_register, Some(14));
    assert_eq!(hint.control_register, 8);
    assert!(hint.has_fp32_value_path());
}

#[test]
fn rejects_other_route_or_leaf() {
    let add = 0x85dc_b618;
    for wrong in [add ^ (1 << 29), add ^ (3 << 25), add | 2, add | 3] {
        assert_eq!(C220VecArithmeticHint::from_word(wrong), None);
    }
}

#[test]
fn maximum_and_minimum_select_their_own_opcode_family() {
    let maximum = C220VecArithmeticHint::from_word(0x87dc_b618).unwrap();
    let minimum = C220VecArithmeticHint::from_word(0x87dc_b619).unwrap();
    assert_eq!(maximum.operation, C220VecArithmeticOperation::Maximum);
    assert_eq!(minimum.operation, C220VecArithmeticOperation::Minimum);
    assert!(maximum.has_fp32_value_path());
    assert!(minimum.has_fp32_value_path());
    let first = [(-0.0_f32).to_bits(), 0x7fc0_1234, f32::INFINITY.to_bits()];
    let second = [
        0.0_f32.to_bits(),
        1.0_f32.to_bits(),
        f32::NEG_INFINITY.to_bits(),
    ];
    let mask = [0b111, 0, 0, 0];
    let max_lanes = maximum.evaluate_fp32_lanes(&first, &second, &mask).unwrap();
    let min_lanes = minimum.evaluate_fp32_lanes(&first, &second, &mask).unwrap();
    assert_eq!(
        max_lanes.iter().map(|lane| lane.bits).collect::<Vec<_>>(),
        [0, 0x7fff_ffff, f32::INFINITY.to_bits()]
    );
    assert_eq!(
        min_lanes.iter().map(|lane| lane.bits).collect::<Vec<_>>(),
        [0x8000_0000, 0x7fff_ffff, f32::NEG_INFINITY.to_bits()]
    );
    assert!(max_lanes[1].status.unwrap().nan_operand);
    assert!(min_lanes[2].status.unwrap().infinity_operand);
}

#[test]
fn dtype_selector_only_enables_the_supported_fp32_value_path() {
    let base = 0x85dc_b618 & !(3 << 22);
    for selector in 0..4 {
        let hint = C220VecArithmeticHint::from_word(base | (selector << 22)).unwrap();
        assert_eq!(hint.dtype_selector, selector as u8);
        assert_eq!(hint.has_fp32_value_path(), selector == 3);
    }
}

#[test]
fn captured_fp32_words_reach_the_masked_value_stage() {
    let first = [1.0_f32.to_bits(), 2.0_f32.to_bits()];
    let second = [3.0_f32.to_bits(), 4.0_f32.to_bits()];
    let mask = [1, 0, 0, 0];
    let add = C220VecArithmeticHint::from_word(0x85dc_b618).unwrap();
    let sub = C220VecArithmeticHint::from_word(0x85dc_b619).unwrap();
    let add_relu = C220VecArithmeticHint::from_word(0x94c0_0000).unwrap();
    let sub_relu = C220VecArithmeticHint::from_word(0x94c0_0001).unwrap();
    let added = add.evaluate_fp32_lanes(&first, &second, &mask).unwrap();
    let subtracted = sub.evaluate_fp32_lanes(&first, &second, &mask).unwrap();
    let added_rectified = add_relu
        .evaluate_fp32_lanes(&first, &second, &mask)
        .unwrap();
    let subtracted_rectified = sub_relu
        .evaluate_fp32_lanes(&first, &second, &mask)
        .unwrap();
    assert_eq!(added[0].bits, 4.0_f32.to_bits());
    assert_eq!(subtracted[0].bits, (-2.0_f32).to_bits());
    assert_eq!(added_rectified[0].bits, 4.0_f32.to_bits());
    assert_eq!(subtracted_rectified[0].bits, 0);
    assert_eq!(added[1].bits, 0);
    assert_eq!(subtracted[1].bits, 0);
}

#[test]
fn captured_movev_add_and_sub_share_live_ub_state() {
    let x = (0..32_u32)
        .flat_map(|lane| (lane as f32).to_le_bytes())
        .collect::<Vec<_>>();
    let y = (0..32)
        .flat_map(|_| 0.5_f32.to_le_bytes())
        .collect::<Vec<_>>();
    let mut ub = UbMemory::new(512, 256);
    ub.write_states(
        0,
        &x.iter()
            .copied()
            .map(MemoryByteState::Known)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    ub.write_states(
        0x80,
        &y.iter()
            .copied()
            .map(MemoryByteState::Known)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let scalar_word = (-123.0_f32).to_bits();
    let fill = execute_c220_movev_to_ub(
        0x1131_2648,
        C220_CAPTURED_MOVEV_WORD,
        movev_control(),
        0x100,
        scalar_word,
        &[[u64::MAX; 4]],
        &mut ub,
    )
    .unwrap();
    assert_eq!(fill.stores.len(), 64);
    assert_eq!(&fill.stores[0].data[..4], &scalar_word.to_le_bytes());
    let add = execute_c220_fp32_to_ub(
        0x1131_2660,
        C220_CAPTURED_VADD_WORD,
        fp32_control(),
        addresses(0, 0x80, 0x100),
        &[[0x5555_5555, 0, 0, 0]],
        &mut ub,
    )
    .unwrap();
    assert_eq!(add.source_0_bytes[..128], x);
    assert_eq!(add.source_0_bytes[128..], [0; 128]);
    assert_eq!(add.source_1_bytes[..128], y);
    assert_eq!(add.source_1_bytes[128..], [0; 128]);
    assert_eq!(add.stores.len(), 16);
    let prior_sub = ub.read_known(0x100, 128).unwrap();
    for lane in 0_usize..32 {
        let at = lane * 4;
        if lane.is_multiple_of(2) {
            assert_eq!(prior_sub[at..at + 4], (lane as f32 + 0.5).to_le_bytes());
        } else {
            assert_eq!(prior_sub[at..at + 4], scalar_word.to_le_bytes());
        }
    }

    let sub = execute_c220_fp32_to_ub(
        0x1131_2660,
        C220_CAPTURED_VSUB_WORD,
        fp32_control(),
        addresses(0, 0x80, 0x100),
        &[[0xffff_ffff, 0, 0, 0]],
        &mut ub,
    )
    .unwrap();
    assert_eq!(sub.source_1_bytes[128..], [0; 128]);
    assert_eq!(sub.stores.len(), 32);
    let output = ub.read_known(0x100, 128).unwrap();
    for lane in 0..32 {
        let at = lane * 4;
        assert_eq!(output[at..at + 4], (lane as f32 - 0.5).to_le_bytes());
    }

    let before = ub.clone();
    assert!(matches!(
        execute_c220_fp32_to_ub(
            0,
            C220_CAPTURED_VADD_WORD | 2,
            fp32_control(),
            addresses(0, 0x80, 0x100),
            &[[u64::MAX, 0, 0, 0]],
            &mut ub,
        ),
        Err(C220VectorError::UnsupportedWord { .. })
    ));
    assert_eq!(ub, before);
    assert!(matches!(
        execute_c220_fp32_to_ub(
            0,
            C220_CAPTURED_VADD_WORD,
            fp32_control(),
            addresses(0, 0x80, u64::MAX - 1),
            &[[u64::MAX, 0, 0, 0]],
            &mut ub,
        ),
        Err(C220VectorError::Ub(UbMemoryError::RangeOverflow))
    ));
    assert_eq!(ub, before);

    ub.write_states(0x17f, &[MemoryByteState::Unknown]).unwrap();
    assert_eq!(
        ub.read_known(0x17f, 1),
        Err(UbMemoryError::UnknownByte { address: 0x17f })
    );
    let accesses = plan_c220_vector_arithmetic_read_accesses(
        C220VecArithmeticHint::from_word(C220_CAPTURED_VADD_WORD).unwrap(),
        fp32_control(),
        addresses(0, 0x80, 0x100),
        0,
        &[0xffff_ffff, 0, 0, 0],
        None,
    )
    .unwrap();
    assert_eq!(accesses.len(), 8);
    assert!(accesses.iter().all(|access| access.block_index < 4));
    execute_c220_fp32_to_ub(
        0,
        C220_CAPTURED_VSUB_WORD,
        fp32_control(),
        addresses(0, 0x80, 0x100),
        &[[0xffff_ffff, 0, 0, 0]],
        &mut ub,
    )
    .unwrap();

    ub.write_states(0xff, &[MemoryByteState::Unknown]).unwrap();
    let before = ub.clone();
    assert!(matches!(
        execute_c220_fp32_to_ub(
            0,
            C220_CAPTURED_VSUB_WORD,
            fp32_control(),
            addresses(0, 0x80, 0x100),
            &[[0xffff_ffff, 0, 0, 0]],
            &mut ub,
        ),
        Err(C220VectorError::Ub(UbMemoryError::UnknownByte {
            address: 0xff
        }))
    ));
    assert_eq!(ub, before);
}

#[test]
fn full_c220_tile_reaches_last_fp32_lane() {
    let mut ub = UbMemory::new(768, 256);
    let ones = 1.0_f32.to_le_bytes().repeat(64);
    let zeros = [MemoryByteState::Known(0); 256];
    ub.write_states(
        0,
        &ones
            .iter()
            .copied()
            .map(MemoryByteState::Known)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    ub.write_states(0x100, &zeros).unwrap();
    let mask = [1_u64 << 63, 0, 0, 0];
    let fill = execute_c220_movev_to_ub(
        0,
        C220_CAPTURED_MOVEV_WORD,
        movev_control(),
        0x100,
        2.0_f32.to_bits(),
        &[mask],
        &mut ub,
    )
    .unwrap();
    assert_eq!(fill.stores[0].address, 0x1fc);
    let add = execute_c220_fp32_to_ub(
        4,
        C220_CAPTURED_VADD_WORD,
        fp32_control(),
        addresses(0, 0x100, 0x200),
        &[mask],
        &mut ub,
    )
    .unwrap();
    assert_eq!(add.lanes.len(), 64);
    assert_eq!(add.stores[0].address, 0x2fc);
    assert_eq!(ub.read_known(0x2fc, 4).unwrap(), 3.0_f32.to_le_bytes());
}

#[test]
fn single_repeat_block_strides_change_vector_addresses() {
    let mut ub = UbMemory::new(1024, 256);
    let ones = 1.0_f32.to_le_bytes().repeat(8);
    let twos = 2.0_f32.to_le_bytes().repeat(8);
    for block in 0..8 {
        for (address, bytes) in [(block * 64, &ones), (0x400 + block * 96, &twos)] {
            ub.write_states(
                address,
                &bytes
                    .iter()
                    .copied()
                    .map(MemoryByteState::Known)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        }
    }
    let control = C220VectorControl {
        destination_block_stride: 2,
        source_0_block_stride: 2,
        source_1_block_stride: 3,
        ..fp32_control()
    };
    let mask = [1 | (1 << 8) | (1 << 63), 0, 0, 0];
    let result = execute_c220_fp32_to_ub(
        0,
        C220_CAPTURED_VADD_WORD,
        control,
        addresses(0, 0x400, 0x800),
        &[mask],
        &mut ub,
    )
    .unwrap();
    assert_eq!(result.stores.len(), 3);
    for store in &result.stores {
        let block = store.lane_index / 8;
        let lane = store.lane_index % 8;
        assert_eq!(store.address, 0x800 + block as u64 * 64 + lane as u64 * 4);
        assert_eq!(
            ub.read_known(store.address, 4).unwrap(),
            3.0_f32.to_le_bytes()
        );
    }
    assert_eq!(
        ub.read_states(0x820, 4).unwrap(),
        [MemoryByteState::Unknown; 4]
    );

    let movev = execute_c220_movev_to_ub(
        4,
        C220_CAPTURED_MOVEV_WORD,
        C220MovevControl {
            destination_block_stride: 2,
            ..movev_control()
        },
        0xc00,
        7,
        &[[1 | (1 << 8), 0, 0, 0]],
        &mut ub,
    )
    .unwrap();
    assert_eq!(movev.stores[0].address, 0xc00);
    assert_eq!(movev.stores[1].address, 0xc40);
}

#[test]
fn count_mask_spans_repeats_and_applies_only_the_final_tail() {
    let raw = (2_u64 << 56) | (8 << 32) | 1;
    let control = decode_c220_movev_control(raw).unwrap();
    let masks = decode_c220_repeat_masks(C220_COUNT_MASK_CONTROL, 65, 0, 64, 2).unwrap();
    assert_eq!(masks, vec![[u64::MAX, 0, 0, 0], [1, 0, 0, 0]]);

    let mut ub = UbMemory::new(512, 256);
    let step =
        execute_c220_movev_to_ub(0, C220_CAPTURED_MOVEV_WORD, control, 0, 7, &masks, &mut ub)
            .unwrap();
    assert_eq!(step.stores.len(), 65);
    assert_eq!(step.stores[64].repeat_index, 1);
    assert_eq!(step.stores[64].address, 0x100);
    assert_eq!(ub.read_known(0x100, 4).unwrap(), 7_u32.to_le_bytes());
    assert_eq!(
        ub.read_states(0x104, 4).unwrap(),
        [MemoryByteState::Unknown; 4]
    );
}

#[test]
fn fp32_repeats_read_previous_repeat_writes_on_aliasing_ub() {
    let control = decode_c220_fp32_control((2_u64 << 56) | (1 << 16) | (1 << 8) | 1).unwrap();
    let mut ub = UbMemory::new(512, 256);
    let ones = 1.0_f32.to_le_bytes().repeat(64);
    let twos = 2.0_f32.to_le_bytes().repeat(64);
    ub.write_states(
        0,
        &ones
            .into_iter()
            .map(MemoryByteState::Known)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    ub.write_states(
        0x100,
        &twos
            .into_iter()
            .map(MemoryByteState::Known)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let step = execute_c220_fp32_to_ub(
        0,
        C220_CAPTURED_VADD_WORD,
        control,
        addresses(0, 0x100, 0),
        &[[1, 0, 0, 0]; 2],
        &mut ub,
    )
    .unwrap();
    assert_eq!(step.stores.len(), 2);
    assert_eq!(step.stores[1].repeat_index, 1);
    assert_eq!(step.source_0_bytes.len(), 512);
    assert_eq!(step.source_0_bytes[256..260], 3.0_f32.to_le_bytes());
    assert_eq!(ub.read_known(0, 4).unwrap(), 5.0_f32.to_le_bytes());
}

#[test]
fn fp32_repeat_strides_advance_each_operand_independently() {
    let raw = (2_u64 << 56) | (16 << 40) | (8 << 32) | (8 << 24) | (1 << 16) | (1 << 8) | 1;
    let control = decode_c220_fp32_control(raw).unwrap();
    let masks = decode_c220_repeat_masks(C220_COUNT_MASK_CONTROL, 65, 0, 64, 2).unwrap();
    let mut ub = UbMemory::new(2048, 256);
    for (address, value) in [(0, 1.0_f32), (0x100, 3.0), (0x400, 2.0), (0x600, 4.0)] {
        let states = value
            .to_le_bytes()
            .repeat(64)
            .into_iter()
            .map(MemoryByteState::Known)
            .collect::<Vec<_>>();
        ub.write_states(address, &states).unwrap();
    }
    let step = execute_c220_fp32_to_ub(
        0,
        C220_CAPTURED_VADD_WORD,
        control,
        addresses(0, 0x400, 0x800),
        &masks,
        &mut ub,
    )
    .unwrap();
    assert_eq!(step.stores.len(), 65);
    assert_eq!(step.stores[64].address, 0x900);
    assert_eq!(step.stores[64].repeat_index, 1);
    assert_eq!(step.source_0_bytes[256..260], 3.0_f32.to_le_bytes());
    assert_eq!(step.source_1_bytes[256..260], 4.0_f32.to_le_bytes());
    assert_eq!(ub.read_known(0x800, 4).unwrap(), 3.0_f32.to_le_bytes());
    assert_eq!(ub.read_known(0x900, 4).unwrap(), 7.0_f32.to_le_bytes());
    assert_eq!(
        ub.read_states(0x904, 4).unwrap(),
        [MemoryByteState::Unknown; 4]
    );
}
