use crate::isa::c310::layout::C310_PB_SLOT_BYTES;
use crate::isa::c310::vector::{
    C310_CAPTURED_PLT32_WORD, C310_CAPTURED_SMOVI32_WORD, C310_CAPTURED_SUB_VST_WORD,
    C310_CAPTURED_VDUPS_WORD, C310_CAPTURED_VLD_V0_WORD, C310_CAPTURED_VLD_V1_WORD,
    C310_CAPTURED_VLDI_V0_WORD, C310_CAPTURED_VLDI_V1_WORD, C310_CAPTURED_VST_WORD,
    C310_MASK0_SPR_INDEX, C310_MASK1_SPR_INDEX, C310ObservedMovemaskHint, C310RvecArithmeticHint,
    C310RvecArithmeticOperation, C310RvecMovpHint, C310RvecVstiHint,
};

use crate::memory::ub::{UbMemory, UbMemoryError};

use crate::numeric::fp32::Fp32VectorError;
use crate::sim::c310::address::C310RvecAddressState;

use super::*;
use crate::architecture::Architecture;
use crate::memory::sparse::MemoryByteState;
use crate::sim::common::scalar::ScalarMachine;

#[test]
fn captured_duplicate_store_and_load_preserve_overlapping_ub_bytes() {
    let old = (-123.0_f32).to_bits();
    let active_mask = [u64::MAX, u64::MAX, 0, 0];
    let mut machine =
        C310RvecValueMachine::from_vector_words(vec![vec![7; 64], vec![0; 64]]).unwrap();
    let duplicate = machine
        .execute_captured_vdups_word(0x10d0_d908, C310_CAPTURED_VDUPS_WORD, old, &active_mask)
        .unwrap();
    assert_eq!(duplicate.written_lanes, (0..32).collect::<Vec<_>>());
    assert_eq!(&machine.vector_register(0).unwrap()[..32], &[old; 32]);
    assert_eq!(&machine.vector_register(0).unwrap()[32..], &[7; 32]);

    let mut ub = UbMemory::new(384, 256);
    ub.write_states(0x80, &[MemoryByteState::Known(0x5a); 128])
        .unwrap();
    let store = machine
        .execute_captured_vst_word(
            0x10d0_d918,
            C310_CAPTURED_VST_WORD,
            0x100,
            &active_mask,
            &mut ub,
        )
        .unwrap();
    assert_eq!(store.stores.len(), 32);
    assert_eq!(store.stores[0].buffer_address, 0x100);
    assert_eq!(store.stores[31].buffer_address, 0x17c);
    assert_eq!(ub.read_known(0x80, 128).unwrap(), [0x5a; 128]);
    assert_eq!(
        ub.read_known(0x100, 128).unwrap(),
        old.to_le_bytes().repeat(32)
    );

    let mut xregs = [0_u64; 32];
    xregs[8] = 0x80;
    let load = machine
        .execute_captured_vector_load_word(0x10d0_db04, C310_CAPTURED_VLDI_V1_WORD, &xregs, &ub)
        .unwrap();
    assert_eq!(&load.loaded_bytes[..128], &[0x5a; 128]);
    assert_eq!(&load.loaded_bytes[128..], old.to_le_bytes().repeat(32));
}

#[test]
fn captured_sub_vld_and_vst_use_their_distinct_words() {
    let bytes = (0..384)
        .map(|index| (index & 0xff) as u8)
        .collect::<Vec<_>>();
    let mut ub = UbMemory::new(384, 384);
    ub.write_states(
        0,
        &bytes
            .iter()
            .copied()
            .map(MemoryByteState::Known)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let mut xregs = [0_u64; 32];
    xregs[12] = 0;
    xregs[16] = 0x80;
    let mut machine = C310RvecValueMachine::from_vector_words(vec![vec![0; 64]; 2]).unwrap();
    for (pc, word, destination, source) in [
        (0x10d0_d90c, C310_CAPTURED_VLD_V0_WORD, 0, 12),
        (0x10d0_d910, C310_CAPTURED_VLD_V1_WORD, 1, 16),
    ] {
        let step = machine
            .execute_captured_vector_load_word(pc, word, &xregs, &ub)
            .unwrap();
        assert_eq!(step.hint.destination_v_register, destination);
        assert_eq!(step.hint.source_s_register, source / 2);
        assert_eq!(step.hint.source_a_register, Some(0));
        let start = xregs[source as usize] as usize;
        assert_eq!(step.loaded_bytes, bytes[start..start + 256]);
    }
    let first = machine.vector_register(0).unwrap().to_vec();
    let active_mask = [0x1111_1111_1111_1111, 0x1111_1111_1111_1111, 0, 0];
    let stored = machine
        .execute_captured_vst_word(
            0x10d0_d91c,
            C310_CAPTURED_SUB_VST_WORD,
            0x100,
            &active_mask,
            &mut ub,
        )
        .unwrap();
    assert_eq!(stored.stores.len(), 32);
    assert_eq!(stored.source_v_register, 0);
    assert_eq!(
        ub.read_known(0x100, 128).unwrap(),
        first[..32]
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>()
    );
}

#[test]
fn captured_loads_resolve_pb_scalars_and_only_vld_uses_a0() {
    let mut machine = C310RvecValueMachine::from_vector_words(vec![vec![0; 64]; 2]).unwrap();
    let mut slot = [0_u8; C310_PB_SLOT_BYTES];
    slot[..4].copy_from_slice(&0x1e_u32.to_le_bytes());
    for (index, value) in [0x100_u32, 0x80, 0, 0x80].into_iter().enumerate() {
        let start = 4 + index * 4;
        slot[start..start + 4].copy_from_slice(&value.to_le_bytes());
    }
    machine.apply_pb_scalar_init(&slot);
    let mut address = C310RvecAddressState::default();
    address
        .configure_vag_word(
            0x10d0_d900,
            crate::sim::c310::address::C310_CAPTURED_VAG_WORD,
        )
        .unwrap();
    address.start_vloop_i1(&machine).unwrap();

    let bytes = (0..640)
        .map(|index| (index & 0xff) as u8)
        .collect::<Vec<_>>();
    let mut ub = UbMemory::new(640, 640);
    ub.write_states(
        0,
        &bytes
            .iter()
            .copied()
            .map(MemoryByteState::Known)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    for (pc, word, expected_s, expected_a, expected_address) in [
        (0x10d0_db00, C310_CAPTURED_VLDI_V0_WORD, 2, None, 0x100),
        (0x10d0_db04, C310_CAPTURED_VLDI_V1_WORD, 4, None, 0x80),
        (0x10d0_d90c, C310_CAPTURED_VLD_V0_WORD, 6, Some(0), 0),
        (0x10d0_d910, C310_CAPTURED_VLD_V1_WORD, 8, Some(0), 0x80),
    ] {
        let resolved = machine
            .resolve_captured_vector_load_address(pc, word, &address)
            .unwrap();
        assert_eq!(resolved.source_s_register, expected_s);
        assert_eq!(resolved.address_a0, expected_a);
        assert_eq!(resolved.effective_address, expected_address);
        let step = machine
            .execute_captured_vector_load_from_scalar_state(pc, word, &address, &ub)
            .unwrap();
        assert_eq!(step.source_address, expected_address);
        assert_eq!(step.loaded_bytes, bytes[expected_address as usize..][..256]);
    }

    address.update_i1(1, &machine).unwrap();
    let vld = machine
        .execute_captured_vector_load_from_scalar_state(
            0x10d0_d910,
            C310_CAPTURED_VLD_V1_WORD,
            &address,
            &ub,
        )
        .unwrap();
    assert_eq!(vld.source_address, 0x180);
    let vldi = machine
        .execute_captured_vector_load_from_scalar_state(
            0x10d0_db04,
            C310_CAPTURED_VLDI_V1_WORD,
            &address,
            &ub,
        )
        .unwrap();
    assert_eq!(vldi.source_address, 0x80);
}

#[test]
fn scalar_resolved_loads_fail_closed_before_touching_vector_state() {
    let mut machine = C310RvecValueMachine::from_vector_words(vec![vec![9; 64]; 2]).unwrap();
    let address = C310RvecAddressState::default();
    let ub = UbMemory::new(256, 256);
    let before = machine.clone();
    assert_eq!(
        machine.execute_captured_vector_load_from_scalar_state(
            0x10d0_db00,
            C310_CAPTURED_VLDI_V0_WORD,
            &address,
            &ub,
        ),
        Err(C310CapturedVectorLoadError::MissingScalar { index: 2 })
    );
    assert_eq!(machine, before);

    let mut slot = [0_u8; C310_PB_SLOT_BYTES];
    slot[..4].copy_from_slice(&(1_u32 << 3).to_le_bytes());
    machine.apply_pb_scalar_init(&slot);
    let before = machine.clone();
    assert_eq!(
        machine.execute_captured_vector_load_from_scalar_state(
            0x10d0_d90c,
            C310_CAPTURED_VLD_V0_WORD,
            &address,
            &ub,
        ),
        Err(C310CapturedVectorLoadError::MissingAddressA0)
    );
    assert_eq!(machine, before);
}

#[test]
fn scalar_resolved_loads_combine_high_halves_and_wrap_with_a0() {
    let mut machine = C310RvecValueMachine::from_vector_words(vec![vec![0; 64]; 2]).unwrap();
    let mut slot = [0_u8; C310_PB_SLOT_BYTES];
    slot[..4].copy_from_slice(&((1_u32 << 1) | (1_u32 << 3)).to_le_bytes());
    slot[4..8].copy_from_slice(&0x0001_2345_u32.to_le_bytes());
    slot[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
    machine.apply_pb_scalar_init(&slot);
    let mut address = C310RvecAddressState::default();
    address
        .configure_vag_word(
            0x10d0_d900,
            crate::sim::c310::address::C310_CAPTURED_VAG_WORD,
        )
        .unwrap();
    address.update_i1(2, &machine).unwrap();

    let immediate = machine
        .resolve_captured_vector_load_address(0x10d0_db00, C310_CAPTURED_VLDI_V0_WORD, &address)
        .unwrap();
    assert_eq!(immediate.source_scalar_low, 0x2345);
    assert_eq!(immediate.source_scalar_high, 1);
    assert_eq!(immediate.address_a0, None);
    assert_eq!(immediate.effective_address, 0x1_2345);

    let normal = machine
        .resolve_captured_vector_load_address(0x10d0_d90c, C310_CAPTURED_VLD_V0_WORD, &address)
        .unwrap();
    assert_eq!(normal.source_scalar_low, 0xffff);
    assert_eq!(normal.source_scalar_high, 0xffff);
    assert_eq!(normal.address_a0, Some(0x2_468a));
    assert_eq!(normal.effective_address, 0x2_4689);
}

#[test]
fn captured_plt32_writes_the_sub_predicate_without_partial_failure() {
    let mut machine = C310RvecValueMachine::from_vector_and_predicate_bytes(
        vec![vec![0; 64]],
        vec![vec![0xff; 32], vec![0xff; 32]],
    )
    .unwrap();
    let before = machine.clone();
    assert_eq!(
        machine.execute_captured_plt32_word(0x10d0_d914, C310_CAPTURED_PLT32_WORD),
        Err(C310CapturedPltError::MissingScalarLimit)
    );
    assert_eq!(machine, before);
    let smovi = machine
        .execute_captured_smovi_word(0x10d0_d904, C310_CAPTURED_SMOVI32_WORD)
        .unwrap();
    assert_eq!(smovi.destination_s_register, 65);
    assert_eq!(smovi.prior_value, None);
    assert_eq!(smovi.value, 32);
    assert_eq!(machine.captured_s65(), Some(32));
    let step = machine
        .execute_captured_plt32_word(0x10d0_d914, C310_CAPTURED_PLT32_WORD)
        .unwrap();
    assert_eq!(step.destination_p_register, 1);
    assert_eq!(step.lane_limit, 32);
    assert_eq!(step.remaining_scalar_value, 0);
    assert_eq!(machine.captured_s65(), Some(0));
    assert_eq!(&step.predicate_bytes[..16], &[0x11; 16]);
    assert_eq!(&step.predicate_bytes[16..], &[0; 16]);
    assert_eq!(machine.predicate_register(0), Some([0xff; 32].as_slice()));
    assert_eq!(
        machine.predicate_register(1),
        Some(step.predicate_bytes.as_slice())
    );
    let before = machine.clone();
    assert_eq!(
        machine.execute_captured_plt32_word(0x10d0_d914, C310_CAPTURED_PLT32_WORD ^ 1),
        Err(C310CapturedPltError::UnsupportedWord {
            pc: 0x10d0_d914,
            word: C310_CAPTURED_PLT32_WORD ^ 1,
        })
    );
    assert_eq!(machine, before);
    assert_eq!(
        machine.execute_captured_smovi_word(0x10d0_d904, C310_CAPTURED_SMOVI32_WORD ^ 1),
        Err(C310CapturedSmoviError::UnsupportedWord {
            pc: 0x10d0_d904,
            word: C310_CAPTURED_SMOVI32_WORD ^ 1,
        })
    );
    assert_eq!(machine, before);

    let repeated = machine
        .execute_captured_smovi_word(0x10d0_d904, C310_CAPTURED_SMOVI32_WORD)
        .unwrap();
    assert_eq!(repeated.prior_value, Some(0));
    assert_eq!(machine.captured_s65(), Some(32));

    let mut short = C310RvecValueMachine::from_vector_and_predicate_bytes(
        vec![vec![0; 32]],
        vec![vec![0; 32], vec![0; 32]],
    )
    .unwrap();
    let before = short.clone();
    assert_eq!(
        short.execute_captured_plt32_word(0x10d0_d914, C310_CAPTURED_PLT32_WORD),
        Err(C310CapturedPltError::RegisterWidth { actual: 128 })
    );
    assert_eq!(short, before);

    let mut missing =
        C310RvecValueMachine::from_vector_and_predicate_bytes(vec![vec![0; 64]], vec![vec![0; 32]])
            .unwrap();
    assert_eq!(
        missing.execute_captured_plt32_word(0x10d0_d914, C310_CAPTURED_PLT32_WORD),
        Err(C310CapturedPltError::MissingP1 { count: 1 })
    );
}

#[test]
fn pb_scalar_init_preserves_unwritten_registers_and_feeds_vector_steps() {
    let mut machine = C310RvecValueMachine::from_vector_and_predicate_bytes(
        vec![vec![0; 64]],
        vec![vec![0; 32], vec![0; 32]],
    )
    .unwrap();
    let mut slot = [0_u8; C310_PB_SLOT_BYTES];
    slot[..4].copy_from_slice(&((1_u32 << 1) | (1_u32 << 3)).to_le_bytes());
    slot[4..8].copy_from_slice(&0x0000_0100_u32.to_le_bytes());
    slot[8..12].copy_from_slice(&0x7654_3210_u32.to_le_bytes());
    let projection = machine.apply_pb_scalar_init(&slot);
    assert_eq!(projection.consumed_payload_words, 2);
    assert_eq!(machine.scalar_register(2), Some(0x100));
    assert_eq!(machine.scalar_register(3), Some(0));
    assert_eq!(machine.scalar_register(65), Some(0x100));
    assert_eq!(machine.scalar_register(6), Some(0x3210));
    assert_eq!(machine.scalar_register(67), Some(0x7654_3210));
    assert_eq!(machine.scalar_register(4), None);
    assert_eq!(machine.scalar_register(C310_SCALAR_REGISTER_COUNT), None);

    slot[..4].copy_from_slice(&(1_u32 << 3).to_le_bytes());
    slot[4..8].copy_from_slice(&0x89ab_cdef_u32.to_le_bytes());
    machine.apply_pb_scalar_init(&slot);
    assert_eq!(machine.scalar_register(2), Some(0x100));
    assert_eq!(machine.scalar_register(65), Some(0x100));
    assert_eq!(machine.scalar_register(67), Some(0x89ab_cdef));

    let smovi = machine
        .execute_captured_smovi_word(0x10d0_d904, C310_CAPTURED_SMOVI32_WORD)
        .unwrap();
    assert_eq!(smovi.prior_value, Some(0x100));
    assert_eq!(machine.scalar_register(65), Some(32));
    machine
        .execute_captured_plt32_word(0x10d0_d914, C310_CAPTURED_PLT32_WORD)
        .unwrap();
    assert_eq!(machine.scalar_register(65), Some(0));
    assert_eq!(machine.scalar_register(67), Some(0x89ab_cdef));
}

#[test]
fn captured_vector_steps_fail_without_partial_mutation() {
    let active_mask = [u64::MAX, u64::MAX, 0, 0];
    let mut machine = C310RvecValueMachine::from_vector_words(vec![vec![3; 64]]).unwrap();
    let before_machine = machine.clone();
    assert_eq!(
        machine.execute_captured_vdups_word(
            0x10d0_d908,
            C310_CAPTURED_VDUPS_WORD ^ 1,
            0,
            &active_mask,
        ),
        Err(C310CapturedVectorError::UnsupportedWord {
            pc: 0x10d0_d908,
            word: C310_CAPTURED_VDUPS_WORD ^ 1,
        })
    );
    assert_eq!(machine, before_machine);

    let mut ub = UbMemory::new(200, 256);
    ub.write_states(0x80, &[MemoryByteState::Known(0x5a); 128])
        .unwrap();
    let before_ub = ub.clone();
    assert_eq!(
        machine.execute_captured_vst_word(
            0x10d0_d918,
            C310_CAPTURED_VST_WORD,
            0x100,
            &active_mask,
            &mut ub,
        ),
        Err(C310CapturedVectorError::Ub(
            UbMemoryError::TrackedLimitExceeded { limit: 200 }
        ))
    );
    assert_eq!(ub, before_ub);
    assert_eq!(
        machine.execute_captured_vst_word(
            0x10d0_d918,
            C310_CAPTURED_VST_WORD,
            u64::MAX - 4,
            &active_mask,
            &mut ub,
        ),
        Err(C310CapturedVectorError::AddressOverflow {
            base: u64::MAX - 4,
            lane: 2,
        })
    );
    assert_eq!(ub, before_ub);
}

#[test]
fn captured_vldi_reads_overlapping_ub_windows_into_distinct_v_registers() {
    let bytes = (0..384)
        .map(|index| (index & 0xff) as u8)
        .collect::<Vec<_>>();
    let mut ub = UbMemory::new(384, 384);
    ub.write_states(
        0,
        &bytes
            .iter()
            .copied()
            .map(MemoryByteState::Known)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let mut xregs = [0_u64; 32];
    xregs[4] = 0;
    xregs[8] = 0x80;
    let mut machine = C310RvecValueMachine::from_vector_words(vec![vec![0; 64]; 2]).unwrap();
    for (pc, word, destination, source, expected) in [
        (0x10d0_db00, C310_CAPTURED_VLDI_V0_WORD, 0, 4, &bytes[..256]),
        (
            0x10d0_db04,
            C310_CAPTURED_VLDI_V1_WORD,
            1,
            8,
            &bytes[128..384],
        ),
    ] {
        let step = machine
            .execute_captured_vector_load_word(pc, word, &xregs, &ub)
            .unwrap();
        assert_eq!(step.hint.destination_v_register, destination);
        assert_eq!(step.hint.source_s_register, source / 2);
        assert_eq!(step.hint.source_a_register, None);
        assert_eq!(step.source_address, xregs[source as usize]);
        assert_eq!(step.loaded_bytes, expected);
        assert_eq!(
            machine.vector_register(destination as usize).unwrap(),
            expected
                .chunks_exact(4)
                .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn captured_vldi_rejects_unknown_bytes_without_mutating_registers() {
    let mut ub = UbMemory::new(256, 256);
    ub.write_states(0, &[MemoryByteState::Known(7); 255])
        .unwrap();
    let xregs = [0_u64; 32];
    let mut machine = C310RvecValueMachine::from_vector_words(vec![vec![9; 64]; 2]).unwrap();
    let before = machine.clone();
    assert_eq!(
        machine.execute_captured_vector_load_word(
            0x10d0_db00,
            C310_CAPTURED_VLDI_V0_WORD,
            &xregs,
            &ub,
        ),
        Err(C310CapturedVectorLoadError::Ub(
            UbMemoryError::UnknownByte { address: 255 }
        ))
    );
    assert_eq!(machine, before);
    assert_eq!(
        machine.execute_captured_vector_load_word(0x10d0_db00, 0, &xregs, &ub),
        Err(C310CapturedVectorLoadError::UnsupportedWord {
            pc: 0x10d0_db00,
            word: 0,
        })
    );
    assert_eq!(machine, before);
}

#[test]
fn captured_vldi_rejects_wrong_register_shape_or_bank() {
    let ub = UbMemory::new(256, 256);
    let xregs = [0_u64; 32];
    let mut short = C310RvecValueMachine::from_vector_words(vec![vec![0; 32]; 2]).unwrap();
    let before = short.clone();
    assert_eq!(
        short.execute_captured_vector_load_word(
            0x10d0_db00,
            C310_CAPTURED_VLDI_V0_WORD,
            &xregs,
            &ub,
        ),
        Err(C310CapturedVectorLoadError::RegisterWidth { actual: 128 })
    );
    assert_eq!(short, before);

    let mut one_register = C310RvecValueMachine::from_vector_words(vec![vec![0; 64]]).unwrap();
    let before = one_register.clone();
    assert_eq!(
        one_register.execute_captured_vector_load_word(
            0x10d0_db04,
            C310_CAPTURED_VLDI_V1_WORD,
            &xregs,
            &ub,
        ),
        Err(C310CapturedVectorLoadError::RegisterIndex { index: 1, count: 1 })
    );
    assert_eq!(one_register, before);
}

#[test]
fn observed_c310_movemask_words_transfer_live_source_x_values_to_mask_sprs() {
    let mut xregs = [0_u64; 32];
    xregs[0] = u64::MAX;
    xregs[9] = 0x8000_0000_0000_0001;
    xregs[13] = 0x5555_5555;
    let mut sprs = C310RvecMaskSprState::default();
    for (pc, word, source, destination, value) in [
        (0x10d0_d130, 0x15c0_0033, 0, 153, u64::MAX),
        (0x10d0_d134, 0x15c0_0013, 0, 152, u64::MAX),
        (0x10d0_d54c, 0x15c3_0033, 3, 153, 0),
        (0x10d0_d560, 0x15c9_0033, 9, 153, 0x8000_0000_0000_0001),
        (0x10d0_d564, 0x15c9_0013, 9, 152, 0x8000_0000_0000_0001),
        (0x10d0_d55c, 0x15cd_0013, 13, 152, 0x5555_5555),
    ] {
        let step = sprs
            .execute_observed_movemask_word(pc, word, &xregs)
            .unwrap();
        assert_eq!(step.hint.source_x_register, source);
        assert_eq!(step.hint.destination_spr, destination);
        assert_eq!(step.value, value);
    }
    assert_eq!(sprs.mask0, 0x5555_5555);
    assert_eq!(sprs.mask1, 0x8000_0000_0000_0001);
    let before = sprs;
    assert_eq!(
        sprs.execute_observed_movemask_word(0x10d0_d55c, 0x15ce_0012, &xregs),
        Err(C310ObservedMovemaskError::UnsupportedWord {
            pc: 0x10d0_d55c,
            word: 0x15ce_0012,
        })
    );
    assert_eq!(sprs, before);
}

#[test]
fn c310_movemask_decode_covers_each_x_register_and_mask_destination() {
    for source in 0..32_u32 {
        for (low_word, destination) in [
            (0x15c0_0013, C310_MASK0_SPR_INDEX),
            (0x15c0_0033, C310_MASK1_SPR_INDEX),
        ] {
            let word = low_word | (source << 16);
            assert_eq!(
                C310ObservedMovemaskHint::from_word(word),
                Some(C310ObservedMovemaskHint {
                    source_x_register: source as u8,
                    destination_spr: destination,
                })
            );
        }
    }
    for word in [0x15c0_0012, 0x15c0_0053, 0x15e0_0013, 0x1580_0013] {
        assert_eq!(C310ObservedMovemaskHint::from_word(word), None);
    }
}

#[test]
fn captured_scalar_movemask_movp_chain_matches_live_p1_readback() {
    let mut scalar = ScalarMachine::new(Architecture::Dav3510, [0; 32], 0);
    let mut mask_sprs = C310RvecMaskSprState::default();
    scalar.execute_word(0x10d0_d548, 0x071a_5555).unwrap();
    assert_eq!(scalar.xregs()[13], 0x5555);
    mask_sprs
        .execute_observed_movemask_word(0x10d0_d54c, 0x15c3_0033, scalar.xregs())
        .unwrap();
    scalar.execute_word(0x10d0_d550, 0x075b_5555).unwrap();
    assert_eq!(scalar.xregs()[13], 0x5555_5555);
    mask_sprs
        .execute_observed_movemask_word(0x10d0_d55c, 0x15cd_0013, scalar.xregs())
        .unwrap();
    assert_eq!(mask_sprs.mask0, 0x5555_5555);
    assert_eq!(mask_sprs.mask1, 0);
    let mut rvec = C310RvecValueMachine::from_vector_and_predicate_bytes(
        vec![vec![0; 32], vec![0; 32]],
        vec![vec![0; 32], vec![0; 32]],
    )
    .unwrap();
    let movp = rvec
        .execute_movp_u32_from_mask_sprs(0x8204_0156, &mask_sprs)
        .unwrap();
    assert_eq!(movp.scalar_mask, 0x5555_5555);
    assert_eq!(&movp.predicate_bytes[..16], &[0x0f; 16]);
    assert_eq!(&movp.predicate_bytes[16..], &[0; 16]);
}

#[test]
fn captured_vsti_plans_only_predicated_four_byte_buffer_writes() {
    let mut predicate = vec![0_u8; 32];
    predicate[..16].fill(0x0f);
    let machine = C310RvecValueMachine::from_vector_and_predicate_bytes(
        vec![(0..64_u32).collect()],
        vec![vec![0; 32], predicate],
    )
    .unwrap();
    let word = 0x4018_010a;
    let stores = machine.plan_normal_u32_vsti(word, 0x100).unwrap();
    assert_eq!(stores.len(), 16);
    assert_eq!(
        stores[0],
        C310RvecVstiStore {
            lane_index: 0,
            buffer_address: 0x100,
            data: 0_u32.to_le_bytes(),
        }
    );
    assert_eq!(stores[1].lane_index, 2);
    assert_eq!(stores[1].buffer_address, 0x108);
    assert_eq!(stores[1].data, 2_u32.to_le_bytes());
    assert_eq!(stores[15].lane_index, 30);
    assert_eq!(stores[15].buffer_address, 0x178);
    assert_eq!(stores[15].data, 30_u32.to_le_bytes());
    assert_eq!(
        machine.plan_normal_u32_vsti(word ^ 3, 0x100),
        Err(C310RvecValueError::UnsupportedVstiWord)
    );
    assert_eq!(
        machine.plan_normal_u32_vsti(word ^ (1 << 2), 0x100),
        Err(C310RvecValueError::UnsupportedVstiDtype)
    );
    assert_eq!(
        machine.plan_normal_u32_vsti(word, u64::MAX - 3),
        Err(C310RvecValueError::VstiAddressOverflow {
            base: u64::MAX - 3,
            lane: 2,
        })
    );
}

#[test]
fn normal_u32_vsti_commits_only_selected_ub_lanes_atomically() {
    let mut predicate = vec![0_u8; 32];
    predicate[..16].fill(0x0f);
    let machine = C310RvecValueMachine::from_vector_and_predicate_bytes(
        vec![(0..64_u32).collect()],
        vec![vec![0; 32], predicate],
    )
    .unwrap();
    let mut ub = UbMemory::new(128, 128);
    ub.write_states(0x100, &[MemoryByteState::Known(0xa5); 128])
        .unwrap();
    let stores = machine
        .execute_normal_u32_vsti_to_ub(0x4018_010a, 0x100, &mut ub)
        .unwrap();
    assert_eq!(stores.len(), 16);
    for lane in 0..32_u32 {
        let bytes = ub.read_known(0x100 + u64::from(lane) * 4, 4).unwrap();
        if lane % 2 == 0 {
            assert_eq!(bytes, lane.to_le_bytes());
        } else {
            assert_eq!(bytes, [0xa5; 4]);
        }
    }

    let mut limited = UbMemory::new(4, 4);
    limited
        .write_states(0x100, &[MemoryByteState::Known(0xa5)])
        .unwrap();
    let before = limited.clone();
    assert_eq!(
        machine.execute_normal_u32_vsti_to_ub(0x4018_010a, 0x100, &mut limited),
        Err(C310RvecValueError::Ub(
            UbMemoryError::TrackedLimitExceeded { limit: 4 }
        ))
    );
    assert_eq!(limited, before);
}

#[test]
fn captured_movp_word_selects_p1_and_u32_value_path() {
    let hint = C310RvecMovpHint::from_word(0x8204_0156).unwrap();
    assert_eq!(hint.destination_p_register, 1);
    assert_eq!(hint.btype, 2);
    assert!(hint.has_u32_value_path());
    assert_eq!(hint.u32_source_spr_index(), Some(C310_MASK0_SPR_INDEX));
    assert_eq!(C310_MASK1_SPR_INDEX, 153);
    let word = 0x8204_0156;
    for wrong in [
        word ^ (1 << 30),
        word ^ (1 << 20),
        word ^ (1 << 9),
        word ^ 1,
    ] {
        assert_eq!(C310RvecMovpHint::from_word(wrong), None);
    }
    let other_btype = C310RvecMovpHint::from_word(word ^ (1 << 7)).unwrap();
    assert!(!other_btype.has_u32_value_path());
    assert_eq!(other_btype.u32_source_spr_index(), None);
}

#[test]
fn movp_reads_caller_mask_spr_snapshot_with_vendor_defaults() {
    let mut machine = C310RvecValueMachine::from_vector_and_predicate_bytes(
        vec![vec![0; 4]],
        vec![vec![0; 32], vec![0; 32]],
    )
    .unwrap();
    let mut mask_sprs = C310RvecMaskSprState::default();
    assert_eq!(mask_sprs.mask0, u64::MAX);
    assert_eq!(mask_sprs.mask1, u64::MAX);
    let default = machine
        .execute_movp_u32_from_mask_sprs(0x8204_0156, &mask_sprs)
        .unwrap();
    assert_eq!(default.predicate_bytes, [0xff; 32]);

    mask_sprs.mask0 = 0x5555_5555;
    let overridden = machine
        .execute_movp_u32_from_mask_sprs(0x8204_0156, &mask_sprs)
        .unwrap();
    assert_eq!(&overridden.predicate_bytes[..16], &[0x0f; 16]);
    assert_eq!(&overridden.predicate_bytes[16..], &[0; 16]);
    assert_eq!(
        machine.predicate_register(1),
        Some(&overridden.predicate_bytes[..])
    );
}

#[test]
fn movp_updates_encoded_p_destination_before_vadd_and_vsti_store() {
    let old = (-123.0_f32).to_bits();
    let mut machine = C310RvecValueMachine::from_vector_and_predicate_bytes(
        vec![
            vec![
                1.0_f32.to_bits(),
                2.0_f32.to_bits(),
                3.0_f32.to_bits(),
                4.0_f32.to_bits(),
            ],
            vec![
                5.0_f32.to_bits(),
                6.0_f32.to_bits(),
                7.0_f32.to_bits(),
                8.0_f32.to_bits(),
            ],
        ],
        vec![vec![0; 32], vec![0; 32]],
    )
    .unwrap();
    let movp = machine.execute_movp_u32_word(0x8204_0156, 0b0101).unwrap();
    assert_eq!(movp.hint.destination_p_register, 1);
    assert_eq!(&movp.predicate_bytes[..2], &[0x0f, 0x0f]);
    assert_eq!(
        machine.predicate_register(1),
        Some(movp.predicate_bytes.as_slice())
    );
    let add = machine
        .execute_fp32_word_from_predicate_registers(0x8008_2780)
        .unwrap();
    assert_eq!(
        add.writeback.words,
        [6.0_f32.to_bits(), 0, 10.0_f32.to_bits(), 0]
    );
    let store_hint = C310RvecVstiHint::from_word(0x4018_010a).unwrap();
    assert!(store_hint.has_normal_u32_store_path());
    assert_eq!(
        store_hint.predicate_register,
        movp.hint.destination_p_register
    );
    let store = c310_normal_u32_masked_store(
        &[old; 4],
        &add.writeback.words,
        machine
            .predicate_register(usize::from(store_hint.predicate_register))
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        store.words,
        [6.0_f32.to_bits(), old, 10.0_f32.to_bits(), old]
    );
}

#[test]
fn movp_rejects_missing_or_short_predicate_bank_without_mutation() {
    let mut missing = C310RvecValueMachine::from_vector_words(vec![vec![0]]).unwrap();
    assert_eq!(
        missing.execute_movp_u32_word(0x8204_0156, 1),
        Err(C310RvecValueError::MissingPredicateBank)
    );
    let mut short = C310RvecValueMachine::from_vector_and_predicate_bytes(
        vec![vec![0]],
        vec![vec![0; 32], vec![0; 16]],
    )
    .unwrap();
    assert_eq!(
        short.execute_movp_u32_word(0x8204_0156, 1),
        Err(C310RvecValueError::MovpPredicateWidth {
            expected: 32,
            actual: 16,
        })
    );
    assert_eq!(short.predicate_register(1), Some([0; 16].as_slice()));
}

#[test]
fn captured_vsti_word_decodes_to_normal_u32_store_path() {
    let hint = C310RvecVstiHint::from_word(0x4018_010a).unwrap();
    assert_eq!(hint.source_v_register, 0);
    assert_eq!(hint.scalar_register, 3);
    assert_eq!(hint.offset, 0);
    assert_eq!(hint.predicate_register, 1);
    assert!(!hint.p);
    assert_eq!(hint.distance, 2);
    assert_eq!(hint.dtype_code(), 26);
    assert!(hint.has_normal_u32_store_path());
}

#[test]
fn vsti_decoder_rejects_neighboring_fixed_leaves() {
    let word = 0x4018_010a;
    for wrong in [word ^ (1 << 30), word ^ (1 << 6), word ^ 1] {
        assert_eq!(C310RvecVstiHint::from_word(wrong), None);
    }
    let other_dist = C310RvecVstiHint::from_word(word ^ (1 << 2)).unwrap();
    assert_eq!(other_dist.dtype_code(), 27);
    assert!(!other_dist.has_normal_u32_store_path());
}

#[test]
fn add_and_sub_vendor_trace_words_select_distinct_registered_leaves() {
    let add = C310RvecArithmeticHint::from_word(0x8008_2780).unwrap();
    let subtract = C310RvecArithmeticHint::from_word(0x8008_2781).unwrap();
    assert_eq!(add.operation, C310RvecArithmeticOperation::Add);
    assert_eq!(subtract.operation, C310RvecArithmeticOperation::Subtract);
    assert_eq!(add.destination_v_register, 0);
    assert_eq!(add.first_source_v_register, 0);
    assert_eq!(add.second_source_v_register, 1);
    assert_eq!(add.predicate_register, 1);
    assert_eq!(add.dtype_selector, 7);
    assert!(add.has_fp32_value_path());
    assert!(subtract.has_fp32_value_path());
}

#[test]
fn multiply_word_selects_predicated_fp32_lane_path() {
    let word = 0x8000_27c0;
    let hint = C310RvecArithmeticHint::from_word(word).unwrap();
    assert_eq!(hint.operation, C310RvecArithmeticOperation::Multiply);
    assert_eq!(hint.destination_v_register, 0);
    assert_eq!(hint.first_source_v_register, 0);
    assert_eq!(hint.second_source_v_register, 1);
    assert_eq!(hint.predicate_register, 1);
    assert!(hint.has_fp32_value_path());
    let mut machine = C310RvecValueMachine::from_vector_and_predicate_bytes(
        vec![
            vec![2.0_f32.to_bits(), 0],
            vec![3.0_f32.to_bits(), f32::INFINITY.to_bits()],
        ],
        vec![vec![0; 32], vec![0x11, 0, 0, 0]],
    )
    .unwrap();
    let step = machine
        .execute_fp32_word_from_predicate_registers(word)
        .unwrap();
    assert_eq!(step.writeback.words, [6.0_f32.to_bits(), 0x7fff_ffff]);
    for wrong in [word ^ (1 << 6), word ^ (1 << 19), word | 1] {
        assert_eq!(C310RvecArithmeticHint::from_word(wrong), None);
    }
}

#[test]
fn only_registered_fixed_opcode_fields_are_accepted() {
    let add = 0x8008_2780;
    for wrong in [add ^ (1 << 30), add ^ (1 << 19), add ^ (1 << 6), add | 2] {
        assert_eq!(C310RvecArithmeticHint::from_word(wrong), None);
    }
}

#[test]
fn variable_register_and_dtype_fields_do_not_change_opcode_leaf() {
    let word = 0x8008_2780 | (3 << 25) | (4 << 20) | (5 << 13) | (2 << 10) | (1 << 18);
    let hint = C310RvecArithmeticHint::from_word(word).unwrap();
    assert_eq!(hint.operation, C310RvecArithmeticOperation::Add);
    assert_eq!(hint.destination_v_register, 3);
    assert_eq!(hint.first_source_v_register, 4);
    assert_eq!(hint.second_source_v_register, 5);
    assert_eq!(hint.predicate_register, 3);
    assert_eq!(hint.dtype_selector, 15);
    assert!(!hint.has_fp32_value_path());
    assert_eq!(
        evaluate_c310_fp32_lanes(hint, &[], &[], &[0; 4]),
        Err(Fp32VectorError::UnsupportedInstruction)
    );
}

#[test]
fn captured_fp32_words_reach_the_masked_value_stage() {
    let first = [1.0_f32.to_bits(), 2.0_f32.to_bits()];
    let second = [3.0_f32.to_bits(), 4.0_f32.to_bits()];
    let mask = [1, 0, 0, 0];
    let add = C310RvecArithmeticHint::from_word(0x8008_2780).unwrap();
    let sub = C310RvecArithmeticHint::from_word(0x8008_2781).unwrap();
    let added = evaluate_c310_fp32_lanes(add, &first, &second, &mask).unwrap();
    let subtracted = evaluate_c310_fp32_lanes(sub, &first, &second, &mask).unwrap();
    assert_eq!(added[0].bits, 4.0_f32.to_bits());
    assert_eq!(subtracted[0].bits, (-2.0_f32).to_bits());
    assert_eq!(added[1].bits, 0);
    assert_eq!(subtracted[1].bits, 0);
}

#[test]
fn predicate_bytes_fill_the_256_bit_mask_low_bit_first() {
    let mut bytes = [0_u8; 32];
    bytes[0] = 0b0001_0001;
    bytes[1] = 0b1000_0000;
    bytes[8] = 1;
    bytes[31] = 0b1000_0000;
    assert_eq!(
        c310_predicate_bytes_to_mask(&bytes).unwrap(),
        [0x8011, 1, 0, 1_u64 << 63]
    );
    assert_eq!(c310_predicate_bytes_to_mask(&[]).unwrap(), [0; 4]);
    assert_eq!(
        c310_predicate_bytes_to_mask(&[0; 33]),
        Err(C310RvecValueError::PredicateWidth { bytes: 33 })
    );
}

#[test]
fn predicate_bank_selects_encoded_p_register_before_add() {
    let mut machine = C310RvecValueMachine::from_vector_and_predicate_bytes(
        vec![
            vec![1.0_f32.to_bits(), 2.0_f32.to_bits()],
            vec![3.0_f32.to_bits(), 4.0_f32.to_bits()],
        ],
        vec![vec![0; 32], vec![0x01, 0, 0, 0]],
    )
    .unwrap();
    assert_eq!(machine.predicate_register(1), Some(&[1, 0, 0, 0][..]));
    let step = machine
        .execute_fp32_word_from_predicate_registers(0x8008_2780)
        .unwrap();
    assert_eq!(step.hint.predicate_register, 1);
    assert_eq!(step.active_mask, [1, 0, 0, 0]);
    assert_eq!(step.writeback.words, [4.0_f32.to_bits(), 0]);

    let masked_by_p0 = machine
        .execute_fp32_word_from_predicate_registers(0x8008_2380)
        .unwrap();
    assert_eq!(masked_by_p0.hint.predicate_register, 0);
    assert_eq!(masked_by_p0.active_mask, [0; 4]);
    assert_eq!(masked_by_p0.writeback.words, [0, 0]);
}

#[test]
fn register_value_machine_handles_aliasing_and_two_dependent_words() {
    let x = vec![1.0_f32.to_bits(), 2.0_f32.to_bits()];
    let y = vec![3.0_f32.to_bits(), 4.0_f32.to_bits()];
    let mut machine = C310RvecValueMachine::from_vector_words(vec![x.clone(), y]).unwrap();
    let mask = [0x11, 0, 0, 0];
    let add = machine.execute_fp32_word(0x8008_2780, &mask).unwrap();
    assert_eq!(add.first_source, x);
    assert_eq!(add.writeback.words, [4.0_f32.to_bits(), 6.0_f32.to_bits()]);
    let sub = machine.execute_fp32_word(0x8008_2781, &mask).unwrap();
    assert_eq!(sub.first_source, add.writeback.words);
    assert_eq!(machine.vector_register(0).unwrap(), x);
    assert_eq!(
        machine.vector_register(1).unwrap(),
        [3.0_f32.to_bits(), 4.0_f32.to_bits()]
    );
    assert_eq!(machine.words_per_register(), 2);
}

#[test]
fn register_value_machine_zeroes_inactive_lanes_and_rejects_unsupported_words() {
    let mut machine = C310RvecValueMachine::from_vector_words(vec![
        vec![1.0_f32.to_bits(), 2.0_f32.to_bits()],
        vec![3.0_f32.to_bits(), 4.0_f32.to_bits()],
    ])
    .unwrap();
    let step = machine
        .execute_fp32_word(0x8008_2780, &[1, 0, 0, 0])
        .unwrap();
    assert_eq!(step.writeback.words, [4.0_f32.to_bits(), 0]);
    assert_eq!(step.writeback.written, [true, true]);
    let before = machine.clone();
    assert_eq!(
        machine.execute_fp32_word(0x8008_2780 | (1 << 18), &[0; 4]),
        Err(C310RvecValueError::UnsupportedDtype)
    );
    assert_eq!(
        machine.execute_fp32_word(0, &[0; 4]),
        Err(C310RvecValueError::UnsupportedWord)
    );
    assert_eq!(machine, before);
}

#[test]
fn register_bank_shape_and_indices_fail_closed() {
    assert_eq!(
        C310RvecValueMachine::from_vector_words(vec![]),
        Err(C310RvecValueError::RegisterCount { count: 0 })
    );
    assert_eq!(
        C310RvecValueMachine::from_vector_words(vec![vec![]]),
        Err(C310RvecValueError::RegisterWidth { words: 0 })
    );
    assert_eq!(
        C310RvecValueMachine::from_vector_words(vec![vec![0], vec![0, 0]]),
        Err(C310RvecValueError::UnequalRegisterWidth {
            index: 1,
            actual: 2,
            expected: 1
        })
    );
    let mut machine = C310RvecValueMachine::from_vector_words(vec![vec![0]]).unwrap();
    assert_eq!(
        machine.execute_fp32_word(0x8008_2780, &[1, 0, 0, 0]),
        Err(C310RvecValueError::RegisterIndex { index: 1, count: 1 })
    );
    assert_eq!(
        machine.execute_fp32_word_from_predicate_registers(0x8008_2780),
        Err(C310RvecValueError::MissingPredicateBank)
    );
    assert_eq!(
        C310RvecValueMachine::from_vector_and_predicate_bytes(vec![vec![0]], vec![]),
        Err(C310RvecValueError::PredicateRegisterCount { count: 0 })
    );
    assert_eq!(
        C310RvecValueMachine::from_vector_and_predicate_bytes(vec![vec![0]], vec![vec![0; 33]]),
        Err(C310RvecValueError::PredicateWidth { bytes: 33 })
    );
    let mut short_p_bank = C310RvecValueMachine::from_vector_and_predicate_bytes(
        vec![vec![0], vec![0]],
        vec![vec![0; 32]],
    )
    .unwrap();
    assert_eq!(
        short_p_bank.execute_fp32_word_from_predicate_registers(0x8008_2780),
        Err(C310RvecValueError::PredicateRegisterIndex { index: 1, count: 1 })
    );
}

#[test]
fn movp_expands_each_scalar_bit_to_one_four_bit_lane_predicate() {
    let even = c310_movp_u32_mask_to_predicate_bytes(0x5555_5555);
    assert_eq!(&even[..16], &[0x0f; 16]);
    assert_eq!(&even[16..], &[0; 16]);
    let odd = c310_movp_u32_mask_to_predicate_bytes(0xaaaa_aaaa);
    assert_eq!(&odd[..16], &[0xf0; 16]);
    assert_eq!(&odd[16..], &[0; 16]);
    assert_eq!(c310_movp_u32_mask_to_predicate_bytes(u64::MAX), [0xff; 32]);
}

#[test]
fn normal_u32_masked_store_preserves_inactive_memory_after_zeroed_vadd_lanes() {
    let predicate = c310_movp_u32_mask_to_predicate_bytes(0b0101);
    let previous = [(-123.0_f32).to_bits(); 5];
    let source = [4.0_f32.to_bits(), 0, 8.0_f32.to_bits(), 0];
    let stored = c310_normal_u32_masked_store(&previous, &source, &predicate).unwrap();
    assert_eq!(
        stored.words,
        [source[0], previous[1], source[2], previous[3], previous[4]]
    );
    assert_eq!(stored.written, [true, false, true, false, false]);
    assert_eq!(
        c310_normal_u32_masked_store(&previous[..2], &source, &predicate),
        Err(Fp32VectorError::DestinationTooSmall {
            destination: 2,
            results: 4
        }
        .into())
    );
    assert_eq!(
        c310_normal_u32_masked_store(&previous, &source, &[0; 33]),
        Err(C310RvecValueError::PredicateWidth { bytes: 33 })
    );
    assert_eq!(
        c310_normal_u32_masked_store(&previous, &[0; 65], &predicate),
        Err(Fp32VectorError::TooManyLanes { lanes: 65 }.into())
    );
}
