use super::*;

#[test]
fn end_identifies_the_terminal_flow_instruction() {
    let step = FlowEnd::decode(0x10d0_d8b8, 0x4160_0000).unwrap();
    assert_eq!(step.sequential_pc, 0x10d0_d8bc);
    assert!(FlowEnd::decode(0x10d0_d8b8, 0x402e_f000).is_none());
    assert!(FlowEnd::decode(0x10d0_d8b8, 0x6160_0000).is_none());
}

#[test]
fn dcci_decodes_addressed_and_entire_cache_forms_on_both_architectures() {
    for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
        let mut regs = [0_u64; 32];
        regs[8] = 0x1131_3000;
        regs[15] = 0x10d0_d000;

        let addressed = DcciInstruction::decode(architecture, 0x402c_8000).unwrap();
        assert_eq!(addressed.source_register, 8);
        assert!(!addressed.entire_cache);
        assert_eq!(addressed.operation_field, 0);
        let step = addressed.resolve(0x1131_216c, &regs);
        assert_eq!(step.source_value, 0x1131_3000);
        assert_eq!(step.effective_address, step.source_value);

        let entire = DcciInstruction::decode(architecture, 0x402e_f000).unwrap();
        assert_eq!(entire.source_register, 15);
        assert!(entire.entire_cache);
        let step = entire.resolve(0x10d0_d8b4, &regs);
        assert_eq!(step.source_value, 0x10d0_d000);
        assert_eq!(step.effective_address, 0);

        for operation_field in 0..=3 {
            let instruction =
                DcciInstruction::decode(architecture, 0x402c_8000 | operation_field).unwrap();
            assert_eq!(instruction.operation_field, operation_field as u8);
        }
        assert!(DcciInstruction::decode(architecture, 0x4020_000a).is_none());
        assert!(DcciInstruction::decode(architecture, 0x422c_8000).is_none());
        assert!(DcciInstruction::decode(architecture, 0x602c_8000).is_none());
    }
}

#[test]
fn dsb_decodes_the_scope_field() {
    let step = DsbStep::decode(0x1131_2170, 0x41c1_0000).unwrap();
    assert_eq!(step.scope_field, 1);
    for scope in 0..=7 {
        let word = 0x41c0_0000 | (scope << 16);
        assert_eq!(
            DsbStep::decode(0x1000, word).unwrap().scope_field,
            scope as u8
        );
    }
    assert!(DsbStep::decode(0x1000, 0x402c_8000).is_none());
    assert!(DsbStep::decode(0x1000, 0x61c1_0000).is_none());
}

#[test]
fn barriers_decode_only_the_supported_scope_words() {
    assert_eq!(
        PipelineBarrierStep::decode(Architecture::Dav2201, 0x1131_2638, 0x40e0_0400),
        Some(PipelineBarrierStep {
            pc: 0x1131_2638,
            word: 0x40e0_0400,
            scope: PipelineBarrierScope::Vector,
        })
    );
    assert_eq!(
        PipelineBarrierStep::decode(Architecture::Dav2201, 0x1131_2090, 0x40e0_1800)
            .unwrap()
            .scope,
        PipelineBarrierScope::All
    );
    assert_eq!(
        PipelineBarrierStep::decode(Architecture::Dav3510, 0x10d0_d090, 0x40e0_1800)
            .unwrap()
            .scope,
        PipelineBarrierScope::All
    );
    for word in [0x40e0_0401, 0x40e0_0800, 0x40e0_1801] {
        assert!(PipelineBarrierStep::decode(Architecture::Dav2201, 0x1000, word).is_none());
    }
    assert_eq!(
        PipelineBarrierStep::decode(Architecture::Dav2201, 0x1000, 0x40e0_2800)
            .unwrap()
            .scope,
        PipelineBarrierScope::Fix
    );
    assert!(PipelineBarrierStep::decode(Architecture::Dav3510, 0x1000, 0x40e0_2800).is_none());
    for word in [0x40e0_0400, 0x40e0_1801] {
        assert!(PipelineBarrierStep::decode(Architecture::Dav3510, 0x1000, word).is_none());
    }
}

#[test]
fn flag_ids_use_the_live_encoded_register_on_both_architectures() {
    for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
        let instruction = FlagInstruction::decode(architecture, 0x40a2_0630).unwrap();
        assert_eq!(instruction.operation, FlagOperation::Set);
        assert_eq!(instruction.source_pipe_code, 1);
        assert_eq!(instruction.trigger_pipe_code, 4);
        assert_eq!(instruction.id_source, FlagIdSource::Register(12));
        let mut xregs = [0_u64; 32];
        assert_eq!(instruction.resolve(0x1131_26f8, &xregs).flag_id, 0);
        xregs[12] = 1;
        let step = instruction.resolve(0x1131_2704, &xregs);
        assert_eq!(step.flag_id, 1);
        assert_eq!(step.source_value, Some(1));

        let output_set = FlagInstruction::decode(architecture, 0x40a2_06b8).unwrap();
        assert_eq!(output_set.id_source, FlagIdSource::Register(14));
        assert_eq!(
            (output_set.source_pipe_code, output_set.trigger_pipe_code),
            (1, 5)
        );
        let output_wait = FlagInstruction::decode(architecture, 0x40c2_06b4).unwrap();
        assert_eq!(output_wait.operation, FlagOperation::Wait);
        assert_eq!(output_wait.id_source, FlagIdSource::Register(13));
        assert_eq!(
            (output_wait.source_pipe_code, output_wait.trigger_pipe_code),
            (1, 5)
        );

        let immediate = FlagInstruction::decode(architecture, 0x40a0_1000).unwrap();
        assert_eq!(immediate.id_source, FlagIdSource::Immediate(0));
        assert_eq!(immediate.resolve(0x1000, &xregs).source_value, None);
        assert!(FlagInstruction::decode(architecture, 0x42a2_0630).is_none());
        assert!(FlagInstruction::decode(architecture, 0x40e0_1800).is_none());
    }
}

#[test]
fn flow_nop_advances_one_word_on_both_architectures() {
    for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
        let nop = FlowNop::decode(architecture, 0x11312160, 0x4140_0000).unwrap();
        assert_eq!(nop.target_pc, 0x11312164);
        assert!(FlowNop::decode(architecture, 0x11312164, 0x4140_0000).is_some());
        assert!(FlowNop::decode(architecture, 0x11312160, 0x4000_0000).is_none());
        assert!(FlowNop::decode(architecture, 0x11312160, 0x4940_0000).is_none());
    }
}

#[test]
fn relative_immediate_jump_uses_signed_word_displacement() {
    for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
        let regs = [0; 32];
        let forward = UnconditionalJump::decode(architecture, 0x4000_03e0).unwrap();
        assert_eq!(forward.resolve(0x112f_5348, &regs).target_pc, 0x112f_62c8);
        let backward = UnconditionalJump::decode(architecture, 0x4000_fff9).unwrap();
        let target = backward.resolve(0x126e_cfb8, &regs);
        assert_eq!(target.effective_offset_words, -7);
        assert_eq!(target.target_pc, 0x126e_cf9c);
    }
}

#[test]
fn register_jump_sign_extends_bit_45_and_rejects_other_flow_routes() {
    for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
        let mut regs = [0; 32];
        regs[3] = 0x3fff_ffff_fff9;
        let jump = UnconditionalJump::decode(architecture, 0x4002_3000).unwrap();
        assert_eq!(jump.offset_source, JumpOffsetSource::Register { index: 3 });
        let target = jump.resolve(0x1000, &regs);
        assert_eq!(target.source_value, Some(0x3fff_ffff_fff9));
        assert_eq!(target.effective_offset_words, -7);
        assert_eq!(target.target_pc, 0x0fe4);
        assert!(UnconditionalJump::decode(architecture, 0x4020_0002).is_none());
        assert!(UnconditionalJump::decode(architecture, 0x4884_1521).is_none());
        assert!(UnconditionalJump::decode(architecture, 0x0000_0000).is_none());
    }
    assert!(UnconditionalJump::decode(Architecture::Dav3510, 0x4200_0000).is_none());
}

#[test]
fn conditional_jump_selects_taken_or_fallthrough_target() {
    for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
        let regs = [0; 32];
        let jump = ConditionalJump::decode(architecture, 0x4020_0002).unwrap();
        let not_taken = jump.resolve(0x126e_cfb4, &regs, 0);
        assert!(!not_taken.branch_taken);
        assert_eq!(not_taken.target_pc, 0x126e_cfb8);
        assert_eq!(not_taken.taken_target.target_pc, 0x126e_cfbc);
        let taken = jump.resolve(0x126e_cfb4, &regs, 1);
        assert!(taken.branch_taken);
        assert_eq!(taken.target_pc, 0x126e_cfbc);

        let backward = ConditionalJump::decode(architecture, 0x4020_fff9).unwrap();
        assert_eq!(
            backward.resolve(0x126e_d00c, &regs, 1).target_pc,
            0x126e_cff0
        );
        assert!(ConditionalJump::decode(architecture, 0x4000_0002).is_none());
        assert!(ConditionalJump::decode(architecture, 0x4024_0002).is_none());
    }
}

#[test]
fn jump_compare_decodes_and_evaluates_integer_operands() {
    for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
        let mut regs = [0; 32];
        regs[8] = 1;
        let immediate = JumpCompare::decode(architecture, 0x4884_1521).unwrap();
        assert_eq!(immediate.first_source_register, 8);
        assert_eq!(immediate.condition_field, 1);
        assert_eq!(
            immediate.second_operand,
            JumpCompareOperand::Immediate { encoded: 1 }
        );
        assert_eq!(
            immediate.offset_source,
            JumpCompareOffset::Immediate {
                encoded_words: 0xa9
            }
        );
        let result = immediate.evaluate(0x112f_50b8, &regs).unwrap();
        assert!(!result.branch_taken);
        assert_eq!(result.target_pc, 0x112f_50bc);
        assert_eq!(result.spr11_value, 0);

        regs[1] = 0x209;
        regs[2] = 0x301;
        let register = JumpCompare::decode(architecture, 0x4a09_83a2).unwrap();
        assert_eq!(register.first_source_register, 1);
        assert_eq!(
            register.second_operand,
            JumpCompareOperand::Register { index: 2 }
        );
        let result = register.evaluate(0x126e_d14c, &regs).unwrap();
        assert!(result.branch_taken);
        assert_eq!(result.target_pc, 0x126e_d1c0);
        assert_eq!(result.spr11_value, 1);
    }
}

#[test]
fn jump_compare_sign_extends_immediate_and_backwards_displacement() {
    for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
        let mut regs = [0; 32];
        regs[8] = 1;
        let word = (0x4884_1521 & !0x7fff) | (0x3ff << 5) | 0x1f;
        let jump = JumpCompare::decode(architecture, word).unwrap();
        let result = jump.evaluate(0x1000, &regs).unwrap();
        assert_eq!(result.second_source_value, u64::MAX);
        assert_eq!(result.taken_target.effective_offset_words, -1);
        assert_eq!(result.target_pc, 0xffc);

        let signed_less =
            JumpCompare::decode(architecture, (word & !(7 << 18)) | (2 << 18)).unwrap();
        let unsigned_less =
            JumpCompare::decode(architecture, signed_less.word | (1 << 25)).unwrap();
        assert!(!signed_less.evaluate(0x1000, &regs).unwrap().branch_taken);
        assert!(unsigned_less.evaluate(0x1000, &regs).unwrap().branch_taken);

        let unsupported = JumpCompare::decode(architecture, word | (3 << 25)).unwrap();
        assert_eq!(
            unsupported.evaluate(0x1000, &regs),
            Err(JumpCompareError::UnsupportedDtype(3))
        );
        let invalid_condition = JumpCompare::decode(architecture, word | (6 << 18)).unwrap();
        assert_eq!(
            invalid_condition.evaluate(0x1000, &regs),
            Err(JumpCompareError::UnsupportedCondition(7))
        );
    }
}

#[test]
fn jump_compare_register_offset_uses_signed_46_bit_words() {
    for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
        let word = (0x4884_1521 & !0x7fe0) | 0x2_0000 | (3 << 5);
        let jump = JumpCompare::decode(architecture, word).unwrap();
        assert_eq!(jump.offset_source, JumpCompareOffset::Register { index: 3 });
        let mut regs = [0; 32];
        regs[3] = 0x3fff_ffff_fff9;
        regs[8] = 2;
        let result = jump.evaluate(0x1000, &regs).unwrap();
        assert!(result.branch_taken);
        assert_eq!(result.taken_target.effective_offset_words, -7);
        assert_eq!(result.target_pc, 0xfe4);
    }
}

#[test]
fn float_compare_uses_low_word_bits_and_ordered_nan_behavior() {
    for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
        let mut regs = [0; 32];
        let equal_word = (0x4884_1521 & !(7 << 18)) | (2 << 25);
        let equal = JumpCompare::decode(architecture, equal_word).unwrap();
        regs[8] = 1;
        assert!(equal.evaluate(0x1000, &regs).unwrap().branch_taken);
        regs[8] = (1_u64 << 32) | 1;
        assert!(equal.evaluate(0x1000, &regs).unwrap().branch_taken);
        regs[8] = (-0.0_f32).to_bits().into();
        let equal_zero = JumpCompare::decode(architecture, equal_word & !0x1f).unwrap();
        assert!(equal_zero.evaluate(0x1000, &regs).unwrap().branch_taken);

        let not_equal = JumpCompare::decode(architecture, equal_word | (1 << 18)).unwrap();
        regs[8] = 0x7fc0_0000;
        assert!(!not_equal.evaluate(0x1000, &regs).unwrap().branch_taken);
        let nan_immediate = JumpCompare::decode(architecture, not_equal.word | 0x1f).unwrap();
        regs[8] = 0;
        assert!(!nan_immediate.evaluate(0x1000, &regs).unwrap().branch_taken);
        regs[8] = 0x7f80_0000;
        assert!(not_equal.evaluate(0x1000, &regs).unwrap().branch_taken);

        let register_word = (0x4a09_83a2 & !(3 << 25) & !(7 << 18)) | (2 << 25) | (3 << 18);
        let greater = JumpCompare::decode(architecture, register_word).unwrap();
        regs[1] = 0x7f80_0000;
        regs[2] = 1.0_f32.to_bits().into();
        let outcome = greater.evaluate(0x2000, &regs).unwrap();
        assert!(outcome.branch_taken);
        assert_eq!(outcome.spr11_value, 1);
        regs[2] = 0x7fc0_0000;
        assert!(!greater.evaluate(0x2000, &regs).unwrap().branch_taken);
    }
}
