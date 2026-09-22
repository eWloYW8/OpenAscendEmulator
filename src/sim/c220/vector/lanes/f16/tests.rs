use crate::sim::c220::vector::C220VectorInstruction;
use std::num::NonZeroU64;

use crate::architecture::Architecture;
use crate::memory::mapped::MappedMemory;
use crate::memory::region::MemoryRegion;
use crate::memory::sparse::{MemoryByteState, SparseMemory};
use crate::memory::ub::UbMemory;
use crate::sim::c220::core::{C220Core, C220CoreInstruction, C220CoreStep, C220CoreTimingRules};
use crate::sim::c220::mte::mte2::C220Mte2TimingRules;
use crate::sim::c220::mte::mte3::C220Mte3TimingRules;
use crate::sim::c220::numeric::fp16::C220Fp16Mode;
use crate::sim::c220::state::C220State;
use crate::sim::c220::vector::pipeline::C220VectorTimingRules;
use crate::sim::common::scalar::ScalarMachine;
use crate::sim::common::scalar::ScalarStepper;

#[test]
fn f16_arithmetic_captures_mode_and_commits_one_native_uop() {
    for (opcode, first, second, expected, mode, execute_ticks) in [
        (0x9440_0000, 0xbc00, 0x3800, 0, C220Fp16Mode::Saturating, 7),
        (
            0x9440_0001,
            0x4000,
            0x3800,
            0x3e00,
            C220Fp16Mode::Saturating,
            7,
        ),
        (
            0x8540_0000,
            0x3c00_u16,
            0x3800_u16,
            0x3e00_u16,
            C220Fp16Mode::Saturating,
            7,
        ),
        (
            0x8540_0001,
            0x3c00,
            0x3800,
            0x3800,
            C220Fp16Mode::Saturating,
            7,
        ),
        (
            0x8940_0000,
            0x3e00,
            0x4000,
            0x4200,
            C220Fp16Mode::Saturating,
            8,
        ),
        (0x8740_0000, 0x8000, 0, 0, C220Fp16Mode::Saturating, 5),
        (0x8740_0001, 0x8000, 0, 0x8000, C220Fp16Mode::Saturating, 5),
        (0x8540_0000, 0x7c00, 0xfc00, 0, C220Fp16Mode::Saturating, 7),
        (
            0x8540_0000,
            0x7c00,
            0xfc00,
            0x7fff,
            C220Fp16Mode::NonSaturating,
            7,
        ),
        (0x8340_0180, 0xbc00, 0, 0, C220Fp16Mode::Saturating, 5),
        (0x8340_0180, 0x8000, 0, 0, C220Fp16Mode::Saturating, 5),
        (0x8340_0180, 0x7c00, 0, 0x7bff, C220Fp16Mode::Saturating, 5),
        (0x8340_0180, 0x7e00, 0, 0, C220Fp16Mode::Saturating, 5),
        (0x8340_0300, 0xbc00, 0, 0x3c00, C220Fp16Mode::Saturating, 5),
        (0x8340_0300, 0x8000, 0, 0, C220Fp16Mode::Saturating, 5),
        (0x8340_0300, 0xfc00, 0, 0x7bff, C220Fp16Mode::Saturating, 5),
        (
            0x8340_0300,
            0x7e00,
            0,
            0x7fff,
            C220Fp16Mode::NonSaturating,
            5,
        ),
        (
            0x8340_0180,
            0x7e00,
            0,
            0x7fff,
            C220Fp16Mode::NonSaturating,
            5,
        ),
    ] {
        let unary = matches!(opcode, 0x8340_0180 | 0x8340_0300);
        let word =
            opcode | (3 << 17) | (4 << 12) | if unary { 6 << 2 } else { (5 << 7) | (6 << 2) };
        let memory = SparseMemory::new(vec![MemoryRegion::unknown(256)], 512, 512);
        let memory = MappedMemory::bind(memory, &[0x2000]).unwrap();
        let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
        machine.set_xreg(3, 0x200).unwrap();
        machine.set_xreg(4, 0).unwrap();
        machine.set_xreg(5, 0x100).unwrap();
        machine
            .set_xreg(
                6,
                if unary {
                    (1_u64 << 56) | (8 << 40) | (8 << 32) | (1 << 16) | 1
                } else {
                    0x0100_0808_0801_0101
                },
            )
            .unwrap();
        let control_spr = if mode == C220Fp16Mode::NonSaturating {
            1_u64 << 48
        } else {
            0
        };
        machine.set_spr_value(3, control_spr).unwrap();
        machine.set_spr_value(100, 1).unwrap();
        machine.set_spr_value(101, 1).unwrap();
        let mut ub = UbMemory::new(1024, 256);
        for address in [0, 0x80, 0x100, 0x180] {
            ub.write_states(address, &[MemoryByteState::Known(0); 32])
                .unwrap();
        }
        for offset in [0, 0x80] {
            ub.write_states(offset, &first.to_le_bytes().map(MemoryByteState::Known))
                .unwrap();
            ub.write_states(
                offset + 0x100,
                &second.to_le_bytes().map(MemoryByteState::Known),
            )
            .unwrap();
            ub.write_states(offset + 0x200, &[MemoryByteState::Known(0xaa); 32])
                .unwrap();
        }
        let execution = C220State::new(ScalarStepper::new(machine, 0x4000), ub);
        let rate = NonZeroU64::new(32).unwrap();
        let mut core = C220Core::new(
            execution,
            memory,
            C220CoreTimingRules {
                mte2: C220Mte2TimingRules {
                    issue_interval: NonZeroU64::new(1).unwrap(),
                    startup_ticks: 0,
                    bytes_per_tick: rate,
                    retire_ticks: 0,
                },
                mte3: C220Mte3TimingRules {
                    issue_interval: NonZeroU64::new(1).unwrap(),
                    startup_ticks: 0,
                    bytes_per_tick: rate,
                    retire_ticks: 0,
                },
                vector: C220VectorTimingRules {
                    dispatch_ticks: 0,
                    uop_issue_interval: NonZeroU64::new(1).unwrap(),
                    ub_response_ticks: 1,
                },
            },
        )
        .unwrap();
        let C220CoreStep::Executed {
            instruction: C220CoreInstruction::Vector(C220VectorInstruction::Arithmetic(issue)),
            ..
        } = core.step_word_at(0, word).unwrap()
        else {
            panic!("F16 vector instruction should issue");
        };
        assert_eq!(issue.modes.fp16_mode, mode);
        assert_eq!(issue.result_element_bytes, 2);
        let uops = C220CoreInstruction::Vector(C220VectorInstruction::Arithmetic(issue))
            .as_vector()
            .unwrap()
            .uops()
            .unwrap();
        assert_eq!(uops.len(), 1);
        assert!(
            uops.iter()
                .all(|uop| uop.stages.execute_ticks == execute_ticks)
        );
        core.advance_to(100).unwrap();
        for address in [0x200, 0x280] {
            assert_eq!(
                core.state().ub().read_known(address, 2).unwrap(),
                expected.to_le_bytes()
            );
        }
        let samples = core.vector_pipeline().last_functional_samples();
        assert_eq!(samples.len(), 1);
        assert!(samples[0].lanes[0].fp16_status.is_some());
        assert!(samples[0].lanes[64].fp16_status.is_some());
    }
}
