use std::num::NonZeroU64;

use crate::architecture::Architecture;
use crate::memory::mapped::MappedMemory;
use crate::memory::region::MemoryRegion;
use crate::memory::sparse::{MemoryByteState, SparseMemory};
use crate::memory::ub::UbMemory;
use crate::sim::c220::core::{C220Core, C220CoreInstruction, C220CoreStep, C220CoreTimingRules};
use crate::sim::c220::fp16::C220Fp16Mode;
use crate::sim::c220::timing::mte2::C220Mte2TimingRules;
use crate::sim::c220::timing::mte3::C220Mte3TimingRules;
use crate::sim::c220::vector::pipeline::C220VectorTimingRules;
use crate::sim::machine::ScalarMachine;
use crate::sim::mte_stepper::MteCoreStepper;
use crate::sim::stepper::ScalarStepper;

#[test]
fn f16_binary_arithmetic_captures_mode_and_commits_both_lane_groups() {
    for (opcode, first, second, expected, mode, execute_ticks) in [
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
    ] {
        let word = opcode | (3 << 17) | (4 << 12) | (5 << 7) | (6 << 2);
        let memory = SparseMemory::new(vec![MemoryRegion::unknown(256)], 512, 512);
        let memory = MappedMemory::bind(memory, &[0x2000]).unwrap();
        let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
        machine.set_xreg(3, 0x200).unwrap();
        machine.set_xreg(4, 0).unwrap();
        machine.set_xreg(5, 0x100).unwrap();
        machine.set_xreg(6, 0x0100_0808_0801_0101).unwrap();
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
        let execution = MteCoreStepper::new(ScalarStepper::new(machine, 0x4000), ub);
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
            instruction: C220CoreInstruction::VectorArithmetic(issue),
            ..
        } = core.step_word_at(0, word).unwrap()
        else {
            panic!("F16 vector instruction should issue");
        };
        assert_eq!(issue.modes.fp16_mode, mode);
        assert_eq!(issue.result_element_bytes, 2);
        let uops = C220CoreInstruction::VectorArithmetic(issue)
            .vector_uops()
            .unwrap();
        assert_eq!(uops.len(), 2);
        assert!(
            uops.iter()
                .all(|uop| uop.stages.execute_ticks == execute_ticks)
        );
        core.advance_to(100).unwrap();
        for address in [0x200, 0x280] {
            assert_eq!(
                core.execution().core().ub().read_known(address, 2).unwrap(),
                expected.to_le_bytes()
            );
        }
        let samples = core.vector_pipeline().last_read_samples();
        assert_eq!(samples.len(), 2);
        assert!(samples[0].lanes[0].fp16_status.is_some());
        assert!(samples[1].lanes[64].fp16_status.is_some());
    }
}
