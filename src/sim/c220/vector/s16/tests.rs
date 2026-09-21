use std::num::NonZeroU64;

use crate::architecture::Architecture;
use crate::memory::mapped::MappedMemory;
use crate::memory::region::MemoryRegion;
use crate::memory::sparse::{MemoryByteState, SparseMemory};
use crate::memory::ub::UbMemory;
use crate::sim::c220::core::{C220Core, C220CoreInstruction, C220CoreStep, C220CoreTimingRules};
use crate::sim::c220::timing::mte2::C220Mte2TimingRules;
use crate::sim::c220::timing::mte3::C220Mte3TimingRules;
use crate::sim::c220::vector::pipeline::C220VectorTimingRules;
use crate::sim::machine::ScalarMachine;
use crate::sim::mte_stepper::MteCoreStepper;
use crate::sim::stepper::ScalarStepper;

#[test]
fn widened_s16_binary_arithmetic_writes_one_s32_uop() {
    for (opcode, first, second, expected, execute_ticks) in [
        (0x8580_0000, i16::MAX, 1_i16, 32_768_i32, 5),
        (0x8580_0001, i16::MIN, 1, -32_769, 5),
        (0x8980_0000, i16::MIN, i16::MIN, 1_073_741_824, 6),
    ] {
        let word = opcode | (3 << 17) | (4 << 12) | (5 << 7) | (6 << 2);
        let memory = SparseMemory::new(vec![MemoryRegion::unknown(256)], 512, 512);
        let memory = MappedMemory::bind(memory, &[0x2000]).unwrap();
        let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
        machine.set_xreg(3, 0x200).unwrap();
        machine.set_xreg(4, 0).unwrap();
        machine.set_xreg(5, 0x100).unwrap();
        machine.set_xreg(6, 0x0100_0808_0801_0101).unwrap();
        machine.set_spr_value(3, (1_u64 << 52) | (1 << 53)).unwrap();
        machine.set_spr_value(100, (1_u64 << 63) | 1).unwrap();
        machine.set_spr_value(101, 1).unwrap();
        let mut ub = UbMemory::new(1024, 256);
        for address in [0, 0x60, 0x100, 0x160] {
            ub.write_states(address, &[MemoryByteState::Known(0); 32])
                .unwrap();
        }
        for offset in [0, 126] {
            ub.write_states(offset, &first.to_le_bytes().map(MemoryByteState::Known))
                .unwrap();
            ub.write_states(
                offset + 0x100,
                &second.to_le_bytes().map(MemoryByteState::Known),
            )
            .unwrap();
        }
        for address in [0x200, 0x2e0] {
            ub.write_states(address, &[MemoryByteState::Known(0xaa); 32])
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
            panic!("widened S16 vector instruction should issue");
        };
        assert!(issue.modes.widen_s16);
        assert_eq!(issue.source_element_bytes, 2);
        assert_eq!(issue.result_element_bytes, 4);
        assert_eq!(issue.write_targets.len(), 2);
        assert_eq!(issue.write_targets[1].address, 0x2fc);
        let uops = C220CoreInstruction::VectorArithmetic(issue)
            .vector_uops()
            .unwrap();
        assert_eq!(uops.len(), 1);
        assert_eq!(uops[0].stages.execute_ticks, execute_ticks);
        core.advance_to(100).unwrap();
        for address in [0x200, 0x2fc] {
            assert_eq!(
                core.execution().core().ub().read_known(address, 4).unwrap(),
                expected.to_le_bytes()
            );
        }
        let samples = core.vector_pipeline().last_read_samples();
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].lanes.len(), 64);
        assert_eq!(samples[0].accesses.len(), 4);
    }
}
