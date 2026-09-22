use super::{C220Core, C220CoreError, C220CoreInstruction};
use crate::isa::c220::cube::C220CubeInstruction;
use crate::sim::c220::cube::{C220CubeExecutionControl, C220CubeIssue, C220CubeTimingControl};

impl C220Core {
    pub(super) fn issue_cube_at(
        &mut self,
        tick: u64,
        pc: u64,
        word: u32,
        decoded: C220CubeInstruction,
    ) -> Result<C220CoreInstruction, C220CoreError> {
        let registers = decoded.capture(self.state.scalar().machine().xregs());
        let parameters = decoded.parameters(registers);
        let control_spr = self
            .state
            .scalar()
            .machine()
            .spr_value(3)
            .ok_or(C220CoreError::MissingCubeControlSpr)?;
        let spr107 = self
            .state
            .scalar()
            .machine()
            .spr_value(107)
            .ok_or(C220CoreError::MissingCubeTimingSpr { spr: 107 })?;
        let spr108 = self
            .state
            .scalar()
            .machine()
            .spr_value(108)
            .ok_or(C220CoreError::MissingCubeTimingSpr { spr: 108 })?;
        let execution_control = C220CubeExecutionControl::from_spr3(control_spr);
        let timing_control = C220CubeTimingControl::from_sprs(control_spr, spr107, spr108);
        let ticket = self
            .cube
            .pipeline
            .preview_issue(tick, decoded, parameters, timing_control)?;
        let issue = C220CubeIssue {
            instruction_id: self.next_instruction_id,
            pc,
            word,
            instruction: decoded,
            registers,
            parameters,
            ticket,
        };
        self.cube
            .issue(issue, execution_control, &mut self.local_memory)?;
        self.state.commit_c220_sequential_issue();
        Ok(C220CoreInstruction::Cube(issue))
    }

    pub(super) fn advance_matrix_to(&mut self, tick: u64) -> Result<(), C220CoreError> {
        self.cube.begin_advance();
        loop {
            let event_tick = self
                .cube
                .pipeline
                .next_event_tick()
                .into_iter()
                .chain(self.mte1.next_data_ready_tick())
                .min()
                .map_or(tick, |next| next.min(tick));
            self.mte1
                .commit_ready_at(event_tick, &mut self.local_memory)?;
            self.cube.advance_event(
                event_tick,
                &mut self.local_memory,
                &mut self.hardware_flags,
                self.state.scalar_mut().machine_mut(),
            )?;
            if event_tick == tick {
                break;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroU64;

    use crate::Architecture;
    use crate::memory::{mapped::MappedMemory, sparse::SparseMemory, ub::UbMemory};
    use crate::sim::c220::core::{C220CoreInstruction, C220CoreStep, C220CoreTimingRules};
    use crate::sim::c220::mte::mte1::C220Mte1TimingRules;
    use crate::sim::c220::mte::mte2::C220Mte2TimingRules;
    use crate::sim::c220::mte::mte3::C220Mte3TimingRules;
    use crate::sim::c220::state::C220State;
    use crate::sim::c220::vector::pipeline::C220VectorTimingRules;
    use crate::sim::common::scalar::{ScalarMachine, ScalarStepper};

    fn matrix_core() -> C220Core {
        let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
        for (register, value) in [
            (0, 0),
            (1, 0),
            (2, 0),
            (3, 1 | (16 << 12) | (1 << 24) | (1 << 63)),
            (4, 1024),
            (6, 0),
            (7, 0),
            (8, (1 << 16) | (1 << 24)),
            (9, 512),
        ] {
            machine.set_xreg(register, value).unwrap();
        }
        let one = NonZeroU64::new(1).unwrap();
        let mut core = C220Core::new(
            C220State::new(ScalarStepper::new(machine, 0), UbMemory::new(4096, 256)),
            MappedMemory::bind(SparseMemory::new(vec![], 4096, 4096), &[]).unwrap(),
            C220CoreTimingRules {
                mte2: C220Mte2TimingRules {
                    issue_interval: one,
                    startup_ticks: 0,
                    bytes_per_tick: one,
                    retire_ticks: 0,
                },
                mte3: C220Mte3TimingRules {
                    issue_interval: one,
                    startup_ticks: 0,
                    bytes_per_tick: one,
                    retire_ticks: 0,
                },
                vector: C220VectorTimingRules {
                    dispatch_ticks: 0,
                    uop_issue_interval: one,
                    ub_response_ticks: 1,
                },
            },
        )
        .unwrap();
        core.configure_mte1_timing(C220Mte1TimingRules::dav2201())
            .unwrap();
        core.local_memory
            .l0a_mut()
            .write_known(0, &0x3c00_u16.to_le_bytes().repeat(256))
            .unwrap();
        core.local_memory
            .l0b_mut()
            .write_known(0, &0x3c00_u16.to_le_bytes().repeat(256))
            .unwrap();
        core.local_memory
            .l1_mut()
            .write_known(0, &0x4000_u16.to_le_bytes().repeat(256))
            .unwrap();
        core.local_memory
            .l1_mut()
            .write_known(512, &0x4200_u16.to_le_bytes().repeat(256))
            .unwrap();
        let cube = (7 << 29) | (3 << 22) | (1 << 12) | (2 << 7) | (3 << 2);
        let load = (3 << 29) | (6 << 17) | (8 << 7) | 8;
        let flag = (2 << 29) | (15 << 21) | (1 << 15) | (3 << 10) | (2 << 7);
        for (tick, word) in [
            (0, cube),
            (1, load | (7 << 12)),
            (2, flag),
            (3, flag | (1 << 5)),
            (4, cube | (4 << 17)),
            (5, load | (9 << 12)),
        ] {
            assert!(matches!(
                core.step_word_at(tick, word).unwrap(),
                C220CoreStep::Executed { .. }
            ));
        }
        core
    }

    #[test]
    fn matrix_advance_is_independent_of_observation_granularity() {
        let mut bulk = matrix_core();
        let mut incremental = matrix_core();
        let mut releases = Vec::new();
        let mut retired = Vec::new();
        let mut outcomes = Vec::new();
        for tick in 6..=64 {
            incremental.advance_to(tick).unwrap();
            releases.extend_from_slice(incremental.cube.pipeline.last_uop_releases());
            retired.extend_from_slice(incremental.cube.pipeline.last_retirements());
            outcomes.extend_from_slice(incremental.last_cube_outcomes());
        }
        bulk.advance_to(64).unwrap();
        assert!(bulk.local_memory == incremental.local_memory);
        assert_eq!(bulk.hardware_flags, incremental.hardware_flags);
        assert_eq!(bulk.cube.pipeline.last_uop_releases(), releases);
        assert_eq!(bulk.cube.pipeline.last_retirements(), retired);
        assert_eq!(bulk.last_cube_outcomes(), outcomes);
        for (address, expected) in [(0, 16.0_f32), (1024, 32.0)] {
            assert_eq!(
                bulk.local_memory
                    .l0c()
                    .buffer()
                    .read_known(address, 4)
                    .unwrap(),
                expected.to_le_bytes()
            );
        }
        assert_eq!(bulk.hardware_flags.pending_cube_wait_count(), 0);
        assert!(matches!(
            bulk.step_word_at(65, 0x40e0_1800).unwrap(),
            C220CoreStep::Executed {
                instruction: C220CoreInstruction::Barrier(_),
                ..
            }
        ));
    }
}
