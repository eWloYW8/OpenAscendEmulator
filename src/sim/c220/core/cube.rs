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
        self.mte1.begin_advance();
        loop {
            let event_tick = self
                .cube
                .pipeline
                .next_event_tick()
                .into_iter()
                .chain(self.mte1.next_event_tick())
                .chain(self.mte_pipeline.as_ref().and_then(|p| p.next_event_tick()))
                .min()
                .map_or(tick, |next| next.min(tick));
            self.mte1.commit_ready_at(
                event_tick,
                &mut self.local_memory,
                &mut self.hardware_flags,
            )?;
            if let Some(pipeline) = &mut self.mte_pipeline {
                pipeline.advance(event_tick)?;
                self.mte1
                    .observe_completions(event_tick, pipeline.mte1_completions());
            }
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
    use crate::sim::c220::memory::l1::C220L1Geometry;
    use crate::sim::c220::mte::mte1::frontend::C220Mte1ReadBandwidths;
    use crate::sim::c220::mte::mte1::{C220Mte1Command, C220Mte1Generator};
    use crate::sim::c220::mte::mte2::C220Mte2TimingRules;
    use crate::sim::c220::mte::mte3::C220Mte3TimingRules;
    use crate::sim::c220::mte::{C220MtePipelineConfig, C220MtePipelineEvent};
    use crate::sim::c220::state::C220State;
    use crate::sim::c220::vector::pipeline::C220VectorTimingRules;
    use crate::sim::common::scalar::{ScalarMachine, ScalarStepper};
    use std::num::NonZeroU32;

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
        core.configure_mte_pipeline(C220MtePipelineConfig {
            l1: C220L1Geometry::new(32, 4, 1, 0).unwrap(),
            read_width: NonZeroU32::new(256).unwrap(),
            output_bandwidths: C220Mte1ReadBandwidths {
                l0a: NonZeroU32::new(256).unwrap(),
                l0b: NonZeroU32::new(128).unwrap(),
                bt: NonZeroU32::new(64).unwrap(),
            },
            set2d_bandwidths: crate::sim::c220::mte::set2d::C220Set2dBandwidths {
                l0a: NonZeroU32::new(64).unwrap(),
                l0b: NonZeroU32::new(64).unwrap(),
                l1: NonZeroU32::new(32).unwrap(),
            },
        })
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
            (1, flag),
            (2, load | (7 << 12)),
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
    fn bt_switches_after_load2d_generation_and_commits_after_local_completion() {
        use crate::sim::c220::mte::interface::C220MteL1EventOutcome;
        use crate::sim::c220::mte::mte1::C220Mte1TransferResult;
        use crate::sim::c220::mte::mte1::frontend::{C220Mte1ReadKind, C220Mte1ReadTransfer};

        let mut core = matrix_core();
        let machine = core.state.scalar_mut().machine_mut();
        machine.set_xreg(10, 0).unwrap();
        machine.set_xreg(11, 0).unwrap();
        machine.set_xreg(12, 8 | (1 << 4) | (2 << 16)).unwrap();
        let bt = (3 << 29) | (2 << 27) | (4 << 23) | (10 << 17) | (11 << 12) | (12 << 7) | (5 << 3);
        let set = (2 << 29) | (15 << 21) | (5 << 15) | (3 << 10) | (2 << 7);
        core.step_word_at(6, set).unwrap();
        assert!(matches!(
            core.step_word_at(7, bt).unwrap(),
            C220CoreStep::Stalled(_)
        ));
        let (issued_at, id) = (8..64)
            .find_map(|tick| match core.step_word_at(tick, bt).unwrap() {
                C220CoreStep::Executed {
                    instruction:
                        C220CoreInstruction::Mte1 {
                            instruction_id,
                            command: C220Mte1Command::Read(C220Mte1ReadTransfer::Bt(_)),
                            ..
                        },
                    ..
                } => Some((tick, instruction_id)),
                C220CoreStep::Stalled(_) => None,
                other => panic!("unexpected BT dispatch: {other:?}"),
            })
            .expect("BT admission");
        assert!(core.pending_mte1_commands().any(|p| p.instruction_id < id));
        assert_eq!(
            core.mte_pipeline().unwrap().selected_generator(),
            Some(C220Mte1Generator::Read(C220Mte1ReadKind::Bt))
        );
        let trigger = set | (1 << 19) | 1;
        assert!(matches!(
            core.step_word_at(issued_at + 1, trigger).unwrap(),
            C220CoreStep::Stalled(_)
        ));
        let mut output_tick = None;
        let mut completion_tick = None;
        let mut retired = None;
        for tick in issued_at + 2..128 {
            core.advance_to(tick).unwrap();
            for event in core.mte_pipeline().unwrap().last_events() {
                if let C220MtePipelineEvent::Interface(C220MteL1EventOutcome::Output(output)) =
                    event
                    && let Some(sent) = output.sent
                    && sent.fragment.instruction_id == id
                    && sent.fragment.last_in_instruction
                {
                    output_tick = Some(tick);
                }
            }
            if let Some(pending) = core
                .pending_mte1_commands()
                .find(|p| p.instruction_id == id)
            {
                completion_tick = pending.completion_tick;
                let mut bytes = [0; 256];
                core.local_memory.bt().read_into(0, &mut bytes).unwrap();
                assert_eq!(bytes, [0; 256]);
            }
            if let Some(outcome) = core
                .last_mte1_outcomes()
                .iter()
                .find(|o| o.instruction_id == id)
            {
                retired = Some(*outcome);
                break;
            }
        }
        let retired = retired.expect("BT retirement");
        assert_eq!(completion_tick, output_tick.map(|tick| tick + 5));
        assert_eq!(retired.retire_tick, completion_tick.unwrap() + 1);
        let C220Mte1TransferResult::Bt(result) = retired.result else {
            panic!("BT result")
        };
        assert_eq!(
            (
                result.input_bytes,
                result.output_bytes,
                result.converted_elements
            ),
            (128, 256, 64)
        );
        let mut output = [0; 256];
        core.local_memory.bt().read_into(0, &mut output).unwrap();
        assert_eq!(output.as_slice(), 2.0_f32.to_le_bytes().repeat(64));
        let wait = set | (1 << 19) | (1 << 5);
        assert!(matches!(
            core.step_word_at(retired.retire_tick + 1, wait).unwrap(),
            C220CoreStep::Executed { .. }
        ));
    }

    #[test]
    fn set2d_l0_uses_shared_write_ports_and_captured_pattern_at_retirement() {
        use crate::sim::c220::mte::interface::{C220L0WriteEventOutcome, C220L0WritePort};
        use crate::sim::c220::mte::mte1::C220Mte1TransferResult;

        for destination in 0..2 {
            let mut core = matrix_core();
            let pattern = 0x7fa0_1234_u64;
            let machine = core.state.scalar_mut().machine_mut();
            machine.set_xreg(10, 2048).unwrap();
            machine.set_xreg(12, 2 | (1 << 16) | (1 << 32)).unwrap();
            machine.set_spr_value(15, pattern).unwrap();
            let fill =
                (3 << 29) | (1 << 22) | (10 << 17) | (12 << 7) | (destination << 2) | destination;
            let set = (2 << 29) | (15 << 21) | ((destination + 1) << 15) | (3 << 10) | (2 << 7) | 2;
            core.step_word_at(6, set).unwrap();
            assert!(matches!(
                core.step_word_at(7, fill).unwrap(),
                C220CoreStep::Stalled(_)
            ));
            let (issued_at, id) = (8..64)
                .find_map(|tick| match core.step_word_at(tick, fill).unwrap() {
                    C220CoreStep::Executed {
                        instruction:
                            C220CoreInstruction::Mte1 {
                                instruction_id,
                                command: C220Mte1Command::Set2d(_),
                                issue,
                                ..
                            },
                        ..
                    } => {
                        assert_eq!(issue.uop_count, 16);
                        Some((tick, instruction_id))
                    }
                    C220CoreStep::Stalled(_) => None,
                    other => panic!("unexpected fill dispatch: {other:?}"),
                })
                .expect("SET_2D admission");
            assert!(core.pending_mte1_commands().any(|p| p.instruction_id < id));
            assert_eq!(
                core.mte_pipeline().unwrap().selected_generator(),
                Some(C220Mte1Generator::Set2d)
            );
            core.state
                .scalar_mut()
                .machine_mut()
                .set_spr_value(15, 0)
                .unwrap();
            let mut fragments = 0;
            let mut completion = None;
            let mut retired = None;
            for tick in issued_at + 1..128 {
                core.advance_to(tick).unwrap();
                for event in core.mte_pipeline().unwrap().last_events() {
                    if let C220MtePipelineEvent::L0a(C220L0WriteEventOutcome::Sent(send))
                    | C220MtePipelineEvent::L0b(C220L0WriteEventOutcome::Sent(send)) = event
                        && let Some(request) = send.sent
                        && request.instruction_id == id
                    {
                        assert_eq!(send.selected_port, Some(C220L0WritePort::Port1));
                        fragments += 1;
                    }
                }
                if let Some(pending) = core
                    .pending_mte1_commands()
                    .find(|p| p.instruction_id == id)
                {
                    completion = pending.completion_tick;
                    let memory = if destination == 0 {
                        core.local_memory.l0a()
                    } else {
                        core.local_memory.l0b()
                    };
                    assert!(memory.read_known(2048, 512).is_err());
                }
                if let Some(outcome) = core
                    .last_mte1_outcomes()
                    .iter()
                    .find(|o| o.instruction_id == id)
                {
                    retired = Some(*outcome);
                    break;
                }
            }
            let retired = retired.expect("fill retirement");
            assert_eq!(fragments, 16);
            assert_eq!(retired.retire_tick, completion.unwrap() + 1);
            let C220Mte1TransferResult::Set2d(result) = retired.result else {
                panic!("fill result")
            };
            assert_eq!((result.repetitions, result.bytes), (2, 1024));
            let element_bytes = if destination == 0 { 2 } else { 4 };
            let expected = pattern.to_le_bytes()[..element_bytes].repeat(512 / element_bytes);
            let memory = if destination == 0 {
                core.local_memory.l0a()
            } else {
                core.local_memory.l0b()
            };
            for address in [2048, 3072] {
                assert_eq!(memory.read_known(address, 512).unwrap(), expected);
            }
            assert!(memory.read_known(2560, 512).is_err());
            assert!(matches!(
                core.step_word_at(retired.retire_tick + 1, set | (1 << 19) | (1 << 5))
                    .unwrap(),
                C220CoreStep::Executed { .. }
            ));
        }
    }

    #[test]
    fn matrix_advance_is_independent_of_observation_granularity() {
        let mut bulk = matrix_core();
        let mut incremental = matrix_core();
        let mut releases = Vec::new();
        let mut retired = Vec::new();
        let mut outcomes = Vec::new();
        let mut mte1_outcomes = Vec::new();
        for tick in 6..=64 {
            incremental.advance_to(tick).unwrap();
            releases.extend_from_slice(incremental.cube.pipeline.last_uop_releases());
            retired.extend_from_slice(incremental.cube.pipeline.last_retirements());
            outcomes.extend_from_slice(incremental.last_cube_outcomes());
            mte1_outcomes.extend_from_slice(incremental.last_mte1_outcomes());
        }
        bulk.advance_to(64).unwrap();
        assert!(bulk.local_memory == incremental.local_memory);
        assert_eq!(bulk.hardware_flags, incremental.hardware_flags);
        assert_eq!(bulk.cube.pipeline.last_uop_releases(), releases);
        assert_eq!(bulk.cube.pipeline.last_retirements(), retired);
        assert_eq!(bulk.last_cube_outcomes(), outcomes);
        assert_eq!(bulk.last_mte1_outcomes(), mte1_outcomes);
        assert!(bulk.mte_pipeline().unwrap().is_idle());
        assert!(bulk.pending_mte1_commands().next().is_none());
        assert_eq!(mte1_outcomes.len(), 2);
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

        let flag = (2 << 29) | (15 << 21) | (1 << 19) | (1 << 15) | (3 << 10) | (2 << 7) | 1;
        let wait = flag | (1 << 5);
        let pc = bulk.state.scalar().pc();
        assert!(matches!(
            bulk.step_word_at(66, wait).unwrap(),
            C220CoreStep::Stalled(crate::sim::c220::schedule::C220Stall {
                resume_tick: 67,
                ..
            })
        ));
        assert_eq!(bulk.state.scalar().pc(), pc);
        let set = crate::isa::c220::hflag::C220HardwareFlagInstruction::decode(flag)
            .unwrap()
            .resolve(pc, bulk.state.scalar().machine().xregs())
            .unwrap();
        bulk.hardware_flags.schedule_set(set, 66).unwrap();
        assert!(matches!(
            bulk.step_word_at(67, wait).unwrap(),
            C220CoreStep::Executed { .. }
        ));
        assert_eq!(bulk.state.scalar().pc(), pc + 4);
        for _ in 0..crate::sim::c220::sync::C220_HARDWARE_FLAG_ALMOST_FULL {
            bulk.hardware_flags.schedule_set(set, 67).unwrap();
        }
        assert!(matches!(
            bulk.step_word_at(68, flag).unwrap(),
            C220CoreStep::Stalled(crate::sim::c220::schedule::C220Stall {
                resume_tick: 69,
                ..
            })
        ));
        assert_eq!(bulk.state.scalar().pc(), pc + 4);
        let bias_set = (flag & !(7 << 15)) | (5 << 15);
        assert!(matches!(
            bulk.step_word_at(69, bias_set).unwrap(),
            C220CoreStep::Executed {
                instruction: C220CoreInstruction::HardwareFlag {
                    token_ready_tick: Some(70),
                    ..
                },
                ..
            }
        ));
        assert!(matches!(
            bulk.step_word_at(70, bias_set | (1 << 5)).unwrap(),
            C220CoreStep::Executed { .. }
        ));

        // A physical client must keep the shared clock running even with no
        // MTE1 commands left to retire.
        let mut registers = [0; 32];
        registers[3] = 1 | (16 << 16);
        let fill = crate::isa::c220::mte::set2d::C220Set2dInstruction::decode(
            (3 << 29) | (1 << 22) | (3 << 7) | 2,
        )
        .unwrap()
        .capture(&registers, 0);
        for core in [&mut bulk, &mut incremental] {
            core.advance_to(80).unwrap();
            core.mte_pipeline
                .as_mut()
                .unwrap()
                .issue_l1_fill(100, fill)
                .unwrap();
            assert!(core.pending_mte1_commands().next().is_none());
            assert_eq!(
                core.pending_compute_drain(),
                Some((
                    81,
                    crate::sim::c220::schedule::C220StallCause::MtePhysicalDependency
                ))
            );
        }
        for tick in 81..=160 {
            incremental.advance_to(tick).unwrap();
        }
        bulk.advance_to(160).unwrap();
        assert!(bulk.mte_pipeline().unwrap().is_idle());
        assert_eq!(bulk.mte_pipeline(), incremental.mte_pipeline());
    }
}
