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
            execution_control,
            ticket,
        };
        self.cube.issue(issue, &mut self.local_memory)?;
        self.state.commit_c220_sequential_issue();
        Ok(C220CoreInstruction::Cube(issue))
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

    #[test]
    fn fixp_instruction_dispatches_and_drains_on_shared_core_clock() {
        use crate::sim::c220::memory::C220LocalBuffer;
        use crate::sim::c220::mte::fixp::{C220FixpEngineConfig, C220FixpStage::*};
        let mut core = matrix_core();
        core.advance_to(300).unwrap();
        let pc = core.state.scalar().pc();
        core.configure_fixp_l1(
            C220FixpEngineConfig {
                instruction_fifo_depth: 1,
                read_bandwidth: 256,
                read_bank_count: 32,
                read_data_latency: 4,
                l0c_capacity: core.local_memory.l0c().buffer().capacity(),
            },
            super::super::C220FixpFrontendConfig {
                issue_queue_depth: NonZeroU32::new(2).unwrap(),
                outstanding_limit: NonZeroU32::new(2).unwrap(),
            },
            C220LocalBuffer::new(4096),
            &[
                GenerateRead,
                SendRead,
                SendL0c,
                ReceiveL0c,
                Convert,
                Slice,
                Packetize,
                GenerateWrite,
                SendWrite,
            ],
        )
        .unwrap();
        let machine = core.state.scalar_mut().machine_mut();
        for (reg, value) in [
            (1, 4096),
            (2, 0),
            (3, (8 << 32) | (8 << 16) | (16 << 4)),
            (4, 1 << 34),
        ] {
            machine.set_xreg(reg, value).unwrap();
        }
        for reg in [3, 61, 64, 97] {
            machine.set_spr_value(reg, 0).unwrap();
        }
        core.local_memory
            .l0c_mut()
            .buffer_mut()
            .write_known_linear(0, &2_f32.to_le_bytes().repeat(128))
            .unwrap();
        let word = (6 << 29) | (3 << 24) | (1 << 17) | (2 << 12) | (3 << 7) | (4 << 2);
        assert!(matches!(
            core.step_word_at(301, word).unwrap(),
            super::super::C220CoreStep::Executed {
                instruction: C220CoreInstruction::FixpQueued {
                    instruction_id: 6,
                    ..
                },
                ..
            }
        ));
        for tick in 302..=304 {
            assert!(matches!(
                core.step_word_at(tick, word).unwrap(),
                C220CoreStep::Executed {
                    instruction: C220CoreInstruction::FixpQueued { .. },
                    ..
                }
            ));
            assert!(core.fixp_engine().unwrap().commands().is_empty());
        }
        assert_eq!(core.queued_fixp_commands(), 4);
        assert_eq!(core.fixp_issue_queue_len(), 2);
        assert_eq!(core.fixp_command_queue_len(), 2);
        assert_eq!(core.outstanding_fixp_commands(), 2);
        assert!(matches!(
            core.step_word_at(305, word).unwrap(),
            C220CoreStep::Stalled(_)
        ));
        assert_eq!(core.state.scalar().pc(), pc + 16);
        assert!(core.pending_compute_drain().is_some());
        core.advance_to(500).unwrap();
        assert!(core.fixp_engine().unwrap().is_idle());
        assert_eq!(
            core.local_memory.l1().read_known(4096, 256).unwrap(),
            0x4000_u16.to_le_bytes().repeat(128)
        );
        assert!(core.pending_compute_drain().is_none());
        let flag = crate::isa::c220::hflag::C220HardwareFlagInstruction::decode(
            (2 << 29) | (15 << 21) | (3 << 15) | (10 << 10) | (2 << 7),
        )
        .unwrap()
        .resolve(0, &[0; 32])
        .unwrap();
        core.hardware_flags.enqueue_mte_flag(0, flag, 500).unwrap();
        core.configure_factor_reads(crate::sim::c220::core::C220FactorReadConfig {
            port: crate::sim::c220::mte::interface::C220MteL1ReadPort::Port2,
            access_width: NonZeroU32::new(32).unwrap(),
            output_bandwidth: NonZeroU32::new(32).unwrap(),
        })
        .unwrap();
        for ub in [false, true] {
            let tick = if ub { 701 } else { 501 };
            let machine = core.state.scalar_mut().machine_mut();
            machine.set_xreg(1, 0).unwrap();
            machine.set_xreg(2, 4096).unwrap();
            machine.set_xreg(3, (1 << 16) | (1 << 4) | 8).unwrap();
            let word = (6 << 29) | (1 << 17) | (2 << 12) | (3 << 7) | if ub { 1 << 22 } else { 0 };
            assert!(matches!(
                core.step_word_at(tick, word).unwrap(),
                C220CoreStep::Executed {
                    instruction: C220CoreInstruction::FixpQueued { .. },
                    ..
                }
            ));
            let input = 0x4200_u16.to_le_bytes().repeat(64);
            if ub {
                let states: Vec<_> = input
                    .iter()
                    .copied()
                    .map(crate::memory::sparse::MemoryByteState::Known)
                    .collect();
                core.state.ub.write_states(4096, &states).unwrap();
            } else {
                core.local_memory
                    .l1_mut()
                    .write_known_linear(4096, &input)
                    .unwrap();
            }
            core.state
                .scalar_mut()
                .machine_mut()
                .set_xreg(3, 0)
                .unwrap();
            assert!(matches!(
                core.step_word_at(tick + 1, word).unwrap(),
                C220CoreStep::Executed {
                    instruction: C220CoreInstruction::FixpQueued { .. },
                    ..
                }
            ));
            assert_eq!(core.queued_fixp_commands(), 2);
            assert!(core.pending_compute_drain().is_some());
            core.advance_to(tick + 199).unwrap();
            assert!(core.fixp_engine().unwrap().is_idle());
            assert_eq!(core.factor_outcomes().len(), 2);
            assert_eq!(core.hardware_flags.pending_mte_flags().count(), 1);
            let result = core.factor_outcomes()[0];
            let empty = core.factor_outcomes()[1];
            assert_eq!(empty.completed_tick, tick + 5);
            assert_eq!(empty.retired_tick, result.retired_tick + 1);
            assert_eq!(empty.result.blocks, 0);
            assert!(result.retired_tick > result.completed_tick);
            assert_eq!(result.result.output_bytes, 256);
            assert_eq!(
                core.fixp_factors().unwrap().read_known(0, 256).unwrap(),
                3_f32.to_le_bytes().repeat(64)
            );
        }
        let flag_word = (2 << 29) | (15 << 21) | (3 << 15) | (10 << 10) | (2 << 7) | 1;
        let flag_id = core.next_instruction_id;
        assert!(matches!(
            core.step_word_at(901, flag_word).unwrap(),
            C220CoreStep::Executed {
                instruction: C220CoreInstruction::FixpQueued {
                    ready_tick: 902,
                    ..
                },
                ..
            }
        ));
        core.advance_to(903).unwrap();
        assert!(core.fixp_engine().unwrap().control_commands().is_empty());
        core.advance_to(905).unwrap();
        assert_eq!(core.queued_fixp_commands(), 0);
        assert_eq!(
            core.fixp_engine().unwrap().control_commands().get(&flag_id),
            Some(&906)
        );
        assert!(core.pending_compute_drain().is_some());
        core.advance_to(906).unwrap();
        assert!(core.fixp_engine().unwrap().is_idle());
        assert!(core.pending_compute_drain().is_none());
        let empty_factor = (6 << 29) | (1 << 17) | (2 << 12) | (3 << 7);
        let predecessor = core.next_instruction_id;
        assert!(matches!(
            core.step_word_at(907, empty_factor).unwrap(),
            C220CoreStep::Executed { .. }
        ));
        for tick in [908, 909] {
            assert!(matches!(core.step_word_at(tick, 0x40e0_2800).unwrap(),
                C220CoreStep::Executed {
                    instruction: C220CoreInstruction::FixpBarrier { barrier, completed_tick: None },
                    ..
                } if barrier.predecessor == Some(predecessor)
            ));
        }
        assert_eq!(core.pending_fixp_barriers().len(), 2);
        assert_eq!(core.queued_fixp_commands(), 1);
        assert_eq!(core.outstanding_fixp_commands(), 1);
        assert!(matches!(
            core.step_word_at(910, empty_factor).unwrap(),
            C220CoreStep::Executed { .. }
        ));
        assert!(matches!(
            core.step_word_at(911, 0x4140_0000).unwrap(),
            C220CoreStep::Executed {
                instruction: C220CoreInstruction::Scalar { .. },
                ..
            }
        ));
        assert_eq!(core.fixp_issue_queue_len(), 1);
        assert!(core.fixp_frontend_outcomes().iter().any(|event| matches!(
            event,
            C220CoreStep::Stalled(crate::sim::c220::schedule::C220Stall {
                cause: crate::sim::c220::schedule::C220StallCause::FixpBarrier,
                ..
            })
        )));
        core.advance_to(912).unwrap();
        assert_eq!(core.pending_fixp_barriers().len(), 0);
        assert_eq!(core.fixp_issue_queue_len(), 0);
        assert_eq!(
            core.fixp_frontend_outcomes()
                .iter()
                .filter(|event| matches!(
                    event,
                    C220CoreStep::Executed {
                        instruction: C220CoreInstruction::FixpBarrier {
                            completed_tick: Some(912),
                            ..
                        },
                        ..
                    }
                ))
                .count(),
            2
        );
        assert!(core.fixp_frontend_outcomes().iter().any(|event| matches!(
            event,
            C220CoreStep::Executed {
                instruction: C220CoreInstruction::FixpScheduled {
                    ready_tick: 915,
                    ..
                },
                ..
            }
        )));
        core.advance_to(916).unwrap();
        assert!(core.fixp_engine().unwrap().is_idle());
        assert!(matches!(
            core.step_word_at(917, 0x40e0_2800).unwrap(),
            C220CoreStep::Executed {
                instruction: C220CoreInstruction::FixpBarrier {
                    completed_tick: Some(917),
                    ..
                },
                ..
            }
        ));
        core.step_word_at(918, empty_factor).unwrap();
        let cross_word = (2 << 29) | (15 << 21) | (4 << 18) | (10 << 10) | (6 << 2);
        core.state
            .scalar_mut()
            .machine_mut()
            .set_xreg(6, 0xe20)
            .unwrap();
        let cross_id = core.next_instruction_id;
        let cross_pc = core.state.scalar().pc();
        assert!(matches!(
            core.step_word_at(919, cross_word).unwrap(),
            C220CoreStep::Executed {
                instruction: C220CoreInstruction::FixpQueued { .. },
                ..
            }
        ));
        core.state
            .scalar_mut()
            .machine_mut()
            .set_xreg(6, 0)
            .unwrap();
        core.step_word_at(920, empty_factor).unwrap();
        assert!(core.fixp_frontend_outcomes().iter().any(|event| matches!(
            event,
            C220CoreStep::Stalled(crate::sim::c220::schedule::C220Stall {
                cause: crate::sim::c220::schedule::C220StallCause::FixpDependency,
                ..
            })
        )));
        core.advance_to(926).unwrap();
        assert!(core.last_fixp_cross_core_outcomes().is_empty());
        let command = core.fixp_engine().unwrap().cross_core_commands()[&cross_id];
        assert_eq!(command.dispatched_tick, 926);
        assert_eq!(command.ready_tick, 927);
        assert_eq!(command.payload.value, 0xe20);
        core.advance_to(927).unwrap();
        let reception = core.last_fixp_cross_core_outcomes()[0];
        assert_eq!(reception.instruction_id, cross_id);
        assert_eq!(reception.pc, cross_pc);
        assert_eq!(reception.tick, 927);
        assert_eq!((reception.payload.mode, reception.payload.flag_id), (2, 14));
        assert!(core.fixp_engine().unwrap().cross_core_commands().is_empty());
        assert!(core.pending_compute_drain().is_some());
        core.advance_to(928).unwrap();
        assert!(core.fixp_engine().unwrap().is_idle());
        assert!(core.pending_compute_drain().is_none());
        assert_eq!(core.hardware_flags.pending_mte_flags().count(), 2);
    }

    #[test]
    fn external_fixp_core_waits_for_transport_before_retirement() {
        use crate::memory::region::MemoryRegion;
        use crate::sim::c220::core::C220CoreFixpConfig;
        use crate::sim::c220::memory::C220LocalBuffer;
        use crate::sim::c220::mte::fixp::{
            C220FixpAtomicConfig, C220FixpEngineConfig, C220FixpRuntimeStage::*,
        };
        use crate::sim::c220::mte::interface::biu_write::command::C220BiuWriteConfig;
        use crate::sim::c220::schedule::{C220Stall, C220StallCause};
        let mut core = matrix_core();
        core.advance_to(300).unwrap();
        core.memory = MappedMemory::bind(
            SparseMemory::new(
                vec![MemoryRegion::new(4096, vec![0; 4096]).unwrap()],
                4096,
                4096,
            ),
            &[4096],
        )
        .unwrap();
        core.configure_fixp_runtime(
            C220CoreFixpConfig {
                frontend: super::super::C220FixpFrontendConfig {
                    issue_queue_depth: NonZeroU32::new(2).unwrap(),
                    outstanding_limit: NonZeroU32::new(1).unwrap(),
                },
                engine: C220FixpEngineConfig {
                    instruction_fifo_depth: 1,
                    read_bandwidth: 256,
                    read_bank_count: 32,
                    read_data_latency: 4,
                    l0c_capacity: core.local_memory.l0c().buffer().capacity(),
                },
                main_transpose_slots: 8,
                total_transpose_slots: 16,
                atomics: C220FixpAtomicConfig::default(),
            },
            C220LocalBuffer::new(4096),
            &[
                GenerateRead,
                SendRead,
                SendL0c,
                ReceiveL0c,
                Convert,
                Slice,
                Transpose,
                Align,
                Packetize,
                GenerateWrite,
                SendWrite,
            ],
            C220BiuWriteConfig {
                outstanding: NonZeroU32::new(1).unwrap(),
                weights: [1; 3],
                source_bandwidth: NonZeroU32::new(32).unwrap(),
            },
        )
        .unwrap();
        core.configure_factor_reads(crate::sim::c220::core::C220FactorReadConfig {
            port: crate::sim::c220::mte::interface::C220MteL1ReadPort::Port2,
            access_width: NonZeroU32::new(32).unwrap(),
            output_bandwidth: NonZeroU32::new(32).unwrap(),
        })
        .unwrap();
        core.local_memory
            .l1_mut()
            .write_known_linear(8192, &0x3f00_0000_u64.to_le_bytes().repeat(16))
            .unwrap();
        let factor_word = (6 << 29) | (1 << 17) | (2 << 12) | (3 << 7);
        let machine = core.state.scalar_mut().machine_mut();
        for (register, value) in [(1, 0), (2, 8192), (3, (1 << 16) | (1 << 4))] {
            machine.set_xreg(register, value).unwrap();
        }
        let issue_pc = core.state.scalar().pc();
        let captured = core.capture_fixp_command(issue_pc, factor_word).unwrap();
        let captured_id = captured.issue.instruction_id;
        core.next_instruction_id += 1;
        let machine = core.state.scalar_mut().machine_mut();
        machine.set_xreg(2, 0).unwrap();
        machine.set_xreg(3, 0).unwrap();
        core.advance_to(301).unwrap();
        assert!(matches!(
            core.dispatch_captured_fixp_at(301, captured).unwrap(),
            C220CoreStep::Executed {
                instruction: C220CoreInstruction::Factor { instruction_id, .. },
                ..
            } if instruction_id == captured_id
        ));
        assert_eq!(core.state.scalar().pc(), issue_pc);
        assert_eq!(core.next_instruction_id, captured_id + 1);
        core.state.commit_c220_sequential_issue();
        core.clock.finish(301).unwrap();
        let machine = core.state.scalar_mut().machine_mut();
        for (register, value) in [
            (1, 4096),
            (2, 0),
            (3, (8 << 32) | (8 << 16) | (16 << 4)),
            (4, 23 << 34),
        ] {
            machine.set_xreg(register, value).unwrap();
        }
        for register in [3, 61, 64, 93, 94, 97] {
            machine.set_spr_value(register, 0).unwrap();
        }
        core.local_memory
            .l0c_mut()
            .buffer_mut()
            .write_known_linear(0, &2_f32.to_le_bytes().repeat(128))
            .unwrap();
        let pc = core.state.scalar().pc();
        let word = (6 << 29) | (2 << 24) | (1 << 17) | (2 << 12) | (3 << 7) | (4 << 2);
        core.advance_to(350).unwrap();
        assert_eq!(core.factor_outcomes().len(), 1);
        assert_eq!(
            core.fixp_factors().unwrap().read_known(0, 128).unwrap(),
            0x3f00_0000_u64.to_le_bytes().repeat(16)
        );
        assert!(matches!(
            core.step_word_at(351, word | (1 << 24)).unwrap(),
            C220CoreStep::Executed {
                instruction: C220CoreInstruction::FixpQueued { .. },
                ..
            }
        ));
        assert_eq!(core.state.scalar().pc(), pc + 4);
        core.advance_to(400).unwrap();
        assert_eq!(
            core.local_memory.l1().read_known(4096, 128).unwrap(),
            [1; 128]
        );
        assert!(core.fixp_engine().unwrap().is_idle());
        assert!(matches!(
            core.step_word_at(401, word).unwrap(),
            C220CoreStep::Executed {
                instruction: C220CoreInstruction::FixpQueued { .. },
                ..
            }
        ));
        assert_eq!(core.state.scalar().pc(), pc + 8);
        assert!(core.pending_compute_drain().is_some());
        assert!(core.fixp_engine().unwrap().hardware_flag_trigger_ready());
        core.state
            .scalar_mut()
            .machine_mut()
            .set_xreg(3, 0)
            .unwrap();
        assert!(matches!(
            core.step_word_at(402, factor_word).unwrap(),
            C220CoreStep::Executed {
                instruction: C220CoreInstruction::FixpQueued { .. },
                ..
            }
        ));
        assert_eq!(core.queued_fixp_commands(), 2);
        let mut responses = std::collections::VecDeque::new();
        let mut sent_bytes = 0;
        let mut last_response = None;
        let mut trigger_blocked = false;
        let mut trigger_released = false;
        let trigger_word = (2 << 29) | (15 << 21) | (1 << 19) | (3 << 15) | (10 << 10) | (2 << 7);
        for tick in 403..900 {
            core.advance_to(tick).unwrap();
            let engine = core.fixp_engine().unwrap();
            if !engine.hardware_flag_trigger_ready() {
                let pc = core.state.scalar().pc();
                for word in [trigger_word, trigger_word | (1 << 5)] {
                    let instruction =
                        crate::isa::c220::hflag::C220HardwareFlagInstruction::decode(word).unwrap();
                    assert!(matches!(
                        core.dispatch_hardware_flag_at(
                            tick,
                            core.next_instruction_id,
                            instruction
                                .resolve(pc, core.state.scalar().machine().xregs())
                                .unwrap()
                        )
                        .unwrap(),
                        C220CoreStep::Stalled(C220Stall {
                            cause: C220StallCause::HardwareFlagDependency,
                            ..
                        })
                    ));
                    assert_eq!(core.state.scalar().pc(), pc);
                }
                trigger_blocked = true;
            } else if !trigger_released
                && engine.instruction_fifo().is_empty()
                && engine.outstanding_external_commands() != 0
            {
                assert!(!core.fixp_runtime().unwrap().is_idle());
                assert_eq!(core.fixp_issue_queue_len(), 1);
                assert_eq!(core.outstanding_fixp_commands(), 1);
                assert!(core.fixp_frontend_outcomes().iter().any(|outcome| matches!(
                    outcome,
                    C220CoreStep::Stalled(C220Stall {
                        cause: C220StallCause::FixpOutstandingLimit,
                        ..
                    })
                )));
                let instruction =
                    crate::isa::c220::hflag::C220HardwareFlagInstruction::decode(trigger_word)
                        .unwrap();
                assert!(matches!(
                    core.dispatch_hardware_flag_at(
                        tick,
                        core.next_instruction_id,
                        instruction
                            .resolve(
                                core.state.scalar().pc(),
                                core.state.scalar().machine().xregs()
                            )
                            .unwrap()
                    )
                    .unwrap(),
                    C220CoreStep::Executed { .. }
                ));
                trigger_released = true;
            }
            if let Some(command) = core.take_biu_write_command_at(tick).unwrap() {
                core.receive_external_fixp_dbid_at(tick, command.command.tag)
                    .unwrap();
            }
            if let Some(data) = core.take_biu_write_data_at(tick).unwrap() {
                sent_bytes += data.source.request.bytes;
                responses.push_back((tick + 10, data.source.request.tag));
            }
            if let Some((_, tag)) = responses.pop_front_if(|(ready, _)| *ready <= tick) {
                assert!(!core.fixp_runtime().unwrap().is_idle());
                core.receive_external_fixp_response_at(tick, tag).unwrap();
                assert!(!core.fixp_runtime().unwrap().is_idle());
                last_response = Some(tick);
            }
            if core.fixp_runtime().unwrap().is_idle() && core.queued_fixp_commands() == 0 {
                assert!(tick > last_response.unwrap() + 1);
                break;
            }
        }
        assert_eq!(sent_bytes, 128);
        assert!(trigger_blocked && trigger_released);
        assert!(core.fixp_runtime().unwrap().is_idle());
        assert!(core.pending_compute_drain().is_none());
        assert_eq!(core.memory.read_known_at(4096, 128).unwrap(), [1; 128]);
        assert_eq!(core.memory.read_known_at(4224, 128).unwrap(), [0; 128]);
        core.advance_to(1000).unwrap();
        core.local_memory
            .l1_mut()
            .write_known_linear(8192, &[0; 128])
            .unwrap();
        let machine = core.state.scalar_mut().machine_mut();
        for (register, value) in [
            (1, 8192),
            (2, 0),
            (3, (32 << 32) | (2 << 16) | (17 << 4)),
            (4, (1 << 43) | (1 << 34) | 2),
        ] {
            machine.set_xreg(register, value).unwrap();
        }
        machine.set_spr_value(97, 1).unwrap();
        assert!(matches!(
            core.step_word_at(1001, word | (1 << 24)).unwrap(),
            C220CoreStep::Executed {
                instruction: C220CoreInstruction::FixpQueued { .. },
                ..
            }
        ));
        core.advance_to(1005).unwrap();
        assert_eq!(
            core.fixp_engine()
                .unwrap()
                .active_resource_key()
                .unwrap()
                .destination,
            crate::isa::c220::mte::fixp::C220FixpDestination::L1
        );
        let mut nz_bytes = 0;
        assert_eq!(
            core.fixp_engine().unwrap().outstanding_external_commands(),
            0
        );
        for tick in 1006..1500 {
            core.advance_to(tick).unwrap();
            if let Some(command) = core.take_biu_write_command_at(tick).unwrap() {
                core.receive_external_fixp_dbid_at(tick, command.command.tag)
                    .unwrap();
            }
            if let Some(data) = core.take_biu_write_data_at(tick).unwrap() {
                nz_bytes += data.source.request.bytes;
                core.receive_external_fixp_response_at(tick, data.source.request.tag)
                    .unwrap();
                assert!(!core.fixp_runtime().unwrap().is_idle());
            }
            if core.fixp_runtime().unwrap().is_idle() {
                break;
            }
        }
        assert!(core.fixp_runtime().unwrap().is_idle());
        assert_eq!(nz_bytes, 68);
        for row in 0..2 {
            assert_eq!(
                core.local_memory
                    .l1()
                    .read_known(8192 + row * 64, 34)
                    .unwrap(),
                0x4000_u16.to_le_bytes().repeat(17)
            );
            assert_eq!(
                core.local_memory
                    .l1()
                    .read_known(8226 + row * 64, 30)
                    .unwrap(),
                [0; 30]
            );
        }
        assert_eq!(core.memory.read_known_at(4096, 128).unwrap(), [1; 128]);
    }

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
            core_kind: crate::sim::c220::device::C220CoreKind::Cube,
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
    fn cube_first_uop_observes_live_delay_controls() {
        for initially_enabled in [false, true] {
            let mut core = matrix_core();
            core.advance_to(300).unwrap();
            let delay = (1 << 4) | (10 << 5);
            core.state
                .scalar_mut()
                .machine_mut()
                .set_spr_value(107, if initially_enabled { delay } else { 0 })
                .unwrap();
            let word = (7 << 29) | (3 << 22) | (1 << 12) | (2 << 7) | (3 << 2);
            assert!(matches!(
                core.step_word_at(301, word).unwrap(),
                C220CoreStep::Executed { .. }
            ));
            core.advance_to(302).unwrap();
            assert!(core.cube.pipeline.last_uop_releases().is_empty());
            core.state
                .scalar_mut()
                .machine_mut()
                .set_spr_value(107, if initially_enabled { 0 } else { delay })
                .unwrap();
            let expected = if initially_enabled { 304 } else { 312 };
            core.advance_to(expected - 1).unwrap();
            assert!(core.cube.pipeline.last_uop_releases().is_empty());
            core.advance_to(expected).unwrap();
            assert_eq!(core.cube.pipeline.last_uop_releases().len(), 1);
            assert_eq!(
                core.cube.pipeline.last_uop_releases()[0].issue_tick,
                expected
            );
            core.advance_to(expected + 21).unwrap();
            let retired = core.cube.pipeline.last_retirements();
            assert_eq!(retired.len(), 1);
            assert_eq!(retired[0].first_uop_tick, Some(expected));
            assert_eq!(
                retired[0].issue_delay_wait_ticks,
                if initially_enabled { 1 } else { 9 }
            );
            assert_eq!(retired[0].resource_wait_ticks, 0);
        }
    }

    #[test]
    fn zero_dimension_cube_retires_without_output_or_status_changes() {
        for (m, k, n) in [(0, 17, 17), (17, 0, 17), (17, 17, 0)] {
            let mut core = matrix_core();
            core.advance_to(300).unwrap();
            let machine = core.state.scalar_mut().machine_mut();
            machine.set_xreg(0, 4096).unwrap();
            machine.set_xreg(1, u64::MAX).unwrap();
            machine.set_xreg(2, u64::MAX).unwrap();
            machine
                .set_xreg(3, m | (k << 12) | (n << 24) | (1 << 63))
                .unwrap();
            machine.set_spr_value(2, 0x1234_5678_9abc_def0).unwrap();
            core.local_memory
                .l0c_mut()
                .buffer_mut()
                .write_known(4096, &[0xa5; 4096])
                .unwrap();
            let mut before = core.local_memory.clone();
            before.l0c_mut().read_banks_mut().advance_to(301).unwrap();
            let word = (7 << 29) | (3 << 22) | (1 << 12) | (2 << 7) | (3 << 2);
            let C220CoreStep::Executed {
                instruction: C220CoreInstruction::Cube(issue),
                ..
            } = core.step_word_at(301, word).unwrap()
            else {
                panic!("Cube admission");
            };
            assert_eq!(issue.ticket.retire_tick, 301);
            assert_eq!(issue.ticket.uop_count, 0);
            core.advance_to(301).unwrap();
            assert!(core.cube.pipeline.last_uop_releases().is_empty());
            assert_eq!(core.cube.pipeline.pending_retirement_count(), 0);
            assert_eq!(core.last_cube_outcomes().len(), 1);
            assert_eq!(core.last_cube_outcomes()[0].written_lanes, 0);
            assert_eq!(core.local_memory, before);
            assert_eq!(core.state.scalar().machine().spr2(), 0x1234_5678_9abc_def0);
        }
    }

    #[test]
    fn cube_publishes_each_output_request_without_retirement_rewrite() {
        let mut core = matrix_core();
        core.advance_to(300).unwrap();
        for (register, value) in [
            (0, 4096),
            (1, 0),
            (2, 0),
            (3, 1 | (16 << 12) | (17 << 24) | (1 << 63)),
        ] {
            core.state
                .scalar_mut()
                .machine_mut()
                .set_xreg(register, value)
                .unwrap();
        }
        core.local_memory
            .l0a_mut()
            .write_known(0, &0x3c00_u16.to_le_bytes().repeat(256))
            .unwrap();
        core.local_memory
            .l0b_mut()
            .write_known(0, &0x3c00_u16.to_le_bytes().repeat(512))
            .unwrap();
        core.local_memory
            .l0c_mut()
            .buffer_mut()
            .write_known(4096, &[0; 2048])
            .unwrap();
        let word = (7 << 29) | (3 << 22) | (1 << 12) | (2 << 7) | (3 << 2);
        let C220CoreStep::Executed {
            instruction: C220CoreInstruction::Cube(issue),
            ..
        } = core.step_word_at(301, word).unwrap()
        else {
            panic!("Cube admission");
        };
        let mut saw_first = false;
        let mut retired = false;
        for tick in 302..400 {
            core.advance_to(tick).unwrap();
            if core
                .cube
                .pipeline
                .last_uop_releases()
                .iter()
                .any(|release| {
                    release.instruction_id == issue.instruction_id && release.uop.id == 0
                })
            {
                assert_eq!(
                    core.local_memory
                        .l0c()
                        .buffer()
                        .read_known(4096, 4)
                        .unwrap(),
                    16.0_f32.to_le_bytes()
                );
                assert!(core.last_cube_outcomes().is_empty());
                core.local_memory
                    .l0c_mut()
                    .buffer_mut()
                    .write_known(4096, &99.0_f32.to_le_bytes())
                    .unwrap();
                core.local_memory
                    .l0a_mut()
                    .write_known(0, &[0; 512])
                    .unwrap();
                saw_first = true;
            }
            if !core.last_cube_outcomes().is_empty() {
                retired = true;
                break;
            }
        }
        assert!(saw_first && retired);
        assert_eq!(
            core.local_memory
                .l0c()
                .buffer()
                .read_known(4096, 4)
                .unwrap(),
            99.0_f32.to_le_bytes()
        );
        assert_eq!(
            core.local_memory
                .l0c()
                .buffer()
                .read_known(5120, 4)
                .unwrap(),
            16.0_f32.to_le_bytes()
        );
    }

    #[test]
    fn sparse_load_reads_both_streams_but_only_weights_emit_output() {
        use crate::sim::c220::mte::interface::{C220MteL1EventOutcome, C220MteL1OutputDestination};
        use crate::sim::c220::mte::mte1::C220Mte1TransferResult;
        let mut core = matrix_core();
        core.advance_to(300).unwrap();
        for (register, value) in [(10, 4096), (11, (16384 << 32) | 8192), (12, 2 << 16)] {
            core.state
                .scalar_mut()
                .machine_mut()
                .set_xreg(register, value)
                .unwrap();
        }
        let word = (3 << 29) | (1 << 27) | (24 << 22) | (10 << 17) | (11 << 12) | (12 << 7);
        let C220CoreStep::Executed {
            instruction:
                C220CoreInstruction::Mte1Queued(super::super::C220Mte1QueuedCommand {
                    instruction_id,
                    ..
                }),
            ..
        } = core.step_word_at(301, word).unwrap()
        else {
            panic!("sparse instruction admission");
        };
        core.local_memory
            .l1_mut()
            .write_known(8192, &[0x42; 1024])
            .unwrap();
        core.local_memory
            .l1_mut()
            .write_known(16384, &[0xa5; 256])
            .unwrap();
        for register in 10..=12 {
            core.state
                .scalar_mut()
                .machine_mut()
                .set_xreg(register, 0)
                .unwrap();
        }
        let mut indices = 0;
        let mut weight_output_bytes = 0;
        let mut retired = None;
        for tick in 302..450 {
            core.advance_to(tick).unwrap();
            if tick == 304 {
                assert!(core.mte1_frontend_outcomes().iter().any(|step| matches!(step,
                    C220CoreStep::Executed { instruction: C220CoreInstruction::Mte1 { issue, .. }, .. } if issue.uop_count == 6
                )));
            }
            for event in core.mte_pipeline().unwrap().last_events() {
                match event {
                    C220MtePipelineEvent::Interface(C220MteL1EventOutcome::Response(Some(
                        request,
                    ))) if request.operation.instruction_id == instruction_id
                        && request.operation.destination
                            == C220MteL1OutputDestination::SparseIndex =>
                    {
                        indices += 1;
                        assert!(!request.operation.completes_logical_uop);
                        assert!(!request.operation.last_in_instruction);
                    }
                    C220MtePipelineEvent::Interface(C220MteL1EventOutcome::Output(output)) => {
                        if let Some(sent) = output.sent
                            && sent.fragment.instruction_id == instruction_id
                        {
                            assert!(matches!(
                                sent.destination,
                                C220MteL1OutputDestination::L0b(_)
                            ));
                            weight_output_bytes += sent.fragment.bytes;
                        }
                    }
                    _ => {}
                }
            }
            if let Some(outcome) = core
                .last_mte1_outcomes()
                .iter()
                .find(|o| o.instruction_id == instruction_id)
            {
                retired = Some(*outcome);
            }
            if retired.is_none() {
                assert!(core.local_memory.l0b().read_known(4096, 512).is_err());
                assert_eq!(core.local_memory.weight_index().tracked_bytes(), 0);
            }
        }
        assert_eq!((indices, weight_output_bytes), (2, 1024));
        let C220Mte1TransferResult::Load2dSparse(result) =
            retired.expect("sparse retirement").result
        else {
            panic!("sparse result")
        };
        assert_eq!((result.weight_bytes, result.index_bytes), (1024, 256));
        assert_eq!(
            core.local_memory.l0b().read_known(4096, 1024).unwrap(),
            vec![0x42; 1024]
        );
        assert_eq!(
            core.local_memory
                .weight_index()
                .read_initialized_linear(1024, 256)
                .unwrap(),
            vec![0xa5; 256]
        );
        assert!(core.pending_mte1_commands().next().is_none());
        assert!(core.mte_pipeline().unwrap().is_idle());
    }

    #[test]
    fn mte1_queue_captures_early_spr_writes_and_preserves_backpressure() {
        use crate::sim::c220::schedule::{C220Stall, C220StallCause};
        let mut core = matrix_core();
        core.advance_to(300).unwrap();
        let machine = core.state.scalar_mut().machine_mut();
        machine.set_spr_value(15, 0x1111).unwrap();
        machine.set_xreg(10, 4096).unwrap();
        machine.set_xreg(11, 16 | (1 << 16)).unwrap();
        machine.set_xreg(12, 0x2222).unwrap();
        let fill = (3 << 29) | (1 << 22) | (10 << 17) | (11 << 7);
        let spr = (2 << 24) | (15 << 17) | (12 << 12) | (18 << 7);
        core.step_word_at(301, fill).unwrap();
        core.step_word_at(302, spr).unwrap();
        assert_eq!(core.state.scalar().machine().spr_value(15), Some(0x2222));
        let machine = core.state.scalar_mut().machine_mut();
        machine.set_xreg(10, 16384).unwrap();
        machine.set_xreg(11, 1 | (1 << 16)).unwrap();
        core.step_word_at(303, fill).unwrap();
        assert_eq!(core.queued_mte1_commands().count(), 3);
        assert!(core.pending_mte1_commands().next().is_none());
        core.state
            .scalar_mut()
            .machine_mut()
            .set_xreg(12, 0x3333)
            .unwrap();
        core.step_word_at(304, spr).unwrap();
        assert_eq!(core.pending_mte1_commands().count(), 1);
        assert_eq!(core.state.scalar().machine().spr_value(15), Some(0x3333));
        core.state
            .scalar_mut()
            .machine_mut()
            .set_xreg(12, 0x4444)
            .unwrap();
        let pc = core.state.scalar().pc();
        assert!(matches!(
            core.step_word_at(305, spr).unwrap(),
            C220CoreStep::Stalled(C220Stall {
                cause: C220StallCause::Mte1CommandQueueFull,
                ..
            })
        ));
        assert_eq!(core.state.scalar().pc(), pc);
        assert_eq!(core.state.scalar().machine().spr_value(15), Some(0x3333));
        assert!(core.mte1_frontend_outcomes().iter().any(|step| matches!(
            step,
            C220CoreStep::Stalled(C220Stall {
                cause: C220StallCause::Mte1IssueRate,
                ..
            })
        )));
        let set = (2 << 29) | (5 << 21) | (3 << 10) | (2 << 7) | 3;
        let wait = (set & !(15 << 21)) | (6 << 21);
        assert!(matches!(
            core.step_word_at(306, set).unwrap(),
            C220CoreStep::Executed { .. }
        ));
        assert!(matches!(
            core.step_word_at(307, wait).unwrap(),
            C220CoreStep::Stalled(_)
        ));
        core.advance_to(700).unwrap();
        assert_eq!(core.outstanding_mte1_commands(), 0);
        assert_eq!(core.last_mte1_outcomes().len(), 4);
        assert!(
            core.last_mte1_outcomes()
                .windows(2)
                .all(|pair| pair[0].instruction_id < pair[1].instruction_id
                    && pair[0].retire_tick < pair[1].retire_tick)
        );
        assert_eq!(core.state.scalar().machine().spr_value(15), Some(0x3333));
        assert_eq!(
            core.local_memory.l0a().read_known(4096, 8192).unwrap(),
            0x1111_u16.to_le_bytes().repeat(4096)
        );
        assert_eq!(
            core.local_memory.l0a().read_known(16384, 512).unwrap(),
            0x2222_u16.to_le_bytes().repeat(256)
        );
        assert!(matches!(
            core.step_word_at(701, wait).unwrap(),
            C220CoreStep::Executed { .. }
        ));
    }

    #[test]
    fn load3dv2_runs_through_shared_l1_and_commits_spr54() {
        use crate::sim::c220::mte::mte1::C220Mte1TransferResult;
        use crate::sim::c220::schedule::{C220Stall, C220StallCause};
        let mut core = matrix_core();
        let machine = core.state.scalar_mut().machine_mut();
        for (register, value) in [
            (10, 4096),
            (11, 8192),
            (12, 40 | (16 << 16)),
            (13, (1 << 12) | (1 << 20) | (40 << 48)),
        ] {
            machine.set_xreg(register, value).unwrap();
        }
        let mut setup_tick = 280;
        let selected = core.mte_pipeline().unwrap().selected_generator();
        for (register, mask) in [
            (10, u64::MAX),
            (13, 0xffff_ffff),
            (15, 0xffff_ffff),
            (22, 0xffff_ffff),
            (53, u64::MAX),
            (54, 0xfff),
            (58, 0xffff_ffff),
            (92, u64::MAX),
        ] {
            core.state
                .scalar_mut()
                .machine_mut()
                .set_xreg(14, u64::MAX)
                .unwrap();
            let word = (2 << 24) | (register << 17) | (14 << 12) | (18 << 7);
            assert_eq!(
                crate::isa::c220::mte::read_register_mask(word),
                Some(1 << 14)
            );
            let C220CoreStep::Executed {
                instruction:
                    C220CoreInstruction::Mte1Queued(super::super::C220Mte1QueuedCommand {
                        command: C220Mte1Command::WriteSpr(step),
                        ..
                    }),
                ..
            } = core.step_word_at(setup_tick, word).unwrap()
            else {
                panic!("MTE1 SPR write should issue");
            };
            assert_eq!(step.value, mask);
            assert_eq!(
                core.state.scalar().machine().spr_value(register as u16),
                Some(mask)
            );
            assert_eq!(core.mte_pipeline().unwrap().selected_generator(), selected);
            setup_tick += 1;
        }
        for (register, value) in [
            (10, 4 | (4 << 16)),
            (92, 0),
            (13, 0),
            (58, 1 << 16),
            (22, 0),
            (54, 0),
        ] {
            core.state
                .scalar_mut()
                .machine_mut()
                .set_xreg(14, value)
                .unwrap();
            let word = (2 << 24) | (register << 17) | (14 << 12) | (18 << 7);
            assert!(matches!(
                core.step_word_at(setup_tick, word).unwrap(),
                C220CoreStep::Executed { .. }
            ));
            setup_tick += 1;
        }
        core.advance_to(300).unwrap();
        let word = (3 << 29) | (20 << 22) | (10 << 17) | (11 << 12) | (12 << 7) | (13 << 2);
        let C220CoreStep::Executed {
            instruction: C220CoreInstruction::Mte1Queued(_),
            ..
        } = core.step_word_at(301, word).unwrap()
        else {
            panic!("LOAD3Dv2 admission");
        };
        let read_spr54 = (2 << 24) | (1 << 22) | (14 << 17) | (22 << 12) | (17 << 7);
        let waiting_pc = core.state.scalar().pc();
        assert!(matches!(
            core.step_word_at(302, read_spr54).unwrap(),
            C220CoreStep::Stalled(C220Stall {
                cause: C220StallCause::Mte1Dependency,
                ..
            })
        ));
        assert_eq!(core.state.scalar().pc(), waiting_pc);
        core.state
            .scalar_mut()
            .machine_mut()
            .set_xreg(11, 0)
            .unwrap();
        core.local_memory
            .l1_mut()
            .write_known(8192, &[0x73; 640])
            .unwrap();
        let outcome = (303..600)
            .find_map(|tick| {
                core.advance_to(tick).unwrap();
                if tick == 304 {
                    assert!(core.mte1_frontend_outcomes().iter().any(|step| matches!(step,
                        C220CoreStep::Executed { instruction: C220CoreInstruction::Mte1 { issue, .. }, .. } if issue.uop_count == 8
                    )));
                }
                core.last_mte1_outcomes().first().copied()
            })
            .expect("LOAD3Dv2 retirement");
        assert!(outcome.retire_tick > 321);
        let C220Mte1TransferResult::Load3dv2(report) = outcome.result else {
            panic!("LOAD3Dv2 result");
        };
        assert_eq!(report.output_packets, 2);
        assert_eq!(core.state.scalar().machine().spr_value(54), Some(40));
        assert_eq!(
            core.local_memory.l0a().read_known(4096, 512).unwrap(),
            [0x73; 512]
        );
        let tail = core.local_memory.l0a().read_known(4608, 512).unwrap();
        for row in tail.chunks_exact(32) {
            assert_eq!(&row[..8], &[0x73; 8]);
            assert_eq!(&row[8..], &[0; 24]);
        }
        assert!(core.pending_mte1_commands().next().is_none());
        assert!(matches!(
            core.step_word_at(outcome.retire_tick + 1, read_spr54)
                .unwrap(),
            C220CoreStep::Executed { .. }
        ));
        assert_eq!(core.state.scalar().machine().xregs()[14], 40);
    }

    #[test]
    fn grouped_transpose_runs_through_mte1_and_retires_captured_operands() {
        use crate::sim::c220::mte::mte1::C220Mte1TransferResult;

        for (format, group) in [(0, 2_usize), (1, 1), (2, 4), (3, 2)] {
            for destination in 0..2 {
                let mut core = matrix_core();
                core.advance_to(300).unwrap();
                let machine = core.state.scalar_mut().machine_mut();
                for (register, value) in [
                    (10, 4096),
                    (11, 8192),
                    (12, (2 << 16) | (1 << 24) | (7 << 44)),
                    (13, 1),
                ] {
                    machine.set_xreg(register, value).unwrap();
                }
                let word = (3 << 29)
                    | (1 << 27)
                    | (5 << 24)
                    | (format << 22)
                    | (10 << 17)
                    | (11 << 12)
                    | (12 << 7)
                    | (13 << 2)
                    | destination;
                let C220CoreStep::Executed {
                    instruction: C220CoreInstruction::Mte1Queued(_),
                    ..
                } = core.step_word_at(301, word).unwrap()
                else {
                    panic!("extended transpose admission");
                };
                for register in 10..=13 {
                    core.state
                        .scalar_mut()
                        .machine_mut()
                        .set_xreg(register, 0)
                        .unwrap();
                }
                // The data is written after issue; only operands are captured at issue.
                core.local_memory
                    .l1_mut()
                    .write_known(8192, &vec![0x55; 2 * group * 512])
                    .unwrap();
                let retired = (302..600)
                    .find_map(|tick| {
                        core.advance_to(tick).unwrap();
                        if tick == 304 {
                            assert!(core.mte1_frontend_outcomes().iter().any(|step| matches!(step,
                                C220CoreStep::Executed { instruction: C220CoreInstruction::Mte1 { issue, .. }, .. } if issue.uop_count == (2 * group * 2) as u64
                            )));
                        }
                        let outcome = core.last_mte1_outcomes().first().copied();
                        let target = if destination == 0 {
                            core.local_memory.l0a()
                        } else {
                            core.local_memory.l0b()
                        };
                        if outcome.is_none() {
                            assert!(target.read_known(4096, 512).is_err());
                        }
                        outcome
                    })
                    .expect("extended transpose retirement");
                let C220Mte1TransferResult::Load2dTranspose(result) = retired.result else {
                    panic!("extended transpose result");
                };
                assert_eq!(result.segment_count, 2 * group);
                let target = if destination == 0 {
                    core.local_memory.l0a()
                } else {
                    core.local_memory.l0b()
                };
                for repeat in 0..2 {
                    for fractal in 0..group {
                        assert_eq!(
                            target
                                .read_known((4096 + (repeat * 8 + fractal * 2) * 512) as u64, 512)
                                .unwrap(),
                            vec![0x55; 512]
                        );
                    }
                }
                assert!(core.pending_mte1_commands().next().is_none());
                assert!(core.mte_pipeline().unwrap().is_idle());
            }
        }
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
            C220CoreStep::Executed {
                instruction: C220CoreInstruction::Mte1Queued(_),
                ..
            }
        ));
        let (issued_at, id) = (8..64)
            .find_map(|tick| {
                core.advance_to(tick).unwrap();
                core.mte1_frontend_outcomes()
                    .iter()
                    .find_map(|step| match *step {
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
                        _ => None,
                    })
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
            C220CoreStep::Executed {
                instruction: C220CoreInstruction::Mte1Queued(_),
                ..
            }
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
                C220CoreStep::Executed {
                    instruction: C220CoreInstruction::Mte1Queued(_),
                    ..
                }
            ));
            let (issued_at, id) = (8..64)
                .find_map(|tick| {
                    core.advance_to(tick).unwrap();
                    core.mte1_frontend_outcomes()
                        .iter()
                        .find_map(|step| match *step {
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
                            _ => None,
                        })
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
    fn mte1_retirement_backpressure_allows_flag_consumers_to_run() {
        use crate::isa::c220::hflag::{C220HardwareFlagInstruction, C220MatrixMemory};
        use crate::sim::c220::sync::C220HardwareFlagTimingError;

        for triggered_wait in [true, false] {
            let mut core = matrix_core();
            core.advance_to(64).unwrap();
            let flag = (2 << 29) | (15 << 21) | (1 << 15) | (3 << 10) | (2 << 7);
            let step = C220HardwareFlagInstruction::decode(flag)
                .unwrap()
                .resolve(
                    core.state.scalar().pc(),
                    core.state.scalar().machine().xregs(),
                )
                .unwrap();
            for _ in 0..crate::sim::c220::sync::C220_HARDWARE_FLAG_ALMOST_FULL {
                core.hardware_flags.schedule_set(step, 64).unwrap();
            }
            let machine = core.state.scalar_mut().machine_mut();
            machine.set_xreg(10, 0).unwrap();
            machine.set_xreg(11, 1 | (1 << 16)).unwrap();
            machine.set_spr_value(15, 0x4000).unwrap();
            core.step_word_at(65, flag).unwrap();
            let fill = (3 << 29) | (1 << 22) | (10 << 17) | (11 << 7);
            assert!(matches!(
                core.step_word_at(66, fill).unwrap(),
                C220CoreStep::Executed { .. }
            ));
            let before = core.local_memory.l0a().read_known(0, 512).unwrap();
            core.advance_to(100).unwrap();
            let pending = core.pending_mte1_commands().next().unwrap();
            assert!(pending.completion_tick.is_some());
            assert!(matches!(
                pending.hardware_flag_stall,
                Some(C220HardwareFlagTimingError::AlmostFull { count: 32, .. })
            ));
            assert!(
                core.last_mte1_outcomes()
                    .iter()
                    .all(|outcome| matches!(outcome.command, C220Mte1Command::HardwareFlag(_)))
            );
            assert_eq!(core.local_memory.l0a().read_known(0, 512).unwrap(), before);
            let pc = core.state.scalar().pc();
            assert!(matches!(
                core.step_word_at(101, 0x40e0_1800).unwrap(),
                C220CoreStep::Stalled(_)
            ));
            assert_eq!(core.state.scalar().pc(), pc);

            let wait = flag | (1 << 5) | (u32::from(triggered_wait) << 19);
            assert!(matches!(
                core.step_word_at(102, wait).unwrap(),
                C220CoreStep::Executed { .. }
            ));
            if !triggered_wait {
                let cube = (7 << 29) | (3 << 22) | (1 << 12) | (2 << 7) | (3 << 2);
                assert!(matches!(
                    core.step_word_at(105, cube).unwrap(),
                    C220CoreStep::Executed { .. }
                ));
            }
            core.advance_to(200).unwrap();
            assert!(core.pending_mte1_commands().next().is_none());
            assert_eq!(core.last_mte1_outcomes().len(), 2);
            assert!(core.last_mte1_outcomes()[0].retire_tick > 102);
            assert_eq!(
                core.local_memory.l0a().read_known(0, 512).unwrap(),
                0x4000_u16.to_le_bytes().repeat(256)
            );
            assert_eq!(core.hardware_flags.count(2, C220MatrixMemory::L0a, 0), 32);
            assert_eq!(core.hardware_flags.pending_cube_wait_count(), 0);
            if !triggered_wait {
                assert_eq!(core.last_cube_outcomes().len(), 1);
                assert_eq!(
                    core.local_memory.l0c().buffer().read_known(0, 4).unwrap(),
                    32.0_f32.to_le_bytes()
                );
            }
            assert!(matches!(
                core.step_word_at(201, 0x40e0_1800).unwrap(),
                C220CoreStep::Executed { .. }
            ));
        }
    }

    #[test]
    fn l1_fill_retires_captured_pattern_before_mte1_flags_release_load2d() {
        use crate::sim::c220::mte::mte2::{C220Mte2Completion, C220Mte2IssueTiming};
        let mut core = matrix_core();
        core.advance_to(64).unwrap();
        let pattern = 0x4000_4000_u64;
        let machine = core.state.scalar_mut().machine_mut();
        machine.set_xreg(1, 2048).unwrap();
        machine.set_xreg(3, 2 | (16 << 16) | (16 << 32)).unwrap();
        machine.set_spr_value(15, pattern).unwrap();
        let word = (3 << 29) | (1 << 22) | (1 << 17) | (3 << 7) | 6;
        let C220CoreStep::Executed {
            instruction: C220CoreInstruction::Mte2(issue),
            ..
        } = core.step_word_at(65, word).unwrap()
        else {
            panic!("L1 fill issue")
        };
        assert!(matches!(issue.timing, C220Mte2IssueTiming::L1(_)));
        let set = (2 << 29) | (5 << 21) | (4 << 10) | (3 << 7) | 3;
        let wait = (set & !(15 << 21)) | (6 << 21);
        for (tick, word) in [(66, set), (67, set), (68, set & !(7 << 7))] {
            assert!(matches!(
                core.step_word_at(tick, word).unwrap(),
                C220CoreStep::Executed { .. }
            ));
        }
        let pc = core.state.scalar().pc();
        assert!(matches!(
            core.step_word_at(69, wait).unwrap(),
            C220CoreStep::Stalled(_)
        ));
        assert_eq!(core.state.scalar().pc(), pc);
        core.state
            .scalar_mut()
            .machine_mut()
            .set_spr_value(15, 0)
            .unwrap();
        let mut completed = None;
        let mut retired = None;
        for tick in 70..180 {
            core.advance_to(tick).unwrap();
            if let Some(command) = core
                .mte2
                .pending_commands()
                .find(|p| p.instruction_id == issue.instruction_id)
                && let C220Mte2Completion::Observed { tick: done } = command.completion
            {
                completed = Some(done);
                assert!(core.local_memory.l1().read_known(2048, 512).is_err());
            }
            if let Some(outcome) = core
                .mte2
                .last_outcomes()
                .iter()
                .find(|o| o.command.instruction_id == issue.instruction_id)
            {
                retired = Some(outcome.retire_tick);
                break;
            }
        }
        let retired = retired.expect("L1 command retirement");
        assert_eq!(retired, completed.unwrap() + 1);
        let expected = (pattern as u32).to_le_bytes().repeat(128);
        for address in [2048, 3072] {
            assert_eq!(
                core.local_memory.l1().read_known(address, 512).unwrap(),
                expected
            );
        }
        assert!(core.local_memory.l1().read_known(2560, 512).is_err());
        for (offset, word) in [(1, wait), (2, wait), (3, wait & !(7 << 7))] {
            assert!(matches!(
                core.step_word_at(retired + offset, word).unwrap(),
                C220CoreStep::Executed { .. }
            ));
        }
        let machine = core.state.scalar_mut().machine_mut();
        machine.set_xreg(1, 8192).unwrap();
        machine.set_xreg(2, 2048).unwrap();
        machine.set_xreg(3, (2 << 16) | (2 << 24)).unwrap();
        let load = (3 << 29) | (1 << 17) | (2 << 12) | (3 << 7) | 8;
        assert!(matches!(
            core.step_word_at(retired + 4, load).unwrap(),
            C220CoreStep::Executed {
                instruction: C220CoreInstruction::Mte1Queued(_),
                ..
            }
        ));
        core.advance_to(retired + 80).unwrap();
        assert_eq!(
            core.local_memory.l0a().read_known(8192, 1024).unwrap(),
            expected.repeat(2)
        );
        assert!(core.mte_pipeline().unwrap().is_idle());
        assert!(!core.mte2.is_busy());
    }

    #[test]
    fn dma_functional_retirement_does_not_require_wait() {
        use crate::isa::c220::mte::{C220MovInstruction, CAPTURED_C220_MOV_OUT_TO_UB_X_WORD};
        use crate::memory::region::MemoryRegion;
        let mut core = matrix_core();
        core.advance_to(64).unwrap();
        // Bind a source after initialization; the transfer reads it at retirement.
        core.memory = MappedMemory::bind(
            SparseMemory::new(vec![MemoryRegion::unknown(32)], 64, 64),
            &[0x1000],
        )
        .unwrap();
        core.memory.write_known_at(0x1000, &[7; 32]).unwrap();
        let word = CAPTURED_C220_MOV_OUT_TO_UB_X_WORD;
        let decoded = C220MovInstruction::decode(word).unwrap();
        let machine = core.state.scalar_mut().machine_mut();
        machine
            .set_xreg(decoded.destination_register, 0x100)
            .unwrap();
        machine.set_xreg(decoded.source_register, 0x1000).unwrap();
        machine
            .set_xreg(decoded.descriptor_register, (1 << 4) | (1 << 16))
            .unwrap();
        assert!(matches!(
            core.step_word_at(65, word).unwrap(),
            C220CoreStep::Executed {
                instruction: C220CoreInstruction::Mte2(_),
                ..
            }
        ));
        assert!(core.state.ub().read_known(0x100, 32).is_err());
        core.advance_to(97).unwrap();
        assert_eq!(core.state.ub().read_known(0x100, 32).unwrap(), vec![7; 32]);
        assert_eq!(core.mte2.last_outcomes().len(), 1);
        assert!(!core.mte2.is_busy());
        assert!(matches!(
            core.step_word_at(98, 0x40e0_1800).unwrap(),
            C220CoreStep::Executed { .. }
        ));

        // A late DMA overwrite must not become visible to earlier vector reads
        // when the caller advances across both operations in one jump.
        core.memory = MappedMemory::bind(
            SparseMemory::new(vec![MemoryRegion::unknown(256)], 512, 512),
            &[0x1000],
        )
        .unwrap();
        core.memory
            .write_known_at(0x1000, &2.0_f32.to_le_bytes().repeat(64))
            .unwrap();
        core.state
            .ub_mut()
            .write_states(
                0x100,
                &1.0_f32
                    .to_le_bytes()
                    .repeat(64)
                    .into_iter()
                    .map(crate::memory::sparse::MemoryByteState::Known)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let machine = core.state.scalar_mut().machine_mut();
        machine.set_xreg(2, 0x100).unwrap();
        machine.set_xreg(3, 0x800).unwrap();
        machine
            .set_xreg(4, (1 << 56) | 1 | (1 << 16) | (1 << 32) | (8 << 40))
            .unwrap();
        machine.set_spr_value(3, 0).unwrap();
        machine.set_spr_value(100, u64::MAX).unwrap();
        machine.set_spr_value(101, 0).unwrap();
        assert!(matches!(
            core.step_word_at(100, 0x83c6_2392).unwrap(),
            C220CoreStep::Executed { .. }
        ));
        let machine = core.state.scalar_mut().machine_mut();
        machine
            .set_xreg(decoded.destination_register, 0x100)
            .unwrap();
        machine.set_xreg(decoded.source_register, 0x1000).unwrap();
        machine
            .set_xreg(decoded.descriptor_register, (1 << 4) | (8 << 16))
            .unwrap();
        assert!(matches!(
            core.step_word_at(101, word).unwrap(),
            C220CoreStep::Executed { .. }
        ));
        core.advance_to(400).unwrap();
        assert_eq!(
            core.state.scalar().machine().spr_value(87),
            Some(u64::from(64.0_f32.to_bits()))
        );
        assert_eq!(
            core.state.ub().read_known(0x100, 256).unwrap(),
            2.0_f32.to_le_bytes().repeat(64)
        );
        assert!(!core.last_vector_releases().is_empty());
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
        assert_eq!(bulk.mte_pipeline(), incremental.mte_pipeline());
        assert!(bulk.mte_pipeline().unwrap().is_idle());
        assert!(bulk.pending_mte1_commands().next().is_none());
        assert_eq!(mte1_outcomes.len(), 3);
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
            C220CoreStep::Executed {
                instruction: C220CoreInstruction::Mte1Queued(_),
                ..
            }
        ));
        assert_eq!(bulk.state.scalar().pc(), pc + 4);
        bulk.advance_to(69).unwrap();
        assert!(matches!(
            bulk.mte1_frontend_outcomes(),
            [C220CoreStep::Stalled(
                crate::sim::c220::schedule::C220Stall {
                    resume_tick: 70,
                    ..
                }
            )]
        ));
        let set = crate::isa::c220::hflag::C220HardwareFlagInstruction::decode(flag)
            .unwrap()
            .resolve(pc, bulk.state.scalar().machine().xregs())
            .unwrap();
        bulk.hardware_flags.schedule_set(set, 69).unwrap();
        bulk.advance_to(70).unwrap();
        assert_eq!(bulk.queued_mte1_commands().count(), 0);
        for _ in 0..crate::sim::c220::sync::C220_HARDWARE_FLAG_ALMOST_FULL {
            bulk.hardware_flags.schedule_set(set, 70).unwrap();
        }
        assert!(matches!(
            bulk.step_word_at(71, flag).unwrap(),
            C220CoreStep::Executed { .. }
        ));
        let bias_set = (flag & !(7 << 15)) | (5 << 15);
        assert!(matches!(
            bulk.step_word_at(72, bias_set).unwrap(),
            C220CoreStep::Executed { .. }
        ));
        assert!(matches!(
            bulk.step_word_at(73, bias_set | (1 << 5)).unwrap(),
            C220CoreStep::Executed { .. }
        ));
        bulk.advance_to(74).unwrap();
        assert_eq!(bulk.queued_mte1_commands().count(), 3);
        assert!(matches!(
            bulk.mte1_frontend_outcomes(),
            [C220CoreStep::Stalled(
                crate::sim::c220::schedule::C220Stall {
                    resume_tick: 75,
                    ..
                }
            )]
        ));
        let resolved_wait = crate::isa::c220::hflag::C220HardwareFlagInstruction::decode(wait)
            .unwrap()
            .resolve(pc, bulk.state.scalar().machine().xregs())
            .unwrap();
        bulk.hardware_flags.consume_wait(resolved_wait).unwrap();
        bulk.advance_to(76).unwrap();
        assert!(bulk.mte1_frontend_outcomes().iter().any(|step| matches!(
            step,
            C220CoreStep::Executed {
                instruction: C220CoreInstruction::HardwareFlag {
                    token_ready_tick: Some(77),
                    ..
                },
                ..
            }
        )));
        bulk.advance_to(78).unwrap();
        assert_eq!(bulk.queued_mte1_commands().count(), 0);
        assert_eq!(bulk.state.scalar().pc(), pc + 16);

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
            core.mte2
                .issue_l1_fill(core.mte_pipeline.as_mut().unwrap(), 100, 0, fill)
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
        assert_eq!(
            bulk.mte_pipeline().unwrap().memory(),
            incremental.mte_pipeline().unwrap().memory()
        );
        assert_eq!(
            bulk.mte_pipeline().unwrap().interface(),
            incremental.mte_pipeline().unwrap().interface()
        );
        let cross = (2 << 29) | (15 << 21) | (4 << 18) | (3 << 10) | (6 << 2);
        bulk.state
            .scalar_mut()
            .machine_mut()
            .set_xreg(6, 0xa20)
            .unwrap();
        let mut flags = bulk.hardware_flags.clone();
        let C220CoreStep::Executed {
            instruction:
                C220CoreInstruction::Mte1Queued(super::super::C220Mte1QueuedCommand {
                    instruction_id,
                    ..
                }),
            ..
        } = bulk.step_word_at(161, cross).unwrap()
        else {
            panic!("MTE1 cross-core admission")
        };

        assert!(bulk.last_mte1_outcomes().is_empty());
        bulk.state
            .scalar_mut()
            .machine_mut()
            .set_xreg(6, 0)
            .unwrap();
        bulk.advance_to(165).unwrap();
        let outcome = bulk.last_mte1_outcomes().first().unwrap();
        assert_eq!(outcome.instruction_id, instruction_id);
        assert_eq!(outcome.retire_tick, 165);
        let crate::sim::c220::mte::mte1::C220Mte1TransferResult::CrossCore(payload) =
            outcome.result
        else {
            panic!("MTE1 cross-core retirement")
        };
        assert_eq!(payload.value, 0xa20);
        assert_eq!((payload.mode, payload.flag_id), (2, 10));
        let reception = outcome.cross_core_reception().unwrap();
        assert_eq!(reception.tick, 165);
        assert_eq!(reception.payload, payload);
        flags.advance_to(165).unwrap();
        assert_eq!(bulk.hardware_flags, flags);
        assert!(bulk.pending_mte1_commands().next().is_none());
        let mte2_cross = (cross & !(15 << 10)) | (4 << 10);
        bulk.state
            .scalar_mut()
            .machine_mut()
            .set_xreg(6, 0xc10)
            .unwrap();
        let C220CoreStep::Executed {
            instruction: C220CoreInstruction::Mte2(issue),
            ..
        } = bulk.step_word_at(167, mte2_cross).unwrap()
        else {
            panic!("MTE2 cross-core admission")
        };
        assert!(matches!(
            issue.timing,
            crate::sim::c220::mte::mte2::C220Mte2IssueTiming::CrossCore { dispatch_tick: 167 }
        ));
        assert!(bulk.mte2.last_outcomes().is_empty());
        assert!(bulk.mte2.is_busy());
        bulk.state
            .scalar_mut()
            .machine_mut()
            .set_xreg(6, 0)
            .unwrap();
        bulk.advance_to(168).unwrap();
        let reception = bulk.mte2.last_outcomes()[0].cross_core_reception().unwrap();
        assert_eq!(reception.instruction_id, issue.instruction_id);
        assert_eq!(reception.tick, 168);
        assert_eq!(reception.payload.value, 0xc10);
        assert_eq!((reception.payload.mode, reception.payload.flag_id), (1, 12));
        assert!(!bulk.mte2.is_busy());
        flags.advance_to(168).unwrap();
        assert_eq!(bulk.hardware_flags, flags);
        bulk.connect_mte3_dma().unwrap();
        bulk.state
            .scalar_mut()
            .machine_mut()
            .set_xreg(6, 0xd30)
            .unwrap();
        let mte3_cross = (cross & !(15 << 10)) | (5 << 10);
        let C220CoreStep::Executed {
            instruction: C220CoreInstruction::Mte3CrossCore(record),
            ..
        } = bulk.step_word_at(169, mte3_cross).unwrap()
        else {
            panic!("MTE3 cross-core admission")
        };
        assert_eq!(record.issue_tick, 169);
        assert_eq!(record.dispatch_tick, None);
        bulk.state
            .scalar_mut()
            .machine_mut()
            .set_xreg(6, 0)
            .unwrap();
        bulk.advance_to(172).unwrap();
        assert!(bulk.last_mte3_cross_core_outcomes().is_empty());
        assert!(bulk.take_mte3_dma_request().is_none());
        assert_eq!(
            bulk.mte_pipeline()
                .unwrap()
                .mte3_frontend()
                .selected_generator(),
            Some(crate::sim::c220::mte::mte3::frontend::C220Mte3Generator::Load3d)
        );
        bulk.advance_to(173).unwrap();
        let reception = bulk.last_mte3_cross_core_outcomes()[0];
        assert_eq!(reception.instruction_id, record.instruction_id);
        assert_eq!(reception.tick, 173);
        assert_eq!(reception.payload.value, 0xd30);
        assert_eq!((reception.payload.mode, reception.payload.flag_id), (3, 13));
        assert!(bulk.mte_pipeline().unwrap().mte3_frontend().is_idle());
        assert!(bulk.mte3.native_commands.is_empty());
    }
}
