use super::{C220Core, C220CoreError};
use crate::isa::c220::cube::C220CubeInstruction;
use crate::sim::c220::cube::frontend::C220CubeCommand;
use crate::sim::c220::cube::{C220CubeExecutionControl, C220CubeTimingControl};

impl C220Core {
    pub(super) fn step_cube_spr_at(
        &mut self,
        tick: u64,
        pc: u64,
        word: u32,
        instruction: crate::isa::c220::cube::spr::C220CubeSprWrite,
    ) -> Result<super::C220CoreStep, C220CoreError> {
        let step = crate::sim::c220::cube::spr::capture_write(
            self.state.scalar().machine(),
            pc,
            word,
            instruction,
        );
        self.enqueue_cube_at(tick, pc, word, C220CubeCommand::WriteSpr(step))
    }

    pub(super) fn issue_cube_at(
        &mut self,
        tick: u64,
        pc: u64,
        word: u32,
        decoded: C220CubeInstruction,
    ) -> Result<super::C220CoreStep, C220CoreError> {
        let registers = decoded.capture(self.state.scalar().machine().xregs());
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
        self.enqueue_cube_at(
            tick,
            pc,
            word,
            C220CubeCommand::Mmad {
                instruction: decoded,
                registers,
                execution_control,
                timing_control,
            },
        )
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
    fn ordinary_fixp_events_connect_cube_and_mte1_at_retirement() {
        use crate::sim::c220::memory::C220LocalBuffer;
        use crate::sim::c220::mte::fixp::{C220FixpEngineConfig, C220FixpStage::*};

        let mut core = matrix_core();
        core.advance_to(300).unwrap();
        core.configure_fixp_l1(
            C220FixpEngineConfig {
                instruction_fifo_depth: 1,
                read_bandwidth: 256,
                read_bank_count: 32,
                read_data_latency: 4,
                l0c_capacity: core.local_memory.l0c().buffer().capacity(),
            },
            super::super::C220FixpFrontendConfig {
                issue_queue_depth: NonZeroU32::new(8).unwrap(),
                outstanding_limit: NonZeroU32::new(4).unwrap(),
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
            (12, 0x1234_5678),
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
        let flag = |op: u32, source: u32, dest: u32| {
            (2 << 29)
                | (op << 21)
                | (1 << 17)
                | (source << 10)
                | ((dest >> 3) << 14)
                | ((dest & 7) << 7)
                | (12 << 2)
        };
        let fix = (6 << 29) | (3 << 24) | (1 << 17) | (2 << 12) | (3 << 7) | (4 << 2);
        let first_fix = core.next_instruction_id + 1;
        for (tick, word) in [
            (301, flag(6, 2, 10)),
            (302, fix),
            (303, flag(5, 10, 3)),
            (304, flag(5, 10, 2)),
            (305, 0x40e0_2800),
            (306, fix),
            (307, flag(6, 10, 3)),
            (308, 0x4140_0000),
        ] {
            assert!(matches!(
                core.step_word_at(tick, word).unwrap(),
                C220CoreStep::Executed { .. }
            ));
        }
        assert_eq!(core.fixp_command_queue_len(), 0);
        assert_eq!(core.outstanding_fixp_commands(), 0);
        assert!(core.pending_fixp_barriers().next().unwrap().requires_idle);
        assert_eq!(core.queued_mte1_instructions().count(), 1);
        core.step_word_at(309, flag(5, 2, 10)).unwrap();
        core.step_word_at(310, flag(6, 10, 2)).unwrap();
        core.state
            .scalar_mut()
            .machine_mut()
            .set_xreg(12, 0)
            .unwrap();
        core.advance_to(500).unwrap();
        let consumed = core.pipeline_events().last_consumptions();
        assert_eq!(consumed.len(), 3);
        assert!(
            consumed
                .iter()
                .all(|event| event.step.flag_id == 0x1234_5678)
        );
        let outputs: Vec<_> = consumed
            .iter()
            .filter(|event| event.step.instruction.source_pipe_code == 10)
            .collect();
        assert_eq!(outputs.len(), 2);
        let published_tick = outputs[0].event.published_tick;
        assert!(
            outputs
                .iter()
                .all(|event| event.event.published_tick == published_tick
                    && event.event.received_tick < published_tick
                    && event.tick >= published_tick)
        );
        assert!(
            core.fixp_frontend_outcomes()
                .iter()
                .any(|step| matches!(step,
            C220CoreStep::Executed { instruction: C220CoreInstruction::FixpBarrier {
                completed_tick: Some(tick), .. }, .. } if *tick == published_tick))
        );
        assert_eq!(
            core.fixp_engine()
                .unwrap()
                .last_retirement()
                .unwrap()
                .instruction_id,
            first_fix + 4
        );
        assert!(core.fixp_engine().unwrap().last_retirement().unwrap().tick > published_tick);
        assert_eq!(core.pending_fixp_barriers().len(), 0);
        assert_eq!(core.queued_fixp_commands(), 0);
        assert_eq!(core.queued_mte1_instructions().count(), 0);
        assert_eq!(core.pipeline_events().pending().count(), 0);
        assert_eq!(
            core.local_memory.l1().read_known(4096, 256).unwrap(),
            0x4000_u16.to_le_bytes().repeat(128)
        );
    }

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

    #[test]
    fn mte1_barriers_order_reception_without_holding_scalar_or_command_slots() {
        let mut core = matrix_core();
        core.advance_to(300).unwrap();
        let signal = (2 << 29) | (15 << 21) | (1 << 19) | (1 << 15) | (2 << 10) | (3 << 7);
        let set = (2 << 29) | (5 << 21) | (3 << 10) | (2 << 7);
        core.step_word_at(301, signal | (1 << 5)).unwrap();
        let predecessor = core.mte1_frontend.last_accepted;
        for tick in [302, 303] {
            assert!(matches!(core.step_word_at(tick, 0x40e0_0c00).unwrap(),
                C220CoreStep::Executed {
                    instruction: C220CoreInstruction::Mte1Barrier { barrier, completed_tick: None }, ..
                } if barrier.predecessor == predecessor));
        }
        core.step_word_at(304, set).unwrap();
        core.step_word_at(305, 0x4140_0000).unwrap();
        assert_eq!(core.pending_mte1_barriers().len(), 2);
        assert_eq!(core.queued_mte1_instructions().count(), 1);
        assert_eq!(core.outstanding_mte1_commands(), 1);
        assert!(core.pipeline_events().ready(3, 2).is_empty());
        core.step_word_at(306, signal).unwrap();
        core.advance_to(340).unwrap();
        assert_eq!(core.pending_mte1_barriers().len(), 0);
        assert_eq!(core.outstanding_mte1_commands(), 0);
        assert_eq!(core.queued_mte1_instructions().count(), 0);
        let retired = core.last_mte1_outcomes().last().unwrap().retire_tick;
        let completed: Vec<_> = core
            .mte1_frontend_outcomes()
            .iter()
            .filter_map(|outcome| match outcome {
                C220CoreStep::Executed {
                    instruction: C220CoreInstruction::Mte1Barrier { completed_tick, .. },
                    ..
                } => *completed_tick,
                _ => None,
            })
            .collect();
        assert_eq!(completed, [retired, retired]);
        assert!(core.pipeline_events().ready(3, 2)[0].published_tick >= retired);
        assert!(matches!(
            core.step_word_at(341, 0x40e0_0c00).unwrap(),
            C220CoreStep::Executed {
                instruction: C220CoreInstruction::Mte1Barrier {
                    completed_tick: Some(341),
                    ..
                },
                ..
            }
        ));
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
    fn mte1_issue_credits_delay_set_events_without_stopping_scalar_progress() {
        use crate::sim::c220::schedule::C220StallCause;
        let mut core = matrix_core();
        core.advance_to(300).unwrap();
        core.mte1_frontend.config.issue_queue_depth = NonZeroU32::new(2).unwrap();
        core.mte1_frontend.config.outstanding_limit = NonZeroU32::new(1).unwrap();
        let signal = (2 << 29) | (15 << 21) | (1 << 19) | (1 << 15) | (2 << 10) | (3 << 7);
        let set = (2 << 29) | (5 << 21) | (1 << 17) | (3 << 10) | (2 << 7) | (12 << 2);
        core.state
            .scalar_mut()
            .machine_mut()
            .set_xreg(12, 0x1234_5678)
            .unwrap();
        for (tick, word) in [(301, signal | (1 << 5)), (302, set), (303, set)] {
            assert!(matches!(
                core.step_word_at(tick, word).unwrap(),
                C220CoreStep::Executed { .. }
            ));
        }
        assert_eq!(core.outstanding_mte1_commands(), 1);
        assert_eq!(core.queued_mte1_instructions().count(), 2);
        let pc = core.state.scalar().pc();
        let id = core.next_instruction_id;
        assert!(
            matches!(core.step_word_at(304, set).unwrap(), C220CoreStep::Stalled(stall)
            if stall.cause == C220StallCause::Mte1IssueQueueFull)
        );
        assert_eq!(
            (core.state.scalar().pc(), core.next_instruction_id),
            (pc, id)
        );
        assert!(
            core.mte1_frontend_outcomes()
                .iter()
                .any(|step| matches!(step,
            C220CoreStep::Stalled(stall) if stall.cause == C220StallCause::Mte1OutstandingLimit))
        );
        assert!(core.pipeline_events().ready(3, 2).is_empty());
        assert_eq!(core.pipeline_events().pending().count(), 0);
        assert!(matches!(
            core.step_word_at(305, 0x4140_0000).unwrap(),
            C220CoreStep::Executed { .. }
        ));
        core.state
            .scalar_mut()
            .machine_mut()
            .set_xreg(12, 0)
            .unwrap();
        core.step_word_at(306, signal).unwrap();
        core.advance_to(320).unwrap();
        assert_eq!(core.queued_mte1_instructions().count(), 0);
        assert_eq!(core.queued_mte1_commands().count(), 0);
        assert_eq!(core.outstanding_mte1_commands(), 0);
        assert_eq!(
            core.pipeline_events()
                .ready(3, 2)
                .iter()
                .map(|event| event.step.flag_id)
                .collect::<Vec<_>>(),
            [0x1234_5678, 0x1234_5678]
        );
        let retired_tick = core.last_mte1_outcomes().last().unwrap().retire_tick;
        let published: Vec<_> = core
            .mte1_frontend_outcomes()
            .iter()
            .filter_map(|step| match step {
                C220CoreStep::Executed {
                    tick,
                    instruction: C220CoreInstruction::Mte1Flag(_),
                } => Some(*tick),
                _ => None,
            })
            .collect();
        assert_eq!(published, [retired_tick, retired_tick + 1]);
    }

    #[test]
    fn cube_events_release_mte1_waits_without_consuming_reverse_route_tokens() {
        let mut core = matrix_core();
        core.advance_to(300).unwrap();
        core.local_memory
            .l0a_mut()
            .write_known(0, &0x3c00_u16.to_le_bytes().repeat(256))
            .unwrap();
        let machine = core.state.scalar_mut().machine_mut();
        for (register, value) in [(0, 4096), (6, 0), (7, 512), (12, 0x1234_5678)] {
            machine.set_xreg(register, value).unwrap();
        }
        let set = (2 << 29) | (5 << 21) | (1 << 17) | (2 << 10) | (3 << 7) | (12 << 2);
        let wait = (set & !(15 << 21)) | (6 << 21);
        let reverse_set = (set & !((7 << 10) | (7 << 7))) | (3 << 10) | (2 << 7);
        let load = (3 << 29) | (6 << 17) | (7 << 12) | (8 << 7) | 8;
        let cube = (7 << 29) | (3 << 22) | (1 << 12) | (2 << 7) | (3 << 2);
        for (tick, word) in [(301, reverse_set), (302, wait), (303, wait), (304, load)] {
            assert!(matches!(
                core.step_word_at(tick, word).unwrap(),
                C220CoreStep::Executed { .. }
            ));
        }
        assert_eq!(core.pipeline_events().ready(3, 2).len(), 1);
        assert!(core.pipeline_events().last_consumptions().is_empty());
        assert_eq!(core.queued_mte1_instructions().count(), 3);
        assert_eq!(core.outstanding_mte1_commands(), 0);
        core.step_word_at(305, cube).unwrap();
        for (tick, word) in [(306, set), (307, set), (308, 0x4140_0000)] {
            assert!(matches!(
                core.step_word_at(tick, word).unwrap(),
                C220CoreStep::Executed { .. }
            ));
        }
        core.state
            .scalar_mut()
            .machine_mut()
            .set_xreg(12, 0)
            .unwrap();
        core.advance_to(500).unwrap();
        let consumed = core.pipeline_events().last_consumptions();
        assert_eq!(consumed.len(), 2);
        let retire_tick = core
            .cube
            .pipeline
            .last_retirements()
            .last()
            .unwrap()
            .retire_tick;
        for token in consumed {
            assert_eq!(token.step.flag_id, 0x1234_5678);
            assert_eq!(token.event.step.flag_id, 0x1234_5678);
            assert_eq!(token.event.published_tick, retire_tick);
            assert!(token.tick >= retire_tick);
        }
        assert_eq!(consumed[1].tick, consumed[0].tick + 1);
        assert_ne!(
            consumed[0].event.instruction_id,
            consumed[1].event.instruction_id
        );
        let load_reception = core
            .mte1_frontend_outcomes()
            .iter()
            .find_map(|step| match step {
                C220CoreStep::Executed {
                    tick,
                    instruction: C220CoreInstruction::Mte1Scheduled(_),
                } => Some(*tick),
                _ => None,
            })
            .unwrap();
        assert_eq!(load_reception, consumed[1].tick + 1);
        assert!(core.pipeline_events().ready(2, 3).is_empty());
        assert_eq!(core.pipeline_events().ready(3, 2).len(), 1);
        assert_eq!(core.pipeline_events().pending().count(), 0);
        assert_eq!(
            core.local_memory.l0a().read_known(0, 512).unwrap(),
            0x4200_u16.to_le_bytes().repeat(256)
        );
        assert_eq!(
            core.local_memory
                .l0c()
                .buffer()
                .read_known(4096, 4)
                .unwrap(),
            16.0_f32.to_le_bytes()
        );
        assert!(core.pending_compute_drain().is_none());
    }

    #[test]
    fn ordinary_mte1_waits_release_queued_cube_work_after_transfer_retirement() {
        let mut core = matrix_core();
        core.advance_to(300).unwrap();
        let event_id = 0x1234_5678;
        let machine = core.state.scalar_mut().machine_mut();
        for (register, value) in [(0, 4096), (6, 0), (7, 0), (12, event_id)] {
            machine.set_xreg(register, value).unwrap();
        }
        let load = (3 << 29) | (6 << 17) | (7 << 12) | (8 << 7) | 8;
        let set = (2 << 29) | (5 << 21) | (1 << 17) | (3 << 10) | (2 << 7) | (12 << 2);
        let wait = (set & !(15 << 21)) | (6 << 21);
        let cube = (7 << 29) | (3 << 22) | (1 << 12) | (2 << 7) | (3 << 2);
        for (tick, word) in [
            (301, load),
            (302, set),
            (303, set),
            (304, wait),
            (305, wait),
            (306, 0x4140_0000),
            (307, cube),
        ] {
            assert!(matches!(
                core.step_word_at(tick, word).unwrap(),
                C220CoreStep::Executed { .. }
            ));
        }
        core.state
            .scalar_mut()
            .machine_mut()
            .set_xreg(12, 0)
            .unwrap();
        assert_eq!(core.queued_cube_commands().count(), 3);
        assert_eq!(core.outstanding_cube_commands(), 0);
        assert!(core.active_cube_control().is_none());
        core.advance_to(500).unwrap();
        let transfer_retirement = core.last_mte1_outcomes().last().unwrap().retire_tick;
        let waits: Vec<_> = core
            .cube_frontend_outcomes()
            .iter()
            .filter_map(|event| match event {
                C220CoreStep::Executed {
                    tick,
                    instruction: C220CoreInstruction::CubeFlag(step),
                } => Some((*tick, *step)),
                _ => None,
            })
            .collect();
        assert_eq!(waits.len(), 2);
        assert_eq!(waits[0].0, transfer_retirement);
        assert_eq!(waits[1].0, transfer_retirement + 1);
        assert!(
            waits
                .iter()
                .all(|(_, step)| step.flag_id == event_id as u32)
        );
        assert_eq!(
            core.local_memory
                .l0c()
                .buffer()
                .read_known(4096, 4)
                .unwrap(),
            32.0_f32.to_le_bytes()
        );
        assert!(core.pending_compute_drain().is_none());

        core.state
            .scalar_mut()
            .machine_mut()
            .set_xreg(12, event_id)
            .unwrap();
        core.step_word_at(501, wait).unwrap();
        core.step_word_at(502, cube).unwrap();
        core.advance_to(505).unwrap();
        assert_eq!(core.queued_cube_commands().count(), 2);
        assert_eq!(core.outstanding_cube_commands(), 0);
        core.step_word_at(506, set).unwrap();
        core.advance_to(600).unwrap();
        assert!(core.pending_compute_drain().is_none());
    }

    #[test]
    fn cube_barriers_hold_later_reception_without_blocking_scalar_admission() {
        let mut core = matrix_core();
        core.advance_to(300).unwrap();
        let word = (7 << 29) | (3 << 22) | (1 << 12) | (2 << 7) | (3 << 2);
        let first_id = core.next_instruction_id();
        core.step_word_at(301, word).unwrap();
        for tick in [302, 303] {
            assert!(matches!(core.step_word_at(tick, 0x40e0_0800).unwrap(),
                C220CoreStep::Executed { instruction: C220CoreInstruction::CubeBarrier {
                    barrier, completed_tick: None }, .. } if barrier.predecessor == Some(first_id)));
            assert_eq!(core.outstanding_cube_commands(), 1);
            assert_eq!(core.queued_cube_commands().count(), 0);
        }
        assert_eq!(core.pending_cube_barriers().len(), 2);
        assert!(matches!(
            core.step_word_at(304, 0x4140_0000).unwrap(),
            C220CoreStep::Executed { .. }
        ));
        let second_id = core.next_instruction_id();
        core.step_word_at(305, word).unwrap();
        let retirement = core.cube.pipeline.pending_drain_tick().unwrap();
        core.advance_to(retirement - 1).unwrap();
        assert_eq!(core.queued_cube_commands().count(), 1);
        assert_eq!(core.pending_cube_barriers().len(), 2);
        assert!(!core.cube.pipeline.has_pending_instruction(second_id));
        core.advance_to(retirement).unwrap();
        assert_eq!(core.pending_cube_barriers().len(), 0);
        assert_eq!(core.queued_cube_commands().count(), 0);
        assert!(core.cube.pipeline.has_pending_instruction(second_id));
        assert_eq!(
            core.cube_frontend_outcomes()
                .iter()
                .filter(|event| matches!(event,
            C220CoreStep::Executed { instruction: C220CoreInstruction::CubeBarrier {
                completed_tick: Some(tick), .. }, .. } if *tick == retirement))
                .count(),
            2
        );
        assert!(
            core.cube_frontend_outcomes()
                .iter()
                .any(|event| matches!(event,
            C220CoreStep::Executed { instruction: C220CoreInstruction::Cube(issue), .. }
                if issue.instruction_id == second_id && issue.ticket.accept_tick == retirement))
        );
        core.advance_to(400).unwrap();
        assert!(matches!(
            core.step_word_at(401, 0x40e0_0800).unwrap(),
            C220CoreStep::Executed {
                instruction: C220CoreInstruction::CubeBarrier {
                    completed_tick: Some(401),
                    ..
                },
                ..
            }
        ));
        assert!(core.pending_compute_drain().is_none());
        let set = (2 << 29) | (5 << 21) | (3 << 10) | (2 << 7) | 1;
        let wait = (set & !(15 << 21)) | (6 << 21);
        core.step_word_at(402, set).unwrap();
        core.step_word_at(403, word).unwrap();
        core.step_word_at(404, wait).unwrap();
        core.step_word_at(405, 0x40e0_0800).unwrap();
        assert!(core.pending_cube_barriers().next().unwrap().requires_idle);
        core.step_word_at(406, word).unwrap();
        let retirement = core.cube.pipeline.pending_drain_tick().unwrap();
        core.advance_to(retirement - 1).unwrap();
        assert_eq!(core.pending_cube_barriers().len(), 1);
        assert_eq!(core.queued_cube_commands().count(), 1);
        assert!(matches!(
            core.queued_cube_commands().next().unwrap().command,
            C220CubeCommand::Mmad { .. }
        ));
        core.advance_to(retirement).unwrap();
        assert_eq!(core.pending_cube_barriers().len(), 0);
        assert_eq!(core.queued_cube_commands().count(), 0);
    }

    #[test]
    fn cube_queue_preserves_operands_and_releases_retirement_credits() {
        use crate::sim::c220::cube::frontend::C220CubeFrontendConfig;
        use crate::sim::c220::schedule::C220StallCause;
        for limit in [1, 15] {
            let mut core = matrix_core();
            core.advance_to(300).unwrap();
            core.cube_frontend =
                super::super::cube_frontend::CubeFrontend::new(C220CubeFrontendConfig {
                    outstanding_limit: NonZeroU32::new(limit).unwrap(),
                    ..Default::default()
                });
            core.local_memory
                .l0a_mut()
                .write_known(0, &0x3c00_u16.to_le_bytes().repeat(256))
                .unwrap();
            let flag = (2 << 29) | (15 << 21) | (1 << 19) | (1 << 15) | (3 << 10) | (2 << 7) | 7;
            core.step_word_at(301, flag | (1 << 5)).unwrap();
            let word = (7 << 29) | (3 << 22) | (1 << 12) | (2 << 7) | (3 << 2);
            for index in 0..16 {
                let machine = core.state.scalar_mut().machine_mut();
                machine.set_xreg(0, 4096 + index * 1024).unwrap();
                machine
                    .set_xreg(3, u64::from(index < 2) | (16 << 12) | (1 << 24) | (1 << 63))
                    .unwrap();
                assert!(matches!(
                    core.step_word_at(302 + index, word).unwrap(),
                    C220CoreStep::Executed {
                        instruction: C220CoreInstruction::CubeQueued(_),
                        ..
                    }
                ));
            }
            assert!(core.active_cube_control().is_some());
            assert_eq!(core.outstanding_cube_commands(), 1);
            assert_eq!(core.queued_cube_commands().count(), 16);
            let pc = core.state.scalar().pc();
            let id = core.next_instruction_id();
            assert!(
                matches!(core.step_word_at(318, word).unwrap(), C220CoreStep::Stalled(stall)
                if stall.cause == C220StallCause::CubeIssueQueueFull)
            );
            assert_eq!(core.state.scalar().pc(), pc);
            assert_eq!(core.next_instruction_id(), id);
            assert!(matches!(core.step_word_at(319, 0x40e0_0800).unwrap(),
                C220CoreStep::Stalled(stall) if stall.cause == C220StallCause::CubeIssueQueueFull));
            assert_eq!(core.pending_cube_barriers().len(), 0);
            assert_eq!(core.next_instruction_id(), id);
            for register in 0..4 {
                core.state
                    .scalar_mut()
                    .machine_mut()
                    .set_xreg(register, u64::MAX)
                    .unwrap();
            }
            let signal = crate::isa::c220::hflag::C220HardwareFlagInstruction::decode(flag)
                .unwrap()
                .resolve(pc, core.state.scalar().machine().xregs())
                .unwrap();
            core.hardware_flags.schedule_set(signal, 320).unwrap();
            core.advance_to(500).unwrap();
            let issued: Vec<_> = core
                .cube_frontend_outcomes()
                .iter()
                .filter_map(|event| match event {
                    C220CoreStep::Executed {
                        instruction: C220CoreInstruction::Cube(issue),
                        ..
                    } => Some(*issue),
                    _ => None,
                })
                .collect();
            assert_eq!(issued.len(), 16);
            let first = issued[0];
            let second = issued[1];
            assert_eq!(first.registers.xd, 4096);
            assert_eq!(second.registers.xd, 5120);
            assert_eq!(
                second.ticket.accept_tick,
                if limit == 1 {
                    first.ticket.retire_tick
                } else {
                    first.ticket.last_uop_tick.unwrap()
                }
            );
            for address in [4096, 5120] {
                assert_eq!(
                    core.local_memory
                        .l0c()
                        .buffer()
                        .read_known(address, 4)
                        .unwrap(),
                    16.0_f32.to_le_bytes()
                );
            }
            assert_eq!(core.queued_cube_commands().count(), 0);
            assert_eq!(core.outstanding_cube_commands(), 0);
            assert!(core.active_cube_control().is_none());
            assert!(core.pending_compute_drain().is_none());
        }
    }

    #[test]
    fn mte1_deferred_waits_gate_sends_and_disabled_decode() {
        use crate::isa::c220::hflag::C220MatrixMemory;
        for disabled in [false, true] {
            let mut core = matrix_core();
            core.advance_to(300).unwrap();
            let signal = (2 << 29) | (15 << 21) | (1 << 19) | (1 << 15) | (2 << 10) | (3 << 7);
            let wait = (signal & !(1 << 19)) | (1 << 5);
            core.state
                .scalar_mut()
                .machine_mut()
                .set_xreg(6, 2048)
                .unwrap();
            core.state
                .scalar_mut()
                .machine_mut()
                .set_xreg(7, 512)
                .unwrap();
            if disabled {
                core.state
                    .scalar_mut()
                    .machine_mut()
                    .set_xreg(8, 0)
                    .unwrap();
            }
            core.local_memory
                .l0a_mut()
                .write_known(2048, &[0; 512])
                .unwrap();
            let load = (3 << 29) | (6 << 17) | (7 << 12) | (8 << 7) | 8;
            for (tick, word) in [(301, signal | 1), (302, wait), (303, wait | 1), (304, load)] {
                assert!(matches!(
                    core.step_word_at(tick, word).unwrap(),
                    C220CoreStep::Executed { .. }
                ));
            }
            core.advance_to(340).unwrap();
            assert_eq!(core.hardware_flags.count(3, C220MatrixMemory::L0a, 1), 0);
            assert_eq!(
                core.local_memory.l0a().read_known(2048, 512).unwrap(),
                [0; 512]
            );
            assert_eq!(core.queued_mte1_commands().count(), usize::from(disabled));
            assert!(core.pending_compute_drain().is_some());
            assert!(matches!(
                core.step_word_at(341, signal).unwrap(),
                C220CoreStep::Executed { .. }
            ));
            core.advance_to(500).unwrap();
            assert_eq!(core.hardware_flags.count(3, C220MatrixMemory::L0a, 0), 0);
            assert!(core.pending_compute_drain().is_none());
            let expected = if disabled {
                vec![0; 512]
            } else {
                0x4200_u16.to_le_bytes().repeat(256)
            };
            assert_eq!(
                core.local_memory.l0a().read_known(2048, 512).unwrap(),
                expected
            );
        }
    }

    #[test]
    fn cube_deferred_signals_follow_uop_checkpoints() {
        use crate::isa::c220::hflag::C220MatrixMemory;
        let mut core = matrix_core();
        core.advance_to(300).unwrap();
        let flags = [
            (1, C220MatrixMemory::L0a, 3, 5),
            (2, C220MatrixMemory::L0b, 3, 5),
            (5, C220MatrixMemory::BiasTable, 3, 4),
            (3, C220MatrixMemory::L0c, 10, 16),
        ];
        for (index, &(code, _, destination, _)) in flags.iter().enumerate() {
            let word = (2 << 29)
                | (15 << 21)
                | (code << 15)
                | ((destination >> 3) << 14)
                | (2 << 10)
                | ((destination & 7) << 7);
            assert!(matches!(
                core.step_word_at(301 + index as u64, word).unwrap(),
                C220CoreStep::Executed {
                    instruction: C220CoreInstruction::CubeQueued(_),
                    ..
                }
            ));
        }
        core.advance_to(305).unwrap();
        assert_eq!(core.hardware_flags.pending_cube_set_count(), 4);
        for (register, value) in [
            (0, 4096),
            (1, 0),
            (2, 0),
            (3, 32 | (32 << 12) | (16 << 24) | (1 << 62) | (1 << 63)),
        ] {
            core.state
                .scalar_mut()
                .machine_mut()
                .set_xreg(register, value)
                .unwrap();
        }
        core.local_memory
            .l0a_mut()
            .write_known(0, &0x3c00_u16.to_le_bytes().repeat(1024))
            .unwrap();
        core.local_memory
            .l0b_mut()
            .write_known(0, &0x3c00_u16.to_le_bytes().repeat(512))
            .unwrap();
        let word = (7 << 29) | (3 << 22) | (1 << 12) | (2 << 7) | (3 << 2);
        let C220CoreStep::Executed {
            instruction: C220CoreInstruction::Cube(issue),
            ..
        } = ({
            core.step_word_at(306, word).unwrap();
            core.advance_to(307).unwrap();
            core.cube_frontend_outcomes().last().cloned().unwrap()
        })
        else {
            panic!("Cube admission")
        };
        assert_eq!(issue.ticket.bias_checkpoint_uop, Some(1));
        let mut releases = Vec::new();
        for tick in 307..350 {
            core.advance_to(tick).unwrap();
            releases.extend_from_slice(core.cube.pipeline.last_uop_releases());
            for &(_, memory, destination, delay) in &flags {
                let checkpoint = if memory == C220MatrixMemory::BiasTable {
                    1
                } else {
                    3
                };
                let visible = releases
                    .get(checkpoint)
                    .is_some_and(|release| tick >= release.issue_tick + delay);
                assert_eq!(
                    core.hardware_flags.count(destination as u8, memory, 0),
                    u8::from(visible)
                );
            }
        }
        assert_eq!(releases.len(), 4);
        assert_eq!(core.hardware_flags.pending_cube_set_count(), 0);
        assert_eq!(core.hardware_flags.pending_cube_checkpoints().count(), 0);
    }

    #[test]
    fn cube_triggered_signals_reach_mte_and_fix_destinations() {
        use crate::isa::c220::hflag::C220MatrixMemory;
        use crate::sim::c220::memory::C220LocalBuffer;
        use crate::sim::c220::mte::fixp::{C220FixpEngineConfig, C220FixpStage::*};

        let mut core = matrix_core();
        core.advance_to(300).unwrap();
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
        for (index, (code, memory, destination, delay)) in [
            (1, C220MatrixMemory::L0a, 3, 2),
            (2, C220MatrixMemory::L0b, 3, 2),
            (5, C220MatrixMemory::BiasTable, 3, 1),
            (3, C220MatrixMemory::L0c, 10, 2),
        ]
        .into_iter()
        .enumerate()
        {
            let tick = 301 + index as u64 * 20;
            let word = (2 << 29)
                | (15 << 21)
                | (1 << 19)
                | (code << 15)
                | ((destination >> 3) << 14)
                | (2 << 10)
                | ((destination & 7) << 7);
            assert!(matches!(
                core.step_word_at(tick, word).unwrap(),
                C220CoreStep::Executed {
                    instruction: C220CoreInstruction::CubeQueued(_),
                    ..
                }
            ));
            core.advance_to(tick + 1).unwrap();
            assert!(matches!(core.cube_frontend_outcomes().last().unwrap(),
                C220CoreStep::Executed {
                    instruction: C220CoreInstruction::HardwareFlag {
                        token_ready_tick: Some(ready), ..
                    }, ..
                } if *ready == tick + 1 + delay));
            assert_eq!(core.hardware_flags.count(destination as u8, memory, 0), 0);
            core.advance_to(tick + 1 + delay).unwrap();
            assert_eq!(core.hardware_flags.count(destination as u8, memory, 0), 1);
            assert!(matches!(
                core.step_word_at(tick + 3, word | (1 << 5)).unwrap(),
                C220CoreStep::Executed { .. }
            ));
            core.advance_to(tick + 15).unwrap();
            assert_eq!(core.hardware_flags.count(destination as u8, memory, 0), 0);
        }
        let deferred_wait = (2 << 29) | (15 << 21) | (1 << 15) | (2 << 10) | (3 << 7) | (1 << 5);
        assert!(matches!(
            core.step_word_at(400, deferred_wait).unwrap(),
            C220CoreStep::Executed {
                instruction: C220CoreInstruction::Mte1Queued(_),
                ..
            }
        ));
        core.advance_to(410).unwrap();
        assert_eq!(core.hardware_flags.pending_mte_flags().count(), 1);
    }

    #[test]
    fn cube_control_instructions_use_their_own_pipeline() {
        let mut core = matrix_core();
        let word = (2 << 24) | (52 << 17) | (14 << 12) | (18 << 7);
        core.state
            .scalar_mut()
            .machine_mut()
            .set_xreg(14, u64::MAX)
            .unwrap();
        let pc = core.state.scalar().pc();
        let prior = core.state.scalar().machine().spr_value(52);
        let id = core.next_instruction_id;
        assert!(matches!(
            core.step_word_at(6, word).unwrap(),
            C220CoreStep::Executed {
                instruction: C220CoreInstruction::CubeQueued(_),
                ..
            }
        ));
        assert_eq!(core.state.scalar().pc(), pc + 4);
        assert_eq!(core.next_instruction_id, id + 1);
        assert_eq!(core.state.scalar().machine().spr_value(52), prior);
        core.state
            .scalar_mut()
            .machine_mut()
            .set_xreg(14, 0)
            .unwrap();
        (7..100).find(|&tick| {
            core.advance_to(tick).unwrap();
            core.cube_frontend_outcomes().iter().any(|event| matches!(event, C220CoreStep::Executed {
                instruction: C220CoreInstruction::CubeSpr { instruction_id, step }, ..
            } if *instruction_id == id && step.value == u64::MAX && step.prior_destination_value == prior))
        }).expect("Cube SPR retirement");
        assert_eq!(core.state.scalar().machine().spr_value(52), Some(u64::MAX));
        assert!(core.cube.pipeline.pending_drain_tick().is_none());
        assert_eq!(core.state.scalar().pc(), pc + 4);
        core.advance_to(200).unwrap();
        core.state
            .scalar_mut()
            .machine_mut()
            .set_xreg(14, 7)
            .unwrap();
        let fill = (3 << 29) | (1 << 22) | (10 << 17) | (11 << 7);
        core.state
            .scalar_mut()
            .machine_mut()
            .set_xreg(10, 4096)
            .unwrap();
        core.state
            .scalar_mut()
            .machine_mut()
            .set_xreg(11, 16 | (1 << 16))
            .unwrap();
        core.step_word_at(201, fill).unwrap();
        assert!(core.pending_mte1_tick().is_some());
        core.step_word_at(202, word).unwrap();
        core.advance_to(203).unwrap();
        assert!(
            matches!(core.cube_frontend_outcomes().last().unwrap(), C220CoreStep::Executed {
            instruction: C220CoreInstruction::CubeSpr { step, .. }, ..
        } if step.value == 7)
        );
        assert!(core.pending_mte1_tick().is_some());
        for (tick, source, memory) in [(205, 3, 1), (207, 10, 3)] {
            let set =
                (2 << 29) | (15 << 21) | (1 << 19) | (memory << 15) | (source << 10) | (2 << 7);
            let instruction =
                crate::isa::c220::hflag::C220HardwareFlagInstruction::decode(set).unwrap();
            let event = instruction
                .resolve(
                    core.state.scalar().pc(),
                    core.state.scalar().machine().xregs(),
                )
                .unwrap();
            core.hardware_flags.schedule_set(event, tick - 1).unwrap();
            assert!(matches!(
                core.step_word_at(tick, set | (1 << 5)).unwrap(),
                C220CoreStep::Executed {
                    instruction: C220CoreInstruction::CubeQueued(_),
                    ..
                }
            ));
            assert!(!core.mte_pipeline().unwrap().selected_generator_idle());
        }
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
            core.advance_to(303).unwrap();
            assert!(core.cube.pipeline.last_uop_releases().is_empty());
            core.state
                .scalar_mut()
                .machine_mut()
                .set_spr_value(107, if initially_enabled { 0 } else { delay })
                .unwrap();
            let expected = if initially_enabled { 305 } else { 313 };
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
            before.l0c_mut().read_banks_mut().advance_to(302).unwrap();
            let word = (7 << 29) | (3 << 22) | (1 << 12) | (2 << 7) | (3 << 2);
            let C220CoreStep::Executed {
                instruction: C220CoreInstruction::Cube(issue),
                ..
            } = ({
                core.step_word_at(301, word).unwrap();
                core.advance_to(302).unwrap();
                core.cube_frontend_outcomes().last().cloned().unwrap()
            })
            else {
                panic!("Cube admission");
            };
            assert_eq!(issue.ticket.retire_tick, 302);
            assert_eq!(issue.ticket.uop_count, 0);
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
            instruction: C220CoreInstruction::CubeQueued(issue),
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
                C220CoreInstruction::Mte1Queued(super::super::C220Mte1IssuedInstruction {
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
            if tick == 305 {
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
        core.mte1_frontend.config.issue_queue_depth = NonZeroU32::new(1).unwrap();
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
        assert_eq!(core.queued_mte1_commands().count(), 2);
        assert_eq!(core.queued_mte1_instructions().count(), 1);
        assert!(core.pending_mte1_commands().next().is_none());
        core.state
            .scalar_mut()
            .machine_mut()
            .set_xreg(12, 0x3333)
            .unwrap();
        core.step_word_at(304, spr).unwrap();
        assert_eq!(core.pending_mte1_commands().count(), 0);
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
                cause: C220StallCause::Mte1IssueQueueFull,
                ..
            })
        ));
        assert_eq!(core.state.scalar().pc(), pc);
        assert_eq!(core.state.scalar().machine().spr_value(15), Some(0x3333));
        assert!(core.mte1_frontend_outcomes().iter().any(|step| matches!(
            step,
            C220CoreStep::Stalled(C220Stall {
                cause: C220StallCause::Mte1CommandQueueFull,
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
            C220CoreStep::Executed {
                instruction: C220CoreInstruction::CubeQueued(_),
                ..
            }
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
        assert!(core.cube_frontend_outcomes().iter().any(|event| matches!(
            event,
            C220CoreStep::Executed {
                instruction: C220CoreInstruction::CubeFlag(_),
                ..
            }
        )));
        assert_eq!(core.queued_cube_commands().count(), 0);
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
        core.advance_to(279).unwrap();
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
                    C220CoreInstruction::Mte1Queued(super::super::C220Mte1IssuedInstruction {
                        operation:
                            super::super::C220Mte1Operation::Command(C220Mte1Command::WriteSpr(step)),
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
                if tick == 305 {
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
                        if tick == 305 {
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
                core.advance_to(103).unwrap();
                assert_eq!(core.hardware_flags.pending_cube_wait_count(), 1);
                let cube = (7 << 29) | (3 << 22) | (1 << 12) | (2 << 7) | (3 << 2);
                assert!(matches!(
                    core.step_word_at(104, cube).unwrap(),
                    C220CoreStep::Executed { .. }
                ));
            }
            core.advance_to(200).unwrap();
            assert!(core.pending_mte1_commands().next().is_none());
            assert_eq!(core.last_mte1_outcomes().len(), 1);
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
        use crate::sim::c220::mte::mte2::C220Mte2Completion;
        let mut core = matrix_core();
        core.advance_to(64).unwrap();
        let pattern = 0x4000_4000_u64;
        let machine = core.state.scalar_mut().machine_mut();
        machine.set_xreg(1, 2048).unwrap();
        machine.set_xreg(3, 2 | (16 << 16) | (16 << 32)).unwrap();
        machine.set_spr_value(15, pattern).unwrap();
        let word = (3 << 29) | (1 << 22) | (1 << 17) | (3 << 7) | 6;
        let C220CoreStep::Executed {
            instruction: C220CoreInstruction::Mte2Queued(issue),
            ..
        } = core.step_word_at(65, word).unwrap()
        else {
            panic!("L1 fill issue")
        };
        assert_eq!(issue.ready_tick, 66);
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
            C220CoreStep::Executed { .. }
        ));
        assert_eq!(core.state.scalar().pc(), pc + 4);
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
        for (offset, word) in [(2, wait), (3, wait & !(7 << 7))] {
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
        assert!(!core.mte2_is_busy());
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
                instruction: C220CoreInstruction::Mte2Queued(_),
                ..
            }
        ));
        assert!(core.state.ub().read_known(0x100, 32).is_err());
        core.advance_to(101).unwrap();
        assert_eq!(core.state.ub().read_known(0x100, 32).unwrap(), vec![7; 32]);
        assert_eq!(core.mte2.last_outcomes().len(), 1);
        assert!(!core.mte2_is_busy());
        assert!(matches!(
            core.step_word_at(102, 0x40e0_1800).unwrap(),
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
            core.step_word_at(104, 0x83c6_2392).unwrap(),
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
            core.step_word_at(105, word).unwrap(),
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
                instruction: C220CoreInstruction::CubeQueued(_),
                ..
            }
        ));
        assert_eq!(bulk.state.scalar().pc(), pc + 4);
        let set = crate::isa::c220::hflag::C220HardwareFlagInstruction::decode(flag)
            .unwrap()
            .resolve(pc, bulk.state.scalar().machine().xregs())
            .unwrap();
        bulk.hardware_flags.schedule_set(set, 66).unwrap();
        bulk.advance_to(67).unwrap();
        assert!(matches!(
            bulk.cube_frontend_outcomes().last().unwrap(),
            C220CoreStep::Executed {
                instruction: C220CoreInstruction::HardwareFlag { .. },
                ..
            }
        ));
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
            C220CoreStep::Executed {
                instruction: C220CoreInstruction::CubeQueued(_),
                ..
            }
        ));
        bulk.advance_to(75).unwrap();
        assert!(bulk.active_cube_control().is_some());
        assert_eq!(bulk.queued_mte1_commands().count(), 2);
        assert!(matches!(
            bulk.mte1_frontend_outcomes(),
            [C220CoreStep::Stalled(
                crate::sim::c220::schedule::C220Stall {
                    resume_tick: 76,
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
        assert!(bulk.active_cube_control().is_none());
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
                C220CoreInstruction::Mte1Queued(super::super::C220Mte1IssuedInstruction {
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
        bulk.advance_to(166).unwrap();
        let outcome = bulk.last_mte1_outcomes().first().unwrap();
        assert_eq!(outcome.instruction_id, instruction_id);
        assert_eq!(outcome.retire_tick, 166);
        let crate::sim::c220::mte::mte1::C220Mte1TransferResult::CrossCore(payload) =
            outcome.result
        else {
            panic!("MTE1 cross-core retirement")
        };
        assert_eq!(payload.value, 0xa20);
        assert_eq!((payload.mode, payload.flag_id), (2, 10));
        let reception = outcome.cross_core_reception().unwrap();
        assert_eq!(reception.tick, 166);
        assert_eq!(reception.payload, payload);
        flags.advance_to(166).unwrap();
        assert_eq!(bulk.hardware_flags, flags);
        assert!(bulk.pending_mte1_commands().next().is_none());
        let mte2_cross = (cross & !(15 << 10)) | (4 << 10);
        bulk.state
            .scalar_mut()
            .machine_mut()
            .set_xreg(6, 0xc10)
            .unwrap();
        let C220CoreStep::Executed {
            instruction: C220CoreInstruction::Mte2Queued(issue),
            ..
        } = bulk.step_word_at(167, mte2_cross).unwrap()
        else {
            panic!("MTE2 cross-core admission")
        };
        assert_eq!(issue.ready_tick, 168);
        assert!(bulk.mte2.last_outcomes().is_empty());
        assert!(bulk.mte2_is_busy());
        bulk.state
            .scalar_mut()
            .machine_mut()
            .set_xreg(6, 0)
            .unwrap();
        bulk.advance_to(172).unwrap();
        let reception = bulk.mte2.last_outcomes()[0].cross_core_reception().unwrap();
        assert_eq!(reception.instruction_id, issue.instruction_id);
        assert_eq!(reception.tick, 172);
        assert_eq!(reception.payload.value, 0xc10);
        assert_eq!((reception.payload.mode, reception.payload.flag_id), (1, 12));
        assert!(!bulk.mte2_is_busy());
        flags.advance_to(172).unwrap();
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
        } = bulk.step_word_at(173, mte3_cross).unwrap()
        else {
            panic!("MTE3 cross-core admission")
        };
        assert_eq!(record.issue_tick, 173);
        assert_eq!(record.dispatch_tick, None);
        bulk.state
            .scalar_mut()
            .machine_mut()
            .set_xreg(6, 0)
            .unwrap();
        bulk.advance_to(176).unwrap();
        assert!(bulk.last_mte3_cross_core_outcomes().is_empty());
        assert!(bulk.take_mte3_dma_request().is_none());
        assert_eq!(
            bulk.mte_pipeline()
                .unwrap()
                .mte3_frontend()
                .selected_generator(),
            Some(crate::sim::c220::mte::mte3::frontend::C220Mte3Generator::Load3d)
        );
        bulk.advance_to(177).unwrap();
        let reception = bulk.last_mte3_cross_core_outcomes()[0];
        assert_eq!(reception.instruction_id, record.instruction_id);
        assert_eq!(reception.tick, 177);
        assert_eq!(reception.payload.value, 0xd30);
        assert_eq!((reception.payload.mode, reception.payload.flag_id), (3, 13));
        assert!(bulk.mte_pipeline().unwrap().mte3_frontend().is_idle());
        assert!(bulk.mte3.native_commands.is_empty());
    }
}
