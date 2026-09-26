use super::*;
use crate::isa::c220::mte::fixp::C220FixpDescriptor;
use crate::memory::mapped::MappedMemory;
use crate::memory::{region::MemoryRegion, sparse::SparseMemory};
use crate::sim::c220::memory::C220LocalBuffer;
use crate::sim::c220::memory::l1::C220L1Geometry;
use crate::sim::c220::mte::{
    interface::biu_write::command::C220BiuWriteConfig, mte1::frontend::C220Mte1ReadBandwidths,
    pipeline::C220MtePipelineConfig, set2d::C220Set2dBandwidths,
};
use std::num::NonZeroU32;

#[test]
fn external_engine_executes_layouts_and_waits_for_ordered_retirement() {
    let cases = [
        0, 1, 8, 9, 10, 11, 12, 13, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
    ]
    .into_iter()
    .map(|mode| (mode, true, false))
    .chain([
        (0, false, false),
        (0, false, true),
        (6, false, false),
        (6, true, false),
        (17, false, false),
        (18, false, false),
        (19, false, false),
        (20, false, false),
    ])
    .map(|(mode, nz2nd, split)| (mode, nz2nd, split, mode == 6))
    .chain(
        [
            1, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
        ]
        .map(|mode| (mode, true, false, true)),
    );
    for (conversion, nz2nd, split, half_source) in cases {
        let cleared = matches!(conversion, 6 | 17..=20);
        use crate::sim::c220::device::C220CoreKind;
        let core_kind = match conversion {
            1 => C220CoreKind::Vector0,
            8 => C220CoreKind::Vector1,
            _ => C220CoreKind::Cube,
        };
        let int4 = matches!(conversion, 19..=22 | 25 | 26);
        use C220FixpRuntimeStage::*;
        let width = NonZeroU32::new(32).unwrap();
        let mut pipeline = C220MtePipeline::new(
            0,
            C220MtePipelineConfig {
                core_kind,
                l1: C220L1Geometry::new(32, 1, 1, 0).unwrap(),
                read_width: width,
                output_bandwidths: C220Mte1ReadBandwidths {
                    l0a: width,
                    l0b: width,
                    bt: width,
                    smask: width,
                },
                set2d_bandwidths: C220Set2dBandwidths {
                    l0a: width,
                    l0b: width,
                    l1: width,
                },
            },
        );
        pipeline
            .connect_fixp_biu(C220BiuWriteConfig {
                outstanding: NonZeroU32::new(1).unwrap(),
                weights: [1; 3],
                source_bandwidth: width,
            })
            .unwrap();
        let mut engine = C220FixpRuntime::new(
            C220FixpEngineConfig {
                instruction_fifo_depth: 1,
                read_bandwidth: 128,
                read_bank_count: 32,
                read_data_latency: 2,
                l0c_capacity: 131072,
            },
            8,
            16,
        )
        .unwrap();
        let mut l0c = C220L0c::new(131072, 12).unwrap();
        let input = if matches!(conversion, 8..=13 | 21 | 22) {
            2_i32.to_le_bytes()
        } else {
            1_f32.to_le_bytes()
        };
        l0c.buffer_mut()
            .write_known_linear(0, &input.repeat(1024))
            .unwrap();
        let mut slopes = C220LocalBuffer::new(2176);
        if matches!(conversion, 8 | 10 | 12 | 21 | 23 | 25) {
            let factor = if matches!(conversion, 23 | 25) {
                0x3f80_0000_u64
            } else if matches!(conversion, 8 | 10 | 21) {
                0x3f00_0000_u64
            } else {
                0
            };
            slopes
                .write_known_linear(0, &factor.to_le_bytes().repeat(32))
                .unwrap();
            slopes.write_known_linear(2048, &[0; 128]).unwrap();
        }
        let mut memory = MappedMemory::bind(
            SparseMemory::new(
                vec![MemoryRegion::new(4096, vec![if cleared { 0xa5 } else { 0 }; 4096]).unwrap()],
                4096,
                4096,
            ),
            &[4096],
        )
        .unwrap();
        let operands = C220FixpExternalCommand {
            command: C220FixpCommand {
                descriptor: C220FixpDescriptor {
                    xt: (32 << 32)
                        | (2 << 16)
                        | ((if split {
                            24
                        } else if !nz2nd {
                            32
                        } else if int4 {
                            19
                        } else {
                            17
                        }) << 4),
                    xm: (u64::from(nz2nd) << 43)
                        | (u64::from(split) << 42)
                        | (conversion << 34)
                        | 2,
                    nd: 1,
                },
                source_format: if half_source {
                    C220FixpSourceFormat::Fp16
                } else if matches!(conversion, 8..=13 | 21 | 22) {
                    C220FixpSourceFormat::Int32
                } else {
                    C220FixpSourceFormat::Fp32
                },
                source_address: 0,
                destination_address: 4096,
                control: 0,
                scalar_slope: 0,
                slope_base_block: 0,
                dequant_base_block: 0,
                scalar_dequant: if matches!(conversion, 24 | 26) {
                    0x3f80_0000
                } else if matches!(conversion, 9 | 11 | 22) {
                    0x3f00_0000
                } else {
                    0
                },
            },
            biu_mode_word: 0,
            output_mode_word: 0,
        };
        if conversion == 6 && !nz2nd {
            let slices: Vec<_> = operands.command.layout().unwrap().slices().collect();
            assert_eq!(
                slices
                    .iter()
                    .map(|slice| slice.source_address)
                    .collect::<Vec<_>>(),
                [0, 64, 32, 96]
            );
            assert!(slices.iter().all(|slice| slice.source_bytes() == 32));
            let reads: Vec<_> = C220FixpReadGenerator::new(operands.command, 7, 0, 128)
                .unwrap()
                .map(|uop| {
                    let op = uop.operation;
                    (op.request.fragments.address, op.data_bytes, op.output_bytes)
                })
                .collect();
            assert_eq!(reads, [(0, 64, 64), (64, 64, 64)]);
        }
        use crate::isa::c220::mte::{C220DmaMovDescriptor, CAPTURED_C220_MOV_UB_TO_OUT_WORD};
        pipeline
            .issue_mte3_dma(
                1,
                crate::sim::c220::mte::mte3::C220Mte3TransferPlan {
                    descriptor: C220DmaMovDescriptor::decode(CAPTURED_C220_MOV_UB_TO_OUT_WORD, 0)
                        .unwrap(),
                    source_address: 0,
                    destination_address: 4096,
                    bytes: 0,
                    dma_mode_word: 0,
                    biu_mode_word: 0,
                },
            )
            .unwrap();
        assert!(!pipeline.external_fixp_admission_blocked(operands));
        for tick in 0..=3 {
            pipeline.advance(tick).unwrap();
            assert_eq!(
                pipeline.external_fixp_admission_blocked(operands),
                tick == 3
            );
        }
        assert!(pipeline.mte3_retirement_candidate().is_none());
        assert_eq!(
            pipeline
                .admit_external_fixp(&mut engine, 7, (0, 0), operands)
                .unwrap(),
            C220FixpAdmission::Mte3RetirementPending
        );
        assert!(engine.commands().is_empty());
        pipeline.advance(4).unwrap();
        assert_eq!(pipeline.mte3_frontend().queued_commands(), 0);
        assert!(pipeline.mte3_retirement_candidate().is_some());
        assert!(pipeline.external_fixp_admission_blocked(operands));
        let mut empty = operands;
        empty.command.descriptor.xt &= !(0xffff << 16);
        assert!(!pipeline.external_fixp_admission_blocked(empty));
        pipeline.retire_mte3(1).unwrap();
        assert!(!pipeline.external_fixp_admission_blocked(operands));
        assert_eq!(
            pipeline
                .admit_external_fixp(&mut engine, 7, (0, 0), operands)
                .unwrap(),
            C220FixpAdmission::Active
        );
        let mut disabled = operands;
        disabled.command.descriptor.nd = 0;
        disabled.command.descriptor.xt &= !(0xffff << 16);
        assert_eq!(
            pipeline
                .admit_external_fixp(&mut engine, 8, (0, 0), disabled)
                .unwrap(),
            C220FixpAdmission::DisabledReady
        );
        assert_eq!(engine.write_completion_tick(8), Some(4));
        assert_eq!(engine.instruction_fifo().len(), 1);
        assert_eq!(engine.retirement_fifo().len(), 2);
        assert_eq!(engine.shared_engine().outstanding_external_commands(), 1);
        pipeline
            .bind_external_fixp_stages(&[
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
            ])
            .unwrap();
        assert!(matches!(
            pipeline.advance(4),
            Err(C220MtePipelineError::FixpContextMismatch)
        ));
        let mut dbids = VecDeque::new();
        let mut responses = VecDeque::new();
        let mut retired = None;
        let mut disabled_retired = None;
        let mut executed = None;
        let mut sent_bytes = 0;
        let mut l1 = C220LocalBuffer::new(131072);
        if !half_source && nz2nd && matches!(conversion, 0 | 1 | 8) {
            let mut probe = pipeline.clone();
            let mut probe_engine = engine.clone();
            let mut probe_l0c = l0c.clone();
            let mut probe_memory = MappedMemory::bind(
                SparseMemory::new(
                    vec![MemoryRegion::new(4096, vec![0; 4096]).unwrap()],
                    4096,
                    4096,
                ),
                &[4096],
            )
            .unwrap();
            let mut probe_l1 = l1.clone();
            for (id, descriptor) in [(10, 0), (11, (1 << 16) | (1 << 4))] {
                probe
                    .issue_mte3_dma(
                        id,
                        crate::sim::c220::mte::mte3::C220Mte3TransferPlan {
                            descriptor: C220DmaMovDescriptor::decode(
                                CAPTURED_C220_MOV_UB_TO_OUT_WORD,
                                descriptor,
                            )
                            .unwrap(),
                            source_address: 0,
                            destination_address: 4096,
                            bytes: if id == 10 { 0 } else { 32 },
                            dma_mode_word: 0,
                            biu_mode_word: 0,
                        },
                    )
                    .unwrap();
            }
            for tick in 5..=8 {
                probe
                    .advance_external_fixp(
                        tick,
                        &mut probe_engine,
                        C220FixpRuntimeMemory {
                            l0c: &mut probe_l0c,
                            l1: &mut probe_l1,
                            slopes: &slopes,
                            external: &mut probe_memory,
                            atomics: C220FixpAtomicConfig::default(),
                        },
                        false,
                    )
                    .unwrap();
            }
            let records: Vec<_> = probe.mte3_frontend().records().collect();
            assert_eq!(records[0].dispatch_tick, Some(7));
            assert_eq!(
                records[1].dispatch_tick,
                (core_kind != C220CoreKind::Cube).then_some(8)
            );
            assert_eq!(
                probe.last_events().iter().any(|event| matches!(
                    event,
                    crate::sim::c220::mte::pipeline::C220MtePipelineEvent::Mte3(
                        crate::sim::c220::mte::mte3::frontend::C220Mte3FrontendEvent::ExternalFixpBlocked {
                            instruction_id: 11,
                            tick: 8
                        }
                    )
                )),
                core_kind == C220CoreKind::Cube
            );
        }
        for tick in 4..512 {
            pipeline
                .advance_external_fixp(
                    tick,
                    &mut engine,
                    C220FixpRuntimeMemory {
                        l0c: &mut l0c,
                        l1: &mut l1,
                        slopes: &slopes,
                        external: &mut memory,
                        atomics: C220FixpAtomicConfig::default(),
                    },
                    false,
                )
                .unwrap();
            for event in pipeline.last_events() {
                if let crate::sim::c220::mte::pipeline::C220MtePipelineEvent::FixpExternal(
                    C220FixpRuntimeEvent::Retired {
                        tick,
                        instruction_id,
                        state,
                    },
                ) = event
                {
                    if *instruction_id == 7 {
                        retired = Some((*tick, *state));
                        assert!(engine.commands().contains_key(&8));
                        assert!(engine.retire_ready_write(*tick).unwrap().is_none());
                    } else {
                        assert_eq!(*instruction_id, 8);
                        assert!(state.lifecycle.executed_tick.is_none());
                        assert!(state.lifecycle.write_dispatched_tick.is_none());
                        disabled_retired = Some(*tick);
                    }
                }
                if let crate::sim::c220::mte::pipeline::C220MtePipelineEvent::FixpExternal(
                    C220FixpRuntimeEvent::Read(C220FixpEvent::ReceivedL0c {
                        functional: Some(event),
                        ..
                    }),
                ) = event
                    && event.executed
                {
                    assert!(executed.replace(tick).is_none());
                }
            }
            if disabled_retired.is_some() {
                break;
            }
            if let Some(command) = pipeline.take_biu_write_command().unwrap() {
                dbids.push_back((tick + 5, command.command.tag));
            }
            if let Some((_, tag)) = dbids.pop_front_if(|(ready, _)| *ready <= tick) {
                pipeline
                    .receive_biu_write_dbid(C220BiuSubcore::Cube, tag)
                    .unwrap();
            }
            if let Some(data) = pipeline.take_biu_write_data().unwrap() {
                sent_bytes += data.source.request.bytes;
                responses.push_back((tick + 10, data.source.request.tag));
                assert!(!engine.is_idle());
                assert_eq!(engine.shared_engine().outstanding_external_commands(), 1);
            }
            if let Some((_, tag)) = responses.pop_front_if(|(ready, _)| *ready <= tick) {
                let response = pipeline.receive_biu_write_response(tag).unwrap();
                if let Some(id) = engine.complete_response(response).unwrap() {
                    assert_eq!(engine.write_completion_tick(id), Some(tick));
                    assert!(engine.commands().contains_key(&id));
                    assert!(engine.retire_ready_write(tick).unwrap().is_none());
                }
            }
        }
        if matches!(conversion, 19 | 20) && !nz2nd {
            assert!(retired.is_none() && disabled_retired.is_none());
            assert!(executed.is_some());
            assert_eq!(sent_bytes, 0);
            assert_eq!(engine.retirement_fifo().len(), 2);
            assert_eq!(engine.shared_engine().outstanding_external_commands(), 1);
            assert_eq!(pipeline.fixp_store_buffer().len(), 2);
            assert!(engine.commands()[&7].write_dispatched_tick.is_some());
            assert_eq!(engine.write_completion_tick(7), None);
            for address in [4096, 5120] {
                assert_eq!(memory.read_known_at(address, 16).unwrap(), [0; 16]);
                assert_eq!(memory.read_known_at(address + 16, 16).unwrap(), [0xa5; 16]);
            }
            continue;
        }
        let (tick, state) = retired.expect("external instruction must retire");
        assert_eq!(disabled_retired, Some(tick + 1));
        assert_eq!(state.lifecycle.executed_tick, executed);
        assert!(executed.unwrap() < state.lifecycle.write_dispatched_tick.unwrap());
        assert!(state.lifecycle.write_dispatched_tick.unwrap() < tick);
        assert!(engine.is_idle() && pipeline.is_idle());
        assert_eq!(engine.shared_engine().outstanding_external_commands(), 0);
        assert_eq!(pipeline.next_external_fixp_event_tick(&engine), None);
        if !nz2nd {
            let blocks = if split { 3 } else { 2 };
            let lanes = if split { 16 } else { 32 };
            let lane_bytes = match conversion {
                6 => 2,
                17 | 18 => 1,
                _ => 4,
            };
            assert_eq!(sent_bytes, blocks * lanes * lane_bytes);
            for block in 0..blocks {
                let address = 4096 + u64::from(block) * 1024;
                assert_eq!(
                    memory
                        .read_known_at(address, (lanes * lane_bytes) as usize)
                        .unwrap(),
                    if cleared {
                        vec![0; (lanes * lane_bytes) as usize]
                    } else {
                        1_f32.to_le_bytes().repeat(lanes as usize)
                    }
                );
                assert_eq!(
                    memory
                        .read_known_at(address + u64::from(lanes * lane_bytes), 32)
                        .unwrap(),
                    [if cleared { 0xa5 } else { 0 }; 32]
                );
            }
            assert!(engine.staging().is_idle());
            continue;
        }
        if int4 {
            assert_eq!(sent_bytes, if cleared { 66 } else { 2 });
            for row in 0..2 {
                let expected = if cleared {
                    [0; 9]
                } else if half_source {
                    [0x11, 0x11, 0x11, 0x11, 0, 0, 0, 0, 1]
                } else {
                    [0x11; 9]
                };
                assert_eq!(memory.read_known_at(4096 + row * 16, 9).unwrap(), expected);
                assert_eq!(
                    memory.read_known_at(4105 + row * 16, 7).unwrap(),
                    [if cleared { 0xa5 } else { 0 }; 7]
                );
            }
            continue;
        }
        let lane = if conversion == 0 {
            1_f32.to_le_bytes().to_vec()
        } else if conversion == 6 {
            vec![0; 2]
        } else if matches!(conversion, 17 | 18) {
            vec![0]
        } else if conversion == 16 {
            0x3f80_u16.to_le_bytes().to_vec()
        } else if matches!(conversion, 8 | 9 | 23 | 24) {
            vec![1]
        } else if matches!(conversion, 12 | 13) {
            1_i16.to_le_bytes().to_vec()
        } else {
            0x3c00u16.to_le_bytes().to_vec()
        };
        assert_eq!(
            sent_bytes as usize,
            if matches!(conversion, 8 | 9 | 23 | 24) {
                2
            } else if matches!(conversion, 17 | 18) {
                66
            } else {
                34 * lane.len()
            }
        );
        for row in 0..2 {
            let address = 4096 + row * 32 * lane.len() as u64;
            assert_eq!(
                memory.read_known_at(address, 17 * lane.len()).unwrap(),
                if half_source {
                    let mut expected = lane.repeat(8);
                    expected.resize(17 * lane.len(), 0);
                    expected
                } else {
                    lane.repeat(17)
                }
            );
            assert_eq!(
                memory
                    .read_known_at(address + 17 * lane.len() as u64, 15 * lane.len())
                    .unwrap(),
                vec![if cleared { 0xa5 } else { 0 }; 15 * lane.len()]
            );
        }
    }
}
