use super::*;
use crate::isa::c220::mte::fixp::C220FixpDescriptor;
use crate::memory::{region::MemoryRegion, sparse::SparseMemory};
use crate::sim::c220::memory::l1::C220L1Geometry;
use crate::sim::c220::mte::{
    interface::biu_write::command::C220BiuWriteConfig, mte1::frontend::C220Mte1ReadBandwidths,
    pipeline::C220MtePipelineConfig, set2d::C220Set2dBandwidths,
};
use std::num::NonZeroU32;

#[test]
fn external_engine_executes_layouts_and_waits_for_ordered_retirement() {
    let cases = [0, 1, 8, 9, 10, 11, 12, 13, 21, 22, 23, 24, 25, 26]
        .into_iter()
        .map(|mode| (mode, true, false))
        .chain([(0, false, false), (0, false, true)]);
    for (conversion, nz2nd, split) in cases {
        let int4 = matches!(conversion, 21 | 22 | 25 | 26);
        use C220FixpExternalStage::*;
        let width = NonZeroU32::new(32).unwrap();
        let mut pipeline = C220MtePipeline::new(
            0,
            C220MtePipelineConfig {
                l1: C220L1Geometry::new(32, 1, 1, 0).unwrap(),
                read_width: width,
                output_bandwidths: C220Mte1ReadBandwidths {
                    l0a: width,
                    l0b: width,
                    bt: width,
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
        let mut engine = C220FixpExternalEngine::new(
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
                vec![MemoryRegion::new(4096, vec![0; 4096]).unwrap()],
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
                source_format: if matches!(conversion, 8..=13 | 21 | 22) {
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
        assert_eq!(
            pipeline
                .admit_external_fixp(&mut engine, 7, (0, 0), operands)
                .unwrap(),
            C220FixpAdmission::Mte3RetirementPending
        );
        assert!(engine.commands().is_empty());
        for tick in 0..=4 {
            pipeline.advance(tick).unwrap();
        }
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
        for tick in 4..512 {
            pipeline
                .advance_external_fixp(
                    tick,
                    &mut engine,
                    C220FixpExternalMemory {
                        l0c: &mut l0c,
                        slopes: &slopes,
                        external: &mut memory,
                        atomics: C220FixpAtomicConfig::default(),
                    },
                    false,
                )
                .unwrap();
            for event in pipeline.last_events() {
                if let crate::sim::c220::mte::pipeline::C220MtePipelineEvent::FixpExternal(
                    C220FixpExternalEvent::Retired {
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
                    C220FixpExternalEvent::Read(C220FixpEvent::ReceivedL0c {
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
        let (tick, state) = retired.expect("external instruction must retire");
        assert_eq!(disabled_retired, Some(tick + 1));
        assert_eq!(state.lifecycle.executed_tick, executed);
        assert!(executed.unwrap() < state.lifecycle.write_dispatched_tick.unwrap());
        assert!(state.lifecycle.write_dispatched_tick.unwrap() < tick);
        assert!(engine.is_idle() && pipeline.is_idle());
        assert_eq!(pipeline.next_external_fixp_event_tick(&engine), None);
        if !nz2nd {
            let blocks = if split { 3 } else { 2 };
            let lanes = if split { 16 } else { 32 };
            assert_eq!(sent_bytes, blocks * lanes * 4);
            for block in 0..blocks {
                let address = 4096 + u64::from(block) * 1024;
                assert_eq!(
                    memory.read_known_at(address, lanes as usize * 4).unwrap(),
                    1_f32.to_le_bytes().repeat(lanes as usize)
                );
                assert_eq!(
                    memory
                        .read_known_at(address + u64::from(lanes) * 4, 32)
                        .unwrap(),
                    [0; 32]
                );
            }
            assert!(engine.staging().is_idle());
            continue;
        }
        if int4 {
            assert_eq!(sent_bytes, 2);
            for row in 0..2 {
                assert_eq!(memory.read_known_at(4096 + row * 16, 9).unwrap(), [0x11; 9]);
                assert_eq!(memory.read_known_at(4105 + row * 16, 7).unwrap(), [0; 7]);
            }
            continue;
        }
        let lane = if conversion == 0 {
            1_f32.to_le_bytes().to_vec()
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
            } else {
                34 * lane.len()
            }
        );
        for row in 0..2 {
            let address = 4096 + row * 32 * lane.len() as u64;
            assert_eq!(
                memory.read_known_at(address, 17 * lane.len()).unwrap(),
                lane.repeat(17)
            );
            assert_eq!(
                memory
                    .read_known_at(address + 17 * lane.len() as u64, 15 * lane.len())
                    .unwrap(),
                vec![0; 15 * lane.len()]
            );
        }
    }
}
