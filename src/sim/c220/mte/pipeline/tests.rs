use super::*;
use crate::isa::c220::mte::load2d::C220Load2dInstruction;
use crate::isa::c220::mte::set2d::C220Set2dInstruction;
use crate::sim::c220::mte::interface::C220MteL1WritePort;
use crate::sim::c220::mte::read::C220MteReadTransfer;
use std::collections::BTreeMap;

#[test]
fn l1_output_uses_shared_events_and_keeps_external_retirement_separate() {
    use crate::isa::c220::mte::l1_to_out::{C220MovL1ToOutInstruction, C220MovL1ToOutTransfer};
    use crate::sim::c220::mte::fixp::C220FixpAdmission;
    use crate::sim::c220::mte::interface::biu_write::command::C220BiuWriteConfig;
    use crate::sim::c220::mte::l1_to_out::{
        C220L1OutputCommand, C220L1OutputEngine, C220L1OutputEngineConfig, C220L1OutputEvent,
        C220L1OutputStage::*,
    };
    let width = NonZeroU32::new(32).unwrap();
    let mut pipeline = C220MtePipeline::new(
        0,
        C220MtePipelineConfig {
            core_kind: crate::sim::c220::device::C220CoreKind::Cube,
            l1: C220L1Geometry::new(32, 1, 1, 0).unwrap(),
            read_width: width,
            output_bandwidths: C220MteReadBandwidths {
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
    pipeline
        .bind_l1_output_stages(&[GenerateRead, SendRead, Packetize, GenerateWrite, SendWrite])
        .unwrap();
    let mut engine = C220L1OutputEngine::new(C220L1OutputEngineConfig {
        read_bandwidth: width,
        instruction_fifo_depth: 1,
        write_outstanding_limit: 1,
    });
    let instruction =
        C220MovL1ToOutInstruction::decode((3 << 29) | (2 << 27) | (4 << 23) | (2 << 3)).unwrap();
    let command = C220L1OutputCommand {
        transfer: C220MovL1ToOutTransfer {
            instruction,
            source_address: 0,
            destination_address: 4096,
            xm: (1 << 4) | (32 << 16),
        },
        control: 0,
        mode: crate::sim::c220::mte::uop::C220DmaUopMode::Wide512,
    };
    assert_eq!(
        pipeline.admit_l1_output(&mut engine, 91, command).unwrap(),
        C220FixpAdmission::Active
    );
    assert!(matches!(
        pipeline.advance(1),
        Err(C220MtePipelineError::L1OutputContextMismatch)
    ));
    let mut read_responses = 0;
    let mut writes = Vec::new();
    let mut delayed = None;
    let mut final_response = None;
    let mut source_completion = None;
    let mut write_completion = None;
    for tick in 1..=300 {
        pipeline.advance_l1_output(tick, &mut engine).unwrap();
        for event in &pipeline.trace {
            match event {
                C220MtePipelineEvent::Interface(C220MteL1EventOutcome::Response(Some(_))) => {
                    read_responses += 1
                }
                C220MtePipelineEvent::L1Output(C220L1OutputEvent::SourceCompleted {
                    tick,
                    instruction_id,
                }) => {
                    assert_eq!(*instruction_id, 91);
                    assert!(source_completion.replace(*tick).is_none());
                }
                C220MtePipelineEvent::L1Output(C220L1OutputEvent::WriteCompleted {
                    tick,
                    instruction_id,
                }) => {
                    assert_eq!(*instruction_id, 91);
                    assert!(write_completion.replace(*tick).is_none());
                }
                _ => {}
            }
        }
        assert!(pipeline.fixp_completions().is_empty());
        assert!(pipeline.mte1_completions().is_empty());
        if let Some(transfer) = pipeline.take_biu_write_command().unwrap() {
            let request = transfer.command;
            writes.push(request.input.generated.request);
            pipeline
                .receive_biu_write_dbid(C220BiuSubcore::Cube, request.tag)
                .unwrap();
        }
        if let Some(data) = pipeline.take_biu_write_data().unwrap() {
            assert!(delayed.is_none());
            delayed = Some((tick + 12, data));
        }
        if let Some((ready, data)) = delayed
            && tick >= ready
        {
            let response = pipeline
                .receive_biu_write_response(data.source.request.tag)
                .unwrap();
            if response.retired_instruction().is_some() {
                final_response = Some(tick);
            }
            delayed = None;
        }
        if engine.commands()[&91].response_tick.is_some() {
            break;
        }
        assert!(pipeline.retire_l1_output(&mut engine, 91).is_err());
    }
    assert_eq!(read_responses, 32);
    assert_eq!(writes.len(), 2);
    assert_eq!(
        (writes[0].destination_address, writes[0].bytes),
        (4096, 512)
    );
    assert_eq!(
        (writes[1].destination_address, writes[1].bytes),
        (4608, 512)
    );
    let state = pipeline.retire_l1_output(&mut engine, 91).unwrap();
    assert_eq!(state.source_completed_tick, source_completion);
    assert_eq!(state.response_tick, final_response);
    assert_eq!(write_completion, final_response);
    assert!(state.source_completed_tick.unwrap() < state.write_dispatched_tick.unwrap());
    assert!(state.write_dispatched_tick.unwrap() < state.response_tick.unwrap());
    assert!(pipeline.is_idle());
    assert!(engine.is_idle());
}

#[test]
fn factor_reads_share_l1_and_complete_on_fix_lane() {
    use crate::isa::c220::mte::factor::{C220FactorDescriptor, C220FactorLoad, C220FactorSource};
    use crate::sim::c220::memory::{C220L0c, C220LocalBuffer};
    use crate::sim::c220::mte::factor::c220_factor_l1_requests;
    use crate::sim::c220::mte::fixp::{C220FixpEngineConfig, C220FixpGates, C220FixpStage::*};
    use crate::sim::c220::mte::interface::C220MteL1ReadPort;
    let width = NonZeroU32::new(32).unwrap();
    let mut pipeline = C220MtePipeline::new(
        0,
        C220MtePipelineConfig {
            core_kind: crate::sim::c220::device::C220CoreKind::Cube,
            l1: C220L1Geometry::new(32, 1, 1, 0).unwrap(),
            read_width: width,
            output_bandwidths: C220MteReadBandwidths {
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
    let load = C220FactorLoad {
        source: C220FactorSource::L1,
        source_address: 0,
        destination_address: 2048,
        descriptor: C220FactorDescriptor((1 << 16) | (1 << 4) | 8),
    };
    pipeline
        .bind_fixp_stages(&[
            GenerateRead,
            SendRead,
            SendL0c,
            ReceiveL0c,
            Convert,
            Slice,
            Packetize,
            GenerateWrite,
            SendWrite,
        ])
        .unwrap();
    let mut engine = C220FixpEngine::new(C220FixpEngineConfig {
        instruction_fifo_depth: 1,
        read_bandwidth: 128,
        read_bank_count: 32,
        read_data_latency: 4,
        l0c_capacity: 131072,
    })
    .unwrap();
    let mut l0c = C220L0c::new(131072, 12).unwrap();
    let mut l1 = C220LocalBuffer::new(4096);
    let factors = C220LocalBuffer::new(4096);
    assert_eq!(
        engine
            .admit_factor_batch(
                0,
                C220MteL1ReadPort::Port2,
                c220_factor_l1_requests(load, 91, width, width),
            )
            .unwrap(),
        crate::sim::c220::mte::fixp::C220FixpAdmission::Active
    );
    assert_eq!(pipeline.next_event_tick(), None);
    assert_eq!(engine.instruction_fifo().front(), Some(&91));
    assert_eq!(
        engine
            .admit_factor_batch(
                0,
                C220MteL1ReadPort::Port2,
                c220_factor_l1_requests(load, 92, width, width)
            )
            .unwrap(),
        crate::sim::c220::mte::fixp::C220FixpAdmission::InstructionFifoFull,
    );
    assert_eq!(engine.factor_commands().len(), 1);
    assert!(matches!(
        engine.retire_factor(0, 91),
        Err(crate::sim::c220::mte::fixp::C220FixpEngineError::FactorIncomplete(91))
    ));
    assert_eq!(pipeline.next_fixp_event_tick(&engine), Some(1));
    assert!(pipeline.fixp_completions().is_empty());
    let mut completions = Vec::new();
    for tick in 1..=100 {
        pipeline
            .advance_fixp(
                tick,
                &mut engine,
                C220FixpMemory {
                    l0c: &mut l0c,
                    l1: &mut l1,
                    slopes: &factors,
                },
                C220FixpGates::default(),
            )
            .unwrap();
        completions.extend_from_slice(pipeline.fixp_completions());
        assert!(pipeline.mte1_completions().is_empty());
    }
    assert_eq!(completions, [91]);
    assert!(pipeline.is_idle());
    assert!(!engine.is_idle());
    assert!(engine.instruction_fifo().is_empty());
    let state = engine.retire_factor(100, 91).unwrap();
    assert!(state.dispatched_tick.unwrap() < state.completed_tick.unwrap());
    assert!(engine.is_idle());
}

#[test]
fn mte3_output_and_biu_split_use_independent_captured_modes() {
    use crate::isa::c220::mte::{C220DmaMovDescriptor, CAPTURED_C220_MOV_UB_TO_OUT_WORD};
    use crate::sim::c220::mte::interface::biu_write::command::C220BiuWriteConfig;
    let width = NonZeroU32::new(32).unwrap();
    let mut pipeline = C220MtePipeline::new(
        0,
        C220MtePipelineConfig {
            core_kind: crate::sim::c220::device::C220CoreKind::Cube,
            l1: C220L1Geometry::new(32, 1, 1, 0).unwrap(),
            read_width: width,
            output_bandwidths: C220MteReadBandwidths {
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
        .connect_mte3_biu(C220BiuWriteConfig {
            outstanding: NonZeroU32::new(4).unwrap(),
            weights: [1; 3],
            source_bandwidth: width,
        })
        .unwrap();
    pipeline
        .issue_mte3_dma(
            8,
            C220Mte3TransferPlan {
                control: 0,
                descriptor: C220DmaMovDescriptor::decode(
                    CAPTURED_C220_MOV_UB_TO_OUT_WORD,
                    (16 << 16) | (1 << 4),
                )
                .unwrap(),
                source_address: 0,
                destination_address: 4096,
                bytes: 512,
                dma_mode_word: 0,
                biu_mode_word: 5,
            },
        )
        .unwrap();
    let mut fragments = Vec::new();
    for tick in 0..40 {
        pipeline.advance(tick).unwrap();
        if let Some(transfer) = pipeline.take_biu_write_command().unwrap() {
            let command = transfer.command;
            fragments.push((
                command.input.generated.uop_index,
                command.byte_offset,
                command.input.generated.request.bytes,
            ));
        }
    }
    assert_eq!(
        fragments,
        [(0, 0, 128), (0, 128, 128), (0, 256, 128), (0, 384, 128)]
    );
}

#[test]
fn fixp_write_runs_on_shared_clock_and_retires_after_contended_response() {
    use crate::sim::c220::mte::interface::C220MteOutputFragment;
    let width = NonZeroU32::new(32).unwrap();
    let mut pipeline = C220MtePipeline::new(
        0,
        C220MtePipelineConfig {
            core_kind: crate::sim::c220::device::C220CoreKind::Cube,
            l1: C220L1Geometry::new(32, 1, 1, 0).unwrap(),
            read_width: width,
            output_bandwidths: C220MteReadBandwidths {
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
    let fragment = C220MteOutputFragment {
        instruction_id: 71,
        request_id: 9,
        destination_address: 0,
        bytes: 32,
        last_in_uop: true,
        last_in_instruction: true,
    };
    pipeline.enqueue_fixp_l1_write(fragment).unwrap();
    assert!(
        pipeline
            .write_interface
            .push(
                0,
                C220MteL1WritePort::Port1,
                C220MteOutputFragment {
                    instruction_id: 72,
                    ..fragment
                }
            )
            .unwrap()
    );
    let mut response_tick = None;
    let mut completed = Vec::new();
    let mut other_completed = Vec::new();
    let mut contended = false;
    for tick in 0..30 {
        pipeline.advance(tick).unwrap();
        for event in pipeline.last_events() {
            match event {
                C220MtePipelineEvent::Memory(C220L1EventOutcome::Received(cycle)) => {
                    if let (Some(fixp), Some(mte)) = (cycle.decisions[0], cycle.decisions[1]) {
                        contended |= fixp.granted && !mte.granted;
                    }
                }
                C220MtePipelineEvent::FixpWrite(C220FixpL1WriteEvent::Response(Some(request))) => {
                    assert_eq!(request.fragment, fragment);
                    assert!(response_tick.replace(tick).is_none());
                }
                _ => {}
            }
        }
        for &id in pipeline.fixp_completions() {
            assert_eq!(tick, response_tick.unwrap() + 1);
            completed.push(id);
        }
        other_completed.extend_from_slice(pipeline.l1_fill_completions());
        if pipeline.is_idle() {
            break;
        }
    }
    assert!(contended && pipeline.is_idle());
    assert_eq!(completed, [71]);
    assert_eq!(other_completed, [72]);
}

#[test]
fn fixp_engine_executes_from_read_acceptance_and_drains_through_shared_l1() {
    for conversion_mode in [0, 1, 16, 23, 24, 25, 26] {
        run_fixp_output(conversion_mode, false, false);
    }
    run_fixp_output(0, true, false);
    run_fixp_output(0, false, true);
    run_fixp_output(0, true, true);
    run_fixp_output(8, true, false);
    run_fixp_output(9, true, false);
    run_fixp_output(21, true, false);
    run_fixp_output(22, true, false);
}

fn run_fixp_output(conversion_mode: u8, integer: bool, split: bool) {
    use crate::isa::c220::mte::fixp::C220FixpDescriptor;
    use crate::sim::c220::memory::{C220L0c, C220LocalBuffer};
    use crate::sim::c220::mte::fixp::{C220FixpCommand, C220FixpEngine, C220FixpEngineConfig};
    use crate::sim::c220::mte::fixp::{C220FixpSyncPoint, C220FixpSyncRequest};
    #[derive(Default)]
    struct Sync {
        requests: Vec<C220FixpSyncRequest>,
    }
    impl C220FixpSync for Sync {
        fn blocked(
            &mut self,
            request: C220FixpSyncRequest,
        ) -> Result<bool, crate::sim::c220::sync::C220HardwareFlagTimingError> {
            let first = !self.requests.iter().any(|r| r.point == request.point);
            self.requests.push(request);
            Ok(first)
        }
    }
    let mut sync = Sync::default();
    let width = NonZeroU32::new(32).unwrap();
    let mut pipeline = C220MtePipeline::new(
        0,
        C220MtePipelineConfig {
            core_kind: crate::sim::c220::device::C220CoreKind::Cube,
            l1: C220L1Geometry::new(32, 16, 2, 9).unwrap(),
            read_width: width,
            output_bandwidths: C220MteReadBandwidths {
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
    let mut engine = C220FixpEngine::new(C220FixpEngineConfig {
        instruction_fifo_depth: 1,
        read_bandwidth: 256,
        read_bank_count: 32,
        read_data_latency: 4,
        l0c_capacity: 131072,
    })
    .unwrap();
    let mut l0c = C220L0c::new(131072, 12).unwrap();
    let byte_output = matches!(conversion_mode, 8 | 9 | 23 | 24);
    let nibble_output = matches!(conversion_mode, 21 | 22 | 25 | 26);
    let quantized = byte_output || nibble_output;
    l0c.buffer_mut()
        .write_known_linear(
            0,
            &if quantized {
                if integer {
                    2_i32.to_le_bytes()
                } else {
                    2_f32.to_le_bytes()
                }
                .repeat(512)
            } else {
                1.5_f32.to_le_bytes().repeat(128)
            },
        )
        .unwrap();
    let mut slopes = C220LocalBuffer::new(4096);
    if quantized {
        slopes
            .write_known_linear(0, &0x3f80_0000_u64.to_le_bytes().repeat(64))
            .unwrap();
    }
    let mut l1 = C220LocalBuffer::new(1048576);
    let command = C220FixpCommand {
        source_format: if integer {
            crate::sim::c220::mte::fixp::C220FixpSourceFormat::Int32
        } else {
            crate::sim::c220::mte::fixp::C220FixpSourceFormat::Fp32
        },
        descriptor: C220FixpDescriptor {
            xt: (8 << 32)
                | (8 << 16)
                | (if nibble_output {
                    64
                } else if byte_output {
                    48
                } else {
                    16
                } << 4),
            xm: (u64::from(conversion_mode) << 34)
                | (u64::from(split) << 42)
                | if quantized { 8 } else { 0 },
            nd: 0,
        },
        source_address: 0,
        destination_address: 0,
        control: 0,
        scalar_slope: 0,
        slope_base_block: 0,
        dequant_base_block: 0,
        scalar_dequant: if quantized { 0x3f80_0000 } else { 0 },
    };
    use crate::sim::c220::mte::fixp::C220FixpAdmission;
    assert_eq!(
        engine.admit(0, 71, 1, command).unwrap(),
        C220FixpAdmission::Active
    );
    assert_eq!(
        engine.admit(0, 72, 100, command).unwrap(),
        C220FixpAdmission::ReadGenerationBusy
    );
    use C220FixpStage::*;
    pipeline
        .bind_fixp_stages(&[
            GenerateRead,
            SendRead,
            SendL0c,
            ReceiveL0c,
            Convert,
            Slice,
            Packetize,
            GenerateWrite,
            SendWrite,
        ])
        .unwrap();
    let mut executed = None;
    let mut retired = None;
    let mut completed = None;
    let mut fifo_full = false;
    let mut released_before_response = false;
    let mut changed_activation = command;
    changed_activation.descriptor.xm |= 1 << 39;
    let mut changed_saturation = command;
    changed_saturation.control |= 1 << 48;
    let mut changed_addresses = command;
    changed_addresses.destination_address = 4096;
    changed_addresses.scalar_slope = 1;
    changed_addresses.control |= 1 << 47;
    assert!(!engine.resource_conflict(changed_addresses));
    for tick in 0..100 {
        pipeline
            .advance_fixp(
                tick,
                &mut engine,
                C220FixpMemory {
                    l0c: &mut l0c,
                    slopes: &slopes,
                    l1: &mut l1,
                },
                &mut sync,
            )
            .unwrap();
        if engine.admission_backpressure() == Some(C220FixpAdmission::InstructionFifoFull) {
            assert_eq!(
                engine.admit(tick, 72, 100, command).unwrap(),
                C220FixpAdmission::InstructionFifoFull
            );
            fifo_full = true;
        }
        if let Some(state) = engine.commands().get(&71)
            && let Some(dispatched) = state.write_dispatched_tick
        {
            assert!(dispatched <= tick);
            assert!(engine.instruction_fifo().is_empty());
            assert_eq!(engine.admission_backpressure(), None);
            assert_eq!(engine.retirement_fifo().front(), Some(&71));
            for changed in [changed_activation, changed_saturation] {
                assert_eq!(
                    engine.admit(tick, 72, 100, changed).unwrap(),
                    C220FixpAdmission::ResourceConflict
                );
            }
            released_before_response |= pipeline.fixp_completions().is_empty();
        }
        for &id in pipeline.fixp_completions() {
            assert_eq!(id, 71);
            assert!(engine.commands().contains_key(&id));
            assert!(executed.unwrap() < tick);
            completed = Some(tick);
        }
        if retired.is_none() && !engine.commands().contains_key(&71) {
            assert!(completed.unwrap() < tick);
            retired = Some(tick);
        }
        if pipeline.last_events().iter().any(|event| matches!(event,
            C220MtePipelineEvent::Fixp(C220FixpEvent::ReceivedL0c { functional: Some(event), .. }) if event.executed
        )) {
            assert!(executed.replace(tick).is_none());
            assert!(retired.is_none());
            let expected = if nibble_output {
                vec![0x22; 256]
            } else if byte_output {
                vec![2; 384]
            } else if conversion_mode == 0 {
                1.5_f32.to_le_bytes().repeat(128)
            } else if conversion_mode == 16 {
                0x3fc0_u16.to_le_bytes().repeat(128)
            } else {
                0x3e00_u16.to_le_bytes().repeat(128)
            };
            assert_eq!(l1.read_known(0, expected.len()).unwrap(), expected);
        }
        if engine.is_idle() && pipeline.is_idle() {
            break;
        }
    }
    assert!(executed.is_some() && retired.is_some());
    assert!(fifo_full && released_before_response);
    assert!(engine.is_idle() && pipeline.is_idle());
    assert!(engine.retirement_fifo().is_empty());
    assert!(!engine.resource_conflict(changed_activation));
    assert!(!engine.resource_conflict(changed_saturation));
    assert!(sync.requests.iter().all(|r| r.instruction_id == 71));
    for (point, retry_delay) in [
        (C220FixpSyncPoint::ReadWait, 1),
        (
            C220FixpSyncPoint::ConversionSet,
            crate::sim::c220::mte::fixp::c220_fixp_conversion_ticks(u32::from(conversion_mode)),
        ),
    ] {
        let attempts: Vec<_> = sync.requests.iter().filter(|r| r.point == point).collect();
        assert!(attempts.len() >= 2);
        assert_eq!(attempts[1].tick - attempts[0].tick, u64::from(retry_delay));
        if point == C220FixpSyncPoint::ConversionSet {
            assert_eq!(attempts.len(), 2);
        }
    }
}

#[test]
fn biu_write_waits_for_dbid_and_all_source_packets_before_data_transport() {
    use crate::sim::c220::mte::interface::biu_write::C220BiuWriteSourceRequest;
    let width = NonZeroU32::new(32).unwrap();
    let mut pipeline = C220MtePipeline::new(
        0,
        C220MtePipelineConfig {
            core_kind: crate::sim::c220::device::C220CoreKind::Cube,
            l1: C220L1Geometry::new(32, 1, 1, 0).unwrap(),
            read_width: width,
            output_bandwidths: C220MteReadBandwidths {
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
    let core = C220BiuSubcore::Vector0;
    pipeline.configure_biu_write_source(core, width).unwrap();
    let request = C220BiuWriteSourceRequest {
        tag: NonZeroU32::new(1).unwrap(),
        instruction_id: 7,
        source_address: 7,
        bytes: 60,
        gather_stride: None,
        last_in_instruction: true,
    };
    pipeline.register_biu_write_source(core, request).unwrap();
    let mut packets = Vec::new();
    let mut completion_ticks = Vec::new();
    let mut sent = None;
    for tick in 0..40 {
        pipeline.advance(tick).unwrap();
        pipeline.advance_ub_service(Default::default()).unwrap();
        if tick == 5 {
            pipeline.receive_biu_write_dbid(core, request.tag).unwrap();
        }
        for event in pipeline.last_events() {
            match event {
                C220MtePipelineEvent::BiuWriteSource(
                    _,
                    C220BiuWriteSourceEvent::ReadPacket(packet),
                ) => {
                    assert!(tick > 5);
                    packets.push((packet.address, packet.bytes));
                }
                C220MtePipelineEvent::BiuWriteSource(
                    _,
                    C220BiuWriteSourceEvent::ReadComplete { .. },
                ) => completion_ticks.push(tick),
                C220MtePipelineEvent::BiuWriteData(event) if event.sent.is_some() => {
                    assert_eq!(completion_ticks.len(), 3);
                    assert_eq!(tick, completion_ticks[2] + 1);
                    sent = event.sent;
                }
                _ => {}
            }
        }
        if let Some(data) = pipeline.take_biu_write_data().unwrap() {
            assert_eq!(Some(data), sent);
            assert_eq!(tick, data.sent_tick + 1);
            assert_eq!(data.source.request, request);
            assert!(!pipeline.is_idle());
            assert!(pipeline.register_biu_write_source(core, request).is_err());
            let response = pipeline.receive_biu_write_response(request.tag).unwrap();
            assert_eq!(response.retired_instruction(), Some(7));
            assert!(pipeline.receive_biu_write_response(request.tag).is_err());
            assert!(pipeline.is_idle());
            break;
        }
    }
    assert_eq!(packets, [(0, 32), (32, 32), (64, 3)]);
    assert!(sent.is_some());
}

#[test]
fn ub_reads_share_vector_banks_and_wait_for_matching_response_tags() {
    use crate::sim::c220::memory::C220UbCycle;
    use crate::sim::c220::mte::interface::ub_read::C220UbReadFragment;

    let width = NonZeroU32::new(32).unwrap();
    let config = C220MtePipelineConfig {
        core_kind: crate::sim::c220::device::C220CoreKind::Cube,
        l1: C220L1Geometry::new(32, 1, 1, 0).unwrap(),
        read_width: width,
        output_bandwidths: C220MteReadBandwidths {
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
    };
    let mut completion_ticks = Vec::new();
    for contend in [false, true] {
        let mut pipeline = C220MtePipeline::new(0, config);
        let core = C220BiuSubcore::Vector0;
        assert!(
            pipeline
                .push_ub_read(
                    core,
                    C220UbReadFragment {
                        tag: 9,
                        address: 0,
                        bytes: 32,
                        completes_read: true,
                    }
                )
                .unwrap()
        );
        let mut observed_response = None;
        for tick in 0..30 {
            pipeline.advance(tick).unwrap();
            let vector = (contend && (5..=10).contains(&tick)).then_some(C220UbCycle {
                tick,
                bank_mask: u64::MAX,
                read_group_mask: 0,
                write_group_mask: 0,
                decisions: Vec::new(),
            });
            pipeline
                .advance_ub_service(crate::sim::c220::memory::ub_service::C220UbVectorActivity {
                    bank_mask: vector.as_ref().map_or(0, |cycle| cycle.bank_mask),
                    triggered: vector.is_some(),
                    ..Default::default()
                })
                .unwrap();
            for event in pipeline.last_events() {
                if let C220MtePipelineEvent::UbReadResponse(_, request) = event {
                    assert_eq!(request.sent_tick, 4);
                    observed_response = Some(tick);
                }
            }
            assert!(pipeline.take_ub_read_completion(core, 8).unwrap().is_none());
            if let Some(ack) = pipeline.take_ub_read_completion(core, 9).unwrap() {
                assert_eq!(Some(tick - 3), observed_response);
                assert_eq!(ack.request.fragment.tag, 9);
                completion_ticks.push(tick);
                assert!(pipeline.is_idle());
                break;
            }
        }
    }
    assert_eq!(completion_ticks, [15, 20]);
    use crate::sim::c220::memory::ub_service::{
        C220UbServicePort, C220UbServiceRequest, C220UbVectorActivity,
    };
    for shared in [false, true] {
        let mut pipeline = C220MtePipeline::new(
            0,
            C220MtePipelineConfig {
                core_kind: crate::sim::c220::device::C220CoreKind::Vector0,
                ..config
            },
        );
        for core in [C220BiuSubcore::Vector0, C220BiuSubcore::Vector1] {
            pipeline.configure_scalar_ub_ports(core, shared).unwrap();
        }
        for core in [C220BiuSubcore::Vector0, C220BiuSubcore::Vector1] {
            let service = pipeline.ub_memory_mut(core).unwrap();
            assert_eq!(service.scalar_uses_vector_ports(), shared);
            assert!(
                service
                    .receive(
                        0,
                        C220UbServicePort::ScalarRead,
                        C220UbServiceRequest {
                            id: 0,
                            address: 0,
                            bytes: 32
                        }
                    )
                    .unwrap()
            );
        }
        assert!(
            pipeline
                .configure_scalar_ub_ports(C220BiuSubcore::Vector0, !shared)
                .is_err()
        );
        pipeline.advance(1).unwrap();
        pipeline
            .advance_ub_service(C220UbVectorActivity {
                read_pending: true,
                triggered: true,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(
            pipeline
                .ub_memory(C220BiuSubcore::Vector0)
                .unwrap()
                .inputs(C220UbServicePort::ScalarRead)
                .len(),
            usize::from(shared)
        );
        assert!(
            pipeline
                .ub_memory(C220BiuSubcore::Vector1)
                .unwrap()
                .inputs(C220UbServicePort::ScalarRead)
                .is_empty()
        );
        pipeline.advance(2).unwrap();
        pipeline.advance_ub_service(Default::default()).unwrap();
        assert!(
            pipeline
                .ub_memory(C220BiuSubcore::Vector0)
                .unwrap()
                .inputs(C220UbServicePort::ScalarRead)
                .is_empty()
        );
    }
}

#[test]
fn l1_fill_contends_with_load2d_and_completes_after_write_response() {
    let width = NonZeroU32::new(32).unwrap();
    let config = C220MtePipelineConfig {
        core_kind: crate::sim::c220::device::C220CoreKind::Cube,
        l1: C220L1Geometry::new(32, 1, 1, 0).unwrap(),
        read_width: width,
        output_bandwidths: C220MteReadBandwidths {
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
    };
    let mut registers = [0; 32];
    registers[3] = (1 << 16) | (1 << 24);
    let read = C220Mte1Command::Read(C220MteReadTransfer::Load2d(
        C220Load2dInstruction::decode((3 << 29) | (1 << 17) | (2 << 12) | (3 << 7))
            .unwrap()
            .capture(&registers)
            .unwrap(),
    ));
    registers[3] = 1 | (32 << 16);
    let fill = C220Set2dInstruction::decode((3 << 29) | (1 << 22) | (1 << 17) | (3 << 7) | 2)
        .unwrap()
        .capture(&registers, 0);
    let mut read_ticks = Vec::new();
    for with_fill in [false, true] {
        let mut pipeline = C220MtePipeline::new(0, config);
        pipeline.issue_mte1(11, read).unwrap();
        if with_fill {
            assert!(!pipeline.selected_generator_idle());
            assert!(pipeline.can_issue_l1_fill(fill));
            assert!(!pipeline.issue_l1_fill(22, fill).unwrap().completion_ready);
        }
        let mut responses = BTreeMap::new();
        let mut sent = BTreeMap::new();
        let mut completions = Vec::new();
        let mut read_blocked = false;
        for tick in 0..300 {
            pipeline.advance(tick).unwrap();
            for event in pipeline.last_events() {
                match event {
                    C220MtePipelineEvent::Memory(C220L1EventOutcome::Received(cycle)) => {
                        if let Some(read) = cycle.decisions[C220L1Port::MteRead as usize]
                            && !read.granted
                        {
                            let write = cycle.decisions[C220L1Port::MteWrite as usize].unwrap();
                            assert!(write.granted);
                            assert_ne!(read.bank_mask & write.bank_mask, 0);
                            read_blocked = true;
                        }
                    }
                    C220MtePipelineEvent::L1Write(C220MteL1WriteEventOutcome::Sent(send)) => {
                        if let Some(request) = send.sent {
                            assert_eq!(request.port, C220MteL1WritePort::Port2);
                            assert!(sent.insert(request.id, request).is_none());
                        }
                    }
                    C220MtePipelineEvent::L1Write(C220MteL1WriteEventOutcome::Response(Some(
                        request,
                    ))) => {
                        assert_eq!(sent.get(&request.id), Some(request));
                        assert!(responses.insert(request.id, tick).is_none());
                    }
                    C220MtePipelineEvent::L1Write(C220MteL1WriteEventOutcome::Acknowledged(
                        Some(ack),
                    )) => {
                        assert_eq!(tick, responses[&ack.request.id] + 1);
                        assert_eq!(ack.ready_tick, tick);
                    }
                    _ => {}
                }
            }
            for &id in pipeline.mte1_completions() {
                assert_eq!(id, 11);
                read_ticks.push(tick);
            }
            completions.extend_from_slice(pipeline.l1_fill_completions());
            if pipeline.is_idle() {
                break;
            }
        }
        assert!(pipeline.is_idle());
        assert_eq!(read_blocked, with_fill);
        assert_eq!(completions, if with_fill { vec![22] } else { vec![] });
        assert_eq!(sent.len(), responses.len());
        assert_eq!(
            sent.values()
                .map(|r| u64::from(r.fragment.bytes))
                .sum::<u64>(),
            if with_fill { fill.byte_count() } else { 0 }
        );
    }
    assert_eq!(read_ticks.len(), 2);
    assert!(read_ticks[1] > read_ticks[0]);
}
