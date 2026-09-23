use super::*;
use crate::isa::c220::mte::load2d::C220Load2dInstruction;
use crate::isa::c220::mte::set2d::C220Set2dInstruction;
use crate::sim::c220::mte::interface::C220MteL1WritePort;
use crate::sim::c220::mte::mte1::frontend::C220Mte1ReadTransfer;
use std::collections::BTreeMap;

#[test]
fn biu_write_waits_for_dbid_and_all_source_packets_before_data_transport() {
    use crate::sim::c220::mte::interface::biu_write::C220BiuWriteSourceRequest;
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
        pipeline.advance_ub_service(std::iter::empty()).unwrap();
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
            pipeline.advance_ub_service(vector.iter()).unwrap();
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
}

#[test]
fn l1_fill_contends_with_load2d_and_completes_after_write_response() {
    let width = NonZeroU32::new(32).unwrap();
    let config = C220MtePipelineConfig {
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
    };
    let mut registers = [0; 32];
    registers[3] = (1 << 16) | (1 << 24);
    let read = C220Mte1Command::Read(C220Mte1ReadTransfer::Load2d(
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
