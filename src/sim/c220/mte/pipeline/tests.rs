use super::*;
use crate::isa::c220::mte::load2d::C220Load2dInstruction;
use crate::isa::c220::mte::set2d::C220Set2dInstruction;
use crate::sim::c220::mte::interface::C220MteL1WritePort;
use crate::sim::c220::mte::mte1::frontend::C220Mte1ReadTransfer;
use std::collections::BTreeMap;

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
