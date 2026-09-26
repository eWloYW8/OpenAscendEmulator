use super::*;
use crate::sim::c220::mte::dma::C220DmaGenerated;
use crate::sim::c220::mte::interface::biu_read::C220BiuReadInput;
use crate::sim::c220::mte::uop::{C220DmaUopMode, C220DmaUopRequest, C220DmaUopRoute};

fn request(tag: u32, id: u64, core: C220BiuSubcore, bytes: u32) -> C220BiuReadRequest {
    C220BiuReadRequest {
        tag: NonZeroU32::new(tag).unwrap(),
        byte_offset: 0,
        input: C220BiuReadInput {
            subcore: core,
            destination: match core {
                C220BiuSubcore::Cube => C220BiuWriteDestination::L1,
                C220BiuSubcore::Vector0 => C220BiuWriteDestination::Ub0,
                C220BiuSubcore::Vector1 => C220BiuWriteDestination::Ub1,
            },
            prefetch: false,
            generated: C220DmaGenerated {
                sid: Some(11),
                instruction_id: id,
                uop_index: 0,
                ready_tick: 0,
                mode: C220DmaUopMode::Wide512,
                out_of_order: false,
                destination: crate::sim::c220::mte::uop::C220DmaDestinationLayout {
                    base: 0,
                    burst_bytes: bytes,
                    burst_stride: u64::from(bytes),
                },
                last_in_instruction: true,
                request: C220DmaUopRequest {
                    route: C220DmaUopRoute::Ordinary,
                    burst_index: 0,
                    source_address: 0,
                    destination_address: 0,
                    bytes,
                    last_in_burst: true,
                },
            },
        },
    }
}

fn bandwidths(bytes: u32) -> C220BiuWriteBandwidths {
    let width = NonZeroU32::new(bytes).unwrap();
    C220BiuWriteBandwidths {
        l1: width,
        l0a: width,
        l0b: width,
        ub: width,
    }
}

fn beat(request: C220BiuReadRequest, id: u32) -> Option<C220BiuReadBeat> {
    Some(C220BiuReadBeat {
        tag: request.tag,
        transaction_id: id,
    })
}

#[test]
fn truncation_keeps_exact_packet_offset_and_emits_raw_partial_tail() {
    let mut aligner = WriteAligner::new(NonZeroU32::new(32).unwrap());
    let output = |id, bytes, last| {
        let mut request = request(1, id, C220BiuSubcore::Cube, bytes);
        request.input.generated.request.route = C220DmaUopRoute::L1Take4;
        request.input.generated.last_in_instruction = last;
        C220BiuReadOutput {
            ready_tick: 0,
            request,
            last_in_instruction: last,
        }
    };
    let full = aligner.plan(output(1, 256, true)).unwrap().front().unwrap();
    assert_eq!((full.destination_address, full.bytes), (0, 32));
    assert!(full.last_in_instruction);
    assert_eq!(aligner.progress.destination_offset, 32);
    assert_eq!(aligner.progress.instruction_id, None);
    let tail = aligner.plan(output(2, 32, true)).unwrap().front().unwrap();
    assert_eq!(
        (tail.destination_address, tail.bytes, tail.logical_bytes),
        (32, 4, 4)
    );
    assert!(tail.last_in_instruction);
    assert_eq!(aligner.progress, C220BiuWriteProgress::default());
    assert!(
        aligner
            .plan(output(3, 127, false))
            .unwrap()
            .front()
            .is_none()
    );
    assert_eq!(aligner.progress.buffered_bytes, 12);
    let tail = aligner.plan(output(3, 1, true)).unwrap().front().unwrap();
    assert_eq!(tail.bytes, 12);
    assert!(tail.last_in_instruction);
    assert_eq!(aligner.progress, C220BiuWriteProgress::default());
}

#[test]
fn out_of_order_tail_waits_for_last_completing_request() {
    let mut returns =
        C220BiuReadReturns::new(NonZeroU32::new(2).unwrap(), true, bandwidths(128)).unwrap();
    let mut first = request(1, 7, C220BiuSubcore::Vector0, 128);
    first.input.generated.last_in_instruction = false;
    first.input.generated.out_of_order = true;
    let mut tail = request(2, 7, C220BiuSubcore::Vector0, 128);
    tail.input.generated.out_of_order = true;
    returns.track(0, first).unwrap();
    returns.track(0, tail).unwrap();
    assert_eq!(
        returns.receive(1, [beat(tail, 0), None]).unwrap(),
        [true, false]
    );
    assert_eq!(returns.ingress(2, 0).unwrap(), None);
    assert_eq!(returns.ingress(3, 0).unwrap(), Some(tail.tag));
    assert_eq!(returns.select(3).unwrap(), [None; 3]);
    assert_eq!(returns.select(4).unwrap(), [None, Some(tail.tag), None]);
    assert!(returns.read(4).unwrap().is_empty());
    assert_eq!(returns.read(5).unwrap().len(), 1);
    assert!(
        !returns
            .egress(6, C220BiuSubcore::Vector0)
            .unwrap()
            .unwrap()
            .last_in_instruction
    );
    assert!(
        returns
            .send_output(6, C220BiuSubcore::Vector0, true)
            .unwrap()
            .sent()
            .is_none()
    );
    returns
        .send_output(7, C220BiuSubcore::Vector0, true)
        .unwrap()
        .sent()
        .unwrap();
    returns.receive(8, [beat(first, 0), None]).unwrap();
    returns.ingress(10, 0).unwrap();
    returns.select(11).unwrap();
    returns.read(12).unwrap();
    let output = returns
        .egress(13, C220BiuSubcore::Vector0)
        .unwrap()
        .unwrap();
    assert_eq!(output.request, first);
    assert!(output.last_in_instruction);
    assert!(returns.contains_instruction(7));
    returns
        .send_output(14, C220BiuSubcore::Vector0, true)
        .unwrap()
        .sent()
        .unwrap();
    assert!(returns.is_idle());
    assert_eq!(returns.occupancy(), [0; 2]);
}

#[test]
fn cube_drains_distinct_ports_together_but_vector_port_conflicts_serialize() {
    let mut returns =
        C220BiuReadReturns::new(NonZeroU32::new(4).unwrap(), true, bandwidths(128)).unwrap();
    let cube = request(1, 1, C220BiuSubcore::Cube, 256);
    let v0 = request(2, 2, C220BiuSubcore::Vector0, 128);
    let v1 = request(3, 3, C220BiuSubcore::Vector1, 128);
    for request in [cube, v0, v1] {
        returns.track(0, request).unwrap();
    }
    assert_eq!(
        returns.receive(1, [beat(cube, 1), beat(cube, 0)]).unwrap(),
        [true; 2]
    );
    returns.receive(2, [beat(v0, 2), None]).unwrap();
    returns.receive(3, [beat(v1, 4), None]).unwrap();
    for tick in 3..=5 {
        returns.ingress(tick, 0).unwrap();
    }
    assert_eq!(returns.select(5).unwrap(), [Some(cube.tag), None, None]);
    let drained = returns.read(6).unwrap();
    assert_eq!(
        drained
            .iter()
            .map(|b| b.beat.transaction_id)
            .collect::<Vec<_>>(),
        [0, 1]
    );
    assert_eq!(drained.iter().map(|b| b.port).collect::<Vec<_>>(), [0, 1]);
    assert_eq!(
        returns.select(6).unwrap(),
        [None, Some(v0.tag), Some(v1.tag)]
    );
    assert_eq!(returns.read(7).unwrap()[0].beat.tag, v0.tag);
    assert_eq!(returns.select(7).unwrap(), [None; 3]);
    assert_eq!(returns.read(8).unwrap()[0].beat.tag, v1.tag);
    assert!(!returns.has_active_tags());
    assert_eq!(returns.occupancy(), [0; 2]);
}

#[test]
fn destination_send_holds_adapter_credit_and_rounds_only_interface_bytes() {
    let core = C220BiuSubcore::Vector0;
    let mut returns =
        C220BiuReadReturns::new(NonZeroU32::new(2).unwrap(), true, bandwidths(64)).unwrap();
    let mut request = request(1, 1, core, 65);
    request.input.generated.out_of_order = true;
    request.input.generated.request.destination_address = 17;
    let output = C220BiuReadOutput {
        ready_tick: 4,
        request,
        last_in_instruction: true,
    };
    returns.adapters[1].push_back(output);
    assert_eq!(
        returns.send_output(3, core, true).unwrap().stall,
        Some(C220BiuWriteStall::NotReady)
    );
    let blocked = returns.send_output(4, core, false).unwrap();
    assert_eq!(blocked.stall, Some(C220BiuWriteStall::DestinationFull));
    assert_eq!(returns.adapter(core).len(), 1);
    let first = returns.send_output(5, core, true).unwrap();
    assert_eq!(first.sent(), blocked.offered);
    let first = first.sent().unwrap();
    assert_eq!(
        (first.destination_address, first.bytes, first.logical_bytes),
        (17, 64, 64)
    );
    assert!(!first.last_in_instruction);
    assert_eq!(returns.adapter(core).len(), 1);
    assert!(returns.send_output(5, core, true).is_err());
    assert!(
        returns
            .send_output(6, core, false)
            .unwrap()
            .consumed
            .is_none()
    );
    let tail = returns.send_output(7, core, true).unwrap();
    let fragment = tail.sent().unwrap();
    assert_eq!(
        (
            fragment.destination_address,
            fragment.bytes,
            fragment.logical_bytes
        ),
        (81, 32, 1)
    );
    assert!(fragment.last_in_instruction);
    assert_eq!(tail.consumed, Some(output));
    assert!(returns.is_idle());
}

#[test]
fn ordered_destination_coalesces_across_requests_and_restores_collapsed_gaps() {
    let core = C220BiuSubcore::Vector0;
    let mut returns =
        C220BiuReadReturns::new(NonZeroU32::new(2).unwrap(), true, bandwidths(64)).unwrap();
    let mut tick = 0;
    let mut send = |bytes, last_in_burst, last_in_instruction, collapsed| {
        let mut request = request(1, 1, core, bytes);
        request.input.generated.destination =
            crate::sim::c220::mte::uop::C220DmaDestinationLayout {
                base: 0x1000,
                burst_bytes: 32,
                burst_stride: 160,
            };
        request.input.generated.request.last_in_burst = last_in_burst;
        if collapsed {
            request.input.generated.request.route = C220DmaUopRoute::DestinationGapCollapse;
        }
        returns.adapters[1].push_back(C220BiuReadOutput {
            ready_tick: tick,
            request,
            last_in_instruction,
        });
        let mut fragments = Vec::new();
        while !returns.adapter(core).is_empty() {
            let sent = returns.send_output(tick, core, true).unwrap();
            fragments.extend(sent.sent());
            tick += 1;
        }
        fragments
    };
    assert!(send(31, false, false, false).is_empty());
    let first = send(33, false, false, false);
    assert_eq!((first[0].destination_address, first[0].bytes), (0x1000, 64));
    assert!(!first[0].last_in_instruction);
    let burst_tail = send(1, true, false, false);
    assert_eq!(
        (burst_tail[0].destination_address, burst_tail[0].bytes),
        (0x1040, 32)
    );
    let tail = send(32, true, true, false);
    assert_eq!((tail[0].destination_address, tail[0].bytes), (0x10a0, 32));
    assert!(tail[0].last_in_instruction);
    let full = send(127, false, false, true);
    assert_eq!(
        full.iter()
            .map(|f| f.destination_address)
            .collect::<Vec<_>>(),
        [0x1000, 0x10a0, 0x1140]
    );
    let tail = send(1, true, true, true);
    assert_eq!((tail[0].destination_address, tail[0].bytes), (0x11e0, 32));
    assert!(tail[0].last_in_instruction);
    assert_eq!(
        returns.write_progress(C220BiuWriteDestination::Ub0),
        C220BiuWriteProgress::default()
    );
    assert!(returns.is_idle());
}
