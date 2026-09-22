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
            prefetch: false,
            generated: C220DmaGenerated {
                instruction_id: id,
                uop_index: 0,
                ready_tick: 0,
                mode: C220DmaUopMode::Wide512,
                out_of_order: false,
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

fn beat(request: C220BiuReadRequest, id: u32) -> Option<C220BiuReadBeat> {
    Some(C220BiuReadBeat {
        tag: request.tag,
        transaction_id: id,
    })
}

#[test]
fn out_of_order_tail_waits_for_last_completing_request() {
    let mut returns = C220BiuReadReturns::new(NonZeroU32::new(2).unwrap(), true).unwrap();
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
            .take_output(6, C220BiuSubcore::Vector0)
            .unwrap()
            .is_none()
    );
    returns
        .take_output(7, C220BiuSubcore::Vector0)
        .unwrap()
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
        .take_output(14, C220BiuSubcore::Vector0)
        .unwrap()
        .unwrap();
    assert!(returns.is_idle());
    assert_eq!(returns.occupancy(), [0; 2]);
}

#[test]
fn cube_drains_distinct_ports_together_but_vector_port_conflicts_serialize() {
    let mut returns = C220BiuReadReturns::new(NonZeroU32::new(4).unwrap(), true).unwrap();
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
