use super::*;
use crate::sim::c220::mte::uop::{C220DmaUopMode, C220DmaUopRequest, C220DmaUopRoute};

fn input(subcore: C220BiuSubcore, id: u64, address: u64, bytes: u32) -> C220BiuReadInput {
    C220BiuReadInput {
        subcore,
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
                source_address: address,
                destination_address: 0,
                bytes,
                last_in_burst: true,
            },
        },
    }
}

#[test]
fn weighted_arbitration_split_backpressure_and_reserved_tags() {
    let mut frontend = C220BiuReadFrontend::new(C220BiuReadConfig {
        outstanding: NonZeroU32::new(2).unwrap(),
        weights: [1, 3, 1],
        group_vector_returns: true,
    });
    for subcore in C220BiuSubcore::ALL {
        assert!(
            frontend
                .push(0, input(subcore, subcore as u64, 1, 512))
                .unwrap()
        );
    }
    assert_eq!(frontend.arbitrate(2).unwrap().selected, None);
    assert_eq!(
        frontend.arbitrate(3).unwrap().selected,
        Some(C220BiuSubcore::Vector0)
    );
    assert_eq!(
        frontend.send(3, true).unwrap().stall,
        Some(C220BiuReadStall::NotReady)
    );
    let held = frontend.send(4, false).unwrap().offered.unwrap();
    assert_eq!(held.tag.get(), 1);
    assert_eq!(held.input.generated.request.bytes, 127);
    assert!(!held.input.generated.last_in_instruction);
    assert_eq!(frontend.free_tag_count(), 1);
    assert!(frontend.arbitrate(4).unwrap().pending_split_blocked);
    assert_eq!(frontend.send(5, true).unwrap().sent, Some(held));
    let second = frontend.send(6, true).unwrap().sent.unwrap();
    assert_eq!(second.input.generated.request.bytes, 128);
    assert_eq!(second.byte_offset, 127);
    assert_eq!(
        frontend.send(7, true).unwrap().stall,
        Some(C220BiuReadStall::NoTag)
    );
    assert_eq!(frontend.release_tag(7, held.tag).unwrap(), held);
    let third = frontend.send(8, true).unwrap().sent.unwrap();
    assert_eq!(third.tag, held.tag);
    assert_eq!(third.input.generated.request.bytes, 256);
    assert_eq!(
        frontend.arbitrate(8).unwrap().selected,
        Some(C220BiuSubcore::Cube)
    );
    frontend.release_tag(8, second.tag).unwrap();
    let tail = frontend.send(9, true).unwrap().sent.unwrap();
    assert_eq!(tail.input.generated.request.bytes, 1);
    assert_eq!(tail.byte_offset, 511);
    assert!(tail.input.generated.last_in_instruction);
    assert!(frontend.contains_instruction(1));
    frontend.release_tag(9, third.tag).unwrap();
    assert!(frontend.release_tag(9, third.tag).is_err());
    frontend.release_tag(9, tail.tag).unwrap();
    assert!(!frontend.contains_instruction(1));
    assert_eq!(
        frontend.send(10, true).unwrap().sent.unwrap().input.subcore,
        C220BiuSubcore::Cube
    );
}
