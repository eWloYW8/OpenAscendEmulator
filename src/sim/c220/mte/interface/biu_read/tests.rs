use super::*;
use crate::sim::c220::mte::uop::{C220DmaUopMode, C220DmaUopRequest, C220DmaUopRoute};

fn input(subcore: C220BiuSubcore, id: u64, address: u64, bytes: u32) -> C220BiuReadInput {
    C220BiuReadInput {
        subcore,
        destination: match subcore {
            C220BiuSubcore::Cube => write::C220BiuWriteDestination::L1,
            C220BiuSubcore::Vector0 => write::C220BiuWriteDestination::Ub0,
            C220BiuSubcore::Vector1 => write::C220BiuWriteDestination::Ub1,
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
        write_bandwidths: write::C220BiuWriteBandwidths {
            l1: NonZeroU32::new(128).unwrap(),
            l0a: NonZeroU32::new(128).unwrap(),
            l0b: NonZeroU32::new(128).unwrap(),
            ub: NonZeroU32::new(128).unwrap(),
        },
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
    assert_eq!(frontend.send(5, true).unwrap().sent(), Some(held));
    let second = frontend.send(6, true).unwrap().sent().unwrap();
    assert_eq!(second.input.generated.request.bytes, 128);
    assert_eq!(second.byte_offset, 127);
    assert_eq!(
        frontend.send(7, true).unwrap().stall,
        Some(C220BiuReadStall::NoTag)
    );
    assert_eq!(frontend.release_tag(7, held.tag).unwrap(), held);
    let third = frontend.send(8, true).unwrap().sent().unwrap();
    assert_eq!(third.tag, held.tag);
    assert_eq!(third.input.generated.request.bytes, 256);
    assert_eq!(
        frontend.arbitrate(8).unwrap().selected,
        Some(C220BiuSubcore::Cube)
    );
    frontend.release_tag(8, second.tag).unwrap();
    let tail = frontend.send(9, true).unwrap().sent().unwrap();
    assert_eq!(tail.input.generated.request.bytes, 1);
    assert_eq!(tail.byte_offset, 511);
    for request in [held, second, third, tail] {
        assert_eq!(request.input.generated.sid, Some(11));
    }
    assert!(tail.input.generated.last_in_instruction);
    assert!(frontend.contains_instruction(1));
    frontend.release_tag(9, third.tag).unwrap();
    assert!(frontend.release_tag(9, third.tag).is_err());
    frontend.release_tag(9, tail.tag).unwrap();
    assert!(!frontend.contains_instruction(1));
    assert_eq!(
        frontend
            .send(10, true)
            .unwrap()
            .sent()
            .unwrap()
            .input
            .subcore,
        C220BiuSubcore::Cube
    );
}
