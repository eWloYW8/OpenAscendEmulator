use super::*;
use crate::isa::c220::mte::set2d::C220Set2dInstruction;
use crate::sim::c220::memory::l1::{C220L1Geometry, C220L1Port, C220L1Transport};
use crate::sim::c220::mte::set2d::{C220Set2dBandwidths, C220Set2dOutputRoute, C220Set2dUops};
use std::num::NonZeroU32;

fn fragment(instruction_id: u64) -> C220MteOutputFragment {
    C220MteOutputFragment {
        instruction_id,
        request_id: 0,
        destination_address: 0,
        bytes: 32,
        last_in_uop: true,
        last_in_instruction: true,
    }
}

#[test]
fn bounded_inputs_rotate_under_backpressure_and_wait_for_responses() {
    let mut interface = C220MteL1WriteInterface::default();
    for port in C220MteL1WritePort::ALL {
        for _ in 0..port.capacity() {
            assert!(interface.push(0, port, fragment(port as u64)).unwrap());
        }
        assert!(!interface.push(0, port, fragment(99)).unwrap());
    }
    assert_eq!(interface.queue_state().inputs, [10, 2, 4, 2]);
    assert_eq!(interface.preview_send(0).eligible, [false; 4]);
    assert_eq!(
        interface.preview_send(1).eligible,
        [false, true, false, true]
    );
    assert_eq!(
        interface.preview_send(3).eligible,
        [false, true, true, true]
    );
    for (tick, port) in (9..13).zip(C220MteL1WritePort::ALL) {
        let cycle = interface.send(tick, false).unwrap();
        assert_eq!(cycle.selected, Some(port));
        assert!(cycle.sent.is_none());
    }
    assert_eq!(interface.queue_state().inputs, [10, 2, 4, 2]);
    let request = interface.send(13, true).unwrap().sent.unwrap();
    assert_eq!(request.port, C220MteL1WritePort::Port0);
    assert_eq!(request.id, 1);
    assert!(interface.retire(50).unwrap().is_none());
    let snapshot = interface.clone();
    assert_eq!(
        interface.receive_response(50, 999),
        Err(C220MteL1WriteError::UnknownResponse(999))
    );
    assert_eq!(interface, snapshot);
    interface.receive_response(50, request.id).unwrap();
    let ack = interface.retire(51).unwrap().unwrap();
    assert_eq!(ack.ready_tick, 51);
    assert_eq!(ack.retired_instruction(), Some(0));
    assert!(interface.retire(51).is_err());
}

#[test]
fn set2d_shares_l1_banks_with_fixp_and_retires_on_final_response() {
    let instruction = C220Set2dInstruction::decode((3 << 29) | (1 << 22) | (1 << 7) | 2).unwrap();
    let mut registers = [0; 32];
    registers[1] = 2 | (1 << 16) | (1 << 32);
    let fill = instruction.capture(&registers, 0);
    let bandwidth = NonZeroU32::new(24).unwrap();
    let uops = C220Set2dUops::new(
        fill,
        C220Set2dBandwidths {
            l0a: bandwidth,
            l0b: bandwidth,
            l1: bandwidth,
        },
    )
    .collect::<Vec<_>>();
    let mut interface = C220MteL1WriteInterface::default();
    let mut transport = C220L1Transport::new(C220L1Geometry::new(32, 4, 1, 0).unwrap());
    let mut retired = Vec::new();
    let mut sent = Vec::new();
    let mut denied = Vec::new();
    // This fixture explicitly chooses producer -> response -> memory phases.
    for tick in 0..16 {
        if let Some(uop) = uops.get(tick as usize) {
            let C220Set2dOutputRoute::L1(port) = uop.route else {
                panic!("L1 route")
            };
            assert!(
                interface
                    .push(tick, port, uop.output_fragment(42, tick))
                    .unwrap()
            );
        }
        if let Some(ack) = interface.retire(tick).unwrap()
            && let Some(id) = ack.retired_instruction()
        {
            retired.push((tick, id));
        }
        if let Some(response) = transport
            .receive_response(tick, C220L1Port::MteWrite)
            .unwrap()
        {
            interface
                .receive_response(tick, response.request.id)
                .unwrap();
        }
        transport
            .receive_response(tick, C220L1Port::FixpWrite)
            .unwrap();
        let cycle = interface
            .send(tick, transport.request_ready(C220L1Port::MteWrite))
            .unwrap();
        if let Some(request) = cycle.sent {
            assert!(
                transport
                    .send_request(tick, C220L1Port::MteWrite, request.l1_request())
                    .unwrap()
            );
            sent.push(tick);
        }
        if tick == 3 {
            assert!(
                transport
                    .send_request(
                        tick,
                        C220L1Port::FixpWrite,
                        C220L1Request {
                            id: 200,
                            access: C220L1Access {
                                address: 0,
                                bytes: 32
                            },
                        }
                    )
                    .unwrap()
            );
        }
        let memory = transport.advance(tick).unwrap();
        if memory.decisions[1].is_some_and(|decision| !decision.granted) {
            denied.push(tick);
        }
    }
    assert_eq!(sent, [3, 4, 6, 7]);
    assert_eq!(denied, [4]);
    assert_eq!(retired, [(11, 42)]);
    assert!(interface.is_idle());
    assert!(transport.is_idle());
}
