use super::*;

#[test]
fn partial_write_retries_after_delay_and_can_stall_again() {
    let mut service = C220UbMteService::default();
    let port = C220UbMtePort::Write0;
    assert!(
        service
            .receive(
                0,
                port,
                C220UbServiceRequest {
                    id: 1,
                    address: 31,
                    bytes: 2
                }
            )
            .unwrap()
    );
    assert!(service.arbitrate(0, 0, false).unwrap().decisions.is_empty());
    assert_eq!(service.arbitrate(1, 0, false).unwrap().bank_mask, 3);
    for tick in 2..=7 {
        let cycle = service.arbitrate(tick, 0, false).unwrap();
        assert!(cycle.decisions.is_empty());
        assert!(cycle.completed.is_empty());
    }
    let conflict = service.arbitrate(8, 1, true).unwrap();
    assert!(
        conflict
            .decisions
            .iter()
            .all(|decision| decision.second_grant)
    );
    assert!(!conflict.decisions[0].granted);
    assert!(conflict.decisions[1].granted);
    assert!(conflict.completed.is_empty());
    let done = service.arbitrate(9, 0, false).unwrap();
    assert_eq!(done.completed.len(), 1);
    assert_eq!(done.completed[0].1.ready_tick, 12);
    assert!(service.send_responses(11).unwrap().is_empty());
    assert_eq!(service.send_responses(12).unwrap().len(), 1);
    assert!(service.take_response(12, port).unwrap().is_none());
    let response = service.take_response(13, port).unwrap().unwrap();
    assert_eq!(response.completion_tick, 9);
    assert_eq!(response.ready_tick, 13);
    assert!(service.is_idle());
}

#[test]
fn mte_ports_share_banks_but_not_vector_group_limits() {
    let mut service = C220UbMteService::default();
    for (port, address) in [
        (C220UbMtePort::Write0, 0),
        (C220UbMtePort::Write1, 0x10000),
        (C220UbMtePort::Read, 0),
    ] {
        service
            .receive(
                0,
                port,
                C220UbServiceRequest {
                    id: 1,
                    address,
                    bytes: 32,
                },
            )
            .unwrap();
    }
    // A higher-priority port can trigger the callback before MTE queue delay
    // elapses; every nonempty port is visited by that callback.
    let first = service.arbitrate(0, 0, true).unwrap();
    assert_eq!(first.bank_mask, 1 | (1 << 16));
    assert_eq!(first.completed.len(), 2);
    assert!(!first.decisions[2].granted);
    let second = service.arbitrate(1, 0, false).unwrap();
    assert_eq!(second.completed[0].0, C220UbMtePort::Read);
    assert_eq!(second.completed[0].1.ready_tick, 6);
    // One ready response queue triggers a scan of all nonempty ports.
    let responses = service.send_responses(3).unwrap();
    assert_eq!(responses.len(), 3);
    assert!(
        responses
            .iter()
            .all(|(_, response)| response.ready_tick == 4)
    );
    for port in C220UbMtePort::ALL {
        assert!(service.take_response(4, port).unwrap().is_some());
    }
    assert!(service.is_idle());
}
