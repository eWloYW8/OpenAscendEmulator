use super::*;

#[test]
fn scalar_ports_compete_with_mte_and_share_response_wakeup() {
    let mut service = C220UbService::default();
    for (port, bytes) in [
        (C220UbServicePort::ScalarWrite, 4),
        (C220UbServicePort::ScalarRead, 4),
        (C220UbServicePort::MteWrite0, 32),
    ] {
        assert!(
            service
                .receive(
                    0,
                    port,
                    C220UbServiceRequest {
                        id: 1,
                        address: 0,
                        bytes,
                    }
                )
                .unwrap()
        );
    }
    for tick in 1..=8 {
        let cycle = service
            .arbitrate(tick, C220UbVectorActivity::default())
            .unwrap();
        if tick == 1 {
            assert_eq!(
                cycle
                    .decisions
                    .iter()
                    .map(|d| (d.port, d.granted))
                    .collect::<Vec<_>>(),
                [
                    (C220UbServicePort::ScalarWrite, true),
                    (C220UbServicePort::ScalarRead, false),
                    (C220UbServicePort::MteWrite0, false),
                ]
            );
        }
        if tick == 2 {
            assert_eq!(cycle.completed[0].1.ready_tick, 5);
        }
        if tick == 3 {
            assert_eq!(cycle.completed[0].1.ready_tick, 6);
        }
        if tick == 5 {
            let sent = service.send_responses(tick).unwrap();
            assert_eq!(sent.len(), 2);
            assert!(sent.iter().all(|(_, response)| response.ready_tick == 6));
        }
        if tick == 8 {
            assert!(cycle.decisions[0].second_grant);
            assert_eq!(cycle.completed[0].1.ready_tick, 8);
            assert_eq!(service.send_responses(tick).unwrap()[0].1.ready_tick, 9);
        }
    }
    for port in [
        C220UbServicePort::ScalarWrite,
        C220UbServicePort::ScalarRead,
        C220UbServicePort::MteWrite0,
    ] {
        assert!(service.take_response(9, port).unwrap().is_some());
    }
    assert!(service.is_idle());
}

#[test]
fn shared_vector_ports_gate_scalar_queues_independently_of_bank_grants() {
    for shared in [false, true] {
        let mut service = C220UbService::new(shared);
        for (port, address) in [
            (C220UbServicePort::ScalarWrite, 0),
            (C220UbServicePort::ScalarRead, 32),
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
        let blocked = service
            .arbitrate(
                1,
                C220UbVectorActivity {
                    write_pending: true,
                    read_pending: true,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(blocked.scalar_write_blocked, shared);
        assert_eq!(blocked.scalar_read_blocked, shared);
        assert_eq!(blocked.completed.len(), if shared { 0 } else { 2 });
        if shared {
            let read = service
                .arbitrate(
                    2,
                    C220UbVectorActivity {
                        write_pending: true,
                        ..Default::default()
                    },
                )
                .unwrap();
            assert_eq!(read.completed[0].0, C220UbServicePort::ScalarRead);
            let write = service
                .arbitrate(3, C220UbVectorActivity::default())
                .unwrap();
            assert_eq!(write.completed[0].0, C220UbServicePort::ScalarWrite);
            // The zero-delay write response also wakes the pending read response.
            assert_eq!(service.send_responses(3).unwrap().len(), 2);
        }
    }
}

#[test]
fn partial_write_retries_after_delay_and_can_stall_again() {
    let mut service = C220UbService::default();
    let port = C220UbServicePort::MteWrite0;
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
    assert!(
        service
            .arbitrate(
                0,
                C220UbVectorActivity {
                    bank_mask: 0,
                    triggered: false,
                    ..Default::default()
                }
            )
            .unwrap()
            .decisions
            .is_empty()
    );
    assert_eq!(
        service
            .arbitrate(
                1,
                C220UbVectorActivity {
                    bank_mask: 0,
                    triggered: false,
                    ..Default::default()
                }
            )
            .unwrap()
            .bank_mask,
        3
    );
    for tick in 2..=7 {
        let cycle = service
            .arbitrate(
                tick,
                C220UbVectorActivity {
                    bank_mask: 0,
                    triggered: false,
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(cycle.decisions.is_empty());
        assert!(cycle.completed.is_empty());
    }
    let conflict = service
        .arbitrate(
            8,
            C220UbVectorActivity {
                bank_mask: 1,
                triggered: true,
                ..Default::default()
            },
        )
        .unwrap();
    assert!(
        conflict
            .decisions
            .iter()
            .all(|decision| decision.second_grant)
    );
    assert!(!conflict.decisions[0].granted);
    assert!(conflict.decisions[1].granted);
    assert!(conflict.completed.is_empty());
    let done = service
        .arbitrate(
            9,
            C220UbVectorActivity {
                bank_mask: 0,
                triggered: false,
                ..Default::default()
            },
        )
        .unwrap();
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
    let mut service = C220UbService::default();
    for (port, address) in [
        (C220UbServicePort::MteWrite0, 0),
        (C220UbServicePort::MteWrite1, 0x10000),
        (C220UbServicePort::MteRead, 0),
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
    let first = service
        .arbitrate(
            0,
            C220UbVectorActivity {
                bank_mask: 0,
                triggered: true,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(first.bank_mask, 1 | (1 << 16));
    assert_eq!(first.completed.len(), 2);
    assert!(!first.decisions[2].granted);
    let second = service
        .arbitrate(
            1,
            C220UbVectorActivity {
                bank_mask: 0,
                triggered: false,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(second.completed[0].0, C220UbServicePort::MteRead);
    assert_eq!(second.completed[0].1.ready_tick, 6);
    // One ready response queue triggers a scan of all nonempty ports.
    let responses = service.send_responses(3).unwrap();
    assert_eq!(responses.len(), 3);
    assert!(
        responses
            .iter()
            .all(|(_, response)| response.ready_tick == 4)
    );
    for port in [
        C220UbServicePort::MteWrite0,
        C220UbServicePort::MteWrite1,
        C220UbServicePort::MteRead,
    ] {
        assert!(service.take_response(4, port).unwrap().is_some());
    }
    assert!(service.is_idle());
}
