use super::*;
use crate::sim::c220::scalar::C220ScalarMappedAddress;
use crate::sim::c220::scalar::lsu::store_buffer::C220LsuMemory;

#[test]
fn split_store_notifications_observe_shared_completion_count() {
    use crate::sim::c220::scalar::C220StoreOperands;
    use crate::sim::c220::scalar::lsu::scheduler::C220LsuStorePath;
    for early in [false, true] {
        for reverse in [false, true] {
            let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
            machine.set_xreg(5, 0x1038).unwrap();
            let operands = C220StoreOperands::capture(
                &machine,
                0,
                (9 << 24) | (3 << 22) | (7 << 17) | (5 << 12) | (8 << 7) | 1,
            )
            .unwrap();
            let mut lane = C220LsuCommitLane::new(C220LoadCommitMode::Retirement);
            lane.admit_store_pair(0, C220LsuRequestId(0), C220LsuRequestId(1), operands)
                .unwrap();
            assert_eq!(lane.pending_count(), 1);
            let mut data = [
                C220LsuStoreValue {
                    request: C220LsuRequestId(0),
                    tick: 2,
                    operands,
                    mapped: C220ScalarMappedAddress {
                        address: 0x1038,
                        memory: C220LsuMemory::External,
                        stack: false,
                    },
                    path: C220LsuStorePath::Cache,
                    part: C220LsuPairPart::First,
                },
                C220LsuStoreValue {
                    request: C220LsuRequestId(1),
                    tick: 2,
                    operands,
                    mapped: C220ScalarMappedAddress {
                        address: 0x1040,
                        memory: C220LsuMemory::External,
                        stack: false,
                    },
                    path: C220LsuStorePath::Refill,
                    part: C220LsuPairPart::Second,
                },
            ];
            if reverse {
                data.reverse();
            }
            lane.complete_store_at(2, data[0]).unwrap();
            assert!(lane.complete_store_at(2, data[0]).is_err());
            let tick = if early {
                assert!(lane.retire_next_at(3, &mut machine).unwrap().is_none());
                assert_eq!(lane.pending_count(), 1);
                4
            } else {
                2
            };
            data[1].tick = tick;
            lane.complete_store_at(tick, data[1]).unwrap();
            let Some(C220LsuRetirement::Store {
                first_request,
                repeated_notification,
                final_notification,
                ..
            }) = lane.retire_next_at(tick + 1, &mut machine).unwrap()
            else {
                panic!("store retirement");
            };
            assert_eq!(first_request, C220LsuRequestId(0));
            assert!(!repeated_notification);
            assert_eq!(final_notification, early);
            let next = lane.retire_next_at(tick + 2, &mut machine).unwrap();
            if early {
                assert!(next.is_none());
            } else {
                assert!(matches!(
                    next,
                    Some(C220LsuRetirement::Store {
                        repeated_notification: true,
                        final_notification: true,
                        ..
                    })
                ));
            }
            assert_eq!(lane.pending_count(), 0);
            assert_eq!(lane.retirement_occupancy(), 0);
        }
    }
}

#[test]
fn split_responses_share_state_but_consume_individual_notifications() {
    for mode in [
        C220LoadCommitMode::DataBypass,
        C220LoadCommitMode::Retirement,
    ] {
        for path in [
            C220LsuLoadPath::Cache,
            C220LsuLoadPath::StoreForward,
            C220LsuLoadPath::Refill,
        ] {
            for reverse in [false, true] {
                for early in [false, true] {
                    let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
                    machine.set_xreg(5, 0x1038).unwrap();
                    machine.set_xreg(7, u64::MAX).unwrap();
                    machine.set_xreg(8, u64::MAX).unwrap();
                    let operands = C220LoadOperands::capture(
                        &machine,
                        0,
                        (9 << 24) | (3 << 22) | (7 << 17) | (5 << 12) | (8 << 7),
                    )
                    .unwrap();
                    let mut lane = C220LsuCommitLane::new(mode);
                    lane.issue(0, C220LoadId(0), operands, &mut machine)
                        .unwrap();
                    lane.admit(
                        1,
                        C220LoadId(0),
                        C220LsuRequestId(0),
                        Some(C220LsuRequestId(1)),
                    )
                    .unwrap();
                    let mut parts = [
                        C220LsuLoadValue {
                            request: C220LsuRequestId(0),
                            tick: 10,
                            operands,
                            mapped: C220ScalarMappedAddress {
                                address: 0x1038,
                                memory: C220LsuMemory::External,
                                stack: false,
                            },
                            value: 17,
                            second_value: Some(0),
                            path,
                            part: C220LsuPairPart::First,
                        },
                        C220LsuLoadValue {
                            request: C220LsuRequestId(1),
                            tick: 10,
                            operands,
                            mapped: C220ScalarMappedAddress {
                                address: 0x1040,
                                memory: C220LsuMemory::External,
                                stack: false,
                            },
                            value: 0,
                            second_value: Some(29),
                            path,
                            part: C220LsuPairPart::Second,
                        },
                    ];
                    if reverse {
                        parts.reverse();
                    }
                    lane.complete_data_at(10, parts[0], &mut machine).unwrap();
                    if mode == C220LoadCommitMode::DataBypass {
                        assert_eq!(
                            [machine.xregs()[7], machine.xregs()[8]],
                            if reverse { [0, 29] } else { [17, 0] }
                        );
                    } else {
                        assert_eq!([machine.xregs()[7], machine.xregs()[8]], [u64::MAX; 2]);
                    }
                    assert_eq!(
                        lane.retirement_occupancy(),
                        usize::from(path != C220LsuLoadPath::Cache)
                    );
                    let tick = if early {
                        assert!(lane.retire_next_at(11, &mut machine).unwrap().is_none());
                        12
                    } else {
                        10
                    };
                    parts[1].tick = tick;
                    lane.complete_data_at(tick, parts[1], &mut machine).unwrap();
                    assert!(lane.complete_data_at(tick, parts[1], &mut machine).is_err());
                    let Some(C220LsuRetirement::Load(done)) =
                        lane.retire_next_at(tick + 1, &mut machine).unwrap()
                    else {
                        panic!("one pair retirement");
                    };
                    assert_eq!([done.data.value, done.data.second_value.unwrap()], [17, 29]);
                    assert_eq!(
                        [done.register_value, done.second_register_value.unwrap()],
                        [17, 29]
                    );
                    assert!(done.responses.iter().all(Option::is_some));
                    assert_eq!(lane.pending_count(), 0);
                    assert!(
                        lane.retire_next_at(tick + 2, &mut machine)
                            .unwrap()
                            .is_none()
                    );
                    assert_eq!(lane.retirement_occupancy(), 0);
                }
            }
        }
    }
}
