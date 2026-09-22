use super::*;

#[test]
fn bank_conflicts_cache_tags_and_response_backpressure() {
    let geometry = C220L1Geometry::new(32, 4, 2, 12).unwrap();
    let mut arbiter = C220L1Arbiter::new(geometry);
    let pair = C220L1Access {
        address: 32,
        bytes: 64,
    };
    let alias = C220L1Access {
        address: 160,
        bytes: 32,
    };
    assert_eq!(
        arbiter.write_bank_mask(C220L1Access {
            address: 31,
            bytes: 2
        }),
        1
    );
    assert_eq!(
        arbiter.write_bank_mask(C220L1Access {
            address: 1_u64 << 32,
            bytes: 32
        }),
        1
    );
    assert_eq!(
        arbiter.write_bank_mask(C220L1Access {
            address: 4096,
            bytes: 32
        }),
        16
    );
    assert_eq!(
        arbiter.write_bank_mask(C220L1Access {
            address: 0,
            bytes: u32::MAX
        }),
        0
    );
    assert_eq!(
        arbiter.arbitrate([None, None, Some(pair)])[2]
            .unwrap()
            .bank_mask,
        6
    );
    assert_eq!(arbiter.cache_tags(), [None, Some(33), Some(65), None]);
    assert_eq!(arbiter.read_bank_mask(pair), 0);
    let decisions = arbiter.arbitrate([Some(alias), None, Some(pair)]);
    assert!(decisions[0].unwrap().granted);
    assert!(decisions[2].unwrap().granted);
    assert_eq!(decisions[2].unwrap().bank_mask, 0);
    let decisions = arbiter.arbitrate([Some(pair), Some(pair), Some(pair)]);
    assert!(decisions[0].unwrap().granted);
    assert!(!decisions[1].unwrap().granted);
    assert!(!decisions[2].unwrap().granted);
    assert_eq!(arbiter.read_bank_mask(pair), 6);

    let grouped = C220L1Access {
        address: 4096,
        bytes: 32,
    };
    arbiter.arbitrate([None, None, Some(grouped)]);
    assert_eq!(arbiter.read_bank_mask(grouped), 16);

    let mut pipeline = C220L1Pipeline::new(geometry);
    let request = C220L1Request {
        id: 7,
        access: pair,
    };
    pipeline
        .step(1, [None, None, Some(request)], [true; 3])
        .unwrap();
    assert_eq!(pipeline.pending(C220L1Port::MteRead)[0].ready_tick, 9);
    let other = C220L1Request {
        id: 8,
        access: alias,
    };
    let cycle = pipeline
        .step(2, [Some(other), Some(other), None], [true; 3])
        .unwrap();
    assert!(cycle.decisions[0].unwrap().granted);
    assert!(!cycle.decisions[1].unwrap().granted);
    assert!(pipeline.pending(C220L1Port::MteWrite).is_empty());
    let cycle = pipeline.step(3, [None; 3], [true; 3]).unwrap();
    assert_eq!(cycle.responses[0].unwrap().request.id, 8);
    assert!(cycle.responses[2].is_none());
    let cycle = pipeline
        .step(9, [None, None, Some(other)], [false; 3])
        .unwrap();
    assert!(cycle.responses[2].is_none());
    assert_eq!(pipeline.pending(C220L1Port::MteRead).len(), 2);
    let cycle = pipeline.step(17, [None; 3], [true; 3]).unwrap();
    assert_eq!(cycle.responses[2].unwrap().request.id, 7);
    let cycle = pipeline.step(18, [None; 3], [true; 3]).unwrap();
    assert_eq!(cycle.responses[2].unwrap().request.id, 8);
    let prior = pipeline.clone();
    assert!(matches!(
        pipeline.step(18, [None; 3], [true; 3]),
        Err(C220L1Error::RepeatedCallback { .. })
    ));
    assert_eq!(pipeline, prior);
    assert_eq!(
        pipeline.step(u64::MAX, [None, None, Some(request)], [true; 3]),
        Err(C220L1Error::TimeOverflow)
    );
    assert_eq!(pipeline, prior);
    assert_eq!(
        C220L1Geometry::new(0, 4, 2, 12),
        Err(C220L1Error::UnsupportedGeometry)
    );

    for tick in [1, u64::from(u32::MAX) + 1] {
        let mut pipeline = C220L1Pipeline::new(geometry);
        let mut heads = [Some(request); 3];
        let write = pipeline
            .receive(tick, C220L1Receiver::Write, heads)
            .unwrap();
        assert_eq!(write.accepted, [true, tick > u64::from(u32::MAX), false]);
        for (head, accepted) in heads.iter_mut().zip(write.accepted) {
            if accepted {
                *head = None;
            }
        }
        let read = pipeline.receive(tick, C220L1Receiver::Read, heads).unwrap();
        assert_eq!(read.accepted[2], tick > u64::from(u32::MAX));
    }
    let mut pipeline = C220L1Pipeline::new(geometry);
    assert_eq!(
        pipeline
            .receive(0, C220L1Receiver::Read, [None, None, Some(request)])
            .unwrap()
            .accepted,
        [false; 3]
    );
}
