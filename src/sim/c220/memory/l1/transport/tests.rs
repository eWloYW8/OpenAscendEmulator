use super::*;
use crate::sim::c220::memory::l1::C220L1Access;

fn request(id: u64, address: u64) -> C220L1Request {
    C220L1Request {
        id,
        access: C220L1Access { address, bytes: 32 },
    }
}

#[test]
fn transport_visibility_shared_write_notifications_and_credit_stalls() {
    use C220L1Port::{FixpWrite, MteRead, MteWrite};
    let geometry = C220L1Geometry::new(32, 4, 1, 0).unwrap();
    let mut link = C220L1Transport::new(geometry);
    link.send_request(0, MteRead, request(0, 0)).unwrap();
    assert_eq!(link.advance(0).unwrap().accepted, [false; 3]);
    link.send_request(1, FixpWrite, request(1, 0)).unwrap();
    let cycle = link.advance(1).unwrap();
    assert!(cycle.decisions[0].unwrap().granted);
    assert!(!cycle.decisions[2].unwrap().granted);
    assert_eq!(cycle.accepted, [false; 3]);
    assert_eq!(link.advance(2).unwrap().accepted, [true, false, false]);
    assert_eq!(link.advance(3).unwrap().accepted, [false, false, true]);
    assert!(link.receive_response(3, FixpWrite).unwrap().is_none());
    assert_eq!(
        link.receive_response(4, FixpWrite)
            .unwrap()
            .unwrap()
            .request
            .id,
        1
    );
    for tick in 4..=11 {
        link.advance(tick).unwrap();
    }
    assert!(link.receive_response(11, MteRead).unwrap().is_none());
    let response = link.receive_response(12, MteRead).unwrap().unwrap();
    assert_eq!((response.accepted_tick, response.ready_tick), (3, 11));
    assert!(link.is_idle());

    let mut link = C220L1Transport::new(geometry);
    link.send_request(0, FixpWrite, request(2, 0)).unwrap();
    link.advance(0).unwrap();
    link.send_request(1, MteWrite, request(3, 32)).unwrap();
    assert_eq!(link.advance(1).unwrap().accepted, [true, true, false]);
    link.advance(2).unwrap();
    assert_eq!(link.responses(MteWrite)[0].payload.accepted_tick, 1);
    assert_eq!(
        link.receive_response(3, FixpWrite)
            .unwrap()
            .unwrap()
            .request
            .id,
        2
    );
    assert_eq!(
        link.receive_response(3, MteWrite)
            .unwrap()
            .unwrap()
            .request
            .id,
        3
    );
    assert!(link.is_idle());

    let mut link = C220L1Transport::new(geometry);
    assert!(link.send_request(0, FixpWrite, request(0, 0)).unwrap());
    assert!(link.send_request(0, FixpWrite, request(1, 32)).unwrap());
    assert!(!link.send_request(0, FixpWrite, request(999, 64)).unwrap());
    link.advance(0).unwrap();
    link.advance(1).unwrap();
    // A send after the service phase can use the newly released credit.
    assert!(link.send_request(1, FixpWrite, request(2, 64)).unwrap());
    link.advance(2).unwrap();
    assert!(link.send_request(2, FixpWrite, request(3, 96)).unwrap());
    for tick in 3..=6 {
        link.advance(tick).unwrap();
    }
    assert_eq!(link.responses(FixpWrite).len(), 2);
    assert_eq!(link.service().pending(FixpWrite).len(), 2);
    let before = link.clone();
    assert!(matches!(
        link.advance(8),
        Err(C220L1TransportError::SkippedCycle { .. })
    ));
    assert_eq!(link, before);
    assert!(matches!(
        link.receive_response(5, FixpWrite),
        Err(C220L1TransportError::TimeReversed { .. })
    ));
    assert_eq!(link, before);
    assert_eq!(
        link.receive_response(7, FixpWrite)
            .unwrap()
            .unwrap()
            .request
            .id,
        0
    );
    let cycle = link.advance(7).unwrap();
    assert_eq!(cycle.responses[0].unwrap().request.id, 2);
    assert_eq!(link.responses(FixpWrite)[1].ready_tick, 8);
    for tick in 8..=11 {
        link.receive_response(tick, FixpWrite).unwrap();
        link.advance(tick).unwrap();
    }
    assert!(link.is_idle());
    let before = link.clone();
    assert_eq!(
        link.send_request(u64::MAX, MteRead, request(99, 0)),
        Err(C220L1TransportError::TimeOverflow)
    );
    assert_eq!(link, before);
    // The service may already have advanced during the issue cycle.
    link.send_request(11, MteRead, request(99, 0)).unwrap();
    assert!(link.advance(12).unwrap().accepted[2]);
}
