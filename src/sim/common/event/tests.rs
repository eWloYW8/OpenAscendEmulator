use super::*;

#[test]
fn fifo_notifications_process_suppression_and_cancellation() {
    let mut events = EventDispatcher::new(0);
    let a = events.add_event();
    let b = events.add_event();
    let normal = events.add_process('n', false);
    let repeat = events.add_process('r', true);
    let disabled = events.add_process('d', false);
    events.subscribe(a, normal);
    events.subscribe(a, repeat);
    events.subscribe(a, repeat);
    events.subscribe(b, normal);
    events.subscribe(b, disabled);
    events.set_process_enabled(disabled, false);
    events.notify_at(a, 2);
    events.notify_at(b, 2);
    events.notify_at(a, 2);
    events.notify_at(a, 8);
    assert_eq!(
        events.advance_to(3),
        Err(EventError::PendingEvents { tick: 2 })
    );
    events.advance_to(2).unwrap();
    assert!(!events.triggered(a));
    let mut calls = Vec::new();
    while let Some(call) = events.next_callback() {
        calls.push((call.event, call.callback));
        if calls.len() == 1 {
            assert!(events.triggered(a));
            events.notify_after(b, 0).unwrap();
        }
    }
    assert_eq!(calls, [(a, 'n'), (a, 'r'), (a, 'r')]);
    assert!(events.triggered(b));
    events.notify_at(b, 8);
    events.notify_at(a, 8);
    assert_eq!(events.cancel(a), 2);
    assert_eq!(events.pending_notifications().collect::<Vec<_>>(), [(8, b)]);
    events.advance_to(8).unwrap();
    assert!(!events.triggered(a));
    assert!(!events.notify_at(a, 7));
    events.set_process_enabled(disabled, true);
    assert_eq!(events.next_callback().unwrap().callback, 'n');
    assert_eq!(events.next_callback().unwrap().callback, 'd');
    assert!(events.next_callback().is_none());
    events.notify_at(a, 9);
    events.set_event_enabled(a, false);
    events.advance_to(9).unwrap();
    assert!(events.next_callback().is_none());
    assert!(!events.triggered(a));
    events.advance_to(u64::MAX).unwrap();
    assert_eq!(events.notify_after(a, 1), Err(EventError::TimeOverflow));
    assert_eq!(events.next_event_tick(), None);
}
