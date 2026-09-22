use super::{
    C220L1Port, C220L1Receiver, C220L1RequestCycle, C220L1ResponseCycle, C220L1Transport,
    C220L1TransportError,
};
use crate::sim::common::event::{EventDispatcher, EventId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220L1Callback {
    RequestReady(C220L1Port),
    PendingReady(C220L1Port),
    Receive(C220L1Receiver),
    Respond(C220L1Receiver),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum C220L1EventOutcome {
    Readiness,
    Received(C220L1RequestCycle),
    Responded(C220L1ResponseCycle),
}

/// One binding for the shared L1 service. Each queue probes its own age,
/// while both write ports wake the same readout/response processes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220L1Events {
    request_valid: [EventId; 3],
    pending_valid: [EventId; 3],
}

impl C220L1Events {
    pub fn register<T: Copy>(
        events: &mut EventDispatcher<T>,
        clock: EventId,
        tag: impl Fn(C220L1Callback) -> T,
    ) -> Self {
        let request_valid = std::array::from_fn(|_| events.add_event());
        let pending_valid = std::array::from_fn(|_| events.add_event());
        let receive = C220L1Receiver::ALL
            .map(|receiver| events.add_process(tag(C220L1Callback::Receive(receiver)), false));
        let respond = C220L1Receiver::ALL
            .map(|receiver| events.add_process(tag(C220L1Callback::Respond(receiver)), false));
        for port in [
            C220L1Port::MteRead,
            C220L1Port::FixpWrite,
            C220L1Port::MteWrite,
        ] {
            let receiver = if port == C220L1Port::MteRead {
                C220L1Receiver::Read
            } else {
                C220L1Receiver::Write
            };
            let input = events.add_process(tag(C220L1Callback::RequestReady(port)), false);
            events.subscribe(clock, input);
            events.subscribe(request_valid[port as usize], receive[receiver as usize]);
            let pending = events.add_process(tag(C220L1Callback::PendingReady(port)), false);
            events.subscribe(clock, pending);
            events.subscribe(pending_valid[port as usize], respond[receiver as usize]);
        }
        Self {
            request_valid,
            pending_valid,
        }
    }

    pub fn handle<T: Copy>(
        &self,
        callback: C220L1Callback,
        events: &mut EventDispatcher<T>,
        transport: &mut C220L1Transport,
    ) -> Result<C220L1EventOutcome, C220L1TransportError> {
        let tick = events.tick();
        let (event, ready) = match callback {
            C220L1Callback::RequestReady(port) => (
                self.request_valid[port as usize],
                transport.requests(port).front().map(|head| head.ready_tick),
            ),
            C220L1Callback::PendingReady(port) => (
                self.pending_valid[port as usize],
                transport
                    .service()
                    .pending(port)
                    .front()
                    .map(|head| head.accepted_tick + 1),
            ),
            C220L1Callback::Receive(receiver) => {
                return transport
                    .receive_requests(tick, receiver)
                    .map(C220L1EventOutcome::Received);
            }
            C220L1Callback::Respond(receiver) => {
                return transport
                    .send_responses(tick, receiver)
                    .map(C220L1EventOutcome::Responded);
            }
        };
        if ready.is_some_and(|ready| ready <= tick) {
            events.notify_at(event, tick);
        }
        Ok(C220L1EventOutcome::Readiness)
    }
}
