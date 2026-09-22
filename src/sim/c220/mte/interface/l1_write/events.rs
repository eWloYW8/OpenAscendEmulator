use super::{
    C220MteL1WriteAcknowledgment, C220MteL1WriteError, C220MteL1WriteInterface, C220MteL1WritePort,
    C220MteL1WriteRequest, C220MteL1WriteSend,
};
use crate::sim::common::event::{EventDispatcher, EventId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220MteL1WriteCallback {
    InputReady(C220MteL1WritePort),
    ResponseReady,
    AcknowledgmentReady,
    Send,
    ReceiveResponse,
    Retire,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MteL1WriteEventInputs {
    pub request_ready: bool,
    pub response: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220MteL1WriteEventOutcome {
    Readiness,
    Sent(C220MteL1WriteSend),
    Response(Option<C220MteL1WriteRequest>),
    Acknowledged(Option<C220MteL1WriteAcknowledgment>),
}

/// Four input queues share one sender. Response acceptance and acknowledgment
/// retirement have independent callbacks. The owner forwards each sent request
/// immediately and removes a transport response only after successful acceptance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MteL1WriteEvents {
    input_valid: [EventId; 4],
    response_valid: EventId,
    acknowledgment_valid: EventId,
}

impl C220MteL1WriteEvents {
    pub fn register<T: Copy>(
        events: &mut EventDispatcher<T>,
        clock: EventId,
        tag: impl Fn(C220MteL1WriteCallback) -> T,
    ) -> Self {
        let input_valid = std::array::from_fn(|_| events.add_event());
        let response_valid = events.add_event();
        let acknowledgment_valid = events.add_event();
        let send = events.add_process(tag(C220MteL1WriteCallback::Send), false);
        for port in C220MteL1WritePort::ALL {
            let probe = events.add_process(tag(C220MteL1WriteCallback::InputReady(port)), false);
            events.subscribe(clock, probe);
            events.subscribe(input_valid[port as usize], send);
        }
        for (probe, callback, valid) in [
            (
                C220MteL1WriteCallback::ResponseReady,
                C220MteL1WriteCallback::ReceiveResponse,
                response_valid,
            ),
            (
                C220MteL1WriteCallback::AcknowledgmentReady,
                C220MteL1WriteCallback::Retire,
                acknowledgment_valid,
            ),
        ] {
            let probe = events.add_process(tag(probe), false);
            events.subscribe(clock, probe);
            let process = events.add_process(tag(callback), false);
            events.subscribe(valid, process);
        }
        Self {
            input_valid,
            response_valid,
            acknowledgment_valid,
        }
    }

    pub fn handle<T: Copy>(
        &self,
        callback: C220MteL1WriteCallback,
        events: &mut EventDispatcher<T>,
        interface: &mut C220MteL1WriteInterface,
        inputs: C220MteL1WriteEventInputs,
    ) -> Result<C220MteL1WriteEventOutcome, C220MteL1WriteError> {
        let tick = events.tick();
        let (valid, ready) = match callback {
            C220MteL1WriteCallback::InputReady(port) => (
                self.input_valid[port as usize],
                interface.queue(port).front().map(|head| head.ready_tick),
            ),
            C220MteL1WriteCallback::ResponseReady => {
                (self.response_valid, inputs.response.map(|_| tick))
            }
            C220MteL1WriteCallback::AcknowledgmentReady => (
                self.acknowledgment_valid,
                interface
                    .acknowledgments()
                    .front()
                    .map(|head| head.ready_tick),
            ),
            C220MteL1WriteCallback::Send => {
                return interface
                    .send(tick, inputs.request_ready)
                    .map(C220MteL1WriteEventOutcome::Sent);
            }
            C220MteL1WriteCallback::ReceiveResponse => {
                return inputs
                    .response
                    .map(|id| interface.receive_response(tick, id))
                    .transpose()
                    .map(C220MteL1WriteEventOutcome::Response);
            }
            C220MteL1WriteCallback::Retire => {
                return interface
                    .retire(tick)
                    .map(C220MteL1WriteEventOutcome::Acknowledged);
            }
        };
        if ready.is_some_and(|ready| ready <= tick) {
            events.notify_at(valid, tick);
        }
        Ok(C220MteL1WriteEventOutcome::Readiness)
    }
}
