use super::{
    C220FixpL1WriteEntry, C220FixpL1WriteError, C220FixpL1WriteInterface, C220FixpL1WriteRequest,
    C220FixpL1WriteSend,
};
use crate::sim::c220::memory::l1::{C220L1Port, C220L1Transport};
use crate::sim::common::event::{EventDispatcher, EventId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220FixpL1WriteCallback {
    InputReady,
    ResponseReady,
    AcknowledgmentReady,
    Send,
    Receive,
    Retire,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220FixpL1WriteEvent {
    Readiness,
    Sent(C220FixpL1WriteSend),
    Response(Option<C220FixpL1WriteRequest>),
    Acknowledged(Option<C220FixpL1WriteEntry>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FixpL1WriteEvents {
    valid: [EventId; 3],
}

impl C220FixpL1WriteEvents {
    pub fn register<T: Copy>(
        events: &mut EventDispatcher<T>,
        clock: EventId,
        tag: impl Fn(C220FixpL1WriteCallback) -> T,
    ) -> Self {
        use C220FixpL1WriteCallback::*;
        let valid = [
            (InputReady, Send),
            (ResponseReady, Receive),
            (AcknowledgmentReady, Retire),
        ]
        .map(|(probe, callback)| {
            let valid = events.add_event();
            let probe = events.add_process(tag(probe), false);
            let process = events.add_process(tag(callback), false);
            events.subscribe(clock, probe);
            events.subscribe(valid, process);
            valid
        });
        Self { valid }
    }

    /// Request and response ownership remains in the shared L1 transport.
    /// Readiness notifications enqueue independent, once-per-tick callbacks.
    pub fn handle<T: Copy>(
        &self,
        callback: C220FixpL1WriteCallback,
        events: &mut EventDispatcher<T>,
        interface: &mut C220FixpL1WriteInterface,
        memory: &mut C220L1Transport,
    ) -> Result<C220FixpL1WriteEvent, C220FixpL1WriteError> {
        use C220FixpL1WriteCallback::*;
        let tick = events.tick();
        let (index, ready) = match callback {
            InputReady => (0, interface.input().front().map(|head| head.ready_tick)),
            ResponseReady => (
                1,
                memory
                    .responses(C220L1Port::FixpWrite)
                    .front()
                    .map(|head| head.ready_tick),
            ),
            AcknowledgmentReady => (
                2,
                interface
                    .acknowledgments()
                    .front()
                    .map(|head| head.ready_tick),
            ),
            Send => return interface.send(tick, memory).map(C220FixpL1WriteEvent::Sent),
            Receive => {
                return interface
                    .receive(tick, memory)
                    .map(C220FixpL1WriteEvent::Response);
            }
            Retire => {
                return interface
                    .retire(tick)
                    .map(C220FixpL1WriteEvent::Acknowledged);
            }
        };
        if ready.is_some_and(|ready| ready <= tick) {
            events.notify_at(self.valid[index], tick);
        }
        Ok(C220FixpL1WriteEvent::Readiness)
    }
}
