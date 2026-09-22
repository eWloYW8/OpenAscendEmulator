use super::{C220UbWriteAcknowledgment, C220UbWriteError, C220UbWriteInterface, C220UbWriteSend};
use crate::sim::common::event::{EventDispatcher, EventId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220UbWriteCallback {
    Probe,
    Send,
    Retire,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220UbWriteEvent {
    Readiness,
    Sent(C220UbWriteSend),
    Acknowledged(Option<C220UbWriteAcknowledgment>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220UbWriteEvents {
    input_valid: EventId,
    acknowledgment_valid: EventId,
}

impl C220UbWriteEvents {
    pub fn register<T: Copy>(
        events: &mut EventDispatcher<T>,
        clock: EventId,
        tag: impl Fn(C220UbWriteCallback) -> T,
    ) -> Self {
        let probe = events.add_process(tag(C220UbWriteCallback::Probe), false);
        events.subscribe(clock, probe);
        let input_valid = events.add_event();
        let send = events.add_process(tag(C220UbWriteCallback::Send), false);
        events.subscribe(input_valid, send);
        let acknowledgment_valid = events.add_event();
        let retire = events.add_process(tag(C220UbWriteCallback::Retire), false);
        events.subscribe(acknowledgment_valid, retire);
        Self {
            input_valid,
            acknowledgment_valid,
        }
    }

    pub fn handle<T: Copy>(
        &self,
        callback: C220UbWriteCallback,
        events: &mut EventDispatcher<T>,
        interface: &mut C220UbWriteInterface,
    ) -> Result<C220UbWriteEvent, C220UbWriteError> {
        let tick = events.tick();
        match callback {
            C220UbWriteCallback::Probe => {
                if interface
                    .inputs()
                    .front()
                    .is_some_and(|head| head.ready_tick <= tick)
                {
                    events.notify_at(self.input_valid, tick);
                }
                if interface
                    .acknowledgments()
                    .front()
                    .is_some_and(|head| head.ready_tick <= tick)
                {
                    events.notify_at(self.acknowledgment_valid, tick);
                }
                Ok(C220UbWriteEvent::Readiness)
            }
            C220UbWriteCallback::Send => interface.send(tick).map(C220UbWriteEvent::Sent),
            C220UbWriteCallback::Retire => {
                interface.retire(tick).map(C220UbWriteEvent::Acknowledged)
            }
        }
    }
}
