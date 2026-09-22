use super::{
    C220BiuReadArbitration, C220BiuReadError, C220BiuReadFrontend, C220BiuReadInput,
    C220BiuReadSend,
};
use crate::sim::common::event::{EventDispatcher, EventId, ProcessId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220BiuReadCallback {
    InputReady,
    RequestReady,
    Arbitrate,
    Send,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220BiuReadEvent {
    Readiness,
    Arbitration(C220BiuReadArbitration),
    Send(C220BiuReadSend),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220BiuReadEvents {
    input_valid: EventId,
    request_valid: EventId,
    input_probe: ProcessId,
    request_probe: ProcessId,
}

impl C220BiuReadEvents {
    pub fn register<T: Copy>(
        events: &mut EventDispatcher<T>,
        clock: EventId,
        tag: impl Fn(C220BiuReadCallback) -> T,
    ) -> Self {
        let input_valid = events.add_event();
        let request_valid = events.add_event();
        let input_probe = events.add_process(tag(C220BiuReadCallback::InputReady), false);
        let request_probe = events.add_process(tag(C220BiuReadCallback::RequestReady), false);
        events.subscribe(clock, input_probe);
        events.subscribe(clock, request_probe);
        events.set_process_enabled(input_probe, false);
        events.set_process_enabled(request_probe, false);
        let arbitrate = events.add_process(tag(C220BiuReadCallback::Arbitrate), false);
        let send = events.add_process(tag(C220BiuReadCallback::Send), false);
        events.subscribe(input_valid, arbitrate);
        events.subscribe(request_valid, send);
        Self {
            input_valid,
            request_valid,
            input_probe,
            request_probe,
        }
    }

    pub fn push<T: Copy>(
        &self,
        events: &mut EventDispatcher<T>,
        frontend: &mut C220BiuReadFrontend,
        input: C220BiuReadInput,
    ) -> Result<bool, C220BiuReadError> {
        let accepted = frontend.push(events.tick(), input)?;
        if accepted {
            events.set_process_enabled(self.input_probe, true);
        }
        Ok(accepted)
    }

    pub fn handle<T: Copy>(
        &self,
        callback: C220BiuReadCallback,
        events: &mut EventDispatcher<T>,
        frontend: &mut C220BiuReadFrontend,
        transport_ready: bool,
    ) -> Result<C220BiuReadEvent, C220BiuReadError> {
        let tick = events.tick();
        match callback {
            C220BiuReadCallback::InputReady | C220BiuReadCallback::RequestReady => {
                let (ready, valid, probe) = if callback == C220BiuReadCallback::InputReady {
                    (
                        frontend.input_ready_tick(),
                        self.input_valid,
                        self.input_probe,
                    )
                } else {
                    (
                        frontend.request_ready_tick(),
                        self.request_valid,
                        self.request_probe,
                    )
                };
                match ready {
                    None => events.set_process_enabled(probe, false),
                    Some(ready) if ready <= tick => {
                        events.notify_at(valid, tick);
                    }
                    Some(_) => {}
                }
                Ok(C220BiuReadEvent::Readiness)
            }
            C220BiuReadCallback::Arbitrate => {
                let outcome = frontend.arbitrate(tick)?;
                if outcome.selected.is_some() {
                    events.set_process_enabled(self.request_probe, true);
                }
                Ok(C220BiuReadEvent::Arbitration(outcome))
            }
            C220BiuReadCallback::Send => frontend
                .send(tick, transport_ready)
                .map(C220BiuReadEvent::Send),
        }
    }
}
