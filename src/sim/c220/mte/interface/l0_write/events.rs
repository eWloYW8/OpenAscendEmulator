use super::{
    C220L0WriteAcknowledgment, C220L0WriteError, C220L0WritePipeline, C220L0WritePort,
    C220L0WriteSend,
};
use crate::sim::common::event::{EventDispatcher, EventId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220L0WriteCallback {
    InputReady(C220L0WritePort),
    AcknowledgmentReady,
    Send,
    Retire,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220L0WriteEventOutcome {
    Readiness,
    Sent(C220L0WriteSend),
    Acknowledged(Option<C220L0WriteAcknowledgment>),
}

/// One binding per L0 destination. Input readiness events share one sending
/// process; acknowledgment readiness drives a separate retirement process.
/// Clock probes are harmless on empty queues, so any producer can enqueue
/// directly without owning or modifying the destination's event registration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220L0WriteEvents {
    input_valid: [EventId; 3],
    acknowledgment_valid: EventId,
}

impl C220L0WriteEvents {
    pub fn register<T: Copy>(
        events: &mut EventDispatcher<T>,
        clock: EventId,
        tag: impl Fn(C220L0WriteCallback) -> T,
    ) -> Self {
        let input_valid = std::array::from_fn(|_| events.add_event());
        let acknowledgment_valid = events.add_event();
        let send = events.add_process(tag(C220L0WriteCallback::Send), false);
        for port in [
            C220L0WritePort::Port0,
            C220L0WritePort::Port1,
            C220L0WritePort::Port2,
        ] {
            let probe = events.add_process(tag(C220L0WriteCallback::InputReady(port)), false);
            events.subscribe(clock, probe);
            events.subscribe(input_valid[port as usize], send);
        }
        let probe = events.add_process(tag(C220L0WriteCallback::AcknowledgmentReady), false);
        events.subscribe(clock, probe);
        let retire = events.add_process(tag(C220L0WriteCallback::Retire), false);
        events.subscribe(acknowledgment_valid, retire);
        Self {
            input_valid,
            acknowledgment_valid,
        }
    }

    pub fn handle<T: Copy>(
        &self,
        callback: C220L0WriteCallback,
        events: &mut EventDispatcher<T>,
        pipeline: &mut C220L0WritePipeline,
    ) -> Result<C220L0WriteEventOutcome, C220L0WriteError> {
        let tick = events.tick();
        match callback {
            C220L0WriteCallback::InputReady(port) => {
                if pipeline
                    .queue(port)
                    .front()
                    .is_some_and(|head| head.ready_tick <= tick)
                {
                    events.notify_at(self.input_valid[port as usize], tick);
                }
                Ok(C220L0WriteEventOutcome::Readiness)
            }
            C220L0WriteCallback::AcknowledgmentReady => {
                if pipeline
                    .acknowledgments()
                    .front()
                    .is_some_and(|head| head.ready_tick <= tick)
                {
                    events.notify_at(self.acknowledgment_valid, tick);
                }
                Ok(C220L0WriteEventOutcome::Readiness)
            }
            C220L0WriteCallback::Send => pipeline.send(tick).map(C220L0WriteEventOutcome::Sent),
            C220L0WriteCallback::Retire => pipeline
                .retire(tick)
                .map(C220L0WriteEventOutcome::Acknowledged),
        }
    }
}
