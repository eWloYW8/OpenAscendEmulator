use super::{
    C220MteReadFrontend, C220MteReadFrontendError, C220MteReadGenerated, C220MteReadIssue,
    C220MteReadSend, C220MteReadTransfer, C220MteReadUop,
};
use crate::sim::c220::mte::generator::{C220MteGeneratorCallback, GeneratorEvents};
use crate::sim::c220::mte::interface::C220MteL1Interface;
use crate::sim::common::event::{EventDispatcher, EventId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220MteReadEventOutcome {
    Readiness,
    Generated(Option<C220MteReadGenerated>),
    Sent(C220MteReadSend),
}

/// One binding per read generator, all feeding the shared L1 interface.
/// Registration order determines producer callback order on the shared clock.
/// Do not mix direct frontend callbacks with this binding for the same engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MteReadEvents {
    queues: GeneratorEvents,
}

impl C220MteReadEvents {
    pub fn register<T: Copy>(
        events: &mut EventDispatcher<T>,
        clock: EventId,
        tag: impl Fn(C220MteGeneratorCallback) -> T,
    ) -> Self {
        Self {
            queues: GeneratorEvents::register(events, clock, tag),
        }
    }

    pub fn issue<T: Copy>(
        &self,
        events: &mut EventDispatcher<T>,
        frontend: &mut C220MteReadFrontend,
        instruction_id: u64,
        transfer: C220MteReadTransfer,
    ) -> Result<C220MteReadIssue, C220MteReadFrontendError> {
        let issue = frontend.issue(events.tick(), instruction_id, transfer)?;
        if !issue.completion_ready {
            self.queues.arm_instruction(events);
        }
        Ok(issue)
    }

    pub fn handle<T: Copy>(
        &self,
        callback: C220MteGeneratorCallback,
        events: &mut EventDispatcher<T>,
        frontend: &mut C220MteReadFrontend,
        hardware_sync_blocked: bool,
        interface: &mut C220MteL1Interface<C220MteReadUop>,
    ) -> Result<C220MteReadEventOutcome, C220MteReadFrontendError> {
        self.handle_mapped(
            callback,
            events,
            frontend,
            hardware_sync_blocked,
            interface,
            |payload| payload,
        )
    }

    pub fn handle_mapped<T: Copy, U: Copy>(
        &self,
        callback: C220MteGeneratorCallback,
        events: &mut EventDispatcher<T>,
        frontend: &mut C220MteReadFrontend,
        hardware_sync_blocked: bool,
        interface: &mut C220MteL1Interface<U>,
        map: impl FnOnce(C220MteReadUop) -> U,
    ) -> Result<C220MteReadEventOutcome, C220MteReadFrontendError> {
        let tick = events.tick();
        match callback {
            C220MteGeneratorCallback::InstructionReady => {
                self.queues
                    .probe_instruction(events, frontend.instruction_ready_tick());
                Ok(C220MteReadEventOutcome::Readiness)
            }
            C220MteGeneratorCallback::GeneratedReady => {
                self.queues.probe_generated(
                    events,
                    frontend.generated().front().map(|head| head.ready_tick),
                );
                Ok(C220MteReadEventOutcome::Readiness)
            }
            C220MteGeneratorCallback::Generate => {
                let generated = frontend.generate(tick)?;
                if generated.is_some() {
                    self.queues.arm_generated(events);
                }
                Ok(C220MteReadEventOutcome::Generated(generated))
            }
            C220MteGeneratorCallback::Send => frontend
                .send_mapped(tick, hardware_sync_blocked, interface, map)
                .map(C220MteReadEventOutcome::Sent),
        }
    }
}
