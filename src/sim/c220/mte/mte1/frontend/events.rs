use super::{
    C220Mte1ReadFrontend, C220Mte1ReadFrontendError, C220Mte1ReadGenerated, C220Mte1ReadIssue,
    C220Mte1ReadSend, C220Mte1ReadTransfer, C220Mte1ReadUop,
};
use crate::sim::c220::mte::generator::{C220MteGeneratorCallback, GeneratorEvents};
use crate::sim::c220::mte::interface::C220MteL1Interface;
use crate::sim::common::event::{EventDispatcher, EventId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220Mte1ReadEventOutcome {
    Readiness,
    Generated(Option<C220Mte1ReadGenerated>),
    Sent(C220Mte1ReadSend),
}

/// One binding per LOAD2D or BT generator; both may feed the same L1 interface.
/// Registration order determines producer callback order on the shared clock.
/// Do not mix direct frontend callbacks with this binding for the same engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Mte1ReadEvents {
    queues: GeneratorEvents,
}

impl C220Mte1ReadEvents {
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
        frontend: &mut C220Mte1ReadFrontend,
        instruction_id: u64,
        transfer: C220Mte1ReadTransfer,
    ) -> Result<C220Mte1ReadIssue, C220Mte1ReadFrontendError> {
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
        frontend: &mut C220Mte1ReadFrontend,
        hardware_sync_blocked: bool,
        interface: &mut C220MteL1Interface<C220Mte1ReadUop>,
    ) -> Result<C220Mte1ReadEventOutcome, C220Mte1ReadFrontendError> {
        let tick = events.tick();
        match callback {
            C220MteGeneratorCallback::InstructionReady => {
                self.queues
                    .probe_instruction(events, frontend.instruction_ready_tick());
                Ok(C220Mte1ReadEventOutcome::Readiness)
            }
            C220MteGeneratorCallback::GeneratedReady => {
                self.queues.probe_generated(
                    events,
                    frontend.generated().front().map(|head| head.ready_tick),
                );
                Ok(C220Mte1ReadEventOutcome::Readiness)
            }
            C220MteGeneratorCallback::Generate => {
                let generated = frontend.generate(tick)?;
                if generated.is_some() {
                    self.queues.arm_generated(events);
                }
                Ok(C220Mte1ReadEventOutcome::Generated(generated))
            }
            C220MteGeneratorCallback::Send => frontend
                .send(tick, hardware_sync_blocked, interface)
                .map(C220Mte1ReadEventOutcome::Sent),
        }
    }
}
