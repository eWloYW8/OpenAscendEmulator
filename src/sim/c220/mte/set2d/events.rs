use super::{
    C220Set2dFill, C220Set2dFrontend, C220Set2dFrontendError, C220Set2dGates, C220Set2dGenerated,
    C220Set2dIssue, C220Set2dOutputs, C220Set2dSend,
};
use crate::sim::c220::mte::generator::{C220MteGeneratorCallback, GeneratorEvents};
use crate::sim::common::event::{EventDispatcher, EventId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220Set2dEventOutcome {
    Readiness,
    Generated(Option<C220Set2dGenerated>),
    Sent(C220Set2dSend),
}

/// Binds a SET_2D engine to a shared clock and event dispatcher. Readiness
/// checks notify consumers at the back of the current event queue. Output
/// interfaces remain shared with other producers and keep their own callbacks.
/// Do not mix direct frontend callbacks with this binding for the same engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Set2dEvents {
    queues: GeneratorEvents,
}

impl C220Set2dEvents {
    /// `tag` embeds these callbacks in the owner's component/instance routing
    /// enum. Registration preserves the clock's existing subscriber order.
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
        frontend: &mut C220Set2dFrontend,
        instruction_id: u64,
        fill: C220Set2dFill,
    ) -> Result<C220Set2dIssue, C220Set2dFrontendError> {
        let issue = frontend.issue(events.tick(), instruction_id, fill)?;
        if !issue.completion_ready {
            self.queues.arm_instruction(events);
        }
        Ok(issue)
    }

    pub fn handle<T: Copy>(
        &self,
        callback: C220MteGeneratorCallback,
        events: &mut EventDispatcher<T>,
        frontend: &mut C220Set2dFrontend,
        gates: C220Set2dGates,
        outputs: C220Set2dOutputs<'_>,
    ) -> Result<C220Set2dEventOutcome, C220Set2dFrontendError> {
        let tick = events.tick();
        match callback {
            C220MteGeneratorCallback::InstructionReady => {
                self.queues
                    .probe_instruction(events, frontend.instruction_ready_tick());
                Ok(C220Set2dEventOutcome::Readiness)
            }
            C220MteGeneratorCallback::GeneratedReady => {
                self.queues.probe_generated(
                    events,
                    frontend.generated().front().map(|head| head.ready_tick),
                );
                Ok(C220Set2dEventOutcome::Readiness)
            }
            C220MteGeneratorCallback::Generate => {
                let generated = frontend.generate(tick)?;
                if generated.is_some() {
                    self.queues.arm_generated(events);
                }
                Ok(C220Set2dEventOutcome::Generated(generated))
            }
            C220MteGeneratorCallback::Send => frontend
                .send(tick, gates, outputs)
                .map(C220Set2dEventOutcome::Sent),
        }
    }
}
