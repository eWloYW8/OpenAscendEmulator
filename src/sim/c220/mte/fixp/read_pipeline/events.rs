use super::{
    C220FixpReadPipeline, C220FixpReadPipelineError, C220FixpReadProgress, C220FixpReadStream,
};
use crate::sim::c220::mte::generator::{C220MteGeneratorCallback, GeneratorEvents};
use crate::sim::c220::mte::interface::C220MteL0cReadInterface;
use crate::sim::common::event::{EventDispatcher, EventId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220FixpReadEventOutcome {
    Readiness,
    Generated(C220FixpReadProgress),
    Sent(C220FixpReadProgress),
}

/// Read generation callbacks on the core-owned clock. The L0C interface has
/// its own callbacks; enqueueing here does not advance memory or retire FIX.
/// Use this binding or direct pipeline callbacks, never both for one engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FixpReadEvents {
    queues: GeneratorEvents,
}

impl C220FixpReadEvents {
    pub fn register<T: Copy>(
        events: &mut EventDispatcher<T>,
        clock: EventId,
        tag: impl Fn(C220MteGeneratorCallback) -> T,
    ) -> Self {
        Self {
            queues: GeneratorEvents::register(events, clock, tag),
        }
    }

    pub fn submit<T: Copy>(
        &self,
        events: &mut EventDispatcher<T>,
        pipeline: &mut C220FixpReadPipeline,
        packets: impl Into<C220FixpReadStream>,
    ) -> Result<bool, C220FixpReadPipelineError> {
        let submitted = pipeline.submit(events.tick(), packets)?;
        if submitted {
            self.queues.arm_instruction(events);
        }
        Ok(submitted)
    }

    pub fn handle<T: Copy>(
        &self,
        callback: C220MteGeneratorCallback,
        events: &mut EventDispatcher<T>,
        pipeline: &mut C220FixpReadPipeline,
        input: &mut C220MteL0cReadInterface,
        hardware_sync_blocked: bool,
    ) -> Result<C220FixpReadEventOutcome, C220FixpReadPipelineError> {
        match callback {
            C220MteGeneratorCallback::InstructionReady => {
                self.queues
                    .probe_instruction(events, pipeline.generated_ready_tick());
                Ok(C220FixpReadEventOutcome::Readiness)
            }
            C220MteGeneratorCallback::GeneratedReady => {
                self.queues.probe_generated(
                    events,
                    pipeline
                        .dispatch_queue()
                        .front()
                        .map(|head| head.ready_tick),
                );
                Ok(C220FixpReadEventOutcome::Readiness)
            }
            C220MteGeneratorCallback::Generate => {
                let generated = pipeline.generate(events.tick())?;
                if matches!(generated, C220FixpReadProgress::Advanced(_)) {
                    self.queues.arm_generated(events);
                }
                Ok(C220FixpReadEventOutcome::Generated(generated))
            }
            C220MteGeneratorCallback::Send => pipeline
                .send(events.tick(), input, hardware_sync_blocked)
                .map(C220FixpReadEventOutcome::Sent),
        }
    }
}
