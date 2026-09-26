use std::collections::BTreeMap;

use crate::isa::flow::FlagStep;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220PipelineEvent {
    pub instruction_id: u64,
    pub step: FlagStep,
    pub received_tick: u64,
    pub published_tick: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220DeferredPipelineEvent {
    pub instruction_id: u64,
    pub step: FlagStep,
    pub received_tick: u64,
    pub predecessor: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220PipelineEventConsumption {
    pub instruction_id: u64,
    pub step: FlagStep,
    pub tick: u64,
    pub event: C220PipelineEvent,
}

/// Ordinary pipeline events are independent of hardware-flag counters.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct C220PipelineEvents {
    ready: BTreeMap<(u8, u8), Vec<C220PipelineEvent>>,
    deferred: BTreeMap<(u8, u64), Vec<C220DeferredPipelineEvent>>,
    publications: Vec<C220PipelineEvent>,
    consumptions: Vec<C220PipelineEventConsumption>,
}

impl C220PipelineEvents {
    pub fn last_publications(&self) -> &[C220PipelineEvent] {
        &self.publications
    }

    pub fn last_consumptions(&self) -> &[C220PipelineEventConsumption] {
        &self.consumptions
    }

    pub(in crate::sim::c220) fn begin_advance(&mut self) {
        self.publications.clear();
        self.consumptions.clear();
    }

    pub fn ready(&self, source: u8, destination: u8) -> &[C220PipelineEvent] {
        self.ready
            .get(&(source, destination))
            .map_or(&[], Vec::as_slice)
    }

    pub fn pending(&self) -> impl Iterator<Item = C220DeferredPipelineEvent> + '_ {
        self.deferred.values().flatten().copied()
    }

    pub(in crate::sim::c220) fn set(
        &mut self,
        instruction_id: u64,
        step: FlagStep,
        predecessor: Option<u64>,
        tick: u64,
    ) {
        if let Some(predecessor) = predecessor {
            self.deferred
                .entry((step.instruction.source_pipe_code, predecessor))
                .or_default()
                .push(C220DeferredPipelineEvent {
                    instruction_id,
                    step,
                    received_tick: tick,
                    predecessor,
                });
        } else {
            self.publish(C220PipelineEvent {
                instruction_id,
                step,
                received_tick: tick,
                published_tick: tick,
            });
        }
    }

    fn publish(&mut self, event: C220PipelineEvent) {
        self.publications.push(event);
        self.ready
            .entry((
                event.step.instruction.source_pipe_code,
                event.step.instruction.trigger_pipe_code,
            ))
            .or_default()
            .push(event);
    }

    pub(in crate::sim::c220) fn retire(&mut self, source: u8, instruction_id: u64, tick: u64) {
        if let Some(events) = self.deferred.remove(&(source, instruction_id)) {
            for event in events {
                self.publish(C220PipelineEvent {
                    instruction_id: event.instruction_id,
                    step: event.step,
                    received_tick: event.received_tick,
                    published_tick: tick,
                });
            }
        }
    }

    pub(in crate::sim::c220) fn consume(
        &mut self,
        instruction_id: u64,
        step: FlagStep,
        tick: u64,
    ) -> Option<C220PipelineEvent> {
        let route = (
            step.instruction.source_pipe_code,
            step.instruction.trigger_pipe_code,
        );
        let events = self.ready.get_mut(&route)?;
        let index = events
            .iter()
            .position(|event| event.step.flag_id == step.flag_id)?;
        let event = events.remove(index);
        if events.is_empty() {
            self.ready.remove(&route);
        }
        self.consumptions.push(C220PipelineEventConsumption {
            instruction_id,
            step,
            tick,
            event,
        });
        Some(event)
    }
}
