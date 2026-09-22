use crate::sim::common::event::{EventDispatcher, EventId, ProcessId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220MteGeneratorCallback {
    InstructionReady,
    GeneratedReady,
    Generate,
    Send,
}

/// Clock probes and consumer events shared by C220 MTE generation engines.
/// Each owner supplies the actual queue heads and executes its own operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::sim::c220::mte) struct GeneratorEvents {
    instruction_valid: EventId,
    generated_valid: EventId,
    instruction_probe: ProcessId,
    generated_probe: ProcessId,
}

impl GeneratorEvents {
    pub(super) fn register<T: Copy>(
        events: &mut EventDispatcher<T>,
        clock: EventId,
        tag: impl Fn(C220MteGeneratorCallback) -> T,
    ) -> Self {
        use C220MteGeneratorCallback::*;
        let instruction_valid = events.add_event();
        let generated_valid = events.add_event();
        let instruction_probe = events.add_process(tag(InstructionReady), false);
        let generated_probe = events.add_process(tag(GeneratedReady), false);
        events.subscribe(clock, generated_probe);
        events.subscribe(clock, instruction_probe);
        events.set_process_enabled(instruction_probe, false);
        events.set_process_enabled(generated_probe, false);
        let generate = events.add_process(tag(Generate), false);
        let send = events.add_process(tag(Send), false);
        events.subscribe(instruction_valid, generate);
        events.subscribe(generated_valid, send);
        Self {
            instruction_valid,
            generated_valid,
            instruction_probe,
            generated_probe,
        }
    }

    pub(super) fn arm_instruction<T: Copy>(&self, events: &mut EventDispatcher<T>) {
        events.set_process_enabled(self.instruction_probe, true);
    }

    pub(super) fn arm_generated<T: Copy>(&self, events: &mut EventDispatcher<T>) {
        events.set_process_enabled(self.generated_probe, true);
    }

    pub(super) fn probe_instruction<T: Copy>(
        &self,
        events: &mut EventDispatcher<T>,
        head: Option<u64>,
    ) {
        Self::probe(events, head, self.instruction_probe, self.instruction_valid);
    }

    pub(super) fn probe_generated<T: Copy>(
        &self,
        events: &mut EventDispatcher<T>,
        head: Option<u64>,
    ) {
        Self::probe(events, head, self.generated_probe, self.generated_valid);
    }

    fn probe<T: Copy>(
        events: &mut EventDispatcher<T>,
        head: Option<u64>,
        probe: ProcessId,
        valid: EventId,
    ) {
        match head {
            None => events.set_process_enabled(probe, false),
            Some(ready) if ready <= events.tick() => {
                events.notify_at(valid, events.tick());
            }
            Some(_) => {}
        }
    }
}
