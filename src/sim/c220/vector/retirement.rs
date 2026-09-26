use super::runtime::VectorEngine;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220VectorFence {
    pub instruction_id: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220VectorRetirement {
    pub instruction_id: u64,
    pub pc: u64,
    pub word: u32,
    pub dispatch_tick: u64,
    pub retirement_tick: u64,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct PendingVectorInstruction {
    pub instruction_id: u64,
    pc: u64,
    word: u32,
    dispatch_tick: u64,
    instruction_group: Option<u64>,
    vmsu_generation: Option<u64>,
}

impl VectorEngine {
    pub(in crate::sim::c220) fn outstanding_instructions(&self) -> usize {
        self.pending_instructions.len()
    }

    pub(super) fn record_dispatch(&mut self, tick: u64, id: u64, pc: u64, word: u32) {
        self.pending_instructions
            .push_back(PendingVectorInstruction {
                instruction_id: id,
                pc,
                word,
                dispatch_tick: tick,
                instruction_group: self.pipeline.instruction_fence(),
                vmsu_generation: self.vmsu.instruction_fence(),
            });
        self.last_dispatched_id = Some(id);
    }

    fn backend_retirement_tick(&self, pending: &PendingVectorInstruction) -> Option<u64> {
        pending
            .instruction_group
            .and_then(|group| self.pipeline.fence_retirement_tick(group))
            .into_iter()
            .chain(
                pending
                    .vmsu_generation
                    .and_then(|generation| self.vmsu.fence_retirement_tick(generation)),
            )
            .max()
    }

    pub(in crate::sim::c220) fn instruction_fence(&self) -> C220VectorFence {
        C220VectorFence {
            instruction_id: self
                .pending_instructions
                .back()
                .map(|entry| entry.instruction_id),
        }
    }

    pub(in crate::sim::c220) fn fence_retirement_tick(
        &self,
        fence: C220VectorFence,
    ) -> Option<u64> {
        let id = fence.instruction_id?;
        let mut previous = self.last_retired_tick;
        let mut result = None;
        for entry in self
            .pending_instructions
            .iter()
            .take_while(|entry| entry.instruction_id <= id)
        {
            let ready = self
                .backend_retirement_tick(entry)
                .unwrap_or(entry.dispatch_tick);
            let ready = ready.max(previous.map_or(0, |tick| tick.saturating_add(1)));
            let ready = ready.max(self.observed_tick.map_or(0, |tick| tick.saturating_add(1)));
            result = Some(ready);
            previous = result;
        }
        result
    }

    pub(super) fn next_retirement_tick(&self) -> Option<u64> {
        let first = self.pending_instructions.front()?;
        self.fence_retirement_tick(C220VectorFence {
            instruction_id: Some(first.instruction_id),
        })
    }

    pub(super) fn retire_ready(&mut self, tick: u64) {
        let Some(&entry) = self.pending_instructions.front() else {
            return;
        };
        if entry
            .instruction_group
            .is_some_and(|group| !self.pipeline.fence_is_retired(group))
            || entry
                .vmsu_generation
                .is_some_and(|generation| !self.vmsu.fence_is_retired(generation))
            || self
                .last_retired_tick
                .is_some_and(|previous| previous >= tick)
            || entry.dispatch_tick > tick
        {
            return;
        }
        self.pending_instructions.pop_front();
        self.last_retired_tick = Some(tick);
        self.retirements.push(C220VectorRetirement {
            instruction_id: entry.instruction_id,
            pc: entry.pc,
            word: entry.word,
            dispatch_tick: entry.dispatch_tick,
            retirement_tick: tick,
        });
    }
}
