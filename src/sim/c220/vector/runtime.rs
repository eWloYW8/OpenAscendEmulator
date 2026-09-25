use std::collections::{BTreeMap, VecDeque};

use crate::sim::c220::state::C220State;
use crate::sim::c220::vector::ops::compare::C220CompareMask;
use crate::sim::c220::vector::pipeline::{C220VectorPipeline, C220VectorTimingRules};
use crate::sim::c220::vector::timing::C220VectorUopRelease;
use crate::sim::c220::vector::va::C220VaRegisters;
use crate::sim::c220::vector::vmsu::C220VmsuPipeline;

use super::pipeline::{C220VectorAdvanceError, C220VectorPipelineError};
use super::vmsu::C220VmsuError;
use super::{C220VectorInstruction, C220VectorUopError};

pub(in crate::sim::c220) struct VectorEngine {
    pub(in crate::sim::c220) pipeline: C220VectorPipeline,
    pub(in crate::sim::c220) vmsu: C220VmsuPipeline,
    pub(in crate::sim::c220) va: C220VaRegisters,
    pub(in crate::sim::c220) releases: Vec<C220VectorUopRelease>,
    scalar_flags: BTreeMap<u32, VecDeque<C220VectorFence>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220VectorFence {
    pub instruction_group: Option<u64>,
    pub vmsu_generation: Option<u64>,
}

impl VectorEngine {
    pub(in crate::sim::c220) fn ub_activity_at(
        &self,
        tick: u64,
    ) -> crate::sim::c220::memory::ub_service::C220UbVectorActivity {
        let (write, read) = self.pipeline.ub_port_occupancy(tick);
        let (merge_write, merge_read) = self.vmsu.ub_port_occupancy();
        let mut activity = crate::sim::c220::memory::ub_service::C220UbVectorActivity {
            write_pending: write || merge_write,
            read_pending: read || merge_read,
            ..Default::default()
        };
        for cycle in self.ub_cycles_at(tick) {
            activity.bank_mask |= cycle.bank_mask;
            activity.triggered = true;
        }
        activity
    }

    pub(in crate::sim::c220) fn signal_scalar(&mut self, flag_id: u32) {
        let fence = self.instruction_fence();
        self.scalar_flags
            .entry(flag_id)
            .or_default()
            .push_back(fence);
    }

    pub(in crate::sim::c220) fn scalar_event_ready_tick(
        &self,
        flag_id: u32,
        tick: u64,
    ) -> Option<u64> {
        let fence = *self.scalar_flags.get(&flag_id)?.front()?;
        Some(self.fence_retirement_tick(fence).unwrap_or(tick))
    }

    pub(in crate::sim::c220) fn consume_scalar_event(&mut self, flag_id: u32) {
        if let Some(events) = self.scalar_flags.get_mut(&flag_id) {
            events.pop_front();
            if events.is_empty() {
                self.scalar_flags.remove(&flag_id);
            }
        }
    }

    pub(in crate::sim::c220) fn instruction_fence(&self) -> C220VectorFence {
        C220VectorFence {
            instruction_group: self.pipeline.instruction_fence(),
            vmsu_generation: self.vmsu.instruction_fence(),
        }
    }

    pub(in crate::sim::c220) fn fence_retirement_tick(
        &self,
        fence: C220VectorFence,
    ) -> Option<u64> {
        fence
            .instruction_group
            .and_then(|group| self.pipeline.fence_retirement_tick(group))
            .into_iter()
            .chain(
                fence
                    .vmsu_generation
                    .and_then(|generation| self.vmsu.fence_retirement_tick(generation)),
            )
            .max()
    }
    pub(in crate::sim::c220) fn ub_cycles_at(
        &self,
        tick: u64,
    ) -> impl Iterator<Item = &crate::sim::c220::memory::C220UbCycle> {
        self.pipeline
            .last_ub_cycles()
            .iter()
            .rev()
            .take_while(move |cycle| cycle.tick == tick)
            .chain(
                self.vmsu
                    .trace()
                    .into_iter()
                    .flat_map(|trace| trace.repeats.iter().rev())
                    .flat_map(|repeat| repeat.ub_cycles.iter().rev())
                    .take_while(move |cycle| cycle.tick == tick),
            )
    }

    pub(in crate::sim::c220) fn new(rules: C220VectorTimingRules, mask: C220CompareMask) -> Self {
        let mut pipeline = C220VectorPipeline::new(rules);
        pipeline.set_compare_mask(mask);
        Self {
            pipeline,
            vmsu: C220VmsuPipeline::new(rules),
            va: C220VaRegisters::default(),
            releases: Vec::new(),
            scalar_flags: BTreeMap::new(),
        }
    }

    pub(in crate::sim::c220) fn begin_advance(&mut self) {
        self.releases.clear();
        self.pipeline.begin_advance();
    }

    pub(in crate::sim::c220) fn next_event_tick(&self) -> Option<u64> {
        self.pipeline
            .next_event_tick()
            .into_iter()
            .chain(self.vmsu.next_event_tick())
            .min()
    }

    pub(in crate::sim::c220) fn advance_event(
        &mut self,
        tick: u64,
        state: &mut C220State,
    ) -> Result<(), C220VectorRuntimeError> {
        let va_updates = self.pipeline.last_va_updates().len();
        self.releases
            .extend(self.pipeline.advance_event(tick, state)?);
        for &update in &self.pipeline.last_va_updates()[va_updates..] {
            self.va.apply(update);
        }
        self.vmsu.advance_to(tick, state)?;
        Ok(())
    }

    pub(in crate::sim::c220) fn issue_at(
        &mut self,
        tick: u64,
        instruction: &C220VectorInstruction,
    ) -> Result<(), C220VectorRuntimeError> {
        let uops = instruction.uops()?;
        let stores = instruction.stores();
        self.pipeline.issue_classified_at(
            tick,
            &uops,
            stores,
            instruction.read_issue(),
            instruction.queue_class(),
        )?;
        Ok(())
    }

    pub(in crate::sim::c220) fn pending_drain_tick(&self) -> Option<u64> {
        self.pipeline
            .pending_drain_tick()
            .into_iter()
            .chain(self.vmsu.pending_drain_tick())
            .max()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum C220VectorRuntimeError {
    #[error(transparent)]
    Execute(#[from] crate::sim::c220::state::C220ExecutionError),
    #[error(transparent)]
    Plan(#[from] super::C220VectorError),
    #[error(transparent)]
    Scalar(#[from] crate::sim::common::scalar::ScalarMachineError),
    #[error(transparent)]
    Advance(#[from] C220VectorAdvanceError),
    #[error(transparent)]
    Issue(#[from] C220VectorPipelineError),
    #[error(transparent)]
    Uop(#[from] C220VectorUopError),
    #[error(transparent)]
    Merge(#[from] C220VmsuError),
}
