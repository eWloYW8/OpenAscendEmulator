use std::collections::BTreeMap;

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
    pub(in crate::sim::c220) scalar_flags: BTreeMap<u32, u64>,
}

impl VectorEngine {
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

    pub(in crate::sim::c220) fn pending_visibility_tick(&self) -> Option<u64> {
        self.pipeline
            .pending_visibility_tick()
            .into_iter()
            .chain(self.vmsu.pending_visibility_tick())
            .max()
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
