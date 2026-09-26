use crate::sim::c220::sync::C220PipelineEvents;
use std::collections::VecDeque;

use crate::sim::c220::state::C220State;
use crate::sim::c220::vector::ops::compare::C220CompareMask;
use crate::sim::c220::vector::pipeline::{C220VectorPipeline, C220VectorTimingRules};
use crate::sim::c220::vector::timing::C220VectorUopRelease;
use crate::sim::c220::vector::va::C220VaRegisters;
use crate::sim::c220::vector::vmsu::C220VmsuPipeline;

use super::frontend::{C220VectorFrontendConfig, VectorFrontend};
use super::pipeline::{C220VectorAdvanceError, C220VectorPipelineError};
use super::retirement::PendingVectorInstruction;
use super::vmsu::C220VmsuError;
use super::{C220VectorFence, C220VectorRetirement};
use super::{C220VectorInstruction, C220VectorUopError};

pub(in crate::sim::c220) struct VectorEngine {
    pub(super) frontend: VectorFrontend,
    pub(in crate::sim::c220) pipeline: C220VectorPipeline,
    pub(in crate::sim::c220) vmsu: C220VmsuPipeline,
    pub(in crate::sim::c220) va: C220VaRegisters,
    pub(in crate::sim::c220) releases: Vec<C220VectorUopRelease>,
    pub(super) pending_instructions: VecDeque<PendingVectorInstruction>,
    pub(super) last_dispatched_id: Option<u64>,
    pub(super) last_retired_tick: Option<u64>,
    pub(super) observed_tick: Option<u64>,
    pub(in crate::sim::c220) retirements: Vec<C220VectorRetirement>,
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

    pub(in crate::sim::c220) fn new(
        mut rules: C220VectorTimingRules,
        mask: C220CompareMask,
        config: C220VectorFrontendConfig,
    ) -> Self {
        let frontend = VectorFrontend::new(config, rules.dispatch_ticks);
        rules.dispatch_ticks = 0;
        let mut pipeline = C220VectorPipeline::new(rules);
        pipeline.set_compare_mask(mask);
        Self {
            frontend,
            pipeline,
            vmsu: C220VmsuPipeline::new(rules),
            va: C220VaRegisters::default(),
            releases: Vec::new(),
            pending_instructions: VecDeque::new(),
            last_dispatched_id: None,
            last_retired_tick: None,
            observed_tick: None,
            retirements: Vec::new(),
        }
    }

    pub(in crate::sim::c220) fn begin_advance(&mut self) {
        self.frontend.events.clear();
        self.releases.clear();
        self.retirements.clear();
        self.pipeline.begin_advance();
    }

    pub(in crate::sim::c220) fn next_event_tick(&self) -> Option<u64> {
        self.pipeline
            .next_event_tick()
            .into_iter()
            .chain(self.vmsu.next_event_tick())
            .chain(self.next_retirement_tick())
            .chain(self.frontend.next_tick)
            .min()
    }

    pub(in crate::sim::c220) fn advance_event(
        &mut self,
        tick: u64,
        state: &mut C220State,
        events: &mut C220PipelineEvents,
    ) -> Result<(), C220VectorRuntimeError> {
        let va_updates = self.pipeline.last_va_updates().len();
        self.releases
            .extend(self.pipeline.advance_event(tick, state)?);
        for &update in &self.pipeline.last_va_updates()[va_updates..] {
            self.va.apply(update);
        }
        self.vmsu.advance_to(tick, state)?;
        let previous_retirements = self.retirements.len();
        self.retire_ready(tick);
        for retirement in &self.retirements[previous_retirements..] {
            events.retire(1, retirement.instruction_id, retirement.retirement_tick);
        }
        self.release_barriers_at(tick);
        self.observed_tick = Some(tick);
        self.advance_frontend(tick, state, events)?;
        Ok(())
    }

    pub(in crate::sim::c220) fn issue_at(
        &mut self,
        tick: u64,
        instruction: &C220VectorInstruction,
    ) -> Result<(), C220VectorRuntimeError> {
        if matches!(
            instruction,
            C220VectorInstruction::MoveAddress { .. } | C220VectorInstruction::WriteSpr(_)
        ) {
            self.pipeline.issue_register_write_at(tick)?;
            return Ok(());
        }
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
        self.pending_execution_drain_tick()
            .into_iter()
            .chain(self.fence_retirement_tick(self.instruction_fence()))
            .max()
    }

    pub(super) fn pending_execution_drain_tick(&self) -> Option<u64> {
        self.pipeline
            .pending_drain_tick()
            .into_iter()
            .chain(self.vmsu.pending_drain_tick())
            .chain(
                self.fence_retirement_tick(C220VectorFence {
                    instruction_id: self
                        .pending_instructions
                        .back()
                        .map(|entry| entry.instruction_id),
                }),
            )
            .max()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum C220VectorRuntimeError {
    #[error("Vector instruction ID {requested} does not follow {previous}")]
    InstructionOrder { previous: u64, requested: u64 },
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
