use crate::architecture::Architecture;
use crate::image::loader::LoadedDeviceKernel;
use crate::memory::hbm_pv_memory::HbmPvMemory;
use crate::memory::mapped::MappedMemory;
use crate::sim::c220::cube::{C220CubeExecutionOutcome, C220CubePipeline};
use crate::sim::c220::device::C220Device;
use crate::sim::c220::memory::{C220LocalMemory, C220LocalMemoryConfig};
use crate::sim::c220::mte::mte1::{C220Mte1CommandState, C220Mte1Outcome};
use crate::sim::c220::mte::mte2::C220Mte2Pipeline;
use crate::sim::c220::mte::{C220MtePipeline, C220MtePipelineConfig};
use crate::sim::c220::scalar::timing::C220ScalarTimingLane;
use crate::sim::c220::schedule::{C220IssueClock, C220Stall, C220StallCause};
use crate::sim::c220::state::C220State;
use crate::sim::c220::sync::C220HardwareFlagState;
use crate::sim::c220::vector::ops::compare::C220CompareMask;
use crate::sim::c220::vector::pipeline::C220VectorPipeline;
use crate::sim::c220::vector::timing::C220VectorUopRelease;
use crate::sim::c220::vector::va::C220VaRegisters;
use crate::sim::c220::vector::vmsu::C220VmsuPipeline;

mod activity;
mod termination;
pub use termination::C220Termination;
mod advance;
pub use activity::C220CoreActivity;
mod cache;
mod lsu;
pub use crate::sim::c220::sync::C220DeviceSync;
pub use lsu::{C220CoreAtomicCompletion, C220CoreAtomicIssue};
pub use lsu::{C220CoreLoadCompletion, C220CoreLoadIssue};
pub use lsu::{C220CoreLsuAdmission, C220CoreLsuCompletion, C220CoreLsuConfig, C220CoreLsuIssue};
pub use lsu::{C220CoreMaintenanceCompletion, C220CoreMaintenanceIssue};
pub use lsu::{C220CorePreloadCompletion, C220CorePreloadIssue};
pub use lsu::{C220CoreStoreCompletion, C220CoreStoreIssue};
mod cube;
mod cube_barrier;
mod cube_frontend;
mod decode;
mod dispatch;
mod external_fixp;
mod factor;
mod fixp;
mod fixp_barrier;
mod fixp_frontend;
mod scalar_flag;
pub use external_fixp::C220CoreFixpConfig;
pub use factor::{C220FactorOutcome, C220FactorReadConfig};
pub use fixp_barrier::C220FixpBarrier;
pub use fixp_frontend::C220FixpFrontendConfig;
mod cross_core;
mod hflag;
mod mte1;
mod mte1_frontend;
pub use mte1_frontend::C220Mte1QueuedCommand;
mod mte1_barrier;
pub use mte1_barrier::C220Mte1Barrier;
mod mte2_barrier;
pub use mte2_barrier::C220Mte2Barrier;
mod mte1_issue;
pub use mte1_issue::{C220Mte1FrontendConfig, C220Mte1IssuedInstruction, C220Mte1Operation};
mod mte2;
mod mte2_frontend;
pub use mte2_frontend::{
    C220Mte2FrontendConfig, C220Mte2IssuedInstruction, C220Mte2Operation, C220Mte2QueuedCommand,
};
mod mte3;
mod mte3_barrier;
pub use mte3_barrier::C220Mte3Barrier;
mod mte3_frontend;
pub use mte3_frontend::{C220Mte3IssueQueueConfig, C220Mte3IssuedInstruction, C220Mte3Operation};

use crate::sim::c220::cube::runtime::CubeEngine;
use crate::sim::c220::mte::mte1::runtime::Mte1Engine;
use crate::sim::c220::mte::mte3::runtime::Mte3Engine;
use crate::sim::c220::vector::runtime::VectorEngine;

mod error;
mod instruction;
use crate::sim::c220::cube::C220CubeConfig;
use crate::sim::c220::mte::mte2::C220Mte2TimingRules;
use crate::sim::c220::mte::mte3::C220Mte3TimingRules;
use crate::sim::c220::vector::pipeline::C220VectorTimingRules;
pub use error::C220CoreError;
pub use instruction::C220CoreInstruction;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220CoreConfig {
    pub device: C220Device,
    pub cube: C220CubeConfig,
    pub cube_frontend: crate::sim::c220::cube::frontend::C220CubeFrontendConfig,
    pub vector_frontend: crate::sim::c220::vector::C220VectorFrontendConfig,
    pub mte1_frontend: C220Mte1FrontendConfig,
    pub mte2_frontend: C220Mte2FrontendConfig,
    pub mte3_issue_queue: C220Mte3IssueQueueConfig,
    pub timing: C220CoreTimingRules,
}

impl C220CoreConfig {
    pub fn new(timing: C220CoreTimingRules) -> Self {
        Self {
            device: C220Device::default(),
            cube: C220CubeConfig::default(),
            cube_frontend: Default::default(),
            vector_frontend: Default::default(),
            mte1_frontend: Default::default(),
            mte2_frontend: Default::default(),
            mte3_issue_queue: Default::default(),
            timing,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220CoreTimingRules {
    pub mte2: C220Mte2TimingRules,
    pub mte3: C220Mte3TimingRules,
    pub vector: C220VectorTimingRules,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(
    clippy::large_enum_variant,
    reason = "Keep per-instruction stepping allocation-free"
)]
pub enum C220CoreStep {
    Executed {
        tick: u64,
        instruction: C220CoreInstruction,
    },
    Stalled(C220Stall),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220RunStop {
    Halted,
    TickBudget,
    EventBudget,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220CoreRun {
    pub start_tick: u64,
    pub next_tick: u64,
    pub next_pc: u64,
    pub events: Vec<C220CoreStep>,
    pub stop: C220RunStop,
}

pub struct C220Core {
    clock: C220IssueClock,
    next_instruction_id: u64,
    device: C220Device,
    state: C220State,
    mte2: C220Mte2Pipeline,
    mte2_frontend: mte2_frontend::Mte2Frontend,
    scalar_timing: C220ScalarTimingLane,
    termination: C220Termination,
    lsu: Option<lsu::CoreLsu>,
    mte1: Mte1Engine,
    mte1_frontend: mte1_frontend::Mte1Frontend,
    mte_pipeline: Option<C220MtePipeline>,
    fixp: Option<fixp::CoreFixp>,
    external_fixp: Option<external_fixp::CoreExternalFixp>,
    fixp_frontend: fixp_frontend::FixpFrontend,
    hardware_flags: C220HardwareFlagState,
    pipeline_events: crate::sim::c220::sync::C220PipelineEvents,
    device_flags: crate::sim::c220::sync::C220DeviceFlagState,
    mte3: Mte3Engine,
    mte3_issue_queue: mte3_frontend::Mte3IssueQueue,
    cube: CubeEngine,
    cube_frontend: cube_frontend::CubeFrontend,
    vector: VectorEngine,
    local_memory: C220LocalMemory,
    memory: MappedMemory,
}

impl C220Core {
    pub fn new(
        state: C220State,
        memory: MappedMemory,
        timing: C220CoreTimingRules,
    ) -> Result<Self, C220CoreError> {
        Self::with_config(state, memory, C220CoreConfig::new(timing))
    }

    pub fn with_config(
        state: C220State,
        memory: MappedMemory,
        config: C220CoreConfig,
    ) -> Result<Self, C220CoreError> {
        let C220CoreConfig {
            device,
            cube: cube_config,
            cube_frontend,
            vector_frontend,
            mte1_frontend,
            mte2_frontend,
            mte3_issue_queue,
            timing,
        } = config;
        if state.scalar().machine().architecture() != Architecture::Dav2201 {
            return Err(C220CoreError::ArchitectureMismatch);
        }
        let initial_compare_mask = C220CompareMask::from_bits([
            state.scalar().machine().spr_value(104).unwrap_or_default(),
            state.scalar().machine().spr_value(105).unwrap_or_default(),
        ]);
        Ok(Self {
            clock: C220IssueClock::default(),
            next_instruction_id: 0,
            device,
            state,
            mte2: C220Mte2Pipeline::new(timing.mte2),
            mte2_frontend: mte2_frontend::Mte2Frontend::new(mte2_frontend),
            scalar_timing: C220ScalarTimingLane::default(),
            termination: C220Termination::default(),
            lsu: None,
            mte1: Mte1Engine::with_outstanding_limit(mte1_frontend.outstanding_limit),
            mte1_frontend: mte1_frontend::Mte1Frontend {
                config: mte1_frontend,
                ..Default::default()
            },
            mte_pipeline: None,
            fixp: None,
            external_fixp: None,
            fixp_frontend: fixp_frontend::FixpFrontend::default(),
            hardware_flags: C220HardwareFlagState::default(),
            pipeline_events: Default::default(),
            device_flags: crate::sim::c220::sync::C220DeviceFlagState::default(),
            mte3: Mte3Engine::new(timing.mte3),
            mte3_issue_queue: mte3_frontend::Mte3IssueQueue::new(mte3_issue_queue),
            cube: CubeEngine::new(cube_config)?,
            cube_frontend: cube_frontend::CubeFrontend::new(cube_frontend),
            vector: VectorEngine::new(timing.vector, initial_compare_mask, vector_frontend),
            local_memory: C220LocalMemory::new(C220LocalMemoryConfig::for_device(device))?,
            memory,
        })
    }

    pub const fn device(&self) -> C220Device {
        self.device
    }

    pub const fn next_instruction_id(&self) -> u64 {
        self.next_instruction_id
    }

    pub const fn state(&self) -> &C220State {
        &self.state
    }

    pub const fn mte2_pipeline(&self) -> &C220Mte2Pipeline {
        &self.mte2
    }

    pub const fn scalar_timing(&self) -> &C220ScalarTimingLane {
        &self.scalar_timing
    }

    pub fn mte_pipeline(&self) -> Option<&C220MtePipeline> {
        self.mte_pipeline.as_ref()
    }

    pub fn configure_mte_pipeline(
        &mut self,
        config: C220MtePipelineConfig,
    ) -> Result<(), C220CoreError> {
        if self.lsu.is_some() {
            return Err(C220CoreError::LsuAlreadyConfigured);
        }
        if self.mte1.pending_commands().next().is_some()
            || self
                .fixp
                .as_ref()
                .is_some_and(|fixp| !fixp.engine.is_idle() || !fixp.bindings.is_idle())
            || self.mte2.is_busy()
            || self
                .external_fixp
                .as_ref()
                .is_some_and(|fixp| !fixp.engine.is_idle() || !fixp.bindings.is_idle())
            || !self.mte3.native_commands.is_empty()
            || self.mte_pipeline.as_ref().is_some_and(|p| !p.is_idle())
        {
            return Err(C220CoreError::MtePipelineBusy);
        }
        self.mte_pipeline = Some(C220MtePipeline::new(self.mte1.tick(), config));
        self.fixp = None;
        self.external_fixp = None;
        Ok(())
    }

    pub fn configure_timed_memory(
        &mut self,
        config: crate::sim::c220::memory::timed_memory::C220TimedMemoryConfig,
    ) -> Result<(), C220CoreError> {
        if self.lsu.is_some() {
            return Err(C220CoreError::LsuAlreadyConfigured);
        }
        self.mte_pipeline
            .as_mut()
            .ok_or(C220CoreError::MteUnconfigured)?
            .connect_timed_memory(config)?;
        Ok(())
    }

    pub fn pending_mte1_commands(&self) -> impl Iterator<Item = C220Mte1CommandState> + '_ {
        self.mte1.pending_commands()
    }

    pub const fn hardware_flags(&self) -> &C220HardwareFlagState {
        &self.hardware_flags
    }

    pub const fn pipeline_events(&self) -> &crate::sim::c220::sync::C220PipelineEvents {
        &self.pipeline_events
    }

    pub const fn memory(&self) -> &MappedMemory {
        &self.memory
    }

    pub const fn local_memory(&self) -> &C220LocalMemory {
        &self.local_memory
    }

    pub const fn vector_pipeline(&self) -> &C220VectorPipeline {
        &self.vector.pipeline
    }

    pub const fn vmsu_pipeline(&self) -> &C220VmsuPipeline {
        &self.vector.vmsu
    }

    pub const fn cube_pipeline(&self) -> &C220CubePipeline {
        &self.cube.pipeline
    }

    pub const fn compare_mask(&self) -> C220CompareMask {
        self.vector.pipeline.compare_mask()
    }

    pub const fn va_registers(&self) -> &C220VaRegisters {
        &self.vector.va
    }

    pub fn last_vector_releases(&self) -> &[C220VectorUopRelease] {
        &self.vector.releases
    }

    pub fn last_vector_retirements(&self) -> &[crate::sim::c220::vector::C220VectorRetirement] {
        &self.vector.retirements
    }

    pub fn outstanding_vector_instructions(&self) -> usize {
        self.vector.outstanding_instructions()
    }

    pub fn queued_vector_instructions(
        &self,
    ) -> impl ExactSizeIterator<Item = crate::sim::c220::vector::C220VectorQueuedInstruction> + '_
    {
        self.vector.queued_instructions()
    }

    pub fn received_vector_instructions(
        &self,
    ) -> impl ExactSizeIterator<Item = crate::sim::c220::vector::C220VectorReception> + '_ {
        self.vector.received_instructions()
    }

    pub fn last_vector_frontend_events(
        &self,
    ) -> &[crate::sim::c220::vector::C220VectorFrontendEvent] {
        self.vector.frontend_events()
    }

    pub fn pending_vector_barriers(
        &self,
    ) -> impl ExactSizeIterator<Item = &crate::sim::c220::vector::C220VectorBarrier> {
        self.vector.pending_barriers()
    }

    pub fn last_cube_outcomes(&self) -> &[C220CubeExecutionOutcome] {
        &self.cube.outcomes
    }

    pub fn last_mte1_outcomes(&self) -> &[C220Mte1Outcome] {
        &self.mte1.outcomes
    }

    pub fn memory_mut(&mut self) -> &mut MappedMemory {
        &mut self.memory
    }

    pub fn local_memory_mut(&mut self) -> &mut C220LocalMemory {
        &mut self.local_memory
    }

    pub fn pending_output_retirement_tick(&self) -> Option<u64> {
        self.mte3.pending_retirement_tick()
    }

    pub fn last_mte3_outcomes(&self) -> &[crate::sim::c220::mte::mte3::C220Mte3Outcome] {
        &self.mte3.outcomes
    }

    pub fn pending_mte3_commands(
        &self,
    ) -> impl Iterator<Item = crate::sim::c220::mte::mte3::C220Mte3CommandState> + '_ {
        self.mte3.pending_commands()
    }

    pub fn advance_to(&mut self, tick: u64) -> Result<Option<C220Stall>, C220CoreError> {
        let gate = self.clock.observe(tick, self.state.scalar().pc())?;
        self.advance_engines_to(tick)?;
        self.hardware_flags.advance_to(tick)?;
        self.local_memory
            .l0c_mut()
            .scoreboard_mut()
            .advance_to(tick);
        Ok(gate)
    }

    pub fn step_loaded_at(
        &mut self,
        tick: u64,
        kernel: &LoadedDeviceKernel,
        code_memory: &mut HbmPvMemory,
    ) -> Result<C220CoreStep, C220CoreError> {
        if kernel.placement().architecture != Architecture::Dav2201 {
            return Err(C220CoreError::ArchitectureMismatch);
        }
        let pc = self.state.scalar().pc();
        let word = kernel.fetch_executable_word(code_memory, pc)?;
        self.step_word_at(tick, word)
    }

    pub fn run_loaded_until(
        &mut self,
        start_tick: u64,
        tick_limit: u64,
        max_events: usize,
        kernel: &LoadedDeviceKernel,
        code_memory: &mut HbmPvMemory,
    ) -> Result<C220CoreRun, C220CoreError> {
        let mut tick = start_tick;
        let mut events = Vec::new();
        loop {
            if self.state.scalar().is_halted() {
                self.advance_to(tick)?;
            }
            let stop = if self.termination.is_complete() {
                Some(C220RunStop::Halted)
            } else if tick >= tick_limit {
                Some(C220RunStop::TickBudget)
            } else if events.len() >= max_events {
                Some(C220RunStop::EventBudget)
            } else {
                None
            };
            if let Some(stop) = stop {
                if !self.state.scalar().is_halted() {
                    self.advance_to(tick)?;
                }
                return Ok(C220CoreRun {
                    start_tick,
                    next_tick: tick,
                    next_pc: self.state.scalar().pc(),
                    events,
                    stop,
                });
            }
            if self.state.scalar().is_halted() {
                tick = tick.checked_add(1).ok_or(C220CoreError::TimeOverflow)?;
                continue;
            }
            let event = self.step_loaded_at(tick, kernel, code_memory)?;
            tick = match &event {
                C220CoreStep::Executed { .. } => {
                    tick.checked_add(1).ok_or(C220CoreError::TimeOverflow)?
                }
                C220CoreStep::Stalled(stall) if stall.resume_tick > tick => {
                    stall.resume_tick.min(tick_limit)
                }
                _ => return Err(C220CoreError::NonprogressingStall),
            };
            events.push(event);
        }
    }

    pub fn step_word_at(&mut self, tick: u64, word: u32) -> Result<C220CoreStep, C220CoreError> {
        let step = self.dispatch_word_at(tick, word)?;
        if matches!(step, C220CoreStep::Executed { .. }) {
            self.clock.finish(tick)?;
            self.next_instruction_id = self
                .next_instruction_id
                .checked_add(1)
                .ok_or(C220CoreError::TimeOverflow)?;
        }
        Ok(step)
    }

    fn pending_compute_drain(&self) -> Option<(u64, C220StallCause)> {
        [
            self.lsu
                .as_ref()
                .and_then(|lsu| lsu.next_tick)
                .map(|tick| (tick, C220StallCause::LsuDependency)),
            self.external_fixp
                .as_ref()
                .and_then(|fixp| {
                    self.mte_pipeline
                        .as_ref()
                        .and_then(|pipeline| pipeline.next_external_fixp_event_tick(&fixp.engine))
                })
                .map(|tick| (tick, C220StallCause::FixpDependency)),
            self.fixp
                .as_ref()
                .and_then(|fixp| {
                    self.mte_pipeline
                        .as_ref()
                        .and_then(|pipeline| pipeline.next_fixp_event_tick(&fixp.engine))
                })
                .map(|tick| (tick, C220StallCause::FixpDependency)),
            self.scalar_timing
                .pending_drain_tick()
                .map(|tick| (tick, C220StallCause::ScalarDependency)),
            self.mte3
                .timing
                .latest_retirement_tick()
                .map(|tick| (tick, C220StallCause::Mte3Dependency)),
            self.vector
                .pending_drain_tick()
                .map(|tick| (tick, C220StallCause::VectorDependency)),
            self.cube
                .pipeline
                .pending_drain_tick()
                .map(|tick| (tick, C220StallCause::CubeDependency)),
            self.cube_frontend
                .next_tick
                .map(|tick| (tick, C220StallCause::CubeDependency)),
            self.pending_mte1_tick()
                .map(|tick| (tick, C220StallCause::Mte1Dependency)),
            self.mte2
                .next_event_tick()
                .map(|tick| (tick, C220StallCause::Mte2Dependency)),
            self.mte_pipeline
                .as_ref()
                .and_then(C220MtePipeline::next_event_tick)
                .map(|tick| (tick, C220StallCause::MtePhysicalDependency)),
        ]
        .into_iter()
        .flatten()
        .max_by_key(|(tick, _)| *tick)
    }
}

#[cfg(test)]
mod tests;
