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

mod advance;
mod cache;
mod lsu;
pub use lsu::{C220CoreLsuCompletion, C220CoreLsuConfig, C220CoreLsuIssue};
pub use lsu::{C220CoreLoadCompletion, C220CoreLoadIssue};
mod cube;
mod decode;
mod dispatch;
mod external_fixp;
mod factor;
mod fixp;
mod fixp_barrier;
mod fixp_frontend;
pub use external_fixp::C220CoreFixpConfig;
pub use factor::{C220FactorOutcome, C220FactorReadConfig};
pub use fixp_barrier::C220FixpBarrier;
pub use fixp_frontend::C220FixpFrontendConfig;
mod hflag;
mod mte1;
mod mte2;
mod mte3;

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
    pub timing: C220CoreTimingRules,
}

impl C220CoreConfig {
    pub fn new(timing: C220CoreTimingRules) -> Self {
        Self {
            device: C220Device::default(),
            cube: C220CubeConfig::default(),
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
    scalar_timing: C220ScalarTimingLane,
    lsu: Option<lsu::CoreLsu>,
    mte1: Mte1Engine,
    mte_pipeline: Option<C220MtePipeline>,
    fixp: Option<fixp::CoreFixp>,
    external_fixp: Option<external_fixp::CoreExternalFixp>,
    fixp_frontend: fixp_frontend::FixpFrontend,
    hardware_flags: C220HardwareFlagState,
    mte3: Mte3Engine,
    cube: CubeEngine,
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
            scalar_timing: C220ScalarTimingLane::default(),
            lsu: None,
            mte1: Mte1Engine::default(),
            mte_pipeline: None,
            fixp: None,
            external_fixp: None,
            fixp_frontend: fixp_frontend::FixpFrontend::default(),
            hardware_flags: C220HardwareFlagState::default(),
            mte3: Mte3Engine::new(timing.mte3),
            cube: CubeEngine::new(cube_config)?,
            vector: VectorEngine::new(timing.vector, initial_compare_mask),
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
            || !self.mte3.dma_commands.is_empty()
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
        self.scalar_timing.advance_to(tick);
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
            let stop = if self.state.scalar().is_halted()
                && self.lsu.as_ref().is_none_or(|lsu| lsu.next_tick.is_none())
            {
                Some(C220RunStop::Halted)
            } else if tick >= tick_limit {
                Some(C220RunStop::TickBudget)
            } else if events.len() >= max_events {
                Some(C220RunStop::EventBudget)
            } else {
                None
            };
            if let Some(stop) = stop {
                self.advance_to(tick)?;
                return Ok(C220CoreRun {
                    start_tick,
                    next_tick: tick,
                    next_pc: self.state.scalar().pc(),
                    events,
                    stop,
                });
            }
            if self.state.scalar().is_halted() {
                self.advance_to(tick)?;
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
            self.mte1
                .next_event_tick()
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
