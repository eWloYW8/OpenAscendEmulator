use std::collections::BTreeMap;

use thiserror::Error;

use crate::architecture::Architecture;
use crate::architecture::c220::C220Device;
use crate::image::loader::{DeviceKernelFetchError, LoadedDeviceKernel};
use crate::isa::c220::axpy::C220AxpyInstruction;
use crate::isa::c220::compare::{
    C220CompareMaskInstruction, C220MoveMaskDirection, C220MoveMaskInstruction,
    C220PackedCompareInstruction,
};
use crate::isa::c220::control::C220VectorControlInstruction;
use crate::isa::c220::conversion::C220ConversionInstruction;
use crate::isa::c220::cube::C220CubeInstruction;
use crate::isa::c220::fused::C220FusedInstruction;
use crate::isa::c220::gather::C220GatherInstruction;
use crate::isa::c220::hflag::{
    C220HardwareFlagError, C220HardwareFlagInstruction, C220HardwareFlagOperation,
    C220HardwareFlagSourcePipe, C220MatrixMemory,
};
use crate::isa::c220::load_va::C220LoadVaInstruction;
use crate::isa::c220::merge::C220MergeInstruction;
use crate::isa::c220::mte::C220DmaMovDescriptor;
use crate::isa::c220::mte1::C220Load2dInstruction;
use crate::isa::c220::no_effect::C220NoEffectVectorInstruction;
use crate::isa::c220::reduce::C220ReductionInstruction;
use crate::isa::c220::scalar::C220ScalarConversionHint;
use crate::isa::c220::select::C220SelectInstruction;
use crate::isa::c220::sort::C220SortInstruction;
use crate::isa::c220::special::C220SpecialUnaryInstruction;
use crate::isa::c220::ternary::C220TernaryInstruction;
use crate::isa::c220::vector::{
    C220BroadcastInstruction, C220CopyInstruction, C220MoveVaInstruction, C220MovemaskHint,
    C220MovevInstruction, C220NchwInstruction, C220ShiftInstruction, C220TransposeInstruction,
    C220VecArithmeticHint,
};
use crate::isa::c220::vector_scalar::C220VectorScalarInstruction;
use crate::isa::flow::{
    FlagInstruction, FlagOperation, PipelineBarrierScope, PipelineBarrierStep,
};
use crate::memory::hbm_pv_memory::HbmPvMemory;
use crate::memory::mapped::{MappedMemory, MappedMemoryError};
use crate::sim::c220::core::functional::{C220FunctionalCore, C220FunctionalError};
use crate::sim::c220::core::functional::C220OutputAction;
use crate::sim::c220::cube::{
    C220CubeConfig, C220CubeControl, C220CubeExecutionError, C220CubeExecutionOutcome,
    C220CubeIssue, C220CubePipeline, C220CubeTimingError, C220PreparedCubeExecution,
    update_cube_status_spr2,
};
use crate::sim::c220::memory::{C220L0cError, C220LocalMemory, C220LocalMemoryConfig};
use crate::sim::c220::mte::load2d::{
    C220Load2dTransferError, C220PreparedLoad2d, prepare_c220_load2d,
};
use crate::sim::c220::mte::transfer::C220PreparedOutput;
use crate::sim::c220::scalar::bus::C220ScalarBusError;
use crate::sim::c220::timing::hflag::{C220HardwareFlagState, C220HardwareFlagTimingError};
use crate::sim::c220::timing::mte1::{
    C220Mte1TimingError, C220Mte1TimingRules, C220TimedMte1Lane,
};
use crate::sim::c220::timing::mte2::{
    C220Mte2Pipeline, C220Mte2Step, C220Mte2TimingRules, C220Stall, C220StallCause,
    C220TimingError, is_mte2_transfer,
};
use crate::sim::c220::timing::mte3::{
    C220Mte3TimingError, C220Mte3TimingRules, C220TimedMte3Lane,
};
use crate::sim::c220::timing::scalar::{C220ScalarTimingLane, C220ScalarTimingTicket};
use crate::sim::c220::va::C220VaRegisters;
use crate::sim::c220::vector::compare::{
    C220CompareMask,
    plan_c220_compare_mask_issue, plan_c220_move_mask_issue, plan_c220_packed_compare_issue,
};
use crate::sim::c220::vector::load_va::plan_c220_load_va_issue;
use crate::sim::c220::vector::nchw::plan_c220_nchw_issue;
use crate::sim::c220::vector::pipeline::{
    C220VectorAdvanceError, C220VectorPipeline, C220VectorPipelineError, C220VectorQueueClass,
    C220VectorTimingRules,
};
use crate::sim::c220::vector::read::C220VectorReadIssue;
use crate::sim::c220::vector::select::{C220SelectMode, plan_c220_select_issue};
use crate::sim::c220::vector::timing::{
    C220VectorUopRelease, C220VectorWritePlanError,
};
use crate::sim::c220::vector::vmsu::{C220VmsuError, C220VmsuPipeline};
use crate::sim::c220::vector::{
    C220VectorError, C220VectorMaskState, decode_c220_repeat_masks,
};
use crate::sim::common::scalar::{ScalarInstructionError, ScalarMachineError};

use super::instruction::C220CoreInstruction;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220CoreTimingRules {
    pub mte2: C220Mte2TimingRules,
    pub mte3: C220Mte3TimingRules,
    pub vector: C220VectorTimingRules,
}

#[derive(Debug, Clone, PartialEq, Eq)]
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

#[derive(Debug, Error)]
pub enum C220CoreError {
    #[error("loaded kernel is not for dav_2201")]
    ArchitectureMismatch,
    #[error(transparent)]
    Fetch(#[from] DeviceKernelFetchError),
    #[error(transparent)]
    Timing(#[from] C220TimingError),
    #[error(transparent)]
    Mte(#[from] C220FunctionalError),
    #[error(transparent)]
    Mte1Timing(#[from] C220Mte1TimingError),
    #[error(transparent)]
    HardwareFlagDecode(#[from] C220HardwareFlagError),
    #[error(transparent)]
    HardwareFlagTiming(#[from] C220HardwareFlagTimingError),
    #[error(transparent)]
    Load2d(#[from] C220Load2dTransferError),
    #[error(transparent)]
    Mte3Timing(#[from] C220Mte3TimingError),
    #[error(transparent)]
    CubeTiming(#[from] C220CubeTimingError),
    #[error(transparent)]
    CubeExecution(#[from] C220CubeExecutionError),
    #[error(transparent)]
    LocalMemory(#[from] C220L0cError),
    #[error(transparent)]
    VectorPlan(#[from] C220VectorWritePlanError),
    #[error(transparent)]
    Vector(#[from] C220VectorError),
    #[error(transparent)]
    VectorPipeline(#[from] C220VectorPipelineError),
    #[error(transparent)]
    VectorAdvance(#[from] C220VectorAdvanceError),
    #[error(transparent)]
    Vmsu(#[from] C220VmsuError),
    #[error(transparent)]
    OutputMemory(#[from] MappedMemoryError),
    #[error(transparent)]
    Scalar(#[from] ScalarInstructionError<C220ScalarBusError<MappedMemoryError>>),
    #[error(transparent)]
    ScalarMachine(#[from] ScalarMachineError),
    #[error("timed core stalled without a future resume tick")]
    NonprogressingStall,
    #[error("tick counter overflowed")]
    TimeOverflow,
    #[error("vector repeat {repeat_index} has no writeback schedule")]
    MissingVectorWriteback { repeat_index: usize },
    #[error("C220 vector-to-scalar flag {flag_id} is already pending")]
    VectorScalarFlagAlreadySet { flag_id: u32 },
    #[error("C220 vector-to-scalar flag {flag_id} was not set before wait")]
    VectorScalarWaitWithoutFlag { flag_id: u32 },
    #[error("C220 Cube execution requires initialized SPR3 control state")]
    MissingCubeControlSpr,
    #[error(
        "C220 hardware flag source {source_pipe:?} for {memory:?} requires an unimplemented data path"
    )]
    UnsupportedHardwareFlagCheckpoint {
        source_pipe: C220HardwareFlagSourcePipe,
        memory: C220MatrixMemory,
    },
}

pub struct C220Core {
    device: C220Device,
    functional: C220FunctionalCore,
    mte2: C220Mte2Pipeline,
    scalar_timing: C220ScalarTimingLane,
    mte1: C220TimedMte1Lane,
    hardware_flags: C220HardwareFlagState,
    mte3: C220TimedMte3Lane,
    cube: C220CubePipeline,
    vector: C220VectorPipeline,
    vmsu: C220VmsuPipeline,
    va: C220VaRegisters,
    last_vector_releases: Vec<C220VectorUopRelease>,
    last_cube_outcomes: Vec<C220CubeExecutionOutcome>,
    pending_output: Option<C220PendingOutput>,
    pending_load2d: Vec<C220PendingLoad2d>,
    pending_cube: Vec<C220PendingCube>,
    vector_to_scalar_flags: BTreeMap<u32, u64>,
    local_memory: C220LocalMemory,
    memory: MappedMemory,
}

struct C220PendingOutput {
    data_ready_tick: u64,
    prepared: C220PreparedOutput,
}

struct C220PendingLoad2d {
    data_ready_tick: u64,
    prepared: C220PreparedLoad2d,
}

struct C220PendingCube {
    pc: u64,
    accept_tick: u64,
    prepared: C220PreparedCubeExecution,
}

impl C220Core {
    pub fn new(
        functional: C220FunctionalCore,
        memory: MappedMemory,
        timing: C220CoreTimingRules,
    ) -> Result<Self, C220CoreError> {
        Self::new_with_cube_config(functional, memory, timing, C220CubeConfig::default())
    }

    pub fn new_with_cube_config(
        functional: C220FunctionalCore,
        memory: MappedMemory,
        timing: C220CoreTimingRules,
        cube_config: C220CubeConfig,
    ) -> Result<Self, C220CoreError> {
        Self::new_for_device_with_cube_config(
            functional,
            memory,
            timing,
            C220Device::default(),
            cube_config,
        )
    }

    pub fn new_for_device(
        functional: C220FunctionalCore,
        memory: MappedMemory,
        timing: C220CoreTimingRules,
        device: C220Device,
    ) -> Result<Self, C220CoreError> {
        Self::new_for_device_with_cube_config(
            functional,
            memory,
            timing,
            device,
            C220CubeConfig::default(),
        )
    }

    pub fn new_for_device_with_cube_config(
        functional: C220FunctionalCore,
        memory: MappedMemory,
        timing: C220CoreTimingRules,
        device: C220Device,
        cube_config: C220CubeConfig,
    ) -> Result<Self, C220CoreError> {
        if functional.scalar().machine().architecture() != Architecture::Dav2201 {
            return Err(C220TimingError::ArchitectureMismatch.into());
        }
        let initial_compare_mask = C220CompareMask::from_bits([
            functional
                .scalar()
                .machine()
                .spr_value(104)
                .unwrap_or_default(),
            functional
                .scalar()
                .machine()
                .spr_value(105)
                .unwrap_or_default(),
        ]);
        let mut vector = C220VectorPipeline::new(timing.vector);
        vector.set_compare_mask(initial_compare_mask);
        Ok(Self {
            device,
            functional,
            mte2: C220Mte2Pipeline::new(timing.mte2),
            scalar_timing: C220ScalarTimingLane::default(),
            mte1: C220TimedMte1Lane::default(),
            hardware_flags: C220HardwareFlagState::default(),
            mte3: C220TimedMte3Lane::new(timing.mte3),
            cube: C220CubePipeline::new(cube_config)?,
            vector,
            vmsu: C220VmsuPipeline::new(timing.vector),
            va: C220VaRegisters::default(),
            last_vector_releases: Vec::new(),
            last_cube_outcomes: Vec::new(),
            pending_output: None,
            pending_load2d: Vec::new(),
            pending_cube: Vec::new(),
            vector_to_scalar_flags: BTreeMap::new(),
            local_memory: C220LocalMemory::new(C220LocalMemoryConfig::for_device(device))?,
            memory,
        })
    }

    pub const fn device(&self) -> C220Device {
        self.device
    }

    pub const fn functional(&self) -> &C220FunctionalCore {
        &self.functional
    }

    pub const fn mte2_pipeline(&self) -> &C220Mte2Pipeline {
        &self.mte2
    }

    pub const fn scalar_timing(&self) -> &C220ScalarTimingLane {
        &self.scalar_timing
    }

    pub const fn mte1_timing(&self) -> &C220TimedMte1Lane {
        &self.mte1
    }

    pub const fn hardware_flags(&self) -> &C220HardwareFlagState {
        &self.hardware_flags
    }

    pub fn configure_mte1_timing(
        &mut self,
        rules: C220Mte1TimingRules,
    ) -> Result<(), C220CoreError> {
        self.mte1.configure(rules)?;
        Ok(())
    }

    pub const fn memory(&self) -> &MappedMemory {
        &self.memory
    }

    pub const fn local_memory(&self) -> &C220LocalMemory {
        &self.local_memory
    }

    pub const fn vector_pipeline(&self) -> &C220VectorPipeline {
        &self.vector
    }

    pub const fn vmsu_pipeline(&self) -> &C220VmsuPipeline {
        &self.vmsu
    }

    pub const fn cube_pipeline(&self) -> &C220CubePipeline {
        &self.cube
    }

    pub const fn compare_mask(&self) -> C220CompareMask {
        self.vector.compare_mask()
    }

    pub const fn va_registers(&self) -> &C220VaRegisters {
        &self.va
    }

    pub fn last_vector_releases(&self) -> &[C220VectorUopRelease] {
        &self.last_vector_releases
    }

    pub fn last_cube_outcomes(&self) -> &[C220CubeExecutionOutcome] {
        &self.last_cube_outcomes
    }

    pub fn memory_mut(&mut self) -> &mut MappedMemory {
        &mut self.memory
    }

    pub fn local_memory_mut(&mut self) -> &mut C220LocalMemory {
        &mut self.local_memory
    }

    pub fn pending_output_ready_tick(&self) -> Option<u64> {
        self.pending_output
            .as_ref()
            .map(|pending| pending.data_ready_tick)
    }

    pub fn advance_to(&mut self, tick: u64) -> Result<Option<C220Stall>, C220CoreError> {
        let gate = self
            .mte2
            .gate_other_at(tick, self.functional.scalar().pc())?;
        self.scalar_timing.advance_to(tick);
        self.mte1.advance_to(tick);
        self.hardware_flags.advance_to(tick)?;
        self.commit_ready_load2d_at(tick)?;
        let retired_cube = {
            let arbiter = self.local_memory.l0c_mut().write_arbiter_mut();
            self.cube.advance_to(tick, arbiter)?.to_vec()
        };
        self.commit_retired_cube(&retired_cube)?;
        self.local_memory
            .l0c_mut()
            .scoreboard_mut()
            .advance_to(tick);
        self.last_vector_releases = self.vector.advance_to(tick, &mut self.functional)?;
        for &update in self.vector.last_va_updates() {
            self.va.apply(update);
        }
        self.vmsu.advance_to(tick, &mut self.functional)?;
        self.commit_ready_output_at(tick)?;
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
        let pc = self.functional.scalar().pc();
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
            let stop = if self.functional.scalar().is_halted() {
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
                    next_pc: self.functional.scalar().pc(),
                    events,
                    stop,
                });
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
        let gate = self.advance_to(tick)?;
        if let Some(stall) = gate {
            return Ok(C220CoreStep::Stalled(stall));
        }
        if let Some(resume_tick) = self.scalar_timing.dependency_tick(word, tick) {
            return Ok(C220CoreStep::Stalled(C220Stall {
                tick,
                pc: self.functional.scalar().pc(),
                resume_tick,
                cause: C220StallCause::ScalarDependency,
            }));
        }
        let pc = self.functional.scalar().pc();
        if is_mte1_word(word) {
            return self.step_mte1_at(tick, pc, word);
        }
        if is_mte2_word(word) {
            return match self
                .mte2
                .step_at(tick, &mut self.functional, word, &self.memory)?
            {
                C220Mte2Step::Stalled(stall) => Ok(C220CoreStep::Stalled(stall)),
                result @ C220Mte2Step::Executed { .. } => Ok(C220CoreStep::Executed {
                    tick,
                    instruction: C220CoreInstruction::Mte2(result),
                }),
            };
        }
        if let Some(decoded) = C220HardwareFlagInstruction::decode(word) {
            return self.step_hardware_flag_at(tick, pc, decoded);
        }
        let cube_instruction = C220CubeInstruction::decode(word);
        if cube_instruction.is_some() && tick < self.cube.next_accept_tick() {
            return Ok(C220CoreStep::Stalled(C220Stall {
                tick,
                pc,
                resume_tick: self.cube.next_accept_tick(),
                cause: C220StallCause::CubeDependency,
            }));
        }
        if is_c220_vector_word(word)
            && let Some(resume_tick) = self.vector.instruction_buffer_ready_tick()
            && tick < resume_tick
        {
            return Ok(C220CoreStep::Stalled(C220Stall {
                tick,
                pc,
                resume_tick,
                cause: C220StallCause::VectorDependency,
            }));
        }
        if is_c220_vector_word(word) {
            let queue_class = C220VectorQueueClass::from_word(word);
            let mut resume_tick = self.vector.pending_queue_hazard_tick(queue_class);
            if queue_class == C220VectorQueueClass::Vms4 {
                resume_tick = resume_tick
                    .into_iter()
                    .chain(self.vmsu.pending_drain_tick())
                    .max();
            }
            if let Some(resume_tick) = resume_tick
                && tick < resume_tick
            {
                return Ok(C220CoreStep::Stalled(C220Stall {
                    tick,
                    pc,
                    resume_tick,
                    cause: C220StallCause::VectorDependency,
                }));
            }
        }
        let instruction = if is_mte3_word(word) {
            if let Some(flag) = FlagInstruction::decode(Architecture::Dav2201, word)
                && flag.source_pipe_code == 1
                && flag.trigger_pipe_code == 5
                && flag.operation == FlagOperation::Wait
                && let Some(resume_tick) = self.pending_vector_visibility_tick()
                && tick < resume_tick
            {
                return Ok(C220CoreStep::Stalled(C220Stall {
                    tick,
                    pc,
                    resume_tick,
                    cause: C220StallCause::VectorDependency,
                }));
            }
            if let Some(flag) = FlagInstruction::decode(Architecture::Dav2201, word)
                && flag.source_pipe_code == 5
                && flag.trigger_pipe_code == 1
                && flag.operation == FlagOperation::Wait
            {
                let flag_id = flag
                    .resolve(pc, self.functional.scalar().machine().xregs())
                    .flag_id;
                if let Ok(flag_id) = u8::try_from(flag_id)
                    && let Some(resume_tick) = self.mte3.completion_ready_tick(flag_id)
                    && tick < resume_tick
                {
                    return Ok(C220CoreStep::Stalled(C220Stall {
                        tick,
                        pc,
                        resume_tick,
                        cause: C220StallCause::Mte3Dependency,
                    }));
                }
            }
            let (ticket, requests) = if C220DmaMovDescriptor::is_word(word) {
                if tick < self.mte3.next_issue_tick() {
                    return Ok(C220CoreStep::Stalled(C220Stall {
                        tick,
                        pc,
                        resume_tick: self.mte3.next_issue_tick(),
                        cause: C220StallCause::Mte3IssueRate,
                    }));
                }
                let plan = self.functional.preview_c220_mte3_transfer(word)?;
                let (ticket, requests) = self.mte3.preview_issue(tick, plan)?;
                if self.pending_output.is_some() {
                    return Err(C220Mte3TimingError::TicketMismatch.into());
                }
                (Some(ticket), requests)
            } else {
                (None, Vec::new())
            };
            let (step, prepared) = self.functional.step_c220_output_word_deferred(word)?;
            match step.action {
                C220OutputAction::CopyToHbm { .. } => {
                    let ticket = ticket.ok_or(C220Mte3TimingError::TicketMismatch)?;
                    self.mte3.issue(ticket)?;
                    self.pending_output = Some(C220PendingOutput {
                        data_ready_tick: ticket.data_ready_tick,
                        prepared: prepared.ok_or(C220Mte3TimingError::TicketMismatch)?,
                    });
                }
                C220OutputAction::SetMte3CompletionFlag { flag_id, .. } => {
                    self.mte3.set_completion_flag(flag_id)?;
                }
                C220OutputAction::WaitMte3CompletionFlag { flag_id, .. } => {
                    self.mte3.wait_completion_flag(flag_id)?;
                }
                _ => {}
            }
            C220CoreInstruction::Mte3 {
                step,
                requests,
                ticket,
            }
        } else {
            match word {
                _ if cube_instruction.is_some() => {
                    let decoded = cube_instruction.expect("matched Cube decode");
                    let registers = decoded.capture(self.functional.scalar().machine().xregs());
                    let parameters = decoded.parameters(registers);
                    let control_spr = self
                        .functional
                        .scalar()
                        .machine()
                        .spr_value(3)
                        .ok_or(C220CoreError::MissingCubeControlSpr)?;
                    let control = C220CubeControl::from_spr3(control_spr);
                    let ticket = self
                        .cube
                        .preview_issue(tick, decoded, parameters, control)?;
                    let issue = C220CubeIssue {
                        pc,
                        word,
                        instruction: decoded,
                        registers,
                        parameters,
                        ticket,
                    };
                    let prepared = issue.prepare(&self.local_memory, control)?;
                    self.cube
                        .issue(ticket, self.local_memory.l0c_mut().write_arbiter_mut())?;
                    self.pending_cube.push(C220PendingCube {
                        pc,
                        accept_tick: ticket.accept_tick,
                        prepared,
                    });
                    self.functional.commit_c220_sequential_issue();
                    C220CoreInstruction::Cube(issue)
                }
                _ if FlagInstruction::decode(Architecture::Dav2201, word).is_some_and(|flag| {
                    flag.source_pipe_code == 1 && flag.trigger_pipe_code == 0
                }) =>
                {
                    let flag = FlagInstruction::decode(Architecture::Dav2201, word)
                        .expect("matched vector-to-scalar flag")
                        .resolve(pc, self.functional.scalar().machine().xregs());
                    match flag.instruction.operation {
                        FlagOperation::Set => {
                            if self.vector_to_scalar_flags.contains_key(&flag.flag_id) {
                                return Err(C220CoreError::VectorScalarFlagAlreadySet {
                                    flag_id: flag.flag_id,
                                });
                            }
                            let ready_tick = self.pending_vector_drain_tick().unwrap_or(tick);
                            self.vector_to_scalar_flags.insert(flag.flag_id, ready_tick);
                        }
                        FlagOperation::Wait => {
                            let ready_tick = self
                                .vector_to_scalar_flags
                                .get(&flag.flag_id)
                                .copied()
                                .ok_or(C220CoreError::VectorScalarWaitWithoutFlag {
                                    flag_id: flag.flag_id,
                                })?;
                            if tick < ready_tick {
                                return Ok(C220CoreStep::Stalled(C220Stall {
                                    tick,
                                    pc,
                                    resume_tick: ready_tick,
                                    cause: C220StallCause::VectorDependency,
                                }));
                            }
                            self.vector_to_scalar_flags.remove(&flag.flag_id);
                        }
                    }
                    self.functional.commit_c220_vector_issue(None);
                    C220CoreInstruction::VectorToScalarFlag(flag)
                }
                _ if C220MoveVaInstruction::decode(word).is_some() => {
                    if let Some(resume_tick) = self.vector.pending_move_va_blocker_tick()
                        && tick < resume_tick
                    {
                        return Ok(C220CoreStep::Stalled(C220Stall {
                            tick,
                            pc,
                            resume_tick,
                            cause: C220StallCause::VectorDependency,
                        }));
                    }
                    let decoded = C220MoveVaInstruction::decode(word).expect("matched decode");
                    let instruction = C220CoreInstruction::VectorMoveAddress {
                        pc,
                        word,
                        instruction: decoded,
                    };
                    self.issue_vector_at(tick, &instruction)?;
                    self.va
                        .write_pair(decoded, self.functional.scalar().machine().xregs());
                    self.functional.commit_c220_vector_issue(None);
                    instruction
                }
                _ if C220LoadVaInstruction::decode(word).is_some() => {
                    let decoded = C220LoadVaInstruction::decode(word).expect("matched decode");
                    if decoded.high_half
                        && let Some(resume_tick) = self.vector.pending_load_va_drain_tick()
                        && tick < resume_tick
                    {
                        return Ok(C220CoreStep::Stalled(C220Stall {
                            tick,
                            pc,
                            resume_tick,
                            cause: C220StallCause::VectorDependency,
                        }));
                    }
                    let source_address = self.functional.scalar().machine().xregs()
                        [usize::from(decoded.source_register)];
                    let step =
                        plan_c220_load_va_issue(pc, word, source_address, self.functional.ub())?;
                    let instruction = C220CoreInstruction::VectorLoadAddress(step);
                    self.issue_vector_at(tick, &instruction)?;
                    self.functional.commit_c220_vector_issue(None);
                    instruction
                }
                _ if C220MovemaskHint::from_word(word).is_some() => {
                    let step = crate::sim::c220::scalar::execute_movemask(
                        self.functional.scalar_mut().machine_mut(),
                        pc,
                        word,
                    )?;
                    let instruction = C220CoreInstruction::VectorMovemask(step);
                    self.issue_vector_at(tick, &instruction)?;
                    self.functional.commit_c220_vector_issue(None);
                    instruction
                }
                _ if C220VectorControlInstruction::decode(word).is_some() => {
                    let decoded =
                        C220VectorControlInstruction::decode(word).expect("matched decode");
                    let instruction = C220CoreInstruction::VectorControl {
                        pc,
                        word,
                        instruction: decoded,
                    };
                    self.issue_vector_at(tick, &instruction)?;
                    self.functional.commit_c220_vector_issue(None);
                    instruction
                }
                _ if C220NoEffectVectorInstruction::decode(word).is_some() => {
                    let decoded =
                        C220NoEffectVectorInstruction::decode(word).expect("matched decode");
                    let machine = self.functional.scalar().machine();
                    let control = machine.xregs()[usize::from(decoded.control_register)];
                    let repeat_count = decode_c220_repeat_masks(
                        machine
                            .spr_value(3)
                            .ok_or(C220VectorError::MissingMaskState)?,
                        machine
                            .spr_value(100)
                            .ok_or(C220VectorError::MissingMaskState)?,
                        machine
                            .spr_value(101)
                            .ok_or(C220VectorError::MissingMaskState)?,
                        decoded.lane_count(),
                        (control >> 56) as u8,
                    )?
                    .len();
                    let instruction = C220CoreInstruction::VectorNoEffect {
                        pc,
                        word,
                        instruction: decoded,
                        repeat_count,
                        lane_groups: decoded.lane_groups(),
                    };
                    self.issue_vector_at(tick, &instruction)?;
                    self.functional.commit_c220_vector_issue(None);
                    instruction
                }
                _ if C220MovevInstruction::decode(word).is_some() => {
                    let step = self.functional.preview_c220_movev_word(word)?;
                    let instruction = C220CoreInstruction::VectorMove(step);
                    self.issue_vector_at(tick, &instruction)?;
                    self.functional.commit_c220_vector_issue(None);
                    instruction
                }
                _ if C220NchwInstruction::decode(word).is_some() => {
                    if let Some(resume_tick) = self.vector.pending_load_va_drain_tick()
                        && tick < resume_tick
                    {
                        return Ok(C220CoreStep::Stalled(C220Stall {
                            tick,
                            pc,
                            resume_tick,
                            cause: C220StallCause::VectorDependency,
                        }));
                    }
                    let decoded = C220NchwInstruction::decode(word).expect("matched decode");
                    let control = self.functional.scalar().machine().xregs()
                        [usize::from(decoded.control_register)];
                    let step =
                        plan_c220_nchw_issue(pc, word, control, &self.va, self.functional.ub())?;
                    let destination = step.rows.first().map(|rows| rows.destination[0]);
                    let instruction = C220CoreInstruction::VectorNchw(step);
                    self.issue_vector_at(tick, &instruction)?;
                    self.functional.commit_c220_vector_issue(destination);
                    instruction
                }
                _ if C220MoveMaskInstruction::decode(word).is_some() => {
                    let decoded = C220MoveMaskInstruction::decode(word).expect("matched decode");
                    if matches!(decoded.direction, C220MoveMaskDirection::FromMemory)
                        && let Some(resume_tick) = self.pending_vector_drain_tick()
                        && tick < resume_tick
                    {
                        return Ok(C220CoreStep::Stalled(C220Stall {
                            tick,
                            pc,
                            resume_tick,
                            cause: C220StallCause::VectorDependency,
                        }));
                    }
                    let step = plan_c220_move_mask_issue(
                        pc,
                        word,
                        self.functional.scalar().machine().xregs(),
                        self.functional.ub(),
                    )?;
                    let destination =
                        matches!(step.instruction.direction, C220MoveMaskDirection::ToMemory)
                            .then_some(step.address);
                    let instruction = C220CoreInstruction::VectorMoveMask(step);
                    self.issue_vector_at(tick, &instruction)?;
                    self.functional.commit_c220_vector_issue(destination);
                    instruction
                }
                _ if C220CompareMaskInstruction::decode(word).is_some() => {
                    let decoded = C220CompareMaskInstruction::decode(word).expect("matched decode");
                    let machine = self.functional.scalar().machine();
                    let registers = machine.xregs();
                    let step = plan_c220_compare_mask_issue(
                        pc,
                        word,
                        registers[usize::from(decoded.control_register)],
                        C220VectorMaskState {
                            control: machine
                                .spr_value(3)
                                .ok_or(C220VectorError::MissingMaskState)?,
                            low: machine
                                .spr_value(100)
                                .ok_or(C220VectorError::MissingMaskState)?,
                            high: machine
                                .spr_value(101)
                                .ok_or(C220VectorError::MissingMaskState)?,
                        },
                        registers,
                        self.functional.ub(),
                    )?;
                    let instruction = C220CoreInstruction::VectorCompareMask(step);
                    self.issue_vector_at(tick, &instruction)?;
                    self.functional.commit_c220_vector_issue(None);
                    instruction
                }
                _ if C220SelectInstruction::decode(word).is_some() => {
                    let decoded = C220SelectInstruction::decode(word).expect("matched decode");
                    let machine = self.functional.scalar().machine();
                    let registers = machine.xregs();
                    let control_value = registers[usize::from(decoded.control_register)];
                    let mode = C220SelectMode::decode(control_value).ok_or(
                        C220VectorError::UnsupportedSelectMode(((control_value >> 48) & 3) as u8),
                    )?;
                    if matches!(mode, C220SelectMode::TensorTensor)
                        && self.vector.has_pending_compare_mask_write()
                    {
                        return Ok(C220CoreStep::Stalled(C220Stall {
                            tick,
                            pc,
                            resume_tick: self.vector.pending_drain_tick().unwrap_or(tick),
                            cause: C220StallCause::VectorDependency,
                        }));
                    }
                    let step = plan_c220_select_issue(
                        pc,
                        word,
                        control_value,
                        C220VectorMaskState {
                            control: machine
                                .spr_value(3)
                                .ok_or(C220VectorError::MissingMaskState)?,
                            low: machine
                                .spr_value(100)
                                .ok_or(C220VectorError::MissingMaskState)?,
                            high: machine
                                .spr_value(101)
                                .ok_or(C220VectorError::MissingMaskState)?,
                        },
                        self.vector.compare_mask(),
                        registers,
                        self.functional.ub(),
                    )?;
                    let destination = step.addresses.destination;
                    let instruction = C220CoreInstruction::VectorSelect(step);
                    self.issue_vector_at(tick, &instruction)?;
                    self.functional.commit_c220_vector_issue(Some(destination));
                    instruction
                }
                _ if C220PackedCompareInstruction::decode(word).is_some() => {
                    let decoded =
                        C220PackedCompareInstruction::decode(word).expect("matched decode");
                    let registers = self.functional.scalar().machine().xregs();
                    let control = registers[usize::from(decoded.control_register)];
                    let step = plan_c220_packed_compare_issue(
                        pc,
                        word,
                        control,
                        registers,
                        self.functional.ub(),
                    )?;
                    let destination = step.addresses.destination;
                    let instruction = C220CoreInstruction::VectorPackedCompare(step);
                    self.issue_vector_at(tick, &instruction)?;
                    self.functional.commit_c220_vector_issue(Some(destination));
                    instruction
                }
                _ if C220ReductionInstruction::decode(word).is_some() => {
                    let step = self.functional.preview_c220_reduction_word(word)?;
                    let produces_output =
                        !step.instruction.writes_accumulator() && !step.iteration_masks.is_empty();
                    let destination = step.addresses.destination;
                    let instruction = C220CoreInstruction::VectorReduction(step);
                    self.issue_vector_at(tick, &instruction)?;
                    self.functional
                        .commit_c220_vector_issue(produces_output.then_some(destination));
                    instruction
                }
                _ if C220SortInstruction::decode(word).is_some() => {
                    let step = self.functional.preview_c220_sort_word(word)?;
                    let destination = step.addresses.destination;
                    let produces_output = step.repeat_count != 0;
                    let instruction = C220CoreInstruction::VectorSort(step);
                    self.issue_vector_at(tick, &instruction)?;
                    self.functional
                        .commit_c220_vector_issue(produces_output.then_some(destination));
                    instruction
                }
                _ if C220MergeInstruction::decode(word).is_some() => {
                    if let Some(resume_tick) = self.vector.pending_drain_tick()
                        && tick < resume_tick
                    {
                        return Ok(C220CoreStep::Stalled(C220Stall {
                            tick,
                            pc,
                            resume_tick,
                            cause: C220StallCause::VectorDependency,
                        }));
                    }
                    let step = self.functional.preview_c220_merge_word(word)?;
                    let destination = step
                        .repeats
                        .first()
                        .map(|repeat| repeat.destination_address);
                    if step.repeat_count() == 0 {
                        self.functional
                            .scalar_mut()
                            .machine_mut()
                            .set_spr_value(17, 0)?;
                    } else {
                        self.vmsu
                            .issue_at(tick, step.clone(), self.functional.ub())?;
                    }
                    self.functional.commit_c220_vector_issue(destination);
                    C220CoreInstruction::VectorMerge(step)
                }
                _ if C220TernaryInstruction::decode(word).is_some() => {
                    let step = self.functional.preview_c220_ternary_word(word)?;
                    let destination = step.addresses.destination;
                    let instruction = C220CoreInstruction::VectorTernary(step);
                    self.issue_vector_at(tick, &instruction)?;
                    self.functional.commit_c220_vector_issue(Some(destination));
                    instruction
                }
                _ if C220AxpyInstruction::decode(word).is_some() => {
                    let step = self.functional.preview_c220_axpy_word(word)?;
                    let destination = step.addresses.destination;
                    let instruction = C220CoreInstruction::VectorAxpy(step);
                    self.issue_vector_at(tick, &instruction)?;
                    self.functional.commit_c220_vector_issue(Some(destination));
                    instruction
                }
                _ if C220SpecialUnaryInstruction::decode(word).is_some() => {
                    let step = self.functional.preview_c220_special_unary_word(word)?;
                    let destination = step.addresses.destination;
                    let instruction = C220CoreInstruction::VectorSpecialUnary(step);
                    self.issue_vector_at(tick, &instruction)?;
                    self.functional.commit_c220_vector_issue(Some(destination));
                    instruction
                }
                _ if C220FusedInstruction::decode(word).is_some() => {
                    let step = self.functional.preview_c220_fused_word(word)?;
                    let destination = step.addresses.destination;
                    let instruction = C220CoreInstruction::VectorFused(step);
                    self.issue_vector_at(tick, &instruction)?;
                    self.functional.commit_c220_vector_issue(Some(destination));
                    instruction
                }
                _ if C220ConversionInstruction::decode(word).is_some() => {
                    let step = self.functional.preview_c220_conversion_word(word)?;
                    let destination = step.addresses.destination;
                    let instruction = C220CoreInstruction::VectorConversion(step);
                    self.issue_vector_at(tick, &instruction)?;
                    self.functional.commit_c220_vector_issue(Some(destination));
                    instruction
                }
                _ if C220GatherInstruction::decode(word).is_some() => {
                    let step = self.functional.preview_c220_gather_word(word)?;
                    let destination = step.destination_address;
                    let instruction = C220CoreInstruction::VectorGather(step);
                    self.issue_vector_at(tick, &instruction)?;
                    self.functional.commit_c220_vector_issue(Some(destination));
                    instruction
                }
                _ if C220VecArithmeticHint::from_word(word).is_some() => {
                    let step = self.functional.preview_c220_vector_word(word)?;
                    let destination_address = step.addresses.destination;
                    let instruction = C220CoreInstruction::VectorArithmetic(step);
                    self.issue_vector_at(tick, &instruction)?;
                    self.functional
                        .commit_c220_vector_issue(Some(destination_address));
                    instruction
                }
                _ if C220VectorScalarInstruction::decode(word).is_some() => {
                    let step = self.functional.preview_c220_vector_scalar_word(word)?;
                    let destination_address = step.addresses.destination;
                    let instruction = C220CoreInstruction::VectorScalar(step);
                    self.issue_vector_at(tick, &instruction)?;
                    self.functional
                        .commit_c220_vector_issue(Some(destination_address));
                    instruction
                }
                _ if C220ShiftInstruction::decode(word).is_some() => {
                    let step = self.functional.preview_c220_shift_word(word)?;
                    let destination_address = step.addresses.destination;
                    let instruction = C220CoreInstruction::VectorShift(step);
                    self.issue_vector_at(tick, &instruction)?;
                    self.functional
                        .commit_c220_vector_issue(Some(destination_address));
                    instruction
                }
                _ if C220CopyInstruction::decode(word).is_some() => {
                    let step = self.functional.preview_c220_copy_word(word)?;
                    let destination_address = step.addresses.destination;
                    let instruction = C220CoreInstruction::VectorCopy(step);
                    self.issue_vector_at(tick, &instruction)?;
                    self.functional
                        .commit_c220_vector_issue(Some(destination_address));
                    instruction
                }
                _ if C220BroadcastInstruction::decode(word).is_some() => {
                    let step = self.functional.preview_c220_broadcast_word(word)?;
                    let destination_address = step.destination_address;
                    let has_repeats = step.control.repeat_count != 0;
                    let instruction = C220CoreInstruction::VectorBroadcast(step);
                    self.issue_vector_at(tick, &instruction)?;
                    self.functional
                        .commit_c220_vector_issue(has_repeats.then_some(destination_address));
                    instruction
                }
                _ if C220TransposeInstruction::decode(word).is_some() => {
                    let step = self.functional.preview_c220_transpose_word(word)?;
                    let destination_address = step.destination_address;
                    let instruction = C220CoreInstruction::VectorTranspose(step);
                    self.issue_vector_at(tick, &instruction)?;
                    self.functional
                        .commit_c220_vector_issue(Some(destination_address));
                    instruction
                }
                _ if matches!(
                    PipelineBarrierStep::decode(Architecture::Dav2201, pc, word),
                    Some(PipelineBarrierStep {
                        scope: PipelineBarrierScope::All,
                        ..
                    })
                ) =>
                {
                    if let Some((resume_tick, cause)) = self.pending_compute_drain()
                        && tick < resume_tick
                    {
                        return Ok(C220CoreStep::Stalled(C220Stall {
                            tick,
                            pc,
                            resume_tick,
                            cause,
                        }));
                    }
                    C220CoreInstruction::Barrier(self.functional.step_barrier_word(word)?)
                }
                _ => {
                    let timing = if let Some(hint) = C220ScalarConversionHint::from_word(word) {
                        Some(
                            C220ScalarTimingTicket::for_conversion(tick, hint)
                                .ok_or(C220CoreError::TimeOverflow)?,
                        )
                    } else {
                        None
                    };
                    let step = self
                        .functional
                        .step_scalar_word_with_ub(word, &mut self.memory)?;
                    if let Some(ticket) = timing {
                        self.scalar_timing.issue(ticket);
                    }
                    C220CoreInstruction::Scalar { step, timing }
                }
            }
        };
        self.mte2.finish_other_at(tick)?;
        Ok(C220CoreStep::Executed { tick, instruction })
    }

    fn issue_vector_at(
        &mut self,
        tick: u64,
        instruction: &C220CoreInstruction,
    ) -> Result<(), C220CoreError> {
        let uops = instruction.vector_uops()?;
        let stores = instruction
            .vector_stores()
            .expect("vector instruction has stores");
        let compute = match instruction {
            C220CoreInstruction::VectorLoadAddress(issue) => {
                Some(C220VectorReadIssue::LoadVa(issue))
            }
            C220CoreInstruction::VectorArithmetic(issue) => {
                Some(C220VectorReadIssue::Arithmetic(issue))
            }
            C220CoreInstruction::VectorScalar(issue) => {
                Some(C220VectorReadIssue::VectorScalar(issue))
            }
            C220CoreInstruction::VectorShift(issue) => Some(C220VectorReadIssue::Shift(issue)),
            C220CoreInstruction::VectorCopy(issue) => Some(C220VectorReadIssue::Copy(issue)),
            C220CoreInstruction::VectorBroadcast(issue) => {
                Some(C220VectorReadIssue::Broadcast(issue))
            }
            C220CoreInstruction::VectorTranspose(issue) => {
                Some(C220VectorReadIssue::Transpose(issue))
            }
            C220CoreInstruction::VectorCompareMask(issue) => {
                Some(C220VectorReadIssue::CompareMask(issue))
            }
            C220CoreInstruction::VectorMoveMask(issue) => {
                Some(C220VectorReadIssue::MoveMask(issue))
            }
            C220CoreInstruction::VectorSelect(issue) => Some(C220VectorReadIssue::Select(issue)),
            C220CoreInstruction::VectorPackedCompare(issue) => {
                Some(C220VectorReadIssue::PackedCompare(issue))
            }
            C220CoreInstruction::VectorReduction(issue) => {
                Some(C220VectorReadIssue::Reduction(issue))
            }
            C220CoreInstruction::VectorSort(issue) => Some(C220VectorReadIssue::Sort(issue)),
            C220CoreInstruction::VectorTernary(issue) => Some(C220VectorReadIssue::Ternary(issue)),
            C220CoreInstruction::VectorAxpy(issue) => Some(C220VectorReadIssue::Axpy(issue)),
            C220CoreInstruction::VectorSpecialUnary(issue) => {
                Some(C220VectorReadIssue::SpecialUnary(issue))
            }
            C220CoreInstruction::VectorConversion(issue) => {
                Some(C220VectorReadIssue::Conversion(issue))
            }
            C220CoreInstruction::VectorFused(issue) => Some(C220VectorReadIssue::Fused(issue)),
            C220CoreInstruction::VectorGather(issue) => Some(C220VectorReadIssue::Gather(issue)),
            C220CoreInstruction::VectorNchw(issue) => Some(C220VectorReadIssue::Nchw(issue)),
            _ => None,
        };
        self.vector.issue_classified_at(
            tick,
            &uops,
            stores,
            compute,
            instruction.vector_queue_class(),
        )?;
        Ok(())
    }

    fn commit_ready_output_at(&mut self, tick: u64) -> Result<(), C220CoreError> {
        if let Some(pending) = self.pending_output.as_ref()
            && tick >= pending.data_ready_tick
        {
            self.memory.write_segments_at(&pending.prepared.writes)?;
            self.pending_output = None;
        }
        Ok(())
    }

    fn step_mte1_at(
        &mut self,
        tick: u64,
        pc: u64,
        word: u32,
    ) -> Result<C220CoreStep, C220CoreError> {
        let instruction = if let Some(decoded) =
            C220Load2dInstruction::decode(word).filter(|instruction| instruction.is_mte1())
        {
            let next_accept_tick = self.mte1.next_accept_tick();
            if tick < next_accept_tick {
                return Ok(C220CoreStep::Stalled(C220Stall {
                    tick,
                    pc,
                    resume_tick: next_accept_tick,
                    cause: C220StallCause::Mte1IssueRate,
                }));
            }
            let transfer = decoded
                .capture(self.functional.scalar().machine().xregs())
                .map_err(C220Load2dTransferError::from)?;
            let ticket = self.mte1.preview_issue(tick, transfer)?;
            let prepared = prepare_c220_load2d(&self.local_memory, transfer)?;
            let result = prepared.result;
            self.mte1.issue(&ticket)?;
            self.pending_load2d.push(C220PendingLoad2d {
                data_ready_tick: ticket.data_ready_tick,
                prepared,
            });
            self.functional.commit_c220_sequential_issue();
            C220CoreInstruction::Mte1Load2d {
                transfer,
                result,
                ticket: Box::new(ticket),
            }
        } else {
            let flag = FlagInstruction::decode(Architecture::Dav2201, word)
                .expect("matched C220 MTE1 flag")
                .resolve(pc, self.functional.scalar().machine().xregs());
            match flag.instruction.operation {
                FlagOperation::Set => self.mte1.set_event(tick, flag.flag_id),
                FlagOperation::Wait => {
                    let ready_tick = self.mte1.event_ready_tick(flag.flag_id).ok_or(
                        C220Mte1TimingError::MissingEvent {
                            event_id: flag.flag_id,
                        },
                    )?;
                    if tick < ready_tick {
                        return Ok(C220CoreStep::Stalled(C220Stall {
                            tick,
                            pc,
                            resume_tick: ready_tick,
                            cause: C220StallCause::Mte1Dependency,
                        }));
                    }
                    self.mte1.wait_event(tick, flag.flag_id)?;
                }
            }
            self.functional.commit_c220_sequential_issue();
            C220CoreInstruction::Mte1Flag(flag)
        };
        self.mte2.finish_other_at(tick)?;
        Ok(C220CoreStep::Executed { tick, instruction })
    }

    fn step_hardware_flag_at(
        &mut self,
        tick: u64,
        pc: u64,
        instruction: C220HardwareFlagInstruction,
    ) -> Result<C220CoreStep, C220CoreError> {
        let step = instruction.resolve(pc, self.functional.scalar().machine().xregs())?;
        let token_ready_tick = match instruction.operation {
            C220HardwareFlagOperation::Set => {
                let checkpoint_tick = if instruction.trigger {
                    tick
                } else {
                    match (instruction.source_pipe, instruction.memory) {
                        (C220HardwareFlagSourcePipe::Mte1, C220MatrixMemory::L0a) => self
                            .mte1
                            .pending_data_ready_tick(
                                crate::isa::c220::mte1::C220Load2dDestination::L0a,
                            )
                            .unwrap_or(tick),
                        (C220HardwareFlagSourcePipe::Mte1, C220MatrixMemory::L0b) => self
                            .mte1
                            .pending_data_ready_tick(
                                crate::isa::c220::mte1::C220Load2dDestination::L0b,
                            )
                            .unwrap_or(tick),
                        (source_pipe, memory) => {
                            return Err(C220CoreError::UnsupportedHardwareFlagCheckpoint {
                                source_pipe,
                                memory,
                            });
                        }
                    }
                };
                Some(
                    self.hardware_flags
                        .schedule_set(step, checkpoint_tick.max(tick))?,
                )
            }
            C220HardwareFlagOperation::Wait => {
                let ready_tick = self.hardware_flags.wait_ready_tick(step)?.ok_or(
                    C220HardwareFlagTimingError::MissingToken {
                        event_id: step.event_id,
                    },
                )?;
                if tick < ready_tick {
                    return Ok(C220CoreStep::Stalled(C220Stall {
                        tick,
                        pc,
                        resume_tick: ready_tick,
                        cause: C220StallCause::HardwareFlagDependency,
                    }));
                }
                self.hardware_flags.consume_wait(step)?;
                None
            }
        };
        self.functional.commit_c220_sequential_issue();
        self.mte2.finish_other_at(tick)?;
        Ok(C220CoreStep::Executed {
            tick,
            instruction: C220CoreInstruction::HardwareFlag {
                step,
                token_ready_tick,
            },
        })
    }

    fn commit_ready_load2d_at(&mut self, tick: u64) -> Result<(), C220CoreError> {
        let mut index = 0;
        while index < self.pending_load2d.len() {
            if self.pending_load2d[index].data_ready_tick <= tick {
                self.pending_load2d
                    .remove(index)
                    .prepared
                    .commit(&mut self.local_memory)?;
            } else {
                index += 1;
            }
        }
        Ok(())
    }

    fn commit_retired_cube(
        &mut self,
        retired: &[crate::sim::c220::cube::C220CubeTicket],
    ) -> Result<(), C220CoreError> {
        self.last_cube_outcomes.clear();
        for ticket in retired {
            let index = self
                .pending_cube
                .iter()
                .position(|pending| pending.accept_tick == ticket.accept_tick)
                .ok_or(C220CubeTimingError::TicketMismatch)?;
            let pending = self.pending_cube.remove(index);
            let outcome = pending.prepared.outcome;
            pending.prepared.commit(&mut self.local_memory)?;
            let machine = self.functional.scalar_mut().machine_mut();
            let spr2 = update_cube_status_spr2(machine.spr2(), pending.pc, outcome);
            machine.set_spr_value(2, spr2)?;
            self.last_cube_outcomes.push(outcome);
        }
        Ok(())
    }

    fn pending_vector_visibility_tick(&self) -> Option<u64> {
        self.vector
            .pending_visibility_tick()
            .into_iter()
            .chain(self.vmsu.pending_visibility_tick())
            .max()
    }

    fn pending_vector_drain_tick(&self) -> Option<u64> {
        self.vector
            .pending_drain_tick()
            .into_iter()
            .chain(self.vmsu.pending_drain_tick())
            .max()
    }

    fn pending_compute_drain(&self) -> Option<(u64, C220StallCause)> {
        [
            self.pending_vector_drain_tick()
                .map(|tick| (tick, C220StallCause::VectorDependency)),
            self.cube
                .pending_drain_tick()
                .map(|tick| (tick, C220StallCause::CubeDependency)),
            self.mte1
                .pending_drain_tick()
                .map(|tick| (tick, C220StallCause::Mte1Dependency)),
        ]
        .into_iter()
        .flatten()
        .max_by_key(|(tick, _)| *tick)
    }
}

fn is_c220_vector_word(word: u32) -> bool {
    C220MoveVaInstruction::decode(word).is_some()
        || C220LoadVaInstruction::decode(word).is_some()
        || C220MovemaskHint::from_word(word).is_some()
        || C220VectorControlInstruction::decode(word).is_some()
        || C220NoEffectVectorInstruction::decode(word).is_some()
        || C220MovevInstruction::decode(word).is_some()
        || C220NchwInstruction::decode(word).is_some()
        || C220MoveMaskInstruction::decode(word).is_some()
        || C220CompareMaskInstruction::decode(word).is_some()
        || C220SelectInstruction::decode(word).is_some()
        || C220PackedCompareInstruction::decode(word).is_some()
        || C220ReductionInstruction::decode(word).is_some()
        || C220SortInstruction::decode(word).is_some()
        || C220MergeInstruction::decode(word).is_some()
        || C220TernaryInstruction::decode(word).is_some()
        || C220AxpyInstruction::decode(word).is_some()
        || C220SpecialUnaryInstruction::decode(word).is_some()
        || C220FusedInstruction::decode(word).is_some()
        || C220ConversionInstruction::decode(word).is_some()
        || C220GatherInstruction::decode(word).is_some()
        || C220VecArithmeticHint::from_word(word).is_some()
        || C220VectorScalarInstruction::decode(word).is_some()
        || C220ShiftInstruction::decode(word).is_some()
        || C220CopyInstruction::decode(word).is_some()
        || C220BroadcastInstruction::decode(word).is_some()
        || C220TransposeInstruction::decode(word).is_some()
}

fn is_mte2_word(word: u32) -> bool {
    is_mte2_transfer(word)
        || FlagInstruction::decode(Architecture::Dav2201, word).is_some_and(|instruction| {
            instruction.source_pipe_code == 4 && matches!(instruction.trigger_pipe_code, 0 | 1)
        })
}

fn is_mte1_word(word: u32) -> bool {
    C220Load2dInstruction::decode(word).is_some_and(|instruction| instruction.is_mte1())
        || FlagInstruction::decode(Architecture::Dav2201, word).is_some_and(|instruction| {
            instruction.source_pipe_code == 3 && instruction.trigger_pipe_code == 2
        })
}

fn is_mte3_word(word: u32) -> bool {
    C220DmaMovDescriptor::is_word(word)
        || FlagInstruction::decode(Architecture::Dav2201, word).is_some_and(|instruction| {
            matches!(
                (instruction.source_pipe_code, instruction.trigger_pipe_code),
                (1, 5) | (1, 4) | (5, 1)
            )
        })
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
