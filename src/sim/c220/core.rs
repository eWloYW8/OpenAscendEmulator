use thiserror::Error;

use crate::architecture::Architecture;
use crate::image::loader::{DeviceKernelFetchError, LoadedDeviceKernel};
use crate::isa::c220::mte::C220DmaMovDescriptor;
use crate::isa::c220::scalar::C220ScalarConversionHint;
use crate::isa::c220::vector::{C220MovevInstruction, C220ShiftInstruction, C220VecArithmeticHint};
use crate::isa::c220::vector_scalar::C220VectorScalarInstruction;
use crate::isa::flow::{FlagInstruction, FlagOperation, PipelineBarrierScope, PipelineBarrierStep};
use crate::memory::hbm_pv_memory::HbmPvMemory;
use crate::memory::mapped::{MappedMemory, MappedMemoryError};
use crate::sim::c220::mte::output::{C220OutputAction, C220OutputStep};
use crate::sim::c220::mte::transfer::C220PreparedOutput;
use crate::sim::c220::mte::uop::C220DmaUopRequest;
use crate::sim::c220::timing::mte2::{
    C220Mte2TimingRules, C220Stall, C220StallCause, C220TimedMte2Core, C220TimedMte2Step,
    C220TimingError, is_mte2_transfer,
};
use crate::sim::c220::timing::mte3::{
    C220Mte3Ticket, C220Mte3TimingError, C220Mte3TimingRules, C220TimedMte3Lane,
};
use crate::sim::c220::timing::scalar::{C220ScalarTimingLane, C220ScalarTimingTicket};
use crate::sim::c220::vector::pipeline::{
    C220VectorAdvanceError, C220VectorPipeline, C220VectorPipelineError, C220VectorTimingRules,
};
use crate::sim::c220::vector::read::C220VectorReadIssue;
use crate::sim::c220::vector::scalar::C220VectorScalarIssue;
use crate::sim::c220::vector::shift::C220ShiftIssue;
use crate::sim::c220::vector::timing::{
    C220VectorUop, C220VectorUopRelease, C220VectorUopStages, C220VectorWritePlan,
    C220VectorWritePlanError,
};
use crate::sim::c220::vector::{C220MovevStep, C220VectorArithmeticIssue, C220VectorStore};
use crate::sim::machine::ScalarInstructionError;
use crate::sim::mte_stepper::{MteCoreStepper, MteStepperError};
use crate::sim::scalar_bus::UbScalarBusError;
use crate::sim::stepper::ScalarProgramStep;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum C220CoreInstruction {
    Scalar {
        step: ScalarProgramStep,
        timing: Option<C220ScalarTimingTicket>,
    },
    Barrier(ScalarProgramStep),
    Mte2(C220TimedMte2Step),
    VectorMove(C220MovevStep),
    VectorArithmetic(C220VectorArithmeticIssue),
    VectorScalar(C220VectorScalarIssue),
    VectorShift(C220ShiftIssue),
    Mte3 {
        step: C220OutputStep,
        requests: Vec<C220DmaUopRequest>,
        ticket: Option<C220Mte3Ticket>,
    },
}

struct VectorUopInputs {
    pc: u64,
    repeat_count: usize,
    lane_groups: u8,
    stages: C220VectorUopStages,
    plan: C220VectorWritePlan,
}

impl C220CoreInstruction {
    fn vector_stores(&self) -> Option<&[C220VectorStore]> {
        match self {
            Self::VectorMove(step) => Some(&step.stores),
            Self::VectorArithmetic(step) => Some(&step.write_targets),
            Self::VectorScalar(step) => Some(&step.write_targets),
            Self::VectorShift(step) => Some(&step.write_targets),
            _ => None,
        }
    }

    pub fn vector_write_plan(
        &self,
    ) -> Result<Option<C220VectorWritePlan>, C220VectorWritePlanError> {
        match self {
            Self::VectorMove(step) => C220VectorWritePlan::from_stores(&step.stores).map(Some),
            Self::VectorArithmetic(step) => {
                C220VectorWritePlan::from_stores(&step.write_targets).map(Some)
            }
            Self::VectorScalar(step) => {
                C220VectorWritePlan::from_stores(&step.write_targets).map(Some)
            }
            Self::VectorShift(step) => {
                C220VectorWritePlan::from_stores(&step.write_targets).map(Some)
            }
            _ => Ok(None),
        }
    }

    fn vector_uop_stages(&self) -> Option<C220VectorUopStages> {
        match self {
            Self::VectorMove(step) => C220VectorUopStages::movev(step.instruction),
            Self::VectorArithmetic(step) => C220VectorUopStages::vector_arithmetic(step.hint),
            Self::VectorScalar(step) => Some(C220VectorUopStages::vector_scalar(step.instruction)),
            Self::VectorShift(_) => Some(C220VectorUopStages::shift()),
            _ => None,
        }
    }

    /// Describes admitted vector work in 64-lane groups.
    pub fn vector_uops(&self) -> Result<Vec<C220VectorUop>, C220CoreError> {
        let Some(inputs) = self.vector_uop_inputs()? else {
            return Ok(Vec::new());
        };
        let mut uops = Vec::new();
        for repeat_index in 0..inputs.repeat_count {
            for lane_group in 0..inputs.lane_groups {
                if let Some(writeback_ticks) = inputs
                    .plan
                    .writeback_ticks_for_lane_group(repeat_index, lane_group)
                {
                    uops.push(C220VectorUop {
                        pc: inputs.pc,
                        repeat_index,
                        lane_group,
                        stages: inputs.stages,
                        writeback_ticks,
                        writes_ub: true,
                    });
                }
            }
        }
        if uops.is_empty() {
            uops.push(C220VectorUop {
                pc: inputs.pc,
                repeat_index: 0,
                lane_group: 0,
                stages: C220VectorUopStages::empty(),
                writeback_ticks: 1,
                writes_ub: false,
            });
        }
        Ok(uops)
    }

    fn vector_uop_inputs(&self) -> Result<Option<VectorUopInputs>, C220CoreError> {
        let (pc, repeat_count, lane_groups) = match self {
            Self::VectorMove(step) if step.instruction.supported_element_bytes() == Some(2) => {
                (step.pc, step.iteration_masks.len(), 2)
            }
            Self::VectorMove(step) if step.instruction.supported_element_bytes() == Some(4) => {
                (step.pc, step.iteration_masks.len(), 1)
            }
            Self::VectorArithmetic(step) => (
                step.pc,
                step.iteration_masks.len(),
                4 / step.result_element_bytes,
            ),
            Self::VectorScalar(step) => (
                step.pc,
                step.iteration_masks.len(),
                4 / step.instruction.dtype.element_bytes(),
            ),
            Self::VectorShift(step) => (
                step.pc,
                step.iteration_masks.len(),
                4 / step.instruction.element_bytes,
            ),
            _ => return Ok(None),
        };
        let Some(stages) = self.vector_uop_stages() else {
            return Ok(None);
        };
        let Some(plan) = self.vector_write_plan()? else {
            return Ok(None);
        };
        Ok(Some(VectorUopInputs {
            pc,
            repeat_count,
            lane_groups,
            stages,
            plan,
        }))
    }
}

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
    Mte(#[from] MteStepperError),
    #[error(transparent)]
    Mte3Timing(#[from] C220Mte3TimingError),
    #[error(transparent)]
    VectorPlan(#[from] C220VectorWritePlanError),
    #[error(transparent)]
    VectorPipeline(#[from] C220VectorPipelineError),
    #[error(transparent)]
    VectorAdvance(#[from] C220VectorAdvanceError),
    #[error(transparent)]
    OutputMemory(#[from] MappedMemoryError),
    #[error(transparent)]
    Scalar(#[from] ScalarInstructionError<UbScalarBusError<MappedMemoryError>>),
    #[error("timed core stalled without a future resume tick")]
    NonprogressingStall,
    #[error("tick counter overflowed")]
    TimeOverflow,
}

pub struct C220Core {
    execution: C220TimedMte2Core,
    scalar_timing: C220ScalarTimingLane,
    mte3: C220TimedMte3Lane,
    vector: C220VectorPipeline,
    last_vector_releases: Vec<C220VectorUopRelease>,
    pending_output: Option<C220PendingOutput>,
    memory: MappedMemory,
}

struct C220PendingOutput {
    data_ready_tick: u64,
    prepared: C220PreparedOutput,
}

impl C220Core {
    pub fn new(
        execution: MteCoreStepper,
        memory: MappedMemory,
        timing: C220CoreTimingRules,
    ) -> Result<Self, C220CoreError> {
        Ok(Self {
            execution: C220TimedMte2Core::new(execution, timing.mte2)?,
            scalar_timing: C220ScalarTimingLane::default(),
            mte3: C220TimedMte3Lane::new(timing.mte3),
            vector: C220VectorPipeline::new(timing.vector),
            last_vector_releases: Vec::new(),
            pending_output: None,
            memory,
        })
    }

    pub const fn execution(&self) -> &C220TimedMte2Core {
        &self.execution
    }

    pub const fn scalar_timing(&self) -> &C220ScalarTimingLane {
        &self.scalar_timing
    }

    pub const fn memory(&self) -> &MappedMemory {
        &self.memory
    }

    pub const fn vector_pipeline(&self) -> &C220VectorPipeline {
        &self.vector
    }

    pub fn last_vector_releases(&self) -> &[C220VectorUopRelease] {
        &self.last_vector_releases
    }

    pub fn memory_mut(&mut self) -> &mut MappedMemory {
        &mut self.memory
    }

    pub fn pending_output_ready_tick(&self) -> Option<u64> {
        self.pending_output
            .as_ref()
            .map(|pending| pending.data_ready_tick)
    }

    pub fn advance_to(&mut self, tick: u64) -> Result<Option<C220Stall>, C220CoreError> {
        let gate = self.execution.gate_other_at(tick)?;
        self.scalar_timing.advance_to(tick);
        self.last_vector_releases = self.vector.advance_to(tick, self.execution.core_mut())?;
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
        let pc = self.execution.core().scalar().pc();
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
            let stop = if self.execution.core().scalar().is_halted() {
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
                    next_pc: self.execution.core().scalar().pc(),
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
                pc: self.execution.core().scalar().pc(),
                resume_tick,
                cause: C220StallCause::ScalarDependency,
            }));
        }
        if is_mte2_word(word) {
            return match self.execution.step_at(tick, word, &self.memory)? {
                C220TimedMte2Step::Stalled(stall) => Ok(C220CoreStep::Stalled(stall)),
                result @ C220TimedMte2Step::Executed { .. } => Ok(C220CoreStep::Executed {
                    tick,
                    instruction: C220CoreInstruction::Mte2(result),
                }),
            };
        }
        let pc = self.execution.core().scalar().pc();
        let instruction = if is_mte3_word(word) {
            if let Some(flag) = FlagInstruction::decode(Architecture::Dav2201, word)
                && flag.source_pipe_code == 1
                && flag.trigger_pipe_code == 5
                && flag.operation == FlagOperation::Wait
                && let Some(resume_tick) = self.vector.pending_visibility_tick()
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
                    .resolve(pc, self.execution.core().scalar().machine().xregs())
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
                let plan = self.execution.core().preview_c220_mte3_transfer(word)?;
                let (ticket, requests) = self.mte3.preview_issue(tick, plan)?;
                if self.pending_output.is_some() {
                    return Err(C220Mte3TimingError::TicketMismatch.into());
                }
                (Some(ticket), requests)
            } else {
                (None, Vec::new())
            };
            let (step, prepared) = self
                .execution
                .core_mut()
                .step_c220_output_word_deferred(word)?;
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
                _ if C220MovevInstruction::decode(word).is_some() => {
                    let step = self.execution.core().preview_c220_movev_word(word)?;
                    let instruction = C220CoreInstruction::VectorMove(step);
                    self.issue_vector_at(tick, &instruction)?;
                    self.execution.core_mut().commit_c220_vector_issue(None);
                    instruction
                }
                _ if C220VecArithmeticHint::from_word(word).is_some() => {
                    let step = self.execution.core().preview_c220_vector_word(word)?;
                    let destination_address = step.addresses.destination;
                    let instruction = C220CoreInstruction::VectorArithmetic(step);
                    self.issue_vector_at(tick, &instruction)?;
                    self.execution
                        .core_mut()
                        .commit_c220_vector_issue(Some(destination_address));
                    instruction
                }
                _ if C220VectorScalarInstruction::decode(word).is_some() => {
                    let step = self
                        .execution
                        .core()
                        .preview_c220_vector_scalar_word(word)?;
                    let destination_address = step.addresses.destination;
                    let instruction = C220CoreInstruction::VectorScalar(step);
                    self.issue_vector_at(tick, &instruction)?;
                    self.execution
                        .core_mut()
                        .commit_c220_vector_issue(Some(destination_address));
                    instruction
                }
                _ if C220ShiftInstruction::decode(word).is_some() => {
                    let step = self.execution.core().preview_c220_shift_word(word)?;
                    let destination_address = step.addresses.destination;
                    let instruction = C220CoreInstruction::VectorShift(step);
                    self.issue_vector_at(tick, &instruction)?;
                    self.execution
                        .core_mut()
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
                    C220CoreInstruction::Barrier(self.execution.core_mut().step_barrier_word(word)?)
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
                        .execution
                        .core_mut()
                        .step_scalar_word_with_ub(word, &mut self.memory)?;
                    if let Some(ticket) = timing {
                        self.scalar_timing.issue(ticket);
                    }
                    C220CoreInstruction::Scalar { step, timing }
                }
            }
        };
        self.execution.finish_other_at(tick)?;
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
            C220CoreInstruction::VectorArithmetic(issue) => {
                Some(C220VectorReadIssue::Arithmetic(issue))
            }
            C220CoreInstruction::VectorScalar(issue) => {
                Some(C220VectorReadIssue::VectorScalar(issue))
            }
            C220CoreInstruction::VectorShift(issue) => Some(C220VectorReadIssue::Shift(issue)),
            _ => None,
        };
        self.vector.issue_at(tick, &uops, stores, compute)?;
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
}

fn is_mte2_word(word: u32) -> bool {
    is_mte2_transfer(word)
        || FlagInstruction::decode(Architecture::Dav2201, word).is_some_and(|instruction| {
            instruction.source_pipe_code == 4 && matches!(instruction.trigger_pipe_code, 0 | 1)
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
mod tests {
    use super::*;
    use std::num::NonZeroU64;

    use crate::isa::c220::mte::CAPTURED_C220_MOV_UB_TO_OUT_WORD;
    use crate::memory::mapped::MappedMemory;
    use crate::memory::region::MemoryRegion;
    use crate::memory::sparse::{MemoryByteState, SparseMemory};
    use crate::memory::ub::UbMemory;
    use crate::sim::c220::fp16::C220Fp16Mode;
    use crate::sim::c220::vector::{
        C220_CAPTURED_MOVEV_CONTROL, C220_CAPTURED_MOVEV_WORD, C220_CAPTURED_VADD_CONTROL,
        C220_CAPTURED_VADD_WORD,
    };
    use crate::sim::machine::ScalarMachine;
    use crate::sim::mte_stepper::{
        C220_MTE3_TO_VECTOR_SET_FLAG_WORD, C220_MTE3_TO_VECTOR_WAIT_FLAG_WORD,
        C220_VECTOR_TO_MTE3_SET_FLAG_WORD, C220_VECTOR_TO_MTE3_WAIT_FLAG_WORD,
    };
    use crate::sim::stepper::ScalarStepper;

    #[test]
    fn vector_scalar_s32_and_f32_capture_scalar_and_delay_writeback() {
        for (opcode, source_bits, scalar_bits, result_bits, execute_ticks, saturating) in [
            (0x9600_0000, u32::MAX, 2, 2, 5, false),
            (0x9600_0001, u32::MAX, 2, u32::MAX, 5, false),
            (0x9700_0000, u32::MAX, 2, 1, 5, false),
            (0x9700_0001, u32::MAX, 2, u32::MAX - 1, 6, false),
            (0x9700_0000, i32::MAX as u32, 1, i32::MAX as u32, 5, true),
            (0x9700_0001, i32::MIN as u32, 2, i32::MIN as u32, 6, true),
            (
                0x96c0_0000,
                1.5_f32.to_bits(),
                2.0_f32.to_bits(),
                2.0_f32.to_bits(),
                5,
                false,
            ),
            (
                0x96c0_0001,
                1.5_f32.to_bits(),
                2.0_f32.to_bits(),
                1.5_f32.to_bits(),
                5,
                false,
            ),
            (
                0x96c0_0000,
                (-0.0_f32).to_bits(),
                0.0_f32.to_bits(),
                0.0_f32.to_bits(),
                5,
                false,
            ),
            (
                0x96c0_0001,
                (-0.0_f32).to_bits(),
                0.0_f32.to_bits(),
                (-0.0_f32).to_bits(),
                5,
                false,
            ),
            (
                0x97c0_0000,
                1.5_f32.to_bits(),
                2.0_f32.to_bits(),
                3.5_f32.to_bits(),
                7,
                false,
            ),
            (
                0x97c0_0001,
                1.5_f32.to_bits(),
                2.0_f32.to_bits(),
                3.0_f32.to_bits(),
                8,
                false,
            ),
        ] {
            let word = opcode | (3 << 17) | (4 << 12) | (6 << 7) | (5 << 2);
            let memory = SparseMemory::new(vec![MemoryRegion::unknown(256)], 512, 512);
            let memory = MappedMemory::bind(memory, &[0x2000]).unwrap();
            let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
            machine.set_xreg(3, 0x100).unwrap();
            machine.set_xreg(4, 0).unwrap();
            machine.set_xreg(5, (1_u64 << 56) | (1 << 16) | 1).unwrap();
            machine.set_xreg(6, u64::from(scalar_bits)).unwrap();
            let control_spr = (1_u64 << 56) | (u64::from(saturating) << 53);
            machine.set_spr_value(3, control_spr).unwrap();
            machine.set_spr_value(100, 1).unwrap();
            machine.set_spr_value(101, 0).unwrap();
            let mut ub = UbMemory::new(512, 256);
            let mut source = [0; 32];
            source[..4].copy_from_slice(&source_bits.to_le_bytes());
            ub.write_states(0, &source.map(MemoryByteState::Known))
                .unwrap();
            ub.write_states(0x100, &[MemoryByteState::Known(0xaa); 32])
                .unwrap();
            let execution = MteCoreStepper::new(ScalarStepper::new(machine, 0x4000), ub);
            let rate = NonZeroU64::new(32).unwrap();
            let mut core = C220Core::new(
                execution,
                memory,
                C220CoreTimingRules {
                    mte2: C220Mte2TimingRules {
                        issue_interval: NonZeroU64::new(1).unwrap(),
                        startup_ticks: 0,
                        bytes_per_tick: rate,
                        retire_ticks: 0,
                    },
                    mte3: C220Mte3TimingRules {
                        issue_interval: NonZeroU64::new(1).unwrap(),
                        startup_ticks: 0,
                        bytes_per_tick: rate,
                        retire_ticks: 0,
                    },
                    vector: C220VectorTimingRules {
                        dispatch_ticks: 0,
                        uop_issue_interval: NonZeroU64::new(1).unwrap(),
                        ub_response_ticks: 1,
                    },
                },
            )
            .unwrap();
            let C220CoreStep::Executed {
                instruction: C220CoreInstruction::VectorScalar(issue),
                ..
            } = core.step_word_at(0, word).unwrap()
            else {
                panic!("vector-scalar instruction should issue");
            };
            assert_eq!(issue.scalar.bits, scalar_bits);
            assert_eq!(issue.scalar.integer_saturating, saturating);
            if saturating {
                core.execution
                    .core_mut()
                    .scalar_mut()
                    .machine_mut()
                    .set_spr_value(3, 1 << 56)
                    .unwrap();
            }
            assert_eq!(
                C220CoreInstruction::VectorScalar(issue)
                    .vector_uops()
                    .unwrap()[0]
                    .stages
                    .execute_ticks,
                execute_ticks
            );
            assert_eq!(
                core.execution().core().ub().read_known(0x100, 4).unwrap(),
                [0xaa; 4]
            );
            core.execution
                .core_mut()
                .scalar_mut()
                .machine_mut()
                .set_xreg(6, 0x8000_0000)
                .unwrap();
            core.advance_to(100).unwrap();
            assert_eq!(
                core.execution().core().ub().read_known(0x100, 4).unwrap(),
                result_bits.to_le_bytes()
            );
            assert_eq!(
                core.execution().core().ub().read_known(0x104, 4).unwrap(),
                [0xaa; 4]
            );
            assert!(
                core.vector_pipeline().last_read_samples()[0]
                    .read1_grants
                    .is_empty()
            );
            assert_eq!(
                core.vector_pipeline().last_read_samples()[0].lanes[0]
                    .fp32_status
                    .is_some(),
                opcode & 0x00c0_0000 == 0x00c0_0000
            );
        }
    }

    #[test]
    fn vector_scalar_16_bit_forms_use_both_lane_groups_and_preserve_inactive_tail() {
        for (opcode, source_bits, scalar_bits, expected, execute_ticks, is_f16, fp_mode, sat) in [
            (
                0x9640_0000,
                0x3c00_u16,
                0x4000_u16,
                0x4000_u16,
                5,
                true,
                false,
                false,
            ),
            (0x9640_0001, 0x3c00, 0x4000, 0x3c00, 5, true, false, false),
            (0x9740_0000, 0x3c00, 0x4000, 0x4200, 7, true, false, false),
            (0x9740_0001, 0x3c00, 0x4000, 0x4000, 8, true, false, false),
            (0x9740_0000, 0x7c00, 0x3c00, 0x7bff, 7, true, false, false),
            (0x9740_0000, 0x7c00, 0x3c00, 0x7c00, 7, true, true, false),
            (0x9680_0000, u16::MAX, 2, 2, 5, false, false, false),
            (0x9680_0001, u16::MAX, 2, u16::MAX, 5, false, false, false),
            (0x9780_0000, u16::MAX, 2, 1, 5, false, false, false),
            (
                0x9780_0001,
                u16::MAX,
                2,
                u16::MAX - 1,
                6,
                false,
                false,
                false,
            ),
            (
                0x9780_0000,
                i16::MAX as u16,
                1,
                i16::MAX as u16,
                5,
                false,
                false,
                true,
            ),
            (
                0x9780_0001,
                i16::MIN as u16,
                2,
                i16::MIN as u16,
                6,
                false,
                false,
                true,
            ),
        ] {
            let word = opcode | (3 << 17) | (4 << 12) | (6 << 7) | (5 << 2);
            let memory = SparseMemory::new(vec![MemoryRegion::unknown(256)], 512, 512);
            let memory = MappedMemory::bind(memory, &[0x2000]).unwrap();
            let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
            machine.set_xreg(3, 0x200).unwrap();
            machine.set_xreg(4, 0).unwrap();
            machine.set_xreg(5, (1_u64 << 56) | (1 << 16) | 1).unwrap();
            machine.set_xreg(6, u64::from(scalar_bits)).unwrap();
            let control_spr = (1_u64 << 56) | (u64::from(fp_mode) << 48) | (u64::from(sat) << 53);
            machine.set_spr_value(3, control_spr).unwrap();
            machine.set_spr_value(100, 65).unwrap();
            machine.set_spr_value(101, 0).unwrap();
            let mut ub = UbMemory::new(1024, 256);
            let mut source = [0; 256];
            source[..2].copy_from_slice(&source_bits.to_le_bytes());
            source[128..130].copy_from_slice(&source_bits.to_le_bytes());
            ub.write_states(0, &source.map(MemoryByteState::Known))
                .unwrap();
            ub.write_states(0x200, &[MemoryByteState::Known(0xaa); 256])
                .unwrap();
            let execution = MteCoreStepper::new(ScalarStepper::new(machine, 0x4000), ub);
            let rate = NonZeroU64::new(32).unwrap();
            let mut core = C220Core::new(
                execution,
                memory,
                C220CoreTimingRules {
                    mte2: C220Mte2TimingRules {
                        issue_interval: NonZeroU64::new(1).unwrap(),
                        startup_ticks: 0,
                        bytes_per_tick: rate,
                        retire_ticks: 0,
                    },
                    mte3: C220Mte3TimingRules {
                        issue_interval: NonZeroU64::new(1).unwrap(),
                        startup_ticks: 0,
                        bytes_per_tick: rate,
                        retire_ticks: 0,
                    },
                    vector: C220VectorTimingRules {
                        dispatch_ticks: 0,
                        uop_issue_interval: NonZeroU64::new(1).unwrap(),
                        ub_response_ticks: 1,
                    },
                },
            )
            .unwrap();
            let C220CoreStep::Executed {
                instruction: C220CoreInstruction::VectorScalar(issue),
                ..
            } = core.step_word_at(0, word).unwrap()
            else {
                panic!("s16 vector-scalar instruction should issue");
            };
            assert_eq!(
                issue.scalar.fp16_mode,
                C220Fp16Mode::from_control_spr(control_spr)
            );
            assert_eq!(issue.scalar.integer_saturating, sat);
            if sat {
                core.execution
                    .core_mut()
                    .scalar_mut()
                    .machine_mut()
                    .set_spr_value(3, 1 << 56)
                    .unwrap();
            }
            if source_bits == 0x7c00 {
                core.execution
                    .core_mut()
                    .scalar_mut()
                    .machine_mut()
                    .set_spr_value(3, control_spr ^ (1 << 48))
                    .unwrap();
            }
            let uops = C220CoreInstruction::VectorScalar(issue)
                .vector_uops()
                .unwrap();
            assert_eq!(uops.len(), 2);
            assert_eq!(uops[0].stages.execute_ticks, execute_ticks);
            core.advance_to(200).unwrap();
            assert_eq!(
                core.execution().core().ub().read_known(0x200, 2).unwrap(),
                expected.to_le_bytes()
            );
            assert_eq!(
                core.execution().core().ub().read_known(0x280, 2).unwrap(),
                expected.to_le_bytes()
            );
            assert_eq!(
                core.execution().core().ub().read_known(0x282, 2).unwrap(),
                [0xaa; 2]
            );
            assert_eq!(core.vector_pipeline().pending_uops(), 0);
            assert_eq!(
                core.vector_pipeline().last_read_samples()[0].lanes[0]
                    .fp16_status
                    .is_some(),
                is_f16
            );
        }
    }

    #[test]
    fn vector_s32_binary_operations_use_delayed_reads_and_captured_saturation() {
        for (opcode, first, second, expected, execute_ticks, saturating) in [
            (
                0x8500_0000,
                i32::MAX as u32,
                1_u32,
                i32::MIN as u32,
                5,
                false,
            ),
            (0x8500_0000, i32::MAX as u32, 1, i32::MAX as u32, 5, true),
            (0x8500_0001, i32::MIN as u32, 1, i32::MAX as u32, 5, false),
            (0x8500_0001, i32::MIN as u32, 1, i32::MIN as u32, 5, true),
            (0x8900_0000, i32::MAX as u32, 2, u32::MAX - 1, 6, false),
            (0x8900_0000, i32::MAX as u32, 2, i32::MAX as u32, 6, true),
            (0x8700_0000, u32::MAX, 2, 2, 5, false),
            (0x8700_0001, u32::MAX, 2, u32::MAX, 5, false),
        ] {
            let word = opcode | (3 << 17) | (4 << 12) | (5 << 7) | (6 << 2);
            let memory = SparseMemory::new(vec![MemoryRegion::unknown(256)], 512, 512);
            let memory = MappedMemory::bind(memory, &[0x2000]).unwrap();
            let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
            machine.set_xreg(3, 0x200).unwrap();
            machine.set_xreg(4, 0).unwrap();
            machine.set_xreg(5, 0x100).unwrap();
            machine.set_xreg(6, 0x0100_0808_0801_0101).unwrap();
            let control_spr = (1_u64 << 56) | (u64::from(saturating) << 53);
            machine.set_spr_value(3, control_spr).unwrap();
            machine.set_spr_value(100, 1).unwrap();
            machine.set_spr_value(101, 0).unwrap();
            let mut ub = UbMemory::new(1024, 256);
            ub.write_states(0, &[MemoryByteState::Known(0); 32])
                .unwrap();
            ub.write_states(0x100, &[MemoryByteState::Known(0); 32])
                .unwrap();
            ub.write_states(0, &first.to_le_bytes().map(MemoryByteState::Known))
                .unwrap();
            ub.write_states(0x100, &second.to_le_bytes().map(MemoryByteState::Known))
                .unwrap();
            ub.write_states(0x200, &[MemoryByteState::Known(0xaa); 32])
                .unwrap();
            let execution = MteCoreStepper::new(ScalarStepper::new(machine, 0x4000), ub);
            let rate = NonZeroU64::new(32).unwrap();
            let mut core = C220Core::new(
                execution,
                memory,
                C220CoreTimingRules {
                    mte2: C220Mte2TimingRules {
                        issue_interval: NonZeroU64::new(1).unwrap(),
                        startup_ticks: 0,
                        bytes_per_tick: rate,
                        retire_ticks: 0,
                    },
                    mte3: C220Mte3TimingRules {
                        issue_interval: NonZeroU64::new(1).unwrap(),
                        startup_ticks: 0,
                        bytes_per_tick: rate,
                        retire_ticks: 0,
                    },
                    vector: C220VectorTimingRules {
                        dispatch_ticks: 0,
                        uop_issue_interval: NonZeroU64::new(1).unwrap(),
                        ub_response_ticks: 1,
                    },
                },
            )
            .unwrap();
            let C220CoreStep::Executed {
                instruction: C220CoreInstruction::VectorArithmetic(issue),
                ..
            } = core.step_word_at(0, word).unwrap()
            else {
                panic!("S32 vector instruction should issue");
            };
            assert_eq!(issue.modes.integer_saturating, saturating);
            assert!(issue.hint.has_s32_value_path());
            assert_eq!(
                C220CoreInstruction::VectorArithmetic(issue)
                    .vector_uops()
                    .unwrap()[0]
                    .stages
                    .execute_ticks,
                execute_ticks
            );
            assert_eq!(
                core.execution().core().ub().read_known(0x200, 4).unwrap(),
                [0xaa; 4]
            );
            core.execution
                .core_mut()
                .scalar_mut()
                .machine_mut()
                .set_spr_value(3, control_spr ^ (1 << 53))
                .unwrap();
            core.advance_to(100).unwrap();
            assert_eq!(
                core.execution().core().ub().read_known(0x200, 4).unwrap(),
                expected.to_le_bytes()
            );
            assert!(
                core.vector_pipeline().last_read_samples()[0].lanes[0]
                    .fp32_status
                    .is_none()
            );
        }
    }

    #[test]
    fn vector_s16_binary_operations_issue_two_lane_groups() {
        for (opcode, first, second, expected, saturating, widen_bit, execute_ticks) in [
            (0x8580_0000, i16::MAX, 1_i16, i16::MIN, false, false, 5),
            (0x8580_0000, i16::MAX, 1, i16::MAX, true, false, 5),
            (0x8580_0001, i16::MIN, 1, i16::MAX, false, false, 5),
            (0x8580_0001, i16::MIN, 1, i16::MIN, true, false, 5),
            (0x8980_0000, i16::MAX, 2, -2, false, false, 6),
            (0x8980_0000, i16::MAX, 2, i16::MAX, true, false, 6),
            (0x8780_0000, -1, 2, 2, false, false, 5),
            (0x8780_0000, -1, 2, 2, false, true, 5),
            (0x8780_0001, -1, 2, -1, false, false, 5),
            (0x9a40_0000, 0x0f0f, 0x3333, 0x3f3f, false, true, 1),
            (0x9a40_0001, 0x0f0f, 0x3333, 0x0303, false, true, 1),
        ] {
            let word = opcode | (3 << 17) | (4 << 12) | (5 << 7) | (6 << 2);
            let memory = SparseMemory::new(vec![MemoryRegion::unknown(256)], 512, 512);
            let memory = MappedMemory::bind(memory, &[0x2000]).unwrap();
            let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
            machine.set_xreg(3, 0x200).unwrap();
            machine.set_xreg(4, 0).unwrap();
            machine.set_xreg(5, 0x100).unwrap();
            machine.set_xreg(6, 0x0100_0808_0801_0101).unwrap();
            let control_spr = (u64::from(saturating) << 53) | (u64::from(widen_bit) << 52);
            machine.set_spr_value(3, control_spr).unwrap();
            machine.set_spr_value(100, 1).unwrap();
            machine.set_spr_value(101, 1).unwrap();
            let mut ub = UbMemory::new(1024, 256);
            for address in [0, 0x80, 0x100, 0x180] {
                ub.write_states(address, &[MemoryByteState::Known(0); 32])
                    .unwrap();
            }
            for offset in [0, 0x80] {
                ub.write_states(offset, &first.to_le_bytes().map(MemoryByteState::Known))
                    .unwrap();
                ub.write_states(
                    offset + 0x100,
                    &second.to_le_bytes().map(MemoryByteState::Known),
                )
                .unwrap();
                ub.write_states(offset + 0x200, &[MemoryByteState::Known(0xaa); 32])
                    .unwrap();
            }
            let execution = MteCoreStepper::new(ScalarStepper::new(machine, 0x4000), ub);
            let rate = NonZeroU64::new(32).unwrap();
            let mut core = C220Core::new(
                execution,
                memory,
                C220CoreTimingRules {
                    mte2: C220Mte2TimingRules {
                        issue_interval: NonZeroU64::new(1).unwrap(),
                        startup_ticks: 0,
                        bytes_per_tick: rate,
                        retire_ticks: 0,
                    },
                    mte3: C220Mte3TimingRules {
                        issue_interval: NonZeroU64::new(1).unwrap(),
                        startup_ticks: 0,
                        bytes_per_tick: rate,
                        retire_ticks: 0,
                    },
                    vector: C220VectorTimingRules {
                        dispatch_ticks: 0,
                        uop_issue_interval: NonZeroU64::new(1).unwrap(),
                        ub_response_ticks: 1,
                    },
                },
            )
            .unwrap();
            let C220CoreStep::Executed {
                instruction: C220CoreInstruction::VectorArithmetic(issue),
                ..
            } = core.step_word_at(0, word).unwrap()
            else {
                panic!("S16 vector instruction should issue");
            };
            assert_eq!(issue.result_element_bytes, 2);
            assert_eq!(issue.modes.integer_saturating, saturating);
            assert_eq!(issue.modes.widen_s16, widen_bit);
            let uops = C220CoreInstruction::VectorArithmetic(issue)
                .vector_uops()
                .unwrap();
            assert_eq!(uops.len(), 2);
            assert_eq!((uops[0].lane_group, uops[1].lane_group), (0, 1));
            assert!(
                uops.iter()
                    .all(|uop| uop.stages.execute_ticks == execute_ticks)
            );
            core.execution
                .core_mut()
                .scalar_mut()
                .machine_mut()
                .set_spr_value(3, control_spr ^ (1 << 53))
                .unwrap();
            core.advance_to(100).unwrap();
            for address in [0x200, 0x280] {
                assert_eq!(
                    core.execution().core().ub().read_known(address, 2).unwrap(),
                    expected.to_le_bytes()
                );
            }
            assert_eq!(
                core.vector_pipeline()
                    .last_read_samples()
                    .iter()
                    .map(|sample| sample.lane_group)
                    .collect::<Vec<_>>(),
                [0, 1]
            );
        }
    }

    #[test]
    fn scalar_conversion_retires_after_two_ticks_and_blocks_dependent_conversion() {
        let memory = SparseMemory::new(vec![MemoryRegion::unknown(64)], 128, 128);
        let memory = MappedMemory::bind(memory, &[0x2000]).unwrap();
        let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
        machine.set_xreg(8, u64::from(4.75_f32.to_bits())).unwrap();
        machine.set_xreg(11, u64::from(6.5_f32.to_bits())).unwrap();
        let execution =
            MteCoreStepper::new(ScalarStepper::new(machine, 0x4000), UbMemory::new(512, 256));
        let rate = NonZeroU64::new(32).unwrap();
        let mut core = C220Core::new(
            execution,
            memory,
            C220CoreTimingRules {
                mte2: C220Mte2TimingRules {
                    issue_interval: NonZeroU64::new(1).unwrap(),
                    startup_ticks: 0,
                    bytes_per_tick: rate,
                    retire_ticks: 0,
                },
                mte3: C220Mte3TimingRules {
                    issue_interval: NonZeroU64::new(1).unwrap(),
                    startup_ticks: 0,
                    bytes_per_tick: rate,
                    retire_ticks: 0,
                },
                vector: C220VectorTimingRules {
                    dispatch_ticks: 0,
                    uop_issue_interval: NonZeroU64::new(1).unwrap(),
                    ub_response_ticks: 1,
                },
            },
        )
        .unwrap();

        let C220CoreStep::Executed {
            instruction:
                C220CoreInstruction::Scalar {
                    timing: Some(ticket),
                    ..
                },
            ..
        } = core.step_word_at(10, 0x0210_8583).unwrap()
        else {
            panic!("scalar conversion should issue");
        };
        assert_eq!((ticket.issue_tick, ticket.retire_tick), (10, 12));
        assert_eq!(ticket.execution_stage, 2);
        assert_eq!(core.scalar_timing().pending_xreg_retirement(8), Some(12));

        let C220CoreStep::Stalled(stall) = core.step_word_at(11, 0x0210_8583).unwrap() else {
            panic!("dependent conversion should wait");
        };
        assert_eq!(stall.cause, C220StallCause::ScalarDependency);
        assert_eq!(stall.resume_tick, 12);
        assert_eq!(core.execution().core().scalar().pc(), 0x4004);

        let C220CoreStep::Stalled(move_stall) = core.step_word_at(11, 0x0202_8800).unwrap() else {
            panic!("scalar register read should wait");
        };
        assert_eq!(move_stall.cause, C220StallCause::ScalarDependency);
        assert_eq!(move_stall.resume_tick, 12);

        assert!(matches!(
            core.step_word_at(11, 0x0216_b583).unwrap(),
            C220CoreStep::Executed {
                instruction: C220CoreInstruction::Scalar { .. },
                ..
            }
        ));
        assert_eq!(core.scalar_timing().pending_xreg_retirement(11), Some(13));
        assert!(matches!(
            core.step_word_at(12, 0x0202_8800).unwrap(),
            C220CoreStep::Executed { .. }
        ));
        assert_eq!(core.scalar_timing().pending_xreg_retirement(8), None);
        assert_eq!(core.execution().core().scalar().machine().xregs()[1], 4);
        assert!(matches!(
            core.step_word_at(13, 0x0210_8583).unwrap(),
            C220CoreStep::Executed { .. }
        ));
        assert_eq!(core.scalar_timing().pending_xreg_retirement(8), Some(15));
    }

    #[test]
    fn vabs_uses_modeled_fifteen_tick_execution_stage() {
        let word = 0x83c0_0300 | (3 << 17) | (4 << 12) | (5 << 2);
        let make_core = || {
            let memory = SparseMemory::new(vec![MemoryRegion::unknown(256)], 512, 512);
            let memory = MappedMemory::bind(memory, &[0x2000]).unwrap();
            let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
            machine.set_xreg(3, 0x100).unwrap();
            machine.set_xreg(4, 0).unwrap();
            machine.set_xreg(5, (1_u64 << 56) | (1 << 16) | 1).unwrap();
            machine.set_spr_value(3, 1 << 56).unwrap();
            machine.set_spr_value(100, 1).unwrap();
            machine.set_spr_value(101, 0).unwrap();
            let mut ub = UbMemory::new(512, 256);
            let mut source = [0; 32];
            source[..4].copy_from_slice(&(-1.0_f32).to_le_bytes());
            ub.write_states(0, &source.map(MemoryByteState::Known))
                .unwrap();
            let execution = MteCoreStepper::new(ScalarStepper::new(machine, 0x4000), ub);
            let rate = NonZeroU64::new(32).unwrap();
            C220Core::new(
                execution,
                memory,
                C220CoreTimingRules {
                    mte2: C220Mte2TimingRules {
                        issue_interval: NonZeroU64::new(1).unwrap(),
                        startup_ticks: 0,
                        bytes_per_tick: rate,
                        retire_ticks: 0,
                    },
                    mte3: C220Mte3TimingRules {
                        issue_interval: NonZeroU64::new(1).unwrap(),
                        startup_ticks: 0,
                        bytes_per_tick: rate,
                        retire_ticks: 0,
                    },
                    vector: C220VectorTimingRules {
                        dispatch_ticks: 0,
                        uop_issue_interval: NonZeroU64::new(1).unwrap(),
                        ub_response_ticks: 1,
                    },
                },
            )
            .unwrap()
        };
        let mut timed = make_core();
        let C220CoreStep::Executed {
            instruction: C220CoreInstruction::VectorArithmetic(issue),
            ..
        } = timed.step_word_at(0, word).unwrap()
        else {
            panic!("VABS should issue to the vector pipeline");
        };
        assert_eq!(
            C220CoreInstruction::VectorArithmetic(issue)
                .vector_uops()
                .unwrap()[0]
                .stages
                .execute_ticks,
            15
        );
        let visible = timed.vector_pipeline().pending_visibility_tick().unwrap();
        timed.advance_to(visible).unwrap();
        assert_eq!(
            timed.execution().core().ub().read_known(0x100, 4).unwrap(),
            1.0_f32.to_le_bytes()
        );
        assert!(
            timed.vector_pipeline().last_read_samples()[0]
                .read1_grants
                .is_empty()
        );
    }

    #[test]
    fn vnot_b16_reads_one_source_and_preserves_inactive_ub_lanes() {
        let word = 0x8240_0800 | (3 << 17) | (4 << 12) | (5 << 2);
        let memory = SparseMemory::new(vec![MemoryRegion::unknown(256)], 512, 512);
        let memory = MappedMemory::bind(memory, &[0x2000]).unwrap();
        let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
        machine.set_xreg(3, 0x100).unwrap();
        machine.set_xreg(4, 0).unwrap();
        machine.set_xreg(5, (1_u64 << 56) | (1 << 16) | 1).unwrap();
        machine.set_spr_value(3, 1 << 52).unwrap();
        machine.set_spr_value(100, 1).unwrap();
        machine.set_spr_value(101, 0).unwrap();
        let mut ub = UbMemory::new(512, 256);
        let mut source = [0; 32];
        source[..2].copy_from_slice(&0x00f0_u16.to_le_bytes());
        ub.write_states(0, &source.map(MemoryByteState::Known))
            .unwrap();
        ub.write_states(0x100, &[MemoryByteState::Known(0xaa); 32])
            .unwrap();
        let execution = MteCoreStepper::new(ScalarStepper::new(machine, 0x4000), ub);
        let rate = NonZeroU64::new(32).unwrap();
        let mut core = C220Core::new(
            execution,
            memory,
            C220CoreTimingRules {
                mte2: C220Mte2TimingRules {
                    issue_interval: NonZeroU64::new(1).unwrap(),
                    startup_ticks: 0,
                    bytes_per_tick: rate,
                    retire_ticks: 0,
                },
                mte3: C220Mte3TimingRules {
                    issue_interval: NonZeroU64::new(1).unwrap(),
                    startup_ticks: 0,
                    bytes_per_tick: rate,
                    retire_ticks: 0,
                },
                vector: C220VectorTimingRules {
                    dispatch_ticks: 0,
                    uop_issue_interval: NonZeroU64::new(1).unwrap(),
                    ub_response_ticks: 1,
                },
            },
        )
        .unwrap();
        let C220CoreStep::Executed {
            instruction: C220CoreInstruction::VectorArithmetic(issue),
            ..
        } = core.step_word_at(0, word).unwrap()
        else {
            panic!("VNOT should issue to the vector pipeline");
        };
        assert_eq!(issue.hint.source_1_register, None);
        assert_eq!(issue.result_element_bytes, 2);
        let uops = C220CoreInstruction::VectorArithmetic(issue)
            .vector_uops()
            .unwrap();
        assert_eq!(uops[0].stages.execute_ticks, 1);
        core.advance_to(100).unwrap();
        let ub = core.execution().core().ub();
        assert_eq!(ub.read_known(0x100, 2).unwrap(), 0xff0f_u16.to_le_bytes());
        assert_eq!(ub.read_known(0x102, 2).unwrap(), [0xaa; 2]);
        assert!(
            core.vector_pipeline().last_read_samples()[0]
                .read1_grants
                .is_empty()
        );
    }

    #[test]
    fn vector_shifts_capture_scalar_and_follow_masked_pipeline() {
        for (opcode, source, shift, expected, width) in [
            (0x9c80_0003_u32, 0x8001_u32, 1_u64, 2_u32, 2_usize),
            (0x9cc0_0003, 0x8000_0001, 32, 0, 4),
            (0x9b00_0001, 0x8001, 1, 0x4000, 2),
            (0x9b40_0000, 0xfffd, 1, 0xfffe, 2),
            (0x9b40_0001, 0xfffd, 1, 0xffff, 2),
            (0x9b40_0000, 0xfffd, 17, 0xffff, 2),
            (0x9b40_0001, 0xfffd, 17, 0, 2),
            (0x9b80_0000, 0x8000_0001, 64, 0x8000_0001, 4),
            (0x9bc0_0001, 0xffff_fffd, 1, 0xffff_ffff, 4),
        ] {
            let word = opcode | (3 << 17) | (4 << 12) | (6 << 7) | (5 << 2);
            let memory = SparseMemory::new(vec![MemoryRegion::unknown(256)], 512, 512);
            let memory = MappedMemory::bind(memory, &[0x2000]).unwrap();
            let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
            machine.set_xreg(3, 0x100).unwrap();
            machine.set_xreg(4, 0).unwrap();
            machine.set_xreg(5, (1_u64 << 56) | (1 << 16) | 1).unwrap();
            machine.set_xreg(6, shift).unwrap();
            machine.set_spr_value(3, 0).unwrap();
            machine.set_spr_value(100, 1).unwrap();
            machine.set_spr_value(101, 0).unwrap();
            let mut ub = UbMemory::new(512, 256);
            let mut tile = [0; 32];
            tile[..width].copy_from_slice(&source.to_le_bytes()[..width]);
            ub.write_states(0, &tile.map(MemoryByteState::Known))
                .unwrap();
            ub.write_states(0x100, &[MemoryByteState::Known(0xaa); 32])
                .unwrap();
            let execution = MteCoreStepper::new(ScalarStepper::new(machine, 0x4000), ub);
            let rate = NonZeroU64::new(32).unwrap();
            let mut core = C220Core::new(
                execution,
                memory,
                C220CoreTimingRules {
                    mte2: C220Mte2TimingRules {
                        issue_interval: NonZeroU64::new(1).unwrap(),
                        startup_ticks: 0,
                        bytes_per_tick: rate,
                        retire_ticks: 0,
                    },
                    mte3: C220Mte3TimingRules {
                        issue_interval: NonZeroU64::new(1).unwrap(),
                        startup_ticks: 0,
                        bytes_per_tick: rate,
                        retire_ticks: 0,
                    },
                    vector: C220VectorTimingRules {
                        dispatch_ticks: 0,
                        uop_issue_interval: NonZeroU64::new(1).unwrap(),
                        ub_response_ticks: 1,
                    },
                },
            )
            .unwrap();
            let C220CoreStep::Executed {
                instruction: C220CoreInstruction::VectorShift(issue),
                ..
            } = core.step_word_at(0, word).unwrap()
            else {
                panic!("shift should issue to the vector pipeline");
            };
            assert_eq!(issue.shift, shift as u32);
            assert_eq!(
                C220CoreInstruction::VectorShift(issue)
                    .vector_uops()
                    .unwrap()[0]
                    .stages
                    .execute_ticks,
                6
            );
            assert_eq!(
                core.execution()
                    .core()
                    .ub()
                    .read_known(0x100, width)
                    .unwrap(),
                vec![0xaa; width]
            );
            core.advance_to(100).unwrap();
            let ub = core.execution().core().ub();
            assert_eq!(
                ub.read_known(0x100, width).unwrap(),
                expected.to_le_bytes()[..width]
            );
            assert_eq!(ub.read_known(0x100 + width as u64, 2).unwrap(), [0xaa; 2]);
            assert!(
                core.vector_pipeline().last_read_samples()[0]
                    .read1_grants
                    .is_empty()
            );
        }
    }

    #[test]
    fn vector_read_samples_ub_after_issue_without_an_implicit_raw_wait() {
        let memory = SparseMemory::new(vec![MemoryRegion::unknown(256)], 512, 512);
        let memory = MappedMemory::bind(memory, &[0x2000]).unwrap();
        let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
        machine.set_xreg(5, C220_CAPTURED_MOVEV_CONTROL).unwrap();
        machine.set_xreg(6, 0x4000_0000).unwrap();
        machine.set_xreg(8, C220_CAPTURED_VADD_CONTROL).unwrap();
        machine.set_xreg(13, 0).unwrap();
        machine.set_xreg(14, 0x200).unwrap();
        machine.set_xreg(16, 0).unwrap();
        machine.set_spr_value(3, 1 << 56).unwrap();
        machine.set_spr_value(100, 32).unwrap();
        machine.set_spr_value(101, 0).unwrap();
        let mut ub = UbMemory::new(1024, 256);
        let ones = 0x3f80_0000_u32
            .to_le_bytes()
            .repeat(64)
            .into_iter()
            .map(MemoryByteState::Known)
            .collect::<Vec<_>>();
        ub.write_states(0, &ones).unwrap();
        ub.write_states(0x200, &ones).unwrap();
        let execution = MteCoreStepper::new(ScalarStepper::new(machine, 0x4000), ub);
        let rate = NonZeroU64::new(32).unwrap();
        let mut core = C220Core::new(
            execution,
            memory,
            C220CoreTimingRules {
                mte2: C220Mte2TimingRules {
                    issue_interval: NonZeroU64::new(1).unwrap(),
                    startup_ticks: 0,
                    bytes_per_tick: rate,
                    retire_ticks: 0,
                },
                mte3: C220Mte3TimingRules {
                    issue_interval: NonZeroU64::new(1).unwrap(),
                    startup_ticks: 0,
                    bytes_per_tick: rate,
                    retire_ticks: 0,
                },
                vector: C220VectorTimingRules {
                    dispatch_ticks: 1,
                    uop_issue_interval: NonZeroU64::new(20).unwrap(),
                    ub_response_ticks: 2,
                },
            },
        )
        .unwrap();
        assert!(matches!(
            core.step_word_at(0, C220_CAPTURED_MOVEV_WORD).unwrap(),
            C220CoreStep::Executed { .. }
        ));
        let movev_visible = core.vector_pipeline().pending_visibility_tick().unwrap();
        assert!(movev_visible < 21);
        core.execution
            .core_mut()
            .scalar_mut()
            .machine_mut()
            .set_xreg(16, 0x400)
            .unwrap();
        assert!(matches!(
            core.step_word_at(1, C220_CAPTURED_VADD_WORD).unwrap(),
            C220CoreStep::Executed {
                instruction: C220CoreInstruction::VectorArithmetic(_),
                ..
            }
        ));
        assert!(core.vector_pipeline().last_read_samples().is_empty());
        assert!(core.execution().core().ub().read_known(0x400, 4).is_err());
        core.advance_to(30).unwrap();
        assert_eq!(
            core.execution().core().ub().read_known(0, 4).unwrap(),
            0x4000_0000_u32.to_le_bytes()
        );
        let sample = &core.vector_pipeline().last_read_samples()[0];
        let last_grant = sample
            .read0_grants
            .iter()
            .chain(&sample.read1_grants)
            .flatten()
            .copied()
            .max()
            .unwrap();
        assert_eq!(sample.tick, last_grant + 6);
        assert_eq!(sample.accesses.len(), 8);
        assert!(sample.accesses.iter().all(|access| access.block_index < 4));
        assert_eq!(&sample.source_0_bytes[..4], &0x4000_0000_u32.to_le_bytes());
        assert_eq!(&sample.source_1_bytes[..4], &0x3f80_0000_u32.to_le_bytes());
        assert_eq!(sample.lanes[0].bits, 0x4040_0000);
        assert!(core.execution().core().ub().read_known(0x400, 4).is_err());
        let arithmetic_visible = core.vector_pipeline().pending_visibility_tick().unwrap();
        core.advance_to(arithmetic_visible).unwrap();
        assert_eq!(
            core.execution().core().ub().read_known(0x400, 4).unwrap(),
            0x4040_0000_u32.to_le_bytes()
        );
    }

    #[test]
    fn halfword_movev_commits_its_tail_group_after_the_first_group() {
        let memory = SparseMemory::new(vec![MemoryRegion::unknown(256)], 512, 512);
        let memory = MappedMemory::bind(memory, &[0x2000]).unwrap();
        let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
        machine.set_xreg(5, C220_CAPTURED_MOVEV_CONTROL).unwrap();
        machine.set_xreg(6, 0x3c00).unwrap();
        machine.set_xreg(16, 0).unwrap();
        machine.set_spr_value(3, 1 << 56).unwrap();
        machine.set_spr_value(100, 65).unwrap();
        machine.set_spr_value(101, 0).unwrap();
        let execution =
            MteCoreStepper::new(ScalarStepper::new(machine, 0x4000), UbMemory::new(256, 256));
        let rate = NonZeroU64::new(32).unwrap();
        let mut core = C220Core::new(
            execution,
            memory,
            C220CoreTimingRules {
                mte2: C220Mte2TimingRules {
                    issue_interval: NonZeroU64::new(1).unwrap(),
                    startup_ticks: 0,
                    bytes_per_tick: rate,
                    retire_ticks: 0,
                },
                mte3: C220Mte3TimingRules {
                    issue_interval: NonZeroU64::new(1).unwrap(),
                    startup_ticks: 0,
                    bytes_per_tick: rate,
                    retire_ticks: 0,
                },
                vector: C220VectorTimingRules {
                    dispatch_ticks: 1,
                    uop_issue_interval: NonZeroU64::new(1).unwrap(),
                    ub_response_ticks: 2,
                },
            },
        )
        .unwrap();
        let halfword_word = (C220_CAPTURED_MOVEV_WORD & !(7 << 22)) | (1 << 22);
        let step = core.step_word_at(0, halfword_word).unwrap();
        let C220CoreStep::Executed {
            instruction: C220CoreInstruction::VectorMove(step),
            ..
        } = step
        else {
            panic!("expected MOVEV");
        };
        let uops = C220CoreInstruction::VectorMove(step).vector_uops().unwrap();
        assert_eq!(uops.len(), 2);
        assert_eq!((uops[0].lane_group, uops[1].lane_group), (0, 1));
        assert_eq!(core.vector_pipeline().pending_ub_responses(), 2);
        assert!(core.execution().core().ub().read_known(0, 2).is_err());
        let final_visibility = core.vector_pipeline().pending_visibility_tick().unwrap();
        core.advance_to(final_visibility - 1).unwrap();
        assert_eq!(
            core.execution().core().ub().read_known(0, 2).unwrap(),
            0x3c00_u16.to_le_bytes()
        );
        assert!(core.execution().core().ub().read_known(128, 2).is_err());
        core.advance_to(final_visibility).unwrap();
        assert_eq!(
            core.execution().core().ub().read_known(128, 2).unwrap(),
            0x3c00_u16.to_le_bytes()
        );
    }

    #[test]
    fn vector_issue_failure_keeps_execution_state_uncommitted() {
        let memory = SparseMemory::new(vec![MemoryRegion::unknown(256)], 512, 512);
        let memory = MappedMemory::bind(memory, &[0x2000]).unwrap();
        let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
        machine.set_xreg(5, C220_CAPTURED_MOVEV_CONTROL).unwrap();
        machine.set_xreg(6, 0x3c00).unwrap();
        machine.set_xreg(16, 0).unwrap();
        machine.set_spr_value(3, 1 << 56).unwrap();
        machine.set_spr_value(100, 1).unwrap();
        machine.set_spr_value(101, 0).unwrap();
        let execution =
            MteCoreStepper::new(ScalarStepper::new(machine, 0x4000), UbMemory::new(256, 256));
        let before = execution.clone();
        let rate = NonZeroU64::new(32).unwrap();
        let mut core = C220Core::new(
            execution,
            memory,
            C220CoreTimingRules {
                mte2: C220Mte2TimingRules {
                    issue_interval: NonZeroU64::new(1).unwrap(),
                    startup_ticks: 0,
                    bytes_per_tick: rate,
                    retire_ticks: 0,
                },
                mte3: C220Mte3TimingRules {
                    issue_interval: NonZeroU64::new(1).unwrap(),
                    startup_ticks: 0,
                    bytes_per_tick: rate,
                    retire_ticks: 0,
                },
                vector: C220VectorTimingRules {
                    dispatch_ticks: u64::MAX,
                    uop_issue_interval: NonZeroU64::new(1).unwrap(),
                    ub_response_ticks: 1,
                },
            },
        )
        .unwrap();
        assert!(matches!(
            core.step_word_at(1, C220_CAPTURED_MOVEV_WORD),
            Err(C220CoreError::VectorPipeline(
                C220VectorPipelineError::TimeOverflow
            ))
        ));
        assert_eq!(core.execution().core(), &before);
        assert_eq!(core.vector_pipeline().pending_uops(), 0);
    }

    #[test]
    fn mte3_completion_wait_uses_the_scheduled_request_service() {
        let regions = vec![
            MemoryRegion::unknown(128),
            MemoryRegion::new(8, 0x2000_u64.to_le_bytes().to_vec()).unwrap(),
        ];
        let memory = SparseMemory::new(regions, 256, 256);
        let memory = MappedMemory::bind(memory, &[0x2000, 0x1000]).unwrap();
        let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
        machine.set_xreg(5, C220_CAPTURED_MOVEV_CONTROL).unwrap();
        machine.set_xreg(6, 0x3f80_0000).unwrap();
        machine.set_xreg(16, 0).unwrap();
        machine.set_spr_value(3, 1 << 56).unwrap();
        machine.set_spr_value(100, 32).unwrap();
        machine.set_spr_value(101, 0).unwrap();
        let execution =
            MteCoreStepper::new(ScalarStepper::new(machine, 0x4000), UbMemory::new(512, 256));
        let rate = NonZeroU64::new(32).unwrap();
        let mut core = C220Core::new(
            execution,
            memory,
            C220CoreTimingRules {
                mte2: C220Mte2TimingRules {
                    issue_interval: NonZeroU64::new(1).unwrap(),
                    startup_ticks: 0,
                    bytes_per_tick: rate,
                    retire_ticks: 0,
                },
                mte3: C220Mte3TimingRules {
                    issue_interval: NonZeroU64::new(1).unwrap(),
                    startup_ticks: 2,
                    bytes_per_tick: rate,
                    retire_ticks: 1,
                },
                vector: C220VectorTimingRules {
                    dispatch_ticks: 2,
                    uop_issue_interval: NonZeroU64::new(1).unwrap(),
                    ub_response_ticks: 3,
                },
            },
        )
        .unwrap();
        core.step_word_at(0, C220_CAPTURED_MOVEV_WORD).unwrap();
        core.execution
            .core_mut()
            .scalar_mut()
            .machine_mut()
            .set_xreg(16, 0x80)
            .unwrap();
        core.step_word_at(1, C220_CAPTURED_MOVEV_WORD).unwrap();
        let machine = core.execution.core_mut().scalar_mut().machine_mut();
        machine.set_xreg(16, 0x100).unwrap();
        core.step_word_at(2, C220_CAPTURED_MOVEV_WORD).unwrap();
        let read_ready_tick = core.vector_pipeline().pending_visibility_tick().unwrap();
        core.execution
            .core_mut()
            .scalar_mut()
            .machine_mut()
            .set_xreg(16, 0x180)
            .unwrap();
        core.step_word_at(3, C220_CAPTURED_MOVEV_WORD).unwrap();
        let machine = core.execution.core_mut().scalar_mut().machine_mut();
        machine.set_xreg(8, C220_CAPTURED_VADD_CONTROL).unwrap();
        machine.set_xreg(13, 0).unwrap();
        machine.set_xreg(14, 0x80).unwrap();
        let movev_visible = core.vector_pipeline().pending_visibility_tick().unwrap();
        assert!(read_ready_tick < movev_visible);
        let last_release = movev_visible - 3;
        core.advance_to(last_release).unwrap();
        assert_eq!(core.last_vector_releases().len(), 4);
        assert!(core.execution().core().ub().read_known(0x180, 4).is_err());
        core.advance_to(read_ready_tick).unwrap();
        core.step_word_at(read_ready_tick, C220_CAPTURED_VADD_WORD)
            .unwrap();
        core.execution
            .core_mut()
            .scalar_mut()
            .machine_mut()
            .set_xreg(14, 0)
            .unwrap();
        let set_tick = read_ready_tick + 1;
        core.step_word_at(set_tick, C220_VECTOR_TO_MTE3_SET_FLAG_WORD)
            .unwrap();
        core.execution
            .core_mut()
            .scalar_mut()
            .machine_mut()
            .set_xreg(13, 0)
            .unwrap();
        let vector_visible = core.vector_pipeline().pending_visibility_tick().unwrap();
        assert!(matches!(
            core.step_word_at(set_tick + 1, C220_VECTOR_TO_MTE3_WAIT_FLAG_WORD)
                .unwrap(),
            C220CoreStep::Stalled(C220Stall {
                resume_tick,
                cause: C220StallCause::VectorDependency,
                ..
            }) if resume_tick == vector_visible
        ));
        core.step_word_at(vector_visible, C220_VECTOR_TO_MTE3_WAIT_FLAG_WORD)
            .unwrap();
        let machine = core.execution.core_mut().scalar_mut().machine_mut();
        machine.set_xreg(14, 0x180).unwrap();
        machine.set_xreg(10, 0x2000).unwrap();
        machine.set_xreg(3, 0x40010).unwrap();
        let issue_tick = vector_visible + 1;
        let issued = core
            .step_word_at(issue_tick, CAPTURED_C220_MOV_UB_TO_OUT_WORD)
            .unwrap();
        let C220CoreStep::Executed {
            instruction:
                C220CoreInstruction::Mte3 {
                    ticket: Some(ticket),
                    ..
                },
            ..
        } = issued
        else {
            panic!("expected a timed MTE3 transfer");
        };
        assert_eq!(ticket.issue_tick, issue_tick);
        assert_eq!(ticket.data_ready_tick, issue_tick + 6);
        assert_eq!(ticket.retire_tick, issue_tick + 7);
        assert_eq!(ticket.uop_count, 1);
        assert!(core.memory().read_known_at(0x2000, 128).is_err());
        core.execution
            .core_mut()
            .scalar_mut()
            .machine_mut()
            .set_xreg(10, 0)
            .unwrap();
        core.step_word_at(issue_tick + 1, C220_MTE3_TO_VECTOR_SET_FLAG_WORD)
            .unwrap();
        core.execution
            .core_mut()
            .scalar_mut()
            .machine_mut()
            .set_xreg(19, 0)
            .unwrap();
        assert!(matches!(
            core.step_word_at(issue_tick + 2, C220_MTE3_TO_VECTOR_WAIT_FLAG_WORD)
                .unwrap(),
            C220CoreStep::Stalled(C220Stall {
                resume_tick,
                cause: C220StallCause::Mte3Dependency,
                ..
            }) if resume_tick == ticket.retire_tick
        ));
        assert!(core.memory().read_known_at(0x2000, 128).is_err());
        assert_eq!(
            core.pending_output_ready_tick(),
            Some(ticket.data_ready_tick)
        );
        assert!(core.advance_to(ticket.data_ready_tick).unwrap().is_none());
        assert_eq!(core.pending_output_ready_tick(), None);
        assert_eq!(
            core.memory().read_known_at(0x2000, 128).unwrap(),
            0x4000_0000_u32.to_le_bytes().repeat(32)
        );
        assert!(matches!(
            core.step_word_at(ticket.data_ready_tick, C220_MTE3_TO_VECTOR_WAIT_FLAG_WORD)
                .unwrap(),
            C220CoreStep::Stalled(C220Stall {
                resume_tick,
                cause: C220StallCause::Mte3Dependency,
                ..
            }) if resume_tick == ticket.retire_tick
        ));
        assert_eq!(
            core.memory().read_known_at(0x2000, 128).unwrap(),
            0x4000_0000_u32.to_le_bytes().repeat(32)
        );
        core.step_word_at(ticket.retire_tick, C220_MTE3_TO_VECTOR_WAIT_FLAG_WORD)
            .unwrap();
        assert_eq!(
            core.memory().read_known_at(0x2000, 128).unwrap(),
            0x4000_0000_u32.to_le_bytes().repeat(32)
        );
    }
}
