use serde::Serialize;
use thiserror::Error;

use crate::device::architecture::Architecture;
use crate::device::device_loader::{DeviceKernelFetchError, LoadedDeviceKernel};
use crate::execution::c220::dma_uop::C220DmaUopRequest;
use crate::execution::c220::mte3_timing::{
    C220Mte3Ticket, C220Mte3TimingError, C220Mte3TimingRules, C220TimedMte3Lane,
};
use crate::execution::c220::timing::{
    C220Mte2TimingRules, C220Stall, C220StallCause, C220TimedMte2Core, C220TimedMte2Step,
    C220TimingError, is_mte2_transfer,
};
use crate::execution::c220::vector_timing::{C220VectorWritePlan, C220VectorWritePlanError};
use crate::execution::machine::ScalarInstructionError;
use crate::execution::mte_stepper::{
    C220OutputAction, C220OutputStep, MteCoreStepper, MteStepperError, UbScalarBusError,
};
use crate::execution::stepper::ScalarProgramStep;
use crate::instruction::flow::{
    FlagInstruction, FlagOperation, PipelineBarrierScope, PipelineBarrierStep,
};
use crate::instruction::mte_c220::C220DmaMovDescriptor;
use crate::instruction::vec_c220::{
    C220Fp32Step, C220MovevInstruction, C220MovevStep, C220VecArithmeticHint,
};
use crate::memory::hbm_pv_memory::HbmPvMemory;
use crate::memory::mapped::{MappedMemory, MappedMemoryError};
use crate::memory::ub::C220PreparedOutput;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "pipeline", rename_all = "snake_case")]
pub enum C220CoreInstruction {
    Scalar(ScalarProgramStep),
    Barrier(ScalarProgramStep),
    Mte2(C220TimedMte2Step),
    VectorMove(C220MovevStep),
    VectorArithmetic(C220Fp32Step),
    Mte3 {
        step: C220OutputStep,
        requests: Vec<C220DmaUopRequest>,
        ticket: Option<C220Mte3Ticket>,
    },
}

impl C220CoreInstruction {
    pub fn vector_write_plan(
        &self,
    ) -> Result<Option<C220VectorWritePlan>, C220VectorWritePlanError> {
        match self {
            Self::VectorMove(step) => C220VectorWritePlan::from_stores(&step.stores).map(Some),
            Self::VectorArithmetic(step) => {
                C220VectorWritePlan::from_stores(&step.stores).map(Some)
            }
            _ => Ok(None),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct C220CoreTimingRules {
    pub mte2: C220Mte2TimingRules,
    pub mte3: C220Mte3TimingRules,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum C220CoreStep {
    Executed {
        tick: u64,
        instruction: C220CoreInstruction,
    },
    Stalled(C220Stall),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum C220RunStop {
    Halted,
    TickBudget,
    EventBudget,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
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
    mte3: C220TimedMte3Lane,
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
            mte3: C220TimedMte3Lane::new(timing.mte3),
            pending_output: None,
            memory,
        })
    }

    pub const fn execution(&self) -> &C220TimedMte2Core {
        &self.execution
    }

    pub const fn memory(&self) -> &MappedMemory {
        &self.memory
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
            let core = self.execution.core_mut();
            match word {
                _ if C220MovevInstruction::decode(word).is_some() => {
                    C220CoreInstruction::VectorMove(core.step_c220_movev_word(word)?)
                }
                _ if C220VecArithmeticHint::from_word(word).is_some() => {
                    C220CoreInstruction::VectorArithmetic(core.step_c220_vector_word(word)?)
                }
                _ if matches!(
                    PipelineBarrierStep::decode(Architecture::Dav2201, pc, word),
                    Some(PipelineBarrierStep {
                        scope: PipelineBarrierScope::All,
                        ..
                    })
                ) =>
                {
                    C220CoreInstruction::Barrier(core.step_barrier_word(word)?)
                }
                _ => C220CoreInstruction::Scalar(
                    core.step_scalar_word_with_ub(word, &mut self.memory)?,
                ),
            }
        };
        self.execution.finish_other_at(tick)?;
        Ok(C220CoreStep::Executed { tick, instruction })
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

    use crate::execution::machine::ScalarMachine;
    use crate::execution::mte_stepper::{
        C220_MTE3_TO_VECTOR_SET_FLAG_WORD, C220_MTE3_TO_VECTOR_WAIT_FLAG_WORD,
        C220_VECTOR_TO_MTE3_SET_FLAG_WORD, C220_VECTOR_TO_MTE3_WAIT_FLAG_WORD,
    };
    use crate::execution::stepper::ScalarStepper;
    use crate::instruction::mte_c220::CAPTURED_C220_MOV_UB_TO_OUT_WORD;
    use crate::instruction::vec_c220::{
        C220_CAPTURED_MOVEV_CONTROL, C220_CAPTURED_MOVEV_WORD, C220_CAPTURED_VADD_CONTROL,
        C220_CAPTURED_VADD_WORD,
    };
    use crate::memory::mapped::MappedMemory;
    use crate::memory::region::MemoryRegion;
    use crate::memory::sparse::SparseMemory;
    use crate::memory::ub::UbMemory;

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
        let machine = core.execution.core_mut().scalar_mut().machine_mut();
        machine.set_xreg(8, C220_CAPTURED_VADD_CONTROL).unwrap();
        machine.set_xreg(13, 0).unwrap();
        machine.set_xreg(14, 0x80).unwrap();
        core.step_word_at(3, C220_CAPTURED_VADD_WORD).unwrap();
        core.execution
            .core_mut()
            .scalar_mut()
            .machine_mut()
            .set_xreg(14, 0)
            .unwrap();
        core.step_word_at(4, C220_VECTOR_TO_MTE3_SET_FLAG_WORD)
            .unwrap();
        core.execution
            .core_mut()
            .scalar_mut()
            .machine_mut()
            .set_xreg(13, 0)
            .unwrap();
        core.step_word_at(5, C220_VECTOR_TO_MTE3_WAIT_FLAG_WORD)
            .unwrap();
        let machine = core.execution.core_mut().scalar_mut().machine_mut();
        machine.set_xreg(14, 0x100).unwrap();
        machine.set_xreg(10, 0x2000).unwrap();
        machine.set_xreg(3, 0x40010).unwrap();
        let issued = core
            .step_word_at(6, CAPTURED_C220_MOV_UB_TO_OUT_WORD)
            .unwrap();
        assert!(matches!(
            issued,
            C220CoreStep::Executed {
                instruction: C220CoreInstruction::Mte3 {
                    ticket: Some(C220Mte3Ticket {
                        issue_tick: 6,
                        data_ready_tick: 12,
                        retire_tick: 13,
                        uop_count: 1,
                        ..
                    }),
                    ..
                },
                ..
            }
        ));
        assert!(core.memory().read_known_at(0x2000, 128).is_err());
        core.execution
            .core_mut()
            .scalar_mut()
            .machine_mut()
            .set_xreg(10, 0)
            .unwrap();
        core.step_word_at(7, C220_MTE3_TO_VECTOR_SET_FLAG_WORD)
            .unwrap();
        core.execution
            .core_mut()
            .scalar_mut()
            .machine_mut()
            .set_xreg(19, 0)
            .unwrap();
        assert!(matches!(
            core.step_word_at(8, C220_MTE3_TO_VECTOR_WAIT_FLAG_WORD)
                .unwrap(),
            C220CoreStep::Stalled(C220Stall {
                resume_tick: 13,
                cause: C220StallCause::Mte3Dependency,
                ..
            })
        ));
        assert!(core.memory().read_known_at(0x2000, 128).is_err());
        assert_eq!(core.pending_output_ready_tick(), Some(12));
        assert!(core.advance_to(12).unwrap().is_none());
        assert_eq!(core.pending_output_ready_tick(), None);
        assert_eq!(
            core.memory().read_known_at(0x2000, 128).unwrap(),
            0x4000_0000_u32.to_le_bytes().repeat(32)
        );
        assert!(matches!(
            core.step_word_at(12, C220_MTE3_TO_VECTOR_WAIT_FLAG_WORD)
                .unwrap(),
            C220CoreStep::Stalled(C220Stall {
                resume_tick: 13,
                cause: C220StallCause::Mte3Dependency,
                ..
            })
        ));
        assert_eq!(
            core.memory().read_known_at(0x2000, 128).unwrap(),
            0x4000_0000_u32.to_le_bytes().repeat(32)
        );
        core.step_word_at(13, C220_MTE3_TO_VECTOR_WAIT_FLAG_WORD)
            .unwrap();
        assert_eq!(
            core.memory().read_known_at(0x2000, 128).unwrap(),
            0x4000_0000_u32.to_le_bytes().repeat(32)
        );
    }
}
