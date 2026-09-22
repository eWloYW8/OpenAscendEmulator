use thiserror::Error;

use crate::architecture::Architecture;
use crate::isa::c220::mte::C220MovOutToUbError;
use crate::isa::c220::scalar::C220ScalarConversionHint;
use crate::isa::flow::{FlagInstruction, FlagOperation, PipelineBarrierScope, PipelineBarrierStep};
use crate::memory::mapped::MappedMemory;
use crate::memory::ub::{UbMemory, UbMemoryError, UbTransferResult};
use crate::sim::c220::core::functional::state::C220ExecutionState;
use crate::sim::c220::mte::decode as c220_mte_decode;
use crate::sim::c220::mte::transfer::{
    C220Mte2TransferPlan, C220TransferError, copy_c220_mov_out_to_ub,
};
use crate::sim::c220::scalar::bus::{C220ScalarBus, C220ScalarBusError};
use crate::sim::c220::vector::C220VectorError;
use crate::sim::common::scalar::stepper::{ScalarProgramStep, ScalarStepper};
use crate::sim::common::scalar::{ScalarInstructionError, ScalarInstructionStep, ScalarMemoryBus};

mod output;
mod state;
mod vector;

pub use output::{C220OutputAction, C220OutputStep};

pub const MAX_PENDING_MTE2_TRANSFERS: usize = 64;

struct DecodedMte2 {
    transfer: C220Mte2TransferPlan,
    source_address: u64,
    destination_address: u64,
    planned_bytes: usize,
}

fn commit_mte2(
    plan: C220Mte2TransferPlan,
    ub: &mut UbMemory,
    source: &MappedMemory,
) -> Result<UbTransferResult, C220FunctionalError> {
    Ok(copy_c220_mov_out_to_ub(
        ub,
        source,
        plan.descriptor,
        plan.source_address,
        plan.destination_address,
    )?)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220FunctionalCore {
    pub(crate) scalar: ScalarStepper,
    pub(crate) ub: UbMemory,
    pending_mte2: Vec<C220Mte2TransferPlan>,
    flag0_set: bool,
    pub(crate) state: C220ExecutionState,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum C220MteAction {
    Issue {
        source_address: u64,
        destination_address: u64,
        planned_bytes: usize,
        pending_count: usize,
    },
    SetFlag {
        pending_count: usize,
    },
    WaitFlag {
        transfers: Vec<UbTransferResult>,
    },
    SetVectorFlag {
        flag_id: u8,
        remaining_pending: usize,
    },
    WaitVectorFlag {
        flag_id: u8,
        transfer: UbTransferResult,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220MteProgramStep {
    pub pc: u64,
    pub word: u32,
    pub next_pc: u64,
    pub action: C220MteAction,
}

#[derive(Debug, Error)]
pub enum C220FunctionalError {
    #[error("program ended before MTE word at PC {pc:#x}")]
    ProgramEnded { pc: u64 },
    #[error("word {word:#010x} at PC {pc:#x} is not an implemented MTE2 or flag operation")]
    UnsupportedWord { pc: u64, word: u32 },
    #[error("SPR {index} is unavailable for MTE word at PC {pc:#x}")]
    MissingSpr { pc: u64, index: u16 },
    #[error("MTE2 pending count exceeds {MAX_PENDING_MTE2_TRANSFERS}")]
    PendingLimit,
    #[error("MTE2-to-Scalar flag 0 has no pending transfer")]
    SetWithoutTransfer,
    #[error("MTE2-to-Scalar flag 0 is already set")]
    FlagAlreadySet,
    #[error("MTE2-to-Scalar flag 0 was not set before wait")]
    WaitWithoutFlag,
    #[error("MTE2-to-Vector flag {flag_id} has no unassigned transfer")]
    VectorSetWithoutTransfer { flag_id: u8 },
    #[error("MTE2-to-Vector flag {flag_id} is already set")]
    VectorFlagAlreadySet { flag_id: u8 },
    #[error("MTE2-to-Vector flag {flag_id} was not set before wait")]
    VectorWaitWithoutFlag { flag_id: u8 },
    #[error("MTE2-to-Vector flag ID {flag_id} is outside the modeled 0..1 range")]
    UnsupportedVectorFlagId { flag_id: u32 },
    #[error("MTE2 transfer byte count overflows usize")]
    TransferSizeOverflow,
    #[error("pipeline barrier at PC {pc:#x} has outstanding modeled work")]
    BarrierBusy { pc: u64 },
    #[error("C220 output was not produced before its flag")]
    OutputNotProduced,
    #[error("C220 output is still waiting for a flag")]
    UnsignaledOutput,
    #[error("C220 output flag {flag_id} is already set")]
    OutputFlagAlreadySet { flag_id: u8 },
    #[error("C220 output flag {flag_id} was not set before wait")]
    OutputWaitWithoutFlag { flag_id: u8 },
    #[error("C220 output flag ID {flag_id} is outside the modeled range")]
    UnsupportedOutputFlagId { flag_id: u32 },
    #[error("C220 output transfer has no matching completed flag")]
    OutputNotReady,
    #[error("C220 output dependency is already waiting for transfer")]
    OutputDependencyOutstanding,
    #[error("C220 MTE2 reuse flag {flag_id} is already set")]
    ReuseFlagAlreadySet { flag_id: u8 },
    #[error("C220 MTE2 reuse flag {flag_id} was not set before wait")]
    ReuseWaitWithoutFlag { flag_id: u8 },
    #[error("C220 MTE3 completion flag {flag_id} is already set")]
    CompletionFlagAlreadySet { flag_id: u8 },
    #[error("C220 MTE3 completion flag {flag_id} was not set before wait")]
    CompletionWaitWithoutFlag { flag_id: u8 },
    #[error("C220 output was not copied before MTE3 completion flag")]
    OutputNotCopied,
    #[error(transparent)]
    C220(#[from] C220MovOutToUbError),
    #[error(transparent)]
    Ub(#[from] UbMemoryError),
    #[error(transparent)]
    C220Transfer(#[from] C220TransferError),
    #[error(transparent)]
    Vector(#[from] C220VectorError),
}

impl C220FunctionalCore {
    pub fn new(scalar: ScalarStepper, ub: UbMemory) -> Self {
        Self {
            scalar,
            ub,
            pending_mte2: Vec::new(),
            flag0_set: false,
            state: C220ExecutionState::default(),
        }
    }

    pub const fn scalar(&self) -> &ScalarStepper {
        &self.scalar
    }

    pub fn set_isa_instance_index(&mut self, index: u32) {
        self.state.isa_instance_index = index;
    }

    pub fn scalar_mut(&mut self) -> &mut ScalarStepper {
        &mut self.scalar
    }

    pub const fn ub(&self) -> &UbMemory {
        &self.ub
    }

    pub(crate) fn ub_mut(&mut self) -> &mut UbMemory {
        &mut self.ub
    }

    pub const fn pending_mte2_count(&self) -> usize {
        self.pending_mte2.len()
    }

    pub const fn flag0_set(&self) -> bool {
        self.flag0_set
    }

    pub const fn vector_flags_set(&self) -> [bool; 2] {
        self.state.vector_flags_set()
    }

    pub const fn output_flags_set(&self) -> [bool; 4] {
        self.state.output_flags_set()
    }

    pub const fn reuse_flags_set(&self) -> [bool; 2] {
        self.state.mte2_reuse_flags
    }

    pub const fn completion_flags_set(&self) -> [bool; 4] {
        self.state.completion_flags_set()
    }

    pub fn step_scalar_word<B: ScalarMemoryBus>(
        &mut self,
        word: u32,
        bus: &mut B,
    ) -> Result<ScalarProgramStep, ScalarInstructionError<B::Error>> {
        self.scalar.step_word(word, bus)
    }

    pub fn step_barrier_word(
        &mut self,
        word: u32,
    ) -> Result<ScalarProgramStep, C220FunctionalError> {
        let pc = self.scalar.pc();
        if self.scalar.is_halted() {
            return Err(C220FunctionalError::ProgramEnded { pc });
        }
        let architecture = self.scalar.machine().architecture();
        let barrier = PipelineBarrierStep::decode(architecture, pc, word)
            .filter(|step| step.scope == PipelineBarrierScope::All)
            .ok_or(C220FunctionalError::UnsupportedWord { pc, word })?;
        if !self.pending_mte2.is_empty() || self.flag0_set || self.state.barrier_busy() {
            return Err(C220FunctionalError::BarrierBusy { pc });
        }
        self.scalar.advance_sequential();
        Ok(ScalarProgramStep {
            pc,
            word,
            next_pc: self.scalar.pc(),
            instruction: ScalarInstructionStep::Barrier(barrier),
            halted_after: false,
        })
    }

    pub fn step_scalar_word_with_ub<B: ScalarMemoryBus>(
        &mut self,
        word: u32,
        fallback: &mut B,
    ) -> Result<ScalarProgramStep, ScalarInstructionError<C220ScalarBusError<B::Error>>> {
        if C220ScalarConversionHint::from_word(word).is_some() {
            let pc = self.scalar.pc();
            let step = crate::sim::c220::scalar::execute_conversion_word(
                self.scalar.machine_mut(),
                pc,
                word,
            )?;
            self.scalar.advance_sequential();
            return Ok(ScalarProgramStep {
                pc,
                word,
                next_pc: self.scalar.pc(),
                instruction: ScalarInstructionStep::Register(step),
                halted_after: false,
            });
        }
        let machine = self.scalar.machine();
        let mut bus = C220ScalarBus::new(
            &mut self.ub,
            fallback,
            machine.spr_value(67).zip(machine.spr_value(68)),
        );
        self.scalar.step_word(word, &mut bus)
    }

    pub fn step_mte_word(
        &mut self,
        word: u32,
        source: &MappedMemory,
    ) -> Result<C220MteProgramStep, C220FunctionalError> {
        let transfer = if FlagInstruction::decode(Architecture::Dav2201, word).is_some() {
            None
        } else {
            Some(self.decode_transfer(self.scalar.pc(), word)?.transfer)
        };
        self.step_mte_word_with_plan(word, source, transfer)
    }

    pub(crate) fn step_mte_word_with_plan(
        &mut self,
        word: u32,
        source: &MappedMemory,
        transfer: Option<C220Mte2TransferPlan>,
    ) -> Result<C220MteProgramStep, C220FunctionalError> {
        let pc = self.scalar.pc();
        if self.scalar.machine().architecture() != Architecture::Dav2201 {
            return Err(C220FunctionalError::UnsupportedWord { pc, word });
        }
        self.step_mte_word_with_decode(word, source, move |_, pc, word| {
            let plan = transfer.ok_or(C220FunctionalError::UnsupportedWord { pc, word })?;
            Ok(DecodedMte2 {
                transfer: plan,
                source_address: plan.source_address,
                destination_address: plan.destination_address,
                planned_bytes: plan.bytes,
            })
        })
    }

    fn step_mte_word_with_decode<F>(
        &mut self,
        word: u32,
        source: &MappedMemory,
        decode: F,
    ) -> Result<C220MteProgramStep, C220FunctionalError>
    where
        F: FnOnce(&Self, u64, u32) -> Result<DecodedMte2, C220FunctionalError>,
    {
        let pc = self.scalar.pc();
        if self.scalar.is_halted() {
            return Err(C220FunctionalError::ProgramEnded { pc });
        }
        let flag = FlagInstruction::decode(self.scalar.machine().architecture(), word);
        let route = flag.map(|instruction| {
            (
                instruction.source_pipe_code,
                instruction.trigger_pipe_code,
                instruction.operation,
            )
        });
        let action = match route {
            Some((4, 0, FlagOperation::Set))
                if flag
                    .unwrap()
                    .resolve(pc, self.scalar.machine().xregs())
                    .flag_id
                    == 0 =>
            {
                if self.flag0_set {
                    return Err(C220FunctionalError::FlagAlreadySet);
                }
                if self.pending_mte2.is_empty() {
                    return Err(C220FunctionalError::SetWithoutTransfer);
                }
                self.flag0_set = true;
                C220MteAction::SetFlag {
                    pending_count: self.pending_mte2.len(),
                }
            }
            Some((4, 0, FlagOperation::Wait))
                if flag
                    .unwrap()
                    .resolve(pc, self.scalar.machine().xregs())
                    .flag_id
                    == 0 =>
            {
                if !self.flag0_set {
                    return Err(C220FunctionalError::WaitWithoutFlag);
                }
                let mut staged = self.ub.clone();
                let mut transfers = Vec::with_capacity(self.pending_mte2.len());
                for transfer in &self.pending_mte2 {
                    transfers.push(commit_mte2(*transfer, &mut staged, source)?);
                }
                self.ub = staged;
                self.pending_mte2.clear();
                self.flag0_set = false;
                C220MteAction::WaitFlag { transfers }
            }
            Some((4, 1, FlagOperation::Set))
                if self.scalar.machine().architecture() == Architecture::Dav2201 =>
            {
                let flag_id = self.resolve_c220_vector_flag_id(pc, word)?;
                let slot = &mut self.state.vector_flags[usize::from(flag_id)];
                if slot.is_some() {
                    return Err(C220FunctionalError::VectorFlagAlreadySet { flag_id });
                }
                if self.pending_mte2.is_empty() {
                    return Err(C220FunctionalError::VectorSetWithoutTransfer { flag_id });
                }
                let plan = self.pending_mte2.remove(0);
                *slot = Some(plan);
                C220MteAction::SetVectorFlag {
                    flag_id,
                    remaining_pending: self.pending_mte2.len(),
                }
            }
            Some((4, 1, FlagOperation::Wait))
                if self.scalar.machine().architecture() == Architecture::Dav2201 =>
            {
                let flag_id = self.resolve_c220_vector_flag_id(pc, word)?;
                let transfer = self.state.vector_flags[usize::from(flag_id)]
                    .ok_or(C220FunctionalError::VectorWaitWithoutFlag { flag_id })?;
                let mut staged = self.ub.clone();
                let result = commit_mte2(transfer, &mut staged, source)?;
                self.ub = staged;
                self.state.vector_flags[usize::from(flag_id)] = None;
                C220MteAction::WaitVectorFlag {
                    flag_id,
                    transfer: result,
                }
            }
            _ => {
                if self.pending_mte2.len()
                    + self
                        .state
                        .vector_flags
                        .iter()
                        .filter(|slot| slot.is_some())
                        .count()
                    >= MAX_PENDING_MTE2_TRANSFERS
                {
                    return Err(C220FunctionalError::PendingLimit);
                }
                if self.flag0_set {
                    return Err(C220FunctionalError::FlagAlreadySet);
                }
                let decoded = decode(self, pc, word)?;
                self.pending_mte2.push(decoded.transfer);
                C220MteAction::Issue {
                    source_address: decoded.source_address,
                    destination_address: decoded.destination_address,
                    planned_bytes: decoded.planned_bytes,
                    pending_count: self.pending_mte2.len(),
                }
            }
        };
        self.scalar.advance_sequential();
        Ok(C220MteProgramStep {
            pc,
            word,
            next_pc: self.scalar.pc(),
            action,
        })
    }

    fn decode_transfer(&self, pc: u64, word: u32) -> Result<DecodedMte2, C220FunctionalError> {
        let machine = self.scalar.machine();
        if machine.architecture() != Architecture::Dav2201 {
            return Err(C220FunctionalError::UnsupportedWord { pc, word });
        }
        let plan = c220_mte_decode::decode_mte2_transfer(
            machine,
            pc,
            word,
            self.state.isa_instance_index,
        )?;
        Ok(DecodedMte2 {
            transfer: plan,
            source_address: plan.source_address,
            destination_address: plan.destination_address,
            planned_bytes: plan.bytes,
        })
    }

    pub(crate) fn preview_c220_mte2_transfer(
        &self,
        word: u32,
    ) -> Result<C220Mte2TransferPlan, C220FunctionalError> {
        c220_mte_decode::decode_mte2_transfer(
            self.scalar.machine(),
            self.scalar.pc(),
            word,
            self.state.isa_instance_index,
        )
    }
}
