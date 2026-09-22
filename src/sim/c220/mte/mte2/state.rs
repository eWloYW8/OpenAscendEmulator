use crate::architecture::Architecture;
use crate::isa::flow::{FlagInstruction, FlagOperation};
use crate::memory::mapped::MappedMemory;
use crate::memory::ub::{UbMemory, UbTransferResult};
use crate::sim::c220::mte::mte2::{C220Mte2TransferPlan, copy_c220_mov_out_to_ub};
use crate::sim::c220::state::C220ExecutionError;
use crate::sim::common::scalar::{ScalarMachine, ScalarStepper};
use std::collections::VecDeque;

pub const MAX_PENDING_MTE2_TRANSFERS: usize = 64;

fn commit_mte2(
    plan: C220Mte2TransferPlan,
    ub: &mut UbMemory,
    source: &MappedMemory,
) -> Result<UbTransferResult, C220ExecutionError> {
    Ok(copy_c220_mov_out_to_ub(
        ub,
        source,
        plan.descriptor,
        plan.source_address,
        plan.destination_address,
    )?)
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct C220Mte2State {
    pending: VecDeque<C220Mte2TransferPlan>,
    scalar_flag: bool,
    vector_flags: [Option<C220Mte2TransferPlan>; 2],
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

impl C220Mte2State {
    pub(crate) fn pending_count(&self) -> usize {
        self.pending.len()
    }

    pub(crate) const fn scalar_flag_set(&self) -> bool {
        self.scalar_flag
    }

    pub(crate) const fn vector_flags_set(&self) -> [bool; 2] {
        [
            self.vector_flags[0].is_some(),
            self.vector_flags[1].is_some(),
        ]
    }

    pub(crate) fn is_busy(&self) -> bool {
        !self.pending.is_empty()
            || self.scalar_flag
            || self.vector_flags.iter().any(Option::is_some)
    }

    pub(super) fn step_word(
        &mut self,
        scalar: &mut ScalarStepper,
        ub: &mut UbMemory,
        word: u32,
        source: &MappedMemory,
        transfer: Option<C220Mte2TransferPlan>,
    ) -> Result<C220MteProgramStep, C220ExecutionError> {
        let pc = scalar.pc();
        if scalar.machine().architecture() != Architecture::Dav2201 {
            return Err(C220ExecutionError::UnsupportedWord { pc, word });
        }
        if scalar.is_halted() {
            return Err(C220ExecutionError::ProgramEnded { pc });
        }
        let flag = FlagInstruction::decode(scalar.machine().architecture(), word);
        let route = flag.map(|instruction| {
            (
                instruction.source_pipe_code,
                instruction.trigger_pipe_code,
                instruction.operation,
            )
        });
        let action = match route {
            Some((4, 0, FlagOperation::Set))
                if flag.unwrap().resolve(pc, scalar.machine().xregs()).flag_id == 0 =>
            {
                if self.scalar_flag {
                    return Err(C220ExecutionError::FlagAlreadySet);
                }
                if self.pending.is_empty() {
                    return Err(C220ExecutionError::SetWithoutTransfer);
                }
                self.scalar_flag = true;
                C220MteAction::SetFlag {
                    pending_count: self.pending.len(),
                }
            }
            Some((4, 0, FlagOperation::Wait))
                if flag.unwrap().resolve(pc, scalar.machine().xregs()).flag_id == 0 =>
            {
                if !self.scalar_flag {
                    return Err(C220ExecutionError::WaitWithoutFlag);
                }
                let mut staged = ub.clone();
                let mut transfers = Vec::with_capacity(self.pending.len());
                for transfer in &self.pending {
                    transfers.push(commit_mte2(*transfer, &mut staged, source)?);
                }
                *ub = staged;
                self.pending.clear();
                self.scalar_flag = false;
                C220MteAction::WaitFlag { transfers }
            }
            Some((4, 1, FlagOperation::Set))
                if scalar.machine().architecture() == Architecture::Dav2201 =>
            {
                let flag_id = Self::resolve_vector_flag_id(scalar.machine(), pc, word)?;
                let slot = &mut self.vector_flags[usize::from(flag_id)];
                if slot.is_some() {
                    return Err(C220ExecutionError::VectorFlagAlreadySet { flag_id });
                }
                if self.pending.is_empty() {
                    return Err(C220ExecutionError::VectorSetWithoutTransfer { flag_id });
                }
                let plan = self.pending.pop_front().expect("pending transfer checked");
                *slot = Some(plan);
                C220MteAction::SetVectorFlag {
                    flag_id,
                    remaining_pending: self.pending.len(),
                }
            }
            Some((4, 1, FlagOperation::Wait))
                if scalar.machine().architecture() == Architecture::Dav2201 =>
            {
                let flag_id = Self::resolve_vector_flag_id(scalar.machine(), pc, word)?;
                let transfer = self.vector_flags[usize::from(flag_id)]
                    .ok_or(C220ExecutionError::VectorWaitWithoutFlag { flag_id })?;
                let mut staged = ub.clone();
                let result = commit_mte2(transfer, &mut staged, source)?;
                *ub = staged;
                self.vector_flags[usize::from(flag_id)] = None;
                C220MteAction::WaitVectorFlag {
                    flag_id,
                    transfer: result,
                }
            }
            _ => {
                if self.pending.len()
                    + self
                        .vector_flags
                        .iter()
                        .filter(|slot| slot.is_some())
                        .count()
                    >= MAX_PENDING_MTE2_TRANSFERS
                {
                    return Err(C220ExecutionError::PendingLimit);
                }
                if self.scalar_flag {
                    return Err(C220ExecutionError::FlagAlreadySet);
                }
                let plan = transfer.ok_or(C220ExecutionError::UnsupportedWord { pc, word })?;
                self.pending.push_back(plan);
                C220MteAction::Issue {
                    source_address: plan.source_address,
                    destination_address: plan.destination_address,
                    planned_bytes: plan.bytes,
                    pending_count: self.pending.len(),
                }
            }
        };
        scalar.advance_sequential();
        Ok(C220MteProgramStep {
            pc,
            word,
            next_pc: scalar.pc(),
            action,
        })
    }
}

impl C220Mte2State {
    pub(super) fn resolve_vector_flag_id(
        machine: &ScalarMachine,
        pc: u64,
        word: u32,
    ) -> Result<u8, C220ExecutionError> {
        let instruction = FlagInstruction::decode(Architecture::Dav2201, word)
            .filter(|instruction| {
                instruction.source_pipe_code == 4 && instruction.trigger_pipe_code == 1
            })
            .ok_or(C220ExecutionError::UnsupportedWord { pc, word })?;
        let flag_id = instruction.resolve(pc, machine.xregs()).flag_id;
        u8::try_from(flag_id)
            .ok()
            .filter(|id| *id < 2)
            .ok_or(C220ExecutionError::UnsupportedVectorFlagId { flag_id })
    }
}
