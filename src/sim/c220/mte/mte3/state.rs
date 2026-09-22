use crate::architecture::Architecture;
use crate::isa::c220::mte::C220DmaMovDescriptor;
use crate::isa::flow::{FlagInstruction, FlagOperation};
use crate::memory::ub::{UbMemory, UbTransferResult};
use crate::sim::c220::mte::mte3::{
    C220Mte3TransferPlan, C220PreparedOutput, prepare_c220_mov_ub_to_hbm,
};
use crate::sim::c220::state::C220ExecutionError;
use crate::sim::common::scalar::{ScalarMachine, ScalarStepper};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum C220OutputAction {
    SetMte3Flag {
        flag_id: u8,
        source_address: u64,
    },
    SetMte2ReuseFlag {
        flag_id: u8,
    },
    WaitMte2ReuseFlag {
        flag_id: u8,
    },
    WaitMte3Flag {
        flag_id: u8,
        source_address: u64,
    },
    CopyToHbm {
        source_address: u64,
        destination_address: u64,
        transfer: UbTransferResult,
    },
    SetMte3CompletionFlag {
        flag_id: u8,
        source_address: u64,
    },
    WaitMte3CompletionFlag {
        flag_id: u8,
        source_address: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220OutputStep {
    pub pc: u64,
    pub word: u32,
    pub next_pc: u64,
    pub action: C220OutputAction,
}

impl C220Mte3State {
    pub(crate) fn step_word(
        &mut self,
        scalar: &mut ScalarStepper,
        ub: &UbMemory,
        word: u32,
        transfer: Option<C220Mte3TransferPlan>,
    ) -> Result<(C220OutputStep, Option<C220PreparedOutput>), C220ExecutionError> {
        let pc = scalar.pc();
        if scalar.is_halted() {
            return Err(C220ExecutionError::ProgramEnded { pc });
        }
        if scalar.machine().architecture() != Architecture::Dav2201 {
            return Err(C220ExecutionError::UnsupportedWord { pc, word });
        }
        let route = FlagInstruction::decode(Architecture::Dav2201, word).map(|instruction| {
            (
                instruction.source_pipe_code,
                instruction.trigger_pipe_code,
                instruction.operation,
            )
        });
        let mut prepared_output = None;
        let action = match route {
            Some((1, 5, FlagOperation::Set)) => {
                let flag_id = Self::resolve_output_flag_id(scalar.machine(), pc, word, 4)?;
                if self.mte3_flags[usize::from(flag_id)].is_some() {
                    return Err(C220ExecutionError::OutputFlagAlreadySet { flag_id });
                }
                let token = self
                    .unsignaled_output
                    .ok_or(C220ExecutionError::OutputNotProduced)?;
                self.unsignaled_output = None;
                self.mte3_flags[usize::from(flag_id)] = Some(token);
                C220OutputAction::SetMte3Flag {
                    flag_id,
                    source_address: token.source_address,
                }
            }
            Some((1, 4, FlagOperation::Set)) => {
                let flag_id = Self::resolve_output_flag_id(scalar.machine(), pc, word, 2)?;
                if self.mte2_reuse_flags[usize::from(flag_id)] {
                    return Err(C220ExecutionError::ReuseFlagAlreadySet { flag_id });
                }
                if self.unsignaled_output.is_some() {
                    return Err(C220ExecutionError::UnsignaledOutput);
                }
                if self.mte3_flags.iter().all(Option::is_none) && self.ready_output.is_none() {
                    return Err(C220ExecutionError::OutputNotProduced);
                }
                self.mte2_reuse_flags[usize::from(flag_id)] = true;
                C220OutputAction::SetMte2ReuseFlag { flag_id }
            }
            Some((1, 4, FlagOperation::Wait)) => {
                let flag_id = Self::resolve_output_flag_id(scalar.machine(), pc, word, 2)?;
                if !self.mte2_reuse_flags[usize::from(flag_id)] {
                    return Err(C220ExecutionError::ReuseWaitWithoutFlag { flag_id });
                }
                self.mte2_reuse_flags[usize::from(flag_id)] = false;
                C220OutputAction::WaitMte2ReuseFlag { flag_id }
            }
            Some((1, 5, FlagOperation::Wait)) => {
                let flag_id = Self::resolve_output_flag_id(scalar.machine(), pc, word, 4)?;
                if self.ready_output.is_some() {
                    return Err(C220ExecutionError::OutputDependencyOutstanding);
                }
                let token = self.mte3_flags[usize::from(flag_id)]
                    .take()
                    .ok_or(C220ExecutionError::OutputWaitWithoutFlag { flag_id })?;
                self.ready_output = Some(token);
                C220OutputAction::WaitMte3Flag {
                    flag_id,
                    source_address: token.source_address,
                }
            }
            None if C220DmaMovDescriptor::is_word(word) => {
                self.ready_output
                    .ok_or(C220ExecutionError::OutputNotReady)?;
                if self.copied_output.is_some()
                    || self.mte3_completion_flags.iter().any(Option::is_some)
                {
                    return Err(C220ExecutionError::OutputDependencyOutstanding);
                }
                let plan = transfer.ok_or(C220ExecutionError::UnsupportedWord { pc, word })?;
                let prepared = prepare_c220_mov_ub_to_hbm(
                    ub,
                    plan.descriptor,
                    plan.source_address,
                    plan.destination_address,
                )?;
                let transfer = prepared.result;
                prepared_output = Some(prepared);
                self.ready_output = None;
                self.copied_output = Some(C220OutputToken {
                    source_address: plan.source_address,
                });
                C220OutputAction::CopyToHbm {
                    source_address: plan.source_address,
                    destination_address: plan.destination_address,
                    transfer,
                }
            }
            Some((5, 1, FlagOperation::Set)) => {
                let flag_id = Self::resolve_output_flag_id(scalar.machine(), pc, word, 4)?;
                if self.mte3_completion_flags[usize::from(flag_id)].is_some() {
                    return Err(C220ExecutionError::CompletionFlagAlreadySet { flag_id });
                }
                let token = self
                    .copied_output
                    .ok_or(C220ExecutionError::OutputNotCopied)?;
                self.copied_output = None;
                self.mte3_completion_flags[usize::from(flag_id)] = Some(token);
                C220OutputAction::SetMte3CompletionFlag {
                    flag_id,
                    source_address: token.source_address,
                }
            }
            Some((5, 1, FlagOperation::Wait)) => {
                let flag_id = Self::resolve_output_flag_id(scalar.machine(), pc, word, 4)?;
                let token = self.mte3_completion_flags[usize::from(flag_id)]
                    .take()
                    .ok_or(C220ExecutionError::CompletionWaitWithoutFlag { flag_id })?;
                C220OutputAction::WaitMte3CompletionFlag {
                    flag_id,
                    source_address: token.source_address,
                }
            }
            _ => return Err(C220ExecutionError::UnsupportedWord { pc, word }),
        };
        scalar.advance_sequential();
        Ok((
            C220OutputStep {
                pc,
                word,
                next_pc: scalar.pc(),
                action,
            },
            prepared_output,
        ))
    }

    fn resolve_output_flag_id(
        machine: &ScalarMachine,
        pc: u64,
        word: u32,
        max_id: u8,
    ) -> Result<u8, C220ExecutionError> {
        let instruction = FlagInstruction::decode(Architecture::Dav2201, word)
            .filter(|instruction| {
                matches!(
                    (instruction.source_pipe_code, instruction.trigger_pipe_code),
                    (1, 5) | (1, 4) | (5, 1)
                )
            })
            .ok_or(C220ExecutionError::UnsupportedWord { pc, word })?;
        let flag_id = instruction.resolve(pc, machine.xregs()).flag_id;
        u8::try_from(flag_id)
            .ok()
            .filter(|id| *id < max_id)
            .ok_or(C220ExecutionError::UnsupportedOutputFlagId { flag_id })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct C220OutputToken {
    source_address: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct C220Mte3State {
    unsignaled_output: Option<C220OutputToken>,
    mte3_flags: [Option<C220OutputToken>; 4],
    mte2_reuse_flags: [bool; 2],
    ready_output: Option<C220OutputToken>,
    copied_output: Option<C220OutputToken>,
    mte3_completion_flags: [Option<C220OutputToken>; 4],
}

impl C220Mte3State {
    pub(crate) fn publish_vector_output(&mut self, source_address: u64) {
        self.unsignaled_output = Some(C220OutputToken { source_address });
    }

    pub(crate) const fn reuse_flags_set(&self) -> [bool; 2] {
        self.mte2_reuse_flags
    }

    pub(crate) fn barrier_busy(&self) -> bool {
        self.output_buffer_busy() || self.mte2_reuse_flags.iter().any(|set| *set)
    }

    pub(crate) fn output_buffer_busy(&self) -> bool {
        self.unsignaled_output.is_some()
            || self.mte3_flags.iter().any(Option::is_some)
            || self.ready_output.is_some()
            || self.copied_output.is_some()
            || self.mte3_completion_flags.iter().any(Option::is_some)
    }

    pub(crate) const fn output_flags_set(&self) -> [bool; 4] {
        [
            self.mte3_flags[0].is_some(),
            self.mte3_flags[1].is_some(),
            self.mte3_flags[2].is_some(),
            self.mte3_flags[3].is_some(),
        ]
    }

    pub(crate) const fn completion_flags_set(&self) -> [bool; 4] {
        [
            self.mte3_completion_flags[0].is_some(),
            self.mte3_completion_flags[1].is_some(),
            self.mte3_completion_flags[2].is_some(),
            self.mte3_completion_flags[3].is_some(),
        ]
    }
}
