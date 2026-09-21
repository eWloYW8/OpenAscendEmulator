use crate::architecture::Architecture;
use crate::isa::c220::mte::{C220DmaMovDescriptor, C220MovInstruction};
use crate::isa::flow::{FlagInstruction, FlagOperation};
use crate::memory::mapped::MappedMemory;
use crate::memory::ub::UbTransferResult;
use crate::sim::c220::mte::state::C220OutputToken;
use crate::sim::c220::mte::transfer::{
    C220Mte3TransferPlan, C220PreparedOutput, C220TransferError, prepare_c220_mov_ub_to_hbm,
};
use crate::sim::mte_stepper::{MteCoreStepper, MteStepperError};

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

impl MteCoreStepper {
    pub fn step_c220_output_word(
        &mut self,
        word: u32,
        destination: &mut MappedMemory,
    ) -> Result<C220OutputStep, MteStepperError> {
        self.step_c220_output_word_impl(word, Some(destination))
            .map(|(step, _)| step)
    }

    pub(crate) fn step_c220_output_word_deferred(
        &mut self,
        word: u32,
    ) -> Result<(C220OutputStep, Option<C220PreparedOutput>), MteStepperError> {
        self.step_c220_output_word_impl(word, None)
    }

    fn step_c220_output_word_impl(
        &mut self,
        word: u32,
        destination: Option<&mut MappedMemory>,
    ) -> Result<(C220OutputStep, Option<C220PreparedOutput>), MteStepperError> {
        let pc = self.scalar.pc();
        if self.scalar.is_halted() {
            return Err(MteStepperError::ProgramEnded { pc });
        }
        if self.scalar.machine().architecture() != Architecture::Dav2201 {
            return Err(MteStepperError::UnsupportedWord { pc, word });
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
                let flag_id = self.resolve_c220_output_flag_id(pc, word, 4)?;
                if self.c220.mte3_flags[usize::from(flag_id)].is_some() {
                    return Err(MteStepperError::OutputFlagAlreadySet { flag_id });
                }
                let token = self
                    .c220
                    .unsignaled_output
                    .ok_or(MteStepperError::OutputNotProduced)?;
                self.c220.unsignaled_output = None;
                self.c220.mte3_flags[usize::from(flag_id)] = Some(token);
                C220OutputAction::SetMte3Flag {
                    flag_id,
                    source_address: token.source_address,
                }
            }
            Some((1, 4, FlagOperation::Set)) => {
                let flag_id = self.resolve_c220_output_flag_id(pc, word, 2)?;
                if self.c220.mte2_reuse_flags[usize::from(flag_id)] {
                    return Err(MteStepperError::ReuseFlagAlreadySet { flag_id });
                }
                if self.c220.unsignaled_output.is_some() {
                    return Err(MteStepperError::UnsignaledOutput);
                }
                if self.c220.mte3_flags.iter().all(Option::is_none)
                    && self.c220.ready_output.is_none()
                {
                    return Err(MteStepperError::OutputNotProduced);
                }
                self.c220.mte2_reuse_flags[usize::from(flag_id)] = true;
                C220OutputAction::SetMte2ReuseFlag { flag_id }
            }
            Some((1, 4, FlagOperation::Wait)) => {
                let flag_id = self.resolve_c220_output_flag_id(pc, word, 2)?;
                if !self.c220.mte2_reuse_flags[usize::from(flag_id)] {
                    return Err(MteStepperError::ReuseWaitWithoutFlag { flag_id });
                }
                self.c220.mte2_reuse_flags[usize::from(flag_id)] = false;
                C220OutputAction::WaitMte2ReuseFlag { flag_id }
            }
            Some((1, 5, FlagOperation::Wait)) => {
                let flag_id = self.resolve_c220_output_flag_id(pc, word, 4)?;
                if self.c220.ready_output.is_some() {
                    return Err(MteStepperError::OutputDependencyOutstanding);
                }
                let token = self.c220.mte3_flags[usize::from(flag_id)]
                    .take()
                    .ok_or(MteStepperError::OutputWaitWithoutFlag { flag_id })?;
                self.c220.ready_output = Some(token);
                C220OutputAction::WaitMte3Flag {
                    flag_id,
                    source_address: token.source_address,
                }
            }
            None if C220DmaMovDescriptor::is_word(word) => {
                self.c220
                    .ready_output
                    .ok_or(MteStepperError::OutputNotReady)?;
                if self.c220.copied_output.is_some()
                    || self.c220.mte3_completion_flags.iter().any(Option::is_some)
                {
                    return Err(MteStepperError::OutputDependencyOutstanding);
                }
                let plan = self.preview_c220_mte3_transfer(word)?;
                let prepared = prepare_c220_mov_ub_to_hbm(
                    &self.ub,
                    plan.descriptor,
                    plan.source_address,
                    plan.destination_address,
                )?;
                if let Some(destination) = destination {
                    destination
                        .write_segments_at(&prepared.writes)
                        .map_err(C220TransferError::from)?;
                }
                let transfer = prepared.result;
                prepared_output = Some(prepared);
                self.c220.ready_output = None;
                self.c220.copied_output = Some(C220OutputToken {
                    source_address: plan.source_address,
                });
                C220OutputAction::CopyToHbm {
                    source_address: plan.source_address,
                    destination_address: plan.destination_address,
                    transfer,
                }
            }
            Some((5, 1, FlagOperation::Set)) => {
                let flag_id = self.resolve_c220_output_flag_id(pc, word, 4)?;
                if self.c220.mte3_completion_flags[usize::from(flag_id)].is_some() {
                    return Err(MteStepperError::CompletionFlagAlreadySet { flag_id });
                }
                let token = self
                    .c220
                    .copied_output
                    .ok_or(MteStepperError::OutputNotCopied)?;
                self.c220.copied_output = None;
                self.c220.mte3_completion_flags[usize::from(flag_id)] = Some(token);
                C220OutputAction::SetMte3CompletionFlag {
                    flag_id,
                    source_address: token.source_address,
                }
            }
            Some((5, 1, FlagOperation::Wait)) => {
                let flag_id = self.resolve_c220_output_flag_id(pc, word, 4)?;
                let token = self.c220.mte3_completion_flags[usize::from(flag_id)]
                    .take()
                    .ok_or(MteStepperError::CompletionWaitWithoutFlag { flag_id })?;
                C220OutputAction::WaitMte3CompletionFlag {
                    flag_id,
                    source_address: token.source_address,
                }
            }
            _ => return Err(MteStepperError::UnsupportedWord { pc, word }),
        };
        self.scalar.advance_sequential();
        Ok((
            C220OutputStep {
                pc,
                word,
                next_pc: self.scalar.pc(),
                action,
            },
            prepared_output,
        ))
    }

    fn resolve_c220_output_flag_id(
        &self,
        pc: u64,
        word: u32,
        max_id: u8,
    ) -> Result<u8, MteStepperError> {
        let instruction = FlagInstruction::decode(Architecture::Dav2201, word)
            .filter(|instruction| {
                matches!(
                    (instruction.source_pipe_code, instruction.trigger_pipe_code),
                    (1, 5) | (1, 4) | (5, 1)
                )
            })
            .ok_or(MteStepperError::UnsupportedWord { pc, word })?;
        let flag_id = instruction
            .resolve(pc, self.scalar.machine().xregs())
            .flag_id;
        u8::try_from(flag_id)
            .ok()
            .filter(|id| *id < max_id)
            .ok_or(MteStepperError::UnsupportedOutputFlagId { flag_id })
    }

    pub(crate) fn preview_c220_mte3_transfer(
        &self,
        word: u32,
    ) -> Result<C220Mte3TransferPlan, MteStepperError> {
        let pc = self.scalar.pc();
        let machine = self.scalar.machine();
        if machine.architecture() != Architecture::Dav2201 {
            return Err(MteStepperError::UnsupportedWord { pc, word });
        }
        let selectors = C220MovInstruction::decode(word)
            .filter(|_| C220DmaMovDescriptor::is_word(word))
            .ok_or(MteStepperError::UnsupportedWord { pc, word })?;
        let x = machine.xregs();
        let source_address = x[usize::from(selectors.source_register)];
        let destination_address = x[usize::from(selectors.destination_register)];
        let descriptor =
            C220DmaMovDescriptor::decode(word, x[usize::from(selectors.descriptor_register)])
                .map_err(C220TransferError::from)?;
        let bytes = usize::from(descriptor.burst_count) * usize::from(descriptor.burst_length) * 32;
        let dma_mode_word = if self.c220.isa_instance_index == 0 {
            0
        } else {
            machine
                .spr_value(94)
                .ok_or(MteStepperError::MissingSpr { pc, index: 94 })?
        };
        Ok(C220Mte3TransferPlan {
            descriptor,
            source_address,
            destination_address,
            bytes,
            dma_mode_word,
        })
    }
}
