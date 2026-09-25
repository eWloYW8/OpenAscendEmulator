use crate::architecture::Architecture;
use crate::isa::scalar::{ScalarInstruction, ScalarLoadStoreOperation};
use crate::sim::common::scalar::{ScalarMachine, ScalarMachineError};

/// Single-register load operands captured before cache execution. Register
/// updates remain deferred until the core delivers architectural completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220LoadOperands {
    pub pc: u64,
    pub word: u32,
    pub destination_register: u8,
    pub prior_destination_value: u64,
    pub base_register: u8,
    pub base_value: u64,
    pub offset_operand: Option<(u8, u64)>,
    pub effective_address: u64,
    pub updated_base: Option<u64>,
    pub width_bytes: u8,
}

impl C220LoadOperands {
    pub fn capture(
        machine: &ScalarMachine,
        pc: u64,
        word: u32,
    ) -> Result<Self, ScalarMachineError> {
        let unsupported = || ScalarMachineError::UnsupportedWord { pc, word };
        if machine.architecture() != Architecture::Dav2201 {
            return Err(unsupported());
        }
        let instruction =
            ScalarInstruction::from_word(Architecture::Dav2201, word).ok_or_else(unsupported)?;
        let (
            destination_register,
            base_register,
            width_bytes,
            offset_operand,
            address,
            updated_base,
        ) = match instruction {
            ScalarInstruction::ScalarLoadStoreImmediate {
                operation: ScalarLoadStoreOperation::Load,
                data_register,
                base_register,
                width_bytes,
                sign_extend,
                ..
            } if sign_extend != Some(true) => {
                let effect = instruction
                    .scalar_address_effect(machine.xregs()[usize::from(base_register)])
                    .expect("load address");
                (
                    data_register,
                    base_register,
                    width_bytes,
                    None,
                    effect.effective_address,
                    effect.updated_base,
                )
            }
            ScalarInstruction::ScalarIndexedLoad {
                destination_register,
                base_register,
                offset_register,
                width_bytes,
                post_index,
            } => {
                let base = machine.xregs()[usize::from(base_register)];
                let offset = machine.xregs()[usize::from(offset_register)];
                let adjusted = base.wrapping_add(offset.wrapping_mul(u64::from(width_bytes)));
                (
                    destination_register,
                    base_register,
                    width_bytes,
                    Some((offset_register, offset)),
                    if post_index { base } else { adjusted },
                    post_index.then_some(adjusted),
                )
            }
            _ => return Err(unsupported()),
        };
        Ok(Self {
            pc,
            word,
            destination_register,
            base_register,
            width_bytes,
            offset_operand,
            effective_address: address,
            updated_base,
            base_value: machine.xregs()[usize::from(base_register)],
            prior_destination_value: machine.xregs()[usize::from(destination_register)],
        })
    }
}
