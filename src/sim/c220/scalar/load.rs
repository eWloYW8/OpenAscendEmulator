use crate::architecture::Architecture;
use crate::isa::scalar::{ScalarInstruction, ScalarLoadStoreOperation};
use crate::sim::common::scalar::{ScalarMachine, ScalarMachineError};

/// Load operands captured before cache execution. Register
/// updates remain deferred until the core delivers architectural completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220LoadOperands {
    pub pc: u64,
    pub word: u32,
    pub destination_register: u8,
    pub prior_destination_value: u64,
    pub second_destination: Option<(u8, u64)>,
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
        if let ScalarInstruction::ScalarPairLoad {
            first_destination_register,
            second_destination_register,
            base_register,
            signed_offset,
            width_bytes,
            sign_extend: false,
            ..
        } = instruction
        {
            let base_value = machine.xreg_value(base_register).unwrap_or(0);
            return Ok(Self {
                pc,
                word,
                destination_register: first_destination_register,
                prior_destination_value: machine
                    .xreg_value(first_destination_register)
                    .unwrap_or(0),
                second_destination: Some((
                    second_destination_register,
                    machine.xreg_value(second_destination_register).unwrap_or(0),
                )),
                base_register,
                base_value,
                offset_operand: None,
                effective_address: base_value.wrapping_add(signed_offset as i64 as u64),
                updated_base: None,
                width_bytes,
            });
        }
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
                    .scalar_address_effect(machine.xreg_value(base_register).unwrap_or(0))
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
                let base = machine.xreg_value(base_register).unwrap_or(0);
                let offset = machine.xreg_value(offset_register).unwrap_or(0);
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
            base_value: machine.xreg_value(base_register).unwrap_or(0),
            prior_destination_value: machine.xreg_value(destination_register).unwrap_or(0),
            second_destination: None,
        })
    }

    pub fn destinations(&self) -> impl Iterator<Item = u8> {
        std::iter::once(self.destination_register)
            .chain(self.second_destination.map(|(register, _)| register))
    }

    /// Encoded registers absent from the C220 register file. Reads produce
    /// zero and writes are discarded; the operand IDs remain observable.
    pub fn missing_registers(&self) -> impl Iterator<Item = u8> {
        [
            Some(self.base_register),
            self.offset_operand.map(|(register, _)| register),
        ]
        .into_iter()
        .flatten()
        .chain(self.destinations())
        .filter(|register| *register > 32)
    }

    pub fn access_bytes(&self) -> usize {
        usize::from(self.width_bytes)
            * if self.second_destination.is_some() {
                2
            } else {
                1
            }
    }
}
