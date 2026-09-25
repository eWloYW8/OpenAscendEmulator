use crate::architecture::Architecture;
use crate::isa::scalar::{ScalarInstruction, ScalarLoadStoreOperation, ScalarStoreImmediateValue};
use crate::sim::common::scalar::{ScalarMachine, ScalarMachineError};

/// Captured scalar store; capture does not update the base or source registers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220StoreOperands {
    pub pc: u64,
    pub word: u32,
    pub source_operand: Option<(u8, u64)>,
    pub second_source_operand: Option<(u8, u64)>,
    pub base_register: u8,
    pub base_value: u64,
    pub offset_operand: Option<(u8, u64)>,
    pub effective_address: u64,
    pub updated_base: Option<u64>,
    pub width_bytes: u8,
    bytes: [u8; 16],
}

impl C220StoreOperands {
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
        let read = |register: u8| machine.xregs()[usize::from(register)];
        if let ScalarInstruction::ScalarPairStore {
            first_source_register,
            second_source_register,
            base_register,
            signed_offset,
            width_bytes,
            ..
        } = instruction
        {
            let first = read(first_source_register);
            let second = read(second_source_register);
            let base_value = read(base_register);
            let width = usize::from(width_bytes);
            let mut bytes = [0; 16];
            bytes[..width].copy_from_slice(&first.to_le_bytes()[..width]);
            bytes[width..2 * width].copy_from_slice(&second.to_le_bytes()[..width]);
            return Ok(Self {
                pc,
                word,
                source_operand: Some((first_source_register, first)),
                second_source_operand: Some((second_source_register, second)),
                base_register,
                base_value,
                offset_operand: None,
                effective_address: base_value.wrapping_add(signed_offset as i64 as u64),
                updated_base: None,
                width_bytes,
                bytes,
            });
        }
        let constant = |value| match value {
            ScalarStoreImmediateValue::Zero => 0,
            ScalarStoreImmediateValue::One => 1,
            ScalarStoreImmediateValue::Ones => u64::MAX,
        };
        let (base_register, width_bytes, source_operand, offset_operand, offset, post_index, value) =
            match instruction {
                ScalarInstruction::ScalarLoadStoreImmediate {
                    operation: ScalarLoadStoreOperation::Store,
                    data_register,
                    base_register,
                    width_bytes,
                    signed_offset,
                    post_index,
                    ..
                } => (
                    base_register,
                    width_bytes,
                    Some((data_register, read(data_register))),
                    None,
                    signed_offset as i64 as u64,
                    post_index,
                    read(data_register),
                ),
                ScalarInstruction::ScalarIndexedStore {
                    source_register,
                    base_register,
                    offset_register,
                    width_bytes,
                    post_index,
                } => (
                    base_register,
                    width_bytes,
                    Some((source_register, read(source_register))),
                    Some((offset_register, read(offset_register))),
                    read(offset_register).wrapping_mul(u64::from(width_bytes)),
                    post_index,
                    read(source_register),
                ),
                ScalarInstruction::ScalarIndexedImmediateStore {
                    base_register,
                    offset_register,
                    width_bytes,
                    post_index,
                    value,
                } => (
                    base_register,
                    width_bytes,
                    None,
                    Some((offset_register, read(offset_register))),
                    read(offset_register).wrapping_mul(u64::from(width_bytes)),
                    post_index,
                    constant(value),
                ),
                ScalarInstruction::ScalarStoreImmediate {
                    base_register,
                    width_bytes,
                    signed_offset,
                    post_index,
                    value,
                } => (
                    base_register,
                    width_bytes,
                    None,
                    None,
                    signed_offset as i64 as u64,
                    post_index,
                    constant(value),
                ),
                _ => return Err(unsupported()),
            };
        let base_value = read(base_register);
        let adjusted = base_value.wrapping_add(offset);
        let mut bytes = [0; 16];
        bytes[..8].copy_from_slice(&value.to_le_bytes());
        Ok(Self {
            pc,
            word,
            source_operand,
            second_source_operand: None,
            base_register,
            base_value,
            offset_operand,
            effective_address: if post_index { base_value } else { adjusted },
            updated_base: post_index.then_some(adjusted),
            width_bytes,
            bytes,
        })
    }

    pub fn bytes(&self) -> &[u8] {
        let count = if self.second_source_operand.is_some() {
            2
        } else {
            1
        };
        &self.bytes[..usize::from(self.width_bytes) * count]
    }
}
