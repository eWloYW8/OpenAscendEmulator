use crate::architecture::Architecture;
use crate::isa::c220::scalar::{C220PreloadOffset, C220ScalarPreload};
use crate::sim::common::scalar::{ScalarMachine, ScalarMachineError};

/// Captured address operands. A preload has no data destination.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220PreloadOperands {
    pub pc: u64,
    pub word: u32,
    pub instruction: C220ScalarPreload,
    pub base_value: u64,
    pub offset_value: u64,
    pub effective_address: u64,
}

impl C220PreloadOperands {
    pub fn capture(
        machine: &ScalarMachine,
        pc: u64,
        word: u32,
    ) -> Result<Self, ScalarMachineError> {
        let unsupported = || ScalarMachineError::UnsupportedWord { pc, word };
        if machine.architecture() != Architecture::Dav2201 {
            return Err(unsupported());
        }
        let instruction = C220ScalarPreload::decode(word).ok_or_else(unsupported)?;
        let read = |register: u8| {
            machine
                .xregs()
                .get(usize::from(register))
                .copied()
                .ok_or(ScalarMachineError::RegisterOutOfRange(register))
        };
        let base_value = read(instruction.base_register)?;
        let offset_value = match instruction.offset {
            C220PreloadOffset::Immediate(value) => u64::from(value),
            C220PreloadOffset::Register(register) => read(register)?,
        };
        Ok(Self {
            pc,
            word,
            instruction,
            base_value,
            offset_value,
            effective_address: base_value.wrapping_add(offset_value),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn address_offsets_are_unsigned_unscaled_and_wrapping() {
        let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
        machine.set_xreg(2, u64::MAX - 1).unwrap();
        machine.set_xreg(3, 3).unwrap();
        let register = C220PreloadOperands::capture(&machine, 0, 0x0100_21d0).unwrap();
        assert_eq!(register.effective_address, 1);
        let immediate = C220PreloadOperands::capture(&machine, 4, 0x08c0_2fff).unwrap();
        assert_eq!(immediate.offset_value, 4095);
        assert_eq!(immediate.effective_address, 4093);
        let extended = C220ScalarPreload::decode(0x0100_21d3).unwrap();
        assert_eq!(extended.base_register, 34);
        assert_eq!(extended.offset, C220PreloadOffset::Register(35));
        assert!(matches!(
            C220PreloadOperands::capture(&machine, 0, 0x0100_21d3),
            Err(ScalarMachineError::RegisterOutOfRange(34))
        ));
        assert_eq!(
            C220ScalarPreload::decode(0x08c2_2fff)
                .unwrap()
                .base_register,
            34
        );
        assert!(C220ScalarPreload::decode(0x08a0_2000).is_none());
        assert!(C220ScalarPreload::decode(0x28c0_2000).is_none());
    }
}
