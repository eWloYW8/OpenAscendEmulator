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
    /// Missing source registers read as zero; retain their identities for diagnostics.
    pub missing_registers: [Option<u8>; 2],
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
        let base = machine.xreg_value(instruction.base_register);
        let (offset, offset_register) = match instruction.offset {
            C220PreloadOffset::Immediate(value) => (Some(u64::from(value)), None),
            C220PreloadOffset::Register(register) => (machine.xreg_value(register), Some(register)),
        };
        let base_value = base.unwrap_or(0);
        let offset_value = offset.unwrap_or(0);
        Ok(Self {
            pc,
            word,
            instruction,
            base_value,
            offset_value,
            effective_address: base_value.wrapping_add(offset_value),
            missing_registers: [
                base.is_none().then_some(instruction.base_register),
                offset_register.filter(|_| offset.is_none()),
            ],
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
        let absent = C220PreloadOperands::capture(&machine, 0, 0x0100_21d3).unwrap();
        assert_eq!(absent.missing_registers, [Some(34), Some(35)]);
        assert_eq!(absent.effective_address, 0);
        machine.set_xreg(32, 0x2000).unwrap();
        let extra = C220PreloadOperands::capture(&machine, 0, 0x08c2_003f).unwrap();
        assert_eq!(extra.effective_address, 0x203f);
        assert_eq!(extra.missing_registers, [None, None]);
        assert_eq!(machine.xreg_value(32), Some(0x2000));
        assert_eq!(machine.xreg_value(33), None);
        let mut c310 = ScalarMachine::from_pem_initial_state(Architecture::Dav3510);
        assert!(c310.set_xreg(32, 1).is_err());
        assert_eq!(c310.xreg_value(32), None);
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
