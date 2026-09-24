use crate::architecture::Architecture;
use crate::isa::c220::scalar::C220ScalarSprImmediate;
use crate::isa::scalar::ScalarInstruction;
use crate::sim::common::scalar::{ScalarMachine, ScalarMachineError, ScalarSprStep};

pub(crate) const fn scalar_write_mask(register: u16) -> Option<u64> {
    match register {
        0 | 2..=6 | 8 | 14 | 20 | 21 | 59 | 72 | 76 | 88 | 89 | 104 | 105 => Some(u64::MAX),
        1 | 9 | 16 | 73 | 85 | 86 => Some(0xffff),
        7 | 74 | 87 | 107 | 108 => Some(0xffff_ffff),
        18 | 90 => Some(0xff),
        11 | 62 | 75 | 77 | 91 => Some(1),
        67 | 68 => Some(0x1_ffff_ffff_ffff),
        _ => None,
    }
}

pub(crate) fn write_destination(word: u32) -> Option<u16> {
    if let Some(instruction) = C220ScalarSprImmediate::decode(word) {
        if instruction.destination_spr == 3 {
            return None;
        }
        scalar_write_mask(instruction.destination_spr)?;
        return Some(instruction.destination_spr);
    }
    let ScalarInstruction::ScalarKey2MoveToSpr {
        encoded_destination_spr,
        ..
    } = ScalarInstruction::from_word(Architecture::Dav2201, word)?
    else {
        return None;
    };
    scalar_write_mask(encoded_destination_spr)?;
    Some(encoded_destination_spr)
}

pub(crate) fn execute_write(
    machine: &mut ScalarMachine,
    pc: u64,
    word: u32,
) -> Result<ScalarSprStep, ScalarMachineError> {
    let encoded_destination_spr =
        write_destination(word).ok_or(ScalarMachineError::UnsupportedWord { pc, word })?;
    let (source_register, source_value) =
        if let Some(instruction) = C220ScalarSprImmediate::decode(word) {
            (None, u64::from(instruction.immediate))
        } else if let Some(ScalarInstruction::ScalarKey2MoveToSpr {
            source_register, ..
        }) = ScalarInstruction::from_word(Architecture::Dav2201, word)
        {
            (
                Some(source_register),
                machine.xregs()[usize::from(source_register)],
            )
        } else {
            return Err(ScalarMachineError::UnsupportedWord { pc, word });
        };
    let mask = scalar_write_mask(encoded_destination_spr)
        .ok_or(ScalarMachineError::UnsupportedWord { pc, word })?;
    let prior_destination_value = machine.spr_value(encoded_destination_spr);
    let value = source_value & mask;
    machine.set_spr_value(encoded_destination_spr, value)?;
    Ok(ScalarSprStep {
        pc,
        word,
        destination_spr: encoded_destination_spr,
        prior_destination_value,
        source_register,
        source_value,
        value,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220ScalarSprTimingTicket {
    pub destination_spr: u16,
    pub issue_tick: u64,
    pub retire_tick: u64,
}
