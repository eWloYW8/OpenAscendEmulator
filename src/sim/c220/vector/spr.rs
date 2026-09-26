use crate::isa::c220::vector::spr::C220VectorSprWrite;
use crate::sim::common::scalar::{ScalarMachine, ScalarMachineError, ScalarSprStep};

pub(super) fn execute_write(
    machine: &mut ScalarMachine,
    pc: u64,
    word: u32,
    instruction: C220VectorSprWrite,
) -> Result<ScalarSprStep, ScalarMachineError> {
    let mask = match instruction.destination_spr {
        12 | 17 | 48..=51 | 56 | 63 => u64::MAX,
        19 | 69 => 0xffff,
        55 | 60 => 0xffff_ffff_ffff,
        57 => 0xffff_ffff,
        _ => return Err(ScalarMachineError::UnsupportedWord { pc, word }),
    };
    let source_value = machine.xregs()[usize::from(instruction.source_register)];
    let prior_destination_value = machine.spr_value(instruction.destination_spr);
    let value = source_value & mask;
    machine.set_spr_value(instruction.destination_spr, value)?;
    Ok(ScalarSprStep {
        pc,
        word,
        destination_spr: instruction.destination_spr,
        prior_destination_value,
        source_register: Some(instruction.source_register),
        source_value,
        value,
    })
}
