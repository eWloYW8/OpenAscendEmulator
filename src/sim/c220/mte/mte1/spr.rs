use crate::isa::c220::mte::spr::C220Mte1SprWrite;
use crate::sim::common::scalar::{ScalarMachine, ScalarMachineError, ScalarSprStep};

pub(crate) fn capture_write(
    machine: &ScalarMachine,
    pc: u64,
    word: u32,
    instruction: C220Mte1SprWrite,
) -> Result<ScalarSprStep, ScalarMachineError> {
    let mask = match instruction.destination_spr {
        10 | 53 | 92 => u64::MAX,
        13 | 15 | 22 | 58 => 0xffff_ffff,
        54 => 0xfff,
        _ => return Err(ScalarMachineError::UnsupportedWord { pc, word }),
    };
    let source_value = machine.xregs()[usize::from(instruction.source_register)];
    Ok(ScalarSprStep {
        pc,
        word,
        destination_spr: instruction.destination_spr,
        prior_destination_value: machine.spr_value(instruction.destination_spr),
        source_register: Some(instruction.source_register),
        source_value,
        value: source_value & mask,
    })
}
