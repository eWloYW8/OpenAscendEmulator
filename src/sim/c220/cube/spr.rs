use crate::isa::c220::cube::spr::C220CubeSprWrite;
use crate::sim::common::scalar::{ScalarMachine, ScalarMachineError, ScalarSprStep};

pub(in crate::sim::c220) fn execute_write(
    machine: &mut ScalarMachine,
    pc: u64,
    word: u32,
    instruction: C220CubeSprWrite,
) -> Result<ScalarSprStep, ScalarMachineError> {
    let source_value = machine.xregs()[usize::from(instruction.source_register)];
    let step = ScalarSprStep {
        pc,
        word,
        destination_spr: instruction.destination_spr,
        prior_destination_value: machine.spr_value(instruction.destination_spr),
        source_register: Some(instruction.source_register),
        source_value,
        value: source_value,
    };
    machine.set_spr_value(step.destination_spr, step.value)?;
    Ok(step)
}
