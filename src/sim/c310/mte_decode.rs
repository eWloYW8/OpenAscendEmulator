use crate::architecture::Architecture;
use crate::isa::c310::mte::{C310MovAlignDecode, C310MovAlignRegisterSelectors};
use crate::sim::c310::transfer::C310TransferError;
use crate::sim::machine::ScalarMachine;
use crate::sim::mte_stepper::MteStepperError;

pub(crate) fn decode_mte2_transfer(
    machine: &ScalarMachine,
    pc: u64,
    word: u32,
) -> Result<(C310MovAlignDecode, usize), MteStepperError> {
    if machine.architecture() != Architecture::Dav3510 {
        return Err(MteStepperError::UnsupportedWord { pc, word });
    }
    let xregs = machine.xregs();
    let spr = |index| {
        machine
            .spr_value(index)
            .ok_or(MteStepperError::MissingSpr { pc, index })
    };
    let decoded = C310MovAlignRegisterSelectors::from_hbm_to_ub_word(word)
        .map_err(|_| MteStepperError::UnsupportedWord { pc, word })?
        .capture(xregs, spr(105)?, spr(106)?, spr(107)?)
        .decode_hbm_to_ub(word)?;
    let coordinates = decoded
        .parameters
        .coordinates()
        .map_err(C310TransferError::from)?;
    let bytes = coordinates
        .len()
        .checked_mul(decoded.burst_bytes as usize)
        .ok_or(MteStepperError::TransferSizeOverflow)?;
    Ok((decoded, bytes))
}
