use crate::architecture::Architecture;
use crate::isa::c220::mte::{C220MovInstruction, C220MovOutToUbDescriptor};
use crate::sim::c220::core::functional::C220FunctionalError;
use crate::sim::c220::mte::transfer::C220Mte2TransferPlan;
use crate::sim::common::scalar::ScalarMachine;

pub(crate) fn decode_mte2_transfer(
    machine: &ScalarMachine,
    pc: u64,
    word: u32,
    isa_instance_index: u32,
) -> Result<C220Mte2TransferPlan, C220FunctionalError> {
    if machine.architecture() != Architecture::Dav2201 || !C220MovOutToUbDescriptor::is_word(word) {
        return Err(C220FunctionalError::UnsupportedWord { pc, word });
    }
    let selectors = C220MovInstruction::decode(word)
        .ok_or(C220FunctionalError::UnsupportedWord { pc, word })?;
    let xregs = machine.xregs();
    let destination_address = xregs[usize::from(selectors.destination_register)];
    let source_address = xregs[usize::from(selectors.source_register)];
    let descriptor =
        C220MovOutToUbDescriptor::decode(word, xregs[usize::from(selectors.descriptor_register)])?;
    let bytes = descriptor
        .segments(source_address, destination_address)?
        .len()
        .checked_mul(32)
        .ok_or(C220FunctionalError::TransferSizeOverflow)?;
    let dma_mode_word = if isa_instance_index == 0 {
        0
    } else {
        machine
            .spr_value(93)
            .ok_or(C220FunctionalError::MissingSpr { pc, index: 93 })?
    };
    Ok(C220Mte2TransferPlan {
        descriptor,
        source_address,
        destination_address,
        bytes,
        dma_mode_word,
    })
}
