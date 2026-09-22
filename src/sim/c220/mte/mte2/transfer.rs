use crate::architecture::Architecture;
use crate::isa::c220::mte::C220MovInstruction;
use crate::isa::c220::mte::{C220MovOutToUbDescriptor, C220MovOutToUbError, C220MovOutToUbSegment};
use crate::memory::mapped::MappedMemory;
use crate::memory::ub::{UbMemory, UbTransferResult};
use crate::sim::c220::mte::C220TransferError;
use crate::sim::c220::state::C220ExecutionError;
use crate::sim::common::scalar::ScalarMachine;

pub fn copy_c220_mov_out_to_ub(
    ub: &mut UbMemory,
    source: &MappedMemory,
    descriptor: C220MovOutToUbDescriptor,
    source_address: u64,
    destination_address: u64,
) -> Result<UbTransferResult, C220TransferError> {
    let segments = descriptor.segments(source_address, destination_address)?;
    Ok(ub.copy_segments(
        source,
        segments.into_iter().map(|segment| {
            (
                segment.source_hbm,
                segment.destination_local,
                segment.bytes as usize,
            )
        }),
    )?)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Mte2TransferPlan {
    pub descriptor: C220MovOutToUbDescriptor,
    pub source_address: u64,
    pub destination_address: u64,
    pub bytes: usize,
    pub dma_mode_word: u64,
}

impl C220Mte2TransferPlan {
    pub fn descriptor_segments(self) -> Result<Vec<C220MovOutToUbSegment>, C220MovOutToUbError> {
        self.descriptor
            .segments(self.source_address, self.destination_address)
    }
}

pub(super) fn decode_mte2_transfer(
    machine: &ScalarMachine,
    pc: u64,
    word: u32,
    isa_instance_index: u32,
) -> Result<C220Mte2TransferPlan, C220ExecutionError> {
    if machine.architecture() != Architecture::Dav2201 || !C220MovOutToUbDescriptor::is_word(word) {
        return Err(C220ExecutionError::UnsupportedWord { pc, word });
    }
    let selectors =
        C220MovInstruction::decode(word).ok_or(C220ExecutionError::UnsupportedWord { pc, word })?;
    let xregs = machine.xregs();
    let destination_address = xregs[usize::from(selectors.destination_register)];
    let source_address = xregs[usize::from(selectors.source_register)];
    let descriptor =
        C220MovOutToUbDescriptor::decode(word, xregs[usize::from(selectors.descriptor_register)])?;
    let bytes = descriptor
        .segments(source_address, destination_address)?
        .len()
        .checked_mul(32)
        .ok_or(C220ExecutionError::TransferSizeOverflow)?;
    let dma_mode_word = if isa_instance_index == 0 {
        0
    } else {
        machine
            .spr_value(93)
            .ok_or(C220ExecutionError::MissingSpr { pc, index: 93 })?
    };
    Ok(C220Mte2TransferPlan {
        descriptor,
        source_address,
        destination_address,
        bytes,
        dma_mode_word,
    })
}
