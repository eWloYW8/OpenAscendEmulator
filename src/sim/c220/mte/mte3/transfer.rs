use crate::architecture::Architecture;
use crate::isa::c220::mte::C220DmaMovDescriptor;
use crate::isa::c220::mte::C220MovInstruction;
use crate::memory::mapped::{MappedMemory, MappedMemoryError};
use crate::memory::sparse::MemoryByteState;
use crate::memory::ub::{UbMemory, UbMemoryError, UbTransferResult};
use crate::sim::c220::mte::C220TransferError;
use crate::sim::c220::state::C220ExecutionError;
use crate::sim::common::scalar::ScalarMachine;

pub struct C220PreparedOutput {
    writes: Vec<(u64, Vec<MemoryByteState>)>,
    pub result: UbTransferResult,
}

impl C220PreparedOutput {
    pub fn commit(
        &self,
        destination: &mut MappedMemory,
    ) -> Result<UbTransferResult, MappedMemoryError> {
        destination.write_segments_at(&self.writes)?;
        Ok(self.result)
    }
}

pub fn copy_c220_mov_ub_to_hbm(
    ub: &UbMemory,
    destination: &mut MappedMemory,
    descriptor: C220DmaMovDescriptor,
    source_address: u64,
    destination_address: u64,
) -> Result<UbTransferResult, C220TransferError> {
    let prepared = prepare_c220_mov_ub_to_hbm(ub, descriptor, source_address, destination_address)?;
    Ok(prepared.commit(destination)?)
}

pub fn prepare_c220_mov_ub_to_hbm(
    ub: &UbMemory,
    descriptor: C220DmaMovDescriptor,
    source_address: u64,
    destination_address: u64,
) -> Result<C220PreparedOutput, C220TransferError> {
    let segments = descriptor.segment_iter(source_address, destination_address)?;
    let segment_count = segments.len();
    let bytes = segments
        .len()
        .checked_mul(32)
        .ok_or(UbMemoryError::ResultSizeOverflow)?;
    let mut writes = Vec::new();
    let mut known_bytes = 0;
    for segment in segments {
        let states = ub.read_states(segment.source_local, segment.bytes as usize)?;
        if writes.len() == writes.capacity() {
            writes
                .try_reserve(1)
                .map_err(|_| UbMemoryError::HostAllocationFailed {
                    requested: writes.len() + 1,
                })?;
        }
        known_bytes += states
            .iter()
            .filter(|state| matches!(state, MemoryByteState::Known(_)))
            .count();
        writes.push((segment.destination_hbm, states));
    }
    Ok(C220PreparedOutput {
        writes,
        result: UbTransferResult {
            segment_count,
            bytes,
            known_bytes,
            unknown_bytes: bytes - known_bytes,
        },
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Mte3TransferPlan {
    pub descriptor: C220DmaMovDescriptor,
    pub source_address: u64,
    pub destination_address: u64,
    pub bytes: usize,
    pub dma_mode_word: u64,
}

pub(crate) fn decode_mte3_transfer(
    machine: &ScalarMachine,
    pc: u64,
    word: u32,
    isa_instance_index: u32,
) -> Result<C220Mte3TransferPlan, C220ExecutionError> {
    if machine.architecture() != Architecture::Dav2201 {
        return Err(C220ExecutionError::UnsupportedWord { pc, word });
    }
    let selectors = C220MovInstruction::decode(word)
        .filter(|_| C220DmaMovDescriptor::is_word(word))
        .ok_or(C220ExecutionError::UnsupportedWord { pc, word })?;
    let x = machine.xregs();
    let source_address = x[usize::from(selectors.source_register)];
    let destination_address = x[usize::from(selectors.destination_register)];
    let descriptor =
        C220DmaMovDescriptor::decode(word, x[usize::from(selectors.descriptor_register)])
            .map_err(C220TransferError::from)?;
    let bytes = usize::from(descriptor.burst_count) * usize::from(descriptor.burst_length) * 32;
    let dma_mode_word = if isa_instance_index == 0 {
        0
    } else {
        machine
            .spr_value(94)
            .ok_or(C220ExecutionError::MissingSpr { pc, index: 94 })?
    };
    Ok(C220Mte3TransferPlan {
        descriptor,
        source_address,
        destination_address,
        bytes,
        dma_mode_word,
    })
}
