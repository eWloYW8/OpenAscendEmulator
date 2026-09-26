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
        .checked_mul(descriptor.unit_bytes() as usize)
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
    pub biu_mode_word: u64,
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
    let bytes = descriptor.byte_count();
    let (biu_mode_word, dma_mode_word) = if isa_instance_index == 0 {
        (0, 0)
    } else {
        (
            machine
                .spr_value(93)
                .ok_or(C220ExecutionError::MissingSpr { pc, index: 93 })?,
            machine
                .spr_value(94)
                .ok_or(C220ExecutionError::MissingSpr { pc, index: 94 })?,
        )
    };
    Ok(C220Mte3TransferPlan {
        descriptor,
        source_address,
        destination_address,
        bytes,
        dma_mode_word,
        biu_mode_word,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{region::MemoryRegion, sparse::SparseMemory};
    use crate::sim::c220::mte::uop::{C220DmaUopRoute, mte3_uops};

    #[test]
    fn byte_lengths_preserve_source_padding_and_destination_gaps() {
        for length in [1_u32, 2, 31, 32, 33, 127, 128, 129] {
            for gap in [0_u64, 1] {
                let descriptor = C220DmaMovDescriptor::decode(
                    crate::isa::c220::mte::CAPTURED_C220_MOV_UB_TO_OUT_WORD | 1,
                    (u64::from(length) << 16) | (3 << 4) | 9 | (gap << 32) | (gap << 48),
                )
                .unwrap();
                let source_stride = length.div_ceil(32) * 32 + gap as u32 * 32;
                let destination_stride = u64::from(length) + gap * 32;
                let mut ub = UbMemory::new(1024, 1024);
                let input: Vec<_> = (0..1024)
                    .map(|i| MemoryByteState::Known((i % 251) as u8))
                    .collect();
                ub.write_states(0, &input).unwrap();
                let mut memory = MappedMemory::bind(
                    SparseMemory::new(vec![MemoryRegion::unknown(1024)], 1024, 1024),
                    &[0x2000],
                )
                .unwrap();
                let result =
                    copy_c220_mov_ub_to_hbm(&ub, &mut memory, descriptor, 1, 0x2003).unwrap();
                assert_eq!(result.bytes, 3 * length as usize);
                for burst in 0..3 {
                    let start = 1 + burst * source_stride as usize;
                    assert_eq!(
                        memory
                            .read_states_at(
                                0x2003 + burst as u64 * destination_stride,
                                length as usize
                            )
                            .unwrap(),
                        input[start..start + length as usize]
                    );
                    if gap != 0 {
                        assert_eq!(
                            memory
                                .read_states_at(
                                    0x2003 + burst as u64 * destination_stride + u64::from(length),
                                    32
                                )
                                .unwrap(),
                            vec![MemoryByteState::Unknown; 32]
                        );
                    }
                }
                let requests = mte3_uops(C220Mte3TransferPlan {
                    descriptor,
                    source_address: 1,
                    destination_address: 0x2003,
                    bytes: descriptor.byte_count(),
                    dma_mode_word: 5,
                    biu_mode_word: 5,
                })
                .unwrap();
                assert_eq!(requests.sid(), 9);
                let batch = gap == 0 && length.is_multiple_of(32);
                let requests: Vec<_> = requests.collect();
                assert_eq!(requests.iter().map(|r| r.bytes).sum::<u32>(), length * 3);
                for request in &requests {
                    assert_eq!(
                        request.route,
                        if batch {
                            C220DmaUopRoute::ContiguousBatch
                        } else {
                            C220DmaUopRoute::Ordinary
                        }
                    );
                    let offset = request.destination_address
                        - 0x2003
                        - u64::from(request.burst_index) * destination_stride;
                    assert_eq!(
                        request.source_address,
                        1 + u64::from(request.burst_index) * u64::from(source_stride) + offset
                    );
                    assert!(request.bytes <= 128);
                }
                assert!(requests.last().unwrap().last_in_burst);
            }
        }
    }
}
