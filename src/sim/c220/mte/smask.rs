use crate::isa::c220::mte::smask::C220SmaskTransfer;
use crate::memory::mapped::{MappedMemory, MappedMemoryError};
use crate::memory::pv_memory::PvMemoryError;
use crate::sim::c220::memory::{C220LocalBufferError, C220LocalMemory};

mod request;
pub use request::{C220SmaskReadUop, C220SmaskRequestPlan};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220SmaskTransferResult {
    pub elements: u8,
    pub bytes: u32,
    pub first_destination: Option<u32>,
    pub next_destination: Option<u8>,
}

#[derive(Debug, thiserror::Error)]
pub enum C220SmaskTransferError {
    #[error("source mode {0} is not an L1 sparse-mask transfer")]
    NotL1(u8),
    #[error("source mode {0} is not an external sparse-mask transfer")]
    NotExternal(u8),
    #[error(transparent)]
    ExternalSource(#[from] MappedMemoryError),
    #[error(transparent)]
    Source(#[from] C220LocalBufferError),
    #[error(transparent)]
    Destination(#[from] PvMemoryError),
}

/// Functional transfer only; timing and instruction retirement are caller-owned.
pub fn execute_c220_mov_l1_to_smask(
    memory: &mut C220LocalMemory,
    transfer: C220SmaskTransfer,
) -> Result<C220SmaskTransferResult, C220SmaskTransferError> {
    if transfer.instruction.source_mode != 2 {
        return Err(C220SmaskTransferError::NotL1(
            transfer.instruction.source_mode,
        ));
    }
    let input = if transfer.descriptor.is_empty() {
        Vec::new()
    } else {
        memory
            .l1()
            .read_initialized_linear(transfer.source_base, transfer.descriptor.bytes() as usize)?
    };
    write_smask(memory, transfer, &input)
}

/// Functional transfer only; this does not issue bus requests or advance time.
pub fn execute_c220_mov_out_to_smask(
    memory: &mut C220LocalMemory,
    source: &MappedMemory,
    transfer: C220SmaskTransfer,
) -> Result<C220SmaskTransferResult, C220SmaskTransferError> {
    if transfer.instruction.source_mode != 0 {
        return Err(C220SmaskTransferError::NotExternal(
            transfer.instruction.source_mode,
        ));
    }
    let input = if transfer.descriptor.is_empty() {
        Vec::new()
    } else {
        source.read_known_at(transfer.source_base, transfer.descriptor.bytes() as usize)?
    };
    write_smask(memory, transfer, &input)
}

fn write_smask(
    memory: &mut C220LocalMemory,
    transfer: C220SmaskTransfer,
    input: &[u8],
) -> Result<C220SmaskTransferResult, C220SmaskTransferError> {
    let mut result = C220SmaskTransferResult {
        elements: transfer.descriptor.count,
        bytes: transfer.descriptor.bytes(),
        first_destination: None,
        next_destination: None,
    };
    if input.is_empty() {
        return Ok(result);
    }
    let mut destination = transfer.destination_base as u32;
    result.first_destination = Some(destination);
    for element in input.chunks_exact(2) {
        memory.smask_mut().write(u64::from(destination), element)?;
        destination = u32::from(destination.wrapping_add(2) as u8);
    }
    result.next_destination = Some(destination as u8);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::c220::mte::smask::C220MovSmaskInstruction;
    use crate::memory::{region::MemoryRegion, sparse::SparseMemory};

    #[test]
    fn external_source_uses_full_address_and_reads_before_writing() {
        let source_base = (1_u64 << 40) + 4096;
        let bytes: Vec<u8> = (0..510).map(|i| (i / 2) as u8).collect();
        let source = MappedMemory::bind(
            SparseMemory::new(
                vec![MemoryRegion::new(512, bytes).unwrap()],
                16,
                1024,
            ),
            &[source_base],
        )
        .unwrap();
        let mut memory = C220LocalMemory::new(Default::default()).unwrap();
        let instruction = C220MovSmaskInstruction::decode(
            (3 << 29) | (17 << 22) | (1 << 17) | (2 << 12) | (3 << 2),
        )
        .unwrap();
        let mut registers = [0; 32];
        registers[1] = (1_u64 << 32) | 255;
        registers[2] = source_base;
        registers[3] = 127 | (1 << 11);
        let transfer = instruction.capture(&registers);
        let result = execute_c220_mov_out_to_smask(&mut memory, &source, transfer).unwrap();
        assert_eq!(result.elements, 255);
        assert_eq!(result.bytes, 510);
        assert_eq!(result.first_destination, Some(255));
        assert_eq!(result.next_destination, Some(253));
        assert_eq!(memory.smask().read_byte(255), 128);
        assert_eq!(memory.smask().read_byte(256), 128);
        assert_eq!(memory.smask().read_byte(251), 254);
        assert_eq!(memory.smask().read_byte(252), 254);
        assert_eq!(memory.smask().read_byte(1_u64 << 32), 0);

        let before = memory.clone();
        registers[2] = source_base + 508;
        registers[3] = 2;
        assert!(matches!(
            execute_c220_mov_out_to_smask(&mut memory, &source, instruction.capture(&registers)),
            Err(C220SmaskTransferError::ExternalSource(_))
        ));
        assert_eq!(memory, before);
        registers[2] = u64::MAX;
        registers[3] = 0;
        assert_eq!(
            execute_c220_mov_out_to_smask(&mut memory, &source, instruction.capture(&registers))
                .unwrap()
                .first_destination,
            None
        );
        assert_eq!(memory, before);
    }

    #[test]
    fn first_address_is_linear_and_each_halfword_advances_with_wrap() {
        let instruction = C220MovSmaskInstruction::decode(
            (3 << 29) | (17 << 22) | (1 << 17) | (2 << 12) | (3 << 2) | 2,
        )
        .unwrap();
        for first in [255_u64, 511, u64::from(u32::MAX)] {
            let mut memory = C220LocalMemory::new(Default::default()).unwrap();
            memory
                .l1_mut()
                .write_known(16, &[1, 2, 3, 4, 5, 6])
                .unwrap();
            let mut registers = [0; 32];
            registers[1] = first;
            registers[2] = 16;
            registers[3] = 3;
            let result =
                execute_c220_mov_l1_to_smask(&mut memory, instruction.capture(&registers)).unwrap();
            assert_eq!(result.first_destination, Some(first as u32));
            assert_eq!(result.next_destination, Some(5));
            assert_eq!(memory.smask().read_byte(first), 1);
            assert_eq!(memory.smask().read_byte(first + 1), 2);
            let mut tail = [0; 4];
            memory.smask().read_into(1, &mut tail).unwrap();
            assert_eq!(tail, [3, 4, 5, 6]);
            assert_eq!(memory.smask().read_byte(0), 0);
            registers[3] = 0;
            registers[2] = u64::MAX;
            let before = memory.clone();
            let empty =
                execute_c220_mov_l1_to_smask(&mut memory, instruction.capture(&registers)).unwrap();
            assert_eq!(empty.bytes, 0);
            assert_eq!(empty.first_destination, None);
            assert_eq!(memory, before);
        }
    }
}
