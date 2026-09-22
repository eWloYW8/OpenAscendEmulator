mod uop;
pub use uop::{C220BtReadUop, C220BtRequestPlan, C220BtUop, C220BtUops, c220_bt_uops};

use thiserror::Error;

use crate::isa::c220::mte::bias::C220BtTransfer;
use crate::memory::pv_memory::PvMemoryError;
use crate::sim::c220::memory::{C220LocalBufferError, C220LocalMemory};
use crate::sim::c220::numeric::fp16::c220_fp16_to_fp32_bits;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct C220BtTransferResult {
    pub blocks: u32,
    pub input_bytes: u64,
    pub output_bytes: u64,
    pub converted_elements: u64,
    pub nan_elements: u64,
    pub infinity_elements: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220PreparedBtTransfer {
    writes: Vec<(u64, Vec<u8>)>,
    pub result: C220BtTransferResult,
}

impl C220PreparedBtTransfer {
    pub fn commit(
        self,
        memory: &mut C220LocalMemory,
    ) -> Result<C220BtTransferResult, C220BtTransferError> {
        for (address, bytes) in self.writes {
            memory.bt_mut().write(address, &bytes)?;
        }
        Ok(self.result)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum C220BtTransferError {
    #[error(transparent)]
    Source(#[from] C220LocalBufferError),
    #[error(transparent)]
    Destination(#[from] PvMemoryError),
    #[error("cannot reserve {requested} BT transfer records or bytes")]
    AllocationFailed { requested: usize },
}

pub fn prepare_c220_mov_l1_to_bt(
    memory: &C220LocalMemory,
    transfer: C220BtTransfer,
) -> Result<C220PreparedBtTransfer, C220BtTransferError> {
    let mut writes = Vec::new();
    let count = transfer.block_count() as usize;
    writes
        .try_reserve_exact(count)
        .map_err(|_| C220BtTransferError::AllocationFailed { requested: count })?;
    let mut result = C220BtTransferResult {
        blocks: transfer.block_count(),
        input_bytes: transfer.input_bytes(),
        output_bytes: transfer.output_bytes(),
        ..C220BtTransferResult::default()
    };
    for segment in transfer.segments() {
        segment
            .destination_address
            .checked_add(u64::from(segment.output_bytes) - 1)
            .ok_or(PvMemoryError::RangeOverflow)?;
        let input = memory
            .l1()
            .read_initialized_linear(segment.source_address, segment.input_bytes as usize)?;
        let output = if transfer.descriptor.convert_f16_to_f32 {
            let mut output = Vec::new();
            output
                .try_reserve_exact(segment.output_bytes as usize)
                .map_err(|_| C220BtTransferError::AllocationFailed {
                    requested: segment.output_bytes as usize,
                })?;
            for bytes in input.chunks_exact(2) {
                let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
                result.converted_elements += 1;
                result.nan_elements += u64::from(bits & 0x7fff > 0x7c00);
                result.infinity_elements += u64::from(bits & 0x7fff == 0x7c00);
                output.extend_from_slice(&c220_fp16_to_fp32_bits(bits).to_le_bytes());
            }
            output
        } else {
            input
        };
        writes.push((segment.destination_address, output));
    }
    Ok(C220PreparedBtTransfer { writes, result })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::c220::mte::bias::C220MovL1ToBtInstruction;
    use crate::sim::c220::memory::C220LocalMemoryConfig;
    use std::num::NonZeroU32;

    #[test]
    fn bt_copy_and_conversion_preserve_descriptor_units_and_special_values() {
        let word = (3 << 29) | (2 << 27) | (4 << 23) | (1 << 17) | (2 << 12) | (3 << 7) | (5 << 3);
        let instruction = C220MovL1ToBtInstruction::decode(word).unwrap();
        let values = [
            0x0000_u16, 0x8000, 0x0001, 0x03ff, 0x3c00, 0x7c00, 0xfc00, 0xfc01,
        ];
        let expected = [
            0,
            0x8000_0000,
            0x3380_0000,
            0x387f_c000,
            0x3f80_0000,
            0x7f80_0000,
            0xff80_0000,
            0x7fff_ffff,
        ];
        for convert in [false, true] {
            let mut registers = [0; 32];
            registers[1] = u64::from(u32::MAX) - 63;
            registers[2] = 128;
            registers[3] =
                (2 << 4) | (2 << 16) | (1 << 32) | (1 << 48) | if convert { 8 } else { 0 };
            let transfer = instruction.capture(&registers);
            let segments = transfer.segments().collect::<Vec<_>>();
            assert_eq!(
                segments
                    .iter()
                    .map(|s| s.source_address)
                    .collect::<Vec<_>>(),
                [128, 192, 288, 352]
            );
            let stride = if convert { 320 } else { 192 };
            assert_eq!(segments[2].destination_address, registers[1] + stride);
            let mut memory = C220LocalMemory::new(C220LocalMemoryConfig::default()).unwrap();
            let input = values
                .iter()
                .cycle()
                .take(32)
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>();
            for segment in &segments {
                memory
                    .l1_mut()
                    .write_known(segment.source_address, &input)
                    .unwrap();
            }
            let prepared = prepare_c220_mov_l1_to_bt(&memory, transfer).unwrap();
            assert_eq!(memory.bt().read_byte(registers[1]), 0);
            let result = prepared.commit(&mut memory).unwrap();
            assert_eq!(result.input_bytes, 256);
            assert_eq!(result.output_bytes, if convert { 512 } else { 256 });
            assert_eq!(result.nan_elements, if convert { 16 } else { 0 });
            assert_eq!(result.infinity_elements, if convert { 32 } else { 0 });
            for segment in &segments {
                let mut output = vec![0; segment.output_bytes as usize];
                memory
                    .bt()
                    .read_into(segment.destination_address, &mut output)
                    .unwrap();
                if convert {
                    let bits = output
                        .chunks_exact(4)
                        .map(|x| u32::from_le_bytes(x.try_into().unwrap()))
                        .collect::<Vec<_>>();
                    assert_eq!(
                        bits,
                        expected.into_iter().cycle().take(32).collect::<Vec<_>>()
                    );
                } else {
                    assert_eq!(output, input);
                }
            }
            let uops = c220_bt_uops(transfer, NonZeroU32::new(48).unwrap()).collect::<Vec<_>>();
            assert_eq!(uops.len(), 6);
            assert_eq!(uops[2].input_bytes, 32);
            assert!(uops[2].last_in_burst);
            assert_eq!(uops[1].destination_address, registers[1] + 48);
            assert_eq!(uops[3].destination_address, registers[1] + stride);
            registers[3] = 8;
            let empty = instruction.capture(&registers);
            assert_eq!(empty.segments().count(), 0);
            assert_eq!(c220_bt_uops(empty, NonZeroU32::MIN).count(), 0);
            assert_eq!(
                prepare_c220_mov_l1_to_bt(&memory, empty).unwrap().result,
                C220BtTransferResult::default()
            );
        }
    }
}
