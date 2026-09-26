use std::iter::FusedIterator;
use std::num::NonZeroU32;

use super::C220SmaskTransferError;
use crate::isa::c220::mte::smask::C220SmaskTransfer;
use crate::sim::c220::memory::l1::C220L1Access;
use crate::sim::c220::mte::interface::{
    C220MteL1OutputDestination, C220MteL1ReadOperation, C220MteL1ReadPort,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220SmaskReadUop {
    pub source_mode: u8,
    pub source_address: u64,
    pub input_offset: u32,
    pub input_bytes: u32,
    pub destination_address: u64,
    pub output_bytes: u32,
    pub completes_logical_uop: bool,
}

impl C220SmaskReadUop {
    pub const INPUT_PORT: C220MteL1ReadPort = C220MteL1ReadPort::Port0;

    pub fn operation(
        self,
        instruction_id: u64,
        output_bandwidth: NonZeroU32,
    ) -> C220MteL1ReadOperation<Self> {
        C220MteL1ReadOperation {
            instruction_id,
            access: C220L1Access {
                address: self.source_address,
                bytes: self.input_bytes,
            },
            destination: C220MteL1OutputDestination::Smask,
            output_address: self.destination_address,
            output_bytes: self.output_bytes,
            output_bandwidth,
            completes_logical_uop: self.completes_logical_uop,
            last_in_instruction: self.completes_logical_uop,
            payload: self,
        }
    }
}

/// One logical transfer split into physical reads, with a single output tail.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220SmaskRequestPlan {
    transfer: C220SmaskTransfer,
    offset: u32,
    access_width: NonZeroU32,
}

impl C220SmaskRequestPlan {
    pub(in crate::sim::c220::mte) fn for_source(
        transfer: C220SmaskTransfer,
        access_width: NonZeroU32,
    ) -> Result<Self, C220SmaskTransferError> {
        if transfer.instruction.source_mode == 0 {
            Self::new_external(transfer, access_width)
        } else {
            Self::new(transfer, access_width)
        }
    }

    pub fn new_external(
        transfer: C220SmaskTransfer,
        access_width: NonZeroU32,
    ) -> Result<Self, C220SmaskTransferError> {
        if transfer.instruction.source_mode != 0 {
            return Err(C220SmaskTransferError::NotExternal(
                transfer.instruction.source_mode,
            ));
        }
        Ok(Self {
            transfer,
            offset: 0,
            access_width,
        })
    }
    pub fn new(
        transfer: C220SmaskTransfer,
        access_width: NonZeroU32,
    ) -> Result<Self, C220SmaskTransferError> {
        if transfer.instruction.source_mode != 2 {
            return Err(C220SmaskTransferError::NotL1(
                transfer.instruction.source_mode,
            ));
        }
        Ok(Self {
            transfer,
            offset: 0,
            access_width,
        })
    }
}

impl Iterator for C220SmaskRequestPlan {
    type Item = C220SmaskReadUop;

    fn next(&mut self) -> Option<Self::Item> {
        let total = self.transfer.descriptor.bytes();
        if self.offset == total {
            return None;
        }
        let bytes = (total - self.offset).min(self.access_width.get());
        let request = C220SmaskReadUop {
            source_mode: self.transfer.instruction.source_mode,
            source_address: self
                .transfer
                .source_base
                .wrapping_add(u64::from(self.offset)),
            input_offset: self.offset,
            input_bytes: bytes,
            destination_address: self.transfer.destination_base,
            output_bytes: total,
            completes_logical_uop: self.offset + bytes == total,
        };
        self.offset += bytes;
        Some(request)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let count = (self.transfer.descriptor.bytes() - self.offset)
            .div_ceil(self.access_width.get()) as usize;
        (count, Some(count))
    }
}

impl ExactSizeIterator for C220SmaskRequestPlan {}
impl FusedIterator for C220SmaskRequestPlan {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::c220::mte::smask::C220MovSmaskInstruction;

    #[test]
    fn split_reads_preserve_full_output_only_final_response_completes() {
        let instruction = C220MovSmaskInstruction::decode(
            (3 << 29) | (17 << 22) | (1 << 17) | (2 << 12) | (3 << 2) | 2,
        )
        .unwrap();
        let mut registers = [0; 32];
        registers[1] = 511;
        registers[2] = u64::MAX - 31;
        registers[3] = 65;
        let mut plan = C220SmaskRequestPlan::new(
            instruction.capture(&registers),
            NonZeroU32::new(64).unwrap(),
        )
        .unwrap();
        assert_eq!(plan.len(), 3);
        for (index, (address, bytes)) in [(u64::MAX - 31, 64), (32, 64), (96, 2)]
            .into_iter()
            .enumerate()
        {
            let request = plan.next().unwrap();
            assert_eq!(request.source_address, address);
            assert_eq!(request.input_bytes, bytes);
            assert_eq!(request.destination_address, 511);
            assert_eq!(request.output_bytes, 130);
            let operation = request.operation(7, NonZeroU32::new(32).unwrap());
            assert_eq!(operation.completes_logical_uop, index == 2);
            assert_eq!(operation.last_in_instruction, index == 2);
            assert_eq!(operation.destination, C220MteL1OutputDestination::Smask);
            assert_eq!(plan.len(), 2 - index);
        }
        assert_eq!(plan.next(), None);
        registers[3] = 0;
        assert_eq!(
            C220SmaskRequestPlan::new(instruction.capture(&registers), NonZeroU32::MIN)
                .unwrap()
                .next(),
            None
        );
    }
}
