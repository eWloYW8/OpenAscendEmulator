use crate::isa::c220::mte::nd2nz::C220Nd2NzTransfer;
use std::num::NonZeroU32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Nd2NzWriteRequest {
    pub matrix_index: u16,
    pub first_row: u16,
    pub rows: u32,
    pub block_index: u32,
    pub destination_address: u64,
    pub bytes: u32,
    pub last_in_instruction: bool,
}

/// Lazy L1 write geometry. Read-response dependencies and send eligibility
/// belong to the execution engine, not to the address plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220Nd2NzWritePlan {
    transfer: C220Nd2NzTransfer,
    rows_per_write: NonZeroU32,
    next: u64,
    count: u64,
}

impl C220Nd2NzWritePlan {
    pub fn new(transfer: C220Nd2NzTransfer, rows_per_write: NonZeroU32) -> Self {
        let groups = u32::from(transfer.rows()).div_ceil(rows_per_write.get());
        Self {
            transfer,
            rows_per_write,
            next: 0,
            count: u64::from(transfer.matrix_count())
                * u64::from(groups)
                * u64::from(transfer.blocks_per_row()),
        }
    }

    pub fn remaining(&self) -> u64 {
        self.count - self.next
    }
}

impl Iterator for C220Nd2NzWritePlan {
    type Item = C220Nd2NzWriteRequest;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next == self.count {
            return None;
        }
        let blocks = u64::from(self.transfer.blocks_per_row());
        let groups = u32::from(self.transfer.rows()).div_ceil(self.rows_per_write.get());
        let block = (self.next % blocks) as u32;
        let matrix = (self.next / blocks / u64::from(groups)) as u16;
        let first_row =
            ((self.next / blocks % u64::from(groups)) as u32 * self.rows_per_write.get()) as u16;
        let rows =
            (u32::from(self.transfer.rows()) - u32::from(first_row)).min(self.rows_per_write.get());
        let segment_index =
            (u64::from(matrix) * u64::from(self.transfer.rows()) + u64::from(first_row)) * blocks
                + u64::from(block);
        let segment = self
            .transfer
            .segment(segment_index)
            .expect("bounded write coordinate");
        self.next += 1;
        Some(C220Nd2NzWriteRequest {
            matrix_index: matrix,
            first_row,
            rows,
            block_index: block,
            destination_address: segment.destination_address,
            bytes: rows * 32,
            last_in_instruction: self.next == self.count,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match usize::try_from(self.remaining()) {
            Ok(remaining) => (remaining, Some(remaining)),
            Err(_) => (usize::MAX, None),
        }
    }
}

impl std::iter::FusedIterator for C220Nd2NzWritePlan {}
