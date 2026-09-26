use std::num::NonZeroU32;

use crate::isa::c220::mte::nd2nz::C220Nd2NzTransfer;
use crate::sim::c220::mte::uop::C220DmaUopMode;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220Nd2NzReadRoute {
    PerRow,
    ContiguousRows,
}

impl C220Nd2NzReadRoute {
    pub fn select(transfer: C220Nd2NzTransfer, alignment_depth: u32) -> Self {
        let row_bytes = transfer.row_bytes();
        if row_bytes > 63
            && row_bytes <= alignment_depth
            && transfer.columns() == transfer.source_row_stride()
            && transfer.columns().is_multiple_of(32)
        {
            Self::ContiguousRows
        } else {
            Self::PerRow
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Nd2NzReadElement {
    pub row_slot: u32,
    pub bytes: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220Nd2NzReadRequest {
    pub matrix_index: u16,
    pub source_address: u64,
    pub bytes: u32,
    pub padding_bytes: u32,
    pub elements: Vec<C220Nd2NzReadElement>,
    pub route: C220Nd2NzReadRoute,
    pub last_in_instruction: bool,
}

/// Request geometry only; both routes use in-order returns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220Nd2NzReadPlan {
    transfer: C220Nd2NzTransfer,
    route: C220Nd2NzReadRoute,
    mode: C220DmaUopMode,
    rows_per_group: NonZeroU32,
    matrix: u16,
    first_row: u32,
    remaining: Vec<u32>,
    contiguous_offset: u32,
}

impl C220Nd2NzReadPlan {
    pub fn new(
        transfer: C220Nd2NzTransfer,
        route: C220Nd2NzReadRoute,
        mode: C220DmaUopMode,
        rows_per_group: NonZeroU32,
    ) -> Self {
        Self {
            transfer,
            route,
            mode,
            rows_per_group,
            matrix: 0,
            first_row: 0,
            remaining: Vec::new(),
            contiguous_offset: 0,
        }
    }

    fn row_base(&self, row: u32) -> u64 {
        self.transfer
            .segment(
                (u64::from(self.matrix) * u64::from(self.transfer.rows()) + u64::from(row))
                    * u64::from(self.transfer.blocks_per_row()),
            )
            .expect("bounded read row")
            .source_address
    }

    fn next_per_row(&mut self) -> Option<C220Nd2NzReadRequest> {
        if self.remaining.is_empty() {
            let rows =
                (u32::from(self.transfer.rows()) - self.first_row).min(self.rows_per_group.get());
            self.remaining
                .resize(rows as usize, self.transfer.row_bytes());
        }
        let slot = self
            .remaining
            .iter()
            .enumerate()
            .max_by(|(a, bytes_a), (b, bytes_b)| bytes_a.cmp(bytes_b).then_with(|| b.cmp(a)))
            .map(|(slot, _)| slot)?;
        let offset = self.transfer.row_bytes() - self.remaining[slot];
        let source_address = self
            .row_base(self.first_row + slot as u32)
            .wrapping_add(u64::from(offset));
        let bytes = self.mode.split_bytes(source_address, self.remaining[slot]);
        self.remaining[slot] -= bytes;
        let padding_bytes = if self.remaining[slot] == 0 {
            self.transfer.blocks_per_row() * self.transfer.instruction.format.block_bytes()
                - self.transfer.row_bytes()
        } else {
            0
        };
        let matrix_index = self.matrix;
        if self.remaining.iter().all(|&remaining| remaining == 0) {
            self.first_row += self.remaining.len() as u32;
            self.remaining.clear();
            if self.first_row == u32::from(self.transfer.rows()) {
                self.matrix += 1;
                self.first_row = 0;
            }
        }
        Some(C220Nd2NzReadRequest {
            matrix_index,
            source_address,
            bytes,
            padding_bytes,
            elements: vec![C220Nd2NzReadElement {
                row_slot: slot as u32,
                bytes,
            }],
            route: self.route,
            last_in_instruction: self.matrix == self.transfer.matrix_count(),
        })
    }

    fn next_contiguous(&mut self) -> Option<C220Nd2NzReadRequest> {
        let padded_row =
            self.transfer.blocks_per_row() * self.transfer.instruction.format.block_bytes();
        let matrix_bytes = padded_row.wrapping_mul(u32::from(self.transfer.rows()));
        if matrix_bytes == 0 {
            self.matrix = self.transfer.matrix_count();
            return None;
        }
        let source_address = self
            .row_base(0)
            .wrapping_add(u64::from(self.contiguous_offset));
        let bytes = self
            .mode
            .split_bytes(source_address, matrix_bytes - self.contiguous_offset);
        let mut elements = Vec::new();
        let mut offset = self.contiguous_offset;
        let end = offset + bytes;
        while offset < end {
            let length = (padded_row - offset % padded_row).min(end - offset);
            elements.push(C220Nd2NzReadElement {
                row_slot: (offset / padded_row) % self.rows_per_group.get(),
                bytes: length,
            });
            offset += length;
        }
        let matrix_index = self.matrix;
        self.contiguous_offset = end;
        if end == matrix_bytes {
            self.matrix += 1;
            self.contiguous_offset = 0;
        }
        Some(C220Nd2NzReadRequest {
            matrix_index,
            source_address,
            bytes,
            padding_bytes: 0,
            elements,
            route: self.route,
            last_in_instruction: self.matrix == self.transfer.matrix_count(),
        })
    }
}

impl Iterator for C220Nd2NzReadPlan {
    type Item = C220Nd2NzReadRequest;

    fn next(&mut self) -> Option<Self::Item> {
        if self.transfer.is_disabled() || self.matrix == self.transfer.matrix_count() {
            return None;
        }
        match self.route {
            C220Nd2NzReadRoute::PerRow => self.next_per_row(),
            C220Nd2NzReadRoute::ContiguousRows => self.next_contiguous(),
        }
    }
}

impl std::iter::FusedIterator for C220Nd2NzReadPlan {}
