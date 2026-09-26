use std::num::NonZeroU32;

use crate::isa::c220::mte::l1_to_out::C220MovL1ToOutTransfer;
use crate::sim::c220::mte::uop::C220DmaUopMode;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220L1OutputRoute {
    PerBurst,
    Contiguous,
    Gather,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220L1OutputRead {
    pub source_address: u64,
    pub bytes: u32,
    pub destination_address: u64,
    pub destination_bytes: u32,
    pub last_in_transaction: bool,
    pub last_in_instruction: bool,
}

/// Lazy L1 reads grouped by their external write transaction.
#[derive(Debug, Clone)]
pub struct C220L1OutputReadPlan {
    transfer: C220MovL1ToOutTransfer,
    mode: C220DmaUopMode,
    bandwidth: NonZeroU32,
    route: C220L1OutputRoute,
    groups: u32,
    group_bytes: u32,
    source_stride: u32,
    destination_stride: u64,
    group: u32,
    offset: u32,
    transaction_bytes: u32,
    read_offset: u32,
    gather_source_offset: u32,
}

impl C220L1OutputReadPlan {
    pub fn new(
        transfer: C220MovL1ToOutTransfer,
        mode: C220DmaUopMode,
        bandwidth: NonZeroU32,
    ) -> Self {
        let count = ((transfer.xm >> 4) & 4095) as u32;
        let length = (transfer.xm >> 16) as u16;
        let source_gap = (transfer.xm >> 32) as u16;
        let destination_gap = (transfer.xm >> 48) as u16;
        let route = if source_gap == 0 && destination_gap == 0 {
            C220L1OutputRoute::Contiguous
        } else if transfer.source_address.is_multiple_of(32)
            && transfer.destination_address.is_multiple_of(64)
            && source_gap != 0
            && destination_gap == 0
            && count > 1
            && length == 2
        {
            C220L1OutputRoute::Gather
        } else {
            C220L1OutputRoute::PerBurst
        };
        let flatten = route != C220L1OutputRoute::PerBurst;
        Self {
            transfer,
            mode,
            bandwidth,
            route,
            groups: if count == 0 || length == 0 {
                0
            } else if flatten {
                1
            } else {
                count
            },
            group_bytes: (u32::from(length) * 32).wrapping_mul(if flatten { count } else { 1 }),
            source_stride: (u32::from(length) + u32::from(source_gap)) * 32,
            destination_stride: (u64::from(length) + u64::from(destination_gap)) * 32,
            group: 0,
            offset: 0,
            transaction_bytes: 0,
            read_offset: 0,
            gather_source_offset: 0,
        }
    }

    pub const fn route(&self) -> C220L1OutputRoute {
        self.route
    }
}

impl Iterator for C220L1OutputReadPlan {
    type Item = C220L1OutputRead;

    fn next(&mut self) -> Option<Self::Item> {
        if self.group >= self.groups || self.group_bytes == 0 {
            return None;
        }
        let destination_address = self
            .transfer
            .destination_address
            .wrapping_add(u64::from(self.group) * self.destination_stride)
            .wrapping_add(u64::from(self.offset));
        if self.transaction_bytes == 0 {
            self.transaction_bytes = self
                .mode
                .split_bytes(destination_address, self.group_bytes - self.offset);
        }
        let (source_offset, bytes) = if self.route == C220L1OutputRoute::Gather {
            let source_offset = self.gather_source_offset;
            self.gather_source_offset = self.gather_source_offset.wrapping_add(self.source_stride);
            (u64::from(source_offset), 64)
        } else {
            (
                u64::from(self.group.wrapping_mul(self.source_stride))
                    + u64::from(self.offset)
                    + u64::from(self.read_offset),
                (self.transaction_bytes - self.read_offset).min(self.bandwidth.get()),
            )
        };
        self.read_offset += bytes;
        let last_in_transaction = self.read_offset == self.transaction_bytes;
        let read = C220L1OutputRead {
            source_address: self.transfer.source_address.wrapping_add(source_offset),
            bytes,
            destination_address,
            destination_bytes: self.transaction_bytes,
            last_in_transaction,
            last_in_instruction: last_in_transaction
                && self.offset + self.transaction_bytes == self.group_bytes
                && self.group + 1 == self.groups,
        };
        if last_in_transaction {
            self.offset += self.transaction_bytes;
            self.transaction_bytes = 0;
            self.read_offset = 0;
            if self.offset == self.group_bytes {
                self.group += 1;
                self.offset = 0;
            }
        }
        Some(read)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::c220::mte::l1_to_out::C220MovL1ToOutInstruction;

    #[test]
    fn transaction_splits_read_bandwidth_and_gather_rows() {
        let instruction =
            C220MovL1ToOutInstruction::decode((3 << 29) | (2 << 27) | (4 << 23) | (2 << 3))
                .unwrap();
        let mut transfer = C220MovL1ToOutTransfer {
            instruction,
            source_address: 0,
            destination_address: 0x1000,
            xm: (3 << 4) | (2 << 16) | (1 << 32),
        };
        let width = NonZeroU32::new(32).unwrap();
        let plan = C220L1OutputReadPlan::new(transfer, C220DmaUopMode::Wide512, width);
        assert_eq!(plan.route(), C220L1OutputRoute::Gather);
        let reads: Vec<_> = plan.collect();
        assert_eq!(
            reads
                .iter()
                .map(|r| (
                    r.source_address,
                    r.bytes,
                    r.destination_bytes,
                    r.last_in_transaction
                ))
                .collect::<Vec<_>>(),
            [
                (0, 64, 128, false),
                (96, 64, 128, true),
                (192, 64, 64, true)
            ]
        );
        assert!(reads.last().unwrap().last_in_instruction);
        transfer.xm = (2 << 4) | (8 << 16);
        transfer.destination_address = 0x1070;
        let plan = C220L1OutputReadPlan::new(transfer, C220DmaUopMode::Wide512, width);
        assert_eq!(plan.route(), C220L1OutputRoute::Contiguous);
        let reads: Vec<_> = plan.collect();
        assert_eq!(reads[0].bytes, 16);
        assert_eq!(reads[1].source_address, 16);
        assert_eq!(reads[1].destination_bytes, 128);
        assert_eq!(reads.iter().map(|r| r.bytes).sum::<u32>(), 512);
        assert_eq!(reads.iter().filter(|r| r.last_in_instruction).count(), 1);
        transfer.xm |= 1 << 48;
        assert_eq!(
            C220L1OutputReadPlan::new(transfer, C220DmaUopMode::Wide512, width).route(),
            C220L1OutputRoute::PerBurst
        );
        transfer.xm = 0;
        assert!(
            C220L1OutputReadPlan::new(transfer, C220DmaUopMode::Wide512, width)
                .next()
                .is_none()
        );
    }
}
