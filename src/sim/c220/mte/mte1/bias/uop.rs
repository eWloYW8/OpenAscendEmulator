use std::iter::FusedIterator;
use std::num::NonZeroU32;

use crate::isa::c220::mte::bias::{C220_BT_INPUT_BLOCK_BYTES, C220BtTransfer};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220BtUop {
    pub burst_index: u16,
    pub source_address: u64,
    /// Timing-request offsets remain in input bytes even when conversion is enabled.
    pub destination_address: u64,
    pub input_bytes: u32,
    pub output_bytes: u32,
    pub last_in_burst: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220BtUops {
    transfer: C220BtTransfer,
    bandwidth: NonZeroU32,
    burst_index: u16,
    offset: u32,
}

pub fn c220_bt_uops(transfer: C220BtTransfer, input_bytes_per_uop: NonZeroU32) -> C220BtUops {
    C220BtUops {
        transfer,
        bandwidth: input_bytes_per_uop,
        burst_index: 0,
        offset: 0,
    }
}

impl Iterator for C220BtUops {
    type Item = C220BtUop;

    fn next(&mut self) -> Option<Self::Item> {
        let descriptor = self.transfer.descriptor;
        if descriptor.is_empty() || self.burst_index >= descriptor.burst_count {
            return None;
        }
        let burst_bytes = u32::from(descriptor.burst_blocks) * C220_BT_INPUT_BLOCK_BYTES;
        let input_bytes = (burst_bytes - self.offset).min(self.bandwidth.get());
        let last_in_burst = self.offset + input_bytes == burst_bytes;
        let uop = C220BtUop {
            burst_index: self.burst_index,
            source_address: self
                .transfer
                .source_base
                .wrapping_add(u64::from(
                    u32::from(self.burst_index).wrapping_mul(descriptor.source_stride()),
                ))
                .wrapping_add(u64::from(self.offset)),
            destination_address: self
                .transfer
                .destination_base
                .wrapping_add(u64::from(self.burst_index) * descriptor.destination_stride())
                .wrapping_add(u64::from(self.offset)),
            input_bytes,
            output_bytes: input_bytes * if descriptor.convert_f16_to_f32 { 2 } else { 1 },
            last_in_burst,
        };
        if last_in_burst {
            self.burst_index += 1;
            self.offset = 0;
        } else {
            self.offset += input_bytes;
        }
        Some(uop)
    }
}

impl FusedIterator for C220BtUops {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220BtReadUop {
    pub logical: C220BtUop,
    pub source_address: u64,
    pub input_bytes: u32,
    pub input_offset: u32,
    /// Only this fragment's response advances the logical uop to output.
    pub completes_logical_uop: bool,
    pub last_in_instruction: bool,
}

impl C220BtReadUop {
    pub const L1_INPUT_PORT: usize = 0;
}

/// Lazy physical L1 requests; storage is independent of the transfer size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220BtRequestPlan {
    logical: C220BtUops,
    current: Option<C220BtUop>,
    offset: u32,
    access_width: NonZeroU32,
    remaining: u64,
}

impl C220BtRequestPlan {
    pub fn new(
        transfer: C220BtTransfer,
        l1_to_bt_bandwidth: NonZeroU32,
        l1_dmac_access_width: NonZeroU32,
    ) -> Self {
        let burst_bytes = u32::from(transfer.descriptor.burst_blocks) * C220_BT_INPUT_BLOCK_BYTES;
        let bandwidth = l1_to_bt_bandwidth.get();
        let width = l1_dmac_access_width.get();
        let full = u64::from(burst_bytes / bandwidth) * u64::from(bandwidth.div_ceil(width));
        let tail = u64::from((burst_bytes % bandwidth).div_ceil(width));
        Self {
            logical: c220_bt_uops(transfer, l1_to_bt_bandwidth),
            current: None,
            offset: 0,
            access_width: l1_dmac_access_width,
            remaining: u64::from(transfer.descriptor.burst_count) * (full + tail),
        }
    }

    pub const fn remaining_requests(&self) -> u64 {
        self.remaining
    }
}

impl Iterator for C220BtRequestPlan {
    type Item = C220BtReadUop;

    fn next(&mut self) -> Option<Self::Item> {
        let logical = match self.current {
            Some(logical) => logical,
            None => self.logical.next()?,
        };
        let input_bytes = (logical.input_bytes - self.offset).min(self.access_width.get());
        let completes_logical_uop = self.offset + input_bytes == logical.input_bytes;
        let uop = C220BtReadUop {
            logical,
            source_address: logical.source_address.wrapping_add(u64::from(self.offset)),
            input_bytes,
            input_offset: self.offset,
            completes_logical_uop,
            last_in_instruction: self.remaining == 1,
        };
        self.remaining -= 1;
        self.current = (!completes_logical_uop).then_some(logical);
        self.offset = if completes_logical_uop {
            0
        } else {
            self.offset + input_bytes
        };
        Some(uop)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match usize::try_from(self.remaining) {
            Ok(remaining) => (remaining, Some(remaining)),
            Err(_) => (usize::MAX, None),
        }
    }
}

impl FusedIterator for C220BtRequestPlan {}
