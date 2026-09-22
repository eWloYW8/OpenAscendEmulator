use std::iter::FusedIterator;
use std::num::NonZeroU32;

use crate::isa::c220::mte::set2d::{C220Set2dDestination, C220Set2dFill};
use crate::sim::c220::mte::interface::{
    C220L0WritePort, C220MteL1WritePort, C220MteOutputFragment,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Set2dBandwidths {
    pub l0a: NonZeroU32,
    pub l0b: NonZeroU32,
    pub l1: NonZeroU32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220Set2dOutputRoute {
    L0a(C220L0WritePort),
    L0b(C220L0WritePort),
    L1(C220MteL1WritePort),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Set2dUop {
    pub repeat_index: u16,
    pub offset_in_repeat: u32,
    pub destination_address: u64,
    pub bytes: u32,
    pub route: C220Set2dOutputRoute,
    pub last_in_repeat: bool,
    pub last_in_instruction: bool,
}

impl C220Set2dUop {
    pub const fn output_fragment(
        self,
        instruction_id: u64,
        request_id: u64,
    ) -> C220MteOutputFragment {
        C220MteOutputFragment {
            instruction_id,
            request_id,
            destination_address: self.destination_address,
            bytes: self.bytes,
            last_in_uop: true,
            last_in_instruction: self.last_in_instruction,
        }
    }
}

/// Each repeat is split independently by the configured destination bandwidth.
/// Generation, output credit and retirement are handled by the owning engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220Set2dUops {
    fill: C220Set2dFill,
    bandwidth: NonZeroU32,
    route: C220Set2dOutputRoute,
    repeat: u16,
    offset: u32,
}

impl C220Set2dUops {
    pub fn new(fill: C220Set2dFill, bandwidths: C220Set2dBandwidths) -> Self {
        let (bandwidth, route) = match fill.instruction.destination {
            C220Set2dDestination::L0a => (
                bandwidths.l0a,
                C220Set2dOutputRoute::L0a(C220L0WritePort::Port1),
            ),
            C220Set2dDestination::L0b => (
                bandwidths.l0b,
                C220Set2dOutputRoute::L0b(C220L0WritePort::Port1),
            ),
            C220Set2dDestination::L1 => (
                bandwidths.l1,
                C220Set2dOutputRoute::L1(C220MteL1WritePort::Port2),
            ),
        };
        Self {
            fill,
            bandwidth,
            route,
            repeat: 0,
            offset: 0,
        }
    }

    pub fn remaining(&self) -> u64 {
        let repeats = self.fill.descriptor.repeat_count - self.repeat;
        if repeats == 0 || self.fill.burst_bytes() == 0 {
            return 0;
        }
        u64::from(repeats - 1) * u64::from(self.fill.burst_bytes().div_ceil(self.bandwidth.get()))
            + u64::from((self.fill.burst_bytes() - self.offset).div_ceil(self.bandwidth.get()))
    }
}

impl Iterator for C220Set2dUops {
    type Item = C220Set2dUop;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining() == 0 {
            return None;
        }
        let bytes = (self.fill.burst_bytes() - self.offset).min(self.bandwidth.get());
        let last_in_repeat = self.offset + bytes == self.fill.burst_bytes();
        let uop = C220Set2dUop {
            repeat_index: self.repeat,
            offset_in_repeat: self.offset,
            destination_address: self
                .fill
                .destination_base
                .wrapping_add(u64::from(self.repeat).wrapping_mul(self.fill.destination_stride()))
                .wrapping_add(u64::from(self.offset)),
            bytes,
            route: self.route,
            last_in_repeat,
            last_in_instruction: last_in_repeat
                && self.repeat + 1 == self.fill.descriptor.repeat_count,
        };
        if last_in_repeat {
            self.repeat += 1;
            self.offset = 0;
        } else {
            self.offset += bytes;
        }
        Some(uop)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match usize::try_from(self.remaining()) {
            Ok(remaining) => (remaining, Some(remaining)),
            Err(_) => (usize::MAX, None),
        }
    }
}

impl FusedIterator for C220Set2dUops {}
