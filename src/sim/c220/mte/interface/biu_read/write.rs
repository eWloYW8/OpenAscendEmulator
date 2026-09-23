use std::num::NonZeroU32;

use super::C220BiuSubcore;
use super::returns::C220BiuReadOutput;
use crate::sim::c220::mte::uop::C220DmaUopRoute;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum C220BiuWriteDestination {
    L1,
    L0A,
    L0B,
    Ub0,
    Ub1,
}

impl C220BiuWriteDestination {
    pub const fn subcore(self) -> C220BiuSubcore {
        match self {
            Self::L1 | Self::L0A | Self::L0B => C220BiuSubcore::Cube,
            Self::Ub0 => C220BiuSubcore::Vector0,
            Self::Ub1 => C220BiuSubcore::Vector1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220BiuWriteBandwidths {
    pub l1: NonZeroU32,
    pub l0a: NonZeroU32,
    pub l0b: NonZeroU32,
    pub ub: NonZeroU32,
}

impl C220BiuWriteBandwidths {
    pub(super) fn widths(self) -> [NonZeroU32; 5] {
        [self.l1, self.l0a, self.l0b, self.ub, self.ub]
    }
}

/// One destination-interface packet. Interface occupancy may exceed the valid
/// byte count; padding must not be used as permission to overwrite memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220BiuWriteFragment {
    pub output: C220BiuReadOutput,
    pub destination_address: u64,
    pub bytes: u32,
    pub logical_bytes: u32,
    pub last_in_request: bool,
    pub last_in_instruction: bool,
}

impl C220BiuWriteFragment {
    pub fn output_fragment(self) -> crate::sim::c220::mte::interface::C220MteOutputFragment {
        crate::sim::c220::mte::interface::C220MteOutputFragment {
            instruction_id: self.output.request.input.generated.instruction_id,
            request_id: u64::from(self.output.request.tag.get()),
            destination_address: self.destination_address,
            bytes: self.bytes,
            last_in_uop: self.last_in_request,
            last_in_instruction: self.last_in_instruction,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220BiuWriteStall {
    NotReady,
    DestinationFull,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220BiuWriteSend {
    pub tick: u64,
    pub core: C220BiuSubcore,
    pub offered: Option<C220BiuWriteFragment>,
    pub stall: Option<C220BiuWriteStall>,
    /// An input may be absorbed without producing a complete output packet.
    pub consumed: Option<C220BiuReadOutput>,
}

impl C220BiuWriteSend {
    pub fn sent(&self) -> Option<C220BiuWriteFragment> {
        self.offered.filter(|_| self.stall.is_none())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum C220BiuWriteError {
    #[error("BIU write destination does not belong to the request subcore")]
    WrongSubcore,
    #[error("BIU collapsed destination burst must contain at least one byte")]
    EmptyBurst,
    #[error("BIU write alignment still holds instruction {pending}, cannot accept {incoming}")]
    InterleavedInstruction { pending: u64, incoming: u64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct C220BiuWriteProgress {
    pub instruction_id: Option<u64>,
    pub buffered_bytes: u32,
    pub destination_offset: u32,
    pub completed_bursts: u32,
}

/// A lazy packet stream keeps even large descriptors bounded in host memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct WritePlan {
    output: C220BiuReadOutput,
    base: u64,
    offset: u32,
    stride: u32,
    chunk: NonZeroU32,
    remaining: u32,
    full_packets: u32,
    round_full_packets: bool,
    round_tail: bool,
    final_marker: bool,
}

impl WritePlan {
    pub(super) fn front(&self) -> Option<C220BiuWriteFragment> {
        if self.remaining == 0 {
            return None;
        }
        let logical_bytes = self.remaining.min(self.chunk.get());
        let bytes = if self.round_full_packets || (self.full_packets == 0 && self.round_tail) {
            (logical_bytes.wrapping_sub(1) & !31).wrapping_add(32)
        } else {
            logical_bytes
        };
        let last_in_request = self.remaining == logical_bytes;
        Some(C220BiuWriteFragment {
            output: self.output,
            destination_address: self.base.wrapping_add(u64::from(self.offset)),
            bytes,
            logical_bytes,
            last_in_request,
            last_in_instruction: last_in_request && self.final_marker,
        })
    }

    pub(super) fn advance(&mut self) {
        let fragment = self.front().expect("nonempty write plan");
        self.remaining -= fragment.logical_bytes;
        self.full_packets = self.full_packets.saturating_sub(1);
        self.offset = self.offset.wrapping_add(self.stride);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct WriteAligner {
    bandwidth: NonZeroU32,
    pub(super) progress: C220BiuWriteProgress,
}

impl WriteAligner {
    pub(super) fn new(bandwidth: NonZeroU32) -> Self {
        Self {
            bandwidth,
            progress: C220BiuWriteProgress::default(),
        }
    }

    pub(super) fn plan(
        &mut self,
        output: C220BiuReadOutput,
    ) -> Result<WritePlan, C220BiuWriteError> {
        let generated = output.request.input.generated;
        let request = generated.request;
        if output.request.input.destination.subcore() != output.request.input.subcore {
            return Err(C220BiuWriteError::WrongSubcore);
        }
        let collapsed = request.route == C220DmaUopRoute::DestinationGapCollapse;
        let chunk = if request.route.padded_unit_bytes().is_some() {
            NonZeroU32::new(32).unwrap()
        } else if collapsed {
            NonZeroU32::new(generated.destination.burst_bytes)
                .ok_or(C220BiuWriteError::EmptyBurst)?
        } else {
            self.bandwidth
        };
        if generated.out_of_order {
            return Ok(WritePlan {
                output,
                base: request.destination_address,
                offset: 0,
                stride: self.bandwidth.get(),
                chunk: self.bandwidth,
                remaining: request.bytes,
                full_packets: 0,
                round_full_packets: true,
                round_tail: true,
                final_marker: output.last_in_instruction,
            });
        }
        if let Some(pending) = self.progress.instruction_id
            && pending != generated.instruction_id
        {
            return Err(C220BiuWriteError::InterleavedInstruction {
                pending,
                incoming: generated.instruction_id,
            });
        }
        let state = &mut self.progress;
        state.instruction_id = Some(generated.instruction_id);
        if let Some(unit_bytes) = request.route.truncated_unit_bytes() {
            let available = (request.bytes / 32 * unit_bytes).wrapping_add(state.buffered_bytes);
            let full_packets = available / self.bandwidth.get();
            let remainder = available % self.bandwidth.get();
            let plan = WritePlan {
                output,
                base: generated.destination.base,
                offset: state.destination_offset,
                stride: self.bandwidth.get(),
                chunk: self.bandwidth,
                remaining: if output.last_in_instruction {
                    available
                } else {
                    available - remainder
                },
                full_packets,
                round_full_packets: false,
                round_tail: false,
                final_marker: output.last_in_instruction,
            };
            state.buffered_bytes = remainder;
            state.destination_offset = state
                .destination_offset
                .wrapping_add(full_packets.wrapping_mul(self.bandwidth.get()));
            if output.last_in_instruction {
                state.instruction_id = None;
                if remainder != 0 {
                    state.buffered_bytes = 0;
                    state.destination_offset = 0;
                }
            }
            return Ok(plan);
        }
        let available = request.bytes.wrapping_add(state.buffered_bytes);
        let input_chunk = request.route.padded_unit_bytes().unwrap_or(chunk.get());
        let full_packets = available / input_chunk;
        let remainder = available % input_chunk;
        let stride = if collapsed {
            generated.destination.burst_stride as u32
        } else {
            chunk.get()
        };
        let plan = WritePlan {
            output,
            base: generated.destination.base,
            offset: state.destination_offset,
            stride,
            chunk,
            remaining: full_packets
                .wrapping_mul(chunk.get())
                .wrapping_add(if request.last_in_burst { remainder } else { 0 }),
            full_packets,
            round_full_packets: false,
            round_tail: true,
            final_marker: request.last_in_burst && output.last_in_instruction,
        };
        state.buffered_bytes = remainder;
        state.destination_offset = state
            .destination_offset
            .wrapping_add(full_packets.wrapping_mul(stride));
        if request.last_in_burst {
            state.buffered_bytes = 0;
            state.completed_bursts = state.completed_bursts.wrapping_add(1);
            state.destination_offset = state
                .completed_bursts
                .wrapping_mul(generated.destination.burst_stride as u32);
            if output.last_in_instruction {
                *state = C220BiuWriteProgress::default();
            }
        }
        Ok(plan)
    }
}
