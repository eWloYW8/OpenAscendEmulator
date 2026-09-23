use std::{collections::VecDeque, num::NonZeroU32};

use super::{C220FixpNz2ndStaging, C220FixpNz2ndStagingEntry, C220FixpNz2ndStagingError};
use crate::sim::c220::mte::interface::C220MteOutputFragment;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FixpNz2ndOutputPolicy {
    /// Preferred large, medium, and minimum transaction sizes.
    pub burst_sizes: [NonZeroU32; 3],
    pub burst_control: u64,
    pub row_stride_bytes: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_packets_preserve_holes_and_gather_respects_alignment_and_control() {
        let policy = C220FixpNz2ndOutputPolicy {
            burst_sizes: [512, 256, 32].map(|n| NonZeroU32::new(n).unwrap()),
            burst_control: 0,
            row_stride_bytes: 1024,
        };
        let mut output = C220FixpNz2ndOutput::default();
        output.bursts.push_back(C220FixpNz2ndBurst {
            instruction_id: 3,
            request_id: 7,
            address: 16,
            bytes: 80,
            closed: true,
            last_in_instruction: true,
            gather: false,
            row_bytes: 40,
            row_offset: 0,
            policy,
            ready_tick: 1,
        });
        assert!(output.take_write(0, true).unwrap().is_none());
        assert!(output.take_write(1, false).unwrap().is_none());
        for (index, (address, bytes)) in [(16, 16), (32, 24), (1040, 16), (1056, 24)]
            .into_iter()
            .enumerate()
        {
            let fragment = output.take_write(index as u64 + 2, true).unwrap().unwrap();
            assert_eq!(
                (fragment.destination_address, fragment.bytes),
                (address, bytes)
            );
            assert_eq!(fragment.last_in_instruction, index == 3);
        }
        assert!(output.bursts().is_empty());
        assert_eq!(policy.gathered_packet(0, 128, false), None);
        assert_eq!(policy.gathered_packet(0, 128, true), Some(32));
        assert_eq!(policy.gathered_packet(16, 8, true), Some(8));
        assert_eq!(policy.gathered_packet(0, 512, false), Some(512));
        for (control, expected) in [(0, 512), (3, 256), (5, 32), (7, 700)] {
            let restricted = C220FixpNz2ndOutputPolicy {
                burst_control: control,
                ..policy
            };
            assert_eq!(restricted.row_packet(0, 700), expected);
        }
    }
}

impl C220FixpNz2ndOutputPolicy {
    fn mode(self) -> u8 {
        if self.burst_control & 1 == 0 {
            0
        } else {
            ((self.burst_control >> 1) & 3) as u8
        }
    }

    fn row_packet(self, address: u64, remaining: u32) -> u32 {
        let [large, medium, small] = self.burst_sizes.map(NonZeroU32::get);
        let offset = (address % u64::from(small)) as u32;
        if offset != 0 {
            return remaining.min(small - offset);
        }
        for (size, max_mode) in [(large, 0), (medium, 1), (small, 2)] {
            if self.mode() <= max_mode
                && remaining >= size
                && address.is_multiple_of(u64::from(size))
            {
                return size;
            }
        }
        remaining
    }

    fn gathered_packet(self, address: u64, available: u32, closed: bool) -> Option<u32> {
        let [large, medium, small] = self.burst_sizes.map(NonZeroU32::get);
        let offset = (address % u64::from(small)) as u32;
        let limit = if offset != 0 {
            (small - offset).min(512)
        } else if closed {
            return Some(self.row_packet(address, available));
        } else {
            [(large, 0), (medium, 1), (small, 2)]
                .into_iter()
                .find(|&(size, max_mode)| {
                    size <= 512
                        && self.mode() <= max_mode
                        && address.is_multiple_of(u64::from(size))
                })
                .map_or(512, |(size, _)| size)
        };
        if closed {
            Some(available.min(limit))
        } else {
            (available >= limit).then_some(limit)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FixpNz2ndBurst {
    pub instruction_id: u64,
    pub request_id: u64,
    pub address: u64,
    pub bytes: u32,
    pub closed: bool,
    pub last_in_instruction: bool,
    pub gather: bool,
    pub row_bytes: u32,
    pub row_offset: u32,
    pub policy: C220FixpNz2ndOutputPolicy,
    pub ready_tick: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum C220FixpNz2ndOutputError {
    #[error(transparent)]
    Staging(#[from] C220FixpNz2ndStagingError),
    #[error("NZ2ND output time reversed from {previous} to {requested}")]
    TimeReversed { previous: u64, requested: u64 },
    #[error("NZ2ND output packet callback repeated at tick {0}")]
    RepeatedPacket(u64),
    #[error("NZ2ND output time overflow")]
    TimeOverflow,
}

/// External-memory packetization following NZ2ND alignment staging. Address
/// holes between rows are retained; downstream transaction credit and write
/// acknowledgments remain the responsibility of the external interface.
#[derive(Debug, Clone, Default)]
pub struct C220FixpNz2ndOutput {
    bursts: VecDeque<C220FixpNz2ndBurst>,
    observed_tick: Option<u64>,
    packet_tick: Option<u64>,
}

impl C220FixpNz2ndOutput {
    pub fn bursts(&self) -> &VecDeque<C220FixpNz2ndBurst> {
        &self.bursts
    }
    pub fn can_receive(&self) -> bool {
        self.bursts.len() <= 2
    }

    pub fn receive(
        &mut self,
        tick: u64,
        staging: &mut C220FixpNz2ndStaging,
        policy: C220FixpNz2ndOutputPolicy,
    ) -> Result<Option<C220FixpNz2ndStagingEntry>, C220FixpNz2ndOutputError> {
        self.observe(tick)?;
        let ready_tick = tick
            .checked_add(1)
            .ok_or(C220FixpNz2ndOutputError::TimeOverflow)?;
        let entry = staging.take_ready(tick, self.can_receive())?;
        if let Some(entry) = entry {
            let op = entry.operation;
            let d = op.descriptor;
            if d.last_in_uop {
                if let Some(tail) = self.bursts.back_mut().filter(|tail| !tail.closed) {
                    tail.bytes = tail.bytes.wrapping_add(d.bytes);
                    tail.closed = d.end_of_burst;
                    tail.last_in_instruction = op.last_in_instruction;
                } else {
                    self.bursts.push_back(C220FixpNz2ndBurst {
                        instruction_id: op.instruction_id,
                        request_id: u64::from(op.request_id),
                        address: d.destination_address,
                        bytes: d.bytes,
                        closed: d.end_of_burst,
                        last_in_instruction: op.last_in_instruction,
                        gather: d.gather,
                        row_bytes: d.burst_bytes,
                        row_offset: 0,
                        policy,
                        ready_tick,
                    });
                }
            }
        }
        Ok(entry)
    }

    pub fn take_write(
        &mut self,
        tick: u64,
        destination_ready: bool,
    ) -> Result<Option<C220MteOutputFragment>, C220FixpNz2ndOutputError> {
        self.observe(tick)?;
        if self.packet_tick == Some(tick) {
            return Err(C220FixpNz2ndOutputError::RepeatedPacket(tick));
        }
        self.packet_tick = Some(tick);
        let Some(head) = self.bursts.front_mut() else {
            return Ok(None);
        };
        if !destination_ready || head.ready_tick > tick {
            return Ok(None);
        }
        let address = if head.gather {
            head.address
        } else {
            head.address.wrapping_add(u64::from(head.row_offset))
        };
        let bytes = if head.gather {
            let Some(bytes) = head
                .policy
                .gathered_packet(address, head.bytes, head.closed)
            else {
                return Ok(None);
            };
            head.address = head.address.wrapping_add(u64::from(bytes));
            bytes
        } else {
            if head.row_offset >= head.row_bytes {
                return Ok(None);
            }
            let bytes = head
                .policy
                .row_packet(address, head.row_bytes - head.row_offset);
            head.row_offset += bytes;
            if head.row_offset == head.row_bytes {
                head.row_offset = 0;
                head.address = head
                    .address
                    .wrapping_add(u64::from(head.policy.row_stride_bytes));
            }
            bytes
        };
        head.bytes = head.bytes.wrapping_sub(bytes);
        let finished = head.closed && head.bytes == 0;
        let fragment = C220MteOutputFragment {
            instruction_id: head.instruction_id,
            request_id: head.request_id,
            destination_address: address,
            bytes,
            last_in_uop: true,
            last_in_instruction: finished && head.last_in_instruction,
        };
        if finished {
            self.bursts.pop_front();
        }
        Ok(Some(fragment))
    }

    fn observe(&mut self, tick: u64) -> Result<(), C220FixpNz2ndOutputError> {
        if let Some(previous) = self.observed_tick
            && tick < previous
        {
            return Err(C220FixpNz2ndOutputError::TimeReversed {
                previous,
                requested: tick,
            });
        }
        self.observed_tick = Some(tick);
        Ok(())
    }
}
