use std::collections::{BTreeMap, VecDeque};
use std::num::NonZeroU32;

use crate::sim::c220::mte::dma::C220DmaGenerated;

mod events;
pub mod returns;
#[cfg(test)]
mod tests;
pub mod write;
pub use events::{C220BiuReadCallback, C220BiuReadEvent, C220BiuReadEvents};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum C220BiuSubcore {
    Cube,
    Vector0,
    Vector1,
}

impl C220BiuSubcore {
    const ALL: [Self; 3] = [Self::Cube, Self::Vector0, Self::Vector1];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220BiuReadConfig {
    pub outstanding: NonZeroU32,
    pub weights: [u32; 3],
    pub group_vector_returns: bool,
    pub write_bandwidths: write::C220BiuWriteBandwidths,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220BiuReadInput {
    pub subcore: C220BiuSubcore,
    pub destination: write::C220BiuWriteDestination,
    pub prefetch: bool,
    pub generated: C220DmaGenerated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220BiuReadRequest {
    pub tag: NonZeroU32,
    /// Offset within the generator's uop, not the full instruction.
    pub byte_offset: u32,
    pub input: C220BiuReadInput,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220BiuReadStall {
    NotReady,
    NoTag,
    TransportFull,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220BiuReadSend {
    pub tick: u64,
    pub offered: Option<C220BiuReadRequest>,
    pub stall: Option<C220BiuReadStall>,
}

impl C220BiuReadSend {
    pub fn sent(&self) -> Option<C220BiuReadRequest> {
        self.offered.filter(|_| self.stall.is_none())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220BiuReadArbitration {
    pub tick: u64,
    pub eligible: [bool; 3],
    pub selected: Option<C220BiuSubcore>,
    pub pending_split_blocked: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum C220BiuReadError {
    #[error("invalid ND2NZ BIU request")]
    InvalidNd2NzRequest,
    #[error("BIU write destination does not belong to the request subcore")]
    WrongDestination,
    #[error("BIU time reversed from {previous} to {requested}")]
    TimeReversed { previous: u64, requested: u64 },
    #[error("BIU {phase} callback already ran at tick {tick}")]
    RepeatedCallback { phase: &'static str, tick: u64 },
    #[error("BIU time overflowed")]
    TimeOverflow,
    #[error("BIU input has no bytes")]
    EmptyInput,
    #[error("BIU tag {0} is not outstanding")]
    UnknownTag(NonZeroU32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct QueuedInput {
    ready_tick: u64,
    input: C220BiuReadInput,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SplitRequests {
    ready_tick: u64,
    input: C220BiuReadInput,
    offset: u32,
}

impl Iterator for SplitRequests {
    type Item = (u32, C220BiuReadInput);

    fn next(&mut self) -> Option<Self::Item> {
        let original = self.input.generated.request;
        let remaining = original.bytes - self.offset;
        if remaining == 0 {
            return None;
        }
        let offset = self.offset;
        let mut input = self.input;
        let generated = &mut input.generated;
        let request = &mut generated.request;
        request.source_address = original.source_address.wrapping_add(u64::from(offset));
        request.destination_address = original.destination_address.wrapping_add(u64::from(offset));
        request.bytes = generated
            .mode
            .split_bytes(request.source_address, remaining);
        self.offset += request.bytes;
        request.last_in_burst = original.last_in_burst && self.offset == original.bytes;
        generated.last_in_instruction &= request.last_in_burst;
        generated.ready_tick = self.ready_tick;
        Some((offset, input))
    }
}

/// Read request admission, splitting, weighted arbitration and tag ownership.
/// The downstream transport supplies credit and acknowledges egress separately;
/// sending or releasing a tag does not acknowledge destination completion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220BiuReadFrontend {
    config: C220BiuReadConfig,
    inputs: [VecDeque<QueuedInput>; 3],
    current: VecDeque<SplitRequests>,
    weights: [i32; 3],
    reserved_tag: Option<NonZeroU32>,
    free_tags: VecDeque<NonZeroU32>,
    next_fresh_tag: u64,
    outstanding: BTreeMap<NonZeroU32, C220BiuReadRequest>,
    prefetch_outstanding: [u32; 3],
    observed_tick: Option<u64>,
    arbitration_tick: Option<u64>,
    send_tick: Option<u64>,
}

impl C220BiuReadFrontend {
    pub fn new(config: C220BiuReadConfig) -> Self {
        Self {
            config,
            inputs: std::array::from_fn(|_| VecDeque::new()),
            current: VecDeque::new(),
            weights: config.weights.map(|weight| weight as i32),
            reserved_tag: None,
            free_tags: VecDeque::new(),
            next_fresh_tag: 1,
            outstanding: BTreeMap::new(),
            prefetch_outstanding: [0; 3],
            observed_tick: None,
            arbitration_tick: None,
            send_tick: None,
        }
    }

    pub fn is_idle(&self) -> bool {
        self.inputs.iter().all(VecDeque::is_empty)
            && self.current.is_empty()
            && self.outstanding.is_empty()
    }

    pub fn can_push(&self, subcore: C220BiuSubcore) -> bool {
        self.inputs[subcore as usize].len() < 4
    }

    pub fn input_occupancy(&self) -> [usize; 3] {
        std::array::from_fn(|index| self.inputs[index].len())
    }

    pub fn free_tag_count(&self) -> usize {
        (u64::from(self.config.outstanding.get()) + 1 - self.next_fresh_tag) as usize
            + self.free_tags.len()
    }

    pub fn reserved_tag(&self) -> Option<NonZeroU32> {
        self.reserved_tag
    }

    pub fn outstanding(&self) -> &BTreeMap<NonZeroU32, C220BiuReadRequest> {
        &self.outstanding
    }

    pub fn prefetch_outstanding(&self) -> [u32; 3] {
        self.prefetch_outstanding
    }

    pub fn contains_instruction(&self, instruction_id: u64) -> bool {
        self.inputs
            .iter()
            .flatten()
            .any(|entry| entry.input.generated.instruction_id == instruction_id)
            || self
                .current
                .iter()
                .any(|entry| entry.input.generated.instruction_id == instruction_id)
            || self
                .outstanding
                .values()
                .any(|request| request.input.generated.instruction_id == instruction_id)
    }

    pub fn push(&mut self, tick: u64, input: C220BiuReadInput) -> Result<bool, C220BiuReadError> {
        self.check_time(tick)?;
        if let write::C220BiuWriteDestination::Nd2Nz { row_slot } = input.destination
            && (row_slot.is_some_and(|row| row >= 8)
                || input.generated.out_of_order
                || input.prefetch
                || input.generated.mode.split_bytes(
                    input.generated.request.source_address,
                    input.generated.request.bytes,
                ) != input.generated.request.bytes)
        {
            return Err(C220BiuReadError::InvalidNd2NzRequest);
        }
        if input.destination.subcore() != input.subcore {
            return Err(C220BiuReadError::WrongDestination);
        }
        if input.generated.request.bytes == 0 {
            return Err(C220BiuReadError::EmptyInput);
        }
        if !self.can_push(input.subcore) {
            return Ok(false);
        }
        let ready_tick = tick.checked_add(3).ok_or(C220BiuReadError::TimeOverflow)?;
        self.inputs[input.subcore as usize].push_back(QueuedInput { ready_tick, input });
        self.observed_tick = Some(tick);
        Ok(true)
    }

    pub fn arbitrate(&mut self, tick: u64) -> Result<C220BiuReadArbitration, C220BiuReadError> {
        self.check_callback(tick, self.arbitration_tick, "arbitration")?;
        let pending_split_blocked =
            self.current.iter().flat_map(|split| *split).take(2).count() > 1;
        let eligible = C220BiuSubcore::ALL.map(|subcore| {
            self.inputs[subcore as usize].front().is_some_and(|entry| {
                entry.ready_tick <= tick
                    && (!entry.input.prefetch || self.prefetch_allowed(subcore))
            })
        });
        let mut selected = None;
        let mut highest = i32::MIN;
        if !pending_split_blocked {
            for (index, &requested) in eligible.iter().enumerate() {
                if requested && self.weights[index] > highest {
                    highest = self.weights[index];
                    selected = Some(C220BiuSubcore::ALL[index]);
                }
            }
        }
        if let Some(subcore) = selected {
            let ready_tick = tick.checked_add(1).ok_or(C220BiuReadError::TimeOverflow)?;
            let total = eligible
                .iter()
                .enumerate()
                .filter(|(_, requested)| **requested)
                .fold(0_i32, |sum, (index, _)| {
                    sum.wrapping_add(self.config.weights[index] as i32)
                });
            self.weights[subcore as usize] = self.weights[subcore as usize].wrapping_sub(total);
            for (index, requested) in eligible.iter().enumerate() {
                if *requested {
                    self.weights[index] =
                        self.weights[index].wrapping_add(self.config.weights[index] as i32);
                }
            }
            let input = self.inputs[subcore as usize]
                .pop_front()
                .expect("eligible input")
                .input;
            self.current.push_back(SplitRequests {
                ready_tick,
                input,
                offset: 0,
            });
        }
        self.arbitration_tick = Some(tick);
        self.observed_tick = Some(tick);
        Ok(C220BiuReadArbitration {
            tick,
            eligible,
            selected,
            pending_split_blocked,
        })
    }

    pub fn send(
        &mut self,
        tick: u64,
        transport_ready: bool,
    ) -> Result<C220BiuReadSend, C220BiuReadError> {
        self.check_callback(tick, self.send_tick, "send")?;
        let mut result = C220BiuReadSend {
            tick,
            offered: None,
            stall: None,
        };
        if let Some(head) = self.current.front().copied() {
            if tick < head.ready_tick {
                result.stall = Some(C220BiuReadStall::NotReady);
            } else {
                if self.reserved_tag.is_none() {
                    self.reserved_tag =
                        if self.next_fresh_tag <= u64::from(self.config.outstanding.get()) {
                            let tag = NonZeroU32::new(self.next_fresh_tag as u32);
                            self.next_fresh_tag += 1;
                            tag
                        } else {
                            self.free_tags.pop_front()
                        };
                }
                if let Some(tag) = self.reserved_tag {
                    let (byte_offset, input) = head.clone().next().expect("nonempty split");
                    let request = C220BiuReadRequest {
                        tag,
                        byte_offset,
                        input,
                    };
                    result.offered = Some(request);
                    if transport_ready {
                        self.current.front_mut().expect("current request").next();
                        if self
                            .current
                            .front()
                            .expect("current request")
                            .clone()
                            .next()
                            .is_none()
                        {
                            self.current.pop_front();
                        }
                        self.reserved_tag = None;
                        self.outstanding.insert(tag, request);
                        if input.prefetch {
                            self.prefetch_outstanding[input.subcore as usize] += 1;
                        }
                    } else {
                        result.stall = Some(C220BiuReadStall::TransportFull);
                    }
                } else {
                    result.stall = Some(C220BiuReadStall::NoTag);
                }
            }
        }
        self.send_tick = Some(tick);
        self.observed_tick = Some(tick);
        Ok(result)
    }

    /// Called by the return-path owner only after successful egress handoff.
    /// It is not a memory-response or instruction-completion notification.
    pub fn release_tag(
        &mut self,
        tick: u64,
        tag: NonZeroU32,
    ) -> Result<C220BiuReadRequest, C220BiuReadError> {
        self.check_time(tick)?;
        let request = self
            .outstanding
            .remove(&tag)
            .ok_or(C220BiuReadError::UnknownTag(tag))?;
        self.free_tags.push_back(tag);
        self.observed_tick = Some(tick);
        Ok(request)
    }

    fn prefetch_allowed(&self, subcore: C220BiuSubcore) -> bool {
        let shift = if subcore == C220BiuSubcore::Cube {
            1
        } else {
            2
        };
        self.prefetch_outstanding[subcore as usize] < (self.config.outstanding.get() >> shift)
    }

    fn input_ready_tick(&self) -> Option<u64> {
        self.inputs
            .iter()
            .filter_map(|queue| queue.front().map(|entry| entry.ready_tick))
            .min()
    }

    fn request_ready_tick(&self) -> Option<u64> {
        self.current.front().map(|head| head.ready_tick)
    }

    fn check_time(&self, tick: u64) -> Result<(), C220BiuReadError> {
        if let Some(previous) = self.observed_tick
            && tick < previous
        {
            return Err(C220BiuReadError::TimeReversed {
                previous,
                requested: tick,
            });
        }
        Ok(())
    }

    fn check_callback(
        &self,
        tick: u64,
        previous: Option<u64>,
        phase: &'static str,
    ) -> Result<(), C220BiuReadError> {
        self.check_time(tick)?;
        if previous == Some(tick) {
            return Err(C220BiuReadError::RepeatedCallback { phase, tick });
        }
        Ok(())
    }
}
