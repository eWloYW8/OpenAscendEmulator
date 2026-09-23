use std::collections::{BTreeMap, VecDeque};
use std::num::NonZeroU32;

use super::C220BiuWriteSourceRequest;
use crate::sim::c220::mte::dma::C220DmaGenerated;
use crate::sim::c220::mte::fixp::C220FixpStoreWrite;
use crate::sim::c220::mte::interface::biu_read::C220BiuSubcore;
use crate::sim::c220::mte::uop::{
    C220DmaDestinationLayout, C220DmaUopMode, C220DmaUopRequest, C220DmaUopRoute,
};

const CORES: [C220BiuSubcore; 3] = [
    C220BiuSubcore::Cube,
    C220BiuSubcore::Vector0,
    C220BiuSubcore::Vector1,
];

#[cfg(test)]
mod tests;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220BiuWriteConfig {
    pub outstanding: NonZeroU32,
    pub weights: [u32; 3],
    pub source_bandwidth: NonZeroU32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220BiuWriteInput {
    pub subcore: C220BiuSubcore,
    pub generated: C220DmaGenerated,
    pub gather_stride: Option<u32>,
    pub store_token: Option<NonZeroU32>,
}

impl C220BiuWriteInput {
    pub fn from_fixp(write: C220FixpStoreWrite, mode: C220DmaUopMode, tick: u64) -> Self {
        let fragment = write.fragment;
        Self {
            subcore: C220BiuSubcore::Cube,
            gather_stride: None,
            store_token: Some(write.token),
            generated: C220DmaGenerated {
                instruction_id: fragment.instruction_id,
                uop_index: fragment.request_id,
                ready_tick: tick,
                request: C220DmaUopRequest {
                    route: C220DmaUopRoute::Ordinary,
                    burst_index: 0,
                    source_address: 0,
                    destination_address: fragment.destination_address,
                    bytes: fragment.bytes,
                    last_in_burst: true,
                },
                destination: C220DmaDestinationLayout {
                    base: fragment.destination_address,
                    burst_bytes: fragment.bytes,
                    burst_stride: u64::from(fragment.bytes),
                },
                mode,
                out_of_order: false,
                last_in_instruction: fragment.last_in_instruction,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220BiuWriteCommand {
    pub tag: NonZeroU32,
    pub byte_offset: u32,
    pub input: C220BiuWriteInput,
}

impl C220BiuWriteCommand {
    pub fn source_request(self) -> C220BiuWriteSourceRequest {
        C220BiuWriteSourceRequest {
            tag: self.tag,
            instruction_id: self.input.generated.instruction_id,
            source_address: self.input.generated.request.source_address,
            bytes: self.input.generated.request.bytes,
            gather_stride: self.input.gather_stride,
            last_in_instruction: self.input.generated.last_in_instruction,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220BiuWriteCommandTransfer {
    pub ready_tick: u64,
    pub command: C220BiuWriteCommand,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220BiuWriteCommandStall {
    NotReady,
    NoTag,
    TransportFull,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220BiuWriteCommandCycle {
    pub tick: u64,
    pub eligible: [bool; 3],
    pub selected: Option<C220BiuSubcore>,
    pub split_blocked: bool,
    pub sent: Option<C220BiuWriteCommand>,
    pub stall: Option<C220BiuWriteCommandStall>,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum C220BiuWriteCommandError {
    #[error("BIU write time reversed from {previous} to {requested}")]
    TimeReversed { previous: u64, requested: u64 },
    #[error("BIU write command callback already ran at tick {0}")]
    RepeatedTick(u64),
    #[error("BIU write command time overflowed")]
    TimeOverflow,
    #[error("BIU write input is empty")]
    EmptyInput,
    #[error(
        "Cube writes require a store token and no UB gather stride; vector writes cannot carry store tokens"
    )]
    InvalidSource,
    #[error("BIU write tag {0} has no command awaiting DBID")]
    UnexpectedDbid(NonZeroU32),
    #[error("BIU write tag {0} has no issued command")]
    UnknownTag(NonZeroU32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Split {
    ready_tick: u64,
    input: C220BiuWriteInput,
    offset: u32,
}

impl Iterator for Split {
    type Item = (u32, C220BiuWriteInput);
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
            .split_bytes(request.destination_address, remaining);
        self.offset += request.bytes;
        request.last_in_burst = original.last_in_burst && self.offset == original.bytes;
        generated.last_in_instruction &= request.last_in_burst;
        generated.ready_tick = self.ready_tick;
        Some((offset, input))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Issued {
    command: C220BiuWriteCommand,
    delivered: bool,
    dbid_received: bool,
    source_started: bool,
}

/// Write-command admission, destination-aligned splitting and tag ownership.
/// Memory supplies DBID and final responses; neither latency is estimated here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220BiuWriteCommands {
    config: C220BiuWriteConfig,
    inputs: [VecDeque<(u64, C220BiuWriteInput)>; 3],
    splits: VecDeque<Split>,
    weights: [i32; 3],
    reserved: Option<NonZeroU32>,
    free: VecDeque<NonZeroU32>,
    next_fresh: u64,
    issued: BTreeMap<NonZeroU32, Issued>,
    retirement: BTreeMap<u64, (u32, bool)>,
    transport: VecDeque<C220BiuWriteCommandTransfer>,
    observed: Option<u64>,
    advanced: Option<u64>,
}

impl C220BiuWriteCommands {
    pub fn new(config: C220BiuWriteConfig) -> Self {
        Self {
            config,
            inputs: std::array::from_fn(|_| VecDeque::new()),
            splits: VecDeque::new(),
            weights: config.weights.map(|w| w as i32),
            reserved: None,
            free: VecDeque::new(),
            next_fresh: 1,
            issued: BTreeMap::new(),
            retirement: BTreeMap::new(),
            transport: VecDeque::new(),
            observed: None,
            advanced: None,
        }
    }
    pub fn is_idle(&self) -> bool {
        self.inputs.iter().all(VecDeque::is_empty)
            && self.splits.is_empty()
            && self.issued.is_empty()
            && self.retirement.is_empty()
    }
    pub fn can_push(&self, core: C220BiuSubcore) -> bool {
        self.inputs[core as usize].len() < 4
    }
    pub fn input_occupancy(&self) -> [usize; 3] {
        self.inputs.each_ref().map(VecDeque::len)
    }
    pub fn reserved_tag(&self) -> Option<NonZeroU32> {
        self.reserved
    }
    pub fn free_tag_count(&self) -> u64 {
        u64::from(self.config.outstanding.get()) + 1 - self.next_fresh + self.free.len() as u64
    }
    pub fn outstanding(&self) -> impl Iterator<Item = C220BiuWriteCommand> + '_ {
        self.issued.values().map(|issued| issued.command)
    }
    pub fn requests(&self) -> &VecDeque<C220BiuWriteCommandTransfer> {
        &self.transport
    }
    pub fn push(
        &mut self,
        tick: u64,
        input: C220BiuWriteInput,
    ) -> Result<bool, C220BiuWriteCommandError> {
        self.check_time(tick)?;
        if (input.subcore == C220BiuSubcore::Cube) != input.store_token.is_some()
            || (input.store_token.is_some() && input.gather_stride.is_some())
        {
            return Err(C220BiuWriteCommandError::InvalidSource);
        }
        if input.generated.request.bytes == 0 {
            return Err(C220BiuWriteCommandError::EmptyInput);
        }
        if !self.can_push(input.subcore) {
            return Ok(false);
        }
        let ready = tick
            .checked_add(3)
            .ok_or(C220BiuWriteCommandError::TimeOverflow)?;
        self.inputs[input.subcore as usize].push_back((ready, input));
        self.observed = Some(tick);
        Ok(true)
    }
    pub fn advance(
        &mut self,
        tick: u64,
    ) -> Result<C220BiuWriteCommandCycle, C220BiuWriteCommandError> {
        self.check_time(tick)?;
        if self.advanced == Some(tick) {
            return Err(C220BiuWriteCommandError::RepeatedTick(tick));
        }
        let next = tick
            .checked_add(1)
            .ok_or(C220BiuWriteCommandError::TimeOverflow)?;
        let eligible = self
            .inputs
            .each_ref()
            .map(|q| q.front().is_some_and(|(ready, _)| *ready <= tick));
        let split_blocked = self.splits.iter().flat_map(|s| *s).take(2).count() > 1;
        let mut result = C220BiuWriteCommandCycle {
            tick,
            eligible,
            selected: None,
            split_blocked,
            sent: None,
            stall: None,
        };
        if !split_blocked {
            let mut highest = i32::MIN;
            for index in 0..3 {
                if eligible[index] && self.weights[index] > highest {
                    highest = self.weights[index];
                    result.selected = Some(CORES[index]);
                }
            }
            if let Some(core) = result.selected {
                let total = (0..3).filter(|&i| eligible[i]).fold(0_i32, |sum, i| {
                    sum.wrapping_add(self.config.weights[i] as i32)
                });
                self.weights[core as usize] = self.weights[core as usize].wrapping_sub(total);
                for (i, requested) in eligible.iter().enumerate() {
                    if *requested {
                        self.weights[i] =
                            self.weights[i].wrapping_add(self.config.weights[i] as i32);
                    }
                }
                let (_, input) = self.inputs[core as usize]
                    .pop_front()
                    .expect("eligible input");
                self.splits.push_back(Split {
                    ready_tick: next,
                    input,
                    offset: 0,
                });
            }
        }
        if let Some(head) = self.splits.front().copied() {
            if head.ready_tick > tick {
                result.stall = Some(C220BiuWriteCommandStall::NotReady);
            } else {
                if self.reserved.is_none() {
                    self.reserved = if self.next_fresh <= u64::from(self.config.outstanding.get()) {
                        let tag = NonZeroU32::new(self.next_fresh as u32);
                        self.next_fresh += 1;
                        tag
                    } else {
                        self.free.pop_front()
                    };
                }
                if let Some(tag) = self.reserved {
                    if self.transport.len() == 2 {
                        result.stall = Some(C220BiuWriteCommandStall::TransportFull);
                    } else {
                        let (byte_offset, input) = self
                            .splits
                            .front_mut()
                            .expect("split head")
                            .next()
                            .expect("nonempty split");
                        let command = C220BiuWriteCommand {
                            tag,
                            byte_offset,
                            input,
                        };
                        if self
                            .splits
                            .front()
                            .copied()
                            .expect("split head")
                            .next()
                            .is_none()
                        {
                            self.splits.pop_front();
                        }
                        self.reserved = None;
                        let count = self
                            .retirement
                            .entry(input.generated.instruction_id)
                            .or_default();
                        count.0 = count.0.wrapping_add(1);
                        count.1 = input.generated.last_in_instruction;
                        self.issued.insert(
                            tag,
                            Issued {
                                command,
                                delivered: false,
                                dbid_received: false,
                                source_started: false,
                            },
                        );
                        self.transport.push_back(C220BiuWriteCommandTransfer {
                            ready_tick: next,
                            command,
                        });
                        result.sent = Some(command);
                    }
                } else {
                    result.stall = Some(C220BiuWriteCommandStall::NoTag);
                }
            }
        }
        self.advanced = Some(tick);
        self.observed = Some(tick);
        Ok(result)
    }
    pub fn take_request(
        &mut self,
        tick: u64,
    ) -> Result<Option<C220BiuWriteCommandTransfer>, C220BiuWriteCommandError> {
        self.check_time(tick)?;
        let request = self.transport.pop_front_if(|head| head.ready_tick <= tick);
        if let Some(request) = request {
            self.issued
                .get_mut(&request.command.tag)
                .expect("issued command")
                .delivered = true;
        }
        self.observed = Some(tick);
        Ok(request)
    }
    pub fn awaiting_dbid(
        &self,
        tag: NonZeroU32,
    ) -> Result<C220BiuWriteCommand, C220BiuWriteCommandError> {
        self.issued
            .get(&tag)
            .filter(|entry| entry.delivered && !entry.dbid_received)
            .map(|entry| entry.command)
            .ok_or(C220BiuWriteCommandError::UnexpectedDbid(tag))
    }
    pub(in crate::sim::c220::mte) fn mark_dbid(&mut self, tag: NonZeroU32) {
        self.issued
            .get_mut(&tag)
            .expect("validated DBID")
            .dbid_received = true;
    }

    pub(in crate::sim::c220::mte) fn begin_source(
        &mut self,
        tag: NonZeroU32,
    ) -> C220BiuWriteSourceRequest {
        let issued = self
            .issued
            .get_mut(&tag)
            .expect("registered source command");
        assert!(issued.dbid_received && !issued.source_started);
        issued.source_started = true;
        let generated = &mut issued.command.input.generated;
        let count = self
            .retirement
            .get_mut(&generated.instruction_id)
            .expect("issued retirement record");
        count.0 = count.0.saturating_sub(1);
        generated.last_in_instruction = count.1 && count.0 == 0;
        if generated.last_in_instruction {
            self.retirement.remove(&generated.instruction_id);
        }
        issued.command.source_request()
    }
    pub(in crate::sim::c220::mte) fn release_tag(
        &mut self,
        tag: NonZeroU32,
    ) -> Result<C220BiuWriteCommand, C220BiuWriteCommandError> {
        let issued = self
            .issued
            .remove(&tag)
            .ok_or(C220BiuWriteCommandError::UnknownTag(tag))?;
        self.free.push_back(tag);
        Ok(issued.command)
    }
    fn check_time(&self, tick: u64) -> Result<(), C220BiuWriteCommandError> {
        if let Some(previous) = self.observed
            && tick < previous
        {
            return Err(C220BiuWriteCommandError::TimeReversed {
                previous,
                requested: tick,
            });
        }
        Ok(())
    }
}
