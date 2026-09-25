use std::collections::{BTreeMap, VecDeque};
use std::num::{NonZeroU32, NonZeroU64};

use super::biu_read::{C220BiuReadCommandTransfer, C220BiuReadReturn};
use super::biu_write::C220BiuWriteReturnKind;
use crate::sim::c220::mte::interface::biu_read::C220BiuReadRequest;
use crate::sim::c220::mte::interface::biu_read::returns::C220BiuReadBeat;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum C220MemoryWriteId {
    Mte(NonZeroU32),
    Cache { port: u32, transaction: u64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MemoryWriteCommand {
    pub ready_tick: u64,
    pub tag: C220MemoryWriteId,
    pub address: u64,
    pub bytes: u32,
}

/// Timing-only data/response token. Memory bytes are applied by the owning
/// execution unit at completion, independently of transport payload lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MemoryWriteTransfer {
    pub ready_tick: u64,
    pub tag: C220MemoryWriteId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MemoryLatency {
    pub minimum: u32,
    pub spread: u32,
}

impl C220MemoryLatency {
    pub const fn fixed(ticks: u32) -> Self {
        Self {
            minimum: ticks,
            spread: 0,
        }
    }

    /// The deterministic selection uses unsigned 32-bit tick arithmetic.
    pub const fn deterministic_ticks(self) -> u32 {
        self.minimum.wrapping_add(self.spread >> 1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MemoryRegionTiming {
    pub read: C220MemoryLatency,
    pub dbid: C220MemoryLatency,
    pub completion: C220MemoryLatency,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MemoryCredits {
    pub limit: u32,
    pub refill: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MemoryReadAdmission {
    pub tick: u64,
    pub request: C220BiuReadRequest,
}

/// Deterministic memory service. Latencies use profile range midpoints; this
/// service does not substitute for a controller-driven memory model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220TimedMemoryConfig {
    pub input_capacity: NonZeroU32,
    pub pending_limit: NonZeroU32,
    pub credit_period: NonZeroU64,
    pub ddr_credits: C220MemoryCredits,
    pub l2_read_credits: C220MemoryCredits,
    pub l2_write_credits: C220MemoryCredits,
    pub ddr: C220MemoryRegionTiming,
    pub l2: C220MemoryRegionTiming,
    pub l2_start: u32,
    pub l2_bytes: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum C220TimedMemoryError {
    #[error("memory time overflowed")]
    TimeOverflow,
    #[error("memory credit configuration exceeds the positive signed range")]
    InvalidCredits,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Transaction {
    region: usize,
    bytes: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220TimedMemory {
    config: C220TimedMemoryConfig,
    credits: [u32; 3],
    credit_epoch: u64,
    pending: u32,
    commands: VecDeque<C220MemoryWriteCommand>,
    data: VecDeque<C220MemoryWriteTransfer>,
    reads: VecDeque<C220BiuReadCommandTransfer>,
    read_admissions: Vec<C220MemoryReadAdmission>,
    scheduled_reads: BTreeMap<u64, VecDeque<C220BiuReadBeat>>,
    ready_reads: VecDeque<C220BiuReadReturn>,
    scheduled: [BTreeMap<u64, VecDeque<C220MemoryWriteId>>; 2],
    ready: [VecDeque<C220MemoryWriteTransfer>; 2],
    transactions: BTreeMap<C220MemoryWriteId, Transaction>,
}

impl C220TimedMemory {
    pub(crate) fn new(
        config: C220TimedMemoryConfig,
        tick: u64,
    ) -> Result<Self, C220TimedMemoryError> {
        let pools = [
            config.ddr_credits,
            config.l2_write_credits,
            config.l2_read_credits,
        ];
        for pool in pools {
            if u64::from(pool.limit) + u64::from(pool.refill) > i32::MAX as u64 {
                return Err(C220TimedMemoryError::InvalidCredits);
            }
        }
        Ok(Self {
            config,
            credits: pools.map(|pool| pool.limit),
            credit_epoch: tick / config.credit_period.get(),
            pending: 0,
            commands: VecDeque::new(),
            data: VecDeque::new(),
            reads: VecDeque::new(),
            read_admissions: Vec::new(),
            scheduled_reads: BTreeMap::new(),
            ready_reads: VecDeque::new(),
            scheduled: Default::default(),
            ready: Default::default(),
            transactions: BTreeMap::new(),
        })
    }

    pub fn config(&self) -> C220TimedMemoryConfig {
        self.config
    }
    /// Shared DDR, L2 write, and L2 read byte credits.
    pub fn credits(&self) -> [u32; 3] {
        self.credits
    }
    pub fn pending_completions(&self) -> u32 {
        self.pending
    }
    /// Write commands, write data, and read requests.
    pub fn input_occupancy(&self) -> [usize; 3] {
        [self.commands.len(), self.data.len(), self.reads.len()]
    }
    pub fn ready_returns(
        &self,
        kind: C220BiuWriteReturnKind,
    ) -> &VecDeque<C220MemoryWriteTransfer> {
        &self.ready[index(kind)]
    }
    pub fn scheduled_returns(
        &self,
        kind: C220BiuWriteReturnKind,
    ) -> &BTreeMap<u64, VecDeque<C220MemoryWriteId>> {
        &self.scheduled[index(kind)]
    }
    pub fn is_idle(&self) -> bool {
        self.transactions.is_empty()
            && self.commands.is_empty()
            && self.reads.is_empty()
            && self.scheduled_reads.is_empty()
            && self.ready_reads.is_empty()
    }

    pub fn ready_reads(&self) -> &VecDeque<C220BiuReadReturn> {
        &self.ready_reads
    }

    pub fn scheduled_reads(&self) -> &BTreeMap<u64, VecDeque<C220BiuReadBeat>> {
        &self.scheduled_reads
    }

    /// Timing requests admitted during the most recent memory-service tick.
    /// These events do not commit instruction-visible memory effects.
    pub fn read_admissions(&self) -> &[C220MemoryReadAdmission] {
        &self.read_admissions
    }

    pub(crate) fn can_push_read(&self) -> bool {
        self.reads.len() < self.config.input_capacity.get() as usize
    }

    pub(crate) fn push_read(
        &mut self,
        tick: u64,
        request: C220BiuReadRequest,
    ) -> Result<(), C220TimedMemoryError> {
        let ready_tick = add(tick, 1)?;
        self.reads.push_back(C220BiuReadCommandTransfer {
            ready_tick,
            request,
        });
        Ok(())
    }

    pub(crate) fn read_front(&self, tick: u64) -> Option<C220BiuReadBeat> {
        self.ready_reads
            .front()
            .filter(|head| head.ready_tick <= tick)
            .map(|head| head.beat)
    }

    pub(crate) fn pop_read(&mut self) {
        self.ready_reads
            .pop_front()
            .expect("accepted read response");
    }

    pub(crate) fn can_push(&self, kind: C220BiuWriteReturnKind) -> bool {
        self.input_occupancy()[index(kind)] < self.config.input_capacity.get() as usize
    }

    pub(crate) fn push_command(
        &mut self,
        tick: u64,
        mut command: C220MemoryWriteCommand,
    ) -> Result<(), C220TimedMemoryError> {
        command.ready_tick = add(tick, 1)?;
        self.commands.push_back(command);
        Ok(())
    }

    pub(crate) fn push_data(
        &mut self,
        tick: u64,
        mut data: C220MemoryWriteTransfer,
    ) -> Result<(), C220TimedMemoryError> {
        data.ready_tick = add(tick, 1)?;
        self.data.push_back(data);
        Ok(())
    }

    pub(crate) fn advance(&mut self, tick: u64) -> Result<(), C220TimedMemoryError> {
        self.read_admissions.clear();
        let epoch = tick / self.config.credit_period.get();
        let elapsed = epoch - self.credit_epoch;
        for (credit, timing) in self.credits.iter_mut().zip([
            self.config.ddr_credits,
            self.config.l2_write_credits,
            self.config.l2_read_credits,
        ]) {
            *credit = (u128::from(*credit) + u128::from(elapsed) * u128::from(timing.refill))
                .min(u128::from(timing.limit)) as u32;
        }
        self.credit_epoch = epoch;
        while let Some(command) = self
            .commands
            .front()
            .copied()
            .filter(|head| head.ready_tick <= tick)
        {
            let region = self.address_region(command.address);
            let ready = add(tick, self.region(region).dbid.deterministic_ticks().into())?;
            self.transactions.insert(
                command.tag,
                Transaction {
                    region,
                    bytes: command.bytes,
                },
            );
            self.scheduled[0]
                .entry(ready)
                .or_default()
                .push_back(command.tag);
            self.commands.pop_front();
        }
        // Responses release pending service slots at preparation, independently
        // of the bounded upstream response transport.
        self.prepare(tick)?;
        while self.pending < self.config.pending_limit.get() {
            let Some(command) = self
                .reads
                .front()
                .copied()
                .filter(|head| head.ready_tick <= tick)
            else {
                break;
            };
            let request = command.request.input.generated.request;
            let region = self.address_region(request.source_address);
            let pool = region * 2;
            if request.bytes >= self.credits[pool] {
                break;
            }
            let ready = add(tick, self.region(region).read.deterministic_ticks().into())?;
            let count = request.bytes.div_ceil(128);
            self.scheduled_reads
                .entry(ready)
                .or_default()
                .extend((0..count).map(|transaction_id| C220BiuReadBeat {
                    tag: command.request.tag,
                    transaction_id,
                }));
            self.pending += count;
            self.credits[pool] -= request.bytes;
            self.read_admissions.push(C220MemoryReadAdmission {
                tick,
                request: command.request,
            });
            self.reads.pop_front();
        }
        while self.pending < self.config.pending_limit.get() {
            let Some(data) = self.data.front().filter(|head| head.ready_tick <= tick) else {
                break;
            };
            let tag = data.tag;
            let transaction = self.transactions[&tag];
            if transaction.bytes >= self.credits[transaction.region] {
                break;
            }
            let ready = add(
                tick,
                self.region(transaction.region)
                    .completion
                    .deterministic_ticks()
                    .into(),
            )?;
            self.credits[transaction.region] -= transaction.bytes;
            self.pending += 1;
            self.scheduled[1].entry(ready).or_default().push_back(tag);
            self.data.pop_front();
        }
        self.prepare(tick)?;
        Ok(())
    }

    fn prepare(&mut self, tick: u64) -> Result<(), C220TimedMemoryError> {
        while self
            .scheduled_reads
            .first_key_value()
            .is_some_and(|(&due, _)| due <= tick)
        {
            let ready_tick = add(tick, 1)?;
            let (_, beats) = self.scheduled_reads.pop_first().expect("due read group");
            self.pending -= beats.len() as u32;
            self.ready_reads.extend(
                beats
                    .into_iter()
                    .map(|beat| C220BiuReadReturn { ready_tick, beat }),
            );
        }
        for channel in 0..2 {
            while self.scheduled[channel]
                .first_key_value()
                .is_some_and(|(&due, _)| due <= tick)
            {
                let ready_tick = add(tick, 1)?;
                let (_, tags) = self.scheduled[channel]
                    .pop_first()
                    .expect("due response group");
                if channel == 1 {
                    self.pending -= tags.len() as u32;
                }
                self.ready[channel].extend(
                    tags.into_iter()
                        .map(|tag| C220MemoryWriteTransfer { ready_tick, tag }),
                );
            }
        }
        Ok(())
    }

    pub(crate) fn front(
        &self,
        tick: u64,
        kind: C220BiuWriteReturnKind,
    ) -> Option<C220MemoryWriteId> {
        self.ready[index(kind)]
            .front()
            .filter(|head| head.ready_tick <= tick)
            .map(|head| head.tag)
    }

    pub(crate) fn pop(&mut self, kind: C220BiuWriteReturnKind) {
        let response = self.ready[index(kind)]
            .pop_front()
            .expect("accepted response");
        if kind == C220BiuWriteReturnKind::Completion {
            self.transactions.remove(&response.tag);
        }
    }

    fn region(&self, region: usize) -> C220MemoryRegionTiming {
        [self.config.ddr, self.config.l2][region]
    }

    fn address_region(&self, address: u64) -> usize {
        usize::from(
            address >= u64::from(self.config.l2_start)
                && address < u64::from(self.config.l2_start.wrapping_add(self.config.l2_bytes)),
        )
    }
}

fn index(kind: C220BiuWriteReturnKind) -> usize {
    match kind {
        C220BiuWriteReturnKind::Dbid => 0,
        C220BiuWriteReturnKind::Completion => 1,
    }
}

fn add(tick: u64, delay: u64) -> Result<u64, C220TimedMemoryError> {
    tick.checked_add(delay)
        .ok_or(C220TimedMemoryError::TimeOverflow)
}

#[cfg(test)]
mod tests;
