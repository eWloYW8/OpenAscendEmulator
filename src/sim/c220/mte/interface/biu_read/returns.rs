use std::collections::{BTreeMap, VecDeque};
use std::num::NonZeroU32;

use super::{C220BiuReadRequest, C220BiuSubcore};

mod events;
#[cfg(test)]
mod tests;
pub use events::{C220BiuReturnCallback, C220BiuReturnEvent, C220BiuReturnEvents};

type Tag = NonZeroU32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220BiuReadBeat {
    pub tag: NonZeroU32,
    pub transaction_id: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220BiuRobBeat {
    pub beat: C220BiuReadBeat,
    pub port: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220BiuReadOutput {
    pub ready_tick: u64,
    pub request: C220BiuReadRequest,
    /// Marks the final completing request, independently of issue order.
    pub last_in_instruction: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220BiuReadProgress {
    pub request: C220BiuReadRequest,
    pub expected_beats: u32,
    pub received_beats: u32,
    pub pending_rob_beats: usize,
    pub ingress_ready: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum C220BiuReturnError {
    #[error("BIU return time reversed from {previous} to {requested}")]
    TimeReversed { previous: u64, requested: u64 },
    #[error("BIU return {phase} callback already ran at tick {tick}")]
    RepeatedCallback { phase: &'static str, tick: u64 },
    #[error("BIU return time overflowed")]
    TimeOverflow,
    #[error("BIU tag {0} is already awaiting responses")]
    DuplicateTag(Tag),
    #[error("BIU tag {0} returned more data beats than requested")]
    ExcessResponse(Tag),
    #[error("BIU ordinary return path cannot accept a prefetch request")]
    PrefetchUnsupported,
    #[error("BIU ROB port capacity must be nonzero")]
    ZeroCapacity,
    #[error("BIU read request must contain at least one byte")]
    EmptyRequest,
    #[error("invalid BIU response port {0}")]
    InvalidPort(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TimedTag {
    tag: Tag,
    ready_tick: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Record {
    request: C220BiuReadRequest,
    expected: u32,
    received: u32,
    transactions: VecDeque<C220BiuRobBeat>,
    ingress_ready: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct InstructionRequests {
    remaining: u64,
    tail_sent: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RoundRobin<const N: usize> {
    next: usize,
}

impl<const N: usize> Default for RoundRobin<N> {
    fn default() -> Self {
        Self { next: 0 }
    }
}

impl<const N: usize> RoundRobin<N> {
    fn select(&mut self, eligible: [bool; N]) -> Option<usize> {
        let selected = (0..N)
            .map(|offset| (self.next + offset) % N)
            .find(|&i| eligible[i])?;
        self.next = (selected + 1) % N;
        Some(selected)
    }
}

/// Ordinary two-port BIU read returns. Receive, ingress, tag selection, ROB
/// reads and egress are independent callbacks. The write adapter is the output
/// boundary; destination service and HBM response latency belong to their owners.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220BiuReadReturns {
    capacity: u32,
    group_vectors: bool,
    occupancy: [u32; 2],
    records: BTreeMap<Tag, Record>,
    instructions: BTreeMap<u64, InstructionRequests>,
    ingress: [VecDeque<TimedTag>; 2],
    order: [[VecDeque<TimedTag>; 2]; 3],
    active: [Option<Tag>; 3],
    active_ready_tick: Option<u64>,
    egress: [VecDeque<C220BiuReadOutput>; 3],
    adapters: [VecDeque<C220BiuReadOutput>; 3],
    receive_arbiter: RoundRobin<2>,
    core_arbiter: RoundRobin<3>,
    order_arbiters: [RoundRobin<2>; 3],
    vector_arbiter: RoundRobin<2>,
    observed_tick: Option<u64>,
    receive_tick: Option<u64>,
    ingress_ticks: [Option<u64>; 2],
    select_tick: Option<u64>,
    read_tick: Option<u64>,
    egress_ticks: [Option<u64>; 3],
}

impl C220BiuReadReturns {
    pub fn new(outstanding: NonZeroU32, group_vectors: bool) -> Result<Self, C220BiuReturnError> {
        let capacity = (outstanding.get().wrapping_shl(9) / 2) >> 7;
        if capacity == 0 {
            return Err(C220BiuReturnError::ZeroCapacity);
        }
        Ok(Self {
            capacity,
            group_vectors,
            occupancy: [0; 2],
            records: BTreeMap::new(),
            instructions: BTreeMap::new(),
            ingress: std::array::from_fn(|_| VecDeque::new()),
            order: std::array::from_fn(|_| std::array::from_fn(|_| VecDeque::new())),
            active: [None; 3],
            active_ready_tick: None,
            egress: std::array::from_fn(|_| VecDeque::new()),
            adapters: std::array::from_fn(|_| VecDeque::new()),
            receive_arbiter: RoundRobin::default(),
            core_arbiter: RoundRobin::default(),
            order_arbiters: [RoundRobin::default(); 3],
            vector_arbiter: RoundRobin::default(),
            observed_tick: None,
            receive_tick: None,
            ingress_ticks: [None; 2],
            select_tick: None,
            read_tick: None,
            egress_ticks: [None; 3],
        })
    }

    pub fn is_idle(&self) -> bool {
        self.records.is_empty() && self.adapters.iter().all(VecDeque::is_empty)
    }

    pub fn contains_instruction(&self, id: u64) -> bool {
        self.records
            .values()
            .any(|record| record.request.input.generated.instruction_id == id)
            || self
                .adapters
                .iter()
                .flatten()
                .any(|output| output.request.input.generated.instruction_id == id)
    }

    pub fn occupancy(&self) -> [u32; 2] {
        self.occupancy
    }
    pub fn capacity_per_port(&self) -> u32 {
        self.capacity
    }
    pub fn progress(&self, tag: Tag) -> Option<C220BiuReadProgress> {
        self.records.get(&tag).map(|record| C220BiuReadProgress {
            request: record.request,
            expected_beats: record.expected,
            received_beats: record.received,
            pending_rob_beats: record.transactions.len(),
            ingress_ready: record.ingress_ready,
        })
    }
    pub fn transactions(&self, tag: Tag) -> Option<&VecDeque<C220BiuRobBeat>> {
        self.records.get(&tag).map(|record| &record.transactions)
    }
    pub fn ingress_occupancy(&self) -> [usize; 2] {
        std::array::from_fn(|port| self.ingress[port].len())
    }
    pub fn queued_tags(&self, core: C220BiuSubcore) -> [usize; 2] {
        std::array::from_fn(|order| self.order[core as usize][order].len())
    }
    pub fn pending_egress(&self, core: C220BiuSubcore) -> &VecDeque<C220BiuReadOutput> {
        &self.egress[core as usize]
    }
    pub fn active_tags(&self) -> [Option<Tag>; 3] {
        self.active
    }
    pub fn has_active_tags(&self) -> bool {
        self.active.iter().any(Option::is_some)
    }
    pub fn adapter(&self, core: C220BiuSubcore) -> &VecDeque<C220BiuReadOutput> {
        &self.adapters[core as usize]
    }

    pub fn track(
        &mut self,
        tick: u64,
        request: C220BiuReadRequest,
    ) -> Result<(), C220BiuReturnError> {
        self.check_time(tick)?;
        if request.input.generated.request.bytes == 0 {
            return Err(C220BiuReturnError::EmptyRequest);
        }
        if request.input.prefetch {
            return Err(C220BiuReturnError::PrefetchUnsupported);
        }
        if self.records.contains_key(&request.tag) {
            return Err(C220BiuReturnError::DuplicateTag(request.tag));
        }
        let ready_tick = tick
            .checked_add(1)
            .ok_or(C220BiuReturnError::TimeOverflow)?;
        let generated = request.input.generated;
        self.records.insert(
            request.tag,
            Record {
                request,
                expected: generated.request.bytes.div_ceil(128),
                received: 0,
                transactions: VecDeque::new(),
                ingress_ready: false,
            },
        );
        if !generated.out_of_order {
            self.order[request.input.subcore as usize][0].push_back(TimedTag {
                tag: request.tag,
                ready_tick,
            });
        }
        let instruction = self
            .instructions
            .entry(generated.instruction_id)
            .or_default();
        instruction.remaining += 1;
        instruction.tail_sent = generated.last_in_instruction;
        self.observed_tick = Some(tick);
        Ok(())
    }

    /// Each element is the current head of one external response channel.
    /// False entries must remain at their channel head for a later callback.
    pub fn receive(
        &mut self,
        tick: u64,
        heads: [Option<C220BiuReadBeat>; 2],
    ) -> Result<[bool; 2], C220BiuReturnError> {
        self.check_callback(tick, self.receive_tick, "receive")?;
        let ready_tick = tick
            .checked_add(2)
            .ok_or(C220BiuReturnError::TimeOverflow)?;
        let free = self.occupancy.map(|used| used < self.capacity);
        let mut mapping = [None; 2];
        let mut arbiter = self.receive_arbiter;
        match heads {
            [Some(a), Some(b)] if free == [true, true] => {
                mapping = if (a.transaction_id ^ b.transaction_id) & 1 != 0 {
                    [
                        Some((a.transaction_id & 1) as usize),
                        Some((b.transaction_id & 1) as usize),
                    ]
                } else {
                    [Some(0), Some(1)]
                };
            }
            [Some(_), Some(_)] => {
                if let Some(port) = free.iter().position(|&available| available) {
                    let channel = arbiter.select([true; 2]).expect("two channels");
                    mapping[channel] = Some(port);
                }
            }
            _ => {
                for (channel, head) in heads.iter().enumerate() {
                    if let Some(beat) = head {
                        let preferred = (beat.transaction_id & 1) as usize;
                        mapping[channel] = [preferred, 1 - preferred]
                            .into_iter()
                            .find(|&port| free[port]);
                    }
                }
            }
        }
        let mut increments = BTreeMap::<Tag, u32>::new();
        for channel in 0..2 {
            if mapping[channel].is_some()
                && let Some(beat) = heads[channel]
                && let Some(record) = self.records.get(&beat.tag)
            {
                let count = increments.entry(beat.tag).or_default();
                *count += 1;
                if record.received.saturating_add(*count) > record.expected {
                    return Err(C220BiuReturnError::ExcessResponse(beat.tag));
                }
            }
        }
        self.receive_arbiter = arbiter;
        let mut accepted = [false; 2];
        for channel in 0..2 {
            if let (Some(port), Some(beat)) = (mapping[channel], heads[channel])
                && let Some(record) = self.records.get_mut(&beat.tag)
            {
                record.received += 1;
                record.transactions.push_back(C220BiuRobBeat { beat, port });
                self.occupancy[port] += 1;
                accepted[channel] = true;
                if record.received == record.expected {
                    record
                        .transactions
                        .make_contiguous()
                        .sort_unstable_by_key(|transaction| transaction.beat.transaction_id);
                    self.ingress[port].push_back(TimedTag {
                        tag: beat.tag,
                        ready_tick,
                    });
                }
            }
        }
        self.receive_tick = Some(tick);
        self.observed_tick = Some(tick);
        Ok(accepted)
    }

    pub fn ingress(&mut self, tick: u64, port: usize) -> Result<Option<Tag>, C220BiuReturnError> {
        if port >= 2 {
            return Err(C220BiuReturnError::InvalidPort(port));
        }
        self.check_callback(tick, self.ingress_ticks[port], "ingress")?;
        let mut accepted = None;
        if let Some(head) = self.ingress[port].front().copied()
            && head.ready_tick <= tick
        {
            let ready_tick = tick
                .checked_add(1)
                .ok_or(C220BiuReturnError::TimeOverflow)?;
            let record = self
                .records
                .get_mut(&head.tag)
                .expect("registered response");
            record.ingress_ready = true;
            if record.request.input.generated.out_of_order {
                self.order[record.request.input.subcore as usize][1].push_back(TimedTag {
                    tag: head.tag,
                    ready_tick,
                });
            }
            self.ingress[port].pop_front();
            accepted = Some(head.tag);
        }
        self.ingress_ticks[port] = Some(tick);
        self.observed_tick = Some(tick);
        Ok(accepted)
    }

    pub fn select(&mut self, tick: u64) -> Result<[Option<Tag>; 3], C220BiuReturnError> {
        self.check_callback(tick, self.select_tick, "selection")?;
        let mut selected = [None; 3];
        if !self.has_active_tags() {
            let ready_tick = tick
                .checked_add(1)
                .ok_or(C220BiuReturnError::TimeOverflow)?;
            let eligible: [[bool; 2]; 3] = std::array::from_fn(|core| {
                std::array::from_fn(|order| {
                    self.order[core][order].front().is_some_and(|head| {
                        head.ready_tick <= tick && self.records[&head.tag].ingress_ready
                    })
                })
            });
            let cores = eligible.map(|order| order.into_iter().any(|ready| ready));
            let requests = if self.group_vectors {
                [cores[0], cores[1] || cores[2], false]
            } else {
                cores
            };
            if let Some(chosen) = self.core_arbiter.select(requests) {
                for core in 0..3 {
                    let selected_core = (core == chosen && !self.group_vectors)
                        || (self.group_vectors
                            && ((chosen == 0 && core == 0) || (chosen == 1 && core > 0)));
                    if selected_core
                        && let Some(order) = self.order_arbiters[core].select(eligible[core])
                    {
                        selected[core] = self.order[core][order].pop_front().map(|head| head.tag);
                    }
                }
                self.active = selected;
                self.active_ready_tick = Some(ready_tick);
            }
        }
        self.select_tick = Some(tick);
        self.observed_tick = Some(tick);
        Ok(selected)
    }

    pub fn read(&mut self, tick: u64) -> Result<Vec<C220BiuRobBeat>, C220BiuReturnError> {
        self.check_callback(tick, self.read_tick, "ROB read")?;
        let ready_tick = tick
            .checked_add(1)
            .ok_or(C220BiuReturnError::TimeOverflow)?;
        let mut drained = Vec::with_capacity(2);
        if self.active_ready_tick.is_some_and(|ready| ready > tick) {
            self.read_tick = Some(tick);
            self.observed_tick = Some(tick);
            return Ok(drained);
        }
        if let Some(tag) = self.active[0] {
            let first = self.pop_beat(tag);
            let port = first.port;
            drained.push(first);
            if self.records[&tag]
                .transactions
                .front()
                .is_some_and(|beat| beat.port != port)
            {
                drained.push(self.pop_beat(tag));
            }
        }
        let mut vectors = [self.active[1].is_some(), self.active[2].is_some()];
        if self.group_vectors && vectors == [true; 2] {
            let ports = [1, 2].map(|core| {
                self.records[&self.active[core].expect("active vector")]
                    .transactions
                    .front()
                    .expect("pending beat")
                    .port
            });
            if ports[0] == ports[1] {
                let chosen = self
                    .vector_arbiter
                    .select([true; 2])
                    .expect("two active vectors");
                vectors[1 - chosen] = false;
            }
        }
        for (index, enabled) in vectors.into_iter().enumerate() {
            if enabled {
                drained.push(self.pop_beat(self.active[index + 1].expect("active vector")));
            }
        }
        for core in 0..3 {
            if let Some(tag) = self.active[core]
                && self.records[&tag].transactions.is_empty()
            {
                let request = self.records[&tag].request;
                let id = request.input.generated.instruction_id;
                let instruction = self.instructions.get_mut(&id).expect("pending instruction");
                instruction.remaining -= 1;
                let last_in_instruction = instruction.remaining == 0 && instruction.tail_sent;
                if last_in_instruction {
                    self.instructions.remove(&id);
                }
                self.egress[core].push_back(C220BiuReadOutput {
                    ready_tick,
                    request,
                    last_in_instruction,
                });
                self.active[core] = None;
            }
        }
        self.read_tick = Some(tick);
        self.active_ready_tick = self.has_active_tags().then_some(ready_tick);
        self.observed_tick = Some(tick);
        Ok(drained)
    }

    pub fn egress(
        &mut self,
        tick: u64,
        core: C220BiuSubcore,
    ) -> Result<Option<C220BiuReadOutput>, C220BiuReturnError> {
        let index = core as usize;
        self.check_callback(tick, self.egress_ticks[index], "egress")?;
        let mut sent = None;
        if self.adapters[index].len() < 2
            && let Some(mut output) = self.egress[index].front().copied()
            && output.ready_tick <= tick
        {
            output.ready_tick = tick
                .checked_add(1)
                .ok_or(C220BiuReturnError::TimeOverflow)?;
            self.egress[index].pop_front();
            self.records.remove(&output.request.tag);
            self.adapters[index].push_back(output);
            sent = Some(output);
        }
        self.egress_ticks[index] = Some(tick);
        self.observed_tick = Some(tick);
        Ok(sent)
    }

    pub fn take_output(
        &mut self,
        tick: u64,
        core: C220BiuSubcore,
    ) -> Result<Option<C220BiuReadOutput>, C220BiuReturnError> {
        self.check_time(tick)?;
        let queue = &mut self.adapters[core as usize];
        let output = if queue.front().is_some_and(|head| head.ready_tick <= tick) {
            queue.pop_front()
        } else {
            None
        };
        self.observed_tick = Some(tick);
        Ok(output)
    }

    fn pop_beat(&mut self, tag: Tag) -> C220BiuRobBeat {
        let beat = self
            .records
            .get_mut(&tag)
            .expect("active tag")
            .transactions
            .pop_front()
            .expect("active beat");
        self.occupancy[beat.port] -= 1;
        beat
    }

    fn check_time(&self, tick: u64) -> Result<(), C220BiuReturnError> {
        if let Some(previous) = self.observed_tick
            && tick < previous
        {
            return Err(C220BiuReturnError::TimeReversed {
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
    ) -> Result<(), C220BiuReturnError> {
        self.check_time(tick)?;
        if previous == Some(tick) {
            return Err(C220BiuReturnError::RepeatedCallback { phase, tick });
        }
        Ok(())
    }
}
