use std::collections::VecDeque;

use super::{C220VmsuError, trace::*, writeback::Writeback};
use crate::isa::c220::vector::merge::C220MergeWidth;
use crate::sim::c220::memory::{C220UbCycle, C220UbRequest};
use crate::sim::c220::numeric::fp16::{is_nan, to_f64};
use crate::sim::c220::vector::ops::merge::{
    C220MergeIssue, C220MergeRecord, C220MergeRepeat, C220MergeRepeatData,
};

const RECORD_BYTES: u64 = 8;
const READ_BATCH_RECORDS: usize = 8;
const WRITE_GROUP_RECORDS: usize = 4;

#[derive(Debug, Clone)]
struct PendingReadBatch {
    list: usize,
    first_record: usize,
    records: usize,
    generation_tick: u64,
    request: C220UbRequest,
}

#[derive(Debug, Clone, Copy)]
struct PairResult {
    record: C220MergeRecord,
    ready_tick: u64,
    cmp0_tick: u64,
}

#[derive(Debug, Clone)]
pub(super) struct RepeatMachine {
    data: C220MergeRepeatData,
    repeat: C220MergeRepeat,
    width: C220MergeWidth,
    ub_response_ticks: u64,
    next_tick: u64,
    generated: [usize; 4],
    credits: [u16; 4],
    ready: [Vec<Option<u64>>; 4],
    source_cursors: [usize; 4],
    read_queue: VecDeque<PendingReadBatch>,
    pair_queues: [VecDeque<PairResult>; 2],
    writeback: Writeback,
    maximum_records: usize,
    produced_records: usize,
    consumed: [u16; 4],
    exhausted_suspension: bool,
    output_complete: bool,
    current_group: u8,
    rr_cursor: usize,
}

impl RepeatMachine {
    pub(super) fn new(
        issue: &C220MergeIssue,
        data: C220MergeRepeatData,
        start_tick: u64,
        ub_response_ticks: u64,
    ) -> Result<Self, C220VmsuError> {
        let maximum_records = data.lists.iter().map(Vec::len).sum();
        if maximum_records == 0 {
            return Err(C220VmsuError::NonprogressingSchedule);
        }
        Ok(Self {
            repeat: issue.repeats[data.repeat_index],
            width: issue.instruction.width,
            ub_response_ticks,
            next_tick: start_tick
                .checked_add(1)
                .ok_or(C220VmsuError::TimeOverflow)?,
            ready: std::array::from_fn(|list| vec![None; data.lists[list].len()]),
            data,
            generated: [0; 4],
            credits: [0; 4],
            source_cursors: [0; 4],
            read_queue: VecDeque::new(),
            pair_queues: std::array::from_fn(|_| VecDeque::new()),
            writeback: Writeback::default(),
            maximum_records,
            produced_records: 0,
            consumed: [0; 4],
            exhausted_suspension: issue.control.exhausted_suspension,
            output_complete: false,
            current_group: 0,
            rr_cursor: 0,
        })
    }

    pub(super) fn empty_trace(&self, start_tick: u64) -> C220VmsuRepeatTrace {
        C220VmsuRepeatTrace {
            repeat_index: self.data.repeat_index,
            start_tick,
            completion_tick: None,
            consumed: [0; 4],
            read_batches: Vec::new(),
            comparisons: Vec::new(),
            write_groups: Vec::new(),
            ub_cycles: Vec::new(),
        }
    }

    pub(super) const fn next_tick(&self) -> u64 {
        self.next_tick
    }

    fn completion_tick(&self) -> Option<u64> {
        self.output_complete
            .then(|| {
                self.writeback
                    .completion_tick(self.produced_records.div_ceil(WRITE_GROUP_RECORDS))
            })
            .flatten()
    }

    pub(super) fn is_complete(&self, tick: u64) -> bool {
        self.completion_tick().is_some_and(|done| done <= tick)
    }

    pub(super) fn projected_completion(&self) -> Result<u64, C220VmsuError> {
        let mut future = self.clone();
        let mut trace = future.empty_trace(future.next_tick);
        let batches: usize = future
            .data
            .lists
            .iter()
            .map(|list| list.len().div_ceil(READ_BATCH_RECORDS))
            .sum();
        let effort = future
            .maximum_records
            .checked_mul(32)
            .and_then(|n| n.checked_add(batches.checked_mul(16)?))
            .and_then(|n| n.checked_add(256))
            .ok_or(C220VmsuError::TimeOverflow)?;
        for _ in 0..effort {
            if let Some(tick) = future.completion_tick() {
                return Ok(tick);
            }
            future.step(&mut trace)?;
            trace.read_batches.clear();
            trace.comparisons.clear();
            trace.write_groups.clear();
            trace.ub_cycles.clear();
        }
        Err(C220VmsuError::NonprogressingSchedule)
    }

    pub(super) fn observe_completion(&self, tick: u64, trace: &mut C220VmsuRepeatTrace) {
        if self.is_complete(tick) {
            trace.completion_tick = self.completion_tick();
            trace.consumed = self.consumed;
        }
    }

    pub(super) fn step(&mut self, trace: &mut C220VmsuRepeatTrace) -> Result<(), C220VmsuError> {
        let tick = self.next_tick;
        if self.completion_tick().is_some() {
            return Ok(());
        }
        self.writeback.prepare(tick)?;
        let data = &self.data;
        let repeat = &self.repeat;
        let width = self.width;
        let ub_response_ticks = self.ub_response_ticks;
        if !self.output_complete && self.read_queue.len() < 2 {
            let selected = (0..4)
                .map(|offset| (self.rr_cursor + offset) % 4)
                .find(|&list| {
                    self.generated[list] < data.lists[list].len() && self.credits[list] <= 8
                });
            if let Some(list) = selected {
                let first_record = self.generated[list];
                let records = (data.lists[list].len() - first_record).min(READ_BATCH_RECORDS);
                let address = repeat.source_addresses[list]
                    .checked_add(first_record as u64 * RECORD_BYTES)
                    .ok_or(C220VmsuError::TimeOverflow)?;
                self.read_queue.push_back(PendingReadBatch {
                    list,
                    first_record,
                    records,
                    generation_tick: tick,
                    request: C220UbRequest::from_accesses(&[(
                        address,
                        records * RECORD_BYTES as usize,
                    )])?,
                });
                self.generated[list] += records;
                self.credits[list] += records as u16;
                self.rr_cursor = (list + 1) % 4;
            }
        }

        let write = self.writeback.request_mut();
        let read_eligible = !self.output_complete
            && self
                .read_queue
                .front()
                .is_some_and(|batch| batch.generation_tick < tick);
        if write.is_some() || read_eligible {
            let read = read_eligible
                .then(|| &mut self.read_queue.front_mut().expect("eligible read").request);
            let cycle = C220UbCycle::arbitrate(tick, write, read, None);
            trace.ub_cycles.push(cycle);
        }
        if read_eligible
            && self
                .read_queue
                .front()
                .is_some_and(|batch| batch.request.is_complete())
        {
            let batch = self.read_queue.pop_front().expect("completed read batch");
            let ready_tick = tick
                .checked_add(ub_response_ticks)
                .ok_or(C220VmsuError::TimeOverflow)?;
            for item in
                &mut self.ready[batch.list][batch.first_record..batch.first_record + batch.records]
            {
                *item = Some(ready_tick);
            }
            trace.read_batches.push(C220VmsuReadBatchTrace {
                repeat_index: data.repeat_index,
                list: batch.list as u8,
                first_record: batch.first_record as u16,
                records: batch.records as u8,
                generation_tick: batch.generation_tick,
                grant_tick: tick,
                ready_tick,
            });
        }
        self.writeback.finish(tick, trace);

        if !self.output_complete {
            for (pair, queue) in self.pair_queues.iter_mut().enumerate() {
                if queue.len() > 1 {
                    continue;
                }
                let left = pair * 2;
                let right = left + 1;
                let left_state = source_state(data, &self.ready, self.source_cursors, left, tick);
                let right_state = source_state(data, &self.ready, self.source_cursors, right, tick);
                let selected = match (left_state, right_state) {
                    (SourceState::Ready(left_record), SourceState::Ready(right_record)) => {
                        Some(if timing_precedes(width, left_record, right_record) {
                            left
                        } else {
                            right
                        })
                    }
                    (SourceState::Ready(_), SourceState::Exhausted) => Some(left),
                    (SourceState::Exhausted, SourceState::Ready(_)) => Some(right),
                    _ => None,
                };
                if let Some(list) = selected {
                    let record = data.lists[list][self.source_cursors[list]];
                    self.source_cursors[list] += 1;
                    self.credits[list] = self.credits[list]
                        .checked_sub(1)
                        .ok_or(C220VmsuError::NonprogressingSchedule)?;
                    queue.push_back(PairResult {
                        record,
                        ready_tick: tick.checked_add(1).ok_or(C220VmsuError::TimeOverflow)?,
                        cmp0_tick: tick,
                    });
                }
            }

            if self.writeback.has_room() {
                let pair_states = std::array::from_fn(|pair| {
                    if let Some(front) = self.pair_queues[pair].front() {
                        if front.ready_tick <= tick {
                            PairState::Ready(*front)
                        } else {
                            PairState::Pending
                        }
                    } else if pair_exhausted(data, self.source_cursors, pair) {
                        PairState::Exhausted
                    } else {
                        PairState::Pending
                    }
                });
                let selected_pair = match pair_states {
                    [PairState::Ready(left), PairState::Ready(right)] => {
                        Some(if timing_precedes(width, &left.record, &right.record) {
                            0
                        } else {
                            1
                        })
                    }
                    [PairState::Ready(_), PairState::Exhausted] => Some(0),
                    [PairState::Exhausted, PairState::Ready(_)] => Some(1),
                    _ => None,
                };
                if let Some(pair) = selected_pair {
                    let result = self.pair_queues[pair]
                        .pop_front()
                        .expect("ready pair result");
                    let record = result.record;
                    let output_index = self.produced_records;
                    self.produced_records += 1;
                    let list = usize::from(record.source_list);
                    self.consumed[list] += 1;
                    self.output_complete = self.produced_records == self.maximum_records
                        || (self.exhausted_suspension
                            && usize::from(self.consumed[list]) == data.lists[list].len());
                    trace.consumed = self.consumed;
                    self.current_group += 1;
                    trace.comparisons.push(C220VmsuCompareTrace {
                        repeat_index: data.repeat_index,
                        output_index,
                        source_list: record.source_list,
                        source_index: record.source_index,
                        cmp0_tick: result.cmp0_tick,
                        cmp1_tick: tick,
                    });
                    if usize::from(self.current_group) == WRITE_GROUP_RECORDS
                        || self.output_complete
                    {
                        let records = std::mem::take(&mut self.current_group);
                        let first_output = self.produced_records - usize::from(records);
                        let destination = repeat
                            .destination_address
                            .checked_add(first_output as u64 * RECORD_BYTES)
                            .ok_or(C220VmsuError::TimeOverflow)?;
                        self.writeback
                            .push(first_output, records, destination, tick);
                    }
                }
            }
        }

        self.next_tick = tick.checked_add(1).ok_or(C220VmsuError::TimeOverflow)?;
        Ok(())
    }
}

fn timing_precedes(width: C220MergeWidth, left: &C220MergeRecord, right: &C220MergeRecord) -> bool {
    match width {
        C220MergeWidth::F16 => {
            !is_nan(left.key_bits as u16)
                && !is_nan(right.key_bits as u16)
                && to_f64(left.key_bits as u16) >= to_f64(right.key_bits as u16)
        }
        C220MergeWidth::F32 => f32::from_bits(left.key_bits) >= f32::from_bits(right.key_bits),
    }
}

#[derive(Debug, Clone, Copy)]
enum SourceState<'a> {
    Ready(&'a C220MergeRecord),
    Pending,
    Exhausted,
}

fn source_state<'a>(
    data: &'a C220MergeRepeatData,
    ready: &[Vec<Option<u64>>; 4],
    cursors: [usize; 4],
    list: usize,
    tick: u64,
) -> SourceState<'a> {
    let cursor = cursors[list];
    let Some(record) = data.lists[list].get(cursor) else {
        return SourceState::Exhausted;
    };
    if ready[list][cursor].is_some_and(|ready_tick| ready_tick <= tick) {
        SourceState::Ready(record)
    } else {
        SourceState::Pending
    }
}

#[derive(Debug, Clone, Copy)]
enum PairState {
    Ready(PairResult),
    Pending,
    Exhausted,
}

fn pair_exhausted(data: &C220MergeRepeatData, cursors: [usize; 4], pair: usize) -> bool {
    let first = pair * 2;
    cursors[first] >= data.lists[first].len() && cursors[first + 1] >= data.lists[first + 1].len()
}
