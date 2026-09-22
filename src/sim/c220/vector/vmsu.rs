use std::collections::VecDeque;

use thiserror::Error;

use crate::memory::sparse::MemoryByteState;
use crate::memory::ub::{UbMemory, UbMemoryError};
use crate::sim::c220::memory::{C220UbCycle, C220UbRequest, C220UbRequestError};
use crate::sim::c220::state::C220State;
use crate::sim::c220::vector::C220VectorError;
use crate::sim::c220::vector::ops::merge::{
    C220MergeIssue, C220MergeRecord, C220MergeRepeatData, load_c220_merge_repeat,
    merge_record_precedes,
};
use crate::sim::c220::vector::pipeline::C220VectorTimingRules;
use crate::sim::common::scalar::ScalarMachineError;

const RECORD_BYTES: u64 = 8;
const READ_BATCH_RECORDS: usize = 8;
const WRITE_GROUP_RECORDS: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220VmsuReadBatchTrace {
    pub repeat_index: usize,
    pub list: u8,
    pub first_record: u16,
    pub records: u8,
    pub generation_tick: u64,
    pub grant_tick: u64,
    pub ready_tick: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220VmsuCompareTrace {
    pub repeat_index: usize,
    pub output_index: usize,
    pub source_list: u8,
    pub source_index: u16,
    pub cmp0_tick: u64,
    pub cmp1_tick: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220VmsuWriteGroupTrace {
    pub repeat_index: usize,
    pub first_output: usize,
    pub records: u8,
    pub destination: u64,
    pub creation_tick: u64,
    pub grant_tick: u64,
    pub done_tick: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220VmsuRepeatTrace {
    pub repeat_index: usize,
    pub start_tick: u64,
    pub completion_tick: u64,
    pub consumed: [u16; 4],
    pub read_batches: Vec<C220VmsuReadBatchTrace>,
    pub comparisons: Vec<C220VmsuCompareTrace>,
    pub write_groups: Vec<C220VmsuWriteGroupTrace>,
    pub ub_cycles: Vec<C220UbCycle>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220VmsuTrace {
    pub issue: C220MergeIssue,
    pub admission_tick: u64,
    pub repeats: Vec<C220VmsuRepeatTrace>,
    pub retirement_tick: Option<u64>,
}

#[derive(Debug, Error)]
pub enum C220VmsuError {
    #[error("C220 VMSU is already active")]
    Busy,
    #[error("C220 VMSU timeline moved backwards from {previous} to {requested}")]
    TimeReversed { previous: u64, requested: u64 },
    #[error("C220 VMSU timeline computation overflowed")]
    TimeOverflow,
    #[error("C220 VMSU merge order diverged from the state result")]
    MergeOrderMismatch,
    #[error("C220 VMSU made no timing progress")]
    NonprogressingSchedule,
    #[error(transparent)]
    Vector(#[from] C220VectorError),
    #[error(transparent)]
    UbRequest(#[from] C220UbRequestError),
    #[error(transparent)]
    Ub(#[from] UbMemoryError),
    #[error(transparent)]
    Scalar(#[from] ScalarMachineError),
}

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
struct PendingWriteGroup {
    first_output: usize,
    records: Vec<C220MergeRecord>,
    destination: u64,
    creation_tick: u64,
    request: C220UbRequest,
}

#[derive(Debug, Clone)]
struct ScheduledWriteGroup {
    trace: C220VmsuWriteGroupTrace,
    records: Vec<C220MergeRecord>,
}

#[derive(Debug, Clone)]
struct RepeatSchedule {
    trace: C220VmsuRepeatTrace,
    writes: Vec<ScheduledWriteGroup>,
}

#[derive(Debug, Clone)]
struct ActiveVmsu {
    trace: C220VmsuTrace,
    repeat_index: usize,
    schedule: Option<RepeatSchedule>,
    committed_writes: usize,
    retirement_tick: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct C220VmsuPipeline {
    rules: C220VectorTimingRules,
    observed_tick: Option<u64>,
    active: Option<ActiveVmsu>,
    last_completed: Option<C220VmsuTrace>,
}

impl C220VmsuPipeline {
    pub const fn new(rules: C220VectorTimingRules) -> Self {
        Self {
            rules,
            observed_tick: None,
            active: None,
            last_completed: None,
        }
    }

    pub const fn is_active(&self) -> bool {
        self.active.is_some()
    }

    pub fn trace(&self) -> Option<&C220VmsuTrace> {
        self.active
            .as_ref()
            .map(|active| &active.trace)
            .or(self.last_completed.as_ref())
    }

    pub fn pending_visibility_tick(&self) -> Option<u64> {
        self.active.as_ref().and_then(|active| {
            active
                .schedule
                .as_ref()
                .map(|schedule| schedule.trace.completion_tick)
                .or_else(|| {
                    active
                        .trace
                        .repeats
                        .last()
                        .map(|repeat| repeat.completion_tick)
                })
        })
    }

    pub fn pending_drain_tick(&self) -> Option<u64> {
        self.active.as_ref().and_then(|active| {
            active.retirement_tick.or_else(|| {
                active.schedule.as_ref().and_then(|schedule| {
                    let remaining = active.trace.issue.repeat_count() - active.repeat_index - 1;
                    schedule
                        .trace
                        .completion_tick
                        .checked_add(remaining as u64)
                        .and_then(|tick| tick.checked_add(2))
                })
            })
        })
    }

    pub fn issue_at(
        &mut self,
        tick: u64,
        issue: C220MergeIssue,
        ub: &UbMemory,
    ) -> Result<Option<u64>, C220VmsuError> {
        if self.active.is_some() {
            return Err(C220VmsuError::Busy);
        }
        if let Some(previous) = self.observed_tick
            && tick < previous
        {
            return Err(C220VmsuError::TimeReversed {
                previous,
                requested: tick,
            });
        }
        if issue.repeat_count() == 0 {
            return Ok(None);
        }
        let admission_tick = tick
            .checked_add(self.rules.dispatch_ticks)
            .ok_or(C220VmsuError::TimeOverflow)?;
        let data = load_c220_merge_repeat(&issue, 0, ub)?;
        let schedule =
            schedule_repeat(&issue, &data, admission_tick, self.rules.ub_response_ticks)?;
        let trace = C220VmsuTrace {
            issue,
            admission_tick,
            repeats: vec![schedule.trace.clone()],
            retirement_tick: None,
        };
        let visibility = schedule.trace.completion_tick;
        self.active = Some(ActiveVmsu {
            trace,
            repeat_index: 0,
            schedule: Some(schedule),
            committed_writes: 0,
            retirement_tick: None,
        });
        Ok(Some(visibility))
    }

    pub(crate) fn next_event_tick(&self) -> Option<u64> {
        let active = self.active.as_ref()?;
        let tick = if let Some(schedule) = &active.schedule {
            schedule
                .writes
                .get(active.committed_writes)
                .map_or(schedule.trace.completion_tick, |write| {
                    write.trace.done_tick.min(schedule.trace.completion_tick)
                })
        } else {
            active.retirement_tick?
        };
        Some(
            tick.max(
                self.observed_tick
                    .map_or(0, |previous| previous.saturating_add(1)),
            ),
        )
    }

    pub fn advance_to(&mut self, tick: u64, core: &mut C220State) -> Result<(), C220VmsuError> {
        if let Some(previous) = self.observed_tick
            && tick < previous
        {
            return Err(C220VmsuError::TimeReversed {
                previous,
                requested: tick,
            });
        }
        while let Some(mut active) = self.active.take() {
            if let Some(schedule) = active.schedule.as_ref() {
                while active.committed_writes < schedule.writes.len()
                    && schedule.writes[active.committed_writes].trace.done_tick <= tick
                {
                    let write = &schedule.writes[active.committed_writes];
                    let mut states =
                        Vec::with_capacity(write.records.len() * RECORD_BYTES as usize);
                    for record in &write.records {
                        states.extend(record.bytes.map(MemoryByteState::Known));
                    }
                    core.ub_mut()
                        .write_states(write.trace.destination, &states)?;
                    active.committed_writes += 1;
                }
                if schedule.trace.completion_tick > tick {
                    self.active = Some(active);
                    break;
                }
                let completion_tick = schedule.trace.completion_tick;
                active.schedule = None;
                active.committed_writes = 0;
                if active.repeat_index + 1 < active.trace.issue.repeat_count() {
                    active.repeat_index += 1;
                    let data = load_c220_merge_repeat(
                        &active.trace.issue,
                        active.repeat_index,
                        core.ub(),
                    )?;
                    let next = schedule_repeat(
                        &active.trace.issue,
                        &data,
                        completion_tick,
                        self.rules.ub_response_ticks,
                    )?;
                    active.trace.repeats.push(next.trace.clone());
                    active.schedule = Some(next);
                    self.active = Some(active);
                    continue;
                }
                let consumed = active
                    .trace
                    .repeats
                    .last()
                    .map_or([0; 4], |repeat| repeat.consumed);
                let status = if active.trace.issue.control.exhausted_suspension {
                    pack_u16x4(consumed)
                } else {
                    0
                };
                core.scalar_mut().machine_mut().set_spr_value(17, status)?;
                let retirement_tick = completion_tick
                    .checked_add(2)
                    .ok_or(C220VmsuError::TimeOverflow)?;
                active.retirement_tick = Some(retirement_tick);
                active.trace.retirement_tick = Some(retirement_tick);
            }
            let retirement_tick = active
                .retirement_tick
                .expect("completed VMSU has a retirement tick");
            if retirement_tick <= tick {
                self.last_completed = Some(active.trace);
                continue;
            }
            self.active = Some(active);
            break;
        }
        self.observed_tick = Some(tick);
        Ok(())
    }
}

fn schedule_repeat(
    issue: &C220MergeIssue,
    data: &C220MergeRepeatData,
    start_tick: u64,
    ub_response_ticks: u64,
) -> Result<RepeatSchedule, C220VmsuError> {
    let repeat = issue
        .repeats
        .get(data.repeat_index)
        .ok_or(C220VectorError::InvalidRepeatIndex(data.repeat_index))?;
    let output_count = data.output.len();
    if output_count == 0 {
        return Err(C220VmsuError::NonprogressingSchedule);
    }
    let mut generated = [0_usize; 4];
    let mut credits = [0_u16; 4];
    let mut ready: [Vec<Option<u64>>; 4] =
        std::array::from_fn(|list| vec![None; data.lists[list].len()]);
    let mut source_cursors = [0_usize; 4];
    let mut read_queue = VecDeque::<PendingReadBatch>::new();
    let mut pair_queues: [VecDeque<PairResult>; 2] = std::array::from_fn(|_| VecDeque::new());
    let mut write_queue = VecDeque::<PendingWriteGroup>::new();
    let mut output = Vec::<C220MergeRecord>::with_capacity(output_count);
    let mut current_group = Vec::<C220MergeRecord>::with_capacity(WRITE_GROUP_RECORDS);
    let mut read_traces = Vec::new();
    let mut compare_traces = Vec::with_capacity(output_count);
    let mut scheduled_writes = Vec::with_capacity(output_count.div_ceil(WRITE_GROUP_RECORDS));
    let mut ub_cycles = Vec::new();
    let mut rr_cursor = 0_usize;
    let total_batches = (0..4)
        .map(|list| data.lists[list].len().div_ceil(READ_BATCH_RECORDS))
        .sum::<usize>();
    let effort = output_count
        .checked_mul(32)
        .and_then(|value| value.checked_add(total_batches.saturating_mul(16)))
        .and_then(|value| value.checked_add(256))
        .ok_or(C220VmsuError::TimeOverflow)?;
    let limit = start_tick
        .checked_add(effort as u64)
        .ok_or(C220VmsuError::TimeOverflow)?;
    let mut tick = start_tick
        .checked_add(1)
        .ok_or(C220VmsuError::TimeOverflow)?;

    while scheduled_writes.len() < output_count.div_ceil(WRITE_GROUP_RECORDS) {
        if tick > limit {
            return Err(C220VmsuError::NonprogressingSchedule);
        }
        if output.len() < output_count && read_queue.len() < 2 {
            let selected = (0..4)
                .map(|offset| (rr_cursor + offset) % 4)
                .find(|&list| generated[list] < data.lists[list].len() && credits[list] <= 8);
            if let Some(list) = selected {
                let first_record = generated[list];
                let records = (data.lists[list].len() - first_record).min(READ_BATCH_RECORDS);
                let address = repeat.source_addresses[list]
                    .checked_add(first_record as u64 * RECORD_BYTES)
                    .ok_or(C220VmsuError::TimeOverflow)?;
                read_queue.push_back(PendingReadBatch {
                    list,
                    first_record,
                    records,
                    generation_tick: tick,
                    request: C220UbRequest::from_accesses(&[(
                        address,
                        records * RECORD_BYTES as usize,
                    )])?,
                });
                generated[list] += records;
                credits[list] += records as u16;
                rr_cursor = (list + 1) % 4;
            }
        }

        let write_eligible = write_queue
            .front()
            .is_some_and(|group| group.creation_tick < tick);
        let read_eligible = output.len() < output_count
            && read_queue
                .front()
                .is_some_and(|batch| batch.generation_tick < tick);
        if write_eligible || read_eligible {
            let write = write_eligible
                .then(|| &mut write_queue.front_mut().expect("eligible write").request);
            let read =
                read_eligible.then(|| &mut read_queue.front_mut().expect("eligible read").request);
            let cycle = C220UbCycle::arbitrate(tick, write, read, None);
            ub_cycles.push(cycle);
        }
        if read_eligible
            && read_queue
                .front()
                .is_some_and(|batch| batch.request.is_complete())
        {
            let batch = read_queue.pop_front().expect("completed read batch");
            let ready_tick = tick
                .checked_add(ub_response_ticks)
                .ok_or(C220VmsuError::TimeOverflow)?;
            for item in
                &mut ready[batch.list][batch.first_record..batch.first_record + batch.records]
            {
                *item = Some(ready_tick);
            }
            read_traces.push(C220VmsuReadBatchTrace {
                repeat_index: data.repeat_index,
                list: batch.list as u8,
                first_record: batch.first_record as u16,
                records: batch.records as u8,
                generation_tick: batch.generation_tick,
                grant_tick: tick,
                ready_tick,
            });
        }
        if write_eligible
            && write_queue
                .front()
                .is_some_and(|group| group.request.is_complete())
        {
            let group = write_queue.pop_front().expect("completed write group");
            let done_tick = tick.checked_add(1).ok_or(C220VmsuError::TimeOverflow)?;
            let trace = C220VmsuWriteGroupTrace {
                repeat_index: data.repeat_index,
                first_output: group.first_output,
                records: group.records.len() as u8,
                destination: group.destination,
                creation_tick: group.creation_tick,
                grant_tick: tick,
                done_tick,
            };
            scheduled_writes.push(ScheduledWriteGroup {
                trace,
                records: group.records,
            });
        }

        if output.len() < output_count {
            for (pair, queue) in pair_queues.iter_mut().enumerate() {
                if queue.len() > 1 {
                    continue;
                }
                let left = pair * 2;
                let right = left + 1;
                let left_state = source_state(data, &ready, source_cursors, left, tick);
                let right_state = source_state(data, &ready, source_cursors, right, tick);
                let selected = match (left_state, right_state) {
                    (SourceState::Ready(left_record), SourceState::Ready(right_record)) => Some(
                        if merge_record_precedes(issue.instruction.width, left_record, right_record)
                        {
                            left
                        } else {
                            right
                        },
                    ),
                    (SourceState::Ready(_), SourceState::Exhausted) => Some(left),
                    (SourceState::Exhausted, SourceState::Ready(_)) => Some(right),
                    _ => None,
                };
                if let Some(list) = selected {
                    let record = data.lists[list][source_cursors[list]];
                    source_cursors[list] += 1;
                    credits[list] = credits[list]
                        .checked_sub(1)
                        .ok_or(C220VmsuError::NonprogressingSchedule)?;
                    queue.push_back(PairResult {
                        record,
                        ready_tick: tick.checked_add(1).ok_or(C220VmsuError::TimeOverflow)?,
                        cmp0_tick: tick,
                    });
                }
            }

            if write_queue.len() < 2 {
                let pair_states = std::array::from_fn(|pair| {
                    if let Some(front) = pair_queues[pair].front() {
                        if front.ready_tick <= tick {
                            PairState::Ready(*front)
                        } else {
                            PairState::Pending
                        }
                    } else if pair_exhausted(data, source_cursors, pair) {
                        PairState::Exhausted
                    } else {
                        PairState::Pending
                    }
                });
                let selected_pair = match pair_states {
                    [PairState::Ready(left), PairState::Ready(right)] => Some(
                        if merge_record_precedes(
                            issue.instruction.width,
                            &left.record,
                            &right.record,
                        ) {
                            0
                        } else {
                            1
                        },
                    ),
                    [PairState::Ready(_), PairState::Exhausted] => Some(0),
                    [PairState::Exhausted, PairState::Ready(_)] => Some(1),
                    _ => None,
                };
                if let Some(pair) = selected_pair {
                    let result = pair_queues[pair].pop_front().expect("ready pair result");
                    let expected = data.output[output.len()];
                    if (result.record.source_list, result.record.source_index)
                        != (expected.source_list, expected.source_index)
                    {
                        return Err(C220VmsuError::MergeOrderMismatch);
                    }
                    let output_index = output.len();
                    output.push(expected);
                    current_group.push(expected);
                    compare_traces.push(C220VmsuCompareTrace {
                        repeat_index: data.repeat_index,
                        output_index,
                        source_list: expected.source_list,
                        source_index: expected.source_index,
                        cmp0_tick: result.cmp0_tick,
                        cmp1_tick: tick,
                    });
                    if current_group.len() == WRITE_GROUP_RECORDS || output.len() == output_count {
                        let records = std::mem::take(&mut current_group);
                        let first_output = output.len() - records.len();
                        let destination = repeat
                            .destination_address
                            .checked_add(first_output as u64 * RECORD_BYTES)
                            .ok_or(C220VmsuError::TimeOverflow)?;
                        write_queue.push_back(PendingWriteGroup {
                            first_output,
                            request: C220UbRequest::from_accesses(&[(
                                destination,
                                records.len() * RECORD_BYTES as usize,
                            )])?,
                            records,
                            destination,
                            creation_tick: tick,
                        });
                    }
                }
            }
        }
        tick = tick.checked_add(1).ok_or(C220VmsuError::TimeOverflow)?;
    }

    let completion_tick = scheduled_writes
        .last()
        .map(|write| write.trace.done_tick)
        .ok_or(C220VmsuError::NonprogressingSchedule)?;
    let write_groups = scheduled_writes.iter().map(|write| write.trace).collect();
    Ok(RepeatSchedule {
        trace: C220VmsuRepeatTrace {
            repeat_index: data.repeat_index,
            start_tick,
            completion_tick,
            consumed: data.consumed,
            read_batches: read_traces,
            comparisons: compare_traces,
            write_groups,
            ub_cycles,
        },
        writes: scheduled_writes,
    })
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

const fn pack_u16x4(values: [u16; 4]) -> u64 {
    values[0] as u64
        | ((values[1] as u64) << 16)
        | ((values[2] as u64) << 32)
        | ((values[3] as u64) << 48)
}
