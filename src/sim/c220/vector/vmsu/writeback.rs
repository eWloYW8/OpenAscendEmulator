use std::collections::VecDeque;

use crate::memory::sparse::MemoryByteState;
use crate::memory::ub::UbMemory;
use crate::sim::c220::memory::{C220UbBank, C220UbRequest};
use crate::sim::c220::vector::ops::merge::C220MergeRecord;
use crate::sim::c220::vector::timing::{C220VectorWriteBlock, C220VectorWritePlan};

use super::{C220VmsuError, C220VmsuRepeatTrace, C220VmsuWriteGroupTrace};

const UOP_CAPACITY: usize = 2;
const REQUEST_CAPACITY: usize = 2;

#[derive(Debug, Clone)]
struct WriteGroup {
    first_output: usize,
    records: Vec<C220MergeRecord>,
    destination: u64,
    creation_tick: u64,
    departure_tick: Option<u64>,
}

#[derive(Debug, Clone)]
struct Transaction {
    group: WriteGroup,
    submission_tick: u64,
    request: C220UbRequest,
}

#[derive(Debug, Clone)]
struct CompletedWrite {
    destination: u64,
    records: Vec<C220MergeRecord>,
    visibility_tick: u64,
}

#[derive(Debug, Clone, Default)]
pub(super) struct Writeback {
    waiting: VecDeque<WriteGroup>,
    requests: VecDeque<Transaction>,
    completed: VecDeque<CompletedWrite>,
    granted_groups: usize,
    last_visibility_tick: Option<u64>,
}

impl Writeback {
    pub(super) fn has_room(&self) -> bool {
        self.waiting.len() < UOP_CAPACITY
    }

    pub(super) fn push(
        &mut self,
        first_output: usize,
        records: Vec<C220MergeRecord>,
        destination: u64,
        creation_tick: u64,
    ) {
        assert!(self.has_room());
        self.waiting.push_back(WriteGroup {
            first_output,
            records,
            destination,
            creation_tick,
            departure_tick: None,
        });
    }

    pub(super) fn prepare(&mut self, tick: u64) -> Result<(), C220VmsuError> {
        let Some(group) = self
            .waiting
            .front_mut()
            .filter(|group| group.creation_tick < tick)
        else {
            return Ok(());
        };
        if group.departure_tick.is_none() {
            if self.requests.len() == REQUEST_CAPACITY {
                return Ok(());
            }
            let bytes = group.records.len() * 8;
            let plan = C220VectorWritePlan {
                blocks: vec![C220VectorWriteBlock {
                    repeat_index: 0,
                    block_index: 0,
                    base_address: group.destination,
                    bank: C220UbBank::from_address(group.destination),
                    element_bytes: 1,
                    active_lane_mask: ((1_u64 << bytes) - 1) as u32,
                }],
            };
            let latency = plan
                .writeback_ticks()
                .ok_or(C220VmsuError::NonprogressingSchedule)?;
            let departure_tick = tick
                .checked_add(latency.saturating_sub(2) as u64)
                .ok_or(C220VmsuError::TimeOverflow)?;
            let request = C220UbRequest::from_writes(&[(group.destination, bytes, true)])?;
            group.departure_tick = Some(departure_tick);
            let submitted = WriteGroup {
                records: std::mem::take(&mut group.records),
                ..group.clone()
            };
            self.requests.push_back(Transaction {
                group: submitted,
                submission_tick: tick,
                request,
            });
        }
        if group
            .departure_tick
            .is_some_and(|departure| departure <= tick)
        {
            self.waiting.pop_front();
        }
        Ok(())
    }

    pub(super) fn request_mut(&mut self) -> Option<&mut C220UbRequest> {
        self.requests
            .front_mut()
            .map(|transaction| &mut transaction.request)
    }

    pub(super) fn finish(
        &mut self,
        tick: u64,
        trace: &mut C220VmsuRepeatTrace,
    ) -> Result<(), C220VmsuError> {
        if !self
            .requests
            .front()
            .is_some_and(|transaction| transaction.request.is_complete())
        {
            return Ok(());
        }
        let done_tick = tick.checked_add(1).ok_or(C220VmsuError::TimeOverflow)?;
        let transaction = self.requests.pop_front().expect("completed write request");
        let group = transaction.group;
        trace.write_groups.push(C220VmsuWriteGroupTrace {
            repeat_index: trace.repeat_index,
            first_output: group.first_output,
            records: group.records.len() as u8,
            destination: group.destination,
            creation_tick: group.creation_tick,
            submission_tick: transaction.submission_tick,
            lane_departure_tick: group.departure_tick.expect("submitted write lane"),
            grant_tick: tick,
            done_tick,
        });
        self.granted_groups += 1;
        self.last_visibility_tick = Some(done_tick);
        self.completed.push_back(CompletedWrite {
            destination: group.destination,
            records: group.records,
            visibility_tick: done_tick,
        });
        Ok(())
    }

    pub(super) fn completion_tick(&self, groups: usize) -> Option<u64> {
        (self.granted_groups == groups && self.waiting.is_empty())
            .then_some(self.last_visibility_tick)
            .flatten()
    }

    pub(super) fn commit(&mut self, tick: u64, ub: &mut UbMemory) -> Result<(), C220VmsuError> {
        while self
            .completed
            .front()
            .is_some_and(|write| write.visibility_tick <= tick)
        {
            let write = self.completed.front().expect("visible write group");
            let states: Vec<_> = write
                .records
                .iter()
                .flat_map(|record| record.bytes.map(MemoryByteState::Known))
                .collect();
            ub.write_states(write.destination, &states)?;
            self.completed.pop_front();
        }
        Ok(())
    }

    pub(super) fn discard_projected_writes(&mut self) {
        self.completed.clear();
    }
}
