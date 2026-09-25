use std::collections::VecDeque;

use crate::sim::c220::memory::C220UbRequest;

use super::{C220VmsuError, C220VmsuRepeatTrace, C220VmsuWriteGroupTrace};

const UOP_CAPACITY: usize = 2;
const REQUEST_CAPACITY: usize = 2;

#[derive(Debug, Clone, Copy)]
struct WriteGroup {
    first_output: usize,
    records: u8,
    destination: u64,
    creation_tick: u64,
}

#[derive(Debug, Clone)]
struct Transaction {
    group: WriteGroup,
    submission_tick: u64,
    done_tick: u64,
    request: C220UbRequest,
}

#[derive(Debug, Clone, Default)]
pub(super) struct Writeback {
    waiting: VecDeque<WriteGroup>,
    requests: VecDeque<Transaction>,
    departed_groups: usize,
    last_done_tick: Option<u64>,
}

impl Writeback {
    pub(super) fn request_pending(&self) -> bool {
        !self.requests.is_empty()
    }

    pub(super) fn has_room(&self) -> bool {
        self.waiting.len() < UOP_CAPACITY
    }

    pub(super) fn push(
        &mut self,
        first_output: usize,
        records: u8,
        destination: u64,
        creation_tick: u64,
    ) {
        assert!(self.has_room());
        self.waiting.push_back(WriteGroup {
            first_output,
            records,
            destination,
            creation_tick,
        });
    }

    pub(super) fn prepare(&mut self, tick: u64) -> Result<(), C220VmsuError> {
        if self.requests.len() == REQUEST_CAPACITY
            || self
                .waiting
                .front()
                .is_none_or(|group| group.creation_tick >= tick)
        {
            return Ok(());
        }
        let done_tick = tick.checked_add(1).ok_or(C220VmsuError::TimeOverflow)?;
        let group = self.waiting.pop_front().expect("eligible write group");
        self.requests.push_back(Transaction {
            group,
            submission_tick: tick,
            done_tick,
            request: C220UbRequest::default(),
        });
        self.departed_groups += 1;
        self.last_done_tick = Some(done_tick);
        Ok(())
    }

    pub(super) fn request_mut(&mut self) -> Option<&mut C220UbRequest> {
        self.requests
            .front_mut()
            .map(|transaction| &mut transaction.request)
    }

    pub(super) fn finish(&mut self, tick: u64, trace: &mut C220VmsuRepeatTrace) {
        if !self
            .requests
            .front()
            .is_some_and(|transaction| transaction.request.is_complete())
        {
            return;
        }
        let transaction = self.requests.pop_front().expect("completed write request");
        let group = transaction.group;
        trace.write_groups.push(C220VmsuWriteGroupTrace {
            repeat_index: trace.repeat_index,
            first_output: group.first_output,
            records: group.records,
            destination: group.destination,
            request_bytes: 0,
            creation_tick: group.creation_tick,
            submission_tick: transaction.submission_tick,
            lane_departure_tick: transaction.submission_tick,
            request_completion_tick: tick,
            done_tick: transaction.done_tick,
        });
    }

    pub(super) fn completion_tick(&self, groups: usize) -> Option<u64> {
        (self.departed_groups == groups && self.waiting.is_empty())
            .then_some(self.last_done_tick)
            .flatten()
    }
}
