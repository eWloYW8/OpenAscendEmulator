use serde::Serialize;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum C310BufferDisposition {
    Accepted,
    Stalled,
    Unsupported,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum C310GetBufGateResult {
    Ready,
    AwaitReleaseCount { expected: u32, observed: u32 },
    SameTick { tick: u64 },
}

pub const fn evaluate_c310_get_buf_gate(
    expected_release_count: u32,
    observed_release_count: u32,
    last_tick: u64,
    current_tick: u64,
) -> C310GetBufGateResult {
    if expected_release_count != observed_release_count {
        return C310GetBufGateResult::AwaitReleaseCount {
            expected: expected_release_count,
            observed: observed_release_count,
        };
    }
    if last_tick == current_tick {
        return C310GetBufGateResult::SameTick { tick: current_tick };
    }
    C310GetBufGateResult::Ready
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct C310BufferCounter {
    pub dispatch_count: u32,
    pub release_count: u32,
    pub total_dispatches: u32,
    pub outstanding_get_bufs: u32,
    pub last_release_tick: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct C310GetBufDispatch {
    pub expected_release_count: u32,
    pub assigned_dispatch_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum C310GetBufIssueTickResult {
    Primed { recorded_tick: u32 },
    Continue,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct C310GetBufIssueTick {
    recorded_tick: u32,
}

impl C310GetBufIssueTick {
    pub const fn new(recorded_tick: u32) -> Self {
        Self { recorded_tick }
    }

    pub const fn recorded_tick(&self) -> u32 {
        self.recorded_tick
    }

    pub fn step(&mut self, kind: u32, current_tick: u64) -> C310GetBufIssueTickResult {
        if kind == 1 && self.recorded_tick == 0 {
            self.recorded_tick = current_tick as u32;
            C310GetBufIssueTickResult::Primed {
                recorded_tick: self.recorded_tick,
            }
        } else {
            C310GetBufIssueTickResult::Continue
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct C310BufferCounters {
    ring_size: u32,
    counters: Vec<C310BufferCounter>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct C310ReleaseEntry {
    pub valid: bool,
    pub pipe_code: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct C310ReleaseEntries {
    entries: Vec<C310ReleaseEntry>,
}

impl C310ReleaseEntries {
    pub fn new(buffer_count: u8) -> Result<Self, C310BufferCounterError> {
        if !(1..=64).contains(&buffer_count) {
            return Err(C310BufferCounterError::InvalidBufferCount);
        }
        Ok(Self {
            entries: vec![C310ReleaseEntry::default(); usize::from(buffer_count)],
        })
    }

    pub fn buffer_count(&self) -> u8 {
        self.entries.len() as u8
    }

    pub fn entry(&self, id: u8) -> Result<C310ReleaseEntry, C310BufferCounterError> {
        self.entries
            .get(usize::from(id))
            .copied()
            .ok_or_else(|| self.invalid_id(id))
    }

    pub fn record_accepted_release(
        &mut self,
        id: u8,
        pipe_code: u8,
    ) -> Result<(), C310BufferCounterError> {
        let count = self.buffer_count();
        let entry = self
            .entries
            .get_mut(usize::from(id))
            .ok_or(C310BufferCounterError::InvalidBufferId { id, count })?;
        *entry = C310ReleaseEntry {
            valid: true,
            pipe_code,
        };
        Ok(())
    }

    pub fn matches(&self, id: u8, pipe_code: u8) -> Result<bool, C310BufferCounterError> {
        let entry = self.entry(id)?;
        Ok(entry.valid && entry.pipe_code == pipe_code)
    }

    pub fn take_match(&mut self, id: u8, pipe_code: u8) -> Result<bool, C310BufferCounterError> {
        if !self.matches(id, pipe_code)? {
            return Ok(false);
        }
        self.entries[usize::from(id)].valid = false;
        Ok(true)
    }

    fn invalid_id(&self, id: u8) -> C310BufferCounterError {
        C310BufferCounterError::InvalidBufferId {
            id,
            count: self.buffer_count(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum C310GetBufAdmission {
    RingFull,
    PipeStalled,
    Direct(C310GetBufDispatch),
    Queued(C310GetBufDispatch),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum C310ReleaseAdmission {
    NoPipe,
    PipeStalled,
    Queued,
}

const fn is_c310_buffer_issue_pipe(pipe_code: u8) -> bool {
    matches!(pipe_code, 1 | 2 | 3 | 4 | 5 | 10)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct C310BufferAdmissionState {
    counters: C310BufferCounters,
    release_entries: C310ReleaseEntries,
    direct_release_enabled: bool,
}

impl C310BufferAdmissionState {
    pub fn new(
        buffer_count: u8,
        ring_size: u32,
        direct_release_enabled: bool,
    ) -> Result<Self, C310BufferCounterError> {
        Ok(Self {
            counters: C310BufferCounters::new(buffer_count, ring_size)?,
            release_entries: C310ReleaseEntries::new(buffer_count)?,
            direct_release_enabled,
        })
    }

    pub const fn counters(&self) -> &C310BufferCounters {
        &self.counters
    }

    pub const fn release_entries(&self) -> &C310ReleaseEntries {
        &self.release_entries
    }

    pub fn try_pop_queued_get(
        &mut self,
        id: u8,
        expected_release_count: u32,
        current_tick: u64,
    ) -> Result<C310GetBufGateResult, C310BufferCounterError> {
        let gate = self
            .counters
            .get_buf_gate(id, expected_release_count, current_tick)?;
        if gate == C310GetBufGateResult::Ready {
            self.counters.consume_get_buf(id)?;
        }
        Ok(gate)
    }

    pub fn retire_queued_release_at(
        &mut self,
        id: u8,
        mode_field: u8,
        tick: u64,
    ) -> Result<u32, C310BufferCounterError> {
        self.counters.retire_queued_release_at(id, mode_field, tick)
    }

    pub fn admit_get(
        &mut self,
        id: u8,
        pipe_code: u8,
        mode_field: u8,
        try_queue: impl FnOnce() -> bool,
    ) -> Result<C310GetBufAdmission, C310BufferCounterError> {
        if self.counters.is_full(id)? {
            return Ok(C310GetBufAdmission::RingFull);
        }
        if mode_field == 1
            || (self.direct_release_enabled && self.release_entries.matches(id, pipe_code)?)
        {
            if mode_field != 1 {
                self.release_entries.take_match(id, pipe_code)?;
            }
            let dispatch = self
                .counters
                .record_direct_get_buf(id)?
                .expect("ring capacity checked before direct admission");
            return Ok(C310GetBufAdmission::Direct(dispatch));
        }
        if !is_c310_buffer_issue_pipe(pipe_code) {
            return Err(C310BufferCounterError::InvalidPipeCode { code: pipe_code });
        }
        if !try_queue() {
            return Ok(C310GetBufAdmission::PipeStalled);
        }
        let dispatch = self
            .counters
            .record_get_buf_admission(id)?
            .expect("ring capacity checked before queue admission");
        Ok(C310GetBufAdmission::Queued(dispatch))
    }

    pub fn admit_release(
        &mut self,
        id: u8,
        pipe_code: u8,
        try_queue: impl FnOnce() -> bool,
    ) -> Result<C310ReleaseAdmission, C310BufferCounterError> {
        if pipe_code == 0 {
            return Ok(C310ReleaseAdmission::NoPipe);
        }
        if !is_c310_buffer_issue_pipe(pipe_code) {
            return Err(C310BufferCounterError::InvalidPipeCode { code: pipe_code });
        }
        self.release_entries.entry(id)?;
        if !try_queue() {
            return Ok(C310ReleaseAdmission::PipeStalled);
        }
        self.release_entries
            .record_accepted_release(id, pipe_code)?;
        Ok(C310ReleaseAdmission::Queued)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum C310BufferCounterError {
    #[error("buffer count must be between 1 and 64")]
    InvalidBufferCount,
    #[error("buffer ring size must be at least 2")]
    InvalidRingSize,
    #[error("buffer ID {id} is outside configured range 0..{count}")]
    InvalidBufferId { id: u8, count: u8 },
    #[error("buffer issue pipe code {code} is unsupported")]
    InvalidPipeCode { code: u8 },
}

impl C310BufferCounters {
    pub fn new(buffer_count: u8, ring_size: u32) -> Result<Self, C310BufferCounterError> {
        if !(1..=64).contains(&buffer_count) {
            return Err(C310BufferCounterError::InvalidBufferCount);
        }
        if ring_size < 2 {
            return Err(C310BufferCounterError::InvalidRingSize);
        }
        Ok(Self {
            ring_size,
            counters: vec![
                C310BufferCounter {
                    dispatch_count: 0,
                    release_count: 0,
                    total_dispatches: 0,
                    outstanding_get_bufs: 0,
                    last_release_tick: 0,
                };
                usize::from(buffer_count)
            ],
        })
    }

    pub const fn ring_size(&self) -> u32 {
        self.ring_size
    }

    pub fn buffer_count(&self) -> u8 {
        self.counters.len() as u8
    }

    pub fn counter(&self, id: u8) -> Result<C310BufferCounter, C310BufferCounterError> {
        self.counters
            .get(usize::from(id))
            .copied()
            .ok_or_else(|| self.invalid_id(id))
    }

    pub fn is_full(&self, id: u8) -> Result<bool, C310BufferCounterError> {
        let counter = self.counter(id)?;
        Ok(counter.release_count == self.next(counter.dispatch_count))
    }

    pub fn reserve_dispatch_slot(&mut self, id: u8) -> Result<bool, C310BufferCounterError> {
        Ok(self.record_get_buf_admission(id)?.is_some())
    }

    pub fn record_get_buf_admission(
        &mut self,
        id: u8,
    ) -> Result<Option<C310GetBufDispatch>, C310BufferCounterError> {
        self.record_dispatch(id, true)
    }

    pub fn record_direct_get_buf(
        &mut self,
        id: u8,
    ) -> Result<Option<C310GetBufDispatch>, C310BufferCounterError> {
        self.record_dispatch(id, false)
    }

    fn record_dispatch(
        &mut self,
        id: u8,
        queued: bool,
    ) -> Result<Option<C310GetBufDispatch>, C310BufferCounterError> {
        if self.is_full(id)? {
            return Ok(None);
        }
        let expected_release_count = self.counter(id)?.dispatch_count;
        let next = self.next(expected_release_count);
        let counter = &mut self.counters[usize::from(id)];
        counter.dispatch_count = next;
        counter.total_dispatches = counter.total_dispatches.wrapping_add(1);
        if queued {
            counter.outstanding_get_bufs = counter.outstanding_get_bufs.wrapping_add(1);
        }
        Ok(Some(C310GetBufDispatch {
            expected_release_count,
            assigned_dispatch_count: next,
        }))
    }

    pub fn retire_release(&mut self, id: u8) -> Result<u32, C310BufferCounterError> {
        let next = self.next(self.counter(id)?.release_count);
        self.counters[usize::from(id)].release_count = next;
        Ok(next)
    }

    pub fn retire_queued_release_at(
        &mut self,
        id: u8,
        mode_field: u8,
        tick: u64,
    ) -> Result<u32, C310BufferCounterError> {
        let next = self.retire_release(id)?;
        if mode_field != 1 {
            self.record_release_tick(id, tick)?;
        }
        Ok(next)
    }

    pub fn consume_get_buf(&mut self, id: u8) -> Result<u32, C310BufferCounterError> {
        let current = self.counter(id)?.outstanding_get_bufs;
        let next = current.wrapping_sub(1);
        self.counters[usize::from(id)].outstanding_get_bufs = next;
        Ok(next)
    }

    pub fn record_release_tick(&mut self, id: u8, tick: u64) -> Result<(), C310BufferCounterError> {
        let index = usize::from(id);
        if index >= self.counters.len() {
            return Err(self.invalid_id(id));
        }
        self.counters[index].last_release_tick = tick;
        Ok(())
    }

    pub fn get_buf_gate(
        &self,
        id: u8,
        expected_release_count: u32,
        current_tick: u64,
    ) -> Result<C310GetBufGateResult, C310BufferCounterError> {
        let counter = self.counter(id)?;
        Ok(evaluate_c310_get_buf_gate(
            expected_release_count,
            counter.release_count,
            counter.last_release_tick,
            current_tick,
        ))
    }

    fn next(&self, current: u32) -> u32 {
        if current == self.ring_size - 1 {
            0
        } else {
            current + 1
        }
    }

    fn invalid_id(&self, id: u8) -> C310BufferCounterError {
        C310BufferCounterError::InvalidBufferId {
            id,
            count: self.buffer_count(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::issue_queue_c310::{C310DequeueOutcome, C310IssueQueue};

    #[test]
    fn admission_checks_ring_before_queue_and_preserves_state_on_pipe_stall() {
        let mut state = C310BufferAdmissionState::new(1, 2, false).unwrap();
        let original = state.clone();
        assert_eq!(
            state.admit_get(0, 4, 0, || false),
            Ok(C310GetBufAdmission::PipeStalled)
        );
        assert_eq!(state, original);
        assert_eq!(
            state.admit_get(0, 4, 0, || true),
            Ok(C310GetBufAdmission::Queued(C310GetBufDispatch {
                expected_release_count: 0,
                assigned_dispatch_count: 1,
            }))
        );
        let full = state.clone();
        assert_eq!(
            state.admit_get(0, 4, 0, || panic!("full ring must not try the pipe")),
            Ok(C310GetBufAdmission::RingFull)
        );
        assert_eq!(state, full);
    }

    #[test]
    fn queued_get_waits_for_release_count_and_a_later_tick_before_pop() {
        let mut state = C310BufferAdmissionState::new(1, 4, false).unwrap();
        let mut pipe = C310IssueQueue::new(3).unwrap();
        let first = match state
            .admit_get(0, 4, 0, || pipe.try_enqueue(1_u8).is_ok())
            .unwrap()
        {
            C310GetBufAdmission::Queued(dispatch) => dispatch,
            other => panic!("expected queued GET, got {other:?}"),
        };
        assert_eq!(
            state.try_pop_queued_get(0, first.expected_release_count, 1),
            Ok(C310GetBufGateResult::Ready)
        );
        assert!(matches!(
            pipe.try_dequeue_if(|_| true),
            C310DequeueOutcome::Dequeued { item: 1, .. }
        ));
        assert_eq!(state.counters().counter(0).unwrap().outstanding_get_bufs, 0);

        let second = match state
            .admit_get(0, 4, 0, || pipe.try_enqueue(2_u8).is_ok())
            .unwrap()
        {
            C310GetBufAdmission::Queued(dispatch) => dispatch,
            other => panic!("expected queued GET, got {other:?}"),
        };
        assert_eq!(second.expected_release_count, 1);
        let before = state.clone();
        assert_eq!(
            state.try_pop_queued_get(0, second.expected_release_count, 2),
            Ok(C310GetBufGateResult::AwaitReleaseCount {
                expected: 1,
                observed: 0,
            })
        );
        assert_eq!(state, before);
        assert_eq!(pipe.front(), Some(&2));

        assert_eq!(state.retire_queued_release_at(0, 0, 2), Ok(1));
        assert_eq!(
            state.try_pop_queued_get(0, second.expected_release_count, 2),
            Ok(C310GetBufGateResult::SameTick { tick: 2 })
        );
        assert_eq!(state.counters().counter(0).unwrap().outstanding_get_bufs, 1);
        assert_eq!(
            state.try_pop_queued_get(0, second.expected_release_count, 3),
            Ok(C310GetBufGateResult::Ready)
        );
        assert!(matches!(
            pipe.try_dequeue_if(|_| true),
            C310DequeueOutcome::Dequeued { item: 2, .. }
        ));
        assert_eq!(state.counters().counter(0).unwrap().outstanding_get_bufs, 0);
    }

    #[test]
    fn queue_capacity_and_dispatch_count_change_together() {
        let mut state = C310BufferAdmissionState::new(1, 4, false).unwrap();
        let mut pipe = C310IssueQueue::new(1).unwrap();
        assert!(matches!(
            state.admit_get(0, 4, 0, || pipe.try_enqueue(7).is_ok()),
            Ok(C310GetBufAdmission::Queued(_))
        ));
        let counters_after_first = state.counters().clone();
        assert_eq!(
            state.admit_get(0, 4, 0, || pipe.try_enqueue(8).is_ok()),
            Ok(C310GetBufAdmission::PipeStalled)
        );
        assert_eq!(state.counters(), &counters_after_first);
        assert_eq!(pipe.front(), Some(&7));
        assert!(matches!(
            pipe.try_dequeue_if(|_| true),
            C310DequeueOutcome::Dequeued { item: 7, .. }
        ));
        assert!(matches!(
            state.admit_get(0, 4, 0, || pipe.try_enqueue(8).is_ok()),
            Ok(C310GetBufAdmission::Queued(_))
        ));
        assert_eq!(state.counters().counter(0).unwrap().dispatch_count, 2);
        assert_eq!(pipe.front(), Some(&8));
    }

    #[test]
    fn accepted_release_shortcuts_only_a_matching_pipe_when_enabled() {
        let mut state = C310BufferAdmissionState::new(1, 4, true).unwrap();
        assert_eq!(
            state.admit_release(0, 4, || false),
            Ok(C310ReleaseAdmission::PipeStalled)
        );
        assert!(!state.release_entries().entry(0).unwrap().valid);
        assert_eq!(
            state.admit_release(0, 4, || true),
            Ok(C310ReleaseAdmission::Queued)
        );
        assert_eq!(
            state.admit_get(0, 5, 0, || true),
            Ok(C310GetBufAdmission::Queued(C310GetBufDispatch {
                expected_release_count: 0,
                assigned_dispatch_count: 1,
            }))
        );
        assert_eq!(state.release_entries().matches(0, 4), Ok(true));
        assert_eq!(
            state.admit_get(0, 4, 0, || panic!("matching release is direct")),
            Ok(C310GetBufAdmission::Direct(C310GetBufDispatch {
                expected_release_count: 1,
                assigned_dispatch_count: 2,
            }))
        );
        assert_eq!(state.release_entries().matches(0, 4), Ok(false));
        assert_eq!(state.counters().counter(0).unwrap().outstanding_get_bufs, 1);
    }

    #[test]
    fn mode_one_is_direct_without_consuming_a_release_entry() {
        let mut state = C310BufferAdmissionState::new(1, 4, false).unwrap();
        state.admit_release(0, 1, || true).unwrap();
        assert!(matches!(
            state.admit_get(0, 1, 1, || panic!("mode one is direct")),
            Ok(C310GetBufAdmission::Direct(_))
        ));
        assert_eq!(state.release_entries().matches(0, 1), Ok(true));
        assert_eq!(state.counters().counter(0).unwrap().outstanding_get_bufs, 0);
    }

    #[test]
    fn disabled_direct_match_still_uses_the_issue_pipe() {
        let mut state = C310BufferAdmissionState::new(1, 4, false).unwrap();
        state.admit_release(0, 4, || true).unwrap();
        assert!(matches!(
            state.admit_get(0, 4, 0, || true),
            Ok(C310GetBufAdmission::Queued(_))
        ));
        assert_eq!(state.release_entries().matches(0, 4), Ok(true));
        assert_eq!(state.counters().counter(0).unwrap().outstanding_get_bufs, 1);
    }

    #[test]
    fn invalid_release_id_does_not_invoke_pipe_callback() {
        let mut state = C310BufferAdmissionState::new(1, 4, true).unwrap();
        let before = state.clone();
        assert_eq!(
            state.admit_release(1, 4, || panic!("invalid ID must not queue")),
            Err(C310BufferCounterError::InvalidBufferId { id: 1, count: 1 })
        );
        assert_eq!(state, before);
    }

    #[test]
    fn release_without_pipe_has_no_queue_or_entry_effect() {
        let mut state = C310BufferAdmissionState::new(1, 4, true).unwrap();
        let before = state.clone();
        assert_eq!(
            state.admit_release(1, 0, || panic!("pipe zero must not queue")),
            Ok(C310ReleaseAdmission::NoPipe)
        );
        assert_eq!(state, before);
        assert_eq!(
            state.admit_release(0, 6, || panic!("unknown pipe must not queue")),
            Err(C310BufferCounterError::InvalidPipeCode { code: 6 })
        );
        assert_eq!(
            state.admit_get(0, 6, 0, || panic!("unknown pipe must not queue")),
            Err(C310BufferCounterError::InvalidPipeCode { code: 6 })
        );
        assert_eq!(state, before);
    }

    #[test]
    fn accepted_release_entry_matches_once_on_the_same_pipe() {
        let mut entries = C310ReleaseEntries::new(2).unwrap();
        assert_eq!(entries.entry(0), Ok(C310ReleaseEntry::default()));
        entries.record_accepted_release(0, 3).unwrap();
        assert_eq!(entries.matches(0, 1), Ok(false));
        assert_eq!(entries.take_match(0, 1), Ok(false));
        assert_eq!(
            entries.entry(0),
            Ok(C310ReleaseEntry {
                valid: true,
                pipe_code: 3,
            })
        );
        assert_eq!(entries.take_match(0, 3), Ok(true));
        assert_eq!(entries.take_match(0, 3), Ok(false));
        assert_eq!(entries.entry(0).unwrap().pipe_code, 3);
        assert_eq!(entries.entry(1), Ok(C310ReleaseEntry::default()));
    }

    #[test]
    fn accepted_release_entry_replaces_previous_pipe() {
        let mut entries = C310ReleaseEntries::new(1).unwrap();
        entries.record_accepted_release(0, 1).unwrap();
        entries.record_accepted_release(0, 10).unwrap();
        assert_eq!(entries.matches(0, 1), Ok(false));
        assert_eq!(entries.take_match(0, 10), Ok(true));
    }

    #[test]
    fn release_entry_rejects_invalid_count_and_id_without_mutation() {
        assert_eq!(
            C310ReleaseEntries::new(0),
            Err(C310BufferCounterError::InvalidBufferCount)
        );
        assert_eq!(
            C310ReleaseEntries::new(65),
            Err(C310BufferCounterError::InvalidBufferCount)
        );
        let mut entries = C310ReleaseEntries::new(64).unwrap();
        entries.record_accepted_release(63, 5).unwrap();
        let original = entries.clone();
        let invalid = C310BufferCounterError::InvalidBufferId { id: 64, count: 64 };
        assert_eq!(entries.entry(64), Err(invalid));
        assert_eq!(entries.matches(64, 5), Err(invalid));
        assert_eq!(entries.take_match(64, 5), Err(invalid));
        assert_eq!(entries.record_accepted_release(64, 5), Err(invalid));
        assert_eq!(entries, original);
    }

    #[test]
    fn get_gate_reports_release_and_tick_blockers_in_order() {
        assert_eq!(
            evaluate_c310_get_buf_gate(2, 1, 7, 7),
            C310GetBufGateResult::AwaitReleaseCount {
                expected: 2,
                observed: 1,
            }
        );
        assert_eq!(
            evaluate_c310_get_buf_gate(2, 2, 7, 7),
            C310GetBufGateResult::SameTick { tick: 7 }
        );
        assert_eq!(
            evaluate_c310_get_buf_gate(2, 2, 7, 8),
            C310GetBufGateResult::Ready
        );
    }

    #[test]
    fn get_stalls_at_ring_capacity_and_retries_after_release() {
        let mut counters = C310BufferCounters::new(32, 32).unwrap();
        for expected in 1..32 {
            assert_eq!(counters.reserve_dispatch_slot(1), Ok(true));
            assert_eq!(counters.counter(1).unwrap().dispatch_count, expected);
        }
        assert_eq!(counters.is_full(1), Ok(true));
        assert_eq!(counters.reserve_dispatch_slot(1), Ok(false));
        assert_eq!(counters.counter(1).unwrap().dispatch_count, 31);
        assert_eq!(counters.retire_release(1), Ok(1));
        assert_eq!(counters.reserve_dispatch_slot(1), Ok(true));
        assert_eq!(counters.counter(1).unwrap().dispatch_count, 0);
        assert_eq!(counters.counter(1).unwrap().total_dispatches, 32);
        assert_eq!(counters.is_full(1), Ok(true));
    }

    #[test]
    fn admission_reports_pre_and_post_dispatch_counts() {
        let mut counters = C310BufferCounters::new(64, 3).unwrap();
        assert_eq!(
            counters.record_get_buf_admission(63),
            Ok(Some(C310GetBufDispatch {
                expected_release_count: 0,
                assigned_dispatch_count: 1,
            }))
        );
        assert_eq!(
            counters.record_get_buf_admission(63),
            Ok(Some(C310GetBufDispatch {
                expected_release_count: 1,
                assigned_dispatch_count: 2,
            }))
        );
        let full = counters.clone();
        assert_eq!(counters.record_get_buf_admission(63), Ok(None));
        assert_eq!(counters, full);
        assert_eq!(counters.retire_release(63), Ok(1));
        assert_eq!(
            counters.record_get_buf_admission(63),
            Ok(Some(C310GetBufDispatch {
                expected_release_count: 2,
                assigned_dispatch_count: 0,
            }))
        );
    }

    #[test]
    fn consumption_changes_outstanding_count_without_changing_dispatch_history() {
        let mut counters = C310BufferCounters::new(2, 32).unwrap();
        assert!(counters.record_get_buf_admission(0).unwrap().is_some());
        assert_eq!(counters.counter(0).unwrap().outstanding_get_bufs, 1);
        assert_eq!(counters.consume_get_buf(0), Ok(0));
        assert_eq!(counters.counter(0).unwrap().dispatch_count, 1);
        assert_eq!(counters.counter(0).unwrap().total_dispatches, 1);
        assert_eq!(counters.counter(0).unwrap().release_count, 0);
        assert_eq!(counters.retire_release(0), Ok(1));
        assert!(counters.record_get_buf_admission(0).unwrap().is_some());
        assert_eq!(counters.counter(0).unwrap().outstanding_get_bufs, 1);
        assert_eq!(counters.counter(0).unwrap().total_dispatches, 2);
        assert_eq!(counters.counter(1).unwrap().outstanding_get_bufs, 0);
    }

    #[test]
    fn direct_get_advances_dispatch_without_occupying_the_issue_queue() {
        let mut counters = C310BufferCounters::new(1, 3).unwrap();
        assert_eq!(
            counters.record_direct_get_buf(0),
            Ok(Some(C310GetBufDispatch {
                expected_release_count: 0,
                assigned_dispatch_count: 1,
            }))
        );
        assert_eq!(counters.counter(0).unwrap().outstanding_get_bufs, 0);
        assert_eq!(counters.counter(0).unwrap().total_dispatches, 1);
        assert_eq!(
            counters
                .record_get_buf_admission(0)
                .unwrap()
                .unwrap()
                .assigned_dispatch_count,
            2
        );
        assert_eq!(counters.counter(0).unwrap().outstanding_get_bufs, 1);
        let full = counters.clone();
        assert_eq!(counters.record_direct_get_buf(0), Ok(None));
        assert_eq!(counters, full);
        assert_eq!(counters.retire_release(0), Ok(1));
        assert_eq!(
            counters
                .record_direct_get_buf(0)
                .unwrap()
                .unwrap()
                .assigned_dispatch_count,
            0
        );
        assert_eq!(counters.counter(0).unwrap().outstanding_get_bufs, 1);
    }

    #[test]
    fn release_retirement_updates_tick_only_for_non_mode_one() {
        let mut counters = C310BufferCounters::new(1, 4).unwrap();
        assert_eq!(counters.retire_queued_release_at(0, 0, 7), Ok(1));
        assert_eq!(counters.counter(0).unwrap().last_release_tick, 7);
        assert_eq!(counters.retire_queued_release_at(0, 1, 8), Ok(2));
        assert_eq!(counters.counter(0).unwrap().last_release_tick, 7);
        assert_eq!(counters.retire_queued_release_at(0, 0, 9), Ok(3));
        assert_eq!(counters.retire_queued_release_at(0, 0, 10), Ok(0));
        assert_eq!(counters.counter(0).unwrap().last_release_tick, 10);
        let prior = counters.clone();
        assert_eq!(
            counters.retire_queued_release_at(1, 0, 11),
            Err(C310BufferCounterError::InvalidBufferId { id: 1, count: 1 })
        );
        assert_eq!(counters, prior);
    }

    #[test]
    fn consumption_uses_wrapping_u32_arithmetic() {
        let mut counters = C310BufferCounters::new(1, 32).unwrap();
        assert_eq!(counters.consume_get_buf(0), Ok(u32::MAX));
        assert_eq!(counters.counter(0).unwrap().outstanding_get_bufs, u32::MAX);
        assert_eq!(
            counters.consume_get_buf(1),
            Err(C310BufferCounterError::InvalidBufferId { id: 1, count: 1 })
        );
    }

    #[test]
    fn issue_tick_is_initialized_once_before_dispatch() {
        let mut tick = C310GetBufIssueTick::new(0);
        assert_eq!(
            tick.step(1, 1849),
            C310GetBufIssueTickResult::Primed {
                recorded_tick: 1849
            }
        );
        assert_eq!(tick.recorded_tick(), 1849);
        assert_eq!(tick.step(1, 1850), C310GetBufIssueTickResult::Continue);
        assert_eq!(tick.recorded_tick(), 1849);

        let mut other_kind = C310GetBufIssueTick::new(0);
        assert_eq!(
            other_kind.step(2, 1849),
            C310GetBufIssueTickResult::Continue
        );
        assert_eq!(other_kind.recorded_tick(), 0);

        let mut zero_time = C310GetBufIssueTick::new(0);
        assert_eq!(
            zero_time.step(1, 0),
            C310GetBufIssueTickResult::Primed { recorded_tick: 0 }
        );
        assert_eq!(
            zero_time.step(1, 1),
            C310GetBufIssueTickResult::Primed { recorded_tick: 1 }
        );

        let mut truncation = C310GetBufIssueTick::new(0);
        assert_eq!(
            truncation.step(1, u64::from(u32::MAX) + 2),
            C310GetBufIssueTickResult::Primed { recorded_tick: 1 }
        );
    }

    #[test]
    fn buffer_ids_have_independent_counters_and_release_wraps() {
        let mut counters = C310BufferCounters::new(2, 3).unwrap();
        assert_eq!(counters.reserve_dispatch_slot(0), Ok(true));
        assert_eq!(counters.reserve_dispatch_slot(0), Ok(true));
        assert_eq!(counters.reserve_dispatch_slot(0), Ok(false));
        assert_eq!(counters.reserve_dispatch_slot(1), Ok(true));
        assert_eq!(counters.retire_release(0), Ok(1));
        assert_eq!(counters.reserve_dispatch_slot(0), Ok(true));
        assert_eq!(counters.retire_release(0), Ok(2));
        assert_eq!(counters.retire_release(0), Ok(0));
        assert_eq!(counters.counter(1).unwrap().release_count, 0);
    }

    #[test]
    fn release_tick_is_separate_from_count_and_used_by_get_gate() {
        let mut counters = C310BufferCounters::new(2, 32).unwrap();
        assert_eq!(counters.retire_release(0), Ok(1));
        assert_eq!(counters.counter(0).unwrap().last_release_tick, 0);
        assert_eq!(counters.record_release_tick(0, 7), Ok(()));
        assert_eq!(
            counters.get_buf_gate(0, 1, 7),
            Ok(C310GetBufGateResult::SameTick { tick: 7 })
        );
        assert_eq!(
            counters.get_buf_gate(0, 1, 8),
            Ok(C310GetBufGateResult::Ready)
        );
        assert_eq!(counters.retire_release(0), Ok(2));
        assert_eq!(counters.counter(0).unwrap().last_release_tick, 7);
        assert_eq!(counters.counter(1).unwrap().last_release_tick, 0);
    }

    #[test]
    fn invalid_configuration_and_ids_do_not_mutate_state() {
        assert_eq!(
            C310BufferCounters::new(0, 32),
            Err(C310BufferCounterError::InvalidBufferCount)
        );
        assert_eq!(
            C310BufferCounters::new(65, 32),
            Err(C310BufferCounterError::InvalidBufferCount)
        );
        let mut max_buffers = C310BufferCounters::new(64, 32).unwrap();
        assert_eq!(max_buffers.buffer_count(), 64);
        assert_eq!(max_buffers.reserve_dispatch_slot(63), Ok(true));
        assert_eq!(max_buffers.counter(63).unwrap().dispatch_count, 1);
        assert_eq!(
            max_buffers.counter(64),
            Err(C310BufferCounterError::InvalidBufferId { id: 64, count: 64 })
        );
        assert_eq!(
            C310BufferCounters::new(32, 1),
            Err(C310BufferCounterError::InvalidRingSize)
        );
        let mut counters = C310BufferCounters::new(1, 32).unwrap();
        let original = counters.clone();
        assert_eq!(
            counters.reserve_dispatch_slot(1),
            Err(C310BufferCounterError::InvalidBufferId { id: 1, count: 1 })
        );
        assert_eq!(
            counters.retire_release(1),
            Err(C310BufferCounterError::InvalidBufferId { id: 1, count: 1 })
        );
        assert_eq!(
            counters.consume_get_buf(1),
            Err(C310BufferCounterError::InvalidBufferId { id: 1, count: 1 })
        );
        assert_eq!(
            counters.record_release_tick(1, 7),
            Err(C310BufferCounterError::InvalidBufferId { id: 1, count: 1 })
        );
        assert_eq!(
            counters.get_buf_gate(1, 0, 7),
            Err(C310BufferCounterError::InvalidBufferId { id: 1, count: 1 })
        );
        assert_eq!(counters, original);
    }
}
