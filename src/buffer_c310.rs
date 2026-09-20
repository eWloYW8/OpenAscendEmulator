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
    pub last_release_tick: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct C310GetBufDispatch {
    pub expected_release_count: u32,
    pub assigned_dispatch_count: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct C310BufferCounters {
    ring_size: u32,
    counters: Vec<C310BufferCounter>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum C310BufferCounterError {
    #[error("buffer count must be between 1 and 64")]
    InvalidBufferCount,
    #[error("buffer ring size must be at least 2")]
    InvalidRingSize,
    #[error("buffer ID {id} is outside configured range 0..{count}")]
    InvalidBufferId { id: u8, count: u8 },
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
        if self.is_full(id)? {
            return Ok(None);
        }
        let expected_release_count = self.counter(id)?.dispatch_count;
        let next = self.next(expected_release_count);
        let counter = &mut self.counters[usize::from(id)];
        counter.dispatch_count = next;
        counter.total_dispatches = counter.total_dispatches.wrapping_add(1);
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
