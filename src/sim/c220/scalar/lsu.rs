use std::collections::VecDeque;

use thiserror::Error;

pub mod cache;
pub mod direct_store;
pub mod miss_buffer;
pub mod read_queue;
pub mod scheduler;
pub mod store_buffer;
pub mod write_queue;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220LsuStage {
    M0,
    M1,
    M2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct C220LsuRequestId(u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220LsuStageEntry {
    pub request: C220LsuRequestId,
    pub ready_tick: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220LsuStageProgress {
    Idle,
    Advanced(C220LsuRequestId),
    Blocked(C220LsuRequestId),
    ReplayScheduled(C220LsuRequestId),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220LsuAdmissionBlock {
    Pipeline,
    RequestCapacity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum C220LsuPipelineError {
    #[error("LSU request capacity must be at least two")]
    InvalidCapacity,
    #[error("LSU clock moved backwards from {previous} to {requested}")]
    TimeReversal { previous: u64, requested: u64 },
    #[error("LSU clock or request sequence overflow")]
    Overflow,
}

/// Request transport through tag lookup and the two data-cache stages.
///
/// Cache hazards are evaluated by the caller at each stage. Leaving M2 releases
/// request capacity, not the destination register or a miss-table entry.
/// Stage callbacks are explicit so the event scheduler controls same-tick order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220LsuRequestPipeline {
    capacity: u32,
    tick: u64,
    admitted: u64,
    issued: u64,
    consumed: u64,
    admission_blocked: bool,
    last_evaluation: [Option<u64>; 3],
    m0_ready: Option<u64>,
    m1: VecDeque<C220LsuStageEntry>,
    m2: VecDeque<C220LsuStageEntry>,
}

impl C220LsuRequestPipeline {
    pub fn new(capacity: u32) -> Result<Self, C220LsuPipelineError> {
        if capacity < 2 {
            return Err(C220LsuPipelineError::InvalidCapacity);
        }
        Ok(Self {
            capacity,
            tick: 0,
            admitted: 0,
            issued: 0,
            consumed: 0,
            admission_blocked: false,
            last_evaluation: [None; 3],
            m0_ready: None,
            m1: VecDeque::new(),
            m2: VecDeque::new(),
        })
    }

    pub const fn issued_occupancy(&self) -> u64 {
        self.issued - self.consumed
    }

    pub const fn queued_requests(&self) -> u64 {
        self.admitted - self.consumed
    }

    pub fn admission_block(&self, split_pair: bool) -> Option<C220LsuAdmissionBlock> {
        let limit = self.capacity - if split_pair { 2 } else { 0 };
        if self.issued_occupancy() >= u64::from(limit) {
            return Some(C220LsuAdmissionBlock::RequestCapacity);
        }
        if self.admission_blocked {
            return Some(C220LsuAdmissionBlock::Pipeline);
        }
        None
    }

    /// Returns the request handles to associate with the caller's payloads.
    /// Split-pair admission is atomic; neither half is queued on rejection.
    pub fn admit(
        &mut self,
        tick: u64,
        split_pair: bool,
    ) -> Result<Option<(C220LsuRequestId, Option<C220LsuRequestId>)>, C220LsuPipelineError> {
        self.check_tick(tick)?;
        if self.admission_block(split_pair).is_some() {
            return Ok(None);
        }
        let ready = tick.checked_add(1).ok_or(C220LsuPipelineError::Overflow)?;
        let count = if split_pair { 2 } else { 1 };
        let admitted = self
            .admitted
            .checked_add(count)
            .ok_or(C220LsuPipelineError::Overflow)?;
        let first = C220LsuRequestId(self.admitted);
        let second = split_pair.then_some(C220LsuRequestId(admitted - 1));
        self.admitted = admitted;
        self.tick = tick;
        self.schedule_m0(ready);
        Ok(Some((first, second)))
    }

    pub fn head(&self, stage: C220LsuStage) -> Option<C220LsuStageEntry> {
        match stage {
            C220LsuStage::M0 => {
                self.m0_ready
                    .filter(|_| self.issued < self.admitted)
                    .map(|ready_tick| C220LsuStageEntry {
                        request: C220LsuRequestId(self.issued),
                        ready_tick,
                    })
            }
            C220LsuStage::M1 => self.m1.front().copied(),
            C220LsuStage::M2 => self.m2.front().copied(),
        }
    }

    /// Earliest stage eligibility, not a replacement for event delivery order.
    pub fn next_ready_tick(&self) -> Option<u64> {
        [C220LsuStage::M0, C220LsuStage::M1, C220LsuStage::M2]
            .into_iter()
            .filter_map(|stage| {
                self.head(stage).and_then(|entry| {
                    let earliest = match self.last_evaluation[stage as usize] {
                        Some(tick) => tick.checked_add(1)?,
                        None => 0,
                    };
                    Some(entry.ready_tick.max(earliest).max(self.tick))
                })
            })
            .min()
    }

    /// `blocked` includes cache hazards and an active cache-maintenance request.
    /// `draining` means this request is waiting for earlier work to drain;
    /// unlike a hazard, it does not close admission.
    pub fn advance_m0(
        &mut self,
        tick: u64,
        blocked: bool,
        draining: bool,
    ) -> Result<C220LsuStageProgress, C220LsuPipelineError> {
        self.check_tick(tick)?;
        if self.last_evaluation[C220LsuStage::M0 as usize] == Some(tick) {
            return Ok(C220LsuStageProgress::Idle);
        }
        let Some(entry) = self
            .head(C220LsuStage::M0)
            .filter(|entry| entry.ready_tick <= tick)
        else {
            self.tick = tick;
            return Ok(C220LsuStageProgress::Idle);
        };
        let next = tick.checked_add(1).ok_or(C220LsuPipelineError::Overflow)?;
        self.tick = tick;
        self.last_evaluation[C220LsuStage::M0 as usize] = Some(tick);
        self.admission_blocked = blocked;
        if blocked || draining {
            self.m0_ready = Some(next);
            return Ok(C220LsuStageProgress::Blocked(entry.request));
        }
        self.m1.push_back(C220LsuStageEntry {
            request: entry.request,
            ready_tick: next,
        });
        self.issued += 1;
        self.m0_ready = (self.issued < self.admitted).then_some(next);
        Ok(C220LsuStageProgress::Advanced(entry.request))
    }

    pub fn advance_m1(
        &mut self,
        tick: u64,
        blocked: bool,
    ) -> Result<C220LsuStageProgress, C220LsuPipelineError> {
        self.advance_data_stage(C220LsuStage::M1, tick, blocked)
    }

    pub fn advance_m2(
        &mut self,
        tick: u64,
        blocked: bool,
    ) -> Result<C220LsuStageProgress, C220LsuPipelineError> {
        self.advance_data_stage(C220LsuStage::M2, tick, blocked)
    }

    fn advance_data_stage(
        &mut self,
        stage: C220LsuStage,
        tick: u64,
        blocked: bool,
    ) -> Result<C220LsuStageProgress, C220LsuPipelineError> {
        self.check_tick(tick)?;
        if self.last_evaluation[stage as usize] == Some(tick) {
            return Ok(C220LsuStageProgress::Idle);
        }
        let Some(entry) = self.head(stage).filter(|entry| entry.ready_tick <= tick) else {
            self.tick = tick;
            return Ok(C220LsuStageProgress::Idle);
        };
        let next = if blocked || stage == C220LsuStage::M1 {
            tick.checked_add(1).ok_or(C220LsuPipelineError::Overflow)?
        } else {
            tick
        };
        self.tick = tick;
        self.last_evaluation[stage as usize] = Some(tick);
        if blocked {
            self.m1.clear();
            if stage == C220LsuStage::M2 {
                self.m2.clear();
            }
            self.issued = entry.request.0;
            self.schedule_m0(next);
            return Ok(C220LsuStageProgress::ReplayScheduled(entry.request));
        }
        if stage == C220LsuStage::M1 {
            self.m1.pop_front();
            self.m2.push_back(C220LsuStageEntry {
                request: entry.request,
                ready_tick: next,
            });
        } else {
            self.m2.pop_front();
            debug_assert_eq!(entry.request.0, self.consumed);
            self.consumed += 1;
        }
        Ok(C220LsuStageProgress::Advanced(entry.request))
    }

    fn check_tick(&self, tick: u64) -> Result<(), C220LsuPipelineError> {
        if tick < self.tick {
            Err(C220LsuPipelineError::TimeReversal {
                previous: self.tick,
                requested: tick,
            })
        } else {
            Ok(())
        }
    }

    fn schedule_m0(&mut self, tick: u64) {
        self.m0_ready = Some(self.m0_ready.map_or(tick, |prior| prior.min(tick)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stages_reissue_without_releasing_or_duplicating_requests() {
        let mut pipeline = C220LsuRequestPipeline::new(4).unwrap();
        let (first, second) = pipeline.admit(0, true).unwrap().unwrap();
        assert_eq!(pipeline.issued_occupancy(), 0);
        assert_eq!(pipeline.queued_requests(), 2);
        assert_eq!(
            pipeline.advance_m0(0, false, false).unwrap(),
            C220LsuStageProgress::Idle
        );
        pipeline.advance_m0(1, false, false).unwrap();
        assert_eq!(
            pipeline.advance_m0(1, false, false).unwrap(),
            C220LsuStageProgress::Idle
        );
        pipeline.advance_m1(2, false).unwrap();
        pipeline.advance_m0(2, false, false).unwrap();
        assert_eq!(pipeline.issued_occupancy(), 2);
        assert_eq!(
            pipeline.admission_block(true),
            Some(C220LsuAdmissionBlock::RequestCapacity)
        );
        assert_eq!(
            pipeline.advance_m1(3, true).unwrap(),
            C220LsuStageProgress::ReplayScheduled(second.unwrap())
        );
        assert_eq!(pipeline.head(C220LsuStage::M2).unwrap().request, first);
        assert_eq!(pipeline.issued_occupancy(), 1);
        assert_eq!(
            pipeline.advance_m2(3, true).unwrap(),
            C220LsuStageProgress::ReplayScheduled(first)
        );
        assert_eq!(pipeline.issued_occupancy(), 0);
        assert_eq!(pipeline.queued_requests(), 2);
        assert_eq!(pipeline.next_ready_tick(), Some(4));
        assert_eq!(
            pipeline.advance_m2(3, false).unwrap(),
            C220LsuStageProgress::Idle
        );
        for tick in 4..=7 {
            pipeline.advance_m2(tick, false).unwrap();
            pipeline.advance_m1(tick, false).unwrap();
            pipeline.advance_m0(tick, false, false).unwrap();
        }
        assert_eq!(pipeline.queued_requests(), 0);
        assert_eq!(pipeline.issued_occupancy(), 0);
        assert_eq!(pipeline.next_ready_tick(), None);
    }

    #[test]
    fn admission_distinguishes_queued_issued_and_blocked_state() {
        let mut pipeline = C220LsuRequestPipeline::new(4).unwrap();
        for _ in 0..8 {
            assert!(pipeline.admit(0, false).unwrap().is_some());
        }
        assert_eq!(pipeline.queued_requests(), 8);
        assert_eq!(pipeline.issued_occupancy(), 0);
        pipeline.advance_m0(1, false, true).unwrap();
        assert_eq!(pipeline.admission_block(false), None);
        pipeline.advance_m0(2, true, false).unwrap();
        assert_eq!(
            pipeline.admission_block(false),
            Some(C220LsuAdmissionBlock::Pipeline)
        );
        assert!(pipeline.admit(2, true).unwrap().is_none());
        for tick in 3..7 {
            pipeline.advance_m0(tick, false, false).unwrap();
        }
        assert_eq!(pipeline.issued_occupancy(), 4);
        assert_eq!(
            pipeline.admission_block(false),
            Some(C220LsuAdmissionBlock::RequestCapacity)
        );
        let before = pipeline.clone();
        assert!(matches!(
            pipeline.advance_m0(u64::MAX, false, false),
            Err(C220LsuPipelineError::Overflow)
        ));
        assert_eq!(pipeline, before);
        assert!(matches!(
            pipeline.advance_m0(0, false, false),
            Err(C220LsuPipelineError::TimeReversal { .. })
        ));
        assert_eq!(pipeline, before);
    }
}
