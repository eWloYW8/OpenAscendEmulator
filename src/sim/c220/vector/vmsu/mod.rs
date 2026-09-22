mod repeat;
mod trace;
mod writeback;

#[cfg(test)]
mod tests;

pub use trace::*;

use crate::memory::ub::{UbMemory, UbMemoryError};
use crate::sim::c220::memory::C220UbRequestError;
use crate::sim::c220::state::C220State;
use crate::sim::c220::vector::C220VectorError;
use crate::sim::c220::vector::ops::merge::{C220MergeIssue, load_c220_merge_repeat};
use crate::sim::c220::vector::pipeline::C220VectorTimingRules;
use crate::sim::common::scalar::ScalarMachineError;
use repeat::RepeatMachine;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum C220VmsuError {
    #[error("C220 VMSU is already active")]
    Busy,
    #[error("C220 VMSU timeline moved backwards from {previous} to {requested}")]
    TimeReversed { previous: u64, requested: u64 },
    #[error("C220 VMSU timeline computation overflowed")]
    TimeOverflow,
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
struct ActiveVmsu {
    trace: C220VmsuTrace,
    repeat: Option<RepeatMachine>,
    projected_visibility: u64,
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

    /// Earliest completion of the current repeat without competing UB traffic.
    pub fn pending_visibility_tick(&self) -> Option<u64> {
        self.active
            .as_ref()
            .map(|active| active.projected_visibility)
    }

    /// Retry boundary; additional repeats must be rechecked after advancing.
    pub fn pending_drain_tick(&self) -> Option<u64> {
        self.active.as_ref().and_then(|active| {
            active
                .trace
                .retirement_tick
                .or_else(|| active.projected_visibility.checked_add(2))
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
        self.check_time(tick)?;
        if issue.repeat_count() == 0 {
            return Ok(None);
        }
        let admission_tick = tick
            .checked_add(self.rules.dispatch_ticks)
            .ok_or(C220VmsuError::TimeOverflow)?;
        let data = load_c220_merge_repeat(&issue, 0, ub)?;
        let repeat =
            RepeatMachine::new(&issue, data, admission_tick, self.rules.ub_response_ticks)?;
        let projected_visibility = repeat.projected_completion()?;
        let trace = C220VmsuTrace {
            issue,
            admission_tick,
            repeats: vec![repeat.empty_trace(admission_tick)],
            retirement_tick: None,
        };
        self.active = Some(ActiveVmsu {
            trace,
            repeat: Some(repeat),
            projected_visibility,
        });
        self.observed_tick = Some(tick);
        Ok(Some(projected_visibility))
    }

    pub(crate) fn next_event_tick(&self) -> Option<u64> {
        let active = self.active.as_ref()?;
        active
            .repeat
            .as_ref()
            .map(RepeatMachine::next_tick)
            .or(active.trace.retirement_tick)
    }

    pub fn advance_to(&mut self, tick: u64, core: &mut C220State) -> Result<(), C220VmsuError> {
        self.check_time(tick)?;
        while self.next_event_tick().is_some_and(|next| next <= tick) {
            let active = self.active.as_mut().expect("pending VMSU event");
            if let Some(repeat) = active.repeat.as_mut() {
                let event_tick = repeat.next_tick();
                let next_index = active.trace.repeats.len();
                let trace = active
                    .trace
                    .repeats
                    .last_mut()
                    .expect("active repeat trace");
                repeat.commit_writes(event_tick, core.ub_mut(), trace)?;
                if !repeat.is_complete(event_tick) {
                    repeat.step(trace)?;
                    continue;
                }
                if next_index < active.trace.issue.repeat_count() {
                    let data = load_c220_merge_repeat(&active.trace.issue, next_index, core.ub())?;
                    let next = RepeatMachine::new(
                        &active.trace.issue,
                        data,
                        event_tick,
                        self.rules.ub_response_ticks,
                    )?;
                    active.projected_visibility = next.projected_completion()?;
                    active.trace.repeats.push(next.empty_trace(event_tick));
                    active.repeat = Some(next);
                } else {
                    let status = if active.trace.issue.control.exhausted_suspension {
                        pack_u16x4(trace.consumed)
                    } else {
                        0
                    };
                    core.scalar_mut().machine_mut().set_spr_value(17, status)?;
                    active.trace.retirement_tick = Some(
                        event_tick
                            .checked_add(2)
                            .ok_or(C220VmsuError::TimeOverflow)?,
                    );
                    active.repeat = None;
                }
            } else {
                self.last_completed = self.active.take().map(|active| active.trace);
            }
        }
        self.observed_tick = Some(tick);
        Ok(())
    }

    fn check_time(&self, tick: u64) -> Result<(), C220VmsuError> {
        if let Some(previous) = self.observed_tick
            && tick < previous
        {
            return Err(C220VmsuError::TimeReversed {
                previous,
                requested: tick,
            });
        }
        Ok(())
    }
}

const fn pack_u16x4(values: [u16; 4]) -> u64 {
    values[0] as u64
        | ((values[1] as u64) << 16)
        | ((values[2] as u64) << 32)
        | ((values[3] as u64) << 48)
}
