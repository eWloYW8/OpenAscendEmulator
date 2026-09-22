use std::collections::VecDeque;

use super::{C220Set2dBandwidths, C220Set2dOutputRoute, C220Set2dUop, C220Set2dUops};
use crate::isa::c220::mte::set2d::C220Set2dFill;
use crate::sim::c220::mte::interface::{
    C220L0WriteError, C220L0WritePipeline, C220MteL1WriteError, C220MteL1WriteInterface,
};

const COMMAND_TICKS: u64 = 1;
const GENERATED_TICKS: u64 = 3;
const GENERATED_CAPACITY: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Set2dGenerated {
    pub instruction_id: u64,
    pub uop_index: u64,
    pub ready_tick: u64,
    pub uop: C220Set2dUop,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Set2dIssue {
    pub tick: u64,
    pub instruction_id: u64,
    pub uop_count: u64,
    /// No output acknowledgment is needed. The command owner can retire it.
    pub completion_ready: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Set2dQueues {
    pub instruction_uops: u64,
    pub generated: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct C220Set2dGates {
    /// Applies to L0 fills only, for the currently offered instruction.
    pub hardware_sync_blocked: bool,
    /// Applies to L1 fills only, for the currently offered instruction.
    pub l1_prefetch_blocked: bool,
}

pub struct C220Set2dOutputs<'a> {
    pub l0a: &'a mut C220L0WritePipeline,
    pub l0b: &'a mut C220L0WritePipeline,
    pub l1: &'a mut C220MteL1WriteInterface,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220Set2dStall {
    NotReady,
    HardwareFlag,
    Prefetch,
    OutputFull,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Set2dSend {
    pub tick: u64,
    pub offered: Option<C220Set2dGenerated>,
    pub stall: Option<C220Set2dStall>,
    pub sent: Option<C220Set2dGenerated>,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum C220Set2dFrontendError {
    #[error("SET_2D generator cannot accept a command")]
    CommandBusy,
    #[error("SET_2D time reversed from {previous} to {requested}")]
    TimeReversed { previous: u64, requested: u64 },
    #[error("SET_2D {phase} callback already ran at tick {tick}")]
    RepeatedCallback { phase: &'static str, tick: u64 },
    #[error("SET_2D time overflowed")]
    TimeOverflow,
    #[error(transparent)]
    L0(#[from] C220L0WriteError),
    #[error(transparent)]
    L1(#[from] C220MteL1WriteError),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Generation {
    instruction_id: u64,
    ready_tick: u64,
    next_index: u64,
    plan: C220Set2dUops,
}

/// One SET_2D generation engine shared by its three destinations. Generation
/// and sending are independent callbacks, each limited to once per tick.
/// Commands expand lazily, and blocked output keeps the exact queue head.
/// Idleness means all uops were sent, not that writes or commands have retired.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220Set2dFrontend {
    bandwidths: C220Set2dBandwidths,
    generation: Option<Generation>,
    generated: VecDeque<C220Set2dGenerated>,
    observed_tick: Option<u64>,
    generation_tick: Option<u64>,
    send_tick: Option<u64>,
}

impl C220Set2dFrontend {
    pub fn new(bandwidths: C220Set2dBandwidths) -> Self {
        Self {
            bandwidths,
            generation: None,
            generated: VecDeque::new(),
            observed_tick: None,
            generation_tick: None,
            send_tick: None,
        }
    }

    pub fn is_idle(&self) -> bool {
        self.generation.is_none() && self.generated.is_empty()
    }

    pub fn can_issue(&self) -> bool {
        self.generation.is_none() && self.generated.len() < GENERATED_CAPACITY
    }

    pub fn generated(&self) -> &VecDeque<C220Set2dGenerated> {
        &self.generated
    }

    pub fn queue_state(&self) -> C220Set2dQueues {
        C220Set2dQueues {
            instruction_uops: self.generation.as_ref().map_or(0, |g| g.plan.remaining()),
            generated: self.generated.len(),
        }
    }

    pub(super) fn instruction_ready_tick(&self) -> Option<u64> {
        self.generation
            .as_ref()
            .map(|generation| generation.ready_tick)
    }

    pub fn issue(
        &mut self,
        tick: u64,
        instruction_id: u64,
        fill: C220Set2dFill,
    ) -> Result<C220Set2dIssue, C220Set2dFrontendError> {
        self.check_time(tick)?;
        if !self.can_issue() {
            return Err(C220Set2dFrontendError::CommandBusy);
        }
        let plan = C220Set2dUops::new(fill, self.bandwidths);
        let uop_count = plan.remaining();
        if uop_count != 0 {
            let ready_tick = tick
                .checked_add(COMMAND_TICKS)
                .ok_or(C220Set2dFrontendError::TimeOverflow)?;
            self.generation = Some(Generation {
                instruction_id,
                ready_tick,
                next_index: 0,
                plan,
            });
        }
        self.observed_tick = Some(tick);
        Ok(C220Set2dIssue {
            tick,
            instruction_id,
            uop_count,
            completion_ready: uop_count == 0,
        })
    }

    pub fn generate(
        &mut self,
        tick: u64,
    ) -> Result<Option<C220Set2dGenerated>, C220Set2dFrontendError> {
        self.check_callback(tick, self.generation_tick, "generation")?;
        let eligible = self
            .generation
            .as_ref()
            .is_some_and(|g| g.ready_tick <= tick)
            && self.generated.len() < GENERATED_CAPACITY;
        let generated = if eligible {
            let ready_tick = tick
                .checked_add(GENERATED_TICKS)
                .ok_or(C220Set2dFrontendError::TimeOverflow)?;
            let generation = self.generation.as_mut().expect("eligible instruction");
            let entry = C220Set2dGenerated {
                instruction_id: generation.instruction_id,
                uop_index: generation.next_index,
                ready_tick,
                uop: generation.plan.next().expect("nonempty plan"),
            };
            generation.next_index += 1;
            self.generated.push_back(entry);
            if generation.plan.remaining() == 0 {
                self.generation = None;
            }
            Some(entry)
        } else {
            None
        };
        self.generation_tick = Some(tick);
        self.observed_tick = Some(tick);
        Ok(generated)
    }

    pub fn send(
        &mut self,
        tick: u64,
        gates: C220Set2dGates,
        outputs: C220Set2dOutputs<'_>,
    ) -> Result<C220Set2dSend, C220Set2dFrontendError> {
        self.check_callback(tick, self.send_tick, "send")?;
        let offered = self.generated.front().copied();
        let mut stall = None;
        let mut sent = None;
        if let Some(head) = offered {
            use C220Set2dOutputRoute::{L0a, L0b, L1};
            stall = if head.ready_tick > tick {
                Some(C220Set2dStall::NotReady)
            } else if matches!(head.uop.route, L0a(_) | L0b(_)) && gates.hardware_sync_blocked {
                Some(C220Set2dStall::HardwareFlag)
            } else if matches!(head.uop.route, L1(_)) && gates.l1_prefetch_blocked {
                Some(C220Set2dStall::Prefetch)
            } else {
                None
            };
            if stall.is_none() {
                let fragment = head
                    .uop
                    .output_fragment(head.instruction_id, head.uop_index);
                let accepted = match head.uop.route {
                    L0a(port) => outputs.l0a.push(tick, port, fragment)?,
                    L0b(port) => outputs.l0b.push(tick, port, fragment)?,
                    L1(port) => outputs.l1.push(tick, port, fragment)?,
                };
                if accepted {
                    sent = self.generated.pop_front();
                } else {
                    stall = Some(C220Set2dStall::OutputFull);
                }
            }
        }
        self.send_tick = Some(tick);
        self.observed_tick = Some(tick);
        Ok(C220Set2dSend {
            tick,
            offered,
            stall,
            sent,
        })
    }

    fn check_time(&self, tick: u64) -> Result<(), C220Set2dFrontendError> {
        if let Some(previous) = self.observed_tick
            && tick < previous
        {
            return Err(C220Set2dFrontendError::TimeReversed {
                previous,
                requested: tick,
            });
        }
        Ok(())
    }

    fn check_callback(
        &self,
        tick: u64,
        last: Option<u64>,
        phase: &'static str,
    ) -> Result<(), C220Set2dFrontendError> {
        self.check_time(tick)?;
        if last == Some(tick) {
            return Err(C220Set2dFrontendError::RepeatedCallback { phase, tick });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
