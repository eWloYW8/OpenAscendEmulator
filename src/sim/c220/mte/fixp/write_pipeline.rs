use std::collections::VecDeque;

use super::{
    C220FixpL1Output, C220FixpL1OutputError, C220FixpL1WriteError, C220FixpL1WriteInterface,
};
use crate::sim::c220::mte::interface::C220MteOutputFragment;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FixpWriteEntry {
    pub fragment: C220MteOutputFragment,
    pub ready_tick: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220FixpWriteProgress {
    Idle,
    Delayed { ready_tick: u64 },
    QueueFull,
    Advanced(C220MteOutputFragment),
}

#[derive(Debug, thiserror::Error)]
pub enum C220FixpWritePipelineError {
    #[error("FIX write time reversed from {previous} to {requested}")]
    TimeReversed { previous: u64, requested: u64 },
    #[error("FIX write {phase} callback repeated at tick {tick}")]
    RepeatedCallback { phase: &'static str, tick: u64 },
    #[error("FIX write queue time overflowed")]
    Overflow,
    #[error(transparent)]
    Output(#[from] C220FixpL1OutputError),
    #[error(transparent)]
    Interface(#[from] C220FixpL1WriteError),
}

/// Ordinary FIX-to-L1 write generation. Packetization, bounded dispatch, and
/// interface enqueue are separate callbacks; the shared scheduler owns their
/// ordering relative to conversion, memory service, and retirement.
#[derive(Debug, Clone, Default)]
pub struct C220FixpWritePipeline {
    packets: VecDeque<C220FixpWriteEntry>,
    dispatch: VecDeque<C220FixpWriteEntry>,
    observed_tick: Option<u64>,
    callbacks: [Option<u64>; 3],
}

impl C220FixpWritePipeline {
    pub fn packets(&self) -> &VecDeque<C220FixpWriteEntry> {
        &self.packets
    }

    pub fn dispatch_queue(&self) -> &VecDeque<C220FixpWriteEntry> {
        &self.dispatch
    }

    pub fn is_idle(&self) -> bool {
        self.packets.is_empty() && self.dispatch.is_empty()
    }

    pub fn packetize(
        &mut self,
        tick: u64,
        output: &mut C220FixpL1Output,
    ) -> Result<Option<C220MteOutputFragment>, C220FixpWritePipelineError> {
        self.begin(tick, 0, "packetize")?;
        let ready_tick = tick
            .checked_add(1)
            .ok_or(C220FixpWritePipelineError::Overflow)?;
        let fragment = output.take_write(tick, true)?;
        if let Some(fragment) = fragment {
            self.packets.push_back(C220FixpWriteEntry {
                fragment,
                ready_tick,
            });
        }
        Ok(fragment)
    }

    pub fn generate(
        &mut self,
        tick: u64,
    ) -> Result<C220FixpWriteProgress, C220FixpWritePipelineError> {
        self.begin(tick, 1, "generate")?;
        let Some(head) = self.packets.front().copied() else {
            return Ok(C220FixpWriteProgress::Idle);
        };
        if head.ready_tick > tick {
            return Ok(C220FixpWriteProgress::Delayed {
                ready_tick: head.ready_tick,
            });
        }
        if self.dispatch.len() >= 6 {
            return Ok(C220FixpWriteProgress::QueueFull);
        }
        let ready_tick = tick
            .checked_add(1)
            .ok_or(C220FixpWritePipelineError::Overflow)?;
        self.dispatch.push_back(C220FixpWriteEntry {
            fragment: head.fragment,
            ready_tick,
        });
        self.packets.pop_front();
        Ok(C220FixpWriteProgress::Advanced(head.fragment))
    }

    pub fn send(
        &mut self,
        tick: u64,
        interface: &mut C220FixpL1WriteInterface,
    ) -> Result<C220FixpWriteProgress, C220FixpWritePipelineError> {
        self.begin(tick, 2, "send")?;
        let Some(head) = self.dispatch.front().copied() else {
            return Ok(C220FixpWriteProgress::Idle);
        };
        if head.ready_tick > tick {
            return Ok(C220FixpWriteProgress::Delayed {
                ready_tick: head.ready_tick,
            });
        }
        interface.enqueue(tick, head.fragment)?;
        self.dispatch.pop_front();
        Ok(C220FixpWriteProgress::Advanced(head.fragment))
    }

    fn begin(
        &mut self,
        tick: u64,
        index: usize,
        phase: &'static str,
    ) -> Result<(), C220FixpWritePipelineError> {
        if let Some(previous) = self.observed_tick
            && tick < previous
        {
            return Err(C220FixpWritePipelineError::TimeReversed {
                previous,
                requested: tick,
            });
        }
        if self.callbacks[index] == Some(tick) {
            return Err(C220FixpWritePipelineError::RepeatedCallback { phase, tick });
        }
        self.observed_tick = Some(tick);
        self.callbacks[index] = Some(tick);
        Ok(())
    }
}
