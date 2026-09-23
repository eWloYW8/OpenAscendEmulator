use std::collections::VecDeque;
use std::iter::Peekable;

use super::{C220FixpReadGenerator, C220FixpReadUop};
use crate::sim::c220::mte::interface::{C220MteL0cReadError, C220MteL0cReadInterface};

mod events;
pub use events::{C220FixpReadEventOutcome, C220FixpReadEvents};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FixpReadEntry {
    pub uop: C220FixpReadUop,
    pub ready_tick: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220FixpReadProgress {
    Idle,
    Delayed { ready_tick: u64 },
    QueueFull,
    HardwareSync,
    Advanced(C220FixpReadUop),
}

#[derive(Debug, thiserror::Error)]
pub enum C220FixpReadPipelineError {
    #[error("FIX read time reversed from {previous} to {requested}")]
    TimeReversed { previous: u64, requested: u64 },
    #[error("FIX read {phase} callback repeated at tick {tick}")]
    RepeatedCallback { phase: &'static str, tick: u64 },
    #[error("FIX read queue time overflow")]
    TimeOverflow,
    #[error(transparent)]
    Interface(#[from] C220MteL0cReadError),
}

#[derive(Debug, Clone)]
struct Batch {
    packets: Peekable<C220FixpReadGenerator>,
    ready_tick: u64,
}

/// Read-side generation and dispatch. Instruction admission/retirement and
/// downstream response processing remain owned by the surrounding scheduler.
#[derive(Debug, Clone, Default)]
pub struct C220FixpReadPipeline {
    generated: VecDeque<Batch>,
    dispatch: VecDeque<C220FixpReadEntry>,
    observed_tick: Option<u64>,
    callbacks: [Option<u64>; 2],
}

impl C220FixpReadPipeline {
    pub fn is_idle(&self) -> bool {
        self.generated.is_empty() && self.dispatch.is_empty()
    }
    pub fn dispatch_queue(&self) -> &VecDeque<C220FixpReadEntry> {
        &self.dispatch
    }
    pub fn generated_batches(&self) -> usize {
        self.generated.len()
    }

    pub fn generated_ready_tick(&self) -> Option<u64> {
        self.generated.front().map(|batch| batch.ready_tick)
    }

    /// Store a lazily expanded instruction batch with the generation timestamp.
    /// Returns false for zero-uop commands; their owner retires them directly.
    pub fn submit(
        &mut self,
        tick: u64,
        packets: C220FixpReadGenerator,
    ) -> Result<bool, C220FixpReadPipelineError> {
        self.check_time(tick)?;
        let mut packets = packets.peekable();
        if packets.peek().is_none() {
            return Ok(false);
        }
        let ready_tick = tick
            .checked_add(2)
            .ok_or(C220FixpReadPipelineError::TimeOverflow)?;
        self.generated.push_back(Batch {
            packets,
            ready_tick,
        });
        Ok(true)
    }

    pub fn generate(
        &mut self,
        tick: u64,
    ) -> Result<C220FixpReadProgress, C220FixpReadPipelineError> {
        self.begin(tick, 0, "generate")?;
        let Some(batch) = self.generated.front_mut() else {
            return Ok(C220FixpReadProgress::Idle);
        };
        if tick < batch.ready_tick {
            return Ok(C220FixpReadProgress::Delayed {
                ready_tick: batch.ready_tick,
            });
        }
        if self.dispatch.len() >= 7 {
            return Ok(C220FixpReadProgress::QueueFull);
        }
        let ready_tick = tick
            .checked_add(3)
            .ok_or(C220FixpReadPipelineError::TimeOverflow)?;
        let uop = batch
            .packets
            .next()
            .expect("only nonempty batches are retained");
        let finished = batch.packets.peek().is_none();
        self.dispatch
            .push_back(C220FixpReadEntry { uop, ready_tick });
        if finished {
            self.generated.pop_front();
        }
        Ok(C220FixpReadProgress::Advanced(uop))
    }

    pub fn send(
        &mut self,
        tick: u64,
        input: &mut C220MteL0cReadInterface,
        hardware_sync_blocked: bool,
    ) -> Result<C220FixpReadProgress, C220FixpReadPipelineError> {
        self.begin(tick, 1, "send")?;
        let Some(entry) = self.dispatch.front().copied() else {
            return Ok(C220FixpReadProgress::Idle);
        };
        if tick < entry.ready_tick {
            return Ok(C220FixpReadProgress::Delayed {
                ready_tick: entry.ready_tick,
            });
        }
        if !input.can_enqueue() {
            return Ok(C220FixpReadProgress::QueueFull);
        }
        if hardware_sync_blocked {
            return Ok(C220FixpReadProgress::HardwareSync);
        }
        if !input.enqueue(tick, entry.uop.operation)? {
            return Ok(C220FixpReadProgress::QueueFull);
        }
        self.dispatch.pop_front();
        Ok(C220FixpReadProgress::Advanced(entry.uop))
    }

    fn check_time(&mut self, tick: u64) -> Result<(), C220FixpReadPipelineError> {
        if let Some(previous) = self.observed_tick
            && tick < previous
        {
            return Err(C220FixpReadPipelineError::TimeReversed {
                previous,
                requested: tick,
            });
        }
        self.observed_tick = Some(tick);
        Ok(())
    }

    fn begin(
        &mut self,
        tick: u64,
        index: usize,
        phase: &'static str,
    ) -> Result<(), C220FixpReadPipelineError> {
        self.check_time(tick)?;
        if self.callbacks[index] == Some(tick) {
            return Err(C220FixpReadPipelineError::RepeatedCallback { phase, tick });
        }
        self.callbacks[index] = Some(tick);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::c220::mte::fixp::C220FixpDescriptor;
    use crate::sim::c220::memory::C220L0c;
    use crate::sim::c220::mte::fixp::C220FixpFp16Command;
    use crate::sim::c220::mte::interface::C220MteL0cReadSend;
    use crate::sim::common::event::EventDispatcher;

    #[test]
    fn clock_events_retry_backpressure_without_dropping_or_duplicating_packets() {
        let command = C220FixpFp16Command {
            descriptor: C220FixpDescriptor {
                xt: (64 << 16) | (16 << 4),
                xm: 1 << 34,
                nd: 0,
            },
            source_address: 0,
            destination_address: 0,
            control: 0,
            scalar_slope: 0,
            slope_base_block: 0,
        };
        let mut events = EventDispatcher::new(0);
        let clock = events.add_event();
        let binding = C220FixpReadEvents::register(&mut events, clock, |phase| phase);
        let mut pipeline = C220FixpReadPipeline::default();
        let mut input = C220MteL0cReadInterface::new(32, 0).unwrap();
        let packets = C220FixpReadGenerator::new(command, 1, 1, 256).unwrap();
        let expected: Vec<_> = packets.clone().collect();
        binding.submit(&mut events, &mut pipeline, packets).unwrap();
        let mut generated = Vec::new();
        let mut sent = Vec::new();
        for tick in 0..40 {
            events.advance_to(tick).unwrap();
            events.notify_at(clock, tick);
            while let Some(invocation) = events.next_callback() {
                match binding
                    .handle(
                        invocation.callback,
                        &mut events,
                        &mut pipeline,
                        &mut input,
                        tick < 8,
                    )
                    .unwrap()
                {
                    C220FixpReadEventOutcome::Generated(C220FixpReadProgress::Advanced(uop)) => {
                        generated.push(uop);
                    }
                    C220FixpReadEventOutcome::Sent(C220FixpReadProgress::Advanced(uop)) => {
                        sent.push((tick, uop));
                    }
                    _ => {}
                }
            }
        }
        assert_eq!(generated, expected[..12]);
        assert_eq!(
            sent.iter().map(|&(tick, _)| tick).collect::<Vec<_>>(),
            [8, 9, 10, 11, 12]
        );
        assert_eq!(
            sent.iter().map(|&(_, uop)| uop).collect::<Vec<_>>(),
            expected[..5]
        );
        assert_eq!(pipeline.dispatch_queue().len(), 7);
        assert_eq!(pipeline.dispatch_queue()[0].uop, expected[5]);
        assert_eq!(pipeline.generated_batches(), 1);
        assert!(!input.can_enqueue());
    }

    #[test]
    fn queue_delays_and_sync_blocking_preserve_the_generated_head() {
        let command = C220FixpFp16Command {
            descriptor: C220FixpDescriptor {
                xt: (1 << 16) | (16 << 4),
                xm: 1 << 34,
                nd: 0,
            },
            source_address: 0,
            destination_address: 0,
            control: 0,
            scalar_slope: 0,
            slope_base_block: 0,
        };
        let mut pipeline = C220FixpReadPipeline::default();
        let mut input = C220MteL0cReadInterface::new(32, 0).unwrap();
        let mut memory = C220L0c::new(131072, 12).unwrap();
        assert!(
            pipeline
                .submit(0, C220FixpReadGenerator::new(command, 1, 1, 256).unwrap())
                .unwrap()
        );
        assert_eq!(
            pipeline.generate(1).unwrap(),
            C220FixpReadProgress::Delayed { ready_tick: 2 }
        );
        assert!(matches!(
            pipeline.generate(2).unwrap(),
            C220FixpReadProgress::Advanced(_)
        ));
        assert_eq!(
            pipeline.send(4, &mut input, false).unwrap(),
            C220FixpReadProgress::Delayed { ready_tick: 5 }
        );
        assert_eq!(
            pipeline.send(5, &mut input, true).unwrap(),
            C220FixpReadProgress::HardwareSync
        );
        assert_eq!(pipeline.dispatch_queue().len(), 1);
        assert!(matches!(
            pipeline.send(6, &mut input, false).unwrap(),
            C220FixpReadProgress::Advanced(_)
        ));
        assert!(pipeline.is_idle());
        assert_eq!(
            input.send_queued(9, &mut memory).unwrap(),
            C220MteL0cReadSend::InputLatency { ready_tick: 10 }
        );
        assert_eq!(
            input.send_queued(10, &mut memory).unwrap(),
            C220MteL0cReadSend::Sent
        );
    }
}
