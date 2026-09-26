use std::collections::VecDeque;

use super::generator::{C220MteGeneratorCallback, GeneratorEvents};
use super::uop::{C220DmaDestinationLayout, C220DmaUopMode, C220DmaUopRequest, C220DmaUops};
use crate::sim::common::event::{EventDispatcher, EventId};

const COMMAND_TICKS: u64 = 1;
mod requests;
use requests::Requests;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220DmaGenerated {
    pub instruction_id: u64,
    pub uop_index: u64,
    pub ready_tick: u64,
    pub request: C220DmaUopRequest,
    pub destination: C220DmaDestinationLayout,
    pub mode: C220DmaUopMode,
    pub out_of_order: bool,
    pub last_in_instruction: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220DmaIssue {
    pub tick: u64,
    pub instruction_id: u64,
    pub completion_ready: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220DmaStall {
    NotReady,
    HardwareFlag,
    OutputFull,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220DmaSend {
    pub tick: u64,
    pub offered: Option<C220DmaGenerated>,
    pub stall: Option<C220DmaStall>,
    pub sent: Option<C220DmaGenerated>,
}

#[derive(Debug, thiserror::Error)]
pub enum C220DmaFrontendError {
    #[error("request stream does not belong to this DMA generation engine")]
    WrongGenerator,
    #[error("DMA generator cannot accept another command")]
    CommandBusy,
    #[error("DMA generator time reversed from {previous} to {requested}")]
    TimeReversed { previous: u64, requested: u64 },
    #[error("DMA {phase} callback already ran at tick {tick}")]
    RepeatedCallback { phase: &'static str, tick: u64 },
    #[error("DMA generator time overflowed")]
    TimeOverflow,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Generation {
    instruction_id: u64,
    ready_tick: u64,
    next_index: u64,
    requests: Requests,
}

/// Ordinary DMA or external LOAD2D generation, using independent instances.
/// Sending the final request permits generator switching but never retires
/// the command. The consumer supplies output credit and synchronization gates.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct C220DmaFrontend {
    load2d: bool,
    generation: Option<Generation>,
    generated: VecDeque<C220DmaGenerated>,
    observed_tick: Option<u64>,
    generation_tick: Option<u64>,
    send_tick: Option<u64>,
}

impl C220DmaFrontend {
    pub fn external_load2d() -> Self {
        Self {
            load2d: true,
            ..Self::default()
        }
    }

    fn generated_ticks(&self) -> u64 {
        if self.load2d { 2 } else { 3 }
    }

    fn generated_capacity(&self) -> usize {
        self.generated_ticks() as usize + 1
    }
    pub fn is_idle(&self) -> bool {
        self.generation.is_none() && self.generated.is_empty()
    }

    pub fn can_issue(&self) -> bool {
        self.generation.is_none() && self.generated.len() < self.generated_capacity()
    }

    pub fn generated(&self) -> &VecDeque<C220DmaGenerated> {
        &self.generated
    }

    pub fn pending_instruction(&self) -> Option<u64> {
        self.generation.as_ref().map(|g| g.instruction_id)
    }

    pub fn issue(
        &mut self,
        tick: u64,
        instruction_id: u64,
        requests: C220DmaUops,
    ) -> Result<C220DmaIssue, C220DmaFrontendError> {
        if self.load2d {
            return Err(C220DmaFrontendError::WrongGenerator);
        }
        self.issue_requests(tick, instruction_id, Requests::Dma(requests))
    }

    pub fn issue_load2d(
        &mut self,
        tick: u64,
        instruction_id: u64,
        requests: super::load2d::C220Load2dExternalRequests,
    ) -> Result<C220DmaIssue, C220DmaFrontendError> {
        if !self.load2d {
            return Err(C220DmaFrontendError::WrongGenerator);
        }
        self.issue_requests(tick, instruction_id, Requests::Load2d(requests))
    }

    fn issue_requests(
        &mut self,
        tick: u64,
        instruction_id: u64,
        requests: Requests,
    ) -> Result<C220DmaIssue, C220DmaFrontendError> {
        self.check_time(tick)?;
        if !self.can_issue() {
            return Err(C220DmaFrontendError::CommandBusy);
        }
        let completion_ready = requests.clone().next().is_none();
        if !completion_ready {
            let ready_tick = tick
                .checked_add(COMMAND_TICKS)
                .ok_or(C220DmaFrontendError::TimeOverflow)?;
            self.generation = Some(Generation {
                instruction_id,
                ready_tick,
                next_index: 0,
                requests,
            });
        }
        self.observed_tick = Some(tick);
        Ok(C220DmaIssue {
            tick,
            instruction_id,
            completion_ready,
        })
    }

    pub fn generate(
        &mut self,
        tick: u64,
    ) -> Result<Option<C220DmaGenerated>, C220DmaFrontendError> {
        self.check_callback(tick, self.generation_tick, "generation")?;
        let eligible = self
            .generation
            .as_ref()
            .is_some_and(|g| g.ready_tick <= tick)
            && self.generated.len() < self.generated_capacity();
        let entry = if eligible {
            let ready_tick = tick
                .checked_add(self.generated_ticks())
                .ok_or(C220DmaFrontendError::TimeOverflow)?;
            let generation = self.generation.as_mut().expect("eligible command");
            let request = generation.requests.next().expect("nonempty command");
            let (destination, mode, out_of_order) = generation.requests.metadata();
            let last_in_instruction = generation.requests.clone().next().is_none();
            let entry = C220DmaGenerated {
                instruction_id: generation.instruction_id,
                uop_index: generation.next_index,
                ready_tick,
                request,
                destination,
                mode,
                out_of_order,
                last_in_instruction,
            };
            generation.next_index += 1;
            if last_in_instruction {
                self.generation = None;
            }
            self.generated.push_back(entry);
            Some(entry)
        } else {
            None
        };
        self.generation_tick = Some(tick);
        self.observed_tick = Some(tick);
        Ok(entry)
    }

    pub fn send(
        &mut self,
        tick: u64,
        hardware_sync_blocked: bool,
        output_ready: bool,
    ) -> Result<C220DmaSend, C220DmaFrontendError> {
        self.check_callback(tick, self.send_tick, "send")?;
        let offered = self.generated.front().copied();
        let stall = offered.and_then(|head| {
            if tick < head.ready_tick {
                Some(C220DmaStall::NotReady)
            } else if hardware_sync_blocked {
                Some(C220DmaStall::HardwareFlag)
            } else if !output_ready {
                Some(C220DmaStall::OutputFull)
            } else {
                None
            }
        });
        let sent = if stall.is_none() {
            self.generated.pop_front()
        } else {
            None
        };
        self.send_tick = Some(tick);
        self.observed_tick = Some(tick);
        Ok(C220DmaSend {
            tick,
            offered,
            stall,
            sent,
        })
    }

    fn check_time(&self, tick: u64) -> Result<(), C220DmaFrontendError> {
        if let Some(previous) = self.observed_tick
            && tick < previous
        {
            return Err(C220DmaFrontendError::TimeReversed {
                previous,
                requested: tick,
            });
        }
        Ok(())
    }

    fn check_callback(
        &self,
        tick: u64,
        previous: Option<u64>,
        phase: &'static str,
    ) -> Result<(), C220DmaFrontendError> {
        self.check_time(tick)?;
        if previous == Some(tick) {
            return Err(C220DmaFrontendError::RepeatedCallback { phase, tick });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220DmaEventOutcome {
    Readiness,
    Generated(Option<C220DmaGenerated>),
    Sent(C220DmaSend),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220DmaEvents {
    queues: GeneratorEvents,
}

impl C220DmaEvents {
    pub fn issue_load2d<T: Copy>(
        &self,
        events: &mut EventDispatcher<T>,
        frontend: &mut C220DmaFrontend,
        instruction_id: u64,
        requests: super::load2d::C220Load2dExternalRequests,
    ) -> Result<C220DmaIssue, C220DmaFrontendError> {
        let issue = frontend.issue_load2d(events.tick(), instruction_id, requests)?;
        if !issue.completion_ready {
            self.queues.arm_instruction(events);
        }
        Ok(issue)
    }
    pub fn register<T: Copy>(
        events: &mut EventDispatcher<T>,
        clock: EventId,
        tag: impl Fn(C220MteGeneratorCallback) -> T,
    ) -> Self {
        Self {
            queues: GeneratorEvents::register(events, clock, tag),
        }
    }

    pub fn issue<T: Copy>(
        &self,
        events: &mut EventDispatcher<T>,
        frontend: &mut C220DmaFrontend,
        instruction_id: u64,
        requests: C220DmaUops,
    ) -> Result<C220DmaIssue, C220DmaFrontendError> {
        let issue = frontend.issue(events.tick(), instruction_id, requests)?;
        if !issue.completion_ready {
            self.queues.arm_instruction(events);
        }
        Ok(issue)
    }

    pub fn handle<T: Copy>(
        &self,
        callback: C220MteGeneratorCallback,
        events: &mut EventDispatcher<T>,
        frontend: &mut C220DmaFrontend,
        hardware_sync_blocked: bool,
        output_ready: bool,
    ) -> Result<C220DmaEventOutcome, C220DmaFrontendError> {
        match callback {
            C220MteGeneratorCallback::InstructionReady => {
                self.queues
                    .probe_instruction(events, frontend.generation.as_ref().map(|g| g.ready_tick));
                Ok(C220DmaEventOutcome::Readiness)
            }
            C220MteGeneratorCallback::GeneratedReady => {
                self.queues
                    .probe_generated(events, frontend.generated.front().map(|g| g.ready_tick));
                Ok(C220DmaEventOutcome::Readiness)
            }
            C220MteGeneratorCallback::Generate => {
                let generated = frontend.generate(events.tick())?;
                if generated.is_some() {
                    self.queues.arm_generated(events);
                }
                Ok(C220DmaEventOutcome::Generated(generated))
            }
            C220MteGeneratorCallback::Send => frontend
                .send(events.tick(), hardware_sync_blocked, output_ready)
                .map(C220DmaEventOutcome::Sent),
        }
    }
}
