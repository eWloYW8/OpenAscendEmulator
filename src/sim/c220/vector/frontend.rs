use std::collections::VecDeque;
use std::num::NonZeroU32;

use super::dispatch::VectorStep;
use super::pipeline::C220VectorPipelineError;
use super::runtime::{C220VectorRuntimeError, VectorEngine};
use super::{C220VectorInstruction, C220VectorRequest};
use crate::sim::c220::schedule::{C220Stall, C220StallCause};
use crate::sim::c220::state::C220State;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220VectorFrontendConfig {
    pub issue_queue_depth: NonZeroU32,
    pub outstanding_limit: NonZeroU32,
}

impl Default for C220VectorFrontendConfig {
    fn default() -> Self {
        Self {
            issue_queue_depth: NonZeroU32::new(64).unwrap(),
            outstanding_limit: NonZeroU32::new(31).unwrap(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220VectorQueuedInstruction {
    pub instruction_id: u64,
    pub pc: u64,
    pub word: u32,
    pub accepted_tick: u64,
    pub ready_tick: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220VectorReception {
    pub instruction: C220VectorQueuedInstruction,
    pub received_tick: u64,
    pub ready_tick: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum C220VectorFrontendEvent {
    Barrier(super::C220VectorBarrierOutcome),
    Received(C220VectorReception),
    Dispatched {
        instruction_id: u64,
        tick: u64,
        instruction: Box<C220VectorInstruction>,
    },
    Stalled(C220Stall),
}

pub(in crate::sim::c220) enum VectorAdmission {
    Queued(C220VectorQueuedInstruction),
    Stalled(C220Stall),
}

struct Issued {
    ticket: C220VectorQueuedInstruction,
    request: C220VectorRequest,
}

struct Received {
    reception: C220VectorReception,
    request: C220VectorRequest,
}

pub(super) struct VectorFrontend {
    config: C220VectorFrontendConfig,
    reception_ticks: u64,
    issued: VecDeque<Issued>,
    received: VecDeque<Received>,
    pub last_accepted: Option<u64>,
    pub barriers: VecDeque<super::C220VectorBarrier>,
    pub next_tick: Option<u64>,
    pub events: Vec<C220VectorFrontendEvent>,
}

impl VectorFrontend {
    pub(super) fn new(config: C220VectorFrontendConfig, reception_ticks: u64) -> Self {
        Self {
            config,
            reception_ticks,
            issued: VecDeque::new(),
            received: VecDeque::new(),
            last_accepted: None,
            barriers: VecDeque::new(),
            next_tick: None,
            events: Vec::new(),
        }
    }

    pub(super) fn fence_id(&self) -> Option<u64> {
        self.issued
            .back()
            .map(|entry| entry.ticket.instruction_id)
            .or_else(|| {
                self.received
                    .back()
                    .map(|entry| entry.reception.instruction.instruction_id)
            })
    }

    pub(super) fn fence_pending(&self, id: u64) -> bool {
        self.issued
            .front()
            .is_some_and(|entry| entry.ticket.instruction_id <= id)
            || self
                .received
                .front()
                .is_some_and(|entry| entry.reception.instruction.instruction_id <= id)
    }

    pub(super) fn received_count(&self) -> usize {
        self.received.len()
    }

    pub(super) fn is_full(&self) -> bool {
        self.issued.len() >= self.config.issue_queue_depth.get() as usize
    }
}

impl VectorEngine {
    pub(in crate::sim::c220) fn queued_instructions(
        &self,
    ) -> impl ExactSizeIterator<Item = C220VectorQueuedInstruction> + '_ {
        self.frontend.issued.iter().map(|entry| entry.ticket)
    }

    pub(in crate::sim::c220) fn received_instructions(
        &self,
    ) -> impl ExactSizeIterator<Item = C220VectorReception> + '_ {
        self.frontend.received.iter().map(|entry| entry.reception)
    }

    pub(in crate::sim::c220) fn frontend_events(&self) -> &[C220VectorFrontendEvent] {
        &self.frontend.events
    }

    pub(in crate::sim::c220) fn enqueue_at(
        &mut self,
        tick: u64,
        instruction_id: u64,
        request: C220VectorRequest,
    ) -> Result<VectorAdmission, C220VectorRuntimeError> {
        let ready_tick = tick
            .checked_add(1)
            .ok_or(C220VectorPipelineError::TimeOverflow)?;
        if self.frontend.is_full() {
            return Ok(VectorAdmission::Stalled(C220Stall {
                tick,
                pc: request.pc,
                resume_tick: ready_tick,
                cause: C220StallCause::VectorIssueQueueFull,
            }));
        }
        let ticket = C220VectorQueuedInstruction {
            instruction_id,
            pc: request.pc,
            word: request.word,
            accepted_tick: tick,
            ready_tick,
        };
        self.frontend.issued.push_back(Issued { ticket, request });
        self.frontend.last_accepted = Some(instruction_id);
        self.frontend.next_tick = Some(
            self.frontend
                .next_tick
                .map_or(ready_tick, |next| next.min(ready_tick)),
        );
        Ok(VectorAdmission::Queued(ticket))
    }

    pub(super) fn advance_frontend(
        &mut self,
        tick: u64,
        state: &mut C220State,
    ) -> Result<(), C220VectorRuntimeError> {
        if self.frontend.next_tick.is_none_or(|next| next > tick) {
            return Ok(());
        }
        let retry = tick
            .checked_add(1)
            .ok_or(C220VectorPipelineError::TimeOverflow)?;
        if let Some(ticket) = self.frontend.issued.front().map(|entry| entry.ticket)
            && ticket.ready_tick <= tick
        {
            let cause = if self.frontend.received.len() as u64 > self.frontend.reception_ticks {
                Some(C220StallCause::VectorReceptionQueueFull)
            } else if self.outstanding_instructions()
                >= self.frontend.config.outstanding_limit.get() as usize
                || self.pending_instructions.len() >= 32
            {
                Some(C220StallCause::VectorOutstandingLimit)
            } else if self.frontend.barriers.iter().any(|barrier| {
                barrier
                    .predecessor
                    .is_some_and(|id| id < ticket.instruction_id)
            }) {
                Some(C220StallCause::VectorBarrier)
            } else {
                None
            };
            if let Some(cause) = cause {
                self.frontend
                    .events
                    .push(C220VectorFrontendEvent::Stalled(C220Stall {
                        tick,
                        pc: ticket.pc,
                        resume_tick: retry,
                        cause,
                    }));
            } else {
                let ready_tick = tick
                    .checked_add(self.frontend.reception_ticks)
                    .ok_or(C220VectorPipelineError::TimeOverflow)?;
                let issued = self
                    .frontend
                    .issued
                    .pop_front()
                    .expect("ready Vector issue");
                let reception = C220VectorReception {
                    instruction: issued.ticket,
                    received_tick: tick,
                    ready_tick,
                };
                self.frontend.received.push_back(Received {
                    reception,
                    request: issued.request,
                });
                self.frontend
                    .events
                    .push(C220VectorFrontendEvent::Received(reception));
            }
        }
        if let Some(received) = self
            .frontend
            .received
            .pop_front_if(|head| head.reception.ready_tick <= tick)
        {
            let id = received.reception.instruction.instruction_id;
            let result = self.dispatch_at(tick, id, &received.request, state);
            match result {
                Ok(VectorStep::Issued(instruction)) => {
                    self.frontend
                        .events
                        .push(C220VectorFrontendEvent::Dispatched {
                            instruction_id: id,
                            tick,
                            instruction: Box::new(instruction),
                        });
                }
                Ok(VectorStep::Stalled(stall)) => {
                    self.frontend.received.push_front(received);
                    self.frontend
                        .events
                        .push(C220VectorFrontendEvent::Stalled(stall));
                }
                Err(error) => {
                    self.frontend.received.push_front(received);
                    return Err(error);
                }
            }
        }
        self.frontend.next_tick = self
            .frontend
            .issued
            .front()
            .map(|head| head.ticket.ready_tick.max(retry))
            .into_iter()
            .chain(
                self.frontend
                    .received
                    .front()
                    .map(|head| head.reception.ready_tick.max(retry)),
            )
            .min();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::architecture::Architecture;
    use crate::memory::ub::UbMemory;
    use crate::sim::c220::vector::ops::compare::C220CompareMask;
    use crate::sim::c220::vector::pipeline::C220VectorTimingRules;
    use crate::sim::common::scalar::{ScalarMachine, ScalarStepper};
    use std::num::NonZeroU64;

    #[test]
    fn outstanding_credit_is_held_from_reception_through_retirement() {
        let mut state = C220State::new(
            ScalarStepper::new(
                ScalarMachine::from_pem_initial_state(Architecture::Dav2201),
                0x1000,
            ),
            UbMemory::new(256, 256),
        );
        let mut engine = VectorEngine::new(
            C220VectorTimingRules {
                dispatch_ticks: 3,
                uop_issue_interval: NonZeroU64::new(1).unwrap(),
                ub_response_ticks: 1,
            },
            C220CompareMask::from_bits([0; 2]),
            C220VectorFrontendConfig {
                issue_queue_depth: NonZeroU32::new(2).unwrap(),
                outstanding_limit: NonZeroU32::new(1).unwrap(),
            },
        );
        let capture =
            |state: &C220State| C220VectorRequest::capture(state.scalar(), 0x8000_6380).unwrap();
        assert!(matches!(
            engine.enqueue_at(0, 0, capture(&state)).unwrap(),
            VectorAdmission::Queued(_)
        ));
        engine.advance_event(1, &mut state).unwrap();
        assert_eq!(engine.outstanding_instructions(), 1);
        assert_eq!(engine.received_instructions().next().unwrap().ready_tick, 4);
        for id in 1..=2 {
            assert!(matches!(
                engine.enqueue_at(id, id, capture(&state)).unwrap(),
                VectorAdmission::Queued(_)
            ));
            assert_eq!(engine.frontend.next_tick, Some(id + 1));
            engine.advance_event(id + 1, &mut state).unwrap();
        }
        assert!(matches!(
            engine.enqueue_at(3, 3, capture(&state)).unwrap(),
            VectorAdmission::Stalled(C220Stall {
                cause: C220StallCause::VectorIssueQueueFull,
                ..
            })
        ));
        let fence = engine.instruction_fence();
        let barrier = crate::isa::flow::PipelineBarrierStep::decode(
            Architecture::Dav2201,
            0x1000,
            0x40e0_0400,
        )
        .unwrap();
        assert!(matches!(
            engine.issue_barrier_at(3, 3, barrier).unwrap(),
            super::super::barrier::VectorBarrierAdmission::Stalled(C220Stall {
                cause: C220StallCause::VectorIssueQueueFull,
                ..
            })
        ));
        assert_eq!(engine.pending_barriers().len(), 0);
        assert_eq!(fence.instruction_id, Some(2));
        for tick in 4..=5 {
            engine.advance_event(tick, &mut state).unwrap();
            assert_eq!(engine.queued_instructions().len(), 2);
            assert_eq!(engine.outstanding_instructions(), 1);
        }
        engine.advance_event(6, &mut state).unwrap();
        assert_eq!(engine.retirements[0].instruction_id, 0);
        assert_eq!(engine.queued_instructions().len(), 1);
        assert_eq!(engine.outstanding_instructions(), 1);
        for tick in 7..=16 {
            assert!(engine.fence_retirement_tick(fence).is_some());
            engine.advance_event(tick, &mut state).unwrap();
        }
        assert_eq!(engine.fence_retirement_tick(fence), None);
        assert_eq!(engine.outstanding_instructions(), 0);
        assert_eq!(
            engine
                .retirements
                .iter()
                .map(|entry| entry.retirement_tick)
                .collect::<Vec<_>>(),
            [6, 11, 16]
        );
    }
}
