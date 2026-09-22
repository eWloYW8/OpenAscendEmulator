use std::collections::{BTreeMap, VecDeque};
use std::num::NonZeroU32;

use super::{
    C220MteL1Output, C220MteL1OutputCredits, C220MteL1OutputCycle, C220MteL1OutputDestination,
    C220MteL1OutputError, C220MteL1OutputQueues, C220MteL1ReadArbiter, C220MteL1ReadDecision,
    C220MteL1ReadDestination, C220MteL1ReadHead, C220MteL1ReadPort, C220MteOutputPlan,
};
use crate::sim::c220::memory::l1::{C220L1Access, C220L1Request};

const INPUT_TICKS: u64 = 4;
const INPUT_CAPACITY: usize = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MteL1ReadOperation<T> {
    pub instruction_id: u64,
    pub access: C220L1Access,
    pub destination: C220MteL1OutputDestination,
    pub output_address: u64,
    pub output_bytes: u32,
    pub output_bandwidth: NonZeroU32,
    pub completes_logical_uop: bool,
    pub last_in_instruction: bool,
    pub payload: T,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MteL1ReadRequest<T> {
    pub id: u64,
    pub operation: C220MteL1ReadOperation<T>,
}

impl<T> C220MteL1ReadRequest<T> {
    pub const fn l1_request(&self) -> C220L1Request {
        C220L1Request {
            id: self.id,
            access: self.operation.access,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct C220MteL1CycleInputs {
    pub request_ready: bool,
    pub response: Option<u64>,
    pub output_credits: C220MteL1OutputCredits,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MteL1Queues {
    pub inputs: [usize; 3],
    pub awaiting_response: usize,
    pub output: C220MteL1OutputQueues,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220MteL1Cycle<T> {
    pub tick: u64,
    pub decision: C220MteL1ReadDecision,
    pub sent: Option<C220MteL1ReadRequest<T>>,
    pub received: Option<C220MteL1ReadRequest<T>>,
    pub output: C220MteL1OutputCycle<C220MteL1ReadRequest<T>>,
    pub queues: C220MteL1Queues,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum C220MteL1Error {
    #[error("L1 interface time reversed from {previous} to {requested}")]
    TimeReversed { previous: u64, requested: u64 },
    #[error("L1 interface expected cycle {expected}, got {requested}")]
    InvalidCycle { expected: u64, requested: u64 },
    #[error("L1 interface time or request ID overflowed")]
    Overflow,
    #[error("L1 response {0} does not identify an outstanding request")]
    UnknownResponse(u64),
    #[error(transparent)]
    Output(#[from] C220MteL1OutputError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Entry<T> {
    ready_tick: u64,
    request: C220MteL1ReadRequest<T>,
}

/// Shared by all producers feeding the MTE L1 read interface. Input queues,
/// request IDs, in-flight responses, and output bandwidth have one owner.
/// Memory-bank arbitration and the request/response transport remain external.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220MteL1Interface<T> {
    inputs: [VecDeque<Entry<T>>; 3],
    arbiter: C220MteL1ReadArbiter,
    in_flight: BTreeMap<u64, C220MteL1ReadRequest<T>>,
    output: C220MteL1Output<C220MteL1ReadRequest<T>>,
    next_request_id: u64,
    observed_tick: Option<u64>,
    next_tick: Option<u64>,
}

impl<T> Default for C220MteL1Interface<T> {
    fn default() -> Self {
        Self {
            inputs: std::array::from_fn(|_| VecDeque::new()),
            arbiter: C220MteL1ReadArbiter::default(),
            in_flight: BTreeMap::new(),
            output: C220MteL1Output::default(),
            next_request_id: 0,
            observed_tick: None,
            next_tick: None,
        }
    }
}

impl<T: Copy> C220MteL1Interface<T> {
    pub fn is_idle(&self) -> bool {
        self.inputs.iter().all(VecDeque::is_empty)
            && self.in_flight.is_empty()
            && self.output.is_idle()
    }

    pub fn can_push(&self, port: C220MteL1ReadPort) -> bool {
        self.inputs[port as usize].len() < INPUT_CAPACITY
    }

    pub fn queue_state(&self) -> C220MteL1Queues {
        C220MteL1Queues {
            inputs: std::array::from_fn(|index| self.inputs[index].len()),
            awaiting_response: self.in_flight.len(),
            output: self.output.queue_state(),
        }
    }

    pub fn outstanding_requests(&self) -> impl Iterator<Item = &C220MteL1ReadRequest<T>> {
        self.in_flight.values()
    }

    pub fn read_decision(&self, tick: u64) -> C220MteL1ReadDecision {
        self.arbiter.preview(
            tick,
            self.heads(),
            self.output.queue_state().output_fragments,
        )
    }

    pub(in crate::sim::c220::mte) fn validate_push(
        &self,
        tick: u64,
        port: C220MteL1ReadPort,
    ) -> Result<(), C220MteL1Error> {
        self.check_time(tick)?;
        if self.can_push(port) {
            tick.checked_add(INPUT_TICKS)
                .ok_or(C220MteL1Error::Overflow)?;
            self.next_request_id
                .checked_add(1)
                .ok_or(C220MteL1Error::Overflow)?;
        }
        Ok(())
    }

    /// Returns None on a full port without consuming the operation or an ID.
    /// Push before or after step to specify the producer's callback phase.
    pub fn push(
        &mut self,
        tick: u64,
        port: C220MteL1ReadPort,
        operation: C220MteL1ReadOperation<T>,
    ) -> Result<Option<C220MteL1ReadRequest<T>>, C220MteL1Error> {
        self.validate_push(tick, port)?;
        if !self.can_push(port) {
            return Ok(None);
        }
        if self.is_idle() {
            self.next_tick = Some(self.next_tick.unwrap_or(tick).max(tick));
        }
        let request = C220MteL1ReadRequest {
            id: self.next_request_id,
            operation,
        };
        self.inputs[port as usize].push_back(Entry {
            ready_tick: tick + INPUT_TICKS,
            request,
        });
        self.next_request_id += 1;
        self.observed_tick = Some(tick);
        Ok(Some(request))
    }

    pub fn step(
        &mut self,
        tick: u64,
        inputs: C220MteL1CycleInputs,
    ) -> Result<C220MteL1Cycle<T>, C220MteL1Error> {
        self.check_time(tick)?;
        if let Some(expected) = self.next_tick
            && tick < expected
        {
            return Err(C220MteL1Error::InvalidCycle {
                expected,
                requested: tick,
            });
        }
        let next_tick = tick.checked_add(1).ok_or(C220MteL1Error::Overflow)?;
        let received = inputs
            .response
            .map(|id| {
                self.in_flight
                    .get(&id)
                    .copied()
                    .ok_or(C220MteL1Error::UnknownResponse(id))
            })
            .transpose()?;
        self.output.validate_step(tick, inputs.output_credits)?;
        if received.is_some_and(|request| request.operation.completes_logical_uop) {
            self.output.validate_receive(tick)?;
        }

        // Read eligibility uses occupancy at callback entry, before output drains.
        let decision = self.arbiter.arbitrate(
            tick,
            self.heads(),
            self.output.queue_state().output_fragments,
        );
        let output = self.output.step(tick, inputs.output_credits)?;
        if let Some(request) = received {
            self.in_flight.remove(&request.id);
            let operation = request.operation;
            if operation.completes_logical_uop {
                self.output.receive(
                    tick,
                    operation.destination,
                    C220MteOutputPlan::new(
                        operation.instruction_id,
                        request.id,
                        operation.output_address,
                        operation.output_bytes,
                        operation.last_in_instruction,
                        operation.output_bandwidth,
                    ),
                    request,
                )?;
            }
        }
        let sent = decision
            .selected
            .filter(|_| inputs.request_ready)
            .map(|port| {
                let request = self.inputs[port as usize]
                    .pop_front()
                    .expect("selected input")
                    .request;
                self.in_flight.insert(request.id, request);
                request
            });
        self.observed_tick = Some(tick);
        self.next_tick = Some(next_tick);
        Ok(C220MteL1Cycle {
            tick,
            decision,
            sent,
            received,
            output,
            queues: self.queue_state(),
        })
    }

    fn heads(&self) -> [Option<C220MteL1ReadHead>; 3] {
        std::array::from_fn(|index| {
            self.inputs[index].front().map(|entry| C220MteL1ReadHead {
                ready_tick: entry.ready_tick,
                destination: match entry.request.operation.destination {
                    C220MteL1OutputDestination::Bt => C220MteL1ReadDestination::Bt,
                    C220MteL1OutputDestination::L0a(_) => C220MteL1ReadDestination::L0a,
                    C220MteL1OutputDestination::L0b(_) => C220MteL1ReadDestination::L0b,
                },
            })
        })
    }

    fn check_time(&self, tick: u64) -> Result<(), C220MteL1Error> {
        if let Some(previous) = self.observed_tick
            && tick < previous
        {
            return Err(C220MteL1Error::TimeReversed {
                previous,
                requested: tick,
            });
        }
        if !self.is_idle()
            && let Some(expected) = self.next_tick
            && tick > expected
        {
            return Err(C220MteL1Error::InvalidCycle {
                expected,
                requested: tick,
            });
        }
        Ok(())
    }
}
