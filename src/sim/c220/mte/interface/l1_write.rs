use std::collections::{BTreeMap, VecDeque};

use super::C220MteOutputFragment;
use crate::sim::c220::memory::l1::{C220L1Access, C220L1Request};

mod events;
pub use events::{
    C220MteL1WriteCallback, C220MteL1WriteEventInputs, C220MteL1WriteEventOutcome,
    C220MteL1WriteEvents,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum C220MteL1WritePort {
    Port0 = 0,
    Port1 = 1,
    Port2 = 2,
    Port3 = 3,
}

impl C220MteL1WritePort {
    pub(super) const ALL: [Self; 4] = [Self::Port0, Self::Port1, Self::Port2, Self::Port3];

    pub const fn input_ticks(self) -> u64 {
        match self {
            Self::Port0 => 9,
            Self::Port1 | Self::Port3 => 1,
            Self::Port2 => 3,
        }
    }

    pub const fn capacity(self) -> usize {
        self.input_ticks() as usize + 1
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MteL1WriteEntry {
    pub ready_tick: u64,
    pub fragment: C220MteOutputFragment,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MteL1WriteRequest {
    /// Unique within this write interface, independent of the producer's ID.
    pub id: u64,
    pub port: C220MteL1WritePort,
    pub fragment: C220MteOutputFragment,
}

impl C220MteL1WriteRequest {
    pub const fn l1_request(self) -> C220L1Request {
        C220L1Request {
            id: self.id,
            access: C220L1Access {
                address: self.fragment.destination_address,
                bytes: self.fragment.bytes,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MteL1WriteAcknowledgment {
    pub ready_tick: u64,
    pub request: C220MteL1WriteRequest,
}

impl C220MteL1WriteAcknowledgment {
    pub const fn retired_instruction(self) -> Option<u64> {
        if self.request.fragment.last_in_instruction {
            Some(self.request.fragment.instruction_id)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MteL1WriteQueues {
    pub inputs: [usize; 4],
    pub awaiting_response: usize,
    pub acknowledgments: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MteL1WriteSend {
    pub tick: u64,
    pub eligible: [bool; 4],
    pub selected: Option<C220MteL1WritePort>,
    pub sent: Option<C220MteL1WriteRequest>,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum C220MteL1WriteError {
    #[error("L1 write interface time reversed from {previous} to {requested}")]
    TimeReversed { previous: u64, requested: u64 },
    #[error("L1 write {phase} callback already ran at tick {tick}")]
    RepeatedCallback { phase: &'static str, tick: u64 },
    #[error("L1 write response {0} does not identify an outstanding request")]
    UnknownResponse(u64),
    #[error("L1 write time or request ID overflowed")]
    Overflow,
}

/// Shared MTE write client of the L1 transport. Input ports arbitrate in round
/// robin order, including when downstream transport is blocked. Responses, not
/// sends, enter the one-tick acknowledgment queue. Memory commits belong to
/// command retirement, not this timing interface.
///
/// Send, response and retirement callbacks are separate so the owner can retain
/// its event ordering. Each callback runs at most once per tick. The interface
/// neither steps the L1 transport nor imposes phases on other memory clients.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220MteL1WriteInterface {
    inputs: [VecDeque<C220MteL1WriteEntry>; 4],
    in_flight: BTreeMap<u64, C220MteL1WriteRequest>,
    acknowledgments: VecDeque<C220MteL1WriteAcknowledgment>,
    last_selected: C220MteL1WritePort,
    next_id: u64,
    observed_tick: Option<u64>,
    send_tick: Option<u64>,
    response_tick: Option<u64>,
    retirement_tick: Option<u64>,
}

impl Default for C220MteL1WriteInterface {
    fn default() -> Self {
        Self {
            inputs: std::array::from_fn(|_| VecDeque::new()),
            in_flight: BTreeMap::new(),
            acknowledgments: VecDeque::new(),
            last_selected: C220MteL1WritePort::Port3,
            next_id: 1,
            observed_tick: None,
            send_tick: None,
            response_tick: None,
            retirement_tick: None,
        }
    }
}

impl C220MteL1WriteInterface {
    pub fn is_idle(&self) -> bool {
        self.inputs.iter().all(VecDeque::is_empty)
            && self.in_flight.is_empty()
            && self.acknowledgments.is_empty()
    }

    pub fn can_push(&self, port: C220MteL1WritePort) -> bool {
        self.queue(port).len() < port.capacity()
    }

    pub fn queue(&self, port: C220MteL1WritePort) -> &VecDeque<C220MteL1WriteEntry> {
        &self.inputs[port as usize]
    }

    pub fn outstanding_requests(&self) -> impl Iterator<Item = &C220MteL1WriteRequest> {
        self.in_flight.values()
    }

    pub fn acknowledgments(&self) -> &VecDeque<C220MteL1WriteAcknowledgment> {
        &self.acknowledgments
    }

    pub fn queue_state(&self) -> C220MteL1WriteQueues {
        C220MteL1WriteQueues {
            inputs: std::array::from_fn(|index| self.inputs[index].len()),
            awaiting_response: self.in_flight.len(),
            acknowledgments: self.acknowledgments.len(),
        }
    }

    /// False leaves the fragment with the producer. No request ID is consumed.
    pub fn push(
        &mut self,
        tick: u64,
        port: C220MteL1WritePort,
        fragment: C220MteOutputFragment,
    ) -> Result<bool, C220MteL1WriteError> {
        self.check_time(tick)?;
        if !self.can_push(port) {
            self.observed_tick = Some(tick);
            return Ok(false);
        }
        let ready_tick = tick
            .checked_add(port.input_ticks())
            .ok_or(C220MteL1WriteError::Overflow)?;
        self.inputs[port as usize].push_back(C220MteL1WriteEntry {
            ready_tick,
            fragment,
        });
        self.observed_tick = Some(tick);
        Ok(true)
    }

    pub fn preview_send(&self, tick: u64) -> C220MteL1WriteSend {
        let eligible = std::array::from_fn(|index| {
            self.inputs[index]
                .front()
                .is_some_and(|head| head.ready_tick <= tick)
        });
        let selected = (1..=4)
            .map(|offset| C220MteL1WritePort::ALL[(self.last_selected as usize + offset) % 4])
            .find(|port| eligible[*port as usize]);
        C220MteL1WriteSend {
            tick,
            eligible,
            selected,
            sent: None,
        }
    }

    /// `request_ready` is the shared L1 MTE-write transport's current credit.
    /// The returned request must be handed to that transport in this callback.
    pub fn send(
        &mut self,
        tick: u64,
        request_ready: bool,
    ) -> Result<C220MteL1WriteSend, C220MteL1WriteError> {
        self.check_callback(tick, self.send_tick, "send")?;
        let mut result = self.preview_send(tick);
        if let Some(port) = result.selected {
            let next_id = if request_ready {
                self.next_id
                    .checked_add(1)
                    .ok_or(C220MteL1WriteError::Overflow)?
            } else {
                self.next_id
            };
            self.last_selected = port;
            if request_ready {
                let fragment = self.inputs[port as usize]
                    .pop_front()
                    .expect("selected head")
                    .fragment;
                let request = C220MteL1WriteRequest {
                    id: self.next_id,
                    port,
                    fragment,
                };
                self.next_id = next_id;
                self.in_flight.insert(request.id, request);
                result.sent = Some(request);
            }
        }
        self.send_tick = Some(tick);
        self.observed_tick = Some(tick);
        Ok(result)
    }

    pub fn receive_response(
        &mut self,
        tick: u64,
        id: u64,
    ) -> Result<C220MteL1WriteRequest, C220MteL1WriteError> {
        self.check_callback(tick, self.response_tick, "response")?;
        let request = *self
            .in_flight
            .get(&id)
            .ok_or(C220MteL1WriteError::UnknownResponse(id))?;
        if request.fragment.last_in_uop {
            let ready_tick = tick.checked_add(1).ok_or(C220MteL1WriteError::Overflow)?;
            self.acknowledgments
                .push_back(C220MteL1WriteAcknowledgment {
                    ready_tick,
                    request,
                });
        }
        self.in_flight.remove(&id);
        self.response_tick = Some(tick);
        self.observed_tick = Some(tick);
        Ok(request)
    }

    pub fn retire(
        &mut self,
        tick: u64,
    ) -> Result<Option<C220MteL1WriteAcknowledgment>, C220MteL1WriteError> {
        self.check_callback(tick, self.retirement_tick, "retirement")?;
        let acknowledgment = self
            .acknowledgments
            .pop_front_if(|head| head.ready_tick <= tick);
        self.retirement_tick = Some(tick);
        self.observed_tick = Some(tick);
        Ok(acknowledgment)
    }

    fn check_time(&self, tick: u64) -> Result<(), C220MteL1WriteError> {
        if let Some(previous) = self.observed_tick
            && tick < previous
        {
            return Err(C220MteL1WriteError::TimeReversed {
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
    ) -> Result<(), C220MteL1WriteError> {
        self.check_time(tick)?;
        if last == Some(tick) {
            return Err(C220MteL1WriteError::RepeatedCallback { phase, tick });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
