use std::collections::{BTreeMap, VecDeque};

use super::biu_read::write::{C220BiuWriteDestination, C220BiuWriteFragment};

mod events;
pub use events::{C220UbWriteCallback, C220UbWriteEvent, C220UbWriteEvents};

pub const C220_UB_WRITE_INPUT_CAPACITY: usize = 8;
pub const C220_UB_WRITE_INPUT_TICKS: u64 = 7;
pub const C220_UB_WRITE_TRANSPORT_CAPACITY: usize = 2;
pub const C220_UB_WRITE_TRANSPORT_TICKS: u64 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220UbWriteEntry {
    pub ready_tick: u64,
    pub fragment: C220BiuWriteFragment,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220UbWriteRequest {
    /// Unique within this interface, independent of the BIU tag and uop index.
    pub id: u64,
    pub sent_tick: u64,
    pub ready_tick: u64,
    pub fragment: C220BiuWriteFragment,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220UbWritePending {
    pub request: C220UbWriteRequest,
    pub delivered: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220UbWriteAcknowledgment {
    pub ready_tick: u64,
    pub request: C220UbWriteRequest,
}

impl C220UbWriteAcknowledgment {
    pub fn retired_instruction(self) -> Option<u64> {
        self.request.fragment.last_in_instruction.then_some(
            self.request
                .fragment
                .output
                .request
                .input
                .generated
                .instruction_id,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220UbWriteSend {
    pub tick: u64,
    pub attempted_id: Option<u64>,
    pub transport_full: bool,
    pub sent: Option<C220UbWriteRequest>,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum C220UbWriteError {
    #[error("UB write time reversed from {previous} to {requested}")]
    TimeReversed { previous: u64, requested: u64 },
    #[error("UB write {phase} callback already ran at tick {tick}")]
    RepeatedCallback { phase: &'static str, tick: u64 },
    #[error("UB write time or request ID overflowed")]
    Overflow,
    #[error("UB write interface only accepts UB destination packets")]
    WrongDestination,
    #[error("UB response {0} does not identify an outstanding request")]
    UnknownResponse(u64),
    #[error("UB request {0} has not reached the memory service")]
    RequestUndelivered(u64),
}

/// One vector subcore's MTE-to-UB write interface. Sending occupies a bounded
/// request channel; only a matching memory response enters the acknowledgment
/// queue. Memory arbitration and the response-channel delay are backend-owned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220UbWriteInterface {
    inputs: VecDeque<C220UbWriteEntry>,
    requests: VecDeque<C220UbWriteRequest>,
    in_flight: BTreeMap<u64, C220UbWritePending>,
    acknowledgments: VecDeque<C220UbWriteAcknowledgment>,
    next_id: u64,
    observed_tick: Option<u64>,
    send_tick: Option<u64>,
    receive_tick: Option<u64>,
    response_tick: Option<u64>,
    retire_tick: Option<u64>,
}

impl Default for C220UbWriteInterface {
    fn default() -> Self {
        Self {
            inputs: VecDeque::new(),
            requests: VecDeque::new(),
            in_flight: BTreeMap::new(),
            acknowledgments: VecDeque::new(),
            next_id: 1,
            observed_tick: None,
            send_tick: None,
            receive_tick: None,
            response_tick: None,
            retire_tick: None,
        }
    }
}

impl C220UbWriteInterface {
    pub fn is_idle(&self) -> bool {
        self.inputs.is_empty() && self.in_flight.is_empty() && self.acknowledgments.is_empty()
    }

    pub fn contains_instruction(&self, id: u64) -> bool {
        self.inputs
            .iter()
            .map(|entry| entry.fragment)
            .chain(
                self.in_flight
                    .values()
                    .map(|pending| pending.request.fragment),
            )
            .chain(self.acknowledgments.iter().map(|ack| ack.request.fragment))
            .any(|fragment| fragment.output.request.input.generated.instruction_id == id)
    }

    pub fn can_push(&self) -> bool {
        self.inputs.len() < C220_UB_WRITE_INPUT_CAPACITY
    }
    pub fn inputs(&self) -> &VecDeque<C220UbWriteEntry> {
        &self.inputs
    }
    pub fn requests(&self) -> &VecDeque<C220UbWriteRequest> {
        &self.requests
    }
    pub fn outstanding(&self) -> &BTreeMap<u64, C220UbWritePending> {
        &self.in_flight
    }
    pub fn acknowledgments(&self) -> &VecDeque<C220UbWriteAcknowledgment> {
        &self.acknowledgments
    }

    pub fn push(
        &mut self,
        tick: u64,
        fragment: C220BiuWriteFragment,
    ) -> Result<bool, C220UbWriteError> {
        self.check_time(tick)?;
        if !matches!(
            fragment.output.request.input.destination,
            C220BiuWriteDestination::Ub0 | C220BiuWriteDestination::Ub1
        ) {
            return Err(C220UbWriteError::WrongDestination);
        }
        if !self.can_push() {
            return Ok(false);
        }
        let ready_tick = tick
            .checked_add(C220_UB_WRITE_INPUT_TICKS)
            .ok_or(C220UbWriteError::Overflow)?;
        self.inputs.push_back(C220UbWriteEntry {
            ready_tick,
            fragment,
        });
        self.observed_tick = Some(tick);
        Ok(true)
    }

    pub fn send(&mut self, tick: u64) -> Result<C220UbWriteSend, C220UbWriteError> {
        self.check_callback(tick, self.send_tick, "send")?;
        let mut result = C220UbWriteSend {
            tick,
            attempted_id: None,
            transport_full: false,
            sent: None,
        };
        if let Some(head) = self
            .inputs
            .front()
            .copied()
            .filter(|head| head.ready_tick <= tick)
        {
            let id = self.next_id;
            let next_id = id.checked_add(1).ok_or(C220UbWriteError::Overflow)?;
            result.attempted_id = Some(id);
            result.transport_full = self.requests.len() == C220_UB_WRITE_TRANSPORT_CAPACITY;
            if !result.transport_full {
                let ready_tick = tick
                    .checked_add(C220_UB_WRITE_TRANSPORT_TICKS)
                    .ok_or(C220UbWriteError::Overflow)?;
                let request = C220UbWriteRequest {
                    id,
                    sent_tick: tick,
                    ready_tick,
                    fragment: head.fragment,
                };
                self.requests.push_back(request);
                self.in_flight.insert(
                    id,
                    C220UbWritePending {
                        request,
                        delivered: false,
                    },
                );
                self.inputs.pop_front();
                result.sent = Some(request);
            }
            // A failed transport attempt still consumes its request identity.
            self.next_id = next_id;
        }
        self.observed_tick = Some(tick);
        self.send_tick = Some(tick);
        Ok(result)
    }

    /// The memory receiver consumes at most one request per tick. Merely
    /// inspecting an unready channel does not consume that tick's receive slot.
    pub fn take_request(
        &mut self,
        tick: u64,
    ) -> Result<Option<C220UbWriteRequest>, C220UbWriteError> {
        self.check_time(tick)?;
        if self.receive_tick == Some(tick) {
            return Ok(None);
        }
        let request = self.requests.pop_front_if(|head| head.ready_tick <= tick);
        if let Some(request) = request {
            self.in_flight
                .get_mut(&request.id)
                .expect("queued request")
                .delivered = true;
            self.receive_tick = Some(tick);
        }
        self.observed_tick = Some(tick);
        Ok(request)
    }

    /// Called when a response arrives at this interface, after memory service
    /// and response transport. Unknown, duplicate and undelivered IDs fail
    /// without consuming the response callback or changing any queue.
    pub fn receive_response(
        &mut self,
        tick: u64,
        id: u64,
    ) -> Result<C220UbWriteRequest, C220UbWriteError> {
        self.check_callback(tick, self.response_tick, "response")?;
        let pending = self
            .in_flight
            .get(&id)
            .ok_or(C220UbWriteError::UnknownResponse(id))?;
        if !pending.delivered {
            return Err(C220UbWriteError::RequestUndelivered(id));
        }
        let ready_tick = tick.checked_add(1).ok_or(C220UbWriteError::Overflow)?;
        let request = pending.request;
        self.in_flight.remove(&id);
        self.acknowledgments.push_back(C220UbWriteAcknowledgment {
            ready_tick,
            request,
        });
        self.observed_tick = Some(tick);
        self.response_tick = Some(tick);
        Ok(request)
    }

    pub fn retire(
        &mut self,
        tick: u64,
    ) -> Result<Option<C220UbWriteAcknowledgment>, C220UbWriteError> {
        self.check_callback(tick, self.retire_tick, "retire")?;
        let ack = self
            .acknowledgments
            .pop_front_if(|ack| ack.ready_tick <= tick);
        self.observed_tick = Some(tick);
        self.retire_tick = Some(tick);
        Ok(ack)
    }

    fn check_time(&self, tick: u64) -> Result<(), C220UbWriteError> {
        if let Some(previous) = self.observed_tick
            && previous > tick
        {
            return Err(C220UbWriteError::TimeReversed {
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
    ) -> Result<(), C220UbWriteError> {
        self.check_time(tick)?;
        if previous == Some(tick) {
            return Err(C220UbWriteError::RepeatedCallback { phase, tick });
        }
        Ok(())
    }
}
