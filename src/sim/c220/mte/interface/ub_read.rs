use std::collections::{BTreeMap, VecDeque};

const INPUT_CAPACITY: usize = 5;
const INPUT_TICKS: u64 = 4;
const TRANSPORT_CAPACITY: usize = 2;
const TRANSPORT_TICKS: u64 = 1;
const RESPONSE_TICKS: u64 = 3;

/// One source-side packet from the BIU write alignment buffer.
/// `tag` identifies the parent BIU request; several packets may share it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220UbReadFragment {
    pub tag: u32,
    pub address: u64,
    pub bytes: u32,
    pub completes_read: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220UbReadEntry {
    pub ready_tick: u64,
    pub fragment: C220UbReadFragment,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220UbReadRequest {
    pub id: u64,
    pub sent_tick: u64,
    pub ready_tick: u64,
    pub fragment: C220UbReadFragment,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220UbReadAcknowledgment {
    pub ready_tick: u64,
    pub request: C220UbReadRequest,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum C220UbReadError {
    #[error("UB read time reversed from {previous} to {requested}")]
    TimeReversed { previous: u64, requested: u64 },
    #[error("UB read {phase} callback already ran at tick {tick}")]
    RepeatedCallback { phase: &'static str, tick: u64 },
    #[error("UB read time or request ID overflowed")]
    Overflow,
    #[error("UB read response {0} has no delivered request")]
    UnexpectedResponse(u64),
}

/// One vector subcore's UB read interface. Acknowledgments remain in response
/// order until the BIU polls the matching head tag. Completing a UB read does
/// not imply that the parent BIU write or MTE command has completed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220UbReadInterface {
    inputs: VecDeque<C220UbReadEntry>,
    transport: VecDeque<C220UbReadRequest>,
    delivered: BTreeMap<u64, C220UbReadRequest>,
    acknowledgments: VecDeque<C220UbReadAcknowledgment>,
    next_id: u64,
    observed_tick: Option<u64>,
    send_tick: Option<u64>,
    receive_tick: Option<u64>,
    response_tick: Option<u64>,
}

impl Default for C220UbReadInterface {
    fn default() -> Self {
        Self {
            inputs: VecDeque::new(),
            transport: VecDeque::new(),
            delivered: BTreeMap::new(),
            acknowledgments: VecDeque::new(),
            next_id: 1,
            observed_tick: None,
            send_tick: None,
            receive_tick: None,
            response_tick: None,
        }
    }
}

impl C220UbReadInterface {
    pub fn is_idle(&self) -> bool {
        self.inputs.is_empty()
            && self.transport.is_empty()
            && self.delivered.is_empty()
            && self.acknowledgments.is_empty()
    }

    pub fn can_push(&self) -> bool {
        self.inputs.len() < INPUT_CAPACITY
    }
    pub fn inputs(&self) -> &VecDeque<C220UbReadEntry> {
        &self.inputs
    }
    pub fn requests(&self) -> &VecDeque<C220UbReadRequest> {
        &self.transport
    }
    pub fn acknowledgments(&self) -> &VecDeque<C220UbReadAcknowledgment> {
        &self.acknowledgments
    }

    pub fn push(
        &mut self,
        tick: u64,
        fragment: C220UbReadFragment,
    ) -> Result<bool, C220UbReadError> {
        self.check_time(tick)?;
        if !self.can_push() {
            return Ok(false);
        }
        let ready_tick = tick
            .checked_add(INPUT_TICKS)
            .ok_or(C220UbReadError::Overflow)?;
        self.inputs.push_back(C220UbReadEntry {
            ready_tick,
            fragment,
        });
        self.observed_tick = Some(tick);
        Ok(true)
    }

    pub fn send(&mut self, tick: u64) -> Result<Option<C220UbReadRequest>, C220UbReadError> {
        self.check_callback(tick, self.send_tick, "send")?;
        let mut sent = None;
        if let Some(head) = self.inputs.front().filter(|head| head.ready_tick <= tick) {
            let id = self.next_id;
            let next_id = id.checked_add(1).ok_or(C220UbReadError::Overflow)?;
            if self.transport.len() < TRANSPORT_CAPACITY {
                let ready_tick = tick
                    .checked_add(TRANSPORT_TICKS)
                    .ok_or(C220UbReadError::Overflow)?;
                let request = C220UbReadRequest {
                    id,
                    sent_tick: tick,
                    ready_tick,
                    fragment: head.fragment,
                };
                self.transport.push_back(request);
                self.inputs.pop_front();
                sent = Some(request);
            }
            self.next_id = next_id;
        }
        self.send_tick = Some(tick);
        self.observed_tick = Some(tick);
        Ok(sent)
    }

    pub fn take_request(
        &mut self,
        tick: u64,
    ) -> Result<Option<C220UbReadRequest>, C220UbReadError> {
        self.check_time(tick)?;
        if self.receive_tick == Some(tick) {
            return Ok(None);
        }
        let request = self.transport.pop_front_if(|head| head.ready_tick <= tick);
        if let Some(request) = request {
            self.delivered.insert(request.id, request);
            self.receive_tick = Some(tick);
        }
        self.observed_tick = Some(tick);
        Ok(request)
    }

    pub fn receive_response(
        &mut self,
        tick: u64,
        id: u64,
    ) -> Result<C220UbReadRequest, C220UbReadError> {
        self.check_callback(tick, self.response_tick, "response")?;
        let request = *self
            .delivered
            .get(&id)
            .ok_or(C220UbReadError::UnexpectedResponse(id))?;
        let ready_tick = tick
            .checked_add(RESPONSE_TICKS)
            .ok_or(C220UbReadError::Overflow)?;
        self.delivered.remove(&id);
        if request.fragment.completes_read {
            self.acknowledgments.push_back(C220UbReadAcknowledgment {
                ready_tick,
                request,
            });
        }
        self.response_tick = Some(tick);
        self.observed_tick = Some(tick);
        Ok(request)
    }

    pub fn take_completion(
        &mut self,
        tick: u64,
        tag: u32,
    ) -> Result<Option<C220UbReadAcknowledgment>, C220UbReadError> {
        self.check_time(tick)?;
        self.observed_tick = Some(tick);
        Ok(self
            .acknowledgments
            .pop_front_if(|head| head.ready_tick <= tick && head.request.fragment.tag == tag))
    }

    fn check_time(&self, tick: u64) -> Result<(), C220UbReadError> {
        if let Some(previous) = self.observed_tick
            && tick < previous
        {
            return Err(C220UbReadError::TimeReversed {
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
    ) -> Result<(), C220UbReadError> {
        self.check_time(tick)?;
        if previous == Some(tick) {
            return Err(C220UbReadError::RepeatedCallback { phase, tick });
        }
        Ok(())
    }
}
