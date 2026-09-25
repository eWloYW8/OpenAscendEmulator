use std::collections::{BTreeMap, VecDeque};

use super::store_buffer::{C220LsuLineKey, C220LsuMemory};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct C220LsuWriteId(u64);

impl C220LsuWriteId {
    pub const fn sequence(self) -> u64 {
        self.0
    }

    pub(crate) const fn from_sequence(value: u64) -> Self {
        Self(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220LsuWriteState {
    Queued,
    InFlight,
    Responding,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220LsuWriteRequest {
    pub id: C220LsuWriteId,
    pub line: C220LsuLineKey,
    pub state: C220LsuWriteState,
    pub queued_tick: u64,
    pub byte_len: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum C220LsuWriteError {
    #[error("write request ID exhausted")]
    Overflow,
    #[error("write request does not exist")]
    MissingRequest,
    #[error("write request is not in the required response state")]
    InvalidState,
    #[error("write-queue clock reversed from {previous} to {requested}")]
    TimeReversal { previous: u64, requested: u64 },
    #[error("write-queue clock exhausted")]
    TimeOverflow,
}

/// Write-port arbitration and lifetime tracking. Payload data remains in cache,
/// store, or eviction storage until response handling. Event scheduling and
/// port readiness are supplied by the transport controller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220LsuWriteQueue {
    capacity: u32,
    line_bytes: usize,
    outstanding: u32,
    next_id: u64,
    ub: VecDeque<C220LsuWriteId>,
    external: VecDeque<C220LsuWriteId>,
    requests: BTreeMap<C220LsuWriteId, C220LsuWriteRequest>,
    ub_addresses: BTreeMap<u64, C220LsuWriteId>,
    tick: u64,
    last_dispatch: Option<u64>,
}

impl C220LsuWriteQueue {
    pub fn new(capacity: u32, line_bytes: usize) -> Self {
        Self {
            capacity,
            line_bytes,
            outstanding: 0,
            next_id: 0,
            ub: VecDeque::new(),
            external: VecDeque::new(),
            requests: BTreeMap::new(),
            ub_addresses: BTreeMap::new(),
            tick: 0,
            last_dispatch: None,
        }
    }

    pub const fn outstanding(&self) -> u32 {
        self.outstanding
    }

    pub const fn tick(&self) -> u64 {
        self.tick
    }

    pub fn check_tick(&self, tick: u64) -> Result<(), C220LsuWriteError> {
        if tick < self.tick {
            return Err(C220LsuWriteError::TimeReversal {
                previous: self.tick,
                requested: tick,
            });
        }
        Ok(())
    }

    /// Set the event time before response handling or request generation.
    pub fn advance_to(&mut self, tick: u64) -> Result<(), C220LsuWriteError> {
        self.check_tick(tick)?;
        self.tick = tick;
        Ok(())
    }

    pub fn next_ready_tick(&self) -> Option<u64> {
        let next_process_tick = self
            .last_dispatch
            .map_or(Some(0), |tick| tick.checked_add(1))?;
        [self.ub.front(), self.external.front()]
            .into_iter()
            .flatten()
            .filter_map(|id| self.requests[id].queued_tick.checked_add(1))
            .map(|tick| tick.max(next_process_tick).max(self.tick))
            .min()
    }

    pub fn requests(&self) -> impl Iterator<Item = &C220LsuWriteRequest> {
        self.requests.values()
    }

    pub fn request(&self, id: C220LsuWriteId) -> Option<&C220LsuWriteRequest> {
        self.requests.get(&id)
    }

    pub fn check_enqueue(&self, count: u64) -> Result<(), C220LsuWriteError> {
        if count != 0 && self.tick == u64::MAX {
            return Err(C220LsuWriteError::TimeOverflow);
        }
        self.next_id
            .checked_add(count)
            .map(|_| ())
            .ok_or(C220LsuWriteError::Overflow)
    }

    pub fn enqueue(&mut self, line: C220LsuLineKey) -> Result<C220LsuWriteId, C220LsuWriteError> {
        self.check_enqueue(1)?;
        let next = self
            .next_id
            .checked_add(1)
            .ok_or(C220LsuWriteError::Overflow)?;
        let id = C220LsuWriteId(self.next_id);
        self.next_id = next;
        self.requests.insert(
            id,
            C220LsuWriteRequest {
                id,
                line,
                state: C220LsuWriteState::Queued,
                queued_tick: self.tick,
                byte_len: self.line_bytes,
            },
        );
        match line.memory {
            C220LsuMemory::Ub => {
                self.ub.push_back(id);
                self.ub_addresses.insert(line.address, id);
            }
            C220LsuMemory::External => self.external.push_back(id),
        }
        Ok(id)
    }

    pub fn has_hazard(&self, line: C220LsuLineKey) -> bool {
        match line.memory {
            C220LsuMemory::Ub => self.ub_addresses.contains_key(&line.address),
            C220LsuMemory::External => self.requests.values().any(|request| request.line == line),
        }
    }

    /// Evaluate queue-valid events at a clock tick. Either aged queue head can
    /// wake the shared send process, which may also send the other, younger head.
    /// Port acceptance is sampled by the caller; sending does not imply arrival.
    pub fn dispatch_clock(
        &mut self,
        tick: u64,
        ub_accepts: bool,
        external_accepts: bool,
    ) -> Result<Vec<C220LsuWriteRequest>, C220LsuWriteError> {
        self.check_tick(tick)?;
        let ready = self.next_ready_tick().is_some_and(|ready| ready <= tick);
        self.tick = tick;
        if !ready || self.last_dispatch == Some(tick) {
            return Ok(Vec::new());
        }
        self.last_dispatch = Some(tick);
        let mut sent = Vec::with_capacity(2);
        for (queue, accepts) in [
            (&mut self.ub, ub_accepts),
            (&mut self.external, external_accepts),
        ] {
            if self.outstanding >= self.capacity {
                break;
            }
            if accepts && let Some(id) = queue.pop_front() {
                let request = self.requests.get_mut(&id).expect("queued write exists");
                request.state = C220LsuWriteState::InFlight;
                self.outstanding += 1;
                sent.push(*request);
            }
        }
        Ok(sent)
    }

    /// Free shared capacity before applying the response's memory effects.
    /// The address remains blocked until `finish_response` is called.
    pub fn begin_response(
        &mut self,
        id: C220LsuWriteId,
    ) -> Result<C220LsuLineKey, C220LsuWriteError> {
        let request = self
            .requests
            .get_mut(&id)
            .ok_or(C220LsuWriteError::MissingRequest)?;
        if request.state != C220LsuWriteState::InFlight {
            return Err(C220LsuWriteError::InvalidState);
        }
        request.state = C220LsuWriteState::Responding;
        self.outstanding -= 1;
        Ok(request.line)
    }

    pub fn finish_response(&mut self, id: C220LsuWriteId) -> Result<(), C220LsuWriteError> {
        let request = self
            .requests
            .get(&id)
            .ok_or(C220LsuWriteError::MissingRequest)?;
        if request.state != C220LsuWriteState::Responding {
            return Err(C220LsuWriteError::InvalidState);
        }
        if request.line.memory == C220LsuMemory::Ub {
            self.ub_addresses.remove(&request.line.address);
        }
        self.requests.remove(&id);
        Ok(())
    }
}
