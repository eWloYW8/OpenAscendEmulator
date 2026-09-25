use std::collections::{BTreeMap, VecDeque};

use super::store_buffer::{C220LsuLineKey, C220LsuMemory};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct C220LsuReadId(u64);

impl C220LsuReadId {
    pub(crate) const fn from_sequence(value: u64) -> Self {
        Self(value)
    }

    pub const fn sequence(self) -> u64 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220LsuReadOwner {
    Miss,
    Store,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220LsuReadState {
    Queued,
    InFlight,
    Responding,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220LsuReadRequest {
    pub id: C220LsuReadId,
    pub line: C220LsuLineKey,
    pub partition_address: u64,
    pub owner: C220LsuReadOwner,
    pub state: C220LsuReadState,
    pub queued_tick: u64,
    pub byte_len: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum C220LsuReadError {
    #[error("read request sequence or clock exhausted")]
    Overflow,
    #[error("read request does not exist")]
    MissingRequest,
    #[error("read request is not in the required state")]
    InvalidState,
    #[error("read-queue clock reversed from {previous} to {requested}")]
    TimeReversal { previous: u64, requested: u64 },
}

/// Lower-level reads retain their initiating buffer across coalesced accesses.
/// UB and external ports share credits, but retain independent FIFO heads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220LsuReadQueue {
    capacity: u32,
    line_bytes: usize,
    outstanding: u32,
    next_id: u64,
    queues: [VecDeque<C220LsuReadId>; 2],
    requests: BTreeMap<C220LsuReadId, C220LsuReadRequest>,
    tick: u64,
    last_dispatch: Option<u64>,
}

impl C220LsuReadQueue {
    pub fn new(capacity: u32, line_bytes: usize) -> Self {
        Self {
            capacity,
            line_bytes,
            outstanding: 0,
            next_id: 0,
            queues: Default::default(),
            requests: BTreeMap::new(),
            tick: 0,
            last_dispatch: None,
        }
    }

    pub const fn outstanding(&self) -> u32 {
        self.outstanding
    }

    pub fn requests(&self) -> impl Iterator<Item = &C220LsuReadRequest> {
        self.requests.values()
    }

    pub fn request(&self, id: C220LsuReadId) -> Option<&C220LsuReadRequest> {
        self.requests.get(&id)
    }

    pub fn check_tick(&self, tick: u64) -> Result<(), C220LsuReadError> {
        if tick < self.tick {
            return Err(C220LsuReadError::TimeReversal {
                previous: self.tick,
                requested: tick,
            });
        }
        Ok(())
    }

    pub fn advance_to(&mut self, tick: u64) -> Result<(), C220LsuReadError> {
        self.check_tick(tick)?;
        self.tick = tick;
        Ok(())
    }

    pub fn check_enqueue(&self) -> Result<(), C220LsuReadError> {
        if self.tick == u64::MAX || self.next_id == u64::MAX {
            return Err(C220LsuReadError::Overflow);
        }
        Ok(())
    }

    pub fn enqueue(
        &mut self,
        line: C220LsuLineKey,
        partition_address: u64,
        owner: C220LsuReadOwner,
    ) -> Result<C220LsuReadId, C220LsuReadError> {
        self.check_enqueue()?;
        let id = C220LsuReadId(self.next_id);
        self.next_id += 1;
        self.requests.insert(
            id,
            C220LsuReadRequest {
                id,
                line,
                partition_address,
                owner,
                state: C220LsuReadState::Queued,
                queued_tick: self.tick,
                byte_len: self.line_bytes,
            },
        );
        let port = usize::from(line.memory == C220LsuMemory::External);
        self.queues[port].push_back(id);
        Ok(id)
    }

    pub fn next_ready_tick(&self) -> Option<u64> {
        let next_process = self
            .last_dispatch
            .map_or(Some(0), |tick| tick.checked_add(1))?;
        self.queues
            .iter()
            .filter_map(|queue| queue.front())
            .filter_map(|id| self.requests[id].queued_tick.checked_add(1))
            .map(|ready| ready.max(next_process).max(self.tick))
            .min()
    }

    /// Either aged head wakes the shared process. Once awake, it can also send
    /// the other port's younger head; each port sends at most once per tick.
    pub fn dispatch_clock(
        &mut self,
        tick: u64,
        ub_accepts: bool,
        external_accepts: bool,
    ) -> Result<Vec<C220LsuReadRequest>, C220LsuReadError> {
        self.check_tick(tick)?;
        let ready = self.next_ready_tick().is_some_and(|ready| ready <= tick);
        self.tick = tick;
        if !ready || self.last_dispatch == Some(tick) {
            return Ok(Vec::new());
        }
        self.last_dispatch = Some(tick);
        let mut sent = Vec::with_capacity(2);
        for (queue, accepts) in self.queues.iter_mut().zip([ub_accepts, external_accepts]) {
            if self.outstanding >= self.capacity {
                break;
            }
            if accepts && let Some(id) = queue.pop_front() {
                let request = self.requests.get_mut(&id).expect("queued read exists");
                request.state = C220LsuReadState::InFlight;
                self.outstanding += 1;
                sent.push(*request);
            }
        }
        Ok(sent)
    }

    pub fn begin_response(
        &mut self,
        id: C220LsuReadId,
    ) -> Result<C220LsuReadRequest, C220LsuReadError> {
        let request = self
            .requests
            .get_mut(&id)
            .ok_or(C220LsuReadError::MissingRequest)?;
        if request.state != C220LsuReadState::InFlight {
            return Err(C220LsuReadError::InvalidState);
        }
        request.state = C220LsuReadState::Responding;
        self.outstanding -= 1;
        Ok(*request)
    }

    pub fn finish_response(&mut self, id: C220LsuReadId) -> Result<(), C220LsuReadError> {
        let request = self
            .requests
            .get(&id)
            .ok_or(C220LsuReadError::MissingRequest)?;
        if request.state != C220LsuReadState::Responding {
            return Err(C220LsuReadError::InvalidState);
        }
        self.requests.remove(&id);
        Ok(())
    }
}
