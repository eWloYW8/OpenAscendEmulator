use std::collections::VecDeque;

use super::{
    C220L1Cycle, C220L1Error, C220L1Geometry, C220L1Pipeline, C220L1Port, C220L1Request,
    C220L1Response,
};

pub const C220_L1_TRANSPORT_CAPACITY: usize = 2;
pub const C220_L1_TRANSPORT_TICKS: u64 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220L1Transit<T> {
    pub sent_tick: u64,
    pub ready_tick: u64,
    pub payload: T,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum C220L1TransportError {
    #[error("L1 transport time reversed from {previous} to {requested}")]
    TimeReversed { previous: u64, requested: u64 },
    #[error("L1 transport skipped a busy cycle: expected {expected}, got {requested}")]
    SkippedCycle { expected: u64, requested: u64 },
    #[error("L1 transport time overflowed")]
    TimeOverflow,
    #[error(transparent)]
    Service(#[from] C220L1Error),
}

/// Shared L1 service with independent, bounded request and response transports.
/// Producers and consumers explicitly enqueue/dequeue at their callback phase;
/// `advance` runs the L1 receiver and service phase once per cycle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220L1Transport {
    service: C220L1Pipeline,
    requests: [VecDeque<C220L1Transit<C220L1Request>>; 3],
    responses: [VecDeque<C220L1Transit<C220L1Response>>; 3],
    observed_tick: Option<u64>,
    next_service_tick: Option<u64>,
}

impl C220L1Transport {
    pub fn new(geometry: C220L1Geometry) -> Self {
        Self {
            service: C220L1Pipeline::new(geometry),
            requests: std::array::from_fn(|_| VecDeque::new()),
            responses: std::array::from_fn(|_| VecDeque::new()),
            observed_tick: None,
            next_service_tick: None,
        }
    }

    pub const fn service(&self) -> &C220L1Pipeline {
        &self.service
    }

    pub fn is_idle(&self) -> bool {
        self.service.is_idle()
            && self.requests.iter().all(VecDeque::is_empty)
            && self.responses.iter().all(VecDeque::is_empty)
    }

    pub fn requests(&self, port: C220L1Port) -> &VecDeque<C220L1Transit<C220L1Request>> {
        &self.requests[port as usize]
    }

    pub fn responses(&self, port: C220L1Port) -> &VecDeque<C220L1Transit<C220L1Response>> {
        &self.responses[port as usize]
    }

    pub fn request_ready(&self, port: C220L1Port) -> bool {
        self.requests[port as usize].len() < C220_L1_TRANSPORT_CAPACITY
    }

    /// False leaves ownership of the request with the producer.
    pub fn send_request(
        &mut self,
        tick: u64,
        port: C220L1Port,
        request: C220L1Request,
    ) -> Result<bool, C220L1TransportError> {
        self.check_time(tick)?;
        if !self.request_ready(port) {
            self.observed_tick = Some(tick);
            return Ok(false);
        }
        let ready_tick = transport_ready(tick)?;
        if self.is_idle() {
            self.next_service_tick = Some(self.next_service_tick.unwrap_or(tick).max(tick));
        }
        self.requests[port as usize].push_back(C220L1Transit {
            sent_tick: tick,
            ready_tick,
            payload: request,
        });
        self.observed_tick = Some(tick);
        Ok(true)
    }

    pub fn receive_response(
        &mut self,
        tick: u64,
        port: C220L1Port,
    ) -> Result<Option<C220L1Response>, C220L1TransportError> {
        self.check_time(tick)?;
        let response = self.responses[port as usize].pop_front_if(|head| head.ready_tick <= tick);
        self.observed_tick = Some(tick);
        Ok(response.map(|transit| transit.payload))
    }

    pub fn advance(&mut self, tick: u64) -> Result<C220L1Cycle, C220L1TransportError> {
        self.check_time(tick)?;
        let next_service_tick = tick
            .checked_add(1)
            .ok_or(C220L1TransportError::TimeOverflow)?;
        let heads =
            std::array::from_fn(|index| self.requests[index].front().map(|head| head.payload));
        let notified: [bool; 3] = std::array::from_fn(|index| {
            self.requests[index]
                .front()
                .is_some_and(|head| head.ready_tick <= tick)
        });
        // Both write ports wake the same receiver, which scans both queue heads.
        let write_notified = notified[0] || notified[1];
        let receivers = [write_notified, write_notified, notified[2]];
        let credits =
            std::array::from_fn(|index| self.responses[index].len() < C220_L1_TRANSPORT_CAPACITY);
        let cycle = self
            .service
            .step_notified(tick, heads, receivers, credits)?;
        for index in 0..3 {
            if cycle.accepted[index] {
                self.requests[index].pop_front();
            }
            if let Some(payload) = cycle.responses[index] {
                self.responses[index].push_back(C220L1Transit {
                    sent_tick: tick,
                    ready_tick: next_service_tick,
                    payload,
                });
            }
        }
        self.next_service_tick = Some(next_service_tick);
        self.observed_tick = Some(tick);
        Ok(cycle)
    }

    fn check_time(&self, tick: u64) -> Result<(), C220L1TransportError> {
        if let Some(previous) = self.observed_tick
            && tick < previous
        {
            return Err(C220L1TransportError::TimeReversed {
                previous,
                requested: tick,
            });
        }
        if !self.is_idle()
            && let Some(expected) = self.next_service_tick
            && tick > expected
        {
            return Err(C220L1TransportError::SkippedCycle {
                expected,
                requested: tick,
            });
        }
        Ok(())
    }
}

fn transport_ready(tick: u64) -> Result<u64, C220L1TransportError> {
    tick.checked_add(C220_L1_TRANSPORT_TICKS)
        .ok_or(C220L1TransportError::TimeOverflow)
}

#[cfg(test)]
mod tests;
