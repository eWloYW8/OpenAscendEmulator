use std::collections::VecDeque;

use super::{
    C220L1Arbiter, C220L1Cycle, C220L1Error, C220L1Geometry, C220L1Port, C220L1Request,
    C220L1Response,
};

/// Services requests that have already passed the input transport latency.
/// Denied requests remain owned by the caller and must be offered again.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220L1Pipeline {
    arbiter: C220L1Arbiter,
    pending: [VecDeque<C220L1Response>; 3],
    last_tick: Option<u64>,
}

impl C220L1Pipeline {
    pub fn new(geometry: C220L1Geometry) -> Self {
        Self {
            arbiter: C220L1Arbiter::new(geometry),
            pending: std::array::from_fn(|_| VecDeque::new()),
            last_tick: None,
        }
    }

    pub const fn arbiter(&self) -> &C220L1Arbiter {
        &self.arbiter
    }

    pub fn pending(&self, port: C220L1Port) -> &VecDeque<C220L1Response> {
        &self.pending[port as usize]
    }

    pub fn is_idle(&self) -> bool {
        self.pending.iter().all(VecDeque::is_empty)
    }

    /// At most one response per port is emitted. A blocked head is retained.
    pub fn step(
        &mut self,
        tick: u64,
        requests: [Option<C220L1Request>; 3],
        response_ready: [bool; 3],
    ) -> Result<C220L1Cycle, C220L1Error> {
        self.step_notified(tick, requests, [true; 3], response_ready)
    }

    pub(super) fn step_notified(
        &mut self,
        tick: u64,
        requests: [Option<C220L1Request>; 3],
        receiver_notified: [bool; 3],
        response_ready: [bool; 3],
    ) -> Result<C220L1Cycle, C220L1Error> {
        if let Some(previous) = self.last_tick
            && tick <= previous
        {
            return Err(C220L1Error::NonIncreasingTick {
                previous,
                requested: tick,
            });
        }
        let mut ready_ticks = [0; 3];
        for port in C220L1Port::ALL {
            if requests[port as usize].is_some() && receiver_notified[port as usize] {
                ready_ticks[port as usize] = tick
                    .checked_add(port.response_ticks())
                    .ok_or(C220L1Error::TimeOverflow)?;
            }
        }
        let responses = std::array::from_fn(|index| {
            if response_ready[index]
                && self.pending[index]
                    .front()
                    .is_some_and(|head| head.ready_tick <= tick)
            {
                self.pending[index].pop_front()
            } else {
                None
            }
        });
        let decisions = if requests
            .iter()
            .zip(receiver_notified)
            .any(|(r, notified)| r.is_some() && notified)
        {
            self.arbiter
                .arbitrate(requests.map(|request| request.map(|r| r.access)))
        } else {
            [None; 3]
        };
        let mut accepted = [false; 3];
        for port in C220L1Port::ALL {
            let index = port as usize;
            if let (Some(request), Some(decision)) = (requests[index], decisions[index])
                && decision.granted
                && receiver_notified[index]
            {
                accepted[index] = true;
                self.pending[index].push_back(C220L1Response {
                    request,
                    accepted_tick: tick,
                    ready_tick: ready_ticks[index],
                    bank_mask: decision.bank_mask,
                });
            }
        }
        self.last_tick = Some(tick);
        Ok(C220L1Cycle {
            tick,
            decisions,
            accepted,
            responses,
        })
    }
}
