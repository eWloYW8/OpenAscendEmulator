use std::collections::VecDeque;

use super::{
    C220L1Arbiter, C220L1Cycle, C220L1Decision, C220L1Error, C220L1Geometry, C220L1Port,
    C220L1Receiver, C220L1Request, C220L1RequestCycle, C220L1Response, C220L1ResponseCycle,
};

/// Services requests that have already passed the input transport latency.
/// Denied requests remain owned by the caller and must be offered again.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220L1Pipeline {
    arbiter: C220L1Arbiter,
    pending: [VecDeque<C220L1Response>; 3],
    observed_tick: Option<u64>,
    receive_ticks: [Option<u64>; 2],
    response_ticks: [Option<u64>; 2],
    arbitration_tick: Option<u64>,
    decisions: [Option<C220L1Decision>; 3],
}

impl C220L1Pipeline {
    pub fn new(geometry: C220L1Geometry) -> Self {
        Self {
            arbiter: C220L1Arbiter::new(geometry),
            pending: std::array::from_fn(|_| VecDeque::new()),
            observed_tick: None,
            receive_ticks: [None; 2],
            response_ticks: [None; 2],
            arbitration_tick: None,
            decisions: [None; 3],
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
        self.validate_step(tick, requests, receiver_notified)?;
        let mut responses = [None; 3];
        let mut accepted = [false; 3];
        let mut decisions = [None; 3];
        for receiver in C220L1Receiver::ALL {
            let cycle = self.respond(tick, receiver, response_ready)?;
            for (response, emitted) in responses.iter_mut().zip(cycle.responses) {
                *response = response.or(emitted);
            }
        }
        let mut heads = requests;
        for receiver in C220L1Receiver::ALL {
            let notified = C220L1Port::ALL
                .into_iter()
                .any(|port| receiver.contains(port) && receiver_notified[port as usize]);
            if notified {
                let cycle = self.receive(tick, receiver, heads)?;
                for port in C220L1Port::ALL {
                    let index = port as usize;
                    if receiver.contains(port) || decisions[index].is_none() {
                        decisions[index] = cycle.decisions[index];
                    }
                    accepted[index] |= cycle.accepted[index];
                    if cycle.accepted[index] {
                        heads[index] = None;
                    }
                }
            }
        }
        Ok(C220L1Cycle {
            tick,
            decisions,
            accepted,
            responses,
        })
    }

    pub(super) fn validate_step(
        &self,
        tick: u64,
        requests: [Option<C220L1Request>; 3],
        receiver_notified: [bool; 3],
    ) -> Result<(), C220L1Error> {
        for receiver in C220L1Receiver::ALL {
            self.check_callback(tick, receiver, false)?;
            self.check_callback(tick, receiver, true)?;
        }
        for port in C220L1Port::ALL {
            if requests[port as usize].is_some() && receiver_notified[port as usize] {
                tick.checked_add(port.response_ticks())
                    .ok_or(C220L1Error::TimeOverflow)?;
            }
        }
        Ok(())
    }

    /// Invoke only when this receiver is notified. The first nonempty receiver
    /// snapshots and arbitrates all heads, including those whose transport
    /// notification is not yet due. Decisions use a low-32-bit time stamp:
    /// tick zero keeps the initial empty grants, and beyond that stamp's range
    /// every arbitration call refreshes the snapshot, even within one callback.
    /// Write notification scans both write ports; it is not a per-port callback.
    pub fn receive(
        &mut self,
        tick: u64,
        receiver: C220L1Receiver,
        mut requests: [Option<C220L1Request>; 3],
    ) -> Result<C220L1RequestCycle, C220L1Error> {
        self.check_callback(tick, receiver, false)?;
        let mut ready_ticks = [0; 3];
        for port in C220L1Port::ALL {
            if receiver.contains(port) && requests[port as usize].is_some() {
                ready_ticks[port as usize] = tick
                    .checked_add(port.response_ticks())
                    .ok_or(C220L1Error::TimeOverflow)?;
            }
        }
        let mut decisions = [None; 3];
        let mut accepted = [false; 3];
        for port in C220L1Port::ALL {
            let index = port as usize;
            if !receiver.contains(port) || requests[index].is_none() {
                continue;
            }
            if u64::from(self.arbitration_tick.unwrap_or(0) as u32) < tick {
                self.decisions = self
                    .arbiter
                    .arbitrate(requests.map(|r| r.map(|r| r.access)));
                self.arbitration_tick = Some(tick);
            }
            decisions[index] = self.decisions[index];
            if let (Some(request), Some(decision)) = (requests[index], decisions[index])
                && decision.granted
            {
                accepted[index] = true;
                self.pending[index].push_back(C220L1Response {
                    request,
                    accepted_tick: tick,
                    ready_tick: ready_ticks[index],
                    bank_mask: decision.bank_mask,
                });
                requests[index] = None;
            }
        }
        for port in C220L1Port::ALL {
            if !receiver.contains(port) && self.arbitration_tick == Some(tick) {
                decisions[port as usize] = self.decisions[port as usize];
            }
        }
        self.observed_tick = Some(tick);
        self.receive_ticks[receiver as usize] = Some(tick);
        Ok(C220L1RequestCycle {
            tick,
            decisions,
            accepted,
        })
    }

    /// Forward at most one eligible response from each port in the receiver.
    pub fn respond(
        &mut self,
        tick: u64,
        receiver: C220L1Receiver,
        response_ready: [bool; 3],
    ) -> Result<C220L1ResponseCycle, C220L1Error> {
        self.check_callback(tick, receiver, true)?;
        let responses = std::array::from_fn(|index| {
            if receiver.contains(C220L1Port::ALL[index]) && response_ready[index] {
                self.pending[index].pop_front_if(|head| head.ready_tick <= tick)
            } else {
                None
            }
        });
        self.observed_tick = Some(tick);
        self.response_ticks[receiver as usize] = Some(tick);
        Ok(C220L1ResponseCycle { tick, responses })
    }

    fn check_callback(
        &self,
        tick: u64,
        receiver: C220L1Receiver,
        response: bool,
    ) -> Result<(), C220L1Error> {
        if let Some(previous) = self.observed_tick
            && tick < previous
        {
            return Err(C220L1Error::TimeReversed {
                previous,
                requested: tick,
            });
        }
        let previous = if response {
            self.response_ticks
        } else {
            self.receive_ticks
        };
        if previous[receiver as usize] == Some(tick) {
            return Err(C220L1Error::RepeatedCallback {
                receiver,
                phase: if response { "respond" } else { "receive" },
                tick,
            });
        }
        Ok(())
    }
}
