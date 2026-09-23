use std::collections::{BTreeMap, VecDeque};

use crate::sim::c220::memory::l1::{
    C220L1Access, C220L1Port, C220L1Request, C220L1Transport, C220L1TransportError,
};
use crate::sim::c220::mte::interface::C220MteOutputFragment;

mod events;
pub use events::{C220FixpL1WriteCallback, C220FixpL1WriteEvent, C220FixpL1WriteEvents};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FixpL1WriteEntry {
    pub ready_tick: u64,
    pub fragment: C220MteOutputFragment,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FixpL1WriteRequest {
    /// Interface-local transport identity, independent of the source fragment.
    pub id: u64,
    pub fragment: C220MteOutputFragment,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220FixpL1WriteSend {
    Idle,
    Delayed { ready_tick: u64 },
    TransportFull,
    Sent(C220FixpL1WriteRequest),
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum C220FixpL1WriteError {
    #[error("FIX L1 write time reversed from {previous} to {requested}")]
    TimeReversed { previous: u64, requested: u64 },
    #[error("FIX L1 write {phase} callback repeated at tick {tick}")]
    RepeatedCallback { phase: &'static str, tick: u64 },
    #[error("FIX L1 write time or request identity overflowed")]
    Overflow,
    #[error("FIX L1 response {0} has no outstanding request")]
    UnknownResponse(u64),
    #[error(transparent)]
    Transport(#[from] C220L1TransportError),
}

/// Dedicated FIX write client. Its unbounded, one-tick input queue is distinct
/// from the bounded L1 transport and the ordinary MTE write client's ports.
/// Memory service is advanced by the owner, allowing other L1 clients to
/// contend. Sending never completes an instruction; only returned responses
/// enter the one-tick retirement queue. This interface moves timing metadata,
/// not numerical bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220FixpL1WriteInterface {
    input: VecDeque<C220FixpL1WriteEntry>,
    outstanding: BTreeMap<u64, C220FixpL1WriteRequest>,
    acknowledged: VecDeque<C220FixpL1WriteEntry>,
    next_id: u64,
    observed_tick: Option<u64>,
    callbacks: [Option<u64>; 3],
}

impl Default for C220FixpL1WriteInterface {
    fn default() -> Self {
        Self {
            input: VecDeque::new(),
            outstanding: BTreeMap::new(),
            acknowledged: VecDeque::new(),
            next_id: 1,
            observed_tick: None,
            callbacks: [None; 3],
        }
    }
}

impl C220FixpL1WriteInterface {
    pub fn input(&self) -> &VecDeque<C220FixpL1WriteEntry> {
        &self.input
    }

    pub fn outstanding(&self) -> impl Iterator<Item = &C220FixpL1WriteRequest> {
        self.outstanding.values()
    }

    pub fn acknowledgments(&self) -> &VecDeque<C220FixpL1WriteEntry> {
        &self.acknowledged
    }

    pub fn is_idle(&self) -> bool {
        self.input.is_empty() && self.outstanding.is_empty() && self.acknowledged.is_empty()
    }

    pub fn enqueue(
        &mut self,
        tick: u64,
        fragment: C220MteOutputFragment,
    ) -> Result<(), C220FixpL1WriteError> {
        self.check_time(tick)?;
        let ready_tick = tick.checked_add(1).ok_or(C220FixpL1WriteError::Overflow)?;
        self.input.push_back(C220FixpL1WriteEntry {
            ready_tick,
            fragment,
        });
        self.observed_tick = Some(tick);
        Ok(())
    }

    pub fn send(
        &mut self,
        tick: u64,
        memory: &mut C220L1Transport,
    ) -> Result<C220FixpL1WriteSend, C220FixpL1WriteError> {
        self.begin(tick, 0, "send")?;
        let Some(head) = self.input.front().copied() else {
            return Ok(C220FixpL1WriteSend::Idle);
        };
        if head.ready_tick > tick {
            return Ok(C220FixpL1WriteSend::Delayed {
                ready_tick: head.ready_tick,
            });
        }
        let request = C220FixpL1WriteRequest {
            id: self.next_id,
            fragment: head.fragment,
        };
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or(C220FixpL1WriteError::Overflow)?;
        if !memory.send_request(
            tick,
            C220L1Port::FixpWrite,
            C220L1Request {
                id: request.id,
                access: C220L1Access {
                    address: request.fragment.destination_address,
                    bytes: request.fragment.bytes,
                },
            },
        )? {
            return Ok(C220FixpL1WriteSend::TransportFull);
        }
        self.outstanding.insert(request.id, request);
        self.input.pop_front();
        Ok(C220FixpL1WriteSend::Sent(request))
    }

    pub fn receive(
        &mut self,
        tick: u64,
        memory: &mut C220L1Transport,
    ) -> Result<Option<C220FixpL1WriteRequest>, C220FixpL1WriteError> {
        self.begin(tick, 1, "receive")?;
        let Some(response) = memory
            .responses(C220L1Port::FixpWrite)
            .front()
            .filter(|response| response.ready_tick <= tick)
        else {
            return Ok(None);
        };
        let id = response.payload.request.id;
        let request = *self
            .outstanding
            .get(&id)
            .ok_or(C220FixpL1WriteError::UnknownResponse(id))?;
        let ready_tick = tick.checked_add(1).ok_or(C220FixpL1WriteError::Overflow)?;
        memory.receive_response(tick, C220L1Port::FixpWrite)?;
        self.outstanding.remove(&id);
        if request.fragment.last_in_uop {
            self.acknowledged.push_back(C220FixpL1WriteEntry {
                ready_tick,
                fragment: request.fragment,
            });
        }
        Ok(Some(request))
    }

    /// Returns every completed logical fragment. The owner retires a command
    /// only when the returned fragment has `last_in_instruction` set.
    pub fn retire(
        &mut self,
        tick: u64,
    ) -> Result<Option<C220FixpL1WriteEntry>, C220FixpL1WriteError> {
        self.begin(tick, 2, "retire")?;
        Ok(self
            .acknowledged
            .pop_front_if(|head| head.ready_tick <= tick))
    }

    fn check_time(&self, tick: u64) -> Result<(), C220FixpL1WriteError> {
        if let Some(previous) = self.observed_tick
            && tick < previous
        {
            return Err(C220FixpL1WriteError::TimeReversed {
                previous,
                requested: tick,
            });
        }
        Ok(())
    }

    fn begin(
        &mut self,
        tick: u64,
        index: usize,
        phase: &'static str,
    ) -> Result<(), C220FixpL1WriteError> {
        self.check_time(tick)?;
        if self.callbacks[index] == Some(tick) {
            return Err(C220FixpL1WriteError::RepeatedCallback { phase, tick });
        }
        self.callbacks[index] = Some(tick);
        self.observed_tick = Some(tick);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::c220::memory::l1::C220L1Geometry;

    #[test]
    fn bounded_transport_holds_input_and_completion_waits_for_response() {
        let mut interface = C220FixpL1WriteInterface::default();
        let mut memory = C220L1Transport::new(C220L1Geometry::new(32, 16, 2, 9).unwrap());
        for index in 0..3 {
            interface
                .enqueue(
                    0,
                    C220MteOutputFragment {
                        instruction_id: 7,
                        request_id: index,
                        destination_address: index * 256,
                        bytes: 256,
                        last_in_uop: true,
                        last_in_instruction: index == 2,
                    },
                )
                .unwrap();
        }
        assert_eq!(
            interface.send(0, &mut memory).unwrap(),
            C220FixpL1WriteSend::Delayed { ready_tick: 1 }
        );
        for tick in 1..=2 {
            assert!(matches!(
                interface.send(tick, &mut memory).unwrap(),
                C220FixpL1WriteSend::Sent(_)
            ));
        }
        assert_eq!(
            interface.send(3, &mut memory).unwrap(),
            C220FixpL1WriteSend::TransportFull
        );
        assert_eq!(interface.input().len(), 1);
        assert_eq!(interface.outstanding().count(), 2);
        assert_eq!(interface.retire(3).unwrap(), None);
        let mut received = BTreeMap::new();
        let mut retired = Vec::new();
        for tick in 4..30 {
            memory.advance(tick).unwrap();
            interface.send(tick, &mut memory).unwrap();
            if let Some(request) = interface.receive(tick, &mut memory).unwrap() {
                received.insert(request.fragment.request_id, tick);
            }
            if let Some(entry) = interface.retire(tick).unwrap() {
                assert_eq!(tick, received[&entry.fragment.request_id] + 1);
                retired.push(entry.fragment);
            }
            if interface.is_idle() && memory.is_idle() {
                break;
            }
        }
        assert_eq!(retired.len(), 3);
        assert_eq!(
            retired
                .iter()
                .map(|fragment| fragment.request_id)
                .collect::<Vec<_>>(),
            [0, 1, 2]
        );
        assert!(!retired[0].last_in_instruction && !retired[1].last_in_instruction);
        assert!(retired[2].last_in_instruction);
        assert!(interface.is_idle() && memory.is_idle());
    }
}
