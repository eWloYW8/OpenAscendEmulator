use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::num::NonZeroU32;

use crate::sim::c220::mte::interface::biu_read::C220BiuReadRequest;
use crate::sim::c220::mte::interface::biu_read::returns::C220BiuReadBeat;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220BiuReadCommandTransfer {
    pub ready_tick: u64,
    pub request: C220BiuReadRequest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220BiuReadReturn {
    pub ready_tick: u64,
    pub beat: C220BiuReadBeat,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum C220BiuBusReadError {
    #[error("BIU bus read time overflowed")]
    TimeOverflow,
    #[error("BIU bus read tag {0} is already in use or has no data")]
    InvalidRequest(NonZeroU32),
    #[error("BIU bus read response is early, duplicate, or outside its request: {0:?}")]
    InvalidResponse(C220BiuReadBeat),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Transaction {
    expected: u32,
    sent: bool,
    received: BTreeSet<u32>,
    delivered: u32,
}

/// MTE read route through the core BIU, excluding cache arbitration.
/// Downstream acceptance owns the memory endpoint's capacity and delay.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220BiuMteBusReads {
    limit: NonZeroU32,
    outstanding: u32,
    input: VecDeque<C220BiuReadCommandTransfer>,
    commands: VecDeque<C220BiuReadCommandTransfer>,
    returns: [VecDeque<C220BiuReadReturn>; 2],
    upstream: [VecDeque<C220BiuReadReturn>; 2],
    transactions: BTreeMap<NonZeroU32, Transaction>,
    last_send: Option<u64>,
}

impl C220BiuMteBusReads {
    pub(crate) fn new(limit: NonZeroU32) -> Self {
        Self {
            limit,
            outstanding: 0,
            input: VecDeque::new(),
            commands: VecDeque::new(),
            returns: Default::default(),
            upstream: Default::default(),
            transactions: BTreeMap::new(),
            last_send: None,
        }
    }

    pub fn outstanding(&self) -> u32 {
        self.outstanding
    }

    pub fn inputs(&self) -> &VecDeque<C220BiuReadCommandTransfer> {
        &self.input
    }

    pub fn queued_commands(&self) -> &VecDeque<C220BiuReadCommandTransfer> {
        &self.commands
    }

    pub fn returns(&self) -> &[VecDeque<C220BiuReadReturn>; 2] {
        &self.returns
    }

    pub fn upstream(&self) -> &[VecDeque<C220BiuReadReturn>; 2] {
        &self.upstream
    }

    pub fn is_idle(&self) -> bool {
        self.transactions.is_empty()
    }

    pub(crate) fn can_push(&self) -> bool {
        self.input.len() < 2
    }

    pub(crate) fn push(
        &mut self,
        tick: u64,
        request: C220BiuReadRequest,
    ) -> Result<(), C220BiuBusReadError> {
        let bytes = request.input.generated.request.bytes;
        if bytes == 0 || self.transactions.contains_key(&request.tag) {
            return Err(C220BiuBusReadError::InvalidRequest(request.tag));
        }
        let ready_tick = next_tick(tick)?;
        assert!(self.can_push());
        self.transactions.insert(
            request.tag,
            Transaction {
                expected: bytes.div_ceil(128),
                sent: false,
                received: BTreeSet::new(),
                delivered: 0,
            },
        );
        self.input.push_back(C220BiuReadCommandTransfer {
            ready_tick,
            request,
        });
        Ok(())
    }

    pub(crate) fn advance_input(&mut self, tick: u64) -> Result<(), C220BiuBusReadError> {
        let ready_tick = next_tick(tick)?;
        if self.outstanding < self.limit.get()
            && let Some(mut command) = self.input.pop_front_if(|head| head.ready_tick <= tick)
        {
            command.ready_tick = ready_tick;
            self.commands.push_back(command);
        }
        Ok(())
    }

    pub(crate) fn take_command(&mut self, tick: u64) -> Option<C220BiuReadRequest> {
        if self.last_send == Some(tick) {
            return None;
        }
        let command = self.commands.pop_front_if(|head| head.ready_tick <= tick)?;
        self.transactions
            .get_mut(&command.request.tag)
            .expect("queued read")
            .sent = true;
        self.outstanding += 1;
        self.last_send = Some(tick);
        Some(command.request)
    }

    pub(crate) fn receive(
        &mut self,
        tick: u64,
        heads: [Option<C220BiuReadBeat>; 2],
    ) -> Result<[bool; 2], C220BiuBusReadError> {
        let ready_tick = next_tick(tick)?;
        let accepted =
            std::array::from_fn(|port| heads[port].is_some() && self.returns[port].len() < 3);
        for (port, head) in heads.iter().enumerate() {
            if let Some(beat) = head {
                let valid = self.transactions.get(&beat.tag).is_some_and(|transaction| {
                    transaction.sent
                        && beat.transaction_id < transaction.expected
                        && !transaction.received.contains(&beat.transaction_id)
                });
                if !valid || (port == 1 && accepted[0] && heads[0] == Some(*beat)) {
                    return Err(C220BiuBusReadError::InvalidResponse(*beat));
                }
            }
        }
        for port in 0..2 {
            if accepted[port] {
                let beat = heads[port].expect("accepted response");
                self.transactions
                    .get_mut(&beat.tag)
                    .expect("validated read")
                    .received
                    .insert(beat.transaction_id);
                self.returns[port].push_back(C220BiuReadReturn { ready_tick, beat });
            }
        }
        Ok(accepted)
    }

    pub(crate) fn advance_returns(&mut self, tick: u64) -> Result<(), C220BiuBusReadError> {
        let ready_tick = next_tick(tick)?;
        for port in 0..2 {
            if self.upstream[port].len() < 2
                && let Some(mut response) =
                    self.returns[port].pop_front_if(|head| head.ready_tick <= tick)
            {
                response.ready_tick = ready_tick;
                if response.beat.transaction_id == 0 {
                    self.outstanding -= 1;
                }
                self.upstream[port].push_back(response);
            }
        }
        Ok(())
    }

    pub(crate) fn heads(&self, tick: u64) -> [Option<C220BiuReadBeat>; 2] {
        self.upstream.each_ref().map(|queue| {
            queue
                .front()
                .filter(|head| head.ready_tick <= tick)
                .map(|head| head.beat)
        })
    }

    pub(crate) fn consume(&mut self, accepted: [bool; 2]) {
        for (port, accepted) in accepted.into_iter().enumerate() {
            if accepted {
                let beat = self.upstream[port].pop_front().expect("accepted head").beat;
                let transaction = self.transactions.get_mut(&beat.tag).expect("pending read");
                transaction.delivered += 1;
                if transaction.delivered == transaction.expected {
                    self.transactions.remove(&beat.tag);
                }
            }
        }
    }
}

fn next_tick(tick: u64) -> Result<u64, C220BiuBusReadError> {
    tick.checked_add(1).ok_or(C220BiuBusReadError::TimeOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beat_zero_releases_credit_only_after_upstream_acceptance() {
        let mut bus = C220BiuMteBusReads::new(NonZeroU32::new(1).unwrap());
        let tag = NonZeroU32::new(1).unwrap();
        bus.transactions.insert(
            tag,
            Transaction {
                expected: 4,
                sent: true,
                received: BTreeSet::new(),
                delivered: 0,
            },
        );
        bus.outstanding = 1;
        let beat = |transaction_id| C220BiuReadBeat {
            tag,
            transaction_id,
        };
        for id in [1, 2, 0] {
            assert_eq!(
                bus.receive(0, [Some(beat(id)), None]).unwrap(),
                [true, false]
            );
        }
        assert_eq!(
            bus.receive(0, [Some(beat(3)), None]).unwrap(),
            [false, false]
        );
        bus.advance_returns(0).unwrap();
        assert_eq!(bus.heads(0), [None; 2]);
        bus.advance_returns(1).unwrap();
        assert_eq!(bus.outstanding(), 1);
        assert_eq!(bus.heads(1), [None; 2]);
        bus.advance_returns(2).unwrap();
        bus.advance_returns(3).unwrap();
        assert_eq!(bus.outstanding(), 1);
        assert_eq!(bus.returns()[0].front().unwrap().beat, beat(0));
        assert_eq!(bus.heads(3), [Some(beat(1)), None]);
        bus.consume([true, false]);
        bus.advance_returns(4).unwrap();
        assert_eq!(bus.outstanding(), 0);
        assert!(!bus.is_idle());
        assert!(bus.receive(4, [None, Some(beat(0))]).is_err());
        assert_eq!(
            bus.receive(4, [None, Some(beat(3))]).unwrap(),
            [false, true]
        );
        bus.advance_returns(5).unwrap();
        assert_eq!(bus.heads(5), [Some(beat(2)), None]);
        bus.consume([true, false]);
        assert_eq!(bus.heads(6), [Some(beat(0)), Some(beat(3))]);
        bus.consume([true; 2]);
        assert!(bus.is_idle());
    }
}
