use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::num::NonZeroU32;

use super::timed_memory::{
    C220MemoryReadBeat, C220MemoryReadCommand, C220MemoryReadId, C220MemoryReadReturn,
};
mod cache;
use cache::CachePort;
pub use cache::{C220BiuReadCacheConfig, C220BiuReadCacheKind};

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum C220BiuBusReadError {
    #[error("BIU bus read time overflowed")]
    TimeOverflow,
    #[error("BIU bus read tag {0:?} is already in use or has no data")]
    InvalidRequest(C220MemoryReadId),
    #[error("invalid or busy BIU cache read endpoint")]
    InvalidPort,
    #[error("BIU bus read response is early, duplicate, or outside its request: {0:?}")]
    InvalidResponse(C220MemoryReadBeat),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Transaction {
    expected: u32,
    sent: bool,
    received: BTreeSet<u32>,
    delivered: u32,
}

/// Shared read route through the core BIU.
/// Downstream acceptance owns the memory endpoint's capacity and delay.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220BiuBusReads {
    limit: NonZeroU32,
    outstanding: u32,
    input: VecDeque<C220MemoryReadCommand>,
    commands: VecDeque<C220MemoryReadCommand>,
    returns: [VecDeque<C220MemoryReadReturn>; 2],
    upstream: [VecDeque<C220MemoryReadReturn>; 2],
    transactions: BTreeMap<C220MemoryReadId, Transaction>,
    cache_ports: [Vec<CachePort>; 2],
    last_cache: [Option<usize>; 2],
    last_admit: Option<u64>,
    last_return: Option<u64>,
    last_send: Option<u64>,
}

impl C220BiuBusReads {
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
            cache_ports: Default::default(),
            last_cache: [None; 2],
            last_admit: None,
            last_return: None,
        }
    }

    pub fn outstanding(&self) -> u32 {
        self.outstanding
    }

    pub fn inputs(&self) -> &VecDeque<C220MemoryReadCommand> {
        &self.input
    }

    pub fn queued_commands(&self) -> &VecDeque<C220MemoryReadCommand> {
        &self.commands
    }

    pub fn returns(&self) -> &[VecDeque<C220MemoryReadReturn>; 2] {
        &self.returns
    }

    pub fn upstream(&self) -> &[VecDeque<C220MemoryReadReturn>; 2] {
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
        mut request: C220MemoryReadCommand,
    ) -> Result<(), C220BiuBusReadError> {
        let bytes = request.bytes;
        if !matches!(request.tag, C220MemoryReadId::Mte(_))
            || bytes == 0
            || self.transactions.contains_key(&request.tag)
        {
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
        request.ready_tick = ready_tick;
        self.input.push_back(request);
        Ok(())
    }

    pub(crate) fn advance_input(&mut self, tick: u64) -> Result<(), C220BiuBusReadError> {
        let ready_tick = next_tick(tick)?;
        if self.last_admit == Some(tick) || self.outstanding >= self.limit.get() {
            return Ok(());
        }
        let command = self
            .select_cache(tick)
            .or_else(|| self.input.pop_front_if(|head| head.ready_tick <= tick));
        if let Some(mut command) = command {
            self.last_admit = Some(tick);
            command.ready_tick = ready_tick;
            self.commands.push_back(command);
        }
        Ok(())
    }

    pub(crate) fn take_command(&mut self, tick: u64) -> Option<C220MemoryReadCommand> {
        if self.last_send == Some(tick) {
            return None;
        }
        let command = self.commands.pop_front_if(|head| head.ready_tick <= tick)?;
        self.transactions
            .get_mut(&command.tag)
            .expect("queued read")
            .sent = true;
        self.outstanding += 1;
        self.last_send = Some(tick);
        Some(command)
    }

    pub(crate) fn receive(
        &mut self,
        tick: u64,
        heads: [Option<C220MemoryReadBeat>; 2],
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
                self.returns[port].push_back(C220MemoryReadReturn { ready_tick, beat });
            }
        }
        Ok(accepted)
    }

    pub(crate) fn advance_returns(&mut self, tick: u64) -> Result<(), C220BiuBusReadError> {
        let ready_tick = next_tick(tick)?;
        if self.last_return == Some(tick) {
            return Ok(());
        }
        self.last_return = Some(tick);
        for port in 0..2 {
            if self.returns[port]
                .front()
                .is_some_and(|head| !matches!(head.beat.tag, C220MemoryReadId::Mte(_)))
            {
                self.forward_cache_return(tick, port)?;
                continue;
            }
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

    pub(crate) fn heads(&self, tick: u64) -> [Option<C220MemoryReadBeat>; 2] {
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
        let mut bus = C220BiuBusReads::new(NonZeroU32::new(1).unwrap());
        let tag = C220MemoryReadId::Mte(NonZeroU32::new(1).unwrap());
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
        let beat = |transaction_id| C220MemoryReadBeat {
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

        let mut bus = C220BiuBusReads::new(NonZeroU32::new(8).unwrap());
        let config = C220BiuReadCacheConfig {
            request_capacity: NonZeroU32::new(2).unwrap(),
            response_capacity: NonZeroU32::new(1).unwrap(),
            request_latency: 1,
            response_latency: 1,
        };
        for kind in [
            C220BiuReadCacheKind::Instruction,
            C220BiuReadCacheKind::Instruction,
            C220BiuReadCacheKind::Data,
        ] {
            bus.add_cache_port(kind, config).unwrap();
        }
        let tags = [
            C220MemoryReadId::InstructionCache {
                port: 0,
                transaction: 1,
            },
            C220MemoryReadId::InstructionCache {
                port: 1,
                transaction: 1,
            },
            C220MemoryReadId::DataCache {
                port: 0,
                transaction: 1,
            },
            C220MemoryReadId::DataCache {
                port: 0,
                transaction: 2,
            },
            C220MemoryReadId::Mte(NonZeroU32::new(1).unwrap()),
        ];
        for tag in tags {
            let command = C220MemoryReadCommand {
                ready_tick: 0,
                tag,
                address: 0x2000,
                bytes: 64,
            };
            if matches!(tag, C220MemoryReadId::Mte(_)) {
                bus.push(0, command).unwrap();
            } else {
                assert!(bus.send_cache_command(0, command).unwrap());
            }
        }
        for (index, tag) in tags.into_iter().enumerate() {
            let tick = index as u64 + 1;
            bus.advance_input(tick).unwrap();
            bus.advance_input(tick).unwrap();
            assert_eq!(bus.take_command(tick + 1).unwrap().tag, tag);
            assert!(bus.take_command(tick + 1).is_none());
        }
        assert_eq!(bus.outstanding(), 5);
        let response = |tag| C220MemoryReadBeat {
            tag,
            transaction_id: 0,
        };
        bus.receive(7, [Some(response(tags[2])), Some(response(tags[3]))])
            .unwrap();
        bus.advance_returns(8).unwrap();
        assert_eq!(bus.outstanding(), 4);
        assert_eq!(bus.returns[1].len(), 1);
        assert_eq!(
            bus.take_cache_return(9, C220BiuReadCacheKind::Data, 0),
            Some(response(tags[2]))
        );
        bus.advance_returns(9).unwrap();
        assert_eq!(bus.outstanding(), 3);
        assert_eq!(
            bus.take_cache_return(10, C220BiuReadCacheKind::Data, 0),
            Some(response(tags[3]))
        );
        bus.receive(10, [Some(response(tags[0])), Some(response(tags[1]))])
            .unwrap();
        bus.advance_returns(11).unwrap();
        assert_eq!(bus.outstanding(), 1);
        for port in 0..2 {
            assert!(
                bus.take_cache_return(12, C220BiuReadCacheKind::Instruction, port)
                    .is_some()
            );
        }
        bus.receive(12, [Some(response(tags[4])), None]).unwrap();
        bus.advance_returns(13).unwrap();
        assert_eq!(bus.outstanding(), 0);
        assert_eq!(bus.heads(14), [Some(response(tags[4])), None]);
        bus.consume([true, false]);
        assert!(bus.is_idle());
    }
}
