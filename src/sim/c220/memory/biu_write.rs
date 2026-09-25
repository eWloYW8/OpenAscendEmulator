use std::collections::{BTreeMap, VecDeque};
use std::num::NonZeroU32;

use super::timed_memory::{C220MemoryWriteCommand, C220MemoryWriteId, C220MemoryWriteTransfer};

mod cache;
use cache::CachePort;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220BiuWriteReturnKind {
    Dbid,
    Completion,
}

impl C220BiuWriteReturnKind {
    fn index(self) -> usize {
        match self {
            Self::Dbid => 0,
            Self::Completion => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Queued,
    CommandSent,
    DbidQueued,
    DbidDelivered,
    DataQueued,
    DataSent,
    CompletionQueued,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum C220BiuBusWriteError {
    #[error("BIU bus write time overflowed")]
    TimeOverflow,
    #[error("BIU bus write tag {0:?} is not in the expected transaction phase")]
    InvalidPhase(C220MemoryWriteId),
    #[error("BIU cache port is not configured or does not match its request")]
    InvalidPort,
}

/// Shared cache/MTE write route through the core BIU.
/// Taking an outgoing item means the downstream endpoint accepted the send;
/// that endpoint owns its receive capacity, transport delay and memory timing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220BiuBusWrites {
    limit: NonZeroU32,
    outstanding: u32,
    commands: VecDeque<C220MemoryWriteCommand>,
    data: VecDeque<C220MemoryWriteTransfer>,
    returns: [VecDeque<C220MemoryWriteTransfer>; 2],
    upstream: [VecDeque<C220MemoryWriteTransfer>; 2],
    phases: BTreeMap<C220MemoryWriteId, Phase>,
    cache_ports: Vec<CachePort>,
    next_cache: usize,
    last_admission: Option<u64>,
    last_return: [Option<u64>; 2],
    last_cache_data: Option<u64>,
    last_command_send: Option<u64>,
    last_data_send: Option<u64>,
}

impl C220BiuBusWrites {
    pub(crate) fn new(limit: NonZeroU32) -> Self {
        Self {
            limit,
            outstanding: 0,
            commands: VecDeque::new(),
            data: VecDeque::new(),
            returns: Default::default(),
            upstream: Default::default(),
            phases: BTreeMap::new(),
            cache_ports: Vec::new(),
            next_cache: 0,
            last_admission: None,
            last_return: [None; 2],
            last_cache_data: None,
            last_command_send: None,
            last_data_send: None,
        }
    }

    pub fn outstanding(&self) -> u32 {
        self.outstanding
    }

    pub fn queued_commands(&self) -> &VecDeque<C220MemoryWriteCommand> {
        &self.commands
    }

    pub fn queued_data(&self) -> &VecDeque<C220MemoryWriteTransfer> {
        &self.data
    }

    pub fn returns(&self, kind: C220BiuWriteReturnKind) -> &VecDeque<C220MemoryWriteTransfer> {
        &self.returns[kind.index()]
    }

    pub fn upstream(&self, kind: C220BiuWriteReturnKind) -> &VecDeque<C220MemoryWriteTransfer> {
        &self.upstream[kind.index()]
    }

    pub fn is_idle(&self) -> bool {
        self.phases.is_empty() && self.cache_ports.iter().all(|port| port.input.is_empty())
    }

    pub(crate) fn can_receive_command(&self) -> bool {
        self.outstanding < self.limit.get()
    }

    pub(crate) fn can_admit_command(&self, tick: u64) -> bool {
        self.can_receive_command() && self.last_admission != Some(tick)
    }

    pub(crate) fn push_command(
        &mut self,
        tick: u64,
        mut command: C220MemoryWriteCommand,
    ) -> Result<(), C220BiuBusWriteError> {
        let tag = command.tag;
        if self.phases.contains_key(&tag) {
            return Err(C220BiuBusWriteError::InvalidPhase(tag));
        }
        command.ready_tick = next_tick(tick)?;
        self.phases.insert(tag, Phase::Queued);
        self.commands.push_back(command);
        Ok(())
    }

    pub(crate) fn take_command(&mut self, tick: u64) -> Option<C220MemoryWriteCommand> {
        if self.last_command_send == Some(tick) {
            return None;
        }
        let command = self.commands.pop_front_if(|head| head.ready_tick <= tick)?;
        self.phases.insert(command.tag, Phase::CommandSent);
        self.outstanding += 1;
        self.last_command_send = Some(tick);
        Some(command)
    }

    pub(crate) fn push_data(
        &mut self,
        tick: u64,
        mut data: C220MemoryWriteTransfer,
    ) -> Result<(), C220BiuBusWriteError> {
        let tag = data.tag;
        self.require(tag, Phase::DbidDelivered)?;
        data.ready_tick = next_tick(tick)?;
        self.phases.insert(tag, Phase::DataQueued);
        self.data.push_back(data);
        Ok(())
    }

    pub(crate) fn take_data(&mut self, tick: u64) -> Option<C220MemoryWriteTransfer> {
        if self.last_data_send == Some(tick) {
            return None;
        }
        let data = self.data.pop_front_if(|head| head.ready_tick <= tick)?;
        self.phases.insert(data.tag, Phase::DataSent);
        self.last_data_send = Some(tick);
        Some(data)
    }

    pub(crate) fn receive(
        &mut self,
        tick: u64,
        kind: C220BiuWriteReturnKind,
        tag: C220MemoryWriteId,
    ) -> Result<bool, C220BiuBusWriteError> {
        let (expected, next) = match kind {
            C220BiuWriteReturnKind::Dbid => (Phase::CommandSent, Phase::DbidQueued),
            C220BiuWriteReturnKind::Completion => (Phase::DataSent, Phase::CompletionQueued),
        };
        self.require(tag, expected)?;
        let ready_tick = next_tick(tick)?;
        let queue = &mut self.returns[kind.index()];
        if queue.len() == 3 {
            return Ok(false);
        }
        queue.push_back(C220MemoryWriteTransfer { ready_tick, tag });
        self.phases.insert(tag, next);
        Ok(true)
    }

    pub(crate) fn advance(&mut self, tick: u64) -> Result<(), C220BiuBusWriteError> {
        let ready_tick = next_tick(tick)?;
        for index in 0..2 {
            if self.last_return[index] == Some(tick) {
                continue;
            }
            self.last_return[index] = Some(tick);
            let Some(head) = self.returns[index]
                .front()
                .filter(|head| head.ready_tick <= tick)
            else {
                continue;
            };
            let (queue, capacity) = match head.tag {
                C220MemoryWriteId::Mte(_) => (&mut self.upstream[index], 2),
                C220MemoryWriteId::Cache { port, .. } => {
                    let cache = self
                        .cache_ports
                        .get_mut(port as usize)
                        .ok_or(C220BiuBusWriteError::InvalidPort)?;
                    (&mut cache.returns[index], cache.capacities[index])
                }
            };
            if queue.len() < capacity {
                let mut response = self.returns[index].pop_front().expect("ready response");
                response.ready_tick = ready_tick;
                queue.push_back(response);
                if index == 1 {
                    self.outstanding -= 1;
                }
            }
        }
        Ok(())
    }

    pub(crate) fn take_return(
        &mut self,
        tick: u64,
        kind: C220BiuWriteReturnKind,
    ) -> Option<NonZeroU32> {
        let response = self.upstream[kind.index()].pop_front_if(|head| head.ready_tick <= tick)?;
        match kind {
            C220BiuWriteReturnKind::Dbid => {
                self.phases.insert(response.tag, Phase::DbidDelivered);
            }
            C220BiuWriteReturnKind::Completion => {
                self.phases.remove(&response.tag);
            }
        }
        let C220MemoryWriteId::Mte(tag) = response.tag else {
            unreachable!("MTE return port")
        };
        Some(tag)
    }

    fn require(&self, tag: C220MemoryWriteId, expected: Phase) -> Result<(), C220BiuBusWriteError> {
        if self.phases.get(&tag) != Some(&expected) {
            return Err(C220BiuBusWriteError::InvalidPhase(tag));
        }
        Ok(())
    }
}

fn next_tick(tick: u64) -> Result<u64, C220BiuBusWriteError> {
    tick.checked_add(1)
        .ok_or(C220BiuBusWriteError::TimeOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn final_response_backpressure_holds_outstanding_credit() {
        let mut bus = C220BiuBusWrites::new(NonZeroU32::new(4).unwrap());
        let tags = [1, 2, 3, 4].map(|tag| NonZeroU32::new(tag).unwrap());
        for tag in tags {
            bus.phases
                .insert(C220MemoryWriteId::Mte(tag), Phase::DataSent);
        }
        bus.outstanding = 4;
        let completion = C220BiuWriteReturnKind::Completion;
        for tag in &tags[..3] {
            assert!(
                bus.receive(0, completion, C220MemoryWriteId::Mte(*tag))
                    .unwrap()
            );
        }
        assert!(
            !bus.receive(0, completion, C220MemoryWriteId::Mte(tags[3]))
                .unwrap()
        );
        assert!(!bus.can_receive_command());
        bus.advance(0).unwrap();
        assert_eq!(bus.outstanding(), 4);
        bus.advance(1).unwrap();
        assert_eq!(bus.outstanding(), 3);
        assert!(bus.can_receive_command());
        assert!(bus.take_return(1, completion).is_none());
        assert!(
            bus.receive(1, completion, C220MemoryWriteId::Mte(tags[3]))
                .unwrap()
        );
        bus.advance(2).unwrap();
        assert_eq!(bus.outstanding(), 2);
        bus.advance(3).unwrap();
        assert_eq!(bus.outstanding(), 2);
        assert_eq!(bus.returns(completion).len(), 2);
        assert_eq!(bus.upstream(completion).len(), 2);
        assert_eq!(bus.take_return(3, completion), Some(tags[0]));
        bus.advance(4).unwrap();
        assert_eq!(bus.outstanding(), 1);
        assert_eq!(bus.take_return(4, completion), Some(tags[1]));
        bus.advance(5).unwrap();
        assert_eq!(bus.outstanding(), 0);
        assert!(!bus.is_idle());
        assert_eq!(bus.take_return(5, completion), Some(tags[2]));
        assert_eq!(bus.take_return(6, completion), Some(tags[3]));
        assert!(bus.is_idle());
        bus.add_cache_port([1, 1]).unwrap();
        bus.add_cache_port([1, 1]).unwrap();
        let cache_tags = [0, 1].map(|port| C220MemoryWriteId::Cache {
            port,
            transaction: 1,
        });
        let command = |tag| C220MemoryWriteCommand {
            ready_tick: 7,
            tag,
            address: 0,
            bytes: 64,
        };
        let mut cache_heads = cache_tags.map(|tag| Some(command(tag)));
        let mte_tag = C220MemoryWriteId::Mte(tags[0]);
        for (tick, expected) in [(7, cache_tags[0]), (8, cache_tags[1]), (9, mte_tag)] {
            assert_eq!(
                bus.admit_commands(tick, &cache_heads, Some(command(mte_tag)))
                    .unwrap(),
                Some(expected)
            );
            assert_eq!(
                bus.admit_commands(tick, &cache_heads, Some(command(mte_tag)))
                    .unwrap(),
                None
            );
            if let C220MemoryWriteId::Cache { port, .. } = expected {
                cache_heads[port as usize] = None;
            }
            assert_eq!(bus.take_command(tick + 1).unwrap().tag, expected);
        }
        assert_eq!(bus.outstanding(), 3);
        let dbid = C220BiuWriteReturnKind::Dbid;
        for tag in [cache_tags[0], cache_tags[1], mte_tag] {
            assert!(bus.receive(11, dbid, tag).unwrap());
        }
        bus.advance(12).unwrap();
        bus.advance_cache_data(12, false).unwrap();
        assert!(bus.queued_data().is_empty());
        bus.advance(13).unwrap();
        bus.advance_cache_data(13, false).unwrap();
        assert_eq!(bus.queued_data().len(), 2);
        assert!(bus.take_data(13).is_none());
        bus.advance(14).unwrap();
        assert_eq!(bus.take_data(14).unwrap().tag, cache_tags[0]);
        assert_eq!(bus.take_data(15).unwrap().tag, cache_tags[1]);
        assert_eq!(bus.take_return(15, dbid), Some(tags[0]));
        bus.push_data(
            15,
            C220MemoryWriteTransfer {
                ready_tick: 15,
                tag: mte_tag,
            },
        )
        .unwrap();
        assert_eq!(bus.take_data(16).unwrap().tag, mte_tag);
        for tag in [cache_tags[0], cache_tags[1], mte_tag] {
            assert!(bus.receive(17, completion, tag).unwrap());
        }
        for tick in 18..=20 {
            bus.advance(tick).unwrap();
        }
        assert_eq!(bus.outstanding(), 0);
        for (port, tag) in cache_tags.into_iter().enumerate() {
            assert_eq!(
                bus.take_cache_return(21, port as u32, completion),
                Some(tag)
            );
        }
        assert_eq!(bus.take_return(21, completion), Some(tags[0]));
        assert!(bus.is_idle());
    }
}
