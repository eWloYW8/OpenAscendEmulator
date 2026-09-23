use std::collections::{BTreeMap, VecDeque};
use std::num::NonZeroU32;

use crate::sim::c220::mte::interface::biu_write::command::C220BiuWriteCommandTransfer;
use crate::sim::c220::mte::interface::biu_write::data::C220BiuWriteData;

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
pub struct C220BiuWriteReturn {
    pub ready_tick: u64,
    pub tag: NonZeroU32,
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
    #[error("BIU bus write tag {0} is not in the expected transaction phase")]
    InvalidPhase(NonZeroU32),
}

/// MTE write route through the core BIU. Cache arbitration is not connected.
/// Taking an outgoing item means the downstream endpoint accepted the send;
/// that endpoint owns its receive capacity, transport delay and memory timing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220BiuMteBusWrites {
    limit: NonZeroU32,
    outstanding: u32,
    commands: VecDeque<C220BiuWriteCommandTransfer>,
    data: VecDeque<C220BiuWriteData>,
    returns: [VecDeque<C220BiuWriteReturn>; 2],
    upstream: [VecDeque<C220BiuWriteReturn>; 2],
    phases: BTreeMap<NonZeroU32, Phase>,
    last_command_send: Option<u64>,
    last_data_send: Option<u64>,
}

impl C220BiuMteBusWrites {
    pub(crate) fn new(limit: NonZeroU32) -> Self {
        Self {
            limit,
            outstanding: 0,
            commands: VecDeque::new(),
            data: VecDeque::new(),
            returns: Default::default(),
            upstream: Default::default(),
            phases: BTreeMap::new(),
            last_command_send: None,
            last_data_send: None,
        }
    }

    pub fn outstanding(&self) -> u32 {
        self.outstanding
    }

    pub fn queued_commands(&self) -> &VecDeque<C220BiuWriteCommandTransfer> {
        &self.commands
    }

    pub fn queued_data(&self) -> &VecDeque<C220BiuWriteData> {
        &self.data
    }

    pub fn returns(&self, kind: C220BiuWriteReturnKind) -> &VecDeque<C220BiuWriteReturn> {
        &self.returns[kind.index()]
    }

    pub fn upstream(&self, kind: C220BiuWriteReturnKind) -> &VecDeque<C220BiuWriteReturn> {
        &self.upstream[kind.index()]
    }

    pub fn is_idle(&self) -> bool {
        self.phases.is_empty()
    }

    pub(crate) fn can_receive_command(&self) -> bool {
        self.outstanding < self.limit.get()
    }

    pub(crate) fn push_command(
        &mut self,
        tick: u64,
        mut command: C220BiuWriteCommandTransfer,
    ) -> Result<(), C220BiuBusWriteError> {
        let tag = command.command.tag;
        if self.phases.contains_key(&tag) {
            return Err(C220BiuBusWriteError::InvalidPhase(tag));
        }
        command.ready_tick = next_tick(tick)?;
        self.phases.insert(tag, Phase::Queued);
        self.commands.push_back(command);
        Ok(())
    }

    pub(crate) fn take_command(&mut self, tick: u64) -> Option<C220BiuWriteCommandTransfer> {
        if self.last_command_send == Some(tick) {
            return None;
        }
        let command = self.commands.pop_front_if(|head| head.ready_tick <= tick)?;
        self.phases.insert(command.command.tag, Phase::CommandSent);
        self.outstanding += 1;
        self.last_command_send = Some(tick);
        Some(command)
    }

    pub(crate) fn push_data(
        &mut self,
        tick: u64,
        mut data: C220BiuWriteData,
    ) -> Result<(), C220BiuBusWriteError> {
        let tag = data.source.request.tag;
        self.require(tag, Phase::DbidDelivered)?;
        data.ready_tick = next_tick(tick)?;
        self.phases.insert(tag, Phase::DataQueued);
        self.data.push_back(data);
        Ok(())
    }

    pub(crate) fn take_data(&mut self, tick: u64) -> Option<C220BiuWriteData> {
        if self.last_data_send == Some(tick) {
            return None;
        }
        let mut data = self.data.pop_front_if(|head| head.ready_tick <= tick)?;
        data.sent_tick = tick;
        self.phases.insert(data.source.request.tag, Phase::DataSent);
        self.last_data_send = Some(tick);
        Some(data)
    }

    pub(crate) fn receive(
        &mut self,
        tick: u64,
        kind: C220BiuWriteReturnKind,
        tag: NonZeroU32,
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
        queue.push_back(C220BiuWriteReturn { ready_tick, tag });
        self.phases.insert(tag, next);
        Ok(true)
    }

    pub(crate) fn advance(&mut self, tick: u64) -> Result<(), C220BiuBusWriteError> {
        let ready_tick = next_tick(tick)?;
        for index in 0..2 {
            if self.upstream[index].len() < 2
                && let Some(mut response) =
                    self.returns[index].pop_front_if(|head| head.ready_tick <= tick)
            {
                response.ready_tick = ready_tick;
                self.upstream[index].push_back(response);
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
        Some(response.tag)
    }

    fn require(&self, tag: NonZeroU32, expected: Phase) -> Result<(), C220BiuBusWriteError> {
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
        let mut bus = C220BiuMteBusWrites::new(NonZeroU32::new(4).unwrap());
        let tags = [1, 2, 3, 4].map(|tag| NonZeroU32::new(tag).unwrap());
        for tag in tags {
            bus.phases.insert(tag, Phase::DataSent);
        }
        bus.outstanding = 4;
        let completion = C220BiuWriteReturnKind::Completion;
        for tag in &tags[..3] {
            assert!(bus.receive(0, completion, *tag).unwrap());
        }
        assert!(!bus.receive(0, completion, tags[3]).unwrap());
        assert!(!bus.can_receive_command());
        bus.advance(0).unwrap();
        assert_eq!(bus.outstanding(), 4);
        bus.advance(1).unwrap();
        assert_eq!(bus.outstanding(), 3);
        assert!(bus.can_receive_command());
        assert!(bus.take_return(1, completion).is_none());
        assert!(bus.receive(1, completion, tags[3]).unwrap());
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
    }
}
