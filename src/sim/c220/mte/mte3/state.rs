use std::collections::{BTreeMap, VecDeque};

use super::C220Mte3TransferPlan;
use crate::isa::flow::FlagStep;
use crate::sim::c220::vector::C220VectorFence;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220OutputDependency {
    NotBefore(u64),
    Vector(C220VectorFence),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum C220OutputAction {
    Event {
        flag: FlagStep,
        dependency: C220OutputDependency,
    },
    CopyToHbm {
        transfer: C220Mte3TransferPlan,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220OutputStep {
    pub pc: u64,
    pub word: u32,
    pub next_pc: u64,
    pub action: C220OutputAction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220OutputEvent {
    pub source_pipe: u8,
    pub destination_pipe: u8,
    pub flag_id: u32,
    pub dependency: C220OutputDependency,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct C220Mte3State {
    events: BTreeMap<(u8, u8, u32), VecDeque<C220OutputDependency>>,
}

impl C220Mte3State {
    pub(crate) fn signal(&mut self, flag: FlagStep, dependency: C220OutputDependency) {
        self.events
            .entry(Self::key(flag))
            .or_default()
            .push_back(dependency);
    }

    pub(crate) fn dependency(&self, flag: FlagStep) -> Option<C220OutputDependency> {
        self.events.get(&Self::key(flag))?.front().copied()
    }

    pub(crate) fn consume(&mut self, flag: FlagStep) -> Option<C220OutputDependency> {
        let key = Self::key(flag);
        let events = self.events.get_mut(&key)?;
        let dependency = events.pop_front();
        if events.is_empty() {
            self.events.remove(&key);
        }
        dependency
    }

    pub(crate) fn pending_events(&self) -> impl Iterator<Item = C220OutputEvent> + '_ {
        self.events
            .iter()
            .flat_map(|(&(source_pipe, destination_pipe, flag_id), events)| {
                events.iter().map(move |&dependency| C220OutputEvent {
                    source_pipe,
                    destination_pipe,
                    flag_id,
                    dependency,
                })
            })
    }

    fn key(flag: FlagStep) -> (u8, u8, u32) {
        (
            flag.instruction.source_pipe_code,
            flag.instruction.trigger_pipe_code,
            flag.flag_id,
        )
    }
}
