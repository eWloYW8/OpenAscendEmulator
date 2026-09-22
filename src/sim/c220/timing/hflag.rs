use std::collections::BTreeMap;

use thiserror::Error;

use crate::isa::c220::hflag::{C220HardwareFlagOperation, C220HardwareFlagStep, C220MatrixMemory};

pub const C220_HARDWARE_FLAG_ALMOST_FULL: u8 = 32;
pub const C220_HARDWARE_FLAG_CAPACITY: u8 = 63;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct C220HardwareFlagKey {
    destination_pipe_code: u8,
    memory: u8,
    event_id: u32,
}

impl C220HardwareFlagKey {
    const fn from_step(step: C220HardwareFlagStep) -> Self {
        Self {
            destination_pipe_code: step.instruction.destination_pipe_code,
            memory: memory_code(step.instruction.memory),
            event_id: step.event_id,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct C220PendingHardwareFlag {
    ready_tick: u64,
    key: C220HardwareFlagKey,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct C220HardwareFlagState {
    now: u64,
    counters: BTreeMap<C220HardwareFlagKey, u8>,
    pending_sets: Vec<C220PendingHardwareFlag>,
}

impl C220HardwareFlagState {
    pub fn advance_to(&mut self, tick: u64) -> Result<(), C220HardwareFlagTimingError> {
        if tick < self.now {
            return Err(C220HardwareFlagTimingError::TimeReversed {
                requested: tick,
                previous: self.now,
            });
        }
        self.now = tick;
        let mut index = 0;
        while index < self.pending_sets.len() {
            if self.pending_sets[index].ready_tick <= tick {
                let pending = self.pending_sets.remove(index);
                let counter = self.counters.entry(pending.key).or_default();
                if *counter >= C220_HARDWARE_FLAG_CAPACITY {
                    return Err(C220HardwareFlagTimingError::CounterOverflow);
                }
                *counter += 1;
            } else {
                index += 1;
            }
        }
        Ok(())
    }

    pub fn schedule_set(
        &mut self,
        step: C220HardwareFlagStep,
        checkpoint_tick: u64,
    ) -> Result<u64, C220HardwareFlagTimingError> {
        if step.instruction.operation != C220HardwareFlagOperation::Set {
            return Err(C220HardwareFlagTimingError::OperationMismatch);
        }
        let key = C220HardwareFlagKey::from_step(step);
        let reserved = u16::from(self.counters.get(&key).copied().unwrap_or(0))
            + self
                .pending_sets
                .iter()
                .filter(|pending| pending.key == key)
                .count() as u16;
        if reserved >= u16::from(C220_HARDWARE_FLAG_ALMOST_FULL) {
            return Err(C220HardwareFlagTimingError::AlmostFull {
                event_id: step.event_id,
                count: reserved,
            });
        }
        let ready_tick = checkpoint_tick
            .checked_add(visibility_ticks(step.instruction.memory))
            .ok_or(C220HardwareFlagTimingError::TimeOverflow)?;
        self.pending_sets
            .push(C220PendingHardwareFlag { ready_tick, key });
        Ok(ready_tick)
    }

    pub fn wait_ready_tick(
        &self,
        step: C220HardwareFlagStep,
    ) -> Result<Option<u64>, C220HardwareFlagTimingError> {
        if step.instruction.operation != C220HardwareFlagOperation::Wait {
            return Err(C220HardwareFlagTimingError::OperationMismatch);
        }
        let key = C220HardwareFlagKey::from_step(step);
        if self.counters.get(&key).copied().unwrap_or(0) != 0 {
            return Ok(Some(self.now));
        }
        Ok(self
            .pending_sets
            .iter()
            .filter(|pending| pending.key == key)
            .map(|pending| pending.ready_tick)
            .min())
    }

    pub fn consume_wait(
        &mut self,
        step: C220HardwareFlagStep,
    ) -> Result<(), C220HardwareFlagTimingError> {
        if step.instruction.operation != C220HardwareFlagOperation::Wait {
            return Err(C220HardwareFlagTimingError::OperationMismatch);
        }
        let key = C220HardwareFlagKey::from_step(step);
        let counter =
            self.counters
                .get_mut(&key)
                .ok_or(C220HardwareFlagTimingError::MissingToken {
                    event_id: step.event_id,
                })?;
        if *counter == 0 {
            return Err(C220HardwareFlagTimingError::MissingToken {
                event_id: step.event_id,
            });
        }
        *counter -= 1;
        if *counter == 0 {
            self.counters.remove(&key);
        }
        Ok(())
    }

    pub fn count(&self, destination_pipe_code: u8, memory: C220MatrixMemory, event_id: u32) -> u8 {
        self.counters
            .get(&C220HardwareFlagKey {
                destination_pipe_code,
                memory: memory_code(memory),
                event_id,
            })
            .copied()
            .unwrap_or(0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum C220HardwareFlagTimingError {
    #[error("C220 hardware flag timing computation overflowed")]
    TimeOverflow,
    #[error("hardware flag tick {requested} precedes observed tick {previous}")]
    TimeReversed { requested: u64, previous: u64 },
    #[error("hardware flag operation does not match the requested state transition")]
    OperationMismatch,
    #[error("hardware flag counter would exceed its storage capacity")]
    CounterOverflow,
    #[error("hardware flag event {event_id} reached almost-full count {count}")]
    AlmostFull { event_id: u32, count: u16 },
    #[error("hardware flag event {event_id} has no visible or scheduled token")]
    MissingToken { event_id: u32 },
}

const fn visibility_ticks(memory: C220MatrixMemory) -> u64 {
    match memory {
        C220MatrixMemory::L0a | C220MatrixMemory::L0b | C220MatrixMemory::L0c => 1,
        C220MatrixMemory::BiasTable => 2,
    }
}

const fn memory_code(memory: C220MatrixMemory) -> u8 {
    match memory {
        C220MatrixMemory::L0a => 1,
        C220MatrixMemory::L0b => 2,
        C220MatrixMemory::L0c => 3,
        C220MatrixMemory::BiasTable => 5,
    }
}
