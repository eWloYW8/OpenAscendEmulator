use std::collections::{BTreeMap, VecDeque};

use thiserror::Error;

use crate::isa::c220::hflag::{C220HardwareFlagOperation, C220HardwareFlagStep, C220MatrixMemory};

pub const C220_HARDWARE_FLAG_ALMOST_FULL: u8 = 32;
pub const C220_HARDWARE_FLAG_CAPACITY: u8 = 63;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220HardwareFlagCounterSnapshot {
    pub visible: u8,
    pub pending: usize,
    pub almost_full: bool,
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct C220QueuedHardwareWait {
    instruction_id: u64,
    step: C220HardwareFlagStep,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct C220QueuedMteSet {
    instruction_id: u64,
    step: C220HardwareFlagStep,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct C220HardwareFlagState {
    now: u64,
    counters: BTreeMap<C220HardwareFlagKey, u8>,
    pending_sets: Vec<C220PendingHardwareFlag>,
    cube_waits: VecDeque<C220QueuedHardwareWait>,
    mte_sets: Vec<C220QueuedMteSet>,
    saturated_sets: u64,
}

impl C220HardwareFlagState {
    pub(in crate::sim::c220) fn enqueue_mte_set(
        &mut self,
        instruction_id: u64,
        step: C220HardwareFlagStep,
    ) -> Result<(), C220HardwareFlagTimingError> {
        if step.instruction.operation != C220HardwareFlagOperation::Set || step.instruction.trigger
        {
            return Err(C220HardwareFlagTimingError::OperationMismatch);
        }
        if self.mte_sets.iter().any(|pending| {
            pending.step.instruction.memory == step.instruction.memory
                && pending.step.event_id == step.event_id
        }) {
            return Ok(());
        }
        self.mte_sets.push(C220QueuedMteSet {
            instruction_id,
            step,
        });
        Ok(())
    }

    pub fn pending_mte_set_count(&self) -> usize {
        self.mte_sets.len()
    }

    pub(in crate::sim::c220) fn take_mte_sets(
        &mut self,
        instruction_id: u64,
        memory: C220MatrixMemory,
    ) -> VecDeque<C220HardwareFlagStep> {
        let mut attached = VecDeque::new();
        self.mte_sets.retain(|pending| {
            if pending.instruction_id < instruction_id && pending.step.instruction.memory == memory
            {
                attached.push_back(pending.step);
                false
            } else {
                true
            }
        });
        attached
    }

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
                    self.saturated_sets = self.saturated_sets.saturating_add(1);
                } else {
                    *counter += 1;
                }
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
        let count = self.counters.get(&key).copied().unwrap_or(0);
        if count >= C220_HARDWARE_FLAG_ALMOST_FULL {
            return Err(C220HardwareFlagTimingError::AlmostFull {
                event_id: step.event_id,
                count,
            });
        }
        let ready_tick = checkpoint_tick
            .checked_add(visibility_ticks(step.instruction.memory))
            .ok_or(C220HardwareFlagTimingError::TimeOverflow)?;
        self.pending_sets
            .push(C220PendingHardwareFlag { ready_tick, key });
        Ok(ready_tick)
    }

    pub fn enqueue_cube_wait(
        &mut self,
        instruction_id: u64,
        step: C220HardwareFlagStep,
    ) -> Result<(), C220HardwareFlagTimingError> {
        if step.instruction.operation != C220HardwareFlagOperation::Wait {
            return Err(C220HardwareFlagTimingError::OperationMismatch);
        }
        if self
            .cube_waits
            .back()
            .is_some_and(|wait| wait.instruction_id >= instruction_id)
        {
            return Err(C220HardwareFlagTimingError::InstructionOrder);
        }
        self.cube_waits.push_back(C220QueuedHardwareWait {
            instruction_id,
            step,
        });
        Ok(())
    }

    pub fn gate_cube_instruction(
        &mut self,
        tick: u64,
        instruction_id: u64,
    ) -> Result<Option<u64>, C220HardwareFlagTimingError> {
        self.advance_to(tick)?;
        while self
            .cube_waits
            .front()
            .is_some_and(|wait| wait.instruction_id < instruction_id)
        {
            let step = self
                .cube_waits
                .front()
                .expect("older Cube wait exists")
                .step;
            if self
                .counters
                .get(&C220HardwareFlagKey::from_step(step))
                .copied()
                .unwrap_or(0)
                == 0
            {
                let resume_tick = self
                    .pending_sets
                    .iter()
                    .filter(|pending| pending.key == C220HardwareFlagKey::from_step(step))
                    .map(|pending| pending.ready_tick)
                    .min()
                    .map_or_else(
                        || {
                            tick.checked_add(1)
                                .ok_or(C220HardwareFlagTimingError::TimeOverflow)
                        },
                        Ok,
                    )?;
                return Ok(Some(resume_tick));
            }
            self.consume_wait(step)?;
            self.cube_waits.pop_front();
        }
        Ok(None)
    }

    pub fn pending_cube_wait_count(&self) -> usize {
        self.cube_waits.len()
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

    pub fn counter_snapshot(
        &self,
        destination_pipe_code: u8,
        memory: C220MatrixMemory,
        event_id: u32,
    ) -> C220HardwareFlagCounterSnapshot {
        let key = C220HardwareFlagKey {
            destination_pipe_code,
            memory: memory_code(memory),
            event_id,
        };
        let visible = self.counters.get(&key).copied().unwrap_or(0);
        C220HardwareFlagCounterSnapshot {
            visible,
            pending: self
                .pending_sets
                .iter()
                .filter(|set| set.key == key)
                .count(),
            almost_full: visible >= C220_HARDWARE_FLAG_ALMOST_FULL,
        }
    }

    /// Number of delivered sets discarded because the visible counter was full.
    pub const fn saturated_set_count(&self) -> u64 {
        self.saturated_sets
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
    #[error("hardware flag event {event_id} reached almost-full count {count}")]
    AlmostFull { event_id: u32, count: u8 },
    #[error("hardware flag event {event_id} has no visible or scheduled token")]
    MissingToken { event_id: u32 },
    #[error("C220 hardware waits must be enqueued in increasing instruction order")]
    InstructionOrder,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::c220::hflag::{C220HardwareFlagInstruction, C220HardwareFlagSourcePipe};

    fn step(operation: C220HardwareFlagOperation) -> C220HardwareFlagStep {
        C220HardwareFlagStep {
            pc: 0,
            instruction: C220HardwareFlagInstruction {
                word: 0,
                operation,
                event_source: crate::isa::c220::hflag::C220HardwareEventSource::Immediate(0),
                source_pipe: C220HardwareFlagSourcePipe::Mte1,
                destination_pipe_code: 2,
                memory: C220MatrixMemory::L0a,
                trigger: false,
            },
            event_id: 0,
            source_value: None,
        }
    }

    #[test]
    fn queued_wait_gates_only_newer_cube_instructions() {
        let mut flags = C220HardwareFlagState::default();
        assert_eq!(
            flags.schedule_set(step(C220HardwareFlagOperation::Set), 10),
            Ok(11)
        );
        flags
            .enqueue_cube_wait(0, step(C220HardwareFlagOperation::Wait))
            .unwrap();
        assert_eq!(flags.gate_cube_instruction(5, 0), Ok(None));
        assert_eq!(flags.gate_cube_instruction(5, 1), Ok(Some(11)));
        assert_eq!(flags.gate_cube_instruction(11, 1), Ok(None));
        assert_eq!(flags.pending_cube_wait_count(), 0);
        assert_eq!(flags.count(2, C220MatrixMemory::L0a, 0), 0);
    }

    #[test]
    fn admission_counts_visible_tokens_and_delivery_saturates() {
        let mut flags = C220HardwareFlagState::default();
        let set = step(C220HardwareFlagOperation::Set);
        for _ in 0..=C220_HARDWARE_FLAG_CAPACITY {
            assert_eq!(flags.schedule_set(set, 0), Ok(1));
        }
        assert_eq!(
            flags.counter_snapshot(2, C220MatrixMemory::L0a, 0),
            C220HardwareFlagCounterSnapshot {
                visible: 0,
                pending: usize::from(C220_HARDWARE_FLAG_CAPACITY) + 1,
                almost_full: false,
            }
        );
        flags.advance_to(1).unwrap();
        assert_eq!(flags.saturated_set_count(), 1);
        assert_eq!(flags.count(2, C220MatrixMemory::L0a, 0), 63);
        assert!(matches!(
            flags.schedule_set(set, 1),
            Err(C220HardwareFlagTimingError::AlmostFull { count: 63, .. })
        ));
        for _ in C220_HARDWARE_FLAG_ALMOST_FULL..=C220_HARDWARE_FLAG_CAPACITY {
            flags
                .consume_wait(step(C220HardwareFlagOperation::Wait))
                .unwrap();
        }
        assert_eq!(flags.schedule_set(set, 1), Ok(2));
        flags.advance_to(2).unwrap();
        assert_eq!(flags.count(2, C220MatrixMemory::L0a, 0), 32);
    }
}
