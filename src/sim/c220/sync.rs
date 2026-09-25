use std::collections::{BTreeMap, BTreeSet, VecDeque};

mod cross_core;
pub use cross_core::{C220CrossCoreReception, C220DeviceSync};

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

/// A captured synchronization event. Zero requests a fresh visibility delay;
/// a nonzero timestamp is retained until a delivery callback observes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220HardwareFlagEvent {
    pub step: C220HardwareFlagStep,
    pub timestamp: u64,
}

impl C220HardwareFlagEvent {
    /// MTE instruction capture uses the low 32 bits of the issue clock.
    pub const fn capture_mte(step: C220HardwareFlagStep, tick: u64) -> Self {
        Self {
            step,
            timestamp: tick as u32 as u64,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220HardwareFlagDelivery {
    pub admitted_tick: u64,
    pub timestamp: u64,
    pub destination_pipe_code: u8,
    pub memory: C220MatrixMemory,
    pub event_id: u32,
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
    admitted_tick: u64,
    event: C220HardwareFlagEvent,
    key: C220HardwareFlagKey,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct C220QueuedHardwareWait {
    instruction_id: u64,
    step: C220HardwareFlagStep,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct C220QueuedMteFlag {
    instruction_id: u64,
    event: C220HardwareFlagEvent,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct C220HardwareFlagState {
    now: u64,
    counters: BTreeMap<C220HardwareFlagKey, u8>,
    pending_sets: Vec<C220PendingHardwareFlag>,
    // The delivery process runs at most once per tick, across all notifications.
    notifications: BTreeSet<u64>,
    cube_waits: VecDeque<C220QueuedHardwareWait>,
    mte_flags: Vec<C220QueuedMteFlag>,
    saturated_sets: u64,
}

impl C220HardwareFlagState {
    pub(in crate::sim::c220) fn enqueue_mte_flag(
        &mut self,
        instruction_id: u64,
        step: C220HardwareFlagStep,
        tick: u64,
    ) -> Result<(), C220HardwareFlagTimingError> {
        if step.instruction.trigger {
            return Err(C220HardwareFlagTimingError::OperationMismatch);
        }
        if self.mte_flags.iter().any(|pending| {
            pending.event.step.instruction.memory == step.instruction.memory
                && pending.event.step.instruction.operation == step.instruction.operation
                && pending.event.step.event_id == step.event_id
        }) {
            return Ok(());
        }
        self.mte_flags.push(C220QueuedMteFlag {
            instruction_id,
            event: C220HardwareFlagEvent::capture_mte(step, tick),
        });
        Ok(())
    }

    pub fn pending_mte_set_count(&self) -> usize {
        self.pending_mte_flags()
            .filter(|event| event.step.instruction.operation == C220HardwareFlagOperation::Set)
            .count()
    }

    pub fn pending_mte_flags(&self) -> impl Iterator<Item = C220HardwareFlagEvent> + '_ {
        self.mte_flags.iter().map(|pending| pending.event)
    }

    pub(in crate::sim::c220) fn take_mte_sets(
        &mut self,
        instruction_id: u64,
        memory: C220MatrixMemory,
    ) -> VecDeque<C220HardwareFlagEvent> {
        self.take_mte_flags_matching(instruction_id, memory, Some(C220HardwareFlagOperation::Set))
    }

    pub(in crate::sim::c220) fn take_mte_flags(
        &mut self,
        instruction_id: u64,
        memory: C220MatrixMemory,
    ) -> VecDeque<C220HardwareFlagEvent> {
        self.take_mte_flags_matching(instruction_id, memory, None)
    }

    fn take_mte_flags_matching(
        &mut self,
        instruction_id: u64,
        memory: C220MatrixMemory,
        operation: Option<C220HardwareFlagOperation>,
    ) -> VecDeque<C220HardwareFlagEvent> {
        let mut attached = VecDeque::new();
        self.mte_flags.retain(|pending| {
            if pending.instruction_id < instruction_id
                && pending.event.step.instruction.memory == memory
                && operation
                    .is_none_or(|operation| pending.event.step.instruction.operation == operation)
            {
                attached.push_back(pending.event);
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
        while let Some(&notification) = self.notifications.first()
            && notification <= tick
        {
            self.notifications.pop_first();
            self.deliver_at(notification);
        }
        self.now = tick;
        Ok(())
    }

    fn deliver_at(&mut self, tick: u64) {
        let mut index = 0;
        while index < self.pending_sets.len() {
            let pending = self.pending_sets[index];
            if pending.admitted_tick <= tick && pending.event.timestamp <= tick {
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
    }

    /// Schedule a new, untimestamped completion signal.
    pub fn schedule_set(
        &mut self,
        step: C220HardwareFlagStep,
        checkpoint_tick: u64,
    ) -> Result<u64, C220HardwareFlagTimingError> {
        self.schedule_event(
            C220HardwareFlagEvent { step, timestamp: 0 },
            checkpoint_tick,
        )
    }

    /// Return the notification tick, not a guaranteed token visibility tick.
    /// Every notification scans all admitted events; an earlier notification
    /// can deliver this event, while a future timestamp can outlive this wake.
    pub fn schedule_event(
        &mut self,
        mut event: C220HardwareFlagEvent,
        checkpoint_tick: u64,
    ) -> Result<u64, C220HardwareFlagTimingError> {
        let step = event.step;
        if step.instruction.operation != C220HardwareFlagOperation::Set {
            return Err(C220HardwareFlagTimingError::OperationMismatch);
        }
        if checkpoint_tick < self.now {
            return Err(C220HardwareFlagTimingError::TimeReversed {
                requested: checkpoint_tick,
                previous: self.now,
            });
        }
        let key = C220HardwareFlagKey::from_step(step);
        let count = self.counters.get(&key).copied().unwrap_or(0);
        if count >= C220_HARDWARE_FLAG_ALMOST_FULL {
            return Err(C220HardwareFlagTimingError::AlmostFull {
                event_id: step.event_id,
                count,
            });
        }
        let delay = if event.timestamp == 0 {
            visibility_ticks(step.instruction.memory)
        } else {
            1
        };
        let notification_tick = checkpoint_tick
            .checked_add(delay)
            .ok_or(C220HardwareFlagTimingError::TimeOverflow)?;
        if event.timestamp == 0 {
            event.timestamp = notification_tick;
        }
        self.notifications.insert(notification_tick);
        self.pending_sets.push(C220PendingHardwareFlag {
            admitted_tick: checkpoint_tick,
            event,
            key,
        });
        Ok(notification_tick)
    }

    pub fn next_notification_tick(&self) -> Option<u64> {
        self.notifications.first().copied()
    }

    pub fn pending_deliveries(&self) -> impl Iterator<Item = C220HardwareFlagDelivery> + '_ {
        self.pending_sets
            .iter()
            .map(|pending| C220HardwareFlagDelivery {
                admitted_tick: pending.admitted_tick,
                timestamp: pending.event.timestamp,
                destination_pipe_code: pending.key.destination_pipe_code,
                memory: pending.event.step.instruction.memory,
                event_id: pending.key.event_id,
            })
    }

    fn next_delivery_tick(&self, key: C220HardwareFlagKey) -> Option<u64> {
        self.pending_sets
            .iter()
            .filter(|pending| pending.key == key)
            .filter_map(|pending| {
                self.notifications
                    .range(pending.admitted_tick.max(pending.event.timestamp)..)
                    .next()
                    .copied()
            })
            .min()
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
                    .next_delivery_tick(C220HardwareFlagKey::from_step(step))
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
        Ok(self.next_delivery_tick(key))
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

    #[test]
    fn timestamped_sets_share_notifications_without_automatic_rearming() {
        let set = step(C220HardwareFlagOperation::Set);
        let wait = step(C220HardwareFlagOperation::Wait);
        let mut flags = C220HardwareFlagState::default();
        // This timestamp remains pending after its initial notification.
        assert_eq!(
            flags.schedule_event(
                C220HardwareFlagEvent {
                    step: set,
                    timestamp: 10,
                },
                0
            ),
            Ok(1)
        );
        flags.advance_to(9).unwrap();
        assert_eq!(flags.wait_ready_tick(wait), Ok(None));
        assert_eq!(flags.next_notification_tick(), None);
        assert_eq!(flags.pending_deliveries().next().unwrap().timestamp, 10);

        // A different counter's notification wakes the whole table.
        let mut other = set;
        other.event_id = 1;
        assert_eq!(flags.schedule_set(other, 9), Ok(10));
        assert_eq!(flags.wait_ready_tick(wait), Ok(Some(10)));
        // Even a set whose own notification is later participates in that scan.
        assert_eq!(
            flags.schedule_event(
                C220HardwareFlagEvent {
                    step: set,
                    timestamp: 3,
                },
                10
            ),
            Ok(11)
        );
        flags.advance_to(10).unwrap();
        assert_eq!(flags.count(2, C220MatrixMemory::L0a, 0), 2);
        assert_eq!(flags.count(2, C220MatrixMemory::L0a, 1), 1);
        assert_eq!(flags.pending_deliveries().count(), 0);
        flags.advance_to(11).unwrap();
        assert_eq!(flags.count(2, C220MatrixMemory::L0a, 0), 2);
    }

    #[test]
    fn mte_capture_preserves_timestamp_through_attachment() {
        let mut set = step(C220HardwareFlagOperation::Set);
        set.instruction.memory = C220MatrixMemory::BiasTable;
        let mut flags = C220HardwareFlagState::default();
        flags.enqueue_mte_flag(1, set, 5).unwrap();
        flags.enqueue_mte_flag(2, set, 6).unwrap();
        let event = flags
            .take_mte_sets(3, C220MatrixMemory::BiasTable)
            .pop_front()
            .unwrap();
        assert_eq!(event.timestamp, 5);
        flags.advance_to(20).unwrap();
        assert_eq!(flags.schedule_event(event, 20), Ok(21));
        flags.advance_to(21).unwrap();
        assert_eq!(flags.count(2, C220MatrixMemory::BiasTable, 0), 1);

        // A wrapped issue timestamp of zero requests the full memory delay.
        let wrapped = C220HardwareFlagEvent::capture_mte(set, 1_u64 << 32);
        assert_eq!(wrapped.timestamp, 0);
        assert_eq!(flags.schedule_event(wrapped, 21), Ok(23));
        flags.advance_to(22).unwrap();
        assert_eq!(flags.count(2, C220MatrixMemory::BiasTable, 0), 1);
        flags.advance_to(23).unwrap();
        assert_eq!(flags.count(2, C220MatrixMemory::BiasTable, 0), 2);
    }
}
