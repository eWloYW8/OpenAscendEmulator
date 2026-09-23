use std::collections::BTreeMap;

use crate::isa::c220::hflag::{C220HardwareFlagOperation, C220MatrixMemory};
use crate::sim::c220::sync::{
    C220HardwareFlagEvent, C220HardwareFlagState, C220HardwareFlagTimingError,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220FixpSyncPoint {
    DisabledSet,
    DisabledWait,
    ReadWait,
    ConversionSet,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FixpSyncRequest {
    pub tick: u64,
    pub instruction_id: u64,
    pub point: C220FixpSyncPoint,
}

/// Called only when the stage has capacity and its queue head is ready.
/// Implementations may consume individual ready events even when other events
/// still block the command; consumed events must stay consumed on retry.
pub trait C220FixpSync {
    fn blocked(
        &mut self,
        request: C220FixpSyncRequest,
    ) -> Result<bool, C220HardwareFlagTimingError>;
}

impl<T: C220FixpSync + ?Sized> C220FixpSync for &mut T {
    fn blocked(
        &mut self,
        request: C220FixpSyncRequest,
    ) -> Result<bool, C220HardwareFlagTimingError> {
        (**self).blocked(request)
    }
}

impl C220FixpSync for bool {
    fn blocked(
        &mut self,
        _request: C220FixpSyncRequest,
    ) -> Result<bool, C220HardwareFlagTimingError> {
        Ok(*self)
    }
}

/// Already-associated command events. Instruction capture and attachment
/// selection belong to the command frontend, not to the execution stages.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct C220FixpSyncBindings {
    waits: BTreeMap<u64, Vec<C220HardwareFlagEvent>>,
    sets: BTreeMap<u64, Vec<C220HardwareFlagEvent>>,
}

impl C220FixpSyncBindings {
    /// Disabled commands publish their sets before attempting waits. A blocked
    /// set prevents any waits from being consumed during that decode attempt.
    pub fn disabled_blocked(
        &mut self,
        tick: u64,
        instruction_id: u64,
        flags: &mut C220HardwareFlagState,
    ) -> Result<bool, C220HardwareFlagTimingError> {
        let mut resolver = self.resolver(flags);
        if resolver.blocked(C220FixpSyncRequest {
            tick,
            instruction_id,
            point: C220FixpSyncPoint::DisabledSet,
        })? {
            return Ok(true);
        }
        resolver.blocked(C220FixpSyncRequest {
            tick,
            instruction_id,
            point: C220FixpSyncPoint::DisabledWait,
        })
    }

    /// Bind older L0C events at the command decode boundary, before dispatch.
    /// Repeating this for the same command does not capture newer flag issues.
    pub fn capture_pending(&mut self, instruction_id: u64, flags: &mut C220HardwareFlagState) {
        for event in flags.take_mte_flags(instruction_id, C220MatrixMemory::L0c) {
            self.attach(instruction_id, event);
        }
    }

    pub fn attach(&mut self, instruction_id: u64, event: C220HardwareFlagEvent) {
        let table = match event.step.instruction.operation {
            C220HardwareFlagOperation::Set => &mut self.sets,
            C220HardwareFlagOperation::Wait => &mut self.waits,
        };
        table.entry(instruction_id).or_default().push(event);
    }

    pub fn pending(
        &self,
        instruction_id: u64,
        point: C220FixpSyncPoint,
    ) -> &[C220HardwareFlagEvent] {
        let table = match point {
            C220FixpSyncPoint::ConversionSet | C220FixpSyncPoint::DisabledSet => &self.sets,
            C220FixpSyncPoint::ReadWait | C220FixpSyncPoint::DisabledWait => &self.waits,
        };
        table.get(&instruction_id).map_or(&[], Vec::as_slice)
    }

    /// The owner advances flag delivery in its shared event topology before
    /// invoking checkpoints; this binding does not run delivery callbacks.
    pub fn resolver<'a>(
        &'a mut self,
        flags: &'a mut C220HardwareFlagState,
    ) -> C220FixpFlagResolver<'a> {
        C220FixpFlagResolver {
            bindings: self,
            flags,
        }
    }
}

pub struct C220FixpFlagResolver<'a> {
    bindings: &'a mut C220FixpSyncBindings,
    flags: &'a mut C220HardwareFlagState,
}

impl C220FixpSync for C220FixpFlagResolver<'_> {
    fn blocked(
        &mut self,
        request: C220FixpSyncRequest,
    ) -> Result<bool, C220HardwareFlagTimingError> {
        let is_set = matches!(
            request.point,
            C220FixpSyncPoint::ConversionSet | C220FixpSyncPoint::DisabledSet
        );
        let table = if is_set {
            &mut self.bindings.sets
        } else {
            &mut self.bindings.waits
        };
        let Some(events) = table.get_mut(&request.instruction_id) else {
            return Ok(false);
        };
        let mut index = 0;
        while index < events.len() {
            let event = events[index];
            let result = if is_set {
                self.flags.schedule_event(event, request.tick).map(|_| ())
            } else {
                self.flags.consume_wait(event.step)
            };
            match result {
                Ok(()) => {
                    events.remove(index);
                }
                Err(C220HardwareFlagTimingError::AlmostFull { .. }) if is_set => index += 1,
                Err(C220HardwareFlagTimingError::MissingToken { .. }) if !is_set => index += 1,
                Err(error) => return Err(error),
            }
        }
        if events.is_empty() {
            table.remove(&request.instruction_id);
            Ok(false)
        } else {
            Ok(true)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::c220::hflag::{
        C220HardwareEventSource, C220HardwareFlagInstruction, C220HardwareFlagSourcePipe,
        C220HardwareFlagStep, C220MatrixMemory,
    };

    fn event(id: u32, operation: C220HardwareFlagOperation) -> C220HardwareFlagEvent {
        C220HardwareFlagEvent {
            timestamp: 0,
            step: C220HardwareFlagStep {
                pc: 0,
                event_id: id,
                source_value: None,
                instruction: C220HardwareFlagInstruction {
                    word: 0,
                    operation,
                    event_source: C220HardwareEventSource::Immediate(id as u8),
                    source_pipe: C220HardwareFlagSourcePipe::Fix,
                    destination_pipe_code: 2,
                    memory: C220MatrixMemory::L0c,
                    trigger: false,
                },
            },
        }
    }

    #[test]
    fn disabled_admission_publishes_sets_before_waiting_without_generating_reads() {
        use super::super::{
            C220FixpAdmission, C220FixpCommand, C220FixpEngine, C220FixpEngineConfig,
        };
        use crate::isa::c220::mte::fixp::C220FixpDescriptor;
        use C220HardwareFlagOperation::{Set, Wait};
        let mut engine = C220FixpEngine::new(C220FixpEngineConfig {
            instruction_fifo_depth: 2,
            read_bandwidth: 256,
            read_bank_count: 32,
            read_data_latency: 4,
            l0c_capacity: 131072,
        })
        .unwrap();
        let command = C220FixpCommand {
            source_format: crate::sim::c220::mte::fixp::C220FixpSourceFormat::Fp32,
            descriptor: C220FixpDescriptor {
                xt: 0,
                xm: 1 << 34,
                nd: 0,
            },
            source_address: 0,
            destination_address: 0,
            control: 0,
            scalar_slope: 0,
            slope_base_block: 0,
            dequant_base_block: 0,
            scalar_dequant: 0,
        };
        let mut flags = C220HardwareFlagState::default();
        let mut bindings = C220FixpSyncBindings::default();
        flags.enqueue_mte_flag(1, event(0, Set).step, 0).unwrap();
        flags.enqueue_mte_flag(2, event(0, Wait).step, 0).unwrap();
        assert_eq!(
            engine
                .admit_with_flags(0, 3, 1, command, &mut bindings, &mut flags)
                .unwrap(),
            C220FixpAdmission::HardwareSync
        );
        assert!(engine.is_idle());
        assert_eq!(flags.pending_deliveries().count(), 1);
        flags.advance_to(1).unwrap();
        assert_eq!(
            engine
                .admit_with_flags(1, 3, 1, command, &mut bindings, &mut flags)
                .unwrap(),
            C220FixpAdmission::DisabledReady
        );
        assert_eq!(flags.count(2, C220MatrixMemory::L0c, 0), 0);
        assert!(engine.is_idle());

        for _ in 0..32 {
            flags.schedule_event(event(0, Set), 1).unwrap();
        }
        flags.schedule_event(event(1, Set), 1).unwrap();
        flags.advance_to(2).unwrap();
        bindings.attach(4, event(0, Set));
        bindings.attach(4, event(1, Wait));
        assert_eq!(
            engine
                .admit_with_flags(2, 4, 1, command, &mut bindings, &mut flags)
                .unwrap(),
            C220FixpAdmission::HardwareSync
        );
        assert_eq!(flags.count(2, C220MatrixMemory::L0c, 1), 1);
        flags.consume_wait(event(0, Wait).step).unwrap();
        assert_eq!(
            engine
                .admit_with_flags(2, 4, 1, command, &mut bindings, &mut flags)
                .unwrap(),
            C220FixpAdmission::DisabledReady
        );
        assert_eq!(flags.count(2, C220MatrixMemory::L0c, 1), 0);
        assert!(engine.is_idle());
    }

    #[test]
    fn capture_selects_older_l0c_events_and_preserves_first_duplicate() {
        use C220HardwareFlagOperation::{Set, Wait};
        let mut flags = C220HardwareFlagState::default();
        let mut bindings = C220FixpSyncBindings::default();
        let set = event(0, Set).step;
        flags.enqueue_mte_flag(1, set, 5).unwrap();
        let mut duplicate = set;
        duplicate.instruction.destination_pipe_code = 3;
        flags.enqueue_mte_flag(2, duplicate, 6).unwrap();
        flags.enqueue_mte_flag(3, event(0, Wait).step, 7).unwrap();
        let mut other = set;
        other.instruction.memory = C220MatrixMemory::L0a;
        flags.enqueue_mte_flag(4, other, 8).unwrap();
        bindings.capture_pending(3, &mut flags);
        assert_eq!(
            bindings.pending(3, C220FixpSyncPoint::ConversionSet),
            &[C220HardwareFlagEvent::capture_mte(set, 5)]
        );
        assert!(bindings.pending(3, C220FixpSyncPoint::ReadWait).is_empty());
        bindings.capture_pending(5, &mut flags);
        assert_eq!(
            bindings.pending(5, C220FixpSyncPoint::ReadWait),
            &[C220HardwareFlagEvent::capture_mte(event(0, Wait).step, 7)]
        );
        assert_eq!(
            flags.pending_mte_flags().collect::<Vec<_>>(),
            [C220HardwareFlagEvent::capture_mte(other, 8)]
        );
        bindings.capture_pending(6, &mut flags);
        assert!(bindings.pending(6, C220FixpSyncPoint::ReadWait).is_empty());
    }

    #[test]
    fn partial_success_survives_retries() {
        use C220FixpSyncPoint::{ConversionSet, ReadWait};
        use C220HardwareFlagOperation::{Set, Wait};
        let mut flags = C220HardwareFlagState::default();
        let mut bindings = C220FixpSyncBindings::default();
        flags.schedule_event(event(1, Set), 0).unwrap();
        flags.advance_to(1).unwrap();
        for id in [0, 1] {
            bindings.attach(7, event(id, Wait));
        }
        let request = |tick, instruction_id, point| C220FixpSyncRequest {
            tick,
            instruction_id,
            point,
        };
        assert!(
            !bindings
                .resolver(&mut flags)
                .blocked(request(1, 8, ReadWait))
                .unwrap()
        );
        assert!(
            bindings
                .resolver(&mut flags)
                .blocked(request(1, 7, ReadWait))
                .unwrap()
        );
        assert_eq!(bindings.pending(7, ReadWait), &[event(0, Wait)]);
        assert_eq!(flags.count(2, C220MatrixMemory::L0c, 1), 0);
        flags.schedule_event(event(0, Set), 1).unwrap();
        flags.advance_to(2).unwrap();
        assert!(
            !bindings
                .resolver(&mut flags)
                .blocked(request(2, 7, ReadWait))
                .unwrap()
        );
        assert!(bindings.pending(7, ReadWait).is_empty());

        for _ in 0..32 {
            flags.schedule_event(event(0, Set), 2).unwrap();
        }
        flags.advance_to(3).unwrap();
        for id in [0, 1] {
            bindings.attach(8, event(id, Set));
        }
        assert!(
            bindings
                .resolver(&mut flags)
                .blocked(request(3, 8, ConversionSet))
                .unwrap()
        );
        assert_eq!(bindings.pending(8, ConversionSet), &[event(0, Set)]);
        flags.advance_to(4).unwrap();
        flags.consume_wait(event(0, Wait).step).unwrap();
        assert!(
            !bindings
                .resolver(&mut flags)
                .blocked(request(4, 8, ConversionSet))
                .unwrap()
        );
        flags.advance_to(5).unwrap();
        assert_eq!(flags.count(2, C220MatrixMemory::L0c, 0), 32);
        assert_eq!(flags.count(2, C220MatrixMemory::L0c, 1), 1);
        assert!(bindings.pending(8, ConversionSet).is_empty());
    }
}
