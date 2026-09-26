use super::*;
use crate::isa::c220::hflag::{C220HardwareFlagOperation, C220MatrixMemory};
use crate::sim::c220::sync::{
    C220HardwareFlagEvent, C220HardwareFlagState, C220HardwareFlagTimingError,
};
use std::collections::{BTreeMap, VecDeque};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct Mte1Sync {
    attached: BTreeMap<u64, Vec<C220HardwareFlagEvent>>,
    pending: Option<(Callback, u64)>,
}

impl C220MtePipeline {
    pub fn pending_mte1_hardware_flags(&self, id: u64) -> &[C220HardwareFlagEvent] {
        self.mte1_sync.attached.get(&id).map_or(&[], Vec::as_slice)
    }
    pub(in crate::sim::c220) fn capture_mte1_flags(
        &mut self,
        id: u64,
        memory: C220MatrixMemory,
        flags: &mut C220HardwareFlagState,
    ) {
        let events = flags.take_mte_flags(id, memory);
        if !events.is_empty() {
            self.mte1_sync
                .attached
                .entry(id)
                .or_default()
                .extend(events);
        }
    }

    pub(in crate::sim::c220) fn take_mte1_sets(
        &mut self,
        id: u64,
    ) -> VecDeque<C220HardwareFlagEvent> {
        let mut sets = VecDeque::new();
        if let Some(events) = self.mte1_sync.attached.get_mut(&id) {
            events.retain(|event| {
                if event.step.instruction.operation == C220HardwareFlagOperation::Set {
                    sets.push_back(*event);
                    false
                } else {
                    true
                }
            });
            if events.is_empty() {
                self.mte1_sync.attached.remove(&id);
            }
        }
        sets
    }

    pub(in crate::sim::c220) fn disabled_mte1_sync_blocked(
        &mut self,
        id: u64,
        flags: &mut C220HardwareFlagState,
    ) -> Result<bool, C220MtePipelineError> {
        if self.resolve_mte1_flags(id, C220HardwareFlagOperation::Set, flags)? {
            return Ok(true);
        }
        self.resolve_mte1_flags(id, C220HardwareFlagOperation::Wait, flags)
    }

    fn resolve_mte1_flags(
        &mut self,
        id: u64,
        operation: C220HardwareFlagOperation,
        flags: &mut C220HardwareFlagState,
    ) -> Result<bool, C220MtePipelineError> {
        let Some(events) = self.mte1_sync.attached.get_mut(&id) else {
            return Ok(false);
        };
        let mut blocked = false;
        let mut index = 0;
        while index < events.len() {
            let event = events[index];
            if event.step.instruction.operation != operation {
                index += 1;
                continue;
            }
            let result = match operation {
                C220HardwareFlagOperation::Set => {
                    flags.schedule_event(event, self.events.tick()).map(|_| ())
                }
                C220HardwareFlagOperation::Wait => flags.consume_wait(event.step),
            };
            match result {
                Ok(()) => {
                    events.remove(index);
                }
                Err(
                    C220HardwareFlagTimingError::MissingToken { .. }
                    | C220HardwareFlagTimingError::AlmostFull { .. },
                ) => {
                    blocked = true;
                    index += 1;
                }
                Err(error) => return Err(error.into()),
            }
        }
        if events.is_empty() {
            self.mte1_sync.attached.remove(&id);
        }
        Ok(blocked)
    }

    pub(crate) fn mte1_sync_pending(&self) -> bool {
        self.mte1_sync.pending.is_some()
    }

    pub(crate) fn resolve_mte1_sync(
        &mut self,
        flags: &mut C220HardwareFlagState,
    ) -> Result<(), C220MtePipelineError> {
        let (callback, id) = self
            .mte1_sync
            .pending
            .expect("pending MTE1 synchronization");
        let blocked = self.resolve_mte1_flags(id, C220HardwareFlagOperation::Wait, flags)?;
        self.handle_mte1_send_callback(callback, blocked)?;
        self.mte1_sync.pending = None;
        Ok(())
    }

    pub(super) fn defer_mte1_sync(&mut self, callback: Callback, tick: u64) -> bool {
        let id = match callback {
            Callback::Generator(kind, C220MteGeneratorCallback::Send) => self.generators
                [kind.index()]
            .generated()
            .front()
            .filter(|head| head.ready_tick <= tick)
            .map(|head| head.operation.instruction_id),
            Callback::Set2d(C220MteGeneratorCallback::Send) => self
                .set2d
                .generated()
                .front()
                .filter(|head| head.ready_tick <= tick)
                .map(|head| head.instruction_id),
            _ => None,
        };
        if let Some(id) = id
            && self.mte1_sync.attached.get(&id).is_some_and(|events| {
                events.iter().any(|event| {
                    event.step.instruction.operation == C220HardwareFlagOperation::Wait
                })
            })
        {
            self.mte1_sync.pending = Some((callback, id));
            true
        } else {
            false
        }
    }

    pub(super) fn handle_mte1_send_callback(
        &mut self,
        callback: Callback,
        blocked: bool,
    ) -> Result<(), C220MtePipelineError> {
        match callback {
            Callback::Generator(kind, phase) => {
                let outcome = self.generator_events[kind.index()].handle_mapped(
                    phase,
                    &mut self.events,
                    &mut self.generators[kind.index()],
                    blocked,
                    &mut self.interface,
                    C220MteReadPayload::Mte1,
                )?;
                if outcome != C220Mte1ReadEventOutcome::Readiness {
                    self.trace
                        .push(C220MtePipelineEvent::Generator(kind, outcome));
                }
            }
            Callback::Set2d(phase) => {
                let [l0a, l0b] = &mut self.l0;
                let outcome = self.set2d_events.handle(
                    phase,
                    &mut self.events,
                    &mut self.set2d,
                    C220Set2dGates {
                        hardware_sync_blocked: blocked,
                        l1_prefetch_blocked: false,
                    },
                    C220Set2dOutputs::L0 { l0a, l0b },
                )?;
                if outcome != C220Set2dEventOutcome::Readiness {
                    self.trace.push(C220MtePipelineEvent::Set2d(outcome));
                }
            }
            _ => unreachable!("MTE1 generator callback"),
        }
        Ok(())
    }
}
