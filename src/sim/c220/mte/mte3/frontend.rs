use std::collections::{BTreeSet, VecDeque};

use super::{C220_MTE3_OUTSTANDING_LIMIT, C220Mte3TransferPlan};
use crate::sim::c220::mte::C220MteGeneratorCallback;
use crate::sim::c220::mte::dma::{
    C220DmaEventOutcome, C220DmaEvents, C220DmaFrontend, C220DmaFrontendError, C220DmaGenerated,
};
use crate::sim::c220::mte::uop::{C220DmaUopError, C220DmaUops, mte3_uops};
use crate::sim::common::event::{EventDispatcher, EventId, ProcessId};

const COMMAND_CAPACITY: usize = 3;
const COMMAND_TICKS: u64 = 3;
const RECORD_TICKS: u64 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220Mte3Callback {
    CommandReady,
    Dispatch,
    Generator(C220MteGeneratorCallback),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220Mte3FrontendEvent {
    Dispatched { instruction_id: u64, tick: u64 },
    Generator(C220DmaEventOutcome),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Mte3Record {
    pub instruction_id: u64,
    pub issue_tick: u64,
    pub dispatch_tick: Option<u64>,
    pub transfer: C220Mte3TransferPlan,
}

#[derive(Debug, thiserror::Error)]
pub enum C220Mte3FrontendError {
    #[error("MTE3 command or outstanding queue is full")]
    QueueFull,
    #[error("MTE3 instruction ID {0} is already active")]
    DuplicateInstruction(u64),
    #[error(
        "MTE3 response does not match a delivered request: instruction {instruction_id}, request {uop_index}"
    )]
    UnexpectedResponse { instruction_id: u64, uop_index: u64 },
    #[error("MTE3 instruction {0} is not eligible for ordered retirement")]
    NotRetirable(u64),
    #[error("MTE3 frontend time overflowed")]
    TimeOverflow,
    #[error(transparent)]
    Dma(#[from] C220DmaFrontendError),
    #[error(transparent)]
    Uop(#[from] C220DmaUopError),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Command {
    instruction_id: u64,
    ready_tick: u64,
    requests: C220DmaUops,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Pending {
    record: C220Mte3Record,
    record_ready_tick: Option<u64>,
    tail_delivered: bool,
    responses: BTreeSet<u64>,
}

/// Command decode, ordinary DMA generation and destination acknowledgments.
/// Output consumption transfers ownership to the transport; every delivered
/// request must be acknowledged. The owner performs the functional operation
/// before explicitly releasing the eligible retirement record.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct C220Mte3Frontend {
    commands: VecDeque<Command>,
    records: VecDeque<Pending>,
    generator: C220DmaFrontend,
    output: Option<C220DmaGenerated>,
    hardware_sync_blocked: bool,
    last_retirement_tick: Option<u64>,
}

impl C220Mte3Frontend {
    pub fn can_issue(&self) -> bool {
        self.commands.len() < COMMAND_CAPACITY && self.records.len() < C220_MTE3_OUTSTANDING_LIMIT
    }

    pub fn is_idle(&self) -> bool {
        self.records.is_empty() && self.generator.is_idle() && self.output.is_none()
    }

    pub fn records(&self) -> impl Iterator<Item = C220Mte3Record> + '_ {
        self.records.iter().map(|entry| entry.record)
    }

    pub fn queued_commands(&self) -> usize {
        self.commands.len()
    }

    pub fn generator(&self) -> &C220DmaFrontend {
        &self.generator
    }

    pub fn output(&self) -> Option<C220DmaGenerated> {
        self.output
    }

    pub fn take_output(&mut self) -> Option<C220DmaGenerated> {
        let output = self.output.take()?;
        let pending = self
            .records
            .iter_mut()
            .find(|p| p.record.instruction_id == output.instruction_id)
            .expect("generated request retains its command record");
        pending.responses.insert(output.uop_index);
        pending.tail_delivered |= output.last_in_instruction;
        Some(output)
    }

    pub fn set_hardware_sync_blocked(&mut self, blocked: bool) {
        self.hardware_sync_blocked = blocked;
    }

    pub fn acknowledge(
        &mut self,
        instruction_id: u64,
        uop_index: u64,
    ) -> Result<(), C220Mte3FrontendError> {
        if self
            .records
            .iter_mut()
            .find(|p| p.record.instruction_id == instruction_id)
            .is_some_and(|p| p.responses.remove(&uop_index))
        {
            Ok(())
        } else {
            Err(C220Mte3FrontendError::UnexpectedResponse {
                instruction_id,
                uop_index,
            })
        }
    }

    pub fn retirement_candidate(&self, tick: u64) -> Option<C220Mte3Record> {
        if self
            .last_retirement_tick
            .is_some_and(|previous| tick <= previous)
        {
            return None;
        }
        self.records
            .front()
            .filter(|p| {
                p.record_ready_tick.is_some_and(|ready| tick >= ready)
                    && p.tail_delivered
                    && p.responses.is_empty()
            })
            .map(|p| p.record)
    }

    pub fn retire(
        &mut self,
        tick: u64,
        instruction_id: u64,
    ) -> Result<C220Mte3Record, C220Mte3FrontendError> {
        let record = self
            .retirement_candidate(tick)
            .filter(|p| p.instruction_id == instruction_id)
            .ok_or(C220Mte3FrontendError::NotRetirable(instruction_id))?;
        self.records.pop_front();
        self.last_retirement_tick = Some(tick);
        Ok(record)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::sim::c220::mte) struct C220Mte3Events {
    command_probe: ProcessId,
    command_valid: EventId,
    dma: C220DmaEvents,
}

impl C220Mte3Events {
    pub(in crate::sim::c220::mte) fn register<T: Copy>(
        events: &mut EventDispatcher<T>,
        clock: EventId,
        tag: impl Fn(C220Mte3Callback) -> T,
    ) -> Self {
        let command_valid = events.add_event();
        let command_probe = events.add_process(tag(C220Mte3Callback::CommandReady), false);
        events.subscribe(clock, command_probe);
        events.set_process_enabled(command_probe, false);
        let dispatch = events.add_process(tag(C220Mte3Callback::Dispatch), false);
        events.subscribe(command_valid, dispatch);
        let dma = C220DmaEvents::register(events, clock, |phase| {
            tag(C220Mte3Callback::Generator(phase))
        });
        Self {
            command_probe,
            command_valid,
            dma,
        }
    }

    pub(in crate::sim::c220::mte) fn issue<T: Copy>(
        &self,
        events: &mut EventDispatcher<T>,
        frontend: &mut C220Mte3Frontend,
        instruction_id: u64,
        transfer: C220Mte3TransferPlan,
    ) -> Result<C220Mte3Record, C220Mte3FrontendError> {
        if !frontend.can_issue() {
            return Err(C220Mte3FrontendError::QueueFull);
        }
        if frontend
            .records
            .iter()
            .any(|p| p.record.instruction_id == instruction_id)
        {
            return Err(C220Mte3FrontendError::DuplicateInstruction(instruction_id));
        }
        let ready_tick = events
            .tick()
            .checked_add(COMMAND_TICKS)
            .ok_or(C220Mte3FrontendError::TimeOverflow)?;
        let requests = mte3_uops(transfer)?;
        let record = C220Mte3Record {
            instruction_id,
            issue_tick: events.tick(),
            dispatch_tick: None,
            transfer,
        };
        frontend.records.push_back(Pending {
            record,
            record_ready_tick: None,
            tail_delivered: false,
            responses: BTreeSet::new(),
        });
        frontend.commands.push_back(Command {
            instruction_id,
            ready_tick,
            requests,
        });
        events.set_process_enabled(self.command_probe, true);
        Ok(record)
    }

    pub(in crate::sim::c220::mte) fn handle<T: Copy>(
        &self,
        phase: C220Mte3Callback,
        events: &mut EventDispatcher<T>,
        frontend: &mut C220Mte3Frontend,
    ) -> Result<Option<C220Mte3FrontendEvent>, C220Mte3FrontendError> {
        match phase {
            C220Mte3Callback::CommandReady => {
                match frontend.commands.front() {
                    None => events.set_process_enabled(self.command_probe, false),
                    Some(command) if command.ready_tick <= events.tick() => {
                        events.notify_at(self.command_valid, events.tick());
                    }
                    Some(_) => {}
                }
                Ok(None)
            }
            C220Mte3Callback::Dispatch => {
                let Some(command) = frontend.commands.front() else {
                    return Ok(None);
                };
                let disabled = command.requests.clone().next().is_none();
                if !disabled && !frontend.generator.can_issue() {
                    return Ok(None);
                }
                let ready = events
                    .tick()
                    .checked_add(RECORD_TICKS)
                    .ok_or(C220Mte3FrontendError::TimeOverflow)?;
                if !disabled {
                    self.dma.issue(
                        events,
                        &mut frontend.generator,
                        command.instruction_id,
                        command.requests.clone(),
                    )?;
                }
                let id = command.instruction_id;
                let pending = frontend
                    .records
                    .iter_mut()
                    .find(|p| p.record.instruction_id == id)
                    .expect("queued command retains its record");
                pending.record.dispatch_tick = Some(events.tick());
                pending.record_ready_tick = Some(ready);
                pending.tail_delivered = disabled;
                frontend.commands.pop_front();
                Ok(Some(C220Mte3FrontendEvent::Dispatched {
                    instruction_id: id,
                    tick: events.tick(),
                }))
            }
            C220Mte3Callback::Generator(phase) => {
                let outcome = self.dma.handle(
                    phase,
                    events,
                    &mut frontend.generator,
                    frontend.hardware_sync_blocked,
                    frontend.output.is_none(),
                )?;
                if let C220DmaEventOutcome::Sent(send) = outcome
                    && let Some(sent) = send.sent
                {
                    frontend.output = Some(sent);
                }
                Ok((outcome != C220DmaEventOutcome::Readiness)
                    .then_some(C220Mte3FrontendEvent::Generator(outcome)))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::c220::mte::{C220DmaMovDescriptor, CAPTURED_C220_MOV_UB_TO_OUT_WORD};

    #[test]
    fn command_delay_backpressure_and_ordered_acknowledgments() {
        let mut events = EventDispatcher::new(0);
        let clock = events.add_event();
        let callbacks = C220Mte3Events::register(&mut events, clock, |phase| phase);
        let mut frontend = C220Mte3Frontend::default();
        let plan = C220Mte3TransferPlan {
            descriptor: C220DmaMovDescriptor::decode(
                CAPTURED_C220_MOV_UB_TO_OUT_WORD,
                (32 << 16) | (1 << 4),
            )
            .unwrap(),
            source_address: 0,
            destination_address: 0x2000,
            bytes: 1024,
            dma_mode_word: 5,
        };
        let mut delivered = Vec::new();
        let disabled = C220Mte3TransferPlan {
            descriptor: C220DmaMovDescriptor::decode(CAPTURED_C220_MOV_UB_TO_OUT_WORD, 0).unwrap(),
            bytes: 0,
            ..plan
        };
        for tick in 0..60 {
            events.advance_to(tick).unwrap();
            if tick < 3 {
                callbacks
                    .issue(
                        &mut events,
                        &mut frontend,
                        tick,
                        if tick == 2 { disabled } else { plan },
                    )
                    .unwrap();
            }
            if tick == 2 {
                assert!(!frontend.can_issue());
                assert!(matches!(
                    callbacks.issue(&mut events, &mut frontend, 99, plan),
                    Err(C220Mte3FrontendError::QueueFull)
                ));
            }
            events.notify_at(clock, tick);
            while let Some(invocation) = events.next_callback() {
                callbacks
                    .handle(invocation.callback, &mut events, &mut frontend)
                    .unwrap();
            }
            if tick == 2 {
                assert_eq!(frontend.records().next().unwrap().dispatch_tick, None);
            }
            if tick == 3 {
                assert_eq!(frontend.records().next().unwrap().dispatch_tick, Some(3));
            }
            if tick < 7 {
                assert!(frontend.output().is_none());
            }
            if tick == 15 {
                assert_eq!(frontend.generator().generated().len(), 4);
                assert_eq!(frontend.output().unwrap().uop_index, 0);
                assert!(frontend.acknowledge(0, 0).is_err());
            }
            if tick >= 16
                && let Some(request) = frontend.take_output()
            {
                delivered.push((request.instruction_id, request.uop_index));
            }
        }
        assert_eq!(delivered.len(), 16);
        for &(id, index) in delivered.iter().rev().filter(|(id, _)| *id != 0) {
            frontend.acknowledge(id, index).unwrap();
        }
        assert!(frontend.retirement_candidate(60).is_none());
        assert!(frontend.retire(60, 1).is_err());
        for &(id, index) in delivered.iter().rev().filter(|(id, _)| *id == 0) {
            frontend.acknowledge(id, index).unwrap();
            assert!(frontend.acknowledge(id, index).is_err());
        }
        assert_eq!(frontend.retire(60, 0).unwrap().instruction_id, 0);
        assert!(frontend.retire(60, 1).is_err());
        frontend.retire(61, 1).unwrap();
        frontend.retire(62, 2).unwrap();
        assert!(frontend.is_idle());
    }
}
