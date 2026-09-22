use super::mte1::{C220Mte1Command, C220Mte1Generator, C220Mte1Issue};
use crate::isa::c220::mte::set2d::{C220Set2dDestination, C220Set2dFill};
use crate::sim::c220::mte::set2d::{
    C220Set2dBandwidths, C220Set2dEventOutcome, C220Set2dEvents, C220Set2dFrontend,
    C220Set2dFrontendError, C220Set2dGates, C220Set2dIssue, C220Set2dOutputs,
};
use std::num::NonZeroU32;

use super::mte1::frontend::{
    C220Mte1ReadBandwidths, C220Mte1ReadEventOutcome, C220Mte1ReadEvents, C220Mte1ReadFrontend,
    C220Mte1ReadFrontendError, C220Mte1ReadKind, C220Mte1ReadUop,
};
use crate::sim::c220::memory::l1::{
    C220L1Callback, C220L1EventOutcome, C220L1Events, C220L1Geometry, C220L1Port, C220L1Transport,
    C220L1TransportError,
};
use crate::sim::c220::mte::C220MteGeneratorCallback;
use crate::sim::c220::mte::interface::{
    C220L0WriteCallback, C220L0WriteError, C220L0WriteEventOutcome, C220L0WriteEvents,
    C220L0WritePipeline, C220L0WritePort, C220MteL1Callback, C220MteL1CycleInputs, C220MteL1Error,
    C220MteL1EventOutcome, C220MteL1Events, C220MteL1Interface, C220MteL1OutputCredits,
    C220MteL1OutputDestination, C220MteL1WriteCallback, C220MteL1WriteError,
    C220MteL1WriteEventInputs, C220MteL1WriteEventOutcome, C220MteL1WriteEvents,
    C220MteL1WriteInterface,
};
use crate::sim::common::event::{EventDispatcher, EventError, EventId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MtePipelineConfig {
    pub l1: C220L1Geometry,
    pub read_width: NonZeroU32,
    pub output_bandwidths: C220Mte1ReadBandwidths,
    pub set2d_bandwidths: C220Set2dBandwidths,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Callback {
    Memory(C220L1Callback),
    Interface(C220MteL1Callback),
    L1Write(C220MteL1WriteCallback),
    L0(bool, C220L0WriteCallback),
    Generator(C220Mte1ReadKind, C220MteGeneratorCallback),
    Set2d(C220MteGeneratorCallback),
    Set2dL1(C220MteGeneratorCallback),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum C220MtePipelineEvent {
    Memory(C220L1EventOutcome),
    Interface(C220MteL1EventOutcome<C220Mte1ReadUop>),
    L1Write(C220MteL1WriteEventOutcome),
    L0a(C220L0WriteEventOutcome),
    L0b(C220L0WriteEventOutcome),
    Generator(C220Mte1ReadKind, C220Mte1ReadEventOutcome),
    Set2d(C220Set2dEventOutcome),
    Set2dL1(C220Set2dEventOutcome),
}

#[derive(Debug, thiserror::Error)]
pub enum C220MtePipelineError {
    #[error("SET_2D to L1 belongs to the MTE2 command lane")]
    WrongCommandLane,
    #[error("the L1 fill generator cannot accept an L0 fill")]
    WrongFillDestination,
    #[error(transparent)]
    L1Write(#[from] C220MteL1WriteError),
    #[error(transparent)]
    Set2d(#[from] C220Set2dFrontendError),
    #[error("MTE1 generator cannot accept the command or switch while its predecessor is active")]
    CommandBusy,
    #[error("MTE active clock skipped tick {expected} to {requested}")]
    SkippedTick { expected: u64, requested: u64 },
    #[error(transparent)]
    Events(#[from] EventError),
    #[error(transparent)]
    Generator(#[from] C220Mte1ReadFrontendError),
    #[error(transparent)]
    Interface(#[from] C220MteL1Error),
    #[error(transparent)]
    Memory(#[from] C220L1TransportError),
    #[error(transparent)]
    L0(#[from] C220L0WriteError),
}

#[cfg(test)]
mod tests;

/// Physical MTE paths. Generators, shared L1 interfaces, L1 service and
/// destinations execute on one event dispatcher. Retirement is an observed
/// destination acknowledgment, not a prediction made at command admission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220MtePipeline {
    events: EventDispatcher<Callback>,
    clock: EventId,
    memory_events: C220L1Events,
    interface_events: C220MteL1Events,
    write_events: C220MteL1WriteEvents,
    l0_events: [C220L0WriteEvents; 2],
    generator_events: [C220Mte1ReadEvents; 2],
    set2d_events: C220Set2dEvents,
    set2d_l1_events: C220Set2dEvents,
    memory: C220L1Transport,
    interface: C220MteL1Interface<C220Mte1ReadUop>,
    write_interface: C220MteL1WriteInterface,
    l0: [C220L0WritePipeline; 2],
    generators: [C220Mte1ReadFrontend; 2],
    set2d: C220Set2dFrontend,
    set2d_l1: C220Set2dFrontend,
    l1_prefetch_blocked: bool,
    selected_generator: Option<C220Mte1Generator>,
    completions: Vec<u64>,
    l1_fill_completions: Vec<u64>,
    trace: Vec<C220MtePipelineEvent>,
    last_advance: Option<u64>,
}

impl C220MtePipeline {
    pub fn new(tick: u64, config: C220MtePipelineConfig) -> Self {
        let mut events = EventDispatcher::new(tick);
        let clock = events.add_event();
        let memory_events = C220L1Events::register(&mut events, clock, Callback::Memory);
        let interface_events = C220MteL1Events::register(&mut events, clock, Callback::Interface);
        let write_events = C220MteL1WriteEvents::register(&mut events, clock, Callback::L1Write);
        let l0_events = [false, true].map(|b| {
            C220L0WriteEvents::register(&mut events, clock, |phase| Callback::L0(b, phase))
        });
        let generator_events = C220Mte1ReadKind::ALL.map(|kind| {
            C220Mte1ReadEvents::register(&mut events, clock, |phase| {
                Callback::Generator(kind, phase)
            })
        });
        let set2d_events = C220Set2dEvents::register(&mut events, clock, Callback::Set2d);
        let set2d_l1_events = C220Set2dEvents::register(&mut events, clock, Callback::Set2dL1);
        Self {
            events,
            clock,
            memory_events,
            interface_events,
            write_events,
            l0_events,
            generator_events,
            set2d_events,
            set2d_l1_events,
            memory: C220L1Transport::new(config.l1),
            interface: C220MteL1Interface::default(),
            write_interface: C220MteL1WriteInterface::default(),
            l0: std::array::from_fn(|_| C220L0WritePipeline::default()),
            generators: C220Mte1ReadKind::ALL.map(|kind| {
                C220Mte1ReadFrontend::new(kind, config.read_width, config.output_bandwidths)
            }),
            selected_generator: None,
            set2d: C220Set2dFrontend::new(config.set2d_bandwidths),
            set2d_l1: C220Set2dFrontend::new(config.set2d_bandwidths),
            l1_prefetch_blocked: false,
            completions: Vec::new(),
            l1_fill_completions: Vec::new(),
            trace: Vec::new(),
            last_advance: None,
        }
    }

    pub fn can_issue_mte1(&self, command: C220Mte1Command) -> bool {
        if matches!(command, C220Mte1Command::Set2d(fill) if fill.instruction.destination == C220Set2dDestination::L1)
        {
            return false;
        }
        command.is_disabled()
            || (match command {
                C220Mte1Command::Read(transfer) => self.generator(transfer.kind()).can_issue(),
                C220Mte1Command::Set2d(_) => self.set2d.can_issue(),
            } && (self.selected_generator == Some(command.generator())
                || self.selected_generator_idle()))
    }
    pub fn selected_generator_idle(&self) -> bool {
        self.selected_generator.is_none_or(|kind| match kind {
            C220Mte1Generator::Read(kind) => self.generator(kind).is_idle(),
            C220Mte1Generator::Set2d => self.set2d.is_idle(),
        })
    }
    pub fn selected_generator(&self) -> Option<C220Mte1Generator> {
        self.selected_generator
    }
    pub fn is_idle(&self) -> bool {
        self.generators.iter().all(C220Mte1ReadFrontend::is_idle)
            && self.set2d.is_idle()
            && self.set2d_l1.is_idle()
            && self.write_interface.is_idle()
            && self.interface.is_idle()
            && self.memory.is_idle()
            && self.l0.iter().all(C220L0WritePipeline::is_idle)
    }
    pub fn next_event_tick(&self) -> Option<u64> {
        (!self.is_idle()).then(|| self.events.tick().saturating_add(1))
    }
    pub fn generator(&self, kind: C220Mte1ReadKind) -> &C220Mte1ReadFrontend {
        &self.generators[kind.index()]
    }
    pub fn set2d_generator(&self) -> &C220Set2dFrontend {
        &self.set2d
    }
    pub fn l1_fill_generator(&self) -> &C220Set2dFrontend {
        &self.set2d_l1
    }
    pub fn l1_write_interface(&self) -> &C220MteL1WriteInterface {
        &self.write_interface
    }
    pub fn l1_fill_completions(&self) -> &[u64] {
        &self.l1_fill_completions
    }
    pub fn set_l1_prefetch_blocked(&mut self, blocked: bool) {
        self.l1_prefetch_blocked = blocked;
    }

    /// Admission to this generator only; the MTE2 command owner must first
    /// resolve its lane-level dependencies and generator switching rules.
    pub fn can_issue_l1_fill(&self, fill: C220Set2dFill) -> bool {
        fill.instruction.destination == C220Set2dDestination::L1
            && (fill.descriptor.is_disabled() || self.set2d_l1.can_issue())
    }

    pub fn issue_l1_fill(
        &mut self,
        instruction_id: u64,
        fill: C220Set2dFill,
    ) -> Result<C220Set2dIssue, C220MtePipelineError> {
        if fill.instruction.destination != C220Set2dDestination::L1 {
            return Err(C220MtePipelineError::WrongFillDestination);
        }
        if fill.descriptor.is_disabled() {
            return Ok(C220Set2dIssue {
                tick: self.events.tick(),
                instruction_id,
                uop_count: 0,
                completion_ready: true,
            });
        }
        Ok(self.set2d_l1_events.issue(
            &mut self.events,
            &mut self.set2d_l1,
            instruction_id,
            fill,
        )?)
    }
    pub fn memory(&self) -> &C220L1Transport {
        &self.memory
    }
    pub fn interface(&self) -> &C220MteL1Interface<C220Mte1ReadUop> {
        &self.interface
    }
    pub fn l0a(&self) -> &C220L0WritePipeline {
        &self.l0[0]
    }
    pub fn l0b(&self) -> &C220L0WritePipeline {
        &self.l0[1]
    }
    pub fn mte1_completions(&self) -> &[u64] {
        &self.completions
    }
    pub fn last_events(&self) -> &[C220MtePipelineEvent] {
        &self.trace
    }

    pub fn issue_mte1(
        &mut self,
        instruction_id: u64,
        command: C220Mte1Command,
    ) -> Result<C220Mte1Issue, C220MtePipelineError> {
        if matches!(command, C220Mte1Command::Set2d(fill) if fill.instruction.destination == C220Set2dDestination::L1)
        {
            return Err(C220MtePipelineError::WrongCommandLane);
        }
        if !self.can_issue_mte1(command) {
            return Err(C220MtePipelineError::CommandBusy);
        }
        if command.is_disabled() {
            return Ok(C220Mte1Issue {
                tick: self.events.tick(),
                instruction_id,
                uop_count: 0,
                completion_ready: true,
            });
        }
        let issue = match command {
            C220Mte1Command::Read(transfer) => {
                let index = transfer.kind().index();
                self.generator_events[index]
                    .issue(
                        &mut self.events,
                        &mut self.generators[index],
                        instruction_id,
                        transfer,
                    )?
                    .into()
            }
            C220Mte1Command::Set2d(fill) => self
                .set2d_events
                .issue(&mut self.events, &mut self.set2d, instruction_id, fill)?
                .into(),
        };
        self.selected_generator = Some(command.generator());
        Ok(issue)
    }

    /// The owner supplies every active clock tick and consumes completions
    /// before advancing again. Idle intervals may be skipped.
    pub fn advance(&mut self, tick: u64) -> Result<(), C220MtePipelineError> {
        if self.last_advance == Some(tick) {
            return Ok(());
        }
        if !self.is_idle()
            && let Some(expected) = self.events.tick().checked_add(1)
            && tick > expected
        {
            return Err(C220MtePipelineError::SkippedTick {
                expected,
                requested: tick,
            });
        }
        self.events.advance_to(tick)?;
        self.completions.clear();
        self.l1_fill_completions.clear();
        self.trace.clear();
        self.events.notify_at(self.clock, tick);
        while let Some(invocation) = self.events.next_callback() {
            match invocation.callback {
                Callback::Set2dL1(phase) => {
                    let outcome = self.set2d_l1_events.handle(
                        phase,
                        &mut self.events,
                        &mut self.set2d_l1,
                        C220Set2dGates {
                            hardware_sync_blocked: false,
                            l1_prefetch_blocked: self.l1_prefetch_blocked,
                        },
                        C220Set2dOutputs::L1(&mut self.write_interface),
                    )?;
                    if outcome != C220Set2dEventOutcome::Readiness {
                        self.trace.push(C220MtePipelineEvent::Set2dL1(outcome));
                    }
                }
                Callback::L1Write(phase) => {
                    let inputs = C220MteL1WriteEventInputs {
                        request_ready: self.memory.request_ready(C220L1Port::MteWrite),
                        response: self
                            .memory
                            .responses(C220L1Port::MteWrite)
                            .front()
                            .filter(|head| head.ready_tick <= tick)
                            .map(|head| head.payload.request.id),
                    };
                    let outcome = self.write_events.handle(
                        phase,
                        &mut self.events,
                        &mut self.write_interface,
                        inputs,
                    )?;
                    match outcome {
                        C220MteL1WriteEventOutcome::Sent(send) => {
                            if let Some(request) = send.sent {
                                assert!(self.memory.send_request(
                                    tick,
                                    C220L1Port::MteWrite,
                                    request.l1_request()
                                )?);
                            }
                        }
                        C220MteL1WriteEventOutcome::Response(Some(request)) => {
                            let response = self
                                .memory
                                .receive_response(tick, C220L1Port::MteWrite)?
                                .expect("accepted write response");
                            assert_eq!(request.id, response.request.id);
                        }
                        C220MteL1WriteEventOutcome::Acknowledged(Some(ack)) => {
                            if let Some(id) = ack.retired_instruction() {
                                self.l1_fill_completions.push(id);
                            }
                        }
                        _ => {}
                    }
                    if outcome != C220MteL1WriteEventOutcome::Readiness {
                        self.trace.push(C220MtePipelineEvent::L1Write(outcome));
                    }
                }
                Callback::Set2d(phase) => {
                    let [l0a, l0b] = &mut self.l0;
                    let outcome = self.set2d_events.handle(
                        phase,
                        &mut self.events,
                        &mut self.set2d,
                        C220Set2dGates::default(),
                        C220Set2dOutputs::L0 { l0a, l0b },
                    )?;
                    if outcome != C220Set2dEventOutcome::Readiness {
                        self.trace.push(C220MtePipelineEvent::Set2d(outcome));
                    }
                }
                Callback::Memory(phase) => {
                    let outcome =
                        self.memory_events
                            .handle(phase, &mut self.events, &mut self.memory)?;
                    if outcome != C220L1EventOutcome::Readiness {
                        self.trace.push(C220MtePipelineEvent::Memory(outcome));
                    }
                }
                Callback::Generator(kind, phase) => {
                    let outcome = self.generator_events[kind.index()].handle(
                        phase,
                        &mut self.events,
                        &mut self.generators[kind.index()],
                        false,
                        &mut self.interface,
                    )?;
                    if outcome != C220Mte1ReadEventOutcome::Readiness {
                        self.trace
                            .push(C220MtePipelineEvent::Generator(kind, outcome));
                    }
                }
                Callback::L0(b, phase) => {
                    let outcome = self.l0_events[usize::from(b)].handle(
                        phase,
                        &mut self.events,
                        &mut self.l0[usize::from(b)],
                    )?;
                    if let C220L0WriteEventOutcome::Acknowledged(Some(ack)) = outcome
                        && let Some(id) = ack.retired_instruction()
                    {
                        self.completions.push(id);
                    }
                    if outcome != C220L0WriteEventOutcome::Readiness {
                        self.trace.push(if b {
                            C220MtePipelineEvent::L0b(outcome)
                        } else {
                            C220MtePipelineEvent::L0a(outcome)
                        });
                    }
                }
                Callback::Interface(phase) => {
                    let response = self
                        .memory
                        .responses(C220L1Port::MteRead)
                        .front()
                        .filter(|head| head.ready_tick <= tick)
                        .map(|head| head.payload.request.id);
                    let inputs = C220MteL1CycleInputs {
                        request_ready: self.memory.request_ready(C220L1Port::MteRead),
                        response,
                        output_credits: C220MteL1OutputCredits {
                            l0a: [self.l0[0].can_push(C220L0WritePort::Port0), false, false],
                            l0b: [self.l0[1].can_push(C220L0WritePort::Port0), false, false],
                        },
                    };
                    let outcome = self.interface_events.handle(
                        phase,
                        &mut self.events,
                        &mut self.interface,
                        inputs,
                    )?;
                    match &outcome {
                        C220MteL1EventOutcome::Request(send) => {
                            if let Some(request) = send.sent {
                                assert!(self.memory.send_request(
                                    tick,
                                    C220L1Port::MteRead,
                                    request.l1_request()
                                )?);
                            }
                        }
                        C220MteL1EventOutcome::Response(Some(request)) => {
                            let response = self
                                .memory
                                .receive_response(tick, C220L1Port::MteRead)?
                                .expect("accepted response head");
                            assert_eq!(request.id, response.request.id);
                        }
                        C220MteL1EventOutcome::Output(output) => {
                            if let Some(sent) = output.sent {
                                let target = match sent.destination {
                                    C220MteL1OutputDestination::L0a(port) => Some((0, port)),
                                    C220MteL1OutputDestination::L0b(port) => Some((1, port)),
                                    C220MteL1OutputDestination::Bt => None,
                                };
                                if let Some((index, port)) = target {
                                    assert!(self.l0[index].push(tick, port, sent.fragment)?);
                                }
                            }
                        }
                        C220MteL1EventOutcome::Retired(Some(output))
                            if output.fragment.last_in_instruction =>
                        {
                            self.completions.push(output.fragment.instruction_id);
                        }
                        _ => {}
                    }
                    if outcome != C220MteL1EventOutcome::Readiness {
                        self.trace.push(C220MtePipelineEvent::Interface(outcome));
                    }
                }
            }
        }
        self.last_advance = Some(tick);
        Ok(())
    }
}
