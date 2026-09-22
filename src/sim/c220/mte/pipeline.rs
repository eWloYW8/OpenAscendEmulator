use super::dma::{
    C220DmaEventOutcome, C220DmaEvents, C220DmaFrontend, C220DmaFrontendError, C220DmaGenerated,
    C220DmaIssue,
};
use super::interface::biu_read::returns::{
    C220BiuReadBeat, C220BiuReadReturns, C220BiuReturnCallback, C220BiuReturnError,
    C220BiuReturnEvent, C220BiuReturnEvents,
};
use super::interface::biu_read::write::C220BiuWriteDestination;
use super::interface::biu_read::{
    C220BiuReadCallback, C220BiuReadConfig, C220BiuReadError, C220BiuReadEvent, C220BiuReadEvents,
    C220BiuReadFrontend, C220BiuReadInput, C220BiuReadRequest, C220BiuSubcore,
};
use super::interface::ub_write::{
    C220UbWriteCallback, C220UbWriteError, C220UbWriteEvent, C220UbWriteEvents,
    C220UbWriteInterface, C220UbWriteRequest,
};
use super::mte1::{C220Mte1Command, C220Mte1Generator, C220Mte1Issue};
use super::mte2::C220Mte2TransferPlan;
use super::uop::{C220DmaUopError, mte2_uops};
use crate::isa::c220::mte::set2d::{C220Set2dDestination, C220Set2dFill};
use crate::sim::c220::memory::ub_service::{
    C220UbMteService, C220UbServiceCycle, C220UbServiceError,
};
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
    Dma(C220MteGeneratorCallback),
    BiuRead(C220BiuReadCallback),
    BiuReturn(C220BiuReturnCallback),
    UbWrite(usize, C220UbWriteCallback),
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
    Dma(C220DmaEventOutcome),
    BiuRead(C220BiuReadEvent),
    BiuReturn(C220BiuReturnEvent),
    UbWrite(C220BiuSubcore, C220UbWriteEvent),
    UbRequest(C220BiuSubcore, C220UbWriteRequest),
    UbResponse(C220BiuSubcore, C220UbWriteRequest),
    UbService(C220BiuSubcore, C220UbServiceCycle),
}

#[derive(Debug, thiserror::Error)]
pub enum C220MtePipelineError {
    #[error(transparent)]
    UbService(#[from] C220UbServiceError),
    #[error(transparent)]
    UbWrite(#[from] C220UbWriteError),
    #[error("BIU-connected DMA completion is owned by the UB write interface")]
    DestinationOwnedCompletion,
    #[error(transparent)]
    BiuReturn(#[from] C220BiuReturnError),
    #[error(transparent)]
    BiuRead(#[from] C220BiuReadError),
    #[error("BIU read frontend is not connected")]
    BiuDisconnected,
    #[error("BIU request has not been consumed by its transport")]
    BiuRequestUndelivered,
    #[error("the current MTE2 DMA generator targets UB and requires a vector subcore")]
    BiuUbSubcoreRequired,
    #[error("MTE2 DMA requires a connected request/response consumer")]
    DmaDisconnected,
    #[error(transparent)]
    Dma(#[from] C220DmaFrontendError),
    #[error(transparent)]
    DmaUop(#[from] C220DmaUopError),
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mte2Generator {
    Dma,
    L1Fill,
}

#[cfg(test)]
mod tests;
mod ub;

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
    dma_events: C220DmaEvents,
    biu_events: C220BiuReadEvents,
    biu_return_events: C220BiuReturnEvents,
    ub_write_events: [C220UbWriteEvents; 2],
    memory: C220L1Transport,
    interface: C220MteL1Interface<C220Mte1ReadUop>,
    write_interface: C220MteL1WriteInterface,
    l0: [C220L0WritePipeline; 2],
    generators: [C220Mte1ReadFrontend; 2],
    set2d: C220Set2dFrontend,
    set2d_l1: C220Set2dFrontend,
    dma: C220DmaFrontend,
    dma_connected: bool,
    dma_output: Option<C220DmaGenerated>,
    biu_read: Option<C220BiuReadFrontend>,
    biu_returns: Option<C220BiuReadReturns>,
    biu_subcore: C220BiuSubcore,
    biu_output: Option<C220BiuReadRequest>,
    ub_write: [C220UbWriteInterface; 2],
    ub_memory: [C220UbMteService; 2],
    last_ub_service: Option<u64>,
    dma_hardware_sync_blocked: bool,
    selected_mte2_generator: Option<Mte2Generator>,
    l1_prefetch_blocked: bool,
    selected_generator: Option<C220Mte1Generator>,
    completions: Vec<u64>,
    l1_fill_completions: Vec<u64>,
    dma_completions: Vec<u64>,
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
        let dma_events = C220DmaEvents::register(&mut events, clock, Callback::Dma);
        let biu_events = C220BiuReadEvents::register(&mut events, clock, Callback::BiuRead);
        let biu_return_events =
            C220BiuReturnEvents::register(&mut events, clock, Callback::BiuReturn);
        let ub_write_events = [0, 1].map(|index| {
            C220UbWriteEvents::register(&mut events, clock, |phase| Callback::UbWrite(index, phase))
        });
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
            dma_events,
            biu_events,
            biu_return_events,
            ub_write_events,
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
            dma: C220DmaFrontend::default(),
            dma_connected: false,
            dma_output: None,
            biu_read: None,
            biu_returns: None,
            biu_subcore: C220BiuSubcore::Vector0,
            biu_output: None,
            ub_write: std::array::from_fn(|_| C220UbWriteInterface::default()),
            ub_memory: std::array::from_fn(|_| C220UbMteService::default()),
            last_ub_service: None,
            dma_hardware_sync_blocked: false,
            selected_mte2_generator: None,
            l1_prefetch_blocked: false,
            completions: Vec::new(),
            l1_fill_completions: Vec::new(),
            dma_completions: Vec::new(),
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
            && self.dma.is_idle()
            && self.dma_output.is_none()
            && self.biu_output.is_none()
            && self.ub_write.iter().all(C220UbWriteInterface::is_idle)
            && self.ub_memory.iter().all(C220UbMteService::is_idle)
            && self
                .biu_read
                .as_ref()
                .is_none_or(C220BiuReadFrontend::is_idle)
            && self
                .biu_returns
                .as_ref()
                .is_none_or(C220BiuReadReturns::is_idle)
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

    /// Enables explicit DMA requests. The caller must consume every request
    /// and report destination completion; this boundary is not a BIU model.
    pub fn connect_mte2_dma(&mut self) -> Result<(), C220MtePipelineError> {
        if !self.is_idle() {
            return Err(C220MtePipelineError::CommandBusy);
        }
        self.dma_connected = true;
        self.biu_read = None;
        self.biu_returns = None;
        Ok(())
    }

    /// Connects BIU reads and return reordering to native UB write service.
    /// The external memory service supplies only the BIU read response beats.
    pub fn connect_mte2_biu(
        &mut self,
        config: C220BiuReadConfig,
        subcore: C220BiuSubcore,
    ) -> Result<(), C220MtePipelineError> {
        if !self.is_idle() {
            return Err(C220MtePipelineError::CommandBusy);
        }
        if subcore == C220BiuSubcore::Cube {
            return Err(C220MtePipelineError::BiuUbSubcoreRequired);
        }
        let returns = C220BiuReadReturns::new(
            config.outstanding,
            config.group_vector_returns,
            config.write_bandwidths,
        )?;
        self.biu_read = Some(C220BiuReadFrontend::new(config));
        self.biu_returns = Some(returns);
        self.biu_subcore = subcore;
        self.dma_connected = true;
        Ok(())
    }

    pub fn biu_read(&self) -> Option<&C220BiuReadFrontend> {
        self.biu_read.as_ref()
    }

    pub fn take_biu_read_request(&mut self) -> Option<C220BiuReadRequest> {
        self.biu_output.take()
    }

    pub fn biu_read_returns(&self) -> Option<&C220BiuReadReturns> {
        self.biu_returns.as_ref()
    }

    pub fn receive_biu_read(
        &mut self,
        heads: [Option<C220BiuReadBeat>; 2],
    ) -> Result<[bool; 2], C220MtePipelineError> {
        if heads.iter().flatten().any(|beat| {
            self.biu_output
                .is_some_and(|request| request.tag == beat.tag)
        }) {
            return Err(C220MtePipelineError::BiuRequestUndelivered);
        }
        Ok(self
            .biu_returns
            .as_mut()
            .ok_or(C220MtePipelineError::BiuDisconnected)?
            .receive(self.events.tick(), heads)?)
    }

    pub fn ub_write_interface(
        &self,
        core: C220BiuSubcore,
    ) -> Result<&C220UbWriteInterface, C220MtePipelineError> {
        Ok(&self.ub_write[self.ub_write_index(core)?])
    }

    fn ub_write_index(&self, core: C220BiuSubcore) -> Result<usize, C220MtePipelineError> {
        if self.biu_returns.is_none() {
            return Err(C220MtePipelineError::BiuDisconnected);
        }
        match core {
            C220BiuSubcore::Vector0 => Ok(0),
            C220BiuSubcore::Vector1 => Ok(1),
            C220BiuSubcore::Cube => Err(C220MtePipelineError::BiuUbSubcoreRequired),
        }
    }

    pub fn check_external_dma_completion(&self) -> Result<(), C220MtePipelineError> {
        if self.biu_read.is_some() {
            return Err(C220MtePipelineError::DestinationOwnedCompletion);
        }
        Ok(())
    }

    pub fn take_dma_completions(&mut self) -> std::vec::Drain<'_, u64> {
        self.dma_completions.drain(..)
    }

    pub fn mte2_dma_connected(&self) -> bool {
        self.dma_connected
    }
    pub fn dma_generator(&self) -> &C220DmaFrontend {
        &self.dma
    }
    pub fn dma_output(&self) -> Option<C220DmaGenerated> {
        self.dma_output
    }
    pub fn take_dma_output(&mut self) -> Option<C220DmaGenerated> {
        self.dma_output.take()
    }
    pub fn set_dma_hardware_sync_blocked(&mut self, blocked: bool) {
        self.dma_hardware_sync_blocked = blocked;
    }

    fn mte2_generator_idle(&self) -> bool {
        self.selected_mte2_generator
            .is_none_or(|generator| match generator {
                Mte2Generator::Dma => self.dma.is_idle(),
                Mte2Generator::L1Fill => self.set2d_l1.is_idle(),
            })
    }

    pub fn can_issue_mte2_dma(&self) -> bool {
        self.dma_connected
            && self.dma.can_issue()
            && (self.selected_mte2_generator == Some(Mte2Generator::Dma)
                || self.mte2_generator_idle())
    }

    pub fn issue_mte2_dma(
        &mut self,
        instruction_id: u64,
        transfer: C220Mte2TransferPlan,
    ) -> Result<C220DmaIssue, C220MtePipelineError> {
        if !self.dma_connected {
            return Err(C220MtePipelineError::DmaDisconnected);
        }
        let requests = mte2_uops(transfer)?;
        if transfer.descriptor.is_disabled() {
            return Ok(C220DmaIssue {
                tick: self.events.tick(),
                instruction_id,
                completion_ready: true,
            });
        }
        if !self.can_issue_mte2_dma() {
            return Err(C220MtePipelineError::CommandBusy);
        }
        let issue =
            self.dma_events
                .issue(&mut self.events, &mut self.dma, instruction_id, requests)?;
        self.selected_mte2_generator = Some(Mte2Generator::Dma);
        Ok(issue)
    }

    /// Resolves generator admission and switching. The command owner still
    /// owns ordered retirement and lane-level dependencies.
    pub fn can_issue_l1_fill(&self, fill: C220Set2dFill) -> bool {
        fill.instruction.destination == C220Set2dDestination::L1
            && (fill.descriptor.is_disabled()
                || (self.set2d_l1.can_issue()
                    && (self.selected_mte2_generator == Some(Mte2Generator::L1Fill)
                        || self.mte2_generator_idle())))
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
        if !self.can_issue_l1_fill(fill) {
            return Err(C220MtePipelineError::CommandBusy);
        }
        let issue = self.set2d_l1_events.issue(
            &mut self.events,
            &mut self.set2d_l1,
            instruction_id,
            fill,
        )?;
        self.selected_mte2_generator = Some(Mte2Generator::L1Fill);
        Ok(issue)
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
        self.dma_completions.clear();
        self.trace.clear();
        self.events.notify_at(self.clock, tick);
        while let Some(invocation) = self.events.next_callback() {
            match invocation.callback {
                Callback::Dma(phase) => {
                    let output_ready = self
                        .biu_read
                        .as_ref()
                        .map_or(self.dma_output.is_none(), |frontend| {
                            frontend.can_push(self.biu_subcore)
                        });
                    let outcome = self.dma_events.handle(
                        phase,
                        &mut self.events,
                        &mut self.dma,
                        self.dma_hardware_sync_blocked,
                        output_ready,
                    )?;
                    if let C220DmaEventOutcome::Sent(send) = outcome
                        && let Some(sent) = send.sent
                    {
                        if let Some(frontend) = &mut self.biu_read {
                            assert!(self.biu_events.push(
                                &mut self.events,
                                frontend,
                                C220BiuReadInput {
                                    subcore: self.biu_subcore,
                                    destination: match self.biu_subcore {
                                        C220BiuSubcore::Vector0 => C220BiuWriteDestination::Ub0,
                                        C220BiuSubcore::Vector1 => C220BiuWriteDestination::Ub1,
                                        C220BiuSubcore::Cube =>
                                            unreachable!("UB connection requires vector"),
                                    },
                                    prefetch: false,
                                    generated: sent,
                                }
                            )?);
                        } else {
                            self.dma_output = Some(sent);
                        }
                    }
                    if outcome != C220DmaEventOutcome::Readiness {
                        self.trace.push(C220MtePipelineEvent::Dma(outcome));
                    }
                }
                Callback::BiuRead(phase) => {
                    if let Some(frontend) = &mut self.biu_read {
                        let outcome = self.biu_events.handle(
                            phase,
                            &mut self.events,
                            frontend,
                            self.biu_output.is_none(),
                        )?;
                        if let C220BiuReadEvent::Send(send) = outcome
                            && let Some(request) = send.sent()
                        {
                            self.biu_returns
                                .as_mut()
                                .expect("connected return path")
                                .track(tick, request)?;
                            self.biu_output = Some(request);
                        }
                        if outcome != C220BiuReadEvent::Readiness {
                            self.trace.push(C220MtePipelineEvent::BiuRead(outcome));
                        }
                    }
                }
                Callback::BiuReturn(phase) => {
                    if let Some(returns) = &mut self.biu_returns {
                        let outcome = self.biu_return_events.handle(
                            phase,
                            &mut self.events,
                            returns,
                            [
                                false,
                                self.ub_write[0].can_push(),
                                self.ub_write[1].can_push(),
                            ],
                        )?;
                        if let C220BiuReturnEvent::Send(send) = &outcome
                            && let Some(fragment) = send.sent()
                        {
                            let index = match send.core {
                                C220BiuSubcore::Vector0 => 0,
                                C220BiuSubcore::Vector1 => 1,
                                C220BiuSubcore::Cube => {
                                    unreachable!("UB connection requires vector")
                                }
                            };
                            assert!(self.ub_write[index].push(tick, fragment)?);
                        }
                        if let C220BiuReturnEvent::Egress(Some(output)) = &outcome {
                            let released = self
                                .biu_read
                                .as_mut()
                                .expect("connected request path")
                                .release_tag(tick, output.request.tag)?;
                            assert_eq!(released, output.request);
                        }
                        if outcome != C220BiuReturnEvent::Readiness {
                            self.trace.push(C220MtePipelineEvent::BiuReturn(outcome));
                        }
                    }
                }
                Callback::UbWrite(index, phase) => {
                    let outcome = self.ub_write_events[index].handle(
                        phase,
                        &mut self.events,
                        &mut self.ub_write[index],
                    )?;
                    if let C220UbWriteEvent::Acknowledged(Some(ack)) = outcome
                        && let Some(id) = ack.retired_instruction()
                    {
                        self.dma_completions.push(id);
                    }
                    if outcome != C220UbWriteEvent::Readiness {
                        self.trace.push(C220MtePipelineEvent::UbWrite(
                            [C220BiuSubcore::Vector0, C220BiuSubcore::Vector1][index],
                            outcome,
                        ));
                    }
                }
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
