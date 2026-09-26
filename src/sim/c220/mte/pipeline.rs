use super::dma::{
    C220DmaEventOutcome, C220DmaEvents, C220DmaFrontend, C220DmaFrontendError, C220DmaGenerated,
    C220DmaIssue,
};
use super::fixp::C220FixpStoreBuffer;
use super::fixp::{
    C220FixpCallback, C220FixpEngine, C220FixpEvent, C220FixpMemory, C220FixpResources,
    C220FixpSliceResult, C220FixpStage, C220FixpStageEvents, C220FixpSync,
};
use super::fixp::{
    C220FixpL1WriteCallback, C220FixpL1WriteError, C220FixpL1WriteEvent, C220FixpL1WriteEvents,
    C220FixpL1WriteInterface,
};
use super::fixp::{
    C220FixpRuntime, C220FixpRuntimeEvent, C220FixpRuntimeMemory, C220FixpRuntimeStage,
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
use super::interface::biu_write::command::{
    C220BiuWriteCommandCycle, C220BiuWriteCommandError, C220BiuWriteCommands,
};
use super::interface::biu_write::cube::{C220BiuCubeWriteError, C220BiuCubeWriteSource};
use super::interface::biu_write::data::{
    C220BiuWriteDataError, C220BiuWriteDataPort, C220BiuWriteDataSend, C220BiuWriteResponse,
};
use super::interface::biu_write::{
    C220BiuWriteSource, C220BiuWriteSourceError, C220BiuWriteSourceEvent,
};
use super::interface::ub_read::{C220UbReadError, C220UbReadInterface, C220UbReadRequest};
use super::interface::ub_write::{
    C220UbWriteCallback, C220UbWriteError, C220UbWriteEvent, C220UbWriteEvents,
    C220UbWriteInterface, C220UbWriteRequest,
};
use super::mte1::{C220Mte1Command, C220Mte1Generator, C220Mte1Issue};
use super::mte2::C220Mte2TransferPlan;
use super::mte3::C220Mte3TransferPlan;
use super::mte3::frontend::{
    C220Mte3Callback, C220Mte3Events, C220Mte3Frontend, C220Mte3FrontendError,
    C220Mte3FrontendEvent, C220Mte3Record,
};
use super::nd2nz::{
    C220Nd2NzCallback, C220Nd2NzEngine, C220Nd2NzEngineError, C220Nd2NzEvent, C220Nd2NzEvents,
};
use super::uop::{C220DmaUopError, C220DmaUops, mte2_l1_uops, mte2_uops};
use crate::isa::c220::mte::out_to_l1::C220L1DmaDescriptor;
use crate::isa::c220::mte::set2d::{C220Set2dDestination, C220Set2dFill};
use crate::sim::c220::memory::biu_read::{C220BiuBusReadError, C220BiuBusReads};
use crate::sim::c220::memory::biu_write::{C220BiuBusWriteError, C220BiuBusWrites};
use crate::sim::c220::memory::timed_memory::{C220TimedMemory, C220TimedMemoryError};
use crate::sim::c220::memory::ub_service::{C220UbService, C220UbServiceCycle, C220UbServiceError};
use crate::sim::c220::mte::set2d::{
    C220Set2dBandwidths, C220Set2dEventOutcome, C220Set2dEvents, C220Set2dFrontend,
    C220Set2dFrontendError, C220Set2dGates, C220Set2dIssue, C220Set2dOutputs,
};
use std::num::NonZeroU32;

use super::read::{
    C220MteReadBandwidths, C220MteReadEventOutcome, C220MteReadEvents, C220MteReadFrontend,
    C220MteReadFrontendError, C220MteReadKind, C220MteReadUop,
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
    C220MteL1WriteInterface, C220MteL1WritePort,
};
use crate::sim::common::event::{EventDispatcher, EventError, EventId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MtePipelineConfig {
    pub core_kind: crate::sim::c220::device::C220CoreKind,
    pub l1: C220L1Geometry,
    pub read_width: NonZeroU32,
    pub output_bandwidths: C220MteReadBandwidths,
    pub set2d_bandwidths: C220Set2dBandwidths,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Callback {
    Nd2Nz(C220Nd2NzCallback),
    FixpIssueProbe,
    FixpIssueTransfer,
    Mte1IssueProbe,
    Mte1IssueTransfer,
    Mte2IssueProbe,
    Mte2IssueTransfer,
    Mte3IssueProbe,
    Mte3IssueTransfer,
    Mte2CommandProbe,
    Mte2CommandDispatch,
    FixpCommandProbe,
    FixpCommandDispatch,
    Mte1CommandProbe,
    Mte1CommandDispatch,
    FixpExternal(usize, C220FixpCallback),
    Fixp(usize, C220FixpCallback),
    FixpWrite(C220FixpL1WriteCallback),
    Memory(C220L1Callback),
    Interface(C220MteL1Callback),
    L1Write(C220MteL1WriteCallback),
    L0(bool, C220L0WriteCallback),
    Generator(C220MteReadKind, C220MteGeneratorCallback),
    Set2d(C220MteGeneratorCallback),
    Set2dL1(C220MteGeneratorCallback),
    Dma(C220MteGeneratorCallback),
    ExternalLoad2d(C220MteGeneratorCallback),
    Mte3(C220Mte3Callback),
    BiuRead(C220BiuReadCallback),
    BiuReturn(C220BiuReturnCallback),
    UbWrite(usize, C220UbWriteCallback),
    UbReadProbe(usize),
    UbReadSend(usize),
    BiuWriteSource(usize),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum C220MtePipelineEvent {
    Nd2Nz(C220Nd2NzEvent),
    FixpExternal(C220FixpRuntimeEvent),
    FixpExternalStored {
        slice: C220FixpSliceResult,
        bytes: Vec<u8>,
    },
    Fixp(C220FixpEvent),
    FixpSlice(C220FixpSliceResult),
    FixpWrite(C220FixpL1WriteEvent),
    Memory(C220L1EventOutcome),
    Interface(C220MteL1EventOutcome<C220MteReadPayload>),
    L1Write(C220MteL1WriteEventOutcome),
    L0a(C220L0WriteEventOutcome),
    L0b(C220L0WriteEventOutcome),
    Generator(C220MteReadKind, C220MteReadEventOutcome),
    Set2d(C220Set2dEventOutcome),
    Set2dL1(C220Set2dEventOutcome),
    Dma(C220DmaEventOutcome),
    ExternalLoad2d(C220DmaEventOutcome),
    Mte3(C220Mte3FrontendEvent),
    BiuRead(C220BiuReadEvent),
    BiuReturn(C220BiuReturnEvent),
    UbWrite(C220BiuSubcore, C220UbWriteEvent),
    UbRequest(C220BiuSubcore, C220UbWriteRequest),
    UbResponse(C220BiuSubcore, C220UbWriteRequest),
    UbService(C220BiuSubcore, C220UbServiceCycle),
    UbReadSent(C220BiuSubcore, C220UbReadRequest),
    UbReadRequest(C220BiuSubcore, C220UbReadRequest),
    UbReadResponse(C220BiuSubcore, C220UbReadRequest),
    BiuWriteSource(C220BiuSubcore, C220BiuWriteSourceEvent),
    BiuWriteData(C220BiuWriteDataSend),
    BiuWriteResponse(C220BiuWriteResponse),
    BiuWriteCommand(C220BiuWriteCommandCycle),
    BiuCubeSourceStarted {
        tick: u64,
        tag: NonZeroU32,
    },
    BiuCubeSourceReady(super::interface::biu_write::C220BiuWriteDataReady),
    BiuCubeSourceProbe(super::interface::biu_write::cube::C220BiuCubeWriteProbe),
}

#[derive(Debug, thiserror::Error)]
pub enum C220MtePipelineError {
    #[error(transparent)]
    Nd2Nz(#[from] C220Nd2NzEngineError),
    #[error("ND2NZ requires staging configuration")]
    Nd2NzUnconfigured,
    #[error("MTE2 MOV_PAD requires an external-to-UB transfer")]
    WrongMovPadDirection,
    #[error(transparent)]
    Load2d(#[from] crate::isa::c220::mte::load2d::C220Load2dError),
    #[error("MTE cycle {active} must finish before advancing to {requested}")]
    UnfinishedCycle { active: u64, requested: u64 },
    #[error("FIX frontend must resolve its pending dispatch before resuming the MTE cycle")]
    FixpDispatchPending,
    #[error("MTE1 frontend must resolve its pending dispatch before resuming the MTE cycle")]
    Mte1DispatchPending,
    #[error("MTE2 frontend must resolve its pending dispatch before resuming the MTE cycle")]
    Mte2DispatchPending,
    #[error("MTE3 frontend must resolve its pending reception before resuming the MTE cycle")]
    Mte3IssuePending,
    #[error("MTE1 synchronization callback must be resolved before resuming the cycle")]
    Mte1SyncPending,
    #[error(transparent)]
    HardwareFlag(#[from] crate::sim::c220::sync::C220HardwareFlagTimingError),
    #[error(transparent)]
    FixpExternal(Box<super::fixp::C220FixpRuntimeError>),
    #[error("external FIX event binding requires each of the eleven stages exactly once")]
    InvalidExternalFixpStages,
    #[error(transparent)]
    BiuCubeWrite(#[from] C220BiuCubeWriteError),
    #[error(transparent)]
    FixpOutput(#[from] super::fixp::C220FixpExternalOutputError),
    #[error(transparent)]
    FixpWritePipeline(#[from] super::fixp::C220FixpWritePipelineError),
    #[error("FIX event binding requires each of the nine stages exactly once")]
    InvalidFixpStages,
    #[error("FIX events are already bound or the pipeline is active")]
    FixpBindingBusy,
    #[error("FIX event binding and execution context must both be present")]
    FixpContextMismatch,
    #[error(transparent)]
    FixpEngine(#[from] super::fixp::C220FixpEngineError),
    #[error(transparent)]
    FixpWrite(#[from] C220FixpL1WriteError),
    #[error("the connected memory service owns BIU read traffic")]
    MemoryOwnedRead,
    #[error(transparent)]
    BiuBusRead(#[from] C220BiuBusReadError),
    #[error(transparent)]
    TimedMemory(#[from] C220TimedMemoryError),
    #[error("the connected memory service owns BIU write traffic")]
    MemoryOwnedWrite,
    #[error(transparent)]
    BiuBusWrite(#[from] C220BiuBusWriteError),
    #[error("BIU bus owns write responses; use its bounded return channels")]
    BiuBusOwnedResponse,
    #[error("BIU bus write route is not connected")]
    BiuBusDisconnected,
    #[error(transparent)]
    BiuWriteCommand(#[from] C220BiuWriteCommandError),
    #[error("BIU write command transport owns source registration")]
    BiuOwnedSource,
    #[error("BIU write command transport is not configured")]
    BiuWriteCommandDisconnected,
    #[error("BIU write DBID does not match the command subcore")]
    BiuWriteWrongSubcore,
    #[error(transparent)]
    BiuWriteData(#[from] C220BiuWriteDataError),
    #[error(transparent)]
    BiuWriteSource(#[from] C220BiuWriteSourceError),
    #[error("BIU write source transport is not configured")]
    BiuWriteSourceDisconnected,
    #[error("UB source packets and completions are owned by the connected BIU write source")]
    BiuOwnedUbRead,
    #[error(transparent)]
    UbRead(#[from] C220UbReadError),
    #[error("UB read service requires a vector subcore")]
    UbReadSubcoreRequired,
    #[error(transparent)]
    Mte3(#[from] C220Mte3FrontendError),
    #[error(transparent)]
    UbService(#[from] C220UbServiceError),
    #[error(transparent)]
    UbWrite(#[from] C220UbWriteError),
    #[error("BIU-connected DMA completion is owned by the destination write interface")]
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
    #[error("L1 DMA requires a connected Cube BIU subcore")]
    BiuL1SubcoreRequired,
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
    Generator(#[from] C220MteReadFrontendError),
    #[error(transparent)]
    Interface(#[from] C220MteL1Error),
    #[error(transparent)]
    Memory(#[from] C220L1TransportError),
    #[error(transparent)]
    L0(#[from] C220L0WriteError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mte2Generator {
    Nd2Nz,
    ExternalLoad2d,
    Default,
    Load3d,
    Dma,
    L1Fill,
}

mod cache;
mod external_fixp;
mod load2d;
mod memory;
mod nd2nz;
mod read;
mod smask;
mod sync;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220MteReadPayload {
    Read(C220MteReadUop),
    Factor(super::factor::C220FactorReadPacket),
}
#[cfg(test)]
mod tests;
mod ub;
mod write;

/// Physical MTE paths. Generators, shared L1 interfaces, L1 service and
/// destinations execute on one event dispatcher. Retirement is an observed
/// destination acknowledgment, not a prediction made at command admission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220MtePipeline {
    nd2nz: Option<C220Nd2NzEngine>,
    nd2nz_events: C220Nd2NzEvents,
    fixp_events: Vec<C220FixpStageEvents>,
    external_fixp_events: Vec<C220FixpStageEvents<C220FixpRuntimeStage>>,
    fixp_write: C220FixpL1WriteInterface,
    fixp_write_events: C220FixpL1WriteEvents,
    fixp_completions: Vec<u64>,
    events: EventDispatcher<Callback>,
    clock: EventId,
    memory_events: C220L1Events,
    interface_events: C220MteL1Events,
    write_events: C220MteL1WriteEvents,
    l0_events: [C220L0WriteEvents; 2],
    generator_events: [C220MteReadEvents; 5],
    set2d_events: C220Set2dEvents,
    set2d_l1_events: C220Set2dEvents,
    dma_events: C220DmaEvents,
    load2d_events: C220DmaEvents,
    mte3_events: C220Mte3Events,
    mte3: C220Mte3Frontend,
    core_kind: crate::sim::c220::device::C220CoreKind,
    biu_events: C220BiuReadEvents,
    biu_return_events: C220BiuReturnEvents,
    ub_write_events: [C220UbWriteEvents; 2],
    memory: C220L1Transport,
    interface: C220MteL1Interface<C220MteReadPayload>,
    write_interface: C220MteL1WriteInterface,
    l0: [C220L0WritePipeline; 2],
    generators: [C220MteReadFrontend; 5],
    set2d: C220Set2dFrontend,
    set2d_l1: C220Set2dFrontend,
    dma: C220DmaFrontend,
    external_load2d: C220DmaFrontend,
    load2d_destinations: std::collections::BTreeMap<u64, C220BiuWriteDestination>,
    dma_connected: bool,
    dma_output: Option<C220DmaGenerated>,
    biu_read: Option<C220BiuReadFrontend>,
    biu_returns: Option<C220BiuReadReturns>,
    biu_subcore: C220BiuSubcore,
    biu_output: Option<C220BiuReadRequest>,
    dma_tails: Vec<u64>,
    biu_bus_reads: Option<C220BiuBusReads>,
    ub_write: [C220UbWriteInterface; 2],
    ub_read: [C220UbReadInterface; 2],
    biu_write_source: [Option<C220BiuWriteSource>; 2],
    biu_cube_source: C220BiuCubeWriteSource,
    fixp_stores: C220FixpStoreBuffer,
    biu_write_data: C220BiuWriteDataPort,
    biu_write_commands: Option<C220BiuWriteCommands>,
    biu_bus_writes: Option<C220BiuBusWrites>,
    timed_memory: Option<C220TimedMemory>,
    ub_read_valid: [EventId; 2],
    ub_memory: [C220UbService; 2],
    last_ub_service: Option<u64>,
    dma_hardware_sync_blocked: bool,
    selected_mte2_generator: Option<Mte2Generator>,
    l1_prefetch_blocked: bool,
    selected_generator: Option<C220Mte1Generator>,
    completions: Vec<u64>,
    l1_fill_completions: Vec<u64>,
    mte2_read_completions: Vec<u64>,
    dma_completions: Vec<u64>,
    trace: Vec<C220MtePipelineEvent>,
    last_advance: Option<u64>,
    active_cycle: Option<u64>,
    fixp_command_valid: EventId,
    mte1_command_valid: EventId,
    mte1_command_ready: Option<u64>,
    mte1_dispatch_pending: bool,
    mte1_sync: sync::Mte1Sync,
    fixp_command_ready: Option<u64>,
    fixp_head_is_convert: bool,
    fixp_dispatch_pending: bool,
    fixp_issue_valid: EventId,
    fixp_issue_ready: Option<u64>,
    fixp_issue_pending: bool,
    mte1_issue_valid: EventId,
    mte1_issue_ready: Option<u64>,
    mte1_issue_pending: bool,
    mte2_issue_valid: EventId,
    mte2_issue_ready: Option<u64>,
    mte2_issue_pending: bool,
    mte3_issue_valid: EventId,
    mte3_issue_ready: Option<u64>,
    mte3_issue_pending: bool,
    mte2_command_valid: EventId,
    mte2_command_ready: Option<u64>,
    mte2_dispatch_pending: bool,
}

impl C220MtePipeline {
    pub(crate) fn set_mte3_issue_head(&mut self, ready: Option<u64>) {
        self.mte3_issue_ready = ready;
    }

    pub(crate) fn mte3_issue_pending(&self) -> bool {
        self.mte3_issue_pending
    }

    pub(crate) fn finish_mte3_issue(&mut self) {
        self.mte3_issue_pending = false;
    }

    pub const fn core_kind(&self) -> crate::sim::c220::device::C220CoreKind {
        self.core_kind
    }

    pub(crate) fn set_mte2_issue_head(&mut self, ready: Option<u64>) {
        self.mte2_issue_ready = ready;
    }

    pub(crate) fn mte2_issue_pending(&self) -> bool {
        self.mte2_issue_pending
    }

    pub(crate) fn finish_mte2_issue(&mut self) {
        self.mte2_issue_pending = false;
    }

    pub(crate) fn set_mte2_command_head(&mut self, ready: Option<u64>) {
        self.mte2_command_ready = ready;
    }

    pub(crate) fn mte2_dispatch_pending(&self) -> bool {
        self.mte2_dispatch_pending
    }

    pub(crate) fn finish_mte2_dispatch(&mut self) {
        self.mte2_dispatch_pending = false;
    }

    pub(crate) fn set_mte1_issue_head(&mut self, ready: Option<u64>) {
        self.mte1_issue_ready = ready;
    }

    pub(crate) fn mte1_issue_pending(&self) -> bool {
        self.mte1_issue_pending
    }

    pub(crate) fn finish_mte1_issue(&mut self) {
        self.mte1_issue_pending = false;
    }

    pub(crate) fn set_mte1_command_head(&mut self, ready: Option<u64>) {
        self.mte1_command_ready = ready;
    }

    pub(crate) fn mte1_dispatch_pending(&self) -> bool {
        self.mte1_dispatch_pending
    }

    pub(crate) fn finish_mte1_dispatch(&mut self) {
        self.mte1_dispatch_pending = false;
    }

    pub(crate) fn set_fixp_issue_head(&mut self, ready: Option<u64>) {
        self.fixp_issue_ready = ready;
    }

    pub(crate) fn fixp_issue_pending(&self) -> bool {
        self.fixp_issue_pending
    }

    pub(crate) fn finish_fixp_issue(&mut self) {
        self.fixp_issue_pending = false;
    }

    pub(crate) fn set_fixp_command_head(&mut self, ready: Option<u64>, is_convert: bool) {
        self.fixp_command_ready = ready;
        self.fixp_head_is_convert = is_convert;
    }

    pub(crate) fn fixp_dispatch_pending(&self) -> bool {
        self.fixp_dispatch_pending
    }

    pub(crate) fn finish_fixp_dispatch(&mut self) {
        self.fixp_dispatch_pending = false;
    }

    pub fn new(tick: u64, config: C220MtePipelineConfig) -> Self {
        let mut events = EventDispatcher::new(tick);
        let clock = events.add_event();
        let fixp_issue_valid = events.add_event();
        let issue_probe = events.add_process(Callback::FixpIssueProbe, false);
        let issue_transfer = events.add_process(Callback::FixpIssueTransfer, false);
        events.subscribe(clock, issue_probe);
        events.subscribe(fixp_issue_valid, issue_transfer);
        let mte1_issue_valid = events.add_event();
        let issue_probe = events.add_process(Callback::Mte1IssueProbe, false);
        let issue_transfer = events.add_process(Callback::Mte1IssueTransfer, false);
        events.subscribe(clock, issue_probe);
        events.subscribe(mte1_issue_valid, issue_transfer);
        let mte2_issue_valid = events.add_event();
        let issue_probe = events.add_process(Callback::Mte2IssueProbe, false);
        let issue_transfer = events.add_process(Callback::Mte2IssueTransfer, false);
        events.subscribe(clock, issue_probe);
        events.subscribe(mte2_issue_valid, issue_transfer);
        let mte3_issue_valid = events.add_event();
        let issue_probe = events.add_process(Callback::Mte3IssueProbe, false);
        let issue_transfer = events.add_process(Callback::Mte3IssueTransfer, false);
        events.subscribe(clock, issue_probe);
        events.subscribe(mte3_issue_valid, issue_transfer);
        let mte3_events = C220Mte3Events::register(&mut events, clock, Callback::Mte3);
        let fixp_command_valid = events.add_event();
        let probe = events.add_process(Callback::FixpCommandProbe, false);
        let dispatch = events.add_process(Callback::FixpCommandDispatch, false);
        events.subscribe(clock, probe);
        events.subscribe(fixp_command_valid, dispatch);
        let mte1_command_valid = events.add_event();
        let probe = events.add_process(Callback::Mte1CommandProbe, false);
        let dispatch = events.add_process(Callback::Mte1CommandDispatch, false);
        events.subscribe(clock, probe);
        events.subscribe(mte1_command_valid, dispatch);
        let mte2_command_valid = events.add_event();
        let probe = events.add_process(Callback::Mte2CommandProbe, false);
        let dispatch = events.add_process(Callback::Mte2CommandDispatch, false);
        events.subscribe(clock, probe);
        events.subscribe(mte2_command_valid, dispatch);
        let memory_events = C220L1Events::register(&mut events, clock, Callback::Memory);
        let fixp_write_events =
            C220FixpL1WriteEvents::register(&mut events, clock, Callback::FixpWrite);
        let interface_events = C220MteL1Events::register(&mut events, clock, Callback::Interface);
        let write_events = C220MteL1WriteEvents::register(&mut events, clock, Callback::L1Write);
        let l0_events = [false, true].map(|b| {
            C220L0WriteEvents::register(&mut events, clock, |phase| Callback::L0(b, phase))
        });
        let generator_events = C220MteReadKind::ALL.map(|kind| {
            C220MteReadEvents::register(&mut events, clock, |phase| {
                Callback::Generator(kind, phase)
            })
        });
        let set2d_events = C220Set2dEvents::register(&mut events, clock, Callback::Set2d);
        let dma_events = C220DmaEvents::register(&mut events, clock, Callback::Dma);
        let load2d_events = C220DmaEvents::register(&mut events, clock, Callback::ExternalLoad2d);
        let nd2nz_events = C220Nd2NzEvents::register(&mut events, clock, Callback::Nd2Nz);
        let set2d_l1_events = C220Set2dEvents::register(&mut events, clock, Callback::Set2dL1);
        let biu_events = C220BiuReadEvents::register(&mut events, clock, Callback::BiuRead);
        let biu_return_events =
            C220BiuReturnEvents::register(&mut events, clock, Callback::BiuReturn);
        let ub_write_events = [0, 1].map(|index| {
            C220UbWriteEvents::register(&mut events, clock, |phase| Callback::UbWrite(index, phase))
        });
        let ub_read_valid = [0, 1].map(|index| {
            let valid = events.add_event();
            let probe = events.add_process(Callback::UbReadProbe(index), false);
            let send = events.add_process(Callback::UbReadSend(index), false);
            events.subscribe(clock, probe);
            events.subscribe(valid, send);
            let source = events.add_process(Callback::BiuWriteSource(index), false);
            events.subscribe(clock, source);
            valid
        });
        Self {
            fixp_events: Vec::new(),
            nd2nz: None,
            nd2nz_events,
            external_fixp_events: Vec::new(),
            fixp_write: C220FixpL1WriteInterface::default(),
            fixp_write_events,
            fixp_completions: Vec::new(),
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
            load2d_events,
            mte3_events,
            mte3: C220Mte3Frontend::default(),
            biu_events,
            biu_return_events,
            ub_write_events,
            ub_read_valid,
            ub_read: std::array::from_fn(|_| C220UbReadInterface::default()),
            biu_write_source: [None, None],
            biu_cube_source: C220BiuCubeWriteSource::default(),
            fixp_stores: C220FixpStoreBuffer::default(),
            biu_write_data: C220BiuWriteDataPort::default(),
            biu_write_commands: None,
            biu_bus_writes: None,
            timed_memory: None,
            memory: C220L1Transport::new(config.l1),
            interface: C220MteL1Interface::default(),
            write_interface: C220MteL1WriteInterface::default(),
            l0: std::array::from_fn(|_| C220L0WritePipeline::default()),
            generators: C220MteReadKind::ALL.map(|kind| {
                C220MteReadFrontend::new(kind, config.read_width, config.output_bandwidths)
            }),
            selected_generator: None,
            set2d: C220Set2dFrontend::new(config.set2d_bandwidths),
            set2d_l1: C220Set2dFrontend::new(config.set2d_bandwidths),
            dma: C220DmaFrontend::default(),
            external_load2d: C220DmaFrontend::external_load2d(),
            load2d_destinations: Default::default(),
            dma_connected: false,
            dma_output: None,
            biu_read: None,
            biu_returns: None,
            biu_subcore: C220BiuSubcore::Vector0,
            core_kind: config.core_kind,
            biu_output: None,
            dma_tails: Vec::new(),
            biu_bus_reads: None,
            ub_write: std::array::from_fn(|_| C220UbWriteInterface::default()),
            ub_memory: std::array::from_fn(|_| C220UbService::default()),
            last_ub_service: None,
            dma_hardware_sync_blocked: false,
            selected_mte2_generator: None,
            l1_prefetch_blocked: false,
            completions: Vec::new(),
            l1_fill_completions: Vec::new(),
            mte2_read_completions: Vec::new(),
            dma_completions: Vec::new(),
            trace: Vec::new(),
            last_advance: None,
            active_cycle: None,
            fixp_command_valid,
            mte1_command_valid,
            mte1_command_ready: None,
            mte1_dispatch_pending: false,
            mte1_sync: sync::Mte1Sync::default(),
            fixp_command_ready: None,
            fixp_head_is_convert: false,
            fixp_dispatch_pending: false,
            fixp_issue_valid,
            fixp_issue_ready: None,
            fixp_issue_pending: false,
            mte1_issue_valid,
            mte1_issue_ready: None,
            mte1_issue_pending: false,
            mte2_issue_valid,
            mte2_issue_ready: None,
            mte2_issue_pending: false,
            mte3_issue_valid,
            mte3_issue_ready: None,
            mte3_issue_pending: false,
            mte2_command_valid,
            mte2_command_ready: None,
            mte2_dispatch_pending: false,
        }
    }

    pub fn can_issue_mte1(&self, command: C220Mte1Command) -> bool {
        if matches!(command, C220Mte1Command::HardwareFlag(_)) {
            return true;
        }
        if matches!(command, C220Mte1Command::WriteSpr(_)) {
            return self.selected_generator_idle();
        }
        if matches!(command, C220Mte1Command::Set2d(fill) if fill.instruction.destination == C220Set2dDestination::L1)
            || matches!(command, C220Mte1Command::Read(super::read::C220MteReadTransfer::Smask(transfer)) if transfer.instruction.source_mode != 2)
        {
            return false;
        }
        command.is_disabled()
            || (match command {
                C220Mte1Command::CrossCore { .. } => true,
                C220Mte1Command::WriteSpr(_) | C220Mte1Command::HardwareFlag(_) => {
                    unreachable!("handled above")
                }
                C220Mte1Command::Read(transfer) => self.generator(transfer.kind()).can_issue(),
                C220Mte1Command::Set2d(_) => self.set2d.can_issue(),
            } && (self.selected_generator == command.generator()
                || self.selected_generator_idle()))
    }
    pub fn selected_generator_idle(&self) -> bool {
        self.selected_generator.is_none_or(|kind| match kind {
            C220Mte1Generator::Load3d => true,
            C220Mte1Generator::Read(kind) => self.generator(kind).is_idle(),
            C220Mte1Generator::Set2d => self.set2d.is_idle(),
        })
    }
    pub fn selected_generator(&self) -> Option<C220Mte1Generator> {
        self.selected_generator
    }
    pub fn is_idle(&self) -> bool {
        self.fixp_command_ready.is_none()
            && self.mte1_command_ready.is_none()
            && self.fixp_issue_ready.is_none()
            && self.mte1_issue_ready.is_none()
            && self.mte2_issue_ready.is_none()
            && self.mte3_issue_ready.is_none()
            && self.mte2_command_ready.is_none()
            && self.active_cycle.is_none()
            && self.fixp_write.is_idle()
            && self.biu_cube_source.is_idle()
            && self.fixp_stores.is_empty()
            && self.generators.iter().all(C220MteReadFrontend::is_idle)
            && self.set2d.is_idle()
            && self.set2d_l1.is_idle()
            && self.external_load2d.is_idle()
            && self.nd2nz.as_ref().is_none_or(C220Nd2NzEngine::is_idle)
            && self.load2d_destinations.is_empty()
            && self.dma.is_idle()
            && self.mte3.is_idle()
            && self.dma_output.is_none()
            && self.biu_output.is_none()
            && self
                .biu_bus_reads
                .as_ref()
                .is_none_or(C220BiuBusReads::is_idle)
            && self.ub_write.iter().all(C220UbWriteInterface::is_idle)
            && self.ub_read.iter().all(C220UbReadInterface::is_idle)
            && self.biu_write_data.is_idle()
            && self
                .timed_memory
                .as_ref()
                .is_none_or(C220TimedMemory::is_idle)
            && self
                .biu_bus_writes
                .as_ref()
                .is_none_or(C220BiuBusWrites::is_idle)
            && self
                .biu_write_commands
                .as_ref()
                .is_none_or(C220BiuWriteCommands::is_idle)
            && self
                .biu_write_source
                .iter()
                .flatten()
                .all(C220BiuWriteSource::is_idle)
            && self.ub_memory.iter().all(C220UbService::is_idle)
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
    pub fn tick(&self) -> u64 {
        self.events.tick()
    }

    /// Include the borrowed engine even before it emits its first L1 request.
    pub fn next_fixp_event_tick(&self, engine: &C220FixpEngine) -> Option<u64> {
        (!self.is_idle() || !engine.is_idle()).then(|| self.events.tick().saturating_add(1))
    }
    pub fn generator(&self, kind: C220MteReadKind) -> &C220MteReadFrontend {
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

    pub fn fixp_write_interface(&self) -> &C220FixpL1WriteInterface {
        &self.fixp_write
    }

    /// Physical write-interface boundary. The producer must already have
    /// completed its output-generation queues and hardware-sync gate.
    pub fn enqueue_fixp_l1_write(
        &mut self,
        fragment: super::interface::C220MteOutputFragment,
    ) -> Result<(), C220MtePipelineError> {
        self.fixp_write.enqueue(self.events.tick(), fragment)?;
        Ok(())
    }

    pub fn fixp_completions(&self) -> &[u64] {
        &self.fixp_completions
    }

    /// Bind the explicit FIX stage order selected by the owning scheduler.
    /// Existing memory and interface subscribers retain their positions.
    pub fn bind_fixp_stages(
        &mut self,
        stages: &[C220FixpStage],
    ) -> Result<(), C220MtePipelineError> {
        if !self.fixp_events.is_empty() || !self.external_fixp_events.is_empty() || !self.is_idle()
        {
            return Err(C220MtePipelineError::FixpBindingBusy);
        }
        if stages.len() != 9
            || stages
                .iter()
                .enumerate()
                .any(|(index, stage)| stages[..index].contains(stage))
        {
            return Err(C220MtePipelineError::InvalidFixpStages);
        }
        for (index, &stage) in stages.iter().enumerate() {
            self.fixp_events.push(C220FixpStageEvents::register(
                &mut self.events,
                self.clock,
                stage,
                |phase| Callback::Fixp(index, phase),
            ));
        }
        Ok(())
    }

    pub fn send_fixp_output(
        &mut self,
        engine: &mut super::fixp::C220FixpEngine,
    ) -> Result<
        super::fixp::C220FixpWriteProgress<super::fixp::C220FixpDispatchPacket>,
        C220MtePipelineError,
    > {
        Ok(engine.send_write(
            self.events.tick(),
            &mut self.fixp_write,
            &mut self.interface,
            self.biu_write_commands.as_mut(),
        )?)
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
        if self.timed_memory.is_some() {
            return Err(C220MtePipelineError::MemoryOwnedRead);
        }
        if !self.is_idle() {
            return Err(C220MtePipelineError::CommandBusy);
        }
        self.dma_connected = true;
        self.biu_read = None;
        self.biu_returns = None;
        self.biu_bus_reads = None;
        self.nd2nz = None;
        Ok(())
    }

    /// Connects BIU reads and return reordering to native L1 or UB write service.
    /// The external memory service supplies only the BIU read response beats.
    pub fn connect_mte2_biu(
        &mut self,
        config: C220BiuReadConfig,
        subcore: C220BiuSubcore,
    ) -> Result<(), C220MtePipelineError> {
        if self.timed_memory.is_some() {
            return Err(C220MtePipelineError::MemoryOwnedRead);
        }
        if !self.is_idle() {
            return Err(C220MtePipelineError::CommandBusy);
        }
        let returns = C220BiuReadReturns::new(
            config.outstanding,
            config.group_vector_returns,
            config.write_bandwidths,
        )?;
        self.biu_read = Some(C220BiuReadFrontend::new(config));
        self.biu_bus_reads = None;
        self.biu_returns = Some(returns);
        self.biu_subcore = subcore;
        if subcore != C220BiuSubcore::Cube {
            self.nd2nz = None;
        } else if self.nd2nz.is_none() {
            self.nd2nz = Some(C220Nd2NzEngine::new(Default::default())?);
        }
        self.dma_connected = true;
        Ok(())
    }

    pub fn biu_read(&self) -> Option<&C220BiuReadFrontend> {
        self.biu_read.as_ref()
    }

    pub fn take_biu_read_request(&mut self) -> Option<C220BiuReadRequest> {
        if self.timed_memory.is_some() {
            return None;
        }
        if let Some(bus) = &mut self.biu_bus_reads {
            if !matches!(
                bus.queued_commands().front()?.tag,
                crate::sim::c220::memory::timed_memory::C220MemoryReadId::Mte(_)
            ) {
                return None;
            }
            let command = bus.take_command(self.events.tick())?;
            let crate::sim::c220::memory::timed_memory::C220MemoryReadId::Mte(tag) = command.tag
            else {
                unreachable!("MTE head");
            };
            return self
                .biu_returns
                .as_ref()?
                .progress(tag)
                .map(|progress| progress.request);
        }
        self.biu_output.take()
    }

    pub fn biu_read_returns(&self) -> Option<&C220BiuReadReturns> {
        self.biu_returns.as_ref()
    }

    pub fn receive_biu_read(
        &mut self,
        heads: [Option<C220BiuReadBeat>; 2],
    ) -> Result<[bool; 2], C220MtePipelineError> {
        if self.timed_memory.is_some() {
            return Err(C220MtePipelineError::MemoryOwnedRead);
        }
        if let Some(bus) = &mut self.biu_bus_reads {
            return Ok(bus.receive(
                self.events.tick(),
                heads.map(|head| {
                    head.map(
                        |beat| crate::sim::c220::memory::timed_memory::C220MemoryReadBeat {
                            tag: crate::sim::c220::memory::timed_memory::C220MemoryReadId::Mte(
                                beat.tag,
                            ),
                            transaction_id: beat.transaction_id,
                        },
                    )
                }),
            )?);
        }
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

    pub fn mte3_frontend(&self) -> &C220Mte3Frontend {
        &self.mte3
    }

    pub fn connect_mte3_biu_retirement(&mut self) -> Result<(), C220MtePipelineError> {
        if !self.is_idle() {
            return Err(C220MtePipelineError::CommandBusy);
        }
        if self.biu_write_source(self.biu_subcore)?.is_none() {
            return Err(C220MtePipelineError::BiuWriteSourceDisconnected);
        }
        Ok(self.mte3.connect_biu_retirement()?)
    }

    pub fn issue_mte3_dma(
        &mut self,
        instruction_id: u64,
        transfer: C220Mte3TransferPlan,
    ) -> Result<C220Mte3Record, C220MtePipelineError> {
        Ok(self
            .mte3_events
            .issue(&mut self.events, &mut self.mte3, instruction_id, transfer)?)
    }

    pub fn take_mte3_dma_output(&mut self) -> Option<C220DmaGenerated> {
        if self.biu_write_commands.is_some() {
            return None;
        }
        self.mte3.take_output()
    }

    pub fn issue_mte3_mov_pad(
        &mut self,
        instruction_id: u64,
        command: super::mov_pad::C220MovPadCommand,
    ) -> Result<C220Mte3Record, C220MtePipelineError> {
        Ok(self.mte3_events.issue_command(
            &mut self.events,
            &mut self.mte3,
            instruction_id,
            super::mte3::frontend::C220Mte3Command::MovPad(command),
        )?)
    }

    pub(crate) fn issue_mte3_cross_core(
        &mut self,
        instruction_id: u64,
        instruction: crate::isa::c220::control::C220SetCrossCoreInstruction,
        payload: crate::sim::c220::sync::C220DeviceSync,
    ) -> Result<C220Mte3Record, C220MtePipelineError> {
        Ok(self.mte3_events.issue_command(
            &mut self.events,
            &mut self.mte3,
            instruction_id,
            super::mte3::frontend::C220Mte3Command::CrossCore {
                instruction,
                payload,
            },
        )?)
    }

    pub fn acknowledge_mte3_dma(
        &mut self,
        instruction_id: u64,
        uop_index: u64,
    ) -> Result<(), C220MtePipelineError> {
        Ok(self.mte3.acknowledge(instruction_id, uop_index)?)
    }

    pub fn set_mte3_hardware_sync_blocked(&mut self, blocked: bool) {
        self.mte3.set_hardware_sync_blocked(blocked);
    }

    pub fn mte3_retirement_candidate(&self) -> Option<C220Mte3Record> {
        self.mte3.retirement_candidate(self.events.tick())
    }

    /// Release the record only after the owner successfully commits its functional effects.
    pub fn retire_mte3(
        &mut self,
        instruction_id: u64,
    ) -> Result<C220Mte3Record, C220MtePipelineError> {
        Ok(self.mte3.retire(self.events.tick(), instruction_id)?)
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

    pub(crate) fn mte2_generator_idle(&self) -> bool {
        self.selected_mte2_generator
            .is_none_or(|generator| match generator {
                Mte2Generator::Nd2Nz => self.nd2nz.as_ref().is_none_or(C220Nd2NzEngine::is_idle),
                Mte2Generator::ExternalLoad2d => self.external_load2d.is_idle(),
                Mte2Generator::Default => self.generator(C220MteReadKind::Default).is_idle(),
                Mte2Generator::Dma => self.dma.is_idle(),
                Mte2Generator::L1Fill => self.set2d_l1.is_idle(),
                Mte2Generator::Load3d => true,
            })
    }

    pub(crate) fn can_issue_mte2_cross_core(&self) -> bool {
        self.mte2_generator_idle()
    }

    pub(crate) fn issue_mte2_cross_core(&mut self) -> Result<u64, C220MtePipelineError> {
        if !self.can_issue_mte2_cross_core() {
            return Err(C220MtePipelineError::CommandBusy);
        }
        self.selected_mte2_generator = Some(Mte2Generator::Load3d);
        Ok(self.events.tick())
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
        if self.biu_read.is_some() && self.biu_subcore == C220BiuSubcore::Cube {
            return Err(C220MtePipelineError::BiuUbSubcoreRequired);
        }
        let requests = mte2_uops(transfer)?;
        self.issue_dma_requests(instruction_id, requests)
    }

    pub fn issue_mte2_l1_dma(
        &mut self,
        instruction_id: u64,
        descriptor: C220L1DmaDescriptor,
        source_address: u64,
        destination_address: u64,
        dma_mode_word: u64,
    ) -> Result<C220DmaIssue, C220MtePipelineError> {
        if self.biu_read.is_none() || self.biu_subcore != C220BiuSubcore::Cube {
            return Err(C220MtePipelineError::BiuL1SubcoreRequired);
        }
        let requests = mte2_l1_uops(
            descriptor,
            source_address,
            destination_address,
            dma_mode_word,
        )?;
        self.issue_dma_requests(instruction_id, requests)
    }

    pub fn issue_mte2_mov_pad(
        &mut self,
        instruction_id: u64,
        command: super::mov_pad::C220MovPadCommand,
    ) -> Result<C220DmaIssue, C220MtePipelineError> {
        if !command.transfer.is_input() {
            return Err(C220MtePipelineError::WrongMovPadDirection);
        }
        if command.transfer.is_disabled() {
            return Ok(C220DmaIssue {
                tick: self.events.tick(),
                instruction_id,
                completion_ready: true,
            });
        }
        if self.biu_read.is_some() && self.biu_subcore == C220BiuSubcore::Cube {
            return Err(C220MtePipelineError::BiuUbSubcoreRequired);
        }
        let requests = super::uop::mov_pad_uops(
            command.transfer,
            super::uop::C220DmaUopMode::from_mode_word(command.biu_mode_word),
        )?;
        self.issue_dma_requests(instruction_id, requests)
    }

    fn issue_dma_requests(
        &mut self,
        instruction_id: u64,
        requests: C220DmaUops,
    ) -> Result<C220DmaIssue, C220MtePipelineError> {
        if !self.dma_connected {
            return Err(C220MtePipelineError::DmaDisconnected);
        }
        if requests.clone().next().is_none() {
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
    pub fn interface(&self) -> &C220MteL1Interface<C220MteReadPayload> {
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
    pub fn mte2_read_completions(&self) -> &[u64] {
        &self.mte2_read_completions
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
            || matches!(command, C220Mte1Command::Read(super::read::C220MteReadTransfer::Smask(transfer)) if transfer.instruction.source_mode != 2)
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
            C220Mte1Command::CrossCore { .. }
            | C220Mte1Command::WriteSpr(_)
            | C220Mte1Command::HardwareFlag(_) => C220Mte1Issue {
                tick: self.events.tick(),
                instruction_id,
                uop_count: 0,
                completion_ready: true,
            },
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
        if let Some(generator) = command.generator() {
            self.selected_generator = Some(generator);
        }
        Ok(issue)
    }

    /// The owner supplies every active clock tick and consumes completions
    /// before advancing again. Idle intervals may be skipped.
    pub fn advance(&mut self, tick: u64) -> Result<(), C220MtePipelineError> {
        self.advance_inner(tick, None, None)
    }

    pub fn advance_fixp(
        &mut self,
        tick: u64,
        engine: &mut C220FixpEngine,
        memory: C220FixpMemory<'_>,
        mut gates: impl C220FixpSync,
    ) -> Result<(), C220MtePipelineError> {
        self.advance_inner(tick, Some((engine, memory, &mut gates)), None)
    }

    fn advance_inner(
        &mut self,
        tick: u64,
        mut fixp: Option<(
            &mut C220FixpEngine,
            C220FixpMemory<'_>,
            &mut dyn C220FixpSync,
        )>,
        mut external_fixp: Option<(
            &mut C220FixpRuntime,
            C220FixpRuntimeMemory<'_>,
            &mut dyn C220FixpSync,
        )>,
    ) -> Result<(), C220MtePipelineError> {
        if self.fixp_events.is_empty() == fixp.is_some()
            || self.external_fixp_events.is_empty() == external_fixp.is_some()
        {
            return Err(C220MtePipelineError::FixpContextMismatch);
        }
        if self.last_advance == Some(tick) {
            return Ok(());
        }
        if let Some(active) = self.active_cycle {
            if active != tick {
                return Err(C220MtePipelineError::UnfinishedCycle {
                    active,
                    requested: tick,
                });
            }
            if self.fixp_dispatch_pending || self.fixp_issue_pending {
                return Err(C220MtePipelineError::FixpDispatchPending);
            }
            if self.mte1_dispatch_pending || self.mte1_issue_pending {
                return Err(C220MtePipelineError::Mte1DispatchPending);
            }
            if self.mte2_dispatch_pending || self.mte2_issue_pending {
                return Err(C220MtePipelineError::Mte2DispatchPending);
            }
            if self.mte3_issue_pending {
                return Err(C220MtePipelineError::Mte3IssuePending);
            }
            if self.mte1_sync_pending() {
                return Err(C220MtePipelineError::Mte1SyncPending);
            }
        } else {
            let active = !self.is_idle()
                || fixp
                    .as_ref()
                    .is_some_and(|(engine, _, _)| !engine.is_idle())
                || external_fixp
                    .as_ref()
                    .is_some_and(|(engine, _, _)| !engine.is_idle());
            if active
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
            self.fixp_completions.clear();
            self.l1_fill_completions.clear();
            self.mte2_read_completions.clear();
            self.dma_completions.clear();
            self.trace.clear();
            if let Some((engine, _, _)) = fixp.as_mut() {
                engine.retire_ready_write(tick)?;
            }
            if let Some((engine, _, _)) = external_fixp.as_mut()
                && let Some(retired) = engine.retire_ready_write(tick)?
            {
                self.trace.push(match retired.external {
                    Some(operands) => {
                        C220MtePipelineEvent::FixpExternal(C220FixpRuntimeEvent::Retired {
                            tick,
                            instruction_id: retired.instruction_id,
                            state: super::fixp::C220FixpExternalCommandState {
                                operands,
                                lifecycle: retired.lifecycle,
                            },
                        })
                    }
                    None => C220MtePipelineEvent::Fixp(C220FixpEvent::Retired {
                        tick,
                        instruction_id: retired.instruction_id,
                        state: retired.lifecycle,
                    }),
                });
            }
            if let Some(memory) = &mut self.timed_memory {
                memory.advance(tick)?;
            }
            self.advance_biu_bus_returns(tick)?;
            if let Some((engine, _, _)) = external_fixp.as_mut() {
                for &id in &self.fixp_completions {
                    engine.complete_write_transport(tick, id)?;
                }
            }
            self.advance_biu_read_returns(tick)?;
            self.events.notify_at(self.clock, tick);
            self.active_cycle = Some(tick);
        }
        while let Some(invocation) = self.events.next_callback() {
            match invocation.callback {
                Callback::FixpIssueProbe => {
                    if self.fixp_issue_ready.is_some_and(|ready| ready <= tick) {
                        self.events.notify_at(self.fixp_issue_valid, tick);
                    }
                }
                Callback::FixpIssueTransfer => {
                    self.fixp_issue_pending = true;
                    return Ok(());
                }
                Callback::Mte1IssueProbe => {
                    if self.mte1_issue_ready.is_some_and(|ready| ready <= tick) {
                        self.events.notify_at(self.mte1_issue_valid, tick);
                    }
                }
                Callback::Mte1IssueTransfer => {
                    self.mte1_issue_pending = true;
                    return Ok(());
                }
                Callback::Mte2IssueProbe => {
                    if self.mte2_issue_ready.is_some_and(|ready| ready <= tick) {
                        self.events.notify_at(self.mte2_issue_valid, tick);
                    }
                }
                Callback::Mte2IssueTransfer => {
                    self.mte2_issue_pending = true;
                    return Ok(());
                }
                Callback::Mte3IssueProbe => {
                    if self.mte3_issue_ready.is_some_and(|ready| ready <= tick) {
                        self.events.notify_at(self.mte3_issue_valid, tick);
                    }
                }
                Callback::Mte3IssueTransfer => {
                    self.mte3_issue_pending = true;
                    return Ok(());
                }
                Callback::Mte2CommandProbe => {
                    if self.mte2_command_ready.is_some_and(|ready| ready <= tick) {
                        self.events.notify_at(self.mte2_command_valid, tick);
                    }
                }
                Callback::Mte2CommandDispatch => {
                    self.mte2_dispatch_pending = true;
                    return Ok(());
                }
                Callback::FixpCommandProbe => {
                    if self.fixp_command_ready.is_some_and(|ready| ready <= tick) {
                        self.events.notify_at(self.fixp_command_valid, tick);
                    }
                }
                Callback::FixpCommandDispatch => {
                    self.fixp_dispatch_pending = true;
                    return Ok(());
                }
                Callback::Mte1CommandProbe => {
                    if self.mte1_command_ready.is_some_and(|ready| ready <= tick) {
                        self.events.notify_at(self.mte1_command_valid, tick);
                    }
                }
                Callback::Mte1CommandDispatch => {
                    self.mte1_dispatch_pending = true;
                    return Ok(());
                }
                Callback::FixpExternal(index, phase) => {
                    let (engine, memory, gates) = external_fixp
                        .as_mut()
                        .expect("validated external FIX context");
                    self.handle_external_fixp(index, phase, engine, memory, &mut **gates)?;
                }
                Callback::Fixp(index, phase) => {
                    let (engine, memory, gates) = fixp.as_mut().expect("validated FIX context");
                    let outcome = self.fixp_events[index].handle(
                        phase,
                        &mut self.events,
                        engine,
                        C220FixpResources {
                            biu: self.biu_write_commands.as_mut(),
                            l0c: memory.l0c,
                            slopes: memory.slopes,
                            l1: memory.l1,
                            writer: &mut self.fixp_write,
                            reader: &mut self.interface,
                            gates: &mut **gates,
                        },
                        |slice| {
                            self.trace
                                .push(C220MtePipelineEvent::FixpSlice(slice.clone()))
                        },
                    )?;
                    if outcome != C220FixpEvent::Readiness {
                        self.trace.push(C220MtePipelineEvent::Fixp(outcome));
                    }
                }
                Callback::FixpWrite(phase) => {
                    let outcome = self.fixp_write_events.handle(
                        phase,
                        &mut self.events,
                        &mut self.fixp_write,
                        &mut self.memory,
                    )?;
                    if let C220FixpL1WriteEvent::Acknowledged(Some(entry)) = outcome
                        && entry.fragment.last_in_instruction
                    {
                        self.fixp_completions.push(entry.fragment.instruction_id);
                        if let Some((engine, _, _)) = fixp.as_mut() {
                            engine.complete_write_transport(tick, entry.fragment.instruction_id)?;
                        }
                        if let Some((engine, _, _)) = external_fixp.as_mut() {
                            engine.complete_write_transport(tick, entry.fragment.instruction_id)?;
                        }
                    }
                    if outcome != C220FixpL1WriteEvent::Readiness {
                        self.trace.push(C220MtePipelineEvent::FixpWrite(outcome));
                    }
                }
                Callback::BiuWriteSource(index) => {
                    if let Some(source) = &mut self.biu_write_source[index] {
                        if let Some(tag) = source.starting_source(tick)
                            && let Some(commands) = &mut self.biu_write_commands
                        {
                            let request = commands.begin_source(tag);
                            source.set_source_tail(tag, request.last_in_instruction);
                        }
                        for event in source.advance(tick, &mut self.ub_read[index])? {
                            self.trace.push(C220MtePipelineEvent::BiuWriteSource(
                                [C220BiuSubcore::Vector0, C220BiuSubcore::Vector1][index],
                                event,
                            ));
                        }
                    }
                }
                Callback::UbReadProbe(index) => {
                    if self.ub_read[index]
                        .inputs()
                        .front()
                        .is_some_and(|head| head.ready_tick <= tick)
                    {
                        self.events.notify_at(self.ub_read_valid[index], tick);
                    }
                }
                Callback::UbReadSend(index) => {
                    if let Some(request) = self.ub_read[index].send(tick)? {
                        self.trace.push(C220MtePipelineEvent::UbReadSent(
                            [C220BiuSubcore::Vector0, C220BiuSubcore::Vector1][index],
                            request,
                        ));
                    }
                }
                Callback::Mte3(phase) => {
                    let outstanding_fixp = external_fixp.as_ref().map_or(0, |(engine, _, _)| {
                        engine.shared_engine().outstanding_external_commands()
                    });
                    self.mte3.set_external_fixp_pending(
                        self.core_kind == crate::sim::c220::device::C220CoreKind::Cube
                            && outstanding_fixp != 0,
                    );
                    self.mte3.set_fixp_head_pending(
                        self.core_kind == crate::sim::c220::device::C220CoreKind::Cube
                            && self.fixp_head_is_convert,
                    );
                    if let Some(outcome) =
                        self.mte3_events
                            .handle(phase, &mut self.events, &mut self.mte3)?
                    {
                        self.trace.push(C220MtePipelineEvent::Mte3(outcome));
                    }
                }
                Callback::ExternalLoad2d(phase) => self.advance_external_load2d(phase)?,
                Callback::Nd2Nz(phase) => self.advance_nd2nz(phase)?,
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
                                        C220BiuSubcore::Cube => C220BiuWriteDestination::L1,
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
                            self.biu_bus_reads
                                .as_ref()
                                .map_or(self.biu_output.is_none(), C220BiuBusReads::can_push),
                        )?;
                        if let C220BiuReadEvent::Send(send) = outcome
                            && let Some(request) = send.sent()
                        {
                            self.biu_returns
                                .as_mut()
                                .expect("connected return path")
                                .track(tick, request)?;
                            if let Some(bus) = &mut self.biu_bus_reads {
                                bus.push(tick, read::memory_read_command(request, tick))?;
                            } else {
                                self.biu_output = Some(request);
                            }
                        }
                        if outcome != C220BiuReadEvent::Readiness {
                            self.trace.push(C220MtePipelineEvent::BiuRead(outcome));
                        }
                    }
                }
                Callback::BiuReturn(phase) => {
                    if self.advance_nd2nz_return(phase)? {
                        continue;
                    }
                    let cube_ready = self.cube_read_output_ready();
                    if let Some(returns) = &mut self.biu_returns {
                        let outcome = self.biu_return_events.handle(
                            phase,
                            &mut self.events,
                            returns,
                            [
                                cube_ready,
                                self.ub_write[0].can_push(),
                                self.ub_write[1].can_push(),
                            ],
                        )?;
                        if let C220BiuReturnEvent::Send(send) = &outcome
                            && let Some(fragment) = send.sent()
                        {
                            match send.core {
                                C220BiuSubcore::Cube => {
                                    match fragment.output.request.input.destination {
                                        C220BiuWriteDestination::L0A => assert!(self.l0[0].push(
                                            tick,
                                            C220L0WritePort::Port2,
                                            fragment.output_fragment()
                                        )?),
                                        C220BiuWriteDestination::L0B => assert!(self.l0[1].push(
                                            tick,
                                            C220L0WritePort::Port2,
                                            fragment.output_fragment()
                                        )?),
                                        C220BiuWriteDestination::L1 => {
                                            assert!(self.write_interface.push(
                                                tick,
                                                C220MteL1WritePort::Port0,
                                                fragment.output_fragment()
                                            )?)
                                        }
                                        _ => unreachable!("cube destination"),
                                    }
                                }
                                C220BiuSubcore::Vector0 => {
                                    assert!(self.ub_write[0].push(tick, fragment)?)
                                }
                                C220BiuSubcore::Vector1 => {
                                    assert!(self.ub_write[1].push(tick, fragment)?)
                                }
                            }
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
                                if ack.request.port == C220MteL1WritePort::Port0 {
                                    self.load2d_destinations.remove(&id);
                                    self.dma_completions.push(id);
                                } else {
                                    self.l1_fill_completions.push(id);
                                }
                            }
                        }
                        _ => {}
                    }
                    if outcome != C220MteL1WriteEventOutcome::Readiness {
                        self.trace.push(C220MtePipelineEvent::L1Write(outcome));
                    }
                }
                callback @ (Callback::Set2d(_) | Callback::Generator(_, _)) => {
                    if self.defer_mte1_sync(callback, tick) {
                        return Ok(());
                    }
                    self.handle_mte1_send_callback(callback, false)?;
                }
                Callback::Memory(phase) => {
                    let outcome =
                        self.memory_events
                            .handle(phase, &mut self.events, &mut self.memory)?;
                    if outcome != C220L1EventOutcome::Readiness {
                        self.trace.push(C220MtePipelineEvent::Memory(outcome));
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
                        if self.load2d_destinations.remove(&id).is_some() {
                            self.dma_completions.push(id);
                        } else {
                            self.completions.push(id);
                        }
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
                                    C220MteL1OutputDestination::Bt
                                    | C220MteL1OutputDestination::Fb
                                    | C220MteL1OutputDestination::Smask
                                    | C220MteL1OutputDestination::SparseIndex => None,
                                };
                                if let Some((index, port)) = target {
                                    assert!(self.l0[index].push(tick, port, sent.fragment)?);
                                }
                            }
                        }
                        C220MteL1EventOutcome::Retired(Some(output))
                            if output.fragment.last_in_instruction =>
                        {
                            match output.payload.operation.payload {
                                C220MteReadPayload::Read(C220MteReadUop::Smask(uop))
                                    if uop.source_mode == 0 =>
                                {
                                    self.mte2_read_completions
                                        .push(output.fragment.instruction_id)
                                }
                                C220MteReadPayload::Read(_) => {
                                    self.completions.push(output.fragment.instruction_id)
                                }
                                C220MteReadPayload::Factor(_) => {
                                    if let Some((engine, _, _)) = fixp.as_mut() {
                                        engine.complete_factor_transport(
                                            tick,
                                            output.fragment.instruction_id,
                                        )?;
                                    }
                                    if let Some((engine, _, _)) = external_fixp.as_mut() {
                                        engine.shared_engine_mut().complete_factor_transport(
                                            tick,
                                            output.fragment.instruction_id,
                                        )?;
                                    }
                                    self.fixp_completions.push(output.fragment.instruction_id)
                                }
                            }
                        }
                        _ => {}
                    }
                    if outcome != C220MteL1EventOutcome::Readiness {
                        self.trace.push(C220MtePipelineEvent::Interface(outcome));
                    }
                }
            }
        }
        self.advance_biu_write_commands(tick)?;
        self.advance_biu_cube_source(tick)?;
        self.advance_biu_write_data(tick)?;
        self.advance_biu_bus_inputs(tick)?;
        self.advance_biu_read_inputs(tick)?;
        self.last_advance = Some(tick);
        self.active_cycle = None;
        Ok(())
    }
}
