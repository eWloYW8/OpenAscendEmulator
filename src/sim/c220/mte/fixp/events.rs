use super::*;
use crate::sim::c220::memory::{C220L0c, C220LocalBuffer};
use crate::sim::c220::mte::interface::{
    C220MteL0cReadResponse, C220MteL0cReadSend, C220MteOutputFragment,
};
use crate::sim::common::event::{EventDispatcher, EventId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220FixpStage {
    GenerateRead,
    SendRead,
    SendL0c,
    ReceiveL0c,
    Convert,
    Slice,
    Packetize,
    GenerateWrite,
    SendWrite,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220FixpCallback {
    Probe,
    Execute,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220FixpEvent {
    Readiness,
    GeneratedRead(C220FixpReadProgress),
    SentRead(C220FixpReadProgress),
    SentL0c(C220MteL0cReadSend),
    ReceivedL0c {
        response: C220MteL0cReadResponse,
        functional: Option<C220FixpFunctionalEvent>,
    },
    Converted(C220FixpConversionReceive),
    Sliced(Option<C220FixpConversionEntry>),
    Packetized(Option<C220MteOutputFragment>),
    GeneratedWrite(C220FixpWriteProgress<C220FixpDispatchPacket>),
    SentWrite(C220FixpWriteProgress<C220FixpDispatchPacket>),
}

/// Read dispatch and final conversion admission have independent gates.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct C220FixpGates {
    pub read_blocked: bool,
    pub conversion_blocked: bool,
}

impl C220FixpSync for C220FixpGates {
    fn blocked(
        &mut self,
        request: C220FixpSyncRequest,
    ) -> Result<bool, crate::sim::c220::sync::C220HardwareFlagTimingError> {
        Ok(match request.point {
            C220FixpSyncPoint::ReadWait | C220FixpSyncPoint::DisabledWait => self.read_blocked,
            C220FixpSyncPoint::ConversionSet | C220FixpSyncPoint::DisabledSet => {
                self.conversion_blocked
            }
        })
    }
}

pub struct C220FixpResources<'a> {
    pub l0c: &'a mut C220L0c,
    pub slopes: &'a C220LocalBuffer,
    pub l1: &'a mut C220LocalBuffer,
    pub writer: &'a mut C220FixpL1WriteInterface,
    pub reader: &'a mut crate::sim::c220::mte::interface::C220MteL1Interface<
        crate::sim::c220::mte::C220MteReadPayload,
    >,
    pub gates: &'a mut dyn C220FixpSync,
}

pub struct C220FixpMemory<'a> {
    pub l0c: &'a mut C220L0c,
    pub slopes: &'a C220LocalBuffer,
    pub l1: &'a mut C220LocalBuffer,
}

/// One queue-readiness binding. Owners register stages where they belong in
/// the shared clock topology; this API does not impose a global phase order.
/// Do not also invoke a bound engine stage directly in the same simulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FixpStageEvents<S = C220FixpStage> {
    stage: S,
    valid: EventId,
}

impl<S: Copy> C220FixpStageEvents<S> {
    pub fn register<T: Copy>(
        events: &mut EventDispatcher<T>,
        clock: EventId,
        stage: S,
        tag: impl Fn(C220FixpCallback) -> T,
    ) -> Self {
        let valid = events.add_event();
        let probe = events.add_process(tag(C220FixpCallback::Probe), false);
        let execute = events.add_process(tag(C220FixpCallback::Execute), false);
        events.subscribe(clock, probe);
        events.subscribe(valid, execute);
        Self { stage, valid }
    }

    pub(crate) fn stage(&self) -> S {
        self.stage
    }

    pub(crate) fn probe<T: Copy>(&self, events: &mut EventDispatcher<T>, ready_tick: Option<u64>) {
        if ready_tick.is_some_and(|ready| ready <= events.tick()) {
            events.notify_at(self.valid, events.tick());
        }
    }
}

impl C220FixpStageEvents {
    pub fn handle<T: Copy>(
        &self,
        callback: C220FixpCallback,
        events: &mut EventDispatcher<T>,
        engine: &mut C220FixpEngine,
        resources: C220FixpResources<'_>,
        observe: impl FnMut(&C220FixpSliceResult),
    ) -> Result<C220FixpEvent, C220FixpEngineError> {
        let tick = events.tick();
        if callback == C220FixpCallback::Probe {
            self.probe(events, engine.stage_ready_tick(self.stage, tick));
            return Ok(C220FixpEvent::Readiness);
        }
        use C220FixpStage::*;
        Ok(match self.stage {
            GenerateRead => C220FixpEvent::GeneratedRead(engine.generate_read(tick)?),
            SendRead => C220FixpEvent::SentRead(engine.send_read(tick, resources.gates)?),
            SendL0c => C220FixpEvent::SentL0c(engine.send_l0c(tick, resources.l0c)?),
            ReceiveL0c => {
                let (response, functional) = engine.receive_l0c(
                    tick,
                    resources.l0c,
                    resources.slopes,
                    resources.l1,
                    observe,
                )?;
                C220FixpEvent::ReceivedL0c {
                    response,
                    functional,
                }
            }
            Convert => C220FixpEvent::Converted(engine.convert(tick, resources.gates)?),
            Slice => C220FixpEvent::Sliced(engine.slice(tick)?),
            Packetize => C220FixpEvent::Packetized(engine.packetize(tick)?),
            GenerateWrite => C220FixpEvent::GeneratedWrite(engine.generate_write(tick)?),
            SendWrite => C220FixpEvent::SentWrite(engine.send_write(
                tick,
                resources.writer,
                resources.reader,
            )?),
        })
    }
}
