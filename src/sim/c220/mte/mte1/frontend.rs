use std::collections::VecDeque;
use std::num::NonZeroU32;

use super::bias::{C220BtReadUop, C220BtRequestPlan};
use super::load2d::{C220Load2dReadUop, C220Load2dRequestPlan};
use crate::isa::c220::mte::bias::C220BtTransfer;
use crate::isa::c220::mte::load2d::{C220Load2dDestination, C220Load2dError, C220Load2dTransfer};
use crate::sim::c220::memory::l1::C220L1Access;
use crate::sim::c220::mte::interface::{
    C220L0WritePort, C220MteL1Error, C220MteL1Interface, C220MteL1OutputDestination,
    C220MteL1ReadOperation, C220MteL1ReadPort, C220MteL1ReadRequest,
};

const COMMAND_TICKS: u64 = 1;
const GENERATED_TICKS: u64 = 3;
const GENERATED_CAPACITY: usize = 4;

mod events;
pub use events::{C220Mte1ReadEventOutcome, C220Mte1ReadEvents};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220Mte1ReadKind {
    Load2d,
    Bt,
}

impl C220Mte1ReadKind {
    pub(in crate::sim::c220::mte) const ALL: [Self; 2] = [Self::Load2d, Self::Bt];

    pub(in crate::sim::c220::mte) const fn index(self) -> usize {
        match self {
            Self::Load2d => 0,
            Self::Bt => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Mte1ReadBandwidths {
    pub l0a: NonZeroU32,
    pub l0b: NonZeroU32,
    pub bt: NonZeroU32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220Mte1ReadTransfer {
    Load2d(C220Load2dTransfer),
    Bt(C220BtTransfer),
}

impl C220Mte1ReadTransfer {
    pub const fn is_empty(self) -> bool {
        match self {
            Self::Load2d(transfer) => transfer.descriptor.repeat_count == 0,
            Self::Bt(transfer) => transfer.descriptor.is_empty(),
        }
    }

    pub const fn kind(self) -> C220Mte1ReadKind {
        match self {
            Self::Load2d(_) => C220Mte1ReadKind::Load2d,
            Self::Bt(_) => C220Mte1ReadKind::Bt,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220Mte1ReadUop {
    Load2d(C220Load2dReadUop),
    Bt(C220BtReadUop),
}

impl C220Mte1ReadUop {
    fn operation(
        self,
        instruction_id: u64,
        bandwidths: C220Mte1ReadBandwidths,
    ) -> C220MteL1ReadOperation<Self> {
        let (
            source_address,
            input_bytes,
            destination,
            output_address,
            output_bytes,
            output_bandwidth,
            completes_logical_uop,
            last_in_instruction,
        ) = match self {
            Self::Bt(uop) => (
                uop.source_address,
                uop.input_bytes,
                C220MteL1OutputDestination::Bt,
                uop.logical.destination_address,
                uop.logical.output_bytes,
                bandwidths.bt,
                uop.completes_logical_uop,
                uop.last_in_instruction,
            ),
            Self::Load2d(uop) => {
                let (destination, bandwidth) = match uop.destination {
                    C220Load2dDestination::L0a => (
                        C220MteL1OutputDestination::L0a(C220L0WritePort::Port0),
                        bandwidths.l0a,
                    ),
                    C220Load2dDestination::L0b => (
                        C220MteL1OutputDestination::L0b(C220L0WritePort::Port0),
                        bandwidths.l0b,
                    ),
                    _ => unreachable!("LOAD2D route validated at admission"),
                };
                (
                    uop.source_address,
                    uop.input_bytes,
                    destination,
                    uop.logical.destination_address,
                    uop.logical.bytes,
                    bandwidth,
                    uop.completes_logical_uop,
                    uop.last_in_instruction,
                )
            }
        };
        C220MteL1ReadOperation {
            instruction_id,
            access: C220L1Access {
                address: source_address,
                bytes: input_bytes,
            },
            destination,
            output_address,
            output_bytes,
            output_bandwidth,
            completes_logical_uop,
            last_in_instruction,
            payload: self,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Plan {
    Load2d(C220Load2dRequestPlan),
    Bt(C220BtRequestPlan),
}

impl Plan {
    fn remaining(&self) -> u64 {
        match self {
            Self::Load2d(plan) => plan.len() as u64,
            Self::Bt(plan) => plan.remaining_requests(),
        }
    }

    fn next(&mut self) -> Option<C220Mte1ReadUop> {
        match self {
            Self::Load2d(plan) => plan.next().map(C220Mte1ReadUop::Load2d),
            Self::Bt(plan) => plan.next().map(C220Mte1ReadUop::Bt),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Generation {
    ready_tick: u64,
    instruction_id: u64,
    plan: Plan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Mte1ReadGenerated {
    pub ready_tick: u64,
    pub operation: C220MteL1ReadOperation<C220Mte1ReadUop>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Mte1ReadIssue {
    pub tick: u64,
    pub instruction_id: u64,
    pub request_count: u64,
    /// No output acknowledgment is needed; the command owner may retire it.
    pub completion_ready: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220Mte1ReadStall {
    NotReady,
    HardwareFlag,
    OutputFull,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Mte1ReadSend {
    pub tick: u64,
    pub offered: Option<C220Mte1ReadGenerated>,
    pub stall: Option<C220Mte1ReadStall>,
    pub queued: Option<C220MteL1ReadRequest<C220Mte1ReadUop>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Mte1ReadFrontendQueues {
    pub instruction_requests: u64,
    pub generated: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220Mte1ReadFrontendCycle {
    pub tick: u64,
    pub generated: Option<C220MteL1ReadOperation<C220Mte1ReadUop>>,
    pub queued: Option<C220MteL1ReadRequest<C220Mte1ReadUop>>,
    pub queues: C220Mte1ReadFrontendQueues,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum C220Mte1ReadFrontendError {
    #[error("MTE1 read generator cannot accept a command")]
    CommandBusy,
    #[error("MTE1 read generator {expected:?} cannot accept {requested:?}")]
    WrongGenerator {
        expected: C220Mte1ReadKind,
        requested: C220Mte1ReadKind,
    },
    #[error("MTE1 read generator time reversed from {previous} to {requested}")]
    TimeReversed { previous: u64, requested: u64 },
    #[error("MTE1 read generator {phase} callback already ran at tick {tick}")]
    RepeatedCallback { phase: &'static str, tick: u64 },
    #[error("MTE1 read generator time overflowed")]
    TimeOverflow,
    #[error(transparent)]
    Load2d(#[from] C220Load2dError),
    #[error(transparent)]
    Interface(#[from] C220MteL1Error),
}

/// One generation engine. LOAD2D (including transpose) and BT use separate
/// instances but feed the same L1 interface and input port. The caller chooses
/// producer callback order explicitly; this type does not invent cross-engine
/// arbitration. An idle frontend does not imply its submitted work has retired.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220Mte1ReadFrontend {
    kind: C220Mte1ReadKind,
    access_width: NonZeroU32,
    bandwidths: C220Mte1ReadBandwidths,
    generation: Option<Generation>,
    generated: VecDeque<C220Mte1ReadGenerated>,
    observed_tick: Option<u64>,
    generation_tick: Option<u64>,
    send_tick: Option<u64>,
}

impl C220Mte1ReadFrontend {
    pub fn new(
        kind: C220Mte1ReadKind,
        access_width: NonZeroU32,
        bandwidths: C220Mte1ReadBandwidths,
    ) -> Self {
        Self {
            kind,
            access_width,
            bandwidths,
            generation: None,
            generated: VecDeque::new(),
            observed_tick: None,
            generation_tick: None,
            send_tick: None,
        }
    }

    pub const fn kind(&self) -> C220Mte1ReadKind {
        self.kind
    }

    pub fn is_idle(&self) -> bool {
        self.generation.is_none() && self.generated.is_empty()
    }

    pub fn can_issue(&self) -> bool {
        self.generation.is_none() && self.generated.len() < GENERATED_CAPACITY
    }

    pub fn queue_state(&self) -> C220Mte1ReadFrontendQueues {
        C220Mte1ReadFrontendQueues {
            instruction_requests: self.generation.as_ref().map_or(0, |g| g.plan.remaining()),
            generated: self.generated.len(),
        }
    }

    pub fn generated(&self) -> &VecDeque<C220Mte1ReadGenerated> {
        &self.generated
    }

    pub(super) fn instruction_ready_tick(&self) -> Option<u64> {
        self.generation
            .as_ref()
            .map(|generation| generation.ready_tick)
    }

    pub fn issue(
        &mut self,
        tick: u64,
        instruction_id: u64,
        transfer: C220Mte1ReadTransfer,
    ) -> Result<C220Mte1ReadIssue, C220Mte1ReadFrontendError> {
        self.check_time(tick)?;
        if transfer.kind() != self.kind {
            return Err(C220Mte1ReadFrontendError::WrongGenerator {
                expected: self.kind,
                requested: transfer.kind(),
            });
        }
        if !self.can_issue() {
            return Err(C220Mte1ReadFrontendError::CommandBusy);
        }
        let plan = match transfer {
            C220Mte1ReadTransfer::Bt(transfer) => Plan::Bt(C220BtRequestPlan::new(
                transfer,
                self.bandwidths.bt,
                self.access_width,
            )),
            C220Mte1ReadTransfer::Load2d(transfer) => {
                Plan::Load2d(C220Load2dRequestPlan::new(transfer, self.access_width)?)
            }
        };
        let request_count = plan.remaining();
        if request_count != 0 {
            let ready_tick = tick
                .checked_add(COMMAND_TICKS)
                .ok_or(C220Mte1ReadFrontendError::TimeOverflow)?;
            self.generation = Some(Generation {
                ready_tick,
                instruction_id,
                plan,
            });
        }
        self.observed_tick = Some(tick);
        Ok(C220Mte1ReadIssue {
            tick,
            instruction_id,
            request_count,
            completion_ready: request_count == 0,
        })
    }

    pub fn generate(
        &mut self,
        tick: u64,
    ) -> Result<Option<C220Mte1ReadGenerated>, C220Mte1ReadFrontendError> {
        self.check_callback(tick, self.generation_tick, "generation")?;
        let eligible = self
            .instruction_ready_tick()
            .is_some_and(|ready| ready <= tick)
            && self.generated.len() < GENERATED_CAPACITY;
        let generated = if eligible {
            let ready_tick = tick
                .checked_add(GENERATED_TICKS)
                .ok_or(C220Mte1ReadFrontendError::TimeOverflow)?;
            let generation = self.generation.as_mut().expect("eligible instruction");
            let operation = generation
                .plan
                .next()
                .expect("nonempty request plan")
                .operation(generation.instruction_id, self.bandwidths);
            let entry = C220Mte1ReadGenerated {
                ready_tick,
                operation,
            };
            self.generated.push_back(entry);
            if generation.plan.remaining() == 0 {
                self.generation = None;
            }
            Some(entry)
        } else {
            None
        };
        self.generation_tick = Some(tick);
        self.observed_tick = Some(tick);
        Ok(generated)
    }

    pub fn send(
        &mut self,
        tick: u64,
        hardware_sync_blocked: bool,
        interface: &mut C220MteL1Interface<C220Mte1ReadUop>,
    ) -> Result<C220Mte1ReadSend, C220Mte1ReadFrontendError> {
        self.check_callback(tick, self.send_tick, "send")?;
        let offered = self.generated.front().copied();
        let mut stall = None;
        let mut queued = None;
        if let Some(head) = offered {
            stall = if head.ready_tick > tick {
                Some(C220Mte1ReadStall::NotReady)
            } else if hardware_sync_blocked {
                Some(C220Mte1ReadStall::HardwareFlag)
            } else {
                queued = interface.push(tick, C220MteL1ReadPort::Port0, head.operation)?;
                if queued.is_some() {
                    self.generated.pop_front();
                    None
                } else {
                    Some(C220Mte1ReadStall::OutputFull)
                }
            };
        }
        self.send_tick = Some(tick);
        self.observed_tick = Some(tick);
        Ok(C220Mte1ReadSend {
            tick,
            offered,
            stall,
            queued,
        })
    }

    /// Explicit send-before-generate convenience; shared event owners invoke
    /// the two callbacks separately. Errors are checked before either mutates.
    pub fn step(
        &mut self,
        tick: u64,
        hardware_sync_blocked: bool,
        interface: &mut C220MteL1Interface<C220Mte1ReadUop>,
    ) -> Result<C220Mte1ReadFrontendCycle, C220Mte1ReadFrontendError> {
        self.check_callback(tick, self.send_tick, "send")?;
        self.check_callback(tick, self.generation_tick, "generation")?;
        let port = C220MteL1ReadPort::Port0;
        let dispatch = !hardware_sync_blocked
            && self
                .generated
                .front()
                .is_some_and(|head| head.ready_tick <= tick)
            && interface.can_push(port);
        if dispatch {
            interface.validate_push(tick, port)?;
        }
        let generate = self
            .generation
            .as_ref()
            .is_some_and(|g| g.ready_tick <= tick)
            && self.generated.len() - usize::from(dispatch) < GENERATED_CAPACITY;
        if generate {
            tick.checked_add(GENERATED_TICKS)
                .ok_or(C220Mte1ReadFrontendError::TimeOverflow)?;
        }
        let queued = self.send(tick, hardware_sync_blocked, interface)?.queued;
        let generated = self.generate(tick)?.map(|entry| entry.operation);
        Ok(C220Mte1ReadFrontendCycle {
            tick,
            generated,
            queued,
            queues: self.queue_state(),
        })
    }

    fn check_time(&self, tick: u64) -> Result<(), C220Mte1ReadFrontendError> {
        if let Some(previous) = self.observed_tick
            && tick < previous
        {
            return Err(C220Mte1ReadFrontendError::TimeReversed {
                previous,
                requested: tick,
            });
        }
        Ok(())
    }

    fn check_callback(
        &self,
        tick: u64,
        previous: Option<u64>,
        phase: &'static str,
    ) -> Result<(), C220Mte1ReadFrontendError> {
        self.check_time(tick)?;
        if previous == Some(tick) {
            return Err(C220Mte1ReadFrontendError::RepeatedCallback { phase, tick });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
