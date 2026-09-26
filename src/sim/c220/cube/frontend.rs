use std::num::NonZeroU32;

use crate::isa::c220::control::C220SetCrossCoreInstruction;
use crate::isa::c220::cube::{C220CubeInstruction, C220CubeRegisterValues};
use crate::isa::c220::hflag::C220HardwareFlagStep;
use crate::isa::flow::{FlagStep, PipelineBarrierStep};
use crate::sim::c220::sync::C220DeviceSync;
use crate::sim::common::scalar::ScalarSprStep;

use super::{C220CubeExecutionControl, C220CubeTimingControl};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220CubeFrontendConfig {
    /// Commands waiting to enter the execution pipeline.
    pub queue_depth: NonZeroU32,
    /// Received commands whose retirement has not completed.
    pub outstanding_limit: NonZeroU32,
}

impl Default for C220CubeFrontendConfig {
    fn default() -> Self {
        Self {
            queue_depth: NonZeroU32::new(16).unwrap(),
            outstanding_limit: NonZeroU32::new(15).unwrap(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220CubeCommand {
    Mmad {
        instruction: C220CubeInstruction,
        registers: C220CubeRegisterValues,
        execution_control: C220CubeExecutionControl,
        timing_control: C220CubeTimingControl,
    },
    WriteSpr(ScalarSprStep),
    HardwareFlag(C220HardwareFlagStep),
    Flag(FlagStep),
    CrossCore {
        instruction: C220SetCrossCoreInstruction,
        payload: C220DeviceSync,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220CubeQueuedCommand {
    pub instruction_id: u64,
    pub pc: u64,
    pub word: u32,
    /// Scalar-side admission, when source operands are captured.
    pub accepted_tick: u64,
    /// Earliest reception; dependencies can delay the actual reception.
    pub ready_tick: u64,
    pub command: C220CubeCommand,
}

/// An ordering marker that occupies neither a queue slot nor a retirement credit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220CubeBarrier {
    pub instruction_id: u64,
    pub step: PipelineBarrierStep,
    pub predecessor: Option<u64>,
    pub issued_tick: u64,
    /// A queue-handled event fence also waits for earlier running instructions.
    pub requires_idle: bool,
}
