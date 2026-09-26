pub(super) mod runtime;
pub use runtime::C220CubeRuntimeError;
mod accumulator;
pub use accumulator::C220CubeAccumulatorSource;
mod control;
mod execute;
pub mod frontend;
mod fsm_v0;
mod fsm_v1;
mod layout;
mod mmad;
mod numeric;
pub mod sparse;
pub(super) mod spr;
#[cfg(test)]
mod tests;
pub mod timing;
mod uop;

use crate::isa::c220::cube::{C220CubeInstruction, C220CubeRegisterValues, C220MmadParameters};

pub use timing::{
    C220CubeConfig, C220CubeFsmVersion, C220CubeL0cStall, C220CubeL0cStallReason, C220CubePipeline,
    C220CubeResourceWaits, C220CubeTicket, C220CubeTimingError, C220CubeV1FrameOrder,
};

pub use control::{
    C220CubeExecutionControl, C220CubeIssueDelay, C220CubeTimingControl, C220F32MmadMode,
};
pub(crate) use execute::update_cube_status_spr2;
pub use execute::{C220CubeExecutionError, C220CubeExecutionOutcome, C220PreparedCubeExecution};
pub use fsm_v0::C220CubeV0UopPlanner;
pub use fsm_v1::C220CubeV1UopPlanner;
pub use numeric::C220CubeFpStatus;
pub use uop::{
    C220CubeL0cAccess, C220CubeL0cRequest, C220CubeTileIndices, C220CubeUnitFlagMode, C220CubeUop,
    C220CubeUopRelease,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum C220CubeUopPlanner {
    V0(C220CubeV0UopPlanner),
    V1(C220CubeV1UopPlanner),
}

impl Iterator for C220CubeUopPlanner {
    type Item = C220CubeUop;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::V0(planner) => planner.next(),
            Self::V1(planner) => planner.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::V0(planner) => planner.size_hint(),
            Self::V1(planner) => planner.size_hint(),
        }
    }
}

impl ExactSizeIterator for C220CubeUopPlanner {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220CubeIssue {
    pub instruction_id: u64,
    pub pc: u64,
    pub word: u32,
    pub instruction: C220CubeInstruction,
    pub registers: C220CubeRegisterValues,
    pub parameters: C220MmadParameters,
    /// Numerical modes captured when the instruction is admitted.
    pub execution_control: C220CubeExecutionControl,
    pub ticket: C220CubeTicket,
}

impl C220CubeIssue {
    pub fn uops(self) -> C220CubeUopPlanner {
        match self.ticket.fsm_version {
            C220CubeFsmVersion::V0 => C220CubeUopPlanner::V0(C220CubeV0UopPlanner::new(
                self.ticket,
                self.instruction,
                self.parameters,
            )),
            C220CubeFsmVersion::V1 => C220CubeUopPlanner::V1(C220CubeV1UopPlanner::new(
                self.ticket,
                self.instruction,
                self.parameters,
            )),
        }
    }
}
