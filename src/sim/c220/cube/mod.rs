mod execute;
mod fsm_v0;
mod fsm_v1;
pub mod timing;
mod uop;

use crate::isa::c220::cube::{C220CubeInstruction, C220CubeRegisterValues, C220MmadParameters};

pub use timing::{
    C220CubeConfig, C220CubeFsmVersion, C220CubePipeline, C220CubeTicket, C220CubeTimingError,
};

pub(crate) use execute::update_cube_status_spr2;
pub use execute::{
    C220CubeControl, C220CubeExecutionError, C220CubeExecutionOutcome, C220CubeFpStatus,
    C220F32MmadMode, C220PreparedCubeExecution,
};
pub use fsm_v0::C220CubeV0UopPlanner;
pub use fsm_v1::C220CubeV1UopPlanner;
pub use uop::{
    C220CubeL0cAccess, C220CubeL0cRequest, C220CubeTileIndices, C220CubeUop, C220CubeUopRelease,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220CubeTimeline {
    ticket: C220CubeTicket,
    uops: C220CubeUopPlanner,
}

impl C220CubeTimeline {
    fn new(issue: C220CubeIssue) -> Self {
        Self {
            ticket: issue.ticket,
            uops: issue.uops(),
        }
    }
}

impl Iterator for C220CubeTimeline {
    type Item = C220CubeUopRelease;

    fn next(&mut self) -> Option<Self::Item> {
        let uop = self.uops.next()?;
        let issue_tick = self
            .ticket
            .uop_issue_tick(uop.id)
            .expect("planner only emits scheduled Cube uops");
        Some(C220CubeUopRelease { uop, issue_tick })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.uops.size_hint()
    }
}

impl ExactSizeIterator for C220CubeTimeline {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220CubeIssue {
    pub pc: u64,
    pub word: u32,
    pub instruction: C220CubeInstruction,
    pub registers: C220CubeRegisterValues,
    pub parameters: C220MmadParameters,
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

    pub fn timeline(self) -> C220CubeTimeline {
        C220CubeTimeline::new(self)
    }
}
