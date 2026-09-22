use crate::isa::c220::hflag::C220HardwareFlagStep;
use crate::isa::c220::mte::load2d::C220Load2dTransfer;
use crate::isa::flow::FlagStep;
use crate::sim::c220::cube::C220CubeIssue;
use crate::sim::c220::mte::mte1::C220Mte1Ticket;
use crate::sim::c220::mte::mte2::C220Mte2Step;
use crate::sim::c220::mte::mte3::C220Mte3Ticket;
use crate::sim::c220::mte::mte3::C220OutputStep;
use crate::sim::c220::mte::uop::C220DmaUopRequest;
use crate::sim::c220::scalar::timing::C220ScalarTimingTicket;
use crate::sim::c220::vector::C220VectorInstruction;
use crate::sim::common::scalar::ScalarProgramStep;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum C220CoreInstruction {
    Scalar {
        step: ScalarProgramStep,
        timing: Option<C220ScalarTimingTicket>,
    },
    Barrier(ScalarProgramStep),
    Mte1Load2d {
        instruction_id: u64,
        pc: u64,
        transfer: C220Load2dTransfer,
        ticket: Box<C220Mte1Ticket>,
    },
    Mte1Flag(FlagStep),
    HardwareFlag {
        instruction_id: u64,
        step: C220HardwareFlagStep,
        token_ready_tick: Option<u64>,
    },
    Mte2(C220Mte2Step),
    Cube(C220CubeIssue),
    Vector(C220VectorInstruction),
    VectorToScalarFlag(FlagStep),
    Mte3 {
        step: C220OutputStep,
        requests: Vec<C220DmaUopRequest>,
        ticket: Option<C220Mte3Ticket>,
    },
}

impl C220CoreInstruction {
    pub const fn as_vector(&self) -> Option<&C220VectorInstruction> {
        match self {
            Self::Vector(instruction) => Some(instruction),
            _ => None,
        }
    }
}
