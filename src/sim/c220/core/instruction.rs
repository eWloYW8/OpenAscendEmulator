use crate::isa::c220::hflag::C220HardwareFlagStep;
use crate::isa::flow::FlagStep;
use crate::sim::c220::cube::C220CubeIssue;
use crate::sim::c220::mte::mte1::{C220Mte1Command, C220Mte1Issue};
use crate::sim::c220::mte::mte2::C220Mte2Issue;
use crate::sim::c220::mte::mte3::C220Mte3Ticket;
use crate::sim::c220::mte::mte3::C220OutputStep;
use crate::sim::c220::scalar::timing::C220ScalarTimingTicket;
use crate::sim::c220::vector::C220VectorInstruction;
use crate::sim::common::scalar::ScalarProgramStep;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum C220CoreInstruction {
    Fixp {
        instruction_id: u64,
        pc: u64,
        word: u32,
        command: crate::sim::c220::mte::fixp::C220FixpCommand,
        admission: crate::sim::c220::mte::fixp::C220FixpAdmission,
    },
    Scalar {
        step: ScalarProgramStep,
        timing: Option<C220ScalarTimingTicket>,
    },
    Barrier(ScalarProgramStep),
    Mte1 {
        instruction_id: u64,
        pc: u64,
        command: C220Mte1Command,
        issue: C220Mte1Issue,
    },
    Mte1Flag(FlagStep),
    HardwareFlag {
        instruction_id: u64,
        step: C220HardwareFlagStep,
        token_ready_tick: Option<u64>,
    },
    Mte2(C220Mte2Issue),
    Mte2Flag(FlagStep),
    Cube(C220CubeIssue),
    Vector(C220VectorInstruction),
    VectorToScalarFlag(FlagStep),
    Mte3 {
        step: C220OutputStep,
        ticket: Option<C220Mte3Ticket>,
    },
    Mte3Dma {
        step: C220OutputStep,
        record: crate::sim::c220::mte::mte3::frontend::C220Mte3Record,
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
