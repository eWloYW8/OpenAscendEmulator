use crate::isa::c220::hflag::C220HardwareFlagStep;
use crate::isa::flow::FlagStep;
use crate::sim::c220::cube::C220CubeIssue;
use crate::sim::c220::mte::mte1::{C220Mte1Command, C220Mte1Issue};
use crate::sim::c220::mte::mte2::C220Mte2Issue;
use crate::sim::c220::mte::mte3::C220Mte3Ticket;
use crate::sim::c220::mte::mte3::C220OutputStep;
use crate::sim::c220::scalar::timing::C220ScalarTimingTicket;
use crate::sim::common::scalar::ScalarProgramStep;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum C220CoreInstruction {
    WaitDeviceFlag {
        instruction_id: u64,
        pc: u64,
        instruction: crate::isa::c220::control::C220WaitDeviceFlagInstruction,
        flag_id: u32,
        remaining: u32,
    },
    Mte3CrossCore(crate::sim::c220::mte::mte3::frontend::C220Mte3Record),
    Mte3Queued(super::C220Mte3IssuedInstruction),
    Mte3Flag(FlagStep),
    Mte3Barrier {
        barrier: super::C220Mte3Barrier,
        completed_tick: Option<u64>,
    },
    CrossCore(crate::sim::c220::sync::C220CrossCoreReception),
    Preload(super::C220CorePreloadIssue),
    AtomicStore(super::C220CoreAtomicIssue),
    Load(super::C220CoreLoadIssue),
    Store(super::C220CoreStoreIssue),
    DirectStore(super::C220CoreLsuIssue),
    Maintenance(super::C220CoreMaintenanceIssue),
    FixpBarrier {
        barrier: super::C220FixpBarrier,
        completed_tick: Option<u64>,
    },
    FixpFlag(FlagStep),
    FixpQueued {
        instruction_id: u64,
        pc: u64,
        word: u32,
        /// Earliest transfer from the issue queue to the command scheduler.
        ready_tick: u64,
    },
    FixpCrossCoreDispatched {
        instruction_id: u64,
    },
    FixpScheduled {
        instruction_id: u64,
        /// Earliest command dispatch; resource dependencies may delay it.
        ready_tick: u64,
    },
    FixpExternal {
        instruction_id: u64,
        pc: u64,
        word: u32,
        command: crate::sim::c220::mte::fixp::C220FixpExternalCommand,
        destination: crate::isa::c220::mte::fixp::C220FixpDestination,
        admission: crate::sim::c220::mte::fixp::C220FixpAdmission,
    },
    Factor {
        instruction_id: u64,
        pc: u64,
        word: u32,
        load: crate::isa::c220::mte::factor::C220FactorLoad,
        admission: crate::sim::c220::mte::fixp::C220FixpAdmission,
    },
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
        spr_timing: Option<crate::sim::c220::scalar::spr::C220ScalarSprTimingTicket>,
    },
    Barrier(ScalarProgramStep),
    Mte1Queued(super::C220Mte1IssuedInstruction),
    Mte1Scheduled(super::C220Mte1QueuedCommand),
    Mte1 {
        instruction_id: u64,
        pc: u64,
        command: C220Mte1Command,
        issue: C220Mte1Issue,
    },
    Mte1Flag(FlagStep),
    Mte1Barrier {
        barrier: super::C220Mte1Barrier,
        completed_tick: Option<u64>,
    },
    HardwareFlag {
        instruction_id: u64,
        step: C220HardwareFlagStep,
        token_ready_tick: Option<u64>,
    },
    Mte2(C220Mte2Issue),
    Mte2Flag(FlagStep),
    Mte2Barrier {
        barrier: super::C220Mte2Barrier,
        completed_tick: Option<u64>,
    },
    Mte2Queued(super::C220Mte2IssuedInstruction),
    Mte2Scheduled(super::C220Mte2QueuedCommand),
    Cube(C220CubeIssue),
    CubeFlag(FlagStep),
    CubeQueued(crate::sim::c220::cube::frontend::C220CubeQueuedCommand),
    CubeBarrier {
        barrier: crate::sim::c220::cube::frontend::C220CubeBarrier,
        completed_tick: Option<u64>,
    },
    CubeSpr {
        instruction_id: u64,
        step: crate::sim::common::scalar::ScalarSprStep,
    },
    VectorQueued(crate::sim::c220::vector::C220VectorQueuedInstruction),
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
