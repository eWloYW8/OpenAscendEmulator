use super::{C220Mte2L1TransferPlan, C220Mte2Ticket, C220Mte2TransferPlan};
use crate::isa::c220::mte::set2d::C220Set2dFill;
use crate::memory::ub::UbTransferResult;
use crate::sim::c220::mte::out_to_l1::C220L1DmaResult;
use crate::sim::c220::mte::set2d::{C220Set2dIssue, C220Set2dResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220Mte2Command {
    MovOutToUb(C220Mte2TransferPlan),
    MovOutToL1(C220Mte2L1TransferPlan),
    Set2d(C220Set2dFill),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220Mte2IssueTiming {
    /// Zero burst count or length; no generator, memory traffic or rate estimate.
    Disabled,
    /// Requests traverse the generator; completion must come from the consumer.
    Dma(crate::sim::c220::mte::dma::C220DmaIssue),
    /// Completion is observed from the shared L1 write path.
    L1(C220Set2dIssue),
    /// Completion uses caller-supplied rates until a physical DMA path is available.
    AggregateDma(C220Mte2Ticket),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Mte2Issue {
    pub instruction_id: u64,
    pub pc: u64,
    pub command: C220Mte2Command,
    pub timing: C220Mte2IssueTiming,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220Mte2Completion {
    AwaitingDestination,
    AwaitingDma { tail_delivered: bool },
    Observed { tick: u64 },
    Estimated { retire_tick: u64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Mte2CommandState {
    pub instruction_id: u64,
    pub pc: u64,
    pub issue_tick: u64,
    pub command: C220Mte2Command,
    pub completion: C220Mte2Completion,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Mte2EventState {
    pub destination_pipe: u8,
    pub event_id: u32,
    pub dependency: Option<u64>,
    pub ready: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum C220Mte2Result {
    MovOutToUb(UbTransferResult),
    MovOutToL1(C220L1DmaResult),
    Set2d(C220Set2dResult),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220Mte2Outcome {
    pub command: C220Mte2CommandState,
    pub retire_tick: u64,
    pub result: C220Mte2Result,
}
