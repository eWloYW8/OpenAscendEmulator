use super::{C220Mte2L1TransferPlan, C220Mte2Ticket, C220Mte2TransferPlan};
use crate::isa::c220::mte::set2d::C220Set2dFill;
use crate::memory::ub::UbTransferResult;
use crate::sim::c220::mte::out_to_l1::C220L1DmaResult;
use crate::sim::c220::mte::set2d::{C220Set2dIssue, C220Set2dResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220Mte2Command {
    MovPad(crate::sim::c220::mte::mov_pad::C220MovPadCommand),
    Load2d {
        transfer: crate::isa::c220::mte::load2d::C220Load2dTransfer,
        mode: crate::sim::c220::mte::uop::C220DmaUopMode,
    },
    WriteSpr(crate::sim::common::scalar::ScalarSprStep),
    CrossCore {
        instruction: crate::isa::c220::control::C220SetCrossCoreInstruction,
        payload: crate::sim::c220::sync::C220DeviceSync,
    },
    MovOutToUb(C220Mte2TransferPlan),
    MovOutToL1(C220Mte2L1TransferPlan),
    MovOutToSmask(crate::isa::c220::mte::smask::C220SmaskTransfer),
    Set2d(C220Set2dFill),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220Mte2IssueTiming {
    WriteSpr {
        dispatch_tick: u64,
    },
    /// Zero-uop notification dispatched through the selected generator.
    CrossCore {
        dispatch_tick: u64,
    },
    /// Zero burst count or length; no generator, memory traffic or rate estimate.
    Disabled,
    /// Requests traverse the generator; completion must come from the consumer.
    Dma(crate::sim::c220::mte::dma::C220DmaIssue),
    /// Completion is observed from the shared L1 write path.
    L1(C220Set2dIssue),
    /// Completion is observed from the shared L1 read output path.
    Read(crate::sim::c220::mte::read::C220MteReadIssue),
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum C220Mte2Result {
    MovPad(UbTransferResult),
    Load2d(crate::sim::c220::mte::load2d::C220Load2dTransferResult),
    WriteSpr(crate::sim::common::scalar::ScalarSprStep),
    CrossCore(crate::sim::c220::sync::C220DeviceSync),
    MovOutToUb(UbTransferResult),
    MovOutToL1(C220L1DmaResult),
    MovOutToSmask(crate::sim::c220::mte::smask::C220SmaskTransferResult),
    Set2d(C220Set2dResult),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220Mte2Outcome {
    pub command: C220Mte2CommandState,
    pub retire_tick: u64,
    pub result: C220Mte2Result,
}

impl C220Mte2Outcome {
    pub fn cross_core_reception(&self) -> Option<crate::sim::c220::sync::C220CrossCoreReception> {
        let C220Mte2Command::CrossCore { instruction, .. } = self.command.command else {
            return None;
        };
        let C220Mte2Result::CrossCore(payload) = self.result else {
            return None;
        };
        Some(crate::sim::c220::sync::C220CrossCoreReception {
            instruction_id: self.command.instruction_id,
            pc: self.command.pc,
            tick: self.retire_tick,
            instruction,
            payload,
        })
    }
}
