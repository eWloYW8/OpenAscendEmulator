use super::frontend::{C220Mte1ReadIssue, C220Mte1ReadKind, C220Mte1ReadTransfer};
use crate::isa::c220::mte::set2d::C220Set2dFill;
use crate::sim::c220::mte::set2d::C220Set2dIssue;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220Mte1Generator {
    Load3d,
    Read(C220Mte1ReadKind),
    Set2d,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220Mte1Command {
    WriteSpr(crate::sim::common::scalar::ScalarSprStep),
    CrossCore {
        instruction: crate::isa::c220::control::C220SetCrossCoreInstruction,
        payload: crate::sim::c220::sync::C220DeviceSync,
    },
    Read(C220Mte1ReadTransfer),
    Set2d(C220Set2dFill),
}

impl C220Mte1Command {
    pub const fn generator(self) -> Option<C220Mte1Generator> {
        match self {
            Self::WriteSpr(_) => None,
            Self::CrossCore { .. } => Some(C220Mte1Generator::Load3d),
            Self::Read(transfer) => Some(C220Mte1Generator::Read(transfer.kind())),
            Self::Set2d(_) => Some(C220Mte1Generator::Set2d),
        }
    }

    /// Disabled commands bypass generation and leave the selected engine intact.
    pub const fn is_disabled(self) -> bool {
        match self {
            Self::CrossCore { .. } | Self::WriteSpr(_) => false,
            Self::Read(transfer) => transfer.is_empty(),
            Self::Set2d(fill) => fill.descriptor.is_disabled(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Mte1Issue {
    pub tick: u64,
    pub instruction_id: u64,
    pub uop_count: u64,
    pub completion_ready: bool,
}

impl From<C220Mte1ReadIssue> for C220Mte1Issue {
    fn from(issue: C220Mte1ReadIssue) -> Self {
        Self {
            tick: issue.tick,
            instruction_id: issue.instruction_id,
            uop_count: issue.request_count,
            completion_ready: issue.completion_ready,
        }
    }
}

impl From<C220Set2dIssue> for C220Mte1Issue {
    fn from(issue: C220Set2dIssue) -> Self {
        Self {
            tick: issue.tick,
            instruction_id: issue.instruction_id,
            uop_count: issue.uop_count,
            completion_ready: issue.completion_ready,
        }
    }
}
