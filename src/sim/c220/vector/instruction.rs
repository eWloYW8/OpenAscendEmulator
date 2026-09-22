use super::C220VectorStore;
use super::pipeline::C220VectorQueueClass;
use crate::isa::c220::control::C220VectorControlInstruction;
use crate::isa::c220::vector::C220MoveVaInstruction;
use crate::isa::c220::vector::no_effect::C220NoEffectVectorInstruction;
use crate::sim::c220::scalar::C220MovemaskStep;
use crate::sim::c220::vector::ops::axpy::C220AxpyIssue;
use crate::sim::c220::vector::ops::broadcast::C220BroadcastIssue;
use crate::sim::c220::vector::ops::compare::{
    C220CompareMaskIssue, C220MoveMaskIssue, C220PackedCompareIssue,
};
use crate::sim::c220::vector::ops::conversion::C220ConversionIssue;
use crate::sim::c220::vector::ops::copy::C220CopyIssue;
use crate::sim::c220::vector::ops::fused::C220FusedIssue;
use crate::sim::c220::vector::ops::gather::C220GatherIssue;
use crate::sim::c220::vector::ops::merge::C220MergeIssue;
use crate::sim::c220::vector::ops::nchw::C220NchwIssue;
use crate::sim::c220::vector::ops::reduce::C220ReductionIssue;
use crate::sim::c220::vector::ops::scalar::C220VectorScalarIssue;
use crate::sim::c220::vector::ops::select::C220SelectIssue;
use crate::sim::c220::vector::ops::shift::C220ShiftIssue;
use crate::sim::c220::vector::ops::sort::C220SortIssue;
use crate::sim::c220::vector::ops::special::C220SpecialUnaryIssue;
use crate::sim::c220::vector::ops::ternary::C220TernaryIssue;
use crate::sim::c220::vector::ops::transpose::C220TransposeIssue;
use crate::sim::c220::vector::va::C220LoadVaIssue;
use crate::sim::c220::vector::{C220MovevStep, C220VectorArithmeticIssue};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum C220VectorInstruction {
    MoveAddress {
        pc: u64,
        word: u32,
        instruction: C220MoveVaInstruction,
    },
    Control {
        pc: u64,
        word: u32,
        instruction: C220VectorControlInstruction,
    },
    NoEffect {
        pc: u64,
        word: u32,
        instruction: C220NoEffectVectorInstruction,
        repeat_count: usize,
        lane_groups: u8,
    },
    LoadAddress(C220LoadVaIssue),
    Movemask(C220MovemaskStep),
    Move(C220MovevStep),
    Arithmetic(C220VectorArithmeticIssue),
    Scalar(C220VectorScalarIssue),
    Shift(C220ShiftIssue),
    Copy(C220CopyIssue),
    Broadcast(C220BroadcastIssue),
    Transpose(C220TransposeIssue),
    CompareMask(C220CompareMaskIssue),
    MoveMask(C220MoveMaskIssue),
    Select(C220SelectIssue),
    PackedCompare(C220PackedCompareIssue),
    Reduction(C220ReductionIssue),
    Sort(C220SortIssue),
    Ternary(C220TernaryIssue),
    Axpy(C220AxpyIssue),
    SpecialUnary(C220SpecialUnaryIssue),
    Conversion(C220ConversionIssue),
    Fused(C220FusedIssue),
    Gather(C220GatherIssue),
    Nchw(C220NchwIssue),
    Merge(C220MergeIssue),
}

impl C220VectorInstruction {
    pub(crate) fn read_issue(&self) -> Option<super::read::C220VectorReadIssue<'_>> {
        use super::read::C220VectorReadIssue;
        match self {
            C220VectorInstruction::LoadAddress(issue) => Some(C220VectorReadIssue::LoadVa(issue)),
            C220VectorInstruction::Arithmetic(issue) => {
                Some(C220VectorReadIssue::Arithmetic(issue))
            }
            C220VectorInstruction::Scalar(issue) => Some(C220VectorReadIssue::VectorScalar(issue)),
            C220VectorInstruction::Shift(issue) => Some(C220VectorReadIssue::Shift(issue)),
            C220VectorInstruction::Copy(issue) => Some(C220VectorReadIssue::Copy(issue)),
            C220VectorInstruction::Broadcast(issue) => Some(C220VectorReadIssue::Broadcast(issue)),
            C220VectorInstruction::Transpose(issue) => Some(C220VectorReadIssue::Transpose(issue)),
            C220VectorInstruction::CompareMask(issue) => {
                Some(C220VectorReadIssue::CompareMask(issue))
            }
            C220VectorInstruction::MoveMask(issue) => Some(C220VectorReadIssue::MoveMask(issue)),
            C220VectorInstruction::Select(issue) => Some(C220VectorReadIssue::Select(issue)),
            C220VectorInstruction::PackedCompare(issue) => {
                Some(C220VectorReadIssue::PackedCompare(issue))
            }
            C220VectorInstruction::Reduction(issue) => Some(C220VectorReadIssue::Reduction(issue)),
            C220VectorInstruction::Sort(issue) => Some(C220VectorReadIssue::Sort(issue)),
            C220VectorInstruction::Ternary(issue) => Some(C220VectorReadIssue::Ternary(issue)),
            C220VectorInstruction::Axpy(issue) => Some(C220VectorReadIssue::Axpy(issue)),
            C220VectorInstruction::SpecialUnary(issue) => {
                Some(C220VectorReadIssue::SpecialUnary(issue))
            }
            C220VectorInstruction::Conversion(issue) => {
                Some(C220VectorReadIssue::Conversion(issue))
            }
            C220VectorInstruction::Fused(issue) => Some(C220VectorReadIssue::Fused(issue)),
            C220VectorInstruction::Gather(issue) => Some(C220VectorReadIssue::Gather(issue)),
            C220VectorInstruction::Nchw(issue) => Some(C220VectorReadIssue::Nchw(issue)),
            Self::MoveAddress { .. }
            | Self::Control { .. }
            | Self::NoEffect { .. }
            | Self::Movemask(_)
            | Self::Move(_)
            | Self::Merge(_) => None,
        }
    }
}

impl C220VectorInstruction {
    pub(crate) fn queue_class(&self) -> C220VectorQueueClass {
        match self {
            Self::NoEffect { instruction, .. } => {
                C220VectorQueueClass::from_no_effect(instruction.operation)
            }
            _ => C220VectorQueueClass::from_compute(self.read_issue()),
        }
    }

    pub(crate) fn stores(&self) -> &[C220VectorStore] {
        match self {
            Self::MoveAddress { .. }
            | Self::Control { .. }
            | Self::NoEffect { .. }
            | Self::LoadAddress(_)
            | Self::Movemask(_) => &[],
            Self::Move(step) => &step.stores,
            Self::Arithmetic(step) => &step.write_targets,
            Self::Scalar(step) => &step.write_targets,
            Self::Shift(step) => &step.write_targets,
            Self::Copy(step) => &step.write_targets,
            Self::Broadcast(step) => &step.write_targets,
            Self::Transpose(step) => &step.write_targets,
            Self::CompareMask(_) => &[],
            Self::MoveMask(step) => &step.write_targets,
            Self::Select(step) => &step.write_targets,
            Self::PackedCompare(step) => &step.write_targets,
            Self::Reduction(step) => &step.write_targets,
            Self::Sort(step) => &step.write_targets,
            Self::Ternary(step) => &step.write_targets,
            Self::Axpy(step) => &step.write_targets,
            Self::SpecialUnary(step) => &step.write_targets,
            Self::Conversion(step) => &step.write_targets,
            Self::Fused(step) => &step.write_targets,
            Self::Gather(step) => &step.write_targets,
            Self::Nchw(step) => &step.write_targets,
            Self::Merge(_) => &[],
        }
    }
}
