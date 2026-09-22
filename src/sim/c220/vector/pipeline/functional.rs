use super::{C220VectorAdvanceError, C220VectorPipeline};
use crate::isa::c220::vector::conversion::C220ConversionKind;
use crate::sim::c220::state::C220State;
use crate::sim::c220::vector::C220_VECTOR_TILE_BYTES;
use crate::sim::c220::vector::ops::{
    arithmetic::C220VectorArithmeticIssue, axpy::C220AxpyIssue, conversion::C220ConversionIssue,
    copy::C220CopyIssue, fused::C220FusedIssue, scalar::C220VectorScalarIssue,
    shift::C220ShiftIssue, special::C220SpecialUnaryIssue, ternary::C220TernaryIssue,
};
use crate::sim::c220::vector::read::{C220VectorReadIssue, PendingVectorRead};
use crate::sim::c220::vector::timing::C220VectorUopKind;

#[derive(Debug, Clone)]
pub(super) enum FunctionalInstruction {
    Arithmetic(Box<C220VectorArithmeticIssue>),
    Scalar(Box<C220VectorScalarIssue>),
    Shift(Box<C220ShiftIssue>),
    Copy(Box<C220CopyIssue>),
    Special(Box<C220SpecialUnaryIssue>),
    Ternary(Box<C220TernaryIssue>),
    Axpy(Box<C220AxpyIssue>),
    Conversion(Box<C220ConversionIssue>),
    Fused(Box<C220FusedIssue>),
}

impl FunctionalInstruction {
    pub(super) fn from_issue(issue: C220VectorReadIssue<'_>) -> Option<Self> {
        match issue {
            C220VectorReadIssue::Arithmetic(issue) => {
                Some(Self::Arithmetic(Box::new(issue.clone())))
            }
            C220VectorReadIssue::VectorScalar(issue) => Some(Self::Scalar(Box::new(issue.clone()))),
            C220VectorReadIssue::Shift(issue) => Some(Self::Shift(Box::new(issue.clone()))),
            C220VectorReadIssue::Copy(issue) => Some(Self::Copy(Box::new(issue.clone()))),
            C220VectorReadIssue::SpecialUnary(issue) => {
                Some(Self::Special(Box::new(issue.clone())))
            }
            C220VectorReadIssue::Ternary(issue) => Some(Self::Ternary(Box::new(issue.clone()))),
            C220VectorReadIssue::Axpy(issue) => Some(Self::Axpy(Box::new(issue.clone()))),
            C220VectorReadIssue::Conversion(issue) => {
                Some(Self::Conversion(Box::new(issue.clone())))
            }
            C220VectorReadIssue::Fused(issue) => Some(Self::Fused(Box::new(issue.clone()))),
            _ => None,
        }
    }

    fn read_issue(&self) -> C220VectorReadIssue<'_> {
        match self {
            Self::Arithmetic(issue) => C220VectorReadIssue::Arithmetic(issue),
            Self::Scalar(issue) => C220VectorReadIssue::VectorScalar(issue),
            Self::Shift(issue) => C220VectorReadIssue::Shift(issue),
            Self::Copy(issue) => C220VectorReadIssue::Copy(issue),
            Self::Special(issue) => C220VectorReadIssue::SpecialUnary(issue),
            Self::Ternary(issue) => C220VectorReadIssue::Ternary(issue),
            Self::Axpy(issue) => C220VectorReadIssue::Axpy(issue),
            Self::Conversion(issue) => C220VectorReadIssue::Conversion(issue),
            Self::Fused(issue) => C220VectorReadIssue::Fused(issue),
        }
    }

    fn prepare_repeat(&mut self, core: &C220State) {
        if let Self::Conversion(issue) = self {
            match issue.instruction.kind {
                C220ConversionKind::VectorDeqS16ToS8 { .. } => {
                    issue.deq_scale = core.scalar().machine().spr_value(12).unwrap_or(0);
                    issue.addresses.source_1 = 32 * (issue.deq_scale & 0x3fff);
                }
                C220ConversionKind::ScalarDeqS16ToS8 { .. }
                | C220ConversionKind::ScalarDeqS32ToF16 => {
                    issue.deq_scale = core.scalar().machine().spr_value(12).unwrap_or(0);
                }
                _ => {}
            }
        }
    }

    fn repeat_shape(&self) -> (usize, usize) {
        let (repeats, element_bytes) = match self {
            Self::Arithmetic(issue) => (issue.iteration_masks.len(), issue.result_element_bytes),
            Self::Scalar(issue) => (
                issue.iteration_masks.len(),
                issue.instruction.dtype.element_bytes(),
            ),
            Self::Shift(issue) => (issue.iteration_masks.len(), issue.instruction.element_bytes),
            Self::Copy(issue) => (issue.iteration_masks.len(), issue.instruction.element_bytes),
            Self::Special(issue) => (
                issue.iteration_masks.len(),
                issue.instruction.width.element_bytes(),
            ),
            Self::Ternary(issue) => (
                issue.iteration_masks.len(),
                issue.instruction.width.destination_element_bytes(),
            ),
            Self::Axpy(issue) => (
                issue.iteration_masks.len(),
                issue.instruction.width.destination_element_bytes(),
            ),
            Self::Conversion(issue) => {
                return (issue.iteration_masks.len(), issue.instruction.lane_count());
            }
            Self::Fused(issue) => {
                return (issue.iteration_masks.len(), issue.instruction.lane_count());
            }
        };
        (repeats, C220_VECTOR_TILE_BYTES / usize::from(element_bytes))
    }
}

impl C220VectorPipeline {
    pub(super) fn complete_functional_instruction(
        &mut self,
        group: u64,
        tick: u64,
        core: &mut C220State,
    ) -> Result<(), C220VectorAdvanceError> {
        let (repeats, lanes) = self.functional_instructions[&group].repeat_shape();
        for repeat in 0..repeats {
            let instruction = self.functional_instructions.get_mut(&group).unwrap();
            instruction.prepare_repeat(core);
            let mut read = PendingVectorRead::new(
                instruction.read_issue(),
                repeat,
                Some(0),
                C220VectorUopKind::LaneSlice {
                    first_lane: 0,
                    lane_count: lanes as u16,
                },
            )?;
            read.capture_functional_inputs(core.ub())?;
            read.set_ready_tick(tick);
            let (sample, stores) = read.sample(core.ub(), self.compare_mask, None)?;
            core.commit_c220_vector_stores(&stores)?;
            self.last_functional_samples.push(sample);
        }
        self.functional_instructions.remove(&group);
        Ok(())
    }
}
