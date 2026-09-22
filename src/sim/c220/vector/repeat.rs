use super::C220VectorControl;
use super::read::C220VectorReadIssue;
use crate::isa::c220::vector::C220VecArithmeticOperation;

#[derive(Debug, Clone, Copy)]
pub(super) struct OrdinaryReadSchedule {
    reuse_input: bool,
    reads_second_source: bool,
}

impl OrdinaryReadSchedule {
    pub(super) fn from_issue(issue: C220VectorReadIssue<'_>) -> Option<Self> {
        let (control, second_source, same_sources, feedback) = match issue {
            C220VectorReadIssue::Arithmetic(issue) => {
                let control = issue.control;
                let second_source = issue.hint.source_1_register.is_some();
                let feedback = matches!(
                    issue.hint.operation,
                    C220VecArithmeticOperation::Add
                        | C220VecArithmeticOperation::Subtract
                        | C220VecArithmeticOperation::Multiply
                        | C220VecArithmeticOperation::Maximum
                        | C220VecArithmeticOperation::Minimum
                ) && (control.destination_repeat_stride == 0
                    || issue.iteration_masks.len() == 1)
                    && issue.addresses.source_1 == issue.addresses.destination
                    && control.source_1_block_stride.max(1)
                        == control.destination_block_stride.max(1);
                let same_sources = second_source
                    && issue.addresses.source_0 == issue.addresses.source_1
                    && same_source_strides(control, issue.iteration_masks.len());
                (control, second_source, same_sources, feedback)
            }
            C220VectorReadIssue::Fused(issue) => (
                issue.control,
                true,
                issue.addresses.source_0 == issue.addresses.source_1
                    && same_source_strides(issue.control, issue.iteration_masks.len()),
                false,
            ),
            C220VectorReadIssue::VectorScalar(issue) => (issue.control, false, false, false),
            C220VectorReadIssue::Shift(issue) => (issue.control, false, false, false),
            C220VectorReadIssue::SpecialUnary(issue) => (issue.control, false, false, false),
            C220VectorReadIssue::Conversion(issue) => (issue.control, false, false, false),
            _ => return None,
        };
        Some(Self {
            reuse_input: control.source_0_repeat_stride == 0 && !feedback,
            reads_second_source: second_source && !same_sources,
        })
    }

    pub(super) fn reads_source(self, repeat: usize, source: u8) -> bool {
        match source {
            0 => repeat == 0 || !self.reuse_input,
            1 => self.reads_second_source,
            _ => false,
        }
    }
}

fn same_source_strides(control: C220VectorControl, repeats: usize) -> bool {
    control.source_0_block_stride.max(1) == control.source_1_block_stride.max(1)
        && (repeats <= 1 || control.source_0_repeat_stride == control.source_1_repeat_stride)
}

#[derive(Debug, Clone, Copy)]
pub(super) struct AccumulatorSchedule {
    pub repeat_count: usize,
    pub lane_count: usize,
    ternary: bool,
    reuse_destination: bool,
    reuse_input: bool,
}

impl AccumulatorSchedule {
    pub(super) fn new(
        control: C220VectorControl,
        repeat_count: usize,
        lane_count: usize,
        ternary: bool,
    ) -> Self {
        Self {
            repeat_count,
            lane_count,
            ternary,
            reuse_destination: control.destination_repeat_stride == 0 || repeat_count == 1,
            reuse_input: control.source_0_repeat_stride == 0,
        }
    }

    pub(super) fn from_issue(issue: C220VectorReadIssue<'_>) -> Option<Self> {
        match issue {
            C220VectorReadIssue::Ternary(issue) => Some(Self::new(
                issue.control,
                issue.iteration_masks.len(),
                issue.instruction.width.lane_count(),
                true,
            )),
            C220VectorReadIssue::Axpy(issue) => Some(Self::new(
                issue.control,
                issue.iteration_masks.len(),
                issue.instruction.width.lane_count(),
                false,
            )),
            _ => None,
        }
    }

    pub(super) fn lanes_per_uop(self, repeat: usize) -> usize {
        if self.ternary && !(self.reuse_destination && repeat > 0) {
            self.lane_count / 2
        } else {
            self.lane_count
        }
    }

    pub(super) fn reads_source(self, repeat: usize, source: u8) -> bool {
        repeat == 0
            || match source {
                0 => !self.reuse_input,
                2 => !self.reuse_destination,
                _ => true,
            }
    }

    pub(super) fn source_ports(self, repeat: usize) -> [usize; 3] {
        [0, usize::from(self.reuse_destination && repeat > 0), 1]
    }

    pub(super) fn writes_destination(self, repeat: usize) -> bool {
        !self.reuse_destination || repeat + 1 == self.repeat_count
    }

    pub(super) fn has_traffic(self, repeat: usize) -> bool {
        self.ternary
            || self.reads_source(repeat, 0)
            || self.reads_source(repeat, 2)
            || self.writes_destination(repeat)
    }

    pub(super) fn issue_gap(self, repeat: usize, first_lane: usize, lanes: usize) -> u64 {
        if self.reuse_destination
            && repeat + 1 < self.repeat_count
            && first_lane + lanes >= self.lane_count
        {
            8
        } else {
            1
        }
    }
}
