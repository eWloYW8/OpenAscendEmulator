use crate::isa::c220::vector::C220VecArithmeticOperation;
use crate::isa::c220::vector::conversion::C220ConversionKind;
use crate::isa::c220::vector::fused::{C220FusedFormat, C220FusedOperation};
use crate::isa::c220::vector::no_effect::{
    C220NoEffectVectorInstruction, C220NoEffectVectorOperation,
};
use crate::isa::c220::vector::reduce::{C220ReductionKind, C220ReductionWidth};
use crate::isa::c220::vector::sort::C220SortWidth;
use crate::isa::c220::vector::special::C220SpecialUnaryOperation;
use crate::isa::c220::vector::ternary::C220TernaryOperation;
use crate::sim::c220::vector::read::C220VectorReadIssue;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RepeatOpcode {
    Arithmetic(C220VecArithmeticOperation, u8),
    Ternary(
        C220TernaryOperation,
        crate::isa::c220::vector::ternary::C220TernaryWidth,
    ),
    Axpy(crate::isa::c220::vector::axpy::C220AxpyWidth),
}

impl RepeatOpcode {
    pub(super) fn from_compute(compute: C220VectorReadIssue<'_>) -> Option<Self> {
        match compute {
            C220VectorReadIssue::Arithmetic(issue) => Some(Self::Arithmetic(
                issue.hint.operation,
                issue.hint.dtype_selector,
            )),
            C220VectorReadIssue::Ternary(issue) => Some(Self::Ternary(
                issue.instruction.operation,
                issue.instruction.width,
            )),
            C220VectorReadIssue::Axpy(issue) => Some(Self::Axpy(issue.instruction.width)),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum C220VectorIssueVariant {
    Other,
    Fma,
    AddReduction {
        grouped: bool,
        width: C220ReductionWidth,
    },
    ReluConversion(C220FusedFormat),
    Conversion(u16),
    MulConversion,
    Log,
    Exp,
    SlowDivide,
    Sort(C220SortWidth),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum C220VectorQueueClass {
    Ordinary,
    MultiplyAccumulate,
    Axpy,
    WholeAdd,
    GroupAdd,
    WholeExtremum,
    GroupExtremum,
    Vrpac,
    Vms4,
}

impl C220VectorQueueClass {
    pub(crate) fn from_word(word: u32) -> Self {
        if let Some(instruction) = C220NoEffectVectorInstruction::decode(word) {
            return Self::from_no_effect(instruction.operation);
        }
        if let Some(instruction) =
            crate::isa::c220::vector::reduce::C220ReductionInstruction::decode(word)
        {
            return Self::from_reduction(instruction.kind);
        }
        if crate::isa::c220::vector::axpy::C220AxpyInstruction::decode(word).is_some() {
            return Self::Axpy;
        }
        if crate::isa::c220::vector::ternary::C220TernaryInstruction::decode(word).is_some_and(
            |instruction| instruction.operation == C220TernaryOperation::MultiplyAccumulate,
        ) {
            return Self::MultiplyAccumulate;
        }
        Self::Ordinary
    }

    pub(crate) fn from_compute(compute: Option<C220VectorReadIssue<'_>>) -> Self {
        match compute {
            Some(C220VectorReadIssue::Reduction(issue)) => {
                Self::from_reduction(issue.instruction.kind)
            }
            Some(C220VectorReadIssue::Ternary(issue))
                if issue.instruction.operation == C220TernaryOperation::MultiplyAccumulate =>
            {
                Self::MultiplyAccumulate
            }
            Some(C220VectorReadIssue::Axpy(_)) => Self::Axpy,
            _ => Self::Ordinary,
        }
    }

    pub(crate) const fn from_no_effect(operation: C220NoEffectVectorOperation) -> Self {
        match operation {
            C220NoEffectVectorOperation::Vrpac => Self::Vrpac,
            C220NoEffectVectorOperation::Vms4 => Self::Vms4,
            _ => Self::Ordinary,
        }
    }

    const fn from_reduction(kind: C220ReductionKind) -> Self {
        match kind {
            C220ReductionKind::WholeAdd { .. } => Self::WholeAdd,
            C220ReductionKind::GroupAdd => Self::GroupAdd,
            C220ReductionKind::WholeExtremum { .. } => Self::WholeExtremum,
            C220ReductionKind::GroupExtremum { .. } => Self::GroupExtremum,
            C220ReductionKind::PairAdd => Self::Ordinary,
        }
    }

    pub(super) const fn is_blocked_by(self, queued: Self) -> bool {
        if matches!(self, Self::Vms4) || matches!(queued, Self::Vms4) {
            return true;
        }
        match self {
            Self::Vrpac => matches!(
                queued,
                Self::MultiplyAccumulate | Self::Axpy | Self::WholeAdd | Self::GroupAdd
            ),
            Self::WholeAdd => matches!(
                queued,
                Self::MultiplyAccumulate | Self::Axpy | Self::GroupAdd
            ),
            Self::GroupAdd => matches!(
                queued,
                Self::MultiplyAccumulate | Self::Axpy | Self::WholeAdd
            ),
            Self::WholeExtremum => matches!(queued, Self::GroupExtremum),
            _ => false,
        }
    }
}

impl C220VectorIssueVariant {
    pub(super) fn minimum_gap_from(self, previous: Self) -> u64 {
        let mut gap = 1;
        if self.is_fma() && (previous.is_add_reduction() || previous.is_relu_conversion()) {
            gap = gap.max(5);
        }
        if self.is_fma() && previous.is_special_conversion() {
            gap = gap.max(3);
        }
        if self == Self::Log && previous.is_fma() {
            gap = gap.max(6);
        }
        if self == Self::Exp && previous.is_long_special() {
            gap = gap.max(2);
        }
        if self == Self::Exp && previous.is_special_conversion() {
            gap = gap.max(5);
        }
        if self == Self::Exp && (previous.is_fma() || previous == Self::MulConversion) {
            gap = gap.max(6);
        }
        if self == Self::SlowDivide && previous.is_special_conversion() {
            gap = gap.max(5);
        }
        if self == Self::SlowDivide && (previous.is_fma() || previous == Self::MulConversion) {
            gap = gap.max(6);
        }
        if self.is_vcadd(C220ReductionWidth::F32) && previous.is_vcadd(C220ReductionWidth::F16) {
            gap = gap.max(4);
        }
        if self.is_vcgadd(C220ReductionWidth::F16) {
            if previous.is_vcadd(C220ReductionWidth::F32) {
                gap = gap.max(4);
            }
            if previous.is_relu_conversion() {
                gap = gap.max(10);
            }
        }
        if self.is_vcgadd(C220ReductionWidth::F32) {
            if previous.is_vcadd(C220ReductionWidth::F16)
                || previous.is_vcgadd(C220ReductionWidth::F16)
            {
                gap = gap.max(4);
            }
            if previous.is_relu_conversion() {
                gap = gap.max(7);
            }
        }
        if self.is_special_conversion() && previous.is_fma() {
            gap = gap.max(2);
        }
        if self.is_scalar_s32_dequant()
            && (previous.is_special_conversion()
                || previous.is_fma()
                || previous == Self::MulConversion)
        {
            gap = gap.max(2);
        }
        if self.is_narrow_relu_conversion()
            && (previous.is_special_conversion() || previous.is_scalar_s32_dequant())
        {
            gap = gap.max(6);
        }
        if self == Self::MulConversion
            && (previous.is_special_conversion() || previous.is_scalar_s32_dequant())
        {
            gap = gap.max(6);
        }
        match (self, previous) {
            (Self::Sort(current), Self::Sort(prior)) if current == prior => gap.max(17),
            (Self::Sort(_), Self::Sort(_)) => gap.max(16),
            _ => gap,
        }
    }

    const fn is_fma(self) -> bool {
        matches!(self, Self::Fma)
    }

    const fn is_add_reduction(self) -> bool {
        matches!(self, Self::AddReduction { .. })
    }

    const fn is_vcadd(self, expected: C220ReductionWidth) -> bool {
        matches!(
            self,
            Self::AddReduction {
                grouped: false,
                width,
            } if width as u8 == expected as u8
        )
    }

    const fn is_vcgadd(self, expected: C220ReductionWidth) -> bool {
        matches!(
            self,
            Self::AddReduction {
                grouped: true,
                width,
            } if width as u8 == expected as u8
        )
    }

    const fn is_relu_conversion(self) -> bool {
        matches!(self, Self::ReluConversion(_))
    }

    const fn is_narrow_relu_conversion(self) -> bool {
        matches!(self, Self::ReluConversion(C220FusedFormat::F16ToS8))
    }

    const fn is_special_conversion(self) -> bool {
        matches!(self, Self::Conversion(1019..=1022))
    }

    const fn is_scalar_s32_dequant(self) -> bool {
        matches!(self, Self::Conversion(1023))
    }

    const fn is_long_special(self) -> bool {
        matches!(self, Self::Log | Self::SlowDivide)
    }
}

impl C220VectorIssueVariant {
    pub(super) fn from_compute(compute: Option<C220VectorReadIssue<'_>>) -> Self {
        match compute {
            Some(C220VectorReadIssue::Arithmetic(issue))
                if issue.hint.operation == C220VecArithmeticOperation::Divide =>
            {
                C220VectorIssueVariant::SlowDivide
            }
            Some(C220VectorReadIssue::Reduction(issue)) => match issue.instruction.kind {
                C220ReductionKind::WholeAdd { .. } => C220VectorIssueVariant::AddReduction {
                    grouped: false,
                    width: issue.instruction.width,
                },
                C220ReductionKind::GroupAdd => C220VectorIssueVariant::AddReduction {
                    grouped: true,
                    width: issue.instruction.width,
                },
                C220ReductionKind::WholeExtremum { .. }
                | C220ReductionKind::GroupExtremum { .. }
                | C220ReductionKind::PairAdd => C220VectorIssueVariant::Other,
            },
            Some(C220VectorReadIssue::Ternary(issue))
                if matches!(
                    issue.instruction.operation,
                    C220TernaryOperation::MultiplyAccumulate
                        | C220TernaryOperation::MultiplyAddRelu
                ) =>
            {
                C220VectorIssueVariant::Fma
            }
            Some(C220VectorReadIssue::Axpy(_)) => C220VectorIssueVariant::Fma,
            Some(C220VectorReadIssue::SpecialUnary(issue)) => match issue.instruction.operation {
                C220SpecialUnaryOperation::Ln => C220VectorIssueVariant::Log,
                C220SpecialUnaryOperation::Exp => C220VectorIssueVariant::Exp,
                C220SpecialUnaryOperation::Sqrt => C220VectorIssueVariant::SlowDivide,
                _ => C220VectorIssueVariant::Other,
            },
            Some(C220VectorReadIssue::Conversion(issue)) => {
                let id = issue.instruction.kind.conversion_id();
                if matches!(
                    issue.instruction.kind,
                    C220ConversionKind::VectorDeqS16ToS8 { .. }
                        | C220ConversionKind::ScalarDeqS16ToS8 { .. }
                        | C220ConversionKind::ScalarDeqS32ToF16
                ) {
                    C220VectorIssueVariant::Conversion(id)
                } else {
                    C220VectorIssueVariant::Other
                }
            }
            Some(C220VectorReadIssue::Fused(issue)) => match issue.instruction.operation {
                C220FusedOperation::AddRelu | C220FusedOperation::SubtractRelu
                    if matches!(
                        issue.instruction.format,
                        C220FusedFormat::F16ToS8 | C220FusedFormat::F32ToF16
                    ) =>
                {
                    C220VectorIssueVariant::ReluConversion(issue.instruction.format)
                }
                C220FusedOperation::Multiply
                    if issue.instruction.format == C220FusedFormat::F16ToS8 =>
                {
                    C220VectorIssueVariant::MulConversion
                }
                _ => C220VectorIssueVariant::Other,
            },
            Some(C220VectorReadIssue::Sort(issue)) => {
                C220VectorIssueVariant::Sort(issue.instruction.width)
            }
            _ => C220VectorIssueVariant::Other,
        }
    }
}
