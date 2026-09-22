use std::collections::{BTreeMap, VecDeque};
use std::num::NonZeroU64;

use thiserror::Error;

use crate::isa::c220::conversion::C220ConversionKind;
use crate::isa::c220::fused::{C220FusedFormat, C220FusedOperation};
use crate::isa::c220::gather::C220GatherKind;
use crate::isa::c220::no_effect::{C220NoEffectVectorInstruction, C220NoEffectVectorOperation};
use crate::isa::c220::reduce::{C220ReductionKind, C220ReductionWidth};
use crate::isa::c220::sort::C220SortWidth;
use crate::isa::c220::special::C220SpecialUnaryOperation;
use crate::isa::c220::ternary::C220TernaryOperation;
use crate::isa::c220::vector::C220VecArithmeticOperation;
use crate::memory::ub::UbMemory;
use crate::sim::c220::core::functional::{C220FunctionalCore, C220FunctionalError};
use crate::sim::c220::ub_arbiter::{C220UbCycle, C220UbPort, C220UbRequest, C220UbRequestError};
use crate::sim::c220::vector::compare::{C220CompareMask, C220CompareMaskUpdate};
use crate::sim::c220::vector::load_va::C220VaUpdate;
use crate::sim::c220::vector::read::{
    C220VectorReadError, C220VectorReadIssue, C220VectorReadSample, PendingVectorRead,
};
use crate::sim::c220::vector::reduce::{
    C220ReductionState, C220ReductionStateUpdate, C220ReductionValue,
};
use crate::sim::c220::vector::select::C220SelectionMaskBlock;
use crate::sim::c220::vector::timing::{
    C220VectorTimelineError, C220VectorUop, C220VectorUopKind, C220VectorUopRelease,
    C220VectorWritePlan, C220VectorWritePlanError,
};
use crate::sim::c220::vector::{C220_VECTOR_BLOCK_BYTES, C220VectorError, C220VectorStore};
use crate::sim::common::scalar::ScalarMachineError;

const VECTOR_RETIREMENT_BOUNDARY_TICKS: u64 = 2;
const C220_VECTOR_READ_QUEUE_CAPACITY: usize = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220VectorTimingRules {
    pub dispatch_ticks: u64,
    pub uop_issue_interval: NonZeroU64,
    pub ub_response_ticks: u64,
}

#[derive(Debug, Error)]
pub enum C220VectorPipelineError {
    #[error("vector timeline computation overflowed")]
    TimeOverflow,
    #[error("vector stores do not match the scheduled uops")]
    StoreUopMismatch,
    #[error("unsupported vector store width {0}")]
    UnsupportedStoreWidth(u8),
    #[error("vector store address overflows")]
    StoreAddressOverflow,
    #[error(transparent)]
    Timeline(#[from] C220VectorTimelineError),
    #[error(transparent)]
    ReadSetup(#[from] C220VectorReadError),
    #[error(transparent)]
    UbRequest(#[from] C220UbRequestError),
    #[error(transparent)]
    WritePlan(#[from] C220VectorWritePlanError),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PendingVectorUop {
    uop: C220VectorUop,
    instruction_group: u64,
    instruction_last: bool,
    issue_variant: C220VectorIssueVariant,
    queue_class: C220VectorQueueClass,
    admission_tick: u64,
    admitted: bool,
    stores: Vec<C220VectorStore>,
    read: Option<PendingVectorRead>,
    shared_read_from_previous: bool,
    write: Option<C220UbRequest>,
    execute_ready_tick: Option<u64>,
    eligible_tick: Option<u64>,
    release_tick: Option<u64>,
    retirement_tick: Option<u64>,
    visible_tick: Option<u64>,
    committed: bool,
    compare_update: Option<C220CompareMaskUpdate>,
    compare_update_applied: bool,
    selection_update: Option<C220SelectionMaskBlock>,
    selection_update_applied: bool,
    reduction_update: Option<PendingReductionUpdate>,
    va_update: Option<C220VaUpdate>,
    va_update_applied: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PendingReductionUpdate {
    group: u64,
    update: Option<C220ReductionStateUpdate>,
    last: bool,
    applied: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum C220VectorIssueVariant {
    Other,
    Fma,
    AddReduction {
        grouped: bool,
        width: C220ReductionWidth,
    },
    ExtremumReduction {
        grouped: bool,
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
            return match instruction.operation {
                C220NoEffectVectorOperation::Vrpac => Self::Vrpac,
                C220NoEffectVectorOperation::Vms4 => Self::Vms4,
                _ => Self::Ordinary,
            };
        }
        if let Some(instruction) = crate::isa::c220::reduce::C220ReductionInstruction::decode(word)
        {
            return match instruction.kind {
                C220ReductionKind::WholeAdd { .. } => Self::WholeAdd,
                C220ReductionKind::GroupAdd => Self::GroupAdd,
                C220ReductionKind::WholeExtremum { .. } => Self::WholeExtremum,
                C220ReductionKind::GroupExtremum { .. } => Self::GroupExtremum,
                C220ReductionKind::PairAdd => Self::Ordinary,
            };
        }
        if crate::isa::c220::axpy::C220AxpyInstruction::decode(word).is_some() {
            return Self::Axpy;
        }
        if crate::isa::c220::ternary::C220TernaryInstruction::decode(word).is_some_and(
            |instruction| instruction.operation == C220TernaryOperation::MultiplyAccumulate,
        ) {
            return Self::MultiplyAccumulate;
        }
        Self::Ordinary
    }

    fn from_compute(compute: Option<C220VectorReadIssue<'_>>) -> Self {
        match compute {
            Some(C220VectorReadIssue::Reduction(issue)) => match issue.instruction.kind {
                C220ReductionKind::WholeAdd { .. } => Self::WholeAdd,
                C220ReductionKind::GroupAdd => Self::GroupAdd,
                C220ReductionKind::WholeExtremum { .. } => Self::WholeExtremum,
                C220ReductionKind::GroupExtremum { .. } => Self::GroupExtremum,
                C220ReductionKind::PairAdd => Self::Ordinary,
            },
            Some(C220VectorReadIssue::Ternary(issue))
                if issue.instruction.operation == C220TernaryOperation::MultiplyAccumulate =>
            {
                Self::MultiplyAccumulate
            }
            Some(C220VectorReadIssue::Axpy(_)) => Self::Axpy,
            _ => Self::Ordinary,
        }
    }

    const fn is_blocked_by(self, queued: Self) -> bool {
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
    fn minimum_gap_from(self, previous: Self) -> u64 {
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

#[derive(Debug, Clone)]
pub struct C220VectorPipeline {
    rules: C220VectorTimingRules,
    next_admission_tick: u64,
    next_service_tick: u64,
    observed_tick: Option<u64>,
    last_release_tick: Option<u64>,
    last_issue_variant: Option<C220VectorIssueVariant>,
    last_uop_admission_tick: Option<u64>,
    pending: VecDeque<PendingVectorUop>,
    last_read_samples: Vec<C220VectorReadSample>,
    last_ub_cycles: Vec<C220UbCycle>,
    last_va_updates: Vec<C220VaUpdate>,
    compare_mask: C220CompareMask,
    selection_mask: Option<C220SelectionMaskBlock>,
    next_reduction_group: u64,
    next_instruction_group: u64,
    reduction_states: BTreeMap<u64, C220ReductionState>,
}

impl C220VectorPipeline {
    pub fn new(rules: C220VectorTimingRules) -> Self {
        Self {
            rules,
            next_admission_tick: 0,
            next_service_tick: 0,
            observed_tick: None,
            last_release_tick: None,
            last_issue_variant: None,
            last_uop_admission_tick: None,
            pending: VecDeque::new(),
            last_read_samples: Vec::new(),
            last_ub_cycles: Vec::new(),
            last_va_updates: Vec::new(),
            compare_mask: C220CompareMask::default(),
            selection_mask: None,
            next_reduction_group: 0,
            next_instruction_group: 0,
            reduction_states: BTreeMap::new(),
        }
    }

    pub const fn rules(&self) -> C220VectorTimingRules {
        self.rules
    }

    pub fn pending_uops(&self) -> usize {
        self.pending
            .iter()
            .filter(|entry| entry.release_tick.is_none())
            .count()
    }

    pub fn instruction_buffer_ready_tick(&self) -> Option<u64> {
        let group = self
            .pending
            .iter()
            .find(|entry| !entry.admitted)?
            .instruction_group;
        let scheduled = self
            .pending
            .iter()
            .filter(|entry| entry.instruction_group == group && !entry.admitted)
            .map(|entry| entry.admission_tick)
            .max()?;
        Some(
            self.observed_tick
                .map_or(scheduled, |tick| scheduled.max(tick.saturating_add(1))),
        )
    }

    pub fn pending_ub_responses(&self) -> usize {
        self.pending
            .iter()
            .filter(|entry| !entry.stores.is_empty() && !entry.committed)
            .count()
    }

    pub fn last_read_samples(&self) -> &[C220VectorReadSample] {
        &self.last_read_samples
    }

    pub fn last_ub_cycles(&self) -> &[C220UbCycle] {
        &self.last_ub_cycles
    }

    pub fn last_va_updates(&self) -> &[C220VaUpdate] {
        &self.last_va_updates
    }

    pub const fn compare_mask(&self) -> C220CompareMask {
        self.compare_mask
    }

    pub(crate) fn set_compare_mask(&mut self, compare_mask: C220CompareMask) {
        self.compare_mask = compare_mask;
    }

    pub fn has_pending_compare_mask_write(&self) -> bool {
        self.pending.iter().any(|entry| {
            entry
                .read
                .as_ref()
                .is_some_and(PendingVectorRead::writes_compare_mask)
                && !entry.compare_update_applied
        })
    }

    fn predicted_ticks(&self) -> Vec<(u64, Option<u64>, u64)> {
        let mut previous_release = self.last_release_tick;
        let mut previous_read_ready = None;
        let mut release_and_visibility = Vec::with_capacity(self.pending.len());
        let mut group_retirements = BTreeMap::new();
        for entry in &self.pending {
            let release = entry.release_tick.unwrap_or_else(|| {
                let read_ready = if entry.shared_read_from_previous {
                    entry
                        .execute_ready_tick
                        .map(|ready| {
                            ready.saturating_sub(u64::from(entry.uop.stages.execute_ticks))
                        })
                        .or(previous_read_ready)
                        .unwrap_or(
                            entry
                                .admission_tick
                                .saturating_add(u64::from(entry.uop.stages.read_ticks)),
                        )
                } else if let Some(read) = &entry.read {
                    read.ready_tick().unwrap_or_else(|| {
                        let earliest_grant =
                            self.observed_tick.map_or(entry.admission_tick, |observed| {
                                entry.admission_tick.max(observed.saturating_add(1))
                            });
                        earliest_grant.saturating_add(u64::from(entry.uop.stages.read_ticks))
                    })
                } else {
                    entry
                        .admission_tick
                        .saturating_add(u64::from(entry.uop.stages.read_ticks))
                };
                previous_read_ready = Some(read_ready);
                let mut eligible = read_ready
                    .saturating_add(u64::from(entry.uop.stages.execute_ticks))
                    .saturating_add(entry.uop.writeback_ticks as u64);
                if entry
                    .write
                    .as_ref()
                    .is_some_and(|write| !write.is_complete())
                {
                    let earliest_grant = self
                        .observed_tick
                        .map_or(0, |observed| observed.saturating_add(1))
                        .max(entry.execute_ready_tick.unwrap_or(0));
                    eligible = eligible.max(earliest_grant.saturating_add(1));
                }
                eligible.max(previous_release.map_or(0, |tick| tick.saturating_add(1)))
            });
            if entry.execute_ready_tick.is_some() {
                previous_read_ready = entry
                    .execute_ready_tick
                    .map(|ready| ready.saturating_sub(u64::from(entry.uop.stages.execute_ticks)));
            }
            previous_release = Some(release);
            let visible = if entry.stores.is_empty() {
                None
            } else {
                Some(
                    entry
                        .visible_tick
                        .unwrap_or_else(|| release.saturating_add(self.rules.ub_response_ticks)),
                )
            };
            release_and_visibility.push((release, visible));
            if entry.instruction_last {
                group_retirements.insert(
                    entry.instruction_group,
                    entry.retirement_tick.unwrap_or_else(|| {
                        release.saturating_add(VECTOR_RETIREMENT_BOUNDARY_TICKS)
                    }),
                );
            }
        }
        self.pending
            .iter()
            .zip(release_and_visibility)
            .map(|(entry, (release, visible))| {
                let retirement = entry.retirement_tick.unwrap_or_else(|| {
                    *group_retirements
                        .get(&entry.instruction_group)
                        .expect("pending vector instruction has a final uop")
                });
                (release, visible, retirement)
            })
            .collect()
    }

    pub fn pending_visibility_tick(&self) -> Option<u64> {
        self.pending
            .iter()
            .zip(self.predicted_ticks())
            .filter(|(entry, _)| !entry.committed)
            .filter_map(|(_, (_, visible, _))| visible)
            .max()
    }

    pub fn pending_drain_tick(&self) -> Option<u64> {
        self.pending
            .iter()
            .zip(self.predicted_ticks())
            .map(|(_, (_, visible, retirement))| {
                visible.map_or(retirement, |tick| tick.max(retirement))
            })
            .max()
    }

    pub fn pending_move_va_blocker_tick(&self) -> Option<u64> {
        let last = self.pending.back()?;
        if matches!(last.uop.kind, C220VectorUopKind::MoveVa) {
            None
        } else {
            self.pending_drain_tick()
        }
    }

    pub(crate) fn pending_queue_hazard_tick(&self, incoming: C220VectorQueueClass) -> Option<u64> {
        self.pending
            .iter()
            .zip(self.predicted_ticks())
            .filter(|(entry, _)| incoming.is_blocked_by(entry.queue_class))
            .map(|(_, (_, _, retirement))| retirement)
            .max()
    }

    pub fn pending_load_va_drain_tick(&self) -> Option<u64> {
        self.pending
            .iter()
            .zip(self.predicted_ticks())
            .filter(|(entry, _)| {
                entry
                    .read
                    .as_ref()
                    .is_some_and(PendingVectorRead::is_load_va)
            })
            .map(|(_, (_, _, retirement))| retirement)
            .max()
    }

    pub fn issue_at(
        &mut self,
        tick: u64,
        uops: &[C220VectorUop],
        stores: &[C220VectorStore],
        compute: Option<C220VectorReadIssue<'_>>,
    ) -> Result<Option<u64>, C220VectorPipelineError> {
        let queue_class = C220VectorQueueClass::from_compute(compute);
        self.issue_classified_at(tick, uops, stores, compute, queue_class)
    }

    pub(crate) fn issue_classified_at(
        &mut self,
        tick: u64,
        uops: &[C220VectorUop],
        stores: &[C220VectorStore],
        compute: Option<C220VectorReadIssue<'_>>,
        queue_class: C220VectorQueueClass,
    ) -> Result<Option<u64>, C220VectorPipelineError> {
        if matches!(compute, Some(C220VectorReadIssue::Transpose(_)))
            && (uops.len() != 2
                || uops[0].repeat_index != 0
                || uops[1].repeat_index != 1
                || uops.iter().any(|uop| uop.lane_group.is_some()))
        {
            return Err(C220VectorPipelineError::StoreUopMismatch);
        }
        if let Some(C220VectorReadIssue::Nchw(issue)) = compute
            && (uops.len() != issue.rows.len() * 2
                || uops
                    .iter()
                    .enumerate()
                    .any(|(index, uop)| uop.repeat_index != index || uop.lane_group.is_some()))
        {
            return Err(C220VectorPipelineError::StoreUopMismatch);
        }
        if let Some(previous) = self.observed_tick
            && tick < previous
        {
            return Err(C220VectorTimelineError::TimeReversed {
                previous,
                requested: tick,
            }
            .into());
        }
        let mut grouped = vec![Vec::new(); uops.len()];
        for &store in stores {
            if !matches!(store.width_bytes, 1 | 2 | 4 | 8) {
                return Err(C220VectorPipelineError::UnsupportedStoreWidth(
                    store.width_bytes,
                ));
            }
            store
                .address
                .checked_add(u64::from(store.width_bytes))
                .ok_or(C220VectorPipelineError::StoreAddressOverflow)?;
            let logical_lane = match compute {
                Some(C220VectorReadIssue::Conversion(issue)) => issue
                    .logical_lane_for_store(&store)
                    .ok_or(C220VectorPipelineError::StoreUopMismatch)?,
                Some(C220VectorReadIssue::Fused(issue)) => issue
                    .logical_lane_for_store(&store)
                    .ok_or(C220VectorPipelineError::StoreUopMismatch)?,
                _ => store.lane_index,
            };
            let lanes_per_group = match compute {
                Some(C220VectorReadIssue::Gather(issue)) => match issue.instruction.kind {
                    C220GatherKind::Elements(_) => 16,
                    C220GatherKind::Blocks => usize::MAX,
                },
                _ => 64,
            };
            let index = uops
                .iter()
                .position(|uop| {
                    uop.writes_ub
                        && uop.repeat_index == store.repeat_index
                        && if matches!(
                            uop.kind,
                            C220VectorUopKind::GatherData { .. }
                                | C220VectorUopKind::LaneSlice { .. }
                        ) {
                            uop.kind.contains_lane(logical_lane, uop.lane_group)
                        } else {
                            let lane_group = logical_lane / lanes_per_group;
                            uop.lane_group
                                .is_none_or(|group| usize::from(group) == lane_group)
                        }
                })
                .ok_or(C220VectorPipelineError::StoreUopMismatch)?;
            grouped[index].push(store);
        }
        if uops
            .iter()
            .enumerate()
            .any(|(index, uop)| uop.writes_ub == grouped[index].is_empty())
        {
            return Err(C220VectorPipelineError::StoreUopMismatch);
        }

        let first_admission = tick
            .checked_add(self.rules.dispatch_ticks)
            .ok_or(C220VectorPipelineError::TimeOverflow)?;
        let mut next_admission_tick = self.next_admission_tick;
        let issue_variant = match compute {
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
                C220ReductionKind::WholeExtremum { .. } => {
                    C220VectorIssueVariant::ExtremumReduction { grouped: false }
                }
                C220ReductionKind::GroupExtremum { .. } => {
                    C220VectorIssueVariant::ExtremumReduction { grouped: true }
                }
                C220ReductionKind::PairAdd => C220VectorIssueVariant::Other,
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
        };
        let mut previous_variant = self.last_issue_variant;
        let mut previous_admission_tick = self.last_uop_admission_tick;
        let mut entries = Vec::with_capacity(uops.len());
        let instruction_group = self.next_instruction_group;
        if !uops.is_empty() {
            self.next_instruction_group = self
                .next_instruction_group
                .checked_add(1)
                .ok_or(C220VectorPipelineError::TimeOverflow)?;
        }
        let reduction_group = match compute {
            Some(C220VectorReadIssue::Reduction(issue))
                if issue.instruction.has_cross_repeat_state() && !uops.is_empty() =>
            {
                let group = self.next_reduction_group;
                self.next_reduction_group = self
                    .next_reduction_group
                    .checked_add(1)
                    .ok_or(C220VectorPipelineError::TimeOverflow)?;
                self.reduction_states.insert(
                    group,
                    C220ReductionState::for_instruction(issue.instruction, issue.fp16_mode)
                        .expect("stateful reduction has state"),
                );
                let synthetic_update = issue.iteration_masks.is_empty().then(|| {
                    C220ReductionStateUpdate::Add(C220ReductionValue::zero(issue.instruction.width))
                });
                Some((group, synthetic_update))
            }
            _ => None,
        };
        for (uop_index, (uop, stores)) in uops.iter().copied().zip(grouped).enumerate() {
            let shared_read_from_previous = match compute {
                Some(C220VectorReadIssue::Transpose(_)) => uop.repeat_index == 1,
                Some(C220VectorReadIssue::Nchw(_)) => uop.repeat_index % 2 == 1,
                _ => false,
            };
            let conflict_tick = previous_variant.zip(previous_admission_tick).map_or(
                0,
                |(previous, previous_tick)| {
                    previous_tick.saturating_add(issue_variant.minimum_gap_from(previous))
                },
            );
            let admission_tick = first_admission.max(next_admission_tick).max(conflict_tick);
            next_admission_tick = admission_tick
                .checked_add(self.rules.uop_issue_interval.get())
                .ok_or(C220VectorPipelineError::TimeOverflow)?;
            previous_variant = Some(issue_variant);
            previous_admission_tick = Some(admission_tick);
            let synthetic_zero_reduction =
                reduction_group.is_some_and(|(_, update)| update.is_some());
            let read = if let Some(issue) = compute
                && !shared_read_from_previous
                && !synthetic_zero_reduction
            {
                Some(PendingVectorRead::new(
                    issue,
                    uop.repeat_index,
                    uop.lane_group,
                    uop.kind,
                )?)
            } else {
                None
            };
            let write = if stores.is_empty() {
                None
            } else {
                let plan = C220VectorWritePlan::from_stores(&stores)?;
                let accesses = plan
                    .blocks
                    .iter()
                    .map(|block| (block.base_address, C220_VECTOR_BLOCK_BYTES))
                    .collect::<Vec<_>>();
                Some(C220UbRequest::from_accesses(&accesses)?)
            };
            entries.push(PendingVectorUop {
                uop,
                instruction_group,
                instruction_last: uop_index + 1 == uops.len(),
                issue_variant,
                queue_class,
                admission_tick,
                admitted: false,
                stores,
                read,
                shared_read_from_previous,
                write,
                execute_ready_tick: None,
                eligible_tick: None,
                release_tick: None,
                retirement_tick: None,
                visible_tick: None,
                committed: false,
                compare_update: None,
                compare_update_applied: false,
                selection_update: None,
                selection_update_applied: false,
                reduction_update: reduction_group.map(|(group, update)| PendingReductionUpdate {
                    group,
                    update,
                    last: uop_index + 1 == uops.len(),
                    applied: false,
                }),
                va_update: None,
                va_update_applied: false,
            });
        }
        self.next_admission_tick = next_admission_tick;
        if !uops.is_empty() {
            self.last_issue_variant = previous_variant;
            self.last_uop_admission_tick = previous_admission_tick;
        }
        self.pending.extend(entries);
        Ok(self.pending_visibility_tick())
    }

    pub fn advance_to(
        &mut self,
        tick: u64,
        core: &mut C220FunctionalCore,
    ) -> Result<Vec<C220VectorUopRelease>, C220VectorAdvanceError> {
        if let Some(previous) = self.observed_tick
            && tick < previous
        {
            return Err(C220VectorTimelineError::TimeReversed {
                previous,
                requested: tick,
            }
            .into());
        }
        self.last_read_samples.clear();
        self.last_ub_cycles.clear();
        self.last_va_updates.clear();
        let mut releases = Vec::new();
        while self.next_service_tick <= tick {
            if self.pending.is_empty() {
                break;
            }
            let cycle_tick = self.next_service_tick;
            if let Some(first_admission) = self
                .pending
                .iter()
                .filter(|entry| entry.release_tick.is_none())
                .map(|entry| entry.admission_tick)
                .min()
                && first_admission > cycle_tick
                && self
                    .pending
                    .iter()
                    .all(|entry| entry.visible_tick.is_none())
            {
                self.next_service_tick = first_admission;
                continue;
            }
            self.commit_ready(cycle_tick, core)?;
            self.admit_ready(cycle_tick)?;
            self.apply_compare_updates(cycle_tick, core)?;
            self.apply_selection_updates(cycle_tick);
            self.apply_reduction_updates(cycle_tick, core)?;
            self.apply_va_updates(cycle_tick);
            self.arbitrate_ub(cycle_tick, core.ub())?;
            self.finish_reads(cycle_tick, core.ub())?;
            self.apply_compare_updates(cycle_tick, core)?;
            self.apply_selection_updates(cycle_tick);
            self.apply_reduction_updates(cycle_tick, core)?;
            self.apply_va_updates(cycle_tick);
            self.finish_writes()?;
            self.release_ready(cycle_tick, &mut releases)?;
            self.commit_ready(cycle_tick, core)?;
            while self.pending.front().is_some_and(|entry| {
                entry
                    .retirement_tick
                    .is_some_and(|retirement| retirement <= cycle_tick)
                    && entry.committed
            }) {
                self.pending.pop_front();
            }
            self.next_service_tick = cycle_tick
                .checked_add(1)
                .ok_or(C220VectorAdvanceError::TimeOverflow)?;
        }
        self.next_service_tick = tick
            .checked_add(1)
            .ok_or(C220VectorAdvanceError::TimeOverflow)?;
        self.observed_tick = Some(tick);
        Ok(releases)
    }

    fn admit_ready(&mut self, tick: u64) -> Result<(), C220VectorAdvanceError> {
        let occupied = self
            .pending
            .iter()
            .filter(|entry| {
                entry.admitted
                    && entry
                        .read
                        .as_ref()
                        .is_some_and(|read| read.ready_tick().is_none())
            })
            .count();
        if occupied >= C220_VECTOR_READ_QUEUE_CAPACITY {
            return Ok(());
        }
        let Some(index) = self
            .pending
            .iter()
            .position(|entry| !entry.admitted && entry.admission_tick <= tick)
        else {
            return Ok(());
        };
        let entry = &mut self.pending[index];
        entry.admitted = true;
        entry.admission_tick = tick;
        if entry.read.is_none() && !entry.shared_read_from_previous {
            let execute_ready_tick = tick
                .checked_add(u64::from(entry.uop.stages.read_ticks))
                .and_then(|value| value.checked_add(u64::from(entry.uop.stages.execute_ticks)))
                .ok_or(C220VectorAdvanceError::TimeOverflow)?;
            entry.execute_ready_tick = Some(execute_ready_tick);
            if entry.write.is_none() {
                entry.eligible_tick = Some(
                    execute_ready_tick
                        .checked_add(entry.uop.writeback_ticks as u64)
                        .ok_or(C220VectorAdvanceError::TimeOverflow)?,
                );
            }
        }
        Ok(())
    }

    fn arbitrate_ub(&mut self, tick: u64, ub: &UbMemory) -> Result<(), C220VectorAdvanceError> {
        let write_index = self.pending.iter().position(|entry| {
            entry.execute_ready_tick.is_some_and(|ready| ready <= tick)
                && entry
                    .write
                    .as_ref()
                    .is_some_and(|write| !write.is_complete())
        });
        let select = |port: C220UbPort| {
            self.pending.iter().position(|entry| {
                entry.admitted
                    && entry.admission_tick <= tick
                    && entry
                        .read
                        .as_ref()
                        .is_some_and(|read| !read.request(port).is_complete())
            })
        };
        let port0_index = select(C220UbPort::VectorRead0);
        let port1_index = select(C220UbPort::VectorRead1);
        let destination_port_index = select(C220UbPort::VectorReadDestination);
        if write_index.is_none()
            && port0_index.is_none()
            && port1_index.is_none()
            && destination_port_index.is_none()
        {
            return Ok(());
        }
        let mut write_request = write_index.map(|index| {
            std::mem::take(self.pending[index].write.as_mut().expect("selected write"))
        });
        let mut port0_request = port0_index.map(|index| {
            std::mem::take(
                self.pending[index]
                    .read
                    .as_mut()
                    .expect("selected read")
                    .request_mut(C220UbPort::VectorRead0),
            )
        });
        let mut port1_request = port1_index.map(|index| {
            std::mem::take(
                self.pending[index]
                    .read
                    .as_mut()
                    .expect("selected read")
                    .request_mut(C220UbPort::VectorRead1),
            )
        });
        let mut destination_port_request = destination_port_index.map(|index| {
            std::mem::take(
                self.pending[index]
                    .read
                    .as_mut()
                    .expect("selected read")
                    .request_mut(C220UbPort::VectorReadDestination),
            )
        });
        let cycle = C220UbCycle::arbitrate_with_destination(
            tick,
            write_request.as_mut(),
            port0_request.as_mut(),
            port1_request.as_mut(),
            destination_port_request.as_mut(),
        );
        if let (Some(index), Some(request)) = (write_index, write_request) {
            self.pending[index].write = Some(request);
        }
        if let (Some(index), Some(request)) = (port0_index, port0_request) {
            *self.pending[index]
                .read
                .as_mut()
                .expect("selected read")
                .request_mut(C220UbPort::VectorRead0) = request;
        }
        if let (Some(index), Some(request)) = (port1_index, port1_request) {
            *self.pending[index]
                .read
                .as_mut()
                .expect("selected read")
                .request_mut(C220UbPort::VectorRead1) = request;
        }
        if let (Some(index), Some(request)) = (destination_port_index, destination_port_request) {
            *self.pending[index]
                .read
                .as_mut()
                .expect("selected read")
                .request_mut(C220UbPort::VectorReadDestination) = request;
        }
        for decision in &cycle.decisions {
            if !decision.granted {
                continue;
            }
            let index = match decision.port {
                C220UbPort::VectorRead0 => port0_index,
                C220UbPort::VectorRead1 => port1_index,
                C220UbPort::VectorReadDestination => destination_port_index,
                C220UbPort::VectorWrite => continue,
            }
            .expect("decision belongs to a selected read");
            self.pending[index]
                .read
                .as_mut()
                .expect("selected read")
                .capture(decision, ub)?;
        }
        self.last_ub_cycles.push(cycle);
        Ok(())
    }

    fn finish_reads(&mut self, tick: u64, ub: &UbMemory) -> Result<(), C220VectorAdvanceError> {
        for index in 0..self.pending.len() {
            if !self.pending[index].admitted {
                continue;
            }
            let waits_for_compare_mask = self.pending[index]
                .read
                .as_ref()
                .is_some_and(PendingVectorRead::uses_compare_mask)
                && self.pending.iter().take(index).any(|entry| {
                    entry
                        .read
                        .as_ref()
                        .is_some_and(PendingVectorRead::writes_compare_mask)
                        && !entry.compare_update_applied
                });
            if waits_for_compare_mask {
                continue;
            }
            let waits_for_selection_mask = self.pending[index]
                .read
                .as_ref()
                .is_some_and(PendingVectorRead::uses_selection_mask)
                && self.pending.iter().take(index).any(|entry| {
                    entry
                        .read
                        .as_ref()
                        .is_some_and(PendingVectorRead::writes_selection_mask)
                        && !entry.selection_update_applied
                });
            if waits_for_selection_mask {
                continue;
            }
            let admission_tick = self.pending[index].admission_tick;
            let read_ticks = self.pending[index].uop.stages.read_ticks;
            let Some(read) = self.pending[index].read.as_mut() else {
                continue;
            };
            if read.ready_tick().is_none()
                && let Some(grant_tick) = read.grant_tick(admission_tick)
            {
                read.set_ready_tick(
                    grant_tick
                        .checked_add(u64::from(read_ticks))
                        .ok_or(C220VectorAdvanceError::TimeOverflow)?,
                );
            }
            let Some(ready_tick) = read.ready_tick() else {
                continue;
            };
            if read.is_sampled() || ready_tick > tick {
                continue;
            }
            let (sample, stores) =
                read.sample(ub, self.compare_mask, self.selection_mask.as_ref())?;
            let compare_update = sample.compare_update;
            let selection_update = sample.selection_update.clone();
            let reduction_update = sample.reduction_update;
            let va_update = sample.va_update;
            let shares_read = read.shares_read_with_next();
            let first_store_count = self.pending[index].stores.len();
            if shares_read && (index + 1 >= self.pending.len() || stores.len() < first_store_count)
            {
                return Err(C220VectorAdvanceError::WriteTargetMismatch);
            }
            let (first_stores, second_stores) = if shares_read {
                stores.split_at(first_store_count)
            } else {
                (stores.as_slice(), &[][..])
            };
            if !write_targets_match(first_stores, &self.pending[index].stores)
                || (shares_read
                    && (!self.pending[index + 1].shared_read_from_previous
                        || !write_targets_match(second_stores, &self.pending[index + 1].stores)))
            {
                return Err(C220VectorAdvanceError::WriteTargetMismatch);
            }
            self.pending[index]
                .read
                .as_mut()
                .expect("read remains pending")
                .mark_sampled();
            self.pending[index].stores = first_stores.to_vec();
            self.pending[index].compare_update = compare_update;
            self.pending[index].selection_update = selection_update;
            if let Some(update) = self.pending[index].reduction_update.as_mut() {
                update.update = reduction_update;
            }
            self.pending[index].va_update = va_update;
            let execute_ready_tick = ready_tick
                .checked_add(u64::from(self.pending[index].uop.stages.execute_ticks))
                .ok_or(C220VectorAdvanceError::TimeOverflow)?;
            self.pending[index].execute_ready_tick = Some(execute_ready_tick);
            if self.pending[index].write.is_none() {
                self.pending[index].eligible_tick = Some(
                    execute_ready_tick
                        .checked_add(self.pending[index].uop.writeback_ticks as u64)
                        .ok_or(C220VectorAdvanceError::TimeOverflow)?,
                );
            }
            if shares_read {
                self.pending[index + 1].stores = second_stores.to_vec();
                self.pending[index + 1].execute_ready_tick = Some(
                    ready_tick
                        .checked_add(u64::from(self.pending[index + 1].uop.stages.execute_ticks))
                        .ok_or(C220VectorAdvanceError::TimeOverflow)?,
                );
            }
            self.last_read_samples.push(sample);
        }
        Ok(())
    }

    fn apply_compare_updates(
        &mut self,
        tick: u64,
        core: &mut C220FunctionalCore,
    ) -> Result<(), C220VectorAdvanceError> {
        for index in 0..self.pending.len() {
            if self.pending[index].compare_update_applied
                || self.pending[index]
                    .execute_ready_tick
                    .is_none_or(|ready| ready > tick)
            {
                continue;
            }
            let earlier_state_access_pending = self.pending.iter().take(index).any(|entry| {
                let Some(read) = entry.read.as_ref() else {
                    return false;
                };
                (read.uses_compare_mask() && !read.is_sampled())
                    || (read.writes_compare_mask() && !entry.compare_update_applied)
            });
            if earlier_state_access_pending {
                continue;
            }
            if let Some(update) = self.pending[index].compare_update {
                self.compare_mask.apply(update);
                let [low, high] = self.compare_mask.bits();
                core.scalar_mut().machine_mut().set_spr_value(104, low)?;
                core.scalar_mut().machine_mut().set_spr_value(105, high)?;
                self.pending[index].compare_update_applied = true;
            }
        }
        Ok(())
    }

    fn apply_selection_updates(&mut self, tick: u64) {
        for index in 0..self.pending.len() {
            if self.pending[index].selection_update_applied
                || self.pending[index]
                    .execute_ready_tick
                    .is_none_or(|ready| ready > tick)
            {
                continue;
            }
            let earlier_state_access_pending = self.pending.iter().take(index).any(|entry| {
                let Some(read) = entry.read.as_ref() else {
                    return false;
                };
                (read.uses_selection_mask() && !read.is_sampled())
                    || (read.writes_selection_mask() && !entry.selection_update_applied)
            });
            if earlier_state_access_pending {
                continue;
            }
            if let Some(update) = self.pending[index].selection_update.clone() {
                self.selection_mask = Some(update);
                self.pending[index].selection_update_applied = true;
            }
        }
    }

    fn apply_reduction_updates(
        &mut self,
        tick: u64,
        core: &mut C220FunctionalCore,
    ) -> Result<(), C220VectorAdvanceError> {
        for index in 0..self.pending.len() {
            let Some(update) = self.pending[index].reduction_update else {
                continue;
            };
            if update.applied
                || update.update.is_none()
                || self.pending[index]
                    .execute_ready_tick
                    .is_none_or(|ready| ready > tick)
            {
                continue;
            }
            if self.pending.iter().take(index).any(|entry| {
                entry
                    .reduction_update
                    .is_some_and(|prior| prior.group == update.group && !prior.applied)
            }) {
                continue;
            }
            let mut state = self
                .reduction_states
                .remove(&update.group)
                .expect("issued reduction has state");
            state.apply(update.update.expect("checked reduction update"));
            if update.last {
                let (register, value) = state.result().expect("completed reduction has a result");
                core.scalar_mut()
                    .machine_mut()
                    .set_spr_value(register, value)?;
            } else {
                self.reduction_states.insert(update.group, state);
            }
            self.pending[index]
                .reduction_update
                .as_mut()
                .expect("reduction update remains pending")
                .applied = true;
        }
        Ok(())
    }

    fn apply_va_updates(&mut self, tick: u64) {
        for index in 0..self.pending.len() {
            if self.pending[index].va_update_applied
                || self.pending[index]
                    .execute_ready_tick
                    .is_none_or(|ready| ready > tick)
            {
                continue;
            }
            if self.pending.iter().take(index).any(|entry| {
                entry
                    .read
                    .as_ref()
                    .is_some_and(PendingVectorRead::is_load_va)
                    && !entry.va_update_applied
            }) {
                continue;
            }
            if let Some(update) = self.pending[index].va_update {
                self.last_va_updates.push(update);
                self.pending[index].va_update_applied = true;
            }
        }
    }

    fn finish_writes(&mut self) -> Result<(), C220VectorAdvanceError> {
        for entry in &mut self.pending {
            if entry.eligible_tick.is_some() {
                continue;
            }
            let (Some(ready_tick), Some(write)) = (entry.execute_ready_tick, &entry.write) else {
                continue;
            };
            if !write.is_complete() {
                continue;
            }
            let baseline = ready_tick
                .checked_add(entry.uop.writeback_ticks as u64)
                .ok_or(C220VectorAdvanceError::TimeOverflow)?;
            let last_grant = write.completion_tick().unwrap_or(ready_tick);
            let granted = last_grant
                .checked_add(1)
                .ok_or(C220VectorAdvanceError::TimeOverflow)?;
            entry.eligible_tick = Some(baseline.max(granted));
        }
        Ok(())
    }

    fn release_ready(
        &mut self,
        tick: u64,
        releases: &mut Vec<C220VectorUopRelease>,
    ) -> Result<(), C220VectorAdvanceError> {
        let mut completed_groups = Vec::new();
        for entry in &mut self.pending {
            if entry.release_tick.is_some() {
                continue;
            }
            if entry.reduction_update.is_some_and(|update| !update.applied) {
                break;
            }
            if entry.va_update.is_some() && !entry.va_update_applied {
                break;
            }
            let Some(eligible_tick) = entry.eligible_tick else {
                break;
            };
            let release_tick = eligible_tick.max(
                self.last_release_tick
                    .map_or(0, |previous| previous.saturating_add(1)),
            );
            if release_tick > tick {
                break;
            }
            entry.release_tick = Some(release_tick);
            self.last_release_tick = Some(release_tick);
            if entry.instruction_last {
                completed_groups.push((entry.instruction_group, release_tick));
            }
            if entry.stores.is_empty() {
                entry.committed = true;
            } else {
                entry.visible_tick = Some(
                    release_tick
                        .checked_add(self.rules.ub_response_ticks)
                        .ok_or(C220VectorAdvanceError::TimeOverflow)?,
                );
            }
            releases.push(C220VectorUopRelease {
                pc: entry.uop.pc,
                repeat_index: entry.uop.repeat_index,
                lane_group: entry.uop.lane_group,
                kind: entry.uop.kind,
                admission_tick: entry.admission_tick,
                eligible_tick,
                release_tick,
                ub_write_requested: entry.uop.writes_ub,
            });
        }
        for (instruction_group, last_release_tick) in completed_groups {
            let retirement_tick = last_release_tick
                .checked_add(VECTOR_RETIREMENT_BOUNDARY_TICKS)
                .ok_or(C220VectorAdvanceError::TimeOverflow)?;
            for entry in self
                .pending
                .iter_mut()
                .filter(|entry| entry.instruction_group == instruction_group)
            {
                entry.retirement_tick = Some(retirement_tick);
            }
        }
        Ok(())
    }

    fn commit_ready(
        &mut self,
        tick: u64,
        core: &mut C220FunctionalCore,
    ) -> Result<(), C220VectorAdvanceError> {
        for entry in &mut self.pending {
            if !entry.committed && entry.visible_tick.is_some_and(|visible| visible <= tick) {
                core.commit_c220_vector_stores(&entry.stores)?;
                entry.committed = true;
            }
        }
        Ok(())
    }
}

fn write_targets_match(actual: &[C220VectorStore], planned: &[C220VectorStore]) -> bool {
    actual.len() == planned.len()
        && actual.iter().zip(planned).all(|(actual, planned)| {
            actual.repeat_index == planned.repeat_index
                && actual.address == planned.address
                && actual.lane_index == planned.lane_index
                && actual.width_bytes == planned.width_bytes
        })
}

#[derive(Debug, Error)]
pub enum C220VectorAdvanceError {
    #[error(transparent)]
    Timeline(#[from] C220VectorTimelineError),
    #[error(transparent)]
    Commit(#[from] C220FunctionalError),
    #[error(transparent)]
    Read(#[from] C220VectorError),
    #[error(transparent)]
    Scalar(#[from] ScalarMachineError),
    #[error("sampled vector stores disagree with their issue-time write targets")]
    WriteTargetMismatch,
    #[error("vector timeline computation overflowed")]
    TimeOverflow,
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::architecture::Architecture;
    use crate::architecture::c220::C220UbBank;
    use crate::memory::sparse::MemoryByteState;
    use crate::sim::c220::vector::{
        C220VectorAddresses, C220VectorArithmeticModes, C220VectorControl,
        plan_c220_vector_arithmetic_issue,
    };
    use crate::sim::common::scalar::ScalarMachine;
    use crate::sim::common::scalar::stepper::ScalarStepper;

    #[test]
    fn conflicting_read_ports_delay_visibility_and_keep_granted_bytes() {
        let mut ub = UbMemory::new(4096, 256);
        for (address, value) in [(0, 1.0_f32), (0x10000, 2.0_f32)] {
            let bytes = value
                .to_le_bytes()
                .repeat(8)
                .into_iter()
                .map(MemoryByteState::Known)
                .collect::<Vec<_>>();
            ub.write_states(address, &bytes).unwrap();
        }
        let issue = plan_c220_vector_arithmetic_issue(
            0,
            0x85dc_b618,
            C220VectorControl {
                encoded_repeat_count: 0,
                destination_block_stride: 1,
                source_0_block_stride: 1,
                source_1_block_stride: 1,
                destination_repeat_stride: 1,
                source_0_repeat_stride: 1,
                source_1_repeat_stride: 1,
            },
            C220VectorAddresses {
                source_0: 0,
                source_1: 0x10000,
                destination: 0x200,
            },
            &[[0xff, 0, 0, 0]],
            C220VectorArithmeticModes::from_control_spr(0),
            &ub,
        )
        .unwrap();
        let machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
        let mut core = C220FunctionalCore::new(ScalarStepper::new(machine, 0), ub);
        let mut pipeline = C220VectorPipeline::new(C220VectorTimingRules {
            dispatch_ticks: 0,
            uop_issue_interval: NonZeroU64::new(1).unwrap(),
            ub_response_ticks: 2,
        });
        let write_stores = (0..8)
            .map(|lane| C220VectorStore {
                repeat_index: 0,
                lane_index: lane,
                address: (lane * 4) as u64,
                bank: C220UbBank::from_address((lane * 4) as u64),
                width_bytes: 4,
                data: super::super::store_data(4.0_f32.to_le_bytes()),
            })
            .collect::<Vec<_>>();
        pipeline
            .issue_at(
                0,
                &[C220VectorUop {
                    pc: 0,
                    repeat_index: 0,
                    lane_group: Some(0),
                    kind: crate::sim::c220::vector::timing::C220VectorUopKind::Ordinary,
                    stages: crate::sim::c220::vector::timing::C220VectorUopStages {
                        read_ticks: 1,
                        execute_ticks: 0,
                    },
                    writeback_ticks: 1,
                    writes_ub: true,
                }],
                &write_stores,
                None,
            )
            .unwrap();
        pipeline
            .issue_at(
                0,
                &[C220VectorUop {
                    pc: 0,
                    repeat_index: 0,
                    lane_group: Some(0),
                    kind: crate::sim::c220::vector::timing::C220VectorUopKind::Ordinary,
                    stages: crate::sim::c220::vector::timing::C220VectorUopStages {
                        read_ticks: 6,
                        execute_ticks: 7,
                    },
                    writeback_ticks: 1,
                    writes_ub: true,
                }],
                &issue.write_targets,
                Some(C220VectorReadIssue::Arithmetic(&issue)),
            )
            .unwrap();
        pipeline.advance_to(1, &mut core).unwrap();
        let cycle = &pipeline.last_ub_cycles()[0];
        assert!(
            cycle
                .decisions
                .iter()
                .any(|decision| { decision.port == C220UbPort::VectorWrite && decision.granted })
        );
        assert!(
            cycle
                .decisions
                .iter()
                .any(|decision| { decision.port == C220UbPort::VectorRead0 && !decision.granted })
        );
        pipeline.advance_to(3, &mut core).unwrap();
        assert_eq!(pipeline.pending_visibility_tick(), Some(18));
        pipeline.advance_to(8, &mut core).unwrap();
        let sample = &pipeline.last_read_samples()[0];
        assert_eq!(sample.tick, 8);
        assert_eq!(sample.read0_grants, [Some(2)]);
        assert_eq!(sample.read1_grants, [Some(1)]);
        assert_eq!(core.ub().read_known(0, 4).unwrap(), 4.0_f32.to_le_bytes());
        assert_eq!(&sample.source_0_bytes[..4], &1.0_f32.to_le_bytes());
        assert_eq!(sample.lanes[0].bits, 3.0_f32.to_bits());
        pipeline.advance_to(18, &mut core).unwrap();
        assert_eq!(
            core.ub().read_known(0x200, 4).unwrap(),
            3.0_f32.to_le_bytes()
        );
    }
}
