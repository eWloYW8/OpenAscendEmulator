use super::hazards::{C220VectorIssueVariant, C220VectorQueueClass};
use super::{
    C220VectorPipeline, C220VectorPipelineError, PendingReductionUpdate, PendingVectorUop,
};
use crate::isa::c220::vector::gather::C220GatherKind;
use crate::sim::c220::memory::C220UbRequest;
use crate::sim::c220::vector::ops::reduce::{
    C220ReductionState, C220ReductionStateUpdate, C220ReductionValue,
};
use crate::sim::c220::vector::read::{C220VectorReadIssue, PendingVectorRead};
use crate::sim::c220::vector::timing::{
    C220VectorTimelineError, C220VectorUop, C220VectorUopKind, C220VectorWritePlan,
};
use crate::sim::c220::vector::{C220_VECTOR_BLOCK_BYTES, C220VectorStore};
impl C220VectorPipeline {
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
        let issue_variant = C220VectorIssueVariant::from_compute(compute);
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
}
