use super::functional::FunctionalInstruction;
use super::hazards::{C220VectorIssueVariant, C220VectorQueueClass, RepeatOpcode};
use super::{
    C220VectorPipeline, C220VectorPipelineError, PendingReductionUpdate, PendingVectorUop,
};
use crate::isa::c220::vector::gather::C220GatherKind;
use crate::sim::c220::memory::C220UbRequest;
use crate::sim::c220::vector::ops::reduce::{
    C220ReductionState, C220ReductionStateUpdate, C220ReductionValue,
};
use crate::sim::c220::vector::read::{C220VectorReadIssue, PendingVectorRead};
use crate::sim::c220::vector::repeat::{AccumulatorSchedule, OrdinaryRepeatSchedule};
use crate::sim::c220::vector::timing::{
    C220VectorTimelineError, C220VectorUop, C220VectorUopKind, C220VectorWritePlan,
};
use crate::sim::c220::vector::{C220_VECTOR_BLOCK_BYTES, C220VectorStore};
impl C220VectorPipeline {
    pub(crate) fn issue_register_write_at(
        &mut self,
        tick: u64,
    ) -> Result<(), C220VectorPipelineError> {
        if let Some(previous) = self.observed_tick
            && tick < previous
        {
            return Err(C220VectorTimelineError::TimeReversed {
                previous,
                requested: tick,
            }
            .into());
        }
        let retirement = tick
            .checked_add(self.rules.dispatch_ticks)
            .and_then(|tick| tick.checked_add(super::VECTOR_RETIREMENT_BOUNDARY_TICKS))
            .ok_or(C220VectorPipelineError::TimeOverflow)?;
        let retirement = match self.register_retirements.values().next_back() {
            Some(&previous) => retirement.max(
                previous
                    .checked_add(1)
                    .ok_or(C220VectorPipelineError::TimeOverflow)?,
            ),
            None => retirement,
        };
        let group = self.next_instruction_group;
        let next_group = group
            .checked_add(1)
            .ok_or(C220VectorPipelineError::TimeOverflow)?;
        self.register_retirements.insert(group, retirement);
        self.next_instruction_group = next_group;
        self.observed_tick = Some(tick);
        Ok(())
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
        let accumulator = compute.and_then(AccumulatorSchedule::from_issue);
        let ordinary_reads = compute.and_then(OrdinaryRepeatSchedule::from_issue);
        let functional = compute.and_then(FunctionalInstruction::from_issue);
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
            if accumulator.is_some_and(|schedule| !schedule.has_traffic(store.repeat_index)) {
                continue;
            }
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
                    (uop.writes_ub || accumulator.is_some() || ordinary_reads.is_some())
                        && uop.repeat_index == store.repeat_index
                        && if matches!(uop.kind, C220VectorUopKind::LaneSlice { .. })
                            || (matches!(uop.kind, C220VectorUopKind::GatherData { .. })
                                && lanes_per_group != usize::MAX)
                        {
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
        if uops.iter().enumerate().any(|(index, uop)| {
            if accumulator.is_some() || ordinary_reads.is_some() {
                uop.lane_group.is_some() == grouped[index].is_empty()
            } else {
                uop.writes_ub == grouped[index].is_empty()
            }
        }) {
            return Err(C220VectorPipelineError::StoreUopMismatch);
        }

        let first_admission = tick
            .checked_add(self.rules.dispatch_ticks)
            .ok_or(C220VectorPipelineError::TimeOverflow)?
            .max(self.next_service_tick);
        let mut next_admission_tick = self.next_admission_tick;
        let issue_variant = C220VectorIssueVariant::from_compute(compute);
        let repeat_opcode = compute.and_then(RepeatOpcode::from_compute);
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
            let admission_tick = first_admission.max(next_admission_tick);
            let issue_gap = accumulator
                .zip(uop.kind.lane_slice())
                .map_or(1, |(schedule, (first, lanes))| {
                    schedule.issue_gap(uop.repeat_index, first, lanes)
                })
                .max(ordinary_reads.map_or(1, |schedule| {
                    schedule.issue_gap(uop.repeat_index, uop.stages.execute_ticks)
                }));
            next_admission_tick = admission_tick
                .checked_add(self.rules.uop_issue_interval.get())
                .ok_or(C220VectorPipelineError::TimeOverflow)?;
            let synthetic_zero_reduction =
                reduction_group.is_some_and(|(_, update)| update.is_some());
            let synthetic_functional = functional.is_some()
                && match compute {
                    Some(C220VectorReadIssue::Select(select)) => select.iteration_masks.is_empty(),
                    Some(C220VectorReadIssue::Broadcast(issue)) => issue.control.repeat_count == 0,
                    Some(C220VectorReadIssue::Gather(issue)) => issue.repeat_count() == 0,
                    Some(
                        C220VectorReadIssue::MoveMask(_)
                        | C220VectorReadIssue::Transpose(_)
                        | C220VectorReadIssue::Sort(_)
                        | C220VectorReadIssue::Nchw(_),
                    ) => false,
                    _ => uop.lane_group.is_none(),
                };
            let mut read = if let Some(issue) = compute
                && !shared_read_from_previous
                && !synthetic_zero_reduction
                && !synthetic_functional
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
            if let (Some(read), Some(schedule)) = (&mut read, accumulator) {
                read.configure_accumulator_timing(schedule)?;
            }
            if let (Some(read), Some(schedule)) = (&mut read, ordinary_reads) {
                read.configure_ordinary_timing(schedule)?;
            }
            let write = if stores.is_empty() || !uop.writes_ub {
                None
            } else {
                let plan = C220VectorWritePlan::from_stores(&stores)?;
                let accesses = plan
                    .blocks
                    .iter()
                    .map(|block| (block.base_address, C220_VECTOR_BLOCK_BYTES, block.full()))
                    .collect::<Vec<_>>();
                Some(C220UbRequest::from_writes(&accesses)?)
            };
            entries.push(PendingVectorUop {
                uop,
                instruction_group,
                instruction_last: uop_index + 1 == uops.len(),
                queue_class,
                admission_tick,
                admitted: false,
                issue_gap,
                issue_variant,
                repeat_opcode,
                conflict_check_tick: None,
                deferred_compute: functional.is_some(),
                stores,
                read,
                shared_read_from_previous,
                write,
                write_completion_tick: None,
                execute_ready_tick: None,
                eligible_tick: None,
                release_tick: None,
                retirement_tick: None,
                visible_tick: None,
                committed: false,
                compare_update_applied: false,
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
        if !uops.is_empty()
            && let Some(functional) = functional
        {
            self.functional_instructions
                .insert(instruction_group, functional);
        }
        self.pending.extend(entries);
        Ok(self.pending_visibility_tick())
    }
}
