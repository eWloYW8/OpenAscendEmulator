mod admission;
mod hazards;
mod updates;
use hazards::C220VectorIssueVariant;
pub(crate) use hazards::C220VectorQueueClass;

use crate::memory::ub::UbMemory;
use crate::sim::c220::memory::{C220UbCycle, C220UbPort, C220UbRequest, C220UbRequestError};
use crate::sim::c220::state::{C220ExecutionError, C220State};
use crate::sim::c220::vector::ops::compare::{C220CompareMask, C220CompareMaskUpdate};
use crate::sim::c220::vector::ops::reduce::{C220ReductionState, C220ReductionStateUpdate};
use crate::sim::c220::vector::ops::select::C220SelectionMaskBlock;
use crate::sim::c220::vector::read::{
    C220VectorReadError, C220VectorReadSample, PendingVectorRead,
};
use crate::sim::c220::vector::timing::{
    C220VectorTimelineError, C220VectorUop, C220VectorUopKind, C220VectorUopRelease,
    C220VectorWritePlanError,
};
use crate::sim::c220::vector::va::C220VaUpdate;
use crate::sim::c220::vector::{C220VectorError, C220VectorStore};
use crate::sim::common::scalar::ScalarMachineError;
use std::collections::{BTreeMap, VecDeque};
use std::num::NonZeroU64;
use thiserror::Error;
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

    pub fn advance_to(
        &mut self,
        tick: u64,
        core: &mut C220State,
    ) -> Result<Vec<C220VectorUopRelease>, C220VectorAdvanceError> {
        self.begin_advance();
        self.advance_event(tick, core)
    }

    pub(crate) fn begin_advance(&mut self) {
        self.last_read_samples.clear();
        self.last_ub_cycles.clear();
        self.last_va_updates.clear();
    }

    pub(crate) fn next_event_tick(&self) -> Option<u64> {
        (!self.pending.is_empty()).then_some(self.next_service_tick)
    }

    pub(crate) fn advance_event(
        &mut self,
        tick: u64,
        core: &mut C220State,
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
        core: &mut C220State,
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
    Commit(#[from] C220ExecutionError),
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
mod tests;
