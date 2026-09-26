mod admission;
mod conflict;
mod functional;
mod hazards;
mod updates;
mod writeback;
use functional::FunctionalInstruction;
pub(crate) use hazards::C220VectorQueueClass;
use hazards::{C220VectorIssueVariant, RepeatOpcode};

use crate::memory::ub::UbMemory;
use crate::sim::c220::memory::{C220UbCycle, C220UbPort, C220UbRequest, C220UbRequestError};
use crate::sim::c220::state::{C220ExecutionError, C220State};
use crate::sim::c220::vector::ops::compare::C220CompareMask;
use crate::sim::c220::vector::ops::reduce::{C220ReductionState, C220ReductionStateUpdate};
use crate::sim::c220::vector::read::{
    C220VectorReadError, C220VectorReadSample, PendingVectorRead,
};
use crate::sim::c220::vector::timing::{
    C220VectorTimelineError, C220VectorUop, C220VectorUopRelease, C220VectorWriteCompletion,
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
    issue_gap: u64,
    issue_variant: C220VectorIssueVariant,
    repeat_opcode: Option<RepeatOpcode>,
    conflict_check_tick: Option<u64>,
    deferred_compute: bool,
    stores: Vec<C220VectorStore>,
    read: Option<PendingVectorRead>,
    shared_read_from_previous: bool,
    write: Option<C220UbRequest>,
    write_completion_tick: Option<u64>,
    execute_ready_tick: Option<u64>,
    eligible_tick: Option<u64>,
    release_tick: Option<u64>,
    retirement_tick: Option<u64>,
    visible_tick: Option<u64>,
    committed: bool,
    compare_update_applied: bool,
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

#[derive(Clone, Copy)]
struct ProjectedVectorTiming {
    visible: Option<u64>,
    retirement: u64,
    write_completion: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct C220VectorPipeline {
    rules: C220VectorTimingRules,
    next_admission_tick: u64,
    next_actual_admission_tick: u64,
    next_service_tick: u64,
    observed_tick: Option<u64>,
    last_release_tick: Option<u64>,
    last_conflict_check_tick: Option<u64>,
    pending: VecDeque<PendingVectorUop>,
    last_read_samples: Vec<C220VectorReadSample>,
    last_functional_samples: Vec<C220VectorReadSample>,
    functional_instructions: BTreeMap<u64, FunctionalInstruction>,
    last_ub_cycles: Vec<C220UbCycle>,
    last_write_completions: Vec<C220VectorWriteCompletion>,
    last_va_updates: Vec<C220VaUpdate>,
    compare_mask: C220CompareMask,
    next_reduction_group: u64,
    next_instruction_group: u64,
    register_retirements: BTreeMap<u64, u64>,
    reduction_states: BTreeMap<u64, C220ReductionState>,
}

impl C220VectorPipeline {
    pub fn new(rules: C220VectorTimingRules) -> Self {
        Self {
            rules,
            next_admission_tick: 0,
            next_actual_admission_tick: 0,
            next_service_tick: 0,
            observed_tick: None,
            last_release_tick: None,
            last_conflict_check_tick: None,
            pending: VecDeque::new(),
            last_read_samples: Vec::new(),
            last_functional_samples: Vec::new(),
            functional_instructions: BTreeMap::new(),
            last_ub_cycles: Vec::new(),
            last_write_completions: Vec::new(),
            last_va_updates: Vec::new(),
            compare_mask: C220CompareMask::default(),
            next_reduction_group: 0,
            next_instruction_group: 0,
            register_retirements: BTreeMap::new(),
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

    /// Planned or submitted UB writes whose bank service has not completed.
    pub fn pending_ub_responses(&self) -> usize {
        self.pending
            .iter()
            .filter(|entry| entry.write.is_some() && entry.write_completion_tick.is_none())
            .count()
    }

    pub fn last_read_samples(&self) -> &[C220VectorReadSample] {
        &self.last_read_samples
    }

    /// Whole-repeat numerical results, separate from timing-only UB request samples.
    pub fn last_functional_samples(&self) -> &[C220VectorReadSample] {
        &self.last_functional_samples
    }

    pub fn last_ub_cycles(&self) -> &[C220UbCycle] {
        &self.last_ub_cycles
    }

    /// Outstanding requests after Vector arbitration, excluding future uops
    /// and completed requests waiting only for response delivery.
    pub fn ub_port_occupancy(&self, tick: u64) -> (bool, bool) {
        let write = self.pending.iter().any(|entry| {
            entry.release_tick.is_some_and(|ready| ready <= tick)
                && entry
                    .write
                    .as_ref()
                    .is_some_and(|request| !request.is_complete())
        });
        let read = self.pending.iter().any(|entry| {
            entry.admitted
                && entry.admission_tick <= tick
                && entry.read.as_ref().is_some_and(|read| {
                    [C220UbPort::VectorRead0, C220UbPort::VectorRead1]
                        .into_iter()
                        .any(|port| !read.request(port).is_complete())
                })
        });
        (write, read)
    }

    pub fn last_write_completions(&self) -> &[C220VectorWriteCompletion] {
        &self.last_write_completions
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

    fn predicted_ticks(&self) -> Vec<ProjectedVectorTiming> {
        let mut previous_release = self.last_release_tick;
        let mut previous_read_ready = None;
        let mut previous_write_completion = None;
        let mut release_and_visibility = Vec::with_capacity(self.pending.len());
        let mut group_retirements = BTreeMap::new();
        for (index, entry) in self.pending.iter().enumerate() {
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
                        self.conflict_ready_tick(index, earliest_grant)
                            .saturating_add(u64::from(entry.uop.stages.read_ticks))
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
                eligible = eligible.max(entry.eligible_tick.unwrap_or(0));
                eligible.max(previous_release.map_or(0, |tick| tick.saturating_add(1)))
            });
            if entry.execute_ready_tick.is_some() {
                previous_read_ready = entry
                    .execute_ready_tick
                    .map(|ready| ready.saturating_sub(u64::from(entry.uop.stages.execute_ticks)));
            }
            previous_release = Some(release);
            let write_completion =
                self.projected_write_completion(entry, release, previous_write_completion);
            if write_completion.is_some() {
                previous_write_completion = write_completion;
            }
            let visible = if entry.deferred_compute {
                entry.instruction_last.then(|| {
                    entry
                        .visible_tick
                        .unwrap_or_else(|| release.saturating_add(1))
                })
            } else if entry.stores.is_empty() {
                None
            } else {
                Some(entry.visible_tick.unwrap_or_else(|| {
                    write_completion
                        .unwrap_or(release)
                        .saturating_add(self.rules.ub_response_ticks)
                }))
            };
            release_and_visibility.push((visible, write_completion));
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
            .map(|(entry, (visible, write_completion))| {
                let retirement = entry.retirement_tick.unwrap_or_else(|| {
                    *group_retirements
                        .get(&entry.instruction_group)
                        .expect("pending vector instruction has a final uop")
                });
                ProjectedVectorTiming {
                    visible,
                    retirement,
                    write_completion,
                }
            })
            .collect()
    }

    pub fn pending_visibility_tick(&self) -> Option<u64> {
        self.pending
            .iter()
            .zip(self.predicted_ticks())
            .filter(|(entry, _)| !entry.committed)
            .filter_map(|(_, timing)| timing.visible)
            .max()
    }

    pub(crate) fn instruction_fence(&self) -> Option<u64> {
        self.pending
            .back()
            .map(|entry| entry.instruction_group)
            .into_iter()
            .chain(self.register_retirements.keys().next_back().copied())
            .max()
    }

    pub(crate) fn fence_retirement_tick(&self, group: u64) -> Option<u64> {
        self.pending
            .iter()
            .zip(self.predicted_ticks())
            .filter(|(entry, _)| entry.instruction_group <= group)
            .map(|(_, timing)| timing.retirement)
            .chain(
                self.register_retirements
                    .range(..=group)
                    .map(|(_, &tick)| tick),
            )
            .filter(|&tick| self.observed_tick.is_none_or(|observed| observed < tick))
            .max()
    }

    pub(crate) fn fence_is_retired(&self, group: u64) -> bool {
        let retired = |due: u64| self.observed_tick.is_some_and(|tick| tick >= due);
        !self.pending.iter().any(|entry| {
            entry.instruction_group <= group && !entry.retirement_tick.is_some_and(retired)
        }) && self
            .register_retirements
            .range(..=group)
            .all(|(_, &due)| retired(due))
    }

    pub fn pending_drain_tick(&self) -> Option<u64> {
        self.predicted_ticks()
            .into_iter()
            .map(|timing| {
                timing
                    .visible
                    .map_or(timing.retirement, |tick| tick.max(timing.retirement))
                    .max(timing.write_completion.unwrap_or(0))
            })
            .chain(self.register_retirements.values().copied())
            .max()
    }

    /// Projected instruction retirement, independent of outstanding UB writes.
    pub fn pending_retirement_tick(&self) -> Option<u64> {
        self.predicted_ticks()
            .into_iter()
            .map(|timing| timing.retirement)
            .chain(self.register_retirements.values().copied())
            .filter(|&retirement| self.observed_tick.is_none_or(|tick| retirement > tick))
            .max()
    }

    pub fn pending_register_write_blocker_tick(&self) -> Option<u64> {
        let last = self.pending.back()?;
        if self
            .register_retirements
            .keys()
            .next_back()
            .is_some_and(|&group| group > last.instruction_group)
        {
            None
        } else {
            self.pending_retirement_tick()
        }
    }

    /// Instruction-group IDs and retirement ticks for writes that bypass execution uops.
    pub fn pending_register_retirements(&self) -> impl Iterator<Item = (u64, u64)> + '_ {
        self.register_retirements
            .iter()
            .map(|(&group, &tick)| (group, tick))
    }

    pub(crate) fn pending_queue_hazard_tick(&self, incoming: C220VectorQueueClass) -> Option<u64> {
        self.pending
            .iter()
            .zip(self.predicted_ticks())
            .filter(|(entry, _)| incoming.is_blocked_by(entry.queue_class))
            .map(|(_, timing)| timing.retirement)
            .filter(|&retirement| self.observed_tick.is_none_or(|tick| retirement > tick))
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
            .map(|(_, timing)| timing.retirement)
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
        self.last_functional_samples.clear();
        self.last_ub_cycles.clear();
        self.last_write_completions.clear();
        self.last_va_updates.clear();
    }

    pub(crate) fn next_event_tick(&self) -> Option<u64> {
        (!self.pending.is_empty())
            .then_some(self.next_service_tick)
            .into_iter()
            .chain(self.register_retirements.values().copied().min())
            .min()
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
            self.apply_reduction_updates(cycle_tick, core)?;
            self.apply_va_updates(cycle_tick);
            self.check_conflicts(cycle_tick)?;
            self.prepare_writeback()?;
            self.release_ready(cycle_tick, &mut releases)?;
            self.arbitrate_ub(cycle_tick, core.ub())?;
            self.check_conflicts(cycle_tick)?;
            self.finish_reads(cycle_tick, core.ub())?;
            self.apply_reduction_updates(cycle_tick, core)?;
            self.apply_va_updates(cycle_tick);
            self.finish_writes()?;
            self.prepare_writeback()?;
            self.release_ready(cycle_tick, &mut releases)?;
            self.commit_ready(cycle_tick, core)?;
            while self.pending.front().is_some_and(|entry| {
                entry
                    .retirement_tick
                    .is_some_and(|retirement| retirement <= cycle_tick)
                    && entry.committed
                    && (entry.write.is_none() || entry.write_completion_tick.is_some())
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
        self.register_retirements
            .retain(|_, retirement| *retirement > tick);
        Ok(releases)
    }

    fn admit_ready(&mut self, tick: u64) -> Result<(), C220VectorAdvanceError> {
        if tick < self.next_actual_admission_tick {
            return Ok(());
        }
        let occupied = self
            .pending
            .iter()
            .filter(|entry| entry.admitted && entry.conflict_check_tick.is_none())
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
        self.next_actual_admission_tick = tick
            .checked_add(self.rules.uop_issue_interval.get())
            .ok_or(C220VectorAdvanceError::TimeOverflow)?;
        Ok(())
    }

    fn arbitrate_ub(&mut self, tick: u64, ub: &UbMemory) -> Result<(), C220VectorAdvanceError> {
        let write_index = self.pending.iter().position(|entry| {
            entry.release_tick.is_some_and(|released| released <= tick)
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
            if self.pending[index].deferred_compute {
                continue;
            }
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
            if self.pending[index].conflict_check_tick.is_none() {
                continue;
            }
            let deferred_compute = self.pending[index].deferred_compute;
            let Some(read) = self.pending[index].read.as_mut() else {
                continue;
            };
            let Some(ready_tick) = read.ready_tick() else {
                continue;
            };
            if read.is_sampled() || ready_tick > tick {
                continue;
            }
            if deferred_compute {
                self.last_read_samples.push(read.timing_sample());
                read.mark_sampled();
                let entry = &mut self.pending[index];
                let execute_ready_tick = ready_tick
                    .checked_add(u64::from(entry.uop.stages.execute_ticks))
                    .ok_or(C220VectorAdvanceError::TimeOverflow)?;
                entry.execute_ready_tick = Some(execute_ready_tick);
                continue;
            }
            let (sample, stores) = read.sample(ub, self.compare_mask)?;
            let reduction_update = sample.reduction_update;
            let va_update = sample.va_update;
            if !write_targets_match(&stores, &self.pending[index].stores) {
                return Err(C220VectorAdvanceError::WriteTargetMismatch);
            }
            self.pending[index]
                .read
                .as_mut()
                .expect("read remains pending")
                .mark_sampled();
            self.pending[index].stores = stores;
            if let Some(update) = self.pending[index].reduction_update.as_mut() {
                update.update = reduction_update;
            }
            self.pending[index].va_update = va_update;
            let execute_ready_tick = ready_tick
                .checked_add(u64::from(self.pending[index].uop.stages.execute_ticks))
                .ok_or(C220VectorAdvanceError::TimeOverflow)?;
            self.pending[index].execute_ready_tick = Some(execute_ready_tick);
            self.last_read_samples.push(sample);
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
            if entry.deferred_compute {
                entry.committed = !entry.instruction_last;
                if entry.instruction_last {
                    entry.visible_tick = Some(
                        release_tick
                            .checked_add(1)
                            .ok_or(C220VectorAdvanceError::TimeOverflow)?,
                    );
                }
            } else if entry.stores.is_empty() {
                entry.committed = true;
            } else if entry.write.is_none() {
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
                conflict_check_tick: entry
                    .conflict_check_tick
                    .expect("released uop passed conflicts"),
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
        for index in 0..self.pending.len() {
            let entry = &self.pending[index];
            if !entry.committed && entry.visible_tick.is_some_and(|visible| visible <= tick) {
                if entry.deferred_compute {
                    self.complete_functional_instruction(entry.instruction_group, tick, core)?;
                } else {
                    core.commit_c220_vector_stores(&entry.stores)?;
                }
                self.pending[index].committed = true;
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
    ReadSetup(#[from] C220VectorReadError),
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
