use std::collections::{BTreeMap, VecDeque};
use std::num::NonZeroU64;

use thiserror::Error;

use crate::execution::c220::ub_arbiter::{
    C220UbCycle, C220UbPort, C220UbRequest, C220UbRequestError,
};
use crate::execution::c220::vector_read::{
    C220Fp32ReadSample, C220VectorReadError, PendingFp32Read,
};
use crate::execution::c220::vector_timing::{
    C220VectorTimelineError, C220VectorUop, C220VectorUopRelease, C220VectorWritePlan,
    C220VectorWritePlanError,
};
use crate::execution::mte_stepper::{MteCoreStepper, MteStepperError};
use crate::instruction::c220::vector::{
    C220_VECTOR_BLOCK_BYTES, C220Fp32Issue, C220VectorError, C220VectorStore,
};
use crate::memory::ub::UbMemory;

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
    admission_tick: u64,
    stores: Vec<C220VectorStore>,
    read: Option<PendingFp32Read>,
    write: Option<C220UbRequest>,
    execute_ready_tick: Option<u64>,
    eligible_tick: Option<u64>,
    release_tick: Option<u64>,
    visible_tick: Option<u64>,
    committed: bool,
}

#[derive(Debug, Clone)]
pub struct C220VectorPipeline {
    rules: C220VectorTimingRules,
    next_admission_tick: u64,
    next_service_tick: u64,
    observed_tick: Option<u64>,
    last_release_tick: Option<u64>,
    pending: VecDeque<PendingVectorUop>,
    last_read_samples: Vec<C220Fp32ReadSample>,
    last_ub_cycles: Vec<C220UbCycle>,
}

impl C220VectorPipeline {
    pub fn new(rules: C220VectorTimingRules) -> Self {
        Self {
            rules,
            next_admission_tick: 0,
            next_service_tick: 0,
            observed_tick: None,
            last_release_tick: None,
            pending: VecDeque::new(),
            last_read_samples: Vec::new(),
            last_ub_cycles: Vec::new(),
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

    pub fn pending_ub_responses(&self) -> usize {
        self.pending
            .iter()
            .filter(|entry| !entry.stores.is_empty() && !entry.committed)
            .count()
    }

    pub fn last_read_samples(&self) -> &[C220Fp32ReadSample] {
        &self.last_read_samples
    }

    pub fn last_ub_cycles(&self) -> &[C220UbCycle] {
        &self.last_ub_cycles
    }

    fn predicted_ticks(&self) -> Vec<(u64, Option<u64>)> {
        let mut previous_release = self.last_release_tick;
        let mut predicted = Vec::with_capacity(self.pending.len());
        for entry in &self.pending {
            let release = entry.release_tick.unwrap_or_else(|| {
                let read_ready = if let Some(read) = &entry.read {
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
            predicted.push((release, visible));
        }
        predicted
    }

    pub fn pending_visibility_tick(&self) -> Option<u64> {
        self.pending
            .iter()
            .zip(self.predicted_ticks())
            .filter(|(entry, _)| !entry.committed)
            .filter_map(|(_, (_, visible))| visible)
            .max()
    }

    pub fn pending_drain_tick(&self) -> Option<u64> {
        self.pending
            .iter()
            .zip(self.predicted_ticks())
            .filter(|(entry, _)| entry.release_tick.is_none() || !entry.committed)
            .map(|(_, (release, visible))| visible.unwrap_or(release))
            .max()
    }

    pub fn issue_at(
        &mut self,
        tick: u64,
        uops: &[C220VectorUop],
        stores: &[C220VectorStore],
        fp32: Option<&C220Fp32Issue>,
    ) -> Result<Option<u64>, C220VectorPipelineError> {
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
        let mut indices = BTreeMap::new();
        for (index, uop) in uops.iter().enumerate() {
            if indices
                .insert((uop.repeat_index, uop.lane_group), index)
                .is_some()
            {
                return Err(C220VectorPipelineError::StoreUopMismatch);
            }
        }
        for &store in stores {
            if !matches!(store.width_bytes, 2 | 4) {
                return Err(C220VectorPipelineError::UnsupportedStoreWidth(
                    store.width_bytes,
                ));
            }
            store
                .address
                .checked_add(u64::from(store.width_bytes))
                .ok_or(C220VectorPipelineError::StoreAddressOverflow)?;
            let lane_group = u8::try_from(store.lane_index / 64)
                .map_err(|_| C220VectorPipelineError::StoreUopMismatch)?;
            let index = indices
                .get(&(store.repeat_index, lane_group))
                .ok_or(C220VectorPipelineError::StoreUopMismatch)?;
            grouped[*index].push(store);
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
        let mut entries = Vec::with_capacity(uops.len());
        for (uop, stores) in uops.iter().copied().zip(grouped) {
            let admission_tick = first_admission.max(next_admission_tick);
            next_admission_tick = admission_tick
                .checked_add(self.rules.uop_issue_interval.get())
                .ok_or(C220VectorPipelineError::TimeOverflow)?;
            let read = if let Some(issue) = fp32
                && !stores.is_empty()
            {
                Some(PendingFp32Read::new(issue, uop.repeat_index)?)
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
            let execute_ready_tick = if read.is_none() {
                Some(
                    admission_tick
                        .checked_add(u64::from(uop.stages.read_ticks))
                        .and_then(|value| value.checked_add(u64::from(uop.stages.execute_ticks)))
                        .ok_or(C220VectorPipelineError::TimeOverflow)?,
                )
            } else {
                None
            };
            let eligible_tick = if write.is_none() {
                execute_ready_tick
                    .map(|ready| {
                        ready
                            .checked_add(uop.writeback_ticks as u64)
                            .ok_or(C220VectorPipelineError::TimeOverflow)
                    })
                    .transpose()?
            } else {
                None
            };
            entries.push(PendingVectorUop {
                uop,
                admission_tick,
                stores,
                read,
                write,
                execute_ready_tick,
                eligible_tick,
                release_tick: None,
                visible_tick: None,
                committed: false,
            });
        }
        self.next_admission_tick = next_admission_tick;
        self.pending.extend(entries);
        Ok(self.pending_visibility_tick())
    }

    pub fn advance_to(
        &mut self,
        tick: u64,
        core: &mut MteCoreStepper,
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
            self.arbitrate_ub(cycle_tick, core.ub())?;
            self.finish_reads(cycle_tick, core.ub())?;
            self.finish_writes()?;
            self.release_ready(cycle_tick, &mut releases)?;
            self.commit_ready(cycle_tick, core)?;
            while self
                .pending
                .front()
                .is_some_and(|entry| entry.release_tick.is_some() && entry.committed)
            {
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
                entry.admission_tick <= tick
                    && entry
                        .read
                        .as_ref()
                        .is_some_and(|read| !read.request(port).is_complete())
            })
        };
        let port0_index = select(C220UbPort::VectorRead0);
        let port1_index = select(C220UbPort::VectorRead1);
        if write_index.is_none() && port0_index.is_none() && port1_index.is_none() {
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
        let cycle = C220UbCycle::arbitrate(
            tick,
            write_request.as_mut(),
            port0_request.as_mut(),
            port1_request.as_mut(),
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
        for decision in &cycle.decisions {
            if !decision.granted {
                continue;
            }
            let index = match decision.port {
                C220UbPort::VectorRead0 => port0_index,
                C220UbPort::VectorRead1 => port1_index,
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
        for entry in &mut self.pending {
            let Some(read) = entry.read.as_mut() else {
                continue;
            };
            if read.ready_tick().is_none()
                && let Some(grant_tick) = read.grant_tick(entry.admission_tick)
            {
                read.set_ready_tick(
                    grant_tick
                        .checked_add(u64::from(entry.uop.stages.read_ticks))
                        .ok_or(C220VectorAdvanceError::TimeOverflow)?,
                );
            }
            let Some(ready_tick) = read.ready_tick() else {
                continue;
            };
            if read.is_sampled() || ready_tick > tick {
                continue;
            }
            let (sample, stores) = read.sample(ub)?;
            if stores.len() != entry.stores.len()
                || stores.iter().zip(&entry.stores).any(|(actual, planned)| {
                    actual.address != planned.address
                        || actual.lane_index != planned.lane_index
                        || actual.width_bytes != planned.width_bytes
                })
            {
                return Err(C220VectorAdvanceError::WriteTargetMismatch);
            }
            entry.stores = stores;
            read.mark_sampled();
            entry.execute_ready_tick = Some(
                ready_tick
                    .checked_add(u64::from(entry.uop.stages.execute_ticks))
                    .ok_or(C220VectorAdvanceError::TimeOverflow)?,
            );
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
        for entry in &mut self.pending {
            if entry.release_tick.is_some() {
                continue;
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
                admission_tick: entry.admission_tick,
                eligible_tick,
                release_tick,
                ub_write_requested: entry.uop.writes_ub,
            });
        }
        Ok(())
    }

    fn commit_ready(
        &mut self,
        tick: u64,
        core: &mut MteCoreStepper,
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

#[derive(Debug, Error)]
pub enum C220VectorAdvanceError {
    #[error(transparent)]
    Timeline(#[from] C220VectorTimelineError),
    #[error(transparent)]
    Commit(#[from] MteStepperError),
    #[error(transparent)]
    Read(#[from] C220VectorError),
    #[error("sampled FP32 stores disagree with their issue-time write targets")]
    WriteTargetMismatch,
    #[error("vector timeline computation overflowed")]
    TimeOverflow,
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::device::architecture::Architecture;
    use crate::execution::machine::ScalarMachine;
    use crate::execution::stepper::ScalarStepper;
    use crate::instruction::c220::vector::{
        C220Fp32Addresses, C220Fp32Control, plan_c220_fp32_issue,
    };
    use crate::memory::sparse::MemoryByteState;
    use crate::memory::ub_bank_c220::C220UbBank;

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
        let issue = plan_c220_fp32_issue(
            0,
            0x85dc_b618,
            C220Fp32Control {
                encoded_repeat_count: 0,
                destination_block_stride: 1,
                source_0_block_stride: 1,
                source_1_block_stride: 1,
                destination_repeat_stride: 1,
                source_0_repeat_stride: 1,
                source_1_repeat_stride: 1,
            },
            C220Fp32Addresses {
                source_0: 0,
                source_1: 0x10000,
                destination: 0x200,
            },
            &[[0xff, 0, 0, 0]],
            &ub,
        )
        .unwrap();
        let machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
        let mut core = MteCoreStepper::new(ScalarStepper::new(machine, 0), ub);
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
                data: 4.0_f32.to_le_bytes(),
            })
            .collect::<Vec<_>>();
        pipeline
            .issue_at(
                0,
                &[C220VectorUop {
                    pc: 0,
                    repeat_index: 0,
                    lane_group: 0,
                    stages: crate::execution::c220::vector_timing::C220VectorUopStages {
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
                    lane_group: 0,
                    stages: crate::execution::c220::vector_timing::C220VectorUopStages {
                        read_ticks: 6,
                        execute_ticks: 7,
                    },
                    writeback_ticks: 1,
                    writes_ub: true,
                }],
                &issue.write_targets,
                Some(&issue),
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
