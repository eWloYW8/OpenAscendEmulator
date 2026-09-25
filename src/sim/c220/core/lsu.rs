use std::collections::{BTreeMap, VecDeque};

mod cache;
mod ingress;
mod retirement;
mod store;
mod write_transport;
use crate::sim::c220::scalar::lsu::commit::{C220LoadCommitMode, C220LsuCommitLane};
use cache::CoreCache;
pub use cache::{C220CoreLoadCompletion, C220CoreLoadIssue};
pub use ingress::C220CoreLsuAdmission;
use ingress::DispatchedLsu;
pub use store::{C220CoreStoreCompletion, C220CoreStoreIssue};

use super::{C220Core, C220CoreError, C220CoreInstruction, C220CoreStep};
use crate::sim::c220::memory::timed_memory::{C220MemoryWriteCommand, C220MemoryWriteId};
use crate::sim::c220::scalar::lsu::cache::C220CacheAddressLayout;
use crate::sim::c220::scalar::lsu::miss_buffer::{C220LsuMissBuffer, C220LsuMissConfig};
use crate::sim::c220::scalar::lsu::scheduler::{
    C220LsuExternalHazards, C220LsuRequestScheduler, C220LsuSchedulerError,
};
use crate::sim::c220::scalar::lsu::store_buffer::{
    C220LsuCompletion, C220LsuStoreBuffer, C220LsuStoreConfig,
};
use crate::sim::c220::scalar::lsu::write_queue::{C220LsuWriteId, C220LsuWriteRequest};
use crate::sim::c220::scalar::lsu::{C220LsuRequestId, C220LsuStage};
use crate::sim::c220::scalar::{C220DirectStoreOperands, C220ScalarMappedAddress};
use crate::sim::c220::schedule::{C220Stall, C220StallCause};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220CoreLsuConfig {
    pub request_capacity: u32,
    pub read_capacity: u32,
    pub write_capacity: u32,
    pub direct_store_capacity: usize,
    pub misses: C220LsuMissConfig,
    pub stores: C220LsuStoreConfig,
    pub layout: C220CacheAddressLayout,
    pub partition_stack: bool,
    /// Refill UB responses into cache RAM. Existing valid tags still participate
    /// in lookup when refills are disabled.
    pub cache_ub: bool,
    pub ub_write_allocate: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220CoreLsuIssue {
    pub instruction_id: u64,
    pub tick: u64,
    pub operands: C220DirectStoreOperands,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220CoreLsuCompletion {
    pub issue: C220CoreLsuIssue,
    pub request: C220LsuRequestId,
    pub tick: u64,
    pub response_tick: u64,
    pub write: C220LsuWriteId,
}

pub(super) struct CoreLsu {
    commits: C220LsuCommitLane,
    stores: BTreeMap<C220LsuRequestId, C220CoreStoreIssue>,
    store_completions: Vec<C220CoreStoreCompletion>,
    ingress: VecDeque<DispatchedLsu>,
    admissions: Vec<C220CoreLsuAdmission>,
    scheduler: C220LsuRequestScheduler,
    config: C220CoreLsuConfig,
    port: u32,
    pending: BTreeMap<C220LsuRequestId, C220CoreLsuIssue>,
    send_pending: Option<C220LsuWriteRequest>,
    ub_writes: VecDeque<(u64, C220LsuWriteRequest)>,
    pub(super) next_tick: Option<u64>,
    completions: Vec<C220CoreLsuCompletion>,
    cache: Option<CoreCache>,
}

impl C220Core {
    /// Attach the scalar LSU to the core's configured shared BIU and memory service.
    pub fn configure_lsu(&mut self, config: C220CoreLsuConfig) -> Result<(), C220CoreError> {
        if self.lsu.is_some() {
            return Err(C220CoreError::LsuAlreadyConfigured);
        }
        if self
            .mte_pipeline
            .as_ref()
            .and_then(|p| p.timed_memory())
            .is_none()
        {
            return Err(C220CoreError::LsuUnconfigured);
        }
        let scheduler = C220LsuRequestScheduler::new(
            config.request_capacity,
            config.read_capacity,
            config.write_capacity,
            config.direct_store_capacity,
            C220LsuMissBuffer::new(config.misses).map_err(C220LsuSchedulerError::from)?,
            C220LsuStoreBuffer::new(config.stores).map_err(C220LsuSchedulerError::from)?,
        )?;
        let port = self.connect_cache_write_port()?;
        self.lsu = Some(CoreLsu {
            commits: C220LsuCommitLane::new(C220LoadCommitMode::Retirement),
            stores: BTreeMap::new(),
            store_completions: Vec::new(),
            ingress: VecDeque::new(),
            admissions: Vec::new(),
            scheduler,
            config,
            port,
            pending: BTreeMap::new(),
            send_pending: None,
            ub_writes: VecDeque::new(),
            next_tick: None,
            completions: Vec::new(),
            cache: None,
        });
        Ok(())
    }

    pub fn lsu_scheduler(&self) -> Option<&C220LsuRequestScheduler> {
        self.lsu.as_ref().map(|lsu| &lsu.scheduler)
    }

    pub fn lsu_config(&self) -> Option<C220CoreLsuConfig> {
        self.lsu.as_ref().map(|lsu| lsu.config)
    }

    pub fn pending_lsu_instructions(&self) -> impl Iterator<Item = &C220CoreLsuIssue> {
        self.lsu.iter().flat_map(|lsu| {
            lsu.pending
                .values()
                .chain(lsu.ingress.iter().filter_map(|entry| match entry {
                    DispatchedLsu::DirectStore(issue) => Some(issue),
                    _ => None,
                }))
        })
    }

    pub fn take_lsu_completions(&mut self) -> Vec<C220CoreLsuCompletion> {
        self.lsu
            .as_mut()
            .map(|lsu| std::mem::take(&mut lsu.completions))
            .unwrap_or_default()
    }

    pub(super) fn step_direct_store_at(
        &mut self,
        tick: u64,
        word: u32,
    ) -> Result<C220CoreStep, C220CoreError> {
        let pc = self.state.scalar().pc();
        if self.state.scalar().is_halted() {
            return Err(crate::sim::c220::state::C220ExecutionError::ProgramEnded { pc }.into());
        }
        let next_tick = tick.checked_add(1).ok_or(C220CoreError::TimeOverflow)?;
        if let Some(resume_tick) = [67, 68]
            .into_iter()
            .filter_map(|spr| self.scalar_timing.pending_spr_retirement(spr))
            .filter(|ready| *ready > tick)
            .max()
        {
            return Ok(C220CoreStep::Stalled(C220Stall {
                tick,
                pc,
                resume_tick,
                cause: C220StallCause::ScalarDependency,
            }));
        }
        let machine = self.state.scalar().machine();
        let operands = C220DirectStoreOperands::capture(machine, pc, word)
            .map_err(crate::sim::common::scalar::ScalarInstructionError::from)?;
        let lsu = self.lsu.as_mut().ok_or(C220CoreError::LsuUnconfigured)?;
        if lsu.ingress.len() == 2 {
            return Ok(C220CoreStep::Stalled(C220Stall {
                tick,
                pc,
                resume_tick: next_tick,
                cause: C220StallCause::LsuDependency,
            }));
        }
        let issue = C220CoreLsuIssue {
            instruction_id: self.next_instruction_id,
            tick,
            operands,
        };
        lsu.ingress.push_back(DispatchedLsu::DirectStore(issue));
        lsu.next_tick = Some(
            lsu.next_tick
                .map_or(next_tick, |prior| prior.min(next_tick)),
        );
        self.state.commit_c220_sequential_issue();
        Ok(C220CoreStep::Executed {
            tick,
            instruction: C220CoreInstruction::DirectStore(issue),
        })
    }

    pub(super) fn advance_lsu_at(&mut self, tick: u64) -> Result<(), C220CoreError> {
        let Some(lsu) = &mut self.lsu else {
            return Ok(());
        };
        if lsu.next_tick.is_none_or(|next| next > tick) {
            return Ok(());
        }
        let pipeline = self
            .mte_pipeline
            .as_mut()
            .ok_or(C220CoreError::LsuUnconfigured)?;
        lsu.retire_at(tick, self.state.scalar_mut().machine_mut())?;
        lsu.admit_ingress_at(
            tick,
            self.state.scalar().machine(),
            pipeline.ub_vector_subcore()
                != crate::sim::c220::mte::interface::biu_read::C220BiuSubcore::Cube,
        )?;
        lsu.receive_ub_write_at(tick, pipeline, &mut self.state.ub)?;
        let scheduler = &mut lsu.scheduler;
        scheduler
            .writes
            .advance_to(tick)
            .map_err(C220LsuSchedulerError::from)?;
        scheduler
            .reads
            .advance_to(tick)
            .map_err(C220LsuSchedulerError::from)?;
        if let Some(loads) = &mut lsu.cache {
            loads.receive_at(
                tick,
                pipeline,
                scheduler,
                &mut self.memory,
                &self.state.ub,
                lsu.config.cache_ub,
            )?;
            scheduler.deliver_values(
                tick,
                &mut lsu.commits,
                self.state.scalar_mut().machine_mut(),
            )?;
        }
        if let Some(C220MemoryWriteId::Cache { transaction, .. }) =
            pipeline.cache_write_completion(lsu.port)
        {
            let write = C220LsuWriteId::from_sequence(transaction);
            if scheduler.writes.request(write).is_some_and(|request| {
                scheduler.evicted_line(request.line.address).is_none()
                    && scheduler
                        .direct_stores()
                        .entry(request.line.address)
                        .is_some()
            }) {
                lsu.commits.check_retirement_send(tick)?;
            }
            let completion =
                scheduler.apply_external_write_response::<C220CoreError>(write, |key, bytes| {
                    self.memory.write_known_at(key.address, bytes)?;
                    Ok(())
                })?;
            pipeline.take_cache_write_completion(lsu.port)?;
            if let Some(C220LsuCompletion::Store(request)) = completion
                && lsu.pending.contains_key(&request)
            {
                lsu.commits.complete_direct_store_at(tick, request, write)?;
            }
        }
        scheduler.process_direct_stores(tick)?;
        if let Some(cache) = &mut lsu.cache {
            scheduler.process_stores(tick, &mut cache.cache, lsu.config.ub_write_allocate)?;
            scheduler.deliver_values(
                tick,
                &mut lsu.commits,
                self.state.scalar_mut().machine_mut(),
            )?;
        }
        let hazards = C220LsuExternalHazards {
            maintenance_active: false,
            maintenance_draining: false,
        };
        for stage in [C220LsuStage::M2, C220LsuStage::M1, C220LsuStage::M0] {
            if let Some(loads) = &mut lsu.cache {
                scheduler.advance_with_cache(stage, tick, hazards, &mut loads.cache)?;
            } else {
                scheduler.advance(stage, tick, hazards)?;
            }
        }
        scheduler.deliver_values(
            tick,
            &mut lsu.commits,
            self.state.scalar_mut().machine_mut(),
        )?;
        if let Some(loads) = &mut lsu.cache {
            loads.send_at(tick, pipeline, scheduler)?;
        }
        lsu.send_writes_at(tick, pipeline)?;
        let scheduler = &lsu.scheduler;
        lsu.next_tick = if lsu.ingress.is_empty()
            && lsu.pending.is_empty()
            && lsu.stores.is_empty()
            && lsu.commits.pending_count() == 0
            && lsu.commits.retirement_occupancy() == 0
            && lsu.cache.as_ref().is_none_or(CoreCache::is_idle)
            && scheduler.reads.requests().next().is_none()
            && scheduler.writes.requests().next().is_none()
            && lsu.send_pending.is_none()
            && lsu.ub_writes.is_empty()
        {
            None
        } else {
            Some(tick.checked_add(1).ok_or(C220CoreError::TimeOverflow)?)
        };
        Ok(())
    }
}
