use std::collections::BTreeMap;

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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220CoreLsuIssue {
    pub instruction_id: u64,
    pub request: C220LsuRequestId,
    pub tick: u64,
    pub operands: C220DirectStoreOperands,
    pub mapped: C220ScalarMappedAddress,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220CoreLsuCompletion {
    pub issue: C220CoreLsuIssue,
    pub tick: u64,
    pub write: C220LsuWriteId,
}

pub(super) struct CoreLsu {
    scheduler: C220LsuRequestScheduler,
    config: C220CoreLsuConfig,
    port: u32,
    pending: BTreeMap<C220LsuRequestId, C220CoreLsuIssue>,
    send_pending: Option<C220LsuWriteRequest>,
    pub(super) next_tick: Option<u64>,
    completions: Vec<C220CoreLsuCompletion>,
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
            scheduler,
            config,
            port,
            pending: BTreeMap::new(),
            send_pending: None,
            next_tick: None,
            completions: Vec::new(),
        });
        Ok(())
    }

    pub fn lsu_scheduler(&self) -> Option<&C220LsuRequestScheduler> {
        self.lsu.as_ref().map(|lsu| &lsu.scheduler)
    }

    pub fn pending_lsu_instructions(&self) -> impl Iterator<Item = &C220CoreLsuIssue> {
        self.lsu.iter().flat_map(|lsu| lsu.pending.values())
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
        let roots =
            machine
                .spr_value(67)
                .zip(machine.spr_value(68))
                .ok_or(C220CoreError::LsuAddress {
                    address: operands.effective_address,
                })?;
        let mapped = C220ScalarMappedAddress::decode(operands.effective_address, roots.0, roots.1)
            .ok_or(C220CoreError::LsuAddress {
                address: operands.effective_address,
            })?;
        let lsu = self.lsu.as_mut().ok_or(C220CoreError::LsuUnconfigured)?;
        let Some(request) = lsu.scheduler.admit_direct_store(
            tick,
            operands,
            mapped,
            lsu.config.partition_stack,
            lsu.config.layout,
        )?
        else {
            return Ok(C220CoreStep::Stalled(C220Stall {
                tick,
                pc,
                resume_tick: next_tick,
                cause: C220StallCause::LsuDependency,
            }));
        };
        let issue = C220CoreLsuIssue {
            instruction_id: self.next_instruction_id,
            request,
            tick,
            operands,
            mapped,
        };
        lsu.pending.insert(request, issue);
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
        let scheduler = &mut lsu.scheduler;
        scheduler
            .writes
            .advance_to(tick)
            .map_err(C220LsuSchedulerError::from)?;
        if let Some(C220MemoryWriteId::Cache { transaction, .. }) =
            pipeline.cache_write_completion(lsu.port)
        {
            let write = C220LsuWriteId::from_sequence(transaction);
            let completion =
                scheduler.apply_external_write_response::<C220CoreError>(write, |key, bytes| {
                    self.memory.write_known_at(key.address, bytes)?;
                    Ok(())
                })?;
            pipeline.take_cache_write_completion(lsu.port)?;
            if let Some(C220LsuCompletion::Store(request)) = completion
                && let Some(issue) = lsu.pending.remove(&request)
            {
                lsu.completions
                    .push(C220CoreLsuCompletion { issue, tick, write });
            }
        }
        scheduler.process_direct_stores(tick)?;
        let hazards = C220LsuExternalHazards {
            maintenance_active: false,
            maintenance_draining: false,
        };
        for stage in [C220LsuStage::M2, C220LsuStage::M1, C220LsuStage::M0] {
            scheduler.advance(stage, tick, hazards)?;
        }
        if lsu.send_pending.is_none() {
            let ready = pipeline
                .biu_bus_writes()
                .is_some_and(|bus| bus.cache_can_send(lsu.port));
            lsu.send_pending = scheduler
                .writes
                .dispatch_clock(tick, false, ready)
                .map_err(C220LsuSchedulerError::from)?
                .into_iter()
                .next();
        }
        if let Some(request) = lsu.send_pending {
            let bytes =
                u32::try_from(request.byte_len).map_err(|_| C220CoreError::InvalidCacheWrite)?;
            if pipeline.send_cache_write(C220MemoryWriteCommand {
                ready_tick: tick,
                tag: C220MemoryWriteId::Cache {
                    port: lsu.port,
                    transaction: request.id.sequence(),
                },
                address: request.line.address,
                bytes,
            })? {
                lsu.send_pending = None;
            }
        }
        lsu.next_tick = if lsu.pending.is_empty()
            && scheduler.writes.requests().next().is_none()
            && lsu.send_pending.is_none()
        {
            None
        } else {
            Some(tick.checked_add(1).ok_or(C220CoreError::TimeOverflow)?)
        };
        Ok(())
    }
}
