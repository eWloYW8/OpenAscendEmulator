use super::*;
use crate::memory::mapped::MappedMemory;
use crate::sim::c220::memory::biu_read::{C220BiuReadCacheConfig, C220BiuReadCacheKind};
use crate::sim::c220::memory::timed_memory::{C220MemoryReadCommand, C220MemoryReadId};
use crate::sim::c220::mte::C220MtePipeline;
use crate::sim::c220::scalar::C220LoadOperands;
use crate::sim::c220::scalar::lsu::cache::C220DataCache;
use crate::sim::c220::scalar::lsu::load_commit::{
    C220LoadCommitLane, C220LoadCommitMode, C220LoadId, C220LoadRetirement,
};
use crate::sim::c220::scalar::lsu::read_queue::{C220LsuReadId, C220LsuReadRequest};
use crate::sim::common::scalar::ScalarMachine;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220CoreLoadIssue {
    pub instruction_id: u64,
    pub tick: u64,
    pub operands: C220LoadOperands,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220CoreLoadCompletion {
    pub issue: C220CoreLoadIssue,
    pub retirement: C220LoadRetirement,
}

pub(super) struct CoreLoads {
    pub(super) cache: C220DataCache,
    pub(super) commits: C220LoadCommitLane,
    port: u32,
    pending: BTreeMap<C220LoadId, C220CoreLoadIssue>,
    send_pending: Option<C220LsuReadRequest>,
    completions: Vec<C220CoreLoadCompletion>,
}

impl C220Core {
    pub fn configure_lsu_loads(
        &mut self,
        cache: C220DataCache,
        mode: C220LoadCommitMode,
        port: C220BiuReadCacheConfig,
    ) -> Result<(), C220CoreError> {
        let lsu = self.lsu.as_ref().ok_or(C220CoreError::LsuUnconfigured)?;
        if lsu.loads.is_some() || lsu.next_tick.is_some() {
            return Err(C220CoreError::LsuAlreadyConfigured);
        }
        if cache.line_bytes() != lsu.config.misses.line_bytes || cache.layout() != lsu.config.layout
        {
            return Err(C220LsuSchedulerError::LineGeometry.into());
        }
        if cache.line_bytes() > 128 {
            return Err(C220CoreError::UnsupportedTimedLsuAccess);
        }
        let port = self.connect_cache_read_port(port)?;
        self.lsu.as_mut().expect("configured LSU").loads = Some(CoreLoads {
            cache,
            commits: C220LoadCommitLane::new(mode),
            port,
            pending: BTreeMap::new(),
            send_pending: None,
            completions: Vec::new(),
        });
        Ok(())
    }

    pub fn data_cache(&self) -> Option<&C220DataCache> {
        self.lsu.as_ref()?.loads.as_ref().map(|loads| &loads.cache)
    }

    pub fn pending_load_instructions(&self) -> impl Iterator<Item = &C220CoreLoadIssue> {
        self.lsu
            .iter()
            .flat_map(|lsu| lsu.loads.iter())
            .flat_map(|loads| loads.pending.values())
    }

    pub fn take_load_completions(&mut self) -> Vec<C220CoreLoadCompletion> {
        self.lsu
            .as_mut()
            .and_then(|lsu| lsu.loads.as_mut())
            .map(|loads| std::mem::take(&mut loads.completions))
            .unwrap_or_default()
    }

    pub(in crate::sim::c220::core) fn load_dependency_tick(
        &self,
        word: u32,
        tick: u64,
    ) -> Option<u64> {
        let loads = self.lsu.as_ref().and_then(|lsu| lsu.loads.as_ref());
        self.scalar_timing
            .dependency_tick_with_loads(word, tick, |register| {
                loads.is_some_and(|loads| loads.commits.pending_destination(register).is_some())
            })
    }

    pub(in crate::sim::c220::core) fn supersede_load_destination(&mut self, register: u8) {
        if let Some(loads) = self.lsu.as_mut().and_then(|lsu| lsu.loads.as_mut()) {
            loads.commits.supersede(register);
        }
    }

    pub(in crate::sim::c220::core) fn step_load_at(
        &mut self,
        tick: u64,
        word: u32,
    ) -> Result<C220CoreStep, C220CoreError> {
        let pc = self.state.scalar().pc();
        if self.state.scalar().is_halted() {
            return Err(crate::sim::c220::state::C220ExecutionError::ProgramEnded { pc }.into());
        }
        let next_tick = tick.checked_add(1).ok_or(C220CoreError::TimeOverflow)?;
        let machine = self.state.scalar().machine();
        let operands = C220LoadOperands::capture(machine, pc, word)
            .map_err(crate::sim::common::scalar::ScalarInstructionError::from)?;
        if let Some(resume_tick) = [67, 68]
            .into_iter()
            .filter_map(|spr| self.scalar_timing.pending_spr_retirement(spr))
            .chain(
                self.scalar_timing
                    .load_waw_tick(operands.destination_register),
            )
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
        let lsu = self.lsu.as_mut().ok_or(C220CoreError::LsuUnconfigured)?;
        let loads = lsu.loads.as_mut().ok_or(C220CoreError::LsuUnconfigured)?;
        if lsu.ingress.len() == 2 {
            return Ok(C220CoreStep::Stalled(C220Stall {
                tick,
                pc,
                resume_tick: next_tick,
                cause: C220StallCause::LsuDependency,
            }));
        }
        let instruction = C220LoadId(self.next_instruction_id);
        loads.commits.issue(
            tick,
            instruction,
            operands,
            self.state.scalar_mut().machine_mut(),
        )?;
        self.scalar_timing
            .supersede_destination(operands.destination_register);
        let issue = C220CoreLoadIssue {
            instruction_id: self.next_instruction_id,
            tick,
            operands,
        };
        loads.pending.insert(instruction, issue);
        lsu.ingress.push_back(DispatchedLsu::Load(issue));
        lsu.next_tick = Some(
            lsu.next_tick
                .map_or(next_tick, |prior| prior.min(next_tick)),
        );
        self.state.commit_c220_sequential_issue();
        Ok(C220CoreStep::Executed {
            tick,
            instruction: C220CoreInstruction::Load(issue),
        })
    }
}

impl CoreLoads {
    pub(super) fn is_idle(&self) -> bool {
        self.pending.is_empty() && self.send_pending.is_none()
    }

    pub(super) fn receive_at(
        &mut self,
        tick: u64,
        pipeline: &mut C220MtePipeline,
        scheduler: &mut C220LsuRequestScheduler,
        memory: &mut MappedMemory,
        machine: &mut ScalarMachine,
    ) -> Result<(), C220CoreError> {
        if let Some(retirement) = self.commits.retire_next_at(tick, machine)? {
            let issue = self
                .pending
                .remove(&retirement.instruction)
                .expect("issued load");
            self.completions
                .push(C220CoreLoadCompletion { issue, retirement });
        }
        let Some(response) = pipeline
            .biu_bus_reads()
            .and_then(|bus| bus.cache_returns(C220BiuReadCacheKind::Data, self.port))
            .and_then(|returns| returns.front())
            .filter(|response| response.ready_tick <= tick)
            .copied()
        else {
            return Ok(());
        };
        let C220MemoryReadId::DataCache { transaction, .. } = response.beat.tag else {
            return Err(C220CoreError::InvalidCacheRead);
        };
        if response.beat.transaction_id != 0 {
            return Err(C220CoreError::InvalidCacheRead);
        }
        scheduler.apply_cached_read_response::<C220CoreError>(
            C220LsuReadId::from_sequence(transaction),
            &mut self.cache,
            |key, size| Ok(memory.read_known_at(key.address, size)?),
        )?;
        let consumed = pipeline.take_cache_read_return(C220BiuReadCacheKind::Data, self.port);
        debug_assert_eq!(consumed, Some(response.beat));
        Ok(())
    }

    pub(super) fn send_at(
        &mut self,
        tick: u64,
        pipeline: &mut C220MtePipeline,
        scheduler: &mut C220LsuRequestScheduler,
        machine: &mut ScalarMachine,
    ) -> Result<(), C220CoreError> {
        scheduler.deliver_load_values(tick, &mut self.commits, machine)?;
        if self.send_pending.is_none() {
            let ready = pipeline
                .biu_bus_reads()
                .is_some_and(|bus| bus.cache_can_send(C220BiuReadCacheKind::Data, self.port));
            self.send_pending = scheduler
                .reads
                .dispatch_clock(tick, false, ready)
                .map_err(C220LsuSchedulerError::from)?
                .into_iter()
                .next();
        }
        if let Some(request) = self.send_pending
            && pipeline.send_cache_read(C220MemoryReadCommand {
                ready_tick: tick,
                tag: C220MemoryReadId::DataCache {
                    port: self.port,
                    transaction: request.id.sequence(),
                },
                address: request.line.address,
                bytes: request.byte_len as u32,
            })?
        {
            self.send_pending = None;
        }
        Ok(())
    }
}
