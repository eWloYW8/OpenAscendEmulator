use super::super::cache::{C220AtomicCacheHit, C220CacheLocation, C220DataCache};
use super::super::store_buffer::C220LsuMemory;
use super::*;
use crate::sim::c220::scalar::C220AtomicStoreOperands;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220LsuAtomicCompletion {
    pub request: C220LsuRequestId,
    pub tick: u64,
    pub operands: C220AtomicStoreOperands,
    pub cache_hit: Option<C220AtomicCacheHit>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PendingAtomic {
    operands: C220AtomicStoreOperands,
    line: C220LsuLineKey,
    offset: usize,
    lookup: Option<Option<C220CacheLocation>>,
}

impl C220LsuRequestScheduler {
    /// Admit captured store data. Functional backing-memory arithmetic must be
    /// executed separately; cache timing never repeats the atomic addition.
    pub fn admit_atomic_store(
        &mut self,
        tick: u64,
        operands: C220AtomicStoreOperands,
    ) -> Result<Option<C220LsuRequestId>, C220LsuSchedulerError> {
        let address = operands.effective_address & 0x0000_ffff_ffff_ffff;
        let offset = (address & 63) as usize;
        if self.stores.line_bytes() != 64
            || !operands.is_external()
            || offset + operands.bytes().len() > 64
        {
            return Err(C220LsuSchedulerError::UnsupportedStoreAccess);
        }
        let line = C220LsuLineKey {
            address: address & !63,
            memory: C220LsuMemory::External,
        };
        let Some((id, _)) = self.admit(
            tick,
            C220LsuRequest {
                line,
                access: C220LsuAccess::AtomicStore,
            },
            None,
        )?
        else {
            return Ok(None);
        };
        self.pending_atomics.insert(
            id,
            PendingAtomic {
                operands,
                line,
                offset,
                lookup: None,
            },
        );
        Ok(Some(id))
    }

    pub fn take_atomic_completions(&mut self) -> Vec<C220LsuAtomicCompletion> {
        std::mem::take(&mut self.atomic_completions)
    }

    pub(super) fn validate_atomic_m2(&self, tick: u64) -> Result<(), C220LsuSchedulerError> {
        if let Some(pending) = self
            .pipeline
            .head(C220LsuStage::M2)
            .filter(|head| head.ready_tick <= tick)
            .and_then(|head| self.pending_atomics.get(&head.request))
        {
            pending
                .lookup
                .ok_or(C220LsuSchedulerError::MissingStoreLookup)?;
            if self.stores.full(pending.line)
                || self
                    .stores
                    .entry(pending.line)
                    .is_some_and(|entry| entry.forbidden())
            {
                return Err(C220LsuStoreError::Blocked.into());
            }
            tick.checked_add(1).ok_or(EventError::TimeOverflow)?;
        }
        Ok(())
    }

    pub(super) fn advance_atomic_data(
        &mut self,
        stage: C220LsuStage,
        tick: u64,
        id: C220LsuRequestId,
        cache: &mut C220DataCache,
    ) -> Result<(), C220LsuSchedulerError> {
        let Some(pending) = self.pending_atomics.get_mut(&id) else {
            return Ok(());
        };
        match stage {
            C220LsuStage::M0 => {
                let hit = cache.lookup_partitioned(
                    pending.line.address,
                    pending.operands.effective_address,
                    pending.line.memory,
                );
                pending.lookup = Some(hit.or(pending.lookup.flatten()));
            }
            C220LsuStage::M1 => {}
            C220LsuStage::M2 => {
                self.stores.store(
                    pending.line,
                    id,
                    pending.offset,
                    pending.operands.bytes(),
                    pending.lookup.flatten().is_some(),
                )?;
                let next = tick.checked_add(1).ok_or(EventError::TimeOverflow)?;
                self.store_next_tick =
                    Some(self.store_next_tick.map_or(next, |prior| prior.min(next)));
            }
        }
        Ok(())
    }

    pub(super) fn flush_atomic_store(
        &mut self,
        tick: u64,
        key: C220LsuLineKey,
        cache: &mut C220DataCache,
    ) -> Result<(), C220LsuSchedulerError> {
        let entry = self
            .stores
            .entry(key)
            .ok_or(C220LsuStoreError::MissingEntry)?;
        let pending = self.pending_atomics[&entry.requests()[0]];
        let hit = if entry.cache_hit() {
            let location = pending
                .lookup
                .flatten()
                .ok_or(C220LsuSchedulerError::MissingStoreLookup)?;
            Some(entry.write_atomic_hit(cache, location)?)
        } else {
            cache.install_atomic_line(
                key.address,
                pending.operands.effective_address,
                entry.bytes(),
            )?;
            None
        };
        let entry = self.stores.remove(key).expect("validated atomic entry");
        for &request in entry.requests() {
            self.finish_atomic_store(request, tick, hit);
            self.finish_store(request, tick, C220LsuStorePath::Cache);
        }
        Ok(())
    }

    pub(super) fn finish_atomic_store(
        &mut self,
        request: C220LsuRequestId,
        tick: u64,
        cache_hit: Option<C220AtomicCacheHit>,
    ) {
        if let Some(pending) = self.pending_atomics.remove(&request) {
            self.atomic_completions.push(C220LsuAtomicCompletion {
                request,
                tick,
                operands: pending.operands,
                cache_hit,
            });
        }
    }
}
