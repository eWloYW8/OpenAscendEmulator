use super::super::cache::{C220CacheLocation, C220DataCache};
use super::super::store_buffer::{C220LsuCompletion, C220LsuMemory};
use super::*;
use crate::sim::c220::scalar::{C220ScalarMappedAddress, C220StoreOperands};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220LsuStorePath {
    Cache,
    Refill,
}

/// Store bytes have reached cache RAM; retirement transport remains separate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220LsuStoreValue {
    pub request: C220LsuRequestId,
    pub tick: u64,
    pub operands: C220StoreOperands,
    pub mapped: C220ScalarMappedAddress,
    pub path: C220LsuStorePath,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PendingStore {
    operands: C220StoreOperands,
    mapped: C220ScalarMappedAddress,
    line: C220LsuLineKey,
    partition_address: u64,
    offset: usize,
    lookup: Option<Option<C220CacheLocation>>,
}

impl C220LsuRequestScheduler {
    pub fn admit_store(
        &mut self,
        tick: u64,
        operands: C220StoreOperands,
        mapped: C220ScalarMappedAddress,
        partition_stack: bool,
    ) -> Result<Option<C220LsuRequestId>, C220LsuSchedulerError> {
        let size = self.stores.line_bytes() as u64;
        let address = mapped.address & 0x0000_ffff_ffff_ffff;
        let offset = (address % size) as usize;
        if ![1, 2, 4, 8].contains(&operands.width_bytes)
            || mapped.memory != C220LsuMemory::External
            || offset + usize::from(operands.width_bytes) > size as usize
        {
            return Err(C220LsuSchedulerError::UnsupportedStoreAccess);
        }
        let line = C220LsuLineKey {
            address: address / size * size,
            memory: mapped.memory,
        };
        let Some((id, _)) = self.admit(
            tick,
            C220LsuRequest {
                line,
                access: C220LsuAccess::Store,
            },
            None,
        )?
        else {
            return Ok(None);
        };
        self.pending_stores.insert(
            id,
            PendingStore {
                operands,
                mapped,
                line,
                partition_address: mapped.cache_address(partition_stack),
                offset,
                lookup: None,
            },
        );
        Ok(Some(id))
    }

    pub fn take_store_values(&mut self) -> Vec<C220LsuStoreValue> {
        std::mem::take(&mut self.store_values)
    }

    pub fn pending_store(&self, request: C220LsuRequestId) -> Option<&C220StoreOperands> {
        self.pending_stores
            .get(&request)
            .map(|pending| &pending.operands)
    }

    pub fn next_store_tick(&self) -> Option<u64> {
        self.store_next_tick
    }

    pub(super) fn validate_store_m2(&self, tick: u64) -> Result<(), C220LsuSchedulerError> {
        if let Some(pending) = self
            .pipeline
            .head(C220LsuStage::M2)
            .filter(|head| head.ready_tick <= tick)
            .and_then(|head| self.pending_stores.get(&head.request))
        {
            pending
                .lookup
                .ok_or(C220LsuSchedulerError::MissingStoreLookup)?;
            tick.checked_add(1).ok_or(EventError::TimeOverflow)?;
        }
        Ok(())
    }

    pub(super) fn advance_store_data(
        &mut self,
        stage: C220LsuStage,
        tick: u64,
        id: C220LsuRequestId,
        cache: &mut C220DataCache,
    ) -> Result<(), C220LsuSchedulerError> {
        let Some(pending) = self.pending_stores.get_mut(&id) else {
            return Ok(());
        };
        match stage {
            C220LsuStage::M0 => {
                let location = cache.lookup_partitioned(
                    pending.line.address,
                    pending.partition_address,
                    pending.line.memory,
                );
                pending.lookup = Some(location.or(pending.lookup.flatten()));
            }
            C220LsuStage::M1 => {}
            C220LsuStage::M2 => {
                self.stores.store(
                    pending.line,
                    id,
                    pending.offset,
                    pending.operands.bytes(),
                    pending.lookup.expect("validated lookup").is_some(),
                )?;
                let next = tick.checked_add(1).ok_or(EventError::TimeOverflow)?;
                self.store_next_tick =
                    Some(self.store_next_tick.map_or(next, |prior| prior.min(next)));
            }
        }
        Ok(())
    }

    /// Process one store-buffer clock. Call at each `next_store_tick`, before
    /// newly arriving M2 stores, so insertion does not consume a timeout tick.
    pub fn process_stores(
        &mut self,
        tick: u64,
        cache: &mut C220DataCache,
    ) -> Result<(), C220LsuSchedulerError> {
        self.reads.check_tick(tick)?;
        if self.store_next_tick.is_none_or(|next| next > tick) {
            return Ok(());
        }
        if cache.line_bytes() != self.stores.line_bytes() {
            return Err(C220LsuSchedulerError::LineGeometry);
        }
        let next = tick.checked_add(1).ok_or(EventError::TimeOverflow)?;
        self.reads.advance_to(tick)?;
        self.stores.advance_timeouts();
        let ready: Vec<_> = self
            .stores
            .ready_to_flush()
            .map(|entry| entry.key())
            .collect();
        for key in ready {
            let entry = self.stores.entry(key).expect("ready store");
            let first = entry.requests()[0];
            let pending = *self
                .pending_stores
                .get(&first)
                .ok_or(C220LsuSchedulerError::MissingStoreLookup)?;
            if entry.cache_hit() {
                let data_location = cache
                    .find_way(key.address, key.memory)
                    .ok_or(C220LsuSchedulerError::MissingCacheLine)?;
                let dirty_location = pending
                    .lookup
                    .flatten()
                    .ok_or(C220LsuSchedulerError::MissingStoreLookup)?;
                entry.write_valid_bytes(cache.line_mut(data_location)?)?;
                cache.mark_dirty(dirty_location)?;
                let entry = self.stores.remove(key).expect("flushed store");
                for request in entry.requests() {
                    self.finish_store(*request, tick, C220LsuStorePath::Cache);
                }
            } else {
                self.enqueue_store_read(key, pending.partition_address)?;
            }
        }
        self.store_next_tick = (!self.stores.entries().is_empty()).then_some(next);
        Ok(())
    }

    fn finish_store(&mut self, request: C220LsuRequestId, tick: u64, path: C220LsuStorePath) {
        if let Some(pending) = self.pending_stores.remove(&request) {
            self.store_values.push(C220LsuStoreValue {
                request,
                tick,
                operands: pending.operands,
                mapped: pending.mapped,
                path,
            });
        }
    }

    pub(super) fn resolve_store_values(&mut self, tick: u64, notifications: &[C220LsuCompletion]) {
        for notification in notifications {
            if let C220LsuCompletion::Store(request) = notification {
                self.finish_store(*request, tick, C220LsuStorePath::Refill);
            }
        }
    }
}
