use super::super::cache::{C220CacheRefill, C220DataCache};
use super::super::miss_buffer::{C220LsuMissError, C220LsuMissState};
use super::super::store_buffer::{
    C220LsuCompletion, C220LsuLineKey, C220LsuMemory, C220LsuStoreError, C220LsuStoreState,
};
use super::{C220LsuRequestScheduler, C220LsuSchedulerError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220LsuReadCompletion {
    pub key: C220LsuLineKey,
    /// Data for completed loads: unmodified for miss-owned responses, merged
    /// with store bytes for store-owned responses.
    pub load_line: Vec<u8>,
    pub notifications: Vec<C220LsuCompletion>,
    pub pending_ub_write: Option<Vec<u8>>,
    pub mark_cache_dirty: bool,
}

impl C220LsuRequestScheduler {
    /// Refill from merged store data, then release external requests or retain
    /// UB requests until their write acknowledgement. Dirty writebacks and UB
    /// follow-up writes are queued automatically; returned data is diagnostic.
    pub fn complete_cached_store_read(
        &mut self,
        key: C220LsuLineKey,
        partition_address: u64,
        returned_line: &[u8],
        cache: &mut C220DataCache,
    ) -> Result<(C220LsuReadCompletion, C220CacheRefill), C220LsuSchedulerError> {
        self.validate_store_read(key, returned_line.len(), Some(cache.line_bytes()))?;
        self.writes.check_enqueue(
            u64::from(cache.replacement_requires_writeback(key.address, partition_address))
                + u64::from(key.memory == C220LsuMemory::Ub),
        )?;
        let mut merged = returned_line.to_vec();
        self.stores
            .entry(key)
            .expect("validated store response")
            .write_valid_bytes(&mut merged)?;
        let refill = cache.refill(
            key.address,
            partition_address,
            key.memory,
            &merged,
            key.memory == C220LsuMemory::External,
        )?;
        self.queue_refill_writeback(&refill)?;
        let completion =
            self.complete_store_read(key, returned_line, Some(cache.line_mut(refill.location)?))?;
        Ok((completion, refill))
    }

    /// Refill cache RAM and complete a cacheable MSHR-owned response together.
    /// Dirty writebacks and UB follow-up writes are queued automatically.
    pub fn complete_cached_miss_read(
        &mut self,
        key: C220LsuLineKey,
        partition_address: u64,
        returned_line: &[u8],
        cache: &mut C220DataCache,
    ) -> Result<(C220LsuReadCompletion, C220CacheRefill), C220LsuSchedulerError> {
        self.validate_miss_read(key, returned_line.len(), Some(cache.line_bytes()))?;
        self.writes.check_enqueue(
            u64::from(cache.replacement_requires_writeback(key.address, partition_address))
                + u64::from(key.memory == C220LsuMemory::Ub && self.stores.entry(key).is_some()),
        )?;
        let refill = cache.refill(
            key.address,
            partition_address,
            key.memory,
            returned_line,
            false,
        )?;
        self.queue_refill_writeback(&refill)?;
        let completion =
            self.complete_miss_read(key, returned_line, Some(cache.line_mut(refill.location)?))?;
        if completion.mark_cache_dirty {
            cache.mark_dirty(refill.location)?;
        }
        Ok((completion, refill))
    }

    fn validate_miss_read(
        &self,
        key: C220LsuLineKey,
        returned_bytes: usize,
        cache_bytes: Option<usize>,
    ) -> Result<(), C220LsuSchedulerError> {
        let width = self.misses.line_bytes();
        if self.stores.line_bytes() != width {
            return Err(C220LsuSchedulerError::LineGeometry);
        }
        if returned_bytes != width || cache_bytes.is_some_and(|bytes| bytes != width) {
            return Err(C220LsuMissError::InvalidLineSize.into());
        }
        if key.memory == C220LsuMemory::External && cache_bytes.is_none() {
            return Err(C220LsuSchedulerError::MissingCacheLine);
        }
        if self
            .misses
            .entry(key)
            .ok_or(C220LsuMissError::MissingEntry)?
            .state()
            != C220LsuMissState::Fetching
        {
            return Err(C220LsuSchedulerError::UnexpectedReadResponse);
        }
        Ok(())
    }

    /// Complete an MSHR-owned read after cache refill. `cache_line` is required
    /// for external memory and present for cacheable UB. UB follow-up writes are
    /// queued automatically. The caller delivers notifications through the timed
    /// completion port. Tag allocation and response ownership are caller-owned.
    pub fn complete_miss_read(
        &mut self,
        key: C220LsuLineKey,
        returned_line: &[u8],
        mut cache_line: Option<&mut [u8]>,
    ) -> Result<C220LsuReadCompletion, C220LsuSchedulerError> {
        self.validate_miss_read(
            key,
            returned_line.len(),
            cache_line.as_ref().map(|line| line.len()),
        )?;
        if key.memory == C220LsuMemory::Ub && self.stores.entry(key).is_some() {
            self.writes.check_enqueue(1)?;
        }
        let entry = self
            .misses
            .entry(key)
            .ok_or(C220LsuMissError::MissingEntry)?;
        let load_line = if key.memory == C220LsuMemory::External {
            cache_line
                .as_deref()
                .expect("validated external cache line")
                .to_vec()
        } else {
            returned_line.to_vec()
        };
        let mut notifications: Vec<_> = entry
            .requests()
            .iter()
            .copied()
            .map(C220LsuCompletion::Load)
            .collect();
        self.misses.receive_line(key, returned_line)?;
        let mut pending_ub_write = None;
        let mut mark_cache_dirty = false;
        if let Some(store) = self.stores.entry(key) {
            if let Some(line) = cache_line.as_mut() {
                store.write_valid_bytes(line)?;
            }
            self.stores.set_state(key, C220LsuStoreState::Ready)?;
            if key.memory == C220LsuMemory::External {
                mark_cache_dirty = true;
                notifications.extend(
                    self.stores
                        .remove(key)
                        .expect("linked store exists")
                        .requests()
                        .iter()
                        .copied()
                        .map(C220LsuCompletion::Store),
                );
            } else {
                self.stores.merge_line(key, returned_line)?;
                pending_ub_write = Some(
                    self.stores
                        .entry(key)
                        .expect("linked store exists")
                        .bytes()
                        .to_vec(),
                );
                self.writes.enqueue(key)?;
            }
        }
        self.misses.remove(key);
        self.resolve_values(self.reads.tick(), key, &notifications, &load_line);
        Ok(C220LsuReadCompletion {
            key,
            load_line,
            notifications,
            pending_ub_write,
            mark_cache_dirty,
        })
    }

    /// Complete an STB-owned read into an allocated cache line when cacheable.
    /// UB stores and their linked loads remain pending until the UB write reply.
    pub fn complete_store_read(
        &mut self,
        key: C220LsuLineKey,
        returned_line: &[u8],
        cache_line: Option<&mut [u8]>,
    ) -> Result<C220LsuReadCompletion, C220LsuSchedulerError> {
        self.validate_store_read(
            key,
            returned_line.len(),
            cache_line.as_ref().map(|line| line.len()),
        )?;
        if key.memory == C220LsuMemory::Ub {
            self.writes.check_enqueue(1)?;
        }
        self.stores.merge_line(key, returned_line)?;
        self.stores.set_state(key, C220LsuStoreState::Ready)?;
        self.stores.set_forbidden(key, true)?;
        let entry = self.stores.entry(key).expect("validated store response");
        let load_line = entry.bytes().to_vec();
        if let Some(line) = cache_line {
            line.copy_from_slice(&load_line);
        }
        let mut notifications = Vec::new();
        let pending_ub_write = if key.memory == C220LsuMemory::Ub {
            self.writes.enqueue(key)?;
            Some(load_line.clone())
        } else {
            notifications.extend(
                entry
                    .requests()
                    .iter()
                    .copied()
                    .map(C220LsuCompletion::Store),
            );
            if let Some(miss) = self.misses.entry(key) {
                notifications.extend(miss.requests().iter().copied().map(C220LsuCompletion::Load));
            }
            self.stores.remove(key);
            self.misses.remove(key);
            None
        };
        self.resolve_values(self.reads.tick(), key, &notifications, &load_line);
        Ok(C220LsuReadCompletion {
            key,
            load_line,
            notifications,
            pending_ub_write,
            mark_cache_dirty: key.memory == C220LsuMemory::External,
        })
    }

    fn validate_store_read(
        &self,
        key: C220LsuLineKey,
        returned_bytes: usize,
        cache_bytes: Option<usize>,
    ) -> Result<(), C220LsuSchedulerError> {
        let width = self.stores.line_bytes();
        if self.misses.line_bytes() != width {
            return Err(C220LsuSchedulerError::LineGeometry);
        }
        if returned_bytes != width || cache_bytes.is_some_and(|bytes| bytes != width) {
            return Err(C220LsuMissError::InvalidLineSize.into());
        }
        if key.memory == C220LsuMemory::External && cache_bytes.is_none() {
            return Err(C220LsuSchedulerError::MissingCacheLine);
        }
        if self
            .stores
            .entry(key)
            .ok_or(C220LsuStoreError::MissingEntry)?
            .state()
            != C220LsuStoreState::Fetching
        {
            return Err(C220LsuSchedulerError::UnexpectedStoreResponse);
        }
        Ok(())
    }

    pub(super) fn resolve_values(
        &mut self,
        tick: u64,
        key: C220LsuLineKey,
        notifications: &[C220LsuCompletion],
        line: &[u8],
    ) {
        for notification in notifications {
            let single = std::slice::from_ref(notification);
            match notification {
                C220LsuCompletion::Load(_) => self.resolve_load_values(tick, key, single, line),
                C220LsuCompletion::Store(_) => self.resolve_store_values(tick, single),
            }
        }
    }
}
