use super::super::cache::{C220CacheRefill, C220DataCache};
use super::super::store_buffer::C220LsuCompletion;
use super::super::store_buffer::{C220LsuLineKey, C220LsuMemory, C220LsuWriteCompletion};
use super::super::write_queue::{C220LsuWriteError, C220LsuWriteId, C220LsuWriteState};
use super::{C220LsuRequestScheduler, C220LsuSchedulerError};

impl C220LsuRequestScheduler {
    /// Complete an ordinary external write. Eviction data takes precedence over
    /// direct stores at the same address. Maintenance replies use a separate path.
    pub fn complete_external_write_response(
        &mut self,
        id: C220LsuWriteId,
        backing_line: &mut [u8],
    ) -> Result<Option<C220LsuCompletion>, C220LsuSchedulerError> {
        self.apply_external_write_response(id, |_, bytes| {
            if bytes.len() != backing_line.len() {
                return Err(super::super::cache::C220CacheError::InvalidLineSize.into());
            }
            backing_line.copy_from_slice(bytes);
            Ok(())
        })
    }

    /// Release transport credit before writing, then notify and remove data
    /// only after the sink succeeds. A failed sink retains a retryable response.
    pub fn apply_external_write_response<E>(
        &mut self,
        id: C220LsuWriteId,
        write: impl FnOnce(C220LsuLineKey, &[u8]) -> Result<(), E>,
    ) -> Result<Option<C220LsuCompletion>, E>
    where
        E: From<C220LsuSchedulerError>,
    {
        let key = self
            .writes
            .request(id)
            .ok_or(C220LsuSchedulerError::Write(
                C220LsuWriteError::MissingRequest,
            ))?
            .line;
        if key.memory != C220LsuMemory::External {
            return Err(C220LsuSchedulerError::MissingWriteData.into());
        }
        self.enter_write_response(id)?;
        let completion = if let Some(bytes) = self.eviction_data.get(&key.address) {
            write(key, bytes)?;
            self.eviction_data.remove(&key.address);
            None
        } else {
            let entry = self
                .direct_stores
                .entry(key.address)
                .ok_or(C220LsuSchedulerError::MissingWriteData)?;
            write(key, entry.bytes())?;
            let request = self
                .direct_stores
                .remove(key.address)
                .map_err(C220LsuSchedulerError::from)?;
            Some(C220LsuCompletion::Store(request))
        };
        self.writes
            .finish_response(id)
            .map_err(C220LsuSchedulerError::from)?;
        Ok(completion)
    }

    /// UB replies without a matching store entry write the current cache line,
    /// rather than a snapshot captured when the request was issued.
    pub fn complete_ub_write_response(
        &mut self,
        id: C220LsuWriteId,
        write_allocate: bool,
        backing_line: &mut [u8],
        cache: &C220DataCache,
    ) -> Result<Option<C220LsuWriteCompletion>, C220LsuSchedulerError> {
        let key = self
            .writes
            .request(id)
            .ok_or(C220LsuWriteError::MissingRequest)?
            .line;
        if key.memory != C220LsuMemory::Ub {
            return Err(C220LsuSchedulerError::MissingWriteData);
        }
        if self.stores.entry(key).is_some() {
            return self
                .complete_ub_store_write(id, write_allocate, backing_line)
                .map(Some);
        }
        let location = cache
            .find_way(key.address, key.memory)
            .ok_or(C220LsuSchedulerError::MissingWriteData)?;
        let bytes = cache.line(location)?;
        if bytes.len() != backing_line.len() {
            return Err(super::super::cache::C220CacheError::InvalidLineSize.into());
        }
        self.enter_write_response(id)?;
        backing_line.copy_from_slice(bytes);
        self.writes.finish_response(id)?;
        Ok(None)
    }

    pub fn evicted_line(&self, address: u64) -> Option<&[u8]> {
        self.eviction_data.get(&address).map(Vec::as_slice)
    }

    pub(super) fn queue_refill_writeback(
        &mut self,
        refill: &C220CacheRefill,
    ) -> Result<(), C220LsuSchedulerError> {
        if let Some(writeback) = &refill.writeback {
            self.writes.enqueue(C220LsuLineKey {
                address: writeback.address,
                memory: writeback.memory,
            })?;
            self.eviction_data
                .insert(writeback.address, writeback.bytes.clone());
        }
        Ok(())
    }

    /// Apply an external eviction response to its live backing-memory line.
    /// Maintenance and direct-store responses have separate data sources.
    pub fn complete_eviction_write(
        &mut self,
        id: C220LsuWriteId,
        backing_line: &mut [u8],
    ) -> Result<(), C220LsuSchedulerError> {
        let key = self
            .writes
            .request(id)
            .ok_or(C220LsuWriteError::MissingRequest)?
            .line;
        if key.memory != C220LsuMemory::External {
            return Err(C220LsuSchedulerError::MissingWriteData);
        }
        let bytes = self
            .eviction_data
            .get(&key.address)
            .ok_or(C220LsuSchedulerError::MissingWriteData)?;
        if bytes.len() != backing_line.len() {
            return Err(super::super::cache::C220CacheError::InvalidLineSize.into());
        }
        self.enter_write_response(id)?;
        backing_line.copy_from_slice(&self.eviction_data[&key.address]);
        self.eviction_data.remove(&key.address);
        self.writes.finish_response(id)?;
        Ok(())
    }

    /// A response that fails data validation remains in Responding state and can
    /// be retried without decrementing shared write capacity a second time.
    pub fn complete_ub_store_write(
        &mut self,
        id: C220LsuWriteId,
        write_allocate: bool,
        backing_line: &mut [u8],
    ) -> Result<C220LsuWriteCompletion, C220LsuSchedulerError> {
        let key = self
            .writes
            .request(id)
            .ok_or(C220LsuWriteError::MissingRequest)?
            .line;
        if key.memory != C220LsuMemory::Ub {
            return Err(C220LsuSchedulerError::MissingWriteData);
        }
        self.enter_write_response(id)?;
        let completion =
            self.stores
                .complete_ub_write(key, write_allocate, backing_line, &mut self.misses)?;
        self.resolve_load_values(
            self.writes.tick(),
            key,
            &completion.notifications,
            &completion.line,
        );
        self.writes.finish_response(id)?;
        Ok(completion)
    }

    fn enter_write_response(&mut self, id: C220LsuWriteId) -> Result<(), C220LsuSchedulerError> {
        match self
            .writes
            .request(id)
            .ok_or(C220LsuWriteError::MissingRequest)?
            .state
        {
            C220LsuWriteState::InFlight => {
                self.writes.begin_response(id)?;
            }
            C220LsuWriteState::Responding => {}
            C220LsuWriteState::Queued => return Err(C220LsuWriteError::InvalidState.into()),
        }
        Ok(())
    }
}
