use super::super::cache::{C220CacheLocation, C220DataCache};
use super::super::store_buffer::{C220LsuLineKey, C220LsuMemory};
use super::super::write_queue::C220LsuWriteId;
use super::{C220LsuRequestScheduler, C220LsuSchedulerError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220LsuMaintenanceWrite {
    pub write: C220LsuWriteId,
    pub location: C220CacheLocation,
    pub line: C220LsuLineKey,
}

impl C220LsuRequestScheduler {
    pub fn maintenance_writes(&self) -> impl Iterator<Item = &C220LsuMaintenanceWrite> {
        self.maintenance_writes.values()
    }

    /// Execute the data action for an already selected external cache line.
    /// The controller handles instruction retirement separately and must keep
    /// cache storage from being reused until all maintenance writes drain.
    pub fn invalidate_external_line(
        &mut self,
        tick: u64,
        cache: &mut C220DataCache,
        location: C220CacheLocation,
    ) -> Result<Option<C220LsuMaintenanceWrite>, C220LsuSchedulerError> {
        self.writes.check_tick(tick)?;
        if cache.line_bytes() != self.stores.line_bytes() {
            return Err(C220LsuSchedulerError::LineGeometry);
        }
        let tag = cache.tag(location)?;
        if tag.memory != C220LsuMemory::External {
            return Err(C220LsuSchedulerError::MissingWriteData);
        }
        if !tag.valid {
            return Ok(None);
        }
        self.writes.advance_to(tick)?;
        let pending = if tag.dirty {
            let line = C220LsuLineKey {
                address: cache.layout().line_address(location.index, tag.tag),
                memory: tag.memory,
            };
            let write = self.writes.enqueue(line)?;
            let pending = C220LsuMaintenanceWrite {
                write,
                location,
                line,
            };
            self.maintenance_writes.insert(write, pending);
            Some(pending)
        } else {
            None
        };
        cache.invalidate(location)?;
        Ok(pending)
    }

    /// Sample live data on response. A failed sink retains the pending entry
    /// and can be retried without releasing transport credit twice.
    pub fn apply_maintenance_write_response<E>(
        &mut self,
        id: C220LsuWriteId,
        cache: &mut C220DataCache,
        write: impl FnOnce(C220LsuLineKey, &[u8]) -> Result<(), E>,
    ) -> Result<C220LsuMaintenanceWrite, E>
    where
        E: From<C220LsuSchedulerError>,
    {
        let pending = *self
            .maintenance_writes
            .get(&id)
            .ok_or(C220LsuSchedulerError::MissingWriteData)?;
        self.enter_write_response(id)?;
        let bytes = cache
            .line(pending.location)
            .map_err(C220LsuSchedulerError::from)?;
        write(pending.line, bytes)?;
        cache
            .reset_after_maintenance_write(pending.location)
            .map_err(C220LsuSchedulerError::from)?;
        self.writes
            .finish_response(id)
            .map_err(C220LsuSchedulerError::from)?;
        self.maintenance_writes.remove(&id);
        Ok(pending)
    }
}
