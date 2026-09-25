use super::super::cache::{C220CacheLocation, C220DataCache};
use super::super::store_buffer::{C220LsuLineKey, C220LsuMemory};
use super::super::write_queue::C220LsuWriteId;
use super::{C220LsuAccess, C220LsuRequest, C220LsuRequestId, C220LsuStage};
use super::{C220LsuRequestScheduler, C220LsuSchedulerError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220LsuMaintenanceScope {
    Line {
        address: u64,
        partition_address: u64,
    },
    All {
        memory: Option<C220LsuMemory>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220LsuMaintenanceCompletion {
    pub request: C220LsuRequestId,
    pub tick: u64,
    pub scope: C220LsuMaintenanceScope,
    pub invalidated: Vec<C220CacheLocation>,
    pub writes: Vec<C220LsuMaintenanceWrite>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PendingMaintenance {
    scope: C220LsuMaintenanceScope,
    location: Option<C220CacheLocation>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220LsuMaintenanceWrite {
    pub write: C220LsuWriteId,
    pub location: C220CacheLocation,
    pub line: C220LsuLineKey,
}

impl C220LsuRequestScheduler {
    /// Enqueue maintenance in instruction order. The caller supplies retirement
    /// port occupancy through the M0 `maintenance_draining` hazard signal.
    pub fn admit_maintenance(
        &mut self,
        tick: u64,
        scope: C220LsuMaintenanceScope,
    ) -> Result<Option<C220LsuRequestId>, C220LsuSchedulerError> {
        let address = match scope {
            C220LsuMaintenanceScope::Line { address, .. } => address,
            C220LsuMaintenanceScope::All { .. } => 0,
        };
        let size = self.stores.line_bytes() as u64;
        let Some((request, _)) = self.admit(
            tick,
            C220LsuRequest {
                line: C220LsuLineKey {
                    address: address / size * size,
                    memory: C220LsuMemory::External,
                },
                access: C220LsuAccess::Other,
            },
            None,
        )?
        else {
            return Ok(None);
        };
        self.pending_maintenance.insert(
            request,
            PendingMaintenance {
                scope,
                location: None,
            },
        );
        Ok(Some(request))
    }

    pub fn maintenance_active(&self) -> bool {
        self.maintenance_active
    }

    pub fn maintenance_head(&self) -> bool {
        self.pipeline
            .head(C220LsuStage::M0)
            .is_some_and(|head| self.pending_maintenance.contains_key(&head.request))
    }

    pub fn take_maintenance_completions(&mut self) -> Vec<C220LsuMaintenanceCompletion> {
        std::mem::take(&mut self.maintenance_completions)
    }

    pub(super) fn maintenance_needs_drain(&self) -> bool {
        self.maintenance_head()
            && (self.pipeline.head(C220LsuStage::M1).is_some()
                || self.pipeline.head(C220LsuStage::M2).is_some()
                || !self.misses.entries().is_empty()
                || !self.stores.entries().is_empty()
                || !self.direct_stores.entries().is_empty())
    }

    fn maintenance_locations(
        pending: PendingMaintenance,
        cache: &C220DataCache,
    ) -> Vec<C220CacheLocation> {
        match pending.scope {
            C220LsuMaintenanceScope::Line { .. } => pending.location.into_iter().collect(),
            C220LsuMaintenanceScope::All { memory } => cache.maintenance_lines(memory),
        }
    }

    pub(super) fn prepare_maintenance_stage(
        &mut self,
        stage: C220LsuStage,
        tick: u64,
        cache: &mut C220DataCache,
    ) -> Result<(), C220LsuSchedulerError> {
        self.writes.check_tick(tick)?;
        if self
            .maintenance_release_tick
            .is_some_and(|ready| ready <= tick)
            && self.maintenance_writes.is_empty()
            && !self
                .writes
                .requests()
                .any(|write| write.line.memory == C220LsuMemory::External)
        {
            self.maintenance_active = false;
            self.maintenance_release_tick = None;
        }
        let Some(head) = self
            .pipeline
            .head(stage)
            .filter(|head| head.ready_tick <= tick)
        else {
            return Ok(());
        };
        let Some(pending) = self.pending_maintenance.get_mut(&head.request) else {
            return Ok(());
        };
        if stage == C220LsuStage::M0
            && !self.maintenance_active
            && let C220LsuMaintenanceScope::Line {
                address,
                partition_address,
            } = pending.scope
        {
            pending.location = cache
                .lookup_maintenance(address, partition_address)
                .or(pending.location);
        }
        if stage == C220LsuStage::M2 {
            let mut dirty = 0;
            for location in Self::maintenance_locations(*pending, cache) {
                let tag = cache.tag(location)?;
                if tag.dirty && tag.memory != C220LsuMemory::External {
                    return Err(C220LsuSchedulerError::MissingWriteData);
                }
                dirty += u64::from(tag.valid && tag.dirty);
            }
            self.writes.check_enqueue(dirty)?;
            tick.checked_add(1).ok_or(super::EventError::TimeOverflow)?;
        }
        Ok(())
    }

    pub(super) fn advance_maintenance_stage(
        &mut self,
        stage: C220LsuStage,
        tick: u64,
        request: C220LsuRequestId,
        cache: &mut C220DataCache,
    ) -> Result<(), C220LsuSchedulerError> {
        let Some(pending) = self.pending_maintenance.get(&request).copied() else {
            return Ok(());
        };
        match stage {
            C220LsuStage::M0 => self.maintenance_active = true,
            C220LsuStage::M1 => {}
            C220LsuStage::M2 => {
                let invalidated = Self::maintenance_locations(pending, cache);
                let mut writes = Vec::new();
                for &location in &invalidated {
                    if cache.tag(location)?.memory == C220LsuMemory::External {
                        if let Some(write) = self.invalidate_external_line(tick, cache, location)? {
                            writes.push(write);
                        }
                    } else {
                        cache.invalidate(location)?;
                    }
                }
                if writes.is_empty() {
                    match pending.scope {
                        C220LsuMaintenanceScope::All { .. } => self.maintenance_active = false,
                        C220LsuMaintenanceScope::Line { .. } => {
                            self.maintenance_release_tick = Some(tick + 1)
                        }
                    }
                }
                self.pending_maintenance.remove(&request);
                self.maintenance_completions
                    .push(C220LsuMaintenanceCompletion {
                        request,
                        tick,
                        scope: pending.scope,
                        invalidated,
                        writes,
                    });
            }
        }
        Ok(())
    }

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
            self.maintenance_release_tick = Some(tick + 1);
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
