use std::collections::BTreeMap;

use super::miss_buffer::{C220LsuMissBuffer, C220LsuMissError, C220LsuMissState};
use super::store_buffer::{
    C220LsuCompletion, C220LsuLineKey, C220LsuMemory, C220LsuStoreBuffer, C220LsuStoreError,
    C220LsuStoreState,
};
use super::{
    C220LsuPipelineError, C220LsuRequestId, C220LsuRequestPipeline, C220LsuStage,
    C220LsuStageProgress,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220LsuAccess {
    Load,
    Store,
    Other,
}

#[cfg(test)]
mod tests;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220LsuRequest {
    pub line: C220LsuLineKey,
    pub access: C220LsuAccess,
}

/// Signals from cache eviction, direct stores and cache maintenance.
/// They must describe the current stage head at the current event tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220LsuExternalHazards {
    pub eviction_conflict: bool,
    pub direct_store_full: bool,
    pub maintenance_active: bool,
    pub maintenance_draining: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220LsuStall {
    MissCapacity,
    MissForbidden,
    StoreCapacity,
    PendingLoad,
    StoreForbidden,
    Eviction,
    DirectStoreCapacity,
    Maintenance,
    Drain,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220LsuStageOutcome {
    pub progress: C220LsuStageProgress,
    pub stall: Option<C220LsuStall>,
    /// Request handed to the data/completion path when it leaves M2.
    pub consumed: Option<C220LsuRequest>,
}

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

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum C220LsuSchedulerError {
    #[error(transparent)]
    Pipeline(#[from] C220LsuPipelineError),
    #[error("LSU buffers must have matching cache-line sizes")]
    LineGeometry,
    #[error("LSU request line is not aligned")]
    UnalignedLine,
    #[error(transparent)]
    Miss(#[from] C220LsuMissError),
    #[error(transparent)]
    Store(#[from] C220LsuStoreError),
    #[error("read response requires a fetching miss entry")]
    UnexpectedReadResponse,
    #[error("external read response requires an allocated cache line")]
    MissingCacheLine,
    #[error("read response requires a fetching store entry")]
    UnexpectedStoreResponse,
}

/// Couples stage replay to live line-buffer hazards. The cache controller owns
/// M2 data actions and completion delivery; consuming a request is not retirement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220LsuRequestScheduler {
    pipeline: C220LsuRequestPipeline,
    requests: BTreeMap<C220LsuRequestId, C220LsuRequest>,
    pub misses: C220LsuMissBuffer,
    pub stores: C220LsuStoreBuffer,
}

impl C220LsuRequestScheduler {
    pub fn new(
        capacity: u32,
        misses: C220LsuMissBuffer,
        stores: C220LsuStoreBuffer,
    ) -> Result<Self, C220LsuSchedulerError> {
        if misses.line_bytes() != stores.line_bytes() {
            return Err(C220LsuSchedulerError::LineGeometry);
        }
        Ok(Self {
            pipeline: C220LsuRequestPipeline::new(capacity)?,
            requests: BTreeMap::new(),
            misses,
            stores,
        })
    }

    pub fn pipeline(&self) -> &C220LsuRequestPipeline {
        &self.pipeline
    }

    pub fn request(&self, id: C220LsuRequestId) -> Option<&C220LsuRequest> {
        self.requests.get(&id)
    }

    /// Complete an MSHR-owned read after cache refill. `cache_line` is required
    /// for external memory and present for cacheable UB. The caller issues any
    /// returned UB write and delivers notifications in order through the timed
    /// completion port. Tag allocation and response ownership are caller-owned.
    pub fn complete_miss_read(
        &mut self,
        key: C220LsuLineKey,
        returned_line: &[u8],
        mut cache_line: Option<&mut [u8]>,
    ) -> Result<C220LsuReadCompletion, C220LsuSchedulerError> {
        let width = self.misses.line_bytes();
        if self.stores.line_bytes() != width {
            return Err(C220LsuSchedulerError::LineGeometry);
        }
        if returned_line.len() != width
            || cache_line.as_ref().is_some_and(|line| line.len() != width)
        {
            return Err(C220LsuMissError::InvalidLineSize.into());
        }
        if key.memory == C220LsuMemory::External && cache_line.is_none() {
            return Err(C220LsuSchedulerError::MissingCacheLine);
        }
        let entry = self
            .misses
            .entry(key)
            .ok_or(C220LsuMissError::MissingEntry)?;
        if entry.state() != C220LsuMissState::Fetching {
            return Err(C220LsuSchedulerError::UnexpectedReadResponse);
        }
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
            }
        }
        self.misses.remove(key);
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
        let width = self.stores.line_bytes();
        if self.misses.line_bytes() != width {
            return Err(C220LsuSchedulerError::LineGeometry);
        }
        if returned_line.len() != width
            || cache_line.as_ref().is_some_and(|line| line.len() != width)
        {
            return Err(C220LsuMissError::InvalidLineSize.into());
        }
        if key.memory == C220LsuMemory::External && cache_line.is_none() {
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
        Ok(C220LsuReadCompletion {
            key,
            load_line,
            notifications,
            pending_ub_write,
            mark_cache_dirty: key.memory == C220LsuMemory::External,
        })
    }

    pub fn admit(
        &mut self,
        tick: u64,
        first: C220LsuRequest,
        second: Option<C220LsuRequest>,
    ) -> Result<Option<(C220LsuRequestId, Option<C220LsuRequestId>)>, C220LsuSchedulerError> {
        for request in std::iter::once(&first).chain(second.iter()) {
            if !request
                .line
                .address
                .is_multiple_of(self.misses.line_bytes() as u64)
            {
                return Err(C220LsuSchedulerError::UnalignedLine);
            }
        }
        let Some(ids) = self.pipeline.admit(tick, second.is_some())? else {
            return Ok(None);
        };
        self.requests.insert(ids.0, first);
        if let (Some(id), Some(request)) = (ids.1, second) {
            self.requests.insert(id, request);
        }
        Ok(Some(ids))
    }

    pub fn hazard(
        &self,
        request: C220LsuRequest,
        external: C220LsuExternalHazards,
    ) -> Option<C220LsuStall> {
        let key = request.line;
        match request.access {
            C220LsuAccess::Load => {
                if self.misses.full(key) {
                    return Some(C220LsuStall::MissCapacity);
                }
                if self
                    .misses
                    .entry(key)
                    .is_some_and(|entry| entry.forbidden())
                {
                    return Some(C220LsuStall::MissForbidden);
                }
            }
            C220LsuAccess::Store => {
                if self.stores.full(key) {
                    return Some(C220LsuStall::StoreCapacity);
                }
                if self.misses.entry(key).is_some() {
                    return Some(C220LsuStall::PendingLoad);
                }
                if self
                    .stores
                    .entry(key)
                    .is_some_and(|entry| entry.forbidden())
                {
                    return Some(C220LsuStall::StoreForbidden);
                }
            }
            C220LsuAccess::Other => return None,
        }
        if external.eviction_conflict {
            return Some(C220LsuStall::Eviction);
        }
        if request.access == C220LsuAccess::Store && external.direct_store_full {
            return Some(C220LsuStall::DirectStoreCapacity);
        }
        None
    }

    pub fn advance(
        &mut self,
        stage: C220LsuStage,
        tick: u64,
        external: C220LsuExternalHazards,
    ) -> Result<C220LsuStageOutcome, C220LsuSchedulerError> {
        let mut stall = self
            .pipeline
            .head(stage)
            .filter(|head| head.ready_tick <= tick)
            .and_then(|head| self.hazard(self.requests[&head.request], external));
        if stage == C220LsuStage::M0 && self.pipeline.head(stage).is_some() {
            if external.maintenance_active {
                stall = Some(C220LsuStall::Maintenance);
            } else if stall.is_none() && external.maintenance_draining {
                stall = Some(C220LsuStall::Drain);
            }
        }
        let progress = match stage {
            C220LsuStage::M0 => self.pipeline.advance_m0(
                tick,
                stall.is_some_and(|s| s != C220LsuStall::Drain),
                stall == Some(C220LsuStall::Drain),
            )?,
            C220LsuStage::M1 => self.pipeline.advance_m1(tick, stall.is_some())?,
            C220LsuStage::M2 => self.pipeline.advance_m2(tick, stall.is_some())?,
        };
        let consumed = match progress {
            C220LsuStageProgress::Advanced(id) if stage == C220LsuStage::M2 => {
                self.requests.remove(&id)
            }
            _ => None,
        };
        if matches!(
            progress,
            C220LsuStageProgress::Idle | C220LsuStageProgress::Advanced(_)
        ) {
            stall = None;
        }
        Ok(C220LsuStageOutcome {
            progress,
            stall,
            consumed,
        })
    }
}
