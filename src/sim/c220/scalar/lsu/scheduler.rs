use super::write_queue::{C220LsuWriteError, C220LsuWriteQueue};
use crate::sim::common::event::{EventDispatcher, EventError, EventId};
use std::collections::{BTreeMap, VecDeque};

use super::cache::C220CacheError;
use super::direct_store::C220LsuDirectStoreBuffer;
use super::read_queue::{C220LsuReadError, C220LsuReadQueue};

use super::miss_buffer::{C220LsuMissBuffer, C220LsuMissError};
use super::store_buffer::{C220LsuLineKey, C220LsuStoreBuffer, C220LsuStoreError};
use super::{
    C220LsuPipelineError, C220LsuRequestId, C220LsuRequestPipeline, C220LsuStage,
    C220LsuStageProgress,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220LsuAccess {
    Load,
    Store,
    AtomicStore,
    Other,
}

#[cfg(test)]
mod tests;

mod direct_store;
use direct_store::PendingDirectStore;
mod atomic;
mod load;
mod preload;
pub use preload::C220LsuPreloadCompletion;
mod store;
pub use atomic::C220LsuAtomicCompletion;
use store::PendingStore;
pub use store::{C220LsuStorePath, C220LsuStoreValue};
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220LsuValue {
    Load(C220LsuLoadValue),
    Store(C220LsuStoreValue),
}
mod read;
use load::PendingLoad;
pub use load::{C220LsuLoadPath, C220LsuLoadValue};
mod maintenance;
mod response;
mod write_response;
pub use maintenance::{
    C220LsuMaintenanceCompletion, C220LsuMaintenanceScope, C220LsuMaintenanceWrite,
};
pub use response::C220LsuReadCompletion;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220LsuRequest {
    pub line: C220LsuLineKey,
    pub access: C220LsuAccess,
}

/// Signals from cache maintenance.
/// They must describe the current stage head at the current event tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220LsuExternalHazards {
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
    #[error("captured memory stages require the attached cache data path")]
    CacheRequired,
    #[error("load has not completed its tag lookup")]
    MissingLoadLookup,
    #[error("store operands must fit their cache-line requests")]
    UnsupportedStoreAccess,
    #[error(
        "atomic store at PC {pc:#x} reached a full STB for {line:?}; no retirement can be modeled for this insertion"
    )]
    AtomicStoreCapacity { pc: u64, line: C220LsuLineKey },
    #[error("store buffer request has no captured cache lookup")]
    MissingStoreLookup,
    #[error("read response requires a fetching store entry")]
    UnexpectedStoreResponse,
    #[error(transparent)]
    Cache(#[from] C220CacheError),
    #[error(transparent)]
    Write(#[from] C220LsuWriteError),
    #[error(transparent)]
    Read(#[from] C220LsuReadError),
    #[error("write response does not match the required data source")]
    MissingWriteData,
    #[error(transparent)]
    Event(#[from] EventError),
}

/// Couples stage replay to live line-buffer hazards. The cache controller owns
/// M2 data actions and completion delivery; consuming a request is not retirement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220LsuRequestScheduler {
    pipeline: C220LsuRequestPipeline,
    requests: BTreeMap<C220LsuRequestId, C220LsuRequest>,
    pub misses: C220LsuMissBuffer,
    pub stores: C220LsuStoreBuffer,
    direct_stores: C220LsuDirectStoreBuffer,
    pending_direct_stores: BTreeMap<C220LsuRequestId, PendingDirectStore>,
    direct_events: EventDispatcher<()>,
    direct_event: EventId,
    pub writes: C220LsuWriteQueue,
    pub reads: C220LsuReadQueue,
    pending_loads: BTreeMap<C220LsuRequestId, PendingLoad>,
    pending_preloads: BTreeMap<C220LsuRequestId, preload::PendingPreload>,
    preload_completions: VecDeque<C220LsuPreloadCompletion>,
    values: VecDeque<C220LsuValue>,
    pending_stores: BTreeMap<C220LsuRequestId, PendingStore>,
    pending_atomics: BTreeMap<C220LsuRequestId, atomic::PendingAtomic>,
    atomic_completions: Vec<C220LsuAtomicCompletion>,
    store_next_tick: Option<u64>,
    eviction_data: BTreeMap<u64, Vec<u8>>,
    maintenance_writes: BTreeMap<super::write_queue::C220LsuWriteId, C220LsuMaintenanceWrite>,
    pending_maintenance: BTreeMap<C220LsuRequestId, maintenance::PendingMaintenance>,
    maintenance_completions: Vec<C220LsuMaintenanceCompletion>,
    maintenance_active: bool,
    maintenance_release_tick: Option<u64>,
}

impl C220LsuRequestScheduler {
    pub fn new(
        capacity: u32,
        read_capacity: u32,
        write_capacity: u32,
        direct_store_capacity: usize,
        misses: C220LsuMissBuffer,
        stores: C220LsuStoreBuffer,
    ) -> Result<Self, C220LsuSchedulerError> {
        if misses.line_bytes() != stores.line_bytes() {
            return Err(C220LsuSchedulerError::LineGeometry);
        }
        let mut direct_events = EventDispatcher::new(0);
        let direct_event = direct_events.add_event();
        let process = direct_events.add_process((), false);
        direct_events.subscribe(direct_event, process);
        let writes = C220LsuWriteQueue::new(write_capacity, stores.line_bytes());
        let reads = C220LsuReadQueue::new(read_capacity, stores.line_bytes());
        Ok(Self {
            pipeline: C220LsuRequestPipeline::new(capacity)?,
            requests: BTreeMap::new(),
            pending_direct_stores: BTreeMap::new(),
            direct_stores: C220LsuDirectStoreBuffer::new(
                stores.line_bytes(),
                direct_store_capacity,
            )?,
            misses,
            stores,
            writes,
            reads,
            pending_loads: BTreeMap::new(),
            pending_preloads: BTreeMap::new(),
            preload_completions: VecDeque::new(),
            values: VecDeque::new(),
            pending_stores: BTreeMap::new(),
            pending_atomics: BTreeMap::new(),
            atomic_completions: Vec::new(),
            store_next_tick: None,
            direct_events,
            direct_event,
            eviction_data: BTreeMap::new(),
            maintenance_writes: BTreeMap::new(),
            pending_maintenance: BTreeMap::new(),
            maintenance_completions: Vec::new(),
            maintenance_active: false,
            maintenance_release_tick: None,
        })
    }

    pub fn pipeline(&self) -> &C220LsuRequestPipeline {
        &self.pipeline
    }

    pub fn values(&self) -> impl Iterator<Item = &C220LsuValue> {
        self.values.iter()
    }

    pub fn request(&self, id: C220LsuRequestId) -> Option<&C220LsuRequest> {
        self.requests.get(&id)
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
        self.reads.check_tick(tick)?;
        self.writes.check_tick(tick)?;
        self.direct_events.check_advance_to(tick)?;
        let admitted = self.pipeline.admit(tick, second.is_some())?;
        self.writes.advance_to(tick)?;
        self.reads.advance_to(tick)?;
        self.direct_events.advance_to(tick)?;
        let Some(ids) = admitted else {
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
        _external: C220LsuExternalHazards,
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
            C220LsuAccess::AtomicStore | C220LsuAccess::Other => return None,
        }
        if self.writes.has_hazard(key) {
            return Some(C220LsuStall::Eviction);
        }
        if request.access == C220LsuAccess::Store && self.direct_stores.full() {
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
        if self.pipeline.head(stage).is_some_and(|head| {
            self.pending_loads.contains_key(&head.request)
                || self.pending_preloads.contains_key(&head.request)
                || self.pending_stores.contains_key(&head.request)
                || self.pending_atomics.contains_key(&head.request)
                || self.pending_maintenance.contains_key(&head.request)
        }) {
            return Err(C220LsuSchedulerError::CacheRequired);
        }
        self.advance_impl(stage, tick, external)
    }

    fn advance_impl(
        &mut self,
        stage: C220LsuStage,
        tick: u64,
        external: C220LsuExternalHazards,
    ) -> Result<C220LsuStageOutcome, C220LsuSchedulerError> {
        self.reads.check_tick(tick)?;
        self.writes.check_tick(tick)?;
        self.direct_events.check_advance_to(tick)?;
        let mut stall = self
            .pipeline
            .head(stage)
            .filter(|head| head.ready_tick <= tick)
            .and_then(|head| self.hazard(self.requests[&head.request], external));
        if stage == C220LsuStage::M0 && self.pipeline.head(stage).is_some() {
            if external.maintenance_active || self.maintenance_active {
                stall = Some(C220LsuStall::Maintenance);
            } else if stall.is_none()
                && (external.maintenance_draining || self.maintenance_needs_drain())
            {
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
            C220LsuStage::M2 => self.advance_direct_store_m2(tick, stall.is_some())?,
        };
        self.writes.advance_to(tick)?;
        self.reads.advance_to(tick)?;
        self.direct_events.advance_to(tick)?;
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
