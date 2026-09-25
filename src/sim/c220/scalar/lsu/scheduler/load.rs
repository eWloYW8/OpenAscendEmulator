use super::super::cache::{C220CacheLocation, C220DataCache};
use super::super::store_buffer::C220LsuCompletion;
use super::*;
use crate::sim::c220::scalar::{C220LoadOperands, C220ScalarMappedAddress};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220LsuLoadPath {
    Cache,
    StoreForward,
    CacheAndStore,
    Refill,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220LsuLoadValue {
    pub request: C220LsuRequestId,
    pub tick: u64,
    pub operands: C220LoadOperands,
    pub mapped: C220ScalarMappedAddress,
    pub value: u64,
    pub path: C220LsuLoadPath,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PendingLoad {
    operands: C220LoadOperands,
    mapped: C220ScalarMappedAddress,
    line: C220LsuLineKey,
    partition_address: u64,
    offset: usize,
    lookup: Option<Option<C220CacheLocation>>,
}

impl C220LsuRequestScheduler {
    pub fn admit_load(
        &mut self,
        tick: u64,
        operands: C220LoadOperands,
        mapped: C220ScalarMappedAddress,
        partition_stack: bool,
    ) -> Result<Option<C220LsuRequestId>, C220LsuSchedulerError> {
        let size = self.misses.line_bytes() as u64;
        let physical_address = mapped.address & 0x0000_ffff_ffff_ffff;
        let offset = (physical_address % size) as usize;
        if ![1, 2, 4, 8].contains(&operands.width_bytes)
            || offset + usize::from(operands.width_bytes) > size as usize
        {
            return Err(C220LsuStoreError::InvalidRange.into());
        }
        let line = C220LsuLineKey {
            address: physical_address / size * size,
            memory: mapped.memory,
        };
        let Some((id, _)) = self.admit(
            tick,
            C220LsuRequest {
                line,
                access: C220LsuAccess::Load,
            },
            None,
        )?
        else {
            return Ok(None);
        };
        self.pending_loads.insert(
            id,
            PendingLoad {
                operands,
                mapped,
                line,
                offset,
                partition_address: mapped.cache_address(partition_stack),
                lookup: None,
            },
        );
        Ok(Some(id))
    }

    pub fn pending_load(&self, id: C220LsuRequestId) -> Option<&C220LoadOperands> {
        self.pending_loads.get(&id).map(|pending| &pending.operands)
    }

    /// Data availability, not architectural retirement. The core owns final
    /// register updates and release of dependent instructions.
    pub fn take_load_values(&mut self) -> Vec<C220LsuLoadValue> {
        std::mem::take(&mut self.load_values)
    }

    pub fn advance_with_cache(
        &mut self,
        stage: C220LsuStage,
        tick: u64,
        external: C220LsuExternalHazards,
        cache: &mut C220DataCache,
    ) -> Result<C220LsuStageOutcome, C220LsuSchedulerError> {
        if cache.line_bytes() != self.misses.line_bytes() {
            return Err(C220LsuSchedulerError::LineGeometry);
        }
        let pending = self
            .pipeline
            .head(stage)
            .filter(|head| head.ready_tick <= tick)
            .and_then(|head| {
                self.pending_loads
                    .get(&head.request)
                    .copied()
                    .map(|load| (head.request, load))
            });
        let mut result = None;
        if let Some((id, pending)) = pending
            && stage == C220LsuStage::M2
            && self.hazard(self.requests[&id], external).is_none()
        {
            let location = pending
                .lookup
                .ok_or(C220LsuSchedulerError::MissingLoadLookup)?;
            let width = usize::from(pending.operands.width_bytes);
            let store = self.stores.entry(pending.line);
            let covered = store
                .map(|entry| entry.covered_bytes(pending.offset, width))
                .transpose()?
                .unwrap_or(0);
            if location.is_some() || covered == width {
                let mut bytes = [0; 8];
                if let Some(location) = location {
                    bytes[..width].copy_from_slice(
                        &cache.line(location)?[pending.offset..pending.offset + width],
                    );
                }
                if covered != 0 {
                    store
                        .expect("covered store bytes")
                        .forward_scalar(pending.offset, &mut bytes[..width])?;
                }
                let path = match (location.is_some(), covered != 0) {
                    (true, false) => C220LsuLoadPath::Cache,
                    (true, true) => C220LsuLoadPath::CacheAndStore,
                    _ => C220LsuLoadPath::StoreForward,
                };
                result = Some((u64::from_le_bytes(bytes), path));
            } else {
                self.reads.check_enqueue()?;
                tick.checked_add(1).ok_or(C220LsuReadError::Overflow)?;
            }
        }
        let outcome = self.advance_impl(stage, tick, external)?;
        if let C220LsuStageProgress::Advanced(id) = outcome.progress
            && let Some((pending_id, pending)) = pending
        {
            debug_assert_eq!(pending_id, id);
            match stage {
                C220LsuStage::M0 => {
                    let location = cache.lookup_partitioned(
                        pending.line.address,
                        pending.partition_address,
                        pending.line.memory,
                    );
                    self.pending_loads
                        .get_mut(&id)
                        .expect("admitted load")
                        .lookup = Some(location.or(pending.lookup.flatten()));
                }
                C220LsuStage::M2 => {
                    if let Some((value, path)) = result {
                        self.finish_load(id, pending, tick, value, path);
                    } else {
                        self.enqueue_load_miss(pending.line, id, pending.partition_address)?;
                    }
                }
                C220LsuStage::M1 => {}
            }
        }
        Ok(outcome)
    }

    fn finish_load(
        &mut self,
        id: C220LsuRequestId,
        pending: PendingLoad,
        tick: u64,
        value: u64,
        path: C220LsuLoadPath,
    ) {
        self.pending_loads.remove(&id);
        self.load_values.push(C220LsuLoadValue {
            request: id,
            tick,
            operands: pending.operands,
            mapped: pending.mapped,
            value,
            path,
        });
    }

    pub(super) fn resolve_load_values(
        &mut self,
        tick: u64,
        key: C220LsuLineKey,
        notifications: &[C220LsuCompletion],
        line: &[u8],
    ) {
        for notification in notifications {
            if let C220LsuCompletion::Load(id) = *notification
                && let Some(pending) = self.pending_loads.get(&id).copied()
            {
                debug_assert_eq!(pending.line, key);
                let width = usize::from(pending.operands.width_bytes);
                let mut bytes = [0; 8];
                bytes[..width].copy_from_slice(&line[pending.offset..pending.offset + width]);
                self.finish_load(
                    id,
                    pending,
                    tick,
                    u64::from_le_bytes(bytes),
                    C220LsuLoadPath::Refill,
                );
            }
        }
    }
}
