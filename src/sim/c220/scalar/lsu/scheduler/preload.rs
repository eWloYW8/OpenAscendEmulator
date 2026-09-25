use super::super::cache::{C220CacheLocation, C220DataCache};
use super::*;
use crate::sim::c220::scalar::{C220PreloadOperands, C220ScalarMappedAddress};

/// Cache data-path completion; architectural retirement is a separate event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220LsuPreloadCompletion {
    pub request: C220LsuRequestId,
    pub tick: u64,
    pub operands: C220PreloadOperands,
    pub mapped: C220ScalarMappedAddress,
    pub cache_hit: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PendingPreload {
    operands: C220PreloadOperands,
    mapped: C220ScalarMappedAddress,
    line: C220LsuLineKey,
    partition_address: u64,
    lookup: Option<Option<C220CacheLocation>>,
}

impl C220LsuRequestScheduler {
    pub fn admit_preload(
        &mut self,
        tick: u64,
        operands: C220PreloadOperands,
        mapped: C220ScalarMappedAddress,
        partition_stack: bool,
    ) -> Result<Option<C220LsuRequestId>, C220LsuSchedulerError> {
        if self.misses.line_bytes() != 64 {
            return Err(C220LsuSchedulerError::LineGeometry);
        }
        let line = C220LsuLineKey {
            address: mapped.address & 0x0000_ffff_ffff_ffc0,
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
        self.pending_preloads.insert(
            id,
            PendingPreload {
                operands,
                mapped,
                line,
                partition_address: mapped.cache_address(partition_stack),
                lookup: None,
            },
        );
        Ok(Some(id))
    }

    pub fn preload_completions(&self) -> impl Iterator<Item = &C220LsuPreloadCompletion> {
        self.preload_completions.iter()
    }

    pub fn take_preload_completions(&mut self) -> Vec<C220LsuPreloadCompletion> {
        self.preload_completions.drain(..).collect()
    }

    pub(super) fn validate_preload_m2(
        &self,
        tick: u64,
        external: C220LsuExternalHazards,
    ) -> Result<(), C220LsuSchedulerError> {
        if let Some(head) = self.pipeline.head(C220LsuStage::M2)
            && head.ready_tick <= tick
            && let Some(pending) = self.pending_preloads.get(&head.request)
            && self
                .hazard(self.requests[&head.request], external)
                .is_none()
        {
            let location = pending
                .lookup
                .ok_or(C220LsuSchedulerError::MissingLoadLookup)?;
            if location.is_none() {
                self.reads.check_enqueue()?;
                tick.checked_add(1).ok_or(C220LsuReadError::Overflow)?;
            }
        }
        Ok(())
    }

    pub(super) fn advance_preload(
        &mut self,
        stage: C220LsuStage,
        tick: u64,
        id: C220LsuRequestId,
        cache: &mut C220DataCache,
    ) -> Result<(), C220LsuSchedulerError> {
        let Some(pending) = self.pending_preloads.get_mut(&id) else {
            return Ok(());
        };
        match stage {
            C220LsuStage::M0 => {
                let hit = cache.lookup_partitioned(
                    pending.line.address,
                    pending.partition_address,
                    pending.line.memory,
                );
                pending.lookup = Some(hit.or(pending.lookup.flatten()));
            }
            C220LsuStage::M1 => {}
            C220LsuStage::M2 => {
                if pending.lookup.flatten().is_some() {
                    self.finish_preload(id, tick, true);
                } else {
                    let (line, partition) = (pending.line, pending.partition_address);
                    self.enqueue_load_miss(line, id, partition)?;
                }
            }
        }
        Ok(())
    }

    pub(super) fn finish_preload(&mut self, request: C220LsuRequestId, tick: u64, cache_hit: bool) {
        if let Some(pending) = self.pending_preloads.remove(&request) {
            self.preload_completions
                .push_back(C220LsuPreloadCompletion {
                    request,
                    tick,
                    operands: pending.operands,
                    mapped: pending.mapped,
                    cache_hit,
                });
        }
    }
}
