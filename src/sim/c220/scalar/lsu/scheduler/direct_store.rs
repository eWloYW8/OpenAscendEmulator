use super::super::C220LsuRequestId;
use super::super::cache::C220CacheAddressLayout;
use super::super::direct_store::C220LsuDirectStoreBuffer;
use super::super::store_buffer::{C220LsuLineKey, C220LsuMemory};
use super::super::write_queue::C220LsuWriteId;
use super::super::{C220LsuStage, C220LsuStageProgress};
use super::{C220LsuAccess, C220LsuRequest};
use super::{C220LsuRequestScheduler, C220LsuSchedulerError};
use crate::sim::c220::scalar::{C220DirectStoreOperands, C220ScalarMappedAddress};
use crate::sim::common::event::EventError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PendingDirectStore {
    operands: C220DirectStoreOperands,
    address: u64,
    layout: C220CacheAddressLayout,
}

impl C220LsuRequestScheduler {
    /// Queue ST_DEV for the clocked cache stages. Address mapping is captured
    /// at issue, so later root-register changes do not redirect this request.
    pub fn admit_direct_store(
        &mut self,
        tick: u64,
        operands: C220DirectStoreOperands,
        mapped: C220ScalarMappedAddress,
        partition_stack: bool,
        layout: C220CacheAddressLayout,
    ) -> Result<Option<C220LsuRequestId>, C220LsuSchedulerError> {
        let address = mapped.cache_address(partition_stack);
        let line_bytes = self.stores.line_bytes() as u64;
        let offset = (address as u32 & (line_bytes as u32).wrapping_sub(1)) as usize;
        if offset
            .checked_add(operands.bytes().len())
            .is_none_or(|end| end > line_bytes as usize)
        {
            return Err(super::super::store_buffer::C220LsuStoreError::InvalidRange.into());
        }
        let request = C220LsuRequest {
            line: C220LsuLineKey {
                address: address / line_bytes * line_bytes,
                memory: mapped.memory,
            },
            access: C220LsuAccess::Store,
        };
        let Some((id, _)) = self.admit(tick, request, None)? else {
            return Ok(None);
        };
        self.pending_direct_stores.insert(
            id,
            PendingDirectStore {
                operands,
                address,
                layout,
            },
        );
        Ok(Some(id))
    }

    pub fn pending_direct_store(&self, id: C220LsuRequestId) -> Option<&C220DirectStoreOperands> {
        self.pending_direct_stores
            .get(&id)
            .map(|pending| &pending.operands)
    }

    pub(super) fn advance_direct_store_m2(
        &mut self,
        tick: u64,
        stalled: bool,
    ) -> Result<C220LsuStageProgress, C220LsuSchedulerError> {
        let pending = self
            .pipeline
            .head(C220LsuStage::M2)
            .and_then(|head| self.pending_direct_stores.get(&head.request).cloned());
        let Some(pending) = pending else {
            return Ok(self.pipeline.advance_m2(tick, stalled)?);
        };
        let mut pipeline = self.pipeline.clone();
        let progress = pipeline.advance_m2(tick, stalled)?;
        if let C220LsuStageProgress::Advanced(id) = progress {
            self.push_direct_store(
                tick,
                pending.address,
                id,
                pending.operands.bytes(),
                pending.layout,
            )?;
            self.pending_direct_stores.remove(&id);
        }
        self.pipeline = pipeline;
        Ok(progress)
    }

    /// Enqueue captured ST_DEV operands when the request leaves M2.
    /// This does not advance the PC or publish instruction completion.
    pub fn execute_direct_store(
        &mut self,
        tick: u64,
        request: C220LsuRequestId,
        operands: &crate::sim::c220::scalar::C220DirectStoreOperands,
        layout: C220CacheAddressLayout,
    ) -> Result<(), C220LsuSchedulerError> {
        self.push_direct_store(
            tick,
            operands.effective_address,
            request,
            operands.bytes(),
            layout,
        )
    }

    pub fn direct_stores(&self) -> &C220LsuDirectStoreBuffer {
        &self.direct_stores
    }

    pub fn next_direct_store_tick(&self) -> Option<u64> {
        self.direct_events.next_event_tick()
    }

    /// Capture a scalar store and schedule its buffer process one tick later.
    /// `address` is the instruction's byte address, before line alignment.
    pub fn push_direct_store(
        &mut self,
        tick: u64,
        address: u64,
        request: C220LsuRequestId,
        bytes: &[u8],
        layout: C220CacheAddressLayout,
    ) -> Result<(), C220LsuSchedulerError> {
        self.writes.check_tick(tick)?;
        self.direct_events.check_advance_to(tick)?;
        let ready = tick.checked_add(1).ok_or(EventError::TimeOverflow)?;
        let line_bytes = self.stores.line_bytes() as u64;
        let aligned = (address / line_bytes * line_bytes) & 0x0000_ffff_ffff_ffff;
        let offset = (address as u32 & (line_bytes as u32).wrapping_sub(1)) as usize;
        let write_address = layout.line_address(layout.index(aligned), layout.tag(aligned));
        self.direct_stores
            .push(aligned, write_address, request, offset, bytes)?;
        self.writes.advance_to(tick)?;
        self.direct_events.advance_to(tick)?;
        self.direct_events.notify_at(self.direct_event, ready);
        Ok(())
    }

    /// Deliver the direct-store event at this tick. Each invocation visits all
    /// resident entries, including those with a previous write still in flight.
    /// Sending these generated descriptors is a separate clocked process.
    pub fn process_direct_stores(
        &mut self,
        tick: u64,
    ) -> Result<Vec<C220LsuWriteId>, C220LsuSchedulerError> {
        self.writes.check_tick(tick)?;
        self.direct_events.check_advance_to(tick)?;
        let count = if self.direct_events.next_event_tick() == Some(tick) {
            self.direct_stores.entries().len() as u64
        } else {
            0
        };
        if count != 0 && tick == u64::MAX {
            return Err(EventError::TimeOverflow.into());
        }
        self.writes.check_enqueue(count)?;
        self.writes.advance_to(tick)?;
        self.direct_events.advance_to(tick)?;
        let mut generated = Vec::new();
        while self.direct_events.next_callback().is_some() {
            for entry in self.direct_stores.entries() {
                generated.push(self.writes.enqueue(C220LsuLineKey {
                    address: entry.write_address,
                    memory: C220LsuMemory::External,
                })?);
            }
        }
        Ok(generated)
    }
}
