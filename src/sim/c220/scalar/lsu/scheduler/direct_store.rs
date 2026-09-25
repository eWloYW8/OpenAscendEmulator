use super::super::C220LsuRequestId;
use super::super::cache::C220CacheAddressLayout;
use super::super::direct_store::C220LsuDirectStoreBuffer;
use super::super::store_buffer::{C220LsuLineKey, C220LsuMemory};
use super::super::write_queue::C220LsuWriteId;
use super::{C220LsuRequestScheduler, C220LsuSchedulerError};
use crate::sim::common::event::EventError;

impl C220LsuRequestScheduler {
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
