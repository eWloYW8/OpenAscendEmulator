use super::{C220Core, C220CoreError};
mod read;
use crate::sim::c220::memory::timed_memory::{C220MemoryWriteCommand, C220MemoryWriteId};
use crate::sim::c220::scalar::lsu::scheduler::C220LsuRequestScheduler;
use crate::sim::c220::scalar::lsu::store_buffer::C220LsuCompletion;
use crate::sim::c220::scalar::lsu::store_buffer::C220LsuMemory;
use crate::sim::c220::scalar::lsu::write_queue::{
    C220LsuWriteId, C220LsuWriteRequest, C220LsuWriteState,
};

impl C220Core {
    /// Commit an ordinary cache response to mapped external memory before
    /// releasing its source data and returning its instruction notification.
    /// Maintenance and atomic replies are not handled by this path.
    pub fn commit_cache_write_at(
        &mut self,
        tick: u64,
        port: u32,
        lsu: &mut C220LsuRequestScheduler,
    ) -> Result<Option<(C220LsuWriteId, Option<C220LsuCompletion>)>, C220CoreError> {
        self.advance_to(tick)?;
        let pipeline = self
            .mte_pipeline
            .as_mut()
            .ok_or(C220CoreError::MteUnconfigured)?;
        let Some(C220MemoryWriteId::Cache { transaction, .. }) =
            pipeline.cache_write_completion(port)
        else {
            return Ok(None);
        };
        let id = C220LsuWriteId::from_sequence(transaction);
        lsu.writes
            .advance_to(tick)
            .map_err(crate::sim::c220::scalar::lsu::scheduler::C220LsuSchedulerError::from)?;
        let completion = lsu.apply_external_write_response::<C220CoreError>(id, |key, bytes| {
            self.memory.write_known_at(key.address, bytes)?;
            Ok(())
        })?;
        let consumed = pipeline.take_cache_write_completion(port)?;
        debug_assert!(
            matches!(consumed, Some(C220MemoryWriteId::Cache { transaction, .. }) if transaction == id.sequence())
        );
        Ok(Some((id, completion)))
    }

    /// Attach a cache endpoint to the existing shared BIU, before work starts.
    pub fn connect_cache_write_port(&mut self) -> Result<u32, C220CoreError> {
        Ok(self
            .mte_pipeline
            .as_mut()
            .ok_or(C220CoreError::MteUnconfigured)?
            .connect_cache_write_port()?)
    }

    /// Observe transport space before dispatching an LSU write. This does not
    /// reserve a slot; a rejected send must retain the dispatched request.
    pub fn cache_write_ready_at(&mut self, tick: u64, port: u32) -> Result<bool, C220CoreError> {
        self.advance_to(tick)?;
        Ok(self
            .mte_pipeline
            .as_ref()
            .and_then(|pipeline| pipeline.biu_bus_writes())
            .is_some_and(|bus| bus.cache_can_send(port)))
    }

    pub fn send_cache_write_at(
        &mut self,
        tick: u64,
        port: u32,
        request: C220LsuWriteRequest,
    ) -> Result<bool, C220CoreError> {
        if request.line.memory != C220LsuMemory::External
            || request.state != C220LsuWriteState::InFlight
        {
            return Err(C220CoreError::InvalidCacheWrite);
        }
        let bytes =
            u32::try_from(request.byte_len).map_err(|_| C220CoreError::InvalidCacheWrite)?;
        self.advance_to(tick)?;
        Ok(self
            .mte_pipeline
            .as_mut()
            .ok_or(C220CoreError::MteUnconfigured)?
            .send_cache_write(C220MemoryWriteCommand {
                ready_tick: tick,
                tag: C220MemoryWriteId::Cache {
                    port,
                    transaction: request.id.sequence(),
                },
                address: request.line.address,
                bytes,
            })?)
    }

    /// Deliver the original LSU handle after the final response traverses BIU.
    /// The LSU response owner applies backing-memory effects and notifications.
    pub fn take_cache_write_completion_at(
        &mut self,
        tick: u64,
        port: u32,
    ) -> Result<Option<C220LsuWriteId>, C220CoreError> {
        self.advance_to(tick)?;
        Ok(self
            .mte_pipeline
            .as_mut()
            .ok_or(C220CoreError::MteUnconfigured)?
            .take_cache_write_completion(port)?
            .map(|tag| {
                let C220MemoryWriteId::Cache { transaction, .. } = tag else {
                    unreachable!("cache return port")
                };
                C220LsuWriteId::from_sequence(transaction)
            }))
    }
}
