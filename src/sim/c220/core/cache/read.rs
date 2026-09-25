use super::{C220Core, C220CoreError, C220LsuMemory, C220LsuRequestScheduler};
use crate::sim::c220::memory::biu_read::{C220BiuReadCacheConfig, C220BiuReadCacheKind};
use crate::sim::c220::memory::timed_memory::{C220MemoryReadCommand, C220MemoryReadId};
use crate::sim::c220::scalar::lsu::cache::{C220CacheRefill, C220DataCache};
use crate::sim::c220::scalar::lsu::read_queue::{
    C220LsuReadId, C220LsuReadRequest, C220LsuReadState,
};
use crate::sim::c220::scalar::lsu::scheduler::{C220LsuReadCompletion, C220LsuSchedulerError};

impl C220Core {
    pub fn connect_cache_read_port(
        &mut self,
        config: C220BiuReadCacheConfig,
    ) -> Result<u32, C220CoreError> {
        Ok(self
            .mte_pipeline
            .as_mut()
            .ok_or(C220CoreError::MteUnconfigured)?
            .connect_cache_read_port(C220BiuReadCacheKind::Data, config)?)
    }

    /// Send a dispatched LSU fetch; transport rejection leaves the request with
    /// the caller for retry. Multi-beat cache lines require a separate assembler.
    pub fn send_cache_read_at(
        &mut self,
        tick: u64,
        port: u32,
        request: C220LsuReadRequest,
    ) -> Result<bool, C220CoreError> {
        if request.line.memory != C220LsuMemory::External
            || request.state != C220LsuReadState::InFlight
            || !(1..=128).contains(&request.byte_len)
        {
            return Err(C220CoreError::InvalidCacheRead);
        }
        self.advance_to(tick)?;
        Ok(self
            .mte_pipeline
            .as_mut()
            .ok_or(C220CoreError::MteUnconfigured)?
            .send_cache_read(C220MemoryReadCommand {
                ready_tick: tick,
                tag: C220MemoryReadId::DataCache {
                    port,
                    transaction: request.id.sequence(),
                },
                address: request.line.address,
                bytes: request.byte_len as u32,
            })?)
    }

    /// Preserve the transport response until backing data and cache refill have
    /// succeeded. The request's original buffer decides completion ordering.
    pub fn commit_cache_read_at(
        &mut self,
        tick: u64,
        port: u32,
        lsu: &mut C220LsuRequestScheduler,
        cache: &mut C220DataCache,
    ) -> Result<Option<(C220LsuReadId, C220LsuReadCompletion, C220CacheRefill)>, C220CoreError>
    {
        self.advance_to(tick)?;
        let pipeline = self
            .mte_pipeline
            .as_mut()
            .ok_or(C220CoreError::MteUnconfigured)?;
        let Some(response) = pipeline
            .biu_bus_reads()
            .and_then(|bus| bus.cache_returns(C220BiuReadCacheKind::Data, port))
            .and_then(|returns| returns.front())
            .filter(|response| response.ready_tick <= tick)
            .copied()
        else {
            return Ok(None);
        };
        let C220MemoryReadId::DataCache { transaction, .. } = response.beat.tag else {
            return Err(C220CoreError::InvalidCacheRead);
        };
        if response.beat.transaction_id != 0 {
            return Err(C220CoreError::InvalidCacheRead);
        }
        let id = C220LsuReadId::from_sequence(transaction);
        let request = lsu
            .reads
            .request(id)
            .ok_or(C220CoreError::InvalidCacheRead)?;
        if request.line.memory != C220LsuMemory::External || !(1..=128).contains(&request.byte_len)
        {
            return Err(C220CoreError::InvalidCacheRead);
        }
        lsu.reads
            .check_tick(tick)
            .map_err(C220LsuSchedulerError::from)?;
        lsu.writes
            .check_tick(tick)
            .map_err(C220LsuSchedulerError::from)?;
        lsu.reads
            .advance_to(tick)
            .map_err(C220LsuSchedulerError::from)?;
        lsu.writes
            .advance_to(tick)
            .map_err(C220LsuSchedulerError::from)?;
        let (completion, refill) =
            lsu.apply_read_response::<C220CoreError>(id, Some(cache), |key, size| {
                Ok(self.memory.read_known_at(key.address, size)?)
            })?;
        let consumed = pipeline.take_cache_read_return(C220BiuReadCacheKind::Data, port);
        debug_assert_eq!(consumed, Some(response.beat));
        Ok(Some((
            id,
            completion,
            refill.expect("external cache refill"),
        )))
    }
}
