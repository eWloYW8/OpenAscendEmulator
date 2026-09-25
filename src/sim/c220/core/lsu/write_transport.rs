use super::*;
use crate::memory::sparse::MemoryByteState;
use crate::memory::ub::UbMemory;
use crate::sim::c220::memory::ub_service::{C220UbServicePort, C220UbServiceRequest};
use crate::sim::c220::mte::C220MtePipeline;
use crate::sim::c220::mte::interface::biu_read::C220BiuSubcore;
use crate::sim::c220::scalar::lsu::store_buffer::C220LsuMemory;

impl CoreLsu {
    pub(super) fn receive_ub_write_at(
        &mut self,
        tick: u64,
        pipeline: &mut C220MtePipeline,
        ub: &mut UbMemory,
    ) -> Result<(), C220CoreError> {
        self.scheduler
            .writes
            .advance_to(tick)
            .map_err(C220LsuSchedulerError::from)?;
        let subcore = pipeline.ub_vector_subcore();
        if subcore == C220BiuSubcore::Cube {
            return Ok(());
        }
        let service = pipeline.ub_memory_mut(subcore)?;
        let port = C220UbServicePort::ScalarWrite;
        if let Some(response) = service
            .transport(port)
            .front()
            .filter(|response| response.ready_tick <= tick)
            .copied()
        {
            let id = C220LsuWriteId::from_sequence(response.request.id);
            let request = self
                .scheduler
                .writes
                .request(id)
                .ok_or(C220CoreError::InvalidCacheWrite)?;
            let prior = if !self.config.ub_write_allocate
                && self.scheduler.stores.entry(request.line).is_some()
            {
                ub.read_known(request.line.address, request.byte_len)?
            } else {
                vec![0; request.byte_len]
            };
            let cache = self.cache.as_ref().ok_or(C220CoreError::LsuUnconfigured)?;
            self.scheduler.apply_ub_write_response::<C220CoreError>(
                id,
                self.config.ub_write_allocate,
                &prior,
                &cache.cache,
                |key, bytes| {
                    let states: Vec<_> =
                        bytes.iter().copied().map(MemoryByteState::Known).collect();
                    ub.write_states(key.address, &states)?;
                    Ok(())
                },
            )?;
            let consumed = service.take_response(tick, port)?;
            debug_assert_eq!(consumed, Some(response));
        }
        if let Some((ready, request)) = self.ub_writes.front().copied()
            && ready <= tick
            && service.receive(
                tick,
                port,
                C220UbServiceRequest {
                    id: request.id.sequence(),
                    address: request.line.address,
                    bytes: request.byte_len as u32,
                },
            )?
        {
            self.ub_writes.pop_front();
        }
        Ok(())
    }

    pub(super) fn send_writes_at(
        &mut self,
        tick: u64,
        pipeline: &mut C220MtePipeline,
    ) -> Result<(), C220CoreError> {
        let external_ready = self.send_pending.is_none()
            && pipeline
                .biu_bus_writes()
                .is_some_and(|bus| bus.cache_can_send(self.port));
        let ub_ready =
            pipeline.ub_vector_subcore() != C220BiuSubcore::Cube && self.ub_writes.len() < 2;
        let receive_tick = tick.checked_add(1).ok_or(C220CoreError::TimeOverflow)?;
        for request in self
            .scheduler
            .writes
            .dispatch_clock(tick, ub_ready, external_ready)
            .map_err(C220LsuSchedulerError::from)?
        {
            match request.line.memory {
                C220LsuMemory::Ub => self.ub_writes.push_back((receive_tick, request)),
                C220LsuMemory::External => self.send_pending = Some(request),
            }
        }
        if let Some(request) = self.send_pending {
            let bytes =
                u32::try_from(request.byte_len).map_err(|_| C220CoreError::InvalidCacheWrite)?;
            if pipeline.send_cache_write(C220MemoryWriteCommand {
                ready_tick: tick,
                tag: C220MemoryWriteId::Cache {
                    port: self.port,
                    transaction: request.id.sequence(),
                },
                address: request.line.address,
                bytes,
            })? {
                self.send_pending = None;
            }
        }
        Ok(())
    }
}
