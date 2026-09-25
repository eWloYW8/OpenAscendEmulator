use std::num::NonZeroU32;

use super::{C220MtePipeline, C220MtePipelineError};
use crate::sim::c220::memory::biu_read::C220BiuBusReads;
use crate::sim::c220::memory::timed_memory::{C220MemoryReadCommand, C220MemoryReadId};
use crate::sim::c220::mte::interface::biu_read::returns::C220BiuReadBeat;

pub(super) fn memory_read_command(
    request: crate::sim::c220::mte::interface::biu_read::C220BiuReadRequest,
    tick: u64,
) -> C220MemoryReadCommand {
    C220MemoryReadCommand {
        ready_tick: tick,
        tag: C220MemoryReadId::Mte(request.tag),
        address: request.input.generated.request.source_address,
        bytes: request.input.generated.request.bytes,
    }
}

impl C220MtePipeline {
    pub fn connect_cache_read_port(
        &mut self,
        kind: crate::sim::c220::memory::biu_read::C220BiuReadCacheKind,
        config: crate::sim::c220::memory::biu_read::C220BiuReadCacheConfig,
    ) -> Result<u32, C220MtePipelineError> {
        if !self.is_idle() {
            return Err(C220MtePipelineError::CommandBusy);
        }
        Ok(self
            .biu_bus_reads
            .as_mut()
            .ok_or(C220MtePipelineError::BiuBusDisconnected)?
            .add_cache_port(kind, config)?)
    }

    pub fn send_cache_read(
        &mut self,
        command: C220MemoryReadCommand,
    ) -> Result<bool, C220MtePipelineError> {
        Ok(self
            .biu_bus_reads
            .as_mut()
            .ok_or(C220MtePipelineError::BiuBusDisconnected)?
            .send_cache_command(self.events.tick(), command)?)
    }

    pub fn take_cache_read_return(
        &mut self,
        kind: crate::sim::c220::memory::biu_read::C220BiuReadCacheKind,
        port: u32,
    ) -> Option<crate::sim::c220::memory::timed_memory::C220MemoryReadBeat> {
        self.biu_bus_reads
            .as_mut()?
            .take_cache_return(self.events.tick(), kind, port)
    }

    pub fn connect_biu_bus_reads(
        &mut self,
        outstanding: NonZeroU32,
    ) -> Result<(), C220MtePipelineError> {
        if !self.is_idle() {
            return Err(C220MtePipelineError::CommandBusy);
        }
        if self.biu_read.is_none() {
            return Err(C220MtePipelineError::BiuDisconnected);
        }
        self.biu_bus_reads = Some(C220BiuBusReads::new(outstanding));
        Ok(())
    }

    pub fn biu_bus_reads(&self) -> Option<&C220BiuBusReads> {
        self.biu_bus_reads.as_ref()
    }

    pub(super) fn advance_biu_read_returns(
        &mut self,
        tick: u64,
    ) -> Result<(), C220MtePipelineError> {
        let Some(bus) = &mut self.biu_bus_reads else {
            return Ok(());
        };
        if let Some(memory) = &mut self.timed_memory {
            let mut port = 0;
            while let Some(beat) = memory.read_front(tick) {
                let mut heads = [None; 2];
                heads[port] = Some(beat);
                if !bus.receive(tick, heads)?[port] {
                    break;
                }
                memory.pop_read();
                port ^= 1;
            }
        }
        bus.advance_returns(tick)?;
        let accepted = self
            .biu_returns
            .as_mut()
            .expect("connected read returns")
            .receive(
                tick,
                bus.heads(tick).map(|head| {
                    head.map(|beat| {
                        let C220MemoryReadId::Mte(tag) = beat.tag else {
                            unreachable!("MTE response endpoint");
                        };
                        C220BiuReadBeat {
                            tag,
                            transaction_id: beat.transaction_id,
                        }
                    })
                }),
            )?;
        bus.consume(accepted);
        Ok(())
    }

    pub(super) fn advance_biu_read_inputs(
        &mut self,
        tick: u64,
    ) -> Result<(), C220MtePipelineError> {
        let Some(bus) = &mut self.biu_bus_reads else {
            return Ok(());
        };
        bus.advance_input(tick)?;
        if let Some(memory) = &mut self.timed_memory
            && memory.can_push_read()
            && let Some(request) = bus.take_command(tick)
        {
            memory.push_read(tick, request)?;
            if let C220MemoryReadId::Mte(tag) = request.tag {
                let request = self
                    .biu_returns
                    .as_ref()
                    .expect("MTE returns")
                    .progress(tag)
                    .expect("tracked request")
                    .request;
                if request.input.generated.last_in_instruction {
                    self.dma_tails.push(request.input.generated.instruction_id);
                }
            }
        }
        Ok(())
    }
}
