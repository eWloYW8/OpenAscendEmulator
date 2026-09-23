use super::{C220MtePipeline, C220MtePipelineError};
use crate::sim::c220::memory::timed_memory::{C220TimedMemory, C220TimedMemoryConfig};

impl C220MtePipeline {
    pub fn connect_timed_memory(
        &mut self,
        config: C220TimedMemoryConfig,
    ) -> Result<(), C220MtePipelineError> {
        if !self.is_idle() {
            return Err(C220MtePipelineError::CommandBusy);
        }
        if self.biu_bus_writes.is_none() && self.biu_bus_reads.is_none() {
            return Err(C220MtePipelineError::BiuBusDisconnected);
        }
        if self.biu_read.is_some() && self.biu_bus_reads.is_none() {
            return Err(C220MtePipelineError::BiuBusDisconnected);
        }
        if self.biu_write_commands.is_some() && self.biu_bus_writes.is_none() {
            return Err(C220MtePipelineError::BiuBusDisconnected);
        }
        self.timed_memory = Some(C220TimedMemory::new(config, self.events.tick())?);
        Ok(())
    }

    pub fn timed_memory(&self) -> Option<&C220TimedMemory> {
        self.timed_memory.as_ref()
    }

    pub(crate) fn take_dma_tails(&mut self) -> Vec<u64> {
        std::mem::take(&mut self.dma_tails)
    }
}
