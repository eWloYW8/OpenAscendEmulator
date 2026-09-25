use super::{C220MtePipeline, C220MtePipelineError};
use crate::sim::c220::memory::biu_write::C220BiuWriteReturnKind;
use crate::sim::c220::memory::timed_memory::{C220MemoryWriteCommand, C220MemoryWriteId};

impl C220MtePipeline {
    pub(crate) fn cache_write_completion(&self, port: u32) -> Option<C220MemoryWriteId> {
        self.biu_bus_writes
            .as_ref()?
            .cache_returns(port, C220BiuWriteReturnKind::Completion)?
            .front()
            .filter(|response| response.ready_tick <= self.events.tick())
            .map(|response| response.tag)
    }

    pub(crate) fn connect_cache_write_port(&mut self) -> Result<u32, C220MtePipelineError> {
        if !self.is_idle() {
            return Err(C220MtePipelineError::CommandBusy);
        }
        Ok(self
            .biu_bus_writes
            .as_mut()
            .ok_or(C220MtePipelineError::BiuBusDisconnected)?
            .add_cache_port([2, 2])?)
    }

    pub(crate) fn send_cache_write(
        &mut self,
        command: C220MemoryWriteCommand,
    ) -> Result<bool, C220MtePipelineError> {
        Ok(self
            .biu_bus_writes
            .as_mut()
            .ok_or(C220MtePipelineError::BiuBusDisconnected)?
            .send_cache_command(self.events.tick(), command)?)
    }

    pub(crate) fn take_cache_write_completion(
        &mut self,
        port: u32,
    ) -> Result<Option<C220MemoryWriteId>, C220MtePipelineError> {
        Ok(self
            .biu_bus_writes
            .as_mut()
            .ok_or(C220MtePipelineError::BiuBusDisconnected)?
            .take_cache_return(self.events.tick(), port, C220BiuWriteReturnKind::Completion))
    }
}
