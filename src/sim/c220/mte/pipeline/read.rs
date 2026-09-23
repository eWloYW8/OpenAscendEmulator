use std::num::NonZeroU32;

use super::{C220MtePipeline, C220MtePipelineError};
use crate::sim::c220::memory::biu_read::C220BiuMteBusReads;

impl C220MtePipeline {
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
        self.biu_bus_reads = Some(C220BiuMteBusReads::new(outstanding));
        Ok(())
    }

    pub fn biu_bus_reads(&self) -> Option<&C220BiuMteBusReads> {
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
            .receive(tick, bus.heads(tick))?;
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
            if request.input.generated.last_in_instruction {
                self.dma_tails.push(request.input.generated.instruction_id);
            }
        }
        Ok(())
    }
}
