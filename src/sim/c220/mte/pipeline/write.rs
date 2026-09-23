use super::{C220MtePipeline, C220MtePipelineError, C220MtePipelineEvent};
use crate::sim::c220::memory::biu_write::{C220BiuMteBusWrites, C220BiuWriteReturnKind};
use crate::sim::c220::mte::interface::biu_write::command::{
    C220BiuWriteCommandTransfer, C220BiuWriteCommands, C220BiuWriteConfig, C220BiuWriteInput,
};
use crate::sim::c220::mte::uop::C220DmaUopRoute;
use std::num::NonZeroU32;

impl C220MtePipeline {
    pub fn connect_biu_bus_writes(
        &mut self,
        outstanding: NonZeroU32,
    ) -> Result<(), C220MtePipelineError> {
        if !self.is_idle() {
            return Err(C220MtePipelineError::CommandBusy);
        }
        if self.biu_write_commands.is_none() {
            return Err(C220MtePipelineError::BiuWriteCommandDisconnected);
        }
        self.biu_bus_writes = Some(C220BiuMteBusWrites::new(outstanding));
        Ok(())
    }

    pub fn biu_bus_writes(&self) -> Option<&C220BiuMteBusWrites> {
        self.biu_bus_writes.as_ref()
    }

    pub fn receive_biu_bus_write_return(
        &mut self,
        kind: C220BiuWriteReturnKind,
        tag: NonZeroU32,
    ) -> Result<bool, C220MtePipelineError> {
        if self.timed_memory.is_some() {
            return Err(C220MtePipelineError::MemoryOwnedWrite);
        }
        Ok(self
            .biu_bus_writes
            .as_mut()
            .ok_or(C220MtePipelineError::BiuBusDisconnected)?
            .receive(self.events.tick(), kind, tag)?)
    }

    pub(super) fn advance_biu_bus_returns(
        &mut self,
        tick: u64,
    ) -> Result<(), C220MtePipelineError> {
        let Some(bus) = &mut self.biu_bus_writes else {
            return Ok(());
        };
        if let Some(memory) = &mut self.timed_memory {
            for kind in [
                C220BiuWriteReturnKind::Dbid,
                C220BiuWriteReturnKind::Completion,
            ] {
                while let Some(tag) = memory.front(tick, kind) {
                    if !bus.receive(tick, kind, tag)? {
                        break;
                    }
                    memory.pop(kind);
                }
            }
        }
        bus.advance(tick)?;
        let dbid = bus.take_return(tick, C220BiuWriteReturnKind::Dbid);
        let completion = bus.take_return(tick, C220BiuWriteReturnKind::Completion);
        if let Some(tag) = dbid {
            self.deliver_biu_write_dbid(self.biu_subcore, tag)?;
        }
        if let Some(tag) = completion {
            self.deliver_biu_write_response(tag)?;
        }
        Ok(())
    }

    pub(super) fn advance_biu_bus_inputs(&mut self, tick: u64) -> Result<(), C220MtePipelineError> {
        let Some(bus) = &mut self.biu_bus_writes else {
            return Ok(());
        };
        if bus.can_receive_command()
            && let Some(command) = self
                .biu_write_commands
                .as_mut()
                .expect("connected command route")
                .take_request(tick)?
        {
            bus.push_command(tick, command)?;
        }
        if let Some(data) = self.biu_write_data.take_request(tick)? {
            bus.push_data(tick, data)?;
        }
        if let Some(memory) = &mut self.timed_memory {
            if memory.can_push(C220BiuWriteReturnKind::Dbid)
                && let Some(command) = bus.take_command(tick)
            {
                memory.push_command(tick, command)?;
            }
            if memory.can_push(C220BiuWriteReturnKind::Completion)
                && let Some(data) = bus.take_data(tick)
            {
                memory.push_data(tick, data)?;
            }
        }
        Ok(())
    }

    pub fn connect_mte3_biu(
        &mut self,
        config: C220BiuWriteConfig,
    ) -> Result<(), C220MtePipelineError> {
        if self.timed_memory.is_some() {
            return Err(C220MtePipelineError::MemoryOwnedWrite);
        }
        self.configure_biu_write_source(self.biu_subcore, config.source_bandwidth)?;
        self.connect_mte3_biu_retirement()?;
        self.biu_write_commands = Some(C220BiuWriteCommands::new(config));
        Ok(())
    }

    pub fn biu_write_commands(&self) -> Option<&C220BiuWriteCommands> {
        self.biu_write_commands.as_ref()
    }

    pub fn take_biu_write_command(
        &mut self,
    ) -> Result<Option<C220BiuWriteCommandTransfer>, C220MtePipelineError> {
        if self.timed_memory.is_some() {
            return Err(C220MtePipelineError::MemoryOwnedWrite);
        }
        if let Some(bus) = &mut self.biu_bus_writes {
            return Ok(bus.take_command(self.events.tick()));
        }
        Ok(self
            .biu_write_commands
            .as_mut()
            .ok_or(C220MtePipelineError::BiuWriteCommandDisconnected)?
            .take_request(self.events.tick())?)
    }

    pub(super) fn advance_biu_write_commands(
        &mut self,
        tick: u64,
    ) -> Result<(), C220MtePipelineError> {
        let Some(commands) = self.biu_write_commands.as_mut() else {
            return Ok(());
        };
        if commands.can_push(self.biu_subcore)
            && let Some(generated) = self.mte3.output()
        {
            let gather_stride = if generated.request.route == C220DmaUopRoute::SourceGapGather {
                let descriptor = self
                    .mte3
                    .records()
                    .find(|record| record.instruction_id == generated.instruction_id)
                    .expect("generated command record")
                    .transfer
                    .descriptor;
                Some((u32::from(descriptor.burst_length) + u32::from(descriptor.source_gap)) * 32)
            } else {
                None
            };
            assert!(commands.push(
                tick,
                C220BiuWriteInput {
                    subcore: self.biu_subcore,
                    generated,
                    gather_stride
                }
            )?);
            assert_eq!(self.mte3.take_output(), Some(generated));
        }
        let cycle = commands.advance(tick)?;
        if let Some(sent) = cycle.sent {
            let index = match sent.input.subcore {
                super::C220BiuSubcore::Vector0 => 0,
                super::C220BiuSubcore::Vector1 => 1,
                super::C220BiuSubcore::Cube => {
                    return Err(C220MtePipelineError::UbReadSubcoreRequired);
                }
            };
            self.biu_write_source[index]
                .as_mut()
                .expect("connected source")
                .register(sent.source_request())?;
        }
        if cycle.selected.is_some() || cycle.stall.is_some() || cycle.sent.is_some() {
            self.trace
                .push(C220MtePipelineEvent::BiuWriteCommand(cycle));
        }
        Ok(())
    }
}
