use super::{C220BiuSubcore, C220MtePipeline, C220MtePipelineError, C220MtePipelineEvent};
use crate::sim::c220::memory::biu_write::{C220BiuMteBusWrites, C220BiuWriteReturnKind};
use crate::sim::c220::memory::timed_memory::{
    C220MemoryWriteCommand, C220MemoryWriteId, C220MemoryWriteTransfer,
};
use crate::sim::c220::mte::interface::biu_write::command::{
    C220BiuWriteCommandTransfer, C220BiuWriteCommands, C220BiuWriteConfig, C220BiuWriteInput,
};
use crate::sim::c220::mte::interface::biu_write::data::{
    C220BiuWriteData, C220BiuWriteDataError, C220BiuWriteDataPort, C220BiuWriteResponse,
};
use crate::sim::c220::mte::uop::C220DmaUopRoute;
use std::num::NonZeroU32;

impl C220MtePipeline {
    pub fn receive_biu_write_dbid(
        &mut self,
        core: C220BiuSubcore,
        tag: NonZeroU32,
    ) -> Result<(), C220MtePipelineError> {
        if self.biu_bus_writes.is_some() {
            return Err(C220MtePipelineError::BiuBusOwnedResponse);
        }
        self.deliver_biu_write_dbid(core, tag)
    }

    pub(super) fn deliver_biu_write_dbid(
        &mut self,
        core: C220BiuSubcore,
        tag: NonZeroU32,
    ) -> Result<(), C220MtePipelineError> {
        if let Some(commands) = &self.biu_write_commands
            && commands.awaiting_dbid(tag)?.input.subcore != core
        {
            return Err(C220MtePipelineError::BiuWriteWrongSubcore);
        }
        if core == C220BiuSubcore::Cube {
            self.biu_cube_source.receive_dbid(self.events.tick(), tag)?;
        } else {
            self.biu_write_source[Self::ub_read_index(core)?]
                .as_mut()
                .ok_or(C220MtePipelineError::BiuWriteSourceDisconnected)?
                .receive_dbid(self.events.tick(), tag)?;
        }
        if let Some(commands) = &mut self.biu_write_commands {
            commands.mark_dbid(tag);
        }
        Ok(())
    }

    pub fn biu_write_data(&self) -> &C220BiuWriteDataPort {
        &self.biu_write_data
    }

    pub fn take_biu_write_data(
        &mut self,
    ) -> Result<Option<C220BiuWriteData>, C220MtePipelineError> {
        if self.timed_memory.is_some() {
            return Err(C220MtePipelineError::MemoryOwnedWrite);
        }
        if let Some(bus) = &mut self.biu_bus_writes {
            return Ok(bus.take_data(self.events.tick()));
        }
        Ok(self.biu_write_data.take_request(self.events.tick())?)
    }

    pub fn receive_biu_write_response(
        &mut self,
        tag: NonZeroU32,
    ) -> Result<C220BiuWriteResponse, C220MtePipelineError> {
        if self.biu_bus_writes.is_some() {
            return Err(C220MtePipelineError::BiuBusOwnedResponse);
        }
        self.deliver_biu_write_response(tag)
    }

    pub(super) fn deliver_biu_write_response(
        &mut self,
        tag: NonZeroU32,
    ) -> Result<C220BiuWriteResponse, C220MtePipelineError> {
        let data = self
            .biu_write_data
            .delivered()
            .find(|data| data.source.request.tag == tag)
            .copied()
            .ok_or(C220BiuWriteDataError::UnexpectedResponse(tag))?;
        let notify = self.mte3.uses_biu_retirement()
            && data.subcore != C220BiuSubcore::Cube
            && data.subcore == self.biu_subcore
            && data.source.request.last_in_instruction;
        if notify {
            self.mte3
                .validate_biu_retirement(data.source.request.instruction_id)?;
        }
        let response = self
            .biu_write_data
            .receive_response(self.events.tick(), tag)?;
        if data.subcore == C220BiuSubcore::Cube {
            self.biu_cube_source
                .release_response(self.events.tick(), tag)?;
            if data.source.request.last_in_instruction {
                self.fixp_completions
                    .push(data.source.request.instruction_id);
            }
        } else {
            let index = Self::ub_read_index(data.subcore)?;
            self.biu_write_source[index]
                .as_mut()
                .expect("response source")
                .release_response(tag);
        }
        if let Some(commands) = &mut self.biu_write_commands {
            let command = commands.release_tag(tag)?;
            assert_eq!(command.source_request(), data.source.request);
        }
        if notify {
            self.mte3
                .notify_biu_retirement(data.source.request.instruction_id)?;
        }
        self.trace
            .push(C220MtePipelineEvent::BiuWriteResponse(response));
        Ok(response)
    }

    pub(super) fn advance_biu_write_data(&mut self, tick: u64) -> Result<(), C220MtePipelineError> {
        let vectors = self.biu_write_source.each_ref().map(|source| {
            source
                .as_ref()
                .and_then(|source| source.data_ready().front().copied())
        });
        let cube = self.biu_cube_source.data_ready().front().copied();
        let outcome = self
            .biu_write_data
            .send(tick, [cube, vectors[0], vectors[1]])?;
        if let Some(sent) = outcome.sent {
            let consumed = if sent.subcore == C220BiuSubcore::Cube {
                self.biu_cube_source.take_data_ready(tick)?
            } else {
                let index = Self::ub_read_index(sent.subcore)?;
                self.biu_write_source[index]
                    .as_mut()
                    .expect("selected source")
                    .take_data_ready(tick)
            };
            assert_eq!(consumed, Some(sent.source));
        }
        if outcome.selected.is_some() {
            self.trace.push(C220MtePipelineEvent::BiuWriteData(outcome));
        }
        Ok(())
    }

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
                while let Some(C220MemoryWriteId::Mte(tag)) = memory.front(tick, kind) {
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
            let core = self
                .biu_write_commands
                .as_ref()
                .expect("connected command route")
                .awaiting_dbid(tag)?
                .input
                .subcore;
            self.deliver_biu_write_dbid(core, tag)?;
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
                let request = command.command.input.generated.request;
                memory.push_command(
                    tick,
                    C220MemoryWriteCommand {
                        ready_tick: tick,
                        tag: C220MemoryWriteId::Mte(command.command.tag),
                        address: request.destination_address,
                        bytes: request.bytes,
                    },
                )?;
            }
            if memory.can_push(C220BiuWriteReturnKind::Completion)
                && let Some(data) = bus.take_data(tick)
            {
                memory.push_data(
                    tick,
                    C220MemoryWriteTransfer {
                        ready_tick: tick,
                        tag: C220MemoryWriteId::Mte(data.source.request.tag),
                    },
                )?;
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

    /// Connect FIX writes without enabling the vector MTE3 source path.
    pub fn connect_fixp_biu(
        &mut self,
        config: C220BiuWriteConfig,
    ) -> Result<(), C220MtePipelineError> {
        if !self.is_idle() {
            return Err(C220MtePipelineError::CommandBusy);
        }
        if self.timed_memory.is_some() || self.biu_write_commands.is_some() {
            return Err(C220MtePipelineError::BiuOwnedSource);
        }
        self.biu_write_commands = Some(C220BiuWriteCommands::new(config));
        Ok(())
    }

    pub fn biu_cube_source(&self) -> &super::C220BiuCubeWriteSource {
        &self.biu_cube_source
    }

    pub fn fixp_store_buffer(&self) -> &super::C220FixpStoreBuffer {
        &self.fixp_stores
    }

    /// Publish FIX source availability at packet generation, independently of
    /// downstream BIU command credit. The packet retains its captured mode.
    pub fn packetize_fixp_biu_output(
        &mut self,
        output: &mut crate::sim::c220::mte::fixp::C220FixpExternalOutput,
        writes: &mut crate::sim::c220::mte::fixp::C220FixpDispatchPipeline,
        mode: crate::sim::c220::mte::uop::C220DmaUopMode,
    ) -> Result<Option<crate::sim::c220::mte::fixp::C220FixpBiuWrite>, C220MtePipelineError> {
        if self.biu_write_commands.is_none() {
            return Err(C220MtePipelineError::BiuWriteCommandDisconnected);
        }
        Ok(writes.packetize_external(self.events.tick(), output, &mut self.fixp_stores, mode)?)
    }

    /// Handoff releases write-dispatch capacity, not the store token or the
    /// instruction's retirement dependency on the final BIU response.
    pub fn send_fixp_biu_output(
        &mut self,
        writes: &mut crate::sim::c220::mte::fixp::C220FixpDispatchPipeline,
    ) -> Result<
        crate::sim::c220::mte::fixp::C220FixpWriteProgress<
            crate::sim::c220::mte::fixp::C220FixpDispatchPacket,
        >,
        C220MtePipelineError,
    > {
        let commands = self
            .biu_write_commands
            .as_mut()
            .ok_or(C220MtePipelineError::BiuWriteCommandDisconnected)?;
        Ok(writes.send_shared(
            self.events.tick(),
            &mut self.fixp_write,
            &mut self.interface,
            Some(commands),
        )?)
    }

    pub(super) fn advance_biu_cube_source(
        &mut self,
        tick: u64,
    ) -> Result<(), C220MtePipelineError> {
        let Some(commands) = &mut self.biu_write_commands else {
            return Ok(());
        };
        if let Some(tag) = self
            .biu_cube_source
            .ingress(tick, |tag| commands.begin_source(tag).last_in_instruction)?
        {
            self.trace
                .push(C220MtePipelineEvent::BiuCubeSourceStarted { tick, tag });
        }
        let ready = self.biu_cube_source.egress(tick, &mut self.fixp_stores)?;
        if let Some(probe) = self.biu_cube_source.last_probe() {
            self.trace
                .push(C220MtePipelineEvent::BiuCubeSourceProbe(probe));
        }
        if let Some(ready) = ready {
            self.trace
                .push(C220MtePipelineEvent::BiuCubeSourceReady(ready));
        }
        Ok(())
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
            let transfer = self
                .mte3
                .records()
                .find(|record| record.instruction_id == generated.instruction_id)
                .expect("generated command record")
                .transfer;
            let gather_stride = if generated.request.route == C220DmaUopRoute::SourceGapGather {
                let descriptor = transfer.descriptor;
                Some((u32::from(descriptor.burst_length) + u32::from(descriptor.source_gap)) * 32)
            } else {
                None
            };
            assert!(commands.push(
                tick,
                C220BiuWriteInput {
                    store_token: None,
                    subcore: self.biu_subcore,
                    generated: super::C220DmaGenerated {
                        mode: super::super::uop::C220DmaUopMode::from_mode_word(
                            transfer.biu_mode_word
                        ),
                        ..generated
                    },
                    gather_stride
                }
            )?);
            assert_eq!(self.mte3.take_output(), Some(generated));
        }
        let cycle = commands.advance(tick)?;
        if let Some(sent) = cycle.sent {
            if sent.input.subcore == super::C220BiuSubcore::Cube {
                self.biu_cube_source.register(
                    sent.source_request(),
                    sent.input.store_token.expect("Cube store token"),
                )?;
            } else {
                let index = Self::ub_read_index(sent.input.subcore)?;
                self.biu_write_source[index]
                    .as_mut()
                    .expect("connected source")
                    .register(sent.source_request())?;
            }
        }
        if cycle.selected.is_some() || cycle.stall.is_some() || cycle.sent.is_some() {
            self.trace
                .push(C220MtePipelineEvent::BiuWriteCommand(cycle));
        }
        Ok(())
    }
}
