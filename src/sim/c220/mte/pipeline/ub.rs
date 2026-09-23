use super::{C220BiuSubcore, C220MtePipeline, C220MtePipelineError, C220MtePipelineEvent};
use crate::sim::c220::memory::C220UbCycle;
use crate::sim::c220::memory::ub_service::{C220UbMtePort, C220UbMteService, C220UbServiceRequest};
use crate::sim::c220::mte::interface::biu_write::data::{
    C220BiuWriteData, C220BiuWriteDataError, C220BiuWriteDataPort, C220BiuWriteResponse,
};
use crate::sim::c220::mte::interface::biu_write::{C220BiuWriteSource, C220BiuWriteSourceRequest};
use crate::sim::c220::mte::interface::ub_read::{
    C220UbReadAcknowledgment, C220UbReadFragment, C220UbReadInterface,
};
use std::num::NonZeroU32;

impl C220MtePipeline {
    pub fn configure_biu_write_source(
        &mut self,
        core: C220BiuSubcore,
        bandwidth: NonZeroU32,
    ) -> Result<(), C220MtePipelineError> {
        let index = Self::ub_read_index(core)?;
        if !self.is_idle() {
            return Err(C220MtePipelineError::CommandBusy);
        }
        self.biu_write_source[index] = Some(C220BiuWriteSource::new(bandwidth));
        Ok(())
    }

    pub fn biu_write_source(
        &self,
        core: C220BiuSubcore,
    ) -> Result<Option<&C220BiuWriteSource>, C220MtePipelineError> {
        Ok(self.biu_write_source[Self::ub_read_index(core)?].as_ref())
    }

    pub fn register_biu_write_source(
        &mut self,
        core: C220BiuSubcore,
        request: C220BiuWriteSourceRequest,
    ) -> Result<(), C220MtePipelineError> {
        if self.biu_write_commands.is_some() {
            return Err(C220MtePipelineError::BiuOwnedSource);
        }
        if self.biu_write_source.iter().flatten().any(|source| {
            source
                .pending_requests()
                .any(|pending| pending.tag == request.tag)
        }) {
            return Err(super::C220BiuWriteSourceError::DuplicateTag(request.tag).into());
        }
        self.biu_write_source[Self::ub_read_index(core)?]
            .as_mut()
            .ok_or(C220MtePipelineError::BiuWriteSourceDisconnected)?
            .register(request)?;
        Ok(())
    }

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
        self.biu_write_source[Self::ub_read_index(core)?]
            .as_mut()
            .ok_or(C220MtePipelineError::BiuWriteSourceDisconnected)?
            .receive_dbid(self.events.tick(), tag)?;
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
            && data.subcore == self.biu_subcore
            && data.source.request.last_in_instruction;
        if notify {
            self.mte3
                .validate_biu_retirement(data.source.request.instruction_id)?;
        }
        let response = self
            .biu_write_data
            .receive_response(self.events.tick(), tag)?;
        let index = Self::ub_read_index(data.subcore)?;
        self.biu_write_source[index]
            .as_mut()
            .expect("response source")
            .release_response(tag);
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
        // Cube source readiness is supplied by FIXP, not by either vector UB.
        let outcome = self
            .biu_write_data
            .send(tick, [None, vectors[0], vectors[1]])?;
        if let Some(sent) = outcome.sent {
            let index = Self::ub_read_index(sent.subcore)?;
            let consumed = self.biu_write_source[index]
                .as_mut()
                .expect("selected source")
                .take_data_ready(tick);
            assert_eq!(consumed, Some(sent.source));
        }
        if outcome.selected.is_some() {
            self.trace.push(C220MtePipelineEvent::BiuWriteData(outcome));
        }
        Ok(())
    }

    pub const fn ub_vector_subcore(&self) -> C220BiuSubcore {
        self.biu_subcore
    }

    fn ub_read_index(core: C220BiuSubcore) -> Result<usize, C220MtePipelineError> {
        match core {
            C220BiuSubcore::Vector0 => Ok(0),
            C220BiuSubcore::Vector1 => Ok(1),
            C220BiuSubcore::Cube => Err(C220MtePipelineError::UbReadSubcoreRequired),
        }
    }

    pub fn ub_read_interface(
        &self,
        core: C220BiuSubcore,
    ) -> Result<&C220UbReadInterface, C220MtePipelineError> {
        Ok(&self.ub_read[Self::ub_read_index(core)?])
    }

    /// The BIU source packetizer supplies aligned fragments. Its write tags,
    /// write requests and destination responses remain separate from UB reads.
    pub fn push_ub_read(
        &mut self,
        core: C220BiuSubcore,
        fragment: C220UbReadFragment,
    ) -> Result<bool, C220MtePipelineError> {
        let index = Self::ub_read_index(core)?;
        if self.biu_write_source[index].is_some() {
            return Err(C220MtePipelineError::BiuOwnedUbRead);
        }
        Ok(self.ub_read[index].push(self.events.tick(), fragment)?)
    }

    pub fn take_ub_read_completion(
        &mut self,
        core: C220BiuSubcore,
        tag: u32,
    ) -> Result<Option<C220UbReadAcknowledgment>, C220MtePipelineError> {
        let index = Self::ub_read_index(core)?;
        if self.biu_write_source[index].is_some() {
            return Err(C220MtePipelineError::BiuOwnedUbRead);
        }
        Ok(self.ub_read[index].take_completion(self.events.tick(), tag)?)
    }

    pub fn ub_memory(
        &self,
        core: C220BiuSubcore,
    ) -> Result<&C220UbMteService, C220MtePipelineError> {
        Ok(&self.ub_memory[Self::ub_read_index(core)?])
    }

    /// Runs after this tick's higher-priority Vector grants. Each vector
    /// subcore owns a separate UB; the connected core supplies only its mask.
    pub fn advance_ub_service<'a>(
        &mut self,
        vector_cycles: impl IntoIterator<Item = &'a C220UbCycle>,
    ) -> Result<(), C220MtePipelineError> {
        let tick = self.events.tick();
        if self.last_ub_service == Some(tick) {
            return Ok(());
        }
        let mut vector_banks = 0;
        let mut vector_trigger = false;
        for cycle in vector_cycles.into_iter().filter(|cycle| cycle.tick == tick) {
            vector_banks |= cycle.bank_mask;
            vector_trigger = true;
        }
        for (index, core) in [C220BiuSubcore::Vector0, C220BiuSubcore::Vector1]
            .into_iter()
            .enumerate()
        {
            let interface = &mut self.ub_write[index];
            let memory = &mut self.ub_memory[index];
            let read = &mut self.ub_read[index];
            if let Some(response) = memory.take_response(tick, C220UbMtePort::Read)? {
                let request = read.receive_response(tick, response.request.id)?;
                self.trace
                    .push(C220MtePipelineEvent::UbReadResponse(core, request));
            }
            if memory.can_receive(tick, C220UbMtePort::Read)
                && let Some(request) = read.take_request(tick)?
            {
                assert!(memory.receive(
                    tick,
                    C220UbMtePort::Read,
                    C220UbServiceRequest {
                        id: request.id,
                        address: request.fragment.address,
                        bytes: request.fragment.bytes,
                    }
                )?);
                self.trace
                    .push(C220MtePipelineEvent::UbReadRequest(core, request));
            }
            if let Some(response) = memory.take_response(tick, C220UbMtePort::Write0)? {
                let request = interface.receive_response(tick, response.request.id)?;
                self.trace
                    .push(C220MtePipelineEvent::UbResponse(core, request));
            }
            if memory.can_receive(tick, C220UbMtePort::Write0)
                && let Some(request) = interface.take_request(tick)?
            {
                assert!(memory.receive(
                    tick,
                    C220UbMtePort::Write0,
                    C220UbServiceRequest {
                        id: request.id,
                        address: request.fragment.destination_address,
                        bytes: request.fragment.bytes,
                    }
                )?);
                self.trace
                    .push(C220MtePipelineEvent::UbRequest(core, request));
            }
            let connected = core == self.biu_subcore;
            let cycle = memory.arbitrate(
                tick,
                if connected { vector_banks } else { 0 },
                connected && vector_trigger,
            )?;
            if !cycle.decisions.is_empty() || !cycle.completed.is_empty() {
                self.trace
                    .push(C220MtePipelineEvent::UbService(core, cycle));
            }
            memory.send_responses(tick)?;
        }
        self.last_ub_service = Some(tick);
        Ok(())
    }
}
