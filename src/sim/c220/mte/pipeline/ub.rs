use super::{C220BiuSubcore, C220MtePipeline, C220MtePipelineError, C220MtePipelineEvent};
use crate::sim::c220::memory::C220UbCycle;
use crate::sim::c220::memory::ub_service::{
    C220UbService, C220UbServicePort, C220UbServiceRequest, C220UbVectorActivity,
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

    pub const fn ub_vector_subcore(&self) -> C220BiuSubcore {
        self.biu_subcore
    }

    pub(super) fn ub_read_index(core: C220BiuSubcore) -> Result<usize, C220MtePipelineError> {
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

    pub fn ub_memory(&self, core: C220BiuSubcore) -> Result<&C220UbService, C220MtePipelineError> {
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
            if let Some(response) = memory.take_response(tick, C220UbServicePort::MteRead)? {
                let request = read.receive_response(tick, response.request.id)?;
                self.trace
                    .push(C220MtePipelineEvent::UbReadResponse(core, request));
            }
            if memory.can_receive(tick, C220UbServicePort::MteRead)
                && let Some(request) = read.take_request(tick)?
            {
                assert!(memory.receive(
                    tick,
                    C220UbServicePort::MteRead,
                    C220UbServiceRequest {
                        id: request.id,
                        address: request.fragment.address,
                        bytes: request.fragment.bytes,
                    }
                )?);
                self.trace
                    .push(C220MtePipelineEvent::UbReadRequest(core, request));
            }
            if let Some(response) = memory.take_response(tick, C220UbServicePort::MteWrite0)? {
                let request = interface.receive_response(tick, response.request.id)?;
                self.trace
                    .push(C220MtePipelineEvent::UbResponse(core, request));
            }
            if memory.can_receive(tick, C220UbServicePort::MteWrite0)
                && let Some(request) = interface.take_request(tick)?
            {
                assert!(memory.receive(
                    tick,
                    C220UbServicePort::MteWrite0,
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
                C220UbVectorActivity {
                    bank_mask: if connected { vector_banks } else { 0 },
                    triggered: connected && vector_trigger,
                    ..Default::default()
                },
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
