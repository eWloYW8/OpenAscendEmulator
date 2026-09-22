use super::{C220BiuSubcore, C220MtePipeline, C220MtePipelineError, C220MtePipelineEvent};
use crate::sim::c220::memory::C220UbCycle;
use crate::sim::c220::memory::ub_service::{C220UbMtePort, C220UbMteService, C220UbServiceRequest};

impl C220MtePipeline {
    pub fn ub_memory(
        &self,
        core: C220BiuSubcore,
    ) -> Result<&C220UbMteService, C220MtePipelineError> {
        Ok(&self.ub_memory[self.ub_write_index(core)?])
    }

    /// Runs after this tick's higher-priority Vector grants. Each vector
    /// subcore owns a separate UB; the connected core supplies only its mask.
    pub fn advance_ub_service<'a>(
        &mut self,
        vector_cycles: impl IntoIterator<Item = &'a C220UbCycle>,
    ) -> Result<(), C220MtePipelineError> {
        let tick = self.events.tick();
        if self.biu_read.is_none() || self.last_ub_service == Some(tick) {
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
