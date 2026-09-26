use super::{
    C220Nd2NzEngine, C220Nd2NzEngineError, C220Nd2NzIssuedRead, C220Nd2NzReadRoute,
    C220Nd2NzResponse,
};
use crate::sim::c220::mte::{
    dma::C220DmaGenerated,
    interface::biu_read::{
        C220BiuReadError, C220BiuReadFrontend, C220BiuReadInput, C220BiuSubcore,
        write::C220BiuWriteDestination,
    },
    uop::{C220DmaDestinationLayout, C220DmaUopRequest, C220DmaUopRoute},
};

impl C220Nd2NzIssuedRead {
    pub fn biu_input(&self) -> Result<C220BiuReadInput, C220Nd2NzEngineError> {
        C220Nd2NzResponse::new(&self.request)?;
        let row_slot = match self.request.route {
            C220Nd2NzReadRoute::PerRow => {
                let row = self.request.elements[0].row_slot;
                if row >= 8 {
                    return Err(C220BiuReadError::InvalidNd2NzRequest.into());
                }
                Some(row as u8)
            }
            C220Nd2NzReadRoute::ContiguousRows => None,
        };
        Ok(C220BiuReadInput {
            subcore: C220BiuSubcore::Cube,
            destination: C220BiuWriteDestination::Nd2Nz { row_slot },
            prefetch: false,
            generated: C220DmaGenerated {
                sid: Some(self.sid),
                instruction_id: self.instruction_id,
                uop_index: self.request_id,
                ready_tick: self.ready_tick,
                mode: self.mode,
                out_of_order: false,
                last_in_instruction: self.request.last_in_instruction,
                request: C220DmaUopRequest {
                    route: C220DmaUopRoute::Ordinary,
                    burst_index: self.request.matrix_index,
                    source_address: self.request.source_address,
                    destination_address: 0,
                    bytes: self.request.bytes,
                    last_in_burst: true,
                },
                destination: C220DmaDestinationLayout {
                    base: 0,
                    burst_bytes: self.request.bytes,
                    burst_stride: 0,
                },
            },
        })
    }
}

impl C220Nd2NzEngine {
    pub fn send_biu(
        &mut self,
        tick: u64,
        frontend: &mut C220BiuReadFrontend,
        hardware_sync_blocked: bool,
    ) -> Result<Option<C220Nd2NzIssuedRead>, C220Nd2NzEngineError> {
        self.observe(tick)?;
        Self::callback(&mut self.read_tick, tick, "read send")?;
        let Some(head) = self.generated.front() else {
            return Ok(None);
        };
        if head.ready_tick > tick || hardware_sync_blocked {
            return Ok(None);
        }
        let response = C220Nd2NzResponse::new(&head.request)?;
        if !frontend.push(tick, head.biu_input()?)? {
            return Ok(None);
        }
        self.responses
            .insert((head.instruction_id, head.request_id), response);
        Ok(self.generated.pop_front())
    }
}
