use super::*;
use crate::isa::c220::mte::load2d::{C220Load2dDestination, C220Load2dTransfer};
use crate::sim::c220::mte::load2d::C220Load2dExternalRequests;
use crate::sim::c220::mte::uop::C220DmaUopMode;

impl C220MtePipeline {
    pub fn external_load2d_generator(&self) -> &C220DmaFrontend {
        &self.external_load2d
    }
    pub(crate) fn validate_external_load2d_connection(&self) -> Result<(), C220MtePipelineError> {
        if self.biu_read.is_none() || self.biu_subcore != C220BiuSubcore::Cube {
            return Err(C220MtePipelineError::BiuL1SubcoreRequired);
        }
        Ok(())
    }
    pub fn can_issue_external_load2d(&self) -> bool {
        self.biu_read.is_some()
            && self.biu_subcore == C220BiuSubcore::Cube
            && self.external_load2d.can_issue()
            && (self.selected_mte2_generator == Some(Mte2Generator::ExternalLoad2d)
                || self.mte2_generator_idle())
    }

    pub fn issue_external_load2d(
        &mut self,
        id: u64,
        transfer: C220Load2dTransfer,
        mode: C220DmaUopMode,
    ) -> Result<C220DmaIssue, C220MtePipelineError> {
        let requests = C220Load2dExternalRequests::new(transfer, mode)?;
        if transfer.descriptor.repeat_count == 0 {
            return Ok(C220DmaIssue {
                tick: self.events.tick(),
                instruction_id: id,
                completion_ready: true,
            });
        }
        self.validate_external_load2d_connection()?;
        if !self.can_issue_external_load2d() {
            return Err(C220MtePipelineError::CommandBusy);
        }
        let issue = self.load2d_events.issue_load2d(
            &mut self.events,
            &mut self.external_load2d,
            id,
            requests,
        )?;
        let destination = match transfer.instruction.destination {
            C220Load2dDestination::L0a => C220BiuWriteDestination::L0A,
            C220Load2dDestination::L0b => C220BiuWriteDestination::L0B,
            C220Load2dDestination::L1 => C220BiuWriteDestination::L1,
            C220Load2dDestination::Reserved => unreachable!("validated request plan"),
        };
        self.load2d_destinations.insert(id, destination);
        self.selected_mte2_generator = Some(Mte2Generator::ExternalLoad2d);
        Ok(issue)
    }

    pub(super) fn advance_external_load2d(
        &mut self,
        phase: C220MteGeneratorCallback,
    ) -> Result<(), C220MtePipelineError> {
        let ready = self
            .biu_read
            .as_ref()
            .is_some_and(|frontend| frontend.can_push(C220BiuSubcore::Cube));
        let outcome = self.load2d_events.handle(
            phase,
            &mut self.events,
            &mut self.external_load2d,
            self.dma_hardware_sync_blocked,
            ready,
        )?;
        if let C220DmaEventOutcome::Sent(send) = outcome
            && let Some(generated) = send.sent
        {
            let destination = self.load2d_destinations[&generated.instruction_id];
            assert!(self.biu_events.push(
                &mut self.events,
                self.biu_read.as_mut().expect("connected Cube BIU"),
                C220BiuReadInput {
                    subcore: C220BiuSubcore::Cube,
                    destination,
                    prefetch: false,
                    generated,
                }
            )?);
        }
        if outcome != C220DmaEventOutcome::Readiness {
            self.trace
                .push(C220MtePipelineEvent::ExternalLoad2d(outcome));
        }
        Ok(())
    }

    pub(super) fn cube_read_output_ready(&self) -> bool {
        let destination = self
            .biu_returns
            .as_ref()
            .and_then(|returns| returns.adapter(C220BiuSubcore::Cube).front())
            .map(|output| output.request.input.destination);
        match destination {
            Some(C220BiuWriteDestination::L0A) => self.l0[0].can_push(C220L0WritePort::Port2),
            Some(C220BiuWriteDestination::L0B) => self.l0[1].can_push(C220L0WritePort::Port2),
            _ => self.write_interface.can_push(C220MteL1WritePort::Port0),
        }
    }
}
