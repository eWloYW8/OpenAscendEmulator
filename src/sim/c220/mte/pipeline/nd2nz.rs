use super::*;
use crate::isa::c220::mte::nd2nz::C220Nd2NzTransfer;
use crate::sim::c220::mte::{
    nd2nz::{C220Nd2NzReadRoute, C220Nd2NzStagingConfig},
    uop::C220DmaUopMode,
};

impl C220MtePipeline {
    pub fn configure_nd2nz(
        &mut self,
        config: C220Nd2NzStagingConfig,
    ) -> Result<(), C220MtePipelineError> {
        if !self.is_idle() || self.active_cycle.is_some() {
            return Err(C220MtePipelineError::CommandBusy);
        }
        self.validate_external_load2d_connection()?;
        self.nd2nz = Some(C220Nd2NzEngine::new(config)?);
        Ok(())
    }

    pub fn nd2nz_engine(&self) -> Option<&C220Nd2NzEngine> {
        self.nd2nz.as_ref()
    }

    pub fn can_issue_nd2nz(&self) -> bool {
        self.biu_subcore == C220BiuSubcore::Cube
            && self.biu_read.is_some()
            && self.nd2nz.as_ref().is_some_and(C220Nd2NzEngine::can_submit)
            && (self.selected_mte2_generator == Some(Mte2Generator::Nd2Nz)
                || self.mte2_generator_idle())
    }

    pub fn issue_nd2nz(
        &mut self,
        id: u64,
        transfer: C220Nd2NzTransfer,
        mode: C220DmaUopMode,
    ) -> Result<C220DmaIssue, C220MtePipelineError> {
        let tick = self.events.tick();
        if transfer.is_disabled() {
            return Ok(C220DmaIssue {
                tick,
                instruction_id: id,
                completion_ready: true,
            });
        }
        self.validate_external_load2d_connection()?;
        self.nd2nz
            .as_ref()
            .ok_or(C220MtePipelineError::Nd2NzUnconfigured)?;
        if !self.can_issue_nd2nz() {
            return Err(C220MtePipelineError::CommandBusy);
        }
        let engine = self.nd2nz.as_mut().expect("configured ND2NZ");
        let route = C220Nd2NzReadRoute::select(transfer, engine.staging().config().alignment_depth);
        assert!(engine.submit(tick, id, transfer, route, mode)?);
        self.nd2nz_events.arm(&mut self.events);
        self.selected_mte2_generator = Some(Mte2Generator::Nd2Nz);
        Ok(C220DmaIssue {
            tick,
            instruction_id: id,
            completion_ready: false,
        })
    }

    pub(super) fn advance_nd2nz(
        &mut self,
        phase: C220Nd2NzCallback,
    ) -> Result<(), C220MtePipelineError> {
        let Some(engine) = self.nd2nz.as_mut() else {
            return Ok(());
        };
        let event = self.nd2nz_events.handle(
            phase,
            &mut self.events,
            engine,
            self.biu_read.as_mut().expect("ND2NZ BIU connection"),
            &mut self.write_interface,
            self.dma_hardware_sync_blocked,
        )?;
        if matches!(event, C220Nd2NzEvent::Read(Some(_))) {
            self.biu_events.arm_input(&mut self.events);
        }
        if event != C220Nd2NzEvent::Readiness {
            self.trace.push(C220MtePipelineEvent::Nd2Nz(event));
        }
        Ok(())
    }

    pub(super) fn advance_nd2nz_return(
        &mut self,
        phase: C220BiuReturnCallback,
    ) -> Result<bool, C220MtePipelineError> {
        let dedicated = phase == C220BiuReturnCallback::Nd2NzPush
            || (phase == C220BiuReturnCallback::Egress(C220BiuSubcore::Cube)
                && self
                    .biu_returns
                    .as_ref()
                    .and_then(|returns| returns.pending_egress(C220BiuSubcore::Cube).front())
                    .is_some_and(|head| {
                        matches!(
                            head.request.input.destination,
                            C220BiuWriteDestination::Nd2Nz { .. }
                        )
                    }));
        if !dedicated {
            return Ok(false);
        }
        let engine = self
            .nd2nz
            .as_mut()
            .ok_or(C220MtePipelineError::Nd2NzUnconfigured)?;
        let returns = self.biu_returns.as_mut().expect("connected returns");
        let tick = self.events.tick();
        let progress = if phase == C220BiuReturnCallback::Nd2NzPush {
            returns.push_nd2nz(tick, engine)?
        } else {
            returns.egress_nd2nz(tick, engine)?
        };
        if let Some(output) = progress.released {
            let released = self
                .biu_read
                .as_mut()
                .expect("connected BIU")
                .release_tag(tick, output.request.tag)?;
            assert_eq!(released, output.request);
        }
        self.trace
            .push(C220MtePipelineEvent::BiuReturn(C220BiuReturnEvent::Nd2Nz(
                progress,
            )));
        Ok(true)
    }
}
