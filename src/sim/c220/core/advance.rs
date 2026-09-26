use super::{C220Core, C220CoreError};

impl C220Core {
    pub(super) fn advance_engines_to(&mut self, tick: u64) -> Result<(), C220CoreError> {
        self.pipeline_events.begin_advance();
        self.cube.begin_advance();
        self.cube_frontend.outcomes.clear();
        self.mte1.begin_advance();
        self.mte1_frontend.outcomes.clear();
        self.mte2.begin_advance();
        self.mte2_frontend.outcomes.clear();
        self.mte3.begin_advance();
        self.mte3_issue_queue.outcomes.clear();
        self.vector.begin_advance();
        self.fixp_frontend.outcomes.clear();
        self.fixp_frontend.cross_core_outcomes.clear();
        if let Some(fixp) = &mut self.fixp {
            fixp.factor_outcomes.clear();
        }
        if let Some(fixp) = &mut self.external_fixp {
            fixp.factor_outcomes.clear();
        }
        loop {
            let event_tick = self
                .cube
                .pipeline
                .next_event_tick()
                .into_iter()
                .chain(self.cube_frontend.next_tick)
                .chain(self.hardware_flags.next_notification_tick())
                .chain(self.mte1.next_event_tick())
                .chain(self.lsu.as_ref().and_then(|lsu| lsu.next_tick))
                .chain(self.mte2.next_event_tick())
                .chain(self.vector.next_event_tick())
                .chain(self.termination.next_tick())
                .chain(self.mte3.pending_retirement_tick())
                .chain(self.mte_pipeline.as_ref().and_then(|p| p.next_event_tick()))
                .chain(self.fixp.as_ref().and_then(|fixp| {
                    self.mte_pipeline
                        .as_ref()
                        .and_then(|pipeline| pipeline.next_fixp_event_tick(&fixp.engine))
                }))
                .chain(self.external_fixp.as_ref().and_then(|fixp| {
                    self.mte_pipeline
                        .as_ref()
                        .and_then(|pipeline| pipeline.next_external_fixp_event_tick(&fixp.engine))
                }))
                .min()
                .map_or(tick, |next| next.min(tick));
            self.scalar_timing.advance_to(event_tick);
            self.hardware_flags.advance_to(event_tick)?;
            let previous_mte1_outcomes = self.mte1.outcomes.len();
            self.mte1.commit_ready_at(
                event_tick,
                &mut self.local_memory,
                &mut self.hardware_flags,
            )?;
            for outcome in &self.mte1.outcomes[previous_mte1_outcomes..] {
                self.pipeline_events
                    .retire(3, outcome.instruction_id, outcome.retire_tick);
                if let crate::sim::c220::mte::mte1::C220Mte1TransferResult::Load3dv2(report) =
                    outcome.result
                    && let Some(value) = report.spr54
                {
                    self.state
                        .scalar_mut()
                        .machine_mut()
                        .set_spr_value(54, value)
                        .map_err(C220CoreError::MteSpr)?;
                }
            }
            self.release_mte1_barriers_at(event_tick);
            self.retire_factor_at(event_tick)?;
            if let Some(engine) = self.fixp_engine_mut() {
                engine.retire_ready_control(event_tick);
                if let Some(reception) = engine.retire_ready_cross_core(event_tick) {
                    self.fixp_frontend.cross_core_outcomes.push(reception);
                }
            }
            let previous_mte2_outcomes = self.mte2.last_outcomes().len();
            self.mte2.commit_ready_at(
                event_tick,
                &mut self.local_memory,
                &mut self.state.ub,
                &self.memory,
            )?;
            for outcome in &self.mte2.last_outcomes()[previous_mte2_outcomes..] {
                self.pipeline_events
                    .retire(4, outcome.command.instruction_id, outcome.retire_tick);
            }
            self.release_mte2_barriers_at(event_tick);
            self.publish_fixp_retirement();
            while self.mte_pipeline.is_some() {
                let pipeline = self.mte_pipeline.as_mut().expect("configured MTE pipeline");
                if let Some(fixp) = &mut self.fixp {
                    self.hardware_flags.advance_to(event_tick)?;
                    let (l0c, l1) = self.local_memory.fixp_destinations_mut();
                    pipeline.advance_fixp(
                        event_tick,
                        &mut fixp.engine,
                        crate::sim::c220::mte::fixp::C220FixpMemory {
                            l0c,
                            l1,
                            slopes: &fixp.factors,
                        },
                        fixp.bindings.resolver(&mut self.hardware_flags),
                    )?;
                } else if let Some(fixp) = &mut self.external_fixp {
                    self.hardware_flags.advance_to(event_tick)?;
                    let (l0c, l1) = self.local_memory.fixp_destinations_mut();
                    pipeline.advance_external_fixp(
                        event_tick,
                        &mut fixp.engine,
                        crate::sim::c220::mte::fixp::C220FixpRuntimeMemory {
                            l0c,
                            l1,
                            slopes: &fixp.factors,
                            external: &mut self.memory,
                            atomics: fixp.atomics,
                        },
                        fixp.bindings.resolver(&mut self.hardware_flags),
                    )?;
                } else {
                    pipeline.advance(event_tick)?;
                }
                self.publish_fixp_retirement();
                let pipeline = self.mte_pipeline.as_mut().expect("configured MTE pipeline");
                if pipeline.fixp_issue_pending() {
                    self.transfer_fixp_issue_at(event_tick)?;
                    continue;
                }
                if pipeline.mte1_issue_pending() {
                    self.transfer_mte1_issue_at(event_tick)?;
                    continue;
                }
                if pipeline.mte2_issue_pending() {
                    self.transfer_mte2_issue_at(event_tick)?;
                    continue;
                }
                if pipeline.mte3_issue_pending() {
                    self.transfer_mte3_issue_at(event_tick)?;
                    continue;
                }
                if pipeline.mte2_dispatch_pending() {
                    self.dispatch_mte2_head_at(event_tick)?;
                    continue;
                }
                if pipeline.mte1_sync_pending() {
                    pipeline.resolve_mte1_sync(&mut self.hardware_flags)?;
                    continue;
                }
                if pipeline.fixp_dispatch_pending() {
                    self.dispatch_fixp_head_at(event_tick)?;
                    continue;
                }
                if pipeline.mte1_dispatch_pending() {
                    self.dispatch_mte1_head_at(event_tick)?;
                    continue;
                }
                self.mte1
                    .observe_completions(event_tick, pipeline.mte1_completions());
                self.mte2
                    .observe_l1_completions(event_tick, pipeline.l1_fill_completions());
                self.mte2
                    .observe_l1_completions(event_tick, pipeline.mte2_read_completions());
                for id in pipeline.take_dma_tails() {
                    self.mte2.observe_dma_tail(id);
                }
                for id in pipeline.take_dma_completions() {
                    self.mte2.complete_dma(id)?;
                }
                break;
            }
            self.release_fixp_barriers_at(event_tick);
            self.advance_lsu_at(event_tick)?;
            self.cube.advance_event(
                event_tick,
                &mut self.local_memory,
                &mut self.hardware_flags,
                self.state.scalar_mut().machine_mut(),
                &mut self.pipeline_events,
            )?;
            self.release_cube_barriers_at(event_tick);
            self.dispatch_cube_head_at(event_tick)?;
            self.vector
                .advance_event(event_tick, &mut self.state, &mut self.pipeline_events)?;
            if let Some(pipeline) = &mut self.mte_pipeline {
                pipeline.advance_ub_service(self.vector.ub_activity_at(event_tick))?;
            }
            let previous_mte3_outcomes = self.mte3.outcomes.len();
            self.mte3
                .commit_ready_at(event_tick, self.state.ub(), &mut self.memory)?;
            for outcome in &self.mte3.outcomes[previous_mte3_outcomes..] {
                self.pipeline_events
                    .retire(5, outcome.instruction_id, outcome.tick);
            }
            if self.mte3.physical
                && let Some(pipeline) = &mut self.mte_pipeline
                && let Some(id) = self.mte3.commit_native_at(
                    event_tick,
                    pipeline,
                    self.state.ub(),
                    &mut self.memory,
                )?
            {
                self.pipeline_events.retire(5, id, event_tick);
            }
            self.release_mte3_barriers_at(event_tick);
            self.advance_termination(event_tick)?;
            if event_tick == tick {
                break;
            }
        }
        Ok(())
    }
}
