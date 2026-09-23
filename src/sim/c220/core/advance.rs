use super::{C220Core, C220CoreError};

impl C220Core {
    pub(super) fn advance_engines_to(&mut self, tick: u64) -> Result<(), C220CoreError> {
        self.cube.begin_advance();
        self.mte1.begin_advance();
        self.mte2.begin_advance();
        self.mte3.begin_advance();
        self.vector.begin_advance();
        loop {
            let event_tick = self
                .cube
                .pipeline
                .next_event_tick()
                .into_iter()
                .chain(self.mte1.next_event_tick())
                .chain(self.mte2.next_event_tick())
                .chain(self.vector.next_event_tick())
                .chain(self.mte3.pending_retirement_tick())
                .chain(self.mte_pipeline.as_ref().and_then(|p| p.next_event_tick()))
                .min()
                .map_or(tick, |next| next.min(tick));
            self.mte1.commit_ready_at(
                event_tick,
                &mut self.local_memory,
                &mut self.hardware_flags,
            )?;
            self.mte2.commit_ready_at(
                event_tick,
                &mut self.local_memory,
                &mut self.state.ub,
                &self.memory,
            )?;
            if let Some(pipeline) = &mut self.mte_pipeline {
                pipeline.advance(event_tick)?;
                self.mte1
                    .observe_completions(event_tick, pipeline.mte1_completions());
                self.mte2
                    .observe_l1_completions(event_tick, pipeline.l1_fill_completions());
                for id in pipeline.take_dma_tails() {
                    self.mte2.observe_dma_tail(id);
                }
                for id in pipeline.take_dma_completions() {
                    self.mte2.complete_dma(id)?;
                }
            }
            self.cube.advance_event(
                event_tick,
                &mut self.local_memory,
                &mut self.hardware_flags,
                self.state.scalar_mut().machine_mut(),
            )?;
            self.vector.advance_event(event_tick, &mut self.state)?;
            if let Some(pipeline) = &mut self.mte_pipeline {
                pipeline.advance_ub_service(self.vector.ub_cycles_at(event_tick))?;
            }
            self.mte3
                .commit_ready_at(event_tick, self.state.ub(), &mut self.memory)?;
            if self.mte3.physical
                && let Some(pipeline) = &mut self.mte_pipeline
            {
                self.mte3
                    .commit_dma_at(event_tick, pipeline, self.state.ub(), &mut self.memory)?;
            }
            if event_tick == tick {
                break;
            }
        }
        Ok(())
    }
}
