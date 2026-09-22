use super::{
    C220VectorAdvanceError, C220VectorPipeline, C220VectorWriteCompletion, PendingVectorUop,
};

impl C220VectorPipeline {
    pub(super) fn projected_write_completion(
        &self,
        entry: &PendingVectorUop,
        release: u64,
        preceding_completion: Option<u64>,
    ) -> Option<u64> {
        let write = entry.write.as_ref()?;
        entry.write_completion_tick.or_else(|| {
            let start = self
                .observed_tick
                .map_or(release, |tick| release.max(tick.saturating_add(1)))
                .max(preceding_completion.map_or(0, |tick| tick.saturating_add(1)));
            write.projected_write_completion(start)
        })
    }

    pub(super) fn prepare_writeback(&mut self) -> Result<(), C220VectorAdvanceError> {
        for entry in &mut self.pending {
            if entry.eligible_tick.is_none()
                && let Some(ready_tick) = entry.execute_ready_tick
            {
                entry.eligible_tick = Some(
                    ready_tick
                        .checked_add(entry.uop.writeback_ticks as u64)
                        .ok_or(C220VectorAdvanceError::TimeOverflow)?,
                );
            }
        }
        Ok(())
    }

    pub(super) fn finish_writes(&mut self) -> Result<(), C220VectorAdvanceError> {
        for entry in &mut self.pending {
            if entry.write_completion_tick.is_some() {
                continue;
            }
            let (Some(submitted_tick), Some(write)) = (entry.release_tick, &entry.write) else {
                continue;
            };
            if !write.is_complete() {
                continue;
            }
            let completion_tick = write.completion_tick().unwrap_or(submitted_tick);
            entry.write_completion_tick = Some(completion_tick);
            if !entry.deferred_compute && !entry.stores.is_empty() {
                entry.visible_tick = Some(
                    completion_tick
                        .checked_add(self.rules.ub_response_ticks)
                        .ok_or(C220VectorAdvanceError::TimeOverflow)?,
                );
            }
            self.last_write_completions.push(C220VectorWriteCompletion {
                pc: entry.uop.pc,
                repeat_index: entry.uop.repeat_index,
                lane_group: entry.uop.lane_group,
                kind: entry.uop.kind,
                submitted_tick,
                completion_tick,
                block_grants: write.grants().to_vec(),
            });
        }
        Ok(())
    }
}
