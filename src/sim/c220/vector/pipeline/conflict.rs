use super::{C220VectorAdvanceError, C220VectorPipeline, PendingVectorUop};

const GATHER_INDEX_READY_TICKS: u64 = 10;

impl PendingVectorUop {
    fn nominal_latency(&self) -> u64 {
        u64::from(self.uop.stages.read_ticks)
            + u64::from(self.uop.stages.execute_ticks)
            + self.uop.writeback_ticks as u64
    }

    fn same_repeat_opcode(&self, previous: &Self) -> bool {
        self.instruction_group == previous.instruction_group
            || self
                .repeat_opcode
                .is_some_and(|opcode| Some(opcode) == previous.repeat_opcode)
    }
}

impl C220VectorPipeline {
    pub(super) fn conflict_ready_tick(&self, index: usize, earliest: u64) -> u64 {
        let entry = &self.pending[index];
        let earliest = entry
            .read
            .as_ref()
            .and_then(|read| read.index_prefetch().1)
            .and_then(|repeat| {
                self.index_prefetch_ready
                    .get(&(entry.instruction_group, repeat))
            })
            .map_or(earliest, |ready| {
                earliest.max(ready.saturating_add(GATHER_INDEX_READY_TICKS))
            });
        let Some(previous) = self.pending.iter().take(index).rev().find(|previous| {
            previous.conflict_check_tick.is_some() && previous.release_tick.is_none()
        }) else {
            return earliest;
        };
        let previous_tick = previous
            .conflict_check_tick
            .expect("executing uop has a start");
        let mut ready = earliest.max(
            previous_tick
                .saturating_add(entry.issue_variant.minimum_gap_from(previous.issue_variant)),
        );
        if entry.same_repeat_opcode(previous) {
            ready = ready.max(previous_tick.saturating_add(previous.issue_gap));
        }
        let previous_end = previous_tick.saturating_add(previous.nominal_latency());
        let candidate_latency =
            u64::from(entry.uop.stages.read_ticks) + u64::from(entry.uop.stages.execute_ticks) + 1;
        ready = ready.max(
            previous_end
                .saturating_add(1)
                .saturating_sub(candidate_latency),
        );
        if entry.write.is_some() && previous.write.is_some() {
            ready = ready.max(previous_tick.saturating_add(previous.uop.writeback_ticks as u64));
        }
        ready
    }

    pub(super) fn check_conflicts(&mut self, tick: u64) -> Result<(), C220VectorAdvanceError> {
        if self.last_conflict_check_tick == Some(tick) {
            return Ok(());
        }
        let Some(index) = self
            .pending
            .iter()
            .position(|entry| entry.conflict_check_tick.is_none())
        else {
            return Ok(());
        };
        let entry = &self.pending[index];
        if !entry.admitted {
            return Ok(());
        }
        if let Some(read) = &entry.read
            && read
                .grant_tick(entry.admission_tick)
                .is_none_or(|grant| grant > tick)
        {
            return Ok(());
        }
        if let Some(read) = &entry.read {
            let (produces, requires) = read.index_prefetch();
            if let Some(repeat) = produces {
                self.index_prefetch_ready
                    .entry((entry.instruction_group, repeat))
                    .or_insert(tick);
            }
            if let Some(repeat) = requires
                && self
                    .index_prefetch_ready
                    .get(&(entry.instruction_group, repeat))
                    .is_none_or(|ready| tick < ready.saturating_add(GATHER_INDEX_READY_TICKS))
            {
                return Ok(());
            }
        }
        let shared_ready = if entry.shared_read_from_previous {
            let Some(ready) = index.checked_sub(1).and_then(|previous| {
                let previous = &self.pending[previous];
                previous
                    .execute_ready_tick
                    .map(|ready| ready.saturating_sub(u64::from(previous.uop.stages.execute_ticks)))
            }) else {
                return Ok(());
            };
            Some(ready)
        } else {
            None
        };
        if self.conflict_ready_tick(index, tick) > tick {
            return Ok(());
        }
        let entry = &mut self.pending[index];
        let ready_tick = if let Some(ready) = shared_ready {
            ready.max(tick)
        } else {
            tick.checked_add(u64::from(entry.uop.stages.read_ticks))
                .ok_or(C220VectorAdvanceError::TimeOverflow)?
        };
        entry.conflict_check_tick = Some(tick);
        self.last_conflict_check_tick = Some(tick);
        if let Some(read) = &mut entry.read {
            read.set_ready_tick(ready_tick);
        } else {
            let execute_ready_tick = ready_tick
                .checked_add(u64::from(entry.uop.stages.execute_ticks))
                .ok_or(C220VectorAdvanceError::TimeOverflow)?;
            entry.execute_ready_tick = Some(execute_ready_tick);
        }
        Ok(())
    }
}
