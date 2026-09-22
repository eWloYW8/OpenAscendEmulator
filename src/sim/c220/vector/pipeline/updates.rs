use super::{C220VectorAdvanceError, C220VectorPipeline};
use crate::sim::c220::state::C220State;
use crate::sim::c220::vector::read::PendingVectorRead;
impl C220VectorPipeline {
    pub(super) fn apply_reduction_updates(
        &mut self,
        tick: u64,
        core: &mut C220State,
    ) -> Result<(), C220VectorAdvanceError> {
        for index in 0..self.pending.len() {
            let Some(update) = self.pending[index].reduction_update else {
                continue;
            };
            if update.applied
                || update.update.is_none()
                || self.pending[index]
                    .execute_ready_tick
                    .is_none_or(|ready| ready > tick)
            {
                continue;
            }
            if self.pending.iter().take(index).any(|entry| {
                entry
                    .reduction_update
                    .is_some_and(|prior| prior.group == update.group && !prior.applied)
            }) {
                continue;
            }
            let mut state = self
                .reduction_states
                .remove(&update.group)
                .expect("issued reduction has state");
            state.apply(update.update.expect("checked reduction update"));
            if update.last {
                let (register, value) = state.result().expect("completed reduction has a result");
                core.scalar_mut()
                    .machine_mut()
                    .set_spr_value(register, value)?;
            } else {
                self.reduction_states.insert(update.group, state);
            }
            self.pending[index]
                .reduction_update
                .as_mut()
                .expect("reduction update remains pending")
                .applied = true;
        }
        Ok(())
    }

    pub(super) fn apply_va_updates(&mut self, tick: u64) {
        for index in 0..self.pending.len() {
            if self.pending[index].va_update_applied
                || self.pending[index]
                    .execute_ready_tick
                    .is_none_or(|ready| ready > tick)
            {
                continue;
            }
            if self.pending.iter().take(index).any(|entry| {
                entry
                    .read
                    .as_ref()
                    .is_some_and(PendingVectorRead::is_load_va)
                    && !entry.va_update_applied
            }) {
                continue;
            }
            if let Some(update) = self.pending[index].va_update {
                self.last_va_updates.push(update);
                self.pending[index].va_update_applied = true;
            }
        }
    }
}
