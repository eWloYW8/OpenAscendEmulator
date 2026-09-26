use super::{C220BiuReadOutput, C220BiuReadReturns, C220BiuReturnError, C220BiuWriteDestination};
use crate::sim::c220::mte::nd2nz::C220Nd2NzEngine;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct C220BiuNd2NzProgress {
    pub accepted_bytes: u32,
    pub queued_row: Option<u8>,
    /// The owner releases this request's frontend tag only after this handoff.
    pub released: Option<C220BiuReadOutput>,
}

impl C220BiuReadReturns {
    pub fn nd2nz_rows(&self) -> &[Option<C220BiuReadOutput>; 8] {
        &self.nd2nz_rows
    }

    pub fn egress_nd2nz(
        &mut self,
        tick: u64,
        engine: &mut C220Nd2NzEngine,
    ) -> Result<C220BiuNd2NzProgress, C220BiuReturnError> {
        self.check_callback(tick, self.egress_ticks[0], "egress")?;
        let mut progress = C220BiuNd2NzProgress::default();
        if let Some(mut output) = self.egress[0].front().copied()
            && output.ready_tick <= tick
        {
            match output.request.input.destination {
                C220BiuWriteDestination::Nd2Nz {
                    row_slot: Some(row),
                } => {
                    let slot = &mut self.nd2nz_rows[row as usize];
                    if slot.is_none() {
                        output.ready_tick = tick
                            .checked_add(1)
                            .ok_or(C220BiuReturnError::TimeOverflow)?;
                        *slot = Some(output);
                        self.egress[0].pop_front();
                        progress.queued_row = Some(row);
                    }
                }
                C220BiuWriteDestination::Nd2Nz { row_slot: None } => {
                    let generated = output.request.input.generated;
                    progress.accepted_bytes =
                        engine.receive(tick, generated.instruction_id, generated.uop_index)?;
                    if engine
                        .response_remaining(generated.instruction_id, generated.uop_index)
                        .is_none()
                    {
                        self.egress[0].pop_front();
                        self.records.remove(&output.request.tag);
                        progress.released = Some(output);
                    }
                }
                _ => return Err(C220BiuReturnError::DedicatedEgressRequired),
            }
        }
        self.egress_ticks[0] = Some(tick);
        self.observed_tick = Some(tick);
        Ok(progress)
    }

    /// Visit row slots in ascending order, stopping after the first accepted
    /// fragment. Occupied slots keep their tag until every byte is consumed.
    pub fn push_nd2nz(
        &mut self,
        tick: u64,
        engine: &mut C220Nd2NzEngine,
    ) -> Result<C220BiuNd2NzProgress, C220BiuReturnError> {
        self.check_callback(tick, self.nd2nz_tick, "ND2NZ row push")?;
        let mut progress = C220BiuNd2NzProgress::default();
        for row in 0..self.nd2nz_rows.len() {
            let Some(output) = self.nd2nz_rows[row] else {
                continue;
            };
            let generated = output.request.input.generated;
            if output.ready_tick > tick
                || engine.active_instruction() != Some(generated.instruction_id)
            {
                continue;
            }
            progress.accepted_bytes =
                engine.receive(tick, generated.instruction_id, generated.uop_index)?;
            if engine
                .response_remaining(generated.instruction_id, generated.uop_index)
                .is_none()
            {
                self.nd2nz_rows[row] = None;
                self.records.remove(&output.request.tag);
                progress.released = Some(output);
            }
            if progress.accepted_bytes != 0 {
                break;
            }
        }
        self.nd2nz_tick = Some(tick);
        self.observed_tick = Some(tick);
        Ok(progress)
    }
}
