use super::{C220Mte3Ticket, C220Mte3TimingError, C220Mte3TimingRules, C220TimedMte3Lane};
use crate::memory::mapped::{MappedMemory, MappedMemoryError};
use crate::sim::c220::mte::mte3::C220PreparedOutput;

pub(in crate::sim::c220) struct Mte3Engine {
    pub(in crate::sim::c220) timing: C220TimedMte3Lane,
    pending: Option<PendingOutput>,
}

struct PendingOutput {
    data_ready_tick: u64,
    prepared: C220PreparedOutput,
}

impl Mte3Engine {
    pub(in crate::sim::c220) fn has_pending(&self) -> bool {
        self.pending.is_some()
    }

    pub(in crate::sim::c220) fn issue(
        &mut self,
        ticket: C220Mte3Ticket,
        prepared: C220PreparedOutput,
    ) -> Result<(), C220Mte3TimingError> {
        if self.pending.is_some() {
            return Err(C220Mte3TimingError::TicketMismatch);
        }
        self.timing.issue(ticket)?;
        self.pending = Some(PendingOutput {
            data_ready_tick: ticket.data_ready_tick,
            prepared,
        });
        Ok(())
    }

    pub(in crate::sim::c220) fn new(rules: C220Mte3TimingRules) -> Self {
        Self {
            timing: C220TimedMte3Lane::new(rules),
            pending: None,
        }
    }

    pub(in crate::sim::c220) fn pending_ready_tick(&self) -> Option<u64> {
        self.pending.as_ref().map(|pending| pending.data_ready_tick)
    }

    pub(in crate::sim::c220) fn commit_ready_at(
        &mut self,
        tick: u64,
        memory: &mut MappedMemory,
    ) -> Result<(), MappedMemoryError> {
        if let Some(pending) = self.pending.as_ref()
            && tick >= pending.data_ready_tick
        {
            pending.prepared.commit(memory)?;
            self.pending = None;
        }
        Ok(())
    }
}
