use super::{C220Mte1Ticket, C220Mte1TimingError, C220TimedMte1Lane};
use crate::sim::c220::memory::C220LocalMemory;
use crate::sim::c220::mte::mte1::load2d::{C220Load2dTransferError, C220PreparedLoad2d};

#[derive(Default)]
pub(in crate::sim::c220) struct Mte1Engine {
    pub(in crate::sim::c220) timing: C220TimedMte1Lane,
    pending: Vec<PendingLoad2d>,
}

struct PendingLoad2d {
    data_ready_tick: u64,
    prepared: C220PreparedLoad2d,
}

impl Mte1Engine {
    pub(in crate::sim::c220) fn issue(
        &mut self,
        ticket: &C220Mte1Ticket,
        prepared: C220PreparedLoad2d,
    ) -> Result<(), C220Mte1TimingError> {
        self.timing.issue(ticket)?;
        self.pending.push(PendingLoad2d {
            data_ready_tick: ticket.data_ready_tick,
            prepared,
        });
        Ok(())
    }

    pub(in crate::sim::c220) fn next_data_ready_tick(&self) -> Option<u64> {
        self.pending
            .iter()
            .map(|pending| pending.data_ready_tick)
            .min()
    }

    pub(in crate::sim::c220) fn commit_ready_at(
        &mut self,
        tick: u64,
        memory: &mut C220LocalMemory,
    ) -> Result<(), C220Load2dTransferError> {
        let mut index = 0;
        while index < self.pending.len() {
            if self.pending[index].data_ready_tick <= tick {
                self.pending.remove(index).prepared.commit(memory)?;
            } else {
                index += 1;
            }
        }
        Ok(())
    }
}
