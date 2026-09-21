use std::num::NonZeroU64;

use thiserror::Error;

use crate::execution::c220::dma_uop::{C220DmaUopError, C220DmaUopRequest, mte3_requests};
use crate::execution::c220::transfer::C220Mte3TransferPlan;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Mte3TimingRules {
    pub issue_interval: NonZeroU64,
    pub startup_ticks: u64,
    pub bytes_per_tick: NonZeroU64,
    pub retire_ticks: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Mte3Ticket {
    pub issue_tick: u64,
    pub data_ready_tick: u64,
    pub retire_tick: u64,
    pub transfer: C220Mte3TransferPlan,
    pub uop_count: usize,
    pub modeled_service_ticks: u64,
}

#[derive(Debug, Error)]
pub enum C220Mte3TimingError {
    #[error("MTE3 timing computation overflowed")]
    TimeOverflow,
    #[error("MTE3 completion state does not match its transfer ticket")]
    TicketMismatch,
    #[error(transparent)]
    Uop(#[from] C220DmaUopError),
}

pub struct C220TimedMte3Lane {
    rules: C220Mte3TimingRules,
    next_issue_tick: u64,
    data_port_tick: u64,
    unsignaled: Option<C220Mte3Ticket>,
    completion_flags: [Option<C220Mte3Ticket>; 4],
}

impl C220TimedMte3Lane {
    pub fn new(rules: C220Mte3TimingRules) -> Self {
        Self {
            rules,
            next_issue_tick: 0,
            data_port_tick: 0,
            unsignaled: None,
            completion_flags: [None; 4],
        }
    }

    pub const fn next_issue_tick(&self) -> u64 {
        self.next_issue_tick
    }

    pub fn completion_ready_tick(&self, flag_id: u8) -> Option<u64> {
        self.completion_flags
            .get(usize::from(flag_id))
            .and_then(|ticket| ticket.map(|ticket| ticket.retire_tick))
    }

    pub fn preview_issue(
        &self,
        tick: u64,
        transfer: C220Mte3TransferPlan,
    ) -> Result<(C220Mte3Ticket, Vec<C220DmaUopRequest>), C220Mte3TimingError> {
        let requests = mte3_requests(transfer)?;
        let rate = self.rules.bytes_per_tick.get();
        let modeled_service_ticks = requests.iter().try_fold(0_u64, |total, request| {
            let bytes = u64::from(request.bytes);
            total
                .checked_add(bytes / rate + u64::from(bytes % rate != 0))
                .ok_or(C220Mte3TimingError::TimeOverflow)
        })?;
        let start = tick
            .checked_add(self.rules.startup_ticks)
            .ok_or(C220Mte3TimingError::TimeOverflow)?
            .max(self.data_port_tick);
        let data_ready_tick = start
            .checked_add(modeled_service_ticks)
            .ok_or(C220Mte3TimingError::TimeOverflow)?;
        let retire_tick = data_ready_tick
            .checked_add(self.rules.retire_ticks)
            .ok_or(C220Mte3TimingError::TimeOverflow)?;
        tick.checked_add(self.rules.issue_interval.get())
            .ok_or(C220Mte3TimingError::TimeOverflow)?;
        Ok((
            C220Mte3Ticket {
                issue_tick: tick,
                data_ready_tick,
                retire_tick,
                transfer,
                uop_count: requests.len(),
                modeled_service_ticks,
            },
            requests,
        ))
    }

    pub fn issue(&mut self, ticket: C220Mte3Ticket) -> Result<(), C220Mte3TimingError> {
        if self.unsignaled.is_some() {
            return Err(C220Mte3TimingError::TicketMismatch);
        }
        self.data_port_tick = ticket.data_ready_tick;
        self.next_issue_tick = ticket
            .issue_tick
            .checked_add(self.rules.issue_interval.get())
            .ok_or(C220Mte3TimingError::TimeOverflow)?;
        self.unsignaled = Some(ticket);
        Ok(())
    }

    pub fn set_completion_flag(&mut self, flag_id: u8) -> Result<(), C220Mte3TimingError> {
        let slot = self
            .completion_flags
            .get_mut(usize::from(flag_id))
            .ok_or(C220Mte3TimingError::TicketMismatch)?;
        if slot.is_some() {
            return Err(C220Mte3TimingError::TicketMismatch);
        }
        *slot = Some(
            self.unsignaled
                .take()
                .ok_or(C220Mte3TimingError::TicketMismatch)?,
        );
        Ok(())
    }

    pub fn wait_completion_flag(&mut self, flag_id: u8) -> Result<(), C220Mte3TimingError> {
        self.completion_flags
            .get_mut(usize::from(flag_id))
            .and_then(Option::take)
            .ok_or(C220Mte3TimingError::TicketMismatch)?;
        Ok(())
    }
}
