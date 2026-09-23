use std::num::NonZeroU64;

use thiserror::Error;

use crate::sim::c220::mte::mte3::C220Mte3TransferPlan;
use crate::sim::c220::mte::uop::{C220DmaUopError, C220DmaUops, mte3_uops};

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

impl C220Mte3Ticket {
    pub fn requests(self) -> Result<C220DmaUops, C220DmaUopError> {
        mte3_uops(self.transfer)
    }
}

#[derive(Debug, Error)]
pub enum C220Mte3TimingError {
    #[error("MTE3 timing computation overflowed")]
    TimeOverflow,
    #[error("MTE3 outstanding command queue is full")]
    QueueFull,
    #[error(transparent)]
    Uop(#[from] C220DmaUopError),
}

pub struct C220TimedMte3Lane {
    rules: C220Mte3TimingRules,
    next_issue_tick: u64,
    data_port_tick: u64,
    latest_retirement_tick: Option<u64>,
}

impl C220TimedMte3Lane {
    pub fn new(rules: C220Mte3TimingRules) -> Self {
        Self {
            rules,
            next_issue_tick: 0,
            data_port_tick: 0,
            latest_retirement_tick: None,
        }
    }

    pub const fn next_issue_tick(&self) -> u64 {
        self.next_issue_tick
    }

    pub const fn latest_retirement_tick(&self) -> Option<u64> {
        self.latest_retirement_tick
    }

    pub fn preview_issue(
        &self,
        tick: u64,
        transfer: C220Mte3TransferPlan,
    ) -> Result<C220Mte3Ticket, C220Mte3TimingError> {
        let mut requests = mte3_uops(transfer)?;
        if transfer.descriptor.is_disabled() {
            let data_ready_tick = tick
                .checked_add(1)
                .ok_or(C220Mte3TimingError::TimeOverflow)?;
            let retire_tick = self.ordered_retirement_tick(data_ready_tick)?;
            return Ok(C220Mte3Ticket {
                issue_tick: tick,
                data_ready_tick,
                retire_tick,
                transfer,
                uop_count: 0,
                modeled_service_ticks: 0,
            });
        }
        let rate = self.rules.bytes_per_tick.get();
        let (uop_count, modeled_service_ticks) =
            requests.try_fold((0, 0_u64), |(count, total), request| {
                let bytes = u64::from(request.bytes);
                let ticks = total
                    .checked_add(bytes / rate + u64::from(bytes % rate != 0))
                    .ok_or(C220Mte3TimingError::TimeOverflow)?;
                Ok::<_, C220Mte3TimingError>((count + 1, ticks))
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
        let retire_tick = self.ordered_retirement_tick(retire_tick)?;
        tick.checked_add(self.rules.issue_interval.get())
            .ok_or(C220Mte3TimingError::TimeOverflow)?;
        Ok(C220Mte3Ticket {
            issue_tick: tick,
            data_ready_tick,
            retire_tick,
            transfer,
            uop_count,
            modeled_service_ticks,
        })
    }

    fn ordered_retirement_tick(&self, ready: u64) -> Result<u64, C220Mte3TimingError> {
        match self.latest_retirement_tick {
            Some(previous) => Ok(ready.max(
                previous
                    .checked_add(1)
                    .ok_or(C220Mte3TimingError::TimeOverflow)?,
            )),
            None => Ok(ready),
        }
    }

    pub fn issue(&mut self, ticket: C220Mte3Ticket) -> Result<(), C220Mte3TimingError> {
        if !ticket.transfer.descriptor.is_disabled() {
            let next_issue_tick = ticket
                .issue_tick
                .checked_add(self.rules.issue_interval.get())
                .ok_or(C220Mte3TimingError::TimeOverflow)?;
            self.data_port_tick = ticket.data_ready_tick;
            self.next_issue_tick = next_issue_tick;
        }
        self.latest_retirement_tick = Some(
            self.latest_retirement_tick
                .unwrap_or(0)
                .max(ticket.retire_tick),
        );
        Ok(())
    }
}
