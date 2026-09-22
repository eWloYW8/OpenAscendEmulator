use super::C220Mte2TransferPlan;
use crate::sim::c220::mte::uop::{C220DmaUopError, C220DmaUops, mte2_uops};
use std::num::NonZeroU64;

/// Caller-supplied aggregate DMA timing, not a physical HBM/UB model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Mte2TimingRules {
    pub issue_interval: NonZeroU64,
    pub startup_ticks: u64,
    pub bytes_per_tick: NonZeroU64,
    pub retire_ticks: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Mte2Ticket {
    pub issue_tick: u64,
    pub data_ready_tick: u64,
    pub retire_tick: u64,
    pub transfer: C220Mte2TransferPlan,
    pub uop_count: usize,
    pub modeled_service_ticks: u64,
}

impl C220Mte2Ticket {
    pub fn requests(self) -> Result<C220DmaUops, C220DmaUopError> {
        mte2_uops(self.transfer)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum C220Mte2TimingError {
    #[error("MTE2 timing computation overflowed")]
    TimeOverflow,
    #[error(transparent)]
    Uop(#[from] C220DmaUopError),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct C220Mte2DmaTiming {
    pub(super) rules: C220Mte2TimingRules,
    pub(super) next_issue_tick: u64,
    mte2_data_port_tick: u64,
}
impl C220Mte2DmaTiming {
    pub(super) fn new(rules: C220Mte2TimingRules) -> Self {
        Self {
            rules,
            next_issue_tick: 0,
            mte2_data_port_tick: 0,
        }
    }
    pub(super) fn accept(&mut self, ticket: C220Mte2Ticket) {
        self.mte2_data_port_tick = ticket.data_ready_tick;
        self.next_issue_tick = ticket.issue_tick + self.rules.issue_interval.get();
    }
    pub(super) fn preview_ticket(
        &self,
        issue_tick: u64,
        transfer: C220Mte2TransferPlan,
    ) -> Result<C220Mte2Ticket, C220Mte2TimingError> {
        let mut requests = mte2_uops(transfer)?;
        let rate = self.rules.bytes_per_tick.get();
        let (uop_count, modeled_service_ticks) =
            requests.try_fold((0, 0_u64), |(count, total), request| {
                let bytes = u64::from(request.bytes);
                let ticks = bytes / rate + u64::from(bytes % rate != 0);
                let ticks = total
                    .checked_add(ticks)
                    .ok_or(C220Mte2TimingError::TimeOverflow)?;
                Ok::<_, C220Mte2TimingError>((count + 1, ticks))
            })?;
        let start = issue_tick
            .checked_add(self.rules.startup_ticks)
            .ok_or(C220Mte2TimingError::TimeOverflow)?
            .max(self.mte2_data_port_tick);
        let data_ready_tick = start
            .checked_add(modeled_service_ticks)
            .ok_or(C220Mte2TimingError::TimeOverflow)?;
        let retire_tick = data_ready_tick
            .checked_add(self.rules.retire_ticks)
            .ok_or(C220Mte2TimingError::TimeOverflow)?;
        issue_tick
            .checked_add(self.rules.issue_interval.get())
            .ok_or(C220Mte2TimingError::TimeOverflow)?;
        Ok(C220Mte2Ticket {
            issue_tick,
            data_ready_tick,
            retire_tick,
            transfer,
            uop_count,
            modeled_service_ticks,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::c220::mte::{C220MovOutToUbDescriptor, CAPTURED_C220_MOV_OUT_TO_UB_X_WORD};

    fn transfer(burst_length: u16) -> C220Mte2TransferPlan {
        let descriptor = C220MovOutToUbDescriptor::decode(
            CAPTURED_C220_MOV_OUT_TO_UB_X_WORD,
            (u64::from(burst_length) << 16) | (1 << 4),
        )
        .unwrap();
        C220Mte2TransferPlan {
            descriptor,
            source_address: 0x1000,
            destination_address: 0x200,
            bytes: usize::from(burst_length) * 32,
            dma_mode_word: 0,
        }
    }

    #[test]
    fn c220_timeline_separates_issue_service_and_retirement() {
        let rules = C220Mte2TimingRules {
            issue_interval: NonZeroU64::new(2).unwrap(),
            startup_ticks: 3,
            bytes_per_tick: NonZeroU64::new(32).unwrap(),
            retire_ticks: 1,
        };
        let mut timed = C220Mte2DmaTiming::new(rules);
        let first = timed.preview_ticket(5, transfer(2)).unwrap();
        assert_eq!((first.data_ready_tick, first.retire_tick), (10, 11));
        assert_eq!(first.transfer.descriptor_segments().unwrap().len(), 2);
        assert_eq!(first.uop_count, 1);
        assert_eq!(first.modeled_service_ticks, 2);
        assert_eq!(first.requests().unwrap().next().unwrap().bytes, 64);
        timed.mte2_data_port_tick = first.data_ready_tick;
        let second = timed.preview_ticket(7, transfer(4)).unwrap();
        assert_eq!((second.data_ready_tick, second.retire_tick), (14, 15));
        assert!(matches!(
            timed.preview_ticket(u64::MAX, transfer(1)),
            Err(C220Mte2TimingError::TimeOverflow)
        ));
    }

    #[test]
    fn split_requests_each_consume_a_service_quantum() {
        let rules = C220Mte2TimingRules {
            issue_interval: NonZeroU64::new(1).unwrap(),
            startup_ticks: 0,
            bytes_per_tick: NonZeroU64::new(64).unwrap(),
            retire_ticks: 0,
        };
        let timed = C220Mte2DmaTiming::new(rules);
        let mut plan = transfer(1);
        plan.descriptor = C220MovOutToUbDescriptor::decode(
            CAPTURED_C220_MOV_OUT_TO_UB_X_WORD,
            (1_u64 << 32) | (1 << 16) | (2 << 4),
        )
        .unwrap();
        plan.bytes = 64;
        let ticket = timed.preview_ticket(5, plan).unwrap();
        assert_eq!(ticket.uop_count, 2);
        assert_eq!(ticket.modeled_service_ticks, 2);
        assert_eq!(ticket.data_ready_tick, 7);
    }
}
