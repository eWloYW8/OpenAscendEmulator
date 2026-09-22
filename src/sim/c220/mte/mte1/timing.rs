use std::collections::{BTreeMap, VecDeque};
use std::num::NonZeroU64;

use thiserror::Error;

use crate::isa::c220::mte::load2d::{
    C220_LOAD_2D_BLOCK_BYTES, C220Load2dDestination, C220Load2dTransfer,
};

const C220_MTE1_COMMAND_TICKS: u64 = 1;
const C220_MTE1_L1_READ_TICKS: u64 = 8;
const C220_MTE1_DESTINATION_PROGRESS_TICKS: u64 = 1;
const C220_MTE1_RETIRE_TICKS: u64 = 1;
const C220_MTE1_DESTINATION_QUEUE_DEPTH: usize = 2;
const C220_MTE1_MAX_OUTSTANDING: usize = 31;
const C220_MTE1_L1_TO_L0A_BYTES_PER_TICK: u64 = 256;
const C220_MTE1_L1_TO_L0B_BYTES_PER_TICK: u64 = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Mte1TimingRules {
    pub issue_interval: NonZeroU64,
    pub l1_to_l0a_bytes_per_tick: NonZeroU64,
    pub l1_to_l0b_bytes_per_tick: NonZeroU64,
}

impl C220Mte1TimingRules {
    pub const fn dav2201() -> Self {
        Self {
            issue_interval: NonZeroU64::MIN,
            l1_to_l0a_bytes_per_tick: NonZeroU64::new(C220_MTE1_L1_TO_L0A_BYTES_PER_TICK).unwrap(),
            l1_to_l0b_bytes_per_tick: NonZeroU64::new(C220_MTE1_L1_TO_L0B_BYTES_PER_TICK).unwrap(),
        }
    }
}

impl Default for C220Mte1TimingRules {
    fn default() -> Self {
        Self::dav2201()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Mte1UopTicket {
    pub repeat_index: u8,
    pub command_tick: u64,
    pub l1_data_ready_tick: u64,
    pub transfer_start_tick: u64,
    pub data_ready_tick: u64,
    pub modeled_service_ticks: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220Mte1Ticket {
    pub issue_tick: u64,
    pub data_ready_tick: u64,
    pub retire_tick: u64,
    pub transfer: C220Load2dTransfer,
    pub uops: Vec<C220Mte1UopTicket>,
    pub modeled_service_ticks: u64,
    resource_after: C220Mte1RouteState,
}

impl C220Mte1Ticket {
    pub fn uop_count(&self) -> usize {
        self.uops.len()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum C220Mte1TimingError {
    #[error("C220 MTE1 timing cannot be reconfigured while work is pending")]
    Busy,
    #[error("C220 MTE1 timing computation overflowed")]
    TimeOverflow,
    #[error("C220 MTE1 ticket does not match the current timing state")]
    TicketMismatch,
    #[error("C220 MTE1 outstanding queue is full until tick {ready_tick}")]
    QueueFull { ready_tick: u64 },
    #[error("C220 MTE1 does not support destination {0:?}")]
    UnsupportedDestination(C220Load2dDestination),
    #[error("C220 MTE1 event {event_id} has no matching token")]
    MissingEvent { event_id: u32 },
    #[error("C220 MTE1 event {event_id} is not visible until tick {ready_tick}")]
    EventNotReady { event_id: u32, ready_tick: u64 },
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct C220Mte1RouteState {
    data_port_tick: u64,
    destination_releases: VecDeque<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct C220TimedMte1Lane {
    rules: C220Mte1TimingRules,
    next_issue_tick: u64,
    l0a: C220Mte1RouteState,
    l0b: C220Mte1RouteState,
    pending_retirements: VecDeque<C220Mte1Ticket>,
    last_retirements: Vec<C220Mte1Ticket>,
    unsignaled_retire_tick: Option<u64>,
    events: BTreeMap<u32, VecDeque<u64>>,
}

impl C220TimedMte1Lane {
    pub fn new(rules: C220Mte1TimingRules) -> Self {
        Self {
            rules,
            ..Self::default()
        }
    }

    pub const fn rules(&self) -> C220Mte1TimingRules {
        self.rules
    }

    pub const fn next_issue_tick(&self) -> u64 {
        self.next_issue_tick
    }

    pub fn next_accept_tick(&self) -> u64 {
        let queue_ready_tick = if self.pending_retirements.len() >= C220_MTE1_MAX_OUTSTANDING {
            self.pending_retirements
                .iter()
                .map(|ticket| ticket.retire_tick)
                .min()
                .unwrap_or(self.next_issue_tick)
        } else {
            0
        };
        self.next_issue_tick.max(queue_ready_tick)
    }

    pub fn configure(&mut self, rules: C220Mte1TimingRules) -> Result<(), C220Mte1TimingError> {
        if !self.pending_retirements.is_empty() || self.unsignaled_retire_tick.is_some() {
            return Err(C220Mte1TimingError::Busy);
        }
        self.rules = rules;
        Ok(())
    }

    pub fn pending_drain_tick(&self) -> Option<u64> {
        self.pending_retirements
            .iter()
            .map(|ticket| ticket.retire_tick)
            .max()
    }

    pub fn pending_data_ready_tick(&self, destination: C220Load2dDestination) -> Option<u64> {
        self.pending_retirements
            .iter()
            .filter(|ticket| ticket.transfer.instruction.destination == destination)
            .map(|ticket| ticket.data_ready_tick)
            .max()
    }

    pub fn pending_visibility_tick(&self, destination: C220Load2dDestination) -> Option<u64> {
        self.pending_retirements
            .iter()
            .filter(|ticket| ticket.transfer.instruction.destination == destination)
            .map(|ticket| ticket.retire_tick)
            .max()
    }

    pub fn pending_retirement_count(&self) -> usize {
        self.pending_retirements.len()
    }

    pub fn last_retirements(&self) -> &[C220Mte1Ticket] {
        &self.last_retirements
    }

    pub fn advance_to(&mut self, tick: u64) -> &[C220Mte1Ticket] {
        self.begin_retirements();
        while self.retire_ready_front(tick) {}
        &self.last_retirements
    }

    pub(super) fn begin_retirements(&mut self) {
        self.last_retirements.clear();
    }

    pub(super) fn retire_ready_front(&mut self, tick: u64) -> bool {
        if self
            .pending_retirements
            .front()
            .is_some_and(|ticket| ticket.retire_tick <= tick)
        {
            self.last_retirements.push(
                self.pending_retirements
                    .pop_front()
                    .expect("ready MTE1 retirement exists"),
            );
            true
        } else {
            false
        }
    }

    pub fn preview_issue(
        &self,
        tick: u64,
        transfer: C220Load2dTransfer,
    ) -> Result<C220Mte1Ticket, C220Mte1TimingError> {
        let rules = self.rules;
        if tick < self.next_issue_tick {
            return Err(C220Mte1TimingError::TicketMismatch);
        }
        if self.pending_retirements.len() >= C220_MTE1_MAX_OUTSTANDING {
            let ready_tick = self
                .pending_retirements
                .iter()
                .map(|ticket| ticket.retire_tick)
                .min()
                .ok_or(C220Mte1TimingError::TicketMismatch)?;
            return Err(C220Mte1TimingError::QueueFull { ready_tick });
        }
        let (rate, initial_state) = match transfer.instruction.destination {
            C220Load2dDestination::L0a => (rules.l1_to_l0a_bytes_per_tick, &self.l0a),
            C220Load2dDestination::L0b => (rules.l1_to_l0b_bytes_per_tick, &self.l0b),
            destination => {
                return Err(C220Mte1TimingError::UnsupportedDestination(destination));
            }
        };
        let mut route = initial_state.clone();
        let service_ticks = u64::from(C220_LOAD_2D_BLOCK_BYTES).div_ceil(rate.get());
        let mut command_tick = tick
            .checked_add(C220_MTE1_COMMAND_TICKS)
            .ok_or(C220Mte1TimingError::TimeOverflow)?;
        let mut uops = Vec::new();
        uops.try_reserve_exact(usize::from(transfer.descriptor.repeat_count))
            .map_err(|_| C220Mte1TimingError::TimeOverflow)?;
        let mut total_service = 0_u64;
        for segment in transfer.segments() {
            discard_released(&mut route.destination_releases, command_tick);
            if route.destination_releases.len() >= C220_MTE1_DESTINATION_QUEUE_DEPTH {
                command_tick = command_tick.max(
                    route
                        .destination_releases
                        .pop_front()
                        .expect("full MTE1 destination queue has a release"),
                );
                discard_released(&mut route.destination_releases, command_tick);
            }
            let l1_data_ready_tick = command_tick
                .checked_add(C220_MTE1_L1_READ_TICKS)
                .ok_or(C220Mte1TimingError::TimeOverflow)?;
            let transfer_start_tick = l1_data_ready_tick.max(route.data_port_tick);
            let data_ready_tick = transfer_start_tick
                .checked_add(service_ticks)
                .and_then(|tick| tick.checked_add(C220_MTE1_DESTINATION_PROGRESS_TICKS))
                .ok_or(C220Mte1TimingError::TimeOverflow)?;
            route.data_port_tick = data_ready_tick;
            route.destination_releases.push_back(data_ready_tick);
            total_service = total_service
                .checked_add(service_ticks)
                .ok_or(C220Mte1TimingError::TimeOverflow)?;
            uops.push(C220Mte1UopTicket {
                repeat_index: segment.repeat_index,
                command_tick,
                l1_data_ready_tick,
                transfer_start_tick,
                data_ready_tick,
                modeled_service_ticks: service_ticks,
            });
            command_tick = command_tick
                .checked_add(1)
                .ok_or(C220Mte1TimingError::TimeOverflow)?;
        }
        let data_ready_tick = uops.last().map_or(tick, |uop| uop.data_ready_tick);
        let mut retire_tick = data_ready_tick
            .checked_add(C220_MTE1_RETIRE_TICKS)
            .ok_or(C220Mte1TimingError::TimeOverflow)?;
        if let Some(previous) = self.pending_retirements.back() {
            retire_tick = retire_tick.max(
                previous
                    .retire_tick
                    .checked_add(1)
                    .ok_or(C220Mte1TimingError::TimeOverflow)?,
            );
        }
        tick.checked_add(rules.issue_interval.get())
            .ok_or(C220Mte1TimingError::TimeOverflow)?;
        Ok(C220Mte1Ticket {
            issue_tick: tick,
            data_ready_tick,
            retire_tick,
            transfer,
            uops,
            modeled_service_ticks: total_service,
            resource_after: route,
        })
    }

    pub fn issue(&mut self, ticket: &C220Mte1Ticket) -> Result<(), C220Mte1TimingError> {
        let rules = self.rules;
        if ticket.issue_tick < self.next_issue_tick {
            return Err(C220Mte1TimingError::TicketMismatch);
        }
        if self.pending_retirements.len() >= C220_MTE1_MAX_OUTSTANDING {
            let ready_tick = self
                .pending_retirements
                .iter()
                .map(|ticket| ticket.retire_tick)
                .min()
                .ok_or(C220Mte1TimingError::TicketMismatch)?;
            return Err(C220Mte1TimingError::QueueFull { ready_tick });
        }
        match ticket.transfer.instruction.destination {
            C220Load2dDestination::L0a => self.l0a = ticket.resource_after.clone(),
            C220Load2dDestination::L0b => self.l0b = ticket.resource_after.clone(),
            destination => {
                return Err(C220Mte1TimingError::UnsupportedDestination(destination));
            }
        }
        self.next_issue_tick = ticket
            .issue_tick
            .checked_add(rules.issue_interval.get())
            .ok_or(C220Mte1TimingError::TimeOverflow)?;
        self.unsignaled_retire_tick = Some(
            self.unsignaled_retire_tick
                .map_or(ticket.retire_tick, |tick| tick.max(ticket.retire_tick)),
        );
        self.pending_retirements.push_back(ticket.clone());
        Ok(())
    }

    pub fn set_event(&mut self, tick: u64, event_id: u32) {
        let ready_tick = self.unsignaled_retire_tick.take().unwrap_or(tick);
        self.events
            .entry(event_id)
            .or_default()
            .push_back(ready_tick);
    }

    pub fn event_ready_tick(&self, event_id: u32) -> Option<u64> {
        self.events
            .get(&event_id)
            .and_then(|tokens| tokens.front().copied())
    }

    pub fn wait_event(&mut self, tick: u64, event_id: u32) -> Result<(), C220Mte1TimingError> {
        let tokens = self
            .events
            .get_mut(&event_id)
            .ok_or(C220Mte1TimingError::MissingEvent { event_id })?;
        let ready_tick = tokens
            .front()
            .copied()
            .ok_or(C220Mte1TimingError::MissingEvent { event_id })?;
        if tick < ready_tick {
            return Err(C220Mte1TimingError::EventNotReady {
                event_id,
                ready_tick,
            });
        }
        tokens.pop_front();
        if tokens.is_empty() {
            self.events.remove(&event_id);
        }
        Ok(())
    }
}

fn discard_released(releases: &mut VecDeque<u64>, tick: u64) {
    while releases.front().is_some_and(|release| *release <= tick) {
        releases.pop_front();
    }
}
