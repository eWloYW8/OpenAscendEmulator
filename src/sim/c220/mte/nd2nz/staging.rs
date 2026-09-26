use std::{collections::VecDeque, num::NonZeroU32};

use super::{C220Nd2NzReadElement, C220Nd2NzReadRequest, C220Nd2NzReadRoute};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Nd2NzStagingConfig {
    pub rows: NonZeroU32,
    pub alignment_depth: u32,
    pub small_data_capacity: NonZeroU32,
    pub receive_bandwidth: NonZeroU32,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum C220Nd2NzStagingError {
    #[error("ND2NZ supports at most eight row buffers, requested {0}")]
    InvalidRowCount(u32),
    #[error("invalid ND2NZ alignment lane {0}")]
    InvalidLane(u32),
    #[error("invalid ND2NZ response geometry")]
    InvalidResponse,
    #[error("ND2NZ row slot {0} is outside the staging buffer")]
    InvalidRow(u32),
    #[error("invalid ND2NZ write row count {0}")]
    InvalidWriteRows(u32),
    #[error("ND2NZ staging time reversed from {previous} to {requested}")]
    TimeReversed { previous: u64, requested: u64 },
    #[error("ND2NZ {callback} callback repeated at tick {tick}")]
    RepeatedCallback { callback: &'static str, tick: u64 },
    #[error("ND2NZ staging time overflow")]
    TimeOverflow,
}

/// Mutable receive progress, separate from the immutable read request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220Nd2NzResponse {
    route: C220Nd2NzReadRoute,
    remaining_bytes: u32,
    padding_bytes: u32,
    elements: VecDeque<C220Nd2NzReadElement>,
}

impl C220Nd2NzResponse {
    pub fn new(request: &C220Nd2NzReadRequest) -> Result<Self, C220Nd2NzStagingError> {
        let total: u64 = request.elements.iter().map(|e| u64::from(e.bytes)).sum();
        if request.bytes == 0
            || total != u64::from(request.bytes)
            || request.elements.iter().any(|e| e.bytes == 0)
            || (request.route == C220Nd2NzReadRoute::PerRow && request.elements.len() != 1)
            || (request.route == C220Nd2NzReadRoute::ContiguousRows && request.padding_bytes != 0)
        {
            return Err(C220Nd2NzStagingError::InvalidResponse);
        }
        Ok(Self {
            route: request.route,
            remaining_bytes: request.bytes,
            padding_bytes: request.padding_bytes,
            elements: request.elements.iter().copied().collect(),
        })
    }

    pub fn remaining_bytes(&self) -> u32 {
        self.remaining_bytes
    }

    pub fn is_complete(&self) -> bool {
        self.remaining_bytes == 0
    }
}

#[derive(Debug, Clone)]
struct RowFragment {
    ready_tick: u64,
    bytes: u32,
    padding_bytes: u32,
}

#[derive(Debug, Clone)]
struct SmallBatch {
    ready_tick: u64,
    elements: VecDeque<C220Nd2NzReadElement>,
}

/// Response queues and per-row write credits. Each method represents one
/// scheduler callback; source instruction ownership is enforced by the engine.
#[derive(Debug, Clone)]
pub struct C220Nd2NzStaging {
    config: C220Nd2NzStagingConfig,
    alignment: Vec<u32>,
    small_bytes: u32,
    small: VecDeque<SmallBatch>,
    rows: Vec<VecDeque<RowFragment>>,
    observed_tick: Option<u64>,
    receive_tick: Option<u64>,
    small_tick: Option<u64>,
    lane_ticks: [Option<u64>; 4],
    write_tick: Option<u64>,
}

impl C220Nd2NzStaging {
    pub fn new(config: C220Nd2NzStagingConfig) -> Result<Self, C220Nd2NzStagingError> {
        if config.rows.get() > 8 {
            return Err(C220Nd2NzStagingError::InvalidRowCount(config.rows.get()));
        }
        let rows = config.rows.get() as usize;
        Ok(Self {
            config,
            alignment: vec![0; rows],
            small_bytes: 0,
            small: VecDeque::new(),
            rows: vec![VecDeque::new(); rows],
            observed_tick: None,
            receive_tick: None,
            small_tick: None,
            lane_ticks: [None; 4],
            write_tick: None,
        })
    }

    pub fn config(&self) -> C220Nd2NzStagingConfig {
        self.config
    }

    pub fn alignment_bytes(&self) -> &[u32] {
        &self.alignment
    }

    pub fn small_data_bytes(&self) -> u32 {
        self.small_bytes
    }

    pub fn pending_small_batches(&self) -> usize {
        self.small.len()
    }

    pub fn pending_row_fragments(&self, row: u32) -> Option<usize> {
        self.rows.get(row as usize).map(VecDeque::len)
    }

    pub fn is_idle(&self) -> bool {
        self.small.is_empty()
            && self.rows.iter().all(VecDeque::is_empty)
            && self.alignment.iter().all(|&bytes| bytes == 0)
    }

    /// Accept at most one receive-bandwidth quantum. The caller retains the
    /// response until `is_complete`, including when this call accepts bytes.
    pub fn receive(
        &mut self,
        tick: u64,
        response: &mut C220Nd2NzResponse,
    ) -> Result<u32, C220Nd2NzStagingError> {
        for element in &response.elements {
            self.check_row(element.row_slot)?;
        }
        let ready_tick = tick
            .checked_add(1)
            .ok_or(C220Nd2NzStagingError::TimeOverflow)?;
        self.observe(tick)?;
        Self::callback(&mut self.receive_tick, tick, "receive")?;
        if response.is_complete() {
            return Ok(0);
        }
        let bandwidth = self.config.receive_bandwidth.get();
        let accepted = match response.route {
            C220Nd2NzReadRoute::PerRow => {
                let bytes = response.remaining_bytes.min(bandwidth);
                let element = response
                    .elements
                    .front_mut()
                    .expect("validated row response");
                self.rows[element.row_slot as usize].push_back(RowFragment {
                    ready_tick,
                    bytes,
                    padding_bytes: response.padding_bytes,
                });
                element.bytes -= bytes;
                bytes
            }
            C220Nd2NzReadRoute::ContiguousRows => {
                let mut budget =
                    bandwidth.min(self.config.small_data_capacity.get() - self.small_bytes);
                let mut elements = VecDeque::new();
                let mut accepted = 0;
                while budget != 0 {
                    let Some(head) = response.elements.front_mut() else {
                        break;
                    };
                    let bytes = head.bytes.min(budget);
                    elements.push_back(C220Nd2NzReadElement {
                        row_slot: head.row_slot,
                        bytes,
                    });
                    head.bytes -= bytes;
                    budget -= bytes;
                    accepted += bytes;
                    if head.bytes == 0 {
                        response.elements.pop_front();
                    }
                }
                if !elements.is_empty() {
                    self.small.push_back(SmallBatch {
                        ready_tick,
                        elements,
                    });
                }
                self.small_bytes += accepted;
                accepted
            }
        };
        response.remaining_bytes -= accepted;
        Ok(accepted)
    }

    /// Each of four lanes serves its lower row first, falling back to the row
    /// four slots above it only when the lower row cannot advance.
    pub fn stage_lane(
        &mut self,
        tick: u64,
        lane: u32,
    ) -> Result<Option<u32>, C220Nd2NzStagingError> {
        if lane >= 4 {
            return Err(C220Nd2NzStagingError::InvalidLane(lane));
        }
        self.observe(tick)?;
        Self::callback(&mut self.lane_ticks[lane as usize], tick, "row staging")?;
        for row in [lane, lane + 4] {
            if row < self.config.rows.get() && self.stage_row(tick, row as usize) {
                return Ok(Some(row));
            }
        }
        Ok(None)
    }

    fn stage_row(&mut self, tick: u64, row: usize) -> bool {
        let Some(head) = self.rows[row].front() else {
            return false;
        };
        if head.ready_tick > tick || !self.fits(row, head.bytes) {
            return false;
        }
        self.alignment[row] = self.alignment[row]
            .wrapping_add(head.bytes)
            .wrapping_add(head.padding_bytes);
        self.rows[row].pop_front();
        true
    }

    /// Drain the eligible front batch only. A blocked element retains every
    /// following element, even if a later row has available capacity.
    pub fn stage_small(&mut self, tick: u64) -> Result<u32, C220Nd2NzStagingError> {
        self.observe(tick)?;
        Self::callback(&mut self.small_tick, tick, "small-data staging")?;
        let Some(batch) = self.small.front_mut() else {
            return Ok(0);
        };
        if batch.ready_tick > tick {
            return Ok(0);
        }
        let mut moved = 0;
        while let Some(head) = batch.elements.front() {
            let occupancy = &mut self.alignment[head.row_slot as usize];
            if occupancy.wrapping_add(head.bytes) >= self.config.alignment_depth.wrapping_add(32) {
                break;
            }
            *occupancy = occupancy.wrapping_add(head.bytes);
            self.small_bytes -= head.bytes;
            moved += head.bytes;
            batch.elements.pop_front();
        }
        if batch.elements.is_empty() {
            self.small.pop_front();
        }
        Ok(moved)
    }

    /// Consume one 32-byte unit from each participating row only when the
    /// destination accepts the write. Instruction completion belongs to send.
    pub fn take_write_credit(
        &mut self,
        tick: u64,
        rows: u32,
        destination_ready: bool,
    ) -> Result<bool, C220Nd2NzStagingError> {
        let ready = self.write_ready(tick, rows)? && destination_ready;
        self.observe(tick)?;
        self.write_tick = Some(tick);
        if ready {
            for bytes in &mut self.alignment[..rows as usize] {
                *bytes -= 32;
            }
        }
        Ok(ready)
    }

    pub(crate) fn write_ready(&self, tick: u64, rows: u32) -> Result<bool, C220Nd2NzStagingError> {
        if rows == 0 || rows > self.config.rows.get() {
            return Err(C220Nd2NzStagingError::InvalidWriteRows(rows));
        }
        if let Some(previous) = self.observed_tick
            && tick < previous
        {
            return Err(C220Nd2NzStagingError::TimeReversed {
                previous,
                requested: tick,
            });
        }
        if self.write_tick == Some(tick) {
            return Err(C220Nd2NzStagingError::RepeatedCallback {
                callback: "write",
                tick,
            });
        }
        Ok(self.alignment[..rows as usize]
            .iter()
            .all(|&bytes| bytes >= 32))
    }

    fn fits(&self, row: usize, bytes: u32) -> bool {
        self.alignment[row].wrapping_add(bytes) < self.config.alignment_depth.wrapping_add(32)
    }

    fn check_row(&self, row: u32) -> Result<(), C220Nd2NzStagingError> {
        if row >= self.config.rows.get() {
            return Err(C220Nd2NzStagingError::InvalidRow(row));
        }
        Ok(())
    }

    fn observe(&mut self, tick: u64) -> Result<(), C220Nd2NzStagingError> {
        if let Some(previous) = self.observed_tick
            && tick < previous
        {
            return Err(C220Nd2NzStagingError::TimeReversed {
                previous,
                requested: tick,
            });
        }
        self.observed_tick = Some(tick);
        Ok(())
    }

    fn callback(
        last: &mut Option<u64>,
        tick: u64,
        callback: &'static str,
    ) -> Result<(), C220Nd2NzStagingError> {
        if *last == Some(tick) {
            return Err(C220Nd2NzStagingError::RepeatedCallback { callback, tick });
        }
        *last = Some(tick);
        Ok(())
    }
}
