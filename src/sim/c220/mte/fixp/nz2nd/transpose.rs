use super::super::{C220FixpConversionEntry, C220FixpConversionError, C220FixpConversionPipeline};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct C220FixpTransposeSlot {
    pub rows: u32,
    pub occupied: bool,
    pub end_of_group: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220FixpTransposeProgress {
    Idle,
    Delayed { ready_tick: u64 },
    OutputBlocked { retry_tick: u64 },
    Released { first_slot: usize, slots: u8 },
}

#[derive(Debug, thiserror::Error)]
pub enum C220FixpTransposeError {
    #[error("FIX transpose buffer capacity must be nonzero")]
    Capacity,
    #[error("FIX transpose time reversed from {previous} to {requested}")]
    TimeReversed { previous: u64, requested: u64 },
    #[error("FIX transpose release callback repeated at tick {0}")]
    RepeatedRelease(u64),
    #[error("FIX transpose time overflow")]
    TimeOverflow,
    #[error(transparent)]
    Conversion(#[from] C220FixpConversionError),
}

/// Row credits between numerical conversion and NZ2ND write staging.
/// Each release consumes one row from up to four slots. The owner supplies
/// the corresponding write descriptor and downstream queue credit.
#[derive(Debug, Clone)]
pub struct C220FixpTransposeBuffer {
    slots: Vec<C220FixpTransposeSlot>,
    input: usize,
    output: usize,
    half: usize,
    ready_tick: Option<u64>,
    observed_tick: Option<u64>,
    release_tick: Option<u64>,
}

impl C220FixpTransposeBuffer {
    pub fn new(capacity: usize) -> Result<Self, C220FixpTransposeError> {
        if capacity == 0 {
            return Err(C220FixpTransposeError::Capacity);
        }
        Ok(Self {
            slots: vec![C220FixpTransposeSlot::default(); capacity],
            input: 0,
            output: 0,
            half: 0,
            ready_tick: None,
            observed_tick: None,
            release_tick: None,
        })
    }

    pub fn slots(&self) -> &[C220FixpTransposeSlot] {
        &self.slots
    }

    pub fn cursors(&self) -> (usize, usize, usize) {
        (self.input, self.output, self.half)
    }

    pub fn ready_tick(&self) -> Option<u64> {
        self.ready_tick
    }

    pub fn is_idle(&self) -> bool {
        self.slots.iter().all(|slot| !slot.occupied)
    }

    pub fn receive(
        &mut self,
        tick: u64,
        conversion: &mut C220FixpConversionPipeline,
    ) -> Result<Option<C220FixpConversionEntry>, C220FixpTransposeError> {
        self.check_time(tick)?;
        let next_tick = tick
            .checked_add(1)
            .ok_or(C220FixpTransposeError::TimeOverflow)?;
        let can_receive = conversion.entries().front().is_some_and(|entry| {
            !entry.acknowledgment.operation.last_in_uop || !self.slots[self.input].occupied
        });
        let entry = conversion.take_ready(tick, can_receive)?;
        if let Some(entry) = entry {
            let operation = entry.acknowledgment.operation;
            if operation.last_in_uop {
                self.push(operation.output_bytes, operation.end_of_burst);
                if self.ready_group().is_some() {
                    self.notify(next_tick);
                }
            }
        }
        self.observed_tick = Some(tick);
        Ok(entry)
    }

    pub fn release(
        &mut self,
        tick: u64,
        destination_ready: bool,
    ) -> Result<C220FixpTransposeProgress, C220FixpTransposeError> {
        self.check_time(tick)?;
        if self.release_tick == Some(tick) {
            return Err(C220FixpTransposeError::RepeatedRelease(tick));
        }
        let next_tick = tick
            .checked_add(1)
            .ok_or(C220FixpTransposeError::TimeOverflow)?;
        self.observed_tick = Some(tick);
        self.release_tick = Some(tick);
        let Some(ready_tick) = self.ready_tick else {
            return Ok(C220FixpTransposeProgress::Idle);
        };
        if ready_tick > tick {
            return Ok(C220FixpTransposeProgress::Delayed { ready_tick });
        }
        self.ready_tick = None;
        let Some((count, ends_group)) = self.ready_group() else {
            return Ok(C220FixpTransposeProgress::Idle);
        };
        if !destination_ready {
            self.notify(next_tick);
            return Ok(C220FixpTransposeProgress::OutputBlocked {
                retry_tick: next_tick,
            });
        }
        let first_slot = self.index(0);
        for offset in 0..count {
            let index = self.index(offset);
            let slot = &mut self.slots[index];
            slot.rows = slot.rows.wrapping_sub(1);
            if slot.rows == 0 {
                slot.occupied = false;
            }
        }
        if ends_group || self.half == 4 {
            let drained = (0..count).all(|offset| !self.slots[self.index(offset)].occupied);
            if drained {
                if ends_group {
                    let last = self.index(count - 1);
                    self.slots[last].end_of_group = false;
                }
                self.output = (self.output + self.half + count) % self.slots.len();
                self.half = 0;
            } else {
                self.half = 0;
            }
        } else {
            self.half = 4;
        }
        if self.ready_group().is_some() {
            self.notify(next_tick);
        }
        Ok(C220FixpTransposeProgress::Released {
            first_slot,
            slots: count as u8,
        })
    }

    fn push(&mut self, output_bytes: u32, end_of_group: bool) {
        let slot = &mut self.slots[self.input];
        slot.rows = slot.rows.wrapping_add(output_bytes / 32);
        slot.occupied = true;
        slot.end_of_group |= end_of_group;
        self.input = (self.input + 1) % self.slots.len();
    }

    fn index(&self, offset: usize) -> usize {
        (self.output + self.half + offset) % self.slots.len()
    }

    fn ready_group(&self) -> Option<(usize, bool)> {
        for offset in 0..4 {
            let slot = self.slots[self.index(offset)];
            if !slot.occupied {
                return None;
            }
            if slot.end_of_group {
                return Some((offset + 1, true));
            }
        }
        Some((4, false))
    }

    fn notify(&mut self, tick: u64) {
        self.ready_tick = Some(self.ready_tick.map_or(tick, |previous| previous.min(tick)));
    }

    fn check_time(&self, tick: u64) -> Result<(), C220FixpTransposeError> {
        if let Some(previous) = self.observed_tick
            && tick < previous
        {
            return Err(C220FixpTransposeError::TimeReversed {
                previous,
                requested: tick,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose_alternates_halves_retries_output_and_wraps_short_groups() {
        let mut buffer = C220FixpTransposeBuffer::new(8).unwrap();
        for index in 0..5 {
            buffer.push(64, index == 4);
        }
        buffer.notify(1);
        let before = buffer.slots().to_vec();
        assert_eq!(
            buffer.release(0, true).unwrap(),
            C220FixpTransposeProgress::Delayed { ready_tick: 1 }
        );
        assert_eq!(
            buffer.release(1, false).unwrap(),
            C220FixpTransposeProgress::OutputBlocked { retry_tick: 2 }
        );
        assert_eq!(buffer.slots(), before);
        for (tick, first_slot, slots) in [(2, 0, 4), (3, 4, 1), (4, 0, 4), (5, 4, 1)] {
            assert_eq!(
                buffer.release(tick, true).unwrap(),
                C220FixpTransposeProgress::Released { first_slot, slots }
            );
        }
        assert!(buffer.is_idle());
        assert_eq!(buffer.cursors(), (5, 5, 0));
        assert_eq!(buffer.ready_tick(), None);
        for index in 0..4 {
            buffer.push(32, index == 3);
        }
        buffer.notify(6);
        assert_eq!(
            buffer.release(6, true).unwrap(),
            C220FixpTransposeProgress::Released {
                first_slot: 5,
                slots: 4
            }
        );
        assert_eq!(buffer.cursors(), (1, 1, 0));
        assert!(buffer.is_idle());
        assert!(
            buffer
                .slots()
                .iter()
                .all(|slot| !slot.end_of_group && slot.rows == 0)
        );
    }
}
