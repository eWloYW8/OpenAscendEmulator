use std::collections::VecDeque;

use crate::sim::c220::memory::C220L0cError;
use crate::sim::c220::mte::interface::{
    C220MteL0cReadAcknowledgment, C220MteL0cReadDelivery, C220MteL0cReadError,
    C220MteL0cReadInterface,
};

pub const fn c220_fixp_conversion_ticks(mode: u32) -> u32 {
    match mode {
        0 | 1 | 12 | 13 | 16 => 4,
        8 | 9 | 21 | 22 => 8,
        10 | 11 => 7,
        23..=26 => 6,
        27 => 3,
        _ => 1,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FixpConversionEntry {
    pub acknowledgment: C220MteL0cReadAcknowledgment,
    pub accepted_tick: u64,
    pub ready_tick: u64,
    pub conversion_ticks: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220FixpConversionReceive {
    AwaitingRead,
    QueueFull { occupancy: usize, limit: u32 },
    HardwareSync { retry_tick: u64 },
    Accepted(C220FixpConversionEntry),
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum C220FixpConversionError {
    #[error("FIXP conversion time reversed from {previous} to {requested}")]
    TimeReversed { previous: u64, requested: u64 },
    #[error("FIXP conversion slice callback already ran at tick {0}")]
    RepeatedSlice(u64),
    #[error(transparent)]
    Read(#[from] C220MteL0cReadError),
}

/// Conversion-stage timing and flow control. Numerical conversion and output
/// slicing belong to the consumer, which must retain this head when blocked.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct C220FixpConversionPipeline {
    entries: VecDeque<C220FixpConversionEntry>,
    observed_tick: Option<u64>,
    slice_tick: Option<u64>,
}

impl C220FixpConversionPipeline {
    pub fn entries(&self) -> &VecDeque<C220FixpConversionEntry> {
        &self.entries
    }

    pub fn receive(
        &mut self,
        tick: u64,
        input: &mut C220MteL0cReadInterface,
        hardware_sync_blocked: bool,
    ) -> Result<C220FixpConversionReceive, C220FixpConversionError> {
        self.check_time(tick)?;
        let mut result = C220FixpConversionReceive::AwaitingRead;
        input.deliver_with(tick, |acknowledgment| {
            let conversion_ticks =
                c220_fixp_conversion_ticks(acknowledgment.operation.conversion_mode);
            if self.entries.len() > conversion_ticks as usize {
                result = C220FixpConversionReceive::QueueFull {
                    occupancy: self.entries.len(),
                    limit: conversion_ticks,
                };
                return Ok(C220MteL0cReadDelivery::Retry);
            }
            let ready_tick = tick
                .checked_add(u64::from(conversion_ticks))
                .ok_or(C220L0cError::TimeOverflow)?;
            if acknowledgment.operation.last_in_instruction && hardware_sync_blocked {
                result = C220FixpConversionReceive::HardwareSync {
                    retry_tick: ready_tick,
                };
                return Ok(C220MteL0cReadDelivery::DeferUntil(ready_tick));
            }
            result = C220FixpConversionReceive::Accepted(C220FixpConversionEntry {
                acknowledgment: *acknowledgment,
                accepted_tick: tick,
                ready_tick,
                conversion_ticks,
            });
            Ok(C220MteL0cReadDelivery::Accept)
        })?;
        if let C220FixpConversionReceive::Accepted(entry) = result {
            self.entries.push_back(entry);
        }
        self.observed_tick = Some(tick);
        Ok(result)
    }

    pub fn take_ready(
        &mut self,
        tick: u64,
        output_ready: bool,
    ) -> Result<Option<C220FixpConversionEntry>, C220FixpConversionError> {
        self.check_time(tick)?;
        if self.slice_tick == Some(tick) {
            return Err(C220FixpConversionError::RepeatedSlice(tick));
        }
        self.observed_tick = Some(tick);
        self.slice_tick = Some(tick);
        Ok(if output_ready {
            self.entries.pop_front_if(|entry| entry.ready_tick <= tick)
        } else {
            None
        })
    }

    fn check_time(&self, tick: u64) -> Result<(), C220FixpConversionError> {
        if let Some(previous) = self.observed_tick
            && tick < previous
        {
            return Err(C220FixpConversionError::TimeReversed {
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
    use crate::sim::c220::{
        cube::C220CubeL0cAccess,
        memory::{C220L0c, C220L0cFragmentRequest, C220L0cReadRequest},
        mte::interface::C220MteL0cReadOperation,
    };

    fn stage_read(input: &mut C220MteL0cReadInterface, memory: &mut C220L0c, tick: u64, mode: u32) {
        input
            .send(
                tick,
                C220MteL0cReadOperation {
                    instruction_id: tick,
                    uop_id: tick as u32,
                    conversion_mode: mode,
                    last_in_instruction: true,
                    data_bytes: 128,
                    destination_address: 0,
                    output_bytes: 128,
                    last_in_uop: true,
                    end_of_burst: true,
                    request: C220L0cReadRequest {
                        id: tick as u32 + 1,
                        data_type: 0,
                        half_accumulator: false,
                        fragments: C220L0cFragmentRequest {
                            address: 0,
                            bytes: 1024,
                            access: C220CubeL0cAccess::Read,
                            check_unit_flags: false,
                            update_unit_flags: false,
                        },
                    },
                },
                memory,
            )
            .unwrap();
        input.receive(tick + 1, memory).unwrap();
    }

    #[test]
    fn hardware_sync_defers_input_retry_and_conversion_starts_on_acceptance() {
        let mut input = C220MteL0cReadInterface::new(32, 0).unwrap();
        let mut memory = C220L0c::new(131072, 12).unwrap();
        let mut conversion = C220FixpConversionPipeline::default();
        stage_read(&mut input, &mut memory, 0, 0);
        assert_eq!(
            conversion.receive(2, &mut input, true).unwrap(),
            C220FixpConversionReceive::HardwareSync { retry_tick: 6 }
        );
        assert!(conversion.entries().is_empty());
        assert_eq!(
            conversion.receive(3, &mut input, false).unwrap(),
            C220FixpConversionReceive::AwaitingRead
        );
        let C220FixpConversionReceive::Accepted(entry) =
            conversion.receive(6, &mut input, false).unwrap()
        else {
            panic!("conversion should accept")
        };
        assert_eq!(entry.ready_tick, 10);
        assert!(input.is_idle());
        assert_eq!(conversion.take_ready(9, true).unwrap(), None);
        assert_eq!(conversion.take_ready(10, false).unwrap(), None);
        assert_eq!(conversion.take_ready(11, true).unwrap(), Some(entry));
    }

    #[test]
    fn incoming_mode_controls_queue_limit_and_does_not_reorder_output() {
        let mut input = C220MteL0cReadInterface::new(32, 0).unwrap();
        let mut memory = C220L0c::new(131072, 12).unwrap();
        let mut conversion = C220FixpConversionPipeline::default();
        for (tick, mode) in [(0, 9), (2, 99)] {
            stage_read(&mut input, &mut memory, tick, mode);
            assert!(matches!(
                conversion.receive(tick + 2, &mut input, false).unwrap(),
                C220FixpConversionReceive::Accepted(_)
            ));
        }
        assert_eq!(conversion.entries()[0].ready_tick, 10);
        assert_eq!(conversion.entries()[1].ready_tick, 5);
        stage_read(&mut input, &mut memory, 4, 99);
        assert_eq!(
            conversion.receive(6, &mut input, false).unwrap(),
            C220FixpConversionReceive::QueueFull {
                occupancy: 2,
                limit: 1
            }
        );
        assert_eq!(input.acknowledgments().len(), 1);
        assert_eq!(conversion.take_ready(6, true).unwrap(), None);
        assert!(conversion.take_ready(10, true).unwrap().is_some());
        assert!(matches!(
            conversion.receive(10, &mut input, false).unwrap(),
            C220FixpConversionReceive::Accepted(_)
        ));
        for (mode, ticks) in [
            (0, 4),
            (1, 4),
            (9, 8),
            (8, 8),
            (26, 6),
            (13, 4),
            (22, 8),
            (21, 8),
            (10, 7),
            (11, 7),
            (16, 4),
            (23, 6),
            (12, 4),
            (24, 6),
            (25, 6),
            (27, 3),
            (99, 1),
        ] {
            assert_eq!(c220_fixp_conversion_ticks(mode), ticks);
        }
    }
}
