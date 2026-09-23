use std::collections::VecDeque;

use super::{C220FixpConversionEntry, C220FixpConversionError, C220FixpConversionPipeline};
use crate::sim::c220::mte::interface::C220MteOutputFragment;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FixpL1Burst {
    pub instruction_id: u64,
    pub request_id: u64,
    pub address: u64,
    pub bytes: u32,
    pub closed: bool,
    pub last_in_instruction: bool,
    pub ready_tick: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum C220FixpL1OutputError {
    #[error("FIXP L1 output time reversed from {previous} to {requested}")]
    TimeReversed { previous: u64, requested: u64 },
    #[error("FIXP L1 output callback already ran at tick {0}")]
    RepeatedWrite(u64),
    #[error("FIXP L1 output time overflowed")]
    TimeOverflow,
    #[error(transparent)]
    Conversion(#[from] C220FixpConversionError),
}

/// Ordinary L1 output aggregation and 256-byte packet timing. This path does
/// not perform numerical conversion or NZ2ND layout transformation.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct C220FixpL1Output {
    bursts: VecDeque<C220FixpL1Burst>,
    observed_tick: Option<u64>,
    write_tick: Option<u64>,
}

impl C220FixpL1Output {
    pub fn bursts(&self) -> &VecDeque<C220FixpL1Burst> {
        &self.bursts
    }

    pub fn can_receive(&self) -> bool {
        self.bursts.len() <= 2
    }

    pub fn receive(
        &mut self,
        tick: u64,
        conversion: &mut C220FixpConversionPipeline,
    ) -> Result<Option<C220FixpConversionEntry>, C220FixpL1OutputError> {
        self.check_time(tick)?;
        let ready_tick = tick
            .checked_add(1)
            .ok_or(C220FixpL1OutputError::TimeOverflow)?;
        let consumed = conversion.take_ready(tick, self.can_receive())?;
        if let Some(entry) = consumed {
            let operation = entry.acknowledgment.operation;
            if operation.last_in_uop {
                if let Some(tail) = self.bursts.back_mut().filter(|tail| !tail.closed) {
                    tail.bytes = tail.bytes.wrapping_add(operation.output_bytes);
                    tail.closed = operation.end_of_burst;
                    tail.last_in_instruction = operation.last_in_instruction;
                } else {
                    self.bursts.push_back(C220FixpL1Burst {
                        instruction_id: operation.instruction_id,
                        request_id: u64::from(operation.request.id),
                        address: operation.destination_address,
                        bytes: operation.output_bytes,
                        closed: operation.end_of_burst,
                        last_in_instruction: operation.last_in_instruction,
                        ready_tick,
                    });
                }
            }
        }
        self.observed_tick = Some(tick);
        Ok(consumed)
    }

    /// The caller supplies credit from the destination write queue. No bytes
    /// are consumed on refusal, and a closed zero-byte burst emits its marker.
    pub fn take_write(
        &mut self,
        tick: u64,
        destination_ready: bool,
    ) -> Result<Option<C220MteOutputFragment>, C220FixpL1OutputError> {
        self.check_time(tick)?;
        if self.write_tick == Some(tick) {
            return Err(C220FixpL1OutputError::RepeatedWrite(tick));
        }
        self.observed_tick = Some(tick);
        self.write_tick = Some(tick);
        let Some(head) = self.bursts.front_mut() else {
            return Ok(None);
        };
        if !destination_ready || head.ready_tick > tick || (!head.closed && head.bytes < 256) {
            return Ok(None);
        }
        let bytes = head.bytes.min(256);
        let finishes_burst = head.closed && bytes == head.bytes;
        let fragment = C220MteOutputFragment {
            instruction_id: head.instruction_id,
            request_id: head.request_id,
            destination_address: head.address,
            bytes,
            last_in_uop: true,
            last_in_instruction: finishes_burst && head.last_in_instruction,
        };
        head.bytes -= bytes;
        head.address = head.address.wrapping_add(u64::from(bytes));
        if finishes_burst {
            self.bursts.pop_front();
        }
        Ok(Some(fragment))
    }

    fn check_time(&self, tick: u64) -> Result<(), C220FixpL1OutputError> {
        if let Some(previous) = self.observed_tick
            && tick < previous
        {
            return Err(C220FixpL1OutputError::TimeReversed {
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
        mte::interface::{C220MteL0cReadInterface, C220MteL0cReadOperation},
    };

    #[test]
    fn merges_converted_slices_and_preserves_write_backpressure_and_tail() {
        let mut memory = C220L0c::new(131072, 12).unwrap();
        let mut input = C220MteL0cReadInterface::new(32, 0).unwrap();
        let mut conversion = C220FixpConversionPipeline::default();
        let mut output = C220FixpL1Output::default();
        for (tick, bytes, end) in [(0, 128, false), (7, 200, true)] {
            input
                .send(
                    tick,
                    C220MteL0cReadOperation {
                        instruction_id: 42,
                        uop_id: tick as u32,
                        conversion_mode: 0,
                        last_in_instruction: end,
                        data_bytes: 256,
                        destination_address: if end { 1152 } else { 1024 },
                        output_bytes: bytes,
                        last_in_uop: true,
                        end_of_burst: end,
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
                    &mut memory,
                )
                .unwrap();
            input.receive(tick + 1, &mut memory).unwrap();
            conversion.receive(tick + 2, &mut input, false).unwrap();
            assert!(output.receive(tick + 6, &mut conversion).unwrap().is_some());
            if !end {
                assert_eq!(output.take_write(6, true).unwrap(), None);
                assert_eq!(output.take_write(7, true).unwrap(), None);
                assert_eq!(output.bursts()[0].bytes, 128);
            }
        }
        assert_eq!(output.bursts()[0].bytes, 328);
        assert_eq!(output.bursts()[0].ready_tick, 7);
        assert_eq!(output.take_write(13, false).unwrap(), None);
        let first = output.take_write(14, true).unwrap().unwrap();
        assert_eq!((first.destination_address, first.bytes), (1024, 256));
        assert!(first.last_in_uop);
        assert!(!first.last_in_instruction);
        let last = output.take_write(15, true).unwrap().unwrap();
        assert_eq!((last.destination_address, last.bytes), (1280, 72));
        assert!(last.last_in_uop && last.last_in_instruction);
        assert!(output.bursts().is_empty());
    }
}
