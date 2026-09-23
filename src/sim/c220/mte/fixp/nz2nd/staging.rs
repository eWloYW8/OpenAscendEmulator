use std::collections::VecDeque;

use super::super::{C220FixpConversionEntry, C220FixpConversionPipeline};
use super::{
    C220FixpNz2ndWriteDescriptor, C220FixpTransposeBuffer, C220FixpTransposeError,
    C220FixpTransposeProgress,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FixpNz2ndWriteUop {
    pub instruction_id: u64,
    pub request_id: u32,
    pub descriptor: C220FixpNz2ndWriteDescriptor,
    pub last_in_instruction: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::c220::mte::fixp::C220FixpDescriptor;
    use crate::sim::c220::{
        memory::C220L0c,
        mte::{
            fixp::{C220FixpCommand, C220FixpNz2ndInstructionPlan, C220FixpSourceFormat},
            interface::C220MteL0cReadInterface,
        },
    };

    #[test]
    fn conversion_credits_wait_for_alignment_capacity_and_output_latency() {
        let command = C220FixpCommand {
            descriptor: C220FixpDescriptor {
                xt: (2 << 32) | (8 << 16) | (1 << 4),
                xm: (1 << 43) | (1 << 34),
                nd: 1,
            },
            source_format: C220FixpSourceFormat::Fp32,
            source_address: 0,
            destination_address: 0,
            control: 0,
            scalar_slope: 0,
            slope_base_block: 0,
        };
        let plan = C220FixpNz2ndInstructionPlan::new(command, 7, 0, 0, 64, 8).unwrap();
        let mut staging = C220FixpNz2ndStaging::new(16).unwrap();
        staging.submit(plan.writes);
        let mut memory = C220L0c::new(131072, 12).unwrap();
        let mut input = C220MteL0cReadInterface::new(32, 0).unwrap();
        let mut conversion = C220FixpConversionPipeline::default();
        for (id, read) in plan.reads.enumerate() {
            let tick = id as u64 * 10;
            input.send(tick, read.operation, &mut memory).unwrap();
            input.receive(tick + 1, &mut memory).unwrap();
            conversion.receive(tick + 2, &mut input, false).unwrap();
            assert!(
                staging
                    .receive(tick + 6, &mut conversion)
                    .unwrap()
                    .is_some()
            );
            let result = staging.stage(tick + 7).unwrap();
            if id < 7 {
                assert!(matches!(
                    result,
                    C220FixpTransposeProgress::Released { slots: 1, .. }
                ));
                assert_eq!(staging.alignment().back().unwrap().ready_tick, tick + 13);
            } else {
                assert_eq!(
                    result,
                    C220FixpTransposeProgress::OutputBlocked { retry_tick: 78 }
                );
            }
        }
        assert_eq!(staging.alignment().len(), 7);
        assert_eq!(staging.pending().len(), 1);
        assert_eq!(staging.transpose().slots()[7].rows, 1);
        assert_eq!(
            staging
                .take_ready(78, true)
                .unwrap()
                .unwrap()
                .operation
                .request_id,
            0
        );
        assert!(matches!(
            staging.stage(78).unwrap(),
            C220FixpTransposeProgress::Released { .. }
        ));
        assert_eq!(staging.alignment().back().unwrap().ready_tick, 84);
        assert!(staging.take_ready(79, false).unwrap().is_none());
        let mut output = super::super::C220FixpNz2ndOutput::default();
        let policy = super::super::C220FixpNz2ndOutputPolicy {
            burst_sizes: [512, 256, 32].map(|n| std::num::NonZeroU32::new(n).unwrap()),
            burst_control: 0,
            row_stride_bytes: 4,
        };
        for id in 1..8 {
            let tick = 84 + u64::from(id);
            let entry = output.receive(tick, &mut staging, policy).unwrap().unwrap();
            assert_eq!(entry.operation.request_id, id);
            assert_eq!(entry.operation.descriptor.bytes, 2);
            assert_eq!(entry.operation.last_in_instruction, id == 7);
            let fragment = output.take_write(tick + 1, true).unwrap().unwrap();
            assert_eq!(fragment.destination_address, u64::from(id) * 4);
            assert_eq!(fragment.bytes, 2);
            assert_eq!(fragment.last_in_instruction, id == 7);
        }
        assert!(output.bursts().is_empty());
        assert!(staging.is_idle());
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FixpNz2ndStagingEntry {
    pub operation: C220FixpNz2ndWriteUop,
    pub ready_tick: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum C220FixpNz2ndStagingError {
    #[error(transparent)]
    Transpose(#[from] C220FixpTransposeError),
    #[error("NZ2ND staging time reversed from {previous} to {requested}")]
    TimeReversed { previous: u64, requested: u64 },
    #[error("NZ2ND staging output callback repeated at tick {0}")]
    RepeatedOutput(u64),
    #[error("NZ2ND staging time overflow")]
    TimeOverflow,
}

/// Couples conversion credits to pre-generated write descriptors. Released
/// descriptors spend six cycles in the bounded alignment queue before the
/// output aggregator can accept them.
#[derive(Debug, Clone)]
pub struct C220FixpNz2ndStaging {
    transpose: C220FixpTransposeBuffer,
    pending: VecDeque<C220FixpNz2ndWriteUop>,
    alignment: VecDeque<C220FixpNz2ndStagingEntry>,
    observed_tick: Option<u64>,
    output_tick: Option<u64>,
}

impl C220FixpNz2ndStaging {
    pub fn new(total_slots: usize) -> Result<Self, C220FixpNz2ndStagingError> {
        Ok(Self {
            transpose: C220FixpTransposeBuffer::new(total_slots)?,
            pending: VecDeque::new(),
            alignment: VecDeque::new(),
            observed_tick: None,
            output_tick: None,
        })
    }

    pub fn submit(&mut self, operations: impl IntoIterator<Item = C220FixpNz2ndWriteUop>) {
        self.pending.extend(operations);
    }

    pub fn transpose(&self) -> &C220FixpTransposeBuffer {
        &self.transpose
    }
    pub fn pending(&self) -> &VecDeque<C220FixpNz2ndWriteUop> {
        &self.pending
    }
    pub fn alignment(&self) -> &VecDeque<C220FixpNz2ndStagingEntry> {
        &self.alignment
    }

    pub fn is_idle(&self) -> bool {
        self.transpose.is_idle() && self.pending.is_empty() && self.alignment.is_empty()
    }

    pub fn receive(
        &mut self,
        tick: u64,
        conversion: &mut C220FixpConversionPipeline,
    ) -> Result<Option<C220FixpConversionEntry>, C220FixpNz2ndStagingError> {
        self.observe(tick)?;
        Ok(self.transpose.receive(tick, conversion)?)
    }

    pub fn stage(
        &mut self,
        tick: u64,
    ) -> Result<C220FixpTransposeProgress, C220FixpNz2ndStagingError> {
        self.observe(tick)?;
        let ready_tick = tick
            .checked_add(6)
            .ok_or(C220FixpNz2ndStagingError::TimeOverflow)?;
        let progress = self
            .transpose
            .release(tick, !self.pending.is_empty() && self.alignment.len() < 7)?;
        if matches!(progress, C220FixpTransposeProgress::Released { .. }) {
            self.alignment.push_back(C220FixpNz2ndStagingEntry {
                operation: self.pending.pop_front().expect("checked descriptor credit"),
                ready_tick,
            });
        }
        Ok(progress)
    }

    pub fn take_ready(
        &mut self,
        tick: u64,
        destination_ready: bool,
    ) -> Result<Option<C220FixpNz2ndStagingEntry>, C220FixpNz2ndStagingError> {
        self.observe(tick)?;
        if self.output_tick == Some(tick) {
            return Err(C220FixpNz2ndStagingError::RepeatedOutput(tick));
        }
        self.output_tick = Some(tick);
        Ok(if destination_ready {
            self.alignment
                .pop_front_if(|entry| entry.ready_tick <= tick)
        } else {
            None
        })
    }

    fn observe(&mut self, tick: u64) -> Result<(), C220FixpNz2ndStagingError> {
        if let Some(previous) = self.observed_tick
            && tick < previous
        {
            return Err(C220FixpNz2ndStagingError::TimeReversed {
                previous,
                requested: tick,
            });
        }
        self.observed_tick = Some(tick);
        Ok(())
    }
}
