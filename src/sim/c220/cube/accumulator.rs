use crate::sim::c220::memory::C220LocalMemory;

use super::{C220CubeExecutionError, C220CubeIssue};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220CubeAccumulatorSource {
    Zero,
    L0c,
    Bias { address: u32, read_bytes: u32 },
}

impl C220CubeIssue {
    pub fn accumulator_source(self) -> C220CubeAccumulatorSource {
        if self.parameters.xt_bit_63 {
            C220CubeAccumulatorSource::Zero
        } else if self.parameters.xt_bit_62 {
            C220CubeAccumulatorSource::Bias {
                address: self.parameters.xd_high,
                read_bytes: u32::from(self.parameters.n).div_ceil(16) * 16 * 8,
            }
        } else {
            C220CubeAccumulatorSource::L0c
        }
    }
}

pub(super) struct C220CubeAccumulator {
    source: C220CubeAccumulatorSource,
    bias: Vec<u8>,
}

impl C220CubeAccumulator {
    pub(super) fn load(
        issue: C220CubeIssue,
        memory: &C220LocalMemory,
    ) -> Result<Self, C220CubeExecutionError> {
        let source = issue.accumulator_source();
        let mut bias = Vec::new();
        if let C220CubeAccumulatorSource::Bias {
            address,
            read_bytes,
        } = source
        {
            bias.resize(read_bytes as usize, 0);
            memory.bt().read_into(u64::from(address), &mut bias)?;
        }
        Ok(Self { source, bias })
    }

    pub(super) fn lane_u32(
        &self,
        memory: &C220LocalMemory,
        output_address: u64,
        column: u64,
    ) -> Result<u32, C220CubeExecutionError> {
        Ok(u32::from_le_bytes(self.lane_bytes(
            memory,
            output_address,
            column,
        )?))
    }

    pub(super) fn lane_u16(
        &self,
        memory: &C220LocalMemory,
        output_address: u64,
        column: u64,
    ) -> Result<u16, C220CubeExecutionError> {
        Ok(u16::from_le_bytes(self.lane_bytes(
            memory,
            output_address,
            column,
        )?))
    }

    fn lane_bytes<const N: usize>(
        &self,
        memory: &C220LocalMemory,
        output_address: u64,
        column: u64,
    ) -> Result<[u8; N], C220CubeExecutionError> {
        let mut bytes = [0; N];
        match self.source {
            C220CubeAccumulatorSource::Zero => {}
            C220CubeAccumulatorSource::L0c => {
                bytes.copy_from_slice(
                    &memory
                        .l0c()
                        .buffer()
                        .read_initialized_linear(output_address, N)?,
                );
            }
            C220CubeAccumulatorSource::Bias { .. } => {
                let offset = column as usize * 4;
                bytes.copy_from_slice(&self.bias[offset..offset + N]);
            }
        }
        Ok(bytes)
    }
}
