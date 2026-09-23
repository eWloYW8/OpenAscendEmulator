use thiserror::Error;

use super::layout::{TILE_EDGE, f16_c_address, f32_c_address};
use super::{C220CubeAccumulatorSource, C220CubeExecutionControl, C220CubeFpStatus, C220CubeIssue};
use crate::isa::c220::cube::{C220CubeDataType, C220CubeOperation};
use crate::memory::pv_memory::PvMemoryError;
use crate::sim::c220::memory::{C220LocalBuffer, C220LocalBufferError, C220LocalMemory};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220CubeExecutionOutcome {
    pub accumulator_source: C220CubeAccumulatorSource,
    pub m: u16,
    pub k: u16,
    pub n: u16,
    pub mac_count: u64,
    /// Number of physical output lanes written, including tile padding.
    pub written_lanes: u64,
    pub padded_lanes: u64,
    pub fp_status: C220CubeFpStatus,
    pub integer_overflow: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220PreparedCubeExecution {
    pub(super) writes: Vec<C220CubeWrite>,
    pub outcome: C220CubeExecutionOutcome,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum C220CubeWrite {
    U16 { address: u64, value: u16 },
    U32 { address: u64, value: u32 },
}

impl C220PreparedCubeExecution {
    pub fn write_count(&self) -> usize {
        self.writes.len()
    }

    pub fn commit(self, memory: &mut C220LocalMemory) -> Result<(), C220CubeExecutionError> {
        self.commit_to_buffer(memory.l0c_mut().buffer_mut())
    }

    pub(super) fn commit_to_buffer(
        &self,
        buffer: &mut C220LocalBuffer,
    ) -> Result<(), C220CubeExecutionError> {
        for &write in &self.writes {
            match write {
                C220CubeWrite::U16 { address, value } => {
                    buffer.write_known_linear(address, &value.to_le_bytes())?;
                }
                C220CubeWrite::U32 { address, value } => {
                    buffer.write_known_linear(address, &value.to_le_bytes())?;
                }
            }
        }
        Ok(())
    }
}

pub(crate) const fn update_cube_status_spr2(
    prior_spr2: u64,
    pc: u64,
    outcome: C220CubeExecutionOutcome,
) -> u64 {
    let mut flags = 0;
    if outcome.integer_overflow || outcome.fp_status.overflow {
        flags |= 1 << 10;
    }
    if outcome.fp_status.underflow {
        flags |= 1 << 11;
    }
    if outcome.fp_status.nan_operand
        || outcome.fp_status.infinity_operand
        || outcome.fp_status.invalid
    {
        flags |= 1 << 15;
    }
    if flags == 0 {
        prior_spr2
    } else {
        (prior_spr2 & 0xffff_ff00_ffff_ffff) | (((pc >> 2) & 0xff) << 32) | flags
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum C220CubeExecutionError {
    #[error("functional Cube execution does not yet support {0:?}")]
    UnsupportedOperation(C220CubeOperation),
    #[error("functional Cube execution does not yet support {0:?}")]
    UnsupportedDataType(C220CubeDataType),
    #[error(
        "sparse selector addresses dense K={dense_k} outside the {loaded_k} loaded input lanes"
    )]
    SparseInputOutsideLoadedTiles { dense_k: u64, loaded_k: u64 },
    #[error(
        "sparse INT4 input row {row} is outside the {initialized_rows} initialized temporary rows"
    )]
    SparseUninitializedRow { row: u64, initialized_rows: u64 },
    #[error(transparent)]
    BiasMemory(#[from] PvMemoryError),
    #[error("exact C220 Cube accumulator exceeded its internal range")]
    ExactAccumulatorOverflow,
    #[error(transparent)]
    LocalBuffer(#[from] C220LocalBufferError),
}

impl C220CubeIssue {
    pub fn prepare(
        self,
        memory: &C220LocalMemory,
        control: C220CubeExecutionControl,
    ) -> Result<C220PreparedCubeExecution, C220CubeExecutionError> {
        if self.parameters.m == 0 || self.parameters.effective_k == 0 || self.parameters.n == 0 {
            return Ok(C220PreparedCubeExecution {
                writes: Vec::new(),
                outcome: C220CubeExecutionOutcome {
                    accumulator_source: self.accumulator_source(),
                    m: self.parameters.m,
                    k: self.parameters.effective_k,
                    n: self.parameters.n,
                    mac_count: 0,
                    written_lanes: 0,
                    padded_lanes: 0,
                    fp_status: C220CubeFpStatus::default(),
                    integer_overflow: false,
                },
            });
        }
        self.validate_functional_mode()?;
        let prepared = super::mmad::prepare(self, memory, control)?;
        Ok(self.complete_output_tiles(prepared))
    }

    fn complete_output_tiles(
        self,
        mut prepared: C220PreparedCubeExecution,
    ) -> C220PreparedCubeExecution {
        let geometry = self.instruction.geometry(self.parameters);
        let m_tiles = u64::from(geometry.m_tiles);
        let n_tiles = u64::from(geometry.n_tiles);
        let m = u64::from(self.parameters.m);
        let n = u64::from(self.parameters.n);
        let mut writes = Vec::with_capacity((m_tiles * n_tiles * TILE_EDGE * TILE_EDGE) as usize);
        for n_tile in 0..n_tiles {
            for m_tile in 0..m_tiles {
                for row in m_tile * TILE_EDGE..(m_tile + 1) * TILE_EDGE {
                    for column in n_tile * TILE_EDGE..(n_tile + 1) * TILE_EDGE {
                        let write = if row < m && column < n {
                            prepared.writes[(row * n + column) as usize]
                        } else if self.instruction.data_type == C220CubeDataType::F16F16 {
                            C220CubeWrite::U16 {
                                address: f16_c_address(
                                    u64::from(self.parameters.xd_low),
                                    m_tiles,
                                    row,
                                    column,
                                ),
                                value: 0,
                            }
                        } else {
                            C220CubeWrite::U32 {
                                address: f32_c_address(
                                    u64::from(self.parameters.xd_low),
                                    m_tiles,
                                    row,
                                    column,
                                ),
                                value: 0,
                            }
                        };
                        writes.push(write);
                    }
                }
            }
        }
        prepared.outcome.written_lanes = writes.len() as u64;
        prepared.outcome.padded_lanes = writes.len() as u64 - m * n;
        prepared.writes = writes;
        prepared
    }

    pub fn execute(
        self,
        memory: &mut C220LocalMemory,
        control: C220CubeExecutionControl,
    ) -> Result<C220CubeExecutionOutcome, C220CubeExecutionError> {
        let prepared = self.prepare(memory, control)?;
        let outcome = prepared.outcome;
        prepared.commit(memory)?;
        Ok(outcome)
    }

    fn validate_functional_mode(self) -> Result<(), C220CubeExecutionError> {
        if self.instruction.operation == C220CubeOperation::SparseMmad
            && !matches!(
                self.instruction.data_type,
                C220CubeDataType::S8S8S32
                    | C220CubeDataType::S4S4S32
                    | C220CubeDataType::U8U8S32
                    | C220CubeDataType::U8S8S32
                    | C220CubeDataType::F16F16
                    | C220CubeDataType::F16F32
                    | C220CubeDataType::Bf16F32
                    | C220CubeDataType::F32F32
            )
        {
            return Err(C220CubeExecutionError::UnsupportedOperation(
                self.instruction.operation,
            ));
        }
        Ok(())
    }
}
