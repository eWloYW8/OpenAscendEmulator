use super::accumulator::C220CubeAccumulator;
use super::execute::C220CubeWrite;
use super::layout::{
    f16_b_address, f16_c_address, f32_b_address, f32_c_address, integer_a_element,
    integer_b_element, read_input,
};
use super::numeric::{
    F32SliceOutcome, evaluate_bf16_f32_slice, evaluate_f16_f16_slice, evaluate_f16_f32_slice,
    evaluate_f32_f32_slice, evaluate_hf32_f32_slice,
};
use super::{
    C220CubeExecutionControl, C220CubeExecutionError, C220CubeExecutionOutcome, C220CubeFpStatus,
    C220CubeIssue, C220F32MmadMode, C220PreparedCubeExecution,
};
use crate::isa::c220::cube::{C220CubeDataType, C220CubeGeometry, C220CubeOperation};
use crate::sim::c220::memory::{C220LocalBuffer, C220LocalBufferError, C220LocalMemory};

pub(super) fn prepare(
    issue: C220CubeIssue,
    memory: &C220LocalMemory,
    control: C220CubeExecutionControl,
) -> Result<C220PreparedCubeExecution, C220CubeExecutionError> {
    if matches!(
        issue.instruction.data_type,
        C220CubeDataType::F16U2 | C220CubeDataType::B8U2
    ) {
        return Err(C220CubeExecutionError::UnsupportedDataType(
            issue.instruction.data_type,
        ));
    }
    let kernel = Mmad {
        issue,
        memory,
        control,
        geometry: issue.instruction.geometry(issue.parameters),
    };
    let initialization = C220CubeAccumulator::load(issue, memory)?;
    let parameters = issue.parameters;
    let mut outcome = C220CubeExecutionOutcome {
        accumulator_source: issue.accumulator_source(),
        m: parameters.m,
        k: parameters.effective_k,
        n: parameters.n,
        mac_count: u64::from(parameters.m)
            * u64::from(parameters.effective_k)
            * u64::from(parameters.n),
        written_lanes: 0,
        padded_lanes: 0,
        fp_status: C220CubeFpStatus::default(),
        integer_overflow: false,
    };
    let mut writes = Vec::new();
    let half_output = issue.instruction.data_type == C220CubeDataType::F16F16;
    for m in 0..u64::from(parameters.m) {
        for n in 0..u64::from(parameters.n) {
            let base = u64::from(parameters.xd_low);
            let m_tiles = u64::from(kernel.geometry.m_tiles);
            let address = if half_output {
                f16_c_address(base, m_tiles, m, n)
            } else {
                f32_c_address(base, m_tiles, m, n)
            };
            let mut accumulator = if half_output {
                u32::from(initialization.lane_u16(memory, address, n)?)
            } else {
                initialization.lane_u32(memory, address, n)?
            };
            for k_base in (0..u64::from(parameters.effective_k)).step_by(kernel.slice_width()) {
                accumulator = kernel.evaluate_slice(m, n, k_base, accumulator, &mut outcome)?;
            }
            writes.push(if half_output {
                C220CubeWrite::U16 {
                    address,
                    value: accumulator as u16,
                }
            } else {
                C220CubeWrite::U32 {
                    address,
                    value: accumulator,
                }
            });
        }
    }
    outcome.written_lanes = writes.len() as u64;
    Ok(C220PreparedCubeExecution { writes, outcome })
}

struct Mmad<'a> {
    issue: C220CubeIssue,
    memory: &'a C220LocalMemory,
    control: C220CubeExecutionControl,
    geometry: C220CubeGeometry,
}

impl Mmad<'_> {
    fn slice_width(&self) -> usize {
        if self.issue.instruction.data_type == C220CubeDataType::F32F32
            && self.control.f32_mode == C220F32MmadMode::Fp32
        {
            4
        } else {
            usize::from(self.geometry.k_tile_elements)
        }
    }

    fn evaluate_slice(
        &self,
        m: u64,
        n: u64,
        k_base: u64,
        accumulator: u32,
        outcome: &mut C220CubeExecutionOutcome,
    ) -> Result<u32, C220CubeExecutionError> {
        let mode = self.control.fp16_mode;
        let slice = match self.issue.instruction.data_type {
            C220CubeDataType::F16F16 => {
                let (left, right) = self.read_float16(m, n, k_base)?;
                let slice = evaluate_f16_f16_slice(left, right, accumulator as u16, mode)?;
                F32SliceOutcome {
                    bits: u32::from(slice.bits),
                    status: slice.status,
                }
            }
            C220CubeDataType::F16F32 => {
                let (left, right) = self.read_float16(m, n, k_base)?;
                evaluate_f16_f32_slice(left, right, accumulator, mode)?
            }
            C220CubeDataType::Bf16F32 => {
                let (left, right) = self.read_float16(m, n, k_base)?;
                evaluate_bf16_f32_slice(left, right, accumulator, mode)?
            }
            C220CubeDataType::F32F32 if self.control.f32_mode == C220F32MmadMode::Fp32 => {
                let (left, right) = self.read_float32::<4>(m, n, k_base)?;
                evaluate_f32_f32_slice(left, right, accumulator, mode)?
            }
            C220CubeDataType::F32F32 => {
                let (left, right) = self.read_float32::<8>(m, n, k_base)?;
                evaluate_hf32_f32_slice(left, right, accumulator, mode, self.control.hf32_rounding)?
            }
            data_type => {
                let input_mode = match data_type {
                    C220CubeDataType::U8U8S32 => IntegerInputMode::UnsignedUnsigned,
                    C220CubeDataType::S8S8S32 => IntegerInputMode::SignedSigned,
                    C220CubeDataType::S4S4S32 => IntegerInputMode::Signed4Signed4,
                    C220CubeDataType::U8S8S32 => IntegerInputMode::UnsignedSigned,
                    _ => return Err(C220CubeExecutionError::UnsupportedDataType(data_type)),
                };
                let mut sum = i64::from(accumulator as i32);
                let k_tile = u64::from(self.geometry.k_tile_elements);
                for lane in 0..self.active_lanes(k_base, k_tile) {
                    let k = k_base + lane;
                    let (left, right) = if self.issue.instruction.operation
                        == C220CubeOperation::SparseMmad
                    {
                        super::sparse::read_byte_pair(self.issue.parameters, self.memory, m, n, k)?
                    } else {
                        let left = input_mode.read(
                            self.memory.l0a(),
                            self.issue.parameters.xn,
                            self.a_element(u64::from(self.geometry.k_tiles), k_tile, m, k),
                        )?;
                        let right = input_mode.read(
                            self.memory.l0b(),
                            self.issue.parameters.xm,
                            integer_b_element(u64::from(self.geometry.n_tiles), k_tile, k, n),
                        )?;
                        (left, right)
                    };
                    sum += input_mode.product(left, right);
                }
                let saturated = sum.clamp(i64::from(i32::MIN), i64::from(i32::MAX));
                outcome.integer_overflow |= saturated != sum;
                return Ok(saturated as i32 as u32);
            }
        };
        outcome.fp_status.merge(slice.status);
        Ok(slice.bits)
    }

    fn active_lanes(&self, k_base: u64, width: u64) -> u64 {
        (u64::from(self.issue.parameters.effective_k) - k_base).min(width)
    }

    fn a_element(&self, k_tiles: u64, k_tile: u64, m: u64, k: u64) -> u64 {
        if self.issue.parameters.m == 1
            && self.issue.instruction.operation == C220CubeOperation::Mmad
        {
            k
        } else {
            integer_a_element(k_tiles, k_tile, m, k)
        }
    }

    fn read_float16(
        &self,
        m: u64,
        n: u64,
        k_base: u64,
    ) -> Result<([u16; 16], [u16; 16]), C220LocalBufferError> {
        let mut left = [0; 16];
        let mut right = [0; 16];
        for lane in 0..self.active_lanes(k_base, 16) {
            let k = k_base + lane;
            left[lane as usize] = u16::from_le_bytes(read_input(
                self.memory.l0a(),
                self.issue.parameters.xn,
                2 * self.a_element(u64::from(self.geometry.k_tiles), 16, m, k),
            )?);
            right[lane as usize] = u16::from_le_bytes(read_input(
                self.memory.l0b(),
                self.issue.parameters.xm,
                f16_b_address(0, u64::from(self.geometry.n_tiles), k, n),
            )?);
        }
        Ok((left, right))
    }

    fn read_float32<const N: usize>(
        &self,
        m: u64,
        n: u64,
        k_base: u64,
    ) -> Result<([u32; N], [u32; N]), C220LocalBufferError> {
        let mut left = [0; N];
        let mut right = [0; N];
        let a_k_tiles = if self.issue.parameters.xt_bit_58 {
            2 * u64::from(self.issue.parameters.effective_k.div_ceil(16))
        } else {
            u64::from(self.geometry.k_tiles)
        };
        for lane in 0..self.active_lanes(k_base, N as u64) {
            let k = k_base + lane;
            left[lane as usize] = u32::from_le_bytes(read_input(
                self.memory.l0a(),
                self.issue.parameters.xn,
                4 * self.a_element(a_k_tiles, 8, m, k),
            )?);
            right[lane as usize] = u32::from_le_bytes(read_input(
                self.memory.l0b(),
                self.issue.parameters.xm,
                f32_b_address(0, u64::from(self.geometry.n_tiles), k, n),
            )?);
        }
        Ok((left, right))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IntegerInputMode {
    UnsignedUnsigned,
    SignedSigned,
    Signed4Signed4,
    UnsignedSigned,
}

impl IntegerInputMode {
    fn read(
        &self,
        buffer: &C220LocalBuffer,
        base: u64,
        element: u64,
    ) -> Result<u8, C220LocalBufferError> {
        if *self == Self::Signed4Signed4 {
            let [packed] = read_input(buffer, base, element / 2)?;
            let nibble = (packed >> (4 * (element & 1))) & 0xf;
            Ok(((nibble << 4) as i8 >> 4) as u8)
        } else {
            Ok(read_input::<1>(buffer, base, element)?[0])
        }
    }

    fn product(self, left: u8, right: u8) -> i64 {
        match self {
            Self::UnsignedUnsigned => i64::from(left) * i64::from(right),
            Self::SignedSigned | Self::Signed4Signed4 => {
                i64::from(left as i8) * i64::from(right as i8)
            }
            Self::UnsignedSigned => i64::from(left) * i64::from(right as i8),
        }
    }
}
