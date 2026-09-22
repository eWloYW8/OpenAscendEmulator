use std::cmp::Ordering;

use thiserror::Error;

use crate::isa::c220::cube::{C220CubeDataType, C220CubeOperation};
use crate::sim::c220::cube::C220CubeIssue;
use crate::sim::c220::fp16::C220Fp16Mode;
use crate::sim::c220::memory::{C220LocalBuffer, C220LocalBufferError, C220LocalMemory};

const TILE_EDGE: u64 = 16;
const F16_TILE_BYTES: u64 = 512;
const F32_INPUT_K_TILE: u64 = 8;
const F32_INPUT_TILE_BYTES: u64 = 512;
const F32_TILE_BYTES: u64 = 1024;
const F16_SIGN: u16 = 0x8000;
const F16_EXPONENT: u16 = 0x7c00;
const F16_FRACTION: u16 = 0x03ff;
const F16_MAX_FINITE: u16 = 0x7bff;
const BF16_SIGN: u16 = 0x8000;
const BF16_EXPONENT: u16 = 0x7f80;
const BF16_FRACTION: u16 = 0x007f;
const BF16_MAX_FINITE: u16 = 0x7f7f;
const F32_SIGN: u32 = 0x8000_0000;
const F32_EXPONENT: u32 = 0x7f80_0000;
const F32_FRACTION: u32 = 0x007f_ffff;
const F32_MAX_FINITE: u32 = 0x7f7f_ffff;
const F32_CANONICAL_NAN: u32 = 0x7fff_ffff;
const EXACT_LIMBS: usize = 9;
const EXACT_SCALE_EXPONENT: i32 = -298;
const F32_MIN_NORMAL_BIT: u32 = 172;
const F32_RAW_EXPONENT_OFFSET: u32 = 171;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct C220CubeFpStatus {
    pub nan_operand: bool,
    pub infinity_operand: bool,
    pub invalid: bool,
    pub overflow: bool,
    pub underflow: bool,
}

impl C220CubeFpStatus {
    fn merge(&mut self, other: Self) {
        self.nan_operand |= other.nan_operand;
        self.infinity_operand |= other.infinity_operand;
        self.invalid |= other.invalid;
        self.overflow |= other.overflow;
        self.underflow |= other.underflow;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220CubeExecutionOutcome {
    pub m: u16,
    pub k: u16,
    pub n: u16,
    pub mac_count: u64,
    pub written_lanes: u64,
    pub fp_status: C220CubeFpStatus,
    pub integer_overflow: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220F32MmadMode {
    Fp32,
    Hf32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220CubeControl {
    pub fp16_mode: C220Fp16Mode,
    pub f32_mode: C220F32MmadMode,
    pub hf32_rounding: bool,
}

impl C220CubeControl {
    pub const fn from_spr3(value: u64) -> Self {
        Self {
            fp16_mode: C220Fp16Mode::from_control_spr(value),
            f32_mode: if value & (1 << 46) == 0 {
                C220F32MmadMode::Fp32
            } else {
                C220F32MmadMode::Hf32
            },
            hf32_rounding: value & (1 << 47) != 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220PreparedCubeExecution {
    writes: Vec<C220CubeWrite>,
    pub outcome: C220CubeExecutionOutcome,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum C220CubeWrite {
    U16 { address: u64, value: u16 },
    U32 { address: u64, value: u32 },
}

impl C220PreparedCubeExecution {
    pub fn write_count(&self) -> usize {
        self.writes.len()
    }

    pub fn commit(self, memory: &mut C220LocalMemory) -> Result<(), C220CubeExecutionError> {
        for write in self.writes {
            match write {
                C220CubeWrite::U16 { address, value } => {
                    write_u16_wrapped(memory.l0c_mut().buffer_mut(), address, value)?;
                }
                C220CubeWrite::U32 { address, value } => {
                    write_u32_wrapped(memory.l0c_mut().buffer_mut(), address, value)?;
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
        "functional MMAD does not support XT controls 44:50={bits_44_50:#x}, 55:56={bits_55_56:#x}, bit58={bit_58}"
    )]
    UnsupportedControl {
        bits_44_50: u8,
        bits_55_56: u8,
        bit_58: bool,
    },
    #[error("functional MMAD requires the high half of XD to be zero, got {0:#x}")]
    UnsupportedXdHigh(u32),
    #[error("exact C220 Cube accumulator exceeded its internal 320-bit range")]
    ExactAccumulatorOverflow,
    #[error(transparent)]
    LocalBuffer(#[from] C220LocalBufferError),
}

impl C220CubeIssue {
    pub fn prepare(
        self,
        memory: &C220LocalMemory,
        control: C220CubeControl,
    ) -> Result<C220PreparedCubeExecution, C220CubeExecutionError> {
        self.validate_functional_mode()?;
        match self.instruction.data_type {
            C220CubeDataType::F16F16 => self.prepare_float16_f16_impl(memory, control.fp16_mode),
            C220CubeDataType::F16F32 => {
                self.prepare_float16_f32_impl(memory, control.fp16_mode, Float16InputFormat::Fp16)
            }
            C220CubeDataType::Bf16F32 => {
                self.prepare_float16_f32_impl(memory, control.fp16_mode, Float16InputFormat::Bf16)
            }
            C220CubeDataType::F32F32 if control.f32_mode == C220F32MmadMode::Fp32 => {
                self.prepare_float32_f32_impl(memory, control.fp16_mode)
            }
            C220CubeDataType::F32F32 => {
                self.prepare_hf32_f32_impl(memory, control.fp16_mode, control.hf32_rounding)
            }
            C220CubeDataType::U8U8S32 => {
                self.prepare_integer_impl(memory, IntegerInputMode::UnsignedUnsigned)
            }
            C220CubeDataType::S8S8S32 => {
                self.prepare_integer_impl(memory, IntegerInputMode::SignedSigned)
            }
            C220CubeDataType::S4S4S32 => {
                self.prepare_integer_impl(memory, IntegerInputMode::SignedSigned)
            }
            C220CubeDataType::U8S8S32 => {
                self.prepare_integer_impl(memory, IntegerInputMode::UnsignedSigned)
            }
            data_type => Err(C220CubeExecutionError::UnsupportedDataType(data_type)),
        }
    }

    pub fn execute(
        self,
        memory: &mut C220LocalMemory,
        control: C220CubeControl,
    ) -> Result<C220CubeExecutionOutcome, C220CubeExecutionError> {
        let prepared = self.prepare(memory, control)?;
        let outcome = prepared.outcome;
        prepared.commit(memory)?;
        Ok(outcome)
    }

    fn validate_functional_mode(self) -> Result<(), C220CubeExecutionError> {
        if self.instruction.operation != C220CubeOperation::Mmad {
            return Err(C220CubeExecutionError::UnsupportedOperation(
                self.instruction.operation,
            ));
        }
        if self.parameters.xt_bits_44_50 != 0
            || self.parameters.xt_bits_55_56 != 0
            || self.parameters.xt_bit_58
        {
            return Err(C220CubeExecutionError::UnsupportedControl {
                bits_44_50: self.parameters.xt_bits_44_50,
                bits_55_56: self.parameters.xt_bits_55_56,
                bit_58: self.parameters.xt_bit_58,
            });
        }
        if self.parameters.xd_high != 0 {
            return Err(C220CubeExecutionError::UnsupportedXdHigh(
                self.parameters.xd_high,
            ));
        }
        Ok(())
    }

    fn prepare_float32_f32_impl(
        self,
        memory: &C220LocalMemory,
        fp_mode: C220Fp16Mode,
    ) -> Result<C220PreparedCubeExecution, C220CubeExecutionError> {
        let geometry = self.instruction.geometry(self.parameters);
        let mut fp_status = C220CubeFpStatus::default();
        let mut written_lanes = 0_u64;
        let mut writes = Vec::new();
        for m in 0..u64::from(self.parameters.m) {
            for n in 0..u64::from(self.parameters.n) {
                let c_address = f32_c_address(
                    u64::from(self.parameters.xd_low),
                    u64::from(geometry.m_tiles),
                    m,
                    n,
                );
                let mut accumulator = if self.parameters.xt_bit_63 || self.parameters.xt_bit_62 {
                    0
                } else {
                    read_u32_wrapped(memory.l0c().buffer(), c_address)?
                };

                for k_base in (0..u64::from(self.parameters.effective_k)).step_by(4) {
                    let mut left = [0_u32; 4];
                    let mut right = [0_u32; 4];
                    let active = (u64::from(self.parameters.effective_k) - k_base).min(4);
                    for lane in 0..active {
                        let k = k_base + lane;
                        left[lane as usize] = read_u32_wrapped(
                            memory.l0a(),
                            f32_a_address(self.parameters.xn, u64::from(geometry.k_tiles), m, k),
                        )?;
                        right[lane as usize] = read_u32_wrapped(
                            memory.l0b(),
                            f32_b_address(self.parameters.xm, u64::from(geometry.n_tiles), k, n),
                        )?;
                    }
                    let slice = evaluate_f32_f32_slice(left, right, accumulator, fp_mode)?;
                    accumulator = slice.bits;
                    fp_status.merge(slice.status);
                }

                writes.push(C220CubeWrite::U32 {
                    address: c_address,
                    value: accumulator,
                });
                written_lanes += 1;
            }
        }
        Ok(C220PreparedCubeExecution {
            writes,
            outcome: C220CubeExecutionOutcome {
                m: self.parameters.m,
                k: self.parameters.effective_k,
                n: self.parameters.n,
                mac_count: u64::from(self.parameters.m)
                    * u64::from(self.parameters.effective_k)
                    * u64::from(self.parameters.n),
                written_lanes,
                fp_status,
                integer_overflow: false,
            },
        })
    }

    fn prepare_hf32_f32_impl(
        self,
        memory: &C220LocalMemory,
        fp_mode: C220Fp16Mode,
        round_ties_away: bool,
    ) -> Result<C220PreparedCubeExecution, C220CubeExecutionError> {
        let geometry = self.instruction.geometry(self.parameters);
        let mut fp_status = C220CubeFpStatus::default();
        let mut written_lanes = 0_u64;
        let mut writes = Vec::new();
        for m in 0..u64::from(self.parameters.m) {
            for n in 0..u64::from(self.parameters.n) {
                let c_address = f32_c_address(
                    u64::from(self.parameters.xd_low),
                    u64::from(geometry.m_tiles),
                    m,
                    n,
                );
                let mut accumulator = if self.parameters.xt_bit_63 || self.parameters.xt_bit_62 {
                    0
                } else {
                    read_u32_wrapped(memory.l0c().buffer(), c_address)?
                };

                for k_base in (0..u64::from(self.parameters.effective_k)).step_by(8) {
                    let mut left = [0_u32; 8];
                    let mut right = [0_u32; 8];
                    let active = (u64::from(self.parameters.effective_k) - k_base).min(8);
                    for lane in 0..active {
                        let k = k_base + lane;
                        left[lane as usize] = read_u32_wrapped(
                            memory.l0a(),
                            f32_a_address(self.parameters.xn, u64::from(geometry.k_tiles), m, k),
                        )?;
                        right[lane as usize] = read_u32_wrapped(
                            memory.l0b(),
                            f32_b_address(self.parameters.xm, u64::from(geometry.n_tiles), k, n),
                        )?;
                    }
                    let slice = evaluate_hf32_f32_slice(
                        left,
                        right,
                        accumulator,
                        fp_mode,
                        round_ties_away,
                    )?;
                    accumulator = slice.bits;
                    fp_status.merge(slice.status);
                }

                writes.push(C220CubeWrite::U32 {
                    address: c_address,
                    value: accumulator,
                });
                written_lanes += 1;
            }
        }
        Ok(C220PreparedCubeExecution {
            writes,
            outcome: C220CubeExecutionOutcome {
                m: self.parameters.m,
                k: self.parameters.effective_k,
                n: self.parameters.n,
                mac_count: u64::from(self.parameters.m)
                    * u64::from(self.parameters.effective_k)
                    * u64::from(self.parameters.n),
                written_lanes,
                fp_status,
                integer_overflow: false,
            },
        })
    }

    fn prepare_float16_f32_impl(
        self,
        memory: &C220LocalMemory,
        fp_mode: C220Fp16Mode,
        input_format: Float16InputFormat,
    ) -> Result<C220PreparedCubeExecution, C220CubeExecutionError> {
        let geometry = self.instruction.geometry(self.parameters);
        let mut fp_status = C220CubeFpStatus::default();
        let mut written_lanes = 0_u64;
        let mut writes = Vec::new();
        for m in 0..u64::from(self.parameters.m) {
            for n in 0..u64::from(self.parameters.n) {
                let c_address = f32_c_address(
                    u64::from(self.parameters.xd_low),
                    u64::from(geometry.m_tiles),
                    m,
                    n,
                );
                let mut accumulator = if self.parameters.xt_bit_63 || self.parameters.xt_bit_62 {
                    0
                } else {
                    read_u32_wrapped(memory.l0c().buffer(), c_address)?
                };

                for k_base in (0..u64::from(self.parameters.effective_k)).step_by(16) {
                    let mut left = [0_u16; 16];
                    let mut right = [0_u16; 16];
                    let active = (u64::from(self.parameters.effective_k) - k_base).min(TILE_EDGE);
                    for lane in 0..active {
                        let k = k_base + lane;
                        left[lane as usize] = read_u16_wrapped(
                            memory.l0a(),
                            f16_a_address(self.parameters.xn, u64::from(geometry.k_tiles), m, k),
                        )?;
                        right[lane as usize] = read_u16_wrapped(
                            memory.l0b(),
                            f16_b_address(self.parameters.xm, u64::from(geometry.n_tiles), k, n),
                        )?;
                    }
                    let slice = input_format.evaluate_slice(left, right, accumulator, fp_mode)?;
                    accumulator = slice.bits;
                    fp_status.merge(slice.status);
                }

                writes.push(C220CubeWrite::U32 {
                    address: c_address,
                    value: accumulator,
                });
                written_lanes += 1;
            }
        }
        Ok(C220PreparedCubeExecution {
            writes,
            outcome: C220CubeExecutionOutcome {
                m: self.parameters.m,
                k: self.parameters.effective_k,
                n: self.parameters.n,
                mac_count: u64::from(self.parameters.m)
                    * u64::from(self.parameters.effective_k)
                    * u64::from(self.parameters.n),
                written_lanes,
                fp_status,
                integer_overflow: false,
            },
        })
    }

    fn prepare_float16_f16_impl(
        self,
        memory: &C220LocalMemory,
        fp_mode: C220Fp16Mode,
    ) -> Result<C220PreparedCubeExecution, C220CubeExecutionError> {
        let geometry = self.instruction.geometry(self.parameters);
        let mut fp_status = C220CubeFpStatus::default();
        let mut written_lanes = 0_u64;
        let mut writes = Vec::new();
        for m in 0..u64::from(self.parameters.m) {
            for n in 0..u64::from(self.parameters.n) {
                let c_address = f16_c_address(
                    u64::from(self.parameters.xd_low),
                    u64::from(geometry.m_tiles),
                    m,
                    n,
                );
                let mut accumulator = if self.parameters.xt_bit_63 || self.parameters.xt_bit_62 {
                    0
                } else {
                    read_u16_wrapped(memory.l0c().buffer(), c_address)?
                };

                for k_base in (0..u64::from(self.parameters.effective_k)).step_by(16) {
                    let mut left = [0_u16; 16];
                    let mut right = [0_u16; 16];
                    let active = (u64::from(self.parameters.effective_k) - k_base).min(TILE_EDGE);
                    for lane in 0..active {
                        let k = k_base + lane;
                        left[lane as usize] = read_u16_wrapped(
                            memory.l0a(),
                            f16_a_address(self.parameters.xn, u64::from(geometry.k_tiles), m, k),
                        )?;
                        right[lane as usize] = read_u16_wrapped(
                            memory.l0b(),
                            f16_b_address(self.parameters.xm, u64::from(geometry.n_tiles), k, n),
                        )?;
                    }
                    let slice = evaluate_f16_f16_slice(left, right, accumulator, fp_mode)?;
                    accumulator = slice.bits;
                    fp_status.merge(slice.status);
                }

                writes.push(C220CubeWrite::U16 {
                    address: c_address,
                    value: accumulator,
                });
                written_lanes += 1;
            }
        }
        Ok(C220PreparedCubeExecution {
            writes,
            outcome: C220CubeExecutionOutcome {
                m: self.parameters.m,
                k: self.parameters.effective_k,
                n: self.parameters.n,
                mac_count: u64::from(self.parameters.m)
                    * u64::from(self.parameters.effective_k)
                    * u64::from(self.parameters.n),
                written_lanes,
                fp_status,
                integer_overflow: false,
            },
        })
    }

    fn prepare_integer_impl(
        self,
        memory: &C220LocalMemory,
        input_mode: IntegerInputMode,
    ) -> Result<C220PreparedCubeExecution, C220CubeExecutionError> {
        let geometry = self.instruction.geometry(self.parameters);
        let k_tile_elements = u64::from(geometry.k_tile_elements);
        let mut integer_overflow = false;
        let mut written_lanes = 0_u64;
        let mut writes = Vec::new();
        for m in 0..u64::from(self.parameters.m) {
            for n in 0..u64::from(self.parameters.n) {
                let c_address = f32_c_address(
                    u64::from(self.parameters.xd_low),
                    u64::from(geometry.m_tiles),
                    m,
                    n,
                );
                let mut accumulator = if self.parameters.xt_bit_63 || self.parameters.xt_bit_62 {
                    0_i32
                } else {
                    read_u32_wrapped(memory.l0c().buffer(), c_address)? as i32
                };
                for k_base in
                    (0..u64::from(self.parameters.effective_k)).step_by(k_tile_elements as usize)
                {
                    let active =
                        (u64::from(self.parameters.effective_k) - k_base).min(k_tile_elements);
                    let mut slice = i64::from(accumulator);
                    for lane in 0..active {
                        let k = k_base + lane;
                        let left = read_u8_wrapped(
                            memory.l0a(),
                            integer_a_address(
                                self.parameters.xn,
                                u64::from(geometry.k_tiles),
                                k_tile_elements,
                                m,
                                k,
                            ),
                        )?;
                        let right = read_u8_wrapped(
                            memory.l0b(),
                            integer_b_address(
                                self.parameters.xm,
                                u64::from(geometry.n_tiles),
                                k_tile_elements,
                                k,
                                n,
                            ),
                        )?;
                        slice += input_mode.product(left, right);
                    }
                    let saturated = slice.clamp(i64::from(i32::MIN), i64::from(i32::MAX));
                    integer_overflow |= saturated != slice;
                    accumulator = saturated as i32;
                }
                writes.push(C220CubeWrite::U32 {
                    address: c_address,
                    value: accumulator as u32,
                });
                written_lanes += 1;
            }
        }
        Ok(C220PreparedCubeExecution {
            writes,
            outcome: C220CubeExecutionOutcome {
                m: self.parameters.m,
                k: self.parameters.effective_k,
                n: self.parameters.n,
                mac_count: u64::from(self.parameters.m)
                    * u64::from(self.parameters.effective_k)
                    * u64::from(self.parameters.n),
                written_lanes,
                fp_status: C220CubeFpStatus::default(),
                integer_overflow,
            },
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IntegerInputMode {
    UnsignedUnsigned,
    SignedSigned,
    UnsignedSigned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Float16InputFormat {
    Fp16,
    Bf16,
}

impl Float16InputFormat {
    fn evaluate_slice(
        self,
        left: [u16; 16],
        right: [u16; 16],
        accumulator: u32,
        mode: C220Fp16Mode,
    ) -> Result<F32SliceOutcome, C220CubeExecutionError> {
        match self {
            Self::Fp16 => evaluate_f16_f32_slice(left, right, accumulator, mode),
            Self::Bf16 => evaluate_bf16_f32_slice(left, right, accumulator, mode),
        }
    }
}

impl IntegerInputMode {
    fn product(self, left: u8, right: u8) -> i64 {
        match self {
            Self::UnsignedUnsigned => i64::from(left) * i64::from(right),
            Self::SignedSigned => i64::from(left as i8) * i64::from(right as i8),
            Self::UnsignedSigned => i64::from(left) * i64::from(right as i8),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct F32SliceOutcome {
    bits: u32,
    status: C220CubeFpStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct F16SliceOutcome {
    bits: u16,
    status: C220CubeFpStatus,
}

fn evaluate_f32_f32_slice(
    mut left: [u32; 4],
    mut right: [u32; 4],
    mut accumulator: u32,
    mode: C220Fp16Mode,
) -> Result<F32SliceOutcome, C220CubeExecutionError> {
    let mut status = C220CubeFpStatus::default();
    if mode == C220Fp16Mode::Saturating {
        for bits in left.iter_mut().chain(right.iter_mut()) {
            status.nan_operand |= is_f32_nan(*bits);
            status.infinity_operand |= is_f32_infinite(*bits);
            *bits = saturate_f32_nonfinite(*bits);
        }
        status.nan_operand |= is_f32_nan(accumulator);
        status.infinity_operand |= is_f32_infinite(accumulator);
        accumulator = saturate_f32_nonfinite(accumulator);
    } else {
        let mut positive_infinity = false;
        let mut negative_infinity = false;
        if is_f32_nan(accumulator) {
            status.nan_operand = true;
            return Ok(F32SliceOutcome {
                bits: F32_CANONICAL_NAN,
                status,
            });
        }
        if is_f32_infinite(accumulator) {
            status.infinity_operand = true;
            if accumulator & F32_SIGN == 0 {
                positive_infinity = true;
            } else {
                negative_infinity = true;
            }
        }
        for (&first, &second) in left.iter().zip(&right) {
            let first_nan = is_f32_nan(first);
            let second_nan = is_f32_nan(second);
            let first_infinite = is_f32_infinite(first);
            let second_infinite = is_f32_infinite(second);
            status.nan_operand |= first_nan || second_nan;
            status.infinity_operand |= first_infinite || second_infinite;
            let invalid = first_nan
                || second_nan
                || (first_infinite && is_f32_zero(second))
                || (second_infinite && is_f32_zero(first));
            status.invalid |= invalid;
            if invalid {
                return Ok(F32SliceOutcome {
                    bits: F32_CANONICAL_NAN,
                    status,
                });
            }
            if first_infinite || second_infinite {
                if (first ^ second) & F32_SIGN == 0 {
                    positive_infinity = true;
                } else {
                    negative_infinity = true;
                }
            }
        }
        if positive_infinity && negative_infinity {
            status.invalid = true;
            return Ok(F32SliceOutcome {
                bits: F32_CANONICAL_NAN,
                status,
            });
        }
        if positive_infinity || negative_infinity {
            return Ok(F32SliceOutcome {
                bits: if negative_infinity {
                    F32_SIGN | F32_EXPONENT
                } else {
                    F32_EXPONENT
                },
                status,
            });
        }
    }

    let mut sum = ExactDyadic::default();
    let accumulator_contributes = accumulator & F32_EXPONENT != 0;
    let mut all_zero_terms_negative = !accumulator_contributes && accumulator & F32_SIGN != 0;
    sum.add_f32(accumulator)?;
    for (&first, &second) in left.iter().zip(&right) {
        if is_f32_infinite(first)
            || is_f32_nan(first)
            || is_f32_infinite(second)
            || is_f32_nan(second)
        {
            unreachable!("nonfinite operands were handled or saturated");
        }
        let product_nonzero = !is_f32_zero(first) && !is_f32_zero(second);
        if !product_nonzero {
            all_zero_terms_negative &= (first ^ second) & F32_SIGN != 0;
        }
        sum.add_f32_product(first, second)?;
    }
    let (bits, rounded_status) = sum.round_f32(
        !accumulator_contributes
            && left
                .iter()
                .zip(&right)
                .all(|(&first, &second)| is_f32_zero(first) || is_f32_zero(second))
            && all_zero_terms_negative,
    );
    status.merge(rounded_status);
    Ok(F32SliceOutcome { bits, status })
}

fn evaluate_hf32_f32_slice(
    mut left: [u32; 8],
    mut right: [u32; 8],
    mut accumulator: u32,
    mode: C220Fp16Mode,
    round_ties_away: bool,
) -> Result<F32SliceOutcome, C220CubeExecutionError> {
    let mut status = C220CubeFpStatus::default();
    if mode == C220Fp16Mode::Saturating {
        for bits in left.iter_mut().chain(right.iter_mut()) {
            status.nan_operand |= is_f32_nan(*bits);
            status.infinity_operand |= is_f32_infinite(*bits);
            *bits = saturate_f32_nonfinite(*bits);
            *bits = round_hf32_operand(*bits, round_ties_away);
            *bits = saturate_hf32_nonfinite(*bits);
        }
        status.nan_operand |= is_f32_nan(accumulator);
        status.infinity_operand |= is_f32_infinite(accumulator);
        accumulator = saturate_f32_nonfinite(accumulator);
    } else {
        let mut positive_infinity = false;
        let mut negative_infinity = false;
        if is_f32_nan(accumulator) {
            status.nan_operand = true;
            return Ok(F32SliceOutcome {
                bits: F32_CANONICAL_NAN,
                status,
            });
        }
        if is_f32_infinite(accumulator) {
            status.infinity_operand = true;
            if accumulator & F32_SIGN == 0 {
                positive_infinity = true;
            } else {
                negative_infinity = true;
            }
        }
        for (first, second) in left.iter_mut().zip(right.iter_mut()) {
            let first_nan = is_f32_nan(*first);
            let second_nan = is_f32_nan(*second);
            let first_infinite = is_f32_infinite(*first);
            let second_infinite = is_f32_infinite(*second);
            status.nan_operand |= first_nan || second_nan;
            status.infinity_operand |= first_infinite || second_infinite;
            let invalid = first_nan
                || second_nan
                || (first_infinite && is_f32_zero(*second))
                || (second_infinite && is_f32_zero(*first));
            status.invalid |= invalid;
            if invalid {
                return Ok(F32SliceOutcome {
                    bits: F32_CANONICAL_NAN,
                    status,
                });
            }
            if first_infinite || second_infinite {
                if (*first ^ *second) & F32_SIGN == 0 {
                    positive_infinity = true;
                } else {
                    negative_infinity = true;
                }
            }
            *first = round_hf32_operand(*first, round_ties_away);
            *second = round_hf32_operand(*second, round_ties_away);
        }
        if positive_infinity && negative_infinity {
            status.invalid = true;
            return Ok(F32SliceOutcome {
                bits: F32_CANONICAL_NAN,
                status,
            });
        }
        if positive_infinity || negative_infinity {
            return Ok(F32SliceOutcome {
                bits: if negative_infinity {
                    F32_SIGN | F32_EXPONENT
                } else {
                    F32_EXPONENT
                },
                status,
            });
        }
    }

    let mut sum = ExactDyadic::default();
    let accumulator_contributes = accumulator & F32_EXPONENT != 0;
    let mut all_zero_terms_negative = !accumulator_contributes && accumulator & F32_SIGN != 0;
    sum.add_f32(accumulator)?;
    for (&first, &second) in left.iter().zip(&right) {
        if is_f32_infinite(first)
            || is_f32_nan(first)
            || is_f32_infinite(second)
            || is_f32_nan(second)
        {
            unreachable!("nonfinite operands were handled or saturated");
        }
        let product_nonzero = !is_f32_zero(first) && !is_f32_zero(second);
        if !product_nonzero {
            all_zero_terms_negative &= (first ^ second) & F32_SIGN != 0;
        }
        sum.add_f32_product(first, second)?;
    }
    let (bits, rounded_status) = sum.round_f32(
        !accumulator_contributes
            && left
                .iter()
                .zip(&right)
                .all(|(&first, &second)| is_f32_zero(first) || is_f32_zero(second))
            && all_zero_terms_negative,
    );
    status.merge(rounded_status);
    Ok(F32SliceOutcome { bits, status })
}

fn round_hf32_operand(bits: u32, round_ties_away: bool) -> u32 {
    let sign = bits & F32_SIGN;
    let mut exponent = (bits & F32_EXPONENT) >> 23;
    let mut significand = bits & F32_FRACTION;
    if exponent != 0 {
        significand |= 1 << 23;
    }
    let retained = significand >> 12;
    let discarded = significand & 0x0fff;
    let increment =
        discarded > 0x0800 || (discarded == 0x0800 && (round_ties_away || retained & 1 != 0));
    let rounded = retained + u32::from(increment);
    if rounded == 1 << 12 {
        exponent = exponent.wrapping_add(1);
        return sign | (exponent << 23);
    }
    let fraction = (rounded << 12) & F32_FRACTION;
    if exponent == 0 && rounded == 1 << 11 {
        sign | 1 << 23
    } else {
        sign | (exponent << 23) | fraction
    }
}

fn evaluate_f16_f16_slice(
    mut left: [u16; 16],
    mut right: [u16; 16],
    mut accumulator: u16,
    mode: C220Fp16Mode,
) -> Result<F16SliceOutcome, C220CubeExecutionError> {
    let mut status = C220CubeFpStatus::default();
    if mode == C220Fp16Mode::Saturating {
        for bits in left.iter_mut().chain(right.iter_mut()) {
            status.nan_operand |= is_f16_nan(*bits);
            status.infinity_operand |= is_f16_infinite(*bits);
            *bits = saturate_f16_nonfinite(*bits);
        }
        status.nan_operand |= is_f16_nan(accumulator);
        status.infinity_operand |= is_f16_infinite(accumulator);
        accumulator = saturate_f16_nonfinite(accumulator);
    } else {
        let mut positive_infinity = false;
        let mut negative_infinity = false;
        if is_f16_nan(accumulator) {
            status.nan_operand = true;
            return Ok(F16SliceOutcome {
                bits: 0x7fff,
                status,
            });
        }
        if is_f16_infinite(accumulator) {
            status.infinity_operand = true;
            if accumulator & F16_SIGN == 0 {
                positive_infinity = true;
            } else {
                negative_infinity = true;
            }
        }
        for (&first, &second) in left.iter().zip(&right) {
            let first_nan = is_f16_nan(first);
            let second_nan = is_f16_nan(second);
            let first_infinite = is_f16_infinite(first);
            let second_infinite = is_f16_infinite(second);
            status.nan_operand |= first_nan || second_nan;
            status.infinity_operand |= first_infinite || second_infinite;
            let invalid = first_nan
                || second_nan
                || (first_infinite && is_f16_zero(second))
                || (second_infinite && is_f16_zero(first));
            status.invalid |= invalid;
            if invalid {
                return Ok(F16SliceOutcome {
                    bits: 0x7fff,
                    status,
                });
            }
            if first_infinite || second_infinite {
                if (first ^ second) & F16_SIGN == 0 {
                    positive_infinity = true;
                } else {
                    negative_infinity = true;
                }
            }
        }
        if positive_infinity && negative_infinity {
            status.invalid = true;
            return Ok(F16SliceOutcome {
                bits: 0x7fff,
                status,
            });
        }
        if positive_infinity || negative_infinity {
            return Ok(F16SliceOutcome {
                bits: if negative_infinity {
                    F16_SIGN | F16_EXPONENT
                } else {
                    F16_EXPONENT
                },
                status,
            });
        }
    }

    let mut sum = ExactDyadic::default();
    let accumulator_contributes = accumulator & F16_EXPONENT != 0;
    let mut all_zero_terms_negative = !accumulator_contributes && accumulator & F16_SIGN != 0;
    sum.add_f16(accumulator)?;
    for (&first, &second) in left.iter().zip(&right) {
        if is_f16_infinite(first)
            || is_f16_nan(first)
            || is_f16_infinite(second)
            || is_f16_nan(second)
        {
            unreachable!("nonfinite operands were handled or saturated");
        }
        let product_nonzero = !is_f16_zero(first) && !is_f16_zero(second);
        if !product_nonzero {
            all_zero_terms_negative &= (first ^ second) & F16_SIGN != 0;
        }
        sum.add_f16_product(first, second)?;
    }
    let (bits, rounded_status) = sum.round_f16(
        !accumulator_contributes
            && left
                .iter()
                .zip(&right)
                .all(|(&first, &second)| is_f16_zero(first) || is_f16_zero(second))
            && all_zero_terms_negative,
        mode,
    );
    status.merge(rounded_status);
    Ok(F16SliceOutcome { bits, status })
}

fn evaluate_f16_f32_slice(
    mut left: [u16; 16],
    mut right: [u16; 16],
    mut accumulator: u32,
    mode: C220Fp16Mode,
) -> Result<F32SliceOutcome, C220CubeExecutionError> {
    let mut status = C220CubeFpStatus::default();
    if mode == C220Fp16Mode::Saturating {
        for bits in left.iter_mut().chain(right.iter_mut()) {
            status.nan_operand |= is_f16_nan(*bits);
            status.infinity_operand |= is_f16_infinite(*bits);
            *bits = saturate_f16_nonfinite(*bits);
        }
        status.nan_operand |= is_f32_nan(accumulator);
        status.infinity_operand |= is_f32_infinite(accumulator);
        accumulator = saturate_f32_nonfinite(accumulator);
    } else {
        let mut positive_infinity = false;
        let mut negative_infinity = false;
        if is_f32_nan(accumulator) {
            status.nan_operand = true;
            return Ok(F32SliceOutcome {
                bits: F32_CANONICAL_NAN,
                status,
            });
        }
        if is_f32_infinite(accumulator) {
            status.infinity_operand = true;
            if accumulator & F32_SIGN == 0 {
                positive_infinity = true;
            } else {
                negative_infinity = true;
            }
        }
        for (&first, &second) in left.iter().zip(&right) {
            let first_nan = is_f16_nan(first);
            let second_nan = is_f16_nan(second);
            let first_infinite = is_f16_infinite(first);
            let second_infinite = is_f16_infinite(second);
            status.nan_operand |= first_nan || second_nan;
            status.infinity_operand |= first_infinite || second_infinite;
            let invalid = first_nan
                || second_nan
                || (first_infinite && is_f16_zero(second))
                || (second_infinite && is_f16_zero(first));
            status.invalid |= invalid;
            if invalid {
                return Ok(F32SliceOutcome {
                    bits: F32_CANONICAL_NAN,
                    status,
                });
            }
            if first_infinite || second_infinite {
                if (first ^ second) & F16_SIGN == 0 {
                    positive_infinity = true;
                } else {
                    negative_infinity = true;
                }
            }
        }
        if positive_infinity && negative_infinity {
            status.invalid = true;
            return Ok(F32SliceOutcome {
                bits: F32_CANONICAL_NAN,
                status,
            });
        }
        if positive_infinity || negative_infinity {
            return Ok(F32SliceOutcome {
                bits: if negative_infinity {
                    F32_SIGN | F32_EXPONENT
                } else {
                    F32_EXPONENT
                },
                status,
            });
        }
    }

    let mut sum = ExactDyadic::default();
    let accumulator_contributes = accumulator & F32_EXPONENT != 0;
    let mut all_zero_terms_negative = !accumulator_contributes && accumulator & F32_SIGN != 0;
    sum.add_f32(accumulator)?;
    for (&first, &second) in left.iter().zip(&right) {
        if is_f16_infinite(first)
            || is_f16_nan(first)
            || is_f16_infinite(second)
            || is_f16_nan(second)
        {
            unreachable!("nonfinite operands were handled or saturated");
        }
        let product_nonzero = !is_f16_zero(first) && !is_f16_zero(second);
        if !product_nonzero {
            all_zero_terms_negative &= (first ^ second) & F16_SIGN != 0;
        }
        sum.add_f16_product(first, second)?;
    }
    let (bits, rounded_status) = sum.round_f32(
        !accumulator_contributes
            && left
                .iter()
                .zip(&right)
                .all(|(&first, &second)| is_f16_zero(first) || is_f16_zero(second))
            && all_zero_terms_negative,
    );
    status.merge(rounded_status);
    Ok(F32SliceOutcome { bits, status })
}

fn evaluate_bf16_f32_slice(
    mut left: [u16; 16],
    mut right: [u16; 16],
    mut accumulator: u32,
    mode: C220Fp16Mode,
) -> Result<F32SliceOutcome, C220CubeExecutionError> {
    let mut status = C220CubeFpStatus::default();
    if mode == C220Fp16Mode::Saturating {
        for bits in left.iter_mut().chain(right.iter_mut()) {
            status.nan_operand |= is_bf16_nan(*bits);
            status.infinity_operand |= is_bf16_infinite(*bits);
            *bits = saturate_bf16_nonfinite(*bits);
        }
        status.nan_operand |= is_f32_nan(accumulator);
        status.infinity_operand |= is_f32_infinite(accumulator);
        accumulator = saturate_f32_nonfinite(accumulator);
    } else {
        let mut positive_infinity = false;
        let mut negative_infinity = false;
        if is_f32_nan(accumulator) {
            status.nan_operand = true;
            return Ok(F32SliceOutcome {
                bits: F32_CANONICAL_NAN,
                status,
            });
        }
        if is_f32_infinite(accumulator) {
            status.infinity_operand = true;
            if accumulator & F32_SIGN == 0 {
                positive_infinity = true;
            } else {
                negative_infinity = true;
            }
        }
        for (&first, &second) in left.iter().zip(&right) {
            let first_nan = is_bf16_nan(first);
            let second_nan = is_bf16_nan(second);
            let first_infinite = is_bf16_infinite(first);
            let second_infinite = is_bf16_infinite(second);
            status.nan_operand |= first_nan || second_nan;
            status.infinity_operand |= first_infinite || second_infinite;
            let invalid = first_nan
                || second_nan
                || (first_infinite && is_bf16_zero(second))
                || (second_infinite && is_bf16_zero(first));
            status.invalid |= invalid;
            if invalid {
                return Ok(F32SliceOutcome {
                    bits: F32_CANONICAL_NAN,
                    status,
                });
            }
            if first_infinite || second_infinite {
                if (first ^ second) & BF16_SIGN == 0 {
                    positive_infinity = true;
                } else {
                    negative_infinity = true;
                }
            }
        }
        if positive_infinity && negative_infinity {
            status.invalid = true;
            return Ok(F32SliceOutcome {
                bits: F32_CANONICAL_NAN,
                status,
            });
        }
        if positive_infinity || negative_infinity {
            return Ok(F32SliceOutcome {
                bits: if negative_infinity {
                    F32_SIGN | F32_EXPONENT
                } else {
                    F32_EXPONENT
                },
                status,
            });
        }
    }

    let mut sum = ExactDyadic::default();
    let accumulator_contributes = accumulator & F32_EXPONENT != 0;
    let mut all_zero_terms_negative = !accumulator_contributes && accumulator & F32_SIGN != 0;
    sum.add_f32(accumulator)?;
    for (&first, &second) in left.iter().zip(&right) {
        if is_bf16_infinite(first)
            || is_bf16_nan(first)
            || is_bf16_infinite(second)
            || is_bf16_nan(second)
        {
            unreachable!("nonfinite operands were handled or saturated");
        }
        let product_nonzero = !is_bf16_zero(first) && !is_bf16_zero(second);
        if !product_nonzero {
            all_zero_terms_negative &= (first ^ second) & BF16_SIGN != 0;
        }
        sum.add_bf16_product(first, second)?;
    }
    let (bits, rounded_status) = sum.round_f32(
        !accumulator_contributes
            && left
                .iter()
                .zip(&right)
                .all(|(&first, &second)| is_bf16_zero(first) || is_bf16_zero(second))
            && all_zero_terms_negative,
    );
    status.merge(rounded_status);
    Ok(F32SliceOutcome { bits, status })
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct ExactDyadic {
    negative: bool,
    magnitude: [u64; EXACT_LIMBS],
}

impl ExactDyadic {
    fn add_f16(&mut self, bits: u16) -> Result<(), C220CubeExecutionError> {
        let exponent = (bits & F16_EXPONENT) >> 10;
        if exponent == 0 {
            return Ok(());
        }
        debug_assert_ne!(exponent, 0x1f);
        self.add_term(
            bits & F16_SIGN != 0,
            u64::from(0x400 | (bits & F16_FRACTION)),
            u32::from(exponent) + 241,
        )
    }

    fn add_f32(&mut self, bits: u32) -> Result<(), C220CubeExecutionError> {
        let exponent = (bits >> 23) & 0xff;
        let fraction = bits & F32_FRACTION;
        if exponent == 0 {
            return Ok(());
        }
        debug_assert_ne!(exponent, 0xff);
        self.add_term(
            bits & F32_SIGN != 0,
            u64::from((1 << 23) | fraction),
            exponent + 116,
        )
    }

    fn add_f16_product(&mut self, first: u16, second: u16) -> Result<(), C220CubeExecutionError> {
        let (first_significand, first_exponent) = finite_f16_parts(first);
        let (second_significand, second_exponent) = finite_f16_parts(second);
        if first_significand == 0 || second_significand == 0 {
            return Ok(());
        }
        let shift = first_exponent + second_exponent - EXACT_SCALE_EXPONENT;
        debug_assert!(shift >= 0);
        self.add_term(
            (first ^ second) & F16_SIGN != 0,
            u64::from(first_significand) * u64::from(second_significand),
            shift as u32,
        )
    }

    fn add_bf16_product(&mut self, first: u16, second: u16) -> Result<(), C220CubeExecutionError> {
        let (first_significand, first_exponent) = finite_bf16_parts(first);
        let (second_significand, second_exponent) = finite_bf16_parts(second);
        if first_significand == 0 || second_significand == 0 {
            return Ok(());
        }
        let shift = first_exponent + second_exponent - EXACT_SCALE_EXPONENT;
        self.add_term(
            (first ^ second) & BF16_SIGN != 0,
            u64::from(first_significand) * u64::from(second_significand),
            shift as u32,
        )
    }

    fn add_f32_product(&mut self, first: u32, second: u32) -> Result<(), C220CubeExecutionError> {
        let (first_significand, first_exponent) = finite_f32_parts(first);
        let (second_significand, second_exponent) = finite_f32_parts(second);
        if first_significand == 0 || second_significand == 0 {
            return Ok(());
        }
        let shift = first_exponent + second_exponent - EXACT_SCALE_EXPONENT;
        debug_assert!(shift >= 0);
        self.add_term(
            (first ^ second) & F32_SIGN != 0,
            u64::from(first_significand) * u64::from(second_significand),
            shift as u32,
        )
    }

    fn add_term(
        &mut self,
        negative: bool,
        significand: u64,
        shift: u32,
    ) -> Result<(), C220CubeExecutionError> {
        if significand == 0 {
            return Ok(());
        }
        let term = shifted(significand, shift)?;
        if self.is_zero() {
            self.negative = negative;
            self.magnitude = term;
            return Ok(());
        }
        if self.negative == negative {
            add_magnitudes(&mut self.magnitude, &term)?;
            return Ok(());
        }
        match compare_magnitudes(&self.magnitude, &term) {
            Ordering::Greater => subtract_magnitudes(&mut self.magnitude, &term),
            Ordering::Equal => {
                self.magnitude = [0; EXACT_LIMBS];
                self.negative = false;
            }
            Ordering::Less => {
                let mut result = term;
                subtract_magnitudes(&mut result, &self.magnitude);
                self.magnitude = result;
                self.negative = negative;
            }
        }
        Ok(())
    }

    fn round_f32(self, negative_zero: bool) -> (u32, C220CubeFpStatus) {
        let sign = if self.is_zero() {
            u32::from(negative_zero) << 31
        } else {
            u32::from(self.negative) << 31
        };
        let Some(high_bit) = highest_bit(&self.magnitude) else {
            return (sign, C220CubeFpStatus::default());
        };
        if high_bit < F32_MIN_NORMAL_BIT {
            const SUBNORMAL_QUANTUM_BIT: u32 = 149;
            let mut fraction = low_u64_after_shift(&self.magnitude, SUBNORMAL_QUANTUM_BIT) as u32;
            let half_bit = SUBNORMAL_QUANTUM_BIT - 1;
            let round = bit(&self.magnitude, half_bit)
                && (any_bits_below(&self.magnitude, half_bit) || fraction & 1 != 0);
            fraction += u32::from(round);
            if fraction == 1 << 23 {
                return (sign | 1 << 23, C220CubeFpStatus::default());
            }
            return (
                sign,
                C220CubeFpStatus {
                    underflow: true,
                    ..C220CubeFpStatus::default()
                },
            );
        }

        let discard = high_bit - 23;
        let mut significand = low_u64_after_shift(&self.magnitude, discard) as u32;
        if discard != 0 {
            let half_bit = discard - 1;
            let round = bit(&self.magnitude, half_bit)
                && (any_bits_below(&self.magnitude, half_bit) || significand & 1 != 0);
            significand += u32::from(round);
        }
        let mut rounded_high_bit = high_bit;
        if significand == 1 << 24 {
            significand >>= 1;
            rounded_high_bit += 1;
        }
        let raw_exponent = rounded_high_bit - F32_RAW_EXPONENT_OFFSET;
        if raw_exponent >= 255 {
            return (
                sign | F32_MAX_FINITE,
                C220CubeFpStatus {
                    overflow: true,
                    ..C220CubeFpStatus::default()
                },
            );
        }
        (
            sign | (raw_exponent << 23) | (significand & F32_FRACTION),
            C220CubeFpStatus::default(),
        )
    }

    fn round_f16(self, negative_zero: bool, mode: C220Fp16Mode) -> (u16, C220CubeFpStatus) {
        const MIN_NORMAL_BIT: u32 = 284;
        const SUBNORMAL_QUANTUM_BIT: u32 = 274;

        let sign = if self.is_zero() {
            u16::from(negative_zero) << 15
        } else {
            u16::from(self.negative) << 15
        };
        let Some(high_bit) = highest_bit(&self.magnitude) else {
            return (sign, C220CubeFpStatus::default());
        };
        if high_bit < MIN_NORMAL_BIT {
            let mut fraction = low_u64_after_shift(&self.magnitude, SUBNORMAL_QUANTUM_BIT) as u16;
            let half_bit = SUBNORMAL_QUANTUM_BIT - 1;
            let round = bit(&self.magnitude, half_bit)
                && (any_bits_below(&self.magnitude, half_bit) || fraction & 1 != 0);
            fraction += u16::from(round);
            if fraction == 0x400 {
                return (sign | 0x0400, C220CubeFpStatus::default());
            }
            return (
                sign,
                C220CubeFpStatus {
                    underflow: true,
                    ..C220CubeFpStatus::default()
                },
            );
        }

        let discard = high_bit - 10;
        let mut significand = low_u64_after_shift(&self.magnitude, discard) as u16;
        if discard != 0 {
            let half_bit = discard - 1;
            let round = bit(&self.magnitude, half_bit)
                && (any_bits_below(&self.magnitude, half_bit) || significand & 1 != 0);
            significand += u16::from(round);
        }
        let mut rounded_high_bit = high_bit;
        if significand == 1 << 11 {
            significand >>= 1;
            rounded_high_bit += 1;
        }
        let raw_exponent = rounded_high_bit - 283;
        if raw_exponent >= 31 {
            return (
                sign | match mode {
                    C220Fp16Mode::Saturating => F16_MAX_FINITE,
                    C220Fp16Mode::NonSaturating => F16_EXPONENT,
                },
                C220CubeFpStatus {
                    overflow: true,
                    ..C220CubeFpStatus::default()
                },
            );
        }
        (
            sign | ((raw_exponent as u16) << 10) | (significand & F16_FRACTION),
            C220CubeFpStatus::default(),
        )
    }

    fn is_zero(&self) -> bool {
        self.magnitude.iter().all(|&limb| limb == 0)
    }
}

fn finite_f16_parts(bits: u16) -> (u16, i32) {
    let exponent = i32::from((bits & F16_EXPONENT) >> 10);
    let fraction = bits & F16_FRACTION;
    if exponent == 0 {
        (fraction, -24)
    } else {
        (0x400 | fraction, exponent - 25)
    }
}

fn finite_bf16_parts(bits: u16) -> (u16, i32) {
    let exponent = i32::from((bits & BF16_EXPONENT) >> 7);
    let fraction = bits & BF16_FRACTION;
    if exponent == 0 {
        (fraction, -133)
    } else {
        (0x80 | fraction, exponent - 134)
    }
}

fn finite_f32_parts(bits: u32) -> (u32, i32) {
    let exponent = ((bits & F32_EXPONENT) >> 23) as i32;
    let fraction = bits & F32_FRACTION;
    if exponent == 0 {
        (fraction, -149)
    } else {
        ((1 << 23) | fraction, exponent - 150)
    }
}

fn shifted(significand: u64, shift: u32) -> Result<[u64; EXACT_LIMBS], C220CubeExecutionError> {
    let mut result = [0_u64; EXACT_LIMBS];
    let limb = (shift / 64) as usize;
    let bit = shift % 64;
    if limb >= EXACT_LIMBS {
        return Err(C220CubeExecutionError::ExactAccumulatorOverflow);
    }
    result[limb] = significand << bit;
    if bit != 0 {
        let high = significand >> (64 - bit);
        if high != 0 {
            let Some(destination) = result.get_mut(limb + 1) else {
                return Err(C220CubeExecutionError::ExactAccumulatorOverflow);
            };
            *destination = high;
        }
    }
    Ok(result)
}

fn add_magnitudes(
    destination: &mut [u64; EXACT_LIMBS],
    source: &[u64; EXACT_LIMBS],
) -> Result<(), C220CubeExecutionError> {
    let mut carry = false;
    for (destination, source) in destination.iter_mut().zip(source) {
        let (sum, first_carry) = destination.overflowing_add(*source);
        let (sum, second_carry) = sum.overflowing_add(u64::from(carry));
        *destination = sum;
        carry = first_carry || second_carry;
    }
    if carry {
        Err(C220CubeExecutionError::ExactAccumulatorOverflow)
    } else {
        Ok(())
    }
}

fn subtract_magnitudes(destination: &mut [u64; EXACT_LIMBS], source: &[u64; EXACT_LIMBS]) {
    let mut borrow = false;
    for (destination, source) in destination.iter_mut().zip(source) {
        let (difference, first_borrow) = destination.overflowing_sub(*source);
        let (difference, second_borrow) = difference.overflowing_sub(u64::from(borrow));
        *destination = difference;
        borrow = first_borrow || second_borrow;
    }
    debug_assert!(!borrow);
}

fn compare_magnitudes(first: &[u64; EXACT_LIMBS], second: &[u64; EXACT_LIMBS]) -> Ordering {
    first.iter().rev().cmp(second.iter().rev())
}

fn highest_bit(value: &[u64; EXACT_LIMBS]) -> Option<u32> {
    value.iter().enumerate().rev().find_map(|(index, &limb)| {
        (limb != 0).then(|| index as u32 * 64 + 63 - limb.leading_zeros())
    })
}

fn bit(value: &[u64; EXACT_LIMBS], index: u32) -> bool {
    value
        .get((index / 64) as usize)
        .is_some_and(|limb| limb & (1_u64 << (index % 64)) != 0)
}

fn any_bits_below(value: &[u64; EXACT_LIMBS], index: u32) -> bool {
    let limb_index = (index / 64) as usize;
    if value[..limb_index].iter().any(|&limb| limb != 0) {
        return true;
    }
    let within = index % 64;
    within != 0 && value[limb_index] & ((1_u64 << within) - 1) != 0
}

fn low_u64_after_shift(value: &[u64; EXACT_LIMBS], shift: u32) -> u64 {
    let limb = (shift / 64) as usize;
    let bit = shift % 64;
    let low = value.get(limb).copied().unwrap_or_default() >> bit;
    if bit == 0 {
        low
    } else {
        low | value.get(limb + 1).copied().unwrap_or_default() << (64 - bit)
    }
}

fn saturate_f16_nonfinite(bits: u16) -> u16 {
    if bits & F16_EXPONENT != F16_EXPONENT {
        return bits;
    }
    (bits & F16_SIGN)
        | if is_f16_infinite(bits) {
            F16_MAX_FINITE
        } else {
            0
        }
}

fn saturate_bf16_nonfinite(bits: u16) -> u16 {
    if bits & BF16_EXPONENT != BF16_EXPONENT {
        return bits;
    }
    (bits & BF16_SIGN)
        | if is_bf16_infinite(bits) {
            BF16_MAX_FINITE
        } else {
            0
        }
}

fn saturate_f32_nonfinite(bits: u32) -> u32 {
    if bits & F32_EXPONENT != F32_EXPONENT {
        return bits;
    }
    (bits & F32_SIGN)
        | if is_f32_infinite(bits) {
            F32_MAX_FINITE
        } else {
            0
        }
}

fn saturate_hf32_nonfinite(bits: u32) -> u32 {
    if bits & F32_EXPONENT != F32_EXPONENT {
        return bits;
    }
    (bits & F32_SIGN)
        | if is_f32_infinite(bits) {
            0x7f7f_f000
        } else {
            0
        }
}

const fn is_f16_zero(bits: u16) -> bool {
    bits & !F16_SIGN == 0
}

const fn is_f16_nan(bits: u16) -> bool {
    bits & F16_EXPONENT == F16_EXPONENT && bits & F16_FRACTION != 0
}

const fn is_f16_infinite(bits: u16) -> bool {
    bits & !F16_SIGN == F16_EXPONENT
}

const fn is_bf16_zero(bits: u16) -> bool {
    bits & !BF16_SIGN == 0
}

const fn is_bf16_nan(bits: u16) -> bool {
    bits & BF16_EXPONENT == BF16_EXPONENT && bits & BF16_FRACTION != 0
}

const fn is_bf16_infinite(bits: u16) -> bool {
    bits & !BF16_SIGN == BF16_EXPONENT
}

const fn is_f32_nan(bits: u32) -> bool {
    bits & F32_EXPONENT == F32_EXPONENT && bits & F32_FRACTION != 0
}

const fn is_f32_infinite(bits: u32) -> bool {
    bits & !F32_SIGN == F32_EXPONENT
}

const fn is_f32_zero(bits: u32) -> bool {
    bits & !F32_SIGN == 0
}

fn f16_a_address(base: u64, k_tiles: u64, m: u64, k: u64) -> u64 {
    let tile = (m / TILE_EDGE) * k_tiles + k / TILE_EDGE;
    let lane = (m % TILE_EDGE) * TILE_EDGE + k % TILE_EDGE;
    base.wrapping_add(tile * F16_TILE_BYTES + lane * 2)
}

fn f16_b_address(base: u64, n_tiles: u64, k: u64, n: u64) -> u64 {
    let tile = (k / TILE_EDGE) * n_tiles + n / TILE_EDGE;
    let lane = (n % TILE_EDGE) * TILE_EDGE + k % TILE_EDGE;
    base.wrapping_add(tile * F16_TILE_BYTES + lane * 2)
}

fn f32_a_address(base: u64, k_tiles: u64, m: u64, k: u64) -> u64 {
    let tile = (m / TILE_EDGE) * k_tiles + k / F32_INPUT_K_TILE;
    let lane = (m % TILE_EDGE) * F32_INPUT_K_TILE + k % F32_INPUT_K_TILE;
    base.wrapping_add(tile * F32_INPUT_TILE_BYTES + lane * 4)
}

fn f32_b_address(base: u64, n_tiles: u64, k: u64, n: u64) -> u64 {
    let tile = (k / F32_INPUT_K_TILE) * n_tiles + n / TILE_EDGE;
    let lane = (n % TILE_EDGE) * F32_INPUT_K_TILE + k % F32_INPUT_K_TILE;
    base.wrapping_add(tile * F32_INPUT_TILE_BYTES + lane * 4)
}

fn integer_a_address(base: u64, k_tiles: u64, k_tile: u64, m: u64, k: u64) -> u64 {
    let tile = (m / TILE_EDGE) * k_tiles + k / k_tile;
    let lane = (m % TILE_EDGE) * k_tile + k % k_tile;
    base.wrapping_add(tile * TILE_EDGE * k_tile + lane)
}

fn integer_b_address(base: u64, n_tiles: u64, k_tile: u64, k: u64, n: u64) -> u64 {
    let tile = (k / k_tile) * n_tiles + n / TILE_EDGE;
    let lane = (n % TILE_EDGE) * k_tile + k % k_tile;
    base.wrapping_add(tile * TILE_EDGE * k_tile + lane)
}

fn f32_c_address(base: u64, m_tiles: u64, m: u64, n: u64) -> u64 {
    let tile = m / TILE_EDGE + m_tiles * (n / TILE_EDGE);
    let lane = (m % TILE_EDGE) * TILE_EDGE + n % TILE_EDGE;
    base.wrapping_add(tile * F32_TILE_BYTES + lane * 4)
}

fn f16_c_address(base: u64, m_tiles: u64, m: u64, n: u64) -> u64 {
    let tile = m / TILE_EDGE + m_tiles * (n / TILE_EDGE);
    let lane = (m % TILE_EDGE) * TILE_EDGE + n % TILE_EDGE;
    base.wrapping_add(tile * F16_TILE_BYTES + lane * 2)
}

fn read_u16_wrapped(buffer: &C220LocalBuffer, address: u64) -> Result<u16, C220LocalBufferError> {
    let bytes = buffer.read_known_wrapped(address, 2)?;
    Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
}

fn read_u8_wrapped(buffer: &C220LocalBuffer, address: u64) -> Result<u8, C220LocalBufferError> {
    Ok(buffer.read_known_wrapped(address, 1)?[0])
}

fn read_u32_wrapped(buffer: &C220LocalBuffer, address: u64) -> Result<u32, C220LocalBufferError> {
    let bytes = buffer.read_known_wrapped(address, 4)?;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn write_u32_wrapped(
    buffer: &mut C220LocalBuffer,
    address: u64,
    value: u32,
) -> Result<(), C220LocalBufferError> {
    buffer.write_known_wrapped(address, &value.to_le_bytes())
}

fn write_u16_wrapped(
    buffer: &mut C220LocalBuffer,
    address: u64,
    value: u16,
) -> Result<(), C220LocalBufferError> {
    buffer.write_known_wrapped(address, &value.to_le_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_slice_rounds_once_and_flushes_fp32_subnormals() {
        let mut left = [0_u16; 16];
        let mut right = [0_u16; 16];
        left[0] = 0x3c00;
        right[0] = 0x3c00;
        left[1] = 0x1000;
        right[1] = 0x1000;
        let outcome = evaluate_f16_f32_slice(left, right, 1, C220Fp16Mode::NonSaturating).unwrap();
        assert_eq!(outcome.bits, (1.0_f32 + 2_f32.powi(-22)).to_bits());

        let flushed = evaluate_f16_f32_slice(
            [F16_SIGN; 16],
            [0; 16],
            F32_SIGN | 1,
            C220Fp16Mode::NonSaturating,
        )
        .unwrap();
        assert_eq!(flushed.bits, F32_SIGN);
    }
}
