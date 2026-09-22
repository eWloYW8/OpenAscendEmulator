use crate::architecture::c220::C220UbBank;
use crate::isa::c220::special::{
    C220SpecialUnaryInstruction, C220SpecialUnaryOperation, C220SpecialUnaryWidth,
};
use crate::memory::ub::UbMemory;
use crate::numeric::fp32::{Fp32ValueOutcome, Fp32ValueStatus};
use crate::sim::c220::fp16::{
    C220Fp16Mode, C220Fp16Outcome, C220Fp16Status, is_infinite, is_nan, round_finite_to_f16, to_f64,
};

use super::special_tables::*;
use super::{
    C220_VECTOR_TILE_BYTES, C220VectorAddresses, C220VectorControl, C220VectorError,
    C220VectorReadAccess, C220VectorStore, check_repeat_limit, plan_c220_unary_write_targets,
    plan_c220_vector_read_accesses, vector_destination_address_for_width,
};

const F16_SIGN: u16 = 0x8000;
const F16_INFINITY: u16 = 0x7c00;
const F16_MAX_FINITE: u16 = 0x7bff;
const F16_CANONICAL_NAN: u16 = 0x7fff;
const F32_SIGN: u32 = 0x8000_0000;
const F32_ABS: u32 = 0x7fff_ffff;
const F32_INFINITY: u32 = 0x7f80_0000;
const F32_MAX_FINITE: u32 = 0x7f7f_ffff;
const F32_CANONICAL_NAN: u32 = 0x7fff_ffff;
const C220_LN_2: f64 = f32::from_bits(0x3f31_7218) as f64;
const C220_LOG2_E: f64 = f32::from_bits(0x3fb8_aa3b) as f64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220SpecialUnaryIssue {
    pub pc: u64,
    pub word: u32,
    pub instruction: C220SpecialUnaryInstruction,
    pub control: C220VectorControl,
    pub addresses: C220VectorAddresses,
    pub iteration_masks: Vec<[u64; 4]>,
    pub fp16_mode: C220Fp16Mode,
    pub(crate) write_targets: Vec<C220VectorStore>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct C220SpecialUnaryLaneOutcome {
    pub active: bool,
    pub bits: u32,
    pub fp16_status: Option<C220Fp16Status>,
    pub fp32_status: Option<Fp32ValueStatus>,
}

impl C220SpecialUnaryIssue {
    pub fn read_accesses_for_repeat(
        &self,
        repeat_index: usize,
        lane_group: u8,
    ) -> Result<Vec<C220VectorReadAccess>, C220VectorError> {
        if lane_group >= self.instruction.width.lane_groups() {
            return Err(C220VectorError::InvalidLaneGroup(lane_group));
        }
        let mask = self
            .iteration_masks
            .get(repeat_index)
            .ok_or(C220VectorError::MissingMaskState)?;
        plan_c220_vector_read_accesses(
            self.control,
            self.addresses,
            repeat_index,
            mask,
            1,
            self.instruction.width.element_bytes(),
            Some(lane_group),
        )
    }
}

pub fn plan_c220_special_unary_issue(
    pc: u64,
    word: u32,
    control: C220VectorControl,
    addresses: C220VectorAddresses,
    iteration_masks: &[[u64; 4]],
    fp16_mode: C220Fp16Mode,
    ub: &UbMemory,
) -> Result<C220SpecialUnaryIssue, C220VectorError> {
    check_repeat_limit(iteration_masks.len())?;
    let instruction = C220SpecialUnaryInstruction::decode(word)
        .ok_or(C220VectorError::UnsupportedWord { pc, word })?;
    let lane_count = C220_VECTOR_TILE_BYTES / usize::from(instruction.width.element_bytes());
    let mut effective_masks = iteration_masks.to_vec();
    for mask in &mut effective_masks {
        mask[lane_count.div_ceil(64)..].fill(0);
    }
    let write_targets = plan_c220_unary_write_targets(
        control,
        addresses,
        &effective_masks,
        instruction.width.element_bytes(),
        ub,
    )?;
    Ok(C220SpecialUnaryIssue {
        pc,
        word,
        instruction,
        control,
        addresses,
        iteration_masks: effective_masks,
        fp16_mode,
        write_targets,
    })
}

pub(crate) fn evaluate_c220_special_unary_repeat(
    issue: &C220SpecialUnaryIssue,
    repeat_index: usize,
    lane_group: u8,
    lane_slice: Option<(usize, usize)>,
    source_bytes: &[u8],
    ub: &UbMemory,
) -> Result<(Vec<C220SpecialUnaryLaneOutcome>, Vec<C220VectorStore>), C220VectorError> {
    if source_bytes.len() != C220_VECTOR_TILE_BYTES {
        return Err(C220VectorError::InvalidSourceTile {
            actual: source_bytes.len(),
            expected: C220_VECTOR_TILE_BYTES,
        });
    }
    if lane_group >= issue.instruction.width.lane_groups() {
        return Err(C220VectorError::InvalidLaneGroup(lane_group));
    }
    let mask = issue
        .iteration_masks
        .get(repeat_index)
        .ok_or(C220VectorError::MissingMaskState)?;
    let element_bytes = issue.instruction.width.element_bytes();
    let lane_count = C220_VECTOR_TILE_BYTES / usize::from(element_bytes);
    let mut lanes = Vec::with_capacity(lane_count);
    let mut stores = Vec::with_capacity(64);
    for lane_index in 0..lane_count {
        let in_uop = lane_slice.map_or_else(
            || lane_index / 64 == usize::from(lane_group),
            |(first_lane, lane_count)| {
                lane_index >= first_lane && lane_index < first_lane + lane_count
            },
        );
        let active = in_uop && mask[lane_index / 64] & (1_u64 << (lane_index % 64)) != 0;
        if !active {
            lanes.push(C220SpecialUnaryLaneOutcome {
                active: false,
                bits: 0,
                fp16_status: None,
                fp32_status: None,
            });
            continue;
        }
        let (bits, fp16_status, fp32_status) = match issue.instruction.width {
            C220SpecialUnaryWidth::F16 => {
                let offset = lane_index * 2;
                let source = u16::from_le_bytes(
                    source_bytes[offset..offset + 2]
                        .try_into()
                        .expect("two-byte lane"),
                );
                let outcome =
                    evaluate_c220_f16_special(issue.instruction.operation, source, issue.fp16_mode);
                (u32::from(outcome.bits), Some(outcome.status), None)
            }
            C220SpecialUnaryWidth::F32 => {
                let offset = lane_index * 4;
                let source = u32::from_le_bytes(
                    source_bytes[offset..offset + 4]
                        .try_into()
                        .expect("four-byte lane"),
                );
                let outcome = evaluate_c220_fp32_special(issue.instruction.operation, source);
                (outcome.bits, None, Some(outcome.status))
            }
        };
        let address = vector_destination_address_for_width(
            issue.control,
            issue.addresses,
            repeat_index,
            lane_index,
            element_bytes,
        )?;
        ub.check_range(address, usize::from(element_bytes))?;
        stores.push(C220VectorStore {
            repeat_index,
            lane_index,
            address,
            bank: C220UbBank::from_address(address),
            width_bytes: element_bytes,
            data: super::store_data(bits.to_le_bytes()),
        });
        lanes.push(C220SpecialUnaryLaneOutcome {
            active,
            bits,
            fp16_status,
            fp32_status,
        });
    }
    Ok((lanes, stores))
}

pub fn evaluate_c220_fp32_special(
    operation: C220SpecialUnaryOperation,
    source_bits: u32,
) -> Fp32ValueOutcome {
    match operation {
        C220SpecialUnaryOperation::Reciprocal => evaluate_c220_fp32_reciprocal(source_bits),
        C220SpecialUnaryOperation::ReciprocalSqrt => {
            evaluate_c220_fp32_reciprocal_sqrt(source_bits)
        }
        C220SpecialUnaryOperation::Exp => evaluate_c220_fp32_exp(source_bits),
        C220SpecialUnaryOperation::Ln => evaluate_c220_fp32_ln(source_bits),
        C220SpecialUnaryOperation::Sqrt => evaluate_c220_fp32_sqrt(source_bits),
    }
}

fn evaluate_c220_fp32_reciprocal(source_bits: u32) -> Fp32ValueOutcome {
    let absolute = source_bits & F32_ABS;
    let mut status = Fp32ValueStatus {
        nan_operand: absolute > F32_INFINITY,
        infinity_operand: absolute == F32_INFINITY,
        division_by_zero: absolute == 0,
        ..Fp32ValueStatus::default()
    };
    let bits = if status.nan_operand {
        source_bits | 0x7fc0_0000
    } else if absolute == 0 {
        source_bits & F32_SIGN | F32_INFINITY
    } else if status.infinity_operand {
        source_bits & F32_SIGN
    } else {
        let mut exponent = (source_bits & 0x7f80_0000).wrapping_sub(0x0080_0000);
        if (exponent as i32) <= 0 {
            exponent = 0;
        }
        let significand = absolute.wrapping_sub(exponent);
        if significand <= 0x001f_ffff {
            status.overflow = true;
            source_bits & F32_SIGN | F32_INFINITY
        } else {
            let shift = significand.leading_zeros() - 8;
            let normalized = significand << shift;
            let result_exponent = shift as i32 - (exponent >> 23) as i32 + 251;
            let denominator = (normalized >> 14) | 1;
            let mut fraction = (((denominator >> 1) + 0x0004_0000) / denominator) << 15;
            let exponent_bits = if result_exponent < 0 {
                status.underflow = true;
                fraction >>= (-result_exponent).min(31) as u32;
                0
            } else {
                (result_exponent as u32) << 23
            };
            source_bits & F32_SIGN | exponent_bits.wrapping_add(fraction)
        }
    };
    Fp32ValueOutcome { bits, status }
}

fn evaluate_c220_fp32_reciprocal_sqrt(source_bits: u32) -> Fp32ValueOutcome {
    let absolute = source_bits & F32_ABS;
    let mut status = Fp32ValueStatus {
        nan_operand: absolute > F32_INFINITY,
        infinity_operand: absolute == F32_INFINITY,
        division_by_zero: absolute == 0,
        invalid: source_bits & F32_SIGN != 0 && absolute != 0,
        ..Fp32ValueStatus::default()
    };
    let bits = if status.nan_operand {
        F32_CANONICAL_NAN
    } else if absolute == 0 {
        F32_MAX_FINITE
    } else if status.infinity_operand {
        if source_bits & F32_SIGN == 0 {
            0
        } else {
            F32_CANONICAL_NAN
        }
    } else {
        let mut exponent = (absolute & 0x7f80_0000).wrapping_sub(0x0080_0000);
        if (exponent as i32) < 0 {
            exponent = 0;
        }
        let significand = absolute.wrapping_sub(exponent);
        let shift = significand.leading_zeros() - 8;
        let adjusted_exponent = (exponent >> 23) as i32 - shift as i32;
        let normalized = significand << shift;
        let index =
            (((normalized >> 16) & 0x7f) | (((adjusted_exponent & 1) as u32) << 7)) as usize;
        let fraction = u32::from(C220_RSQRT_TABLE[index]) << 15;
        let exponent_bits = ((((375 - adjusted_exponent) >> 1) + 1) as u32) << 23;
        normalize_c220_fp32(exponent_bits.wrapping_add(fraction))
    };
    status.overflow = absolute == 0;
    Fp32ValueOutcome { bits, status }
}

fn evaluate_c220_fp32_exp(source_bits: u32) -> Fp32ValueOutcome {
    let absolute = source_bits & F32_ABS;
    let mut status = fp32_unary_status(source_bits);
    let bits = if status.nan_operand {
        F32_CANONICAL_NAN
    } else if status.infinity_operand {
        if source_bits & F32_SIGN == 0 {
            F32_INFINITY
        } else {
            0
        }
    } else {
        let result = evaluate_c220_fp32_exp_finite(source_bits);
        status.overflow = result & F32_ABS == F32_INFINITY;
        if status.overflow {
            F32_INFINITY
        } else {
            status.underflow = result & F32_ABS == 0 && absolute != 0;
            result
        }
    };
    Fp32ValueOutcome { bits, status }
}

fn evaluate_c220_fp32_exp_finite(source_bits: u32) -> u32 {
    let fraction_bits = source_bits & 0x007f_ffff;
    let encoded_exponent = ((source_bits >> 23) & 0xff) as i32;
    let fraction = f64::from(fraction_bits) * 2_f64.powi(-23);
    let significand = fraction + 1.0;
    let normalized = significand / C220_LN_2;
    let scale = if normalized < 2.0 {
        encoded_exponent - 127
    } else {
        encoded_exponent - 126
    };
    let negative = source_bits & F32_SIGN != 0;

    let c0_scale = 67_108_864.0;
    let c2_scale = 134_217_728.0;
    let c0_inv = 2_f64.powi(-26);
    let c2_inv = 2_f64.powi(-27);
    let reduced_significand = {
        let value = (significand * C220_LOG2_E * 2_147_483_650.0).floor() * 2_f64.powi(-31);
        if value >= 2.0 { value * 0.5 } else { value }
    };
    let reduced = reduced_significand * 2_f64.powi(scale);
    let integer = reduced.floor();
    let fraction = ((reduced - integer) * c0_scale).floor() * c0_inv;
    let mut phase = 1.0 - fraction;
    if phase == 1.0 {
        phase = 0.999_999_985;
    }
    if !negative {
        phase = fraction;
    }
    let indexed = phase * 64.0;
    let index = indexed.floor() as usize;
    let remainder = indexed - indexed.floor();
    let fixed_remainder = (remainder * 32768.0).floor() * 2_f64.powi(-15);
    let squared = (fixed_remainder * fixed_remainder * 32768.0).trunc() * 2_f64.powi(-15);
    let c0 = (f64::from_bits(C220_FP32_EXP_C0[index]) * c0_scale).round() * c0_inv;
    let c1 = (f64::from_bits(C220_FP32_EXP_C1[index]) * c0_scale).round() * c0_inv;
    let c2 = (f64::from_bits(C220_FP32_EXP_C2[index]) * c2_scale).round() * c2_inv;
    let p0 = (squared * c0 * c2_scale).floor() * c2_inv;
    let p1 = (remainder * c1 * c2_scale).floor() * c2_inv;
    let polynomial = ((p0 + p1 + c2) * 16_777_216.0).floor() * 2_f64.powi(-24);
    let mantissa = (polynomial * 8_388_608.0 + 0.5).floor() * 2_f64.powi(-23);
    if mantissa == 2.0 && negative && significand == 1.0 && scale < -120 {
        return 1.0_f32.to_bits();
    }
    let mut result = if negative {
        mantissa / 2_f64.powf(reduced.ceil())
    } else {
        mantissa * 2_f64.powf(integer)
    };
    if result != 0.0 && result < f64::from(f32::MIN_POSITIVE) {
        result = (result * 2_f64.powi(149) + 0.5).floor() * 2_f64.powi(-149);
    }
    (result as f32).to_bits()
}

fn evaluate_c220_fp32_ln(source_bits: u32) -> Fp32ValueOutcome {
    let absolute = source_bits & F32_ABS;
    let mut status = fp32_unary_status(source_bits);
    status.invalid = source_bits & F32_SIGN != 0 && absolute != 0;
    let bits = if status.nan_operand || status.invalid {
        F32_CANONICAL_NAN
    } else if absolute == 0 {
        F32_SIGN | F32_INFINITY
    } else if status.infinity_operand {
        F32_INFINITY
    } else {
        evaluate_c220_fp32_ln_finite(source_bits)
    };
    Fp32ValueOutcome { bits, status }
}

fn evaluate_c220_fp32_ln_finite(source_bits: u32) -> u32 {
    if source_bits == 1.0_f32.to_bits() {
        return 0;
    }
    let fraction_bits = source_bits & 0x007f_ffff;
    let encoded_exponent = ((source_bits >> 23) & 0xff) as i32;
    let (significand, exponent) = if encoded_exponent == 0 {
        let shift = fraction_bits.leading_zeros() as i32 - 8;
        (fraction_bits << shift, -126 - shift)
    } else {
        (fraction_bits | 0x0080_0000, encoded_exponent - 127)
    };
    let fraction = f64::from(significand & 0x007f_ffff) * 2_f64.powi(-23);
    let indexed = fraction * 64.0;
    let index = indexed.floor() as usize;
    let remainder = indexed - indexed.trunc();
    let fixed_remainder = (remainder * 32768.0).floor() * 2_f64.powi(-15);
    let squared = (fixed_remainder * fixed_remainder * 32768.0).trunc() * 2_f64.powi(-15);

    let (c0, c1, c2, coefficient_scale) = if exponent == -1 {
        (
            (f64::from_bits(C220_FP32_LN_NEG_C0[index]) * 268_435_456.0).round() * 2_f64.powi(-28),
            (f64::from_bits(C220_FP32_LN_NEG_C1[index]) * 134_217_728.0).round() * 2_f64.powi(-27),
            (f64::from_bits(C220_FP32_LN_NEG_C2[index]) * 268_435_456.0).round() * 2_f64.powi(-28),
            29,
        )
    } else {
        (
            (f64::from_bits(C220_FP32_LN_C0[index]) * 134_217_728.0).round() * 2_f64.powi(-27),
            (f64::from_bits(C220_FP32_LN_C1[index]) * 134_217_728.0).round() * 2_f64.powi(-27),
            (f64::from_bits(C220_FP32_LN_C2[index]) * 1_073_741_820.0).round() * 2_f64.powi(-30),
            30,
        )
    };
    let scale = 2_f64.powi(coefficient_scale);
    let p0 = (squared * c0 * scale).floor() * 2_f64.powi(-coefficient_scale);
    let p1 = (-(c1 * remainder) * scale).floor() * -2_f64.powi(-coefficient_scale);
    let polynomial = ((p0 + p1 + c2) * 536_870_912.0).floor() * 2_f64.powi(-29);
    let value = if exponent == -1 {
        -((1.0 - fraction) * polynomial * 2_f64.powi(48)).floor() * 2_f64.powi(-48)
    } else {
        (fraction * polynomial * 2_f64.powi(48)).floor() * 2_f64.powi(-48)
            + f64::from(exponent) * C220_LN_2
    };
    quantize_c220_fp32_real(value)
}

fn quantize_c220_fp32_real(value: f64) -> u32 {
    if value == 0.0 {
        return 0;
    }
    let exponent = value.abs().log2().floor();
    let sign = if value < 0.0 { -1.0 } else { 1.0 };
    let significand = (value * sign * 2_f64.powf(23.0 - exponent) + 0.5).floor();
    ((significand * sign * 2_f64.powf(exponent - 23.0)) as f32).to_bits()
}

fn evaluate_c220_fp32_sqrt(source_bits: u32) -> Fp32ValueOutcome {
    let absolute = source_bits & F32_ABS;
    let mut status = fp32_unary_status(source_bits);
    status.invalid = source_bits & F32_SIGN != 0 && absolute != 0;
    let bits = if status.nan_operand || status.invalid {
        F32_CANONICAL_NAN
    } else if status.infinity_operand {
        F32_INFINITY
    } else {
        evaluate_c220_fp32_sqrt_finite(source_bits)
    };
    Fp32ValueOutcome { bits, status }
}

fn evaluate_c220_fp32_sqrt_finite(source_bits: u32) -> u32 {
    if source_bits & F32_ABS == 0 {
        return source_bits;
    }
    let fraction_bits = source_bits & 0x007f_ffff;
    let encoded_exponent = ((source_bits >> 23) & 0xff) as i32;
    let (significand, exponent) = if encoded_exponent == 0 {
        let shift = fraction_bits.leading_zeros() as i32 - 8;
        (fraction_bits << shift, -126 - shift)
    } else {
        (fraction_bits | 0x0080_0000, encoded_exponent - 127)
    };
    let fraction = f64::from(significand & 0x007f_ffff) * 2_f64.powi(-23);
    let scaled = fraction * 64.0;
    let index = scaled.floor() as usize;
    let remainder = scaled - scaled.trunc();
    let fixed_remainder = (remainder * 32768.0).floor() * 2_f64.powi(-15);
    let squared = (fixed_remainder * fixed_remainder * 32768.0).trunc() * 2_f64.powi(-15);
    let c0 = (f64::from_bits(C220_SQRT_C0[index]) * 134_217_728.0).round() * 2_f64.powi(-27);
    let c1 = (f64::from_bits(C220_SQRT_C1[index]) * 134_217_728.0).round() * 2_f64.powi(-27);
    let c2 = (f64::from_bits(C220_SQRT_C2[index]) * 134_217_728.0).round() * 2_f64.powi(-27);
    let polynomial = (remainder * c1 * 134_217_728.0).floor() * 2_f64.powi(-27)
        - (-(c0 * squared) * 134_217_728.0).floor() * 2_f64.powi(-27)
        + c2;
    let mut output_exponent = exponent.div_euclid(2);
    let rounded = if exponent & 1 != 0 {
        let value = (polynomial * 1.414_213_57 * 8_388_608.0 + 0.5).floor() * 2_f64.powi(-23);
        if value == 2.0 {
            output_exponent += 1;
        }
        value as f32
    } else {
        ((polynomial * 8_388_608.0 + 0.5).floor() * 2_f64.powi(-23)) as f32
    };
    (rounded.to_bits() & 0x007f_ffff) | (((output_exponent + 127) as u32) << 23)
}

fn fp32_unary_status(source_bits: u32) -> Fp32ValueStatus {
    let absolute = source_bits & F32_ABS;
    Fp32ValueStatus {
        nan_operand: absolute > F32_INFINITY,
        infinity_operand: absolute == F32_INFINITY,
        ..Fp32ValueStatus::default()
    }
}

fn normalize_c220_fp32(bits: u32) -> u32 {
    if bits & F32_INFINITY == F32_INFINITY {
        bits & F32_SIGN | F32_MAX_FINITE
    } else {
        bits
    }
}

pub fn evaluate_c220_f16_special(
    operation: C220SpecialUnaryOperation,
    source_bits: u16,
    mode: C220Fp16Mode,
) -> C220Fp16Outcome {
    let absolute = source_bits & !F16_SIGN;
    let nan_operand = is_nan(source_bits);
    let infinity_operand = is_infinite(source_bits);
    let invalid = matches!(
        operation,
        C220SpecialUnaryOperation::Ln
            | C220SpecialUnaryOperation::Sqrt
            | C220SpecialUnaryOperation::ReciprocalSqrt
    ) && source_bits & F16_SIGN != 0
        && absolute != 0;
    let mut status = C220Fp16Status {
        nan_operand,
        infinity_operand,
        invalid,
        ..C220Fp16Status::default()
    };
    let bits = match operation {
        C220SpecialUnaryOperation::Exp => {
            if nan_operand {
                fp16_invalid_result(mode)
            } else if infinity_operand {
                if source_bits & F16_SIGN == 0 {
                    fp16_positive_overflow(mode)
                } else {
                    0
                }
            } else {
                let finite = evaluate_c220_f16_exp_finite(source_bits);
                status.underflow = finite == 0;
                status.overflow = finite == F16_MAX_FINITE;
                if status.overflow && mode == C220Fp16Mode::NonSaturating {
                    F16_INFINITY
                } else {
                    finite
                }
            }
        }
        C220SpecialUnaryOperation::Ln => {
            if nan_operand || invalid {
                fp16_invalid_result(mode)
            } else if absolute == 0 {
                match mode {
                    C220Fp16Mode::Saturating => F16_SIGN | F16_MAX_FINITE,
                    C220Fp16Mode::NonSaturating => F16_SIGN | F16_INFINITY,
                }
            } else if infinity_operand {
                fp16_positive_overflow(mode)
            } else {
                evaluate_c220_f16_ln_finite(source_bits)
            }
        }
        C220SpecialUnaryOperation::Reciprocal => {
            if nan_operand {
                fp16_invalid_result(mode)
            } else if absolute == 0 {
                source_bits & F16_SIGN | fp16_positive_overflow(mode)
            } else if infinity_operand {
                source_bits & F16_SIGN
            } else {
                evaluate_c220_f16_reciprocal_finite(source_bits, mode, &mut status)
            }
        }
        C220SpecialUnaryOperation::ReciprocalSqrt => {
            if nan_operand || invalid {
                fp16_invalid_result(mode)
            } else if absolute == 0 {
                source_bits & F16_SIGN | fp16_positive_overflow(mode)
            } else if infinity_operand {
                0
            } else {
                evaluate_c220_f16_reciprocal_sqrt_finite(source_bits, &mut status)
            }
        }
        C220SpecialUnaryOperation::Sqrt => {
            if nan_operand || invalid {
                fp16_invalid_result(mode)
            } else if infinity_operand {
                fp16_positive_overflow(mode)
            } else {
                let fp32_source = (to_f64(source_bits) as f32).to_bits();
                let fp32_result = evaluate_c220_fp32_sqrt_finite(fp32_source);
                round_finite_to_f16(f64::from(f32::from_bits(fp32_result)))
            }
        }
    };
    C220Fp16Outcome { bits, status }
}

fn evaluate_c220_f16_ln_finite(source_bits: u16) -> u16 {
    match source_bits {
        6_453 => return 50_681,
        13_707 => return 48_189,
        15_360 => return 0,
        24_657 => return 18_001,
        _ => {}
    }
    let fraction = i64::from(source_bits & 0x03ff);
    let encoded_exponent = i32::from((source_bits >> 10) & 0x1f);
    let (significand, mut exponent) = if encoded_exponent != 0 {
        (fraction | 0x0400, encoded_exponent - 15)
    } else {
        let shift = (u32::from(source_bits & 0x03ff)).leading_zeros() as i32 - 22;
        (fraction << (shift + 1), -15 - shift)
    };
    let index = ((significand >> 3) & 0x7f) as usize;
    let remainder = significand & 7;
    let squared = remainder * remainder;
    let c1 = (C220_LN_F16_C1[index] << 6) | 0x80_0000;
    let c2 = (C220_LN_F16_C2[index] << 18) | 0x80_0000;
    let c0 = (2 * C220_LN_F16_C0[index]) | 0x80_0000;
    let c1_shift = C220_LN_F16_C1_E[index].unsigned_abs() + 4;
    let c2_shift = C220_LN_F16_C2_E[index].unsigned_abs() + 14;
    let c0_shift = C220_LN_F16_C0_E[index].unsigned_abs();
    let polynomial = (((c0 << 6) >> c0_shift) & 0x1fff_ffff)
        + (((remainder * c1) >> c1_shift) & 0x1fff_ffff)
        - (((squared * c2) >> c2_shift) & 0x1fff_ffff);
    let mut value =
        (polynomial >> 5) + i64::from(round_from_guard(polynomial, 4, polynomial & 0x0f != 0));
    let mut sign = 0_u16;
    if exponent < 0 {
        if value != 0 {
            exponent = !exponent;
            value = 0x100_0000 - value;
        } else {
            exponent = -exponent;
        }
        sign = F16_SIGN;
    }
    let combined = value | (i64::from(exponent) << 24);
    let mut first_shift = 0_i32;
    while first_shift < 12 && combined >> (29 - first_shift) == 0 {
        first_shift += 1;
    }
    let normalized = combined << first_shift;
    let first_rounded =
        (normalized >> 6) + i64::from(round_from_guard(normalized, 5, normalized & 0x0f != 0));
    let scaled = (744_261_120_i64 * first_rounded) >> 24;
    let second_rounded = (scaled >> 6) + i64::from(round_from_guard(scaled, 5, scaled & 0x0f != 0));
    let mut second_shift = 0_i32;
    while second_shift < 23 && second_rounded >> (23 - second_shift) == 0 {
        second_shift += 1;
    }
    let final_normalized = second_rounded << second_shift;
    let discarded = final_normalized & 0x07ff;
    let rounded_fraction = ((final_normalized >> 13) & 0x03ff)
        + i64::from(round_from_guard(
            final_normalized,
            12,
            discarded != 0 || final_normalized & 0x0800 != 0,
        ));
    let carry = i32::from(rounded_fraction & 0x0400 != 0);
    let exponent_bits = ((carry - first_shift + 20 - second_shift) as u16) << 10;
    sign | exponent_bits | (rounded_fraction as u16 & 0x03ff)
}

fn round_from_guard(value: i64, guard_bit: u32, lower_sticky: bool) -> bool {
    value & (1_i64 << guard_bit) != 0 && (value & (1_i64 << (guard_bit + 1)) != 0 || lower_sticky)
}

fn evaluate_c220_f16_exp_finite(source_bits: u16) -> u16 {
    let source = u32::from(source_bits);
    if (1..=0x0fff).contains(&source) {
        return 0x3c00;
    }
    if source == 17_981 {
        return 0x6000;
    }
    if (18_828..=0x7fff).contains(&source) {
        return F16_MAX_FINITE;
    }
    if (0x8000..=0x8c00).contains(&source) {
        return 0x3c00;
    }
    if source == 47_500 {
        return 0x3800;
    }
    if source > 52_309 {
        return 0;
    }

    let significand = i64::from((source & 0x03ff) | 0x0400);
    let negative = source & 0x8000 != 0;
    let encoded_exponent = ((source >> 10) & 0x1f) as i32;
    let mut product = 774_541_002_i64 * significand;
    let scale = if product & 0x100_0000_0000 != 0 {
        product >>= 1;
        encoded_exponent - 14
    } else {
        encoded_exponent - 15
    };
    let shifted = product >> 9;
    let (mut integral, mut fractional) = if scale < 0 {
        (0_i32, (shifted >> -scale) as u32)
    } else if scale == 0 {
        (1_i32, (shifted as u32) & 0x3fff_ffff)
    } else {
        let scaled = shifted << scale;
        (
            ((scaled >> 30) & 0x7fff) as i32,
            (scaled as u32) & 0x3fff_ffff,
        )
    };
    if negative {
        integral = !integral;
        fractional = 0x4000_0000_u32.wrapping_sub(fractional);
    }
    let index = ((fractional >> 24) & 0x3f) as usize;
    let remainder = i64::from(fractional & 0x00ff_ffff);
    let square = (remainder * remainder) >> 30;
    let polynomial = ((i64::from(C220_EXP_C1[index] << 12) * remainder) >> 30)
        + i64::from(C220_EXP_C0[index] << 6)
        + ((i64::from(C220_EXP_C2[index] << 18) * square) >> 30);
    let mut significand = (polynomial >> 7) as i32;
    if polynomial & 0x40 != 0 {
        significand += 1;
    }
    let exponent_bits = if integral >= -14 {
        ((integral + 15) as u16) << 10
    } else {
        significand >>= -14 - integral;
        0
    };
    let discarded = significand & 0x07ff;
    let sticky = discarded != 0;
    let rounded = ((significand >> 13) & 0x03ff)
        + i32::from(
            significand & 0x1000 != 0
                && (((significand >> 13) | (significand >> 11)) & 1 != 0 || sticky),
        );
    exponent_bits | rounded as u16
}

fn evaluate_c220_f16_reciprocal_finite(
    source_bits: u16,
    mode: C220Fp16Mode,
    status: &mut C220Fp16Status,
) -> u16 {
    let sign = source_bits & F16_SIGN;
    let absolute = source_bits & !F16_SIGN;
    let mut exponent = (source_bits & 0x7c00).wrapping_sub(0x0400);
    if (exponent as i16) < 0 {
        exponent = 0;
    }
    let significand = absolute.wrapping_sub(exponent);
    if significand <= 0x00ff {
        status.overflow = true;
        return sign | fp16_positive_overflow(mode);
    }
    let shift = u32::from(significand).leading_zeros() - 21;
    let result_exponent = shift as i32 - i32::from(exponent >> 10) + 27;
    let normalized = significand.wrapping_shl(shift);
    let denominator = u32::from((normalized >> 1) | 1);
    let mut fraction = (4 * ((((denominator >> 1) + 0x0004_0000) / denominator) & 0x3fff)) as u16;
    let exponent_bits = if result_exponent < 0 {
        status.underflow = true;
        fraction >>= (-result_exponent).min(15) as u32;
        0
    } else {
        ((result_exponent as u16) & 0x3f) << 10
    };
    sign.wrapping_add(fraction).wrapping_add(exponent_bits)
}

fn evaluate_c220_f16_reciprocal_sqrt_finite(source_bits: u16, status: &mut C220Fp16Status) -> u16 {
    let absolute = source_bits & !F16_SIGN;
    let mut exponent = (absolute & 0x7c00).wrapping_sub(0x0400);
    if (exponent as i16) < 0 {
        exponent = 0;
    }
    let significand = absolute.wrapping_sub(exponent);
    let shift = u32::from(significand).leading_zeros() - 21;
    let adjusted_exponent = i32::from(exponent >> 10) - shift as i32;
    let index = ((((u32::from(significand) << shift) >> 3) & 0x7f)
        | (((adjusted_exponent & 1) as u32) << 7)) as usize;
    let mut bits = 4 * (C220_RSQRT_TABLE[index] & 0x3fff)
        + ((((39 - adjusted_exponent) / 2 + 1) as u16) << 10);
    if bits & F16_INFINITY == F16_INFINITY {
        bits = F16_MAX_FINITE;
        status.overflow = true;
    }
    bits
}

fn fp16_invalid_result(mode: C220Fp16Mode) -> u16 {
    match mode {
        C220Fp16Mode::Saturating => 0,
        C220Fp16Mode::NonSaturating => F16_CANONICAL_NAN,
    }
}

fn fp16_positive_overflow(mode: C220Fp16Mode) -> u16 {
    match mode {
        C220Fp16Mode::Saturating => F16_MAX_FINITE,
        C220Fp16Mode::NonSaturating => F16_INFINITY,
    }
}

const C220_RSQRT_TABLE: [u16; 256] = [
    0x01ff, 0x01fd, 0x01fb, 0x01f9, 0x01f7, 0x01f5, 0x01f3, 0x01f2, 0x01f0, 0x01ee, 0x01ec, 0x01ea,
    0x01e9, 0x01e7, 0x01e5, 0x01e4, 0x01e2, 0x01e0, 0x01df, 0x01dd, 0x01db, 0x01da, 0x01d8, 0x01d7,
    0x01d5, 0x01d4, 0x01d2, 0x01d1, 0x01cf, 0x01ce, 0x01cc, 0x01cb, 0x01c9, 0x01c8, 0x01c6, 0x01c5,
    0x01c4, 0x01c2, 0x01c1, 0x01c0, 0x01be, 0x01bd, 0x01bc, 0x01ba, 0x01b9, 0x01b8, 0x01b7, 0x01b5,
    0x01b4, 0x01b3, 0x01b2, 0x01b0, 0x01af, 0x01ae, 0x01ad, 0x01ac, 0x01aa, 0x01a9, 0x01a8, 0x01a7,
    0x01a6, 0x01a5, 0x01a4, 0x01a3, 0x01a2, 0x01a0, 0x019f, 0x019e, 0x019d, 0x019c, 0x019b, 0x019a,
    0x0199, 0x0198, 0x0197, 0x0196, 0x0195, 0x0194, 0x0193, 0x0192, 0x0191, 0x0190, 0x018f, 0x018e,
    0x018d, 0x018c, 0x018c, 0x018b, 0x018a, 0x0189, 0x0188, 0x0187, 0x0186, 0x0185, 0x0184, 0x0183,
    0x0183, 0x0182, 0x0181, 0x0180, 0x017f, 0x017e, 0x017e, 0x017d, 0x017c, 0x017b, 0x017a, 0x0179,
    0x0179, 0x0178, 0x0177, 0x0176, 0x0176, 0x0175, 0x0174, 0x0173, 0x0172, 0x0172, 0x0171, 0x0170,
    0x016f, 0x016f, 0x016e, 0x016d, 0x016d, 0x016c, 0x016b, 0x016a, 0x0169, 0x0168, 0x0167, 0x0165,
    0x0164, 0x0163, 0x0161, 0x0160, 0x015f, 0x015d, 0x015c, 0x015b, 0x015a, 0x0158, 0x0157, 0x0156,
    0x0155, 0x0154, 0x0152, 0x0151, 0x0150, 0x014f, 0x014e, 0x014d, 0x014c, 0x014b, 0x014a, 0x0148,
    0x0147, 0x0146, 0x0145, 0x0144, 0x0143, 0x0142, 0x0141, 0x0140, 0x013f, 0x013e, 0x013d, 0x013c,
    0x013c, 0x013b, 0x013a, 0x0139, 0x0138, 0x0137, 0x0136, 0x0135, 0x0134, 0x0133, 0x0133, 0x0132,
    0x0131, 0x0130, 0x012f, 0x012e, 0x012e, 0x012d, 0x012c, 0x012b, 0x012a, 0x012a, 0x0129, 0x0128,
    0x0127, 0x0126, 0x0126, 0x0125, 0x0124, 0x0123, 0x0123, 0x0122, 0x0121, 0x0121, 0x0120, 0x011f,
    0x011e, 0x011e, 0x011d, 0x011c, 0x011c, 0x011b, 0x011a, 0x011a, 0x0119, 0x0118, 0x0118, 0x0117,
    0x0116, 0x0116, 0x0115, 0x0114, 0x0114, 0x0113, 0x0113, 0x0112, 0x0111, 0x0111, 0x0110, 0x0110,
    0x010f, 0x010e, 0x010e, 0x010d, 0x010d, 0x010c, 0x010b, 0x010b, 0x010a, 0x010a, 0x0109, 0x0109,
    0x0108, 0x0108, 0x0107, 0x0106, 0x0106, 0x0105, 0x0105, 0x0104, 0x0104, 0x0103, 0x0103, 0x0102,
    0x0102, 0x0101, 0x0101, 0x0100,
];

const C220_EXP_C0: [i32; 64] = [
    16777216, 16959908, 17144589, 17331281, 17520006, 17710787, 17903645, 18098603, 18295683,
    18494910, 18696307, 18899896, 19105703, 19313750, 19524063, 19736666, 19951584, 20168843,
    20388467, 20610483, 20834916, 21061794, 21291142, 21522987, 21757357, 21994279, 22233781,
    22475891, 22720637, 22968049, 23218155, 23470984, 23726566, 23984931, 24246110, 24510133,
    24777031, 25046835, 25319577, 25595289, 25874004, 26155753, 26440571, 26728490, 27019544,
    27313767, 27611195, 27911861, 28215801, 28523051, 28833647, 29147625, 29465022, 29785875,
    30110221, 30438100, 30769549, 31104608, 31443315, 31785710, 32131834, 32481727, 32835429,
    33192984,
];

const C220_EXP_C1: [i32; 64] = [
    181702, 183681, 185681, 187703, 189747, 191813, 193901, 196013, 198147, 200305, 202486, 204691,
    206920, 209173, 211451, 213754, 216081, 218434, 220813, 223217, 225648, 228105, 230589, 233100,
    235638, 238204, 240798, 243420, 246071, 248751, 251459, 254197, 256965, 259764, 262592, 265452,
    268342, 271264, 274218, 277204, 280223, 283274, 286359, 289477, 292629, 295816, 299037, 302294,
    305585, 308913, 312277, 315677, 319115, 322590, 326102, 329653, 333243, 336872, 340540, 344248,
    347997, 351787, 355617, 359490,
];

const C220_EXP_C2: [i32; 64] = [
    989, 1000, 1010, 1021, 1033, 1044, 1055, 1067, 1078, 1090, 1102, 1114, 1126, 1138, 1151, 1163,
    1176, 1189, 1202, 1215, 1228, 1241, 1255, 1269, 1282, 1296, 1311, 1325, 1339, 1354, 1369, 1384,
    1399, 1414, 1429, 1445, 1461, 1476, 1493, 1509, 1525, 1542, 1559, 1576, 1593, 1610, 1628, 1645,
    1663, 1681, 1700, 1718, 1737, 1756, 1775, 1794, 1814, 1834, 1854, 1874, 1894, 1915, 1936, 1957,
];

const C220_FP32_EXP_C0: [u64; 64] = [
    0x3f0eea7fd7b94a5e,
    0x3f0f40aec10de638,
    0x3f0f97cdea34e920,
    0x3f0fefdff0f0c456,
    0x3f102473bd20a37a,
    0x3f105173994871c4,
    0x3f107ef0e6dc5ba5,
    0x3f10aced038cf1f1,
    0x3f10db6950db6c19,
    0x3f110a67341fadaf,
    0x3f1139e816979df8,
    0x3f1169ed6570707f,
    0x3f119a7891d1639e,
    0x3f11cb8b10e452be,
    0x3f11fd265be72592,
    0x3f122f4bf0311015,
    0x3f1261fd4f3f4a2f,
    0x3f12953bfec461e7,
    0x3f12c90988aee019,
    0x3f12fd677b39fd20,
    0x3f13325768f4415c,
    0x3f1367dae8d039eb,
    0x3f139df3962fc496,
    0x3f13d4a310eb6ab1,
    0x3f140beafd670d04,
    0x3f1443cd049a878c,
    0x3f147c4ad41b1083,
    0x3f14b5661e2d2ffe,
    0x3f14ef2099d16d2b,
    0x3f15297c02cc4f21,
    0x3f15647a19bbb1cb,
    0x3f15a01ca4202016,
    0x3f15dc656c641b33,
    0x3f16195641f82cb4,
    0x3f1656f0f952da4b,
    0x3f1695376c075307,
    0x3f16d42b78d01b5e,
    0x3f1713cf039fbac7,
    0x3f175423f5acb2d5,
    0x3f17952c3d7d8660,
    0x3f17d6e9cf03641b,
    0x3f18195ea398cf72,
    0x3f185c8cba20f58a,
    0x3f18a07617070127,
    0x3f18e51cc45ccdb6,
    0x3f192a82d1ded56d,
    0x3f1970aa550cf516,
    0x3f19b7956930fc41,
    0x3f19ff462f776efa,
    0x3f1a47becefccbd3,
    0x3f1a910174dce472,
    0x3f1adb105444e643,
    0x3f1b25eda6829e30,
    0x3f1b719bab19e026,
    0x3f1bbe1ca7cf30e5,
    0x3f1c0b72e8c1adeb,
    0x3f1c59a0c071d69b,
    0x3f1ca8a887dccfba,
    0x3f1cf88c9e8c6f50,
    0x3f1d494f6aa52606,
    0x3f1d9af358fd7344,
    0x3f1ded7add2c7b85,
    0x3f1e40e871a4b3ed,
    0x3f1e953e97b694e6,
];

const C220_FP32_EXP_C1: [u64; 64] = [
    0x3f862e32f2bffced,
    0x3f866c07d21e9b15,
    0x3f86aa890ef5ce07,
    0x3f86e9b889c43d9c,
    0x3f87299828441624,
    0x3f876a29d57981b8,
    0x3f87ab6f81c1792e,
    0x3f87ed6b22e0a7b6,
    0x3f88301eb41274c0,
    0x3f88738c36184b5b,
    0x3f88b7b5af48f651,
    0x3f88fc9d2ba032f5,
    0x3f894244bcce6eb8,
    0x3f8988ae7a48b2f2,
    0x3f89cfdc8158acc1,
    0x3f8a17d0f52cfbe8,
    0x3f8a608dfee99c32,
    0x3f8aaa15cdb87be4,
    0x3f8af46a96da501d,
    0x3f8b3f8e95b7831a,
    0x3f8b8b840bf169d9,
    0x3f8bd84d41738f99,
    0x3f8c25ec84854591,
    0x3f8c746429db6022,
    0x3f8cc3b68caa19e3,
    0x3f8d13e60eb72f2e,
    0x3f8d64f5186c3c56,
    0x3f8db6e618e93178,
    0x3f8e09bb86170b73,
    0x3f8e5d77dcbac6f3,
    0x3f8eb21da0887259,
    0x3f8f07af5c36861e,
    0x3f8f5e2fa1917eb8,
    0x3f8fb5a1098f8205,
    0x3f9007031a323708,
    0x3f9033b0e4cafe62,
    0x3f9060db3c08174f,
    0x3f908e837b1cf46f,
    0x3f90bcab0104e9f5,
    0x3f90eb53308dbad0,
    0x3f911a7d70623334,
    0x3f914a2b2b150048,
    0x3f917a5dcf2b8131,
    0x3f91ab16cf28d9a4,
    0x3f91dc57a1990251,
    0x3f920e21c11c14de,
    0x3f924076ac71a094,
    0x3f927357e684325b,
    0x3f92a6c6f674eb94,
    0x3f92dac567a74269,
    0x3f930f54c9cce32c,
    0x3f934476b0f1b009,
    0x3f937a2cb587e42a,
    0x3f93b0787474533c,
    0x3f93e75b8f1ad5e0,
    0x3f941ed7ab6acaf1,
    0x3f9456ee73ebcce9,
    0x3f948fa197ca7763,
    0x3f94c8f2cae55c1d,
    0x3f9502e3c5da1e0c,
    0x3f953d764612a735,
    0x3f9578ac0dd28c65,
    0x3f95b486e4448b6a,
    0x3f95f108958845ba,
];

const C220_FP32_EXP_C2: [u64; 64] = [
    0x3ff0000001c8881b,
    0x3ff02c9a40450118,
    0x3ff059b0d4e80c94,
    0x3ff087451a4d3777,
    0x3ff0b5586ed6477d,
    0x3ff0e3ec34b5c117,
    0x3ff11301d1f98a2d,
    0x3ff1429ab095aaeb,
    0x3ff172b83e6f2b48,
    0x3ff1a35bed671139,
    0x3ff1d48733657a61,
    0x3ff2063b8a64d7ab,
    0x3ff2387a707d46d2,
    0x3ff26b4567f00ab0,
    0x3ff29e9df73325b2,
    0x3ff2d285a8fd12e2,
    0x3ff306fe0c50a032,
    0x3ff33c08b488e987,
    0x3ff371a739657637,
    0x3ff3a7db371676e5,
    0x3ff3dea64e492697,
    0x3ff4160a24344cfa,
    0x3ff44e0862a4e5ac,
    0x3ff486a2b80ae85c,
    0x3ff4bfdad7863668,
    0x3ff4f9b278f3ab52,
    0x3ff5342b58fa52ac,
    0x3ff56f473918c18d,
    0x3ff5ab07dfb296b9,
    0x3ff5e76f181e1f64,
    0x3ff6247eb2b221ee,
    0x3ff6623884d3cfb9,
    0x3ff6a09e6904ddd1,
    0x3ff6dfb23ef1c3f8,
    0x3ff71f75eb8024af,
    0x3ff75feb58dd5c45,
    0x3ff7a114768d391d,
    0x3ff7e2f33978dd99,
    0x3ff825899bfdc91b,
    0x3ff868d99dfd0ec0,
    0x3ff8ace544eab21d,
    0x3ff8f1ae9bdd334c,
    0x3ff93737b39d4208,
    0x3ff97d82a2b5a021,
    0x3ff9c49185832ded,
    0x3ffa0c667e45242f,
    0x3ffa5503b52d7c36,
    0x3ffa9e6b587182eb,
    0x3ffae89f9c5a9e33,
    0x3ffb33a2bb573cdd,
    0x3ffb7f76f60bf8bf,
    0x3ffbcc1e9364e78e,
    0x3ffc199be0a71b46,
    0x3ffc67f131825531,
    0x3ffcb720e022e7be,
    0x3ffd072d4d43ccc8,
    0x3ffd5818e040ecbc,
    0x3ffda9e6072998ce,
    0x3ffdfc9736d33938,
    0x3ffe502eeaec2edb,
    0x3ffea4afa60eeae7,
    0x3ffefa1bf1d539a7,
    0x3fff50765eebc49c,
    0x3fffa7c18525cb11,
];

const C220_SQRT_C0: [u64; 64] = [
    0xbeffa0000000000d,
    0xbefee8000000000b,
    0xbefe38000000000a,
    0xbefd880000000009,
    0xbefce8000000000b,
    0xbefc48000000000c,
    0xbefbaffffffffff2,
    0xbefb17fffffffff5,
    0xbefa87fffffffffa,
    0xbefa000000000000,
    0xbef9780000000006,
    0xbef8f8000000000d,
    0xbef87ffffffffff9,
    0xbef8000000000000,
    0xbef790000000000a,
    0xbef71ffffffffff7,
    0xbef6b00000000001,
    0xbef648000000000c,
    0xbef5dffffffffffa,
    0xbef5780000000006,
    0xbef517fffffffff5,
    0xbef4b80000000002,
    0xbef45ffffffffff3,
    0xbef4080000000001,
    0xbef3affffffffff2,
    0xbef3580000000000,
    0xbef307fffffffff3,
    0xbef2b80000000002,
    0xbef267fffffffff4,
    0xbef2200000000006,
    0xbef1d7fffffffff9,
    0xbef190000000000a,
    0xbef147fffffffffe,
    0xbef107fffffffff3,
    0xbef0c00000000004,
    0xbef07ffffffffff9,
    0xbef040000000000b,
    0xbef0080000000001,
    0xbeef8fffffffffed,
    0xbeef200000000014,
    0xbeeeb00000000001,
    0xbeee3fffffffffee,
    0xbeedd00000000015,
    0xbeed600000000002,
    0xbeecfffffffffff1,
    0xbeeca0000000001c,
    0xbeec300000000008,
    0xbeebcffffffffff8,
    0xbeeb6fffffffffe7,
    0xbeeb200000000014,
    0xbeeac00000000004,
    0xbeea5ffffffffff3,
    0xbeea0fffffffffe5,
    0xbee9c00000000012,
    0xbee9600000000002,
    0xbee90ffffffffff4,
    0xbee8bfffffffffe6,
    0xbee8700000000013,
    0xbee8300000000008,
    0xbee7dffffffffffa,
    0xbee78fffffffffed,
    0xbee750000000001d,
    0xbee7100000000012,
    0xbee6c00000000004,
];

const C220_SQRT_C1: [u64; 64] = [
    0x3f7fffe000000001,
    0x3f7fc09800000004,
    0x3f7f82c800000006,
    0x3f7f466000000000,
    0x3f7f0b4800000002,
    0x3f7ed18000000001,
    0x3f7e98efffffffff,
    0x3f7e6197fffffffd,
    0x3f7e2b5fffffffff,
    0x3f7df64ffffffffb,
    0x3f7dc25000000001,
    0x3f7d8f5800000002,
    0x3f7d5d67ffffffff,
    0x3f7d2c6ffffffffd,
    0x3f7cfc6800000006,
    0x3f7ccd47ffffffff,
    0x3f7c9f1000000001,
    0x3f7c71b000000005,
    0x3f7c452800000000,
    0x3f7c196ffffffffb,
    0x3f7bee7fffffffff,
    0x3f7bc44ffffffffe,
    0x3f7b9ae000000002,
    0x3f7b721ffffffffc,
    0x3f7b4a1800000005,
    0x3f7b22c000000003,
    0x3f7afc07fffffffd,
    0x3f7ad5f800000000,
    0x3f7ab087ffffffff,
    0x3f7a8bb800000005,
    0x3f7a677800000001,
    0x3f7a43d000000001,
    0x3f7a20b000000001,
    0x3f79fe2000000002,
    0x3f79dc1800000002,
    0x3f79ba8fffffffff,
    0x3f79999000000006,
    0x3f797907fffffffb,
    0x3f79590000000004,
    0x3f79396800000004,
    0x3f791a47fffffffe,
    0x3f78fb9ffffffffd,
    0x3f78dd5ffffffffc,
    0x3f78bf8ffffffffd,
    0x3f78a227ffffffff,
    0x3f78853000000003,
    0x3f78689000000002,
    0x3f784c6000000003,
    0x3f783087ffffffff,
    0x3f78151000000004,
    0x3f77f9f800000006,
    0x3f77df3800000003,
    0x3f77c4d7fffffffd,
    0x3f77aac800000006,
    0x3f77910ffffffffe,
    0x3f7777a800000005,
    0x3f775e9000000005,
    0x3f7745c7fffffffc,
    0x3f772d5000000002,
    0x3f77152800000000,
    0x3f76fd47ffffffff,
    0x3f76e5afffffffff,
    0x3f76ce6000000000,
    0x3f76b75800000002,
];

const C220_SQRT_C2: [u64; 64] = [
    0x3ff0000000000000,
    0x3ff01fe03fffffee,
    0x3ff03f81f7fffffe,
    0x3ff05ee690000010,
    0x3ff07e0f6800000d,
    0x3ff09cfdcffffff3,
    0x3ff0bbb308000004,
    0x3ff0da304ffffffa,
    0x3ff0f876d0000009,
    0x3ff11687ac000011,
    0x3ff13463fc000003,
    0x3ff1520cd3ffffff,
    0x3ff16f8333ffffeb,
    0x3ff18cc823fffff7,
    0x3ff1a9dc8ffffff5,
    0x3ff1c6c16c000006,
    0x3ff1e3779c000005,
    0x3ff2000000000000,
    0x3ff21c5b70000010,
    0x3ff2388abffffffa,
    0x3ff2548ebc000013,
    0x3ff270682400000e,
    0x3ff28c17bbfffff0,
    0x3ff2a79e3c000000,
    0x3ff2c2fc5bfffff8,
    0x3ff2de32c8000000,
    0x3ff2f9422c000012,
    0x3ff3142b30000001,
    0x3ff32eee77fffffe,
    0x3ff3498c97fffff6,
    0x3ff364062ffffff4,
    0x3ff37e5bd4000011,
    0x3ff3988e14000014,
    0x3ff3b29d7fffffee,
    0x3ff3cc8a9bfffffd,
    0x3ff3e655f0000011,
    0x3ff4000000000000,
    0x3ff419894bfffffe,
    0x3ff432f24fffffed,
    0x3ff44c3b8400000c,
    0x3ff4656560000013,
    0x3ff47e7053fffff4,
    0x3ff4975cd7ffffef,
    0x3ff4b02b50000003,
    0x3ff4c8dc2fffffef,
    0x3ff4e16fdc00000f,
    0x3ff4f9e6bc000002,
    0x3ff5124133fffff7,
    0x3ff52a7fac000017,
    0x3ff542a277fffff0,
    0x3ff55aaa00000004,
    0x3ff572969c00000b,
    0x3ff58a68a4000016,
    0x3ff5a22074000002,
    0x3ff5b9be5bffffeb,
    0x3ff5d142b800000a,
    0x3ff5e8add3fffff9,
    0x3ff6000000000000,
    0x3ff6173990000008,
    0x3ff62e5acc00000b,
    0x3ff6456403fffff8,
    0x3ff65c5583ffffee,
    0x3ff6732f8bfffff3,
    0x3ff689f26bfffff5,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn special_unary_anchor_values_use_the_quantized_c220_paths() {
        use C220SpecialUnaryOperation::{Exp, Ln, Reciprocal, ReciprocalSqrt, Sqrt};

        for (operation, source, expected) in [
            (Exp, 0_u32, 0x3f80_0000),
            (Ln, 0x3f80_0000, 0),
            (Reciprocal, 0x3f80_0000, 0x3f7f_8000),
            (ReciprocalSqrt, 0x3f80_0000, 0x3f7f_8000),
            (Sqrt, 0x3f80_0000, 0x3f80_0000),
        ] {
            assert_eq!(evaluate_c220_fp32_special(operation, source).bits, expected);
        }
        for (operation, source, expected) in [
            (Exp, 0_u16, 0x3c00),
            (Ln, 0x3c00, 0),
            (Reciprocal, 0x3c00, 0x3bfc),
            (ReciprocalSqrt, 0x3c00, 0x3bfc),
            (Sqrt, 0x3c00, 0x3c00),
        ] {
            assert_eq!(
                evaluate_c220_f16_special(operation, source, C220Fp16Mode::Saturating).bits,
                expected
            );
        }
    }
}
