use crate::isa::c220::vector::scalar::C220VectorScalarOperation;
mod conversion;
pub use conversion::{C220Fp16Rounding, c220_f32_to_fp16};

const SIGN: u16 = 0x8000;
const EXPONENT: u16 = 0x7c00;
const FRACTION: u16 = 0x03ff;
const MAX_FINITE: u16 = 0x7bff;
const CANONICAL_NAN: u16 = 0x7fff;
const MIN_NORMAL: f64 = 1.0 / 16_384.0;
const HALF_MIN_SUBNORMAL: f64 = 1.0 / 33_554_432.0;
const OVERFLOW_MIDPOINT: f64 = 65520.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220Fp16Mode {
    Saturating,
    NonSaturating,
}

impl C220Fp16Mode {
    pub const fn from_control_spr(value: u64) -> Self {
        if value & (1 << 48) == 0 {
            Self::Saturating
        } else {
            Self::NonSaturating
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct C220Fp16Status {
    pub nan_operand: bool,
    pub infinity_operand: bool,
    pub invalid: bool,
    pub overflow: bool,
    pub underflow: bool,
}

impl C220Fp16Status {
    pub const fn merge(self, other: Self) -> Self {
        Self {
            nan_operand: self.nan_operand || other.nan_operand,
            infinity_operand: self.infinity_operand || other.infinity_operand,
            invalid: self.invalid || other.invalid,
            overflow: self.overflow || other.overflow,
            underflow: self.underflow || other.underflow,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Fp16Outcome {
    pub bits: u16,
    pub status: C220Fp16Status,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum C220Fp16AddRounding {
    #[default]
    NearestEven,
    TowardZero,
}

impl C220Fp16AddRounding {
    pub const fn from_model_value(value: u32) -> Self {
        if value == 0 {
            Self::NearestEven
        } else {
            Self::TowardZero
        }
    }
}

/// Overflow classification precedes finite-result rounding. Consequently,
/// truncation does not change the overflow threshold or saturation policy.
pub fn evaluate_c220_fp16_add(
    first: u16,
    second: u16,
    mode: C220Fp16Mode,
    rounding: C220Fp16AddRounding,
) -> C220Fp16Outcome {
    let mut result = evaluate_c220_fp16(C220VectorScalarOperation::Add, first, second, mode);
    if rounding == C220Fp16AddRounding::TowardZero
        && !result.status.nan_operand
        && !result.status.infinity_operand
        && !result.status.overflow
    {
        let exact = to_f64(first) + to_f64(second);
        if to_f64(result.bits).abs() > exact.abs() {
            result.bits -= 1;
        }
    }
    result
}

pub fn evaluate_c220_fp16_relu(bits: u16, mode: C220Fp16Mode) -> C220Fp16Outcome {
    let nan_operand = is_nan(bits);
    let infinity_operand = is_infinite(bits);
    let result = if nan_operand {
        match mode {
            C220Fp16Mode::Saturating => 0,
            C220Fp16Mode::NonSaturating => CANONICAL_NAN,
        }
    } else if bits & SIGN != 0 {
        0
    } else if infinity_operand && mode == C220Fp16Mode::Saturating {
        MAX_FINITE
    } else {
        bits
    };
    C220Fp16Outcome {
        bits: result,
        status: C220Fp16Status {
            nan_operand,
            infinity_operand,
            ..C220Fp16Status::default()
        },
    }
}

pub fn evaluate_c220_fp16_abs(bits: u16, mode: C220Fp16Mode) -> C220Fp16Outcome {
    let nan_operand = is_nan(bits);
    let infinity_operand = is_infinite(bits);
    let result = if nan_operand {
        match mode {
            C220Fp16Mode::Saturating => 0,
            C220Fp16Mode::NonSaturating => CANONICAL_NAN,
        }
    } else if infinity_operand && mode == C220Fp16Mode::Saturating {
        MAX_FINITE
    } else {
        bits & !SIGN
    };
    C220Fp16Outcome {
        bits: result,
        status: C220Fp16Status {
            nan_operand,
            infinity_operand,
            ..C220Fp16Status::default()
        },
    }
}

pub fn evaluate_c220_fp16_lrelu(
    source_bits: u16,
    slope_bits: u16,
    mode: C220Fp16Mode,
) -> C220Fp16Outcome {
    if source_bits & SIGN == 0 {
        return evaluate_c220_fp16_abs(source_bits, mode);
    }
    evaluate_c220_fp16(
        C220VectorScalarOperation::Multiply,
        source_bits,
        slope_bits,
        mode,
    )
}

pub fn evaluate_c220_fp16(
    operation: C220VectorScalarOperation,
    first_bits: u16,
    second_bits: u16,
    mode: C220Fp16Mode,
) -> C220Fp16Outcome {
    if operation == C220VectorScalarOperation::LeakyRelu {
        return evaluate_c220_fp16_lrelu(first_bits, second_bits, mode);
    }
    let first_nan = is_nan(first_bits);
    let second_nan = is_nan(second_bits);
    let first_infinite = is_infinite(first_bits);
    let second_infinite = is_infinite(second_bits);
    let opposite_infinities =
        first_infinite && second_infinite && (first_bits ^ second_bits) & SIGN != 0;
    let zero_times_infinity = (first_infinite && second_bits & !SIGN == 0)
        || (second_infinite && first_bits & !SIGN == 0);
    let mut status = C220Fp16Status {
        nan_operand: first_nan || second_nan,
        infinity_operand: first_infinite || second_infinite,
        invalid: match operation {
            C220VectorScalarOperation::Add => opposite_infinities,
            C220VectorScalarOperation::Multiply => zero_times_infinity,
            C220VectorScalarOperation::Maximum | C220VectorScalarOperation::Minimum => false,
            C220VectorScalarOperation::LeakyRelu => {
                unreachable!("handled before binary evaluation")
            }
        },
        ..C220Fp16Status::default()
    };
    if status.nan_operand || status.invalid {
        return C220Fp16Outcome {
            bits: match mode {
                C220Fp16Mode::Saturating => 0,
                C220Fp16Mode::NonSaturating => CANONICAL_NAN,
            },
            status,
        };
    }

    if matches!(
        operation,
        C220VectorScalarOperation::Maximum | C220VectorScalarOperation::Minimum
    ) {
        let first = to_f64(first_bits);
        let second = to_f64(second_bits);
        let choose_first = if first == second {
            match operation {
                C220VectorScalarOperation::Maximum => first_bits & SIGN <= second_bits & SIGN,
                C220VectorScalarOperation::Minimum => first_bits & SIGN >= second_bits & SIGN,
                C220VectorScalarOperation::LeakyRelu => unreachable!("not an extremum"),
                _ => unreachable!(),
            }
        } else {
            match operation {
                C220VectorScalarOperation::Maximum => first > second,
                C220VectorScalarOperation::Minimum => first < second,
                C220VectorScalarOperation::LeakyRelu => unreachable!("not an extremum"),
                _ => unreachable!(),
            }
        };
        let selected = if choose_first {
            first_bits
        } else {
            second_bits
        };
        return C220Fp16Outcome {
            bits: if mode == C220Fp16Mode::Saturating && is_infinite(selected) {
                selected & SIGN | MAX_FINITE
            } else {
                selected
            },
            status,
        };
    }

    if status.infinity_operand {
        let sign = match operation {
            C220VectorScalarOperation::Add => {
                if first_infinite {
                    first_bits & SIGN
                } else {
                    second_bits & SIGN
                }
            }
            C220VectorScalarOperation::Multiply => (first_bits ^ second_bits) & SIGN,
            C220VectorScalarOperation::LeakyRelu => {
                unreachable!("handled before binary evaluation")
            }
            _ => unreachable!(),
        };
        return C220Fp16Outcome {
            bits: sign
                | match mode {
                    C220Fp16Mode::Saturating => MAX_FINITE,
                    C220Fp16Mode::NonSaturating => EXPONENT,
                },
            status,
        };
    }

    let first = to_f64(first_bits);
    let second = to_f64(second_bits);
    let exact = match operation {
        C220VectorScalarOperation::Add => first + second,
        C220VectorScalarOperation::Multiply => first * second,
        C220VectorScalarOperation::LeakyRelu => unreachable!("handled before binary evaluation"),
        _ => unreachable!(),
    };
    status.overflow = exact.abs() >= OVERFLOW_MIDPOINT;
    status.underflow = exact != 0.0 && exact.abs() <= HALF_MIN_SUBNORMAL;
    let bits = if status.overflow {
        (if exact.is_sign_negative() { SIGN } else { 0 })
            | match mode {
                C220Fp16Mode::Saturating => MAX_FINITE,
                C220Fp16Mode::NonSaturating => EXPONENT,
            }
    } else {
        round_finite_to_f16(exact)
    };
    C220Fp16Outcome { bits, status }
}

pub(crate) fn is_nan(bits: u16) -> bool {
    bits & EXPONENT == EXPONENT && bits & FRACTION != 0
}

pub fn c220_fp16_to_fp32_bits(bits: u16) -> u32 {
    let sign = u32::from(bits & SIGN) << 16;
    let exponent = (bits & EXPONENT) >> 10;
    let fraction = u32::from(bits & FRACTION);
    match (exponent, fraction) {
        (31, 0) => sign | 0x7f80_0000,
        (31, _) => 0x7fff_ffff,
        (0, 0) => sign,
        (0, _) => {
            let shift = fraction.leading_zeros() - 21;
            sign | ((113 - shift) << 23) | (((fraction << shift) & 0x3ff) << 13)
        }
        _ => sign | ((u32::from(exponent) + 112) << 23) | (fraction << 13),
    }
}

pub(crate) fn is_infinite(bits: u16) -> bool {
    bits & !SIGN == EXPONENT
}

pub(crate) fn to_f64(bits: u16) -> f64 {
    let sign = if bits & SIGN == 0 { 1.0 } else { -1.0 };
    let exponent = i32::from((bits & EXPONENT) >> 10);
    let fraction = u32::from(bits & FRACTION);
    let magnitude = if exponent == 0 {
        f64::from(fraction) * 2_f64.powi(-24)
    } else if exponent == 31 {
        f64::INFINITY
    } else {
        f64::from(1024 + fraction) * 2_f64.powi(exponent - 25)
    };
    sign * magnitude
}

pub(crate) fn round_finite_to_f16(value: f64) -> u16 {
    let sign = if value.is_sign_negative() { SIGN } else { 0 };
    let magnitude = value.abs();
    if magnitude == 0.0 {
        return sign;
    }
    if magnitude < MIN_NORMAL {
        let fraction = (magnitude * 2_f64.powi(24)).round_ties_even() as u16;
        return sign | fraction;
    }
    let exponent = ((magnitude.to_bits() >> 52) & 0x7ff) as i32 - 1023;
    let significand = (magnitude * 2_f64.powi(10 - exponent)).round_ties_even() as u16;
    if significand == 2048 {
        return sign | (((exponent + 16) as u16) << 10);
    }
    sign | (((exponent + 15) as u16) << 10) | (significand - 1024)
}

#[cfg(test)]
mod tests {
    use super::*;
    use C220VectorScalarOperation::{Add, LeakyRelu, Maximum, Minimum, Multiply};

    #[test]
    fn add_rounding_is_independent_of_overflow_and_saturation() {
        for (first, second, nearest, truncated) in [
            (0x3c01, 0x1000, 0x3c02, 0x3c01),
            (0xbc01, 0x9000, 0xbc02, 0xbc01),
            (0x3c00, 0x9000, 0x3bff, 0x3bff),
            (1, 1, 2, 2),
            (0x0400, 0x8001, 0x03ff, 0x03ff),
            (0x8000, 0x8000, 0x8000, 0x8000),
            (0x3c00, 0xbc00, 0, 0),
        ] {
            for (rounding, expected) in [
                (C220Fp16AddRounding::NearestEven, nearest),
                (C220Fp16AddRounding::TowardZero, truncated),
            ] {
                assert_eq!(
                    evaluate_c220_fp16_add(first, second, C220Fp16Mode::NonSaturating, rounding)
                        .bits,
                    expected
                );
            }
        }
        for value in [0, 1, u32::MAX] {
            let rounding = C220Fp16AddRounding::from_model_value(value);
            for (mode, expected) in [
                (C220Fp16Mode::Saturating, 0x7bff),
                (C220Fp16Mode::NonSaturating, 0x7c00),
            ] {
                let result = evaluate_c220_fp16_add(0x7bff, 0x4c00, mode, rounding);
                assert_eq!(result.bits, expected);
                assert!(result.status.overflow);
            }
        }
    }

    #[test]
    fn f16_rounding_and_nonfinite_modes_keep_their_distinct_bit_patterns() {
        for (operation, first, second, mode, expected) in [
            (Add, 0x3c00, 0x1000, C220Fp16Mode::Saturating, 0x3c00),
            (Add, 0x3c01, 0x1000, C220Fp16Mode::Saturating, 0x3c02),
            (Add, 0x8000, 0x8000, C220Fp16Mode::Saturating, 0x8000),
            (Multiply, 0x0001, 0x3800, C220Fp16Mode::Saturating, 0),
            (LeakyRelu, 0xc400, 0x3800, C220Fp16Mode::Saturating, 0xc000),
            (Maximum, 0x8000, 0, C220Fp16Mode::Saturating, 0),
            (Minimum, 0x8000, 0, C220Fp16Mode::Saturating, 0x8000),
            (Add, 0x7c00, 0xfc00, C220Fp16Mode::Saturating, 0),
            (Add, 0x7c00, 0xfc00, C220Fp16Mode::NonSaturating, 0x7fff),
            (Multiply, 0x7c00, 0, C220Fp16Mode::NonSaturating, 0x7fff),
            (Add, 0x7bff, 0x7bff, C220Fp16Mode::Saturating, 0x7bff),
            (Add, 0x7bff, 0x7bff, C220Fp16Mode::NonSaturating, 0x7c00),
            (Maximum, 0x7c00, 0x3c00, C220Fp16Mode::Saturating, 0x7bff),
        ] {
            assert_eq!(
                evaluate_c220_fp16(operation, first, second, mode).bits,
                expected,
                "{operation:?}: {first:#06x}, {second:#06x}, {mode:?}"
            );
        }
        assert!(
            evaluate_c220_fp16(Multiply, 1, 0x3800, C220Fp16Mode::Saturating)
                .status
                .underflow
        );
        assert!(
            evaluate_c220_fp16(Add, 0x7bff, 0x7bff, C220Fp16Mode::Saturating)
                .status
                .overflow
        );
    }
}
