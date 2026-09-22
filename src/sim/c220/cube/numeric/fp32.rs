use super::exact::ExactDyadic;
use super::{
    C220CubeExecutionError, C220CubeFpStatus, C220Fp16Mode, F32_CANONICAL_NAN, F32_EXPONENT,
    F32_FRACTION, F32_SIGN, F32SliceOutcome, is_f32_infinite, is_f32_nan, is_f32_zero,
    saturate_f32_nonfinite, saturate_hf32_nonfinite,
};

pub(in crate::sim::c220::cube) fn evaluate_f32_f32_slice(
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

pub(in crate::sim::c220::cube) fn evaluate_hf32_f32_slice(
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
