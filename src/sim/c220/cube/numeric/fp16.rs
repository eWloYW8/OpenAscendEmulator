use super::exact::ExactDyadic;
use super::{
    BF16_SIGN, C220CubeExecutionError, C220CubeFpStatus, C220Fp16Mode, F16_EXPONENT, F16_SIGN,
    F16SliceOutcome, F32_CANONICAL_NAN, F32_EXPONENT, F32_SIGN, F32SliceOutcome, is_bf16_infinite,
    is_bf16_nan, is_bf16_zero, is_f16_infinite, is_f16_nan, is_f16_zero, is_f32_infinite,
    is_f32_nan, saturate_bf16_nonfinite, saturate_f16_nonfinite, saturate_f32_nonfinite,
};

pub(in crate::sim::c220::cube) fn evaluate_f16_f16_slice(
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

pub(in crate::sim::c220::cube) fn evaluate_f16_f32_slice(
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

pub(in crate::sim::c220::cube) fn evaluate_bf16_f32_slice(
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
