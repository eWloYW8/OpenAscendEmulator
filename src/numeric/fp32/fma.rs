use super::{
    ABS_MASK, CANONICAL_NAN_BITS, Fp32ValueOutcome, Fp32ValueStatus, INFINITY_BITS, SIGN_BIT,
};

pub fn evaluate_fp32_fused_multiply_add(
    first: u32,
    second: u32,
    accumulator: u32,
) -> Fp32ValueOutcome {
    let first_abs = first & ABS_MASK;
    let second_abs = second & ABS_MASK;
    let accumulator_abs = accumulator & ABS_MASK;
    let first_infinite = first_abs == INFINITY_BITS;
    let second_infinite = second_abs == INFINITY_BITS;
    let accumulator_infinite = accumulator_abs == INFINITY_BITS;
    let product_sign = (first ^ second) & SIGN_BIT;
    let product = f64::from(f32::from_bits(first)) * f64::from(f32::from_bits(second));
    let opposite_infinities = accumulator_infinite
        && ((product_sign ^ accumulator) & SIGN_BIT != 0)
        && (first_infinite
            || second_infinite
            || product.abs() >= f64::from_bits(0x47ef_ffff_f000_0000));
    let mut status = Fp32ValueStatus {
        nan_operand: first_abs > INFINITY_BITS
            || second_abs > INFINITY_BITS
            || accumulator_abs > INFINITY_BITS,
        infinity_operand: first_infinite || second_infinite || accumulator_infinite,
        zero_times_infinity: (first_abs == 0 && second_infinite)
            || (second_abs == 0 && first_infinite),
        opposite_infinities,
        ..Fp32ValueStatus::default()
    };
    status.invalid = status.zero_times_infinity || opposite_infinities;
    if status.nan_operand || status.invalid {
        return Fp32ValueOutcome {
            bits: CANONICAL_NAN_BITS,
            status,
        };
    }
    if status.infinity_operand {
        return Fp32ValueOutcome {
            bits: if accumulator_infinite {
                accumulator
            } else {
                product_sign | INFINITY_BITS
            },
            status,
        };
    }
    let result = f32::from_bits(first).mul_add(f32::from_bits(second), f32::from_bits(accumulator));
    status.overflow = result.is_infinite();
    status.underflow = result == 0.0 && product + f64::from(f32::from_bits(accumulator)) != 0.0;
    Fp32ValueOutcome {
        bits: result.to_bits(),
        status,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fused_rounding_and_special_values_keep_distinct_status() {
        for (first, second, accumulator, expected) in [
            (0x3f80_0001, 0x3f7f_fffe, 0xbf80_0000, 0xa880_0000),
            (0x7f7f_ffff, 0x4000_0000, 0xff7f_ffff, 0x7f7f_ffff),
            (0x7f80_0000, 0, 0x3f80_0000, CANONICAL_NAN_BITS),
            (0x7f80_0000, 0x3f80_0000, 0xff80_0000, CANONICAL_NAN_BITS),
            (0x7f7f_ffff, 0x4000_0000, 0xff80_0000, CANONICAL_NAN_BITS),
            (0x3f80_0000, 0x3f80_0000, 0xff80_0000, 0xff80_0000),
            (0xff80_0000, 0xbf80_0000, 0, 0x7f80_0000),
            (0x7fc0_1234, 0, 0, CANONICAL_NAN_BITS),
            (SIGN_BIT, 0x3f80_0000, SIGN_BIT, SIGN_BIT),
            (SIGN_BIT, 0x3f80_0000, 0, 0),
            (1, 0x3f00_0000, 0, 0),
            (2, 0x3f00_0000, 0, 1),
        ] {
            assert_eq!(
                evaluate_fp32_fused_multiply_add(first, second, accumulator).bits,
                expected
            );
        }
        assert!(
            evaluate_fp32_fused_multiply_add(1, 0x3f00_0000, 0)
                .status
                .underflow
        );
        assert!(
            !evaluate_fp32_fused_multiply_add(2, 0x3f00_0000, 0)
                .status
                .underflow
        );
        assert!(
            evaluate_fp32_fused_multiply_add(0x7f7f_ffff, 0x4000_0000, 0)
                .status
                .overflow
        );
        let invalid = evaluate_fp32_fused_multiply_add(0x7f80_0000, 0, 0);
        assert!(invalid.status.invalid && invalid.status.infinity_operand);
        assert!(!invalid.status.nan_operand);
    }
}
