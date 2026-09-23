use super::fp16::{C220Fp16Mode, C220Fp16Outcome, C220Fp16Status};
use crate::numeric::fp32::{Fp32ValueOutcome, Fp32ValueStatus};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct C220FixpInt16Status {
    pub positive_saturation: bool,
    pub negative_saturation: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FixpInt16Outcome {
    pub value: i16,
    /// Range status is captured before activation clamps negative values.
    pub status: C220FixpInt16Status,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220FixpDequantActivation {
    None,
    Relu,
    NegativeSlope(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FixpDequantFp16Outcome {
    pub input_after_preshift: i32,
    pub scaled_fp32_bits: u32,
    pub factor_status: C220Fp16Status,
    pub conversion: C220Fp16Outcome,
}

/// Zero-bias integer dequantization. Activation selects the multiplier before
/// FP16 rounding; CTRL does not change this operation's saturation policy.
pub fn c220_fixp_i32_to_f16(
    input: i32,
    factor: u64,
    activation: C220FixpDequantActivation,
) -> C220FixpDequantFp16Outcome {
    let input = if factor & (1 << 36) != 0 {
        i32::from(c220_fixp_i32_to_i16(input, factor, false).value)
    } else {
        input
    };
    let factor_bits = factor as u32 & 0xffff_e000;
    let magnitude = factor_bits & 0x7fff_ffff;
    let factor_status = C220Fp16Status {
        nan_operand: magnitude > 0x7f80_0000,
        infinity_operand: magnitude == 0x7f80_0000,
        ..C220Fp16Status::default()
    };
    let scale = if factor_status.nan_operand {
        0.0
    } else {
        f32::from_bits(factor_bits)
    };
    let operand = f64::from(input as f32);
    let product = if input < 0 {
        match activation {
            C220FixpDequantActivation::None => operand * f64::from(scale),
            C220FixpDequantActivation::Relu => 0.0,
            C220FixpDequantActivation::NegativeSlope(bits) => {
                operand * f64::from(f32::from_bits(bits & 0xffff_e000))
            }
        }
    } else {
        operand * f64::from(scale)
    };
    let scaled = if product.is_nan() || product > f64::from(f32::MAX) {
        f32::MAX
    } else if product < -f64::from(f32::MAX) {
        -f32::MAX
    } else {
        product as f32
    };
    C220FixpDequantFp16Outcome {
        input_after_preshift: input,
        scaled_fp32_bits: scaled.to_bits(),
        factor_status,
        conversion: c220_fixp_f32_to_f16(
            scaled.to_bits(),
            C220FixpRoundMode::NearestEven,
            C220Fp16Mode::Saturating,
        ),
    }
}

/// Integer dequantization with an arithmetic shift of 1–16 bits, followed by
/// signed saturation and optional ReLU. Only factor bits 32–35 select the shift.
pub fn c220_fixp_i32_to_i16(input: i32, factor: u64, relu: bool) -> C220FixpInt16Outcome {
    let shift = ((factor >> 32) & 15) + 1;
    let shifted = input >> shift;
    let status = C220FixpInt16Status {
        positive_saturation: shifted > i32::from(i16::MAX),
        negative_saturation: shifted < i32::from(i16::MIN),
    };
    let value = shifted.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
    C220FixpInt16Outcome {
        value: if relu { value.max(0) } else { value },
        status,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FixpBf16Outcome {
    pub bits: u16,
    pub status: Fp32ValueStatus,
}

/// Mode 16 rounds to BF16 before applying activation code 1. Other activation
/// codes do not change the converted value or access slope operands.
pub fn c220_fixp_f32_to_bf16(bits: u32, control: u64, activation: u8) -> C220FixpBf16Outcome {
    let magnitude = bits & 0x7fff_ffff;
    let sign = ((bits >> 16) as u16) & 0x8000;
    let saturating = control & (1 << 48) == 0;
    let limit = if saturating { 0x7f7f } else { 0x7f80 };
    let mut status = Fp32ValueStatus::default();
    let mut result = if magnitude > 0x7f80_0000 {
        status.nan_operand = true;
        if saturating { 0 } else { 0x7fff }
    } else if magnitude == 0x7f80_0000 {
        status.infinity_operand = true;
        sign | limit
    } else {
        let rounded = (magnitude + 0x7fff + ((magnitude >> 16) & 1)) >> 16;
        if rounded == 0x7f80 {
            status.overflow = true;
            sign | limit
        } else {
            status.underflow = rounded == 0 && magnitude != 0;
            sign | rounded as u16
        }
    };
    if activation == 1 {
        let magnitude = result & 0x7fff;
        status.nan_operand |= magnitude > 0x7f80;
        status.infinity_operand |= magnitude == 0x7f80;
        result = if magnitude > 0x7f80 {
            0x7fff
        } else if result & 0x8000 != 0 {
            0
        } else {
            result
        };
    }
    C220FixpBf16Outcome {
        bits: result,
        status,
    }
}

/// Mode-zero FP32 output: only activation code 1 applies ReLU. Other codes
/// preserve every input bit, including NaN payloads and signed zero.
pub fn c220_fixp_f32_output(bits: u32, activation: u8) -> Fp32ValueOutcome {
    let mut status = Fp32ValueStatus::default();
    let result = if activation == 1 {
        let magnitude = bits & 0x7fff_ffff;
        status.nan_operand = magnitude > 0x7f80_0000;
        status.infinity_operand = magnitude == 0x7f80_0000;
        if status.nan_operand {
            0x7fff_ffff
        } else if bits >> 31 != 0 {
            0
        } else {
            bits
        }
    } else {
        bits
    };
    Fp32ValueOutcome {
        bits: result,
        status,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum C220FixpRoundMode {
    NearestEven = 0,
    NearestAway = 1,
    Down = 2,
    Up = 3,
    TowardZero = 4,
    Odd = 5,
}

/// FP32-to-FP16 conversion, including operand and range status.
pub fn c220_fixp_f32_to_f16(
    bits: u32,
    rounding: C220FixpRoundMode,
    mode: C220Fp16Mode,
) -> C220Fp16Outcome {
    let negative = bits >> 31 != 0;
    let sign = ((bits >> 16) as u16) & 0x8000;
    let exponent = (bits >> 23) & 255;
    let fraction = bits & 0x7f_ffff;
    let mut status = C220Fp16Status::default();
    let overflow_bits = match mode {
        C220Fp16Mode::Saturating => 0x7bff,
        C220Fp16Mode::NonSaturating => 0x7c00,
    };
    if exponent == 255 {
        if fraction != 0 {
            status.nan_operand = true;
            return C220Fp16Outcome {
                bits: match mode {
                    C220Fp16Mode::Saturating => 0,
                    C220Fp16Mode::NonSaturating => 0x7fff,
                },
                status,
            };
        }
        status.infinity_operand = true;
        return C220Fp16Outcome {
            bits: sign | overflow_bits,
            status,
        };
    }

    let overflow = exponent > 142
        || (exponent == 142
            && match rounding {
                C220FixpRoundMode::NearestEven | C220FixpRoundMode::NearestAway => {
                    fraction >= 0x7f_f000
                }
                C220FixpRoundMode::Down => negative && fraction > 0x7f_e000,
                C220FixpRoundMode::Up => !negative && fraction > 0x7f_e000,
                C220FixpRoundMode::TowardZero | C220FixpRoundMode::Odd => false,
            });
    if overflow {
        status.overflow = true;
        return C220Fp16Outcome {
            bits: sign | overflow_bits,
            status,
        };
    }

    let magnitude = if exponent > 112 {
        let significand = fraction | 0x80_0000;
        let rounded = round_significand(significand, 13, negative, rounding);
        // Adding the retained significand also propagates a carry into exponent.
        ((exponent - 113) * 1024 + rounded).min(0x7bff) as u16
    } else if exponent > 102 {
        round_significand(fraction | 0x80_0000, 126 - exponent, negative, rounding) as u16
    } else {
        let nonzero = bits & 0x7fff_ffff != 0;
        let rounds_up = match rounding {
            C220FixpRoundMode::NearestEven => exponent == 102 && fraction != 0,
            C220FixpRoundMode::NearestAway => exponent == 102,
            C220FixpRoundMode::Down => negative && nonzero,
            C220FixpRoundMode::Up => !negative && nonzero,
            C220FixpRoundMode::TowardZero => false,
            C220FixpRoundMode::Odd => nonzero,
        };
        status.underflow = nonzero && !rounds_up;
        u16::from(rounds_up)
    };
    C220Fp16Outcome {
        bits: sign | magnitude,
        status,
    }
}

fn round_significand(value: u32, shift: u32, negative: bool, mode: C220FixpRoundMode) -> u32 {
    let retained = value >> shift;
    let discarded = value & ((1 << shift) - 1);
    let halfway = 1 << (shift - 1);
    let increment = match mode {
        C220FixpRoundMode::NearestEven => {
            discarded > halfway || (discarded == halfway && retained & 1 != 0)
        }
        C220FixpRoundMode::NearestAway => discarded >= halfway,
        C220FixpRoundMode::Down => negative && discarded != 0,
        C220FixpRoundMode::Up => !negative && discarded != 0,
        C220FixpRoundMode::TowardZero => false,
        C220FixpRoundMode::Odd => discarded != 0 && retained & 1 == 0,
    };
    retained + u32::from(increment)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp16_dequantization_preserves_stage_order_and_special_values() {
        use C220FixpDequantActivation::{NegativeSlope, None, Relu};
        for (input, factor, activation, expected) in [
            (3, 0x3f80_0000, None, 0x4200),
            (-3, 0x3f80_0000, Relu, 0),
            (-3, 0x4000_0000, NegativeSlope(0x3f00_0000), 0xbe00),
            (3, 0x4000_0000, NegativeSlope(0x3f00_0000), 0x4600),
            (3, 0x3f80_1fff, None, 0x4200),
            (-1, 0x7fc0_0000, None, 0x8000),
            (0, 0x7f80_0000, None, 0x7bff),
            (-1, 0x3f80_0000, NegativeSlope(0x7fc0_0000), 0x7bff),
            (i32::MAX, (1 << 36) | 0x3f80_0000, None, 0x7800),
            (65536, (1 << 36) | (15 << 32) | 0x3f80_0000, None, 0x3c00),
        ] {
            assert_eq!(
                c220_fixp_i32_to_f16(input, factor, activation)
                    .conversion
                    .bits,
                expected
            );
        }
        let nan = c220_fixp_i32_to_f16(-1, 0x7fc0_0000, None);
        assert!(nan.factor_status.nan_operand);
        assert!(!nan.conversion.status.nan_operand);
        let infinity = c220_fixp_i32_to_f16(0, 0x7f80_0000, None);
        assert!(infinity.factor_status.infinity_operand);
        assert!(infinity.conversion.status.overflow);
        assert_eq!(infinity.scaled_fp32_bits, f32::MAX.to_bits());
    }

    #[test]
    fn integer_dequantization_shift_saturation_and_activation() {
        for shift in 1..=16 {
            let factor = (shift - 1) << 32;
            for input in [
                i32::MIN,
                -65537,
                -65536,
                -3,
                -1,
                0,
                1,
                65534,
                65536,
                i32::MAX,
            ] {
                let shifted = i64::from(input).div_euclid(1_i64 << shift);
                for relu in [false, true] {
                    let result = c220_fixp_i32_to_i16(input, factor, relu);
                    let clamped = shifted.clamp(i64::from(i16::MIN), i64::from(i16::MAX));
                    assert_eq!(
                        result.value,
                        if relu { clamped.max(0) } else { clamped } as i16
                    );
                    assert_eq!(
                        result.status.positive_saturation,
                        shifted > i64::from(i16::MAX)
                    );
                    assert_eq!(
                        result.status.negative_saturation,
                        shifted < i64::from(i16::MIN)
                    );
                    assert_eq!(
                        result,
                        c220_fixp_i32_to_i16(input, factor | !(15_u64 << 32), relu)
                    );
                }
            }
        }
    }

    #[test]
    fn bf16_rounds_before_relu_and_distinguishes_operand_from_range_status() {
        for (input, saturated, unbounded, overflow, underflow) in [
            (0, 0, 0, false, false),
            (0x8000_0000, 0x8000, 0x8000, false, false),
            (1, 0, 0, false, true),
            (0x0000_8000, 0, 0, false, true),
            (0x0000_8001, 1, 1, false, false),
            (0x007f_8000, 0x0080, 0x0080, false, false),
            (0x3f80_8000, 0x3f80, 0x3f80, false, false),
            (0xbf81_8000, 0xbf82, 0xbf82, false, false),
            (0x7f7f_7fff, 0x7f7f, 0x7f7f, false, false),
            (0x7f7f_8000, 0x7f7f, 0x7f80, true, false),
            (0xff7f_ffff, 0xff7f, 0xff80, true, false),
            (0x7f80_0000, 0x7f7f, 0x7f80, false, false),
            (0xff80_0000, 0xff7f, 0xff80, false, false),
            (0xff80_0001, 0, 0x7fff, false, false),
        ] {
            for (control, converted) in [(0, saturated), (1 << 48, unbounded)] {
                for activation in 0..8 {
                    let result = c220_fixp_f32_to_bf16(input, control, activation);
                    let expected = if activation == 1 && converted & 0x8000 != 0 {
                        0
                    } else {
                        converted
                    };
                    assert_eq!(
                        result.bits, expected,
                        "input={input:#x}, control={control:#x}, activation={activation}"
                    );
                    assert_eq!(result.status.overflow, overflow);
                    assert_eq!(result.status.underflow, underflow);
                    assert_eq!(result.status.nan_operand, input & 0x7fff_ffff > 0x7f80_0000);
                    assert_eq!(
                        result.status.infinity_operand,
                        input & 0x7fff_ffff == 0x7f80_0000
                            || (activation == 1 && converted & 0x7fff == 0x7f80)
                    );
                }
            }
        }
    }

    #[test]
    fn fp32_output_preserves_bits_except_explicit_relu() {
        for (input, rectified) in [
            (0x8000_0000, 0),
            (0xff80_0000, 0),
            (0x7f80_0000, 0x7f80_0000),
            (0xff80_0001, 0x7fff_ffff),
            (0x7fc0_1234, 0x7fff_ffff),
            (0x8000_0001, 0),
            (1, 1),
            (0x3f80_0000, 0x3f80_0000),
        ] {
            for activation in 0..8 {
                let result = c220_fixp_f32_output(input, activation);
                assert_eq!(result.bits, if activation == 1 { rectified } else { input });
                assert_eq!(
                    result.status.nan_operand,
                    activation == 1 && input & 0x7fff_ffff > 0x7f80_0000
                );
                assert_eq!(
                    result.status.infinity_operand,
                    activation == 1 && input & 0x7fff_ffff == 0x7f80_0000
                );
            }
        }
    }

    fn c220_fixp_f32_to_f16(bits: u32, rounding: C220FixpRoundMode) -> C220Fp16Outcome {
        super::c220_fixp_f32_to_f16(bits, rounding, C220Fp16Mode::Saturating)
    }

    #[test]
    fn rounding_modes_cover_ties_tiny_values_and_saturation_status() {
        use C220FixpRoundMode::*;
        let modes = [NearestEven, NearestAway, Down, Up, TowardZero, Odd];
        for (bits, expected) in [
            (
                1.0_f32.to_bits() | (1 << 12),
                [0x3c00, 0x3c01, 0x3c00, 0x3c01, 0x3c00, 0x3c01],
            ),
            (
                (-1.0_f32).to_bits() | (1 << 12),
                [0xbc00, 0xbc01, 0xbc01, 0xbc00, 0xbc00, 0xbc01],
            ),
            (102 << 23, [0, 1, 0, 1, 0, 1]),
            (1, [0, 0, 0, 1, 0, 1]),
        ] {
            for (mode, expected) in modes.into_iter().zip(expected) {
                let value = c220_fixp_f32_to_f16(bits, mode);
                assert_eq!(value.bits, expected);
                assert_eq!(value.status.underflow, bits < (103 << 23) && expected == 0);
            }
        }
        for mode in modes {
            assert_eq!(c220_fixp_f32_to_f16(0x8000_0000, mode).bits, 0x8000);
            assert_eq!(c220_fixp_f32_to_f16(65504_f32.to_bits(), mode).bits, 0x7bff);
            assert!(
                !c220_fixp_f32_to_f16(65504_f32.to_bits(), mode)
                    .status
                    .overflow
            );
            assert!(
                c220_fixp_f32_to_f16(65536_f32.to_bits(), mode)
                    .status
                    .overflow
            );
            let nan = c220_fixp_f32_to_f16(0xff80_0001, mode);
            assert_eq!(nan.bits, 0);
            assert!(nan.status.nan_operand);
            let inf = c220_fixp_f32_to_f16(f32::NEG_INFINITY.to_bits(), mode);
            assert_eq!(inf.bits, 0xfbff);
            assert!(inf.status.infinity_operand && !inf.status.overflow);
        }
    }
}
