use super::fixp::{C220FixpDequantActivation, c220_fixp_i32_to_i16};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220FixpQuantizedWidth {
    Bits4,
    Bits8,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct C220FixpRequantStatus {
    pub nan_factor: bool,
    pub infinity_factor: bool,
    pub negative_to_unsigned: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220FixpQuantizationInput {
    Int32 { after_preshift: i32 },
    Fp32 { bits: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FixpRequantOutcome {
    /// Int4 occupies the low nibble; Int8 occupies the full byte.
    pub bits: u8,
    pub value: i16,
    pub input: C220FixpQuantizationInput,
    pub scaled_fp32_bits: u32,
    pub rounded: i32,
    pub before_offset: i16,
    pub offset: i16,
    pub status: C220FixpRequantStatus,
}

/// Zero-bias integer requantization. Activation runs before the zero-point
/// offset, so ReLU does not imply a nonnegative final signed result.
pub fn c220_fixp_requantize(
    input: i32,
    factor: u64,
    activation: C220FixpDequantActivation,
    width: C220FixpQuantizedWidth,
) -> C220FixpRequantOutcome {
    let input = if factor & (1 << 36) != 0 {
        i32::from(c220_fixp_i32_to_i16(input, factor, false).value)
    } else {
        input
    };
    quantize_scaled(
        input as f32,
        C220FixpQuantizationInput::Int32 {
            after_preshift: input,
        },
        factor,
        activation,
        width,
    )
}

/// FP32 quantization does not apply the integer pre-shift. The sign bit selects
/// activation, including for negative zero and signed NaNs.
pub fn c220_fixp_quantize_f32(
    bits: u32,
    factor: u64,
    activation: C220FixpDequantActivation,
    width: C220FixpQuantizedWidth,
) -> C220FixpRequantOutcome {
    quantize_scaled(
        f32::from_bits(bits),
        C220FixpQuantizationInput::Fp32 { bits },
        factor,
        activation,
        width,
    )
}

fn quantize_scaled(
    operand: f32,
    input: C220FixpQuantizationInput,
    factor: u64,
    activation: C220FixpDequantActivation,
    width: C220FixpQuantizedWidth,
) -> C220FixpRequantOutcome {
    let negative = operand.is_sign_negative();
    let mut status = C220FixpRequantStatus::default();
    let scale_bits = factor as u32 & 0xffff_e000;
    let operand = f64::from(operand);
    let product = if negative {
        match activation {
            C220FixpDequantActivation::None => operand * f64::from(f32::from_bits(scale_bits)),
            C220FixpDequantActivation::Relu => 0.0,
            C220FixpDequantActivation::NegativeSlope(bits) => {
                operand * f64::from(f32::from_bits(bits & 0xffff_e000))
            }
        }
    } else {
        let magnitude = scale_bits & 0x7fff_ffff;
        status.nan_factor = magnitude > 0x7f80_0000;
        status.infinity_factor = magnitude == 0x7f80_0000;
        let scale = if status.nan_factor {
            0.0
        } else {
            f32::from_bits(scale_bits)
        };
        operand * f64::from(scale)
    };
    // NaN survives this clamp and becomes zero at integer conversion.
    let scaled = product.clamp(-f64::from(f32::MAX), f64::from(f32::MAX)) as f32;
    let rounded = scaled.round_ties_even() as i32;
    let (before_offset, offset, lower, upper) = match width {
        C220FixpQuantizedWidth::Bits4 => {
            let offset = ((factor >> 37) & 31) as i16;
            let offset = if offset & 16 != 0 {
                offset - 32
            } else {
                offset
            };
            (rounded.clamp(-16, 15) as i16, offset, -8, 7)
        }
        C220FixpQuantizedWidth::Bits8 => {
            let offset = ((factor >> 37) & 511) as i16;
            let offset = if offset & 256 != 0 {
                offset - 512
            } else {
                offset
            };
            let (lower, upper) = if factor & (1 << 46) != 0 {
                (-128, 127)
            } else {
                (0, 255)
            };
            (rounded.clamp(-256, 255) as i16, offset, lower, upper)
        }
    };
    let shifted = before_offset + offset;
    status.negative_to_unsigned =
        width == C220FixpQuantizedWidth::Bits8 && lower == 0 && shifted < 0;
    let value = shifted.clamp(lower, upper);
    let bits = match width {
        C220FixpQuantizedWidth::Bits4 => value as u8 & 15,
        C220FixpQuantizedWidth::Bits8 => value as u8,
    };
    C220FixpRequantOutcome {
        bits,
        value,
        input,
        scaled_fp32_bits: scaled.to_bits(),
        rounded,
        before_offset,
        offset,
        status,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp32_quantization_keeps_fraction_sign_and_ignores_integer_preshift() {
        use C220FixpDequantActivation::{None, Relu};
        use C220FixpQuantizedWidth::{Bits4, Bits8};
        for (input, activation, expected) in [
            (2.5_f32, None, 2),
            (3.5, None, 4),
            (-3.5, None, -4),
            (f32::INFINITY, None, 127),
            (f32::NEG_INFINITY, Relu, 0),
        ] {
            let result = c220_fixp_quantize_f32(
                input.to_bits(),
                (1 << 46) | (1 << 36) | 0x3f80_0000,
                activation,
                Bits8,
            );
            assert_eq!(result.value, expected);
            assert_eq!(
                result.input,
                C220FixpQuantizationInput::Fp32 {
                    bits: input.to_bits()
                }
            );
        }
        for (bits, nan_factor) in [
            (0, true),
            (0x8000_0000, false),
            (0x7fc0_0000, true),
            (0xffc0_0000, false),
        ] {
            let result = c220_fixp_quantize_f32(bits, 0x7fc0_0000, None, Bits4);
            assert_eq!(result.value, 0);
            assert_eq!(result.status.nan_factor, nan_factor);
        }
    }

    #[test]
    fn rounding_offset_and_signedness_follow_separate_stages() {
        use C220FixpDequantActivation::{None, Relu};
        use C220FixpQuantizedWidth::{Bits4, Bits8};
        for (input, factor, activation, width, expected) in [
            (5, 0x3f00_0000, None, Bits8, 2),
            (7, 0x3f00_0000, None, Bits8, 4),
            (1000, (1 << 46) | (256 << 37) | 0x3f80_0000, None, Bits8, -1),
            (
                -1000,
                (1 << 46) | (255 << 37) | 0x3f80_0000,
                None,
                Bits8,
                -1,
            ),
            (-3, (1 << 46) | (511 << 37) | 0x3f80_0000, Relu, Bits8, -1),
            (1000, (16 << 37) | 0x3f80_0000, None, Bits4, -1),
            (-1000, (15 << 37) | 0x3f80_0000, None, Bits4, -1),
            (-1, 0x3f80_0000, None, Bits8, 0),
            (0, 0x7f80_0000, None, Bits8, 0),
        ] {
            assert_eq!(
                c220_fixp_requantize(input, factor, activation, width).value,
                expected
            );
        }
        assert!(
            c220_fixp_requantize(-1, 0x3f80_0000, None, Bits8)
                .status
                .negative_to_unsigned
        );
        assert!(
            c220_fixp_requantize(1, 0x7fc0_0000, None, Bits8)
                .status
                .nan_factor
        );
        assert!(
            !c220_fixp_requantize(-1, 0x7fc0_0000, None, Bits8)
                .status
                .nan_factor
        );
        assert_eq!(
            c220_fixp_requantize(-1, (1 << 46) | 0x3f80_0000, None, Bits8).bits,
            255
        );
        assert_eq!(c220_fixp_requantize(-1, 0x3f80_0000, None, Bits4).bits, 15);
    }
}
