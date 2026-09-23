use super::fp16::{C220Fp16Mode, C220Fp16Outcome, C220Fp16Status};

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
