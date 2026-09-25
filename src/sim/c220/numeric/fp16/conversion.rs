use super::{C220Fp16Mode, C220Fp16Outcome, C220Fp16Status};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum C220Fp16Rounding {
    NearestEven = 0,
    NearestAway = 1,
    Down = 2,
    Up = 3,
    TowardZero = 4,
    Odd = 5,
}

/// FP32-to-FP16 conversion, including operand and range status.
pub fn c220_f32_to_fp16(
    bits: u32,
    rounding: C220Fp16Rounding,
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
                C220Fp16Rounding::NearestEven | C220Fp16Rounding::NearestAway => {
                    fraction >= 0x7f_f000
                }
                C220Fp16Rounding::Down => negative && fraction > 0x7f_e000,
                C220Fp16Rounding::Up => !negative && fraction > 0x7f_e000,
                C220Fp16Rounding::TowardZero | C220Fp16Rounding::Odd => false,
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
            C220Fp16Rounding::NearestEven => exponent == 102 && fraction != 0,
            C220Fp16Rounding::NearestAway => exponent == 102,
            C220Fp16Rounding::Down => negative && nonzero,
            C220Fp16Rounding::Up => !negative && nonzero,
            C220Fp16Rounding::TowardZero => false,
            C220Fp16Rounding::Odd => nonzero,
        };
        status.underflow = nonzero && !rounds_up;
        u16::from(rounds_up)
    };
    C220Fp16Outcome {
        bits: sign | magnitude,
        status,
    }
}

fn round_significand(value: u32, shift: u32, negative: bool, mode: C220Fp16Rounding) -> u32 {
    let retained = value >> shift;
    let discarded = value & ((1 << shift) - 1);
    let halfway = 1 << (shift - 1);
    let increment = match mode {
        C220Fp16Rounding::NearestEven => {
            discarded > halfway || (discarded == halfway && retained & 1 != 0)
        }
        C220Fp16Rounding::NearestAway => discarded >= halfway,
        C220Fp16Rounding::Down => negative && discarded != 0,
        C220Fp16Rounding::Up => !negative && discarded != 0,
        C220Fp16Rounding::TowardZero => false,
        C220Fp16Rounding::Odd => discarded != 0 && retained & 1 == 0,
    };
    retained + u32::from(increment)
}
