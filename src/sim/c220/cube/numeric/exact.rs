use std::cmp::Ordering;

use super::{
    BF16_EXPONENT, BF16_FRACTION, BF16_SIGN, C220CubeExecutionError, C220CubeFpStatus,
    C220Fp16Mode, F16_EXPONENT, F16_FRACTION, F16_MAX_FINITE, F16_SIGN, F32_EXPONENT, F32_FRACTION,
    F32_MAX_FINITE, F32_SIGN,
};

const EXACT_LIMBS: usize = 9;
const EXACT_SCALE_EXPONENT: i32 = -298;
const F32_MIN_NORMAL_BIT: u32 = 172;
const F32_RAW_EXPONENT_OFFSET: u32 = 171;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct ExactDyadic {
    negative: bool,
    magnitude: [u64; EXACT_LIMBS],
}

impl ExactDyadic {
    pub(super) fn add_f16(&mut self, bits: u16) -> Result<(), C220CubeExecutionError> {
        let exponent = (bits & F16_EXPONENT) >> 10;
        if exponent == 0 {
            return Ok(());
        }
        debug_assert_ne!(exponent, 0x1f);
        let (significand, exponent) = finite_f16_parts(bits);
        self.add_term(
            bits & F16_SIGN != 0,
            u64::from(significand),
            (exponent - EXACT_SCALE_EXPONENT) as u32,
        )
    }

    pub(super) fn add_f32(&mut self, bits: u32) -> Result<(), C220CubeExecutionError> {
        let exponent = (bits >> 23) & 0xff;
        if exponent == 0 {
            return Ok(());
        }
        debug_assert_ne!(exponent, 0xff);
        let (significand, exponent) = finite_f32_parts(bits);
        self.add_term(
            bits & F32_SIGN != 0,
            u64::from(significand),
            (exponent - EXACT_SCALE_EXPONENT) as u32,
        )
    }

    pub(super) fn add_f16_product(
        &mut self,
        first: u16,
        second: u16,
    ) -> Result<(), C220CubeExecutionError> {
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

    pub(super) fn add_bf16_product(
        &mut self,
        first: u16,
        second: u16,
    ) -> Result<(), C220CubeExecutionError> {
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

    pub(super) fn add_f32_product(
        &mut self,
        first: u32,
        second: u32,
    ) -> Result<(), C220CubeExecutionError> {
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

    pub(super) fn round_f32(self, negative_zero: bool) -> (u32, C220CubeFpStatus) {
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

    pub(super) fn round_f16(
        self,
        negative_zero: bool,
        mode: C220Fp16Mode,
    ) -> (u16, C220CubeFpStatus) {
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
