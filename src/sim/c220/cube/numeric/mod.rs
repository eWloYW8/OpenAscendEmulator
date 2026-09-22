mod exact;
mod fp16;
mod fp32;

use super::C220CubeExecutionError;
use crate::sim::c220::numeric::fp16::C220Fp16Mode;

pub(super) use fp16::{evaluate_bf16_f32_slice, evaluate_f16_f16_slice, evaluate_f16_f32_slice};
pub(super) use fp32::{evaluate_f32_f32_slice, evaluate_hf32_f32_slice};

const F16_SIGN: u16 = 0x8000;
const F16_EXPONENT: u16 = 0x7c00;
const F16_FRACTION: u16 = 0x03ff;
const F16_MAX_FINITE: u16 = 0x7bff;
const BF16_SIGN: u16 = 0x8000;
const BF16_EXPONENT: u16 = 0x7f80;
const BF16_FRACTION: u16 = 0x007f;
const BF16_MAX_FINITE: u16 = 0x7f7f;
const F32_SIGN: u32 = 0x8000_0000;
const F32_EXPONENT: u32 = 0x7f80_0000;
const F32_FRACTION: u32 = 0x007f_ffff;
const F32_MAX_FINITE: u32 = 0x7f7f_ffff;
const F32_CANONICAL_NAN: u32 = 0x7fff_ffff;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct C220CubeFpStatus {
    pub nan_operand: bool,
    pub infinity_operand: bool,
    pub invalid: bool,
    pub overflow: bool,
    pub underflow: bool,
}

impl C220CubeFpStatus {
    pub(super) fn merge(&mut self, other: Self) {
        self.nan_operand |= other.nan_operand;
        self.infinity_operand |= other.infinity_operand;
        self.invalid |= other.invalid;
        self.overflow |= other.overflow;
        self.underflow |= other.underflow;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct F32SliceOutcome {
    pub(super) bits: u32,
    pub(super) status: C220CubeFpStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct F16SliceOutcome {
    pub(super) bits: u16,
    pub(super) status: C220CubeFpStatus,
}

fn saturate_f16_nonfinite(bits: u16) -> u16 {
    if bits & F16_EXPONENT != F16_EXPONENT {
        return bits;
    }
    (bits & F16_SIGN)
        | if is_f16_infinite(bits) {
            F16_MAX_FINITE
        } else {
            0
        }
}

fn saturate_bf16_nonfinite(bits: u16) -> u16 {
    if bits & BF16_EXPONENT != BF16_EXPONENT {
        return bits;
    }
    (bits & BF16_SIGN)
        | if is_bf16_infinite(bits) {
            BF16_MAX_FINITE
        } else {
            0
        }
}

fn saturate_f32_nonfinite(bits: u32) -> u32 {
    if bits & F32_EXPONENT != F32_EXPONENT {
        return bits;
    }
    (bits & F32_SIGN)
        | if is_f32_infinite(bits) {
            F32_MAX_FINITE
        } else {
            0
        }
}

fn saturate_hf32_nonfinite(bits: u32) -> u32 {
    if bits & F32_EXPONENT != F32_EXPONENT {
        return bits;
    }
    (bits & F32_SIGN)
        | if is_f32_infinite(bits) {
            0x7f7f_f000
        } else {
            0
        }
}

const fn is_f16_zero(bits: u16) -> bool {
    bits & !F16_SIGN == 0
}

const fn is_f16_nan(bits: u16) -> bool {
    bits & F16_EXPONENT == F16_EXPONENT && bits & F16_FRACTION != 0
}

const fn is_f16_infinite(bits: u16) -> bool {
    bits & !F16_SIGN == F16_EXPONENT
}

const fn is_bf16_zero(bits: u16) -> bool {
    bits & !BF16_SIGN == 0
}

const fn is_bf16_nan(bits: u16) -> bool {
    bits & BF16_EXPONENT == BF16_EXPONENT && bits & BF16_FRACTION != 0
}

const fn is_bf16_infinite(bits: u16) -> bool {
    bits & !BF16_SIGN == BF16_EXPONENT
}

const fn is_f32_nan(bits: u32) -> bool {
    bits & F32_EXPONENT == F32_EXPONENT && bits & F32_FRACTION != 0
}

const fn is_f32_infinite(bits: u32) -> bool {
    bits & !F32_SIGN == F32_EXPONENT
}

const fn is_f32_zero(bits: u32) -> bool {
    bits & !F32_SIGN == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_slice_rounds_once_and_flushes_fp32_subnormals() {
        let mut left = [0_u16; 16];
        let mut right = [0_u16; 16];
        left[0] = 0x3c00;
        right[0] = 0x3c00;
        left[1] = 0x1000;
        right[1] = 0x1000;
        let outcome = evaluate_f16_f32_slice(left, right, 1, C220Fp16Mode::NonSaturating).unwrap();
        assert_eq!(outcome.bits, (1.0_f32 + 2_f32.powi(-22)).to_bits());

        let flushed = evaluate_f16_f32_slice(
            [F16_SIGN; 16],
            [0; 16],
            F32_SIGN | 1,
            C220Fp16Mode::NonSaturating,
        )
        .unwrap();
        assert_eq!(flushed.bits, F32_SIGN);
    }
}
