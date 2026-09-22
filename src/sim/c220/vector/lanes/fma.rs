use crate::isa::c220::vector::scalar::C220VectorScalarOperation;
use crate::numeric::fp32::{Fp32ValueStatus, Fp32VectorOperation, evaluate_fp32_value};
use crate::sim::c220::numeric::fp16::{C220Fp16Mode, C220Fp16Status, evaluate_c220_fp16};

pub(crate) fn evaluate_c220_f16_mla(
    first: u16,
    second: u16,
    addend: u16,
    mode: C220Fp16Mode,
) -> (u16, C220Fp16Status) {
    let product = evaluate_c220_fp16(C220VectorScalarOperation::Multiply, first, second, mode);
    let sum = evaluate_c220_fp16(C220VectorScalarOperation::Add, product.bits, addend, mode);
    (sum.bits, product.status.merge(sum.status))
}

pub(crate) fn evaluate_c220_fp32_mla(
    first: u32,
    second: u32,
    addend: u32,
) -> (u32, Fp32ValueStatus) {
    let product = evaluate_fp32_value(Fp32VectorOperation::Multiply, first, second);
    let sum = evaluate_fp32_value(Fp32VectorOperation::Add, product.bits, addend);
    (sum.bits, product.status.merge(sum.status))
}

pub(crate) fn evaluate_c220_mixed_mla(
    first: u16,
    second: u16,
    addend: u32,
) -> (u32, Fp32ValueStatus) {
    let first = f16_to_f32_bits(first);
    let second = f16_to_f32_bits(second);
    let product = evaluate_fp32_value(Fp32VectorOperation::Multiply, first, second);
    let sum = evaluate_fp32_value(Fp32VectorOperation::Add, product.bits, addend);
    let mut status = product.status.merge(sum.status);
    if status.nan_operand
        || status.infinity_operand
        || status.zero_times_infinity
        || status.opposite_infinities
    {
        return (sum.bits, status);
    }
    let exact = (f32::from_bits(first) as f64) * (f32::from_bits(second) as f64)
        + f32::from_bits(addend) as f64;
    let value = exact as f32;
    status.overflow = value.is_infinite() && exact.is_finite();
    status.underflow = exact != 0.0 && value == 0.0;
    (value.to_bits(), status)
}

fn f16_to_f32_bits(bits: u16) -> u32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = u32::from((bits >> 10) & 0x1f);
    let fraction = u32::from(bits & 0x03ff);
    match (exponent, fraction) {
        (0, 0) => sign,
        (0, _) => {
            let highest = 31 - fraction.leading_zeros();
            let exponent = highest + 103;
            let mantissa = (fraction << (23 - highest)) & 0x007f_ffff;
            sign | (exponent << 23) | mantissa
        }
        (0x1f, 0) => sign | 0x7f80_0000,
        (0x1f, _) => sign | 0x7fc0_0000,
        _ => sign | ((exponent + 112) << 23) | (fraction << 13),
    }
}
