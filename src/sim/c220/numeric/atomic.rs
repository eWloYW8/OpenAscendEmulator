use crate::isa::c220::vector::scalar::C220VectorScalarOperation;
use crate::numeric::fp32::{Fp32VectorOperation, evaluate_fp32_value};
use crate::sim::c220::numeric::fixp::c220_fixp_f32_to_bf16;
use crate::sim::c220::numeric::fp16::{
    C220Fp16AddRounding, C220Fp16Mode, evaluate_c220_fp16, evaluate_c220_fp16_add,
};

pub(crate) fn combine_atomic(
    bytes: &mut [u8],
    previous: &[u8],
    data_type: u8,
    operation: u8,
    control: u64,
    rounding: C220Fp16AddRounding,
) {
    macro_rules! combine {
        ($ty:ty, $width:expr) => {
            for (next, old) in bytes
                .chunks_exact_mut($width)
                .zip(previous.chunks_exact($width))
            {
                let next_value = <$ty>::from_le_bytes(next.try_into().expect("lane width"));
                let old_value = <$ty>::from_le_bytes(old.try_into().expect("lane width"));
                let value = match operation {
                    0 => next_value.wrapping_add(old_value),
                    1 => next_value.max(old_value),
                    2 => next_value.min(old_value),
                    _ => unreachable!("atomic operation is decoded before execution"),
                };
                next.copy_from_slice(&value.to_le_bytes());
            }
        };
    }
    match data_type {
        1 | 6 => {
            let operation = match operation {
                0 => Fp32VectorOperation::Add,
                1 => Fp32VectorOperation::Maximum,
                2 => Fp32VectorOperation::Minimum,
                _ => unreachable!("atomic operation is decoded before execution"),
            };
            let width = if data_type == 1 { 4 } else { 2 };
            let expand = |lane: &[u8]| {
                if data_type == 1 {
                    u32::from_le_bytes(lane.try_into().expect("lane width"))
                } else {
                    u32::from(u16::from_le_bytes(lane.try_into().expect("lane width"))) << 16
                }
            };
            for (next, old) in bytes
                .chunks_exact_mut(width)
                .zip(previous.chunks_exact(width))
            {
                let result = evaluate_fp32_value(operation, expand(next), expand(old));
                if data_type == 1 {
                    next.copy_from_slice(&result.bits.to_le_bytes());
                } else {
                    let result = c220_fixp_f32_to_bf16(result.bits, control, 0);
                    next.copy_from_slice(&result.bits.to_le_bytes());
                }
            }
        }
        2 => {
            let operation = match operation {
                0 => C220VectorScalarOperation::Add,
                1 => C220VectorScalarOperation::Maximum,
                2 => C220VectorScalarOperation::Minimum,
                _ => unreachable!("atomic operation is decoded before execution"),
            };
            for (next, old) in bytes.chunks_exact_mut(2).zip(previous.chunks_exact(2)) {
                let first = u16::from_le_bytes(next.try_into().expect("lane width"));
                let second = u16::from_le_bytes(old.try_into().expect("lane width"));
                let mode = C220Fp16Mode::from_control_spr(control);
                let result = if operation == C220VectorScalarOperation::Add {
                    evaluate_c220_fp16_add(first, second, mode, rounding)
                } else {
                    evaluate_c220_fp16(operation, first, second, mode)
                };
                next.copy_from_slice(&result.bits.to_le_bytes());
            }
        }
        3 => combine!(i16, 2),
        4 => combine!(i32, 4),
        5 => combine!(i8, 1),
        0 | 7 => {}
        _ => unreachable!("atomic data type is a three-bit field"),
    }
}
