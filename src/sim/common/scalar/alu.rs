use crate::architecture::Architecture;
use crate::isa::scalar::{ScalarInstruction, ScalarKey8Operation};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScalarIntegerOutcome {
    pub destination_register: u8,
    pub value: u64,
    pub signed_overflow: bool,
    pub spr2: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum ScalarIntegerError {
    #[error("the decoder hint is not a supported scalar integer arithmetic operation")]
    UnsupportedHint,
    #[error("scalar key-8 immediate {0:#x} exceeds 12 bits")]
    ImmediateOutOfRange(u16),
}

pub fn evaluate_scalar_integer_immediate(
    hint: ScalarInstruction,
    source_value: u64,
    runtime_isa_pc: u64,
    prior_spr2: u64,
) -> Result<ScalarIntegerOutcome, ScalarIntegerError> {
    let ScalarInstruction::ScalarKey8 {
        operation,
        destination_register: Some(destination_register),
        encoded_immediate,
        ..
    } = hint
    else {
        return Err(ScalarIntegerError::UnsupportedHint);
    };
    if encoded_immediate > 0xfff {
        return Err(ScalarIntegerError::ImmediateOutOfRange(encoded_immediate));
    }

    let source_signed = source_value as i64;
    let (value, signed_overflow) = match operation {
        ScalarKey8Operation::AddImmediate => {
            source_signed.overflowing_add(i64::from(encoded_immediate))
        }
        ScalarKey8Operation::SubtractImmediate => {
            source_signed.overflowing_sub(i64::from(encoded_immediate))
        }
        ScalarKey8Operation::MultiplyImmediate => {
            let signed_immediate = if encoded_immediate & 0x800 != 0 {
                i64::from(encoded_immediate) - 0x1000
            } else {
                i64::from(encoded_immediate)
            };
            source_signed.overflowing_mul(signed_immediate)
        }
        ScalarKey8Operation::DcPreload => return Err(ScalarIntegerError::UnsupportedHint),
    };

    let spr2 = update_overflow_spr2(prior_spr2, runtime_isa_pc, signed_overflow);
    Ok(ScalarIntegerOutcome {
        destination_register,
        value: value as u64,
        signed_overflow,
        spr2,
    })
}

pub(crate) const fn update_overflow_spr2(
    prior_spr2: u64,
    runtime_isa_pc: u64,
    signed_overflow: bool,
) -> u64 {
    if signed_overflow {
        (prior_spr2 & 0xffff_ffff_ff00_ffff) | (((runtime_isa_pc >> 2) & 0xff) << 16) | 0x10
    } else {
        prior_spr2
    }
}

pub(crate) const fn update_neg_overflow_spr2(
    architecture: Architecture,
    prior_spr2: u64,
    runtime_isa_pc: u64,
    signed_overflow: bool,
) -> u64 {
    if !signed_overflow {
        return prior_spr2;
    }
    let pc_byte = ((runtime_isa_pc >> 2) & 0xff) << 16;
    match architecture {
        Architecture::Dav2201 => (prior_spr2 & 0xffff_ffff_ff00_ffff) | pc_byte | 8,
        Architecture::Dav3510 => (prior_spr2 & 0xffff_ffff_ffff_f007) | pc_byte | 8,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::architecture::Architecture;

    fn hint(word: u32) -> ScalarInstruction {
        ScalarInstruction::from_word(Architecture::Dav3510, word).unwrap()
    }

    #[test]
    fn addition_uses_unsigned_immediate_and_sticky_overflow_spr() {
        let add = hint(0x0802_1001);
        let initial_spr2 = 0xabcd_ef12_3456_7801;
        let overflow =
            evaluate_scalar_integer_immediate(add, i64::MAX as u64, 0x1234, initial_spr2).unwrap();
        assert_eq!(overflow.destination_register, 1);
        assert_eq!(overflow.value, i64::MIN as u64);
        assert!(overflow.signed_overflow);
        assert_eq!(
            overflow.spr2,
            (initial_spr2 & 0xffff_ffff_ff00_ffff) | (((0x1234 >> 2) & 0xff) << 16) | 0x10
        );
        let regular = evaluate_scalar_integer_immediate(add, u64::MAX, 0, initial_spr2).unwrap();
        assert_eq!(regular.value, 0);
        assert!(!regular.signed_overflow);
        assert_eq!(regular.spr2, initial_spr2);
    }

    #[test]
    fn subtraction_uses_unsigned_immediate_and_detects_underflow() {
        let subtract = hint(0x0882_1001);
        let result = evaluate_scalar_integer_immediate(subtract, i64::MIN as u64, 0x4, 0).unwrap();
        assert_eq!(result.value, i64::MAX as u64);
        assert!(result.signed_overflow);
        assert_eq!(result.spr2, 0x0001_0010);
    }

    #[test]
    fn multiplication_sign_extends_only_its_immediate() {
        let negative = hint(0x0842_1800);
        let ordinary = evaluate_scalar_integer_immediate(negative, 2, 0, 0).unwrap();
        assert_eq!(ordinary.value, (-4096_i64) as u64);
        assert!(!ordinary.signed_overflow);

        let minus_one = hint(0x0842_1fff);
        let overflow = evaluate_scalar_integer_immediate(minus_one, i64::MIN as u64, 0, 0).unwrap();
        assert_eq!(overflow.value, i64::MIN as u64);
        assert!(overflow.signed_overflow);
        assert_eq!(overflow.spr2, 0x10);
    }

    #[test]
    fn rejects_unmodeled_instructions_and_invalid_immediates() {
        assert_eq!(
            evaluate_scalar_integer_immediate(hint(0x08c0_0000), 0, 0, 0),
            Err(ScalarIntegerError::UnsupportedHint)
        );
        assert_eq!(
            evaluate_scalar_integer_immediate(
                ScalarInstruction::ScalarKey8 {
                    operation: ScalarKey8Operation::AddImmediate,
                    destination_register: Some(1),
                    source_register: 1,
                    encoded_immediate: 0x1000,
                },
                0,
                0,
                0
            ),
            Err(ScalarIntegerError::ImmediateOutOfRange(0x1000))
        );
    }
}
