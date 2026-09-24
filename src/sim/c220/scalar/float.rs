use crate::architecture::Architecture;
use crate::isa::scalar::{ScalarInstruction, ScalarKey0Operation};
use crate::numeric::fp32::{
    Fp32ValueOutcome, Fp32ValueStatus, Fp32VectorOperation, evaluate_fp32_fused_multiply_add,
    evaluate_fp32_value,
};
use crate::sim::common::scalar::{ScalarMachine, ScalarMachineError, ScalarStep};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220ScalarFp32Outcome {
    pub step: ScalarStep,
    pub status: Fp32ValueStatus,
}

pub(crate) fn supports(word: u32) -> bool {
    FloatInstruction::decode(word).is_some()
}

#[derive(Clone, Copy)]
enum FloatOperation {
    Binary(Fp32VectorOperation),
    MultiplyAdd,
    Negate,
    Absolute,
    SquareRoot,
}

struct FloatInstruction {
    operation: FloatOperation,
    destination: u8,
    first: u8,
    second: Option<u8>,
}

impl FloatInstruction {
    fn decode(word: u32) -> Option<Self> {
        let instruction = ScalarInstruction::from_word(Architecture::Dav2201, word)?;
        let (operation, destination, first, second) = match instruction {
            ScalarInstruction::ScalarKey0 {
                dtype_field: 2,
                operation,
                destination_register,
                first_source_register,
                second_source_register,
            } => {
                let operation = match operation {
                    ScalarKey0Operation::Add => FloatOperation::Binary(Fp32VectorOperation::Add),
                    ScalarKey0Operation::Subtract => {
                        FloatOperation::Binary(Fp32VectorOperation::Subtract)
                    }
                    ScalarKey0Operation::Multiply => {
                        FloatOperation::Binary(Fp32VectorOperation::Multiply)
                    }
                    ScalarKey0Operation::Divide => {
                        FloatOperation::Binary(Fp32VectorOperation::Divide)
                    }
                    ScalarKey0Operation::Minimum => {
                        FloatOperation::Binary(Fp32VectorOperation::Minimum)
                    }
                    ScalarKey0Operation::Maximum => {
                        FloatOperation::Binary(Fp32VectorOperation::Maximum)
                    }
                    ScalarKey0Operation::MultiplyAdd => FloatOperation::MultiplyAdd,
                    _ => return None,
                };
                (
                    operation,
                    destination_register,
                    first_source_register,
                    Some(second_source_register),
                )
            }
            ScalarInstruction::ScalarKey2Negate {
                dtype_field: 2,
                destination_register,
                source_register,
            } => (
                FloatOperation::Negate,
                destination_register,
                source_register,
                None,
            ),
            ScalarInstruction::ScalarKey2Absolute {
                dtype_field: 2,
                destination_register,
                source_register,
            } => (
                FloatOperation::Absolute,
                destination_register,
                source_register,
                None,
            ),
            ScalarInstruction::ScalarKey2IntegerSqrt {
                dtype_field: 2,
                destination_register,
                source_register,
            } => (
                FloatOperation::SquareRoot,
                destination_register,
                source_register,
                None,
            ),
            _ => return None,
        };
        Some(Self {
            operation,
            destination,
            first,
            second,
        })
    }
}

pub fn execute_fp32_word(
    machine: &mut ScalarMachine,
    pc: u64,
    word: u32,
) -> Result<C220ScalarFp32Outcome, ScalarMachineError> {
    let unsupported = || ScalarMachineError::UnsupportedWord { pc, word };
    if machine.architecture() != Architecture::Dav2201 {
        return Err(unsupported());
    }
    let hint = FloatInstruction::decode(word).ok_or_else(unsupported)?;
    let read = |register: u8| {
        machine
            .xregs()
            .get(usize::from(register))
            .copied()
            .ok_or_else(unsupported)
    };
    let first = read(hint.first)?;
    let second = hint.second.map(read).transpose()?;
    let prior_destination_value = read(hint.destination)?;
    let prior_spr2 = machine.spr2();
    let result = match hint.operation {
        FloatOperation::Binary(operation) => {
            evaluate_fp32_value(operation, first as u32, second.unwrap() as u32)
        }
        FloatOperation::MultiplyAdd => evaluate_fp32_fused_multiply_add(
            first as u32,
            second.unwrap() as u32,
            prior_destination_value as u32,
        ),
        FloatOperation::Absolute => {
            evaluate_fp32_value(Fp32VectorOperation::Absolute, first as u32, 0)
        }
        FloatOperation::Negate | FloatOperation::SquareRoot => {
            let bits = first as u32;
            let magnitude = bits & 0x7fff_ffff;
            let mut status = Fp32ValueStatus {
                nan_operand: magnitude > 0x7f80_0000,
                infinity_operand: magnitude == 0x7f80_0000,
                ..Fp32ValueStatus::default()
            };
            let bits = if matches!(hint.operation, FloatOperation::Negate) {
                if status.nan_operand || status.infinity_operand {
                    0
                } else {
                    bits ^ 0x8000_0000
                }
            } else {
                status.invalid = bits >> 31 != 0 && magnitude < 0x7f80_0000;
                if status.nan_operand || (bits >> 31 != 0 && magnitude != 0) {
                    0x7fff_ffff
                } else {
                    f32::from_bits(bits).sqrt().to_bits()
                }
            };
            Fp32ValueOutcome { bits, status }
        }
    };
    let signed_overflow = matches!(hint.operation, FloatOperation::Negate) && first == 1_u64 << 63;
    let flags = if matches!(hint.operation, FloatOperation::Negate) {
        if signed_overflow { 8 } else { 0 }
    } else {
        (if result.status.overflow { 0x20 } else { 0 })
            | if result.status.underflow { 0x40 } else { 0 }
            | if result.status.nan_operand || result.status.infinity_operand {
                0x2000
            } else {
                0
            }
    };
    let spr2 = if flags == 0 {
        prior_spr2
    } else {
        (prior_spr2 & 0xffff_ffff_ff00_ffff) | (((pc >> 2) & 0xff) << 16) | flags
    };
    let value = u64::from(result.bits);
    machine.set_xreg(hint.destination, value)?;
    machine.set_spr_value(2, spr2)?;
    Ok(C220ScalarFp32Outcome {
        step: ScalarStep {
            pc,
            word,
            destination_register: hint.destination,
            prior_destination_value,
            source_register: Some(hint.first),
            source_value: Some(first),
            second_source_register: hint.second,
            second_source_value: second,
            value,
            signed_overflow,
            prior_spr2,
            spr2,
        },
        status: result.status,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unary_operations_keep_architecture_specific_flags_and_results() {
        for (opcode, source, expected, flags) in [
            (0x80, 0x3f80_0000_u64, 0xbf80_0000, 0),
            (0x80, 0x8000_0000, 0, 0),
            (0x80, 0x7f80_0000, 0, 0),
            (0x80, 0xff80_0000, 0, 0),
            (0x80, 0x7fc0_1234, 0, 0),
            (0x80, 1_u64 << 63, 0x8000_0000, 8),
            (0x100, 0xbf80_0000, 0x3f80_0000, 0),
            (0x100, 0xff80_0000, 0x7f80_0000, 0x2000),
            (0x100, 0x7fc0_1234, 0x7fff_ffff, 0x2000),
            (0, 0x4080_0000, 0x4000_0000, 0),
            (0, 0x8000_0000, 0x8000_0000, 0),
            (0, 0xbf80_0000, 0x7fff_ffff, 0),
            (0, 0xff80_0000, 0x7fff_ffff, 0x2000),
            (0, 0x7f80_0000, 0x7f80_0000, 0x2000),
            (0, 1, 0x1a35_04f3, 0),
        ] {
            let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
            machine.set_xreg(1, source).unwrap();
            machine.set_spr_value(2, 0xab_0100).unwrap();
            let outcome = execute_fp32_word(&mut machine, 0x104, 0x0282_1000 | opcode).unwrap();
            assert_eq!(outcome.step.value, expected);
            assert_eq!(outcome.step.second_source_register, None);
            assert_eq!(outcome.step.second_source_value, None);
            assert_eq!(outcome.step.signed_overflow, flags == 8);
            assert_eq!(
                outcome.step.spr2,
                if flags == 0 {
                    0xab_0100
                } else {
                    0x41_0100 | flags
                }
            );
            if opcode == 0x80 && matches!(source, 0x7f80_0000 | 0xff80_0000) {
                assert!(outcome.status.infinity_operand);
            }
            if opcode == 0 && source == 0xbf80_0000 {
                assert!(outcome.status.invalid);
            }
        }
    }

    #[test]
    fn arithmetic_zero_extends_results_and_accumulates_exception_flags() {
        for (opcode, first, second, expected, flags) in [
            (1, 0x3f80_0000, 0x4000_0000, 0x4040_0000, 0),
            (2, 0x3f80_0000, 0x4000_0000, 0xbf80_0000, 0),
            (3, 0x3fc0_0000, 0x4000_0000, 0x4040_0000, 0),
            (1, 0x7f7f_ffff, 0x7f7f_ffff, 0x7f80_0000, 0x20),
            (3, 1, 0x3f00_0000, 0, 0x40),
            (1, 0x7fc0_1234, 0x3f80_0000, 0x7fff_ffff, 0x2000),
            (1, 0x7f80_0000, 0xff80_0000, 0x7fff_ffff, 0x2000),
            (3, 0, 0x7f80_0000, 0x7fff_ffff, 0x2000),
            (3, 0x8000_0000, 0x4000_0000, 0x8000_0000, 0),
            (5, 0x4040_0000, 0x4000_0000, 0x3fc0_0000, 0),
            (5, 0x7f7f_ffff, 0x3f80_0000, 0x7f7f_ffff, 0),
            (5, 0x7f7f_ffff, 0x3f00_0000, 0x7f80_0000, 0x20),
            (5, 0x8000_0001, 0x4000_0000, 0x8000_0000, 0x40),
            (5, 0x3f80_0000, 0x8000_0000, 0xff80_0000, 0x20),
            (5, 0, 0, 0x7fff_ffff, 0x20),
            (5, 0x7f80_0000, 0x7f80_0000, 0x7fff_ffff, 0x2000),
            (7, 0x8000_0000, 0, 0, 0),
            (8, 0, 0x8000_0000, 0x8000_0000, 0),
            (7, 0x7fc0_1234, 0x3f80_0000, 0x7fff_ffff, 0x2000),
            (8, 0x7f80_0000, 0x3f80_0000, 0x3f80_0000, 0x2000),
        ] {
            let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
            machine.set_xreg(1, 0xffff_ffff_0000_0000 | first).unwrap();
            machine.set_xreg(2, second).unwrap();
            machine.set_spr_value(2, 0x1234_56ab_0180).unwrap();
            let outcome = execute_fp32_word(&mut machine, 0x104, 0x0082_1100 | opcode).unwrap();
            assert_eq!(outcome.step.value, expected);
            assert_eq!(machine.xregs()[1], expected);
            assert_eq!(
                outcome.step.spr2,
                if flags == 0 {
                    0x1234_56ab_0180
                } else {
                    0x1234_5641_0180 | flags
                }
            );
            assert_eq!(outcome.status.overflow, flags == 0x20);
            assert_eq!(outcome.status.underflow, flags == 0x40);
        }
        let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
        let before = machine.clone();
        for word in [0x0082_1106, 0x0082_1141, 0x0082_1121, 0x0082_1111] {
            assert!(execute_fp32_word(&mut machine, 0, word).is_err());
            assert_eq!(machine, before);
        }
    }
}
