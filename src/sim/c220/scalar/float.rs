use crate::architecture::Architecture;
use crate::isa::scalar::{ScalarInstruction, ScalarKey0Operation};
use crate::numeric::fp32::{
    Fp32ValueStatus, Fp32VectorOperation, evaluate_fp32_fused_multiply_add, evaluate_fp32_value,
};
use crate::sim::common::scalar::{ScalarMachine, ScalarMachineError, ScalarStep};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220ScalarFp32Outcome {
    pub step: ScalarStep,
    pub status: Fp32ValueStatus,
}

pub(crate) fn supports(word: u32) -> bool {
    matches!(
        ScalarInstruction::from_word(Architecture::Dav2201, word),
        Some(ScalarInstruction::ScalarKey0 {
            dtype_field: 2,
            operation: ScalarKey0Operation::Add
                | ScalarKey0Operation::Subtract
                | ScalarKey0Operation::Multiply
                | ScalarKey0Operation::MultiplyAdd,
            ..
        })
    )
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
    let Some(ScalarInstruction::ScalarKey0 {
        dtype_field: 2,
        operation,
        destination_register,
        first_source_register,
        second_source_register,
    }) = ScalarInstruction::from_word(Architecture::Dav2201, word)
    else {
        return Err(unsupported());
    };
    let operation = match operation {
        ScalarKey0Operation::Add => Some(Fp32VectorOperation::Add),
        ScalarKey0Operation::Subtract => Some(Fp32VectorOperation::Subtract),
        ScalarKey0Operation::Multiply => Some(Fp32VectorOperation::Multiply),
        ScalarKey0Operation::MultiplyAdd => None,
        _ => return Err(unsupported()),
    };
    let read = |register: u8| {
        machine
            .xregs()
            .get(usize::from(register))
            .copied()
            .ok_or_else(unsupported)
    };
    let first = read(first_source_register)?;
    let second = read(second_source_register)?;
    let prior_destination_value = read(destination_register)?;
    let prior_spr2 = machine.spr2();
    let result = match operation {
        Some(operation) => evaluate_fp32_value(operation, first as u32, second as u32),
        None => evaluate_fp32_fused_multiply_add(
            first as u32,
            second as u32,
            prior_destination_value as u32,
        ),
    };
    let flags = if result.status.overflow { 0x20 } else { 0 }
        | if result.status.underflow { 0x40 } else { 0 }
        | if result.status.nan_operand || result.status.infinity_operand {
            0x2000
        } else {
            0
        };
    let spr2 = if flags == 0 {
        prior_spr2
    } else {
        (prior_spr2 & 0xffff_ffff_ff00_ffff) | (((pc >> 2) & 0xff) << 16) | flags
    };
    let value = u64::from(result.bits);
    machine.set_xreg(destination_register, value)?;
    machine.set_spr_value(2, spr2)?;
    Ok(C220ScalarFp32Outcome {
        step: ScalarStep {
            pc,
            word,
            destination_register,
            prior_destination_value,
            source_register: Some(first_source_register),
            source_value: Some(first),
            second_source_register: Some(second_source_register),
            second_source_value: Some(second),
            value,
            signed_overflow: false,
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
