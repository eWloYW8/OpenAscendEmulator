use crate::architecture::Architecture;
use crate::isa::scalar::ScalarInstruction;
use crate::numeric::fp32::Fp32ValueStatus;
use crate::sim::common::scalar::{
    ScalarCompareRegisterStep, ScalarCompareStep, ScalarInstructionStep, ScalarMachine,
    ScalarMachineError,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220ScalarFp32CompareOutcome {
    pub instruction: ScalarInstructionStep,
    pub status: Fp32ValueStatus,
    pub prior_spr2: u64,
    pub spr2: u64,
}

pub(crate) fn supports(word: u32) -> bool {
    matches!(
        ScalarInstruction::from_word(Architecture::Dav2201, word),
        Some(
            ScalarInstruction::ScalarCompare { dtype_field: 2, .. }
                | ScalarInstruction::ScalarCompareRegister { dtype_field: 2, .. }
        )
    )
}

pub fn execute_fp32_compare_word(
    machine: &mut ScalarMachine,
    pc: u64,
    word: u32,
) -> Result<C220ScalarFp32CompareOutcome, ScalarMachineError> {
    let unsupported = || ScalarMachineError::UnsupportedWord { pc, word };
    if machine.architecture() != Architecture::Dav2201 {
        return Err(unsupported());
    }
    let (condition_field, first_source_register, second_source_register, destination) =
        match ScalarInstruction::from_word(Architecture::Dav2201, word) {
            Some(ScalarInstruction::ScalarCompare {
                dtype_field: 2,
                condition_field,
                first_source_register,
                second_source_register,
            }) => (
                condition_field,
                first_source_register,
                second_source_register,
                None,
            ),
            Some(ScalarInstruction::ScalarCompareRegister {
                dtype_field: 2,
                condition_field,
                first_source_register,
                second_source_register,
                destination_register,
            }) => (
                condition_field,
                first_source_register,
                second_source_register,
                Some(destination_register),
            ),
            _ => return Err(unsupported()),
        };
    let first_source_value = machine.xregs()[usize::from(first_source_register)];
    let second_source_value = machine.xregs()[usize::from(second_source_register)];
    let first = f32::from_bits(first_source_value as u32);
    let second = f32::from_bits(second_source_value as u32);
    let status = if condition_field <= 5 {
        Fp32ValueStatus {
            nan_operand: first.is_nan() || second.is_nan(),
            infinity_operand: first.is_infinite() || second.is_infinite(),
            ..Fp32ValueStatus::default()
        }
    } else {
        Fp32ValueStatus::default()
    };
    let value = u64::from(
        !status.nan_operand
            && match condition_field {
                0 => first == second,
                1 => first != second,
                2 => first < second,
                3 => first > second,
                4 => first >= second,
                5 => first <= second,
                _ => false,
            },
    );
    let prior_spr2 = machine.spr2();
    let spr2 = if status.nan_operand || status.infinity_operand {
        (prior_spr2 & 0xffff_ffff_ff00_ffff) | (((pc >> 2) & 0xff) << 16) | 0x2000
    } else {
        prior_spr2
    };
    let instruction = if let Some(destination_register) = destination {
        let prior_destination_value = machine.xregs()[usize::from(destination_register)];
        machine.set_xreg(destination_register, value)?;
        ScalarInstructionStep::CompareRegister(ScalarCompareRegisterStep {
            pc,
            word,
            dtype_field: 2,
            condition_field,
            destination_register,
            prior_destination_value,
            first_source_register,
            first_source_value,
            second_source_register,
            second_source_value,
            value,
        })
    } else {
        let prior_spr11 = machine.spr_value(11);
        machine.set_spr_value(11, value)?;
        ScalarInstructionStep::Compare(ScalarCompareStep {
            pc,
            word,
            dtype_field: 2,
            condition_field,
            first_source_register,
            first_source_value,
            second_source_register,
            second_source_value,
            prior_spr11,
            spr11: value,
        })
    };
    machine.set_spr_value(2, spr2)?;
    Ok(C220ScalarFp32CompareOutcome {
        instruction,
        status,
        prior_spr2,
        spr2,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comparisons_handle_nan_signed_zero_and_both_destinations() {
        for (first, second, results, exceptional) in [
            (0x3f80_0000, 0x4000_0000, [0, 1, 1, 0, 0, 1], false),
            (0x8000_0000, 0, [1, 0, 0, 0, 1, 1], false),
            (0x7fc0_1234, 0x3f80_0000, [0; 6], true),
            (0x3f80_0000, 0x7f80_0001, [0; 6], true),
            (0x7f80_0000, 0x7f80_0000, [1, 0, 0, 0, 1, 1], true),
        ] {
            for condition in 0..8_u32 {
                for register_destination in [false, true] {
                    let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
                    machine.set_xreg(1, 0xaaaa_aaaa_0000_0000 | first).unwrap();
                    machine.set_xreg(2, 0xbbbb_bbbb_0000_0000 | second).unwrap();
                    machine.set_spr_value(2, 0xab_0100).unwrap();
                    let word = 0x0082_110e | (condition << 4) | u32::from(register_destination);
                    let outcome = execute_fp32_compare_word(&mut machine, 0x104, word).unwrap();
                    let expected = results.get(condition as usize).copied().unwrap_or(0);
                    let result = if register_destination {
                        machine.xregs()[1]
                    } else {
                        machine.spr_value(11).unwrap()
                    };
                    assert_eq!(result, expected);
                    assert_eq!(
                        outcome.spr2,
                        if exceptional && condition <= 5 {
                            0x41_2100
                        } else {
                            0xab_0100
                        }
                    );
                    if !register_destination {
                        let selection = machine.execute_select_word(0x108, 0x0086_1109).unwrap();
                        assert_eq!(
                            selection.value,
                            if expected != 0 {
                                0xaaaa_aaaa_0000_0000 | first
                            } else {
                                0xbbbb_bbbb_0000_0000 | second
                            }
                        );
                    }
                }
            }
        }
    }
}
