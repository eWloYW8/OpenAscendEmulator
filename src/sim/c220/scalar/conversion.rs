use crate::architecture::Architecture;
use crate::isa::c220::scalar::{C220ScalarConversion, C220ScalarConversionHint};
use crate::numeric::conversion::{F32ToS32Status, f32_to_s32_truncate, s32_to_f32_bits};
use crate::sim::common::scalar::{ScalarMachine, ScalarMachineError, ScalarStep};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220ScalarConversionStep {
    /// Computed result; consult `destination_written` before treating it as a register update.
    pub step: ScalarStep,
    pub missing_source_register: Option<u8>,
    pub destination_written: bool,
}

pub fn execute_conversion_word(
    machine: &mut ScalarMachine,
    pc: u64,
    word: u32,
) -> Result<C220ScalarConversionStep, ScalarMachineError> {
    if machine.architecture() != Architecture::Dav2201 {
        return Err(ScalarMachineError::UnsupportedWord { pc, word });
    }
    let hint = C220ScalarConversionHint::from_word(word)
        .ok_or(ScalarMachineError::UnsupportedWord { pc, word })?;
    let source = machine.xreg_value(hint.source_register);
    let destination = machine.xreg_value(hint.destination_register);
    let source_value = source.unwrap_or(0);
    let prior_destination_value = destination.unwrap_or(0);
    let prior_spr2 = machine.spr2();
    let spr3 = machine
        .spr_value(3)
        .ok_or(ScalarMachineError::SprValueUnavailable { pc, spr: 3 })?;
    let outcome = execute_scalar_conversion(hint.conversion, source_value, spr3, prior_spr2, pc);
    if destination.is_some() {
        machine.set_xreg(hint.destination_register, outcome.value)?;
    }
    machine.set_spr_value(2, outcome.spr2)?;
    Ok(C220ScalarConversionStep {
        missing_source_register: source.is_none().then_some(hint.source_register),
        destination_written: destination.is_some(),
        step: ScalarStep {
            pc,
            word,
            destination_register: hint.destination_register,
            prior_destination_value,
            source_register: Some(hint.source_register),
            source_value: Some(source_value),
            second_source_register: None,
            second_source_value: None,
            value: outcome.value,
            signed_overflow: false,
            prior_spr2,
            spr2: outcome.spr2,
        },
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220ScalarConversionOutcome {
    pub value: u64,
    pub spr2: u64,
    pub status: F32ToS32Status,
}

pub fn execute_scalar_conversion(
    conversion: C220ScalarConversion,
    source_value: u64,
    spr3: u64,
    prior_spr2: u64,
    pc: u64,
) -> C220ScalarConversionOutcome {
    match conversion {
        C220ScalarConversion::F32ToS32Truncate => {
            let result = f32_to_s32_truncate(source_value as u32, spr3 & (1 << 59) != 0);
            let status_flag = match result.status {
                F32ToS32Status::None => 0,
                F32ToS32Status::Overflow => 0x20,
                F32ToS32Status::NaN | F32ToS32Status::Infinity => 0x2000,
            };
            let spr2 = if status_flag == 0 {
                prior_spr2
            } else {
                (prior_spr2 & 0xffff_ffff_ff00_ffff) | (((pc >> 2) & 0xff) << 16) | status_flag
            };
            C220ScalarConversionOutcome {
                value: u64::from(result.value),
                spr2,
                status: result.status,
            }
        }
        C220ScalarConversion::S32ToF32 => C220ScalarConversionOutcome {
            value: u64::from(s32_to_f32_bits(source_value as u32)),
            spr2: prior_spr2,
            status: F32ToS32Status::None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversion_reports_exception_in_spr2_without_losing_prior_flags() {
        for source_register in [0_u32, 31, 32, 33, 63] {
            for destination_register in [0_u32, 31, 32, 33, 63] {
                let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
                if source_register <= 32 {
                    machine
                        .set_xreg(source_register as u8, 3.75_f32.to_bits() as u64)
                        .unwrap();
                }
                let word = 0x0200_0583
                    | ((source_register & 31) << 12)
                    | ((destination_register & 31) << 17)
                    | (source_register & 32)
                    | ((destination_register & 32) << 1);
                let result = execute_conversion_word(&mut machine, 0x100, word).unwrap();
                let expected = if source_register <= 32 { 3 } else { 0 };
                assert_eq!(result.step.value, expected);
                assert_eq!(
                    result.missing_source_register,
                    (source_register > 32).then_some(source_register as u8)
                );
                assert_eq!(result.destination_written, destination_register <= 32);
                assert_eq!(
                    machine.xreg_value(destination_register as u8),
                    (destination_register <= 32).then_some(expected)
                );
            }
        }
        let overflow = execute_scalar_conversion(
            C220ScalarConversion::F32ToS32Truncate,
            u64::from(2_147_483_648.0_f32.to_bits()),
            0,
            0x100,
            0x104,
        );
        assert_eq!(overflow.value, i32::MAX as u32 as u64);
        assert_eq!(overflow.spr2, 0x41_0120);
        assert_eq!(overflow.status, F32ToS32Status::Overflow);

        let convert_back = execute_scalar_conversion(
            C220ScalarConversion::S32ToF32,
            (-3_i32) as u32 as u64,
            0,
            overflow.spr2,
            0x108,
        );
        assert_eq!(convert_back.value, u64::from((-3.0_f32).to_bits()));
        assert_eq!(convert_back.spr2, overflow.spr2);
    }
}
