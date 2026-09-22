mod execute;
use crate::architecture::Architecture;
use crate::isa::c220::scalar::{C220ScalarConversion, C220ScalarConversionHint};
use crate::isa::c220::vector::C220MovemaskHint;
use crate::numeric::conversion::{F32ToS32Status, f32_to_s32_truncate, s32_to_f32_bits};
use crate::sim::common::scalar::{ScalarMachine, ScalarMachineError, ScalarStep};

pub(crate) mod address;
pub(crate) mod bus;
pub mod timing;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MovemaskStep {
    pub pc: u64,
    pub word: u32,
    pub source_register: u8,
    pub source_value: u64,
    pub destination_spr: u16,
    pub prior_destination_value: Option<u64>,
}

pub fn execute_movemask(
    machine: &mut ScalarMachine,
    pc: u64,
    word: u32,
) -> Result<C220MovemaskStep, ScalarMachineError> {
    if machine.architecture() != Architecture::Dav2201 {
        return Err(ScalarMachineError::UnsupportedWord { pc, word });
    }
    let hint = C220MovemaskHint::from_word(word)
        .ok_or(ScalarMachineError::UnsupportedWord { pc, word })?;
    let source_value = machine.xregs()[usize::from(hint.source_register)];
    let prior_destination_value = machine.spr_value(hint.destination_spr);
    machine.set_spr_value(hint.destination_spr, source_value)?;
    Ok(C220MovemaskStep {
        pc,
        word,
        source_register: hint.source_register,
        source_value,
        destination_spr: hint.destination_spr,
        prior_destination_value,
    })
}

pub fn execute_conversion_word(
    machine: &mut ScalarMachine,
    pc: u64,
    word: u32,
) -> Result<ScalarStep, ScalarMachineError> {
    if machine.architecture() != Architecture::Dav2201 {
        return Err(ScalarMachineError::UnsupportedWord { pc, word });
    }
    let hint = C220ScalarConversionHint::from_word(word)
        .ok_or(ScalarMachineError::UnsupportedWord { pc, word })?;
    let source_value = machine.xregs()[usize::from(hint.source_register)];
    let prior_destination_value = machine.xregs()[usize::from(hint.destination_register)];
    let prior_spr2 = machine.spr2();
    let spr3 = machine
        .spr_value(3)
        .ok_or(ScalarMachineError::SprValueUnavailable { pc, spr: 3 })?;
    let outcome = execute_scalar_conversion(hint.conversion, source_value, spr3, prior_spr2, pc);
    machine.set_xreg(hint.destination_register, outcome.value)?;
    machine.set_spr_value(2, outcome.spr2)?;
    Ok(ScalarStep {
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
