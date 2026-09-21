use crate::isa::c220::scalar::C220ScalarConversion;
use crate::numeric::conversion::{F32ToS32Status, f32_to_s32_truncate, s32_to_f32_bits};

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
