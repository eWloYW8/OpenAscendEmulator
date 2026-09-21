#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum F32ToS32Status {
    None,
    Overflow,
    NaN,
    Infinity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct F32ToS32Result {
    pub value: u32,
    pub status: F32ToS32Status,
}

pub fn f32_to_s32_truncate(bits: u32, wrap_overflow: bool) -> F32ToS32Result {
    let sign = bits >> 31 != 0;
    let exponent = (bits >> 23) & 0xff;
    let fraction = bits & 0x7f_ffff;
    if exponent == 0xff {
        if fraction != 0 {
            return F32ToS32Result {
                value: 0,
                status: F32ToS32Status::NaN,
            };
        }
        return F32ToS32Result {
            value: if wrap_overflow {
                if sign { u32::MAX } else { 1 }
            } else if sign {
                i32::MIN as u32
            } else {
                i32::MAX as u32
            },
            status: F32ToS32Status::Infinity,
        };
    }

    let value = f32::from_bits(bits);
    let overflow = !(-2_147_483_648.0..2_147_483_648.0).contains(&value);
    let result = if wrap_overflow && overflow {
        let significand = fraction | if exponent == 0 { 0 } else { 0x80_0000 };
        let magnitude = if exponent < 127 {
            0
        } else if exponent <= 150 {
            significand >> (150 - exponent)
        } else {
            significand.checked_shl(exponent - 150).unwrap_or(0)
        };
        if sign {
            magnitude.wrapping_neg()
        } else {
            magnitude
        }
    } else {
        (value as i32) as u32
    };
    F32ToS32Result {
        value: result,
        status: if overflow {
            F32ToS32Status::Overflow
        } else {
            F32ToS32Status::None
        },
    }
}

pub fn s32_to_f32_bits(bits: u32) -> u32 {
    ((bits as i32) as f32).to_bits()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_conversions_cover_rounding_and_exception_values() {
        assert_eq!(
            f32_to_s32_truncate(3.75_f32.to_bits(), false),
            F32ToS32Result {
                value: 3,
                status: F32ToS32Status::None
            }
        );
        assert_eq!(
            f32_to_s32_truncate((-3.75_f32).to_bits(), false).value,
            (-3_i32) as u32
        );
        assert_eq!(
            f32_to_s32_truncate(2_147_483_648.0_f32.to_bits(), false),
            F32ToS32Result {
                value: i32::MAX as u32,
                status: F32ToS32Status::Overflow
            }
        );
        assert_eq!(
            f32_to_s32_truncate(2_147_483_648.0_f32.to_bits(), true).value,
            i32::MIN as u32
        );
        assert_eq!(
            f32_to_s32_truncate(f32::NAN.to_bits(), false).status,
            F32ToS32Status::NaN
        );
        assert_eq!(
            f32_to_s32_truncate(f32::INFINITY.to_bits(), false).value,
            i32::MAX as u32
        );
        assert_eq!(s32_to_f32_bits((-3_i32) as u32), (-3.0_f32).to_bits());
    }
}
