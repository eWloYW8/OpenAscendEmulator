use crate::numeric::fp32::{Fp32VectorOperation, evaluate_fp32_value};
use crate::sim::c220::numeric::fixp::{C220FixpRoundMode, c220_fixp_f32_to_f16};
use crate::sim::c220::numeric::fp16::{C220Fp16Mode, C220Fp16Outcome, C220Fp16Status};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220FixpActivation<'a> {
    None,
    Relu,
    LeakyRelu { slope: u32 },
    ParametricRelu { slopes: &'a [u32] },
}

/// Functional conversion for FIX mode 1. Activation precedes narrowing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FixpFp16Conversion<'a> {
    mode: C220Fp16Mode,
    activation: C220FixpActivation<'a>,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum C220FixpFp16Error {
    #[error("FIX PReLU slope lane {lane} is outside {available} supplied slopes")]
    MissingSlope { lane: usize, available: usize },
    #[error("FIX FP32 source has {0} bytes, which is not a whole number of lanes")]
    IncompleteLane(usize),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220FixpFp16Result {
    pub bytes: Vec<u8>,
    /// Per-lane diagnostic status; does not imply an architectural status write.
    pub lane_status: Vec<C220Fp16Status>,
}

impl<'a> C220FixpFp16Conversion<'a> {
    pub fn new(control: u64, activation: C220FixpActivation<'a>) -> Self {
        Self {
            mode: C220Fp16Mode::from_control_spr(control),
            activation,
        }
    }

    pub fn evaluate_lane(
        self,
        bits: u32,
        lane: usize,
    ) -> Result<C220Fp16Outcome, C220FixpFp16Error> {
        let slope = match self.activation {
            C220FixpActivation::LeakyRelu { slope } => Some(slope),
            C220FixpActivation::ParametricRelu { slopes } => {
                Some(*slopes.get(lane).ok_or(C220FixpFp16Error::MissingSlope {
                    lane,
                    available: slopes.len(),
                })?)
            }
            _ => None,
        };
        let mut activation_status = C220Fp16Status::default();
        let bits = if self.activation == C220FixpActivation::Relu || slope.is_some() {
            let magnitude = bits & 0x7fff_ffff;
            activation_status.nan_operand = magnitude > 0x7f80_0000;
            activation_status.infinity_operand = magnitude == 0x7f80_0000;
            if let Some(slope) = slope.filter(|_| bits >> 31 != 0) {
                let product =
                    evaluate_fp32_value(Fp32VectorOperation::Multiply, bits, slope & 0xffff_e000);
                activation_status = C220Fp16Status {
                    nan_operand: product.status.nan_operand,
                    infinity_operand: product.status.infinity_operand,
                    overflow: product.status.overflow,
                    underflow: product.status.underflow,
                    invalid: product.status.invalid || product.status.zero_times_infinity,
                };
                product.bits
            } else if activation_status.nan_operand {
                0x7fff_ffff
            } else if bits >> 31 != 0 {
                0
            } else {
                bits
            }
        } else {
            bits
        };
        let mode = if slope.is_some() {
            C220Fp16Mode::Saturating
        } else {
            self.mode
        };
        let mut result = c220_fixp_f32_to_f16(bits, C220FixpRoundMode::NearestEven, mode);
        result.status = result.status.merge(activation_status);
        Ok(result)
    }

    pub fn evaluate(self, source: &[u8]) -> Result<C220FixpFp16Result, C220FixpFp16Error> {
        if !source.len().is_multiple_of(4) {
            return Err(C220FixpFp16Error::IncompleteLane(source.len()));
        }
        let mut bytes = Vec::with_capacity(source.len() / 2);
        let mut lane_status = Vec::with_capacity(source.len() / 4);
        for (index, lane) in source.chunks_exact(4).enumerate() {
            let bits = u32::from_le_bytes([lane[0], lane[1], lane[2], lane[3]]);
            let result = self.evaluate_lane(bits, index)?;
            bytes.extend_from_slice(&result.bits.to_le_bytes());
            lane_status.push(result.status);
        }
        Ok(C220FixpFp16Result { bytes, lane_status })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn control_and_relu_preserve_lane_order_and_exception_distinctions() {
        let source: Vec<u8> = [
            0x8000_0000_u32,
            0xff80_0000,
            0xff80_0001,
            65536_f32.to_bits(),
            1.0_f32.to_bits() | (1 << 12),
        ]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect();
        for (control, relu, expected) in [
            (0, 0, [0x8000_u16, 0xfbff, 0, 0x7bff, 0x3c00]),
            (1 << 48, 0, [0x8000, 0xfc00, 0x7fff, 0x7c00, 0x3c00]),
            (0, 1, [0, 0, 0, 0x7bff, 0x3c00]),
            (1 << 48, 1, [0, 0, 0x7fff, 0x7c00, 0x3c00]),
        ] {
            let activation = if relu == 0 {
                C220FixpActivation::None
            } else {
                C220FixpActivation::Relu
            };
            let conversion = C220FixpFp16Conversion::new(control, activation);
            let result = conversion.evaluate(&source).unwrap();
            assert_eq!(
                result.bytes,
                expected
                    .into_iter()
                    .flat_map(u16::to_le_bytes)
                    .collect::<Vec<_>>()
            );
            assert!(result.lane_status[1].infinity_operand);
            assert!(!result.lane_status[1].overflow);
            assert!(result.lane_status[2].nan_operand);
            assert!(result.lane_status[3].overflow);
            assert!(conversion.evaluate(&source[..3]).is_err());
        }
    }

    #[test]
    fn leaky_slopes_truncate_and_force_saturation_after_activation() {
        let slopes = [0x3f00_1fff, 0x7f80_0001, 0x3f80_0000, 0];
        let conversion = C220FixpFp16Conversion::new(
            1 << 48,
            C220FixpActivation::ParametricRelu { slopes: &slopes },
        );
        assert_eq!(
            conversion
                .evaluate_lane((-2.0_f32).to_bits(), 0)
                .unwrap()
                .bits,
            0xbc00
        );
        let invalid = conversion.evaluate_lane(0x8000_0000, 1).unwrap();
        assert_eq!(invalid.bits, 0);
        assert!(invalid.status.invalid);
        assert_eq!(
            conversion.evaluate_lane(0x7f80_0000, 2).unwrap().bits,
            0x7bff
        );
        assert_eq!(conversion.evaluate_lane(0xff80_0000, 3).unwrap().bits, 0);
        assert!(conversion.evaluate_lane(0, 4).is_err());
        let scalar =
            C220FixpFp16Conversion::new(0, C220FixpActivation::LeakyRelu { slope: 0x3f00_1fff });
        assert_eq!(
            scalar.evaluate_lane((-2.0_f32).to_bits(), 12).unwrap().bits,
            0xbc00
        );
    }
}
