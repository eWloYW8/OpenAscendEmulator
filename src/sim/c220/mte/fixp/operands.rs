use super::{C220FixpCommand, C220FixpExecutionError, C220FixpSlice};
use crate::sim::c220::memory::C220LocalBuffer;

/// Full factor blocks sampled for one functional slice, including inactive
/// tail lanes. Values are retained before numeric masking or interpretation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220FixpFactorOperands {
    pub dequant: [u64; 16],
    pub slopes: [u32; 16],
    pub dequant_read_address: Option<u32>,
    pub slope_read_address: Option<u32>,
}

impl C220FixpCommand {
    pub fn read_factor_operands(
        self,
        coordinate: C220FixpSlice,
        memory: &C220LocalBuffer,
    ) -> Result<C220FixpFactorOperands, C220FixpExecutionError> {
        self.validate_activation()?;
        let mode = self.descriptor.conversion_mode();
        let activation = self.descriptor.activation_mode();
        let dequantized = matches!(mode, 8..=13 | 21..=26);
        let vector_dequant = matches!(mode, 8 | 10 | 12 | 21 | 23 | 25);
        let dequant_read_address =
            vector_dequant.then(|| coordinate.dequant_address(self.dequant_base_block));
        let slope_read_address = ((vector_dequant && activation == 0)
            || ((mode == 1 || dequantized) && activation == 3))
            .then(|| coordinate.slope_address(self.slope_base_block));
        let mut operands = C220FixpFactorOperands {
            dequant: [if dequantized { self.scalar_dequant } else { 0 }; 16],
            slopes: [match activation {
                0 if dequantized && !vector_dequant => self.scalar_dequant as u32,
                2 if mode == 1 || dequantized => self.scalar_slope,
                _ => 0,
            }; 16],
            dequant_read_address,
            slope_read_address,
        };
        if let Some(address) = dequant_read_address {
            let bytes = memory.read_initialized_linear(u64::from(address), 128)?;
            for (factor, lane) in operands.dequant.iter_mut().zip(bytes.chunks_exact(8)) {
                *factor = u64::from_le_bytes(lane.try_into().expect("eight-byte factor"));
            }
        }
        if let Some(address) = slope_read_address {
            let bytes = memory.read_initialized_linear(u64::from(address), 64)?;
            for (factor, lane) in operands.slopes.iter_mut().zip(bytes.chunks_exact(4)) {
                *factor = u32::from_le_bytes(lane.try_into().expect("four-byte slope"));
            }
        }
        Ok(operands)
    }
}
