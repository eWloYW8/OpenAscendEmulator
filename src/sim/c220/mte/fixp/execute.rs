use crate::isa::c220::mte::fixp::{C220FixpDescriptor, C220FixpDestination, C220FixpInstruction};
use crate::sim::c220::memory::{C220LocalBuffer, C220LocalBufferError};

use super::{
    C220FixpActivation, C220FixpFp16Conversion, C220FixpFp16Error, C220FixpFp16Layout,
    C220FixpFp16Result, C220FixpLayoutError, C220FixpSlice,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FixpFp16Command {
    pub descriptor: C220FixpDescriptor,
    pub source_address: u64,
    pub destination_address: u64,
    pub control: u64,
    pub scalar_slope: u32,
    pub slope_base_block: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220FixpSliceResult {
    pub coordinate: C220FixpSlice,
    pub conversion: C220FixpFp16Result,
    pub slope_read_address: Option<u32>,
}

#[derive(Debug, thiserror::Error)]
pub enum C220FixpExecutionError {
    #[error("word {0:#010x} is not an L0C FIX instruction")]
    Instruction(u32),
    #[error("FIX FP16 L1 command does not support destination {0:?}")]
    Destination(C220FixpDestination),
    #[error("FIX FP16 command requires FP32 source format, got {0}")]
    SourceFormat(u8),
    #[error("FIX source register X{0} is unavailable")]
    MissingGpr(u8),
    #[error("FIX special register {0} is unavailable")]
    MissingSpr(u8),
    #[error(transparent)]
    Layout(#[from] C220FixpLayoutError),
    #[error(transparent)]
    Conversion(#[from] C220FixpFp16Error),
    #[error(transparent)]
    Memory(#[from] C220LocalBufferError),
    #[error("unsupported FIX activation mode {0}")]
    Activation(u8),
}

impl C220FixpFp16Command {
    /// Capture operands at admission. `control` is the instruction's captured
    /// CTRL value, not a later read of live scalar state.
    pub fn capture_l1(
        word: u32,
        control: u64,
        mut read_gpr: impl FnMut(u8) -> Option<u64>,
        mut read_spr: impl FnMut(u8) -> Option<u64>,
    ) -> Result<Self, C220FixpExecutionError> {
        let instruction =
            C220FixpInstruction::decode(word).ok_or(C220FixpExecutionError::Instruction(word))?;
        if instruction.destination != C220FixpDestination::L1 {
            return Err(C220FixpExecutionError::Destination(instruction.destination));
        }
        if instruction.source_format != 0 {
            return Err(C220FixpExecutionError::SourceFormat(
                instruction.source_format,
            ));
        }
        let mut gpr =
            |register| read_gpr(register).ok_or(C220FixpExecutionError::MissingGpr(register));
        let destination_address = gpr(instruction.destination_register)?;
        let source_address = gpr(instruction.source_register)?;
        let xt = gpr(instruction.shape_register)?;
        let xm = gpr(instruction.control_register)?;
        let mut spr =
            |register| read_spr(register).ok_or(C220FixpExecutionError::MissingSpr(register));
        let slope_base_block = spr(64)? as u8;
        let scalar_slope = spr(61)? as u32;
        let nd = spr(97)?;
        let command = Self {
            descriptor: C220FixpDescriptor { xt, xm, nd },
            source_address,
            destination_address,
            control,
            scalar_slope,
            slope_base_block,
        };
        command.layout()?;
        Ok(command)
    }

    pub fn layout(self) -> Result<C220FixpFp16Layout, C220FixpLayoutError> {
        C220FixpFp16Layout::new(
            self.descriptor,
            self.source_address,
            self.destination_address,
        )
    }

    /// Computes slices lazily without publishing destination writes. The timing
    /// consumer decides when to advance this iterator and commit each result.
    pub fn evaluate<'a>(
        self,
        l0c: &'a C220LocalBuffer,
        slopes: &'a C220LocalBuffer,
    ) -> Result<
        impl Iterator<Item = Result<C220FixpSliceResult, C220FixpExecutionError>> + 'a,
        C220FixpExecutionError,
    > {
        let coordinates = self.layout()?.slices();
        if self.descriptor.activation_mode() > 3 && !self.descriptor.is_disabled() {
            return Err(C220FixpExecutionError::Activation(
                self.descriptor.activation_mode(),
            ));
        }
        Ok(coordinates.map(move |coordinate| {
            let input = l0c.read_initialized_linear(
                coordinate.source_address,
                coordinate.source_bytes() as usize,
            )?;
            let slope_read_address = (self.descriptor.activation_mode() == 3)
                .then(|| coordinate.slope_address(self.slope_base_block));
            let mut slope_words = [0; 16];
            if let Some(address) = slope_read_address {
                let bytes = slopes.read_initialized_linear(u64::from(address), 64)?;
                for (word, bytes) in slope_words.iter_mut().zip(bytes.chunks_exact(4)) {
                    *word = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                }
            }
            let activation = match self.descriptor.activation_mode() {
                0 => C220FixpActivation::None,
                1 => C220FixpActivation::Relu,
                2 => C220FixpActivation::LeakyRelu {
                    slope: self.scalar_slope,
                },
                3 => C220FixpActivation::ParametricRelu {
                    slopes: &slope_words,
                },
                other => return Err(C220FixpExecutionError::Activation(other)),
            };
            let conversion =
                C220FixpFp16Conversion::new(self.control, activation).evaluate(&input)?;
            Ok(C220FixpSliceResult {
                coordinate,
                conversion,
                slope_read_address,
            })
        }))
    }

    /// Functional L0C-to-L1 execution. This entry does not advance cycle state.
    /// Earlier writes remain visible if a later source access fails.
    pub fn execute_to_l1(
        self,
        l0c: &C220LocalBuffer,
        slopes: &C220LocalBuffer,
        l1: &mut C220LocalBuffer,
        mut observe: impl FnMut(&C220FixpSliceResult),
    ) -> Result<(), C220FixpExecutionError> {
        for result in self.evaluate(l0c, slopes)? {
            let result = result?;
            l1.write_known_linear(
                result.coordinate.destination_address,
                &result.conversion.bytes,
            )?;
            observe(&result);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instruction_capture_binds_operands_and_rejects_other_destinations() {
        let word = (6 << 29) | (3 << 24) | (1 << 17) | (2 << 12) | (3 << 7) | (4 << 2);
        let gpr = |r| match r {
            1 => Some(128),
            2 => Some(256),
            3 => Some((1 << 16) | (16 << 4)),
            4 => Some(1 << 34),
            _ => None,
        };
        let spr = |r| match r {
            64 => Some(0x1234),
            61 => Some(0x1_3f00_0000),
            97 => Some(1),
            _ => None,
        };
        let command = C220FixpFp16Command::capture_l1(word, 1 << 48, gpr, spr).unwrap();
        assert_eq!(
            (command.source_address, command.destination_address),
            (256, 128)
        );
        assert_eq!(
            (command.slope_base_block, command.scalar_slope),
            (0x34, 0x3f00_0000)
        );
        assert_eq!(command.control, 1 << 48);
        assert!(matches!(
            C220FixpFp16Command::capture_l1(word ^ (1 << 24), 0, gpr, spr),
            Err(C220FixpExecutionError::Destination(_))
        ));
        assert!(matches!(
            C220FixpFp16Command::capture_l1(word, 0, gpr, |_| None),
            Err(C220FixpExecutionError::MissingSpr(64))
        ));
    }

    #[test]
    fn nd_execution_fetches_slopes_per_column_and_preserves_destination_tail() {
        let mut l0c = C220LocalBuffer::new(4096);
        let mut slopes = C220LocalBuffer::new(4096);
        let mut l1 = C220LocalBuffer::new(4096);
        l0c.write_known_linear(0, &(-2_f32).to_le_bytes().repeat(16))
            .unwrap();
        l0c.write_known_linear(256, &(-4_f32).to_le_bytes())
            .unwrap();
        slopes
            .write_known_linear(2048, &0.5_f32.to_le_bytes().repeat(16))
            .unwrap();
        slopes
            .write_known_linear(2112, &0.25_f32.to_le_bytes().repeat(16))
            .unwrap();
        l1.write_known_linear(0, &[0xaa; 40]).unwrap();
        let command = C220FixpFp16Command {
            descriptor: C220FixpDescriptor {
                xt: (32 << 32) | (1 << 16) | (17 << 4),
                xm: (1 << 43) | (3 << 39) | (1 << 34) | 4,
                nd: 1,
            },
            source_address: 0,
            destination_address: 0,
            control: 1 << 48,
            scalar_slope: 0,
            slope_base_block: 0,
        };
        let mut reads = Vec::new();
        command
            .execute_to_l1(&l0c, &slopes, &mut l1, |slice| {
                reads.push(slice.slope_read_address)
            })
            .unwrap();
        assert_eq!(reads, [Some(2048), Some(2112)]);
        assert_eq!(
            l1.read_known(0, 34).unwrap(),
            0xbc00_u16.to_le_bytes().repeat(17)
        );
        assert_eq!(l1.read_known(34, 6).unwrap(), [0xaa; 6]);
    }
}
