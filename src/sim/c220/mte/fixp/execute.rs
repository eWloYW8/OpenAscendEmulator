use crate::isa::c220::mte::fixp::{C220FixpDescriptor, C220FixpDestination, C220FixpInstruction};
use crate::sim::c220::memory::{C220LocalBuffer, C220LocalBufferError};

use super::{
    C220FixpActivation, C220FixpConversionResult, C220FixpFp16Conversion, C220FixpFp16Error,
    C220FixpLaneStatus, C220FixpLayout, C220FixpLayoutError, C220FixpOutputFormat, C220FixpSlice,
    C220FixpSourceFormat,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FixpCommand {
    pub descriptor: C220FixpDescriptor,
    pub source_format: C220FixpSourceFormat,
    pub source_address: u64,
    pub destination_address: u64,
    pub control: u64,
    pub scalar_slope: u32,
    pub scalar_dequant: u64,
    pub slope_base_block: u8,
    pub dequant_base_block: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220FixpSliceResult {
    pub coordinate: C220FixpSlice,
    pub conversion: C220FixpConversionResult,
    pub slope_read_address: Option<u32>,
    pub dequant_read_address: Option<u32>,
}

#[derive(Debug, thiserror::Error)]
pub enum C220FixpExecutionError {
    #[error("word {0:#010x} is not an L0C FIX instruction")]
    Instruction(u32),
    #[error("FIX command destination {0:?} does not match the selected execution path")]
    Destination(C220FixpDestination),
    #[error("unsupported FIX source format {0}")]
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
    #[error(transparent)]
    ExternalMemory(#[from] crate::memory::mapped::MappedMemoryError),
    #[error("unsupported FIX activation mode {0}")]
    Activation(u8),
}

impl C220FixpCommand {
    /// Capture operands at admission. `control` is the instruction's captured
    /// CTRL value, not a later read of live scalar state.
    pub fn capture_l1(
        word: u32,
        control: u64,
        read_gpr: impl FnMut(u8) -> Option<u64>,
        read_spr: impl FnMut(u8) -> Option<u64>,
    ) -> Result<Self, C220FixpExecutionError> {
        Self::capture_destination(word, C220FixpDestination::L1, control, read_gpr, read_spr)
    }

    pub(super) fn capture_destination(
        word: u32,
        destination: C220FixpDestination,
        control: u64,
        mut read_gpr: impl FnMut(u8) -> Option<u64>,
        mut read_spr: impl FnMut(u8) -> Option<u64>,
    ) -> Result<Self, C220FixpExecutionError> {
        let instruction =
            C220FixpInstruction::decode(word).ok_or(C220FixpExecutionError::Instruction(word))?;
        if instruction.destination != destination {
            return Err(C220FixpExecutionError::Destination(instruction.destination));
        }
        let source_format = match instruction.source_format {
            0 => C220FixpSourceFormat::Fp32,
            1 => C220FixpSourceFormat::Int32,
            other => return Err(C220FixpExecutionError::SourceFormat(other)),
        };
        let mut gpr =
            |register| read_gpr(register).ok_or(C220FixpExecutionError::MissingGpr(register));
        let destination_address = gpr(instruction.destination_register)?;
        let source_address = gpr(instruction.source_register)?;
        let xt = gpr(instruction.shape_register)?;
        let xm = gpr(instruction.control_register)?;
        let mut spr =
            |register| read_spr(register).ok_or(C220FixpExecutionError::MissingSpr(register));
        let bases = spr(64)?;
        let slope_base_block = bases as u8;
        let dequant_base_block = (bases >> 8) as u8;
        let scalar_slope = spr(61)? as u32;
        let scalar_dequant = if matches!(
            (C220FixpDescriptor { xt, xm, nd: 0 }).conversion_mode(),
            9 | 11 | 13 | 22 | 24 | 26
        ) {
            spr(65)?
        } else {
            0
        };
        let nd = spr(97)?;
        let command = Self {
            descriptor: C220FixpDescriptor { xt, xm, nd },
            source_format,
            source_address,
            destination_address,
            control,
            scalar_slope,
            scalar_dequant,
            slope_base_block,
            dequant_base_block,
        };
        command.layout()?;
        Ok(command)
    }

    pub fn layout(self) -> Result<C220FixpLayout, C220FixpLayoutError> {
        C220FixpLayout::new(
            self.descriptor,
            self.source_format,
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
        self.validate_activation()?;
        Ok(coordinates.map(move |coordinate| {
            let input = l0c.read_initialized_linear(
                coordinate.source_address,
                coordinate.source_bytes() as usize,
            )?;
            if self.descriptor.conversion_mode() != 1 {
                let mode = self.descriptor.conversion_mode();
                let activation = self.descriptor.activation_mode();
                let dequant_read_address = matches!(mode, 8 | 10 | 12 | 21 | 23 | 25)
                    .then(|| coordinate.dequant_address(self.dequant_base_block));
                let mut factors = [self.scalar_dequant; 16];
                if let Some(address) = dequant_read_address {
                    let data = slopes.read_initialized_linear(u64::from(address), 128)?;
                    for (factor, lane) in factors.iter_mut().zip(data.chunks_exact(8)) {
                        *factor = u64::from_le_bytes(lane.try_into().expect("eight-byte factor"));
                    }
                }
                let slope_read_address = ((dequant_read_address.is_some() && activation == 0)
                    || (matches!(mode, 8..=11 | 21..=26) && activation == 3))
                    .then(|| coordinate.slope_address(self.slope_base_block));
                let mut slope_words = [if activation == 2 {
                    self.scalar_slope
                } else {
                    0
                }; 16];
                if let Some(address) = slope_read_address {
                    let data = slopes.read_initialized_linear(u64::from(address), 64)?;
                    for (word, lane) in slope_words.iter_mut().zip(data.chunks_exact(4)) {
                        *word = u32::from_le_bytes(lane.try_into().expect("four-byte slope"));
                    }
                }
                let mut bytes = Vec::with_capacity(coordinate.destination_bytes() as usize);
                let mut lane_status = Vec::with_capacity(input.len() / 4);
                for (index, lane) in input.chunks_exact(4).enumerate() {
                    let bits = u32::from_le_bytes([lane[0], lane[1], lane[2], lane[3]]);
                    if matches!(
                        coordinate.output_format,
                        C220FixpOutputFormat::Bits8 | C220FixpOutputFormat::Int4
                    ) {
                        use crate::sim::c220::numeric::fixp::C220FixpDequantActivation;
                        use crate::sim::c220::numeric::requant::{
                            C220FixpQuantizedWidth, c220_fixp_quantize_f32, c220_fixp_requantize,
                        };
                        let activation = match activation {
                            0 => C220FixpDequantActivation::None,
                            1 => C220FixpDequantActivation::Relu,
                            _ => C220FixpDequantActivation::NegativeSlope(slope_words[index]),
                        };
                        let width = if coordinate.output_format == C220FixpOutputFormat::Int4 {
                            C220FixpQuantizedWidth::Bits4
                        } else {
                            C220FixpQuantizedWidth::Bits8
                        };
                        let result = match self.source_format {
                            C220FixpSourceFormat::Int32 => {
                                c220_fixp_requantize(bits as i32, factors[index], activation, width)
                            }
                            C220FixpSourceFormat::Fp32 => {
                                c220_fixp_quantize_f32(bits, factors[index], activation, width)
                            }
                        };
                        bytes.push(result.bits);
                        lane_status.push(C220FixpLaneStatus::Requant(result));
                    } else if coordinate.output_format == C220FixpOutputFormat::Int32 {
                        let result = if self.descriptor.activation_mode() == 1 {
                            (bits as i32).max(0) as u32
                        } else {
                            bits
                        };
                        bytes.extend_from_slice(&result.to_le_bytes());
                        lane_status.push(C220FixpLaneStatus::Integer);
                    } else if coordinate.output_format == C220FixpOutputFormat::Int16 {
                        let result = crate::sim::c220::numeric::fixp::c220_fixp_i32_to_i16(
                            bits as i32,
                            factors[index],
                            self.descriptor.activation_mode() == 1,
                        );
                        bytes.extend_from_slice(&result.value.to_le_bytes());
                        lane_status.push(C220FixpLaneStatus::Int16(result.status));
                    } else if coordinate.output_format == C220FixpOutputFormat::Fp16 {
                        use crate::sim::c220::numeric::fixp::{
                            C220FixpDequantActivation, c220_fixp_i32_to_f16,
                        };
                        let activation = match activation {
                            0 => C220FixpDequantActivation::None,
                            1 => C220FixpDequantActivation::Relu,
                            _ => C220FixpDequantActivation::NegativeSlope(slope_words[index]),
                        };
                        let result = c220_fixp_i32_to_f16(bits as i32, factors[index], activation);
                        bytes.extend_from_slice(&result.conversion.bits.to_le_bytes());
                        lane_status.push(C220FixpLaneStatus::DequantFp16(result));
                    } else if coordinate.output_format == C220FixpOutputFormat::Bf16 {
                        let result = crate::sim::c220::numeric::fixp::c220_fixp_f32_to_bf16(
                            bits,
                            self.control,
                            self.descriptor.activation_mode(),
                        );
                        bytes.extend_from_slice(&result.bits.to_le_bytes());
                        lane_status.push(C220FixpLaneStatus::Bf16(result.status));
                    } else {
                        let result = crate::sim::c220::numeric::fixp::c220_fixp_f32_output(
                            bits,
                            self.descriptor.activation_mode(),
                        );
                        bytes.extend_from_slice(&result.bits.to_le_bytes());
                        lane_status.push(C220FixpLaneStatus::Fp32(result.status));
                    }
                }
                if coordinate.output_format == C220FixpOutputFormat::Int4 {
                    for index in 0..bytes.len() / 2 {
                        bytes[index] = (bytes[2 * index] & 15) | ((bytes[2 * index + 1] & 15) << 4);
                    }
                    bytes.truncate(coordinate.destination_bytes() as usize);
                }
                return Ok(C220FixpSliceResult {
                    coordinate,
                    conversion: C220FixpConversionResult {
                        format: coordinate.output_format,
                        bytes,
                        lane_status,
                    },
                    slope_read_address,
                    dequant_read_address,
                });
            }
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
                conversion: conversion.into(),
                slope_read_address,
                dequant_read_address: None,
            })
        }))
    }

    pub(super) fn validate_activation(self) -> Result<(), C220FixpExecutionError> {
        let limit = match self.descriptor.conversion_mode() {
            1 => 3,
            12 | 13 => 1,
            _ => return Ok(()),
        };
        if !self.descriptor.is_disabled() && self.descriptor.activation_mode() > limit {
            return Err(C220FixpExecutionError::Activation(
                self.descriptor.activation_mode(),
            ));
        }
        Ok(())
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
        let command = C220FixpCommand::capture_l1(word, 1 << 48, gpr, spr).unwrap();
        assert_eq!(
            (command.source_address, command.destination_address),
            (256, 128)
        );
        assert_eq!(
            (command.slope_base_block, command.scalar_slope),
            (0x34, 0x3f00_0000)
        );
        assert_eq!(command.control, 1 << 48);
        assert_eq!(command.dequant_base_block, 0x12);
        let dequant = C220FixpCommand::capture_l1(
            word | 1,
            0,
            |r| if r == 4 { Some(13 << 34) } else { gpr(r) },
            |r| if r == 65 { Some(15 << 32) } else { spr(r) },
        )
        .unwrap();
        assert_eq!(dequant.scalar_dequant, 15 << 32);
        assert_eq!(
            dequant
                .layout()
                .unwrap()
                .slices()
                .next()
                .unwrap()
                .output_format,
            C220FixpOutputFormat::Int16
        );
        let mut invalid = dequant;
        invalid.descriptor.xm |= 2 << 39;
        assert!(matches!(
            invalid.validate_activation(),
            Err(C220FixpExecutionError::Activation(2))
        ));
        let integer = C220FixpCommand::capture_l1(
            word | 1,
            0,
            |r| if r == 4 { Some(0) } else { gpr(r) },
            spr,
        )
        .unwrap();
        assert_eq!(integer.source_format, C220FixpSourceFormat::Int32);
        assert!(C220FixpCommand::capture_l1(word | 1, 0, gpr, spr).is_err());
        assert!(matches!(
            C220FixpCommand::capture_l1(word ^ (1 << 24), 0, gpr, spr),
            Err(C220FixpExecutionError::Destination(_))
        ));
        assert!(matches!(
            C220FixpCommand::capture_l1(word, 0, gpr, |_| None),
            Err(C220FixpExecutionError::MissingSpr(64))
        ));
    }

    #[test]
    fn vector_dequantization_reads_full_blocks_and_selects_each_lane() {
        let mut l0c = C220LocalBuffer::new(128);
        l0c.write_known_linear(0, &65536_i32.to_le_bytes().repeat(32))
            .unwrap();
        let mut factors = C220LocalBuffer::new(4096);
        factors
            .write_states(256, &[crate::memory::sparse::MemoryByteState::Unknown; 128])
            .unwrap();
        factors
            .write_states(
                2240,
                &[crate::memory::sparse::MemoryByteState::Unknown; 128],
            )
            .unwrap();
        let words: Vec<_> = (0_u64..16).map(|shift| shift << 32).collect();
        factors
            .write_known_linear(
                128,
                &words
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let mut command = C220FixpCommand {
            descriptor: C220FixpDescriptor {
                xt: (32 << 32) | (1 << 16) | (17 << 4),
                xm: (12 << 34) | (1 << 39) | (1 << 43) | 1,
                nd: 1,
            },
            source_format: C220FixpSourceFormat::Int32,
            source_address: 0,
            destination_address: 0,
            control: 0,
            scalar_slope: 0,
            scalar_dequant: u64::MAX,
            slope_base_block: 3,
            dequant_base_block: 1,
        };
        let mut slices = command.evaluate(&l0c, &factors).unwrap();
        let first = slices.next().unwrap().unwrap();
        assert_eq!(first.dequant_read_address, Some(128));
        assert_eq!(first.slope_read_address, None);
        let expected: Vec<_> = (1..=16)
            .flat_map(|shift| ((65536_i32 >> shift).min(32767) as i16).to_le_bytes())
            .collect();
        assert_eq!(first.conversion.bytes, expected);
        assert!(slices.next().unwrap().is_err());
        drop(slices);
        factors
            .write_known_linear(256, &0_u64.to_le_bytes())
            .unwrap();
        assert!(
            command
                .evaluate(&l0c, &factors)
                .unwrap()
                .nth(1)
                .unwrap()
                .is_err()
        );
        factors
            .write_known_linear(256, &0_u64.to_le_bytes().repeat(16))
            .unwrap();
        let tail = command
            .evaluate(&l0c, &factors)
            .unwrap()
            .nth(1)
            .unwrap()
            .unwrap();
        assert_eq!(tail.conversion.bytes, i16::MAX.to_le_bytes());
        assert_eq!(tail.dequant_read_address, Some(256));
        command.descriptor.xm &= !(7 << 39);
        assert!(
            command
                .evaluate(&l0c, &factors)
                .unwrap()
                .next()
                .unwrap()
                .is_err()
        );
        factors.write_known_linear(2240, &[0; 128]).unwrap();
        let slices = command
            .evaluate(&l0c, &factors)
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(
            slices
                .iter()
                .map(|slice| slice.slope_read_address)
                .collect::<Vec<_>>(),
            [Some(2240), Some(2304)]
        );
    }

    #[test]
    fn byte_output_merges_columns_and_keeps_nd_tail_unpadded() {
        let mut command = C220FixpCommand {
            descriptor: C220FixpDescriptor {
                xt: (8 << 32) | (2 << 16) | (48 << 4),
                xm: (9 << 34) | 2,
                nd: 1,
            },
            source_format: C220FixpSourceFormat::Int32,
            source_address: 0,
            destination_address: 0,
            control: 0,
            scalar_slope: 0,
            scalar_dequant: (1 << 46) | 0x3f80_0000,
            slope_base_block: 0,
            dequant_base_block: 0,
        };
        let mut l0c = C220LocalBuffer::new(384);
        l0c.write_known_linear(0, &(-3_i32).to_le_bytes().repeat(96))
            .unwrap();
        let factors = C220LocalBuffer::new(0);
        let results = command
            .evaluate(&l0c, &factors)
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(
            results
                .iter()
                .map(|r| r.coordinate.destination_address)
                .collect::<Vec<_>>(),
            [0, 16, 256, 32, 48, 272]
        );
        assert!(results.iter().all(|r| r.conversion.bytes == [253; 16]));
        let packets = super::super::C220FixpReadGenerator::new(command, 0, 0, 128)
            .unwrap()
            .collect::<Vec<_>>();
        assert_eq!(
            packets
                .iter()
                .map(|p| (
                    p.operation.destination_address,
                    p.operation.output_bytes,
                    p.operation.last_in_uop
                ))
                .collect::<Vec<_>>(),
            [(0, 64, false), (0, 64, true), (256, 32, true)]
        );
        command.descriptor.xt = (32 << 32) | (2 << 16) | (17 << 4);
        command.descriptor.xm |= 1 << 43;
        let results = command
            .evaluate(&l0c, &factors)
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(
            results
                .iter()
                .map(|r| (r.coordinate.destination_address, r.conversion.bytes.len()))
                .collect::<Vec<_>>(),
            [(0, 16), (16, 1), (32, 16), (48, 1)]
        );
    }

    #[test]
    fn int4_packs_low_lane_first_and_selects_column_layout() {
        let mut command = C220FixpCommand {
            descriptor: C220FixpDescriptor {
                xt: (8 << 32) | (2 << 16) | (64 << 4),
                xm: (22 << 34) | 2,
                nd: 1,
            },
            source_format: C220FixpSourceFormat::Int32,
            source_address: 0,
            destination_address: 0,
            control: 0,
            scalar_slope: 0,
            scalar_dequant: 0x3f80_0000,
            slope_base_block: 0,
            dequant_base_block: 0,
        };
        let mut l0c = C220LocalBuffer::new(512);
        let input: Vec<_> = (-8_i32..8).flat_map(i32::to_le_bytes).collect();
        l0c.write_known_linear(0, &input.repeat(8)).unwrap();
        let mut factors = C220LocalBuffer::new(4096);
        factors
            .write_known_linear(0, &0x3f80_0000_u64.to_le_bytes().repeat(64))
            .unwrap();
        for mode in [21_u64, 22] {
            command.descriptor.xm = (mode << 34) | 2;
            for (columns, addresses) in [
                (64, vec![0, 8, 16, 24, 32, 40, 48, 56]),
                (48, vec![0, 256, 512, 8, 264, 520]),
            ] {
                command.descriptor.xt = (8 << 32) | (2 << 16) | (columns << 4);
                let results = command
                    .evaluate(&l0c, &factors)
                    .unwrap()
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap();
                assert_eq!(
                    results
                        .iter()
                        .map(|r| r.coordinate.destination_address)
                        .collect::<Vec<_>>(),
                    addresses
                );
                assert!(results.iter().all(
                    |r| r.conversion.bytes == [0x98, 0xba, 0xdc, 0xfe, 0x10, 0x32, 0x54, 0x76]
                ));
                let packets: Vec<_> = super::super::C220FixpReadGenerator::new(command, 0, 0, 128)
                    .unwrap()
                    .collect();
                assert_eq!(
                    packets
                        .iter()
                        .filter(|p| p.operation.last_in_uop)
                        .map(|p| p.operation.output_bytes)
                        .sum::<u32>(),
                    if columns == 64 { 64 } else { 48 }
                );
            }
            command.descriptor.xt = (32 << 32) | (2 << 16) | (19 << 4);
            command.descriptor.xm |= 1 << 43;
            let results = command
                .evaluate(&l0c, &factors)
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            assert_eq!(
                results
                    .iter()
                    .map(|r| (r.coordinate.destination_address, r.conversion.bytes.len()))
                    .collect::<Vec<_>>(),
                [(0, 8), (8, 1), (16, 8), (24, 1)]
            );
            assert_eq!(results[1].conversion.bytes, [0x98]);
            assert_eq!(results[1].conversion.lane_status.len(), 3);
            let mut destination = C220LocalBuffer::new(32);
            destination.write_known_linear(0, &[0xaa; 32]).unwrap();
            command
                .execute_to_l1(&l0c, &factors, &mut destination, |_| {})
                .unwrap();
            assert_eq!(destination.read_known(8, 2).unwrap(), [0x98, 0xaa]);
        }
    }

    #[test]
    fn integer_output_preserves_bits_and_uses_signed_relu() {
        let lanes = [i32::MIN, -1, 0, 1, i32::MAX, 0x7fc0_0001, -42, 42];
        let input: Vec<_> = lanes
            .iter()
            .cycle()
            .take(16)
            .flat_map(|x| x.to_le_bytes())
            .collect();
        let mut l0c = C220LocalBuffer::new(64);
        l0c.write_known_linear(0, &input).unwrap();
        let slopes = C220LocalBuffer::new(0);
        for activation in 0..8 {
            let command = C220FixpCommand {
                source_format: C220FixpSourceFormat::Int32,
                descriptor: C220FixpDescriptor {
                    xt: (1 << 16) | (16 << 4),
                    xm: activation << 39,
                    nd: 0,
                },
                source_address: 0,
                destination_address: 0,
                control: 1 << 48,
                scalar_slope: 0,
                slope_base_block: 0,
                dequant_base_block: 0,
                scalar_dequant: 0,
            };
            let result = command
                .evaluate(&l0c, &slopes)
                .unwrap()
                .next()
                .unwrap()
                .unwrap();
            let expected: Vec<_> = lanes
                .iter()
                .cycle()
                .take(16)
                .flat_map(|&x| if activation == 1 { x.max(0) } else { x }.to_le_bytes())
                .collect();
            assert_eq!(result.conversion.bytes, expected);
            assert_eq!(result.conversion.format, C220FixpOutputFormat::Int32);
            assert_eq!(
                result.conversion.lane_status,
                vec![C220FixpLaneStatus::Integer; 16]
            );
            assert_eq!(result.slope_read_address, None);
        }
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
        let command = C220FixpCommand {
            source_format: crate::sim::c220::mte::fixp::C220FixpSourceFormat::Fp32,
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
            dequant_base_block: 0,
            scalar_dequant: 0,
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
        let mut bf16 = command;
        bf16.descriptor.xm = (bf16.descriptor.xm & !(31 << 34)) | (16 << 34);
        bf16.execute_to_l1(&l0c, &C220LocalBuffer::new(0), &mut l1, |slice| {
            assert_eq!(slice.conversion.format, C220FixpOutputFormat::Bf16);
            assert_eq!(slice.slope_read_address, None);
        })
        .unwrap();
        assert_eq!(
            l1.read_known(0, 32).unwrap(),
            0xc000_u16.to_le_bytes().repeat(16)
        );
        assert_eq!(l1.read_known(32, 2).unwrap(), 0xc080_u16.to_le_bytes());
        assert_eq!(l1.read_known(34, 6).unwrap(), [0xaa; 6]);
    }
}
