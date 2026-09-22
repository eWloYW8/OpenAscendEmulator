use crate::architecture::c220::C220UbBank;
use crate::isa::c220::vector_scalar::{
    C220VectorScalarInstruction, C220VectorScalarOperation, C220VectorScalarType,
};
use crate::memory::ub::UbMemory;
use crate::numeric::fp32::{Fp32ValueStatus, Fp32VectorOperation, evaluate_fp32_value};
use crate::sim::c220::fp16::{C220Fp16Mode, C220Fp16Status, evaluate_c220_fp16};
use crate::sim::c220::vector::{
    C220_VECTOR_TILE_BYTES, C220VectorAddresses, C220VectorControl, C220VectorError,
    C220VectorReadAccess, C220VectorStore, plan_c220_unary_write_targets,
    plan_c220_vector_read_accesses, vector_destination_address_for_width,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220VectorScalarLaneOutcome {
    pub bits: u32,
    pub fp16_status: Option<C220Fp16Status>,
    pub fp32_status: Option<Fp32ValueStatus>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220VectorScalarOperand {
    pub bits: u32,
    pub fp16_mode: C220Fp16Mode,
    pub integer_saturating: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220VectorScalarValueInputs {
    pub instruction: C220VectorScalarInstruction,
    pub scalar: C220VectorScalarOperand,
    pub control: C220VectorControl,
    pub addresses: C220VectorAddresses,
    pub mask: [u64; 4],
    pub repeat_index: usize,
    pub lane_group: u8,
    pub lane_slice: Option<(usize, usize)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220VectorScalarIssue {
    pub pc: u64,
    pub word: u32,
    pub instruction: C220VectorScalarInstruction,
    pub scalar: C220VectorScalarOperand,
    pub control: C220VectorControl,
    pub addresses: C220VectorAddresses,
    pub iteration_masks: Vec<[u64; 4]>,
    pub(crate) write_targets: Vec<C220VectorStore>,
}

impl C220VectorScalarIssue {
    pub fn read_accesses_for_repeat(
        &self,
        repeat_index: usize,
        lane_group: u8,
    ) -> Result<Vec<C220VectorReadAccess>, C220VectorError> {
        let mask = self
            .iteration_masks
            .get(repeat_index)
            .ok_or(C220VectorError::MissingMaskState)?;
        plan_c220_vector_read_accesses(
            self.control,
            self.addresses,
            repeat_index,
            mask,
            1,
            self.instruction.dtype.element_bytes(),
            Some(lane_group),
        )
    }
}

pub fn plan_c220_vector_scalar_issue(
    pc: u64,
    word: u32,
    scalar: C220VectorScalarOperand,
    control: C220VectorControl,
    addresses: C220VectorAddresses,
    iteration_masks: &[[u64; 4]],
    ub: &UbMemory,
) -> Result<C220VectorScalarIssue, C220VectorError> {
    let instruction = C220VectorScalarInstruction::decode(word)
        .ok_or(C220VectorError::UnsupportedWord { pc, word })?;
    let write_targets = plan_c220_unary_write_targets(
        control,
        addresses,
        iteration_masks,
        instruction.dtype.element_bytes(),
        ub,
    )?;
    Ok(C220VectorScalarIssue {
        pc,
        word,
        instruction,
        scalar,
        control,
        addresses,
        iteration_masks: iteration_masks.to_vec(),
        write_targets,
    })
}

pub fn evaluate_c220_vector_scalar_repeat(
    inputs: C220VectorScalarValueInputs,
    source_bytes: &[u8],
) -> Result<(Vec<C220VectorScalarLaneOutcome>, Vec<C220VectorStore>), C220VectorError> {
    if source_bytes.len() != C220_VECTOR_TILE_BYTES {
        return Err(C220VectorError::InvalidSourceTile {
            actual: source_bytes.len(),
            expected: C220_VECTOR_TILE_BYTES,
        });
    }
    let element_bytes = inputs.instruction.dtype.element_bytes();
    let lane_count = C220_VECTOR_TILE_BYTES / usize::from(element_bytes);
    let mut values = Vec::with_capacity(lane_count);
    let mut stores = Vec::new();
    for (lane_index, chunk) in source_bytes
        .chunks_exact(usize::from(element_bytes))
        .enumerate()
    {
        let in_uop = inputs.lane_slice.map_or_else(
            || lane_index / 64 == usize::from(inputs.lane_group),
            |(first_lane, lane_count)| {
                lane_index >= first_lane && lane_index < first_lane + lane_count
            },
        );
        if !in_uop || inputs.mask[lane_index / 64] & (1_u64 << (lane_index % 64)) == 0 {
            values.push(C220VectorScalarLaneOutcome {
                bits: 0,
                fp16_status: None,
                fp32_status: None,
            });
            continue;
        }
        let value = match inputs.instruction.dtype {
            C220VectorScalarType::F16 => {
                let source = u16::from_le_bytes(chunk.try_into().expect("two-byte lane"));
                let result = evaluate_c220_fp16(
                    inputs.instruction.operation,
                    source,
                    inputs.scalar.bits as u16,
                    inputs.scalar.fp16_mode,
                );
                C220VectorScalarLaneOutcome {
                    bits: u32::from(result.bits),
                    fp16_status: Some(result.status),
                    fp32_status: None,
                }
            }
            C220VectorScalarType::S16 => {
                let source = u16::from_le_bytes(chunk.try_into().expect("two-byte lane"));
                let scalar = inputs.scalar.bits as u16;
                let bits = match inputs.instruction.operation {
                    C220VectorScalarOperation::Add if inputs.scalar.integer_saturating => {
                        (source as i16).saturating_add(scalar as i16) as u16
                    }
                    C220VectorScalarOperation::Add => source.wrapping_add(scalar),
                    C220VectorScalarOperation::Multiply if inputs.scalar.integer_saturating => {
                        (source as i16).saturating_mul(scalar as i16) as u16
                    }
                    C220VectorScalarOperation::Multiply => source.wrapping_mul(scalar),
                    C220VectorScalarOperation::Maximum => (source as i16).max(scalar as i16) as u16,
                    C220VectorScalarOperation::Minimum => (source as i16).min(scalar as i16) as u16,
                    C220VectorScalarOperation::LeakyRelu => {
                        unreachable!("integer VLRELU has no encoding")
                    }
                };
                C220VectorScalarLaneOutcome {
                    bits: u32::from(bits),
                    fp16_status: None,
                    fp32_status: None,
                }
            }
            C220VectorScalarType::S32 => {
                let source = u32::from_le_bytes(chunk.try_into().expect("four-byte lane"));
                let bits = match inputs.instruction.operation {
                    C220VectorScalarOperation::Add if inputs.scalar.integer_saturating => {
                        (source as i32).saturating_add(inputs.scalar.bits as i32) as u32
                    }
                    C220VectorScalarOperation::Add => source.wrapping_add(inputs.scalar.bits),
                    C220VectorScalarOperation::Multiply if inputs.scalar.integer_saturating => {
                        (source as i32).saturating_mul(inputs.scalar.bits as i32) as u32
                    }
                    C220VectorScalarOperation::Multiply => source.wrapping_mul(inputs.scalar.bits),
                    C220VectorScalarOperation::Maximum => {
                        (source as i32).max(inputs.scalar.bits as i32) as u32
                    }
                    C220VectorScalarOperation::Minimum => {
                        (source as i32).min(inputs.scalar.bits as i32) as u32
                    }
                    C220VectorScalarOperation::LeakyRelu => {
                        unreachable!("integer VLRELU has no encoding")
                    }
                };
                C220VectorScalarLaneOutcome {
                    bits,
                    fp16_status: None,
                    fp32_status: None,
                }
            }
            C220VectorScalarType::F32 => {
                let source = u32::from_le_bytes(chunk.try_into().expect("four-byte lane"));
                let result = if inputs.instruction.operation == C220VectorScalarOperation::LeakyRelu
                {
                    evaluate_c220_fp32_lrelu(source, inputs.scalar.bits)
                } else {
                    let operation = match inputs.instruction.operation {
                        C220VectorScalarOperation::Add => Fp32VectorOperation::Add,
                        C220VectorScalarOperation::Multiply => Fp32VectorOperation::Multiply,
                        C220VectorScalarOperation::Maximum => Fp32VectorOperation::Maximum,
                        C220VectorScalarOperation::Minimum => Fp32VectorOperation::Minimum,
                        C220VectorScalarOperation::LeakyRelu => unreachable!(),
                    };
                    evaluate_fp32_value(operation, source, inputs.scalar.bits)
                };
                C220VectorScalarLaneOutcome {
                    bits: result.bits,
                    fp16_status: None,
                    fp32_status: Some(result.status),
                }
            }
        };
        values.push(value);
        let address = vector_destination_address_for_width(
            inputs.control,
            inputs.addresses,
            inputs.repeat_index,
            lane_index,
            element_bytes,
        )?;
        stores.push(C220VectorStore {
            repeat_index: inputs.repeat_index,
            lane_index,
            address,
            bank: C220UbBank::from_address(address),
            width_bytes: element_bytes,
            data: super::store_data(value.bits.to_le_bytes()),
        });
    }
    Ok((values, stores))
}

fn evaluate_c220_fp32_lrelu(
    source_bits: u32,
    slope_bits: u32,
) -> crate::numeric::fp32::Fp32ValueOutcome {
    if source_bits & 0x8000_0000 != 0 {
        return evaluate_fp32_value(Fp32VectorOperation::Multiply, source_bits, slope_bits);
    }
    let nan_operand = source_bits & 0x7fff_ffff > 0x7f80_0000;
    let infinity_operand = source_bits == 0x7f80_0000;
    crate::numeric::fp32::Fp32ValueOutcome {
        bits: if nan_operand {
            0x7fff_ffff
        } else {
            source_bits
        },
        status: Fp32ValueStatus {
            nan_operand,
            infinity_operand,
            ..Fp32ValueStatus::default()
        },
    }
}
