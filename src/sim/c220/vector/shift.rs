use crate::architecture::c220::C220UbBank;
use crate::isa::c220::vector::{C220ShiftInstruction, C220ShiftOperation};
use crate::memory::ub::UbMemory;
use crate::sim::c220::vector::{
    C220_VECTOR_TILE_BYTES, C220VectorAddresses, C220VectorControl, C220VectorError,
    C220VectorReadAccess, C220VectorStore, plan_c220_unary_write_targets,
    plan_c220_vector_read_accesses, vector_destination_address_for_width,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220ShiftIssue {
    pub pc: u64,
    pub word: u32,
    pub instruction: C220ShiftInstruction,
    pub shift: u32,
    pub control: C220VectorControl,
    pub addresses: C220VectorAddresses,
    pub iteration_masks: Vec<[u64; 4]>,
    pub(crate) write_targets: Vec<C220VectorStore>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220ShiftValueInputs {
    pub instruction: C220ShiftInstruction,
    pub shift: u32,
    pub control: C220VectorControl,
    pub addresses: C220VectorAddresses,
    pub mask: [u64; 4],
    pub repeat_index: usize,
    pub lane_group: u8,
}

impl C220ShiftIssue {
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
            self.instruction.element_bytes,
            Some(lane_group),
        )
    }
}

pub fn plan_c220_shift_issue(
    pc: u64,
    word: u32,
    shift: u32,
    control: C220VectorControl,
    addresses: C220VectorAddresses,
    iteration_masks: &[[u64; 4]],
    ub: &UbMemory,
) -> Result<C220ShiftIssue, C220VectorError> {
    let instruction =
        C220ShiftInstruction::decode(word).ok_or(C220VectorError::UnsupportedWord { pc, word })?;
    let write_targets = plan_c220_unary_write_targets(
        control,
        addresses,
        iteration_masks,
        instruction.element_bytes,
        ub,
    )?;
    Ok(C220ShiftIssue {
        pc,
        word,
        instruction,
        shift,
        control,
        addresses,
        iteration_masks: iteration_masks.to_vec(),
        write_targets,
    })
}

pub fn evaluate_c220_shift_repeat(
    inputs: C220ShiftValueInputs,
    source_bytes: &[u8],
) -> Result<(Vec<u32>, Vec<C220VectorStore>), C220VectorError> {
    if source_bytes.len() != C220_VECTOR_TILE_BYTES {
        return Err(C220VectorError::InvalidSourceTile {
            actual: source_bytes.len(),
            expected: C220_VECTOR_TILE_BYTES,
        });
    }
    let element_bytes = inputs.instruction.element_bytes;
    let lane_count = C220_VECTOR_TILE_BYTES / usize::from(element_bytes);
    let mut values = Vec::with_capacity(lane_count);
    let mut stores = Vec::new();
    for (lane_index, chunk) in source_bytes
        .chunks_exact(usize::from(element_bytes))
        .enumerate()
    {
        if lane_index / 64 != usize::from(inputs.lane_group)
            || inputs.mask[lane_index / 64] & (1_u64 << (lane_index % 64)) == 0
        {
            values.push(0);
            continue;
        }
        let source = if element_bytes == 2 {
            u32::from(u16::from_le_bytes(chunk.try_into().expect("two-byte lane")))
        } else {
            u32::from_le_bytes(chunk.try_into().expect("four-byte lane"))
        };
        let bits = shift_lane(inputs.instruction, source, inputs.shift);
        values.push(bits);
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
            data: bits.to_le_bytes(),
        });
    }
    Ok((values, stores))
}

fn shift_lane(instruction: C220ShiftInstruction, source: u32, shift: u32) -> u32 {
    let width = u32::from(instruction.element_bytes) * 8;
    let amount = shift & (width * 2 - 1);
    let mask = if width == 16 { 0xffff } else { u32::MAX };
    let source = source & mask;
    match instruction.operation {
        C220ShiftOperation::Left => {
            if amount >= width {
                0
            } else {
                (source << amount) & mask
            }
        }
        C220ShiftOperation::RightUnsigned => {
            if amount >= width {
                0
            } else {
                source >> amount
            }
        }
        C220ShiftOperation::RightSigned => {
            let signed = if width == 16 {
                source as u16 as i16 as i32
            } else {
                source as i32
            };
            if amount > width {
                if instruction.round {
                    0
                } else {
                    (signed >> (width - 1)) as u32 & mask
                }
            } else if amount == 0 {
                source
            } else {
                let shifted = signed >> amount.min(width - 1);
                let rounded = if instruction.round && (signed >> (amount - 1)) & 1 != 0 {
                    shifted.wrapping_add(1)
                } else {
                    shifted
                };
                rounded as u32 & mask
            }
        }
    }
}
