use crate::isa::c220::vector::C220CopyInstruction;
use crate::memory::ub::UbMemory;
use crate::sim::c220::memory::C220UbBank;
use crate::sim::c220::vector::{
    C220_VECTOR_TILE_BYTES, C220VectorAddresses, C220VectorControl, C220VectorError,
    C220VectorReadAccess, C220VectorStore, plan_c220_unary_write_targets,
    plan_c220_vector_read_accesses, vector_destination_address_for_width,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220CopyIssue {
    pub pc: u64,
    pub word: u32,
    pub instruction: C220CopyInstruction,
    pub control: C220VectorControl,
    pub addresses: C220VectorAddresses,
    pub iteration_masks: Vec<[u64; 4]>,
    pub(crate) write_targets: Vec<C220VectorStore>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220CopyValueInputs {
    pub instruction: C220CopyInstruction,
    pub control: C220VectorControl,
    pub addresses: C220VectorAddresses,
    pub mask: [u64; 4],
    pub repeat_index: usize,
    pub lane_group: u8,
    pub lane_slice: Option<(usize, usize)>,
}

impl C220CopyIssue {
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

pub fn plan_c220_copy_issue(
    pc: u64,
    word: u32,
    mut control: C220VectorControl,
    addresses: C220VectorAddresses,
    iteration_masks: &[[u64; 4]],
    ub: &UbMemory,
) -> Result<C220CopyIssue, C220VectorError> {
    let instruction =
        C220CopyInstruction::decode(word).ok_or(C220VectorError::UnsupportedWord { pc, word })?;
    control.source_0_block_stride = control.source_0_block_stride.max(1);
    let write_targets = plan_c220_unary_write_targets(
        control,
        addresses,
        iteration_masks,
        instruction.element_bytes,
        ub,
    )?;
    Ok(C220CopyIssue {
        pc,
        word,
        instruction,
        control,
        addresses,
        iteration_masks: iteration_masks.to_vec(),
        write_targets,
    })
}

pub fn evaluate_c220_copy_repeat(
    inputs: C220CopyValueInputs,
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
        let in_uop = inputs.lane_slice.map_or_else(
            || lane_index / 64 == usize::from(inputs.lane_group),
            |(first_lane, lane_count)| {
                lane_index >= first_lane && lane_index < first_lane + lane_count
            },
        );
        if !in_uop || inputs.mask[lane_index / 64] & (1_u64 << (lane_index % 64)) == 0 {
            values.push(0);
            continue;
        }
        let bits = if element_bytes == 2 {
            u32::from(u16::from_le_bytes(chunk.try_into().expect("two-byte lane")))
        } else {
            u32::from_le_bytes(chunk.try_into().expect("four-byte lane"))
        };
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
            data: crate::sim::c220::vector::access::store_data(bits.to_le_bytes()),
        });
    }
    Ok((values, stores))
}
