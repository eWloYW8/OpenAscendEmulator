use crate::isa::c220::vector::{C220MovevControl, C220MovevInstruction};
use crate::memory::ub::UbMemory;
use crate::sim::c220::memory::C220UbBank;
use crate::sim::c220::vector::{
    C220_VECTOR_BLOCK_BYTES, C220_VECTOR_TILE_BYTES, C220VectorError, C220VectorStore,
    check_repeat_limit,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220MovevStep {
    pub pc: u64,
    pub word: u32,
    pub instruction: C220MovevInstruction,
    pub control: C220MovevControl,
    pub destination_address: u64,
    pub scalar_word: u32,
    pub iteration_masks: Vec<[u64; 4]>,
    pub stores: Vec<C220VectorStore>,
}

pub fn execute_c220_movev_to_ub(
    pc: u64,
    word: u32,
    control: C220MovevControl,
    destination_address: u64,
    scalar_word: u32,
    iteration_masks: &[[u64; 4]],
    ub: &mut UbMemory,
) -> Result<C220MovevStep, C220VectorError> {
    let step = plan_c220_movev_to_ub(
        pc,
        word,
        control,
        destination_address,
        scalar_word,
        iteration_masks,
        ub,
    )?;
    crate::sim::c220::vector::access::commit_vector_stores(ub, &step.stores)?;
    Ok(step)
}

pub(crate) fn plan_c220_movev_to_ub(
    pc: u64,
    word: u32,
    control: C220MovevControl,
    destination_address: u64,
    scalar_word: u32,
    iteration_masks: &[[u64; 4]],
    ub: &UbMemory,
) -> Result<C220MovevStep, C220VectorError> {
    check_repeat_limit(iteration_masks.len())?;
    let instruction =
        C220MovevInstruction::decode(word).ok_or(C220VectorError::UnsupportedWord { pc, word })?;
    let element_bytes = usize::from(
        instruction
            .supported_element_bytes()
            .ok_or(C220VectorError::UnsupportedWord { pc, word })?,
    );
    let lane_count = C220_VECTOR_TILE_BYTES / element_bytes;
    let scalar_bytes = scalar_word.to_le_bytes();
    let mut stores = Vec::new();
    let max_lanes = lane_count * iteration_masks.len();
    stores
        .try_reserve_exact(max_lanes)
        .map_err(|_| C220VectorError::HostAllocationFailed { lanes: max_lanes })?;
    for (repeat_index, active_mask) in iteration_masks.iter().enumerate() {
        for lane_index in 0..lane_count {
            if active_mask[lane_index / 64] & (1_u64 << (lane_index % 64)) == 0 {
                continue;
            }
            let lanes_per_block = C220_VECTOR_BLOCK_BYTES / element_bytes;
            let block = lane_index / lanes_per_block;
            let offset = repeat_index as u64
                * u64::from(control.destination_repeat_stride)
                * C220_VECTOR_BLOCK_BYTES as u64
                + block as u64
                    * u64::from(control.destination_block_stride.max(1))
                    * C220_VECTOR_BLOCK_BYTES as u64
                + ((lane_index % lanes_per_block) * element_bytes) as u64;
            let address = destination_address.checked_add(offset).ok_or(
                C220VectorError::AddressOverflow {
                    base: destination_address,
                    lane: repeat_index * lane_count + lane_index,
                },
            )?;
            ub.check_range(address, element_bytes)?;
            stores.push(C220VectorStore {
                repeat_index,
                lane_index,
                address,
                bank: C220UbBank::from_address(address),
                width_bytes: element_bytes as u8,
                data: crate::sim::c220::vector::access::store_data(scalar_bytes),
            });
        }
    }
    Ok(C220MovevStep {
        pc,
        word,
        instruction,
        control,
        destination_address,
        scalar_word,
        iteration_masks: iteration_masks.to_vec(),
        stores,
    })
}
