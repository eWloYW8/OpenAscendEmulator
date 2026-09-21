use crate::architecture::c220::C220UbBank;
use crate::isa::c220::vector::C220MovevInstruction;
use crate::memory::sparse::MemoryByteState;
use crate::memory::ub::UbMemory;
use crate::sim::c220::vector::{
    C220_VECTOR_BLOCK_BYTES, C220_VECTOR_TILE_BYTES, C220VectorError, C220VectorStore,
    check_repeat_limit,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MovevControl {
    pub encoded_repeat_count: u8,
    pub destination_block_stride: u16,
    pub destination_repeat_stride: u16,
}

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

pub fn decode_c220_movev_control(control: u64) -> Result<C220MovevControl, C220VectorError> {
    Ok(C220MovevControl {
        encoded_repeat_count: (control >> 56) as u8,
        destination_block_stride: (control & 0xffff) as u16,
        destination_repeat_stride: (((control >> 32) & 0xff) | (((control >> 52) & 0xf) << 8))
            as u16,
    })
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
    write_c220_movev_to_ub(&step, ub)?;
    Ok(step)
}

pub(crate) fn write_c220_movev_to_ub(
    step: &C220MovevStep,
    ub: &mut UbMemory,
) -> Result<(), C220VectorError> {
    let writes = step
        .stores
        .iter()
        .map(|store| {
            (
                store.address,
                store.data[..usize::from(store.width_bytes)]
                    .iter()
                    .copied()
                    .map(MemoryByteState::Known)
                    .collect(),
            )
        })
        .collect::<Vec<_>>();
    ub.write_segments(&writes)?;
    Ok(())
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
                data: scalar_bytes,
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
