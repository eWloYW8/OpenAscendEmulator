use crate::architecture::c220::C220UbBank;
use crate::isa::c220::vector::C220BroadcastInstruction;
use crate::memory::ub::UbMemory;
use crate::sim::c220::vector::{
    C220_VECTOR_BLOCK_BYTES, C220_VECTOR_BLOCK_COUNT, C220_VECTOR_TILE_BYTES, C220VectorAddresses,
    C220VectorControl, C220VectorError, C220VectorReadAccess, C220VectorStore,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220BroadcastControl {
    pub repeat_count: u8,
    pub destination_block_stride: u16,
    pub destination_repeat_stride: u16,
}

impl C220BroadcastControl {
    pub const fn decode(word: u64) -> Self {
        Self {
            repeat_count: (word >> 56) as u8,
            destination_block_stride: if word as u16 == 0 { 1 } else { word as u16 },
            destination_repeat_stride: (((word >> 32) & 0xff) | ((word >> 52) & 0xf) << 8) as u16,
        }
    }

    pub const fn vector_control(self) -> C220VectorControl {
        C220VectorControl {
            encoded_repeat_count: self.repeat_count,
            destination_block_stride: self.destination_block_stride,
            destination_repeat_stride: self.destination_repeat_stride,
            source_0_block_stride: 0,
            source_0_repeat_stride: 0,
            source_1_block_stride: 0,
            source_1_repeat_stride: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220BroadcastIssue {
    pub pc: u64,
    pub word: u32,
    pub instruction: C220BroadcastInstruction,
    pub control: C220BroadcastControl,
    pub source_address: u64,
    pub destination_address: u64,
    pub(crate) write_targets: Vec<C220VectorStore>,
}

impl C220BroadcastIssue {
    pub fn addresses(&self) -> C220VectorAddresses {
        C220VectorAddresses {
            source_0: self.source_address,
            source_1: 0,
            destination: self.destination_address,
        }
    }

    pub fn read_accesses_for_repeat(
        &self,
        repeat_index: usize,
    ) -> Result<Vec<C220VectorReadAccess>, C220VectorError> {
        if repeat_index >= usize::from(self.control.repeat_count) {
            return Err(C220VectorError::InvalidRepeatIndex(repeat_index));
        }
        Ok(vec![C220VectorReadAccess {
            source_index: 0,
            block_index: 0,
            buffer_offset: 0,
            bytes: 32,
            address: broadcast_source_address(
                self.source_address,
                repeat_index,
                self.instruction.element_bytes,
            )?,
            active_lane_mask: 0xff,
        }])
    }
}

pub fn plan_c220_broadcast_issue(
    pc: u64,
    word: u32,
    control_word: u64,
    source_address: u64,
    destination_address: u64,
    ub: &UbMemory,
) -> Result<C220BroadcastIssue, C220VectorError> {
    let instruction = C220BroadcastInstruction::decode(word)
        .ok_or(C220VectorError::UnsupportedWord { pc, word })?;
    let control = C220BroadcastControl::decode(control_word);
    let lanes_per_block = C220_VECTOR_BLOCK_BYTES / usize::from(instruction.element_bytes);
    let count = usize::from(control.repeat_count) * C220_VECTOR_BLOCK_COUNT * lanes_per_block;
    let mut write_targets = Vec::new();
    write_targets
        .try_reserve_exact(count)
        .map_err(|_| C220VectorError::HostAllocationFailed { lanes: count })?;
    for repeat_index in 0..usize::from(control.repeat_count) {
        ub.check_range(
            broadcast_source_address(source_address, repeat_index, instruction.element_bytes)?,
            C220_VECTOR_BLOCK_BYTES,
        )?;
        for block_index in 0..C220_VECTOR_BLOCK_COUNT {
            let address =
                broadcast_block_address(destination_address, control, repeat_index, block_index)?;
            ub.check_range(address, C220_VECTOR_BLOCK_BYTES)?;
            for lane_in_block in 0..lanes_per_block {
                let lane_index = block_index * lanes_per_block + lane_in_block;
                let lane_address =
                    address + (lane_in_block * usize::from(instruction.element_bytes)) as u64;
                write_targets.push(C220VectorStore {
                    repeat_index,
                    lane_index,
                    address: lane_address,
                    bank: C220UbBank::from_address(lane_address),
                    width_bytes: instruction.element_bytes,
                    data: [0; 4],
                });
            }
        }
    }
    Ok(C220BroadcastIssue {
        pc,
        word,
        instruction,
        control,
        source_address,
        destination_address,
        write_targets,
    })
}

pub fn evaluate_c220_broadcast_repeat(
    instruction: C220BroadcastInstruction,
    control: C220BroadcastControl,
    destination_address: u64,
    repeat_index: usize,
    source_bytes: &[u8],
) -> Result<(Vec<u32>, Vec<C220VectorStore>), C220VectorError> {
    if source_bytes.len() != C220_VECTOR_TILE_BYTES {
        return Err(C220VectorError::InvalidSourceTile {
            actual: source_bytes.len(),
            expected: C220_VECTOR_TILE_BYTES,
        });
    }
    if repeat_index >= usize::from(control.repeat_count) {
        return Err(C220VectorError::InvalidRepeatIndex(repeat_index));
    }
    let width = usize::from(instruction.element_bytes);
    let lanes_per_block = C220_VECTOR_BLOCK_BYTES / width;
    let mut values = Vec::with_capacity(C220_VECTOR_TILE_BYTES / width);
    let mut stores = Vec::with_capacity(C220_VECTOR_TILE_BYTES / width);
    for block_index in 0..C220_VECTOR_BLOCK_COUNT {
        let source = &source_bytes[block_index * width..(block_index + 1) * width];
        let mut data = [0; 4];
        data[..width].copy_from_slice(source);
        let bits = u32::from_le_bytes(data);
        let address =
            broadcast_block_address(destination_address, control, repeat_index, block_index)?;
        for lane_in_block in 0..lanes_per_block {
            let lane_index = block_index * lanes_per_block + lane_in_block;
            let lane_address = address + (lane_in_block * width) as u64;
            values.push(bits);
            stores.push(C220VectorStore {
                repeat_index,
                lane_index,
                address: lane_address,
                bank: C220UbBank::from_address(lane_address),
                width_bytes: instruction.element_bytes,
                data,
            });
        }
    }
    Ok((values, stores))
}

fn broadcast_source_address(
    base: u64,
    repeat_index: usize,
    element_bytes: u8,
) -> Result<u64, C220VectorError> {
    let offset = (8 * repeat_index * usize::from(element_bytes)) as u64;
    base.checked_add(offset)
        .ok_or(C220VectorError::SourceAddressOverflow {
            source_index: 0,
            base,
            block: repeat_index,
        })
}

fn broadcast_block_address(
    base: u64,
    control: C220BroadcastControl,
    repeat_index: usize,
    block_index: usize,
) -> Result<u64, C220VectorError> {
    let blocks = block_index * usize::from(control.destination_block_stride)
        + repeat_index * usize::from(control.destination_repeat_stride);
    base.checked_add((blocks * C220_VECTOR_BLOCK_BYTES) as u64)
        .ok_or(C220VectorError::AddressOverflow {
            base,
            lane: block_index,
        })
}
