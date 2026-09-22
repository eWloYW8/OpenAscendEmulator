use crate::isa::c220::vector::gather::{C220GatherInstruction, C220GatherKind, C220GatherWidth};
use crate::memory::ub::UbMemory;
use crate::sim::c220::memory::C220UbBank;

use crate::sim::c220::vector::{
    C220_VECTOR_BLOCK_BYTES, C220VectorError, C220VectorReadAccess, C220VectorStore,
    check_repeat_limit,
};

const ELEMENTS_PER_UOP: usize = 16;
const ELEMENTS_PER_READ_PORT: usize = 8;
const BLOCKS_PER_REPEAT: usize = 8;
const BLOCK_WORDS: usize = C220_VECTOR_BLOCK_BYTES / 4;
const INDEX_WINDOW_BYTES: usize = 256;
const MAX_BLOCK_REPEATS: usize = INDEX_WINDOW_BYTES / (BLOCKS_PER_REPEAT * 4);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220GatherControl {
    pub base_offset: u32,
    pub destination_repeat_stride: u16,
    pub destination_block_stride: u8,
    pub repeat_count: u8,
}

impl C220GatherControl {
    pub const fn decode(value: u64) -> Self {
        let block_stride = ((value >> 40) & 0xff) as u8;
        Self {
            base_offset: value as u32,
            destination_repeat_stride: (((value >> 32) & 0xff) | (((value >> 52) & 0xf) << 8))
                as u16,
            destination_block_stride: if block_stride == 0 { 1 } else { block_stride },
            repeat_count: (value >> 56) as u8,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220GatherIssue {
    pub pc: u64,
    pub word: u32,
    pub instruction: C220GatherInstruction,
    pub control: C220GatherControl,
    pub destination_address: u64,
    pub index_address: u64,
    pub iteration_masks: Vec<[u64; 4]>,
    indices: Vec<Vec<u32>>,
    pub(crate) write_targets: Vec<C220VectorStore>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct C220GatherLaneOutcome {
    pub active: bool,
    pub bits: u32,
}

impl C220GatherIssue {
    pub const fn repeat_count(&self) -> usize {
        self.indices.len()
    }

    pub const fn groups_per_repeat(&self) -> u8 {
        match self.instruction.kind {
            C220GatherKind::Elements(width) => width.groups_per_repeat(),
            C220GatherKind::Blocks => 1,
        }
    }

    pub const fn index_groups_per_repeat(&self) -> u8 {
        match self.instruction.kind {
            C220GatherKind::Elements(width) => width.groups_per_repeat().div_ceil(4),
            C220GatherKind::Blocks => 0,
        }
    }

    pub fn index_read_access(
        &self,
        repeat_index: usize,
        group: u8,
    ) -> Result<C220VectorReadAccess, C220VectorError> {
        let address = match self.instruction.kind {
            C220GatherKind::Elements(width) => {
                if repeat_index >= self.repeat_count()
                    || group >= width.groups_per_repeat().div_ceil(4)
                {
                    return Err(C220VectorError::InvalidRepeatIndex(repeat_index));
                }
                self.index_address
                    .checked_add((repeat_index * width.lane_count() * 4) as u64)
                    .and_then(|address| {
                        address.checked_add(u64::from(group) * INDEX_WINDOW_BYTES as u64)
                    })
                    .ok_or(C220VectorError::AddressOverflow {
                        base: self.index_address,
                        lane: repeat_index,
                    })?
            }
            C220GatherKind::Blocks => {
                if repeat_index != 0 || group != 0 {
                    return Err(C220VectorError::InvalidRepeatIndex(repeat_index));
                }
                self.index_address
            }
        };
        Ok(C220VectorReadAccess {
            source_index: 0,
            block_index: 0,
            buffer_offset: 0,
            bytes: INDEX_WINDOW_BYTES as u16,
            address,
            active_lane_mask: u16::MAX,
        })
    }

    pub fn data_read_accesses(
        &self,
        repeat_index: usize,
        group: u8,
    ) -> Result<Vec<C220VectorReadAccess>, C220VectorError> {
        let indices = self
            .indices
            .get(repeat_index)
            .ok_or(C220VectorError::InvalidRepeatIndex(repeat_index))?;
        match self.instruction.kind {
            C220GatherKind::Elements(width) => {
                if group >= width.groups_per_repeat() {
                    return Err(C220VectorError::InvalidLaneGroup(group));
                }
                let mask = self
                    .iteration_masks
                    .get(repeat_index)
                    .ok_or(C220VectorError::MissingMaskState)?;
                let element_bytes = width.element_bytes();
                let first_lane = usize::from(group) * ELEMENTS_PER_UOP;
                let mut accesses = Vec::with_capacity(ELEMENTS_PER_UOP);
                for local_lane in 0..ELEMENTS_PER_UOP {
                    let lane = first_lane + local_lane;
                    if mask[lane / 64] & (1_u64 << (lane % 64)) == 0 {
                        continue;
                    }
                    let address = u64::from(self.control.base_offset.wrapping_add(indices[lane]));
                    accesses.push(C220VectorReadAccess {
                        source_index: (local_lane / ELEMENTS_PER_READ_PORT) as u8,
                        block_index: ((local_lane % ELEMENTS_PER_READ_PORT)
                            * usize::from(element_bytes)
                            / C220_VECTOR_BLOCK_BYTES) as u8,
                        buffer_offset: ((local_lane % ELEMENTS_PER_READ_PORT)
                            * usize::from(element_bytes))
                            as u16,
                        bytes: u16::from(element_bytes),
                        address,
                        active_lane_mask: 1 << (local_lane % ELEMENTS_PER_READ_PORT),
                    });
                }
                Ok(accesses)
            }
            C220GatherKind::Blocks => {
                if group != 0 {
                    return Err(C220VectorError::InvalidLaneGroup(group));
                }
                Ok(indices
                    .iter()
                    .enumerate()
                    .map(|(block, &index)| C220VectorReadAccess {
                        source_index: (block / 4) as u8,
                        block_index: (block % 4) as u8,
                        buffer_offset: ((block % 4) * C220_VECTOR_BLOCK_BYTES) as u16,
                        bytes: C220_VECTOR_BLOCK_BYTES as u16,
                        address: u64::from(self.control.base_offset.wrapping_add(index)),
                        active_lane_mask: u16::MAX,
                    })
                    .collect())
            }
        }
    }

    pub(crate) fn stores_for_data_uop(
        &self,
        repeat_index: usize,
        group: u8,
    ) -> Vec<C220VectorStore> {
        match self.instruction.kind {
            C220GatherKind::Elements(_) => {
                let first = usize::from(group) * ELEMENTS_PER_UOP;
                let end = first + ELEMENTS_PER_UOP;
                self.write_targets
                    .iter()
                    .copied()
                    .filter(|store| {
                        store.repeat_index == repeat_index
                            && (first..end).contains(&store.lane_index)
                    })
                    .collect()
            }
            C220GatherKind::Blocks => self
                .write_targets
                .iter()
                .copied()
                .filter(|store| store.repeat_index == repeat_index)
                .collect(),
        }
    }
}

pub fn plan_c220_gather_issue(
    pc: u64,
    word: u32,
    control_value: u64,
    destination_address: u64,
    index_address: u64,
    iteration_masks: Vec<[u64; 4]>,
    ub: &UbMemory,
) -> Result<C220GatherIssue, C220VectorError> {
    let instruction =
        C220GatherInstruction::decode(word).ok_or(C220VectorError::UnsupportedWord { pc, word })?;
    let control = C220GatherControl::decode(control_value);
    let repeat_count = match instruction.kind {
        C220GatherKind::Elements(_) => iteration_masks.len(),
        C220GatherKind::Blocks => usize::from(control.repeat_count),
    };
    check_repeat_limit(repeat_count)?;
    if matches!(instruction.kind, C220GatherKind::Blocks) && repeat_count > MAX_BLOCK_REPEATS {
        return Err(C220VectorError::RepeatLimitExceeded {
            count: repeat_count as u64,
            limit: MAX_BLOCK_REPEATS,
        });
    }

    let index_count = match instruction.kind {
        C220GatherKind::Elements(width) => width.lane_count(),
        C220GatherKind::Blocks => BLOCKS_PER_REPEAT,
    };
    let mut indices = Vec::with_capacity(repeat_count);
    for repeat_index in 0..repeat_count {
        let repeat_address = index_address
            .checked_add((repeat_index * index_count * 4) as u64)
            .ok_or(C220VectorError::AddressOverflow {
                base: index_address,
                lane: repeat_index,
            })?;
        let bytes = ub.read_known(repeat_address, index_count * 4)?;
        indices.push(
            bytes
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("four-byte index")))
                .collect(),
        );
    }

    let mut write_targets = Vec::new();
    match instruction.kind {
        C220GatherKind::Elements(width) => {
            let element_bytes = width.element_bytes();
            for (repeat_index, mask) in iteration_masks.iter().enumerate() {
                for lane_index in 0..width.lane_count() {
                    if mask[lane_index / 64] & (1_u64 << (lane_index % 64)) == 0 {
                        continue;
                    }
                    let address = packed_destination_address(
                        destination_address,
                        control.destination_repeat_stride,
                        repeat_index,
                        lane_index * usize::from(element_bytes),
                    )?;
                    ub.check_range(address, usize::from(element_bytes))?;
                    write_targets.push(C220VectorStore {
                        repeat_index,
                        lane_index,
                        address,
                        bank: C220UbBank::from_address(address),
                        width_bytes: element_bytes,
                        data: [0; 8],
                    });
                }
            }
        }
        C220GatherKind::Blocks => {
            for repeat_index in 0..repeat_count {
                for block in 0..BLOCKS_PER_REPEAT {
                    for word_index in 0..BLOCK_WORDS {
                        let byte_offset = block
                            * C220_VECTOR_BLOCK_BYTES
                            * usize::from(control.destination_block_stride)
                            + word_index * 4;
                        let address = packed_destination_address(
                            destination_address,
                            control.destination_repeat_stride,
                            repeat_index,
                            byte_offset,
                        )?;
                        ub.check_range(address, 4)?;
                        write_targets.push(C220VectorStore {
                            repeat_index,
                            lane_index: block * BLOCK_WORDS + word_index,
                            address,
                            bank: C220UbBank::from_address(address),
                            width_bytes: 4,
                            data: [0; 8],
                        });
                    }
                }
            }
        }
    }

    Ok(C220GatherIssue {
        pc,
        word,
        instruction,
        control,
        destination_address,
        index_address,
        iteration_masks,
        indices,
        write_targets,
    })
}

pub(crate) fn evaluate_c220_gather_data_uop(
    issue: &C220GatherIssue,
    repeat_index: usize,
    group: u8,
    source_0_bytes: &[u8],
    source_1_bytes: &[u8],
) -> Result<(Vec<C220GatherLaneOutcome>, Vec<C220VectorStore>), C220VectorError> {
    match issue.instruction.kind {
        C220GatherKind::Elements(width) => evaluate_elements(
            issue,
            width,
            repeat_index,
            group,
            source_0_bytes,
            source_1_bytes,
        ),
        C220GatherKind::Blocks => {
            evaluate_blocks(issue, repeat_index, group, source_0_bytes, source_1_bytes)
        }
    }
}

fn evaluate_elements(
    issue: &C220GatherIssue,
    width: C220GatherWidth,
    repeat_index: usize,
    group: u8,
    source_0_bytes: &[u8],
    source_1_bytes: &[u8],
) -> Result<(Vec<C220GatherLaneOutcome>, Vec<C220VectorStore>), C220VectorError> {
    if group >= width.groups_per_repeat() {
        return Err(C220VectorError::InvalidLaneGroup(group));
    }
    let mask = issue
        .iteration_masks
        .get(repeat_index)
        .ok_or(C220VectorError::InvalidRepeatIndex(repeat_index))?;
    let element_bytes = usize::from(width.element_bytes());
    let first_lane = usize::from(group) * ELEMENTS_PER_UOP;
    let mut lanes = Vec::with_capacity(ELEMENTS_PER_UOP);
    let mut stores = Vec::with_capacity(ELEMENTS_PER_UOP);
    for local_lane in 0..ELEMENTS_PER_UOP {
        let lane_index = first_lane + local_lane;
        let active = mask[lane_index / 64] & (1_u64 << (lane_index % 64)) != 0;
        if !active {
            lanes.push(C220GatherLaneOutcome {
                active: false,
                bits: 0,
            });
            continue;
        }
        let source = if local_lane < ELEMENTS_PER_READ_PORT {
            source_0_bytes
        } else {
            source_1_bytes
        };
        let offset = (local_lane % ELEMENTS_PER_READ_PORT) * element_bytes;
        let mut data = [0_u8; 4];
        data[..element_bytes].copy_from_slice(&source[offset..offset + element_bytes]);
        let address = packed_destination_address(
            issue.destination_address,
            issue.control.destination_repeat_stride,
            repeat_index,
            lane_index * element_bytes,
        )?;
        lanes.push(C220GatherLaneOutcome {
            active: true,
            bits: u32::from_le_bytes(data),
        });
        stores.push(C220VectorStore {
            repeat_index,
            lane_index,
            address,
            bank: C220UbBank::from_address(address),
            width_bytes: width.element_bytes(),
            data: crate::sim::c220::vector::access::store_data(data),
        });
    }
    Ok((lanes, stores))
}

fn evaluate_blocks(
    issue: &C220GatherIssue,
    repeat_index: usize,
    group: u8,
    source_0_bytes: &[u8],
    source_1_bytes: &[u8],
) -> Result<(Vec<C220GatherLaneOutcome>, Vec<C220VectorStore>), C220VectorError> {
    if group != 0 {
        return Err(C220VectorError::InvalidLaneGroup(group));
    }
    if repeat_index >= issue.repeat_count() {
        return Err(C220VectorError::InvalidRepeatIndex(repeat_index));
    }
    let mut lanes = Vec::with_capacity(BLOCKS_PER_REPEAT * BLOCK_WORDS);
    let mut stores = Vec::with_capacity(BLOCKS_PER_REPEAT * BLOCK_WORDS);
    for block in 0..BLOCKS_PER_REPEAT {
        let source = if block < 4 {
            source_0_bytes
        } else {
            source_1_bytes
        };
        let block_offset = (block % 4) * C220_VECTOR_BLOCK_BYTES;
        for word_index in 0..BLOCK_WORDS {
            let offset = block_offset + word_index * 4;
            let data: [u8; 4] = source[offset..offset + 4]
                .try_into()
                .expect("four-byte gathered word");
            let byte_offset = block
                * C220_VECTOR_BLOCK_BYTES
                * usize::from(issue.control.destination_block_stride)
                + word_index * 4;
            let address = packed_destination_address(
                issue.destination_address,
                issue.control.destination_repeat_stride,
                repeat_index,
                byte_offset,
            )?;
            lanes.push(C220GatherLaneOutcome {
                active: true,
                bits: u32::from_le_bytes(data),
            });
            stores.push(C220VectorStore {
                repeat_index,
                lane_index: block * BLOCK_WORDS + word_index,
                address,
                bank: C220UbBank::from_address(address),
                width_bytes: 4,
                data: crate::sim::c220::vector::access::store_data(data),
            });
        }
    }
    Ok((lanes, stores))
}

fn packed_destination_address(
    base: u64,
    repeat_stride: u16,
    repeat_index: usize,
    byte_offset: usize,
) -> Result<u64, C220VectorError> {
    let repeat_offset =
        repeat_index as u64 * C220_VECTOR_BLOCK_BYTES as u64 * u64::from(repeat_stride);
    base.checked_add(repeat_offset)
        .and_then(|address| address.checked_add(byte_offset as u64))
        .ok_or(C220VectorError::AddressOverflow {
            base,
            lane: byte_offset,
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::sparse::MemoryByteState;
    use crate::sim::c220::core::C220CoreInstruction;
    use crate::sim::c220::vector::C220VectorInstruction;
    use crate::sim::c220::vector::timing::C220VectorUopKind;

    #[test]
    fn gather_plans_index_windows_and_copies_indirect_values() {
        let mut ub = UbMemory::new(4096, 256);
        let indices = (0_u32..64).flat_map(|index| (index * 4).to_le_bytes());
        ub.write_states(
            0x400,
            &indices.map(MemoryByteState::Known).collect::<Vec<_>>(),
        )
        .unwrap();
        let source = (0_u32..64).flat_map(|value| (0x1000 + value).to_le_bytes());
        ub.write_states(
            0x100,
            &source.map(MemoryByteState::Known).collect::<Vec<_>>(),
        )
        .unwrap();
        let word = 0x8000_004a | (3 << 17) | (4 << 12) | (5 << 7);
        let issue = plan_c220_gather_issue(
            0,
            word,
            (1_u64 << 56) | (8_u64 << 32) | 0x100,
            0x800,
            0x400,
            vec![[u64::MAX, 0, 0, 0]],
            &ub,
        )
        .unwrap();
        let uops = C220CoreInstruction::Vector(C220VectorInstruction::Gather(issue.clone()))
            .as_vector()
            .unwrap()
            .uops()
            .unwrap();
        assert_eq!(uops.len(), 5);
        assert!(matches!(
            uops[0].kind,
            C220VectorUopKind::GatherIndex { group: 0 }
        ));
        assert!(uops[1..].iter().all(|uop| uop.stages.execute_ticks == 1));

        let accesses = issue.data_read_accesses(0, 0).unwrap();
        let mut source_0 = vec![0; 256];
        let mut source_1 = vec![0; 256];
        for access in accesses {
            let target = if access.source_index == 0 {
                &mut source_0
            } else {
                &mut source_1
            };
            let offset = usize::from(access.buffer_offset);
            let bytes = ub
                .read_known(access.address, usize::from(access.bytes))
                .unwrap();
            target[offset..offset + bytes.len()].copy_from_slice(&bytes);
        }
        let (_, stores) =
            evaluate_c220_gather_data_uop(&issue, 0, 0, &source_0, &source_1).unwrap();
        assert_eq!(stores.len(), 16);
        assert_eq!(stores[0].address, 0x800);
        assert_eq!(
            u32::from_le_bytes(stores[15].data[..4].try_into().unwrap()),
            0x100f
        );
    }
}
