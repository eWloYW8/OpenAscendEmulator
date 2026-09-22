use crate::architecture::c220::C220UbBank;
use crate::isa::c220::vector::C220TransposeInstruction;
use crate::memory::ub::UbMemory;
use crate::sim::c220::vector::{
    C220_VECTOR_BLOCK_BYTES, C220VectorAddresses, C220VectorError, C220VectorReadAccess,
    C220VectorStore,
};

const TRANSPOSE_BYTES: usize = 512;
const TRANSPOSE_DIMENSION: usize = 16;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220TransposeIssue {
    pub pc: u64,
    pub word: u32,
    pub instruction: C220TransposeInstruction,
    pub source_address: u64,
    pub destination_address: u64,
    pub(crate) write_targets: Vec<C220VectorStore>,
}

impl C220TransposeIssue {
    pub fn addresses(&self) -> C220VectorAddresses {
        C220VectorAddresses {
            source_0: self.source_address,
            source_1: 0,
            destination: self.destination_address,
        }
    }

    pub fn read_accesses(&self) -> Result<Vec<C220VectorReadAccess>, C220VectorError> {
        (0..TRANSPOSE_BYTES / C220_VECTOR_BLOCK_BYTES)
            .map(|block_index| {
                Ok(C220VectorReadAccess {
                    source_index: 0,
                    block_index: block_index as u8,
                    buffer_offset: (block_index * C220_VECTOR_BLOCK_BYTES) as u16,
                    bytes: C220_VECTOR_BLOCK_BYTES as u16,
                    address: self
                        .source_address
                        .checked_add((block_index * C220_VECTOR_BLOCK_BYTES) as u64)
                        .ok_or(C220VectorError::SourceAddressOverflow {
                            source_index: 0,
                            base: self.source_address,
                            block: block_index,
                        })?,
                    active_lane_mask: u16::MAX,
                })
            })
            .collect()
    }
}

pub fn plan_c220_transpose_issue(
    pc: u64,
    word: u32,
    source_address: u64,
    destination_address: u64,
    ub: &UbMemory,
) -> Result<C220TransposeIssue, C220VectorError> {
    let instruction = C220TransposeInstruction::decode(word)
        .ok_or(C220VectorError::UnsupportedWord { pc, word })?;
    source_address.checked_add(TRANSPOSE_BYTES as u64).ok_or(
        C220VectorError::SourceAddressOverflow {
            source_index: 0,
            base: source_address,
            block: 15,
        },
    )?;
    destination_address
        .checked_add(TRANSPOSE_BYTES as u64)
        .ok_or(C220VectorError::AddressOverflow {
            base: destination_address,
            lane: 255,
        })?;
    let mut write_targets = Vec::with_capacity(TRANSPOSE_BYTES / 2);
    for block_index in 0..TRANSPOSE_BYTES / C220_VECTOR_BLOCK_BYTES {
        ub.check_range(
            source_address + (block_index * C220_VECTOR_BLOCK_BYTES) as u64,
            C220_VECTOR_BLOCK_BYTES,
        )?;
        ub.check_range(
            destination_address + (block_index * C220_VECTOR_BLOCK_BYTES) as u64,
            C220_VECTOR_BLOCK_BYTES,
        )?;
    }
    for lane in 0..TRANSPOSE_BYTES / 2 {
        let address = destination_address + (lane * 2) as u64;
        write_targets.push(C220VectorStore {
            repeat_index: lane / 128,
            lane_index: lane % 128,
            address,
            bank: C220UbBank::from_address(address),
            width_bytes: 2,
            data: [0; 8],
        });
    }
    Ok(C220TransposeIssue {
        pc,
        word,
        instruction,
        source_address,
        destination_address,
        write_targets,
    })
}

pub fn evaluate_c220_transpose(
    destination_address: u64,
    source_bytes: &[u8],
) -> Result<(Vec<u32>, Vec<C220VectorStore>), C220VectorError> {
    if source_bytes.len() != TRANSPOSE_BYTES {
        return Err(C220VectorError::InvalidSourceTile {
            actual: source_bytes.len(),
            expected: TRANSPOSE_BYTES,
        });
    }
    destination_address
        .checked_add(TRANSPOSE_BYTES as u64)
        .ok_or(C220VectorError::AddressOverflow {
            base: destination_address,
            lane: 255,
        })?;
    let mut values = Vec::with_capacity(TRANSPOSE_BYTES / 2);
    let mut stores = Vec::with_capacity(TRANSPOSE_BYTES / 2);
    for destination_lane in 0..TRANSPOSE_BYTES / 2 {
        let source_lane = (destination_lane % TRANSPOSE_DIMENSION) * TRANSPOSE_DIMENSION
            + destination_lane / TRANSPOSE_DIMENSION;
        let bytes: [u8; 2] = source_bytes[source_lane * 2..source_lane * 2 + 2]
            .try_into()
            .expect("two-byte lane");
        let address = destination_address + (destination_lane * 2) as u64;
        values.push(u32::from(u16::from_le_bytes(bytes)));
        stores.push(C220VectorStore {
            repeat_index: destination_lane / 128,
            lane_index: destination_lane % 128,
            address,
            bank: C220UbBank::from_address(address),
            width_bytes: 2,
            data: super::store_data(bytes),
        });
    }
    Ok((values, stores))
}
