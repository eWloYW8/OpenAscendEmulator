use super::{
    C220_VECTOR_BLOCK_BYTES, C220_VECTOR_BLOCK_COUNT, C220_VECTOR_TILE_BYTES, C220VectorControl,
    C220VectorError, check_repeat_limit,
};
use crate::isa::c220::vector::C220VecArithmeticHint;
use crate::memory::sparse::MemoryByteState;
use crate::memory::ub::{UbMemory, UbMemoryError};
use crate::sim::c220::memory::C220UbBank;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220VectorStore {
    pub repeat_index: usize,
    pub lane_index: usize,
    pub address: u64,
    pub bank: C220UbBank,
    pub width_bytes: u8,
    pub data: [u8; 8],
}

pub(crate) fn store_data<const N: usize>(bytes: [u8; N]) -> [u8; 8] {
    assert!(N <= 8, "C220 vector store payload exceeds one lane");
    let mut data = [0; 8];
    data[..N].copy_from_slice(&bytes);
    data
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220VectorReadAccess {
    pub source_index: u8,
    pub block_index: u8,
    pub buffer_offset: u16,
    pub bytes: u16,
    pub address: u64,
    pub active_lane_mask: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220VectorAddresses {
    pub source_0: u64,
    pub source_1: u64,
    pub destination: u64,
}

pub(crate) fn plan_c220_unary_write_targets(
    control: C220VectorControl,
    addresses: C220VectorAddresses,
    iteration_masks: &[[u64; 4]],
    element_bytes: u8,
    ub: &UbMemory,
) -> Result<Vec<C220VectorStore>, C220VectorError> {
    let lane_count = C220_VECTOR_TILE_BYTES / usize::from(element_bytes);
    check_repeat_limit(iteration_masks.len())?;
    let max_lanes = lane_count * iteration_masks.len();
    let mut write_targets = Vec::new();
    write_targets
        .try_reserve_exact(max_lanes)
        .map_err(|_| C220VectorError::HostAllocationFailed { lanes: max_lanes })?;
    for (repeat_index, mask) in iteration_masks.iter().enumerate() {
        for access in plan_c220_vector_read_accesses(
            control,
            addresses,
            repeat_index,
            mask,
            1,
            element_bytes,
            None,
        )? {
            ub.check_range(access.address, C220_VECTOR_BLOCK_BYTES)?;
        }
        for lane_index in 0..lane_count {
            if mask[lane_index / 64] & (1_u64 << (lane_index % 64)) == 0 {
                continue;
            }
            let address = vector_destination_address_for_width(
                control,
                addresses,
                repeat_index,
                lane_index,
                element_bytes,
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
    Ok(write_targets)
}

pub(crate) fn vector_destination_address(
    control: C220VectorControl,
    addresses: C220VectorAddresses,
    repeat_index: usize,
    lane_index: usize,
) -> Result<u64, C220VectorError> {
    vector_destination_address_for_width(control, addresses, repeat_index, lane_index, 4)
}

pub(crate) fn vector_destination_address_for_width(
    control: C220VectorControl,
    addresses: C220VectorAddresses,
    repeat_index: usize,
    lane_index: usize,
    element_bytes: u8,
) -> Result<u64, C220VectorError> {
    if !matches!(element_bytes, 1 | 2 | 4 | 8) {
        return Err(C220VectorError::UnsupportedElementWidth(element_bytes));
    }
    let lanes_per_block = C220_VECTOR_BLOCK_BYTES / usize::from(element_bytes);
    let block = lane_index / lanes_per_block;
    let offset = repeat_index as u64
        * u64::from(control.destination_repeat_stride)
        * C220_VECTOR_BLOCK_BYTES as u64
        + block as u64
            * u64::from(control.destination_block_stride.max(1))
            * C220_VECTOR_BLOCK_BYTES as u64
        + ((lane_index % lanes_per_block) * usize::from(element_bytes)) as u64;
    addresses
        .destination
        .checked_add(offset)
        .ok_or(C220VectorError::AddressOverflow {
            base: addresses.destination,
            lane: repeat_index * (C220_VECTOR_TILE_BYTES / usize::from(element_bytes)) + lane_index,
        })
}

pub fn plan_c220_vector_arithmetic_read_accesses(
    hint: C220VecArithmeticHint,
    control: C220VectorControl,
    addresses: C220VectorAddresses,
    repeat_index: usize,
    active_mask: &[u64; 4],
    lane_group: Option<u8>,
) -> Result<Vec<C220VectorReadAccess>, C220VectorError> {
    let element_bytes =
        hint.modeled_element_bytes()
            .ok_or(C220VectorError::UnsupportedArithmeticType(
                hint.dtype_selector,
            ))?;
    plan_c220_vector_read_accesses(
        control,
        addresses,
        repeat_index,
        active_mask,
        1 + usize::from(hint.source_1_register.is_some()),
        element_bytes,
        lane_group,
    )
}

pub(crate) fn plan_c220_vector_read_accesses(
    control: C220VectorControl,
    addresses: C220VectorAddresses,
    repeat_index: usize,
    active_mask: &[u64; 4],
    source_count: usize,
    element_bytes: u8,
    lane_group: Option<u8>,
) -> Result<Vec<C220VectorReadAccess>, C220VectorError> {
    if !matches!(element_bytes, 1 | 2 | 4 | 8) {
        return Err(C220VectorError::UnsupportedElementWidth(element_bytes));
    }
    let lanes_per_block = C220_VECTOR_BLOCK_BYTES / usize::from(element_bytes);
    let mut accesses = Vec::new();
    accesses
        .try_reserve_exact(C220_VECTOR_BLOCK_COUNT * source_count)
        .map_err(|_| UbMemoryError::HostAllocationFailed {
            requested: C220_VECTOR_BLOCK_COUNT * source_count,
        })?;
    let sources = [
        (
            0,
            addresses.source_0,
            control.source_0_block_stride,
            control.source_0_repeat_stride,
        ),
        (
            1,
            addresses.source_1,
            control.source_1_block_stride,
            control.source_1_repeat_stride,
        ),
    ];
    for (source_index, base, block_stride, repeat_stride) in sources.into_iter().take(source_count)
    {
        for block in 0..C220_VECTOR_BLOCK_COUNT {
            let first_lane = block * lanes_per_block;
            if lane_group.is_some_and(|group| first_lane / 64 != usize::from(group)) {
                continue;
            }
            let active_lane_mask = ((active_mask[first_lane / 64] >> (first_lane % 64))
                & ((1_u64 << lanes_per_block) - 1)) as u16;
            if active_lane_mask == 0 {
                continue;
            }
            let offset =
                repeat_index as u64 * C220_VECTOR_BLOCK_BYTES as u64 * u64::from(repeat_stride)
                    + block as u64 * C220_VECTOR_BLOCK_BYTES as u64 * u64::from(block_stride);
            let address =
                base.checked_add(offset)
                    .ok_or(C220VectorError::SourceAddressOverflow {
                        source_index,
                        base,
                        block: repeat_index * C220_VECTOR_BLOCK_COUNT + block,
                    })?;
            address
                .checked_add(C220_VECTOR_BLOCK_BYTES as u64)
                .ok_or(UbMemoryError::RangeOverflow)?;
            accesses.push(C220VectorReadAccess {
                source_index,
                block_index: block as u8,
                buffer_offset: (block * C220_VECTOR_BLOCK_BYTES) as u16,
                bytes: C220_VECTOR_BLOCK_BYTES as u16,
                address,
                active_lane_mask,
            });
        }
    }
    Ok(accesses)
}

pub(crate) fn plan_c220_destination_read_accesses(
    control: C220VectorControl,
    addresses: C220VectorAddresses,
    repeat_index: usize,
    active_mask: &[u64; 4],
    element_bytes: u8,
    lane_group: Option<u8>,
) -> Result<Vec<C220VectorReadAccess>, C220VectorError> {
    if !matches!(element_bytes, 1 | 2 | 4 | 8) {
        return Err(C220VectorError::UnsupportedElementWidth(element_bytes));
    }
    let lanes_per_block = C220_VECTOR_BLOCK_BYTES / usize::from(element_bytes);
    let lane_count = C220_VECTOR_TILE_BYTES / usize::from(element_bytes);
    let mut accesses = Vec::with_capacity(C220_VECTOR_BLOCK_COUNT);
    for block_index in 0..C220_VECTOR_BLOCK_COUNT {
        let first_lane = block_index * lanes_per_block;
        if first_lane >= lane_count
            || lane_group.is_some_and(|group| first_lane / 64 != usize::from(group))
        {
            continue;
        }
        let active_lane_mask = ((active_mask[first_lane / 64] >> (first_lane % 64))
            & ((1_u64 << lanes_per_block) - 1)) as u16;
        if active_lane_mask == 0 {
            continue;
        }
        accesses.push(C220VectorReadAccess {
            source_index: 2,
            block_index: block_index as u8,
            buffer_offset: (block_index * C220_VECTOR_BLOCK_BYTES) as u16,
            bytes: C220_VECTOR_BLOCK_BYTES as u16,
            address: vector_destination_address_for_width(
                control,
                addresses,
                repeat_index,
                first_lane,
                element_bytes,
            )?,
            active_lane_mask,
        });
    }
    Ok(accesses)
}

pub(super) fn commit_vector_stores(
    ub: &mut UbMemory,
    stores: &[C220VectorStore],
) -> Result<(), UbMemoryError> {
    let segments = stores
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
    ub.write_segments(&segments)
}
