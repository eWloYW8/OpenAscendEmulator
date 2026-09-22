use std::cmp::Ordering;

use crate::architecture::c220::C220UbBank;
use crate::isa::c220::sort::{C220SortInstruction, C220SortWidth};
use crate::memory::ub::UbMemory;
use crate::sim::c220::fp16::to_f64;

use super::{
    C220_VECTOR_BLOCK_BYTES, C220_VECTOR_TILE_BYTES, C220VectorAddresses, C220VectorControl,
    C220VectorError, C220VectorReadAccess, C220VectorStore, check_repeat_limit, store_data,
};

const SORT_LANES: usize = 32;
const RECORD_BYTES: usize = 8;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220SortIssue {
    pub pc: u64,
    pub word: u32,
    pub instruction: C220SortInstruction,
    pub repeat_count: u8,
    pub addresses: C220VectorAddresses,
    pub(crate) write_targets: Vec<C220VectorStore>,
}

impl C220SortIssue {
    pub const fn control(&self) -> C220VectorControl {
        C220VectorControl {
            encoded_repeat_count: self.repeat_count,
            destination_block_stride: 1,
            source_0_block_stride: 1,
            source_1_block_stride: 1,
            destination_repeat_stride: 8,
            source_0_repeat_stride: 8,
            source_1_repeat_stride: 8,
        }
    }

    pub fn read_accesses_for_repeat(
        &self,
        repeat_index: usize,
    ) -> Result<Vec<C220VectorReadAccess>, C220VectorError> {
        if repeat_index >= usize::from(self.repeat_count) {
            return Err(C220VectorError::InvalidRepeatIndex(repeat_index));
        }
        let value_bytes = SORT_LANES * usize::from(self.instruction.width.element_bytes());
        let repeat_offset = repeat_index * C220_VECTOR_TILE_BYTES;
        let mut accesses = Vec::with_capacity(value_bytes.div_ceil(32) + 4);
        for (source_index, bytes, base) in [
            (0_u8, value_bytes, self.addresses.source_0),
            (1_u8, SORT_LANES * 4, self.addresses.source_1),
        ] {
            for block in 0..bytes.div_ceil(C220_VECTOR_BLOCK_BYTES) {
                let offset = block * C220_VECTOR_BLOCK_BYTES;
                let address = base.checked_add((repeat_offset + offset) as u64).ok_or(
                    C220VectorError::SourceAddressOverflow {
                        source_index,
                        base,
                        block,
                    },
                )?;
                accesses.push(C220VectorReadAccess {
                    source_index,
                    block_index: block as u8,
                    buffer_offset: offset as u16,
                    bytes: C220_VECTOR_BLOCK_BYTES as u16,
                    address,
                    active_lane_mask: u16::MAX,
                });
            }
        }
        Ok(accesses)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220SortLaneOutcome {
    pub input_lane: u8,
    pub value_bits: u32,
    pub index: u32,
}

pub fn plan_c220_sort_issue(
    pc: u64,
    word: u32,
    control_value: u64,
    addresses: C220VectorAddresses,
    ub: &UbMemory,
) -> Result<C220SortIssue, C220VectorError> {
    let instruction =
        C220SortInstruction::decode(word).ok_or(C220VectorError::UnsupportedWord { pc, word })?;
    let repeat_count = (control_value >> 56) as u8;
    check_repeat_limit(usize::from(repeat_count))?;
    let value_bytes = SORT_LANES * usize::from(instruction.width.element_bytes());
    let index_bytes = SORT_LANES * 4;
    let mut write_targets = Vec::new();
    write_targets
        .try_reserve_exact(usize::from(repeat_count) * SORT_LANES)
        .map_err(|_| C220VectorError::HostAllocationFailed {
            lanes: usize::from(repeat_count) * SORT_LANES,
        })?;
    for repeat_index in 0..usize::from(repeat_count) {
        let repeat_offset = repeat_index * C220_VECTOR_TILE_BYTES;
        let value_address = addresses.source_0.checked_add(repeat_offset as u64).ok_or(
            C220VectorError::SourceAddressOverflow {
                source_index: 0,
                base: addresses.source_0,
                block: repeat_index,
            },
        )?;
        let index_address = addresses.source_1.checked_add(repeat_offset as u64).ok_or(
            C220VectorError::SourceAddressOverflow {
                source_index: 1,
                base: addresses.source_1,
                block: repeat_index,
            },
        )?;
        let destination = addresses
            .destination
            .checked_add(repeat_offset as u64)
            .ok_or(C220VectorError::AddressOverflow {
                base: addresses.destination,
                lane: repeat_index,
            })?;
        ub.check_range(value_address, value_bytes)?;
        ub.check_range(index_address, index_bytes)?;
        ub.check_range(destination, C220_VECTOR_TILE_BYTES)?;
        for lane_index in 0..SORT_LANES {
            let address = destination + (lane_index * RECORD_BYTES) as u64;
            write_targets.push(C220VectorStore {
                repeat_index,
                lane_index,
                address,
                bank: C220UbBank::from_address(address),
                width_bytes: RECORD_BYTES as u8,
                data: [0; 8],
            });
        }
    }
    Ok(C220SortIssue {
        pc,
        word,
        instruction,
        repeat_count,
        addresses,
        write_targets,
    })
}

pub fn evaluate_c220_sort_repeat(
    issue: &C220SortIssue,
    repeat_index: usize,
    value_bytes: &[u8],
    index_bytes: &[u8],
) -> Result<(Vec<C220SortLaneOutcome>, Vec<C220VectorStore>), C220VectorError> {
    if value_bytes.len() != C220_VECTOR_TILE_BYTES {
        return Err(C220VectorError::InvalidSourceTile {
            actual: value_bytes.len(),
            expected: C220_VECTOR_TILE_BYTES,
        });
    }
    if index_bytes.len() != C220_VECTOR_TILE_BYTES {
        return Err(C220VectorError::InvalidSourceTile {
            actual: index_bytes.len(),
            expected: C220_VECTOR_TILE_BYTES,
        });
    }
    if repeat_index >= usize::from(issue.repeat_count) {
        return Err(C220VectorError::InvalidRepeatIndex(repeat_index));
    }

    let width = usize::from(issue.instruction.width.element_bytes());
    let mut lanes = (0..SORT_LANES)
        .map(|lane| {
            let value_offset = lane * width;
            let value_bits = match issue.instruction.width {
                C220SortWidth::F16 => u32::from(u16::from_le_bytes(
                    value_bytes[value_offset..value_offset + 2]
                        .try_into()
                        .expect("two-byte sort value"),
                )),
                C220SortWidth::F32 => u32::from_le_bytes(
                    value_bytes[value_offset..value_offset + 4]
                        .try_into()
                        .expect("four-byte sort value"),
                ),
            };
            let index_offset = lane * 4;
            C220SortLaneOutcome {
                input_lane: lane as u8,
                value_bits,
                index: u32::from_le_bytes(
                    index_bytes[index_offset..index_offset + 4]
                        .try_into()
                        .expect("four-byte sort index"),
                ),
            }
        })
        .collect::<Vec<_>>();
    sort_records(issue.instruction.width, &mut lanes);

    let destination = issue
        .addresses
        .destination
        .checked_add((repeat_index * C220_VECTOR_TILE_BYTES) as u64)
        .ok_or(C220VectorError::AddressOverflow {
            base: issue.addresses.destination,
            lane: repeat_index,
        })?;
    let stores = lanes
        .iter()
        .enumerate()
        .map(|(lane_index, lane)| {
            let data = match issue.instruction.width {
                C220SortWidth::F16 => {
                    let mut bytes = [0; 8];
                    bytes[..2].copy_from_slice(&(lane.value_bits as u16).to_le_bytes());
                    bytes[4..].copy_from_slice(&lane.index.to_le_bytes());
                    bytes
                }
                C220SortWidth::F32 => {
                    let mut bytes = [0; 8];
                    bytes[..4].copy_from_slice(&lane.value_bits.to_le_bytes());
                    bytes[4..].copy_from_slice(&lane.index.to_le_bytes());
                    bytes
                }
            };
            let address = destination + (lane_index * RECORD_BYTES) as u64;
            C220VectorStore {
                repeat_index,
                lane_index,
                address,
                bank: C220UbBank::from_address(address),
                width_bytes: RECORD_BYTES as u8,
                data: store_data(data),
            }
        })
        .collect();
    Ok((lanes, stores))
}

fn compare_records(
    width: C220SortWidth,
    left: &C220SortLaneOutcome,
    right: &C220SortLaneOutcome,
) -> Ordering {
    let left_precedes = c220_sort_value_precedes(width, left.value_bits, right.value_bits);
    let right_precedes = c220_sort_value_precedes(width, right.value_bits, left.value_bits);
    match (left_precedes, right_precedes) {
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        _ => left.input_lane.cmp(&right.input_lane),
    }
}

fn sort_records(width: C220SortWidth, records: &mut [C220SortLaneOutcome]) {
    if records.len() < 2 {
        return;
    }
    let depth_limit = 2 * (usize::BITS - records.len().leading_zeros() - 1) as usize;
    introsort_loop(width, records, depth_limit);
    final_insertion_sort(width, records);
}

fn introsort_loop(
    width: C220SortWidth,
    mut records: &mut [C220SortLaneOutcome],
    mut depth_limit: usize,
) {
    while records.len() > 16 {
        if depth_limit == 0 {
            heap_sort(width, records);
            return;
        }
        depth_limit -= 1;
        let cut = unguarded_partition_pivot(width, records);
        let (left, right) = records.split_at_mut(cut);
        introsort_loop(width, right, depth_limit);
        records = left;
    }
}

fn unguarded_partition_pivot(width: C220SortWidth, records: &mut [C220SortLaneOutcome]) -> usize {
    let middle = records.len() / 2;
    move_median_to_first(width, records, 1, middle, records.len() - 1);
    let pivot = records[0];
    let mut first = 1;
    let mut last = records.len();
    loop {
        while precedes(width, records[first], pivot) {
            first += 1;
        }
        last -= 1;
        while precedes(width, pivot, records[last]) {
            last -= 1;
        }
        if first >= last {
            return first;
        }
        records.swap(first, last);
        first += 1;
    }
}

fn move_median_to_first(
    width: C220SortWidth,
    records: &mut [C220SortLaneOutcome],
    first: usize,
    middle: usize,
    last: usize,
) {
    if precedes(width, records[first], records[middle]) {
        if precedes(width, records[middle], records[last]) {
            records.swap(0, middle);
        } else if precedes(width, records[first], records[last]) {
            records.swap(0, last);
        } else {
            records.swap(0, first);
        }
    } else if precedes(width, records[first], records[last]) {
        records.swap(0, first);
    } else if precedes(width, records[middle], records[last]) {
        records.swap(0, last);
    } else {
        records.swap(0, middle);
    }
}

fn final_insertion_sort(width: C220SortWidth, records: &mut [C220SortLaneOutcome]) {
    if records.len() > 16 {
        insertion_sort(width, &mut records[..16]);
        for index in 16..records.len() {
            unguarded_linear_insert(width, records, index);
        }
    } else {
        insertion_sort(width, records);
    }
}

fn insertion_sort(width: C220SortWidth, records: &mut [C220SortLaneOutcome]) {
    for index in 1..records.len() {
        let value = records[index];
        if precedes(width, value, records[0]) {
            records.copy_within(0..index, 1);
            records[0] = value;
        } else {
            unguarded_linear_insert(width, records, index);
        }
    }
}

fn unguarded_linear_insert(
    width: C220SortWidth,
    records: &mut [C220SortLaneOutcome],
    index: usize,
) {
    let value = records[index];
    let mut hole = index;
    while precedes(width, value, records[hole - 1]) {
        records[hole] = records[hole - 1];
        hole -= 1;
    }
    records[hole] = value;
}

fn heap_sort(width: C220SortWidth, records: &mut [C220SortLaneOutcome]) {
    for root in (0..records.len() / 2).rev() {
        sift_down(width, records, root, records.len());
    }
    for end in (1..records.len()).rev() {
        records.swap(0, end);
        sift_down(width, records, 0, end);
    }
}

fn sift_down(
    width: C220SortWidth,
    records: &mut [C220SortLaneOutcome],
    mut root: usize,
    end: usize,
) {
    loop {
        let left = root * 2 + 1;
        if left >= end {
            return;
        }
        let right = left + 1;
        let child = if right < end && precedes(width, records[left], records[right]) {
            right
        } else {
            left
        };
        if !precedes(width, records[root], records[child]) {
            return;
        }
        records.swap(root, child);
        root = child;
    }
}

fn precedes(width: C220SortWidth, left: C220SortLaneOutcome, right: C220SortLaneOutcome) -> bool {
    compare_records(width, &left, &right) == Ordering::Less
}

pub(crate) fn c220_sort_value_precedes(width: C220SortWidth, left: u32, right: u32) -> bool {
    let sign = match width {
        C220SortWidth::F16 => 0x8000,
        C220SortWidth::F32 => 0x8000_0000,
    };
    let left_zero = left & !sign == 0;
    let right_zero = right & !sign == 0;
    if left == right || left_zero && right_zero {
        return false;
    }
    let is_special = |bits| match width {
        C220SortWidth::F16 => bits & 0x7c00 == 0x7c00,
        C220SortWidth::F32 => bits & 0x7f80_0000 == 0x7f80_0000,
    };
    let left_special = is_special(left);
    let right_special = is_special(right);
    match (left_special, right_special) {
        (true, false) => left & sign == 0,
        (false, true) => right & sign != 0,
        (true, true) if left & sign != 0 && right & sign != 0 => left < right,
        (true, true) => (left as i32) > (right as i32),
        (false, false) => match width {
            C220SortWidth::F16 => to_f64(left as u16) > to_f64(right as u16),
            C220SortWidth::F32 => f32::from_bits(left) > f32::from_bits(right),
        },
    }
}
