use crate::isa::c220::vector::merge::{C220MergeInstruction, C220MergeWidth};
use crate::isa::c220::vector::sort::C220SortWidth;
use crate::memory::ub::UbMemory;

use super::sort::c220_sort_value_precedes;
use crate::sim::c220::vector::C220VectorError;

const RECORD_BYTES: u64 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MergeControl {
    pub encoded_repeat_count: u8,
    pub valid_lists: u8,
    pub exhausted_suspension: bool,
}

impl C220MergeControl {
    pub const fn decode(value: u64) -> Self {
        Self {
            encoded_repeat_count: value as u8,
            valid_lists: ((value >> 8) & 0xf) as u8,
            exhausted_suspension: value & (1 << 12) != 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MergeRepeat {
    pub source_addresses: [u64; 4],
    pub destination_address: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220MergeIssue {
    pub pc: u64,
    pub word: u32,
    pub instruction: C220MergeInstruction,
    pub control: C220MergeControl,
    pub list_lengths: [u16; 4],
    pub repeats: Vec<C220MergeRepeat>,
}

impl C220MergeIssue {
    pub fn repeat_count(&self) -> usize {
        self.repeats.len()
    }

    pub const fn list_is_active(&self, list: usize) -> bool {
        self.control.valid_lists & (1 << list) != 0
    }

    pub fn maximum_output_records(&self) -> usize {
        (0..4)
            .filter(|&list| self.list_is_active(list))
            .map(|list| usize::from(self.list_lengths[list]))
            .sum()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MergeRecord {
    pub source_list: u8,
    pub source_index: u16,
    pub key_bits: u32,
    pub bytes: [u8; 8],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220MergeRepeatData {
    pub repeat_index: usize,
    pub lists: [Vec<C220MergeRecord>; 4],
    pub output: Vec<C220MergeRecord>,
    pub consumed: [u16; 4],
}

pub fn plan_c220_merge_issue(
    pc: u64,
    word: u32,
    registers: &[u64; 32],
    ub: &UbMemory,
) -> Result<C220MergeIssue, C220VectorError> {
    let instruction =
        C220MergeInstruction::decode(word).ok_or(C220VectorError::UnsupportedWord { pc, word })?;
    let packed_sources = registers[usize::from(instruction.sources_register)];
    let packed_lengths = registers[usize::from(instruction.lengths_register)];
    let destination = registers[usize::from(instruction.destination_register)];
    let control = C220MergeControl::decode(registers[usize::from(instruction.control_register)]);
    let list_lengths = unpack_u16x4(packed_lengths);

    let active_lengths_valid = (0..4)
        .filter(|&list| control.valid_lists & (1 << list) != 0)
        .all(|list| list_lengths[list] != 0);
    let mut repeat_count = usize::from(control.encoded_repeat_count);
    if control.valid_lists == 0 || !active_lengths_valid {
        repeat_count = 0;
    } else if control.exhausted_suspension {
        repeat_count = 1;
    } else if repeat_count > 1
        && (control.valid_lists != 0xf || !list_lengths.iter().all(|&len| len == list_lengths[0]))
    {
        repeat_count = 0;
    }

    let packed_source_fields = unpack_u16x4(packed_sources);
    let mut repeats = Vec::new();
    repeats
        .try_reserve_exact(repeat_count)
        .map_err(|_| C220VectorError::HostAllocationFailed {
            lanes: repeat_count,
        })?;
    for repeat_index in 0..repeat_count {
        let source_addresses = if repeat_count > 1 {
            let length = u64::from(list_lengths[0]);
            let repeat_stride =
                length
                    .checked_mul(32)
                    .ok_or(C220VectorError::SourceAddressOverflow {
                        source_index: 0,
                        base: packed_sources,
                        block: repeat_index,
                    })?;
            let base = u64::from(packed_source_fields[0])
                .checked_mul(RECORD_BYTES)
                .and_then(|value| {
                    value.checked_add(repeat_stride.checked_mul(repeat_index as u64)?)
                })
                .ok_or(C220VectorError::SourceAddressOverflow {
                    source_index: 0,
                    base: packed_sources,
                    block: repeat_index,
                })?;
            let mut addresses = [0; 4];
            for (list, address) in addresses.iter_mut().enumerate() {
                *address = base
                    .checked_add(length * RECORD_BYTES * list as u64)
                    .ok_or(C220VectorError::SourceAddressOverflow {
                        source_index: list as u8,
                        base,
                        block: repeat_index,
                    })?;
            }
            addresses
        } else {
            packed_source_fields.map(|field| u64::from(field) * RECORD_BYTES)
        };
        let destination_stride =
            u64::from(list_lengths[0])
                .checked_mul(32)
                .ok_or(C220VectorError::AddressOverflow {
                    base: destination,
                    lane: repeat_index,
                })?;
        let destination_address = destination
            .checked_add(destination_stride * repeat_index as u64)
            .ok_or(C220VectorError::AddressOverflow {
                base: destination,
                lane: repeat_index,
            })?;
        validate_repeat_ranges(
            ub,
            control.valid_lists,
            list_lengths,
            source_addresses,
            destination_address,
            repeat_index,
        )?;
        repeats.push(C220MergeRepeat {
            source_addresses,
            destination_address,
        });
    }

    Ok(C220MergeIssue {
        pc,
        word,
        instruction,
        control,
        list_lengths,
        repeats,
    })
}

pub fn load_c220_merge_repeat(
    issue: &C220MergeIssue,
    repeat_index: usize,
    ub: &UbMemory,
) -> Result<C220MergeRepeatData, C220VectorError> {
    let repeat = issue
        .repeats
        .get(repeat_index)
        .ok_or(C220VectorError::InvalidRepeatIndex(repeat_index))?;
    let mut lists: [Vec<C220MergeRecord>; 4] = std::array::from_fn(|_| Vec::new());
    for (list, records) in lists.iter_mut().enumerate() {
        if !issue.list_is_active(list) {
            continue;
        }
        let length = usize::from(issue.list_lengths[list]);
        records
            .try_reserve_exact(length)
            .map_err(|_| C220VectorError::HostAllocationFailed { lanes: length })?;
        for source_index in 0..length {
            let address = repeat.source_addresses[list]
                .checked_add(RECORD_BYTES * source_index as u64)
                .ok_or(C220VectorError::SourceAddressOverflow {
                    source_index: list as u8,
                    base: repeat.source_addresses[list],
                    block: source_index,
                })?;
            let bytes: [u8; 8] = ub
                .read_known(address, RECORD_BYTES as usize)?
                .try_into()
                .expect("eight-byte merge record");
            let key_bits = match issue.instruction.width {
                C220MergeWidth::F16 => u32::from(u16::from_le_bytes([bytes[0], bytes[1]])),
                C220MergeWidth::F32 => {
                    u32::from_le_bytes(bytes[..4].try_into().expect("four-byte key"))
                }
            };
            records.push(C220MergeRecord {
                source_list: list as u8,
                source_index: source_index as u16,
                key_bits,
                bytes,
            });
        }
    }
    let (output, consumed) = merge_loaded_lists(issue, &lists)?;
    Ok(C220MergeRepeatData {
        repeat_index,
        lists,
        output,
        consumed,
    })
}

fn merge_loaded_lists(
    issue: &C220MergeIssue,
    lists: &[Vec<C220MergeRecord>; 4],
) -> Result<(Vec<C220MergeRecord>, [u16; 4]), C220VectorError> {
    let maximum = issue.maximum_output_records();
    let mut output = Vec::new();
    output
        .try_reserve_exact(maximum)
        .map_err(|_| C220VectorError::HostAllocationFailed { lanes: maximum })?;
    let mut consumed = [0_u16; 4];
    while output.len() < maximum {
        let mut selected = None;
        for list in 0..4 {
            if !issue.list_is_active(list) {
                continue;
            }
            let index = usize::from(consumed[list]);
            let Some(candidate) = lists[list].get(index) else {
                continue;
            };
            if selected.is_none_or(|current: usize| {
                merge_record_precedes(
                    issue.instruction.width,
                    candidate,
                    &lists[current][usize::from(consumed[current])],
                )
            }) {
                selected = Some(list);
            }
        }
        let Some(list) = selected else {
            break;
        };
        let mut record = lists[list][usize::from(consumed[list])];
        if issue.instruction.width == C220MergeWidth::F16 {
            record.bytes[2..4].fill(0);
        }
        output.push(record);
        consumed[list] += 1;
        if issue.control.exhausted_suspension && consumed[list] == issue.list_lengths[list] {
            break;
        }
    }
    Ok((output, consumed))
}

pub(crate) fn merge_record_precedes(
    width: C220MergeWidth,
    left: &C220MergeRecord,
    right: &C220MergeRecord,
) -> bool {
    let sort_width = match width {
        C220MergeWidth::F16 => C220SortWidth::F16,
        C220MergeWidth::F32 => C220SortWidth::F32,
    };
    let left_precedes = c220_sort_value_precedes(sort_width, left.key_bits, right.key_bits);
    let right_precedes = c220_sort_value_precedes(sort_width, right.key_bits, left.key_bits);
    match (left_precedes, right_precedes) {
        (true, false) => true,
        (false, true) => false,
        _ => left.source_list < right.source_list,
    }
}

fn validate_repeat_ranges(
    ub: &UbMemory,
    valid_lists: u8,
    lengths: [u16; 4],
    sources: [u64; 4],
    destination: u64,
    repeat_index: usize,
) -> Result<(), C220VectorError> {
    for list in 0..4 {
        if valid_lists & (1 << list) == 0 {
            continue;
        }
        let last = sources[list]
            .checked_add((u64::from(lengths[list]) - 1) * RECORD_BYTES)
            .ok_or(C220VectorError::SourceAddressOverflow {
                source_index: list as u8,
                base: sources[list],
                block: usize::from(lengths[list]),
            })?;
        ub.check_range(last, RECORD_BYTES as usize)?;
    }
    let records = (0..4)
        .filter(|&list| valid_lists & (1 << list) != 0)
        .map(|list| u64::from(lengths[list]))
        .sum::<u64>();
    let last = destination
        .checked_add((records - 1) * RECORD_BYTES)
        .ok_or(C220VectorError::AddressOverflow {
            base: destination,
            lane: repeat_index,
        })?;
    ub.check_range(last, RECORD_BYTES as usize)?;
    Ok(())
}

const fn unpack_u16x4(value: u64) -> [u16; 4] {
    [
        value as u16,
        (value >> 16) as u16,
        (value >> 32) as u16,
        (value >> 48) as u16,
    ]
}
