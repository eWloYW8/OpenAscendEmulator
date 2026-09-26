use super::{C220GatherIssue, C220GatherKind, packed_destination_address};
use crate::memory::{sparse::MemoryByteState, ub::UbMemory};
use crate::sim::c220::vector::C220VectorError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220GatherTransfer {
    pub lane: usize,
    pub source: u64,
    pub destination: u64,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220GatherExecution {
    pub pc: u64,
    pub tick: u64,
    pub repeat_index: usize,
    pub index_address: u64,
    pub indices: Vec<u32>,
    pub transfers: Vec<C220GatherTransfer>,
}

pub(crate) fn execute_gather_repeat(
    issue: &C220GatherIssue,
    repeat_index: usize,
    tick: u64,
    ub: &mut UbMemory,
) -> Result<C220GatherExecution, C220VectorError> {
    if repeat_index >= issue.repeat_count() {
        return Err(C220VectorError::InvalidRepeatIndex(repeat_index));
    }
    let (count, bytes, stride) = match issue.instruction.kind {
        C220GatherKind::Elements(width) => {
            let bytes = usize::from(width.element_bytes());
            (width.lane_count(), bytes, bytes)
        }
        C220GatherKind::Blocks => (
            8,
            32,
            32 * usize::from(issue.control.destination_block_stride),
        ),
    };
    let index_address = issue
        .index_address
        .checked_add((repeat_index * count * 4) as u64)
        .ok_or(C220VectorError::AddressOverflow {
            base: issue.index_address,
            lane: repeat_index,
        })?;
    let indices = ub
        .read_known(index_address, count * 4)?
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    let mut transfers = Vec::with_capacity(count);
    for (lane, &index) in indices.iter().enumerate() {
        if matches!(issue.instruction.kind, C220GatherKind::Elements(_)) {
            let mask = issue
                .iteration_masks
                .get(repeat_index)
                .ok_or(C220VectorError::MissingMaskState)?;
            if mask[lane / 64] & (1 << (lane % 64)) == 0 {
                continue;
            }
        }
        let source = u64::from(issue.control.base_offset.wrapping_add(index));
        let destination = packed_destination_address(
            issue.destination_address,
            issue.control.destination_repeat_stride,
            repeat_index,
            lane * stride,
        )?;
        let bytes = ub.read_known(source, bytes)?;
        ub.write_states(
            destination,
            &bytes
                .iter()
                .copied()
                .map(MemoryByteState::Known)
                .collect::<Vec<_>>(),
        )?;
        transfers.push(C220GatherTransfer {
            lane,
            source,
            destination,
            bytes,
        });
    }
    Ok(C220GatherExecution {
        pc: issue.pc,
        tick,
        repeat_index,
        index_address,
        indices,
        transfers,
    })
}
