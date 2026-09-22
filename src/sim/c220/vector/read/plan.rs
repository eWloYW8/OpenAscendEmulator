use super::{
    C220VectorReadError, C220VectorReadIssue, C220VectorReadOperation, PendingVectorRead, ReadPort,
};
use crate::sim::c220::vector::ops::select::C220SelectMode;
use crate::sim::c220::vector::timing::C220VectorUopKind;
use crate::sim::c220::vector::{
    C220_VECTOR_TILE_BYTES, C220VectorAddresses, C220VectorControl, C220VectorReadAccess,
};

fn restrict_mask_to_lane_slice(mask: &mut [u64; 4], first_lane: usize, lane_count: usize) {
    let end_lane = first_lane.saturating_add(lane_count).min(256);
    for lane in 0..256 {
        if lane < first_lane || lane >= end_lane {
            mask[lane / 64] &= !(1_u64 << (lane % 64));
        }
    }
}

fn retain_accesses_in_lane_slice(
    accesses: &mut Vec<C220VectorReadAccess>,
    kind: C220VectorUopKind,
    element_bytes: usize,
) {
    let Some((first_lane, lane_count)) = kind.lane_slice() else {
        return;
    };
    let end_lane = first_lane + lane_count;
    accesses.retain(|access| {
        let block_first = usize::from(access.buffer_offset) / element_bytes;
        block_first >= first_lane && block_first < end_lane
    });
}

fn retain_mixed_accesses_in_lane_slice(
    accesses: &mut Vec<C220VectorReadAccess>,
    kind: C220VectorUopKind,
    source_element_bytes: usize,
    destination_element_bytes: usize,
) {
    let Some((first_lane, lane_count)) = kind.lane_slice() else {
        return;
    };
    let end_lane = first_lane + lane_count;
    accesses.retain(|access| {
        let element_bytes = if access.source_index == 2 {
            destination_element_bytes
        } else {
            source_element_bytes
        };
        let block_first = usize::from(access.buffer_offset) / element_bytes;
        block_first >= first_lane && block_first < end_lane
    });
}

fn uop_lane_groups(kind: C220VectorUopKind, ordinary_group: u8) -> (usize, usize) {
    kind.lane_slice().map_or_else(
        || {
            let group = usize::from(ordinary_group);
            (group, group)
        },
        |(first_lane, lane_count)| (first_lane / 64, (first_lane + lane_count - 1) / 64),
    )
}

impl PendingVectorRead {
    pub(in crate::sim::c220::vector) fn new(
        issue: C220VectorReadIssue<'_>,
        repeat_index: usize,
        lane_group: Option<u8>,
        kind: C220VectorUopKind,
    ) -> Result<Self, C220VectorReadError> {
        let select_mask_load = matches!(
            issue,
            C220VectorReadIssue::Select(select)
                if matches!(select.mode, C220SelectMode::TensorTensor) && lane_group.is_none()
        );
        if !matches!(issue, C220VectorReadIssue::Gather(_))
            && (matches!(
                issue,
                C220VectorReadIssue::Broadcast(_)
                    | C220VectorReadIssue::LoadVa(_)
                    | C220VectorReadIssue::Transpose(_)
                    | C220VectorReadIssue::MoveMask(_)
                    | C220VectorReadIssue::Reduction(_)
                    | C220VectorReadIssue::Sort(_)
                    | C220VectorReadIssue::Nchw(_)
            ) || select_mask_load)
                != lane_group.is_none()
        {
            return Err(C220VectorReadError::InvalidUopScope);
        }
        let ordinary_group = lane_group.unwrap_or_default();
        let (pc, word, control, addresses, mask, accesses, operation) = match issue {
            C220VectorReadIssue::LoadVa(issue) => (
                issue.pc,
                issue.word,
                C220VectorControl {
                    encoded_repeat_count: 0,
                    destination_block_stride: 0,
                    source_0_block_stride: 0,
                    source_1_block_stride: 0,
                    destination_repeat_stride: 0,
                    source_0_repeat_stride: 0,
                    source_1_repeat_stride: 0,
                },
                C220VectorAddresses {
                    destination: 0,
                    source_0: issue.source_address,
                    source_1: 0,
                },
                [u64::MAX; 4],
                vec![issue.read_access()],
                C220VectorReadOperation::LoadVa { issue: *issue },
            ),
            C220VectorReadIssue::Arithmetic(issue) => {
                let mut mask = *issue
                    .iteration_masks
                    .get(repeat_index)
                    .ok_or(C220VectorReadError::MissingRepeatMask)?;
                if let Some((first_lane, lane_count)) = kind.lane_slice() {
                    restrict_mask_to_lane_slice(&mut mask, first_lane, lane_count);
                }
                let mut accesses = if let Some((first_lane, lane_count)) = kind.lane_slice() {
                    let first_group = first_lane / 64;
                    let last_group = (first_lane + lane_count - 1) / 64;
                    let mut accesses = Vec::new();
                    for group in first_group..=last_group {
                        accesses.extend(issue.read_accesses_for_repeat(repeat_index, group as u8)?);
                    }
                    accesses
                } else {
                    issue.read_accesses_for_repeat(repeat_index, ordinary_group)?
                };
                retain_accesses_in_lane_slice(
                    &mut accesses,
                    kind,
                    usize::from(issue.source_element_bytes),
                );
                (
                    issue.pc,
                    issue.word,
                    issue.control,
                    issue.addresses,
                    mask,
                    accesses,
                    C220VectorReadOperation::Arithmetic {
                        hint: issue.hint,
                        modes: issue.modes,
                    },
                )
            }
            C220VectorReadIssue::VectorScalar(issue) => {
                let mut mask = *issue
                    .iteration_masks
                    .get(repeat_index)
                    .ok_or(C220VectorReadError::MissingRepeatMask)?;
                if let Some((first_lane, lane_count)) = kind.lane_slice() {
                    restrict_mask_to_lane_slice(&mut mask, first_lane, lane_count);
                }
                let (first_group, last_group) = uop_lane_groups(kind, ordinary_group);
                let mut accesses = Vec::new();
                for group in first_group..=last_group {
                    accesses.extend(issue.read_accesses_for_repeat(repeat_index, group as u8)?);
                }
                retain_accesses_in_lane_slice(
                    &mut accesses,
                    kind,
                    usize::from(issue.instruction.dtype.element_bytes()),
                );
                (
                    issue.pc,
                    issue.word,
                    issue.control,
                    issue.addresses,
                    mask,
                    accesses,
                    C220VectorReadOperation::VectorScalar {
                        instruction: issue.instruction,
                        scalar: issue.scalar,
                    },
                )
            }
            C220VectorReadIssue::Shift(issue) => {
                let mut mask = *issue
                    .iteration_masks
                    .get(repeat_index)
                    .ok_or(C220VectorReadError::MissingRepeatMask)?;
                if let Some((first_lane, lane_count)) = kind.lane_slice() {
                    restrict_mask_to_lane_slice(&mut mask, first_lane, lane_count);
                }
                let (first_group, last_group) = uop_lane_groups(kind, ordinary_group);
                let mut accesses = Vec::new();
                for group in first_group..=last_group {
                    accesses.extend(issue.read_accesses_for_repeat(repeat_index, group as u8)?);
                }
                retain_accesses_in_lane_slice(
                    &mut accesses,
                    kind,
                    usize::from(issue.instruction.element_bytes),
                );
                (
                    issue.pc,
                    issue.word,
                    issue.control,
                    issue.addresses,
                    mask,
                    accesses,
                    C220VectorReadOperation::Shift {
                        instruction: issue.instruction,
                        shift: issue.shift,
                    },
                )
            }
            C220VectorReadIssue::Copy(issue) => {
                let mut mask = *issue
                    .iteration_masks
                    .get(repeat_index)
                    .ok_or(C220VectorReadError::MissingRepeatMask)?;
                if let Some((first_lane, lane_count)) = kind.lane_slice() {
                    restrict_mask_to_lane_slice(&mut mask, first_lane, lane_count);
                }
                let (first_group, last_group) = uop_lane_groups(kind, ordinary_group);
                let mut accesses = Vec::new();
                for group in first_group..=last_group {
                    accesses.extend(issue.read_accesses_for_repeat(repeat_index, group as u8)?);
                }
                retain_accesses_in_lane_slice(
                    &mut accesses,
                    kind,
                    usize::from(issue.instruction.element_bytes),
                );
                (
                    issue.pc,
                    issue.word,
                    issue.control,
                    issue.addresses,
                    mask,
                    accesses,
                    C220VectorReadOperation::Copy {
                        instruction: issue.instruction,
                    },
                )
            }
            C220VectorReadIssue::Broadcast(issue) => (
                issue.pc,
                issue.word,
                issue.control.vector_control(),
                issue.addresses(),
                [u64::MAX; 4],
                issue.read_accesses_for_repeat(repeat_index)?,
                C220VectorReadOperation::Broadcast {
                    instruction: issue.instruction,
                    control: issue.control,
                },
            ),
            C220VectorReadIssue::Transpose(issue) => (
                issue.pc,
                issue.word,
                C220VectorControl {
                    encoded_repeat_count: 0,
                    destination_block_stride: 0,
                    source_0_block_stride: 0,
                    source_1_block_stride: 0,
                    destination_repeat_stride: 0,
                    source_0_repeat_stride: 0,
                    source_1_repeat_stride: 0,
                },
                issue.addresses(),
                [u64::MAX; 4],
                issue.read_accesses()?,
                C220VectorReadOperation::Transpose,
            ),
            C220VectorReadIssue::CompareMask(issue) => {
                let mask = *issue
                    .iteration_masks
                    .get(repeat_index)
                    .ok_or(C220VectorReadError::MissingRepeatMask)?;
                (
                    issue.pc,
                    issue.word,
                    issue.control,
                    issue.addresses,
                    mask,
                    issue.read_accesses_for_uop(repeat_index)?,
                    C220VectorReadOperation::CompareMask {
                        issue: Box::new(issue.clone()),
                    },
                )
            }
            C220VectorReadIssue::MoveMask(issue) => (
                issue.pc,
                issue.word,
                C220VectorControl {
                    encoded_repeat_count: 1,
                    destination_block_stride: 0,
                    source_0_block_stride: 0,
                    source_1_block_stride: 0,
                    destination_repeat_stride: 0,
                    source_0_repeat_stride: 0,
                    source_1_repeat_stride: 0,
                },
                C220VectorAddresses {
                    destination: issue.address,
                    source_0: issue.address,
                    source_1: 0,
                },
                [u64::MAX; 4],
                issue.read_accesses(),
                C220VectorReadOperation::MoveMask {
                    issue: issue.clone(),
                },
            ),
            C220VectorReadIssue::Select(issue) if select_mask_load => (
                issue.pc,
                issue.word,
                issue.control,
                issue.addresses,
                [u64::MAX; 4],
                issue.mask_block_accesses(repeat_index, 0)?,
                C220VectorReadOperation::SelectMaskLoad {
                    issue: Box::new(issue.clone()),
                },
            ),
            C220VectorReadIssue::Select(issue) => (
                issue.pc,
                issue.word,
                issue.control,
                issue.addresses,
                issue.iteration_masks[repeat_index],
                issue.read_accesses_for_repeat(repeat_index, ordinary_group)?,
                C220VectorReadOperation::Select {
                    issue: Box::new(issue.clone()),
                },
            ),
            C220VectorReadIssue::PackedCompare(issue) => (
                issue.pc,
                issue.word,
                issue.control,
                issue.addresses,
                [u64::MAX; 4],
                issue.read_accesses_for_uop(repeat_index)?,
                C220VectorReadOperation::PackedCompare {
                    instruction: issue.instruction,
                    scalar_bits: issue.scalar_bits,
                },
            ),
            C220VectorReadIssue::Reduction(issue) => (
                issue.pc,
                issue.word,
                issue.control,
                issue.addresses,
                *issue
                    .iteration_masks
                    .get(repeat_index)
                    .ok_or(C220VectorReadError::MissingRepeatMask)?,
                issue.read_accesses_for_repeat(repeat_index)?,
                C220VectorReadOperation::Reduction {
                    issue: Box::new(issue.clone()),
                },
            ),
            C220VectorReadIssue::Sort(issue) => (
                issue.pc,
                issue.word,
                issue.control(),
                issue.addresses,
                [u64::MAX; 4],
                issue.read_accesses_for_repeat(repeat_index)?,
                C220VectorReadOperation::Sort {
                    issue: Box::new(issue.clone()),
                },
            ),
            C220VectorReadIssue::Ternary(issue) => {
                let mut mask = *issue
                    .iteration_masks
                    .get(repeat_index)
                    .ok_or(C220VectorReadError::MissingRepeatMask)?;
                if let Some((first_lane, lane_count)) = kind.lane_slice() {
                    restrict_mask_to_lane_slice(&mut mask, first_lane, lane_count);
                }
                let (first_group, last_group) = uop_lane_groups(kind, ordinary_group);
                let mut accesses = Vec::new();
                for group in first_group..=last_group {
                    accesses.extend(issue.read_accesses_for_repeat(repeat_index, group as u8)?);
                }
                retain_mixed_accesses_in_lane_slice(
                    &mut accesses,
                    kind,
                    usize::from(issue.instruction.width.source_element_bytes()),
                    usize::from(issue.instruction.width.destination_element_bytes()),
                );
                (
                    issue.pc,
                    issue.word,
                    issue.control,
                    issue.addresses,
                    mask,
                    accesses,
                    C220VectorReadOperation::Ternary {
                        issue: Box::new(issue.clone()),
                    },
                )
            }
            C220VectorReadIssue::Axpy(issue) => {
                let mut mask = *issue
                    .iteration_masks
                    .get(repeat_index)
                    .ok_or(C220VectorReadError::MissingRepeatMask)?;
                if let Some((first_lane, lane_count)) = kind.lane_slice() {
                    restrict_mask_to_lane_slice(&mut mask, first_lane, lane_count);
                }
                let (first_group, last_group) = uop_lane_groups(kind, ordinary_group);
                let mut accesses = Vec::new();
                for group in first_group..=last_group {
                    accesses.extend(issue.read_accesses_for_repeat(repeat_index, group as u8)?);
                }
                retain_mixed_accesses_in_lane_slice(
                    &mut accesses,
                    kind,
                    usize::from(issue.instruction.width.source_element_bytes()),
                    usize::from(issue.instruction.width.destination_element_bytes()),
                );
                (
                    issue.pc,
                    issue.word,
                    issue.control,
                    issue.addresses,
                    mask,
                    accesses,
                    C220VectorReadOperation::Axpy {
                        issue: Box::new(issue.clone()),
                    },
                )
            }
            C220VectorReadIssue::SpecialUnary(issue) => {
                let mut scoped_issue = issue.clone();
                let mask = scoped_issue
                    .iteration_masks
                    .get_mut(repeat_index)
                    .ok_or(C220VectorReadError::MissingRepeatMask)?;
                if let Some((first_lane, lane_count)) = kind.lane_slice() {
                    restrict_mask_to_lane_slice(mask, first_lane, lane_count);
                }
                let mask = *mask;
                let mut accesses = if let Some((first_lane, lane_count)) = kind.lane_slice() {
                    let first_group = first_lane / 64;
                    let last_group = (first_lane + lane_count - 1) / 64;
                    let mut accesses = Vec::new();
                    for group in first_group..=last_group {
                        accesses.extend(issue.read_accesses_for_repeat(repeat_index, group as u8)?);
                    }
                    accesses
                } else {
                    issue.read_accesses_for_repeat(repeat_index, ordinary_group)?
                };
                retain_accesses_in_lane_slice(
                    &mut accesses,
                    kind,
                    usize::from(issue.instruction.width.element_bytes()),
                );
                (
                    issue.pc,
                    issue.word,
                    issue.control,
                    issue.addresses,
                    mask,
                    accesses,
                    C220VectorReadOperation::SpecialUnary {
                        issue: Box::new(scoped_issue),
                    },
                )
            }
            C220VectorReadIssue::Conversion(issue) => {
                let mut mask = *issue
                    .iteration_masks
                    .get(repeat_index)
                    .ok_or(C220VectorReadError::MissingRepeatMask)?;
                if let Some((first_lane, lane_count)) = kind.lane_slice() {
                    restrict_mask_to_lane_slice(&mut mask, first_lane, lane_count);
                }
                let (first_group, last_group) = uop_lane_groups(kind, ordinary_group);
                let mut accesses = Vec::new();
                for group in first_group..=last_group {
                    for access in issue.read_accesses_for_repeat(repeat_index, group as u8)? {
                        if !accesses.contains(&access) {
                            accesses.push(access);
                        }
                    }
                }
                (
                    issue.pc,
                    issue.word,
                    issue.control,
                    issue.addresses,
                    mask,
                    accesses,
                    C220VectorReadOperation::Conversion {
                        issue: Box::new(issue.clone()),
                    },
                )
            }
            C220VectorReadIssue::Fused(issue) => {
                let mut mask = *issue
                    .iteration_masks
                    .get(repeat_index)
                    .ok_or(C220VectorReadError::MissingRepeatMask)?;
                if let Some((first_lane, lane_count)) = kind.lane_slice() {
                    restrict_mask_to_lane_slice(&mut mask, first_lane, lane_count);
                }
                let (first_group, last_group) = uop_lane_groups(kind, ordinary_group);
                let mut accesses = Vec::new();
                for group in first_group..=last_group {
                    for access in issue.read_accesses_for_repeat(repeat_index, group as u8)? {
                        if !accesses.contains(&access) {
                            accesses.push(access);
                        }
                    }
                }
                (
                    issue.pc,
                    issue.word,
                    issue.control,
                    issue.addresses,
                    mask,
                    accesses,
                    C220VectorReadOperation::Fused {
                        issue: Box::new(issue.clone()),
                    },
                )
            }
            C220VectorReadIssue::Gather(issue) => {
                let (accesses, operation) = match kind {
                    C220VectorUopKind::GatherIndex { group } => (
                        vec![issue.index_read_access(repeat_index, group)?],
                        C220VectorReadOperation::GatherIndex,
                    ),
                    C220VectorUopKind::GatherData { group } if lane_group == Some(group) => (
                        issue.data_read_accesses(repeat_index, group)?,
                        C220VectorReadOperation::GatherData {
                            issue: Box::new(issue.clone()),
                            group,
                        },
                    ),
                    _ => return Err(C220VectorReadError::InvalidUopScope),
                };
                (
                    issue.pc,
                    issue.word,
                    C220VectorControl {
                        encoded_repeat_count: issue.control.repeat_count,
                        destination_block_stride: u16::from(issue.control.destination_block_stride),
                        source_0_block_stride: 0,
                        source_1_block_stride: 0,
                        destination_repeat_stride: issue.control.destination_repeat_stride,
                        source_0_repeat_stride: 0,
                        source_1_repeat_stride: 0,
                    },
                    C220VectorAddresses {
                        source_0: issue.index_address,
                        source_1: 0,
                        destination: issue.destination_address,
                    },
                    [u64::MAX; 4],
                    accesses,
                    operation,
                )
            }
            C220VectorReadIssue::Nchw(issue) => {
                let tile_index = repeat_index / 2;
                (
                    issue.pc,
                    issue.word,
                    C220VectorControl {
                        encoded_repeat_count: issue.control.repeat_count,
                        destination_block_stride: 0,
                        source_0_block_stride: 0,
                        source_1_block_stride: 0,
                        destination_repeat_stride: issue.control.destination_repeat_stride,
                        source_0_repeat_stride: issue.control.source_repeat_stride,
                        source_1_repeat_stride: 0,
                    },
                    C220VectorAddresses {
                        source_0: issue.rows[tile_index].source[0],
                        source_1: 0,
                        destination: issue.rows[tile_index].destination[0],
                    },
                    [u64::MAX; 4],
                    issue.read_accesses_for_repeat(tile_index)?,
                    C220VectorReadOperation::Nchw {
                        instruction: issue.instruction,
                        rows: Box::new(issue.rows[tile_index]),
                    },
                )
            }
        };
        let source_0_bytes = vec![
            0;
            if matches!(
                &operation,
                C220VectorReadOperation::Transpose | C220VectorReadOperation::Nchw { .. }
            ) {
                512
            } else {
                C220_VECTOR_TILE_BYTES
            }
        ];
        let (port0, port1, destination_port) = match &operation {
            C220VectorReadOperation::GatherData { .. } => {
                let (port0, port1) = ReadPort::routed_pair(&accesses, [0, 1, 1])?;
                (port0, port1, ReadPort::new(&[])?)
            }
            C220VectorReadOperation::Ternary { .. } | C220VectorReadOperation::Axpy { .. } => {
                let (port0, port1) = ReadPort::routed_pair(&accesses, [0, 0, 1])?;
                (port0, port1, ReadPort::new(&[])?)
            }
            _ => {
                let for_source = |source_index| {
                    accesses
                        .iter()
                        .copied()
                        .filter(|access| access.source_index == source_index)
                        .collect::<Vec<_>>()
                };
                (
                    ReadPort::new(&for_source(0))?,
                    ReadPort::new(&for_source(1))?,
                    ReadPort::new(&for_source(2))?,
                )
            }
        };
        Ok(Self {
            pc,
            word,
            operation,
            control,
            addresses,
            mask,
            repeat_index,
            lane_group,
            lane_slice: kind.lane_slice(),
            accesses,
            port0,
            port1,
            destination_port,
            source_0_bytes,
            source_1_bytes: vec![0; C220_VECTOR_TILE_BYTES],
            destination_bytes: vec![0; C220_VECTOR_TILE_BYTES],
            ready_tick: None,
            sampled: false,
        })
    }
}
