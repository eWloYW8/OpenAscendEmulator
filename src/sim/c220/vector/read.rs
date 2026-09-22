use std::collections::BTreeMap;

use thiserror::Error;

use crate::isa::c220::compare::{C220MoveMaskDirection, C220PackedCompareInstruction};
use crate::isa::c220::vector::{
    C220BroadcastInstruction, C220CopyInstruction, C220NchwElement, C220NchwInstruction,
    C220ShiftInstruction, C220VecArithmeticHint,
};
use crate::isa::c220::vector_scalar::C220VectorScalarInstruction;
use crate::memory::ub::UbMemory;
use crate::numeric::fp32::Fp32ValueStatus;
use crate::sim::c220::fp16::C220Fp16Status;
use crate::sim::c220::ub_arbiter::{C220UbDecision, C220UbPort, C220UbRequest, C220UbRequestError};
use crate::sim::c220::vector::axpy::{C220AxpyIssue, evaluate_c220_axpy_repeat};
use crate::sim::c220::vector::broadcast::{
    C220BroadcastControl, C220BroadcastIssue, evaluate_c220_broadcast_repeat,
};
use crate::sim::c220::vector::compare::{
    C220CompareMask, C220CompareMaskIssue, C220CompareMaskUpdate, C220MoveMaskIssue,
    C220PackedCompareIssue, C220PackedCompareValueInputs, evaluate_c220_compare_mask_uop,
    evaluate_c220_packed_compare_uop,
};
use crate::sim::c220::vector::conversion::{
    C220ConversionIssue, C220ConversionLaneOutcome, evaluate_c220_conversion_repeat,
};
use crate::sim::c220::vector::copy::{
    C220CopyIssue, C220CopyValueInputs, evaluate_c220_copy_repeat,
};
use crate::sim::c220::vector::f16::{C220F16ValueInputs, evaluate_c220_f16_repeat_from_bytes};
use crate::sim::c220::vector::fused::{
    C220FusedIssue, C220FusedLaneOutcome, evaluate_c220_fused_repeat,
};
use crate::sim::c220::vector::gather::{C220GatherIssue, evaluate_c220_gather_data_uop};
use crate::sim::c220::vector::nchw::{C220NchwIssue, C220NchwRows, evaluate_c220_nchw_repeat};
use crate::sim::c220::vector::reduce::{
    C220ReductionIssue, C220ReductionStateUpdate, evaluate_c220_reduction_repeat,
};
use crate::sim::c220::vector::s16::{
    C220S16ValueInputs, C220S16WidenInputs, evaluate_c220_s16_repeat_from_bytes,
    evaluate_c220_s16_widen_repeat_from_bytes,
};
use crate::sim::c220::vector::s32::{C220S32ValueInputs, evaluate_c220_s32_repeat_from_bytes};
use crate::sim::c220::vector::scalar::{
    C220VectorScalarIssue, C220VectorScalarOperand, C220VectorScalarValueInputs,
    evaluate_c220_vector_scalar_repeat,
};
use crate::sim::c220::vector::select::{
    C220SelectIssue, C220SelectMode, C220SelectionMaskBlock, evaluate_c220_select_uop,
};
use crate::sim::c220::vector::shift::{
    C220ShiftIssue, C220ShiftValueInputs, evaluate_c220_shift_repeat,
};
use crate::sim::c220::vector::sort::{
    C220SortIssue, C220SortLaneOutcome, evaluate_c220_sort_repeat,
};
use crate::sim::c220::vector::special::{
    C220SpecialUnaryIssue, evaluate_c220_special_unary_repeat,
};
use crate::sim::c220::vector::ternary::{C220TernaryIssue, evaluate_c220_ternary_repeat};
use crate::sim::c220::vector::timing::C220VectorUopKind;
use crate::sim::c220::vector::transpose::{C220TransposeIssue, evaluate_c220_transpose};
use crate::sim::c220::vector::{
    C220_VECTOR_TILE_BYTES, C220VectorAddresses, C220VectorArithmeticIssue,
    C220VectorArithmeticModes, C220VectorControl, C220VectorError, C220VectorReadAccess,
    C220VectorStore, evaluate_c220_fp32_repeat_from_bytes,
};

#[derive(Debug, Error)]
pub enum C220VectorReadError {
    #[error("vector repeat has no iteration mask")]
    MissingRepeatMask,
    #[error("vector uop lane scope does not match the instruction")]
    InvalidUopScope,
    #[error(transparent)]
    ReadPlan(#[from] C220VectorError),
    #[error(transparent)]
    UbRequest(#[from] C220UbRequestError),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PendingVectorRead {
    pc: u64,
    word: u32,
    operation: C220VectorReadOperation,
    control: C220VectorControl,
    addresses: C220VectorAddresses,
    mask: [u64; 4],
    repeat_index: usize,
    lane_group: Option<u8>,
    accesses: Vec<C220VectorReadAccess>,
    port0: ReadPort,
    port1: ReadPort,
    destination_port: ReadPort,
    source_0_bytes: Vec<u8>,
    source_1_bytes: Vec<u8>,
    destination_bytes: Vec<u8>,
    ready_tick: Option<u64>,
    sampled: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum C220VectorReadOperation {
    Arithmetic {
        hint: C220VecArithmeticHint,
        modes: C220VectorArithmeticModes,
    },
    VectorScalar {
        instruction: C220VectorScalarInstruction,
        scalar: C220VectorScalarOperand,
    },
    Shift {
        instruction: C220ShiftInstruction,
        shift: u32,
    },
    Copy {
        instruction: C220CopyInstruction,
    },
    Broadcast {
        instruction: C220BroadcastInstruction,
        control: C220BroadcastControl,
    },
    Transpose,
    CompareMask {
        issue: Box<C220CompareMaskIssue>,
    },
    MoveMask {
        issue: C220MoveMaskIssue,
    },
    Select {
        issue: Box<C220SelectIssue>,
    },
    SelectMaskLoad {
        issue: Box<C220SelectIssue>,
    },
    PackedCompare {
        instruction: C220PackedCompareInstruction,
        scalar_bits: Option<u32>,
    },
    Reduction {
        issue: Box<C220ReductionIssue>,
    },
    Sort {
        issue: Box<C220SortIssue>,
    },
    Ternary {
        issue: Box<C220TernaryIssue>,
    },
    Axpy {
        issue: Box<C220AxpyIssue>,
    },
    SpecialUnary {
        issue: Box<C220SpecialUnaryIssue>,
    },
    Conversion {
        issue: Box<C220ConversionIssue>,
    },
    Fused {
        issue: Box<C220FusedIssue>,
    },
    GatherIndex,
    GatherData {
        issue: Box<C220GatherIssue>,
        group: u8,
    },
    Nchw {
        instruction: C220NchwInstruction,
        rows: Box<C220NchwRows>,
    },
}

#[derive(Debug, Clone, Copy)]
pub enum C220VectorReadIssue<'a> {
    Arithmetic(&'a C220VectorArithmeticIssue),
    VectorScalar(&'a C220VectorScalarIssue),
    Shift(&'a C220ShiftIssue),
    Copy(&'a C220CopyIssue),
    Broadcast(&'a C220BroadcastIssue),
    Transpose(&'a C220TransposeIssue),
    CompareMask(&'a C220CompareMaskIssue),
    MoveMask(&'a C220MoveMaskIssue),
    Select(&'a C220SelectIssue),
    PackedCompare(&'a C220PackedCompareIssue),
    Reduction(&'a C220ReductionIssue),
    Sort(&'a C220SortIssue),
    Ternary(&'a C220TernaryIssue),
    Axpy(&'a C220AxpyIssue),
    SpecialUnary(&'a C220SpecialUnaryIssue),
    Conversion(&'a C220ConversionIssue),
    Fused(&'a C220FusedIssue),
    Gather(&'a C220GatherIssue),
    Nchw(&'a C220NchwIssue),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ReadPort {
    request: C220UbRequest,
    destinations: Vec<Vec<ReadDestination>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ReadDestination {
    source_index: u8,
    offset: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ReadSpan {
    address: u64,
    bytes: u16,
    destinations: Vec<ReadDestination>,
}

impl ReadPort {
    fn new(accesses: &[C220VectorReadAccess]) -> Result<Self, C220UbRequestError> {
        Self::from_spans(
            accesses
                .iter()
                .map(|access| ReadSpan {
                    address: access.address,
                    bytes: access.bytes,
                    destinations: vec![ReadDestination {
                        source_index: access.source_index,
                        offset: usize::from(access.buffer_offset),
                    }],
                })
                .collect(),
        )
    }

    fn gather_pair(accesses: &[C220VectorReadAccess]) -> Result<(Self, Self), C220UbRequestError> {
        let mut spans = [Vec::<ReadSpan>::new(), Vec::<ReadSpan>::new()];
        let mut unique = BTreeMap::<(u64, u16), (usize, usize)>::new();
        for access in accesses {
            let destination = ReadDestination {
                source_index: access.source_index,
                offset: usize::from(access.buffer_offset),
            };
            if let Some(&(port, index)) = unique.get(&(access.address, access.bytes)) {
                spans[port][index].destinations.push(destination);
                continue;
            }
            let port = usize::from(access.source_index.min(1));
            let index = spans[port].len();
            spans[port].push(ReadSpan {
                address: access.address,
                bytes: access.bytes,
                destinations: vec![destination],
            });
            unique.insert((access.address, access.bytes), (port, index));
        }
        let [port0, port1] = spans;
        Ok((Self::from_spans(port0)?, Self::from_spans(port1)?))
    }

    fn from_spans(spans: Vec<ReadSpan>) -> Result<Self, C220UbRequestError> {
        let requests = spans
            .iter()
            .map(|span| (span.address, usize::from(span.bytes)))
            .collect::<Vec<_>>();
        let request = C220UbRequest::from_accesses(&requests)?;
        let mut destinations = Vec::with_capacity(request.blocks().len());
        let mut next_block = 0;
        for span in spans {
            let mut address = span.address;
            let end = address + u64::from(span.bytes);
            while address < end {
                let block = request.blocks()[next_block];
                debug_assert_eq!(block.address, address);
                destinations.push(
                    span.destinations
                        .iter()
                        .map(|destination| ReadDestination {
                            source_index: destination.source_index,
                            offset: destination.offset + (address - span.address) as usize,
                        })
                        .collect(),
                );
                address += u64::from(block.bytes);
                next_block += 1;
            }
        }
        Ok(Self {
            request,
            destinations,
        })
    }
}

impl PendingVectorRead {
    pub(super) fn new(
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
                    | C220VectorReadIssue::Transpose(_)
                    | C220VectorReadIssue::MoveMask(_)
                    | C220VectorReadIssue::PackedCompare(_)
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
            C220VectorReadIssue::Arithmetic(issue) => (
                issue.pc,
                issue.word,
                issue.control,
                issue.addresses,
                *issue
                    .iteration_masks
                    .get(repeat_index)
                    .ok_or(C220VectorReadError::MissingRepeatMask)?,
                issue.read_accesses_for_repeat(repeat_index, ordinary_group)?,
                C220VectorReadOperation::Arithmetic {
                    hint: issue.hint,
                    modes: issue.modes,
                },
            ),
            C220VectorReadIssue::VectorScalar(issue) => (
                issue.pc,
                issue.word,
                issue.control,
                issue.addresses,
                *issue
                    .iteration_masks
                    .get(repeat_index)
                    .ok_or(C220VectorReadError::MissingRepeatMask)?,
                issue.read_accesses_for_repeat(repeat_index, ordinary_group)?,
                C220VectorReadOperation::VectorScalar {
                    instruction: issue.instruction,
                    scalar: issue.scalar,
                },
            ),
            C220VectorReadIssue::Shift(issue) => (
                issue.pc,
                issue.word,
                issue.control,
                issue.addresses,
                *issue
                    .iteration_masks
                    .get(repeat_index)
                    .ok_or(C220VectorReadError::MissingRepeatMask)?,
                issue.read_accesses_for_repeat(repeat_index, ordinary_group)?,
                C220VectorReadOperation::Shift {
                    instruction: issue.instruction,
                    shift: issue.shift,
                },
            ),
            C220VectorReadIssue::Copy(issue) => (
                issue.pc,
                issue.word,
                issue.control,
                issue.addresses,
                *issue
                    .iteration_masks
                    .get(repeat_index)
                    .ok_or(C220VectorReadError::MissingRepeatMask)?,
                issue.read_accesses_for_repeat(repeat_index, ordinary_group)?,
                C220VectorReadOperation::Copy {
                    instruction: issue.instruction,
                },
            ),
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
            C220VectorReadIssue::CompareMask(issue) => (
                issue.pc,
                issue.word,
                issue.control,
                issue.addresses,
                issue.iteration_masks[repeat_index / issue.instruction.width.groups_per_repeat()],
                issue.read_accesses_for_uop(repeat_index)?,
                C220VectorReadOperation::CompareMask {
                    issue: Box::new(issue.clone()),
                },
            ),
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
            C220VectorReadIssue::Ternary(issue) => (
                issue.pc,
                issue.word,
                issue.control,
                issue.addresses,
                *issue
                    .iteration_masks
                    .get(repeat_index)
                    .ok_or(C220VectorReadError::MissingRepeatMask)?,
                issue.read_accesses_for_repeat(repeat_index, ordinary_group)?,
                C220VectorReadOperation::Ternary {
                    issue: Box::new(issue.clone()),
                },
            ),
            C220VectorReadIssue::Axpy(issue) => (
                issue.pc,
                issue.word,
                issue.control,
                issue.addresses,
                *issue
                    .iteration_masks
                    .get(repeat_index)
                    .ok_or(C220VectorReadError::MissingRepeatMask)?,
                issue.read_accesses_for_repeat(repeat_index, ordinary_group)?,
                C220VectorReadOperation::Axpy {
                    issue: Box::new(issue.clone()),
                },
            ),
            C220VectorReadIssue::SpecialUnary(issue) => (
                issue.pc,
                issue.word,
                issue.control,
                issue.addresses,
                *issue
                    .iteration_masks
                    .get(repeat_index)
                    .ok_or(C220VectorReadError::MissingRepeatMask)?,
                issue.read_accesses_for_repeat(repeat_index, ordinary_group)?,
                C220VectorReadOperation::SpecialUnary {
                    issue: Box::new(issue.clone()),
                },
            ),
            C220VectorReadIssue::Conversion(issue) => (
                issue.pc,
                issue.word,
                issue.control,
                issue.addresses,
                *issue
                    .iteration_masks
                    .get(repeat_index)
                    .ok_or(C220VectorReadError::MissingRepeatMask)?,
                issue.read_accesses_for_repeat(repeat_index, ordinary_group)?,
                C220VectorReadOperation::Conversion {
                    issue: Box::new(issue.clone()),
                },
            ),
            C220VectorReadIssue::Fused(issue) => (
                issue.pc,
                issue.word,
                issue.control,
                issue.addresses,
                *issue
                    .iteration_masks
                    .get(repeat_index)
                    .ok_or(C220VectorReadError::MissingRepeatMask)?,
                issue.read_accesses_for_repeat(repeat_index, ordinary_group)?,
                C220VectorReadOperation::Fused {
                    issue: Box::new(issue.clone()),
                },
            ),
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
        let port0_accesses = accesses
            .iter()
            .copied()
            .filter(|access| access.source_index == 0)
            .collect::<Vec<_>>();
        let port1_accesses = accesses
            .iter()
            .copied()
            .filter(|access| access.source_index == 1)
            .collect::<Vec<_>>();
        let destination_accesses = accesses
            .iter()
            .copied()
            .filter(|access| access.source_index == 2)
            .collect::<Vec<_>>();
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
        let (port0, port1) = if matches!(operation, C220VectorReadOperation::GatherData { .. }) {
            ReadPort::gather_pair(&accesses)?
        } else {
            (
                ReadPort::new(&port0_accesses)?,
                ReadPort::new(&port1_accesses)?,
            )
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
            accesses,
            port0,
            port1,
            destination_port: ReadPort::new(&destination_accesses)?,
            source_0_bytes,
            source_1_bytes: vec![0; C220_VECTOR_TILE_BYTES],
            destination_bytes: vec![0; C220_VECTOR_TILE_BYTES],
            ready_tick: None,
            sampled: false,
        })
    }

    fn port(&self, port: C220UbPort) -> &ReadPort {
        match port {
            C220UbPort::VectorRead0 => &self.port0,
            C220UbPort::VectorRead1 => &self.port1,
            C220UbPort::VectorReadDestination => &self.destination_port,
            C220UbPort::VectorWrite => unreachable!("read request selected a write port"),
        }
    }

    fn port_mut(&mut self, port: C220UbPort) -> &mut ReadPort {
        match port {
            C220UbPort::VectorRead0 => &mut self.port0,
            C220UbPort::VectorRead1 => &mut self.port1,
            C220UbPort::VectorReadDestination => &mut self.destination_port,
            C220UbPort::VectorWrite => unreachable!("read request selected a write port"),
        }
    }

    pub(super) fn request(&self, port: C220UbPort) -> &C220UbRequest {
        &self.port(port).request
    }

    pub(super) fn request_mut(&mut self, port: C220UbPort) -> &mut C220UbRequest {
        &mut self.port_mut(port).request
    }

    pub(super) fn ready_tick(&self) -> Option<u64> {
        self.ready_tick
    }

    pub(super) fn set_ready_tick(&mut self, tick: u64) {
        self.ready_tick = Some(tick);
    }

    pub(super) fn is_sampled(&self) -> bool {
        self.sampled
    }

    pub(super) fn shares_read_with_next(&self) -> bool {
        matches!(
            &self.operation,
            C220VectorReadOperation::Transpose | C220VectorReadOperation::Nchw { .. }
        )
    }

    pub(super) fn writes_compare_mask(&self) -> bool {
        matches!(self.operation, C220VectorReadOperation::CompareMask { .. })
            || matches!(
                &self.operation,
                C220VectorReadOperation::MoveMask { issue }
                    if matches!(issue.instruction.direction, C220MoveMaskDirection::FromMemory)
            )
    }

    pub(super) fn uses_compare_mask(&self) -> bool {
        matches!(
            &self.operation,
            C220VectorReadOperation::Select { issue }
                if !matches!(issue.mode, C220SelectMode::TensorTensor)
        ) || matches!(
            &self.operation,
            C220VectorReadOperation::SelectMaskLoad { issue }
                if matches!(issue.mode, C220SelectMode::TensorTensor)
        ) || matches!(
            &self.operation,
            C220VectorReadOperation::MoveMask { issue }
                if matches!(issue.instruction.direction, C220MoveMaskDirection::ToMemory)
        )
    }

    pub(super) fn writes_selection_mask(&self) -> bool {
        matches!(
            &self.operation,
            C220VectorReadOperation::SelectMaskLoad { .. }
        ) || matches!(
            &self.operation,
            C220VectorReadOperation::Select { issue }
                if issue.loads_selection_mask(
                    self.repeat_index,
                    self.lane_group.unwrap_or_default()
                )
        )
    }

    pub(super) fn uses_selection_mask(&self) -> bool {
        matches!(
            &self.operation,
            C220VectorReadOperation::Select { issue } if issue.mode.uses_tensor_mask()
        )
    }

    pub(super) fn mark_sampled(&mut self) {
        self.sampled = true;
    }

    pub(super) fn capture(
        &mut self,
        decision: &C220UbDecision,
        ub: &UbMemory,
    ) -> Result<(), C220VectorError> {
        let destinations = self.port(decision.port).destinations[decision.block_index].clone();
        if let C220VectorReadOperation::Nchw {
            instruction:
                C220NchwInstruction {
                    element: C220NchwElement::Byte,
                    source_high,
                    ..
                },
            ..
        } = &self.operation
        {
            let half_offset = usize::from(*source_high) * 16;
            let bytes = ub.read_known(decision.block.address + half_offset as u64, 16)?;
            for read_destination in destinations {
                let destination = match read_destination.source_index {
                    0 => &mut self.source_0_bytes,
                    1 => &mut self.source_1_bytes,
                    2 => &mut self.destination_bytes,
                    _ => unreachable!("unknown vector read destination"),
                };
                let offset = read_destination.offset;
                destination[offset + half_offset..offset + half_offset + 16]
                    .copy_from_slice(&bytes);
            }
            return Ok(());
        }
        let first_offset = destinations
            .first()
            .map_or(0, |destination| destination.offset);
        let length = match &self.operation {
            C220VectorReadOperation::Broadcast { instruction, .. }
                if instruction.element_bytes == 2 =>
            {
                16_usize
                    .saturating_sub(first_offset)
                    .min(usize::from(decision.block.bytes))
            }
            C220VectorReadOperation::MoveMask { issue }
                if matches!(
                    issue.instruction.direction,
                    C220MoveMaskDirection::FromMemory
                ) =>
            {
                16_usize
                    .saturating_sub(first_offset)
                    .min(usize::from(decision.block.bytes))
            }
            _ => usize::from(decision.block.bytes),
        };
        let bytes = ub.read_known(decision.block.address, length)?;
        for read_destination in destinations {
            let destination = match read_destination.source_index {
                0 => &mut self.source_0_bytes,
                1 => &mut self.source_1_bytes,
                2 => &mut self.destination_bytes,
                _ => unreachable!("unknown vector read destination"),
            };
            let offset = read_destination.offset;
            destination[offset..offset + bytes.len()].copy_from_slice(&bytes);
        }
        Ok(())
    }

    pub(super) fn grant_tick(&self, admission_tick: u64) -> Option<u64> {
        if !self.port0.request.is_complete()
            || !self.port1.request.is_complete()
            || !self.destination_port.request.is_complete()
        {
            return None;
        }
        Some(
            self.port0
                .request
                .completion_tick()
                .into_iter()
                .chain(self.port1.request.completion_tick())
                .chain(self.destination_port.request.completion_tick())
                .max()
                .unwrap_or(admission_tick),
        )
    }

    pub(super) fn sample(
        &self,
        ub: &UbMemory,
        compare_mask: C220CompareMask,
        selection_mask: Option<&C220SelectionMaskBlock>,
    ) -> Result<(C220VectorReadSample, Vec<C220VectorStore>), C220VectorError> {
        let lane_group = self.lane_group.unwrap_or_default();
        let mut reduction_update = None;
        let mut conversion_lanes = None;
        let mut fused_lanes = None;
        let mut sort_lanes = None;
        let (lanes, stores) = match self.operation.clone() {
            C220VectorReadOperation::Broadcast {
                instruction,
                control,
            } => {
                let (values, stores) = evaluate_c220_broadcast_repeat(
                    instruction,
                    control,
                    self.addresses.destination,
                    self.repeat_index,
                    &self.source_0_bytes,
                )?;
                (
                    values
                        .into_iter()
                        .map(|bits| C220VectorLaneOutcome {
                            active: true,
                            bits,
                            fp16_status: None,
                            fp32_status: None,
                        })
                        .collect(),
                    stores,
                )
            }
            C220VectorReadOperation::Transpose => {
                let (values, stores) =
                    evaluate_c220_transpose(self.addresses.destination, &self.source_0_bytes)?;
                (
                    values
                        .into_iter()
                        .map(|bits| C220VectorLaneOutcome {
                            active: true,
                            bits,
                            fp16_status: None,
                            fp32_status: None,
                        })
                        .collect(),
                    stores,
                )
            }
            C220VectorReadOperation::CompareMask { issue } => {
                let values = evaluate_c220_compare_mask_uop(
                    &issue,
                    self.repeat_index,
                    &self.source_0_bytes,
                    &self.source_1_bytes,
                )?;
                (
                    values
                        .into_iter()
                        .map(|(active, value)| C220VectorLaneOutcome {
                            active,
                            bits: u32::from(value),
                            fp16_status: None,
                            fp32_status: None,
                        })
                        .collect(),
                    Vec::new(),
                )
            }
            C220VectorReadOperation::MoveMask { issue } => (Vec::new(), issue.stores(compare_mask)),
            C220VectorReadOperation::SelectMaskLoad { .. } => (Vec::new(), Vec::new()),
            C220VectorReadOperation::Select { issue } => {
                let loaded_selection_mask = issue
                    .loads_selection_mask(self.repeat_index, lane_group)
                    .then(|| C220SelectionMaskBlock {
                        first_repeat: self.repeat_index,
                        bytes: self.source_1_bytes[..C220_VECTOR_TILE_BYTES]
                            .try_into()
                            .expect("selection-mask block"),
                    });
                let (values, stores) = evaluate_c220_select_uop(
                    &issue,
                    self.repeat_index,
                    lane_group,
                    compare_mask,
                    loaded_selection_mask.as_ref().or(selection_mask),
                    &self.source_0_bytes,
                    &self.source_1_bytes,
                )?;
                (
                    values
                        .into_iter()
                        .map(|bits| C220VectorLaneOutcome {
                            active: true,
                            bits,
                            fp16_status: None,
                            fp32_status: None,
                        })
                        .collect(),
                    stores,
                )
            }
            C220VectorReadOperation::Nchw { instruction, rows } => {
                let (values, stores) = evaluate_c220_nchw_repeat(
                    instruction,
                    self.repeat_index / 2,
                    *rows,
                    &self.source_0_bytes,
                )?;
                (
                    values
                        .into_iter()
                        .map(|bits| C220VectorLaneOutcome {
                            active: true,
                            bits,
                            fp16_status: None,
                            fp32_status: None,
                        })
                        .collect(),
                    stores,
                )
            }
            C220VectorReadOperation::PackedCompare {
                instruction,
                scalar_bits,
            } => {
                let (values, stores) = evaluate_c220_packed_compare_uop(
                    C220PackedCompareValueInputs {
                        instruction,
                        control: self.control,
                        addresses: self.addresses,
                        scalar_bits,
                        uop_index: self.repeat_index,
                    },
                    &self.source_0_bytes,
                    &self.source_1_bytes,
                )?;
                (
                    values
                        .into_iter()
                        .map(|bits| C220VectorLaneOutcome {
                            active: true,
                            bits,
                            fp16_status: None,
                            fp32_status: None,
                        })
                        .collect(),
                    stores,
                )
            }
            C220VectorReadOperation::Reduction { issue } => {
                let outcome = evaluate_c220_reduction_repeat(
                    &issue,
                    self.repeat_index,
                    &self.source_0_bytes,
                )?;
                reduction_update = outcome.state_update;
                (
                    outcome
                        .lanes
                        .into_iter()
                        .map(|lane| C220VectorLaneOutcome {
                            active: lane.active,
                            bits: lane.bits,
                            fp16_status: lane.fp16_status,
                            fp32_status: lane.fp32_status,
                        })
                        .collect(),
                    outcome.stores,
                )
            }
            C220VectorReadOperation::Sort { issue } => {
                let (values, stores) = evaluate_c220_sort_repeat(
                    &issue,
                    self.repeat_index,
                    &self.source_0_bytes,
                    &self.source_1_bytes,
                )?;
                sort_lanes = Some(values.clone());
                (
                    values
                        .into_iter()
                        .map(|lane| C220VectorLaneOutcome {
                            active: true,
                            bits: lane.value_bits,
                            fp16_status: None,
                            fp32_status: None,
                        })
                        .collect(),
                    stores,
                )
            }
            C220VectorReadOperation::Ternary { issue } => {
                let (values, stores) = evaluate_c220_ternary_repeat(
                    &issue,
                    self.repeat_index,
                    lane_group,
                    &self.source_0_bytes,
                    &self.source_1_bytes,
                    &self.destination_bytes,
                    ub,
                )?;
                (
                    values
                        .into_iter()
                        .map(|lane| C220VectorLaneOutcome {
                            active: lane.active,
                            bits: lane.bits,
                            fp16_status: lane.fp16_status,
                            fp32_status: lane.fp32_status,
                        })
                        .collect(),
                    stores,
                )
            }
            C220VectorReadOperation::Axpy { issue } => {
                let (values, stores) = evaluate_c220_axpy_repeat(
                    &issue,
                    self.repeat_index,
                    lane_group,
                    &self.source_0_bytes,
                    &self.destination_bytes,
                    ub,
                )?;
                (
                    values
                        .into_iter()
                        .map(|lane| C220VectorLaneOutcome {
                            active: lane.active,
                            bits: lane.bits,
                            fp16_status: lane.fp16_status,
                            fp32_status: lane.fp32_status,
                        })
                        .collect(),
                    stores,
                )
            }
            C220VectorReadOperation::SpecialUnary { issue } => {
                let (values, stores) = evaluate_c220_special_unary_repeat(
                    &issue,
                    self.repeat_index,
                    lane_group,
                    &self.source_0_bytes,
                    ub,
                )?;
                (
                    values
                        .into_iter()
                        .map(|lane| C220VectorLaneOutcome {
                            active: lane.active,
                            bits: lane.bits,
                            fp16_status: lane.fp16_status,
                            fp32_status: lane.fp32_status,
                        })
                        .collect(),
                    stores,
                )
            }
            C220VectorReadOperation::Conversion { issue } => {
                let (values, stores) = evaluate_c220_conversion_repeat(
                    &issue,
                    self.repeat_index,
                    lane_group,
                    &self.source_0_bytes,
                    &self.source_1_bytes,
                    ub,
                )?;
                conversion_lanes = Some(values.clone());
                (
                    values
                        .into_iter()
                        .map(|lane| C220VectorLaneOutcome {
                            active: lane.active,
                            bits: lane.result_bits as u32,
                            fp16_status: None,
                            fp32_status: None,
                        })
                        .collect(),
                    stores,
                )
            }
            C220VectorReadOperation::Fused { issue } => {
                let (values, stores) = evaluate_c220_fused_repeat(
                    &issue,
                    self.repeat_index,
                    lane_group,
                    &self.source_0_bytes,
                    &self.source_1_bytes,
                    &self.destination_bytes,
                    ub,
                )?;
                fused_lanes = Some(values.clone());
                (
                    values
                        .into_iter()
                        .map(|lane| C220VectorLaneOutcome {
                            active: lane.active,
                            bits: u32::from(lane.result_bits),
                            fp16_status: None,
                            fp32_status: None,
                        })
                        .collect(),
                    stores,
                )
            }
            C220VectorReadOperation::GatherIndex => (Vec::new(), Vec::new()),
            C220VectorReadOperation::GatherData { issue, group } => {
                let (values, stores) = evaluate_c220_gather_data_uop(
                    &issue,
                    self.repeat_index,
                    group,
                    &self.source_0_bytes,
                    &self.source_1_bytes,
                )?;
                (
                    values
                        .into_iter()
                        .map(|lane| C220VectorLaneOutcome {
                            active: lane.active,
                            bits: lane.bits,
                            fp16_status: None,
                            fp32_status: None,
                        })
                        .collect(),
                    stores,
                )
            }
            C220VectorReadOperation::Arithmetic { hint, modes, .. } if modes.widens(hint) => {
                let (values, stores) = evaluate_c220_s16_widen_repeat_from_bytes(
                    C220S16WidenInputs {
                        hint,
                        control: self.control,
                        addresses: self.addresses,
                        repeat_index: self.repeat_index,
                        mask: self.mask,
                    },
                    &self.source_0_bytes,
                    &self.source_1_bytes,
                    ub,
                )?;
                (
                    values
                        .into_iter()
                        .map(|lane| C220VectorLaneOutcome {
                            active: lane.active,
                            bits: lane.bits,
                            fp16_status: None,
                            fp32_status: None,
                        })
                        .collect(),
                    stores,
                )
            }
            C220VectorReadOperation::Arithmetic { hint, modes, .. }
                if hint.has_f16_value_path() =>
            {
                let (values, stores) = evaluate_c220_f16_repeat_from_bytes(
                    C220F16ValueInputs {
                        hint,
                        control: self.control,
                        addresses: self.addresses,
                        repeat_index: self.repeat_index,
                        lane_group,
                        mask: self.mask,
                        mode: modes.fp16_mode,
                    },
                    &self.source_0_bytes,
                    &self.source_1_bytes,
                    ub,
                )?;
                (
                    values
                        .into_iter()
                        .map(|lane| C220VectorLaneOutcome {
                            active: lane.active,
                            bits: lane.bits,
                            fp16_status: lane.status,
                            fp32_status: None,
                        })
                        .collect(),
                    stores,
                )
            }
            C220VectorReadOperation::Arithmetic { hint, modes, .. }
                if hint.has_s16_value_path() || hint.has_bitwise_b16_value_path() =>
            {
                let (values, stores) = evaluate_c220_s16_repeat_from_bytes(
                    C220S16ValueInputs {
                        hint,
                        control: self.control,
                        addresses: self.addresses,
                        repeat_index: self.repeat_index,
                        lane_group,
                        mask: self.mask,
                        saturating: modes.integer_saturating,
                    },
                    &self.source_0_bytes,
                    &self.source_1_bytes,
                    ub,
                )?;
                (
                    values
                        .into_iter()
                        .map(|lane| C220VectorLaneOutcome {
                            active: lane.active,
                            bits: lane.bits,
                            fp16_status: None,
                            fp32_status: None,
                        })
                        .collect(),
                    stores,
                )
            }
            C220VectorReadOperation::Arithmetic { hint, modes, .. }
                if hint.has_s32_value_path() =>
            {
                let (values, stores) = evaluate_c220_s32_repeat_from_bytes(
                    C220S32ValueInputs {
                        hint,
                        control: self.control,
                        addresses: self.addresses,
                        repeat_index: self.repeat_index,
                        mask: self.mask,
                        saturating: modes.integer_saturating,
                    },
                    &self.source_0_bytes,
                    &self.source_1_bytes,
                    ub,
                )?;
                (
                    values
                        .into_iter()
                        .map(|lane| C220VectorLaneOutcome {
                            active: lane.active,
                            bits: lane.bits,
                            fp16_status: None,
                            fp32_status: None,
                        })
                        .collect(),
                    stores,
                )
            }
            C220VectorReadOperation::Arithmetic { hint, .. } => {
                let result = evaluate_c220_fp32_repeat_from_bytes(
                    hint,
                    self.control,
                    self.addresses,
                    self.repeat_index,
                    &self.mask,
                    self.source_0_bytes.clone(),
                    self.source_1_bytes.clone(),
                    ub,
                )?;
                (
                    result
                        .lanes
                        .into_iter()
                        .map(|lane| C220VectorLaneOutcome {
                            active: lane.active,
                            bits: lane.bits,
                            fp16_status: None,
                            fp32_status: lane.status,
                        })
                        .collect(),
                    result.stores,
                )
            }
            C220VectorReadOperation::VectorScalar {
                instruction,
                scalar,
            } => {
                let (values, stores) = evaluate_c220_vector_scalar_repeat(
                    C220VectorScalarValueInputs {
                        instruction,
                        scalar,
                        control: self.control,
                        addresses: self.addresses,
                        mask: self.mask,
                        repeat_index: self.repeat_index,
                        lane_group,
                    },
                    &self.source_0_bytes,
                )?;
                (
                    values
                        .into_iter()
                        .enumerate()
                        .map(|(index, value)| C220VectorLaneOutcome {
                            active: index / 64 == usize::from(lane_group)
                                && self.mask[index / 64] & (1_u64 << (index % 64)) != 0,
                            bits: value.bits,
                            fp16_status: value.fp16_status,
                            fp32_status: value.fp32_status,
                        })
                        .collect(),
                    stores,
                )
            }
            C220VectorReadOperation::Shift { instruction, shift } => {
                let (values, stores) = evaluate_c220_shift_repeat(
                    C220ShiftValueInputs {
                        instruction,
                        shift,
                        control: self.control,
                        addresses: self.addresses,
                        mask: self.mask,
                        repeat_index: self.repeat_index,
                        lane_group,
                    },
                    &self.source_0_bytes,
                )?;
                (
                    values
                        .into_iter()
                        .enumerate()
                        .map(|(index, bits)| C220VectorLaneOutcome {
                            active: index / 64 == usize::from(lane_group)
                                && self.mask[index / 64] & (1_u64 << (index % 64)) != 0,
                            bits,
                            fp16_status: None,
                            fp32_status: None,
                        })
                        .collect(),
                    stores,
                )
            }
            C220VectorReadOperation::Copy { instruction } => {
                let (values, stores) = evaluate_c220_copy_repeat(
                    C220CopyValueInputs {
                        instruction,
                        control: self.control,
                        addresses: self.addresses,
                        mask: self.mask,
                        repeat_index: self.repeat_index,
                        lane_group,
                    },
                    &self.source_0_bytes,
                )?;
                (
                    values
                        .into_iter()
                        .enumerate()
                        .map(|(index, bits)| C220VectorLaneOutcome {
                            active: index / 64 == usize::from(lane_group)
                                && self.mask[index / 64] & (1_u64 << (index % 64)) != 0,
                            bits,
                            fp16_status: None,
                            fp32_status: None,
                        })
                        .collect(),
                    stores,
                )
            }
        };
        let compare_update = match &self.operation {
            C220VectorReadOperation::CompareMask { .. } => Some({
                let mut write_mask = [0_u64; 2];
                let mut values = [0_u64; 2];
                let first_lane = usize::from(self.lane_group.unwrap_or_default()) * 64;
                for (index, lane) in lanes.iter().enumerate() {
                    if !lane.active {
                        continue;
                    }
                    let bit = first_lane + index;
                    write_mask[bit / 64] |= 1_u64 << (bit % 64);
                    if lane.bits != 0 {
                        values[bit / 64] |= 1_u64 << (bit % 64);
                    }
                }
                C220CompareMaskUpdate { write_mask, values }
            }),
            C220VectorReadOperation::MoveMask { issue }
                if matches!(
                    issue.instruction.direction,
                    C220MoveMaskDirection::FromMemory
                ) =>
            {
                Some(C220CompareMaskUpdate {
                    write_mask: [u64::MAX; 2],
                    values: [
                        u64::from_le_bytes(
                            self.source_0_bytes[..8].try_into().expect("low mask word"),
                        ),
                        u64::from_le_bytes(
                            self.source_0_bytes[8..16]
                                .try_into()
                                .expect("high mask word"),
                        ),
                    ],
                })
            }
            _ => None,
        };
        let selection_update = match &self.operation {
            C220VectorReadOperation::SelectMaskLoad { .. } => Some(C220SelectionMaskBlock {
                first_repeat: self.repeat_index,
                bytes: self.source_0_bytes[..C220_VECTOR_TILE_BYTES]
                    .try_into()
                    .expect("selection-mask block"),
            }),
            C220VectorReadOperation::Select { issue }
                if issue.loads_selection_mask(
                    self.repeat_index,
                    self.lane_group.unwrap_or_default(),
                ) =>
            {
                Some(C220SelectionMaskBlock {
                    first_repeat: self.repeat_index,
                    bytes: self.source_1_bytes[..C220_VECTOR_TILE_BYTES]
                        .try_into()
                        .expect("selection-mask block"),
                })
            }
            _ => None,
        };
        Ok((
            C220VectorReadSample {
                pc: self.pc,
                word: self.word,
                repeat_index: self.repeat_index,
                lane_group: self.lane_group,
                tick: self.ready_tick.expect("read sample has a ready tick"),
                accesses: self.accesses.clone(),
                read0_grants: self.port0.request.grants().to_vec(),
                read1_grants: self.port1.request.grants().to_vec(),
                source_0_bytes: self.source_0_bytes.clone(),
                source_1_bytes: self.source_1_bytes.clone(),
                lanes,
                conversion_lanes,
                fused_lanes,
                sort_lanes,
                compare_update,
                selection_update,
                reduction_update,
            },
            stores,
        ))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220VectorReadSample {
    pub pc: u64,
    pub word: u32,
    pub repeat_index: usize,
    pub lane_group: Option<u8>,
    pub tick: u64,
    pub accesses: Vec<C220VectorReadAccess>,
    pub read0_grants: Vec<Option<u64>>,
    pub read1_grants: Vec<Option<u64>>,
    pub source_0_bytes: Vec<u8>,
    pub source_1_bytes: Vec<u8>,
    pub lanes: Vec<C220VectorLaneOutcome>,
    pub conversion_lanes: Option<Vec<C220ConversionLaneOutcome>>,
    pub fused_lanes: Option<Vec<C220FusedLaneOutcome>>,
    pub sort_lanes: Option<Vec<C220SortLaneOutcome>>,
    pub(crate) compare_update: Option<C220CompareMaskUpdate>,
    pub(crate) selection_update: Option<C220SelectionMaskBlock>,
    pub(crate) reduction_update: Option<C220ReductionStateUpdate>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220VectorLaneOutcome {
    pub active: bool,
    pub bits: u32,
    pub fp16_status: Option<C220Fp16Status>,
    pub fp32_status: Option<Fp32ValueStatus>,
}
