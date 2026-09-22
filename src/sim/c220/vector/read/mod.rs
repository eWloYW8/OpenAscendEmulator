mod evaluate;
mod plan;

use crate::isa::c220::vector::compare::{C220MoveMaskDirection, C220PackedCompareInstruction};
use crate::isa::c220::vector::scalar::C220VectorScalarInstruction;
use crate::isa::c220::vector::{
    C220BroadcastInstruction, C220CopyInstruction, C220NchwElement, C220NchwInstruction,
    C220ShiftInstruction, C220VecArithmeticHint,
};
use crate::memory::ub::UbMemory;
use crate::numeric::fp32::Fp32ValueStatus;
use crate::sim::c220::memory::{C220UbDecision, C220UbPort, C220UbRequest, C220UbRequestError};
use crate::sim::c220::numeric::fp16::C220Fp16Status;
use crate::sim::c220::vector::ops::axpy::C220AxpyIssue;
use crate::sim::c220::vector::ops::broadcast::{C220BroadcastControl, C220BroadcastIssue};
use crate::sim::c220::vector::ops::compare::{
    C220CompareMaskIssue, C220CompareMaskUpdate, C220MoveMaskIssue, C220PackedCompareIssue,
};
use crate::sim::c220::vector::ops::conversion::{C220ConversionIssue, C220ConversionLaneOutcome};
use crate::sim::c220::vector::ops::copy::C220CopyIssue;
use crate::sim::c220::vector::ops::fused::{C220FusedIssue, C220FusedLaneOutcome};
use crate::sim::c220::vector::ops::gather::C220GatherIssue;
use crate::sim::c220::vector::ops::nchw::{C220NchwIssue, C220NchwRows};
use crate::sim::c220::vector::ops::reduce::{C220ReductionIssue, C220ReductionStateUpdate};
use crate::sim::c220::vector::ops::scalar::{C220VectorScalarIssue, C220VectorScalarOperand};
use crate::sim::c220::vector::ops::select::{
    C220SelectIssue, C220SelectMode, C220SelectionMaskBlock,
};
use crate::sim::c220::vector::ops::shift::C220ShiftIssue;
use crate::sim::c220::vector::ops::sort::{C220SortIssue, C220SortLaneOutcome};
use crate::sim::c220::vector::ops::special::C220SpecialUnaryIssue;
use crate::sim::c220::vector::ops::ternary::C220TernaryIssue;
use crate::sim::c220::vector::ops::transpose::C220TransposeIssue;
use crate::sim::c220::vector::va::{C220LoadVaIssue, C220VaUpdate};
use crate::sim::c220::vector::{
    C220VectorAddresses, C220VectorArithmeticIssue, C220VectorArithmeticModes, C220VectorControl,
    C220VectorError, C220VectorReadAccess,
};
use std::collections::BTreeMap;
use thiserror::Error;
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
    lane_slice: Option<(usize, usize)>,
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
    LoadVa {
        issue: C220LoadVaIssue,
    },
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
    LoadVa(&'a C220LoadVaIssue),
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

    pub(super) fn is_load_va(&self) -> bool {
        matches!(self.operation, C220VectorReadOperation::LoadVa { .. })
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
    pub(crate) va_update: Option<C220VaUpdate>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220VectorLaneOutcome {
    pub active: bool,
    pub bits: u32,
    pub fp16_status: Option<C220Fp16Status>,
    pub fp32_status: Option<Fp32ValueStatus>,
}
