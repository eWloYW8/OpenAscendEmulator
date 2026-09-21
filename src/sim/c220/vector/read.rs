use thiserror::Error;

use crate::isa::c220::vector::{C220ShiftInstruction, C220VecArithmeticHint};
use crate::isa::c220::vector_scalar::C220VectorScalarInstruction;
use crate::memory::ub::UbMemory;
use crate::numeric::fp32::Fp32ValueStatus;
use crate::sim::c220::fp16::C220Fp16Status;
use crate::sim::c220::ub_arbiter::{C220UbDecision, C220UbPort, C220UbRequest, C220UbRequestError};
use crate::sim::c220::vector::f16::{C220F16ValueInputs, evaluate_c220_f16_repeat_from_bytes};
use crate::sim::c220::vector::s16::{
    C220S16ValueInputs, C220S16WidenInputs, evaluate_c220_s16_repeat_from_bytes,
    evaluate_c220_s16_widen_repeat_from_bytes,
};
use crate::sim::c220::vector::s32::{C220S32ValueInputs, evaluate_c220_s32_repeat_from_bytes};
use crate::sim::c220::vector::scalar::{
    C220VectorScalarIssue, C220VectorScalarOperand, C220VectorScalarValueInputs,
    evaluate_c220_vector_scalar_repeat,
};
use crate::sim::c220::vector::shift::{
    C220ShiftIssue, C220ShiftValueInputs, evaluate_c220_shift_repeat,
};
use crate::sim::c220::vector::{
    C220_VECTOR_BLOCK_BYTES, C220_VECTOR_TILE_BYTES, C220VectorAddresses,
    C220VectorArithmeticIssue, C220VectorArithmeticModes, C220VectorControl, C220VectorError,
    C220VectorReadAccess, C220VectorStore, evaluate_c220_fp32_repeat_from_bytes,
};

#[derive(Debug, Error)]
pub enum C220VectorReadError {
    #[error("vector repeat has no iteration mask")]
    MissingRepeatMask,
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
    lane_group: u8,
    accesses: Vec<C220VectorReadAccess>,
    port0: ReadPort,
    port1: ReadPort,
    source_0_bytes: Vec<u8>,
    source_1_bytes: Vec<u8>,
    ready_tick: Option<u64>,
    sampled: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
}

#[derive(Debug, Clone, Copy)]
pub enum C220VectorReadIssue<'a> {
    Arithmetic(&'a C220VectorArithmeticIssue),
    VectorScalar(&'a C220VectorScalarIssue),
    Shift(&'a C220ShiftIssue),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ReadPort {
    request: C220UbRequest,
    destination_offsets: Vec<usize>,
}

impl ReadPort {
    fn new(accesses: &[C220VectorReadAccess]) -> Result<Self, C220UbRequestError> {
        let spans = accesses
            .iter()
            .map(|access| (access.address, C220_VECTOR_BLOCK_BYTES))
            .collect::<Vec<_>>();
        let request = C220UbRequest::from_accesses(&spans)?;
        let mut destination_offsets = Vec::with_capacity(request.blocks().len());
        let mut next_block = 0;
        for access in accesses {
            let mut address = access.address;
            let end = address + C220_VECTOR_BLOCK_BYTES as u64;
            while address < end {
                let block = request.blocks()[next_block];
                debug_assert_eq!(block.address, address);
                destination_offsets.push(
                    usize::from(access.block_index) * C220_VECTOR_BLOCK_BYTES
                        + (address - access.address) as usize,
                );
                address += u64::from(block.bytes);
                next_block += 1;
            }
        }
        Ok(Self {
            request,
            destination_offsets,
        })
    }
}

impl PendingVectorRead {
    pub(super) fn new(
        issue: C220VectorReadIssue<'_>,
        repeat_index: usize,
        lane_group: u8,
    ) -> Result<Self, C220VectorReadError> {
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
                issue.read_accesses_for_repeat(repeat_index, lane_group)?,
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
                issue.read_accesses_for_repeat(repeat_index, lane_group)?,
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
                issue.read_accesses_for_repeat(repeat_index, lane_group)?,
                C220VectorReadOperation::Shift {
                    instruction: issue.instruction,
                    shift: issue.shift,
                },
            ),
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
            port0: ReadPort::new(&port0_accesses)?,
            port1: ReadPort::new(&port1_accesses)?,
            source_0_bytes: vec![0; C220_VECTOR_TILE_BYTES],
            source_1_bytes: vec![0; C220_VECTOR_TILE_BYTES],
            ready_tick: None,
            sampled: false,
        })
    }

    fn port(&self, port: C220UbPort) -> &ReadPort {
        match port {
            C220UbPort::VectorRead0 => &self.port0,
            C220UbPort::VectorRead1 => &self.port1,
            C220UbPort::VectorWrite => unreachable!("read request selected a write port"),
        }
    }

    fn port_mut(&mut self, port: C220UbPort) -> &mut ReadPort {
        match port {
            C220UbPort::VectorRead0 => &mut self.port0,
            C220UbPort::VectorRead1 => &mut self.port1,
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

    pub(super) fn mark_sampled(&mut self) {
        self.sampled = true;
    }

    pub(super) fn capture(
        &mut self,
        decision: &C220UbDecision,
        ub: &UbMemory,
    ) -> Result<(), C220VectorError> {
        let offset = self.port(decision.port).destination_offsets[decision.block_index];
        let destination = match decision.port {
            C220UbPort::VectorRead0 => &mut self.source_0_bytes,
            C220UbPort::VectorRead1 => &mut self.source_1_bytes,
            C220UbPort::VectorWrite => unreachable!("read request selected a write port"),
        };
        let bytes = ub.read_known(decision.block.address, usize::from(decision.block.bytes))?;
        destination[offset..offset + bytes.len()].copy_from_slice(&bytes);
        Ok(())
    }

    pub(super) fn grant_tick(&self, admission_tick: u64) -> Option<u64> {
        if !self.port0.request.is_complete() || !self.port1.request.is_complete() {
            return None;
        }
        Some(
            self.port0
                .request
                .completion_tick()
                .into_iter()
                .chain(self.port1.request.completion_tick())
                .max()
                .unwrap_or(admission_tick),
        )
    }

    pub(super) fn sample(
        &self,
        ub: &UbMemory,
    ) -> Result<(C220VectorReadSample, Vec<C220VectorStore>), C220VectorError> {
        let (lanes, stores) = match self.operation {
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
                        lane_group: self.lane_group,
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
                        lane_group: self.lane_group,
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
                        lane_group: self.lane_group,
                    },
                    &self.source_0_bytes,
                )?;
                (
                    values
                        .into_iter()
                        .enumerate()
                        .map(|(index, value)| C220VectorLaneOutcome {
                            active: index / 64 == usize::from(self.lane_group)
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
                        lane_group: self.lane_group,
                    },
                    &self.source_0_bytes,
                )?;
                (
                    values
                        .into_iter()
                        .enumerate()
                        .map(|(index, bits)| C220VectorLaneOutcome {
                            active: index / 64 == usize::from(self.lane_group)
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
    pub lane_group: u8,
    pub tick: u64,
    pub accesses: Vec<C220VectorReadAccess>,
    pub read0_grants: Vec<Option<u64>>,
    pub read1_grants: Vec<Option<u64>>,
    pub source_0_bytes: Vec<u8>,
    pub source_1_bytes: Vec<u8>,
    pub lanes: Vec<C220VectorLaneOutcome>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220VectorLaneOutcome {
    pub active: bool,
    pub bits: u32,
    pub fp16_status: Option<C220Fp16Status>,
    pub fp32_status: Option<Fp32ValueStatus>,
}
