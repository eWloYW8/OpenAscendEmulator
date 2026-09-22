use crate::architecture::c220::C220UbBank;
use crate::isa::c220::axpy::{C220AxpyInstruction, C220AxpyWidth};
use crate::memory::ub::UbMemory;
use crate::numeric::fp32::Fp32ValueStatus;
use crate::sim::c220::fp16::{C220Fp16Mode, C220Fp16Status};

use super::fma::{evaluate_c220_f16_mla, evaluate_c220_fp32_mla, evaluate_c220_mixed_mla};
use super::{
    C220_VECTOR_TILE_BYTES, C220VectorAddresses, C220VectorControl, C220VectorError,
    C220VectorReadAccess, C220VectorStore, check_repeat_limit, plan_c220_destination_read_accesses,
    plan_c220_unary_write_targets, plan_c220_vector_read_accesses,
    vector_destination_address_for_width,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220AxpyIssue {
    pub pc: u64,
    pub word: u32,
    pub instruction: C220AxpyInstruction,
    pub scalar_bits: u32,
    pub control: C220VectorControl,
    pub addresses: C220VectorAddresses,
    pub iteration_masks: Vec<[u64; 4]>,
    pub fp16_mode: C220Fp16Mode,
    pub(crate) write_targets: Vec<C220VectorStore>,
}

pub(crate) struct C220AxpyIssueInputs<'a> {
    pub pc: u64,
    pub word: u32,
    pub scalar_bits: u32,
    pub control: C220VectorControl,
    pub addresses: C220VectorAddresses,
    pub iteration_masks: &'a [[u64; 4]],
    pub fp16_mode: C220Fp16Mode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct C220AxpyLaneOutcome {
    pub active: bool,
    pub bits: u32,
    pub fp16_status: Option<C220Fp16Status>,
    pub fp32_status: Option<Fp32ValueStatus>,
}

impl C220AxpyIssue {
    pub fn read_accesses_for_repeat(
        &self,
        repeat_index: usize,
        lane_group: u8,
    ) -> Result<Vec<C220VectorReadAccess>, C220VectorError> {
        if lane_group >= self.instruction.width.lane_groups() {
            return Err(C220VectorError::InvalidLaneGroup(lane_group));
        }
        let mask = self
            .iteration_masks
            .get(repeat_index)
            .ok_or(C220VectorError::MissingMaskState)?;
        plan_axpy_reads(
            self.instruction,
            self.control,
            self.addresses,
            repeat_index,
            mask,
            lane_group,
        )
    }
}

pub(crate) fn plan_c220_axpy_issue(
    inputs: C220AxpyIssueInputs<'_>,
    ub: &UbMemory,
) -> Result<C220AxpyIssue, C220VectorError> {
    check_repeat_limit(inputs.iteration_masks.len())?;
    let instruction =
        C220AxpyInstruction::decode(inputs.word).ok_or(C220VectorError::UnsupportedWord {
            pc: inputs.pc,
            word: inputs.word,
        })?;
    let lane_count = instruction.width.lane_count();
    let mut effective_masks = inputs.iteration_masks.to_vec();
    for mask in &mut effective_masks {
        mask[lane_count.div_ceil(64)..].fill(0);
    }
    let write_targets = plan_c220_unary_write_targets(
        inputs.control,
        inputs.addresses,
        &effective_masks,
        instruction.width.destination_element_bytes(),
        ub,
    )?;
    for (repeat_index, mask) in effective_masks.iter().enumerate() {
        for lane_group in 0..instruction.width.lane_groups() {
            for access in plan_axpy_reads(
                instruction,
                inputs.control,
                inputs.addresses,
                repeat_index,
                mask,
                lane_group,
            )? {
                ub.check_range(access.address, usize::from(access.bytes))?;
            }
        }
    }
    Ok(C220AxpyIssue {
        pc: inputs.pc,
        word: inputs.word,
        instruction,
        scalar_bits: inputs.scalar_bits,
        control: inputs.control,
        addresses: inputs.addresses,
        iteration_masks: effective_masks,
        fp16_mode: inputs.fp16_mode,
        write_targets,
    })
}

fn plan_axpy_reads(
    instruction: C220AxpyInstruction,
    control: C220VectorControl,
    addresses: C220VectorAddresses,
    repeat_index: usize,
    mask: &[u64; 4],
    lane_group: u8,
) -> Result<Vec<C220VectorReadAccess>, C220VectorError> {
    let group = Some(lane_group);
    let mut accesses = plan_c220_vector_read_accesses(
        control,
        addresses,
        repeat_index,
        mask,
        1,
        instruction.width.source_element_bytes(),
        group,
    )?;
    accesses.extend(plan_c220_destination_read_accesses(
        control,
        addresses,
        repeat_index,
        mask,
        instruction.width.destination_element_bytes(),
        group,
    )?);
    Ok(accesses)
}

pub(crate) fn evaluate_c220_axpy_repeat(
    issue: &C220AxpyIssue,
    repeat_index: usize,
    lane_group: u8,
    lane_slice: Option<(usize, usize)>,
    source_bytes: &[u8],
    destination_bytes: &[u8],
    ub: &UbMemory,
) -> Result<(Vec<C220AxpyLaneOutcome>, Vec<C220VectorStore>), C220VectorError> {
    for source in [source_bytes, destination_bytes] {
        if source.len() != C220_VECTOR_TILE_BYTES {
            return Err(C220VectorError::InvalidSourceTile {
                actual: source.len(),
                expected: C220_VECTOR_TILE_BYTES,
            });
        }
    }
    if lane_group >= issue.instruction.width.lane_groups() {
        return Err(C220VectorError::InvalidLaneGroup(lane_group));
    }
    let mask = issue
        .iteration_masks
        .get(repeat_index)
        .ok_or(C220VectorError::MissingMaskState)?;
    let lane_count = issue.instruction.width.lane_count();
    let destination_bytes_per_lane = issue.instruction.width.destination_element_bytes();
    let mut lanes = Vec::with_capacity(lane_count);
    let mut stores = Vec::with_capacity(64);
    for lane_index in 0..lane_count {
        let in_uop = lane_slice.map_or_else(
            || lane_index / 64 == usize::from(lane_group),
            |(first_lane, lane_count)| {
                lane_index >= first_lane && lane_index < first_lane + lane_count
            },
        );
        let active = in_uop && mask[lane_index / 64] & (1_u64 << (lane_index % 64)) != 0;
        if !active {
            lanes.push(C220AxpyLaneOutcome {
                active: false,
                bits: 0,
                fp16_status: None,
                fp32_status: None,
            });
            continue;
        }
        let (bits, fp16_status, fp32_status) = match issue.instruction.width {
            C220AxpyWidth::F16 => {
                let offset = lane_index * 2;
                let source = read_u16(source_bytes, offset);
                let destination = read_u16(destination_bytes, offset);
                let (bits, status) = evaluate_c220_f16_mla(
                    source,
                    issue.scalar_bits as u16,
                    destination,
                    issue.fp16_mode,
                );
                (u32::from(bits), Some(status), None)
            }
            C220AxpyWidth::F16ToF32 => {
                let source = read_u16(source_bytes, lane_index * 2);
                let destination = read_u32(destination_bytes, lane_index * 4);
                let (bits, status) =
                    evaluate_c220_mixed_mla(source, issue.scalar_bits as u16, destination);
                (bits, None, Some(status))
            }
            C220AxpyWidth::F32 => {
                let offset = lane_index * 4;
                let source = read_u32(source_bytes, offset);
                let destination = read_u32(destination_bytes, offset);
                let (bits, status) = evaluate_c220_fp32_mla(source, issue.scalar_bits, destination);
                (bits, None, Some(status))
            }
        };
        let address = vector_destination_address_for_width(
            issue.control,
            issue.addresses,
            repeat_index,
            lane_index,
            destination_bytes_per_lane,
        )?;
        ub.check_range(address, usize::from(destination_bytes_per_lane))?;
        stores.push(C220VectorStore {
            repeat_index,
            lane_index,
            address,
            bank: C220UbBank::from_address(address),
            width_bytes: destination_bytes_per_lane,
            data: super::store_data(bits.to_le_bytes()),
        });
        lanes.push(C220AxpyLaneOutcome {
            active,
            bits,
            fp16_status,
            fp32_status,
        });
    }
    Ok((lanes, stores))
}

fn read_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(bytes[offset..offset + 2].try_into().expect("two-byte lane"))
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .expect("four-byte lane"),
    )
}
