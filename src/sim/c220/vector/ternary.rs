use crate::architecture::c220::C220UbBank;
use crate::isa::c220::ternary::{C220TernaryInstruction, C220TernaryOperation, C220TernaryWidth};
use crate::memory::ub::UbMemory;
use crate::numeric::fp32::{Fp32ValueStatus, Fp32VectorOperation, evaluate_fp32_value};
use crate::sim::c220::fp16::{C220Fp16Mode, C220Fp16Status, evaluate_c220_fp16_relu};

use super::fma::{
    evaluate_c220_f16_mla, evaluate_c220_fp32_mla, evaluate_c220_mixed_mla, merge_fp16_status,
    merge_fp32_status,
};
use super::{
    C220_VECTOR_BLOCK_BYTES, C220_VECTOR_TILE_BYTES, C220VectorAddresses, C220VectorControl,
    C220VectorError, C220VectorReadAccess, C220VectorStore, check_repeat_limit,
    plan_c220_destination_read_accesses, plan_c220_vector_read_accesses,
    vector_destination_address_for_width,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220TernaryIssue {
    pub pc: u64,
    pub word: u32,
    pub instruction: C220TernaryInstruction,
    pub control: C220VectorControl,
    pub addresses: C220VectorAddresses,
    pub iteration_masks: Vec<[u64; 4]>,
    pub fp16_mode: C220Fp16Mode,
    pub(crate) write_targets: Vec<C220VectorStore>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct C220TernaryLaneOutcome {
    pub active: bool,
    pub bits: u32,
    pub fp16_status: Option<C220Fp16Status>,
    pub fp32_status: Option<Fp32ValueStatus>,
}

impl C220TernaryIssue {
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
        plan_ternary_reads(
            self.instruction,
            self.control,
            self.addresses,
            repeat_index,
            mask,
            lane_group,
        )
    }
}

pub fn plan_c220_ternary_issue(
    pc: u64,
    word: u32,
    control: C220VectorControl,
    addresses: C220VectorAddresses,
    iteration_masks: &[[u64; 4]],
    fp16_mode: C220Fp16Mode,
    ub: &UbMemory,
) -> Result<C220TernaryIssue, C220VectorError> {
    check_repeat_limit(iteration_masks.len())?;
    let instruction = C220TernaryInstruction::decode(word)
        .ok_or(C220VectorError::UnsupportedWord { pc, word })?;
    let lane_count = instruction.width.lane_count();
    let mut effective_masks = iteration_masks.to_vec();
    for mask in &mut effective_masks {
        let used_words = lane_count.div_ceil(64);
        mask[used_words..].fill(0);
    }
    let mut write_targets = Vec::new();
    write_targets
        .try_reserve_exact(lane_count.saturating_mul(effective_masks.len()))
        .map_err(|_| C220VectorError::HostAllocationFailed {
            lanes: lane_count.saturating_mul(effective_masks.len()),
        })?;
    for (repeat_index, mask) in effective_masks.iter().enumerate() {
        for lane_group in 0..instruction.width.lane_groups() {
            for access in plan_ternary_reads(
                instruction,
                control,
                addresses,
                repeat_index,
                mask,
                lane_group,
            )? {
                ub.check_range(access.address, C220_VECTOR_BLOCK_BYTES)?;
            }
        }
        let element_bytes = instruction.width.destination_element_bytes();
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
    Ok(C220TernaryIssue {
        pc,
        word,
        instruction,
        control,
        addresses,
        iteration_masks: effective_masks,
        fp16_mode,
        write_targets,
    })
}

fn plan_ternary_reads(
    instruction: C220TernaryInstruction,
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
        2,
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

#[derive(Debug, Clone, Copy)]
pub(crate) struct C220TernaryValueInputs<'a> {
    pub issue: &'a C220TernaryIssue,
    pub repeat_index: usize,
    pub lane_group: u8,
    pub lane_slice: Option<(usize, usize)>,
}

pub(crate) fn evaluate_c220_ternary_repeat(
    inputs: C220TernaryValueInputs<'_>,
    source_0_bytes: &[u8],
    source_1_bytes: &[u8],
    destination_bytes: &[u8],
    ub: &UbMemory,
) -> Result<(Vec<C220TernaryLaneOutcome>, Vec<C220VectorStore>), C220VectorError> {
    for source in [source_0_bytes, source_1_bytes, destination_bytes] {
        if source.len() != C220_VECTOR_TILE_BYTES {
            return Err(C220VectorError::InvalidSourceTile {
                actual: source.len(),
                expected: C220_VECTOR_TILE_BYTES,
            });
        }
    }
    if inputs.lane_group >= inputs.issue.instruction.width.lane_groups() {
        return Err(C220VectorError::InvalidLaneGroup(inputs.lane_group));
    }
    let mask = inputs
        .issue
        .iteration_masks
        .get(inputs.repeat_index)
        .ok_or(C220VectorError::MissingMaskState)?;
    let lane_count = inputs.issue.instruction.width.lane_count();
    let result_bytes = inputs.issue.instruction.width.destination_element_bytes();
    let mut lanes = Vec::with_capacity(lane_count);
    let mut stores = Vec::with_capacity(64);
    for lane_index in 0..lane_count {
        let in_uop = inputs.lane_slice.map_or_else(
            || lane_index / 64 == usize::from(inputs.lane_group),
            |(first_lane, lane_count)| {
                lane_index >= first_lane && lane_index < first_lane + lane_count
            },
        );
        let active = in_uop && mask[lane_index / 64] & (1_u64 << (lane_index % 64)) != 0;
        if !active {
            lanes.push(C220TernaryLaneOutcome {
                active: false,
                bits: 0,
                fp16_status: None,
                fp32_status: None,
            });
            continue;
        }
        let (bits, fp16_status, fp32_status) = match inputs.issue.instruction.width {
            C220TernaryWidth::F16 => {
                let source_offset = lane_index * 2;
                let source_0 = read_u16(source_0_bytes, source_offset);
                let source_1 = read_u16(source_1_bytes, source_offset);
                let destination = read_u16(destination_bytes, source_offset);
                let (bits, status) = evaluate_f16(
                    inputs.issue.instruction.operation,
                    source_0,
                    source_1,
                    destination,
                    inputs.issue.fp16_mode,
                );
                (u32::from(bits), Some(status), None)
            }
            C220TernaryWidth::F32 => {
                let source_offset = lane_index * 4;
                let source_0 = read_u32(source_0_bytes, source_offset);
                let source_1 = read_u32(source_1_bytes, source_offset);
                let destination = read_u32(destination_bytes, source_offset);
                let (bits, status) = evaluate_f32(
                    inputs.issue.instruction.operation,
                    source_0,
                    source_1,
                    destination,
                );
                (bits, None, Some(status))
            }
            C220TernaryWidth::F16ToF32 => {
                let source_offset = lane_index * 2;
                let destination_offset = lane_index * 4;
                let source_0 = read_u16(source_0_bytes, source_offset);
                let source_1 = read_u16(source_1_bytes, source_offset);
                let destination = read_u32(destination_bytes, destination_offset);
                let (bits, status) = evaluate_mixed_mla(source_0, source_1, destination);
                (bits, None, Some(status))
            }
        };
        let address = vector_destination_address_for_width(
            inputs.issue.control,
            inputs.issue.addresses,
            inputs.repeat_index,
            lane_index,
            result_bytes,
        )?;
        ub.check_range(address, usize::from(result_bytes))?;
        let data = bits.to_le_bytes();
        stores.push(C220VectorStore {
            repeat_index: inputs.repeat_index,
            lane_index,
            address,
            bank: C220UbBank::from_address(address),
            width_bytes: result_bytes,
            data: super::store_data(data),
        });
        lanes.push(C220TernaryLaneOutcome {
            active,
            bits,
            fp16_status,
            fp32_status,
        });
    }
    Ok((lanes, stores))
}

fn evaluate_f16(
    operation: C220TernaryOperation,
    source_0: u16,
    source_1: u16,
    destination: u16,
    mode: C220Fp16Mode,
) -> (u16, C220Fp16Status) {
    let (first_factor, second_factor, addend) = match operation {
        C220TernaryOperation::MultiplyAccumulate => (source_0, source_1, destination),
        C220TernaryOperation::MultiplyAdd | C220TernaryOperation::MultiplyAddRelu => {
            (source_0, destination, source_1)
        }
    };
    let (sum, mut status) = evaluate_c220_f16_mla(first_factor, second_factor, addend, mode);
    let bits = if operation == C220TernaryOperation::MultiplyAddRelu {
        let relu = evaluate_c220_fp16_relu(sum, mode);
        status = merge_fp16_status(status, relu.status);
        relu.bits
    } else {
        sum
    };
    (bits, status)
}

fn evaluate_f32(
    operation: C220TernaryOperation,
    source_0: u32,
    source_1: u32,
    destination: u32,
) -> (u32, Fp32ValueStatus) {
    let (first_factor, second_factor, addend) = match operation {
        C220TernaryOperation::MultiplyAccumulate => (source_0, source_1, destination),
        C220TernaryOperation::MultiplyAdd | C220TernaryOperation::MultiplyAddRelu => {
            (source_0, destination, source_1)
        }
    };
    let (sum, mut status) = evaluate_c220_fp32_mla(first_factor, second_factor, addend);
    let bits = if operation == C220TernaryOperation::MultiplyAddRelu {
        let relu = evaluate_fp32_value(Fp32VectorOperation::Rectify, sum, 0);
        status = merge_fp32_status(status, relu.status);
        relu.bits
    } else {
        sum
    };
    (bits, status)
}

fn evaluate_mixed_mla(source_0: u16, source_1: u16, destination: u32) -> (u32, Fp32ValueStatus) {
    evaluate_c220_mixed_mla(source_0, source_1, destination)
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

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;

    use crate::architecture::Architecture;
    use crate::memory::mapped::MappedMemory;
    use crate::memory::region::MemoryRegion;
    use crate::memory::sparse::{MemoryByteState, SparseMemory};
    use crate::sim::c220::core::functional::C220FunctionalCore;
    use crate::sim::c220::core::{
        C220Core, C220CoreInstruction, C220CoreStep, C220CoreTimingRules,
    };
    use crate::sim::c220::timing::mte2::C220Mte2TimingRules;
    use crate::sim::c220::timing::mte3::C220Mte3TimingRules;
    use crate::sim::c220::vector::pipeline::C220VectorTimingRules;
    use crate::sim::common::scalar::ScalarMachine;
    use crate::sim::common::scalar::stepper::ScalarStepper;

    use super::*;

    #[test]
    fn ternary_words_read_old_destination_and_use_their_staged_arithmetic() {
        let memory = SparseMemory::new(vec![MemoryRegion::unknown(256)], 512, 512);
        let memory = MappedMemory::bind(memory, &[0x2000]).unwrap();
        let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
        machine.set_xreg(3, 0x200).unwrap();
        machine.set_xreg(4, 0).unwrap();
        machine.set_xreg(5, 0x100).unwrap();
        machine.set_xreg(6, 0x0100_0808_0801_0101).unwrap();
        machine.set_spr_value(3, 0).unwrap();
        machine.set_spr_value(100, 1).unwrap();
        machine.set_spr_value(101, 1).unwrap();
        let mut ub = UbMemory::new(1024, 256);
        for address in [0, 0x80, 0x100, 0x180, 0x200, 0x280] {
            ub.write_states(address, &[MemoryByteState::Known(0); 32])
                .unwrap();
        }
        for offset in [0, 0x80] {
            for (base, value) in [(0, 0x4000_u16), (0x100, 0x4200), (0x200, 0x4400)] {
                ub.write_states(
                    base + offset,
                    &value.to_le_bytes().map(MemoryByteState::Known),
                )
                .unwrap();
            }
        }
        let execution = C220FunctionalCore::new(ScalarStepper::new(machine, 0x4000), ub);
        let rate = NonZeroU64::new(32).unwrap();
        let mut core = C220Core::new(
            execution,
            memory,
            C220CoreTimingRules {
                mte2: C220Mte2TimingRules {
                    issue_interval: NonZeroU64::new(1).unwrap(),
                    startup_ticks: 0,
                    bytes_per_tick: rate,
                    retire_ticks: 0,
                },
                mte3: C220Mte3TimingRules {
                    issue_interval: NonZeroU64::new(1).unwrap(),
                    startup_ticks: 0,
                    bytes_per_tick: rate,
                    retire_ticks: 0,
                },
                vector: C220VectorTimingRules {
                    dispatch_ticks: 0,
                    uop_issue_interval: NonZeroU64::new(1).unwrap(),
                    ub_response_ticks: 1,
                },
            },
        )
        .unwrap();
        let word = 0x8940_0001 | (3 << 17) | (4 << 12) | (5 << 7) | (6 << 2);
        let C220CoreStep::Executed {
            instruction: C220CoreInstruction::VectorTernary(issue),
            ..
        } = core.step_word_at(0, word).unwrap()
        else {
            panic!("ternary instruction should issue");
        };
        let uops = C220CoreInstruction::VectorTernary(issue)
            .vector_uops()
            .unwrap();
        assert_eq!(uops.len(), 1);
        assert!(matches!(
            uops[0].kind,
            crate::sim::c220::vector::timing::C220VectorUopKind::LaneSlice {
                first_lane: 0,
                lane_count: 128
            }
        ));
        assert!(uops.iter().all(|uop| uop.stages.execute_ticks == 11));
        core.advance_to(100).unwrap();
        for address in [0x200, 0x280] {
            assert_eq!(
                core.functional().ub().read_known(address, 2).unwrap(),
                0x4900_u16.to_le_bytes()
            );
        }

        assert_eq!(
            evaluate_f16(
                C220TernaryOperation::MultiplyAdd,
                0x4000,
                0x4200,
                0x4400,
                C220Fp16Mode::Saturating,
            )
            .0,
            0x4980
        );
        assert_eq!(
            evaluate_f16(
                C220TernaryOperation::MultiplyAddRelu,
                0xc000,
                0x4200,
                0x4400,
                C220Fp16Mode::Saturating,
            )
            .0,
            0
        );
    }
}
