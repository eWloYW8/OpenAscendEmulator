use crate::isa::c220::vector::compare::{
    C220CompareCondition, C220CompareMaskInstruction, C220CompareWidth, C220MoveMaskDirection,
    C220MoveMaskInstruction, C220PackedCompareInstruction, C220PackedCompareOperand,
};
use crate::memory::ub::UbMemory;
use crate::sim::c220::memory::C220UbBank;
use crate::sim::c220::vector::{
    C220_VECTOR_TILE_BYTES, C220VectorAddresses, C220VectorControl, C220VectorError,
    C220VectorMaskState, C220VectorReadAccess, C220VectorStore, decode_c220_repeat_masks,
    plan_c220_vector_read_accesses,
};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct C220CompareMask {
    bits: [u64; 2],
}

impl C220CompareMask {
    pub const fn from_bits(bits: [u64; 2]) -> Self {
        Self { bits }
    }

    pub const fn bits(self) -> [u64; 2] {
        self.bits
    }

    pub const fn test(self, lane: usize) -> bool {
        self.bits[lane / 64] & (1_u64 << (lane % 64)) != 0
    }

    pub(crate) fn apply(&mut self, update: C220CompareMaskUpdate) {
        for index in 0..2 {
            self.bits[index] =
                (self.bits[index] & !update.write_mask[index]) | update.values[index];
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct C220CompareMaskUpdate {
    pub write_mask: [u64; 2],
    pub values: [u64; 2],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220MoveMaskIssue {
    pub pc: u64,
    pub word: u32,
    pub instruction: C220MoveMaskInstruction,
    pub address: u64,
    pub(crate) write_targets: Vec<C220VectorStore>,
}

impl C220MoveMaskIssue {
    pub fn read_accesses(&self) -> Vec<C220VectorReadAccess> {
        if matches!(
            self.instruction.direction,
            C220MoveMaskDirection::FromMemory
        ) {
            vec![C220VectorReadAccess {
                source_index: 0,
                block_index: 0,
                buffer_offset: 0,
                bytes: 32,
                address: self.address,
                active_lane_mask: u16::MAX,
            }]
        } else {
            Vec::new()
        }
    }

    pub fn stores(&self, compare_mask: C220CompareMask) -> Vec<C220VectorStore> {
        if !matches!(self.instruction.direction, C220MoveMaskDirection::ToMemory) {
            return Vec::new();
        }
        compare_mask
            .bits()
            .into_iter()
            .flat_map(u64::to_le_bytes)
            .collect::<Vec<_>>()
            .chunks_exact(4)
            .enumerate()
            .map(|(index, bytes)| {
                let address = self.address + (index * 4) as u64;
                C220VectorStore {
                    repeat_index: 0,
                    lane_index: index,
                    address,
                    bank: C220UbBank::from_address(address),
                    width_bytes: 4,
                    data: crate::sim::c220::vector::access::store_data::<4>(
                        bytes.try_into().expect("four-byte mask chunk"),
                    ),
                }
            })
            .collect()
    }
}

pub fn plan_c220_move_mask_issue(
    pc: u64,
    word: u32,
    registers: &[u64; 32],
    ub: &UbMemory,
) -> Result<C220MoveMaskIssue, C220VectorError> {
    let instruction = C220MoveMaskInstruction::decode(word)
        .ok_or(C220VectorError::UnsupportedWord { pc, word })?;
    let address = registers[usize::from(instruction.address_register)];
    ub.check_range(address, 16)?;
    let mut issue = C220MoveMaskIssue {
        pc,
        word,
        instruction,
        address,
        write_targets: Vec::new(),
    };
    issue.write_targets = issue.stores(C220CompareMask::default());
    Ok(issue)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220CompareMaskIssue {
    pub pc: u64,
    pub word: u32,
    pub instruction: C220CompareMaskInstruction,
    pub control: C220VectorControl,
    pub addresses: C220VectorAddresses,
    pub iteration_masks: Vec<[u64; 4]>,
    pub initial_compare_mask: C220CompareMask,
}

impl C220CompareMaskIssue {
    pub fn uop_count(&self) -> usize {
        self.iteration_masks.len()
    }

    pub fn split_uop(&self, uop_index: usize) -> Result<(usize, u8), C220VectorError> {
        if uop_index >= self.iteration_masks.len() {
            return Err(C220VectorError::InvalidRepeatIndex(uop_index));
        }
        Ok((uop_index, 0))
    }

    pub fn read_accesses_for_uop(
        &self,
        uop_index: usize,
    ) -> Result<Vec<C220VectorReadAccess>, C220VectorError> {
        let (repeat_index, _) = self.split_uop(uop_index)?;
        let mut accesses = Vec::new();
        for group in 0..self.instruction.width.groups_per_repeat() {
            accesses.extend(plan_c220_vector_read_accesses(
                self.control,
                self.addresses,
                repeat_index,
                &self.iteration_masks[repeat_index],
                2,
                self.instruction.width.element_bytes(),
                Some(group as u8),
            )?);
        }
        Ok(accesses)
    }
}

pub fn plan_c220_compare_mask_issue(
    pc: u64,
    word: u32,
    control_value: u64,
    mask: C220VectorMaskState,
    initial_compare_mask: C220CompareMask,
    registers: &[u64; 32],
    ub: &UbMemory,
) -> Result<C220CompareMaskIssue, C220VectorError> {
    let instruction = C220CompareMaskInstruction::decode(word)
        .ok_or(C220VectorError::UnsupportedWord { pc, word })?;
    let control = C220VectorControl::decode_binary(control_value);
    let addresses = C220VectorAddresses {
        destination: 0,
        source_0: registers[usize::from(instruction.source_0_register)],
        source_1: registers[usize::from(instruction.source_1_register)],
    };
    let lane_count = C220_VECTOR_TILE_BYTES / usize::from(instruction.width.element_bytes());
    let iteration_masks = decode_c220_repeat_masks(
        mask.control,
        mask.low,
        mask.high,
        lane_count,
        control.encoded_repeat_count,
    )?;
    let issue = C220CompareMaskIssue {
        pc,
        word,
        instruction,
        control,
        addresses,
        iteration_masks,
        initial_compare_mask,
    };
    for uop_index in 0..issue.uop_count() {
        for access in issue.read_accesses_for_uop(uop_index)? {
            ub.check_range(access.address, 32)?;
        }
    }
    Ok(issue)
}

pub fn evaluate_c220_compare_mask_uop(
    issue: &C220CompareMaskIssue,
    repeat_index: usize,
    lane_slice: Option<(usize, usize)>,
    source_0_bytes: &[u8],
    source_1_bytes: &[u8],
) -> Result<Vec<(bool, bool)>, C220VectorError> {
    for source in [source_0_bytes, source_1_bytes] {
        if source.len() != C220_VECTOR_TILE_BYTES {
            return Err(C220VectorError::InvalidSourceTile {
                actual: source.len(),
                expected: C220_VECTOR_TILE_BYTES,
            });
        }
    }
    issue.split_uop(repeat_index)?;
    let element_bytes = usize::from(issue.instruction.width.element_bytes());
    let (first_lane, lane_count) = lane_slice.unwrap_or((0, issue.instruction.width.lane_count()));
    if first_lane + lane_count > issue.instruction.width.lane_count() {
        return Err(C220VectorError::InvalidLaneGroup((first_lane / 64) as u8));
    }
    let mask = issue.iteration_masks[repeat_index];
    Ok((first_lane..first_lane + lane_count)
        .map(|lane| {
            let active = mask[lane / 64] & (1_u64 << (lane % 64)) != 0;
            let offset = lane * element_bytes;
            let value = active
                && evaluate_compare(
                    issue.instruction.width,
                    issue.instruction.condition,
                    &source_0_bytes[offset..],
                    &source_1_bytes[offset..],
                );
            (active, value)
        })
        .collect())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220PackedCompareIssue {
    pub pc: u64,
    pub word: u32,
    pub instruction: C220PackedCompareInstruction,
    pub control: C220VectorControl,
    pub addresses: C220VectorAddresses,
    pub scalar_bits: Option<u32>,
    pub(crate) write_targets: Vec<C220VectorStore>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220PackedCompareValueInputs {
    pub instruction: C220PackedCompareInstruction,
    pub control: C220VectorControl,
    pub addresses: C220VectorAddresses,
    pub scalar_bits: Option<u32>,
    pub repeat_index: usize,
}

impl C220PackedCompareIssue {
    pub fn uop_count(&self) -> usize {
        usize::from(self.control.encoded_repeat_count)
    }

    pub fn split_uop(&self, uop_index: usize) -> Result<(usize, u8), C220VectorError> {
        if uop_index >= self.uop_count() {
            return Err(C220VectorError::InvalidRepeatIndex(uop_index));
        }
        Ok((uop_index, 0))
    }

    pub fn read_accesses_for_uop(
        &self,
        uop_index: usize,
    ) -> Result<Vec<C220VectorReadAccess>, C220VectorError> {
        let (repeat_index, _) = self.split_uop(uop_index)?;
        let mut accesses = Vec::new();
        for group in 0..self.instruction.width.groups_per_repeat() {
            accesses.extend(plan_c220_vector_read_accesses(
                self.control,
                self.addresses,
                repeat_index,
                &[u64::MAX; 4],
                if self.instruction.operand.is_vector() {
                    2
                } else {
                    1
                },
                self.instruction.width.element_bytes(),
                Some(group as u8),
            )?);
        }
        Ok(accesses)
    }
}

pub fn plan_c220_packed_compare_issue(
    pc: u64,
    word: u32,
    control_value: u64,
    registers: &[u64; 32],
    ub: &UbMemory,
) -> Result<C220PackedCompareIssue, C220VectorError> {
    let instruction = C220PackedCompareInstruction::decode(word)
        .ok_or(C220VectorError::UnsupportedWord { pc, word })?;
    let control = if instruction.operand.is_vector() {
        C220VectorControl::decode_binary(control_value)
    } else {
        C220VectorControl::decode_unary(control_value)
    };
    let (source_1, scalar_bits) = match instruction.operand {
        C220PackedCompareOperand::VectorRegister(register) => {
            (registers[usize::from(register)], None)
        }
        C220PackedCompareOperand::ScalarRegister(register) => {
            (0, Some(registers[usize::from(register)] as u32))
        }
    };
    let addresses = C220VectorAddresses {
        destination: registers[usize::from(instruction.destination_register)],
        source_0: registers[usize::from(instruction.source_0_register)],
        source_1,
    };
    let mut issue = C220PackedCompareIssue {
        pc,
        word,
        instruction,
        control,
        addresses,
        scalar_bits,
        write_targets: Vec::new(),
    };
    for uop_index in 0..issue.uop_count() {
        for access in issue.read_accesses_for_uop(uop_index)? {
            ub.check_range(access.address, 32)?;
        }
        let packed_bytes = instruction.width.packed_bytes_per_repeat();
        for packed_offset in 0..packed_bytes {
            let address = addresses
                .destination
                .checked_add((uop_index * packed_bytes + packed_offset) as u64)
                .ok_or(C220VectorError::AddressOverflow {
                    base: addresses.destination,
                    lane: packed_offset * 8,
                })?;
            ub.check_range(address, 1)?;
            issue.write_targets.push(C220VectorStore {
                repeat_index: uop_index,
                lane_index: packed_offset,
                address,
                bank: C220UbBank::from_address(address),
                width_bytes: 1,
                data: [0; 8],
            });
        }
    }
    Ok(issue)
}

pub fn evaluate_c220_packed_compare_uop(
    inputs: C220PackedCompareValueInputs,
    source_0_bytes: &[u8],
    source_1_bytes: &[u8],
) -> Result<(Vec<u32>, Vec<C220VectorStore>), C220VectorError> {
    if source_0_bytes.len() != C220_VECTOR_TILE_BYTES {
        return Err(C220VectorError::InvalidSourceTile {
            actual: source_0_bytes.len(),
            expected: C220_VECTOR_TILE_BYTES,
        });
    }
    if inputs.instruction.operand.is_vector() && source_1_bytes.len() != C220_VECTOR_TILE_BYTES {
        return Err(C220VectorError::InvalidSourceTile {
            actual: source_1_bytes.len(),
            expected: C220_VECTOR_TILE_BYTES,
        });
    }
    if inputs.repeat_index >= usize::from(inputs.control.encoded_repeat_count) {
        return Err(C220VectorError::InvalidRepeatIndex(inputs.repeat_index));
    }
    let repeat_index = inputs.repeat_index;
    let element_bytes = usize::from(inputs.instruction.width.element_bytes());
    let lane_count = inputs.instruction.width.lane_count();
    let mut outcomes = Vec::with_capacity(lane_count);
    let mut stores = Vec::with_capacity(lane_count / 8);
    let repeat_offset = (repeat_index as u64)
        .checked_mul(inputs.instruction.width.packed_bytes_per_repeat() as u64)
        .ok_or(C220VectorError::AddressOverflow {
            base: inputs.addresses.destination,
            lane: 0,
        })?;
    let repeat_base = inputs
        .addresses
        .destination
        .checked_add(repeat_offset)
        .ok_or(C220VectorError::AddressOverflow {
            base: inputs.addresses.destination,
            lane: 0,
        })?;
    let scalar = inputs.scalar_bits.map(u32::to_le_bytes);
    for packed_offset in 0..lane_count / 8 {
        let mut packed = 0_u8;
        for bit_index in 0..8 {
            let lane = packed_offset * 8 + bit_index;
            let offset = lane * element_bytes;
            let second = scalar
                .as_ref()
                .map_or_else(|| &source_1_bytes[offset..], |bytes| bytes.as_slice());
            let result = evaluate_compare(
                inputs.instruction.width,
                inputs.instruction.condition,
                &source_0_bytes[offset..],
                second,
            );
            packed |= u8::from(result) << bit_index;
            outcomes.push(u32::from(result));
        }
        let address = repeat_base.checked_add(packed_offset as u64).ok_or(
            C220VectorError::AddressOverflow {
                base: inputs.addresses.destination,
                lane: packed_offset * 8,
            },
        )?;
        stores.push(C220VectorStore {
            repeat_index,
            lane_index: packed_offset,
            address,
            bank: C220UbBank::from_address(address),
            width_bytes: 1,
            data: crate::sim::c220::vector::access::store_data([packed]),
        });
    }
    Ok((outcomes, stores))
}

fn read_float(width: C220CompareWidth, bytes: &[u8]) -> f32 {
    match width {
        C220CompareWidth::F16 => {
            let bits = u16::from_le_bytes(bytes[..2].try_into().expect("f16 lane"));
            let sign = u32::from(bits & 0x8000) << 16;
            let exponent = u32::from((bits >> 10) & 0x1f);
            let fraction = u32::from(bits & 0x3ff);
            let converted = if exponent == 0 {
                if fraction == 0 {
                    sign
                } else {
                    let leading = 31 - fraction.leading_zeros();
                    let exponent = leading + 103;
                    sign | (exponent << 23) | ((fraction << (10 - leading) & 0x3ff) << 13)
                }
            } else if exponent == 31 {
                sign | 0x7f80_0000 | (fraction << 13)
            } else {
                sign | ((exponent + 112) << 23) | (fraction << 13)
            };
            f32::from_bits(converted)
        }
        C220CompareWidth::F32 => {
            f32::from_bits(u32::from_le_bytes(bytes[..4].try_into().expect("f32 lane")))
        }
        C220CompareWidth::S32 => unreachable!("signed integer is not a floating-point width"),
    }
}

fn evaluate_compare(
    width: C220CompareWidth,
    condition: C220CompareCondition,
    first: &[u8],
    second: &[u8],
) -> bool {
    if matches!(width, C220CompareWidth::S32) {
        let first = i32::from_le_bytes(first[..4].try_into().expect("s32 lane"));
        let second = i32::from_le_bytes(second[..4].try_into().expect("s32 lane"));
        return match condition {
            C220CompareCondition::Equal => first == second,
            C220CompareCondition::NotEqual => first != second,
            C220CompareCondition::Less => first < second,
            C220CompareCondition::Greater => first > second,
            C220CompareCondition::GreaterEqual => first >= second,
            C220CompareCondition::LessEqual => first <= second,
        };
    }
    let first = read_float(width, first);
    let second = read_float(width, second);
    match condition {
        C220CompareCondition::Equal => first == second,
        C220CompareCondition::NotEqual => !first.is_nan() && !second.is_nan() && first != second,
        C220CompareCondition::Less => first < second,
        C220CompareCondition::Greater => first > second,
        C220CompareCondition::GreaterEqual => first >= second,
        C220CompareCondition::LessEqual => first <= second,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::c220::vector::C220VectorInstruction;
    use std::num::NonZeroU64;

    use crate::architecture::Architecture;
    use crate::memory::sparse::MemoryByteState;
    use crate::sim::c220::core::C220CoreInstruction;
    use crate::sim::c220::state::C220State;
    use crate::sim::c220::vector::pipeline::{C220VectorPipeline, C220VectorTimingRules};
    use crate::sim::c220::vector::read::C220VectorReadIssue;
    use crate::sim::common::scalar::ScalarMachine;
    use crate::sim::common::scalar::ScalarStepper;

    #[test]
    fn compare_tail_preserves_issue_snapshot_not_previous_repeat() {
        for opcode in [0x9940_0000, 0x99c0_0000] {
            for zero_count in [false, true] {
                let word = opcode | (1 << 12) | (2 << 7) | (3 << 2);
                let width = C220CompareMaskInstruction::decode(word).unwrap().width;
                let mut ub = UbMemory::new(4096, 256);
                for address in [0x400, 0x500, 0x800, 0x900] {
                    ub.write_states(address, &[MemoryByteState::Known(0); 256])
                        .unwrap();
                }
                let mut registers = [0; 32];
                registers[1] = 0x400;
                registers[2] = 0x800;
                let seed = [0xaaaa_aaaa_aaaa_aaaa, 0x5555_5555_5555_5555];
                let issue = plan_c220_compare_mask_issue(
                    0,
                    word,
                    (8 << 40) | (8 << 32) | (1 << 16) | (1 << 8),
                    C220VectorMaskState {
                        control: 1 << 56,
                        low: if zero_count {
                            0
                        } else {
                            width.lane_count() as u64 + 1
                        },
                        high: 0,
                    },
                    C220CompareMask::from_bits(seed),
                    &registers,
                    &ub,
                )
                .unwrap();
                let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
                machine.set_spr_value(104, 0x1234).unwrap();
                machine.set_spr_value(105, 0x5678).unwrap();
                let mut core = C220State::new(ScalarStepper::new(machine, 0), ub);
                let mut pipeline = C220VectorPipeline::new(C220VectorTimingRules {
                    dispatch_ticks: 0,
                    uop_issue_interval: NonZeroU64::new(1).unwrap(),
                    ub_response_ticks: 2,
                });
                let uops = C220VectorInstruction::CompareMask(issue.clone())
                    .uops()
                    .unwrap();
                pipeline
                    .issue_at(
                        0,
                        &uops,
                        &[],
                        Some(C220VectorReadIssue::CompareMask(&issue)),
                    )
                    .unwrap();
                pipeline.advance_to(1000, &mut core).unwrap();
                let expected = if zero_count {
                    [0x1234, 0x5678]
                } else {
                    [
                        seed[0] | 1,
                        if width == C220CompareWidth::F16 {
                            seed[1]
                        } else {
                            0x5678
                        },
                    ]
                };
                assert_eq!(core.scalar().machine().spr_value(104), Some(expected[0]));
                assert_eq!(core.scalar().machine().spr_value(105), Some(expected[1]));
                assert!(!pipeline.has_pending_compare_mask_write());
                assert_eq!(
                    pipeline.last_functional_samples().len(),
                    if zero_count { 0 } else { 2 }
                );
                if !zero_count {
                    assert_eq!(
                        pipeline.last_functional_samples()[0]
                            .compare_update
                            .unwrap()
                            .values[0],
                        u64::MAX
                    );
                    assert_eq!(
                        pipeline.last_functional_samples()[1]
                            .compare_update
                            .unwrap()
                            .values[0],
                        seed[0] | 1
                    );
                }
            }
        }
    }

    #[test]
    fn packed_compare_writes_dense_results_for_supported_operands() {
        let mut registers = [0_u64; 32];
        registers[0] = 0x100;
        registers[1] = 0x400;
        let vector_control =
            (2_u64 << 56) | (8 << 40) | (8 << 32) | (0xab << 24) | (1 << 16) | (1 << 8) | 0x12;
        let scalar_control = (2_u64 << 56) | (8 << 40) | (8 << 32) | (1 << 16) | 0x12;
        for (word, bytes_per_repeat, uops, operand, control) in [
            (0x9800_110d, 16_usize, 2_usize, 0x800, vector_control),
            (0x9800_110e, 8_usize, 2_usize, 0x800, vector_control),
            (0x9800_110f, 8_usize, 2_usize, 0x800, vector_control),
            (0x9a00_110e, 16_usize, 2_usize, 0, scalar_control),
            (0x9a00_110f, 8_usize, 2_usize, 0, scalar_control),
            (0x9900_110e, 8_usize, 2_usize, 0, scalar_control),
            (
                0x9900_110e,
                8_usize,
                2_usize,
                0x1234_5678_dead_beef,
                scalar_control,
            ),
        ] {
            registers[2] = operand;
            registers[3] = control;
            let mut ub = UbMemory::new(4096, 256);
            for address in [0x400, 0x500, 0x800, 0x900] {
                ub.write_states(address, &[MemoryByteState::Known(0); 256])
                    .unwrap();
            }
            if word == 0x9900_110e {
                let scalar = (operand as u32).to_le_bytes();
                let source = std::array::from_fn::<_, 256, _>(|index| {
                    MemoryByteState::Known(scalar[index % scalar.len()])
                });
                for address in [0x400, 0x500] {
                    ub.write_states(address, &source).unwrap();
                }
            }
            let issue =
                plan_c220_packed_compare_issue(0x2000, word, control, &registers, &ub).unwrap();
            if !issue.instruction.operand.is_vector() {
                assert_eq!(issue.scalar_bits, Some(operand as u32));
                assert_eq!(issue.addresses.source_1, 0);
            }
            assert_eq!(issue.uop_count(), uops);
            assert_eq!(issue.write_targets.len(), bytes_per_repeat * 2);
            assert_eq!(issue.write_targets.first().unwrap().address, 0x100);
            assert_eq!(
                issue.write_targets.last().unwrap().address,
                0xff + (bytes_per_repeat * 2) as u64
            );
            assert!(issue.write_targets.iter().all(|store| store.data == [0; 8]));

            let instruction =
                C220CoreInstruction::Vector(C220VectorInstruction::PackedCompare(issue.clone()));
            let uops = instruction.as_vector().unwrap().uops().unwrap();
            let mut pipeline = C220VectorPipeline::new(C220VectorTimingRules {
                dispatch_ticks: 0,
                uop_issue_interval: NonZeroU64::new(1).unwrap(),
                ub_response_ticks: 1,
            });
            pipeline
                .issue_at(
                    0,
                    &uops,
                    &issue.write_targets,
                    Some(C220VectorReadIssue::PackedCompare(&issue)),
                )
                .unwrap();
            let machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
            let mut core = C220State::new(ScalarStepper::new(machine, 0x2000), ub);
            pipeline.advance_to(200, &mut core).unwrap();
            assert_eq!(
                core.ub().read_known(0x100, bytes_per_repeat * 2).unwrap(),
                vec![0xff; bytes_per_repeat * 2]
            );
            let mut alias_registers = registers;
            alias_registers[0] = 0x500;
            let alias =
                plan_c220_packed_compare_issue(0x2004, word, control, &alias_registers, core.ub())
                    .unwrap();
            let alias_uops = C220VectorInstruction::PackedCompare(alias.clone())
                .uops()
                .unwrap();
            pipeline
                .issue_at(
                    201,
                    &alias_uops,
                    &alias.write_targets,
                    Some(C220VectorReadIssue::PackedCompare(&alias)),
                )
                .unwrap();
            pipeline.advance_to(400, &mut core).unwrap();
            let overwritten_lanes =
                bytes_per_repeat / usize::from(alias.instruction.width.element_bytes());
            let mut expected = vec![0xff; bytes_per_repeat * 2];
            expected[bytes_per_repeat] = !((1_u16 << overwritten_lanes) - 1) as u8;
            assert_eq!(
                core.ub().read_known(0x500, bytes_per_repeat * 2).unwrap(),
                expected
            );
            assert_eq!(pipeline.last_functional_samples().len(), 2);
            assert!(
                pipeline
                    .last_read_samples()
                    .iter()
                    .all(|sample| sample.lanes.is_empty())
            );
        }
    }
}
