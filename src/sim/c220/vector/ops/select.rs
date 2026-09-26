use crate::isa::c220::vector::select::C220SelectInstruction;
use crate::memory::ub::UbMemory;
use crate::sim::c220::memory::C220UbBank;
use crate::sim::c220::vector::ops::compare::C220CompareMask;
use crate::sim::c220::vector::{
    C220_VECTOR_TILE_BYTES, C220VectorAddresses, C220VectorControl, C220VectorError,
    C220VectorMaskState, C220VectorReadAccess, C220VectorStore, decode_c220_repeat_masks,
    plan_c220_unary_write_targets, plan_c220_vector_read_accesses,
    vector_destination_address_for_width,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220SelectMode {
    CompareMask,
    TensorScalar,
    TensorTensor,
}

impl C220SelectMode {
    pub const fn decode(control: u64) -> Option<Self> {
        match (control >> 48) & 3 {
            0 => Some(Self::CompareMask),
            1 => Some(Self::TensorScalar),
            2 => Some(Self::TensorTensor),
            _ => None,
        }
    }

    pub const fn uses_tensor_mask(self) -> bool {
        matches!(self, Self::TensorScalar | Self::TensorTensor)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220SelectIssue {
    pub pc: u64,
    pub word: u32,
    pub instruction: C220SelectInstruction,
    pub mode: C220SelectMode,
    pub control: C220VectorControl,
    pub addresses: C220VectorAddresses,
    pub selection_mask_base: Option<u64>,
    pub iteration_masks: Vec<[u64; 4]>,
    pub(crate) write_targets: Vec<C220VectorStore>,
}

impl C220SelectIssue {
    pub(crate) fn functional_read_accesses(
        &self,
        repeat_index: usize,
    ) -> Result<Vec<C220VectorReadAccess>, C220VectorError> {
        if repeat_index >= self.iteration_masks.len() {
            return Err(C220VectorError::InvalidRepeatIndex(repeat_index));
        }
        let mut accesses = plan_c220_vector_read_accesses(
            self.control,
            self.addresses,
            repeat_index,
            &[u64::MAX; 4],
            if self.mode == C220SelectMode::TensorScalar {
                1
            } else {
                2
            },
            self.instruction.width.element_bytes(),
            None,
        )?;
        if self.mode.uses_tensor_mask() {
            let base = self
                .selection_mask_base
                .ok_or(C220VectorError::MissingSelectionMask)?;
            let offset = repeat_index * self.mask_bytes_per_repeat();
            let address = if self.mode == C220SelectMode::TensorTensor {
                u64::from((base as u32).wrapping_add(offset as u32))
            } else {
                base.checked_add(offset as u64)
                    .ok_or(C220VectorError::AddressOverflow {
                        base,
                        lane: repeat_index,
                    })?
            };
            accesses.push(C220VectorReadAccess {
                source_index: if self.mode == C220SelectMode::TensorScalar {
                    1
                } else {
                    2
                },
                block_index: 0,
                buffer_offset: 0,
                bytes: if self.mode == C220SelectMode::TensorScalar {
                    32
                } else {
                    self.mask_bytes_per_repeat() as u16
                },
                address,
                active_lane_mask: u16::MAX,
            });
        }
        Ok(accesses)
    }

    pub fn uop_count(&self) -> usize {
        self.iteration_masks.len() * self.instruction.width.groups_per_repeat()
    }

    pub fn split_uop(&self, uop_index: usize) -> Result<(usize, u8), C220VectorError> {
        let groups = self.instruction.width.groups_per_repeat();
        if uop_index >= self.iteration_masks.len() * groups {
            return Err(C220VectorError::InvalidRepeatIndex(uop_index));
        }
        Ok((uop_index / groups, (uop_index % groups) as u8))
    }

    pub fn read_accesses_for_repeat(
        &self,
        repeat_index: usize,
        lane_group: u8,
    ) -> Result<Vec<C220VectorReadAccess>, C220VectorError> {
        if repeat_index >= self.iteration_masks.len()
            || usize::from(lane_group) >= self.instruction.width.groups_per_repeat()
        {
            return Err(C220VectorError::InvalidRepeatIndex(repeat_index));
        }
        let mut accesses = plan_c220_vector_read_accesses(
            self.control,
            self.addresses,
            repeat_index,
            &self.iteration_masks[repeat_index],
            if matches!(self.mode, C220SelectMode::TensorScalar) {
                1
            } else {
                2
            },
            self.instruction.width.element_bytes(),
            Some(lane_group),
        )?;
        if self.loads_selection_mask(repeat_index, lane_group) {
            accesses.extend(self.mask_block_accesses(repeat_index, 1)?);
        }
        Ok(accesses)
    }

    pub const fn mask_bytes_per_repeat(&self) -> usize {
        C220_VECTOR_TILE_BYTES / self.instruction.width.element_bytes() as usize / 8
    }

    pub const fn mask_repeats_per_block(&self) -> usize {
        C220_VECTOR_TILE_BYTES / self.mask_bytes_per_repeat()
    }

    pub const fn loads_selection_mask(&self, repeat_index: usize, lane_group: u8) -> bool {
        matches!(self.mode, C220SelectMode::TensorScalar)
            && lane_group == 0
            && repeat_index.is_multiple_of(self.mask_repeats_per_block())
    }

    pub fn mask_block_accesses(
        &self,
        first_repeat: usize,
        source_index: u8,
    ) -> Result<Vec<C220VectorReadAccess>, C220VectorError> {
        if !self.mode.uses_tensor_mask()
            || !first_repeat.is_multiple_of(self.mask_repeats_per_block())
            || first_repeat >= self.iteration_masks.len()
        {
            return Err(C220VectorError::InvalidRepeatIndex(first_repeat));
        }
        let base = self
            .selection_mask_base
            .ok_or(C220VectorError::MissingSelectionMask)?;
        let block_index = first_repeat / self.mask_repeats_per_block();
        let address = base
            .checked_add((block_index * C220_VECTOR_TILE_BYTES) as u64)
            .ok_or(C220VectorError::AddressOverflow {
                base,
                lane: block_index,
            })?;
        Ok((0..8)
            .map(|index| C220VectorReadAccess {
                source_index,
                block_index: index,
                buffer_offset: u16::from(index) * 32,
                bytes: 32,
                address: address + u64::from(index) * 32,
                active_lane_mask: u16::MAX,
            })
            .collect())
    }
}

pub fn plan_c220_select_issue(
    pc: u64,
    word: u32,
    control_value: u64,
    mask: C220VectorMaskState,
    compare_mask: C220CompareMask,
    registers: &[u64; 32],
    ub: &UbMemory,
) -> Result<C220SelectIssue, C220VectorError> {
    let instruction =
        C220SelectInstruction::decode(word).ok_or(C220VectorError::UnsupportedWord { pc, word })?;
    let mode = C220SelectMode::decode(control_value).ok_or(
        C220VectorError::UnsupportedSelectMode(((control_value >> 48) & 3) as u8),
    )?;
    let control = C220VectorControl::decode_binary(control_value);
    let encoded_source_1 = registers[usize::from(instruction.source_1_register)];
    let selection_mask_base = match mode {
        C220SelectMode::CompareMask => None,
        C220SelectMode::TensorScalar => Some(encoded_source_1),
        C220SelectMode::TensorTensor => Some(compare_mask.bits()[0]),
    };
    let addresses = C220VectorAddresses {
        destination: registers[usize::from(instruction.destination_register)],
        source_0: registers[usize::from(instruction.source_0_register)],
        source_1: if matches!(mode, C220SelectMode::TensorScalar) {
            0
        } else {
            encoded_source_1
        },
    };
    let lane_count = C220_VECTOR_TILE_BYTES / usize::from(instruction.width.element_bytes());
    let iteration_masks = decode_c220_repeat_masks(
        mask.control,
        mask.low,
        mask.high,
        lane_count,
        control.encoded_repeat_count,
    )?;
    let write_targets = plan_c220_unary_write_targets(
        control,
        addresses,
        &iteration_masks,
        instruction.width.element_bytes(),
        ub,
    )?;
    let issue = C220SelectIssue {
        pc,
        word,
        instruction,
        mode,
        control,
        addresses,
        selection_mask_base,
        iteration_masks,
        write_targets,
    };
    for uop_index in 0..issue.uop_count() {
        let (repeat_index, lane_group) = issue.split_uop(uop_index)?;
        for access in issue.read_accesses_for_repeat(repeat_index, lane_group)? {
            ub.check_range(access.address, 32)?;
        }
    }
    if mode.uses_tensor_mask() {
        for first_repeat in (0..issue.iteration_masks.len()).step_by(issue.mask_repeats_per_block())
        {
            for access in issue.mask_block_accesses(first_repeat, 0)? {
                ub.check_range(access.address, 32)?;
            }
        }
    }
    Ok(issue)
}

pub(crate) fn evaluate_c220_select_uop(
    issue: &C220SelectIssue,
    repeat_index: usize,
    lane_group: u8,
    compare_mask: C220CompareMask,
    selection_mask: C220CompareMask,
    source_0_bytes: &[u8],
    source_1_bytes: &[u8],
) -> Result<(Vec<u32>, Vec<C220VectorStore>), C220VectorError> {
    for source in [source_0_bytes, source_1_bytes] {
        if source.len() != C220_VECTOR_TILE_BYTES {
            return Err(C220VectorError::InvalidSourceTile {
                actual: source.len(),
                expected: C220_VECTOR_TILE_BYTES,
            });
        }
    }
    if repeat_index >= issue.iteration_masks.len()
        || usize::from(lane_group) >= issue.instruction.width.groups_per_repeat()
    {
        return Err(C220VectorError::InvalidRepeatIndex(repeat_index));
    }
    let element_bytes = usize::from(issue.instruction.width.element_bytes());
    let first_lane = usize::from(lane_group) * 64;
    let mask = issue.iteration_masks[repeat_index];
    let mut values = Vec::with_capacity(64);
    let mut stores = Vec::with_capacity(64);
    for lane in first_lane..first_lane + 64 {
        let active = mask[lane / 64] & (1_u64 << (lane % 64)) != 0;
        let offset = lane * element_bytes;
        let selected = match issue.mode {
            C220SelectMode::CompareMask => compare_mask.test(lane),
            C220SelectMode::TensorScalar | C220SelectMode::TensorTensor => {
                selection_mask.test(lane)
            }
        };
        let bits = if !active {
            0
        } else if !selected && matches!(issue.mode, C220SelectMode::TensorScalar) {
            match element_bytes {
                2 => compare_mask.bits()[0] as u16 as u32,
                4 => compare_mask.bits()[0] as u32,
                _ => unreachable!("validated select width"),
            }
        } else {
            let source = if selected {
                source_0_bytes
            } else {
                source_1_bytes
            };
            u32::from_le_bytes(match element_bytes {
                2 => [source[offset], source[offset + 1], 0, 0],
                4 => source[offset..offset + 4]
                    .try_into()
                    .expect("four-byte lane"),
                _ => unreachable!("validated select width"),
            })
        };
        values.push(bits);
        if !active {
            continue;
        }
        let address = vector_destination_address_for_width(
            issue.control,
            issue.addresses,
            repeat_index,
            lane,
            issue.instruction.width.element_bytes(),
        )?;
        stores.push(C220VectorStore {
            repeat_index,
            lane_index: lane,
            address,
            bank: C220UbBank::from_address(address),
            width_bytes: issue.instruction.width.element_bytes(),
            data: crate::sim::c220::vector::access::store_data(bits.to_le_bytes()),
        });
    }
    Ok((values, stores))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::c220::vector::C220VectorInstruction;
    use std::num::NonZeroU64;

    use crate::architecture::Architecture;
    use crate::memory::sparse::MemoryByteState;
    use crate::sim::c220::state::C220State;
    use crate::sim::c220::vector::ops::compare::{
        plan_c220_compare_mask_issue, plan_c220_move_mask_issue,
    };
    use crate::sim::c220::vector::pipeline::{C220VectorPipeline, C220VectorTimingRules};
    use crate::sim::c220::vector::read::C220VectorReadIssue;
    use crate::sim::common::scalar::ScalarMachine;
    use crate::sim::common::scalar::ScalarStepper;

    #[test]
    fn select_repeats_read_current_inputs_and_masks() {
        for opcode in [0x9d40_0000, 0x9dc0_0000] {
            for mode in 0..3_u64 {
                for repeats in [0, 2_u64] {
                    let mut ub = UbMemory::new(4096, 4096);
                    ub.write_states(0, &[MemoryByteState::Known(0); 4096])
                        .unwrap();
                    let mut registers = [0; 32];
                    registers[0] = if mode == 0 { 0x500 } else { 0x800 };
                    registers[1] = 0x400;
                    registers[2] = if mode == 1 { 0x800 } else { 0xa00 };
                    if mode == 0 {
                        ub.write_states(0x400, &[MemoryByteState::Known(0x11); 256])
                            .unwrap();
                    }
                    ub.write_states(0x800, &[MemoryByteState::Known(0xff); 256])
                        .unwrap();
                    ub.write_states(0xa00, &[MemoryByteState::Known(0x22); 512])
                        .unwrap();
                    let control = (repeats << 56)
                        | (mode << 48)
                        | (8 << 40)
                        | (8 << 32)
                        | (8 << 24)
                        | (1 << 16)
                        | (1 << 8)
                        | 1;
                    let issue = plan_c220_select_issue(
                        0,
                        opcode | (1 << 12) | (2 << 7) | (3 << 2),
                        control,
                        C220VectorMaskState {
                            control: 0,
                            low: u64::MAX,
                            high: u64::MAX,
                        },
                        C220CompareMask::from_bits([0xc00, 0]),
                        &registers,
                        &ub,
                    )
                    .unwrap();
                    let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
                    let low = match mode {
                        0 => u64::MAX,
                        1 => 0x3333_3333,
                        _ => (1 << 32) | 0x800,
                    };
                    machine.set_spr_value(104, low).unwrap();
                    machine.set_spr_value(105, u64::MAX).unwrap();
                    let mut core = C220State::new(ScalarStepper::new(machine, 0), ub);
                    let mut pipeline = C220VectorPipeline::new(C220VectorTimingRules {
                        dispatch_ticks: 0,
                        uop_issue_interval: NonZeroU64::new(1).unwrap(),
                        ub_response_ticks: 2,
                    });
                    let uops = C220VectorInstruction::Select(issue.clone()).uops().unwrap();
                    pipeline
                        .issue_at(
                            0,
                            &uops,
                            &issue.write_targets,
                            Some(C220VectorReadIssue::Select(&issue)),
                        )
                        .unwrap();
                    pipeline.advance_to(1000, &mut core).unwrap();
                    assert_eq!(pipeline.last_functional_samples().len(), repeats as usize);
                    assert!(
                        pipeline
                            .last_read_samples()
                            .iter()
                            .all(|sample| sample.lanes.is_empty())
                    );
                    if repeats != 0 {
                        let expected = match mode {
                            0 => 0x11,
                            1 => 0x33,
                            _ => 0x22,
                        };
                        assert_eq!(
                            core.ub().read_known(registers[0] + 256, 256).unwrap(),
                            vec![expected; 256]
                        );
                        assert_eq!(
                            pipeline.last_functional_samples()[1].lanes.len(),
                            issue.instruction.width.lane_count()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn compare_mask_becomes_visible_to_dependent_select() {
        let mut registers = [0_u64; 32];
        registers[..6].copy_from_slice(&[
            0x100,
            0x400,
            0x600,
            (1_u64 << 56) | (8 << 40) | (8 << 32) | (1 << 16) | (1 << 8) | 1,
            0x800,
            0xa00,
        ]);
        registers[6] = 0xc00;
        registers[7] = 0xd00;
        registers[8] = 0xb00;
        let mut ub = UbMemory::new(4096, 256);
        for (address, value) in [
            (0x400, 0_u8),
            (0x600, 0),
            (0x800, 0x11),
            (0xa00, 0x22),
            (0xb00, 0x22),
            (0x100, 0),
            (0xe00, 0xaa),
        ] {
            ub.write_states(address, &[MemoryByteState::Known(value); 256])
                .unwrap();
        }
        ub.write_states(0xa00, &[MemoryByteState::Known(0x55); 16])
            .unwrap();
        let mut tensor_mask_address = [MemoryByteState::Known(0); 16];
        for (destination, byte) in tensor_mask_address.iter_mut().zip(0xe00_u64.to_le_bytes()) {
            *destination = MemoryByteState::Known(byte);
        }
        ub.write_states(0xd00, &tensor_mask_address).unwrap();
        let compare = plan_c220_compare_mask_issue(
            0x2000,
            0x9940_110c,
            registers[3],
            C220VectorMaskState {
                control: 0,
                low: u64::MAX,
                high: u64::MAX,
            },
            C220CompareMask::default(),
            &registers,
            &ub,
        )
        .unwrap();
        let select = plan_c220_select_issue(
            0x2004,
            0x9d40_428c,
            registers[3],
            C220VectorMaskState {
                control: 0,
                low: u64::MAX,
                high: u64::MAX,
            },
            C220CompareMask::default(),
            &registers,
            &ub,
        )
        .unwrap();
        let mut pipeline = C220VectorPipeline::new(C220VectorTimingRules {
            dispatch_ticks: 0,
            uop_issue_interval: NonZeroU64::new(1).unwrap(),
            ub_response_ticks: 1,
        });
        pipeline
            .issue_at(
                0,
                &C220VectorInstruction::CompareMask(compare.clone())
                    .uops()
                    .unwrap(),
                &[],
                Some(C220VectorReadIssue::CompareMask(&compare)),
            )
            .unwrap();
        pipeline
            .issue_at(
                1,
                &C220VectorInstruction::Select(select.clone())
                    .uops()
                    .unwrap(),
                &select.write_targets,
                Some(C220VectorReadIssue::Select(&select)),
            )
            .unwrap();
        let machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
        let mut core = C220State::new(ScalarStepper::new(machine, 0x2000), ub);
        pipeline.advance_to(300, &mut core).unwrap();
        assert_eq!(pipeline.compare_mask().bits(), [u64::MAX; 2]);
        assert_eq!(core.ub().read_known(0x100, 256).unwrap(), vec![0x11; 256]);

        let save = plan_c220_move_mask_issue(0x2008, 0x9e0c_0002, &registers, core.ub()).unwrap();
        pipeline
            .issue_at(
                301,
                &C220VectorInstruction::MoveMask(save.clone())
                    .uops()
                    .unwrap(),
                &save.write_targets,
                Some(C220VectorReadIssue::MoveMask(&save)),
            )
            .unwrap();
        pipeline.advance_to(400, &mut core).unwrap();
        assert_eq!(core.ub().read_known(0xc00, 16).unwrap(), vec![0xff; 16]);

        let clear = plan_c220_compare_mask_issue(
            0x200c,
            0x9942_110c,
            registers[3],
            C220VectorMaskState {
                control: 0,
                low: u64::MAX,
                high: u64::MAX,
            },
            pipeline.compare_mask(),
            &registers,
            core.ub(),
        )
        .unwrap();
        pipeline
            .issue_at(
                401,
                &C220VectorInstruction::CompareMask(clear.clone())
                    .uops()
                    .unwrap(),
                &[],
                Some(C220VectorReadIssue::CompareMask(&clear)),
            )
            .unwrap();
        pipeline.advance_to(500, &mut core).unwrap();
        assert_eq!(pipeline.compare_mask().bits(), [0; 2]);

        let restore =
            plan_c220_move_mask_issue(0x2010, 0x9e0c_0003, &registers, core.ub()).unwrap();
        core.ub_mut()
            .write_states(0xc00, &[MemoryByteState::Known(0); 16])
            .unwrap();
        pipeline
            .issue_at(
                501,
                &C220VectorInstruction::MoveMask(restore.clone())
                    .uops()
                    .unwrap(),
                &[],
                Some(C220VectorReadIssue::MoveMask(&restore)),
            )
            .unwrap();
        let mut released = false;
        for tick in 502..600 {
            let releases = pipeline.advance_to(tick, &mut core).unwrap();
            if releases.iter().any(|release| release.pc == 0x2010) {
                assert_eq!(pipeline.compare_mask().bits(), [0; 2]);
                assert!(
                    pipeline.has_pending_compare_mask_write(),
                    "tick={tick} releases={releases:?}"
                );
                core.ub_mut()
                    .write_states(0xc00, &[MemoryByteState::Known(0xff); 16])
                    .unwrap();
                pipeline.advance_to(tick + 1, &mut core).unwrap();
                assert_eq!(pipeline.compare_mask().bits(), [u64::MAX; 2]);
                assert!(!pipeline.has_pending_compare_mask_write());
                released = true;
                break;
            }
        }
        assert!(released);
        pipeline.advance_to(600, &mut core).unwrap();
        assert_eq!(pipeline.compare_mask().bits(), [u64::MAX; 2]);

        let tensor_scalar = plan_c220_select_issue(
            0x2014,
            0x9d40_428c,
            registers[3] | (1 << 48),
            C220VectorMaskState {
                control: 0,
                low: u64::MAX,
                high: u64::MAX,
            },
            pipeline.compare_mask(),
            &registers,
            core.ub(),
        )
        .unwrap();
        pipeline
            .issue_at(
                601,
                &C220VectorInstruction::Select(tensor_scalar.clone())
                    .uops()
                    .unwrap(),
                &tensor_scalar.write_targets,
                Some(C220VectorReadIssue::Select(&tensor_scalar)),
            )
            .unwrap();
        pipeline.advance_to(900, &mut core).unwrap();
        assert_eq!(
            core.ub().read_known(0x100, 8).unwrap(),
            vec![0x11, 0x11, 0xff, 0xff, 0x11, 0x11, 0xff, 0xff]
        );

        let load_tensor_mask =
            plan_c220_move_mask_issue(0x2018, 0x9e0e_0003, &registers, core.ub()).unwrap();
        pipeline
            .issue_at(
                901,
                &C220VectorInstruction::MoveMask(load_tensor_mask.clone())
                    .uops()
                    .unwrap(),
                &[],
                Some(C220VectorReadIssue::MoveMask(&load_tensor_mask)),
            )
            .unwrap();
        pipeline.advance_to(1000, &mut core).unwrap();
        assert_eq!(pipeline.compare_mask().bits(), [0xe00, 0]);

        let tensor_tensor = plan_c220_select_issue(
            0x201c,
            0x9d40_440c,
            registers[3] | (2 << 48),
            C220VectorMaskState {
                control: 0,
                low: u64::MAX,
                high: u64::MAX,
            },
            pipeline.compare_mask(),
            &registers,
            core.ub(),
        )
        .unwrap();
        pipeline
            .issue_at(
                1001,
                &C220VectorInstruction::Select(tensor_tensor.clone())
                    .uops()
                    .unwrap(),
                &tensor_tensor.write_targets,
                Some(C220VectorReadIssue::Select(&tensor_tensor)),
            )
            .unwrap();
        pipeline.advance_to(1300, &mut core).unwrap();
        assert_eq!(
            core.ub().read_known(0x100, 8).unwrap(),
            vec![0x22, 0x22, 0x11, 0x11, 0x22, 0x22, 0x11, 0x11]
        );
    }
}
