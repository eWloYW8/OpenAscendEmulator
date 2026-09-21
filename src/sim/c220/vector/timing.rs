use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

use crate::architecture::c220::C220UbBank;
use crate::isa::c220::gather::C220GatherKind;
use crate::isa::c220::reduce::{
    C220ExtremumOperation, C220ReductionInstruction, C220ReductionKind, C220ReductionWidth,
};
use crate::isa::c220::ternary::C220TernaryInstruction;
use crate::isa::c220::vector::{
    C220MovevInstruction, C220VecArithmeticHint, C220VecArithmeticOperation,
};
use crate::isa::c220::vector_scalar::{
    C220VectorScalarInstruction, C220VectorScalarOperation, C220VectorScalarType,
};
use crate::sim::c220::ub_arbiter::{C220UbCycle, C220UbRequest};
use crate::sim::c220::vector::{C220_VECTOR_BLOCK_BYTES, C220_VECTOR_BLOCK_COUNT, C220VectorStore};

const PARTIAL_WRITE_OCCUPANCY_TICKS: usize = 6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Latency components before the UB write stage of an admitted vector uop.
pub struct C220VectorUopStages {
    pub read_ticks: u8,
    pub execute_ticks: u8,
}

impl C220VectorUopStages {
    pub const fn empty() -> Self {
        Self {
            read_ticks: 6,
            execute_ticks: 1,
        }
    }

    pub const fn movev(instruction: C220MovevInstruction) -> Option<Self> {
        if instruction.supported_element_bytes().is_none() {
            return None;
        }
        Some(Self {
            read_ticks: 6,
            execute_ticks: 1,
        })
    }

    pub const fn floating_binary_arithmetic(operation: C220VecArithmeticOperation) -> Option<Self> {
        let execute_ticks = match operation {
            C220VecArithmeticOperation::Absolute
            | C220VecArithmeticOperation::Rectify
            | C220VecArithmeticOperation::Not
            | C220VecArithmeticOperation::Or
            | C220VecArithmeticOperation::And => {
                return None;
            }
            C220VecArithmeticOperation::Add
            | C220VecArithmeticOperation::Subtract
            | C220VecArithmeticOperation::AddRectify
            | C220VecArithmeticOperation::SubtractRectify => 7,
            C220VecArithmeticOperation::Maximum | C220VecArithmeticOperation::Minimum => 5,
            C220VecArithmeticOperation::Multiply => 8,
            C220VecArithmeticOperation::Divide => 11,
        };
        Some(Self {
            read_ticks: 6,
            execute_ticks,
        })
    }

    pub const fn vector_arithmetic(hint: C220VecArithmeticHint) -> Option<Self> {
        if hint.has_bitwise_b16_value_path() {
            Some(Self {
                read_ticks: 6,
                execute_ticks: 1,
            })
        } else if hint.has_s32_value_path() || hint.has_s16_value_path() {
            let execute_ticks = match hint.operation {
                C220VecArithmeticOperation::Add
                | C220VecArithmeticOperation::Subtract
                | C220VecArithmeticOperation::AddRectify
                | C220VecArithmeticOperation::SubtractRectify
                | C220VecArithmeticOperation::Maximum
                | C220VecArithmeticOperation::Minimum => 5,
                C220VecArithmeticOperation::Multiply => 6,
                _ => return None,
            };
            Some(Self {
                read_ticks: 6,
                execute_ticks,
            })
        } else if hint.has_f16_value_path() || hint.has_fp32_value_path() {
            match hint.operation {
                C220VecArithmeticOperation::Absolute => Some(Self::abs()),
                C220VecArithmeticOperation::Rectify => Some(Self::relu()),
                operation => Self::floating_binary_arithmetic(operation),
            }
        } else {
            None
        }
    }

    pub const fn abs() -> Self {
        Self {
            read_ticks: 6,
            execute_ticks: 15,
        }
    }

    pub const fn relu() -> Self {
        Self {
            read_ticks: 6,
            execute_ticks: 6,
        }
    }

    pub const fn vector_scalar(instruction: C220VectorScalarInstruction) -> Self {
        Self {
            read_ticks: 6,
            execute_ticks: match (instruction.operation, instruction.dtype) {
                (C220VectorScalarOperation::LeakyRelu, _) => 8,
                (C220VectorScalarOperation::Maximum | C220VectorScalarOperation::Minimum, _) => 5,
                (
                    C220VectorScalarOperation::Add,
                    C220VectorScalarType::S16 | C220VectorScalarType::S32,
                ) => 5,
                (
                    C220VectorScalarOperation::Add,
                    C220VectorScalarType::F16 | C220VectorScalarType::F32,
                ) => 7,
                (
                    C220VectorScalarOperation::Multiply,
                    C220VectorScalarType::S16 | C220VectorScalarType::S32,
                ) => 6,
                (
                    C220VectorScalarOperation::Multiply,
                    C220VectorScalarType::F16 | C220VectorScalarType::F32,
                ) => 8,
            },
        }
    }

    pub const fn shift() -> Self {
        Self {
            read_ticks: 6,
            execute_ticks: 6,
        }
    }

    pub const fn copy() -> Self {
        Self {
            read_ticks: 6,
            execute_ticks: 1,
        }
    }

    pub const fn broadcast() -> Self {
        Self {
            read_ticks: 6,
            execute_ticks: 2,
        }
    }

    pub const fn transpose() -> Self {
        Self {
            read_ticks: 6,
            execute_ticks: 1,
        }
    }

    pub const fn packed_compare() -> Self {
        Self {
            read_ticks: 6,
            execute_ticks: 5,
        }
    }

    pub const fn select() -> Self {
        Self {
            read_ticks: 6,
            execute_ticks: 1,
        }
    }

    pub const fn reduction(instruction: C220ReductionInstruction) -> Self {
        let execute_ticks = match (instruction.kind, instruction.width) {
            (C220ReductionKind::WholeAdd { .. }, _) => 5,
            (
                C220ReductionKind::WholeExtremum {
                    operation: C220ExtremumOperation::Maximum,
                    ..
                },
                C220ReductionWidth::F16,
            ) => 24,
            (
                C220ReductionKind::WholeExtremum {
                    operation: C220ExtremumOperation::Maximum,
                    ..
                },
                C220ReductionWidth::F32,
            ) => 21,
            (
                C220ReductionKind::WholeExtremum {
                    operation: C220ExtremumOperation::Minimum,
                    ..
                },
                C220ReductionWidth::F16,
            ) => 10,
            (
                C220ReductionKind::WholeExtremum {
                    operation: C220ExtremumOperation::Minimum,
                    ..
                },
                C220ReductionWidth::F32,
            ) => 9,
            (C220ReductionKind::GroupAdd, C220ReductionWidth::F16) => 7,
            (C220ReductionKind::GroupAdd, C220ReductionWidth::F32) => 6,
            (
                C220ReductionKind::GroupExtremum {
                    operation: C220ExtremumOperation::Maximum,
                },
                C220ReductionWidth::F16,
            ) => 10,
            (
                C220ReductionKind::GroupExtremum {
                    operation: C220ExtremumOperation::Maximum,
                },
                C220ReductionWidth::F32,
            ) => 9,
            (
                C220ReductionKind::GroupExtremum {
                    operation: C220ExtremumOperation::Minimum,
                },
                C220ReductionWidth::F16,
            ) => 7,
            (
                C220ReductionKind::GroupExtremum {
                    operation: C220ExtremumOperation::Minimum,
                },
                C220ReductionWidth::F32,
            ) => 6,
            (C220ReductionKind::PairAdd, _) => 1,
        };
        Self {
            read_ticks: 6,
            execute_ticks,
        }
    }

    pub const fn ternary(_instruction: C220TernaryInstruction) -> Self {
        Self {
            read_ticks: 6,
            execute_ticks: 11,
        }
    }

    pub const fn gather(kind: C220GatherKind) -> Self {
        Self {
            read_ticks: 6,
            execute_ticks: match kind {
                C220GatherKind::Elements(_) => 1,
                C220GatherKind::Blocks => 2,
            },
        }
    }

    pub const fn gather_index() -> Self {
        Self {
            read_ticks: 0,
            execute_ticks: 0,
        }
    }

    /// Delay until write-lane release; a fully masked uop sends no UB request.
    pub const fn release_offset(self, writeback_ticks: usize) -> usize {
        self.read_ticks as usize + self.execute_ticks as usize + writeback_ticks
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220VectorUopKind {
    Ordinary,
    GatherIndex { group: u8 },
    GatherData { group: u8 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220VectorUop {
    pub pc: u64,
    pub repeat_index: usize,
    pub lane_group: Option<u8>,
    pub kind: C220VectorUopKind,
    pub stages: C220VectorUopStages,
    pub writeback_ticks: usize,
    pub writes_ub: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Departure from the vector write lane, not completion of the UB write.
pub struct C220VectorUopRelease {
    pub pc: u64,
    pub repeat_index: usize,
    pub lane_group: Option<u8>,
    pub kind: C220VectorUopKind,
    pub admission_tick: u64,
    pub eligible_tick: u64,
    pub release_tick: u64,
    pub ub_write_requested: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum C220VectorTimelineError {
    #[error("vector timeline moved backward from tick {previous} to {requested}")]
    TimeReversed { previous: u64, requested: u64 },
}

#[derive(Default)]
struct GroupWriteState {
    next_slot: usize,
    completion: usize,
    partial_slots: BTreeSet<usize>,
}

impl GroupWriteState {
    fn accept(&mut self, full: bool) {
        let slot = self.next_slot;
        if !full {
            self.partial_slots.insert(slot);
        }
        let mut next_slot = slot + 1;
        if slot >= PARTIAL_WRITE_OCCUPANCY_TICKS {
            let mut prior_slot = slot - PARTIAL_WRITE_OCCUPANCY_TICKS;
            let limit = slot + PARTIAL_WRITE_OCCUPANCY_TICKS;
            while self.partial_slots.contains(&prior_slot) {
                next_slot += 1;
                if next_slot > limit {
                    break;
                }
                prior_slot += 1;
            }
        }
        self.next_slot = next_slot;
        self.completion = self
            .completion
            .max(next_slot + usize::from(!full) * PARTIAL_WRITE_OCCUPANCY_TICKS);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220VectorWriteBlock {
    pub repeat_index: usize,
    pub block_index: u8,
    pub base_address: u64,
    pub bank: C220UbBank,
    pub element_bytes: u8,
    pub active_lane_mask: u32,
}

impl C220VectorWriteBlock {
    pub const fn active_lanes(self) -> u32 {
        self.active_lane_mask.count_ones()
    }

    pub const fn full(self) -> bool {
        match self.element_bytes {
            1 => self.active_lane_mask == u32::MAX,
            2 => self.active_lane_mask == u16::MAX as u32,
            4 => self.active_lane_mask == 0xff,
            _ => false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220VectorWritePlan {
    pub blocks: Vec<C220VectorWriteBlock>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum C220VectorWritePlanError {
    #[error("C220 vector store uses unsupported element width {0}")]
    ElementWidth(u8),
    #[error("C220 vector store lane {lane} is outside its tile")]
    LaneOutsideTile { lane: usize },
    #[error("C220 vector stores disagree on the address or width of repeat {repeat} block {block}")]
    InconsistentBlock { repeat: usize, block: usize },
    #[error("C220 vector store address underflows at lane {lane}")]
    AddressUnderflow { lane: usize },
}

impl C220VectorWritePlan {
    pub fn from_stores(stores: &[C220VectorStore]) -> Result<Self, C220VectorWritePlanError> {
        let mut blocks: BTreeMap<(usize, usize), C220VectorWriteBlock> = BTreeMap::new();
        for store in stores {
            let width = usize::from(store.width_bytes);
            if !matches!(width, 1 | 2 | 4) {
                return Err(C220VectorWritePlanError::ElementWidth(store.width_bytes));
            }
            let lanes_per_block = C220_VECTOR_BLOCK_BYTES / width;
            let block = store.lane_index / lanes_per_block;
            if block >= C220_VECTOR_BLOCK_COUNT {
                return Err(C220VectorWritePlanError::LaneOutsideTile {
                    lane: store.lane_index,
                });
            }
            let lane = store.lane_index % lanes_per_block;
            let base_address = store.address.checked_sub((lane * width) as u64).ok_or(
                C220VectorWritePlanError::AddressUnderflow {
                    lane: store.lane_index,
                },
            )?;
            let key = (store.repeat_index, block);
            match blocks.get_mut(&key) {
                Some(existing) => {
                    if existing.base_address != base_address
                        || existing.element_bytes != store.width_bytes
                    {
                        return Err(C220VectorWritePlanError::InconsistentBlock {
                            repeat: store.repeat_index,
                            block,
                        });
                    }
                    existing.active_lane_mask |= 1 << lane;
                }
                None => {
                    blocks.insert(
                        key,
                        C220VectorWriteBlock {
                            repeat_index: store.repeat_index,
                            block_index: block as u8,
                            base_address,
                            bank: C220UbBank::from_address(base_address),
                            element_bytes: store.width_bytes,
                            active_lane_mask: 1 << lane,
                        },
                    );
                }
            }
        }
        Ok(Self {
            blocks: blocks.into_values().collect(),
        })
    }

    pub fn writeback_ticks(&self) -> Option<usize> {
        let repeat = self.blocks.first()?.repeat_index;
        if self.blocks.iter().any(|block| block.repeat_index != repeat) {
            return None;
        }
        self.writeback_ticks_for_repeat(repeat)
    }

    pub fn writeback_ticks_for_repeat(&self, repeat_index: usize) -> Option<usize> {
        self.writeback_ticks_matching(repeat_index, None)
    }

    pub fn writeback_ticks_for_lane_group(
        &self,
        repeat_index: usize,
        lane_group: u8,
    ) -> Option<usize> {
        self.writeback_ticks_matching(repeat_index, Some(lane_group))
    }

    fn writeback_ticks_matching(
        &self,
        repeat_index: usize,
        lane_group: Option<u8>,
    ) -> Option<usize> {
        let blocks = self
            .blocks
            .iter()
            .filter(|block| block.repeat_index == repeat_index)
            .filter(|block| {
                let first_lane = usize::from(block.block_index)
                    * (C220_VECTOR_BLOCK_BYTES / usize::from(block.element_bytes));
                lane_group.is_none_or(|group| first_lane / 64 == usize::from(group))
            })
            .collect::<Vec<_>>();
        if blocks.is_empty() {
            return None;
        }
        if blocks.iter().all(|block| block.full()) {
            let accesses = blocks
                .iter()
                .map(|block| (block.base_address, C220_VECTOR_BLOCK_BYTES))
                .collect::<Vec<_>>();
            let mut request = C220UbRequest::from_accesses(&accesses).ok()?;
            let mut tick = 0_u64;
            while !request.is_complete() {
                C220UbCycle::arbitrate(tick, Some(&mut request), None, None);
                tick = tick.checked_add(1)?;
            }
            return usize::try_from(tick).ok();
        }
        let mut groups: [GroupWriteState; 16] = std::array::from_fn(|_| GroupWriteState::default());
        for block in blocks {
            groups
                .get_mut(usize::from(block.bank.group))?
                .accept(block.full());
        }
        groups.iter().map(|group| group.completion).max()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::c220::vector::C220_CAPTURED_MOVEV_WORD;

    #[test]
    fn supported_vector_uops_have_distinct_execution_stages() {
        let movev = C220VectorUopStages::movev(
            C220MovevInstruction::decode(C220_CAPTURED_MOVEV_WORD).unwrap(),
        )
        .unwrap();
        assert_eq!(movev.release_offset(1), 8);
        assert_eq!(
            C220VectorUopStages::floating_binary_arithmetic(C220VecArithmeticOperation::Add)
                .unwrap()
                .release_offset(1),
            14
        );
        assert_eq!(
            C220VectorUopStages::floating_binary_arithmetic(C220VecArithmeticOperation::Multiply)
                .unwrap()
                .release_offset(1),
            15
        );
        assert_eq!(
            C220VectorUopStages::floating_binary_arithmetic(C220VecArithmeticOperation::Maximum)
                .unwrap()
                .execute_ticks,
            5
        );
        assert_eq!(
            C220VectorUopStages::floating_binary_arithmetic(C220VecArithmeticOperation::Minimum)
                .unwrap()
                .execute_ticks,
            5
        );
        assert_eq!(
            C220VectorUopStages::floating_binary_arithmetic(C220VecArithmeticOperation::Divide)
                .unwrap()
                .execute_ticks,
            11
        );
        assert_eq!(C220VectorUopStages::relu().execute_ticks, 6);
    }

    #[test]
    fn block_masks_and_bank_groups_determine_writeback() {
        let store = |lane_index, address| C220VectorStore {
            repeat_index: 0,
            lane_index,
            address,
            bank: C220UbBank::from_address(address),
            width_bytes: 4,
            data: [0; 4],
        };
        let full = (0..8)
            .map(|lane| store(lane, lane as u64 * 4))
            .collect::<Vec<_>>();
        let plan = C220VectorWritePlan::from_stores(&full).unwrap();
        assert_eq!(plan.blocks[0].active_lane_mask, 0xff);
        assert_eq!(plan.writeback_ticks(), Some(1));

        let unaligned = (0..16)
            .map(|lane| {
                let block_base = if lane < 8 { 0x1f } else { 0x3f };
                store(lane, block_base + (lane % 8) as u64 * 4)
            })
            .collect::<Vec<_>>();
        let unaligned = C220VectorWritePlan::from_stores(&unaligned).unwrap();
        assert_eq!(unaligned.writeback_ticks(), Some(2));

        let partial = C220VectorWritePlan::from_stores(&[store(0, 0), store(8, 0x20)]).unwrap();
        assert_eq!(partial.blocks.len(), 2);
        assert_eq!(partial.writeback_ticks(), Some(7));

        let contended = C220VectorWritePlan::from_stores(&[store(0, 0), store(8, 0x200)]).unwrap();
        assert_eq!(
            contended.blocks[0].bank.group,
            contended.blocks[1].bank.group
        );
        assert_eq!(contended.writeback_ticks(), Some(8));

        let mut second_repeat = store(0, 0x100);
        second_repeat.repeat_index = 1;
        let repeated = C220VectorWritePlan::from_stores(&[store(0, 0), second_repeat]).unwrap();
        assert_eq!(repeated.blocks.len(), 2);
        assert_eq!(repeated.writeback_ticks(), None);
        assert_eq!(repeated.writeback_ticks_for_repeat(0), Some(7));
        assert_eq!(repeated.writeback_ticks_for_repeat(1), Some(7));
    }

    #[test]
    fn same_group_writes_serialize_and_partial_slots_delay_later_writes() {
        let make_stores = |partial_count: usize| {
            (0..7)
                .flat_map(|block| {
                    let width = if block < partial_count { 1 } else { 8 };
                    (0..width).map(move |lane| {
                        let address = block as u64 * 0x200 + lane as u64 * 4;
                        C220VectorStore {
                            repeat_index: 0,
                            lane_index: block * 8 + lane,
                            address,
                            bank: C220UbBank::from_address(address),
                            width_bytes: 4,
                            data: [0; 4],
                        }
                    })
                })
                .collect::<Vec<_>>()
        };
        let full = C220VectorWritePlan::from_stores(&make_stores(0)).unwrap();
        assert_eq!(full.writeback_ticks(), Some(7));

        let partial = C220VectorWritePlan::from_stores(&make_stores(6)).unwrap();
        assert_eq!(partial.writeback_ticks(), Some(13));
    }
}
