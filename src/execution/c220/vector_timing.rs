use serde::Serialize;
use thiserror::Error;

use crate::instruction::vec_c220::{
    C220_VECTOR_BLOCK_BYTES, C220_VECTOR_BLOCK_COUNT, C220VectorStore,
};
use crate::memory::ub_bank_c220::C220UbBank;

const UNCONTENDED_FULL_WRITE_TICKS: u8 = 1;
const UNCONTENDED_PARTIAL_WRITE_TICKS: u8 = 7;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct C220VectorWriteBlock {
    pub block_index: u8,
    pub base_address: u64,
    pub bank: C220UbBank,
    pub element_bytes: u8,
    pub active_lane_mask: u16,
}

impl C220VectorWriteBlock {
    pub const fn active_lanes(self) -> u32 {
        self.active_lane_mask.count_ones()
    }

    pub const fn full(self) -> bool {
        match self.element_bytes {
            2 => self.active_lane_mask == u16::MAX,
            4 => self.active_lane_mask == 0xff,
            _ => false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct C220VectorWritePlan {
    pub blocks: Vec<C220VectorWriteBlock>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum C220VectorWritePlanError {
    #[error("C220 vector store uses unsupported element width {0}")]
    ElementWidth(u8),
    #[error("C220 vector store lane {lane} is outside its tile")]
    LaneOutsideTile { lane: usize },
    #[error("C220 vector stores disagree on the address or width of block {block}")]
    InconsistentBlock { block: usize },
    #[error("C220 vector store address underflows at lane {lane}")]
    AddressUnderflow { lane: usize },
}

impl C220VectorWritePlan {
    pub fn from_stores(stores: &[C220VectorStore]) -> Result<Self, C220VectorWritePlanError> {
        let mut blocks: [Option<C220VectorWriteBlock>; C220_VECTOR_BLOCK_COUNT] =
            [None; C220_VECTOR_BLOCK_COUNT];
        for store in stores {
            let width = usize::from(store.width_bytes);
            if !matches!(width, 2 | 4) {
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
            let slot = &mut blocks[block];
            match slot {
                Some(existing) => {
                    if existing.base_address != base_address
                        || existing.element_bytes != store.width_bytes
                    {
                        return Err(C220VectorWritePlanError::InconsistentBlock { block });
                    }
                    existing.active_lane_mask |= 1 << lane;
                }
                None => {
                    *slot = Some(C220VectorWriteBlock {
                        block_index: block as u8,
                        base_address,
                        bank: C220UbBank::from_address(base_address),
                        element_bytes: store.width_bytes,
                        active_lane_mask: 1 << lane,
                    });
                }
            }
        }
        Ok(Self {
            blocks: blocks.into_iter().flatten().collect(),
        })
    }

    pub fn uncontended_writeback_ticks(&self) -> Option<u8> {
        let mut seen_groups = [false; 16];
        let mut ticks = UNCONTENDED_FULL_WRITE_TICKS;
        for block in &self.blocks {
            let group = usize::from(block.bank.group);
            let seen = seen_groups.get_mut(group)?;
            if *seen {
                return None;
            }
            *seen = true;
            ticks = ticks.max(if block.full() {
                UNCONTENDED_FULL_WRITE_TICKS
            } else {
                UNCONTENDED_PARTIAL_WRITE_TICKS
            });
        }
        Some(ticks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_masks_and_bank_groups_bound_uncontended_writeback() {
        let store = |lane_index, address| C220VectorStore {
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
        assert_eq!(plan.uncontended_writeback_ticks(), Some(1));

        let partial = C220VectorWritePlan::from_stores(&[store(0, 0), store(8, 0x20)]).unwrap();
        assert_eq!(partial.blocks.len(), 2);
        assert_eq!(partial.uncontended_writeback_ticks(), Some(7));

        let contended = C220VectorWritePlan::from_stores(&[store(0, 0), store(8, 0x200)]).unwrap();
        assert_eq!(
            contended.blocks[0].bank.group,
            contended.blocks[1].bank.group
        );
        assert_eq!(contended.uncontended_writeback_ticks(), None);
    }
}
