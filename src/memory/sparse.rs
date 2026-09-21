use crate::memory::region::MemoryRegion;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum MemoryByteState {
    Known(u8),
    Unknown,
}

pub struct SparseMemory {
    regions: Vec<MemoryRegion>,
    overlays: Vec<BTreeMap<u64, MemoryByteState>>,
    overlay_bytes: usize,
    max_overlay_bytes: usize,
    max_transfer_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum SparseMemoryError {
    #[error("region {0} does not exist")]
    InvalidRegion(usize),
    #[error("region {region} transfer range overflows")]
    RangeOverflow { region: usize },
    #[error(
        "region {region} range [{offset}, {end}) exceeds allocation of {allocation_bytes} bytes"
    )]
    OutOfBounds {
        region: usize,
        offset: u64,
        end: u64,
        allocation_bytes: u64,
    },
    #[error("transfer of {requested} bytes exceeds limit of {limit} bytes")]
    TransferLimitExceeded { requested: usize, limit: usize },
    #[error("overlaid byte count would exceed limit of {limit} bytes")]
    OverlayLimitExceeded { limit: usize },
    #[error("cannot reserve {requested} host bytes for a memory read")]
    HostAllocationFailed { requested: usize },
    #[error("region {region} byte at offset {offset} is unknown")]
    UnknownBytes { region: usize, offset: u64 },
}

impl SparseMemory {
    pub fn new(
        regions: Vec<MemoryRegion>,
        max_overlay_bytes: usize,
        max_transfer_bytes: usize,
    ) -> Self {
        let overlays = (0..regions.len()).map(|_| BTreeMap::new()).collect();
        Self {
            regions,
            overlays,
            overlay_bytes: 0,
            max_overlay_bytes,
            max_transfer_bytes,
        }
    }

    pub fn regions(&self) -> &[MemoryRegion] {
        &self.regions
    }

    pub const fn overlay_bytes(&self) -> usize {
        self.overlay_bytes
    }

    pub fn read_states(
        &self,
        region: usize,
        offset: u64,
        len: usize,
    ) -> Result<Vec<MemoryByteState>, SparseMemoryError> {
        self.check_range(region, offset, len)?;
        let mut result = Vec::new();
        result
            .try_reserve_exact(len)
            .map_err(|_| SparseMemoryError::HostAllocationFailed { requested: len })?;
        for index in 0..len {
            result.push(self.byte_state(region, offset + index as u64));
        }
        Ok(result)
    }

    pub fn read_known(
        &self,
        region: usize,
        offset: u64,
        len: usize,
    ) -> Result<Vec<u8>, SparseMemoryError> {
        self.check_range(region, offset, len)?;
        let mut result = Vec::new();
        result
            .try_reserve_exact(len)
            .map_err(|_| SparseMemoryError::HostAllocationFailed { requested: len })?;
        for index in 0..len {
            let current = offset + index as u64;
            match self.byte_state(region, current) {
                MemoryByteState::Known(value) => result.push(value),
                MemoryByteState::Unknown => {
                    return Err(SparseMemoryError::UnknownBytes {
                        region,
                        offset: current,
                    });
                }
            }
        }
        Ok(result)
    }

    pub fn write_known(
        &mut self,
        region: usize,
        offset: u64,
        bytes: &[u8],
    ) -> Result<(), SparseMemoryError> {
        self.write_with(region, offset, bytes.len(), |index| {
            MemoryByteState::Known(bytes[index])
        })
    }

    pub fn write_states(
        &mut self,
        region: usize,
        offset: u64,
        states: &[MemoryByteState],
    ) -> Result<(), SparseMemoryError> {
        self.write_with(region, offset, states.len(), |index| states[index])
    }

    pub fn write_segments(
        &mut self,
        segments: &[(usize, u64, &[MemoryByteState])],
    ) -> Result<(), SparseMemoryError> {
        let mut newly_overlaid = BTreeSet::new();
        for &(region, offset, states) in segments {
            self.check_range(region, offset, states.len())?;
            for index in 0..states.len() {
                let address = offset + index as u64;
                if !self.overlays[region].contains_key(&address) {
                    newly_overlaid.insert((region, address));
                }
            }
        }
        let next_count = self
            .overlay_bytes
            .checked_add(newly_overlaid.len())
            .filter(|count| *count <= self.max_overlay_bytes)
            .ok_or(SparseMemoryError::OverlayLimitExceeded {
                limit: self.max_overlay_bytes,
            })?;
        for &(region, offset, states) in segments {
            for (index, &state) in states.iter().enumerate() {
                self.overlays[region].insert(offset + index as u64, state);
            }
        }
        self.overlay_bytes = next_count;
        Ok(())
    }

    pub fn write_unknown(
        &mut self,
        region: usize,
        offset: u64,
        len: usize,
    ) -> Result<(), SparseMemoryError> {
        self.write_with(region, offset, len, |_| MemoryByteState::Unknown)
    }

    fn byte_state(&self, region: usize, offset: u64) -> MemoryByteState {
        if let Some(&state) = self.overlays[region].get(&offset) {
            return state;
        }
        usize::try_from(offset)
            .ok()
            .and_then(|index| self.regions[region].known_prefix().get(index))
            .copied()
            .map_or(MemoryByteState::Unknown, MemoryByteState::Known)
    }

    fn check_range(&self, region: usize, offset: u64, len: usize) -> Result<(), SparseMemoryError> {
        let allocation_bytes = self
            .regions
            .get(region)
            .ok_or(SparseMemoryError::InvalidRegion(region))?
            .allocation_bytes();
        if len > self.max_transfer_bytes {
            return Err(SparseMemoryError::TransferLimitExceeded {
                requested: len,
                limit: self.max_transfer_bytes,
            });
        }
        let len = u64::try_from(len).map_err(|_| SparseMemoryError::RangeOverflow { region })?;
        let end = offset
            .checked_add(len)
            .ok_or(SparseMemoryError::RangeOverflow { region })?;
        if end > allocation_bytes {
            return Err(SparseMemoryError::OutOfBounds {
                region,
                offset,
                end,
                allocation_bytes,
            });
        }
        Ok(())
    }

    fn write_with<F>(
        &mut self,
        region: usize,
        offset: u64,
        len: usize,
        value: F,
    ) -> Result<(), SparseMemoryError>
    where
        F: Fn(usize) -> MemoryByteState,
    {
        self.check_range(region, offset, len)?;
        let newly_overlaid = (0..len)
            .filter(|&index| !self.overlays[region].contains_key(&(offset + index as u64)))
            .count();
        let next_count = self
            .overlay_bytes
            .checked_add(newly_overlaid)
            .filter(|count| *count <= self.max_overlay_bytes)
            .ok_or(SparseMemoryError::OverlayLimitExceeded {
                limit: self.max_overlay_bytes,
            })?;
        for index in 0..len {
            self.overlays[region].insert(offset + index as u64, value(index));
        }
        self.overlay_bytes = next_count;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::region::MemoryRegion;

    fn staged_memory() -> SparseMemory {
        let regions = vec![
            MemoryRegion::new(3, vec![1, 2, 3]).unwrap(),
            MemoryRegion::unknown(4),
            MemoryRegion::unknown(16),
            MemoryRegion::new(64, vec![4, 5]).unwrap(),
        ];
        SparseMemory::new(regions, 4, 4)
    }

    #[test]
    fn initialized_prefix_and_sparse_writes_preserve_unknown_bytes() {
        let mut memory = staged_memory();
        assert_eq!(memory.read_known(0, 0, 3).unwrap(), [1, 2, 3]);
        assert_eq!(memory.regions().len(), 4);
        assert_eq!(
            memory.read_states(1, 0, 4).unwrap(),
            [MemoryByteState::Unknown; 4]
        );
        assert_eq!(memory.read_known(3, 0, 2).unwrap(), [4, 5]);
        assert_eq!(
            memory.read_known(3, 2, 1),
            Err(SparseMemoryError::UnknownBytes {
                region: 3,
                offset: 2
            })
        );

        memory.write_known(1, 1, &[9, 8]).unwrap();
        assert_eq!(memory.overlay_bytes(), 2);
        assert_eq!(
            memory.read_states(1, 0, 4).unwrap(),
            [
                MemoryByteState::Unknown,
                MemoryByteState::Known(9),
                MemoryByteState::Known(8),
                MemoryByteState::Unknown
            ]
        );
        memory.write_unknown(0, 1, 1).unwrap();
        assert_eq!(
            memory.read_known(0, 0, 3),
            Err(SparseMemoryError::UnknownBytes {
                region: 0,
                offset: 1
            })
        );
        memory
            .write_states(1, 0, &[MemoryByteState::Known(7), MemoryByteState::Unknown])
            .unwrap();
        assert_eq!(memory.overlay_bytes(), 4);
        assert_eq!(memory.read_known(1, 0, 1).unwrap(), [7]);
        assert_eq!(
            memory.read_known(1, 1, 1),
            Err(SparseMemoryError::UnknownBytes {
                region: 1,
                offset: 1
            })
        );
        assert_eq!(memory.regions()[0].known_prefix(), [1, 2, 3]);
    }

    #[test]
    fn limits_and_bounds_fail_before_partial_mutation() {
        let mut memory = staged_memory();
        memory.write_known(1, 1, &[9, 8]).unwrap();
        assert_eq!(
            memory.write_known(1, 2, &[6, 5, 4]),
            Err(SparseMemoryError::OutOfBounds {
                region: 1,
                offset: 2,
                end: 5,
                allocation_bytes: 4
            })
        );
        assert_eq!(memory.read_known(1, 2, 1).unwrap(), [8]);
        assert_eq!(
            memory.write_unknown(1, 0, 5),
            Err(SparseMemoryError::TransferLimitExceeded {
                requested: 5,
                limit: 4
            })
        );
        memory.write_known(0, 1, &[7, 6]).unwrap();
        assert_eq!(memory.overlay_bytes(), 4);
        assert_eq!(
            memory.write_known(1, 0, &[5, 4]),
            Err(SparseMemoryError::OverlayLimitExceeded { limit: 4 })
        );
        assert_eq!(memory.read_known(1, 1, 1).unwrap(), [9]);
        assert_eq!(
            memory.read_known(1, 0, 1),
            Err(SparseMemoryError::UnknownBytes {
                region: 1,
                offset: 0
            })
        );
        assert_eq!(memory.read_states(1, 4, 0).unwrap(), []);
        assert_eq!(
            memory.read_states(1, u64::MAX, 2),
            Err(SparseMemoryError::RangeOverflow { region: 1 })
        );
        assert_eq!(
            memory.write_unknown(10, 0, 1),
            Err(SparseMemoryError::InvalidRegion(10))
        );
    }
}
