
use crate::replay_seed::{ReplaySeed, SeedArgument};
use serde::Serialize;
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum MemoryByteState {
    Known(u8),
    Unknown,
}

pub struct ReplayMemory {
    seed: ReplaySeed,
    overlays: Vec<BTreeMap<u64, MemoryByteState>>,
    overlay_bytes: usize,
    max_overlay_bytes: usize,
    max_transfer_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum ReplayMemoryError {
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

impl ReplayMemory {
    pub fn new(seed: ReplaySeed, max_overlay_bytes: usize, max_transfer_bytes: usize) -> Self {
        let overlays = (0..seed.regions.len()).map(|_| BTreeMap::new()).collect();
        Self {
            seed,
            overlays,
            overlay_bytes: 0,
            max_overlay_bytes,
            max_transfer_bytes,
        }
    }

    pub fn arguments(&self) -> &[SeedArgument] {
        &self.seed.arguments
    }

    pub fn seed(&self) -> &ReplaySeed {
        &self.seed
    }

    pub const fn overlay_bytes(&self) -> usize {
        self.overlay_bytes
    }

    pub fn read_states(
        &self,
        region: usize,
        offset: u64,
        len: usize,
    ) -> Result<Vec<MemoryByteState>, ReplayMemoryError> {
        self.check_range(region, offset, len)?;
        let mut result = Vec::new();
        result
            .try_reserve_exact(len)
            .map_err(|_| ReplayMemoryError::HostAllocationFailed { requested: len })?;
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
    ) -> Result<Vec<u8>, ReplayMemoryError> {
        self.check_range(region, offset, len)?;
        let mut result = Vec::new();
        result
            .try_reserve_exact(len)
            .map_err(|_| ReplayMemoryError::HostAllocationFailed { requested: len })?;
        for index in 0..len {
            let current = offset + index as u64;
            match self.byte_state(region, current) {
                MemoryByteState::Known(value) => result.push(value),
                MemoryByteState::Unknown => {
                    return Err(ReplayMemoryError::UnknownBytes {
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
    ) -> Result<(), ReplayMemoryError> {
        self.write_with(region, offset, bytes.len(), |index| {
            MemoryByteState::Known(bytes[index])
        })
    }

    pub fn write_states(
        &mut self,
        region: usize,
        offset: u64,
        states: &[MemoryByteState],
    ) -> Result<(), ReplayMemoryError> {
        self.write_with(region, offset, states.len(), |index| states[index])
    }

    pub fn write_unknown(
        &mut self,
        region: usize,
        offset: u64,
        len: usize,
    ) -> Result<(), ReplayMemoryError> {
        self.write_with(region, offset, len, |_| MemoryByteState::Unknown)
    }

    fn byte_state(&self, region: usize, offset: u64) -> MemoryByteState {
        if let Some(&state) = self.overlays[region].get(&offset) {
            return state;
        }
        usize::try_from(offset)
            .ok()
            .and_then(|index| self.seed.regions[region].known_prefix().get(index))
            .copied()
            .map_or(MemoryByteState::Unknown, MemoryByteState::Known)
    }

    fn check_range(&self, region: usize, offset: u64, len: usize) -> Result<(), ReplayMemoryError> {
        let allocation_bytes = self
            .seed
            .regions
            .get(region)
            .ok_or(ReplayMemoryError::InvalidRegion(region))?
            .allocation_bytes();
        if len > self.max_transfer_bytes {
            return Err(ReplayMemoryError::TransferLimitExceeded {
                requested: len,
                limit: self.max_transfer_bytes,
            });
        }
        let len = u64::try_from(len).map_err(|_| ReplayMemoryError::RangeOverflow { region })?;
        let end = offset
            .checked_add(len)
            .ok_or(ReplayMemoryError::RangeOverflow { region })?;
        if end > allocation_bytes {
            return Err(ReplayMemoryError::OutOfBounds {
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
    ) -> Result<(), ReplayMemoryError>
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
            .ok_or(ReplayMemoryError::OverlayLimitExceeded {
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
    use crate::kernel_config::KernelConfigDocument;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

    struct TempTree(PathBuf);

    impl TempTree {
        fn new() -> Self {
            let id = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "open-ascend-emulator-memory-{}-{id}",
                std::process::id()
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TempTree {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn staged_memory() -> (TempTree, ReplayMemory) {
        let tree = TempTree::new();
        let input = tree.0.join("input.bin");
        let tiling = tree.0.join("tiling.bin");
        fs::write(&input, [1_u8, 2, 3]).unwrap();
        fs::write(&tiling, [4_u8, 5]).unwrap();
        let config = KernelConfigDocument::from_slice(
            format!(
                "{{\"old_mode\":\"0\",\"input_path\":\"{}\",\"input_size\":\"3\",\"output_name\":\"out.bin\",\"output_size\":\"4\",\"tiling_data_path\":\"{};2\",\"workspace_size\":\"16\"}}",
                input.display(),
                tiling.display()
            )
            .as_bytes(),
        )
        .unwrap()
        .decode()
        .unwrap();
        let seed = ReplaySeed::load(&config, 5).unwrap();
        (tree, ReplayMemory::new(seed, 4, 4))
    }

    #[test]
    fn initialized_prefix_and_sparse_writes_preserve_unknown_bytes() {
        let (_tree, mut memory) = staged_memory();
        assert_eq!(memory.read_known(0, 0, 3).unwrap(), [1, 2, 3]);
        assert_eq!(memory.arguments().len(), 4);
        assert_eq!(
            memory.read_states(1, 0, 4).unwrap(),
            [MemoryByteState::Unknown; 4]
        );
        assert_eq!(memory.read_known(2, 0, 2).unwrap(), [4, 5]);
        assert_eq!(
            memory.read_known(2, 2, 1),
            Err(ReplayMemoryError::UnknownBytes {
                region: 2,
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
            Err(ReplayMemoryError::UnknownBytes {
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
            Err(ReplayMemoryError::UnknownBytes {
                region: 1,
                offset: 1
            })
        );
        assert_eq!(memory.seed().regions[0].known_prefix(), [1, 2, 3]);
    }

    #[test]
    fn limits_and_bounds_fail_before_partial_mutation() {
        let (_tree, mut memory) = staged_memory();
        memory.write_known(1, 1, &[9, 8]).unwrap();
        assert_eq!(
            memory.write_known(1, 2, &[6, 5, 4]),
            Err(ReplayMemoryError::OutOfBounds {
                region: 1,
                offset: 2,
                end: 5,
                allocation_bytes: 4
            })
        );
        assert_eq!(memory.read_known(1, 2, 1).unwrap(), [8]);
        assert_eq!(
            memory.write_unknown(1, 0, 5),
            Err(ReplayMemoryError::TransferLimitExceeded {
                requested: 5,
                limit: 4
            })
        );
        memory.write_known(0, 1, &[7, 6]).unwrap();
        assert_eq!(memory.overlay_bytes(), 4);
        assert_eq!(
            memory.write_known(1, 0, &[5, 4]),
            Err(ReplayMemoryError::OverlayLimitExceeded { limit: 4 })
        );
        assert_eq!(memory.read_known(1, 1, 1).unwrap(), [9]);
        assert_eq!(
            memory.read_known(1, 0, 1),
            Err(ReplayMemoryError::UnknownBytes {
                region: 1,
                offset: 0
            })
        );
        assert_eq!(memory.read_states(1, 4, 0).unwrap(), []);
        assert_eq!(
            memory.read_states(1, u64::MAX, 2),
            Err(ReplayMemoryError::RangeOverflow { region: 1 })
        );
        assert_eq!(
            memory.write_unknown(10, 0, 1),
            Err(ReplayMemoryError::InvalidRegion(10))
        );
    }
}
