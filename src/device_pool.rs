use std::collections::BTreeMap;

use serde::Serialize;
use thiserror::Error;

use crate::architecture::Architecture;
use crate::binary_alloc::BINARY_POOL_BYTES;
use crate::hbm_pv_memory::{HbmPvMemory, HbmPvMemoryError};

const RETAINED_IDLE_POOLS: usize = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct DevicePoolBlock {
    pub address: u64,
    pub bytes: u64,
    pub backing_address: u64,
    pub read_only: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct DevicePoolSummary {
    pub backing_address: u64,
    pub read_only: bool,
    pub used_bytes: u64,
    pub free_block_count: usize,
    pub free_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum DevicePoolError {
    #[error("pool manager and memory target different architectures")]
    ArchitectureMismatch,
    #[error("pool block was not allocated by this manager")]
    UnknownBlock,
    #[error(transparent)]
    Memory(#[from] HbmPvMemoryError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FreeBlock {
    address: u64,
    bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Pool {
    backing_address: u64,
    read_only: bool,
    used_bytes: u64,
    free_blocks: Vec<FreeBlock>,
    allocated: BTreeMap<u64, u64>,
}

impl Pool {
    fn new(backing_address: u64, read_only: bool) -> Self {
        Self {
            backing_address,
            read_only,
            used_bytes: 0,
            free_blocks: vec![FreeBlock {
                address: backing_address,
                bytes: BINARY_POOL_BYTES,
            }],
            allocated: BTreeMap::new(),
        }
    }

    fn allocate(&mut self, bytes: u64) -> Option<DevicePoolBlock> {
        let index = self
            .free_blocks
            .iter()
            .position(|block| block.bytes >= bytes)?;
        let address = self.free_blocks[index].address;
        if self.free_blocks[index].bytes == bytes {
            self.free_blocks.remove(index);
        } else {
            self.free_blocks[index].address += bytes;
            self.free_blocks[index].bytes -= bytes;
        }
        self.used_bytes += bytes;
        self.allocated.insert(address, bytes);
        Some(DevicePoolBlock {
            address,
            bytes,
            backing_address: self.backing_address,
            read_only: self.read_only,
        })
    }

    fn summary(&self) -> DevicePoolSummary {
        DevicePoolSummary {
            backing_address: self.backing_address,
            read_only: self.read_only,
            used_bytes: self.used_bytes,
            free_block_count: self.free_blocks.len(),
            free_bytes: self.free_blocks.iter().map(|block| block.bytes).sum(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceMemoryPoolManager {
    architecture: Architecture,
    pools: Vec<Pool>,
}

impl DeviceMemoryPoolManager {
    pub fn new(architecture: Architecture) -> Self {
        Self {
            architecture,
            pools: Vec::new(),
        }
    }

    pub fn pool_summaries(&self) -> Vec<DevicePoolSummary> {
        self.pools.iter().map(Pool::summary).collect()
    }

    pub fn allocate(
        &mut self,
        memory: &mut HbmPvMemory,
        bytes: u64,
        read_only: bool,
    ) -> Result<Option<DevicePoolBlock>, DevicePoolError> {
        self.check_architecture(memory)?;
        if bytes == 0 || bytes > BINARY_POOL_BYTES {
            return Ok(None);
        }
        for pool in &mut self.pools {
            if pool.read_only == read_only
                && let Some(block) = pool.allocate(bytes)
            {
                return Ok(Some(block));
            }
        }
        let backing_address = memory.allocate_driver_request(BINARY_POOL_BYTES)?;
        let mut pool = Pool::new(backing_address, read_only);
        let block = pool.allocate(bytes);
        self.pools.push(pool);
        Ok(block)
    }

    pub fn release(
        &mut self,
        memory: &mut HbmPvMemory,
        block: DevicePoolBlock,
    ) -> Result<(), DevicePoolError> {
        self.check_architecture(memory)?;
        let pool = self
            .pools
            .iter_mut()
            .find(|pool| pool.backing_address == block.backing_address)
            .ok_or(DevicePoolError::UnknownBlock)?;
        if pool.read_only != block.read_only
            || pool.allocated.get(&block.address) != Some(&block.bytes)
        {
            return Err(DevicePoolError::UnknownBlock);
        }
        pool.allocated.remove(&block.address);
        pool.used_bytes -= block.bytes;
        pool.free_blocks.push(FreeBlock {
            address: block.address,
            bytes: block.bytes,
        });
        self.reap_idle_pools(memory)?;
        Ok(())
    }

    fn check_architecture(&self, memory: &HbmPvMemory) -> Result<(), DevicePoolError> {
        if memory.allocator().architecture() != self.architecture {
            return Err(DevicePoolError::ArchitectureMismatch);
        }
        Ok(())
    }

    fn reap_idle_pools(&mut self, memory: &mut HbmPvMemory) -> Result<(), DevicePoolError> {
        let mut idle_count = self
            .pools
            .iter()
            .filter(|pool| pool.used_bytes == 0)
            .count();
        let mut index = 0;
        while idle_count > RETAINED_IDLE_POOLS && index < self.pools.len() {
            if self.pools[index].used_bytes == 0 {
                memory.free(self.pools[index].backing_address)?;
                self.pools.remove(index);
                idle_count -= 1;
            } else {
                index += 1;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hbm::CAMODEL_HBM_BASE;

    #[test]
    fn first_fit_splits_blocks_and_separates_read_only_pools() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut memory = HbmPvMemory::new(architecture, 0, 1);
            let mut pools = DeviceMemoryPoolManager::new(architecture);
            assert_eq!(pools.allocate(&mut memory, 0, false).unwrap(), None);
            assert_eq!(
                pools
                    .allocate(&mut memory, BINARY_POOL_BYTES + 1, false)
                    .unwrap(),
                None
            );
            assert!(pools.pool_summaries().is_empty());
            let first = pools.allocate(&mut memory, 4096, false).unwrap().unwrap();
            let second = pools.allocate(&mut memory, 8192, false).unwrap().unwrap();
            let read_only = pools.allocate(&mut memory, 4096, true).unwrap().unwrap();
            assert_eq!(first.address, CAMODEL_HBM_BASE);
            assert_eq!(second.address, CAMODEL_HBM_BASE + 4096);
            assert_eq!(read_only.address, CAMODEL_HBM_BASE + BINARY_POOL_BYTES);
            assert_eq!(pools.pool_summaries().len(), 2);
            assert_eq!(pools.pool_summaries()[0].used_bytes, 12288);
        }
    }

    #[test]
    fn released_blocks_append_without_coalescing() {
        let mut memory = HbmPvMemory::new(Architecture::Dav2201, 0, 1);
        let mut pools = DeviceMemoryPoolManager::new(Architecture::Dav2201);
        let first = pools
            .allocate(&mut memory, BINARY_POOL_BYTES / 2, false)
            .unwrap()
            .unwrap();
        let second = pools
            .allocate(&mut memory, BINARY_POOL_BYTES / 2, false)
            .unwrap()
            .unwrap();
        pools.release(&mut memory, second).unwrap();
        pools.release(&mut memory, first).unwrap();
        assert_eq!(pools.pool_summaries()[0].free_block_count, 2);
        let next = pools
            .allocate(&mut memory, BINARY_POOL_BYTES / 2, false)
            .unwrap()
            .unwrap();
        assert_eq!(next.address, second.address);
        assert_eq!(
            pools.release(&mut memory, first),
            Err(DevicePoolError::UnknownBlock)
        );
    }

    #[test]
    fn retains_only_five_idle_pools() {
        let mut memory = HbmPvMemory::new(Architecture::Dav3510, 0, 1);
        let mut pools = DeviceMemoryPoolManager::new(Architecture::Dav3510);
        let blocks: Vec<_> = (0..6)
            .map(|_| {
                pools
                    .allocate(&mut memory, BINARY_POOL_BYTES, false)
                    .unwrap()
                    .unwrap()
            })
            .collect();
        assert_eq!(pools.pool_summaries().len(), 6);
        for block in blocks {
            pools.release(&mut memory, block).unwrap();
        }
        assert_eq!(pools.pool_summaries().len(), 5);
        assert_eq!(
            pools.pool_summaries()[0].backing_address,
            CAMODEL_HBM_BASE + BINARY_POOL_BYTES
        );
    }

    #[test]
    fn rejects_a_memory_instance_for_another_architecture() {
        let mut memory = HbmPvMemory::new(Architecture::Dav3510, 0, 1);
        let mut pools = DeviceMemoryPoolManager::new(Architecture::Dav2201);
        assert_eq!(
            pools.allocate(&mut memory, 4096, false),
            Err(DevicePoolError::ArchitectureMismatch)
        );
        assert!(!memory.allocator().spans()[0].allocated);
    }
}
