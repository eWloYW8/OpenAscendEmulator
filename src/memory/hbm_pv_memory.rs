use thiserror::Error;

use crate::device::architecture::Architecture;
use crate::memory::hbm::{HbmAllocationError, HbmAllocator, HbmResolveError};
use crate::memory::pv_memory::{PvMemory, PvMemoryError};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum HbmPvMemoryError {
    #[error(transparent)]
    Allocation(#[from] HbmAllocationError),
    #[error(transparent)]
    Resolve(#[from] HbmResolveError),
    #[error(transparent)]
    Store(#[from] PvMemoryError),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HbmPvMemory {
    allocator: HbmAllocator,
    store: PvMemory,
}

impl HbmPvMemory {
    pub fn new(architecture: Architecture, default_byte: u8, max_pages: usize) -> Self {
        Self {
            allocator: HbmAllocator::new(architecture),
            store: PvMemory::new(default_byte, max_pages),
        }
    }

    pub fn with_region(
        architecture: Architecture,
        base: u64,
        bytes: u64,
        default_byte: u8,
        max_pages: usize,
    ) -> Result<Self, HbmPvMemoryError> {
        Ok(Self {
            allocator: HbmAllocator::with_region(architecture, base, bytes)?,
            store: PvMemory::new(default_byte, max_pages),
        })
    }

    pub fn allocator(&self) -> &HbmAllocator {
        &self.allocator
    }

    pub fn store(&self) -> &PvMemory {
        &self.store
    }

    pub fn allocate(&mut self, bytes: u64) -> Result<u64, HbmPvMemoryError> {
        Ok(self.allocator.allocate(bytes)?)
    }

    pub fn free(&mut self, pointer: u64) -> Result<(), HbmPvMemoryError> {
        Ok(self.allocator.free(pointer)?)
    }

    pub fn host_to_device(
        &mut self,
        device_address: u64,
        source: &[u8],
    ) -> Result<(), HbmPvMemoryError> {
        if source.is_empty() {
            return Ok(());
        }
        self.allocator
            .resolve_live(device_address, source.len() as u64)?;
        self.store.write(device_address, source)?;
        Ok(())
    }

    pub fn device_to_host(
        &mut self,
        device_address: u64,
        destination: &mut [u8],
    ) -> Result<(), HbmPvMemoryError> {
        if destination.is_empty() {
            return Ok(());
        }
        self.allocator
            .resolve_live(device_address, destination.len() as u64)?;
        self.store.read_into(device_address, destination)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::machine::ScalarMemoryBus;

    #[test]
    fn scalar_bus_uses_one_live_hbm_allocation() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut memory = HbmPvMemory::new(architecture, 0, 1);
            let address = memory.allocate(16).unwrap();
            ScalarMemoryBus::write(&mut memory, address + 4, &[1, 2, 3, 4]).unwrap();
            let mut bytes = [0; 4];
            ScalarMemoryBus::read(&mut memory, address + 4, &mut bytes).unwrap();
            assert_eq!(bytes, [1, 2, 3, 4]);
            assert!(matches!(
                ScalarMemoryBus::read(&mut memory, address + 14, &mut bytes),
                Err(HbmPvMemoryError::Resolve(HbmResolveError::Unmapped { .. }))
            ));
            memory.free(address).unwrap();
            assert!(matches!(
                ScalarMemoryBus::write(&mut memory, address, &[5]),
                Err(HbmPvMemoryError::Resolve(HbmResolveError::Unmapped { .. }))
            ));
            ScalarMemoryBus::read(&mut memory, address, &mut []).unwrap();
        }
    }

    #[test]
    fn host_copy_uses_absolute_hbm_address_on_both_architectures() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut memory = HbmPvMemory::with_region(architecture, 0x4000, 16, 0xa5, 1).unwrap();
            let pointer = memory.allocate(16).unwrap();
            assert_eq!(pointer, 0x4000);
            memory.host_to_device(pointer + 3, &[1, 2, 3]).unwrap();
            let mut result = [0; 5];
            memory.device_to_host(pointer + 2, &mut result).unwrap();
            assert_eq!(result, [0xa5, 1, 2, 3, 0xa5]);
        }
    }

    #[test]
    fn missing_direct_read_does_not_allocate_pages() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut memory = HbmPvMemory::new(architecture, 0x7f, 1);
            let pointer = memory.allocate(4).unwrap();
            let mut bytes = [0; 4];
            memory.device_to_host(pointer, &mut bytes).unwrap();
            assert_eq!(bytes, [0x7f; 4]);
            assert_eq!(memory.store().page_count(), 0);
        }
    }

    #[test]
    fn host_transfers_require_live_allocations() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut memory = HbmPvMemory::new(architecture, 0, 1);
            let first = memory.allocate(64).unwrap();
            let pattern: [u8; 16] = [
                0x01, 0x80, 0x00, 0xff, 0x5a, 0xa5, 0x12, 0x34, 0xde, 0xad, 0xbe, 0xef, 0x7f, 0x20,
                0x09, 0xc3,
            ];
            memory.host_to_device(first + 3, &pattern).unwrap();
            let mut result = [0; 16];
            memory.device_to_host(first + 3, &mut result).unwrap();
            assert_eq!(result, pattern);

            memory.free(first).unwrap();
            assert!(matches!(
                memory.device_to_host(first + 3, &mut result),
                Err(HbmPvMemoryError::Resolve(HbmResolveError::Unmapped { .. }))
            ));
            assert!(matches!(
                memory.host_to_device(first + 3, &pattern),
                Err(HbmPvMemoryError::Resolve(HbmResolveError::Unmapped { .. }))
            ));
        }
    }

    #[test]
    fn free_and_reallocate_preserves_stored_bytes() {
        let mut memory = HbmPvMemory::new(Architecture::Dav2201, 0, 1);
        let pointer = memory.allocate(4).unwrap();
        memory.host_to_device(pointer, &[1, 2, 3, 4]).unwrap();
        memory.free(pointer).unwrap();
        assert_eq!(memory.allocate(4).unwrap(), pointer);
        let mut bytes = [0; 4];
        memory.device_to_host(pointer, &mut bytes).unwrap();
        assert_eq!(bytes, [1, 2, 3, 4]);
    }
}
