use thiserror::Error;

use crate::architecture::Architecture;
use crate::hbm::{CamodelHbmAllocator, HbmAllocationError};
use crate::pv_memory::{PvMemory, PvMemoryError};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum HbmPvMemoryError {
    #[error(transparent)]
    Allocation(#[from] HbmAllocationError),
    #[error(transparent)]
    Store(#[from] PvMemoryError),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HbmPvMemory {
    allocator: CamodelHbmAllocator,
    store: PvMemory,
}

impl HbmPvMemory {
    pub fn new(architecture: Architecture, default_byte: u8, max_pages: usize) -> Self {
        Self {
            allocator: CamodelHbmAllocator::new(architecture),
            store: PvMemory::new(architecture, default_byte, max_pages),
        }
    }

    pub fn allocator(&self) -> &CamodelHbmAllocator {
        &self.allocator
    }

    pub fn store(&self) -> &PvMemory {
        &self.store
    }

    pub fn allocate(&mut self, bytes: u64) -> Result<u64, HbmPvMemoryError> {
        Ok(self.allocator.allocate(bytes)?)
    }

    pub fn allocate_driver_request(&mut self, bytes: u64) -> Result<u64, HbmPvMemoryError> {
        Ok(self.allocator.allocate_driver_request(bytes)?)
    }

    pub fn free(&mut self, pointer: u64) -> Result<(), HbmPvMemoryError> {
        Ok(self.allocator.free(pointer)?)
    }

    pub fn host_to_device(
        &mut self,
        device_address: u64,
        source: &[u8],
    ) -> Result<(), HbmPvMemoryError> {
        self.store.write(device_address, source)?;
        Ok(())
    }

    pub fn device_to_host(
        &mut self,
        device_address: u64,
        destination: &mut [u8],
    ) -> Result<(), HbmPvMemoryError> {
        self.store.read_into(device_address, destination)?;
        Ok(())
    }

    pub fn device_to_device(
        &mut self,
        destination_address: u64,
        source_address: u64,
        length: usize,
    ) -> Result<(), HbmPvMemoryError> {
        self.store
            .copy(destination_address, source_address, length)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hbm::CAMODEL_HBM_BASE;

    #[test]
    fn host_copy_uses_absolute_hbm_address_on_both_architectures() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut memory = HbmPvMemory::new(architecture, 0xa5, 1);
            let pointer = memory.allocate(16).unwrap();
            assert_eq!(pointer, CAMODEL_HBM_BASE);
            memory.host_to_device(pointer + 3, &[1, 2, 3]).unwrap();
            let mut result = [0; 5];
            memory.device_to_host(pointer + 2, &mut result).unwrap();
            assert_eq!(result, [0xa5, 1, 2, 3, 0xa5]);
            assert_eq!(memory.store().dirty_byte(pointer + 2), Some(0));
            assert_eq!(memory.store().dirty_byte(pointer + 3), Some(1));
        }
    }

    #[test]
    fn missing_direct_read_preserves_architecture_specific_page_effect() {
        for (architecture, expected_pages) in
            [(Architecture::Dav2201, 1), (Architecture::Dav3510, 0)]
        {
            let mut memory = HbmPvMemory::new(architecture, 0x7f, 1);
            let pointer = memory.allocate(4).unwrap();
            let mut bytes = [0; 4];
            memory.device_to_host(pointer, &mut bytes).unwrap();
            assert_eq!(bytes, [0x7f; 4]);
            assert_eq!(memory.store().page_count(), expected_pages);
        }
    }

    #[test]
    fn host_copies_follow_vendor_bytes_outside_live_allocations() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut memory = HbmPvMemory::new(architecture, 0, 1);
            let first = memory.allocate_driver_request(64).unwrap();
            let second = memory.allocate_driver_request(64).unwrap();
            assert_eq!(second, first + 512);
            let pattern: [u8; 16] = [
                0x01, 0x80, 0x00, 0xff, 0x5a, 0xa5, 0x12, 0x34, 0xde, 0xad, 0xbe, 0xef, 0x7f, 0x20,
                0x09, 0xc3,
            ];
            memory.host_to_device(first + 3, &pattern).unwrap();
            memory.device_to_device(second + 5, first + 3, 16).unwrap();
            let mut result = [0; 16];
            memory.device_to_host(second + 5, &mut result).unwrap();
            assert_eq!(result, pattern);

            memory.free(second).unwrap();
            memory.device_to_host(second + 5, &mut result).unwrap();
            assert_eq!(result, pattern);
            result.fill(0x55);
            memory.device_to_host(second + 512, &mut result).unwrap();
            assert_eq!(result, [0; 16]);
            memory.host_to_device(second + 5, &pattern).unwrap();
            memory.device_to_host(second + 5, &mut result).unwrap();
            assert_eq!(result, pattern);
            memory.host_to_device(second + 512, &pattern).unwrap();
            memory.device_to_host(second + 512, &mut result).unwrap();
            assert_eq!(result, pattern);
        }
    }

    #[test]
    fn free_and_reallocate_do_not_clear_pem_bytes() {
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
