use crate::memory::hbm_pv_memory::{HbmPvMemory, HbmPvMemoryError};
use crate::memory::mapped::{MappedMemory, MappedMemoryError};
use crate::sim::common::scalar::ScalarMemoryBus;

impl ScalarMemoryBus for MappedMemory {
    type Error = MappedMemoryError;

    fn read(&mut self, address: u64, destination: &mut [u8]) -> Result<(), Self::Error> {
        let bytes = self.read_known_at(address, destination.len())?;
        destination.copy_from_slice(&bytes);
        Ok(())
    }

    fn write(&mut self, address: u64, source: &[u8]) -> Result<(), Self::Error> {
        self.write_known_at(address, source)
    }
}

impl ScalarMemoryBus for HbmPvMemory {
    type Error = HbmPvMemoryError;

    fn read(&mut self, address: u64, destination: &mut [u8]) -> Result<(), Self::Error> {
        if destination.is_empty() {
            return Ok(());
        }
        self.device_to_host(address, destination)
    }

    fn write(&mut self, address: u64, source: &[u8]) -> Result<(), Self::Error> {
        if source.is_empty() {
            return Ok(());
        }
        self.host_to_device(address, source)
    }
}
