use thiserror::Error;

use crate::device::architecture::Architecture;
use crate::execution::machine::{ScalarMachine, ScalarMemoryBus};
use crate::execution::stepper::LoadedScalarMemoryBus;
use crate::memory::hbm_pv_memory::{HbmPvMemory, HbmPvMemoryError};
use crate::memory::pv_memory::{PvMemory, PvMemoryError};

const ADDRESS_ROOT_MASK: u64 = 0x1fffffe000000;
const UB_OFFSET_MASK: u64 = 0x7ffff;
pub(crate) const C220_UB_BYTES: u64 = 0x30000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum C220ScalarAddressSpaceError {
    #[error("C220 scalar address space requires a dav_2201 HBM store")]
    WrongArchitecture,
    #[error("scalar memory range beginning at {address:#x} overflows")]
    RangeOverflow { address: u64 },
    #[error("scalar memory range [{address:#x}, {end:#x}) is outside supported local buffers")]
    UnsupportedLocal { address: u64, end: u64 },
    #[error(transparent)]
    Hbm(#[from] HbmPvMemoryError),
    #[error(transparent)]
    Ub(#[from] PvMemoryError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum C220ScalarRoute {
    Hbm,
    Ub(u64),
    Unsupported,
}

pub(crate) fn classify_c220_scalar_address(
    address: u64,
    spr67: u64,
    spr68: u64,
) -> C220ScalarRoute {
    let root = address & ADDRESS_ROOT_MASK;
    if address & 0x1000000 != 0 || root != spr67 & ADDRESS_ROOT_MASK {
        return C220ScalarRoute::Hbm;
    }
    let bank = (address >> 20) & 0x1f;
    if bank == 0 && address & 0x80000 != 0 {
        return C220ScalarRoute::Ub(address & UB_OFFSET_MASK);
    }
    if bank & 0xf == 0 {
        return C220ScalarRoute::Unsupported;
    }
    if (spr67 & ADDRESS_ROOT_MASK) != (spr68 & ADDRESS_ROOT_MASK) {
        return C220ScalarRoute::Hbm;
    }
    let offset = address & UB_OFFSET_MASK;
    if offset < C220_UB_BYTES {
        C220ScalarRoute::Ub(offset)
    } else {
        C220ScalarRoute::Unsupported
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220ScalarAddressSpace {
    hbm: HbmPvMemory,
    ub: PvMemory,
    spr67: u64,
    spr68: u64,
}

impl C220ScalarAddressSpace {
    pub fn new(
        hbm: HbmPvMemory,
        spr67: u64,
        spr68: u64,
    ) -> Result<Self, C220ScalarAddressSpaceError> {
        if hbm.allocator().architecture() != Architecture::Dav2201 {
            return Err(C220ScalarAddressSpaceError::WrongArchitecture);
        }
        Ok(Self {
            hbm,
            ub: PvMemory::new(0, 1),
            spr67,
            spr68,
        })
    }

    pub fn hbm(&self) -> &HbmPvMemory {
        &self.hbm
    }

    pub fn hbm_mut(&mut self) -> &mut HbmPvMemory {
        &mut self.hbm
    }

    pub fn ub(&self) -> &PvMemory {
        &self.ub
    }

    pub fn local_roots(&self) -> (u64, u64) {
        (self.spr67, self.spr68)
    }

    pub fn set_local_roots(&mut self, spr67: u64, spr68: u64) {
        self.spr67 = spr67;
        self.spr68 = spr68;
    }

    fn route(
        &self,
        address: u64,
        length: usize,
    ) -> Result<C220ScalarRoute, C220ScalarAddressSpaceError> {
        let length = u64::try_from(length)
            .map_err(|_| C220ScalarAddressSpaceError::RangeOverflow { address })?;
        let end = address
            .checked_add(length)
            .ok_or(C220ScalarAddressSpaceError::RangeOverflow { address })?;
        let route = classify_c220_scalar_address(address, self.spr67, self.spr68);
        let end_route = classify_c220_scalar_address(end - 1, self.spr67, self.spr68);
        match (route, end_route) {
            (C220ScalarRoute::Hbm, C220ScalarRoute::Hbm) => Ok(C220ScalarRoute::Hbm),
            (C220ScalarRoute::Ub(offset), C220ScalarRoute::Ub(_))
                if length <= C220_UB_BYTES.saturating_sub(offset) =>
            {
                Ok(C220ScalarRoute::Ub(offset))
            }
            _ => Err(C220ScalarAddressSpaceError::UnsupportedLocal { address, end }),
        }
    }
}

impl ScalarMemoryBus for C220ScalarAddressSpace {
    type Error = C220ScalarAddressSpaceError;

    fn read(&mut self, address: u64, destination: &mut [u8]) -> Result<(), Self::Error> {
        if destination.is_empty() {
            return Ok(());
        }
        match self.route(address, destination.len())? {
            C220ScalarRoute::Hbm => ScalarMemoryBus::read(&mut self.hbm, address, destination)?,
            C220ScalarRoute::Ub(offset) => self.ub.read_into(offset, destination)?,
            C220ScalarRoute::Unsupported => unreachable!(),
        }
        Ok(())
    }

    fn write(&mut self, address: u64, source: &[u8]) -> Result<(), Self::Error> {
        if source.is_empty() {
            return Ok(());
        }
        match self.route(address, source.len())? {
            C220ScalarRoute::Hbm => ScalarMemoryBus::write(&mut self.hbm, address, source)?,
            C220ScalarRoute::Ub(offset) => self.ub.write(offset, source)?,
            C220ScalarRoute::Unsupported => unreachable!(),
        }
        Ok(())
    }
}

impl LoadedScalarMemoryBus for C220ScalarAddressSpace {
    fn code_memory(&mut self) -> &mut HbmPvMemory {
        &mut self.hbm
    }

    fn sync_machine(&mut self, machine: &ScalarMachine) {
        if let (Some(spr67), Some(spr68)) = (machine.spr_value(67), machine.spr_value(68)) {
            self.set_local_roots(spr67, spr68);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_canonical_and_alias_addresses_to_the_same_ub_bytes() {
        let hbm = HbmPvMemory::new(Architecture::Dav2201, 0, 2);
        let mut memory = C220ScalarAddressSpace::new(hbm, 0, 0).unwrap();
        memory.write(0x100000, &[1, 2, 3, 4]).unwrap();
        let mut bytes = [0; 4];
        memory.read(0x80000, &mut bytes).unwrap();
        assert_eq!(bytes, [1, 2, 3, 4]);
        assert_eq!(memory.ub().dirty_byte(0), Some(1));
        assert_eq!(memory.hbm().store().dirty_byte(0x100000), None);
    }

    #[test]
    fn preserves_hbm_allocation_checks_and_rejects_local_boundary_crossing() {
        let mut hbm = HbmPvMemory::new(Architecture::Dav2201, 0, 2);
        let address = hbm.allocate(16).unwrap();
        let mut memory = C220ScalarAddressSpace::new(hbm, 0, 0).unwrap();
        memory.write(address, &[7, 8]).unwrap();
        let mut bytes = [0; 2];
        memory.read(address, &mut bytes).unwrap();
        assert_eq!(bytes, [7, 8]);
        assert!(matches!(
            memory.read(address + 15, &mut bytes),
            Err(C220ScalarAddressSpaceError::Hbm(HbmPvMemoryError::Resolve(
                _
            )))
        ));
        assert!(matches!(
            memory.write(0x12ffff, &[1, 2]),
            Err(C220ScalarAddressSpaceError::UnsupportedLocal { .. })
        ));
        memory.write(0x100000, &[1]).unwrap();
        memory.set_local_roots(0x2000000, 0x2000000);
        assert!(matches!(
            memory.write(0x100000, &[1]),
            Err(C220ScalarAddressSpaceError::Hbm(HbmPvMemoryError::Resolve(
                _
            )))
        ));
        let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
        machine.set_spr_value(67, 0).unwrap();
        LoadedScalarMemoryBus::sync_machine(&mut memory, &machine);
        assert_eq!(memory.local_roots(), (0, 0));
        memory.write(0x100000, &[9]).unwrap();
    }

    #[test]
    fn rejects_other_architectures() {
        let hbm = HbmPvMemory::new(Architecture::Dav3510, 0, 1);
        assert_eq!(
            C220ScalarAddressSpace::new(hbm, 0, 0),
            Err(C220ScalarAddressSpaceError::WrongArchitecture)
        );
    }
}
