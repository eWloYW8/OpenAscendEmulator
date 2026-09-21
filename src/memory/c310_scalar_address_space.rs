use crate::device::architecture::Architecture;
use crate::execution::machine::{ScalarMachine, ScalarMemoryBus};
use crate::execution::stepper::LoadedScalarMemoryBus;
use crate::memory::hbm_pv_memory::{HbmPvMemory, HbmPvMemoryError};
use crate::memory::pv_memory::{PvMemory, PvMemoryError};
use thiserror::Error;

const ADDRESS_ROOT_MASK: u64 = 0x1fffffe000000;
const BIU_ADDRESS_MASK: u64 = 0xffffffffffff;
const LOCAL_OFFSET_MASK: u64 = 0x7ffff;
pub(crate) const C310_UB_ROUTE_BYTES: u64 = 0x40000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum C310ScalarRoute {
    Hbm(u64),
    Ub(u64),
    Unsupported,
}

pub(crate) fn classify_c310_scalar_address(
    address: u64,
    spr67: u64,
    spr68: u64,
) -> C310ScalarRoute {
    let root = address & ADDRESS_ROOT_MASK;
    let spr67_root = spr67 & ADDRESS_ROOT_MASK;
    let spr68_root = spr68 & ADDRESS_ROOT_MASK;
    if root != spr67_root || address & 0x1000000 != 0 {
        return C310ScalarRoute::Hbm(address & BIU_ADDRESS_MASK);
    }
    let bank = (address >> 20) & 0x1f;
    if bank == 0 && address & 0x80000 != 0 {
        let offset = address & LOCAL_OFFSET_MASK;
        return if offset < C310_UB_ROUTE_BYTES {
            C310ScalarRoute::Ub(offset)
        } else {
            C310ScalarRoute::Unsupported
        };
    }
    if bank & 0xf == 0 {
        return C310ScalarRoute::Unsupported;
    }
    if spr67_root != spr68_root {
        return (address & 0xffffff)
            .checked_sub(0x100000)
            .and_then(|offset| (spr68 & BIU_ADDRESS_MASK).checked_add(offset))
            .map_or(C310ScalarRoute::Unsupported, C310ScalarRoute::Hbm);
    }
    let offset = address & LOCAL_OFFSET_MASK;
    if offset < C310_UB_ROUTE_BYTES {
        C310ScalarRoute::Ub(offset)
    } else {
        C310ScalarRoute::Unsupported
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum C310ScalarAddressSpaceError {
    #[error("C310 scalar address space requires a dav_3510 HBM store")]
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C310ScalarAddressSpace {
    hbm: HbmPvMemory,
    ub: PvMemory,
    spr67: u64,
    spr68: u64,
}

impl C310ScalarAddressSpace {
    pub fn new(
        hbm: HbmPvMemory,
        spr67: u64,
        spr68: u64,
    ) -> Result<Self, C310ScalarAddressSpaceError> {
        if hbm.allocator().architecture() != Architecture::Dav3510 {
            return Err(C310ScalarAddressSpaceError::WrongArchitecture);
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
    ) -> Result<C310ScalarRoute, C310ScalarAddressSpaceError> {
        let length = u64::try_from(length)
            .map_err(|_| C310ScalarAddressSpaceError::RangeOverflow { address })?;
        let end = address
            .checked_add(length)
            .ok_or(C310ScalarAddressSpaceError::RangeOverflow { address })?;
        let start = classify_c310_scalar_address(address, self.spr67, self.spr68);
        let last = classify_c310_scalar_address(end - 1, self.spr67, self.spr68);
        match (start, last) {
            (C310ScalarRoute::Hbm(mapped), C310ScalarRoute::Hbm(last_mapped))
                if mapped.checked_add(length - 1) == Some(last_mapped) =>
            {
                Ok(C310ScalarRoute::Hbm(mapped))
            }
            (C310ScalarRoute::Ub(offset), C310ScalarRoute::Ub(_))
                if length <= C310_UB_ROUTE_BYTES.saturating_sub(offset) =>
            {
                Ok(C310ScalarRoute::Ub(offset))
            }
            _ => Err(C310ScalarAddressSpaceError::UnsupportedLocal { address, end }),
        }
    }
}

impl ScalarMemoryBus for C310ScalarAddressSpace {
    type Error = C310ScalarAddressSpaceError;

    fn read(&mut self, address: u64, destination: &mut [u8]) -> Result<(), Self::Error> {
        if destination.is_empty() {
            return Ok(());
        }
        match self.route(address, destination.len())? {
            C310ScalarRoute::Hbm(mapped) => {
                ScalarMemoryBus::read(&mut self.hbm, mapped, destination)?
            }
            C310ScalarRoute::Ub(offset) => self.ub.read_into(offset, destination)?,
            C310ScalarRoute::Unsupported => unreachable!(),
        }
        Ok(())
    }

    fn write(&mut self, address: u64, source: &[u8]) -> Result<(), Self::Error> {
        if source.is_empty() {
            return Ok(());
        }
        match self.route(address, source.len())? {
            C310ScalarRoute::Hbm(mapped) => ScalarMemoryBus::write(&mut self.hbm, mapped, source)?,
            C310ScalarRoute::Ub(offset) => self.ub.write(offset, source)?,
            C310ScalarRoute::Unsupported => unreachable!(),
        }
        Ok(())
    }
}

impl LoadedScalarMemoryBus for C310ScalarAddressSpace {
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
    fn canonical_and_alias_addresses_share_ub_bytes() {
        let hbm = HbmPvMemory::new(Architecture::Dav3510, 0, 2);
        let mut memory = C310ScalarAddressSpace::new(hbm, 0, 0).unwrap();
        memory.write(0x107f40, &[1, 2]).unwrap();
        let mut actual = [0; 2];
        memory.read(0x87f40, &mut actual).unwrap();
        assert_eq!(actual, [1, 2]);
        assert_eq!(memory.ub().dirty_byte(0x7f40), Some(1));
        assert_eq!(memory.hbm().store().dirty_byte(0x107f40), None);
    }

    #[test]
    fn rejects_unsupported_banks_and_local_boundaries() {
        let hbm = HbmPvMemory::new(Architecture::Dav3510, 0, 2);
        let mut memory = C310ScalarAddressSpace::new(hbm, 0, 0).unwrap();
        for address in [0x40000, 0x140000, 0x80000 + 0x40000] {
            assert!(matches!(
                memory.write(address, &[1]),
                Err(C310ScalarAddressSpaceError::UnsupportedLocal { .. })
            ));
        }
        assert!(matches!(
            memory.write(0x13ffff, &[1, 2]),
            Err(C310ScalarAddressSpaceError::UnsupportedLocal { .. })
        ));
    }

    #[test]
    fn root_mismatch_forwards_to_hbm_and_local_remap_uses_spr68() {
        let mut hbm = HbmPvMemory::new(Architecture::Dav3510, 0, 2);
        let pointer = hbm.allocate(16).unwrap();
        let mut memory = C310ScalarAddressSpace::new(hbm, 0, 0).unwrap();
        memory.write(pointer, &[5]).unwrap();
        let mut actual = [0];
        memory.read(pointer, &mut actual).unwrap();
        assert_eq!(actual, [5]);
        memory.set_local_roots(0, pointer);
        memory.write(0x100000, &[7]).unwrap();
        memory.read(pointer, &mut actual).unwrap();
        assert_eq!(actual, [7]);
        assert_eq!(memory.ub().dirty_byte(0), None);
    }

    #[test]
    fn requires_c310_hbm() {
        let hbm = HbmPvMemory::new(Architecture::Dav2201, 0, 1);
        assert_eq!(
            C310ScalarAddressSpace::new(hbm, 0, 0),
            Err(C310ScalarAddressSpaceError::WrongArchitecture)
        );
    }
}
