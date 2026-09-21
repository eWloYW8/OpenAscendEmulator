use crate::memory::sparse::{MemoryByteState, SparseMemory, SparseMemoryError};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryMapping {
    pub region_index: usize,
    pub base: u64,
    pub allocation_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedAddress {
    pub region_index: usize,
    pub offset: u64,
}

pub struct MappedMemory {
    memory: SparseMemory,
    bindings: Vec<MemoryMapping>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum MemoryMappingError {
    #[error("expected {expected} region bases, got {actual}")]
    BaseCount { expected: usize, actual: usize },
    #[error("region {region} has a null base address")]
    NullBase { region: usize },
    #[error("region {region} has a zero-byte allocation")]
    ZeroSize { region: usize },
    #[error("region {region} at [{base:#x}, +{bytes}) overflows u64")]
    AddressOverflow {
        region: usize,
        base: u64,
        bytes: u64,
    },
    #[error("mapped regions {first} and {second} overlap")]
    OverlappingRegions { first: usize, second: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum MappedMemoryError {
    #[error("zero-length address transfer has no region identity")]
    ZeroLength,
    #[error("address transfer length cannot fit in u64")]
    LengthOverflow,
    #[error("address transfer at {address:#x} overflows u64")]
    RangeOverflow { address: u64 },
    #[error("address range [{address:#x}, {end:#x}) is outside one bound region")]
    Unmapped { address: u64, end: u64 },
    #[error(transparent)]
    Memory(#[from] SparseMemoryError),
}

impl MappedMemory {
    pub fn bind(memory: SparseMemory, region_bases: &[u64]) -> Result<Self, MemoryMappingError> {
        let regions = memory.regions();
        if region_bases.len() != regions.len() {
            return Err(MemoryMappingError::BaseCount {
                expected: regions.len(),
                actual: region_bases.len(),
            });
        }
        let mut bindings = Vec::with_capacity(regions.len());
        for (region_index, (region, &base)) in regions.iter().zip(region_bases).enumerate() {
            if base == 0 {
                return Err(MemoryMappingError::NullBase {
                    region: region_index,
                });
            }
            let allocation_bytes = region.allocation_bytes();
            if allocation_bytes == 0 {
                return Err(MemoryMappingError::ZeroSize {
                    region: region_index,
                });
            }
            base.checked_add(allocation_bytes)
                .ok_or(MemoryMappingError::AddressOverflow {
                    region: region_index,
                    base,
                    bytes: allocation_bytes,
                })?;
            bindings.push(MemoryMapping {
                region_index,
                base,
                allocation_bytes,
            });
        }
        bindings.sort_unstable_by_key(|binding| binding.base);
        for pair in bindings.windows(2) {
            let end = pair[0].base + pair[0].allocation_bytes;
            if end > pair[1].base {
                return Err(MemoryMappingError::OverlappingRegions {
                    first: pair[0].region_index,
                    second: pair[1].region_index,
                });
            }
        }
        Ok(Self { memory, bindings })
    }

    pub fn bindings(&self) -> &[MemoryMapping] {
        &self.bindings
    }

    pub fn memory(&self) -> &SparseMemory {
        &self.memory
    }

    pub fn into_memory(self) -> SparseMemory {
        self.memory
    }

    pub fn resolve(&self, address: u64, len: usize) -> Result<ResolvedAddress, MappedMemoryError> {
        if len == 0 {
            return Err(MappedMemoryError::ZeroLength);
        }
        let bytes = u64::try_from(len).map_err(|_| MappedMemoryError::LengthOverflow)?;
        let end = address
            .checked_add(bytes)
            .ok_or(MappedMemoryError::RangeOverflow { address })?;
        self.bindings
            .iter()
            .find(|binding| {
                address >= binding.base && end <= binding.base + binding.allocation_bytes
            })
            .map(|binding| ResolvedAddress {
                region_index: binding.region_index,
                offset: address - binding.base,
            })
            .ok_or(MappedMemoryError::Unmapped { address, end })
    }

    pub fn read_states_at(
        &self,
        address: u64,
        len: usize,
    ) -> Result<Vec<MemoryByteState>, MappedMemoryError> {
        let location = self.resolve(address, len)?;
        Ok(self
            .memory
            .read_states(location.region_index, location.offset, len)?)
    }

    pub fn read_known_at(&self, address: u64, len: usize) -> Result<Vec<u8>, MappedMemoryError> {
        let location = self.resolve(address, len)?;
        Ok(self
            .memory
            .read_known(location.region_index, location.offset, len)?)
    }

    pub fn write_known_at(&mut self, address: u64, bytes: &[u8]) -> Result<(), MappedMemoryError> {
        let location = self.resolve(address, bytes.len())?;
        Ok(self
            .memory
            .write_known(location.region_index, location.offset, bytes)?)
    }

    pub fn write_states_at(
        &mut self,
        address: u64,
        states: &[MemoryByteState],
    ) -> Result<(), MappedMemoryError> {
        let location = self.resolve(address, states.len())?;
        Ok(self
            .memory
            .write_states(location.region_index, location.offset, states)?)
    }

    pub fn write_segments_at(
        &mut self,
        segments: &[(u64, Vec<MemoryByteState>)],
    ) -> Result<(), MappedMemoryError> {
        let resolved = segments
            .iter()
            .map(|(address, states)| {
                let location = self.resolve(*address, states.len())?;
                Ok((location.region_index, location.offset, states.as_slice()))
            })
            .collect::<Result<Vec<_>, MappedMemoryError>>()?;
        self.memory.write_segments(&resolved)?;
        Ok(())
    }

    pub fn write_unknown_at(&mut self, address: u64, len: usize) -> Result<(), MappedMemoryError> {
        let location = self.resolve(address, len)?;
        Ok(self
            .memory
            .write_unknown(location.region_index, location.offset, len)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::architecture::Architecture;
    use crate::memory::region::MemoryRegion;
    use crate::sim::machine::ScalarMachine;

    fn sub_memory() -> SparseMemory {
        let regions = vec![
            MemoryRegion::new(4096, vec![0x17; 4096]).unwrap(),
            MemoryRegion::new(4096, vec![0x29; 4096]).unwrap(),
            MemoryRegion::unknown(4096),
            MemoryRegion::unknown(0x100_0000),
            MemoryRegion::new(64, 1024_u32.to_le_bytes().to_vec()).unwrap(),
        ];
        SparseMemory::new(regions, 16, 4096)
    }

    #[test]
    fn observed_sub_vectors_resolve_on_both_architectures_and_drive_scalar_store() {
        for (architecture, pointers, word) in [
            (
                Architecture::Dav2201,
                [0x11511c00, 0x11512e00, 0x11514000, 0x11515200, 0x12515400],
                0x04c6_6000,
            ),
            (
                Architecture::Dav3510,
                [0x1b92da00, 0x1b92ec00, 0x1b92fe00, 0x1b931000, 0x1c931200],
                0x03c6_6000,
            ),
        ] {
            let memory = sub_memory();
            let mut addressed = MappedMemory::bind(memory, &pointers).unwrap();
            assert_eq!(addressed.bindings().len(), 5);
            assert_eq!(addressed.read_known_at(pointers[0], 1).unwrap(), [0x17]);
            assert_eq!(addressed.read_known_at(pointers[1], 1).unwrap(), [0x29]);
            assert_eq!(
                addressed.read_known_at(pointers[4], 4).unwrap(),
                1024_u32.to_le_bytes()
            );
            assert_eq!(
                addressed.read_states_at(pointers[2], 1).unwrap(),
                [MemoryByteState::Unknown]
            );
            let mut machine = ScalarMachine::new(architecture, [0; 32], 0);
            machine.set_xreg(6, pointers[2]).unwrap();
            machine.set_xreg(3, 0x0123_4567_89ab_cdef).unwrap();
            machine
                .execute_memory_word(0x100, word, &mut addressed)
                .unwrap();
            assert_eq!(
                addressed.read_known_at(pointers[2], 8).unwrap(),
                0x0123_4567_89ab_cdef_u64.to_le_bytes()
            );
            assert_eq!(
                addressed.read_known_at(pointers[2] + 8, 1),
                Err(MappedMemoryError::Memory(SparseMemoryError::UnknownBytes {
                    region: 2,
                    offset: 8,
                }))
            );
            addressed
                .write_states_at(
                    pointers[2] + 16,
                    &[MemoryByteState::Known(3), MemoryByteState::Unknown],
                )
                .unwrap();
            assert_eq!(addressed.read_known_at(pointers[2] + 16, 1).unwrap(), [3]);
            assert!(matches!(
                addressed.read_known_at(pointers[2] + 17, 1),
                Err(MappedMemoryError::Memory(
                    SparseMemoryError::UnknownBytes { .. }
                ))
            ));
        }
    }

    #[test]
    fn binding_and_address_checks_fail_closed() {
        let memory = sub_memory();
        assert!(matches!(
            MappedMemory::bind(memory, &[1, 2]),
            Err(MemoryMappingError::BaseCount {
                expected: 5,
                actual: 2
            })
        ));
        let memory = sub_memory();
        let mut pointers = [0x10000, 0x10f00, 0x12000, 0x13000, 0x10013000];
        assert!(matches!(
            MappedMemory::bind(memory, &pointers),
            Err(MemoryMappingError::OverlappingRegions {
                first: 0,
                second: 1
            })
        ));
        let memory = sub_memory();
        pointers[1] = 0;
        assert!(matches!(
            MappedMemory::bind(memory, &pointers),
            Err(MemoryMappingError::NullBase { region: 1 })
        ));
        let memory = sub_memory();
        pointers[1] = 0x12000;
        pointers[4] = u64::MAX - 16;
        assert!(matches!(
            MappedMemory::bind(memory, &pointers),
            Err(MemoryMappingError::AddressOverflow { region: 4, .. })
        ));
        let memory = sub_memory();
        pointers = [0x10000, 0x12000, 0x14000, 0x16000, 0x10016000];
        let mut addressed = MappedMemory::bind(memory, &pointers).unwrap();
        assert_eq!(
            addressed.resolve(pointers[0] + 4095, 2),
            Err(MappedMemoryError::Unmapped {
                address: pointers[0] + 4095,
                end: pointers[0] + 4097
            })
        );
        assert_eq!(
            addressed.resolve(pointers[0], 0),
            Err(MappedMemoryError::ZeroLength)
        );
        assert_eq!(
            addressed.resolve(u64::MAX, 2),
            Err(MappedMemoryError::RangeOverflow { address: u64::MAX })
        );
        assert_eq!(
            addressed.write_known_at(pointers[2] + 4095, &[1, 2]),
            Err(MappedMemoryError::Unmapped {
                address: pointers[2] + 4095,
                end: pointers[2] + 4097
            })
        );
        assert_eq!(addressed.memory().overlay_bytes(), 0);
    }
}
