use crate::machine::ScalarMemoryBus;
use crate::replay_memory::{MemoryByteState, ReplayMemory, ReplayMemoryError};
use crate::replay_seed::SeedArgument;
use serde::Serialize;
use thiserror::Error;

const POINTER_BYTES: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ReplayAddressBinding {
    pub argument_index: usize,
    pub region_index: usize,
    pub base: u64,
    pub allocation_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ResolvedReplayAddress {
    pub argument_index: usize,
    pub region_index: usize,
    pub offset: u64,
}

pub struct AddressedReplayMemory {
    memory: ReplayMemory,
    bindings: Vec<ReplayAddressBinding>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum ReplayAddressBindingError {
    #[error("expected {expected} observed argument pointers, got {actual}")]
    PointerCount { expected: usize, actual: usize },
    #[error("seed pointer-array length is inconsistent with its argument count")]
    AppendLength,
    #[error("null argument {argument} has nonzero observed pointer {pointer:#x}")]
    NonzeroNull { argument: usize, pointer: u64 },
    #[error("argument {argument} refers to nonexistent region {region}")]
    InvalidRegion { argument: usize, region: usize },
    #[error("argument {argument} to region {region} has a null observed pointer")]
    NullRegion { argument: usize, region: usize },
    #[error("argument {argument} to region {region} has a zero-byte allocation")]
    ZeroSize { argument: usize, region: usize },
    #[error("region {region} is used by more than one argument")]
    DuplicateRegion { region: usize },
    #[error("region {region} has no argument pointer")]
    UnboundRegion { region: usize },
    #[error("argument {argument} allocation [{pointer:#x}, +{bytes}) overflows u64")]
    AddressOverflow {
        argument: usize,
        pointer: u64,
        bytes: u64,
    },
    #[error("observed argument allocations {first} and {second} overlap")]
    OverlappingAllocations { first: usize, second: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum ReplayAddressError {
    #[error("zero-length address transfer has no region identity")]
    ZeroLength,
    #[error("address transfer length cannot fit in u64")]
    LengthOverflow,
    #[error("address transfer at {address:#x} overflows u64")]
    RangeOverflow { address: u64 },
    #[error("address range [{address:#x}, {end:#x}) is outside one bound region")]
    Unmapped { address: u64, end: u64 },
    #[error(transparent)]
    Memory(#[from] ReplayMemoryError),
}

impl AddressedReplayMemory {
    pub fn bind(
        memory: ReplayMemory,
        observed_pointers: &[u64],
    ) -> Result<Self, ReplayAddressBindingError> {
        let seed = memory.seed();
        if observed_pointers.len() != seed.arguments.len() {
            return Err(ReplayAddressBindingError::PointerCount {
                expected: seed.arguments.len(),
                actual: observed_pointers.len(),
            });
        }
        if seed.arguments.len().checked_mul(POINTER_BYTES) != Some(seed.append_bytes) {
            return Err(ReplayAddressBindingError::AppendLength);
        }
        let mut bindings = Vec::new();
        let mut seen = vec![false; seed.regions.len()];
        for (argument_index, (&argument, &pointer)) in
            seed.arguments.iter().zip(observed_pointers).enumerate()
        {
            let SeedArgument::Region(region_index) = argument else {
                if pointer != 0 {
                    return Err(ReplayAddressBindingError::NonzeroNull {
                        argument: argument_index,
                        pointer,
                    });
                }
                continue;
            };
            let region =
                seed.regions
                    .get(region_index)
                    .ok_or(ReplayAddressBindingError::InvalidRegion {
                        argument: argument_index,
                        region: region_index,
                    })?;
            if pointer == 0 {
                return Err(ReplayAddressBindingError::NullRegion {
                    argument: argument_index,
                    region: region_index,
                });
            }
            let allocation_bytes = region.allocation_bytes();
            if allocation_bytes == 0 {
                return Err(ReplayAddressBindingError::ZeroSize {
                    argument: argument_index,
                    region: region_index,
                });
            }
            if seen[region_index] {
                return Err(ReplayAddressBindingError::DuplicateRegion {
                    region: region_index,
                });
            }
            pointer.checked_add(allocation_bytes).ok_or(
                ReplayAddressBindingError::AddressOverflow {
                    argument: argument_index,
                    pointer,
                    bytes: allocation_bytes,
                },
            )?;
            seen[region_index] = true;
            bindings.push(ReplayAddressBinding {
                argument_index,
                region_index,
                base: pointer,
                allocation_bytes,
            });
        }
        if let Some(region) = seen.iter().position(|&bound| !bound) {
            return Err(ReplayAddressBindingError::UnboundRegion { region });
        }
        bindings.sort_unstable_by_key(|binding| binding.base);
        for pair in bindings.windows(2) {
            let end = pair[0].base + pair[0].allocation_bytes;
            if end > pair[1].base {
                return Err(ReplayAddressBindingError::OverlappingAllocations {
                    first: pair[0].argument_index,
                    second: pair[1].argument_index,
                });
            }
        }
        Ok(Self { memory, bindings })
    }

    pub fn bindings(&self) -> &[ReplayAddressBinding] {
        &self.bindings
    }

    pub fn memory(&self) -> &ReplayMemory {
        &self.memory
    }

    pub fn into_memory(self) -> ReplayMemory {
        self.memory
    }

    pub fn resolve(
        &self,
        address: u64,
        len: usize,
    ) -> Result<ResolvedReplayAddress, ReplayAddressError> {
        if len == 0 {
            return Err(ReplayAddressError::ZeroLength);
        }
        let bytes = u64::try_from(len).map_err(|_| ReplayAddressError::LengthOverflow)?;
        let end = address
            .checked_add(bytes)
            .ok_or(ReplayAddressError::RangeOverflow { address })?;
        self.bindings
            .iter()
            .find(|binding| {
                address >= binding.base && end <= binding.base + binding.allocation_bytes
            })
            .map(|binding| ResolvedReplayAddress {
                argument_index: binding.argument_index,
                region_index: binding.region_index,
                offset: address - binding.base,
            })
            .ok_or(ReplayAddressError::Unmapped { address, end })
    }

    pub fn read_states_at(
        &self,
        address: u64,
        len: usize,
    ) -> Result<Vec<MemoryByteState>, ReplayAddressError> {
        let location = self.resolve(address, len)?;
        Ok(self
            .memory
            .read_states(location.region_index, location.offset, len)?)
    }

    pub fn read_known_at(&self, address: u64, len: usize) -> Result<Vec<u8>, ReplayAddressError> {
        let location = self.resolve(address, len)?;
        Ok(self
            .memory
            .read_known(location.region_index, location.offset, len)?)
    }

    pub fn write_known_at(&mut self, address: u64, bytes: &[u8]) -> Result<(), ReplayAddressError> {
        let location = self.resolve(address, bytes.len())?;
        Ok(self
            .memory
            .write_known(location.region_index, location.offset, bytes)?)
    }

    pub fn write_unknown_at(&mut self, address: u64, len: usize) -> Result<(), ReplayAddressError> {
        let location = self.resolve(address, len)?;
        Ok(self
            .memory
            .write_unknown(location.region_index, location.offset, len)?)
    }
}

impl ScalarMemoryBus for AddressedReplayMemory {
    type Error = ReplayAddressError;

    fn read(&mut self, effective_address: u64, destination: &mut [u8]) -> Result<(), Self::Error> {
        let bytes = self.read_known_at(effective_address, destination.len())?;
        destination.copy_from_slice(&bytes);
        Ok(())
    }

    fn write(&mut self, effective_address: u64, source: &[u8]) -> Result<(), Self::Error> {
        self.write_known_at(effective_address, source)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::architecture::Architecture;
    use crate::kernel_config::KernelConfigDocument;
    use crate::machine::ScalarMachine;
    use crate::replay_seed::ReplaySeed;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

    struct TempTree(PathBuf);

    impl TempTree {
        fn new() -> Self {
            let id = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "open-ascend-addressed-replay-{}-{id}",
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

    fn sub_memory() -> (TempTree, ReplayMemory) {
        let tree = TempTree::new();
        let x = tree.0.join("x.bin");
        let y = tree.0.join("y.bin");
        let tiling = tree.0.join("tiling.bin");
        fs::write(&x, vec![0x17; 4096]).unwrap();
        fs::write(&y, vec![0x29; 4096]).unwrap();
        fs::write(&tiling, 1024_u32.to_le_bytes()).unwrap();
        let config = KernelConfigDocument::from_slice(
            format!(
                "{{\"old_mode\":\"0\",\"input_path\":\"{};{}\",\"input_size\":\"4096;4096\",\"output_name\":\"z.bin\",\"output_size\":\"4096\",\"workspace_size\":\"0\",\"tiling_data_path\":\"{};4\"}}",
                x.display(),
                y.display(),
                tiling.display()
            )
            .as_bytes(),
        )
        .unwrap()
        .decode()
        .unwrap();
        let seed = ReplaySeed::load(&config, 8196).unwrap();
        (tree, ReplayMemory::new(seed, 16, 4096))
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
            let (_tree, memory) = sub_memory();
            let mut addressed = AddressedReplayMemory::bind(memory, &pointers).unwrap();
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
                Err(ReplayAddressError::Memory(
                    ReplayMemoryError::UnknownBytes {
                        region: 2,
                        offset: 8,
                    }
                ))
            );
        }
    }

    #[test]
    fn binding_and_address_checks_fail_closed() {
        let (_tree, memory) = sub_memory();
        assert!(matches!(
            AddressedReplayMemory::bind(memory, &[1, 2]),
            Err(ReplayAddressBindingError::PointerCount {
                expected: 5,
                actual: 2
            })
        ));
        let (_tree, memory) = sub_memory();
        let mut pointers = [0x10000, 0x10f00, 0x12000, 0x13000, 0x10013000];
        assert!(matches!(
            AddressedReplayMemory::bind(memory, &pointers),
            Err(ReplayAddressBindingError::OverlappingAllocations {
                first: 0,
                second: 1
            })
        ));
        let (_tree, memory) = sub_memory();
        pointers[1] = 0;
        assert!(matches!(
            AddressedReplayMemory::bind(memory, &pointers),
            Err(ReplayAddressBindingError::NullRegion {
                argument: 1,
                region: 1
            })
        ));
        let (_tree, memory) = sub_memory();
        pointers[1] = 0x12000;
        pointers[4] = u64::MAX - 16;
        assert!(matches!(
            AddressedReplayMemory::bind(memory, &pointers),
            Err(ReplayAddressBindingError::AddressOverflow { argument: 4, .. })
        ));
        let (_tree, memory) = sub_memory();
        pointers = [0x10000, 0x12000, 0x14000, 0x16000, 0x10016000];
        let mut addressed = AddressedReplayMemory::bind(memory, &pointers).unwrap();
        assert_eq!(
            addressed.resolve(pointers[0] + 4095, 2),
            Err(ReplayAddressError::Unmapped {
                address: pointers[0] + 4095,
                end: pointers[0] + 4097
            })
        );
        assert_eq!(
            addressed.resolve(pointers[0], 0),
            Err(ReplayAddressError::ZeroLength)
        );
        assert_eq!(
            addressed.resolve(u64::MAX, 2),
            Err(ReplayAddressError::RangeOverflow { address: u64::MAX })
        );
        assert_eq!(
            addressed.write_known_at(pointers[2] + 4095, &[1, 2]),
            Err(ReplayAddressError::Unmapped {
                address: pointers[2] + 4095,
                end: pointers[2] + 4097
            })
        );
        assert_eq!(addressed.memory().overlay_bytes(), 0);
    }

    #[test]
    fn null_input_slot_requires_an_observed_zero_pointer() {
        let config = KernelConfigDocument::from_slice(
            br#"{"old_mode":"0","input_path":"n","output_name":"z.bin","output_size":"4"}"#,
        )
        .unwrap()
        .decode()
        .unwrap();
        let seed = ReplaySeed::load(&config, 0).unwrap();
        let memory = ReplayMemory::new(seed, 4, 4);
        assert!(matches!(
            AddressedReplayMemory::bind(memory, &[1, 0x1000]),
            Err(ReplayAddressBindingError::NonzeroNull {
                argument: 0,
                pointer: 1
            })
        ));
        let seed = ReplaySeed::load(&config, 0).unwrap();
        let memory = ReplayMemory::new(seed, 4, 4);
        let addressed = AddressedReplayMemory::bind(memory, &[0, 0x1000]).unwrap();
        assert_eq!(addressed.bindings().len(), 1);
        assert_eq!(addressed.bindings()[0].argument_index, 1);
    }
}
