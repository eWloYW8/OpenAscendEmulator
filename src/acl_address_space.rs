use crate::acl_args::AclArgumentPlan;
use crate::addressed_replay::{AddressedReplayMemory, ReplayAddressError};
use crate::machine::ScalarMemoryBus;
use crate::replay_memory::MemoryByteState;
use serde::Serialize;
use thiserror::Error;

pub const MAX_ACL_ARGUMENT_IMAGE_BYTES: usize = 65_536;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AclArgumentImage {
    device_address: u64,
    user_pointer_count: usize,
    opaque_suffix_bytes: usize,
    overflow_address_offset: Option<usize>,
    bytes: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct AclArgumentImageSummary {
    pub device_address: u64,
    pub total_bytes: usize,
    pub user_pointer_count: usize,
    pub user_pointer_bytes: usize,
    pub opaque_suffix_bytes: usize,
    pub overflow_address_offset: Option<usize>,
    pub overflow_address: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum AclArgumentImageError {
    #[error("ACL argument pointer count {actual} does not match planned count {expected}")]
    PointerCount { expected: usize, actual: usize },
    #[error("ACL argument plan has no user pointers to append")]
    EmptyUserArguments,
    #[error("ACL argument {index} requires a null pointer")]
    ExpectedNull { index: usize },
    #[error("ACL argument {index} requires a non-null pointer")]
    ExpectedNonNull { index: usize },
    #[error("ACL argument plan specifies {expected} pointer bytes, but {actual} were provided")]
    PlanAppendLength { expected: usize, actual: usize },
    #[error("ACL argument image has a null device address")]
    NullDeviceAddress,
    #[error("ACL argument image exceeds the {MAX_ACL_ARGUMENT_IMAGE_BYTES}-byte limit")]
    TooLarge,
    #[error("ACL argument image address range overflows u64")]
    AddressOverflow,
    #[error("cannot reserve {requested} host bytes for the ACL argument image")]
    HostAllocationFailed { requested: usize },
}

impl AclArgumentImage {
    pub fn new(
        plan: &AclArgumentPlan,
        device_address: u64,
        user_pointers: &[u64],
        opaque_suffix: &[u8],
    ) -> Result<Self, AclArgumentImageError> {
        Self::build(plan, device_address, user_pointers, opaque_suffix, false)
    }

    pub fn new_with_overflow_address(
        plan: &AclArgumentPlan,
        device_address: u64,
        user_pointers: &[u64],
        overflow_address: u64,
    ) -> Result<Self, AclArgumentImageError> {
        Self::build(
            plan,
            device_address,
            user_pointers,
            &overflow_address.to_le_bytes(),
            true,
        )
    }

    fn build(
        plan: &AclArgumentPlan,
        device_address: u64,
        user_pointers: &[u64],
        suffix: &[u8],
        has_overflow_address: bool,
    ) -> Result<Self, AclArgumentImageError> {
        if plan.slots.len() != user_pointers.len() {
            return Err(AclArgumentImageError::PointerCount {
                expected: plan.slots.len(),
                actual: user_pointers.len(),
            });
        }
        if user_pointers.is_empty() {
            return Err(AclArgumentImageError::EmptyUserArguments);
        }
        if device_address == 0 {
            return Err(AclArgumentImageError::NullDeviceAddress);
        }
        for (slot, &pointer) in plan.slots.iter().zip(user_pointers) {
            if slot.passes_null_pointer && pointer != 0 {
                return Err(AclArgumentImageError::ExpectedNull { index: slot.index });
            }
            if !slot.passes_null_pointer && pointer == 0 {
                return Err(AclArgumentImageError::ExpectedNonNull { index: slot.index });
            }
        }
        let user_pointer_bytes = user_pointers
            .len()
            .checked_mul(8)
            .ok_or(AclArgumentImageError::TooLarge)?;
        if user_pointer_bytes != plan.append_bytes {
            return Err(AclArgumentImageError::PlanAppendLength {
                expected: plan.append_bytes,
                actual: user_pointer_bytes,
            });
        }
        let total_bytes = user_pointer_bytes
            .checked_add(suffix.len())
            .filter(|bytes| *bytes <= MAX_ACL_ARGUMENT_IMAGE_BYTES)
            .ok_or(AclArgumentImageError::TooLarge)?;
        device_address
            .checked_add(total_bytes as u64)
            .ok_or(AclArgumentImageError::AddressOverflow)?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(total_bytes).map_err(|_| {
            AclArgumentImageError::HostAllocationFailed {
                requested: total_bytes,
            }
        })?;
        for &pointer in user_pointers {
            bytes.extend_from_slice(&pointer.to_le_bytes());
        }
        bytes.extend_from_slice(suffix);
        Ok(Self {
            device_address,
            user_pointer_count: user_pointers.len(),
            opaque_suffix_bytes: if has_overflow_address {
                0
            } else {
                suffix.len()
            },
            overflow_address_offset: has_overflow_address.then_some(user_pointer_bytes),
            bytes,
        })
    }

    pub fn summary(&self) -> AclArgumentImageSummary {
        AclArgumentImageSummary {
            device_address: self.device_address,
            total_bytes: self.bytes.len(),
            user_pointer_count: self.user_pointer_count,
            user_pointer_bytes: self.user_pointer_count * 8,
            opaque_suffix_bytes: self.opaque_suffix_bytes,
            overflow_address_offset: self.overflow_address_offset,
            overflow_address: self.overflow_address(),
        }
    }

    pub fn overflow_address(&self) -> Option<u64> {
        self.overflow_address_offset.map(|offset| {
            u64::from_le_bytes(
                self.bytes[offset..offset + 8]
                    .try_into()
                    .expect("the overflow address occupies eight bytes"),
            )
        })
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn device_address(&self) -> u64 {
        self.device_address
    }

    fn end_address(&self) -> u64 {
        self.device_address + self.bytes.len() as u64
    }
}

pub struct AclReplayAddressSpace {
    arguments: AclArgumentImage,
    regions: AddressedReplayMemory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum AclAddressSpaceBuildError {
    #[error("ACL argument image overlaps replay argument {argument_index}")]
    OverlappingRegion { argument_index: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum AclAddressSpaceError {
    #[error("ACL address transfer length cannot fit in u64")]
    LengthOverflow,
    #[error("ACL address range beginning at {address:#x} overflows")]
    RangeOverflow { address: u64 },
    #[error("ACL access [{address:#x}, {end:#x}) crosses the argument image boundary")]
    CrossesArgumentImage { address: u64, end: u64 },
    #[error(transparent)]
    Region(#[from] ReplayAddressError),
}

impl AclReplayAddressSpace {
    pub fn new(
        arguments: AclArgumentImage,
        regions: AddressedReplayMemory,
    ) -> Result<Self, AclAddressSpaceBuildError> {
        for binding in regions.bindings() {
            let region_end = binding.base + binding.allocation_bytes;
            if binding.base < arguments.end_address() && arguments.device_address < region_end {
                return Err(AclAddressSpaceBuildError::OverlappingRegion {
                    argument_index: binding.argument_index,
                });
            }
        }
        Ok(Self { arguments, regions })
    }

    pub fn arguments(&self) -> &AclArgumentImage {
        &self.arguments
    }

    pub fn regions(&self) -> &AddressedReplayMemory {
        &self.regions
    }

    pub fn regions_mut(&mut self) -> &mut AddressedReplayMemory {
        &mut self.regions
    }

    pub fn into_parts(self) -> (AclArgumentImage, AddressedReplayMemory) {
        (self.arguments, self.regions)
    }

    pub fn read_states_at(
        &self,
        address: u64,
        len: usize,
    ) -> Result<Vec<MemoryByteState>, AclAddressSpaceError> {
        if let Some(range) = self.argument_range(address, len)? {
            Ok(self.arguments.bytes[range]
                .iter()
                .copied()
                .map(MemoryByteState::Known)
                .collect())
        } else {
            Ok(self.regions.read_states_at(address, len)?)
        }
    }

    fn argument_range(
        &self,
        address: u64,
        len: usize,
    ) -> Result<Option<std::ops::Range<usize>>, AclAddressSpaceError> {
        let bytes = u64::try_from(len).map_err(|_| AclAddressSpaceError::LengthOverflow)?;
        let end = address
            .checked_add(bytes)
            .ok_or(AclAddressSpaceError::RangeOverflow { address })?;
        if address >= self.arguments.device_address && end <= self.arguments.end_address() {
            let start = (address - self.arguments.device_address) as usize;
            return Ok(Some(start..start + len));
        }
        if address < self.arguments.end_address() && end > self.arguments.device_address {
            return Err(AclAddressSpaceError::CrossesArgumentImage { address, end });
        }
        Ok(None)
    }
}

impl ScalarMemoryBus for AclReplayAddressSpace {
    type Error = AclAddressSpaceError;

    fn read(&mut self, address: u64, destination: &mut [u8]) -> Result<(), Self::Error> {
        if destination.is_empty() {
            return Ok(());
        }
        if let Some(range) = self.argument_range(address, destination.len())? {
            destination.copy_from_slice(&self.arguments.bytes[range]);
            Ok(())
        } else {
            self.regions.read(address, destination)?;
            Ok(())
        }
    }

    fn write(&mut self, address: u64, source: &[u8]) -> Result<(), Self::Error> {
        if source.is_empty() {
            return Ok(());
        }
        if let Some(range) = self.argument_range(address, source.len())? {
            self.arguments.bytes[range].copy_from_slice(source);
            Ok(())
        } else {
            self.regions.write(address, source)?;
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::architecture::Architecture;
    use crate::kernel_config::KernelConfigDocument;
    use crate::machine::{ScalarInstructionStep, ScalarMachine};
    use crate::replay_memory::ReplayMemory;
    use crate::replay_seed::ReplaySeed;

    fn one_output() -> (AclArgumentPlan, AddressedReplayMemory) {
        let config = KernelConfigDocument::from_slice(
            br#"{"old_mode":"0","output_name":"z","output_size":"16"}"#,
        )
        .unwrap()
        .decode()
        .unwrap();
        let plan = AclArgumentPlan::from_config(&config).unwrap();
        let seed = ReplaySeed::load(&config, 0).unwrap();
        let memory = ReplayMemory::new(seed, 64, 64);
        let regions = AddressedReplayMemory::bind(memory, &[0x2000]).unwrap();
        (plan, regions)
    }

    #[test]
    fn scalar_pair_load_can_read_the_argument_image_then_write_an_output_region() {
        for (architecture, pair_word, store_word) in [
            (Architecture::Dav2201, 0x09ca_0180, 0x04c6_5000),
            (Architecture::Dav3510, 0x0cca_0180, 0x03c6_5000),
        ] {
            let (plan, regions) = one_output();
            let image =
                AclArgumentImage::new(&plan, 0x1000, &[0x2000], &0x3000_u64.to_le_bytes()).unwrap();
            assert_eq!(image.summary().user_pointer_bytes, 8);
            assert_eq!(image.summary().opaque_suffix_bytes, 8);
            let mut space = AclReplayAddressSpace::new(image, regions).unwrap();
            let mut machine = ScalarMachine::new(architecture, [0; 32], 0);
            machine.set_xreg(0, 0x1000).unwrap();
            assert!(matches!(
                machine.execute_instruction(0x80, pair_word, &mut space),
                Ok(ScalarInstructionStep::PairLoad(_))
            ));
            assert_eq!(machine.xregs()[5], 0x2000);
            assert_eq!(machine.xregs()[3], 0x3000);
            assert!(matches!(
                machine.execute_instruction(0x84, store_word, &mut space),
                Ok(ScalarInstructionStep::Memory(_))
            ));
            assert_eq!(
                space.regions().read_known_at(0x2000, 8).unwrap(),
                0x3000_u64.to_le_bytes()
            );
            assert_eq!(space.arguments().bytes()[..8], 0x2000_u64.to_le_bytes());
        }
    }

    #[test]
    fn argument_image_bytes_are_mutable_but_cross_boundary_accesses_fail() {
        let (plan, regions) = one_output();
        let image = AclArgumentImage::new(&plan, 0x1000, &[0x2000], &[0; 8]).unwrap();
        let mut space = AclReplayAddressSpace::new(image, regions).unwrap();
        assert_eq!(
            space.read_states_at(0x1000, 8).unwrap(),
            0x2000_u64.to_le_bytes().map(MemoryByteState::Known)
        );
        assert_eq!(
            space.read_states_at(0x2000, 1).unwrap(),
            [MemoryByteState::Unknown]
        );
        space.write(0x1008, &0x3000_u64.to_le_bytes()).unwrap();
        let mut readback = [0; 8];
        space.read(0x1008, &mut readback).unwrap();
        assert_eq!(readback, 0x3000_u64.to_le_bytes());
        assert_eq!(
            space.read(0x100c, &mut readback),
            Err(AclAddressSpaceError::CrossesArgumentImage {
                address: 0x100c,
                end: 0x1014,
            })
        );
        assert_eq!(
            space.write(0xffc, &[0; 8]),
            Err(AclAddressSpaceError::CrossesArgumentImage {
                address: 0xffc,
                end: 0x1004,
            })
        );
    }

    #[test]
    fn named_overflow_address_stays_in_sync_with_mutable_image_bytes() {
        let (plan, regions) = one_output();
        let image =
            AclArgumentImage::new_with_overflow_address(&plan, 0x1000, &[0x2000], 0x3000).unwrap();
        assert_eq!(image.summary().total_bytes, 16);
        assert_eq!(image.summary().opaque_suffix_bytes, 0);
        assert_eq!(image.summary().overflow_address_offset, Some(8));
        assert_eq!(image.summary().overflow_address, Some(0x3000));
        let mut space = AclReplayAddressSpace::new(image, regions).unwrap();
        space.write(0x1008, &0x4000_u64.to_le_bytes()).unwrap();
        assert_eq!(space.arguments().overflow_address(), Some(0x4000));
        assert_eq!(space.arguments().summary().overflow_address, Some(0x4000));

        let opaque =
            AclArgumentImage::new(&plan, 0x1000, &[0x2000], &0x3000_u64.to_le_bytes()).unwrap();
        assert_eq!(opaque.summary().opaque_suffix_bytes, 8);
        assert_eq!(opaque.summary().overflow_address, None);
        assert_eq!(opaque.summary().overflow_address_offset, None);
    }

    #[test]
    fn image_rejects_invalid_pointers_sizes_and_overlapping_regions() {
        let (plan, regions) = one_output();
        assert_eq!(
            AclArgumentImage::new(&plan, 0x1000, &[], &[]),
            Err(AclArgumentImageError::PointerCount {
                expected: 1,
                actual: 0,
            })
        );
        assert_eq!(
            AclArgumentImage::new(&plan, 0x1000, &[0], &[]),
            Err(AclArgumentImageError::ExpectedNonNull { index: 0 })
        );
        let mut invalid_plan = plan.clone();
        invalid_plan.append_bytes = 16;
        assert_eq!(
            AclArgumentImage::new(&invalid_plan, 0x1000, &[0x2000], &[]),
            Err(AclArgumentImageError::PlanAppendLength {
                expected: 16,
                actual: 8,
            })
        );
        assert_eq!(
            AclArgumentImage::new(&plan, 0, &[0x2000], &[]),
            Err(AclArgumentImageError::NullDeviceAddress)
        );
        assert_eq!(
            AclArgumentImage::new(&plan, u64::MAX - 7, &[0x2000], &[0; 8]),
            Err(AclArgumentImageError::AddressOverflow)
        );
        assert_eq!(
            AclArgumentImage::new(
                &plan,
                0x1000,
                &[0x2000],
                &vec![0; MAX_ACL_ARGUMENT_IMAGE_BYTES],
            ),
            Err(AclArgumentImageError::TooLarge)
        );
        let overlapping = AclArgumentImage::new(&plan, 0x1ff8, &[0x2000], &[0; 8]).unwrap();
        assert!(matches!(
            AclReplayAddressSpace::new(overlapping, regions),
            Err(AclAddressSpaceBuildError::OverlappingRegion { argument_index: 0 })
        ));
    }

    #[test]
    fn null_argument_requires_a_zero_encoded_pointer() {
        let config = KernelConfigDocument::from_slice(br#"{"old_mode":"0","input_path":"n"}"#)
            .unwrap()
            .decode()
            .unwrap();
        let plan = AclArgumentPlan::from_config(&config).unwrap();
        assert_eq!(
            AclArgumentImage::new(&plan, 0x1000, &[0x2000], &[]),
            Err(AclArgumentImageError::ExpectedNull { index: 0 })
        );
        assert_eq!(
            AclArgumentImage::new(&plan, 0x1000, &[0], &[])
                .unwrap()
                .bytes(),
            &[0; 8]
        );
    }
}
