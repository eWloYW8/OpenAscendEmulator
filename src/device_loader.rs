use serde::Serialize;
use thiserror::Error;

use crate::architecture::Architecture;
use crate::binary_alloc::{
    BINARY_ALIGNMENT_BYTES, BinaryAllocationPlanError, BinaryDeviceAllocationPlan,
};
use crate::device_elf::{
    DeviceElf, DeviceElfError, DeviceGlobalAddresses, DeviceKernelSummary, PreparedDeviceLoadImage,
};
use crate::device_pool::{DeviceMemoryPoolManager, DevicePoolBlock, DevicePoolError};
use crate::hbm::HbmResolveError;
use crate::hbm_pv_memory::{HbmPvMemory, HbmPvMemoryError};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct DeviceBinaryPlacement {
    pub architecture: Architecture,
    pub raw_device_address: u64,
    pub aligned_code_address: u64,
    pub allocation_base: u64,
    pub allocation_bytes: u64,
    pub source_file_offset: u64,
    pub image_bytes: u32,
    pub global_patch_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum DeviceBinaryAllocation {
    Pool(DevicePoolBlock),
    DirectFallback { raw_device_address: u64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum DevicePoolAttempt {
    Disabled,
    Allocated,
    RejectedRequest,
    BackingAllocationFailed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct DeviceBinaryLoad {
    pub placement: DeviceBinaryPlacement,
    pub allocation: DeviceBinaryAllocation,
    pub pool_attempt: DevicePoolAttempt,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LoadedDeviceKernel {
    load: DeviceBinaryLoad,
    kernel: DeviceKernelSummary,
    entry_address: u64,
    executable_start_address: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct InstructionFetchWindow {
    pub pc: u64,
    pub requested_bytes: u8,
    pub copied_image_bytes: u8,
    pub bytes: [u8; 16],
}

impl LoadedDeviceKernel {
    pub const fn load(&self) -> DeviceBinaryLoad {
        self.load
    }

    pub const fn kernel(&self) -> &DeviceKernelSummary {
        &self.kernel
    }

    pub const fn entry_address(&self) -> u64 {
        self.entry_address
    }

    pub const fn executable_start_address(&self) -> u64 {
        self.executable_start_address
    }

    pub fn fetch_word(
        &self,
        memory: &mut HbmPvMemory,
        pc: u64,
    ) -> Result<u32, DeviceKernelFetchError> {
        self.fetch_in_range(
            memory,
            pc,
            self.entry_address,
            self.kernel.byte_count,
            DeviceKernelFetchError::OutsideKernel(pc),
        )
    }

    pub fn fetch_executable_word(
        &self,
        memory: &mut HbmPvMemory,
        pc: u64,
    ) -> Result<u32, DeviceKernelFetchError> {
        self.fetch_in_range(
            memory,
            pc,
            self.executable_start_address,
            self.kernel.section_byte_count,
            DeviceKernelFetchError::OutsideExecutableSection(pc),
        )
    }

    pub fn fetch_window(
        &self,
        memory: &mut HbmPvMemory,
        pc: u64,
    ) -> Result<InstructionFetchWindow, DeviceKernelFetchError> {
        self.validate_fetch_pc(
            memory,
            pc,
            self.executable_start_address,
            self.kernel.section_byte_count,
            DeviceKernelFetchError::OutsideExecutableSection(pc),
        )?;
        let requested_bytes = 16 - (pc & 15) as u8;
        let span = memory
            .allocator()
            .resolve_live(pc, u64::from(requested_bytes))?;
        if span.allocation_base != self.load.placement.allocation_base {
            return Err(DeviceKernelFetchError::ImageAllocationChanged);
        }
        let mut bytes = [0; 16];
        memory.device_to_host(pc, &mut bytes[..usize::from(requested_bytes)])?;
        let image_offset = pc
            .checked_sub(self.load.placement.aligned_code_address)
            .ok_or(DeviceKernelFetchError::ImageAllocationChanged)?;
        let copied_image_bytes = u64::from(self.load.placement.image_bytes)
            .saturating_sub(image_offset)
            .min(u64::from(requested_bytes)) as u8;
        Ok(InstructionFetchWindow {
            pc,
            requested_bytes,
            copied_image_bytes,
            bytes,
        })
    }

    fn fetch_in_range(
        &self,
        memory: &mut HbmPvMemory,
        pc: u64,
        start: u64,
        byte_count: u64,
        outside_error: DeviceKernelFetchError,
    ) -> Result<u32, DeviceKernelFetchError> {
        self.validate_fetch_pc(memory, pc, start, byte_count, outside_error)?;
        let mut bytes = [0; 4];
        memory.device_to_host(pc, &mut bytes)?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn validate_fetch_pc(
        &self,
        memory: &HbmPvMemory,
        pc: u64,
        start: u64,
        byte_count: u64,
        outside_error: DeviceKernelFetchError,
    ) -> Result<(), DeviceKernelFetchError> {
        if memory.allocator().architecture() != self.load.placement.architecture {
            return Err(DeviceKernelFetchError::ArchitectureMismatch);
        }
        let relative = pc.checked_sub(start).ok_or(outside_error)?;
        if !pc.is_multiple_of(4) {
            return Err(DeviceKernelFetchError::UnalignedPc(pc));
        }
        if relative.checked_add(4).is_none_or(|end| end > byte_count) {
            return Err(outside_error);
        }
        let span = memory.allocator().resolve_live(pc, 4)?;
        if span.allocation_base != self.load.placement.allocation_base {
            return Err(DeviceKernelFetchError::ImageAllocationChanged);
        }
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum DeviceKernelLoadError {
    #[error(transparent)]
    Elf(#[from] DeviceElfError),
    #[error(transparent)]
    Binary(#[from] DeviceBinaryLoadError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum DeviceKernelFetchError {
    #[error("loaded kernel and memory target different architectures")]
    ArchitectureMismatch,
    #[error("device instruction PC {0:#x} is not four-byte aligned")]
    UnalignedPc(u64),
    #[error("device instruction PC {0:#x} is outside the selected kernel")]
    OutsideKernel(u64),
    #[error("device instruction PC {0:#x} is outside the executable section")]
    OutsideExecutableSection(u64),
    #[error("the loaded image is no longer in its original allocation")]
    ImageAllocationChanged,
    #[error(transparent)]
    Allocation(#[from] HbmResolveError),
    #[error(transparent)]
    Memory(#[from] HbmPvMemoryError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum DeviceBinaryLoadError {
    #[error("allocation plan and memory target different architectures")]
    ArchitectureMismatch,
    #[error("the prepared image is empty or exceeds the runtime's 32-bit byte count")]
    InvalidImageSize,
    #[error("the prepared image declares {declared} bytes but contains {actual} bytes")]
    ImageSummaryMismatch { declared: u64, actual: usize },
    #[error("the allocation plan expects {planned} bytes but the image has {actual} bytes")]
    PlanSizeMismatch { planned: u32, actual: u32 },
    #[error("the raw pointer and aligned image span lie in different allocations")]
    AllocationMismatch,
    #[error(transparent)]
    Alignment(#[from] BinaryAllocationPlanError),
    #[error(transparent)]
    Allocation(#[from] HbmResolveError),
    #[error(transparent)]
    Memory(#[from] HbmPvMemoryError),
    #[error(transparent)]
    Pool(#[from] DevicePoolError),
}

pub fn load_named_kernel(
    memory: &mut HbmPvMemory,
    pools: &mut DeviceMemoryPoolManager,
    plan: &BinaryDeviceAllocationPlan,
    elf: &DeviceElf<'_>,
    kernel_name: &str,
    global_addresses: DeviceGlobalAddresses,
    read_only: bool,
) -> Result<LoadedDeviceKernel, DeviceKernelLoadError> {
    let projected = elf.project_kernel(kernel_name, BINARY_ALIGNMENT_BYTES)?;
    let prepared = elf.prepare_load_image(global_addresses)?;
    let load = load_with_pool_preference(memory, pools, plan, &prepared, read_only)?;
    let entry_address = projected
        .summary
        .function_device_address(load.placement.aligned_code_address)?;
    let executable_start_address = load
        .placement
        .aligned_code_address
        .checked_add(projected.summary.section_virtual_address)
        .ok_or(DeviceElfError::RuntimeAddressOverflow)?;
    Ok(LoadedDeviceKernel {
        load,
        kernel: projected.summary,
        entry_address,
        executable_start_address,
    })
}

pub fn load_with_pool_preference(
    memory: &mut HbmPvMemory,
    pools: &mut DeviceMemoryPoolManager,
    plan: &BinaryDeviceAllocationPlan,
    image: &PreparedDeviceLoadImage,
    read_only: bool,
) -> Result<DeviceBinaryLoad, DeviceBinaryLoadError> {
    validate_image(memory, plan, image)?;
    let pool_attempt = if let Some(pool_bytes) = plan.pool_request_bytes {
        match pools.allocate(memory, pool_bytes, read_only) {
            Ok(Some(block)) => {
                let placement = match copy_prepared_image_at(memory, plan, image, block.address) {
                    Ok(placement) => placement,
                    Err(error) => {
                        pools.release(memory, block)?;
                        return Err(error);
                    }
                };
                return Ok(DeviceBinaryLoad {
                    placement,
                    allocation: DeviceBinaryAllocation::Pool(block),
                    pool_attempt: DevicePoolAttempt::Allocated,
                });
            }
            Ok(None) => DevicePoolAttempt::RejectedRequest,
            Err(DevicePoolError::Memory(_)) => DevicePoolAttempt::BackingAllocationFailed,
            Err(error) => return Err(error.into()),
        }
    } else {
        DevicePoolAttempt::Disabled
    };
    let placement = load_direct_fallback(memory, plan, image)?;
    Ok(DeviceBinaryLoad {
        allocation: DeviceBinaryAllocation::DirectFallback {
            raw_device_address: placement.raw_device_address,
        },
        placement,
        pool_attempt,
    })
}

pub fn copy_prepared_image_at(
    memory: &mut HbmPvMemory,
    plan: &BinaryDeviceAllocationPlan,
    image: &PreparedDeviceLoadImage,
    raw_device_address: u64,
) -> Result<DeviceBinaryPlacement, DeviceBinaryLoadError> {
    let image_bytes = validate_image(memory, plan, image)?;
    let aligned_code_address = plan.aligned_code_address(raw_device_address)?;
    let raw_span = memory.allocator().resolve_live(raw_device_address, 1)?;
    let code_span = memory
        .allocator()
        .resolve_live(aligned_code_address, u64::from(image_bytes))?;
    if raw_span.allocation_base != code_span.allocation_base {
        return Err(DeviceBinaryLoadError::AllocationMismatch);
    }
    memory.host_to_device(aligned_code_address, &image.bytes)?;
    Ok(DeviceBinaryPlacement {
        architecture: plan.architecture,
        raw_device_address,
        aligned_code_address,
        allocation_base: code_span.allocation_base,
        allocation_bytes: code_span.allocation_bytes,
        source_file_offset: image.summary.file_offset,
        image_bytes,
        global_patch_count: image.patches.len(),
    })
}

pub fn load_direct_fallback(
    memory: &mut HbmPvMemory,
    plan: &BinaryDeviceAllocationPlan,
    image: &PreparedDeviceLoadImage,
) -> Result<DeviceBinaryPlacement, DeviceBinaryLoadError> {
    validate_image(memory, plan, image)?;
    let raw_device_address = memory.allocate_driver_request(plan.fallback_request_bytes)?;
    match copy_prepared_image_at(memory, plan, image, raw_device_address) {
        Ok(placement) => Ok(placement),
        Err(error) => {
            memory.free(raw_device_address)?;
            Err(error)
        }
    }
}

fn validate_image(
    memory: &HbmPvMemory,
    plan: &BinaryDeviceAllocationPlan,
    image: &PreparedDeviceLoadImage,
) -> Result<u32, DeviceBinaryLoadError> {
    if memory.allocator().architecture() != plan.architecture {
        return Err(DeviceBinaryLoadError::ArchitectureMismatch);
    }
    let image_bytes =
        u32::try_from(image.bytes.len()).map_err(|_| DeviceBinaryLoadError::InvalidImageSize)?;
    if image_bytes == 0 {
        return Err(DeviceBinaryLoadError::InvalidImageSize);
    }
    if image.summary.byte_count != u64::from(image_bytes) {
        return Err(DeviceBinaryLoadError::ImageSummaryMismatch {
            declared: image.summary.byte_count,
            actual: image.bytes.len(),
        });
    }
    if plan.binary_bytes != image_bytes {
        return Err(DeviceBinaryLoadError::PlanSizeMismatch {
            planned: plan.binary_bytes,
            actual: image_bytes,
        });
    }
    Ok(image_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device_elf::DeviceLoadImageSummary;
    use crate::hbm::CAMODEL_HBM_BASE;

    fn image(bytes: Vec<u8>) -> PreparedDeviceLoadImage {
        PreparedDeviceLoadImage {
            summary: DeviceLoadImageSummary {
                file_offset: 4096,
                byte_count: bytes.len() as u64,
                first_alloc_section: ".text".to_owned(),
                last_alloc_section: ".text".to_owned(),
                alloc_section_count: 1,
            },
            address_meta_flags: 0,
            bytes,
            patches: Vec::new(),
        }
    }

    #[test]
    fn copies_aligned_image_bytes_for_both_architectures() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut memory = HbmPvMemory::new(architecture, 0xa5, 1);
            memory.allocate_driver_request(512).unwrap();
            let raw = memory.allocate_driver_request(8192).unwrap();
            let prepared = image((0..176).map(|value| value as u8).collect());
            let plan = BinaryDeviceAllocationPlan::new(architecture, 176, false, true).unwrap();
            let placement = copy_prepared_image_at(&mut memory, &plan, &prepared, raw).unwrap();
            assert_eq!(placement.raw_device_address, CAMODEL_HBM_BASE + 512);
            assert_eq!(placement.aligned_code_address, CAMODEL_HBM_BASE + 4096);
            assert_eq!(placement.allocation_base, raw);
            assert_eq!(placement.allocation_bytes, 8192);
            assert_eq!(placement.source_file_offset, 4096);
            assert_eq!(placement.image_bytes, 176);
            let mut actual = vec![0; 176];
            memory
                .device_to_host(placement.aligned_code_address, &mut actual)
                .unwrap();
            assert_eq!(actual, prepared.bytes);
            assert_eq!(
                memory.store().dirty_byte(placement.aligned_code_address),
                Some(1)
            );
        }
    }

    #[test]
    fn rejects_cross_allocation_alignment_without_writing() {
        let mut memory = HbmPvMemory::new(Architecture::Dav2201, 0, 1);
        let raw = memory.allocate(4096).unwrap() + 1;
        memory.allocate(4096).unwrap();
        let prepared = image(vec![7; 16]);
        let plan =
            BinaryDeviceAllocationPlan::new(Architecture::Dav2201, 16, false, false).unwrap();
        assert_eq!(
            copy_prepared_image_at(&mut memory, &plan, &prepared, raw),
            Err(DeviceBinaryLoadError::AllocationMismatch)
        );
        assert_eq!(memory.store().page_count(), 0);
    }

    #[test]
    fn validates_architecture_and_image_size_before_copy() {
        let mut memory = HbmPvMemory::new(Architecture::Dav2201, 0, 1);
        let raw = memory.allocate(8192).unwrap();
        let mut prepared = image(vec![1; 16]);
        let other_plan =
            BinaryDeviceAllocationPlan::new(Architecture::Dav3510, 16, false, false).unwrap();
        assert_eq!(
            copy_prepared_image_at(&mut memory, &other_plan, &prepared, raw),
            Err(DeviceBinaryLoadError::ArchitectureMismatch)
        );
        let plan =
            BinaryDeviceAllocationPlan::new(Architecture::Dav2201, 16, false, false).unwrap();
        prepared.summary.byte_count = 15;
        assert!(matches!(
            copy_prepared_image_at(&mut memory, &plan, &prepared, raw),
            Err(DeviceBinaryLoadError::ImageSummaryMismatch { .. })
        ));
        prepared.summary.byte_count = 16;
        prepared.bytes.push(2);
        assert!(matches!(
            copy_prepared_image_at(&mut memory, &plan, &prepared, raw),
            Err(DeviceBinaryLoadError::ImageSummaryMismatch { .. })
        ));
        prepared.summary.byte_count = 17;
        assert!(matches!(
            copy_prepared_image_at(&mut memory, &plan, &prepared, raw),
            Err(DeviceBinaryLoadError::PlanSizeMismatch { .. })
        ));
        assert_eq!(memory.store().page_count(), 0);
    }

    #[test]
    fn direct_fallback_allocates_and_rolls_back_on_copy_failure() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let prepared = image(vec![1; 176]);
            let plan = BinaryDeviceAllocationPlan::new(architecture, 176, false, false).unwrap();
            let mut memory = HbmPvMemory::new(architecture, 0, 1);
            let placement = load_direct_fallback(&mut memory, &plan, &prepared).unwrap();
            assert_eq!(placement.raw_device_address, CAMODEL_HBM_BASE);
            assert_eq!(placement.aligned_code_address, CAMODEL_HBM_BASE);
            assert_eq!(placement.allocation_bytes, 4608);

            let mut limited = HbmPvMemory::new(architecture, 0, 0);
            let initial_spans = limited.allocator().spans().to_vec();
            assert!(matches!(
                load_direct_fallback(&mut limited, &plan, &prepared),
                Err(DeviceBinaryLoadError::Memory(HbmPvMemoryError::Store(_)))
            ));
            assert_eq!(limited.allocator().spans(), initial_spans);
            assert_eq!(limited.store().page_count(), 0);
        }
    }

    #[test]
    fn pool_preference_allocates_backing_and_copies_image() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut memory = HbmPvMemory::new(architecture, 0, 1);
            memory.allocate_driver_request(512).unwrap();
            let mut pools = DeviceMemoryPoolManager::new(architecture);
            let prepared = image((0..176).map(|value| value as u8).collect());
            let plan = BinaryDeviceAllocationPlan::new(architecture, 176, false, true).unwrap();
            let loaded =
                load_with_pool_preference(&mut memory, &mut pools, &plan, &prepared, false)
                    .unwrap();
            assert_eq!(loaded.pool_attempt, DevicePoolAttempt::Allocated);
            let DeviceBinaryAllocation::Pool(block) = loaded.allocation else {
                panic!("expected pool allocation");
            };
            assert_eq!(block.address, CAMODEL_HBM_BASE + 512);
            assert_eq!(block.bytes, 4096);
            assert_eq!(
                loaded.placement.aligned_code_address,
                CAMODEL_HBM_BASE + 4096
            );
            assert_eq!(pools.pool_summaries()[0].used_bytes, 4096);
            let mut actual = vec![0; 176];
            memory
                .device_to_host(loaded.placement.aligned_code_address, &mut actual)
                .unwrap();
            assert_eq!(actual, prepared.bytes);
        }
    }

    #[test]
    fn disabled_pool_uses_direct_fallback() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut memory = HbmPvMemory::new(architecture, 0, 1);
            let mut pools = DeviceMemoryPoolManager::new(architecture);
            let prepared = image(vec![3; 176]);
            let plan = BinaryDeviceAllocationPlan::new(architecture, 176, false, false).unwrap();
            let loaded =
                load_with_pool_preference(&mut memory, &mut pools, &plan, &prepared, false)
                    .unwrap();
            assert_eq!(loaded.pool_attempt, DevicePoolAttempt::Disabled);
            assert_eq!(
                loaded.allocation,
                DeviceBinaryAllocation::DirectFallback {
                    raw_device_address: CAMODEL_HBM_BASE
                }
            );
            assert!(pools.pool_summaries().is_empty());
        }
    }

    #[test]
    fn oversized_pool_request_uses_direct_fallback() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut memory = HbmPvMemory::new(architecture, 0, 2);
            let mut pools = DeviceMemoryPoolManager::new(architecture);
            let prepared = image(vec![0x5a; crate::binary_alloc::BINARY_POOL_BYTES as usize]);
            let plan = BinaryDeviceAllocationPlan::new(
                architecture,
                crate::binary_alloc::BINARY_POOL_BYTES as u32,
                true,
                true,
            )
            .unwrap();
            let loaded =
                load_with_pool_preference(&mut memory, &mut pools, &plan, &prepared, false)
                    .unwrap();
            assert_eq!(loaded.pool_attempt, DevicePoolAttempt::RejectedRequest);
            assert_eq!(loaded.placement.image_bytes, 2 * 1024 * 1024);
            assert!(matches!(
                loaded.allocation,
                DeviceBinaryAllocation::DirectFallback { .. }
            ));
            assert!(pools.pool_summaries().is_empty());
            assert_eq!(memory.store().page_count(), 2);
        }
    }

    #[test]
    fn failed_pool_copy_releases_its_block() {
        let architecture = Architecture::Dav2201;
        let mut memory = HbmPvMemory::new(architecture, 0, 0);
        let mut pools = DeviceMemoryPoolManager::new(architecture);
        let prepared = image(vec![3; 176]);
        let plan = BinaryDeviceAllocationPlan::new(architecture, 176, false, true).unwrap();
        assert!(matches!(
            load_with_pool_preference(&mut memory, &mut pools, &plan, &prepared, false),
            Err(DeviceBinaryLoadError::Memory(HbmPvMemoryError::Store(_)))
        ));
        assert_eq!(pools.pool_summaries()[0].used_bytes, 0);
        assert_eq!(pools.pool_summaries()[0].free_block_count, 2);
        assert_eq!(pools.pool_summaries()[0].free_bytes, 2 * 1024 * 1024);
        assert_eq!(memory.store().page_count(), 0);
    }
}
