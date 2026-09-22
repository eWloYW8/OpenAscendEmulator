use thiserror::Error;

use crate::architecture::Architecture;
use crate::image::elf::{DeviceElf, DeviceElfError, DeviceKernelSummary, DeviceLoadImage};
use crate::memory::hbm::HbmResolveError;
use crate::memory::hbm_pv_memory::{HbmPvMemory, HbmPvMemoryError};

const CODE_ALIGNMENT: u64 = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceBinaryPlacement {
    pub architecture: Architecture,
    pub aligned_code_address: u64,
    pub allocation_base: u64,
    pub allocation_bytes: u64,
    pub image_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadedDeviceKernel {
    placement: DeviceBinaryPlacement,
    kernel: DeviceKernelSummary,
    entry_address: u64,
    executable_start_address: u64,
}

impl LoadedDeviceKernel {
    pub const fn placement(&self) -> DeviceBinaryPlacement {
        self.placement
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

    pub fn fetch_executable_word(
        &self,
        memory: &mut HbmPvMemory,
        pc: u64,
    ) -> Result<u32, DeviceKernelFetchError> {
        if memory.allocator().architecture() != self.placement.architecture {
            return Err(DeviceKernelFetchError::ArchitectureMismatch);
        }
        let outside = DeviceKernelFetchError::OutsideExecutableSection(pc);
        let relative = pc
            .checked_sub(self.executable_start_address)
            .ok_or(outside)?;
        if !pc.is_multiple_of(4) {
            return Err(DeviceKernelFetchError::UnalignedPc(pc));
        }
        if relative
            .checked_add(4)
            .is_none_or(|end| end > self.kernel.section_byte_count)
        {
            return Err(outside);
        }
        let span = memory.allocator().resolve_live(pc, 4)?;
        if span.allocation_base != self.placement.allocation_base {
            return Err(DeviceKernelFetchError::ImageAllocationChanged);
        }
        let mut bytes = [0; 4];
        memory.device_to_host(pc, &mut bytes)?;
        Ok(u32::from_le_bytes(bytes))
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
    #[error("the load image is empty")]
    InvalidImageSize,
    #[error("image allocation size overflows")]
    AllocationSizeOverflow,
    #[error(transparent)]
    Allocation(#[from] HbmResolveError),
    #[error(transparent)]
    Memory(#[from] HbmPvMemoryError),
}

pub fn load_named_kernel(
    memory: &mut HbmPvMemory,
    elf: &DeviceElf<'_>,
    kernel_name: &str,
) -> Result<LoadedDeviceKernel, DeviceKernelLoadError> {
    let kernel = elf.loadable_kernel(kernel_name)?;
    let image = elf.load_image()?;
    let placement = load_image(memory, &image)?;
    let addresses = (|| {
        let entry = kernel.device_address(placement.aligned_code_address)?;
        let end = entry
            .checked_add(kernel.byte_count)
            .ok_or(DeviceElfError::DeviceAddressOverflow)?;
        let section = placement
            .aligned_code_address
            .checked_add(kernel.section_virtual_address)
            .ok_or(DeviceElfError::DeviceAddressOverflow)?;
        let section_end = section
            .checked_add(kernel.section_byte_count)
            .ok_or(DeviceElfError::DeviceAddressOverflow)?;
        if end > section_end {
            return Err(DeviceElfError::InvalidKernelRange);
        }
        Ok::<_, DeviceElfError>((entry, section))
    })();
    let (entry_address, executable_start_address) = match addresses {
        Ok(addresses) => addresses,
        Err(error) => {
            memory
                .free(placement.allocation_base)
                .map_err(DeviceBinaryLoadError::from)?;
            return Err(error.into());
        }
    };
    Ok(LoadedDeviceKernel {
        placement,
        kernel,
        entry_address,
        executable_start_address,
    })
}

pub fn load_image(
    memory: &mut HbmPvMemory,
    image: &DeviceLoadImage<'_>,
) -> Result<DeviceBinaryPlacement, DeviceBinaryLoadError> {
    let image_bytes = validate_image(image)?;
    let allocation_bytes = image_bytes
        .checked_add(CODE_ALIGNMENT - 1)
        .ok_or(DeviceBinaryLoadError::AllocationSizeOverflow)?;
    let allocation_base = memory.allocate(allocation_bytes)?;
    let loaded = (|| {
        let aligned_code_address = allocation_base
            .checked_add(CODE_ALIGNMENT - 1)
            .map(|address| address & !(CODE_ALIGNMENT - 1))
            .ok_or(DeviceBinaryLoadError::AllocationSizeOverflow)?;
        memory
            .allocator()
            .resolve_live(aligned_code_address, image_bytes)?;
        memory.host_to_device(aligned_code_address, image.bytes)?;
        Ok(DeviceBinaryPlacement {
            architecture: memory.allocator().architecture(),
            aligned_code_address,
            allocation_base,
            allocation_bytes,
            image_bytes,
        })
    })();
    match loaded {
        Ok(placement) => Ok(placement),
        Err(error) => {
            memory.free(allocation_base)?;
            Err(error)
        }
    }
}

fn validate_image(image: &DeviceLoadImage<'_>) -> Result<u64, DeviceBinaryLoadError> {
    let image_bytes = u64::try_from(image.bytes.len())
        .map_err(|_| DeviceBinaryLoadError::AllocationSizeOverflow)?;
    if image_bytes == 0 {
        return Err(DeviceBinaryLoadError::InvalidImageSize);
    }
    Ok(image_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::c310::dispatch::C310VfQueueStep;
    use crate::memory::hbm::DEFAULT_HBM_BASE;
    use crate::sim::c310::scalar::{C310ScalarBus, C310ScalarStepper};
    use crate::sim::c310::vector_queue::C310VfQueueDisposition;
    use crate::sim::common::scalar::ScalarStepper;
    use crate::sim::common::scalar::{ScalarMachine, ScalarMemoryBus};
    use std::io;

    struct NoMemoryBus;

    impl ScalarMemoryBus for NoMemoryBus {
        type Error = io::Error;

        fn read(&mut self, _address: u64, _destination: &mut [u8]) -> Result<(), Self::Error> {
            Err(io::Error::other("unexpected data read"))
        }

        fn write(&mut self, _address: u64, _source: &[u8]) -> Result<(), Self::Error> {
            Err(io::Error::other("unexpected data write"))
        }
    }

    struct VfQueueBus {
        steps: Vec<C310VfQueueStep>,
    }

    impl ScalarMemoryBus for VfQueueBus {
        type Error = io::Error;

        fn read(&mut self, _address: u64, _destination: &mut [u8]) -> Result<(), Self::Error> {
            Err(io::Error::other("unexpected data read"))
        }

        fn write(&mut self, _address: u64, _source: &[u8]) -> Result<(), Self::Error> {
            Err(io::Error::other("unexpected data write"))
        }
    }

    impl C310ScalarBus for VfQueueBus {
        fn enqueue_vf(
            &mut self,
            step: C310VfQueueStep,
        ) -> Result<C310VfQueueDisposition, Self::Error> {
            self.steps.push(step);
            Ok(C310VfQueueDisposition::Accepted)
        }
    }

    fn image(bytes: &[u8]) -> DeviceLoadImage<'_> {
        DeviceLoadImage {
            file_offset: 4096,
            bytes,
        }
    }

    fn loaded_program(
        architecture: Architecture,
        words: &[u32],
    ) -> (HbmPvMemory, LoadedDeviceKernel) {
        let bytes: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        let byte_count = bytes.len() as u64;
        let prepared = image(&bytes);
        let mut memory = HbmPvMemory::new(architecture, 0, 1);
        let placement = load_image(&mut memory, &prepared).unwrap();
        let entry = placement.aligned_code_address;
        (
            memory,
            LoadedDeviceKernel {
                placement,
                kernel: DeviceKernelSummary {
                    name: "ScalarProgram".to_owned(),
                    section: ".text".to_owned(),
                    section_virtual_address: 0,
                    section_file_offset: 4096,
                    section_byte_count: byte_count,
                    virtual_address: 0,
                    file_offset: 4096,
                    byte_count,
                },
                entry_address: entry,
                executable_start_address: entry,
            },
        )
    }

    #[test]
    fn loaded_scalar_program_steps_until_end() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let words = [0x0706_0001_u32, 0x4160_0000];
            let (mut memory, kernel) = loaded_program(architecture, &words);
            let entry = kernel.entry_address();
            let mut bus = NoMemoryBus;
            let mut stepper =
                ScalarStepper::new(ScalarMachine::new(architecture, [0; 32], 0), entry);
            let first = stepper.step_loaded(&kernel, &mut memory, &mut bus).unwrap();
            assert_eq!(first.word, words[0]);
            assert_eq!(first.next_pc, entry + 4);
            let last = stepper.step_loaded(&kernel, &mut memory, &mut bus).unwrap();
            assert_eq!(last.word, words[1]);
            assert!(last.halted_after);
            assert_eq!(last.next_pc, entry + 8);
            assert!(stepper.is_halted());
        }
    }

    #[test]
    fn loaded_c310_vf_pair_consumes_eight_code_bytes() {
        let words = [0x154d_0000, 0x15e0_0105, 0x4160_0000];
        let (mut memory, kernel) = loaded_program(Architecture::Dav3510, &words);
        let entry = kernel.entry_address();
        let mut machine = ScalarMachine::new(Architecture::Dav3510, [0; 32], 0);
        machine.set_xreg(13, 0x10d0_d900).unwrap();
        let mut stepper = C310ScalarStepper::new(machine, entry);
        let mut bus = VfQueueBus { steps: Vec::new() };
        let first = stepper.step_loaded(&kernel, &mut memory, &mut bus).unwrap();
        assert_eq!(first.word, words[0]);
        assert_eq!(first.next_pc, entry + 8);
        assert_eq!(bus.steps.len(), 1);
        assert_eq!(stepper.pc(), entry + 8);
        let last = stepper.step_loaded(&kernel, &mut memory, &mut bus).unwrap();
        assert_eq!(last.word, words[2]);
        assert!(last.halted_after);
        assert_eq!(last.next_pc, entry + 12);
    }

    #[test]
    fn copies_aligned_image_bytes_for_both_architectures() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut memory = HbmPvMemory::new(architecture, 0xa5, 1);
            memory.allocate(512).unwrap();
            let bytes: Vec<_> = (0..176).map(|value| value as u8).collect();
            let prepared = image(&bytes);
            let placement = load_image(&mut memory, &prepared).unwrap();
            assert_eq!(placement.aligned_code_address, DEFAULT_HBM_BASE + 4096);
            assert_eq!(placement.allocation_base, DEFAULT_HBM_BASE + 512);
            assert_eq!(placement.allocation_bytes, 176 + CODE_ALIGNMENT - 1);
            assert_eq!(placement.image_bytes, 176);
            let mut actual = vec![0; 176];
            memory
                .device_to_host(placement.aligned_code_address, &mut actual)
                .unwrap();
            assert_eq!(actual, prepared.bytes);
        }
    }

    #[test]
    fn rejects_empty_image_before_allocating() {
        let mut memory = HbmPvMemory::new(Architecture::Dav2201, 0, 1);
        let prepared = image(&[]);
        assert_eq!(
            load_image(&mut memory, &prepared),
            Err(DeviceBinaryLoadError::InvalidImageSize)
        );
        assert_eq!(memory.allocator().spans().len(), 1);
        assert_eq!(memory.store().page_count(), 0);
    }

    #[test]
    fn failed_copy_releases_image_allocation() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let bytes = vec![1; 176];
            let prepared = image(&bytes);
            let mut limited = HbmPvMemory::new(architecture, 0, 0);
            let initial_spans = limited.allocator().spans().to_vec();
            assert!(matches!(
                load_image(&mut limited, &prepared),
                Err(DeviceBinaryLoadError::Memory(HbmPvMemoryError::Store(_)))
            ));
            assert_eq!(limited.allocator().spans(), initial_spans);
            assert_eq!(limited.store().page_count(), 0);
        }
    }
}
