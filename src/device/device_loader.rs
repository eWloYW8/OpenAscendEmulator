use serde::Serialize;
use thiserror::Error;

use crate::device::architecture::Architecture;
use crate::device::device_elf::{
    DeviceElf, DeviceElfError, DeviceGlobalAddresses, DeviceKernelSummary, PreparedDeviceLoadImage,
};
use crate::memory::hbm::HbmResolveError;
use crate::memory::hbm_pv_memory::{HbmPvMemory, HbmPvMemoryError};

const CODE_ALIGNMENT: u64 = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct DeviceBinaryPlacement {
    pub architecture: Architecture,
    pub aligned_code_address: u64,
    pub allocation_base: u64,
    pub allocation_bytes: u64,
    pub source_file_offset: u64,
    pub image_bytes: u64,
    pub global_patch_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LoadedDeviceKernel {
    placement: DeviceBinaryPlacement,
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
        if span.allocation_base != self.placement.allocation_base {
            return Err(DeviceKernelFetchError::ImageAllocationChanged);
        }
        let mut bytes = [0; 16];
        memory.device_to_host(pc, &mut bytes[..usize::from(requested_bytes)])?;
        let image_offset = pc
            .checked_sub(self.placement.aligned_code_address)
            .ok_or(DeviceKernelFetchError::ImageAllocationChanged)?;
        let copied_image_bytes = self
            .placement
            .image_bytes
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
        if memory.allocator().architecture() != self.placement.architecture {
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
        if span.allocation_base != self.placement.allocation_base {
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
    #[error("the prepared image is empty")]
    InvalidImageSize,
    #[error("the prepared image declares {declared} bytes but contains {actual} bytes")]
    ImageSummaryMismatch { declared: u64, actual: usize },
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
    global_addresses: DeviceGlobalAddresses,
) -> Result<LoadedDeviceKernel, DeviceKernelLoadError> {
    let projected = elf.project_kernel(kernel_name, CODE_ALIGNMENT)?;
    let prepared = elf.prepare_load_image(global_addresses)?;
    let placement = load_prepared_image(memory, &prepared)?;
    let addresses = (|| {
        let entry = projected
            .summary
            .function_device_address(placement.aligned_code_address)?;
        let section = placement
            .aligned_code_address
            .checked_add(projected.summary.section_virtual_address)
            .ok_or(DeviceElfError::RuntimeAddressOverflow)?;
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
        kernel: projected.summary,
        entry_address,
        executable_start_address,
    })
}

pub fn load_prepared_image(
    memory: &mut HbmPvMemory,
    image: &PreparedDeviceLoadImage,
) -> Result<DeviceBinaryPlacement, DeviceBinaryLoadError> {
    let image_bytes = validate_image(image)?;
    let allocation_bytes = image_bytes
        .checked_add(CODE_ALIGNMENT - 1)
        .and_then(|bytes| bytes.checked_add(15))
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
        memory.host_to_device(aligned_code_address, &image.bytes)?;
        Ok(DeviceBinaryPlacement {
            architecture: memory.allocator().architecture(),
            aligned_code_address,
            allocation_base,
            allocation_bytes,
            source_file_offset: image.summary.file_offset,
            image_bytes,
            global_patch_count: image.patches.len(),
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

fn validate_image(image: &PreparedDeviceLoadImage) -> Result<u64, DeviceBinaryLoadError> {
    let image_bytes = u64::try_from(image.bytes.len())
        .map_err(|_| DeviceBinaryLoadError::AllocationSizeOverflow)?;
    if image_bytes == 0 {
        return Err(DeviceBinaryLoadError::InvalidImageSize);
    }
    if image.summary.byte_count != image_bytes {
        return Err(DeviceBinaryLoadError::ImageSummaryMismatch {
            declared: image.summary.byte_count,
            actual: image.bytes.len(),
        });
    }
    Ok(image_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::device_elf::DeviceLoadImageSummary;
    use crate::execution::machine::{ScalarInstructionStep, ScalarMachine, ScalarMemoryBus};
    use crate::execution::stepper::{ScalarProgramStop, ScalarStepper};
    use crate::execution::vec_queue_c310::{C310VfQueueDisposition, C310VfQueueStep};
    use crate::memory::c220_scalar_address_space::C220ScalarAddressSpace;
    use crate::memory::hbm::DEFAULT_HBM_BASE;
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

        fn enqueue_c310_vf(
            &mut self,
            step: C310VfQueueStep,
        ) -> Result<C310VfQueueDisposition, Self::Error> {
            self.steps.push(step);
            Ok(C310VfQueueDisposition::Accepted)
        }
    }

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

    fn loaded_program(
        architecture: Architecture,
        words: &[u32],
    ) -> (HbmPvMemory, LoadedDeviceKernel) {
        let bytes: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        let byte_count = bytes.len() as u64;
        let prepared = image(bytes);
        let mut memory = HbmPvMemory::new(architecture, 0, 1);
        let placement = load_prepared_image(&mut memory, &prepared).unwrap();
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
    fn loaded_scalar_program_runs_until_end_and_resumes_after_budget() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let words = [0x0706_0001_u32, 0x4160_0000];
            let (mut memory, kernel) = loaded_program(architecture, &words);
            let entry = kernel.entry_address();
            let mut bus = NoMemoryBus;
            let mut stepper =
                ScalarStepper::new(ScalarMachine::new(architecture, [0; 32], 0), entry);
            let empty = stepper
                .run_loaded(&kernel, &mut memory, &mut bus, 0)
                .unwrap();
            assert_eq!(empty.stop, ScalarProgramStop::StepBudgetReached);
            assert!(empty.steps.is_empty());
            assert_eq!(empty.next_pc, entry);
            assert_eq!(stepper.pc(), entry);
            let run = stepper
                .run_loaded(&kernel, &mut memory, &mut bus, 2)
                .unwrap();
            assert_eq!(run.start_pc, entry);
            assert_eq!(run.next_pc, entry + 8);
            assert_eq!(run.steps.len(), 2);
            assert_eq!(run.steps[0].word, words[0]);
            assert_eq!(run.steps[1].word, words[1]);
            assert!(run.steps[1].halted_after);
            assert_eq!(run.stop, ScalarProgramStop::Halted);
            assert!(stepper.is_halted());
            assert!(
                stepper
                    .run_loaded(&kernel, &mut memory, &mut bus, 0)
                    .unwrap()
                    .steps
                    .is_empty()
            );

            let mut resumed =
                ScalarStepper::new(ScalarMachine::new(architecture, [0; 32], 0), entry);
            let head = resumed
                .run_loaded(&kernel, &mut memory, &mut bus, 1)
                .unwrap();
            assert_eq!(head.stop, ScalarProgramStop::StepBudgetReached);
            assert_eq!(head.steps.len(), 1);
            assert_eq!(head.steps[0].word, words[0]);
            assert_eq!(head.next_pc, entry + 4);
            assert_eq!(resumed.pc(), entry + 4);
            let tail = resumed
                .run_loaded(&kernel, &mut memory, &mut bus, 1)
                .unwrap();
            assert_eq!(tail.start_pc, entry + 4);
            assert_eq!(tail.steps.len(), 1);
            assert!(tail.steps[0].halted_after);
            assert_eq!(tail.stop, ScalarProgramStop::Halted);

            memory
                .host_to_device(entry + 4, &0x6000_0000_u32.to_le_bytes())
                .unwrap();
            let mut failing =
                ScalarStepper::new(ScalarMachine::new(architecture, [0; 32], 0), entry);
            let mut observed = Vec::new();
            assert!(
                failing
                    .run_loaded_with(&kernel, &mut memory, &mut bus, 2, |step| {
                        observed.push(step);
                    })
                    .is_err()
            );
            assert_eq!(observed.len(), 1);
            assert_eq!(observed[0].word, words[0]);
            assert_eq!(failing.pc(), entry + 4);
        }
    }

    #[test]
    fn loaded_c310_vf_pair_consumes_one_step_and_eight_code_bytes() {
        let words = [0x154d_0000, 0x15e0_0105, 0x4160_0000];
        let (mut memory, kernel) = loaded_program(Architecture::Dav3510, &words);
        let entry = kernel.entry_address();
        let mut machine = ScalarMachine::new(Architecture::Dav3510, [0; 32], 0);
        machine.set_xreg(13, 0x10d0_d900).unwrap();
        let mut stepper = ScalarStepper::new(machine, entry);
        let mut bus = VfQueueBus { steps: Vec::new() };
        let first = stepper
            .run_loaded(&kernel, &mut memory, &mut bus, 1)
            .unwrap();
        assert_eq!(first.stop, ScalarProgramStop::StepBudgetReached);
        assert_eq!(first.steps.len(), 1);
        assert_eq!(first.steps[0].word, words[0]);
        assert_eq!(first.steps[0].next_pc, entry + 8);
        assert_eq!(bus.steps.len(), 1);
        assert_eq!(stepper.pc(), entry + 8);
        let tail = stepper
            .run_loaded(&kernel, &mut memory, &mut bus, 1)
            .unwrap();
        assert_eq!(tail.stop, ScalarProgramStop::Halted);
        assert_eq!(tail.steps[0].word, words[2]);
        assert_eq!(tail.next_pc, entry + 12);
    }

    #[test]
    fn unified_loaded_program_reads_and_writes_live_hbm_on_both_architectures() {
        for (architecture, pair_word, store_word) in [
            (Architecture::Dav2201, 0x09ca_0180, 0x04c6_5000),
            (Architecture::Dav3510, 0x0cca_0180, 0x03c6_5000),
        ] {
            let words = [pair_word, store_word, 0x4160_0000];
            let (mut memory, kernel) = loaded_program(architecture, &words);
            let args_address = memory.allocate(16).unwrap();
            let output_address = memory.allocate(16).unwrap();
            let expected = 0x1234_5678_9abc_def0_u64;
            let arguments = [
                output_address.to_le_bytes().as_slice(),
                expected.to_le_bytes().as_slice(),
            ]
            .concat();
            memory.host_to_device(args_address, &arguments).unwrap();
            let mut machine = ScalarMachine::new(architecture, [0; 32], 0);
            machine.set_xreg(0, args_address).unwrap();
            let mut stepper = ScalarStepper::new(machine, kernel.entry_address());
            let run = stepper.run_loaded_unified(&kernel, &mut memory, 3).unwrap();
            assert_eq!(run.stop, ScalarProgramStop::Halted);
            assert_eq!(run.steps.len(), 3);
            assert!(matches!(
                run.steps[0].instruction,
                ScalarInstructionStep::PairLoad(_)
            ));
            assert!(matches!(
                run.steps[1].instruction,
                ScalarInstructionStep::Memory(_)
            ));
            let mut actual = [0; 8];
            memory.device_to_host(output_address, &mut actual).unwrap();
            assert_eq!(actual, expected.to_le_bytes());
        }
    }

    #[test]
    fn mapped_loaded_program_writes_c220_local_buffer_without_corrupting_code() {
        let (hbm, kernel) = loaded_program(Architecture::Dav2201, &[0x04c6_7000, 0x4160_0000]);
        let mut memory = C220ScalarAddressSpace::new(hbm, 0, 0).unwrap();
        let mut machine = ScalarMachine::new(Architecture::Dav2201, [0; 32], 0);
        machine.set_xreg(3, 0x1234_5678_9abc_def0).unwrap();
        machine.set_xreg(7, 0x100000).unwrap();
        let mut stepper = ScalarStepper::new(machine, kernel.entry_address());
        let run = stepper.run_loaded_mapped(&kernel, &mut memory, 2).unwrap();
        assert_eq!(run.stop, ScalarProgramStop::Halted);
        let mut actual = [0; 8];
        ScalarMemoryBus::read(&mut memory, 0x100000, &mut actual).unwrap();
        assert_eq!(actual, 0x1234_5678_9abc_def0_u64.to_le_bytes());
        assert_eq!(
            kernel
                .fetch_executable_word(memory.hbm_mut(), kernel.entry_address())
                .unwrap(),
            0x04c6_7000
        );
    }

    #[test]
    fn copies_aligned_image_bytes_for_both_architectures() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut memory = HbmPvMemory::new(architecture, 0xa5, 1);
            memory.allocate(512).unwrap();
            let prepared = image((0..176).map(|value| value as u8).collect());
            let placement = load_prepared_image(&mut memory, &prepared).unwrap();
            assert_eq!(placement.aligned_code_address, DEFAULT_HBM_BASE + 4096);
            assert_eq!(placement.allocation_base, DEFAULT_HBM_BASE + 512);
            assert_eq!(placement.allocation_bytes, 176 + CODE_ALIGNMENT - 1 + 15);
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
    fn rejects_mismatched_or_empty_image_before_allocating() {
        let mut memory = HbmPvMemory::new(Architecture::Dav2201, 0, 1);
        let mut prepared = image(vec![1; 16]);
        prepared.summary.byte_count = 15;
        assert_eq!(
            load_prepared_image(&mut memory, &prepared),
            Err(DeviceBinaryLoadError::ImageSummaryMismatch {
                declared: 15,
                actual: 16
            })
        );
        prepared = image(Vec::new());
        assert_eq!(
            load_prepared_image(&mut memory, &prepared),
            Err(DeviceBinaryLoadError::InvalidImageSize)
        );
        assert_eq!(memory.allocator().spans().len(), 1);
        assert_eq!(memory.store().page_count(), 0);
    }

    #[test]
    fn failed_copy_releases_image_allocation() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let prepared = image(vec![1; 176]);
            let mut limited = HbmPvMemory::new(architecture, 0, 0);
            let initial_spans = limited.allocator().spans().to_vec();
            assert!(matches!(
                load_prepared_image(&mut limited, &prepared),
                Err(DeviceBinaryLoadError::Memory(HbmPvMemoryError::Store(_)))
            ));
            assert_eq!(limited.allocator().spans(), initial_spans);
            assert_eq!(limited.store().page_count(), 0);
        }
    }
}
