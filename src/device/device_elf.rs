use thiserror::Error;

const ELF_HEADER_SIZE: usize = 64;
const SECTION_HEADER_SIZE: usize = 64;
const SYMBOL_SIZE: usize = 24;
const ASCEND_MACHINE: u16 = 0x1029;
const SHT_PROGBITS: u32 = 1;
const SHT_SYMTAB: u32 = 2;
const SHT_STRTAB: u32 = 3;
const SHF_ALLOC: u64 = 2;
const SHF_EXECINSTR: u64 = 4;
const STT_FUNC: u8 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceElfHeader {
    pub file_type: u16,
    pub machine: u16,
    pub flags: u32,
    pub section_count: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceKernelSummary {
    pub name: String,
    pub section: String,
    pub section_virtual_address: u64,
    pub section_file_offset: u64,
    pub section_byte_count: u64,
    pub virtual_address: u64,
    pub file_offset: u64,
    pub byte_count: u64,
}

impl DeviceKernelSummary {
    pub fn device_address(&self, image_base: u64) -> Result<u64, DeviceElfError> {
        image_base
            .checked_add(self.virtual_address)
            .ok_or(DeviceElfError::DeviceAddressOverflow)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceKernel<'a> {
    pub summary: DeviceKernelSummary,
    pub bytes: &'a [u8],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceLoadImage<'a> {
    pub file_offset: u64,
    pub bytes: &'a [u8],
}

impl DeviceKernel<'_> {
    pub fn words(&self) -> impl Iterator<Item = (u64, u32)> + '_ {
        self.bytes
            .chunks_exact(4)
            .enumerate()
            .map(|(index, chunk)| {
                let pc = self.summary.virtual_address + (index as u64) * 4;
                let word = u32::from_le_bytes(chunk.try_into().expect("four-byte chunk"));
                (pc, word)
            })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SectionHeader {
    name_offset: u32,
    section_type: u32,
    flags: u64,
    address: u64,
    file_offset: u64,
    byte_count: u64,
    link: u32,
    entry_size: u64,
}

#[derive(Debug, Clone)]
pub struct DeviceElf<'a> {
    bytes: &'a [u8],
    header: DeviceElfHeader,
    sections: Vec<SectionHeader>,
    section_names: Vec<String>,
}

impl<'a> DeviceElf<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self, DeviceElfError> {
        if bytes.len() < ELF_HEADER_SIZE {
            return Err(DeviceElfError::TooShort);
        }
        if &bytes[..4] != b"\x7fELF" || bytes[4] != 2 || bytes[5] != 1 || bytes[6] != 1 {
            return Err(DeviceElfError::UnsupportedFormat);
        }
        if read_u32(bytes, 20)? != 1 || read_u16(bytes, 52)? as usize != ELF_HEADER_SIZE {
            return Err(DeviceElfError::UnsupportedFormat);
        }
        let file_type = read_u16(bytes, 16)?;
        if file_type != 1 && file_type != 2 {
            return Err(DeviceElfError::UnsupportedFileType(file_type));
        }
        let machine = read_u16(bytes, 18)?;
        if machine != ASCEND_MACHINE {
            return Err(DeviceElfError::UnexpectedMachine(machine));
        }
        let section_table_offset = read_u64(bytes, 40)?;
        let flags = read_u32(bytes, 48)?;
        let section_header_size = read_u16(bytes, 58)?;
        let section_count = read_u16(bytes, 60)?;
        let section_names_index = read_u16(bytes, 62)?;
        if section_header_size as usize != SECTION_HEADER_SIZE
            || section_count == 0
            || section_names_index == u16::MAX
            || section_names_index >= section_count
        {
            return Err(DeviceElfError::UnsupportedSectionTable);
        }

        let table_len = u64::from(section_count)
            .checked_mul(SECTION_HEADER_SIZE as u64)
            .ok_or(DeviceElfError::RangeOverflow)?;
        checked_slice(bytes, section_table_offset, table_len)?;
        let mut sections = Vec::with_capacity(usize::from(section_count));
        for index in 0..section_count {
            let offset = section_table_offset
                .checked_add(u64::from(index) * SECTION_HEADER_SIZE as u64)
                .ok_or(DeviceElfError::RangeOverflow)?;
            let record = checked_slice(bytes, offset, SECTION_HEADER_SIZE as u64)?;
            sections.push(SectionHeader {
                name_offset: read_u32(record, 0)?,
                section_type: read_u32(record, 4)?,
                flags: read_u64(record, 8)?,
                address: read_u64(record, 16)?,
                file_offset: read_u64(record, 24)?,
                byte_count: read_u64(record, 32)?,
                link: read_u32(record, 40)?,
                entry_size: read_u64(record, 56)?,
            });
        }
        let names_section = sections[usize::from(section_names_index)];
        if names_section.section_type != SHT_STRTAB {
            return Err(DeviceElfError::InvalidStringTable);
        }
        let names = checked_slice(bytes, names_section.file_offset, names_section.byte_count)?;
        let section_names = sections
            .iter()
            .map(|section| {
                Ok(
                    String::from_utf8_lossy(read_c_string(names, section.name_offset)?)
                        .into_owned(),
                )
            })
            .collect::<Result<Vec<_>, DeviceElfError>>()?;

        Ok(Self {
            bytes,
            header: DeviceElfHeader {
                file_type,
                machine,
                flags,
                section_count,
            },
            sections,
            section_names,
        })
    }

    pub const fn header(&self) -> DeviceElfHeader {
        self.header
    }

    pub fn load_image(&self) -> Result<DeviceLoadImage<'a>, DeviceElfError> {
        let mut first: Option<u64> = None;
        let mut last: Option<(u64, u64)> = None;
        for section in &self.sections {
            if section.flags & SHF_ALLOC == 0 || section.byte_count == 0 {
                continue;
            }
            if first == Some(0) {
                return Err(DeviceElfError::AmbiguousZeroOffsetLoadImage);
            }
            let end = section
                .file_offset
                .checked_add(section.byte_count)
                .ok_or(DeviceElfError::RangeOverflow)?;
            if let Some((previous_offset, previous_end)) = last
                && (section.file_offset < previous_offset || end < previous_end)
            {
                return Err(DeviceElfError::UnorderedLoadImageSections);
            }
            if first.is_none() {
                first = Some(section.file_offset);
            }
            last = Some((section.file_offset, end));
        }
        let file_offset = first.ok_or(DeviceElfError::NoLoadImage)?;
        let (_, end) = last.expect("first section implies last section");
        let byte_count = end
            .checked_sub(file_offset)
            .ok_or(DeviceElfError::UnorderedLoadImageSections)?;
        let bytes = checked_slice(self.bytes, file_offset, byte_count)?;
        Ok(DeviceLoadImage { file_offset, bytes })
    }

    pub fn kernels(&self) -> Result<Vec<DeviceKernelSummary>, DeviceElfError> {
        let mut result = Vec::new();
        for section in &self.sections {
            if section.section_type != SHT_SYMTAB {
                continue;
            }
            if section.entry_size != SYMBOL_SIZE as u64
                || section.byte_count % SYMBOL_SIZE as u64 != 0
            {
                return Err(DeviceElfError::InvalidSymbolTable);
            }
            let string_section = self
                .sections
                .get(section.link as usize)
                .ok_or(DeviceElfError::InvalidSymbolTable)?;
            if string_section.section_type != SHT_STRTAB {
                return Err(DeviceElfError::InvalidSymbolTable);
            }
            let string_bytes = checked_slice(
                self.bytes,
                string_section.file_offset,
                string_section.byte_count,
            )?;
            let symbol_bytes = checked_slice(self.bytes, section.file_offset, section.byte_count)?;
            for symbol in symbol_bytes.chunks_exact(SYMBOL_SIZE) {
                if symbol[4] & 0xf != STT_FUNC || read_u32(symbol, 0)? == 0 {
                    continue;
                }
                let section_index = usize::from(read_u16(symbol, 6)?);
                let Some(code_section) = self.sections.get(section_index) else {
                    continue;
                };
                if code_section.section_type != SHT_PROGBITS
                    || code_section.flags & SHF_EXECINSTR == 0
                {
                    continue;
                }
                let byte_count = read_u64(symbol, 16)?;
                if byte_count == 0 {
                    continue;
                }
                let virtual_address = read_u64(symbol, 8)?;
                virtual_address
                    .checked_add(byte_count)
                    .ok_or(DeviceElfError::RangeOverflow)?;
                let relative = virtual_address
                    .checked_sub(code_section.address)
                    .ok_or(DeviceElfError::InvalidKernelRange)?;
                let relative_end = relative
                    .checked_add(byte_count)
                    .ok_or(DeviceElfError::RangeOverflow)?;
                if relative_end > code_section.byte_count {
                    return Err(DeviceElfError::InvalidKernelRange);
                }
                let file_offset = code_section
                    .file_offset
                    .checked_add(relative)
                    .ok_or(DeviceElfError::RangeOverflow)?;
                checked_slice(self.bytes, file_offset, byte_count)?;
                let name_bytes = read_c_string(string_bytes, read_u32(symbol, 0)?)?;
                result.push(DeviceKernelSummary {
                    name: String::from_utf8_lossy(name_bytes).into_owned(),
                    section: self.section_names[section_index].clone(),
                    section_virtual_address: code_section.address,
                    section_file_offset: code_section.file_offset,
                    section_byte_count: code_section.byte_count,
                    virtual_address,
                    file_offset,
                    byte_count,
                });
            }
        }
        Ok(result)
    }

    pub fn kernel(&self, name: &str) -> Result<DeviceKernel<'a>, DeviceElfError> {
        let mut matches = self.kernels()?.into_iter().filter(|item| item.name == name);
        let summary = matches
            .next()
            .ok_or_else(|| DeviceElfError::KernelNotFound(name.to_owned()))?;
        if matches.next().is_some() {
            return Err(DeviceElfError::AmbiguousKernel(name.to_owned()));
        }
        if summary.byte_count % 4 != 0 {
            return Err(DeviceElfError::UnalignedKernelSize(summary.byte_count));
        }
        let bytes = checked_slice(self.bytes, summary.file_offset, summary.byte_count)?;
        Ok(DeviceKernel { summary, bytes })
    }

    pub fn loadable_kernel(&self, name: &str) -> Result<DeviceKernelSummary, DeviceElfError> {
        let image = self.load_image()?;
        let kernel = self.kernel(name)?;
        if !kernel.summary.virtual_address.is_multiple_of(4) {
            return Err(DeviceElfError::UnalignedInstructionAddress(
                kernel.summary.virtual_address,
            ));
        }
        let image_relative_file_offset = kernel
            .summary
            .file_offset
            .checked_sub(image.file_offset)
            .ok_or(DeviceElfError::NonlinearKernelCopyMapping)?;
        if image_relative_file_offset != kernel.summary.virtual_address {
            return Err(DeviceElfError::NonlinearKernelCopyMapping);
        }
        let section_relative_file_offset = kernel
            .summary
            .section_file_offset
            .checked_sub(image.file_offset)
            .ok_or(DeviceElfError::NonlinearKernelCopyMapping)?;
        if section_relative_file_offset != kernel.summary.section_virtual_address {
            return Err(DeviceElfError::NonlinearKernelCopyMapping);
        }
        checked_slice(
            image.bytes,
            section_relative_file_offset,
            kernel.summary.section_byte_count,
        )
        .map_err(|_| DeviceElfError::KernelOutsideLoadImage)?;
        checked_slice(
            image.bytes,
            image_relative_file_offset,
            kernel.summary.byte_count,
        )
        .map_err(|_| DeviceElfError::KernelOutsideLoadImage)?;
        Ok(kernel.summary)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum DeviceElfError {
    #[error("device ELF header is shorter than 64 bytes")]
    TooShort,
    #[error("expected little-endian ELF64 version 1 with a 64-byte header")]
    UnsupportedFormat,
    #[error("unsupported device ELF file type {0}")]
    UnsupportedFileType(u16),
    #[error("unexpected ELF machine {0:#x}, expected 0x1029")]
    UnexpectedMachine(u16),
    #[error("extended or malformed section table is unsupported")]
    UnsupportedSectionTable,
    #[error("ELF byte range is outside the file")]
    OutOfBounds,
    #[error("ELF byte range arithmetic overflow")]
    RangeOverflow,
    #[error("ELF string table is malformed")]
    InvalidStringTable,
    #[error("ELF symbol table is malformed")]
    InvalidSymbolTable,
    #[error("ELF function symbol is outside its executable section")]
    InvalidKernelRange,
    #[error("device function address overflows 64 bits")]
    DeviceAddressOverflow,
    #[error("kernel symbol does not map linearly into the copied device image")]
    NonlinearKernelCopyMapping,
    #[error("kernel symbol bytes fall outside the copied device image")]
    KernelOutsideLoadImage,
    #[error("device instruction address {0:#x} is not four-byte aligned")]
    UnalignedInstructionAddress(u64),
    #[error("device ELF has no nonempty SHF_ALLOC section to load")]
    NoLoadImage,
    #[error("SHF_ALLOC sections are not ordered by increasing file span")]
    UnorderedLoadImageSections,
    #[error("a zero-offset SHF_ALLOC section is followed by another loadable section")]
    AmbiguousZeroOffsetLoadImage,
    #[error("kernel symbol has {0} bytes, not a whole number of 32-bit words")]
    UnalignedKernelSize(u64),
    #[error("kernel symbol {0:?} was not found")]
    KernelNotFound(String),
    #[error("kernel symbol {0:?} is ambiguous")]
    AmbiguousKernel(String),
}

fn checked_slice(bytes: &[u8], offset: u64, length: u64) -> Result<&[u8], DeviceElfError> {
    let start = usize::try_from(offset).map_err(|_| DeviceElfError::RangeOverflow)?;
    let length = usize::try_from(length).map_err(|_| DeviceElfError::RangeOverflow)?;
    let end = start
        .checked_add(length)
        .ok_or(DeviceElfError::RangeOverflow)?;
    bytes.get(start..end).ok_or(DeviceElfError::OutOfBounds)
}

fn read_c_string(bytes: &[u8], offset: u32) -> Result<&[u8], DeviceElfError> {
    let rest = bytes
        .get(offset as usize..)
        .ok_or(DeviceElfError::InvalidStringTable)?;
    let length = rest
        .iter()
        .position(|byte| *byte == 0)
        .ok_or(DeviceElfError::InvalidStringTable)?;
    Ok(&rest[..length])
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, DeviceElfError> {
    let field = bytes
        .get(offset..offset + 2)
        .ok_or(DeviceElfError::OutOfBounds)?;
    Ok(u16::from_le_bytes(
        field.try_into().expect("two-byte field"),
    ))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, DeviceElfError> {
    let field = bytes
        .get(offset..offset + 4)
        .ok_or(DeviceElfError::OutOfBounds)?;
    Ok(u32::from_le_bytes(
        field.try_into().expect("four-byte field"),
    ))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, DeviceElfError> {
    let field = bytes
        .get(offset..offset + 8)
        .ok_or(DeviceElfError::OutOfBounds)?;
    Ok(u64::from_le_bytes(
        field.try_into().expect("eight-byte field"),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::architecture::Architecture;
    use crate::device::device_loader::{
        DeviceKernelFetchError, DeviceKernelLoadError, load_named_kernel,
    };
    use crate::execution::machine::{ScalarInstructionStep, ScalarMachine, ScalarMemoryBus};
    use crate::execution::stepper::{ScalarStepper, ScalarStepperError};
    use crate::memory::hbm_pv_memory::HbmPvMemory;

    struct RejectMemoryBus;

    impl ScalarMemoryBus for RejectMemoryBus {
        type Error = std::io::Error;

        fn read(&mut self, _address: u64, _destination: &mut [u8]) -> Result<(), Self::Error> {
            Err(std::io::Error::other("unexpected data read"))
        }

        fn write(&mut self, _address: u64, _source: &[u8]) -> Result<(), Self::Error> {
            Err(std::io::Error::other("unexpected data write"))
        }
    }

    fn put_u16(bytes: &mut [u8], offset: usize, value: u16) {
        bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
        bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    fn section(bytes: &mut [u8], index: usize, name: u32, kind: u32, offset: u64, size: u64) {
        let base = 0x200 + index * SECTION_HEADER_SIZE;
        put_u32(bytes, base, name);
        put_u32(bytes, base + 4, kind);
        put_u64(bytes, base + 24, offset);
        put_u64(bytes, base + 32, size);
    }

    fn fixture() -> Vec<u8> {
        let mut bytes = vec![0; 0x400];
        bytes[0..4].copy_from_slice(b"\x7fELF");
        bytes[4] = 2;
        bytes[5] = 1;
        bytes[6] = 1;
        put_u16(&mut bytes, 16, 2);
        put_u16(&mut bytes, 18, ASCEND_MACHINE);
        put_u32(&mut bytes, 20, 1);
        put_u64(&mut bytes, 40, 0x200);
        put_u32(&mut bytes, 48, 0x930000);
        put_u16(&mut bytes, 52, ELF_HEADER_SIZE as u16);
        put_u16(&mut bytes, 58, SECTION_HEADER_SIZE as u16);
        put_u16(&mut bytes, 60, 5);
        put_u16(&mut bytes, 62, 3);

        bytes[0x100..0x108].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let shstr = b"\0.text\0.symtab\0.shstrtab\0.strtab\0";
        bytes[0x110..0x110 + shstr.len()].copy_from_slice(shstr);
        bytes[0x1b0..0x1b8].copy_from_slice(b"\0Kernel\0");

        section(&mut bytes, 1, 1, SHT_PROGBITS, 0x100, 8);
        put_u64(&mut bytes, 0x200 + SECTION_HEADER_SIZE + 8, SHF_EXECINSTR);
        section(&mut bytes, 2, 7, SHT_SYMTAB, 0x180, 48);
        put_u32(&mut bytes, 0x200 + 2 * SECTION_HEADER_SIZE + 40, 4);
        put_u64(
            &mut bytes,
            0x200 + 2 * SECTION_HEADER_SIZE + 56,
            SYMBOL_SIZE as u64,
        );
        section(&mut bytes, 3, 15, SHT_STRTAB, 0x110, shstr.len() as u64);
        section(&mut bytes, 4, 25, SHT_STRTAB, 0x1b0, 8);

        let symbol = 0x180 + SYMBOL_SIZE;
        put_u32(&mut bytes, symbol, 1);
        bytes[symbol + 4] = 0x12;
        put_u16(&mut bytes, symbol + 6, 1);
        put_u64(&mut bytes, symbol + 16, 8);
        bytes
    }

    #[test]
    fn extracts_function_bytes_and_little_endian_words() {
        let bytes = fixture();
        let elf = DeviceElf::parse(&bytes).unwrap();
        assert_eq!(elf.header().machine, ASCEND_MACHINE);
        assert_eq!(elf.header().flags, 0x930000);
        let kernel = elf.kernel("Kernel").unwrap();
        assert_eq!(kernel.summary.section, ".text");
        assert_eq!(kernel.summary.file_offset, 0x100);
        assert_eq!(kernel.bytes, &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(kernel.summary.device_address(0x1000).unwrap(), 0x1000);
        assert_eq!(
            kernel.words().collect::<Vec<_>>(),
            vec![(0, 0x04030201), (4, 0x08070605)]
        );
    }

    #[test]
    fn accepts_named_function_symbols_regardless_of_binding_or_visibility() {
        let mut bytes = fixture();
        let symbol = 0x180 + SYMBOL_SIZE;
        bytes[symbol + 4] = 0x02;
        assert_eq!(
            DeviceElf::parse(&bytes).unwrap().kernels().unwrap().len(),
            1
        );
        bytes[symbol + 4] = 0x22;
        assert_eq!(
            DeviceElf::parse(&bytes).unwrap().kernels().unwrap().len(),
            1
        );
        bytes[symbol + 4] = 0x12;
        bytes[symbol + 5] = 2;
        assert_eq!(
            DeviceElf::parse(&bytes).unwrap().kernels().unwrap().len(),
            1
        );
        bytes[symbol + 5] = 3;
        assert_eq!(
            DeviceElf::parse(&bytes).unwrap().kernels().unwrap().len(),
            1
        );
        put_u32(&mut bytes, symbol, 0);
        assert!(
            DeviceElf::parse(&bytes)
                .unwrap()
                .kernels()
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn computes_bounded_device_addresses_without_runtime_field_limits() {
        let bytes = fixture();
        let mut kernel = DeviceElf::parse(&bytes)
            .unwrap()
            .kernels()
            .unwrap()
            .remove(0);
        kernel.virtual_address = 0x1234;
        assert_eq!(kernel.device_address(0x2000).unwrap(), 0x3234);
        kernel.virtual_address = u64::from(u32::MAX) + 1;
        assert_eq!(kernel.device_address(0x2000), Ok(0x1_0000_2000));
        kernel.virtual_address = 0x1000;
        assert_eq!(
            kernel.device_address(u64::MAX - 4095),
            Err(DeviceElfError::DeviceAddressOverflow)
        );
    }

    #[test]
    fn named_kernel_load_fetches_current_device_bytes_on_both_architectures() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut bytes = fixture();
            put_u64(
                &mut bytes,
                0x200 + SECTION_HEADER_SIZE + 8,
                SHF_ALLOC | SHF_EXECINSTR,
            );
            put_u64(&mut bytes, 0x200 + SECTION_HEADER_SIZE + 32, 20);
            put_u32(&mut bytes, 0x100, 0x073a_7f80);
            put_u32(&mut bytes, 0x104, 0x077b_0010);
            put_u32(&mut bytes, 0x108, 0xc200_001d);
            let elf = DeviceElf::parse(&bytes).unwrap();
            let mut memory = HbmPvMemory::new(architecture, 0, 3);
            let loaded = load_named_kernel(&mut memory, &elf, "Kernel").unwrap();
            let pc = loaded.entry_address();
            assert_eq!(loaded.kernel().name, "Kernel");
            assert_eq!(loaded.placement().aligned_code_address, pc);
            assert_eq!(
                loaded.fetch_executable_word(&mut memory, pc),
                Ok(0x073a_7f80)
            );
            assert_eq!(
                loaded.fetch_executable_word(&mut memory, pc + 4),
                Ok(0x077b_0010)
            );
            assert_eq!(loaded.executable_start_address(), pc);
            assert_eq!(
                loaded.fetch_executable_word(&mut memory, pc + 8),
                Ok(0xc200_001d)
            );
            assert_eq!(
                loaded.fetch_executable_word(&mut memory, pc + 1),
                Err(DeviceKernelFetchError::UnalignedPc(pc + 1))
            );
            assert_eq!(
                loaded.fetch_executable_word(&mut memory, pc + 20),
                Err(DeviceKernelFetchError::OutsideExecutableSection(pc + 20))
            );
            memory
                .host_to_device(pc, &0x0200_4880_u32.to_le_bytes())
                .unwrap();
            let mut scalar = ScalarMachine::new(architecture, [0; 32], 0);
            scalar.set_spr_value(4, 0x1022_be00).unwrap();
            let word = loaded.fetch_executable_word(&mut memory, pc).unwrap();
            let step = scalar.execute_spr_read_word(pc, word).unwrap();
            assert_eq!(step.value, 0x1022_be00);
            assert_eq!(scalar.xregs()[0], 0x1022_be00);
            let mut stepper = ScalarStepper::new(ScalarMachine::new(architecture, [0; 32], 0), pc);
            stepper.machine_mut().set_spr_value(4, 0x1022_be00).unwrap();
            let program_step = stepper
                .step_loaded(&loaded, &mut memory, &mut RejectMemoryBus)
                .unwrap();
            assert_eq!(program_step.word, 0x0200_4880);
            assert_eq!(program_step.next_pc, pc + 4);
            assert!(matches!(
                program_step.instruction,
                ScalarInstructionStep::SprRead(_)
            ));
            assert_eq!(stepper.machine().xregs()[0], 0x1022_be00);
            let mut outside =
                ScalarStepper::new(ScalarMachine::new(architecture, [0; 32], 0), pc + 20);
            assert!(matches!(
                outside.step_loaded(&loaded, &mut memory, &mut RejectMemoryBus),
                Err(ScalarStepperError::Fetch(
                    DeviceKernelFetchError::OutsideExecutableSection(_)
                ))
            ));
            assert_eq!(outside.pc(), pc + 20);
            memory
                .host_to_device(pc + 4, &0x0102_0304_u32.to_le_bytes())
                .unwrap();
            assert_eq!(
                loaded.fetch_executable_word(&mut memory, pc + 4),
                Ok(0x0102_0304)
            );

            let other_architecture = match architecture {
                Architecture::Dav2201 => Architecture::Dav3510,
                Architecture::Dav3510 => Architecture::Dav2201,
            };
            let mut wrong_memory = HbmPvMemory::new(other_architecture, 0, 1);
            assert_eq!(
                loaded.fetch_executable_word(&mut wrong_memory, pc),
                Err(DeviceKernelFetchError::ArchitectureMismatch)
            );
        }
    }

    #[test]
    fn invalid_named_kernel_is_rejected_before_device_allocation() {
        let mut bytes = fixture();
        put_u64(
            &mut bytes,
            0x200 + SECTION_HEADER_SIZE + 8,
            SHF_ALLOC | SHF_EXECINSTR,
        );
        let elf = DeviceElf::parse(&bytes).unwrap();
        let architecture = Architecture::Dav2201;
        let mut memory = HbmPvMemory::new(architecture, 0, 1);
        assert!(matches!(
            load_named_kernel(&mut memory, &elf, "Missing"),
            Err(DeviceKernelLoadError::Elf(DeviceElfError::KernelNotFound(
                _
            )))
        ));
        assert_eq!(memory.allocator().spans().len(), 1);
        assert_eq!(memory.store().page_count(), 0);
    }

    #[test]
    fn loadable_kernel_rejects_a_symbol_that_does_not_match_the_image() {
        let mut bytes = fixture();
        assert_eq!(
            DeviceElf::parse(&bytes).unwrap().loadable_kernel("Kernel"),
            Err(DeviceElfError::NoLoadImage)
        );

        put_u64(
            &mut bytes,
            0x200 + SECTION_HEADER_SIZE + 8,
            SHF_ALLOC | SHF_EXECINSTR,
        );
        put_u64(&mut bytes, 0x200 + SECTION_HEADER_SIZE + 16, 0x100);
        put_u64(&mut bytes, 0x180 + SYMBOL_SIZE + 8, 0x100);
        assert_eq!(
            DeviceElf::parse(&bytes).unwrap().loadable_kernel("Kernel"),
            Err(DeviceElfError::NonlinearKernelCopyMapping)
        );

        put_u64(&mut bytes, 0x200 + SECTION_HEADER_SIZE + 16, 0);
        put_u64(&mut bytes, 0x180 + SYMBOL_SIZE + 8, 1);
        put_u64(&mut bytes, 0x180 + SYMBOL_SIZE + 16, 4);
        assert_eq!(
            DeviceElf::parse(&bytes).unwrap().loadable_kernel("Kernel"),
            Err(DeviceElfError::UnalignedInstructionAddress(1))
        );
    }

    #[test]
    fn extracts_the_contiguous_allocated_span_including_inter_section_gaps() {
        let mut bytes = fixture();
        put_u16(&mut bytes, 60, 6);
        put_u64(
            &mut bytes,
            0x200 + SECTION_HEADER_SIZE + 8,
            SHF_ALLOC | SHF_EXECINSTR,
        );
        section(&mut bytes, 5, 0, SHT_PROGBITS, 0x140, 4);
        put_u64(&mut bytes, 0x200 + 5 * SECTION_HEADER_SIZE + 8, SHF_ALLOC);
        bytes[0x140..0x144].copy_from_slice(&[9, 10, 11, 12]);

        let elf = DeviceElf::parse(&bytes).unwrap();
        let image = elf.load_image().unwrap();
        assert_eq!(image.file_offset, 0x100);
        assert_eq!(image.bytes.len(), 0x44);
        assert_eq!(&image.bytes[..8], &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(&image.bytes[0x40..], &[9, 10, 11, 12]);
    }

    #[test]
    fn load_image_rejects_unproven_or_out_of_bounds_section_layouts() {
        let mut bytes = fixture();
        assert_eq!(
            DeviceElf::parse(&bytes).unwrap().load_image(),
            Err(DeviceElfError::NoLoadImage)
        );

        put_u16(&mut bytes, 60, 6);
        put_u64(
            &mut bytes,
            0x200 + SECTION_HEADER_SIZE + 8,
            SHF_ALLOC | SHF_EXECINSTR,
        );
        section(&mut bytes, 5, 0, SHT_PROGBITS, 0xf0, 4);
        put_u64(&mut bytes, 0x200 + 5 * SECTION_HEADER_SIZE + 8, SHF_ALLOC);
        assert_eq!(
            DeviceElf::parse(&bytes).unwrap().load_image(),
            Err(DeviceElfError::UnorderedLoadImageSections)
        );

        section(&mut bytes, 5, 0, SHT_PROGBITS, 0x3f0, 0x40);
        assert_eq!(
            DeviceElf::parse(&bytes).unwrap().load_image(),
            Err(DeviceElfError::OutOfBounds)
        );

        section(&mut bytes, 5, 0, SHT_PROGBITS, u64::MAX, 2);
        assert_eq!(
            DeviceElf::parse(&bytes).unwrap().load_image(),
            Err(DeviceElfError::RangeOverflow)
        );

        section(&mut bytes, 1, 1, SHT_PROGBITS, 0, 8);
        section(&mut bytes, 5, 0, SHT_PROGBITS, 0x140, 4);
        assert_eq!(
            DeviceElf::parse(&bytes).unwrap().load_image(),
            Err(DeviceElfError::AmbiguousZeroOffsetLoadImage)
        );
    }

    #[test]
    fn rejects_malformed_header_and_symbol_ranges() {
        let mut bytes = fixture();
        assert_eq!(
            DeviceElf::parse(&bytes[..63]).unwrap_err(),
            DeviceElfError::TooShort
        );
        bytes[18] = 0;
        assert_eq!(
            DeviceElf::parse(&bytes).unwrap_err(),
            DeviceElfError::UnexpectedMachine(0x1000)
        );
        bytes = fixture();
        put_u32(&mut bytes, 20, 0);
        assert_eq!(
            DeviceElf::parse(&bytes).unwrap_err(),
            DeviceElfError::UnsupportedFormat
        );
        bytes = fixture();
        put_u16(&mut bytes, 52, 0);
        assert_eq!(
            DeviceElf::parse(&bytes).unwrap_err(),
            DeviceElfError::UnsupportedFormat
        );
        bytes = fixture();
        put_u64(&mut bytes, 0x180 + SYMBOL_SIZE + 16, 12);
        assert_eq!(
            DeviceElf::parse(&bytes).unwrap().kernels(),
            Err(DeviceElfError::InvalidKernelRange)
        );
    }

    #[test]
    fn does_not_accept_host_aarch64_elf_as_device_code() {
        let mut bytes = fixture();
        put_u16(&mut bytes, 18, 183);
        assert_eq!(
            DeviceElf::parse(&bytes).unwrap_err(),
            DeviceElfError::UnexpectedMachine(183)
        );
    }

    #[test]
    fn lists_but_does_not_extract_a_partial_instruction() {
        let mut bytes = fixture();
        put_u64(&mut bytes, 0x180 + SYMBOL_SIZE + 16, 7);
        let elf = DeviceElf::parse(&bytes).unwrap();
        assert_eq!(elf.kernels().unwrap()[0].byte_count, 7);
        assert_eq!(
            elf.kernel("Kernel").unwrap_err(),
            DeviceElfError::UnalignedKernelSize(7)
        );
    }
}
