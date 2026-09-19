
use std::collections::{BTreeMap, BTreeSet};

use thiserror::Error;

use crate::architecture::Architecture;

pub const PV_PAGE_BYTES: usize = 0x10_0000;
const PV_PAGE_MASK: u64 = !(PV_PAGE_BYTES as u64 - 1);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum PvMemoryError {
    #[error("PV_MEM address range overflows u64")]
    RangeOverflow,
    #[error("PV_MEM page limit of {limit} would be exceeded")]
    PageLimit { limit: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PvPage {
    bytes: Box<[u8]>,
    dirty: Box<[u8]>,
}

impl PvPage {
    fn new(default_byte: u8) -> Self {
        Self {
            bytes: vec![default_byte; PV_PAGE_BYTES].into_boxed_slice(),
            dirty: vec![0; PV_PAGE_BYTES].into_boxed_slice(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PvMemory {
    architecture: Architecture,
    default_byte: u8,
    max_pages: usize,
    pages: BTreeMap<u64, PvPage>,
}

impl PvMemory {
    pub fn new(architecture: Architecture, default_byte: u8, max_pages: usize) -> Self {
        Self {
            architecture,
            default_byte,
            max_pages,
            pages: BTreeMap::new(),
        }
    }

    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    pub fn read_byte(&self, address: u64) -> u8 {
        let page_base = address & PV_PAGE_MASK;
        let page_offset = (address - page_base) as usize;
        self.pages
            .get(&page_base)
            .map_or(self.default_byte, |page| page.bytes[page_offset])
    }

    pub fn dirty_byte(&self, address: u64) -> Option<u8> {
        let page_base = address & PV_PAGE_MASK;
        let page_offset = (address - page_base) as usize;
        self.pages
            .get(&page_base)
            .map(|page| page.dirty[page_offset])
    }

    pub fn read_into(&mut self, address: u64, destination: &mut [u8]) -> Result<(), PvMemoryError> {
        self.check_range(address, destination.len())?;
        if self.architecture == Architecture::Dav2201 {
            self.check_page_budget(address, destination.len())?;
        }
        self.for_each_chunk(address, destination.len(), |this, base, offset, range| {
            if this.architecture == Architecture::Dav2201 {
                let page = this
                    .pages
                    .entry(base)
                    .or_insert_with(|| PvPage::new(this.default_byte));
                destination[range.clone()]
                    .copy_from_slice(&page.bytes[offset..offset + range.len()]);
            } else if let Some(page) = this.pages.get(&base) {
                destination[range.clone()]
                    .copy_from_slice(&page.bytes[offset..offset + range.len()]);
            } else {
                destination[range].fill(this.default_byte);
            }
        });
        Ok(())
    }

    pub fn write(&mut self, address: u64, source: &[u8]) -> Result<(), PvMemoryError> {
        self.check_range(address, source.len())?;
        self.check_page_budget(address, source.len())?;
        self.for_each_chunk(address, source.len(), |this, base, offset, range| {
            let page = this
                .pages
                .entry(base)
                .or_insert_with(|| PvPage::new(this.default_byte));
            page.bytes[offset..offset + range.len()].copy_from_slice(&source[range.clone()]);
            page.dirty[offset..offset + range.len()].fill(1);
        });
        Ok(())
    }

    pub fn copy(
        &mut self,
        destination_address: u64,
        source_address: u64,
        length: usize,
    ) -> Result<(), PvMemoryError> {
        self.check_range(destination_address, length)?;
        self.check_range(source_address, length)?;
        if length == 0 {
            return Ok(());
        }
        let source_end = source_address + length as u64 - 1;
        let materialized = if self.architecture == Architecture::Dav2201 {
            &[(source_address, length), (destination_address, length)][..]
        } else {
            &[(destination_address, length)][..]
        };
        self.check_page_budget_for_ranges(materialized)?;

        const SCRATCH_BYTES: usize = 64 * 1024;
        let mut scratch = [0_u8; SCRATCH_BYTES];
        if destination_address > source_address && destination_address <= source_end {
            let mut remaining = length;
            while remaining != 0 {
                let count = remaining.min(SCRATCH_BYTES);
                let offset = remaining - count;
                self.read_into(source_address + offset as u64, &mut scratch[..count])?;
                self.write(destination_address + offset as u64, &scratch[..count])?;
                remaining = offset;
            }
        } else {
            let mut offset = 0;
            while offset < length {
                let count = (length - offset).min(SCRATCH_BYTES);
                self.read_into(source_address + offset as u64, &mut scratch[..count])?;
                self.write(destination_address + offset as u64, &scratch[..count])?;
                offset += count;
            }
        }
        Ok(())
    }

    fn check_range(&self, address: u64, length: usize) -> Result<(), PvMemoryError> {
        if length == 0 {
            return Ok(());
        }
        let length = u64::try_from(length).map_err(|_| PvMemoryError::RangeOverflow)?;
        address
            .checked_add(length - 1)
            .ok_or(PvMemoryError::RangeOverflow)?;
        Ok(())
    }

    fn check_page_budget(&self, address: u64, length: usize) -> Result<(), PvMemoryError> {
        self.check_page_budget_for_ranges(&[(address, length)])
    }

    fn check_page_budget_for_ranges(&self, ranges: &[(u64, usize)]) -> Result<(), PvMemoryError> {
        let mut missing = BTreeSet::new();
        for &(address, length) in ranges {
            if length == 0 {
                continue;
            }
            let final_address = address + length as u64 - 1;
            let mut base = address & PV_PAGE_MASK;
            let last_base = final_address & PV_PAGE_MASK;
            loop {
                if !self.pages.contains_key(&base) {
                    missing.insert(base);
                    if missing.len() > self.max_pages.saturating_sub(self.pages.len()) {
                        return Err(PvMemoryError::PageLimit {
                            limit: self.max_pages,
                        });
                    }
                }
                if base == last_base {
                    break;
                }
                base += PV_PAGE_BYTES as u64;
            }
        }
        Ok(())
    }

    fn for_each_chunk(
        &mut self,
        address: u64,
        length: usize,
        mut action: impl FnMut(&mut Self, u64, usize, std::ops::Range<usize>),
    ) {
        let mut done = 0usize;
        while done < length {
            let at = address + done as u64;
            let base = at & PV_PAGE_MASK;
            let offset = (at - base) as usize;
            let count = (PV_PAGE_BYTES - offset).min(length - done);
            action(self, base, offset, done..done + count);
            done += count;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_bulk_read_materializes_only_on_c220() {
        for (architecture, expected_pages) in
            [(Architecture::Dav2201, 2), (Architecture::Dav3510, 0)]
        {
            let mut memory = PvMemory::new(architecture, 0xa5, 2);
            let mut result = [0; 4];
            memory
                .read_into(PV_PAGE_BYTES as u64 - 2, &mut result)
                .unwrap();
            assert_eq!(result, [0xa5; 4]);
            assert_eq!(memory.page_count(), expected_pages);
            assert_eq!(memory.read_byte(0), 0xa5);
        }
    }

    #[test]
    fn cross_page_write_changes_bytes_and_dirty_flags() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut memory = PvMemory::new(architecture, 0x7f, 2);
            let address = PV_PAGE_BYTES as u64 - 1;
            memory.write(address, &[3, 4, 5]).unwrap();
            let mut result = [0; 5];
            memory.read_into(address - 1, &mut result).unwrap();
            assert_eq!(result, [0x7f, 3, 4, 5, 0x7f]);
            assert_eq!(memory.dirty_byte(address - 1), Some(0));
            assert_eq!(memory.dirty_byte(address), Some(1));
            assert_eq!(memory.dirty_byte(address + 2), Some(1));
        }
    }

    #[test]
    fn rejected_ranges_and_page_limits_preserve_state() {
        let mut memory = PvMemory::new(Architecture::Dav2201, 0, 1);
        assert_eq!(
            memory.write(PV_PAGE_BYTES as u64 - 1, &[1, 2]),
            Err(PvMemoryError::PageLimit { limit: 1 })
        );
        assert_eq!(memory.page_count(), 0);
        assert_eq!(
            memory.write(u64::MAX, &[1, 2]),
            Err(PvMemoryError::RangeOverflow)
        );
        assert_eq!(memory.page_count(), 0);
        let mut bytes = [0; 2];
        assert_eq!(
            memory.read_into(PV_PAGE_BYTES as u64 - 1, &mut bytes),
            Err(PvMemoryError::PageLimit { limit: 1 })
        );
        assert_eq!(memory.page_count(), 0);
    }

    #[test]
    fn overlapping_copy_matches_staged_bytes_on_both_architectures() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut memory = PvMemory::new(architecture, 0, 1);
            let pattern: [u8; 16] = [
                0x01, 0x80, 0x00, 0xff, 0x5a, 0xa5, 0x12, 0x34, 0xde, 0xad, 0xbe, 0xef, 0x7f, 0x20,
                0x09, 0xc3,
            ];
            memory.write(0x10000003, &pattern).unwrap();
            memory.copy(0x10000007, 0x10000003, 16).unwrap();
            let mut result = [0; 20];
            memory.read_into(0x10000003, &mut result).unwrap();
            assert_eq!(&result[..4], &pattern[..4]);
            assert_eq!(&result[4..], &pattern);
        }
    }

    #[test]
    fn overlapping_copy_across_scratch_chunks_keeps_original_source() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut memory = PvMemory::new(architecture, 0, 1);
            let source: Vec<u8> = (0..64 * 1024 + 37)
                .map(|index| (index % 251) as u8)
                .collect();
            memory.write(0x10001000, &source).unwrap();
            memory.copy(0x10001005, 0x10001000, source.len()).unwrap();
            let mut result = vec![0; source.len() + 5];
            memory.read_into(0x10001000, &mut result).unwrap();
            assert_eq!(&result[..5], &source[..5]);
            assert_eq!(&result[5..], source);
        }
    }

    #[test]
    fn copy_page_limit_rejects_without_partial_mutation() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut memory = PvMemory::new(architecture, 0, 1);
            memory.write(0x10000000, &[1, 2, 3, 4]).unwrap();
            let before = memory.clone();
            assert_eq!(
                memory.copy(0x11000000, 0x10000000, 4),
                Err(PvMemoryError::PageLimit { limit: 1 })
            );
            assert_eq!(memory, before);
        }
    }
}
