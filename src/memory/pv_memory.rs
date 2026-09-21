use std::collections::BTreeMap;

use thiserror::Error;

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
}

impl PvPage {
    fn new(default_byte: u8) -> Self {
        Self {
            bytes: vec![default_byte; PV_PAGE_BYTES].into_boxed_slice(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PvMemory {
    default_byte: u8,
    max_pages: usize,
    pages: BTreeMap<u64, PvPage>,
}

impl PvMemory {
    pub fn new(default_byte: u8, max_pages: usize) -> Self {
        Self {
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

    pub fn read_into(&self, address: u64, destination: &mut [u8]) -> Result<(), PvMemoryError> {
        self.check_range(address, destination.len())?;
        let mut done = 0;
        while done < destination.len() {
            let at = address + done as u64;
            let base = at & PV_PAGE_MASK;
            let offset = (at - base) as usize;
            let count = (PV_PAGE_BYTES - offset).min(destination.len() - done);
            if let Some(page) = self.pages.get(&base) {
                destination[done..done + count]
                    .copy_from_slice(&page.bytes[offset..offset + count]);
            } else {
                destination[done..done + count].fill(self.default_byte);
            }
            done += count;
        }
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
        });
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
        if length == 0 {
            return Ok(());
        }
        let final_address = address + length as u64 - 1;
        let mut base = address & PV_PAGE_MASK;
        let last_base = final_address & PV_PAGE_MASK;
        let available = self.max_pages.saturating_sub(self.pages.len());
        let mut missing = 0;
        loop {
            if !self.pages.contains_key(&base) {
                missing += 1;
                if missing > available {
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
    fn missing_bulk_read_returns_default_without_allocating() {
        let memory = PvMemory::new(0xa5, 0);
        let mut result = [0; 4];
        memory
            .read_into(PV_PAGE_BYTES as u64 - 2, &mut result)
            .unwrap();
        assert_eq!(result, [0xa5; 4]);
        assert_eq!(memory.page_count(), 0);
        assert_eq!(memory.read_byte(0), 0xa5);
    }

    #[test]
    fn cross_page_write_changes_only_addressed_bytes() {
        let mut memory = PvMemory::new(0x7f, 2);
        let address = PV_PAGE_BYTES as u64 - 1;
        memory.write(address, &[3, 4, 5]).unwrap();
        let mut result = [0; 5];
        memory.read_into(address - 1, &mut result).unwrap();
        assert_eq!(result, [0x7f, 3, 4, 5, 0x7f]);
    }

    #[test]
    fn rejected_ranges_and_page_limits_preserve_state() {
        let mut memory = PvMemory::new(0, 1);
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
        memory
            .read_into(PV_PAGE_BYTES as u64 - 1, &mut bytes)
            .unwrap();
        assert_eq!(bytes, [0; 2]);
        assert_eq!(memory.page_count(), 0);
    }
}
