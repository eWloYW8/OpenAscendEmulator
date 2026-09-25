use super::C220LsuRequestId;
use super::store_buffer::C220LsuStoreError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220LsuDirectStoreEntry {
    pub address: u64,
    pub request: C220LsuRequestId,
    pub write_address: u64,
    bytes: Vec<u8>,
}

impl C220LsuDirectStoreEntry {
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// Non-coalescing external stores. Entries remain resident until their write
/// response; matching responses consume the first entry at the address.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220LsuDirectStoreBuffer {
    line_bytes: usize,
    capacity: usize,
    entries: Vec<C220LsuDirectStoreEntry>,
}

impl C220LsuDirectStoreBuffer {
    pub fn new(line_bytes: usize, capacity: usize) -> Result<Self, C220LsuStoreError> {
        if line_bytes == 0 {
            return Err(C220LsuStoreError::InvalidConfig);
        }
        Ok(Self {
            line_bytes,
            capacity,
            entries: Vec::new(),
        })
    }

    pub fn full(&self) -> bool {
        self.entries.len() >= self.capacity
    }

    pub fn entries(&self) -> &[C220LsuDirectStoreEntry] {
        &self.entries
    }

    pub fn entry(&self, address: u64) -> Option<&C220LsuDirectStoreEntry> {
        self.entries.iter().find(|entry| entry.address == address)
    }

    /// Capture store bytes in a fresh zero-filled line. Address translation and
    /// lower-level request generation are owned by the cache controller.
    pub fn push(
        &mut self,
        address: u64,
        write_address: u64,
        request: C220LsuRequestId,
        offset: usize,
        bytes: &[u8],
    ) -> Result<(), C220LsuStoreError> {
        let end = offset
            .checked_add(bytes.len())
            .filter(|end| *end <= self.line_bytes)
            .ok_or(C220LsuStoreError::InvalidRange)?;
        if self.full() {
            return Err(C220LsuStoreError::Blocked);
        }
        let mut data = vec![0; self.line_bytes];
        data[offset..end].copy_from_slice(bytes);
        self.entries.push(C220LsuDirectStoreEntry {
            address,
            request,
            write_address,
            bytes: data,
        });
        Ok(())
    }

    pub(super) fn complete(
        &mut self,
        address: u64,
        backing: &mut [u8],
    ) -> Result<C220LsuRequestId, C220LsuStoreError> {
        if backing.len() != self.line_bytes {
            return Err(C220LsuStoreError::InvalidRange);
        }
        let index = self
            .entries
            .iter()
            .position(|entry| entry.address == address)
            .ok_or(C220LsuStoreError::MissingEntry)?;
        backing.copy_from_slice(&self.entries[index].bytes);
        Ok(self.entries.remove(index).request)
    }
}
