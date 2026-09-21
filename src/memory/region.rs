use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryRegion {
    allocation_bytes: u64,
    known_prefix: Vec<u8>,
}

impl MemoryRegion {
    pub fn new(allocation_bytes: u64, known_prefix: Vec<u8>) -> Result<Self, MemoryRegionError> {
        if known_prefix.len() as u64 > allocation_bytes {
            return Err(MemoryRegionError::InitializedBytesExceedAllocation);
        }
        Ok(Self {
            allocation_bytes,
            known_prefix,
        })
    }

    pub fn unknown(allocation_bytes: u64) -> Self {
        Self {
            allocation_bytes,
            known_prefix: Vec::new(),
        }
    }

    pub const fn allocation_bytes(&self) -> u64 {
        self.allocation_bytes
    }

    pub fn known_prefix(&self) -> &[u8] {
        &self.known_prefix
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum MemoryRegionError {
    #[error("initialized bytes exceed region allocation")]
    InitializedBytesExceedAllocation,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regions_preserve_known_prefix_and_unknown_tail() {
        let region = MemoryRegion::new(8, vec![1, 2, 3]).unwrap();
        assert_eq!(region.allocation_bytes(), 8);
        assert_eq!(region.known_prefix(), [1, 2, 3]);
        assert_eq!(
            MemoryRegion::new(2, vec![1, 2, 3]),
            Err(MemoryRegionError::InitializedBytesExceedAllocation)
        );
    }
}
