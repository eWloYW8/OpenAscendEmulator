use crate::device::architecture::Architecture;
use serde::Serialize;
use thiserror::Error;

pub const DEFAULT_HBM_BASE: u64 = 0x1000_0000;
pub const DEFAULT_HBM_BYTES: u64 = 0x5_0000_0000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct HbmSpan {
    pub base: u64,
    pub bytes: u64,
    pub allocated: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct HbmResolvedSpan {
    pub allocation_base: u64,
    pub allocation_bytes: u64,
    pub offset: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum HbmAllocationError {
    #[error("HBM region [{base:#x}, +{bytes}) is empty or overflows the address space")]
    InvalidRegion { base: u64, bytes: u64 },
    #[error("cannot allocate zero bytes")]
    ZeroSize,
    #[error("allocation of {requested} bytes exceeds HBM capacity")]
    TooLarge { requested: u64 },
    #[error("no free HBM span can satisfy {requested} bytes")]
    OutOfMemory { requested: u64 },
    #[error("host metadata allocation failed")]
    HostAllocationFailed,
    #[error("no HBM allocation begins at device pointer {0:#x}")]
    UnknownPointer(u64),
    #[error("HBM allocation at device pointer {0:#x} has already been freed")]
    AlreadyFreed(u64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum HbmResolveError {
    #[error("zero-length HBM range has no allocation identity")]
    ZeroLength,
    #[error("HBM range beginning at {address:#x} overflows")]
    RangeOverflow { address: u64 },
    #[error("HBM range [{address:#x}, {end:#x}) is not within one live allocation")]
    Unmapped { address: u64, end: u64 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HbmAllocator {
    architecture: Architecture,
    region_bytes: u64,
    spans: Vec<HbmSpan>,
}

impl HbmAllocator {
    pub fn new(architecture: Architecture) -> Self {
        Self {
            architecture,
            region_bytes: DEFAULT_HBM_BYTES,
            spans: vec![HbmSpan {
                base: DEFAULT_HBM_BASE,
                bytes: DEFAULT_HBM_BYTES,
                allocated: false,
            }],
        }
    }

    pub fn with_region(
        architecture: Architecture,
        base: u64,
        bytes: u64,
    ) -> Result<Self, HbmAllocationError> {
        if bytes == 0 || base.checked_add(bytes).is_none() {
            return Err(HbmAllocationError::InvalidRegion { base, bytes });
        }
        Ok(Self {
            architecture,
            region_bytes: bytes,
            spans: vec![HbmSpan {
                base,
                bytes,
                allocated: false,
            }],
        })
    }

    pub const fn architecture(&self) -> Architecture {
        self.architecture
    }

    pub fn spans(&self) -> &[HbmSpan] {
        &self.spans
    }

    pub fn allocate(&mut self, bytes: u64) -> Result<u64, HbmAllocationError> {
        if bytes == 0 {
            return Err(HbmAllocationError::ZeroSize);
        }
        if bytes > self.region_bytes {
            return Err(HbmAllocationError::TooLarge { requested: bytes });
        }
        let index = self
            .spans
            .iter()
            .position(|span| !span.allocated && span.bytes >= bytes)
            .ok_or(HbmAllocationError::OutOfMemory { requested: bytes })?;
        let original = self.spans[index];
        if original.bytes != bytes {
            self.spans
                .try_reserve(1)
                .map_err(|_| HbmAllocationError::HostAllocationFailed)?;
            self.spans.insert(
                index + 1,
                HbmSpan {
                    base: original.base + bytes,
                    bytes: original.bytes - bytes,
                    allocated: false,
                },
            );
        }
        self.spans[index].bytes = bytes;
        self.spans[index].allocated = true;
        Ok(original.base)
    }

    pub fn free(&mut self, pointer: u64) -> Result<(), HbmAllocationError> {
        let mut index = self
            .spans
            .iter()
            .position(|span| span.base == pointer)
            .ok_or(HbmAllocationError::UnknownPointer(pointer))?;
        if !self.spans[index].allocated {
            return Err(HbmAllocationError::AlreadyFreed(pointer));
        }
        self.spans[index].allocated = false;
        if index > 0 && !self.spans[index - 1].allocated {
            let bytes = self.spans[index].bytes;
            self.spans[index - 1].bytes += bytes;
            self.spans.remove(index);
            index -= 1;
        }
        if index + 1 < self.spans.len() && !self.spans[index + 1].allocated {
            let bytes = self.spans[index + 1].bytes;
            self.spans[index].bytes += bytes;
            self.spans.remove(index + 1);
        }
        Ok(())
    }

    pub fn resolve_live(
        &self,
        address: u64,
        bytes: u64,
    ) -> Result<HbmResolvedSpan, HbmResolveError> {
        if bytes == 0 {
            return Err(HbmResolveError::ZeroLength);
        }
        let end = address
            .checked_add(bytes)
            .ok_or(HbmResolveError::RangeOverflow { address })?;
        self.spans
            .iter()
            .find(|span| span.allocated && address >= span.base && end <= span.base + span.bytes)
            .map(|span| HbmResolvedSpan {
                allocation_base: span.base,
                allocation_bytes: span.bytes,
                offset: address - span.base,
            })
            .ok_or(HbmResolveError::Unmapped { address, end })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_fit_splits_and_reuses_freed_spans_on_both_architectures() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut hbm = HbmAllocator::new(architecture);
            assert_eq!(hbm.allocate(32), Ok(DEFAULT_HBM_BASE));
            assert_eq!(hbm.allocate(64), Ok(DEFAULT_HBM_BASE + 32));
            assert_eq!(hbm.allocate(16), Ok(DEFAULT_HBM_BASE + 96));
            hbm.free(DEFAULT_HBM_BASE + 32).unwrap();
            assert_eq!(hbm.allocate(48), Ok(DEFAULT_HBM_BASE + 32));
            assert_eq!(hbm.allocate(16), Ok(DEFAULT_HBM_BASE + 80));
            assert_eq!(
                hbm.resolve_live(DEFAULT_HBM_BASE + 52, 8),
                Ok(HbmResolvedSpan {
                    allocation_base: DEFAULT_HBM_BASE + 32,
                    allocation_bytes: 48,
                    offset: 20,
                })
            );
            assert_eq!(
                hbm.resolve_live(DEFAULT_HBM_BASE + 76, 8),
                Err(HbmResolveError::Unmapped {
                    address: DEFAULT_HBM_BASE + 76,
                    end: DEFAULT_HBM_BASE + 84,
                })
            );
        }
    }

    #[test]
    fn coalescing_restores_full_capacity_and_rejects_duplicate_free() {
        let mut hbm = HbmAllocator::new(Architecture::Dav2201);
        let a = hbm.allocate(32).unwrap();
        let b = hbm.allocate(64).unwrap();
        let c = hbm.allocate(16).unwrap();
        hbm.free(b).unwrap();
        hbm.free(a).unwrap();
        hbm.free(c).unwrap();
        assert_eq!(hbm.free(a), Err(HbmAllocationError::AlreadyFreed(a)));
        assert_eq!(
            hbm.spans(),
            [HbmSpan {
                base: DEFAULT_HBM_BASE,
                bytes: DEFAULT_HBM_BYTES,
                allocated: false
            }]
        );
        assert_eq!(hbm.allocate(DEFAULT_HBM_BYTES), Ok(DEFAULT_HBM_BASE));
        assert_eq!(
            hbm.allocate(1),
            Err(HbmAllocationError::OutOfMemory { requested: 1 })
        );
    }

    #[test]
    fn invalid_requests_do_not_change_allocator_state() {
        assert_eq!(
            HbmAllocator::with_region(Architecture::Dav2201, 0x1000, 0),
            Err(HbmAllocationError::InvalidRegion {
                base: 0x1000,
                bytes: 0
            })
        );
        let mut hbm = HbmAllocator::new(Architecture::Dav3510);
        let initial = hbm.clone();
        assert_eq!(hbm.allocate(0), Err(HbmAllocationError::ZeroSize));
        assert_eq!(
            hbm.allocate(DEFAULT_HBM_BYTES + 1),
            Err(HbmAllocationError::TooLarge {
                requested: DEFAULT_HBM_BYTES + 1
            })
        );
        assert_eq!(
            hbm.free(DEFAULT_HBM_BASE + 1),
            Err(HbmAllocationError::UnknownPointer(DEFAULT_HBM_BASE + 1))
        );
        assert_eq!(hbm, initial);
        assert_eq!(
            hbm.resolve_live(DEFAULT_HBM_BASE, 0),
            Err(HbmResolveError::ZeroLength)
        );
        assert_eq!(
            hbm.resolve_live(u64::MAX, 2),
            Err(HbmResolveError::RangeOverflow { address: u64::MAX })
        );
    }
}
