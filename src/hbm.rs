
use crate::architecture::Architecture;
use serde::Serialize;
use thiserror::Error;

pub const CAMODEL_HBM_BASE: u64 = 0x1000_0000;
pub const CAMODEL_HBM_BYTES: u64 = 0x5_0000_0000;
pub const C310_DRIVER_ALLOCATION_ALIGNMENT: u64 = 512;

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
    #[error("the CA-model driver rejects a zero-byte allocation")]
    ZeroSize,
    #[error("allocation of {requested} bytes exceeds CA-model HBM capacity")]
    TooLarge { requested: u64 },
    #[error("no free CA-model HBM span can satisfy {requested} bytes")]
    OutOfMemory { requested: u64 },
    #[error("host metadata allocation failed")]
    HostAllocationFailed,
    #[error("no HBM allocation begins at device pointer {0:#x}")]
    UnknownPointer(u64),
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
pub struct CamodelHbmAllocator {
    architecture: Architecture,
    spans: Vec<HbmSpan>,
}

impl CamodelHbmAllocator {
    pub fn new(architecture: Architecture) -> Self {
        Self {
            architecture,
            spans: vec![HbmSpan {
                base: CAMODEL_HBM_BASE,
                bytes: CAMODEL_HBM_BYTES,
                allocated: false,
            }],
        }
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
        if bytes > CAMODEL_HBM_BYTES {
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

    pub fn allocate_driver_request(&mut self, bytes: u64) -> Result<u64, HbmAllocationError> {
        if bytes == 0 {
            return Err(HbmAllocationError::ZeroSize);
        }
        if bytes > CAMODEL_HBM_BYTES {
            return Err(HbmAllocationError::TooLarge { requested: bytes });
        }
        let actual_bytes = if self.architecture == Architecture::Dav3510 {
            (bytes + C310_DRIVER_ALLOCATION_ALIGNMENT - 1) & !(C310_DRIVER_ALLOCATION_ALIGNMENT - 1)
        } else {
            bytes
        };
        self.allocate(actual_bytes)
    }

    pub fn free(&mut self, pointer: u64) -> Result<(), HbmAllocationError> {
        let mut index = self
            .spans
            .iter()
            .position(|span| span.base == pointer)
            .ok_or(HbmAllocationError::UnknownPointer(pointer))?;
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
            let mut hbm = CamodelHbmAllocator::new(architecture);
            assert_eq!(hbm.allocate(32), Ok(CAMODEL_HBM_BASE));
            assert_eq!(hbm.allocate(64), Ok(CAMODEL_HBM_BASE + 32));
            assert_eq!(hbm.allocate(16), Ok(CAMODEL_HBM_BASE + 96));
            hbm.free(CAMODEL_HBM_BASE + 32).unwrap();
            assert_eq!(hbm.allocate(48), Ok(CAMODEL_HBM_BASE + 32));
            assert_eq!(hbm.allocate(16), Ok(CAMODEL_HBM_BASE + 80));
            assert_eq!(
                hbm.resolve_live(CAMODEL_HBM_BASE + 52, 8),
                Ok(HbmResolvedSpan {
                    allocation_base: CAMODEL_HBM_BASE + 32,
                    allocation_bytes: 48,
                    offset: 20,
                })
            );
            assert_eq!(
                hbm.resolve_live(CAMODEL_HBM_BASE + 76, 8),
                Err(HbmResolveError::Unmapped {
                    address: CAMODEL_HBM_BASE + 76,
                    end: CAMODEL_HBM_BASE + 84,
                })
            );
        }
    }

    #[test]
    fn coalescing_restores_full_capacity_and_duplicate_free_is_accepted() {
        let mut hbm = CamodelHbmAllocator::new(Architecture::Dav2201);
        let a = hbm.allocate(32).unwrap();
        let b = hbm.allocate(64).unwrap();
        let c = hbm.allocate(16).unwrap();
        hbm.free(b).unwrap();
        hbm.free(a).unwrap();
        hbm.free(c).unwrap();
        hbm.free(a).unwrap();
        assert_eq!(
            hbm.spans(),
            [HbmSpan {
                base: CAMODEL_HBM_BASE,
                bytes: CAMODEL_HBM_BYTES,
                allocated: false
            }]
        );
        assert_eq!(hbm.allocate(CAMODEL_HBM_BYTES), Ok(CAMODEL_HBM_BASE));
        assert_eq!(
            hbm.allocate(1),
            Err(HbmAllocationError::OutOfMemory { requested: 1 })
        );
    }

    #[test]
    fn invalid_requests_do_not_change_allocator_state() {
        let mut hbm = CamodelHbmAllocator::new(Architecture::Dav3510);
        let initial = hbm.clone();
        assert_eq!(hbm.allocate(0), Err(HbmAllocationError::ZeroSize));
        assert_eq!(
            hbm.allocate(CAMODEL_HBM_BYTES + 1),
            Err(HbmAllocationError::TooLarge {
                requested: CAMODEL_HBM_BYTES + 1
            })
        );
        assert_eq!(
            hbm.free(CAMODEL_HBM_BASE + 1),
            Err(HbmAllocationError::UnknownPointer(CAMODEL_HBM_BASE + 1))
        );
        assert_eq!(hbm, initial);
        assert_eq!(
            hbm.resolve_live(CAMODEL_HBM_BASE, 0),
            Err(HbmResolveError::ZeroLength)
        );
        assert_eq!(
            hbm.resolve_live(u64::MAX, 2),
            Err(HbmResolveError::RangeOverflow { address: u64::MAX })
        );
    }

    #[test]
    fn c310_driver_entry_rounds_to_512_before_hbm_first_fit() {
        let mut c220 = CamodelHbmAllocator::new(Architecture::Dav2201);
        let mut c310 = CamodelHbmAllocator::new(Architecture::Dav3510);
        assert_eq!(c220.allocate_driver_request(1), Ok(CAMODEL_HBM_BASE));
        assert_eq!(c310.allocate_driver_request(1), Ok(CAMODEL_HBM_BASE));
        assert_eq!(c220.allocate_driver_request(513), Ok(CAMODEL_HBM_BASE + 1));
        assert_eq!(
            c310.allocate_driver_request(513),
            Ok(CAMODEL_HBM_BASE + 512)
        );
        assert_eq!(c220.spans()[1].bytes, 513);
        assert_eq!(c310.spans()[1].bytes, 1024);
        assert_eq!(
            c310.allocate_driver_request(CAMODEL_HBM_BYTES + 1),
            Err(HbmAllocationError::TooLarge {
                requested: CAMODEL_HBM_BYTES + 1
            })
        );
    }

    #[test]
    fn conditional_device_open_prefix_reserves_swap_buffer_then_cq() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut hbm = CamodelHbmAllocator::new(architecture);
            assert_eq!(hbm.allocate_driver_request(0x20000), Ok(CAMODEL_HBM_BASE));
            assert_eq!(
                hbm.allocate_driver_request(0x8000),
                Ok(CAMODEL_HBM_BASE + 0x20000)
            );
            assert_eq!(
                hbm.allocate_driver_request(512),
                Ok(CAMODEL_HBM_BASE + 0x28000)
            );
        }
    }
}
