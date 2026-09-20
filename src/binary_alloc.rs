use crate::architecture::Architecture;
use serde::Serialize;
use thiserror::Error;

pub const BINARY_ALIGNMENT_BYTES: u64 = 0x1000;
pub const BINARY_POOL_BYTES: u64 = 0x20_0000;
pub const BINARY_FEATURE_42_EXTRA_BYTES: u64 = 1280;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum BinaryAllocationPlanError {
    #[error("a zero-byte device binary is not loadable")]
    ZeroBinary,
    #[error("device binary allocation arithmetic overflows")]
    ArithmeticOverflow,
    #[error("returned device pointer cannot be aligned without overflow")]
    AddressOverflow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BinaryDeviceAllocationPlan {
    pub architecture: Architecture,
    pub binary_bytes: u32,
    pub feature_42_extra_bytes: bool,
    pub feature_76_pool: bool,
    pub expanded_bytes: u64,
    pub pool_request_bytes: Option<u64>,
    pub pool_request_fits: bool,
    pub fallback_request_bytes: u64,
}

impl BinaryDeviceAllocationPlan {
    pub fn new(
        architecture: Architecture,
        binary_bytes: u32,
        feature_42_extra_bytes: bool,
        feature_76_pool: bool,
    ) -> Result<Self, BinaryAllocationPlanError> {
        if binary_bytes == 0 {
            return Err(BinaryAllocationPlanError::ZeroBinary);
        }
        let expanded_u32 = binary_bytes
            .checked_add(if feature_42_extra_bytes {
                BINARY_FEATURE_42_EXTRA_BYTES as u32
            } else {
                0
            })
            .ok_or(BinaryAllocationPlanError::ArithmeticOverflow)?;
        let expanded_bytes = u64::from(expanded_u32);
        let pool_request_bytes = if feature_76_pool {
            let rounded = expanded_u32
                .checked_add((BINARY_ALIGNMENT_BYTES - 1) as u32)
                .ok_or(BinaryAllocationPlanError::ArithmeticOverflow)?
                & !((BINARY_ALIGNMENT_BYTES - 1) as u32);
            Some(u64::from(rounded))
        } else {
            None
        };
        let fallback_request_bytes = expanded_bytes
            .checked_add(BINARY_ALIGNMENT_BYTES)
            .ok_or(BinaryAllocationPlanError::ArithmeticOverflow)?;
        Ok(Self {
            architecture,
            binary_bytes,
            feature_42_extra_bytes,
            feature_76_pool,
            expanded_bytes,
            pool_request_fits: pool_request_bytes.is_some_and(|bytes| bytes <= BINARY_POOL_BYTES),
            pool_request_bytes,
            fallback_request_bytes,
        })
    }

    pub fn aligned_code_address(
        &self,
        raw_device_pointer: u64,
    ) -> Result<u64, BinaryAllocationPlanError> {
        align_up(raw_device_pointer, BINARY_ALIGNMENT_BYTES)
            .ok_or(BinaryAllocationPlanError::AddressOverflow)
    }
}

const fn align_up(value: u64, alignment: u64) -> Option<u64> {
    match value.checked_add(alignment - 1) {
        Some(value) => Some(value & !(alignment - 1)),
        None => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_runtime_binaries_share_pool_and_fallback_arithmetic() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let plan = BinaryDeviceAllocationPlan::new(architecture, 4097, true, true).unwrap();
            assert_eq!(plan.expanded_bytes, 5377);
            assert_eq!(plan.pool_request_bytes, Some(8192));
            assert!(plan.pool_request_fits);
            assert_eq!(plan.fallback_request_bytes, 9473);
            assert_eq!(plan.aligned_code_address(0x1000_0020), Ok(0x1000_1000));
        }
    }

    #[test]
    fn direct_fallback_and_pool_limit_are_explicit() {
        let direct =
            BinaryDeviceAllocationPlan::new(Architecture::Dav2201, 1, false, false).unwrap();
        assert_eq!(direct.pool_request_bytes, None);
        assert!(!direct.pool_request_fits);
        assert_eq!(direct.fallback_request_bytes, 4097);

        let oversized = BinaryDeviceAllocationPlan::new(
            Architecture::Dav3510,
            BINARY_POOL_BYTES as u32,
            true,
            true,
        )
        .unwrap();
        assert!(oversized.pool_request_bytes.unwrap() > BINARY_POOL_BYTES);
        assert!(!oversized.pool_request_fits);
        assert_eq!(
            oversized.aligned_code_address(u64::MAX),
            Err(BinaryAllocationPlanError::AddressOverflow)
        );
        assert_eq!(
            BinaryDeviceAllocationPlan::new(Architecture::Dav2201, 0, false, false),
            Err(BinaryAllocationPlanError::ZeroBinary)
        );
        assert_eq!(
            BinaryDeviceAllocationPlan::new(Architecture::Dav2201, u32::MAX, true, false),
            Err(BinaryAllocationPlanError::ArithmeticOverflow)
        );
        assert_eq!(
            BinaryDeviceAllocationPlan::new(Architecture::Dav3510, u32::MAX, false, true),
            Err(BinaryAllocationPlanError::ArithmeticOverflow)
        );
    }
}
