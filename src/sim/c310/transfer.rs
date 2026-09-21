use crate::isa::c310::mte::{C310MovAlignCoordinateError, C310MovAlignDecode};
use crate::memory::mapped::MappedMemory;
use crate::memory::ub::{UbMemory, UbMemoryError, UbTransferResult};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum C310TransferError {
    #[error("MOV_ALIGN_V2 route {source_class}->{destination_class} is not HBM-to-UB")]
    UnsupportedRoute {
        source_class: u8,
        destination_class: u8,
    },
    #[error("UB transfer length is zero")]
    ZeroTransferLength,
    #[error(transparent)]
    Coordinates(#[from] C310MovAlignCoordinateError),
    #[error(transparent)]
    Ub(#[from] UbMemoryError),
}

pub fn copy_c310_mov_align_hbm_to_ub(
    ub: &mut UbMemory,
    source: &MappedMemory,
    decoded: C310MovAlignDecode,
) -> Result<UbTransferResult, C310TransferError> {
    if decoded.source_memory_class != 10 || decoded.destination_memory_class != 9 {
        return Err(C310TransferError::UnsupportedRoute {
            source_class: decoded.source_memory_class,
            destination_class: decoded.destination_memory_class,
        });
    }
    if decoded.burst_bytes == 0 {
        return Err(C310TransferError::ZeroTransferLength);
    }
    let coordinates = decoded.parameters.coordinates()?;
    Ok(ub.copy_segments(
        source,
        coordinates.into_iter().map(|coordinate| {
            (
                coordinate.source_address,
                coordinate.destination_address,
                decoded.burst_bytes as usize,
            )
        }),
    )?)
}
