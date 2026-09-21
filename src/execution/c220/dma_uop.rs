use thiserror::Error;

use crate::execution::c220::transfer::{C220Mte2TransferPlan, C220Mte3TransferPlan};
use crate::instruction::c220::mte::{
    C220DmaMovDescriptor, C220DmaMovError, C220MovOutToUbDescriptor, C220MovOutToUbError,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220DmaUopMode {
    Wide512,
    Wide256,
    Fixed128,
    Unbounded,
}

impl C220DmaUopMode {
    pub const fn from_mode_word(value: u64) -> Self {
        if value & 1 == 0 {
            return Self::Wide512;
        }
        match (value >> 1) & 3 {
            0 => Self::Wide512,
            1 => Self::Wide256,
            2 => Self::Fixed128,
            _ => Self::Unbounded,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220DmaUopRoute {
    Ordinary,
    ContiguousBatch,
    DestinationGapCollapse,
    SourceGapGather,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220DmaUopRequest {
    pub route: C220DmaUopRoute,
    pub burst_index: u16,
    pub source_address: u64,
    pub destination_address: u64,
    pub bytes: u32,
    pub last_in_burst: bool,
}

#[derive(Debug, Error)]
pub enum C220DmaUopError {
    #[error(transparent)]
    Descriptor(#[from] C220MovOutToUbError),
    #[error(transparent)]
    OutputDescriptor(#[from] C220DmaMovError),
    #[error("DMA transfer byte count differs from its descriptor")]
    PlanSizeMismatch,
    #[error("DMA uop byte count exceeds u32")]
    SizeOverflow,
    #[error("DMA uop address calculation overflowed")]
    AddressOverflow,
    #[error("cannot reserve {requested} DMA uop records")]
    AllocationFailed { requested: usize },
}

#[derive(Debug, Clone, Copy)]
struct DmaRequestGeometry {
    source_base: u64,
    destination_base: u64,
    burst_count: u16,
    burst_bytes: u64,
    source_stride: u64,
    destination_stride: u64,
    split_on_destination: bool,
    split_enabled: bool,
}

pub fn mte2_requests(
    transfer: C220Mte2TransferPlan,
) -> Result<Vec<C220DmaUopRequest>, C220DmaUopError> {
    let descriptor = checked_descriptor(transfer)?;
    let route = if descriptor.source_gap == 0 && descriptor.destination_gap == 0 {
        Some(C220DmaUopRoute::ContiguousBatch)
    } else if descriptor.source_gap == 0
        && descriptor.destination_gap != 0
        && (1..=2).contains(&descriptor.burst_length)
    {
        Some(C220DmaUopRoute::DestinationGapCollapse)
    } else {
        None
    };
    if route == Some(C220DmaUopRoute::ContiguousBatch) {
        return split_mte2_requests(
            transfer,
            descriptor,
            C220DmaUopRoute::ContiguousBatch,
            C220DmaUopMode::from_mode_word(transfer.dma_mode_word),
        );
    }
    if let Some(route) = route {
        let bytes = u32::try_from(transfer.bytes).map_err(|_| C220DmaUopError::SizeOverflow)?;
        return Ok(vec![C220DmaUopRequest {
            route,
            burst_index: 0,
            source_address: transfer.source_address,
            destination_address: transfer.destination_address,
            bytes,
            last_in_burst: true,
        }]);
    }
    ordinary_mte2_requests(
        transfer,
        C220DmaUopMode::from_mode_word(transfer.dma_mode_word),
    )
}

fn checked_descriptor(
    transfer: C220Mte2TransferPlan,
) -> Result<C220MovOutToUbDescriptor, C220DmaUopError> {
    let descriptor = C220MovOutToUbDescriptor::decode(
        transfer.descriptor.instruction_word,
        transfer.descriptor.xm,
    )?;
    if descriptor != transfer.descriptor {
        return Err(C220MovOutToUbError::InconsistentDescriptor.into());
    }
    let expected_bytes =
        usize::from(descriptor.burst_count) * usize::from(descriptor.burst_length) * 32;
    if transfer.bytes != expected_bytes {
        return Err(C220DmaUopError::PlanSizeMismatch);
    }
    for (base, gap) in [
        (transfer.source_address, descriptor.source_gap),
        (transfer.destination_address, descriptor.destination_gap),
    ] {
        check_address_extent(base, descriptor.burst_count, descriptor.burst_length, gap)?;
    }
    Ok(descriptor)
}

pub fn ordinary_mte2_requests(
    transfer: C220Mte2TransferPlan,
    mode: C220DmaUopMode,
) -> Result<Vec<C220DmaUopRequest>, C220DmaUopError> {
    let descriptor = checked_descriptor(transfer)?;
    split_mte2_requests(transfer, descriptor, C220DmaUopRoute::Ordinary, mode)
}

fn split_mte2_requests(
    transfer: C220Mte2TransferPlan,
    descriptor: C220MovOutToUbDescriptor,
    route: C220DmaUopRoute,
    mode: C220DmaUopMode,
) -> Result<Vec<C220DmaUopRequest>, C220DmaUopError> {
    let batch = route == C220DmaUopRoute::ContiguousBatch;
    split_requests(
        DmaRequestGeometry {
            source_base: transfer.source_address,
            destination_base: transfer.destination_address,
            burst_count: if batch { 1 } else { descriptor.burst_count },
            burst_bytes: if batch {
                transfer.bytes as u64
            } else {
                u64::from(descriptor.burst_length) * 32
            },
            source_stride: (u64::from(descriptor.burst_length) + u64::from(descriptor.source_gap))
                * 32,
            destination_stride: (u64::from(descriptor.burst_length)
                + u64::from(descriptor.destination_gap))
                * 32,
            split_on_destination: false,
            split_enabled: transfer.source_address.is_multiple_of(32),
        },
        route,
        mode,
    )
}

pub fn mte3_requests(
    transfer: C220Mte3TransferPlan,
) -> Result<Vec<C220DmaUopRequest>, C220DmaUopError> {
    let descriptor =
        C220DmaMovDescriptor::decode(transfer.descriptor.instruction_word, transfer.descriptor.xm)?;
    if descriptor != transfer.descriptor {
        return Err(C220DmaMovError::InconsistentDescriptor.into());
    }
    let expected_bytes =
        usize::from(descriptor.burst_count) * usize::from(descriptor.burst_length) * 32;
    if transfer.bytes != expected_bytes {
        return Err(C220DmaUopError::PlanSizeMismatch);
    }
    for (base, gap) in [
        (transfer.source_address, descriptor.source_gap),
        (transfer.destination_address, descriptor.destination_gap),
    ] {
        check_address_extent(base, descriptor.burst_count, descriptor.burst_length, gap)?;
    }
    let batch = descriptor.source_gap == 0 && descriptor.destination_gap == 0;
    let gather = !batch
        && transfer.source_address.is_multiple_of(32)
        && transfer.destination_address.is_multiple_of(64)
        && descriptor.source_gap != 0
        && descriptor.destination_gap == 0
        && descriptor.burst_count > 1
        && descriptor.burst_length == 2;
    let flatten = batch || gather;
    split_requests(
        DmaRequestGeometry {
            source_base: transfer.source_address,
            destination_base: transfer.destination_address,
            burst_count: if flatten { 1 } else { descriptor.burst_count },
            burst_bytes: if flatten {
                transfer.bytes as u64
            } else {
                u64::from(descriptor.burst_length) * 32
            },
            source_stride: (u64::from(descriptor.burst_length) + u64::from(descriptor.source_gap))
                * 32,
            destination_stride: (u64::from(descriptor.burst_length)
                + u64::from(descriptor.destination_gap))
                * 32,
            split_on_destination: true,
            split_enabled: true,
        },
        if batch {
            C220DmaUopRoute::ContiguousBatch
        } else if gather {
            C220DmaUopRoute::SourceGapGather
        } else {
            C220DmaUopRoute::Ordinary
        },
        C220DmaUopMode::from_mode_word(transfer.dma_mode_word),
    )
}

fn check_address_extent(
    base: u64,
    burst_count: u16,
    burst_length: u16,
    gap: u16,
) -> Result<(), C220DmaUopError> {
    let stride = (u64::from(burst_length) + u64::from(gap)) * 32;
    let extent = (u64::from(burst_count) - 1)
        .checked_mul(stride)
        .and_then(|offset| offset.checked_add(u64::from(burst_length) * 32))
        .ok_or(C220DmaUopError::AddressOverflow)?;
    base.checked_add(extent)
        .ok_or(C220DmaUopError::AddressOverflow)?;
    Ok(())
}

fn split_requests(
    geometry: DmaRequestGeometry,
    route: C220DmaUopRoute,
    mode: C220DmaUopMode,
) -> Result<Vec<C220DmaUopRequest>, C220DmaUopError> {
    let requested = usize::from(geometry.burst_count)
        * usize::try_from(geometry.burst_bytes / 32).map_err(|_| C220DmaUopError::SizeOverflow)?
        * 2;
    let mut requests = Vec::new();
    requests
        .try_reserve_exact(requested)
        .map_err(|_| C220DmaUopError::AllocationFailed { requested })?;
    for burst_index in 0..geometry.burst_count {
        let source_start = geometry
            .source_base
            .checked_add(u64::from(burst_index) * geometry.source_stride)
            .ok_or(C220DmaUopError::AddressOverflow)?;
        let destination_start = geometry
            .destination_base
            .checked_add(u64::from(burst_index) * geometry.destination_stride)
            .ok_or(C220DmaUopError::AddressOverflow)?;
        let mut offset = 0;
        while offset < geometry.burst_bytes {
            let source_address = source_start
                .checked_add(offset)
                .ok_or(C220DmaUopError::AddressOverflow)?;
            let destination_address = destination_start
                .checked_add(offset)
                .ok_or(C220DmaUopError::AddressOverflow)?;
            let aligned_address = if geometry.split_on_destination {
                destination_address
            } else {
                source_address
            };
            let remaining = geometry.burst_bytes - offset;
            let bytes = if !geometry.split_enabled {
                remaining
            } else if !aligned_address.is_multiple_of(128) {
                remaining.min(128 - aligned_address % 128)
            } else if mode == C220DmaUopMode::Wide512
                && aligned_address.is_multiple_of(512)
                && remaining >= 512
            {
                512
            } else if matches!(mode, C220DmaUopMode::Wide512 | C220DmaUopMode::Wide256)
                && aligned_address.is_multiple_of(256)
                && remaining >= 256
            {
                256
            } else if mode == C220DmaUopMode::Unbounded {
                remaining
            } else {
                remaining.min(128)
            };
            source_address
                .checked_add(bytes)
                .ok_or(C220DmaUopError::AddressOverflow)?;
            destination_address
                .checked_add(bytes)
                .ok_or(C220DmaUopError::AddressOverflow)?;
            offset += bytes;
            requests.push(C220DmaUopRequest {
                route,
                burst_index,
                source_address,
                destination_address,
                bytes: bytes as u32,
                last_in_burst: offset == geometry.burst_bytes,
            });
        }
    }
    Ok(requests)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instruction::c220::mte::{
        CAPTURED_C220_MOV_OUT_TO_UB_X_WORD, CAPTURED_C220_MOV_UB_TO_OUT_WORD,
    };

    fn transfer(xm: u64, source_address: u64) -> C220Mte2TransferPlan {
        let descriptor =
            C220MovOutToUbDescriptor::decode(CAPTURED_C220_MOV_OUT_TO_UB_X_WORD, xm).unwrap();
        C220Mte2TransferPlan {
            descriptor,
            source_address,
            destination_address: 0x200,
            bytes: usize::from(descriptor.burst_count) * usize::from(descriptor.burst_length) * 32,
            dma_mode_word: 0,
        }
    }

    fn output_transfer(xm: u64, destination_address: u64) -> C220Mte3TransferPlan {
        let descriptor =
            C220DmaMovDescriptor::decode(CAPTURED_C220_MOV_UB_TO_OUT_WORD, xm).unwrap();
        C220Mte3TransferPlan {
            descriptor,
            source_address: 0x200,
            destination_address,
            bytes: usize::from(descriptor.burst_count) * usize::from(descriptor.burst_length) * 32,
            dma_mode_word: 0,
        }
    }

    #[test]
    fn mte3_requests_split_by_destination_alignment_and_keep_burst_strides() {
        let batch = output_transfer((16 << 16) | (1 << 4), 0x1020);
        let requests = mte3_requests(batch).unwrap();
        assert_eq!(
            requests
                .iter()
                .map(|request| request.bytes)
                .collect::<Vec<_>>(),
            [96, 128, 256, 32]
        );
        assert!(
            requests
                .iter()
                .all(|request| request.route == C220DmaUopRoute::ContiguousBatch)
        );

        let strided = output_transfer((2_u64 << 48) | (1 << 32) | (2 << 16) | (2 << 4), 0x1000);
        let requests = mte3_requests(strided).unwrap();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].source_address, 0x200);
        assert_eq!(requests[1].source_address, 0x260);
        assert_eq!(requests[0].destination_address, 0x1000);
        assert_eq!(requests[1].destination_address, 0x1080);
        assert!(requests.iter().all(|request| request.last_in_burst));

        let gathered = output_transfer((1_u64 << 32) | (2 << 16) | (3 << 4), 0x1000);
        let requests = mte3_requests(gathered).unwrap();
        assert_eq!(
            requests
                .iter()
                .map(|request| request.bytes)
                .collect::<Vec<_>>(),
            [128, 64]
        );
        assert!(
            requests
                .iter()
                .all(|request| request.route == C220DmaUopRoute::SourceGapGather)
        );
        assert!(requests.last().unwrap().last_in_burst);

        let unaligned_destination = output_transfer((1_u64 << 32) | (2 << 16) | (3 << 4), 0x1020);
        let requests = mte3_requests(unaligned_destination).unwrap();
        assert!(requests.len() >= 3);
        assert!(
            requests
                .iter()
                .all(|request| request.route == C220DmaUopRoute::Ordinary)
        );
        let second_burst = requests
            .iter()
            .find(|request| request.burst_index == 1)
            .unwrap();
        assert_eq!(second_burst.source_address, 0x260);
        assert_eq!(second_burst.destination_address, 0x1060);
    }

    #[test]
    fn wide_modes_split_at_alignment_boundaries() {
        let plan = transfer((32 << 16) | (1 << 4), 0x1000);
        for (mode, lengths) in [
            (C220DmaUopMode::Wide512, vec![512, 512]),
            (C220DmaUopMode::Wide256, vec![256; 4]),
            (C220DmaUopMode::Fixed128, vec![128; 8]),
            (C220DmaUopMode::Unbounded, vec![1024]),
        ] {
            let requests = ordinary_mte2_requests(plan, mode).unwrap();
            assert_eq!(
                requests
                    .iter()
                    .map(|request| request.bytes)
                    .collect::<Vec<_>>(),
                lengths
            );
            assert!(requests.last().unwrap().last_in_burst);
        }
        let requests = ordinary_mte2_requests(
            transfer((16 << 16) | (1 << 4), 0x1020),
            C220DmaUopMode::Wide512,
        )
        .unwrap();
        assert_eq!(
            requests
                .iter()
                .map(|request| request.bytes)
                .collect::<Vec<_>>(),
            [96, 128, 256, 32]
        );
        let unaligned = ordinary_mte2_requests(
            transfer((16 << 16) | (1 << 4), 0x1001),
            C220DmaUopMode::Wide512,
        )
        .unwrap();
        assert_eq!(unaligned.len(), 1);
        assert_eq!(unaligned[0].bytes, 512);
    }

    #[test]
    fn bursts_keep_independent_source_and_destination_strides() {
        let plan = transfer((2_u64 << 48) | (1 << 32) | (2 << 16) | (2 << 4), 0x1000);
        let requests = ordinary_mte2_requests(plan, C220DmaUopMode::Fixed128).unwrap();
        assert_eq!(requests.len(), 3);
        assert_eq!(
            (
                requests[0].source_address,
                requests[0].destination_address,
                requests[0].bytes
            ),
            (0x1000, 0x200, 64)
        );
        assert_eq!(
            (
                requests[1].source_address,
                requests[1].destination_address,
                requests[1].bytes
            ),
            (0x1060, 0x280, 32)
        );
        assert_eq!(
            (
                requests[2].source_address,
                requests[2].destination_address,
                requests[2].bytes
            ),
            (0x1080, 0x2a0, 32)
        );
        assert_eq!(
            requests
                .iter()
                .filter(|request| request.last_in_burst)
                .count(),
            2
        );
    }

    #[test]
    fn mode_word_and_descriptor_gaps_choose_the_request_route() {
        assert_eq!(C220DmaUopMode::from_mode_word(0), C220DmaUopMode::Wide512);
        assert_eq!(C220DmaUopMode::from_mode_word(3), C220DmaUopMode::Wide256);
        assert_eq!(C220DmaUopMode::from_mode_word(5), C220DmaUopMode::Fixed128);
        assert_eq!(C220DmaUopMode::from_mode_word(7), C220DmaUopMode::Unbounded);
        assert_eq!(C220DmaUopMode::from_mode_word(6), C220DmaUopMode::Wide512);

        let mut contiguous = transfer((32 << 16) | (1 << 4), 0x1000);
        contiguous.dma_mode_word = 5;
        let requests = mte2_requests(contiguous).unwrap();
        assert_eq!(requests.len(), 8);
        assert!(requests.iter().all(
            |request| request.route == C220DmaUopRoute::ContiguousBatch && request.bytes == 128
        ));
        assert!(requests.last().unwrap().last_in_burst);

        let mut strided = transfer((2_u64 << 48) | (1 << 32) | (2 << 16) | (2 << 4), 0x1000);
        strided.dma_mode_word = 5;
        let requests = mte2_requests(strided).unwrap();
        assert_eq!(requests.len(), 3);
        assert!(
            requests
                .iter()
                .all(|request| request.route == C220DmaUopRoute::Ordinary)
        );

        let collapsed = transfer((1_u64 << 48) | (2 << 16) | (2 << 4), 0x1000);
        let requests = mte2_requests(collapsed).unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].route, C220DmaUopRoute::DestinationGapCollapse);
        assert_eq!(requests[0].bytes, 128);
    }
}
