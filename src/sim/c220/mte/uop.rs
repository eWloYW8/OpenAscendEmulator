use crate::isa::c220::mte::out_to_l1::{C220L1DmaDescriptor, C220L1DmaLayout};
use thiserror::Error;

use crate::isa::c220::mte::{
    C220DmaMovDescriptor, C220DmaMovError, C220MovOutToUbDescriptor, C220MovOutToUbError,
};
use crate::sim::c220::mte::mte2::C220Mte2TransferPlan;
use crate::sim::c220::mte::mte3::C220Mte3TransferPlan;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220DmaUopMode {
    Wide512,
    Wide256,
    Fixed128,
    Unbounded,
}

impl C220DmaUopMode {
    pub(super) fn split_bytes(self, address: u64, remaining: u32) -> u32 {
        if !address.is_multiple_of(128) {
            remaining.min(128 - (address % 128) as u32)
        } else if self == Self::Wide512 && address.is_multiple_of(512) && remaining >= 512 {
            512
        } else if matches!(self, Self::Wide512 | Self::Wide256)
            && address.is_multiple_of(256)
            && remaining >= 256
        {
            256
        } else if self == Self::Unbounded {
            remaining
        } else {
            remaining.min(128)
        }
    }

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
    L1Pad1,
    L1Pad2,
    L1Pad4,
    L1Pad8,
    L1Pad16,
    L1Take4,
    L1Take8,
    L1Take16,
}

impl C220DmaUopRoute {
    pub const fn padded_unit_bytes(self) -> Option<u32> {
        match self {
            Self::L1Pad1 => Some(1),
            Self::L1Pad2 => Some(2),
            Self::L1Pad4 => Some(4),
            Self::L1Pad8 => Some(8),
            Self::L1Pad16 => Some(16),
            _ => None,
        }
    }

    pub const fn truncated_unit_bytes(self) -> Option<u32> {
        match self {
            Self::L1Take4 => Some(4),
            Self::L1Take8 => Some(8),
            Self::L1Take16 => Some(16),
            _ => None,
        }
    }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220DmaDestinationLayout {
    pub base: u64,
    pub burst_bytes: u32,
    pub burst_stride: u64,
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
    #[error("cannot reserve {requested} DMA uop records")]
    AllocationFailed { requested: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    mte2_uops(transfer)?.collect_records()
}

pub fn mte2_uops(transfer: C220Mte2TransferPlan) -> Result<C220DmaUops, C220DmaUopError> {
    let mut requests = mte2_request_stream(transfer)?;
    requests.destination = C220DmaDestinationLayout {
        base: transfer.destination_address,
        burst_bytes: u32::from(transfer.descriptor.burst_length) * 32,
        burst_stride: (u64::from(transfer.descriptor.burst_length)
            + u64::from(transfer.descriptor.destination_gap))
            * 32,
    };
    Ok(requests)
}

fn mte2_request_stream(transfer: C220Mte2TransferPlan) -> Result<C220DmaUops, C220DmaUopError> {
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
        return split_requests(
            DmaRequestGeometry {
                source_base: transfer.source_address,
                destination_base: transfer.destination_address,
                burst_count: 1,
                burst_bytes: u64::from(bytes),
                source_stride: 0,
                destination_stride: 0,
                split_on_destination: false,
                split_enabled: false,
            },
            route,
            C220DmaUopMode::from_mode_word(transfer.dma_mode_word),
        );
    }
    split_mte2_requests(
        transfer,
        descriptor,
        C220DmaUopRoute::Ordinary,
        C220DmaUopMode::from_mode_word(transfer.dma_mode_word),
    )
}

pub fn mte2_l1_uops(
    descriptor: C220L1DmaDescriptor,
    source_address: u64,
    destination_address: u64,
    dma_mode_word: u64,
) -> Result<C220DmaUops, C220DmaUopError> {
    let transform = match descriptor.layout {
        C220L1DmaLayout::Pad1 => Some(C220DmaUopRoute::L1Pad1),
        C220L1DmaLayout::Pad2 => Some(C220DmaUopRoute::L1Pad2),
        C220L1DmaLayout::Pad4 => Some(C220DmaUopRoute::L1Pad4),
        C220L1DmaLayout::Pad8 => Some(C220DmaUopRoute::L1Pad8),
        C220L1DmaLayout::Pad16 => Some(C220DmaUopRoute::L1Pad16),
        C220L1DmaLayout::Take4 => Some(C220DmaUopRoute::L1Take4),
        C220L1DmaLayout::Take8 => Some(C220DmaUopRoute::L1Take8),
        C220L1DmaLayout::Take16 => Some(C220DmaUopRoute::L1Take16),
        C220L1DmaLayout::Copy32 => None,
    };
    if let Some(route) = transform {
        let source_unit = descriptor.layout.source_bytes();
        let padding = route.padded_unit_bytes().is_some();
        return split_requests(
            DmaRequestGeometry {
                source_base: source_address,
                destination_base: destination_address,
                burst_count: u16::from(!descriptor.is_disabled()),
                burst_bytes: u64::from(
                    (if padding {
                        source_unit
                    } else {
                        u32::from(descriptor.burst_length()) * source_unit
                    })
                    .wrapping_mul(u32::from(descriptor.burst_count())),
                ),
                source_stride: (u64::from(descriptor.burst_length())
                    + u64::from(descriptor.source_gap()))
                    * u64::from(source_unit),
                destination_stride: (u64::from(descriptor.burst_length())
                    + u64::from(descriptor.destination_gap()))
                    * u64::from(descriptor.layout.destination_bytes()),
                split_on_destination: false,
                split_enabled: padding && source_address.is_multiple_of(32),
            },
            route,
            C220DmaUopMode::from_mode_word(dma_mode_word),
        );
    }
    let count = descriptor.burst_count();
    let length = descriptor.burst_length();
    let source_gap = descriptor.source_gap();
    let destination_gap = descriptor.destination_gap();
    let batch = source_gap == 0 && destination_gap == 0;
    let collapsed = !batch && source_gap == 0 && (1..=2).contains(&length);
    let burst_bytes = u32::from(length) * 32;
    let destination_stride = (u64::from(length) + u64::from(destination_gap)) * 32;
    let mut requests = split_requests(
        DmaRequestGeometry {
            source_base: source_address,
            destination_base: destination_address,
            burst_count: if count == 0 || length == 0 {
                0
            } else if batch || collapsed {
                1
            } else {
                count
            },
            burst_bytes: u64::from(if batch || collapsed {
                burst_bytes.wrapping_mul(u32::from(count))
            } else {
                burst_bytes
            }),
            source_stride: (u64::from(length) + u64::from(source_gap)) * 32,
            destination_stride,
            split_on_destination: false,
            split_enabled: !collapsed && source_address.is_multiple_of(32),
        },
        if batch {
            C220DmaUopRoute::ContiguousBatch
        } else if collapsed {
            C220DmaUopRoute::DestinationGapCollapse
        } else {
            C220DmaUopRoute::Ordinary
        },
        C220DmaUopMode::from_mode_word(dma_mode_word),
    )?;
    requests.destination = C220DmaDestinationLayout {
        base: destination_address,
        burst_bytes,
        burst_stride: destination_stride,
    };
    Ok(requests)
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
    let _ = descriptor.segment_iter(transfer.source_address, transfer.destination_address)?;
    Ok(descriptor)
}

pub fn ordinary_mte2_requests(
    transfer: C220Mte2TransferPlan,
    mode: C220DmaUopMode,
) -> Result<Vec<C220DmaUopRequest>, C220DmaUopError> {
    let descriptor = checked_descriptor(transfer)?;
    split_mte2_requests(transfer, descriptor, C220DmaUopRoute::Ordinary, mode)?.collect_records()
}

fn split_mte2_requests(
    transfer: C220Mte2TransferPlan,
    descriptor: C220MovOutToUbDescriptor,
    route: C220DmaUopRoute,
    mode: C220DmaUopMode,
) -> Result<C220DmaUops, C220DmaUopError> {
    let batch = route == C220DmaUopRoute::ContiguousBatch;
    split_requests(
        DmaRequestGeometry {
            source_base: transfer.source_address,
            destination_base: transfer.destination_address,
            burst_count: if batch { 1 } else { descriptor.burst_count },
            burst_bytes: if batch {
                u64::from(transfer.bytes as u32)
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
    mte3_uops(transfer)?.collect_records()
}

pub fn mte3_uops(transfer: C220Mte3TransferPlan) -> Result<C220DmaUops, C220DmaUopError> {
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
    let _ = descriptor.segment_iter(transfer.source_address, transfer.destination_address)?;
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
                u64::from(transfer.bytes as u32)
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

fn split_requests(
    geometry: DmaRequestGeometry,
    route: C220DmaUopRoute,
    mode: C220DmaUopMode,
) -> Result<C220DmaUops, C220DmaUopError> {
    if geometry.burst_bytes > u64::from(u32::MAX) {
        return Err(C220DmaUopError::SizeOverflow);
    }
    Ok(C220DmaUops {
        destination: C220DmaDestinationLayout {
            base: geometry.destination_base,
            burst_bytes: geometry.burst_bytes as u32,
            burst_stride: geometry.destination_stride,
        },
        geometry,
        route,
        mode,
        burst_index: 0,
        offset: 0,
    })
}

/// Lazy physical requests. Burst source offsets use a 32-bit accumulator;
/// destination offsets and both base addresses remain 64-bit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220DmaUops {
    destination: C220DmaDestinationLayout,
    geometry: DmaRequestGeometry,
    route: C220DmaUopRoute,
    mode: C220DmaUopMode,
    burst_index: u16,
    offset: u64,
}

impl C220DmaUops {
    pub(super) fn destination(&self) -> C220DmaDestinationLayout {
        self.destination
    }

    pub(super) fn out_of_order(&self) -> bool {
        self.geometry.split_enabled
    }

    pub(super) fn mode(&self) -> C220DmaUopMode {
        self.mode
    }

    pub fn collect_records(self) -> Result<Vec<C220DmaUopRequest>, C220DmaUopError> {
        let mut records = Vec::new();
        for request in self {
            if records.len() == records.capacity() {
                records
                    .try_reserve(1)
                    .map_err(|_| C220DmaUopError::AllocationFailed {
                        requested: records.len() + 1,
                    })?;
            }
            records.push(request);
        }
        Ok(records)
    }
}

impl Iterator for C220DmaUops {
    type Item = C220DmaUopRequest;

    fn next(&mut self) -> Option<Self::Item> {
        let geometry = self.geometry;
        if self.burst_index >= geometry.burst_count || geometry.burst_bytes == 0 {
            return None;
        }
        let burst_index = self.burst_index;
        let source_address = geometry
            .source_base
            .wrapping_add(u64::from(
                u32::from(burst_index).wrapping_mul(geometry.source_stride as u32),
            ))
            .wrapping_add(self.offset);
        let destination_address = geometry
            .destination_base
            .wrapping_add(u64::from(burst_index) * geometry.destination_stride)
            .wrapping_add(self.offset);
        let aligned_address = if geometry.split_on_destination {
            destination_address
        } else {
            source_address
        };
        let remaining = geometry.burst_bytes - self.offset;
        let bytes = if !geometry.split_enabled {
            remaining
        } else {
            u64::from(self.mode.split_bytes(aligned_address, remaining as u32))
        };
        self.offset += bytes;
        let last_in_burst = self.offset == geometry.burst_bytes;
        if last_in_burst {
            self.burst_index += 1;
            self.offset = 0;
        }
        Some(C220DmaUopRequest {
            route: self.route,
            burst_index,
            source_address,
            destination_address,
            bytes: bytes as u32,
            last_in_burst,
        })
    }
}

impl std::iter::FusedIterator for C220DmaUops {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::c220::mte::{
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
    fn full_descriptor_range_streams_wrapped_offsets_and_disabled_commands() {
        let xm = (1_u64 << 48) | (0xffff << 32) | 0xffff_fff0;
        let plan = transfer(xm, 1);
        let requests = mte2_uops(plan).unwrap();
        assert_eq!(requests.clone().count(), 4095);
        let last = requests.last().unwrap();
        let segment = plan
            .descriptor_segments()
            .unwrap()
            .nth(4094 * 65535)
            .unwrap();
        assert_eq!(last.source_address, segment.source_hbm);
        assert_eq!(last.destination_address, segment.destination_local);
        assert_eq!(last.bytes, 65535 * 32);
        assert!(last.last_in_burst);
        assert!(last.destination_address > u64::from(u32::MAX));

        let mut batch = transfer(0xffff_fff0, 0x1000);
        batch.dma_mode_word = 7;
        assert!(batch.bytes > u32::MAX as usize);
        let mut requests = mte2_uops(batch).unwrap();
        let request = requests.next().unwrap();
        assert_eq!(request.bytes, batch.bytes as u32);
        assert!(request.last_in_burst);
        assert_eq!(requests.next(), None);

        for xm in [0, 0x10, 0x10000] {
            let mut input = transfer(xm, u64::MAX);
            input.destination_address = u64::MAX;
            assert_eq!(mte2_uops(input).unwrap().next(), None);
            let mut output = output_transfer(xm, u64::MAX);
            output.source_address = u64::MAX;
            assert_eq!(mte3_uops(output).unwrap().next(), None);
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
