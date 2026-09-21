use std::collections::{BTreeMap, BTreeSet};

use thiserror::Error;

use crate::instruction::c220::mte::{
    C220DmaMovDescriptor, C220DmaMovError, C220MovOutToUbDescriptor, C220MovOutToUbError,
};
use crate::instruction::c310::mte::{C310MovAlignCoordinateError, C310MovAlignDecode};
use crate::memory::mapped::{MappedMemory, MappedMemoryError};
use crate::memory::sparse::MemoryByteState;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum UbMemoryError {
    #[error("UB address range overflows u64")]
    RangeOverflow,
    #[error("UB transfer of {requested} bytes exceeds limit of {limit} bytes")]
    TransferLimitExceeded { requested: usize, limit: usize },
    #[error("UB tracked byte count would exceed limit of {limit} bytes")]
    TrackedLimitExceeded { limit: usize },
    #[error("cannot reserve {requested} host bytes for a UB read")]
    HostAllocationFailed { requested: usize },
    #[error("UB byte at {address:#x} is unknown")]
    UnknownByte { address: u64 },
    #[error("UB transfer length is zero")]
    ZeroTransferLength,
    #[error("UB transfer result size overflows usize")]
    ResultSizeOverflow,
    #[error("MOV_ALIGN_V2 route {source_class}->{destination_class} is not HBM-to-UB")]
    UnsupportedRoute {
        source_class: u8,
        destination_class: u8,
    },
    #[error(transparent)]
    C220Descriptor(#[from] C220MovOutToUbError),
    #[error(transparent)]
    C220OutputDescriptor(#[from] C220DmaMovError),
    #[error(transparent)]
    C310Coordinates(#[from] C310MovAlignCoordinateError),
    #[error(transparent)]
    Source(MappedMemoryError),
    #[error(transparent)]
    Destination(#[from] MappedMemoryError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UbTransferResult {
    pub segment_count: usize,
    pub bytes: usize,
    pub known_bytes: usize,
    pub unknown_bytes: usize,
}

pub struct C220PreparedOutput {
    pub(crate) writes: Vec<(u64, Vec<MemoryByteState>)>,
    pub result: UbTransferResult,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UbMemory {
    bytes: BTreeMap<u64, MemoryByteState>,
    max_tracked_bytes: usize,
    max_transfer_bytes: usize,
}

impl UbMemory {
    pub fn new(max_tracked_bytes: usize, max_transfer_bytes: usize) -> Self {
        Self {
            bytes: BTreeMap::new(),
            max_tracked_bytes,
            max_transfer_bytes,
        }
    }

    pub fn tracked_bytes(&self) -> usize {
        self.bytes.len()
    }

    pub fn read_states(
        &self,
        address: u64,
        len: usize,
    ) -> Result<Vec<MemoryByteState>, UbMemoryError> {
        self.check_range(address, len)?;
        let mut result = Vec::new();
        result
            .try_reserve_exact(len)
            .map_err(|_| UbMemoryError::HostAllocationFailed { requested: len })?;
        for offset in 0..len {
            result.push(
                self.bytes
                    .get(&(address + offset as u64))
                    .copied()
                    .unwrap_or(MemoryByteState::Unknown),
            );
        }
        Ok(result)
    }

    pub fn read_known(&self, address: u64, len: usize) -> Result<Vec<u8>, UbMemoryError> {
        self.check_range(address, len)?;
        let mut result = Vec::new();
        result
            .try_reserve_exact(len)
            .map_err(|_| UbMemoryError::HostAllocationFailed { requested: len })?;
        for offset in 0..len {
            let at = address + offset as u64;
            match self.bytes.get(&at).copied() {
                Some(MemoryByteState::Known(value)) => result.push(value),
                _ => return Err(UbMemoryError::UnknownByte { address: at }),
            }
        }
        Ok(result)
    }

    pub fn write_states(
        &mut self,
        address: u64,
        states: &[MemoryByteState],
    ) -> Result<(), UbMemoryError> {
        self.check_range(address, states.len())?;
        let new_bytes = (0..states.len())
            .filter(|offset| !self.bytes.contains_key(&(address + *offset as u64)))
            .count();
        if self
            .bytes
            .len()
            .checked_add(new_bytes)
            .is_none_or(|count| count > self.max_tracked_bytes)
        {
            return Err(UbMemoryError::TrackedLimitExceeded {
                limit: self.max_tracked_bytes,
            });
        }
        for (offset, &state) in states.iter().enumerate() {
            self.bytes.insert(address + offset as u64, state);
        }
        Ok(())
    }

    pub fn write_segments(
        &mut self,
        segments: &[(u64, Vec<MemoryByteState>)],
    ) -> Result<(), UbMemoryError> {
        let mut new_addresses = BTreeSet::new();
        for (address, states) in segments {
            self.check_range(*address, states.len())?;
            for offset in 0..states.len() {
                let at = *address + offset as u64;
                if !self.bytes.contains_key(&at) {
                    new_addresses.insert(at);
                }
            }
        }
        if self
            .bytes
            .len()
            .checked_add(new_addresses.len())
            .is_none_or(|count| count > self.max_tracked_bytes)
        {
            return Err(UbMemoryError::TrackedLimitExceeded {
                limit: self.max_tracked_bytes,
            });
        }
        for (address, states) in segments {
            for (offset, &state) in states.iter().enumerate() {
                self.bytes.insert(*address + offset as u64, state);
            }
        }
        Ok(())
    }

    pub fn copy_from_hbm(
        &mut self,
        source: &MappedMemory,
        source_address: u64,
        destination_address: u64,
        len: usize,
    ) -> Result<(), UbMemoryError> {
        self.check_range(destination_address, len)?;
        let states = source
            .read_states_at(source_address, len)
            .map_err(UbMemoryError::Source)?;
        self.write_states(destination_address, &states)
    }

    pub fn copy_c220_mov_out_to_ub(
        &mut self,
        source: &MappedMemory,
        descriptor: C220MovOutToUbDescriptor,
        source_address: u64,
        destination_address: u64,
    ) -> Result<UbTransferResult, UbMemoryError> {
        let segments = descriptor.segments(source_address, destination_address)?;
        self.copy_segments(
            source,
            segments.into_iter().map(|segment| {
                (
                    segment.source_hbm,
                    segment.destination_local,
                    segment.bytes as usize,
                )
            }),
        )
    }

    pub fn copy_c310_mov_align_hbm_to_ub(
        &mut self,
        source: &MappedMemory,
        decoded: C310MovAlignDecode,
    ) -> Result<UbTransferResult, UbMemoryError> {
        if decoded.source_memory_class != 10 || decoded.destination_memory_class != 9 {
            return Err(UbMemoryError::UnsupportedRoute {
                source_class: decoded.source_memory_class,
                destination_class: decoded.destination_memory_class,
            });
        }
        if decoded.burst_bytes == 0 {
            return Err(UbMemoryError::ZeroTransferLength);
        }
        let coordinates = decoded.parameters.coordinates()?;
        self.copy_segments(
            source,
            coordinates.into_iter().map(|coordinate| {
                (
                    coordinate.source_address,
                    coordinate.destination_address,
                    decoded.burst_bytes as usize,
                )
            }),
        )
    }

    pub fn copy_c220_mov_ub_to_hbm(
        &self,
        destination: &mut MappedMemory,
        descriptor: C220DmaMovDescriptor,
        source_address: u64,
        destination_address: u64,
    ) -> Result<UbTransferResult, UbMemoryError> {
        let prepared =
            self.prepare_c220_mov_ub_to_hbm(descriptor, source_address, destination_address)?;
        destination.write_segments_at(&prepared.writes)?;
        Ok(prepared.result)
    }

    pub fn prepare_c220_mov_ub_to_hbm(
        &self,
        descriptor: C220DmaMovDescriptor,
        source_address: u64,
        destination_address: u64,
    ) -> Result<C220PreparedOutput, UbMemoryError> {
        let segments = descriptor.segments(source_address, destination_address)?;
        let bytes = segments
            .len()
            .checked_mul(32)
            .ok_or(UbMemoryError::ResultSizeOverflow)?;
        let mut writes = Vec::new();
        writes.try_reserve_exact(segments.len()).map_err(|_| {
            UbMemoryError::HostAllocationFailed {
                requested: segments.len(),
            }
        })?;
        let mut known_bytes = 0;
        for segment in &segments {
            let states = self.read_states(segment.source_local, segment.bytes as usize)?;
            known_bytes += states
                .iter()
                .filter(|state| matches!(state, MemoryByteState::Known(_)))
                .count();
            writes.push((segment.destination_hbm, states));
        }
        Ok(C220PreparedOutput {
            writes,
            result: UbTransferResult {
                segment_count: segments.len(),
                bytes,
                known_bytes,
                unknown_bytes: bytes - known_bytes,
            },
        })
    }

    fn copy_segments(
        &mut self,
        source: &MappedMemory,
        segments: impl Iterator<Item = (u64, u64, usize)>,
    ) -> Result<UbTransferResult, UbMemoryError> {
        let mut writes = Vec::new();
        let mut result = UbTransferResult {
            segment_count: 0,
            bytes: 0,
            known_bytes: 0,
            unknown_bytes: 0,
        };
        for (source_address, destination_address, len) in segments {
            self.check_range(destination_address, len)?;
            let states = source
                .read_states_at(source_address, len)
                .map_err(UbMemoryError::Source)?;
            result.segment_count = result
                .segment_count
                .checked_add(1)
                .ok_or(UbMemoryError::ResultSizeOverflow)?;
            result.bytes = result
                .bytes
                .checked_add(len)
                .ok_or(UbMemoryError::ResultSizeOverflow)?;
            let known = states
                .iter()
                .filter(|state| matches!(state, MemoryByteState::Known(_)))
                .count();
            result.known_bytes = result
                .known_bytes
                .checked_add(known)
                .ok_or(UbMemoryError::ResultSizeOverflow)?;
            writes.push((destination_address, states));
        }
        result.unknown_bytes = result.bytes - result.known_bytes;
        self.write_segments(&writes)?;
        Ok(result)
    }

    pub(crate) fn check_range(&self, address: u64, len: usize) -> Result<(), UbMemoryError> {
        if len > self.max_transfer_bytes {
            return Err(UbMemoryError::TransferLimitExceeded {
                requested: len,
                limit: self.max_transfer_bytes,
            });
        }
        let len = u64::try_from(len).map_err(|_| UbMemoryError::RangeOverflow)?;
        address
            .checked_add(len)
            .ok_or(UbMemoryError::RangeOverflow)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::architecture::Architecture;
    use crate::instruction::c220::mte::{
        C220DmaMovDescriptor, C220MovOutToUbDescriptor, CAPTURED_C220_MOV_OUT_TO_UB_X_WORD,
        CAPTURED_C220_MOV_UB_TO_OUT_WORD, CAPTURED_C220_SUB_MOV_UB_TO_OUT_WORD,
        CAPTURED_C220_TILING_MOV_OUT_TO_UB_WORD,
    };
    use crate::instruction::c310::mte::{
        C310_TILING_MOV_ALIGN_WORD, C310MovAlignDecode, C310MovAlignRegisters,
    };
    use crate::memory::mapped::MappedMemory;
    use crate::memory::pv_memory::PvMemory;
    use crate::memory::region::MemoryRegion;
    use crate::memory::sparse::SparseMemory;

    fn tiling_space(tiling_pointer: u64) -> MappedMemory {
        let pointers = [0x2000_u64, tiling_pointer];
        let image: Vec<u8> = pointers.iter().flat_map(|p| p.to_le_bytes()).collect();
        let regions = vec![
            MemoryRegion::unknown(32),
            MemoryRegion::new(64, 1024_u32.to_le_bytes().to_vec()).unwrap(),
            MemoryRegion::new(image.len() as u64, image).unwrap(),
        ];
        let memory = SparseMemory::new(regions, 64, 64);
        MappedMemory::bind(memory, &[0x2000, tiling_pointer, 0x1000]).unwrap()
    }

    fn input_space(input_pointer: u64, bytes: &[u8]) -> MappedMemory {
        let pointers = [input_pointer, 0x2000_u64];
        let image: Vec<u8> = pointers.iter().flat_map(|p| p.to_le_bytes()).collect();
        let regions = vec![
            MemoryRegion::new(bytes.len() as u64, bytes.to_vec()).unwrap(),
            MemoryRegion::unknown(32),
            MemoryRegion::new(image.len() as u64, image).unwrap(),
        ];
        let memory = SparseMemory::new(regions, 64, 128);
        MappedMemory::bind(memory, &[input_pointer, 0x2000, 0x1000]).unwrap()
    }

    fn output_regions() -> MappedMemory {
        let memory = SparseMemory::new(vec![MemoryRegion::unknown(128)], 128, 128);
        MappedMemory::bind(memory, &[0x3000]).unwrap()
    }

    fn c310_tiling_decode(source: u64) -> C310MovAlignDecode {
        C310MovAlignRegisters {
            destination: 0,
            source,
            shape: 0x4000_0010,
            stride: 0,
            loop_spr: 0x20_0001,
            inner_stride_spr: 0,
            outer_stride_spr: 0,
        }
        .decode_hbm_to_ub(C310_TILING_MOV_ALIGN_WORD)
        .unwrap()
    }

    #[test]
    fn tiling_transfer_preserves_unknown_padding_and_separate_local_namespace() {
        for (architecture, tiling_pointer) in [
            (Architecture::Dav2201, 0x1251_5400),
            (Architecture::Dav3510, 0x1c93_1200),
        ] {
            let hbm = tiling_space(tiling_pointer);
            let mut ub = UbMemory::new(64, 64);
            let result = if architecture == Architecture::Dav2201 {
                let descriptor = C220MovOutToUbDescriptor::decode(
                    CAPTURED_C220_TILING_MOV_OUT_TO_UB_WORD,
                    0x10010,
                )
                .unwrap();
                ub.copy_c220_mov_out_to_ub(&hbm, descriptor, tiling_pointer, 0)
                    .unwrap()
            } else {
                let decoded = c310_tiling_decode(tiling_pointer);
                ub.copy_c310_mov_align_hbm_to_ub(&hbm, decoded).unwrap()
            };
            assert_eq!(
                result,
                UbTransferResult {
                    segment_count: 1,
                    bytes: 32,
                    known_bytes: 4,
                    unknown_bytes: 28,
                }
            );
            assert_eq!(ub.read_known(0, 4).unwrap(), 1024_u32.to_le_bytes());
            assert_eq!(
                ub.read_states(4, 28).unwrap(),
                [MemoryByteState::Unknown; 28]
            );
            assert_eq!(
                ub.read_known(0, 32),
                Err(UbMemoryError::UnknownByte { address: 4 })
            );

            let scalar_local = PvMemory::new(0, 1);
            assert_eq!(scalar_local.read_byte(0x80000), 0);
            assert_eq!(ub.read_known(0, 4).unwrap(), 1024_u32.to_le_bytes());
        }
    }

    #[test]
    fn limits_and_failed_hbm_reads_leave_ub_unchanged() {
        let hbm = tiling_space(0x3000);
        let mut ub = UbMemory::new(4, 32);
        ub.write_states(0, &[MemoryByteState::Known(7)]).unwrap();
        assert_eq!(
            ub.copy_from_hbm(&hbm, 0x3000, 0, 32),
            Err(UbMemoryError::TrackedLimitExceeded { limit: 4 })
        );
        assert_eq!(ub.read_known(0, 1).unwrap(), [7]);
        assert_eq!(ub.tracked_bytes(), 1);
        assert!(matches!(
            ub.copy_from_hbm(&hbm, 0x4000, 0, 1),
            Err(UbMemoryError::Source(_))
        ));
        assert_eq!(ub.read_known(0, 1).unwrap(), [7]);
        assert_eq!(
            ub.write_states(u64::MAX, &[MemoryByteState::Known(1)]),
            Err(UbMemoryError::RangeOverflow)
        );
        assert_eq!(
            ub.read_states(0, 33),
            Err(UbMemoryError::TransferLimitExceeded {
                requested: 33,
                limit: 32,
            })
        );
    }

    #[test]
    fn multi_segment_copy_is_atomic_when_later_source_is_unmapped() {
        let hbm = input_space(0x3000, &[0x5a; 32]);
        let descriptor = C220MovOutToUbDescriptor::decode(
            crate::instruction::c220::mte::CAPTURED_C220_MOV_OUT_TO_UB_X_WORD,
            0x40010,
        )
        .unwrap();
        let mut ub = UbMemory::new(256, 32);
        ub.write_states(0, &[MemoryByteState::Known(0xa5)]).unwrap();
        let before = ub.clone();
        assert!(matches!(
            ub.copy_c220_mov_out_to_ub(&hbm, descriptor, 0x3000, 0),
            Err(UbMemoryError::Source(_))
        ));
        assert_eq!(ub, before);

        let mut c310 = c310_tiling_decode(0x3000);
        c310.parameters.burst_count = 2;
        c310.parameters.source_burst_stride = 32;
        assert!(matches!(
            ub.copy_c310_mov_align_hbm_to_ub(&hbm, c310),
            Err(UbMemoryError::Source(_))
        ));
        assert_eq!(ub, before);
    }

    #[test]
    fn hbm_to_ub_routes_copy_the_decoded_input_spans() {
        let input: Vec<u8> = (0..128).collect();
        let hbm = input_space(0x3000, &input);
        let mut ub = UbMemory::new(512, 128);
        let c220 =
            C220MovOutToUbDescriptor::decode(CAPTURED_C220_MOV_OUT_TO_UB_X_WORD, 0x40010).unwrap();
        let result = ub
            .copy_c220_mov_out_to_ub(&hbm, c220, 0x3000, 0x80)
            .unwrap();
        assert_eq!(
            result,
            UbTransferResult {
                segment_count: 4,
                bytes: 128,
                known_bytes: 128,
                unknown_bytes: 0,
            }
        );
        assert_eq!(ub.read_known(0x80, 128).unwrap(), input);

        let c310 = C310MovAlignRegisters {
            destination: 0x180,
            source: 0x3000,
            shape: 0x0400_0001_0000_0010,
            stride: 0x0000_8000_0000_0080,
            loop_spr: 0x20_0001,
            inner_stride_spr: 0,
            outer_stride_spr: 0,
        }
        .decode_hbm_to_ub(0x74ad_8bae)
        .unwrap();
        assert_eq!(
            ub.copy_c310_mov_align_hbm_to_ub(&hbm, c310).unwrap(),
            UbTransferResult {
                segment_count: 1,
                bytes: 128,
                known_bytes: 128,
                unknown_bytes: 0,
            }
        );
        assert_eq!(ub.read_known(0x180, 128).unwrap(), input);
        assert_eq!(ub.tracked_bytes(), 256);
    }

    #[test]
    fn c310_hbm_to_ub_rejects_other_memory_routes_without_changes() {
        let hbm = tiling_space(0x3000);
        let mut decoded = c310_tiling_decode(0x3000);
        decoded.destination_memory_class = 10;
        let mut ub = UbMemory::new(64, 32);
        assert_eq!(
            ub.copy_c310_mov_align_hbm_to_ub(&hbm, decoded),
            Err(UbMemoryError::UnsupportedRoute {
                source_class: 10,
                destination_class: 10,
            })
        );
        assert_eq!(ub.tracked_bytes(), 0);
        decoded.destination_memory_class = 9;
        decoded.burst_bytes = 0;
        assert_eq!(
            ub.copy_c310_mov_align_hbm_to_ub(&hbm, decoded),
            Err(UbMemoryError::ZeroTransferLength)
        );
        assert_eq!(ub.tracked_bytes(), 0);
    }

    #[test]
    fn c220_captured_output_route_commits_four_segments_atomically() {
        let mut ub = UbMemory::new(384, 32);
        let bytes = (0..128_u8).collect::<Vec<_>>();
        for (index, chunk) in bytes.chunks_exact(32).enumerate() {
            ub.write_states(
                0x100 + (index * 32) as u64,
                &chunk
                    .iter()
                    .copied()
                    .map(MemoryByteState::Known)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        }
        for word in [
            CAPTURED_C220_MOV_UB_TO_OUT_WORD,
            CAPTURED_C220_SUB_MOV_UB_TO_OUT_WORD,
        ] {
            let descriptor = C220DmaMovDescriptor::decode(word, 0x40010).unwrap();
            let mut destination = output_regions();
            assert_eq!(
                ub.copy_c220_mov_ub_to_hbm(&mut destination, descriptor, 0x100, 0x3000)
                    .unwrap(),
                UbTransferResult {
                    segment_count: 4,
                    bytes: 128,
                    known_bytes: 128,
                    unknown_bytes: 0,
                }
            );
            assert_eq!(destination.read_known_at(0x3000, 128).unwrap(), bytes);

            let mut malformed = descriptor;
            malformed.burst_count = 2;
            let before = destination.read_states_at(0x3000, 128).unwrap();
            assert!(matches!(
                ub.copy_c220_mov_ub_to_hbm(&mut destination, malformed, 0x100, 0x3000),
                Err(UbMemoryError::C220OutputDescriptor(_))
            ));
            assert_eq!(destination.read_states_at(0x3000, 128).unwrap(), before);

            let mut outside = output_regions();
            let before = outside.read_states_at(0x3000, 128).unwrap();
            assert!(matches!(
                ub.copy_c220_mov_ub_to_hbm(&mut outside, descriptor, 0x100, 0x3001),
                Err(UbMemoryError::Destination(_))
            ));
            assert_eq!(outside.read_states_at(0x3000, 128).unwrap(), before);
        }

        ub.write_states(0x17f, &[MemoryByteState::Unknown]).unwrap();
        let descriptor =
            C220DmaMovDescriptor::decode(CAPTURED_C220_MOV_UB_TO_OUT_WORD, 0x40010).unwrap();
        let mut destination = output_regions();
        let result = ub
            .copy_c220_mov_ub_to_hbm(&mut destination, descriptor, 0x100, 0x3000)
            .unwrap();
        assert_eq!(result.known_bytes, 127);
        assert_eq!(result.unknown_bytes, 1);
        assert_eq!(
            destination.read_states_at(0x307f, 1).unwrap(),
            [MemoryByteState::Unknown]
        );
    }

    #[test]
    fn c220_gapped_mov_preserves_skipped_regions_and_rejects_partial_writeback() {
        let xm = (1_u64 << 48) | (1 << 32) | (1 << 16) | (2 << 4);
        let data = (0..128_u8).collect::<Vec<_>>();
        let source = input_space(0x4000, &data);
        let input =
            C220MovOutToUbDescriptor::decode(CAPTURED_C220_MOV_OUT_TO_UB_X_WORD, xm).unwrap();
        let mut ub = UbMemory::new(512, 32);
        ub.copy_c220_mov_out_to_ub(&source, input, 0x4000, 0x80)
            .unwrap();
        assert_eq!(ub.read_known(0x80, 32).unwrap(), data[..32]);
        assert_eq!(ub.read_known(0xc0, 32).unwrap(), data[64..96]);
        assert_eq!(
            ub.read_states(0xa0, 32).unwrap(),
            [MemoryByteState::Unknown; 32]
        );

        let output = C220DmaMovDescriptor::decode(CAPTURED_C220_MOV_UB_TO_OUT_WORD, xm).unwrap();
        let mut destination = output_regions();
        ub.copy_c220_mov_ub_to_hbm(&mut destination, output, 0x80, 0x3000)
            .unwrap();
        assert_eq!(destination.read_known_at(0x3000, 32).unwrap(), data[..32]);
        assert_eq!(destination.read_known_at(0x3040, 32).unwrap(), data[64..96]);
        assert_eq!(
            destination.read_states_at(0x3020, 32).unwrap(),
            [MemoryByteState::Unknown; 32]
        );

        let mut rejected = output_regions();
        assert!(matches!(
            ub.copy_c220_mov_ub_to_hbm(&mut rejected, output, 0x80, 0x3040),
            Err(UbMemoryError::Destination(_))
        ));
        assert_eq!(
            rejected.read_states_at(0x3040, 32).unwrap(),
            [MemoryByteState::Unknown; 32]
        );
    }
}
