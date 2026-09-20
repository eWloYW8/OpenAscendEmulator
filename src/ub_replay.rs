use std::collections::BTreeMap;

use serde::Serialize;
use thiserror::Error;

use crate::acl_address_space::{AclAddressSpaceError, AclReplayAddressSpace};
use crate::addressed_replay::{AddressedReplayMemory, ReplayAddressError};
use crate::mte_c220::{C220MovOutToUbDescriptor, C220MovOutToUbError};
use crate::mte_c310::{C310CapturedMovAlignDecode, C310MovAlignCoordinateError};
use crate::replay_memory::MemoryByteState;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum UbReplayError {
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
    #[error("MOV_ALIGN_V2 route {source_class}->{destination_class} is not UB-to-HBM")]
    UnsupportedOutputRoute {
        source_class: u8,
        destination_class: u8,
    },
    #[error("captured UB-to-HBM route requires one coordinate, got {count}")]
    OutputCoordinateCount { count: usize },
    #[error(transparent)]
    C220Descriptor(#[from] C220MovOutToUbError),
    #[error(transparent)]
    C310Coordinates(#[from] C310MovAlignCoordinateError),
    #[error(transparent)]
    Source(#[from] AclAddressSpaceError),
    #[error(transparent)]
    Destination(#[from] ReplayAddressError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct UbTransferResult {
    pub segment_count: usize,
    pub bytes: usize,
    pub known_bytes: usize,
    pub unknown_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UbReplayMemory {
    bytes: BTreeMap<u64, MemoryByteState>,
    max_tracked_bytes: usize,
    max_transfer_bytes: usize,
}

impl UbReplayMemory {
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
    ) -> Result<Vec<MemoryByteState>, UbReplayError> {
        self.check_range(address, len)?;
        let mut result = Vec::new();
        result
            .try_reserve_exact(len)
            .map_err(|_| UbReplayError::HostAllocationFailed { requested: len })?;
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

    pub fn read_known(&self, address: u64, len: usize) -> Result<Vec<u8>, UbReplayError> {
        self.check_range(address, len)?;
        let mut result = Vec::new();
        result
            .try_reserve_exact(len)
            .map_err(|_| UbReplayError::HostAllocationFailed { requested: len })?;
        for offset in 0..len {
            let at = address + offset as u64;
            match self.bytes.get(&at).copied() {
                Some(MemoryByteState::Known(value)) => result.push(value),
                _ => return Err(UbReplayError::UnknownByte { address: at }),
            }
        }
        Ok(result)
    }

    pub fn write_states(
        &mut self,
        address: u64,
        states: &[MemoryByteState],
    ) -> Result<(), UbReplayError> {
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
            return Err(UbReplayError::TrackedLimitExceeded {
                limit: self.max_tracked_bytes,
            });
        }
        for (offset, &state) in states.iter().enumerate() {
            self.bytes.insert(address + offset as u64, state);
        }
        Ok(())
    }

    pub fn copy_from_hbm(
        &mut self,
        source: &AclReplayAddressSpace,
        source_address: u64,
        destination_address: u64,
        len: usize,
    ) -> Result<(), UbReplayError> {
        self.check_range(destination_address, len)?;
        let states = source.read_states_at(source_address, len)?;
        self.write_states(destination_address, &states)
    }

    pub fn copy_c220_mov_out_to_ub(
        &mut self,
        source: &AclReplayAddressSpace,
        descriptor: C220MovOutToUbDescriptor,
        source_address: u64,
        destination_address: u64,
    ) -> Result<UbTransferResult, UbReplayError> {
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
        source: &AclReplayAddressSpace,
        decoded: C310CapturedMovAlignDecode,
    ) -> Result<UbTransferResult, UbReplayError> {
        if decoded.source_memory_class != 10 || decoded.destination_memory_class != 9 {
            return Err(UbReplayError::UnsupportedRoute {
                source_class: decoded.source_memory_class,
                destination_class: decoded.destination_memory_class,
            });
        }
        if decoded.burst_bytes == 0 {
            return Err(UbReplayError::ZeroTransferLength);
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

    pub fn copy_c310_mov_align_ub_to_hbm(
        &self,
        destination: &mut AddressedReplayMemory,
        decoded: C310CapturedMovAlignDecode,
    ) -> Result<UbTransferResult, UbReplayError> {
        if decoded.source_memory_class != 9 || decoded.destination_memory_class != 10 {
            return Err(UbReplayError::UnsupportedOutputRoute {
                source_class: decoded.source_memory_class,
                destination_class: decoded.destination_memory_class,
            });
        }
        if decoded.burst_bytes == 0 {
            return Err(UbReplayError::ZeroTransferLength);
        }
        let coordinates = decoded.parameters.coordinates()?;
        if coordinates.len() != 1 {
            return Err(UbReplayError::OutputCoordinateCount {
                count: coordinates.len(),
            });
        }
        let coordinate = coordinates[0];
        let states = self.read_states(coordinate.source_address, decoded.burst_bytes as usize)?;
        destination.write_states_at(coordinate.destination_address, &states)?;
        let known_bytes = states
            .iter()
            .filter(|state| matches!(state, MemoryByteState::Known(_)))
            .count();
        Ok(UbTransferResult {
            segment_count: 1,
            bytes: states.len(),
            known_bytes,
            unknown_bytes: states.len() - known_bytes,
        })
    }

    fn copy_segments(
        &mut self,
        source: &AclReplayAddressSpace,
        segments: impl Iterator<Item = (u64, u64, usize)>,
    ) -> Result<UbTransferResult, UbReplayError> {
        let mut staged = self.clone();
        let mut result = UbTransferResult {
            segment_count: 0,
            bytes: 0,
            known_bytes: 0,
            unknown_bytes: 0,
        };
        for (source_address, destination_address, len) in segments {
            staged.check_range(destination_address, len)?;
            let states = source.read_states_at(source_address, len)?;
            staged.write_states(destination_address, &states)?;
            result.segment_count = result
                .segment_count
                .checked_add(1)
                .ok_or(UbReplayError::ResultSizeOverflow)?;
            result.bytes = result
                .bytes
                .checked_add(len)
                .ok_or(UbReplayError::ResultSizeOverflow)?;
            let known = states
                .iter()
                .filter(|state| matches!(state, MemoryByteState::Known(_)))
                .count();
            result.known_bytes = result
                .known_bytes
                .checked_add(known)
                .ok_or(UbReplayError::ResultSizeOverflow)?;
        }
        result.unknown_bytes = result.bytes - result.known_bytes;
        *self = staged;
        Ok(result)
    }

    fn check_range(&self, address: u64, len: usize) -> Result<(), UbReplayError> {
        if len > self.max_transfer_bytes {
            return Err(UbReplayError::TransferLimitExceeded {
                requested: len,
                limit: self.max_transfer_bytes,
            });
        }
        let len = u64::try_from(len).map_err(|_| UbReplayError::RangeOverflow)?;
        address
            .checked_add(len)
            .ok_or(UbReplayError::RangeOverflow)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use crate::acl_address_space::AclArgumentImage;
    use crate::acl_args::AclArgumentPlan;
    use crate::addressed_replay::AddressedReplayMemory;
    use crate::architecture::Architecture;
    use crate::kernel_config::KernelConfigDocument;
    use crate::mte_c220::{
        C220MovOutToUbDescriptor, CAPTURED_C220_MOV_OUT_TO_UB_X_WORD,
        CAPTURED_C220_TILING_MOV_OUT_TO_UB_WORD,
    };
    use crate::mte_c310::{
        C310_TILING_MOV_ALIGN_WORD, C310CapturedMovAlignRegisters, C310TilingMovAlignRegisters,
    };
    use crate::pv_memory::PvMemory;
    use crate::replay_memory::ReplayMemory;
    use crate::replay_seed::ReplaySeed;

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

    struct TempTree(PathBuf);

    impl TempTree {
        fn new() -> Self {
            let id = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir()
                .join(format!("open-ascend-ub-replay-{}-{id}", std::process::id()));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TempTree {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn tiling_space(tiling_pointer: u64) -> (TempTree, AclReplayAddressSpace) {
        let tree = TempTree::new();
        let path = tree.0.join("tiling.bin");
        fs::write(&path, 1024_u32.to_le_bytes()).unwrap();
        let config = KernelConfigDocument::from_slice(
            format!(
                "{{\"old_mode\":\"0\",\"output_name\":\"z.bin\",\"output_size\":\"32\",\"tiling_data_path\":\"{};4\"}}",
                path.display()
            )
            .as_bytes(),
        )
        .unwrap()
        .decode()
        .unwrap();
        let plan = AclArgumentPlan::from_config(&config).unwrap();
        let seed = ReplaySeed::load(&config, 4).unwrap();
        let memory = ReplayMemory::new(seed, 64, 64);
        let regions = AddressedReplayMemory::bind(memory, &[0x2000, tiling_pointer]).unwrap();
        let image = AclArgumentImage::new(&plan, 0x1000, &[0x2000, tiling_pointer], &[]).unwrap();
        (tree, AclReplayAddressSpace::new(image, regions).unwrap())
    }

    fn input_space(input_pointer: u64, bytes: &[u8]) -> (TempTree, AclReplayAddressSpace) {
        let tree = TempTree::new();
        let path = tree.0.join("input.bin");
        fs::write(&path, bytes).unwrap();
        let config = KernelConfigDocument::from_slice(
            format!(
                "{{\"old_mode\":\"0\",\"input_path\":\"{}\",\"input_size\":\"{}\",\"output_name\":\"z.bin\",\"output_size\":\"32\"}}",
                path.display(),
                bytes.len(),
            )
            .as_bytes(),
        )
        .unwrap()
        .decode()
        .unwrap();
        let plan = AclArgumentPlan::from_config(&config).unwrap();
        let seed = ReplaySeed::load(&config, bytes.len() as u64).unwrap();
        let memory = ReplayMemory::new(seed, 64, 128);
        let regions = AddressedReplayMemory::bind(memory, &[input_pointer, 0x2000]).unwrap();
        let image = AclArgumentImage::new(&plan, 0x1000, &[input_pointer, 0x2000], &[]).unwrap();
        (tree, AclReplayAddressSpace::new(image, regions).unwrap())
    }

    fn output_regions() -> AddressedReplayMemory {
        let config = KernelConfigDocument::from_slice(
            br#"{"old_mode":"0","output_name":"z.bin","output_size":"128"}"#,
        )
        .unwrap()
        .decode()
        .unwrap();
        let seed = ReplaySeed::load(&config, 0).unwrap();
        let memory = ReplayMemory::new(seed, 128, 128);
        AddressedReplayMemory::bind(memory, &[0x3000]).unwrap()
    }

    #[test]
    fn tiling_transfer_preserves_unknown_padding_and_separate_local_namespace() {
        for (architecture, tiling_pointer) in [
            (Architecture::Dav2201, 0x1251_5400),
            (Architecture::Dav3510, 0x1c93_1200),
        ] {
            let (_tree, hbm) = tiling_space(tiling_pointer);
            let mut ub = UbReplayMemory::new(64, 64);
            let result = if architecture == Architecture::Dav2201 {
                let descriptor = C220MovOutToUbDescriptor::decode(
                    CAPTURED_C220_TILING_MOV_OUT_TO_UB_WORD,
                    0x10010,
                )
                .unwrap();
                ub.copy_c220_mov_out_to_ub(&hbm, descriptor, tiling_pointer, 0)
                    .unwrap()
            } else {
                let decoded = C310TilingMovAlignRegisters {
                    source_xreg1: tiling_pointer,
                    shape_xreg4: 0x4000_0010,
                    destination_and_stride_xreg7: 0,
                    loop_spr105: 0x20_0001,
                    inner_stride_spr106: 0,
                    outer_stride_spr107: 0,
                }
                .decode(C310_TILING_MOV_ALIGN_WORD)
                .unwrap();
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
                Err(UbReplayError::UnknownByte { address: 4 })
            );

            let scalar_local = PvMemory::new(architecture, 0, 1);
            assert_eq!(scalar_local.read_byte(0x80000), 0);
            assert_eq!(ub.read_known(0, 4).unwrap(), 1024_u32.to_le_bytes());
        }
    }

    #[test]
    fn limits_and_failed_hbm_reads_leave_ub_unchanged() {
        let (_tree, hbm) = tiling_space(0x3000);
        let mut ub = UbReplayMemory::new(4, 32);
        ub.write_states(0, &[MemoryByteState::Known(7)]).unwrap();
        assert_eq!(
            ub.copy_from_hbm(&hbm, 0x3000, 0, 32),
            Err(UbReplayError::TrackedLimitExceeded { limit: 4 })
        );
        assert_eq!(ub.read_known(0, 1).unwrap(), [7]);
        assert_eq!(ub.tracked_bytes(), 1);
        assert!(matches!(
            ub.copy_from_hbm(&hbm, 0x4000, 0, 1),
            Err(UbReplayError::Source(_))
        ));
        assert_eq!(ub.read_known(0, 1).unwrap(), [7]);
        assert_eq!(
            ub.write_states(u64::MAX, &[MemoryByteState::Known(1)]),
            Err(UbReplayError::RangeOverflow)
        );
        assert_eq!(
            ub.read_states(0, 33),
            Err(UbReplayError::TransferLimitExceeded {
                requested: 33,
                limit: 32,
            })
        );
    }

    #[test]
    fn multi_segment_copy_is_atomic_when_later_source_is_unmapped() {
        let (_tree, hbm) = input_space(0x3000, &[0x5a; 32]);
        let descriptor = C220MovOutToUbDescriptor::decode(
            crate::mte_c220::CAPTURED_C220_MOV_OUT_TO_UB_X_WORD,
            0x40010,
        )
        .unwrap();
        let mut ub = UbReplayMemory::new(256, 32);
        ub.write_states(0, &[MemoryByteState::Known(0xa5)]).unwrap();
        let before = ub.clone();
        assert!(matches!(
            ub.copy_c220_mov_out_to_ub(&hbm, descriptor, 0x3000, 0),
            Err(UbReplayError::Source(_))
        ));
        assert_eq!(ub, before);

        let mut c310 = C310TilingMovAlignRegisters {
            source_xreg1: 0x3000,
            shape_xreg4: 0x4000_0010,
            destination_and_stride_xreg7: 0,
            loop_spr105: 0x20_0001,
            inner_stride_spr106: 0,
            outer_stride_spr107: 0,
        }
        .decode(C310_TILING_MOV_ALIGN_WORD)
        .unwrap();
        c310.parameters.burst_count = 2;
        c310.parameters.source_burst_stride = 32;
        assert!(matches!(
            ub.copy_c310_mov_align_hbm_to_ub(&hbm, c310),
            Err(UbReplayError::Source(_))
        ));
        assert_eq!(ub, before);
    }

    #[test]
    fn hbm_to_ub_routes_copy_the_decoded_input_spans() {
        let input: Vec<u8> = (0..128).collect();
        let (_tree, hbm) = input_space(0x3000, &input);
        let mut ub = UbReplayMemory::new(512, 128);
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

        let c310 = C310CapturedMovAlignRegisters {
            destination: 0x180,
            source: 0x3000,
            shape: 0x0400_0001_0000_0010,
            stride: 0x0000_8000_0000_0080,
            loop_spr: 0x20_0001,
            inner_stride_spr: 0,
            outer_stride_spr: 0,
        }
        .decode_hbm_to_ub_word(0x74ad_8bae)
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
        let (_tree, hbm) = tiling_space(0x3000);
        let mut decoded = C310TilingMovAlignRegisters {
            source_xreg1: 0x3000,
            shape_xreg4: 0x4000_0010,
            destination_and_stride_xreg7: 0,
            loop_spr105: 0x20_0001,
            inner_stride_spr106: 0,
            outer_stride_spr107: 0,
        }
        .decode(C310_TILING_MOV_ALIGN_WORD)
        .unwrap();
        decoded.destination_memory_class = 10;
        let mut ub = UbReplayMemory::new(64, 32);
        assert_eq!(
            ub.copy_c310_mov_align_hbm_to_ub(&hbm, decoded),
            Err(UbReplayError::UnsupportedRoute {
                source_class: 10,
                destination_class: 10,
            })
        );
        assert_eq!(ub.tracked_bytes(), 0);
        decoded.destination_memory_class = 9;
        decoded.burst_bytes = 0;
        assert_eq!(
            ub.copy_c310_mov_align_hbm_to_ub(&hbm, decoded),
            Err(UbReplayError::ZeroTransferLength)
        );
        assert_eq!(ub.tracked_bytes(), 0);
    }

    #[test]
    fn c310_captured_output_route_moves_one_ub_span_to_hbm() {
        let mut ub = UbReplayMemory::new(384, 128);
        let bytes = (0..128_u8).collect::<Vec<_>>();
        ub.write_states(
            0x100,
            &bytes
                .iter()
                .copied()
                .map(MemoryByteState::Known)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let mut destination = output_regions();
        let decoded = C310CapturedMovAlignRegisters {
            destination: 0x3000,
            source: 0x100,
            shape: 0x0000_0001_0000_0010,
            stride: 0x0000_8000_0000_0080,
            loop_spr: 0x20_0001,
            inner_stride_spr: 0,
            outer_stride_spr: 0,
        }
        .decode_captured_word(0x74c4_16a0)
        .unwrap();
        assert_eq!(
            ub.copy_c310_mov_align_ub_to_hbm(&mut destination, decoded)
                .unwrap(),
            UbTransferResult {
                segment_count: 1,
                bytes: 128,
                known_bytes: 128,
                unknown_bytes: 0,
            }
        );
        assert_eq!(destination.read_known_at(0x3000, 128).unwrap(), bytes);

        ub.write_states(0x17f, &[MemoryByteState::Unknown]).unwrap();
        let mut mixed = output_regions();
        assert_eq!(
            ub.copy_c310_mov_align_ub_to_hbm(&mut mixed, decoded)
                .unwrap(),
            UbTransferResult {
                segment_count: 1,
                bytes: 128,
                known_bytes: 127,
                unknown_bytes: 1,
            }
        );
        assert_eq!(
            mixed.read_states_at(0x307f, 1).unwrap(),
            [MemoryByteState::Unknown]
        );

        let mut rejected = output_regions();
        let mut wrong_route = decoded;
        wrong_route.source_memory_class = 10;
        assert_eq!(
            ub.copy_c310_mov_align_ub_to_hbm(&mut rejected, wrong_route),
            Err(UbReplayError::UnsupportedOutputRoute {
                source_class: 10,
                destination_class: 10,
            })
        );
        let mut two_coordinates = decoded;
        two_coordinates.parameters.burst_count = 2;
        assert_eq!(
            ub.copy_c310_mov_align_ub_to_hbm(&mut rejected, two_coordinates),
            Err(UbReplayError::OutputCoordinateCount { count: 2 })
        );
        let mut outside = decoded;
        outside.parameters.destination_base = 0x4000;
        assert!(matches!(
            ub.copy_c310_mov_align_ub_to_hbm(&mut rejected, outside),
            Err(UbReplayError::Destination(
                ReplayAddressError::Unmapped { .. }
            ))
        ));
        assert_eq!(
            rejected.read_states_at(0x3000, 128).unwrap(),
            [MemoryByteState::Unknown; 128]
        );
    }
}
