use std::collections::{BTreeMap, BTreeSet};

use thiserror::Error;

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
    #[error("UB transfer result size overflows usize")]
    ResultSizeOverflow,
    #[error(transparent)]
    Source(MappedMemoryError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UbTransferResult {
    pub segment_count: usize,
    pub bytes: usize,
    pub known_bytes: usize,
    pub unknown_bytes: usize,
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

    pub(crate) fn copy_segments(
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
