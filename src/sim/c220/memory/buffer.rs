use std::collections::BTreeMap;

use thiserror::Error;

use crate::memory::sparse::MemoryByteState;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum C220LocalBufferError {
    #[error("local-buffer range overflows u64")]
    RangeOverflow,
    #[error("local-buffer range [{address:#x}, {end:#x}) exceeds {capacity} bytes")]
    OutOfBounds {
        address: u64,
        end: u64,
        capacity: u64,
    },
    #[error("cannot wrap an access through a zero-sized local buffer")]
    ZeroCapacity,
    #[error("cannot reserve {requested} host bytes for a local-buffer read")]
    HostAllocationFailed { requested: usize },
    #[error("local-buffer byte at {address:#x} is unknown")]
    UnknownByte { address: u64 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220LocalBuffer {
    capacity: u64,
    bytes: BTreeMap<u64, MemoryByteState>,
}

impl C220LocalBuffer {
    pub fn new(capacity: u64) -> Self {
        Self {
            capacity,
            bytes: BTreeMap::new(),
        }
    }

    pub const fn capacity(&self) -> u64 {
        self.capacity
    }

    pub fn tracked_bytes(&self) -> usize {
        self.bytes.len()
    }

    pub fn read_states(
        &self,
        address: u64,
        len: usize,
    ) -> Result<Vec<MemoryByteState>, C220LocalBufferError> {
        self.check_range(address, len)?;
        self.read_with(
            len,
            |offset| address + offset as u64,
            MemoryByteState::Unknown,
        )
    }

    pub fn read_states_wrapped(
        &self,
        address: u64,
        len: usize,
    ) -> Result<Vec<MemoryByteState>, C220LocalBufferError> {
        if self.capacity == 0 {
            return Err(C220LocalBufferError::ZeroCapacity);
        }
        self.read_with(
            len,
            |offset| address.wrapping_add(offset as u64) % self.capacity,
            MemoryByteState::Unknown,
        )
    }

    /// Reads without capacity wrapping. Untouched bytes are zero; explicitly
    /// unknown bytes remain errors.
    pub fn read_initialized_linear(
        &self,
        address: u64,
        len: usize,
    ) -> Result<Vec<u8>, C220LocalBufferError> {
        if len != 0 {
            address
                .checked_add((len - 1) as u64)
                .ok_or(C220LocalBufferError::RangeOverflow)?;
        }
        let states = self.read_with(
            len,
            |offset| address + offset as u64,
            MemoryByteState::Known(0),
        )?;
        collect_known(states, |offset| address + offset as u64)
    }

    pub fn read_known(&self, address: u64, len: usize) -> Result<Vec<u8>, C220LocalBufferError> {
        let states = self.read_states(address, len)?;
        collect_known(states, |offset| address + offset as u64)
    }

    pub fn read_known_wrapped(
        &self,
        address: u64,
        len: usize,
    ) -> Result<Vec<u8>, C220LocalBufferError> {
        if self.capacity == 0 {
            return Err(C220LocalBufferError::ZeroCapacity);
        }
        let states = self.read_states_wrapped(address, len)?;
        collect_known(states, |offset| {
            address.wrapping_add(offset as u64) % self.capacity
        })
    }

    pub fn write_states(
        &mut self,
        address: u64,
        states: &[MemoryByteState],
    ) -> Result<(), C220LocalBufferError> {
        self.check_range(address, states.len())?;
        for (offset, &state) in states.iter().enumerate() {
            self.bytes.insert(address + offset as u64, state);
        }
        Ok(())
    }

    pub fn write_states_wrapped(
        &mut self,
        address: u64,
        states: &[MemoryByteState],
    ) -> Result<(), C220LocalBufferError> {
        if self.capacity == 0 {
            return Err(C220LocalBufferError::ZeroCapacity);
        }
        for (offset, &state) in states.iter().enumerate() {
            let at = address.wrapping_add(offset as u64) % self.capacity;
            self.bytes.insert(at, state);
        }
        Ok(())
    }

    pub fn write_known(&mut self, address: u64, bytes: &[u8]) -> Result<(), C220LocalBufferError> {
        let states = bytes
            .iter()
            .copied()
            .map(MemoryByteState::Known)
            .collect::<Vec<_>>();
        self.write_states(address, &states)
    }

    pub fn write_known_wrapped(
        &mut self,
        address: u64,
        bytes: &[u8],
    ) -> Result<(), C220LocalBufferError> {
        let states = bytes
            .iter()
            .copied()
            .map(MemoryByteState::Known)
            .collect::<Vec<_>>();
        self.write_states_wrapped(address, &states)
    }

    fn read_with(
        &self,
        len: usize,
        address: impl Fn(usize) -> u64,
        default: MemoryByteState,
    ) -> Result<Vec<MemoryByteState>, C220LocalBufferError> {
        let mut result = Vec::new();
        result
            .try_reserve_exact(len)
            .map_err(|_| C220LocalBufferError::HostAllocationFailed { requested: len })?;
        for offset in 0..len {
            result.push(self.bytes.get(&address(offset)).copied().unwrap_or(default));
        }
        Ok(result)
    }

    fn check_range(&self, address: u64, len: usize) -> Result<(), C220LocalBufferError> {
        let len = u64::try_from(len).map_err(|_| C220LocalBufferError::RangeOverflow)?;
        let end = address
            .checked_add(len)
            .ok_or(C220LocalBufferError::RangeOverflow)?;
        if end > self.capacity {
            return Err(C220LocalBufferError::OutOfBounds {
                address,
                end,
                capacity: self.capacity,
            });
        }
        Ok(())
    }
}

fn collect_known(
    states: Vec<MemoryByteState>,
    address: impl Fn(usize) -> u64,
) -> Result<Vec<u8>, C220LocalBufferError> {
    states
        .into_iter()
        .enumerate()
        .map(|(offset, state)| match state {
            MemoryByteState::Known(byte) => Ok(byte),
            MemoryByteState::Unknown => Err(C220LocalBufferError::UnknownByte {
                address: address(offset),
            }),
        })
        .collect()
}
