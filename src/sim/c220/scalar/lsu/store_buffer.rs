use super::C220LsuRequestId;
use super::miss_buffer::C220LsuMissBuffer;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum C220LsuMemory {
    Ub,
    External,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct C220LsuLineKey {
    pub address: u64,
    pub memory: C220LsuMemory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220LsuStoreState {
    Idle,
    Fetching,
    Ready,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220LsuStoreConfig {
    pub line_bytes: usize,
    pub main_entries: usize,
    pub sub_entries: usize,
    pub timeout_ticks: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum C220LsuStoreError {
    #[error("store-buffer geometry must be nonzero")]
    InvalidConfig,
    #[error("store-buffer line address is not aligned")]
    UnalignedLine,
    #[error("store-buffer access extends beyond its line")]
    InvalidRange,
    #[error("store-buffer entry does not exist")]
    MissingEntry,
    #[error("store-buffer entry cannot accept a request")]
    Blocked,
    #[error("write response requires a pending UB store")]
    InvalidWriteResponse,
    #[error("write-allocated store has not received a complete line")]
    IncompleteLine,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220LsuCompletion {
    Store(C220LsuRequestId),
    Load(C220LsuRequestId),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220LsuPairPart {
    Both,
    First,
    Second,
}

/// Completion notifications in delivery order, with the line used by linked loads.
/// The receiver still owns register writeback and paired-instruction retirement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220LsuWriteCompletion {
    pub key: C220LsuLineKey,
    pub line: Vec<u8>,
    pub notifications: Vec<C220LsuCompletion>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220LsuStoreEntry {
    key: C220LsuLineKey,
    state: C220LsuStoreState,
    cache_hit: bool,
    forbidden: bool,
    remaining_ticks: u32,
    requests: Vec<C220LsuRequestId>,
    bytes: Vec<u8>,
    valid: Vec<bool>,
}

impl C220LsuStoreEntry {
    /// Apply this entry's current data and byte mask to an atomic cache hit.
    /// Request retirement and removal remain the controller's responsibility.
    pub fn write_atomic_hit(
        &self,
        cache: &mut super::cache::C220DataCache,
        location: super::cache::C220CacheLocation,
    ) -> Result<super::cache::C220AtomicCacheHit, super::cache::C220CacheError> {
        cache.write_atomic_hit(location, &self.bytes, &self.valid)
    }

    pub const fn key(&self) -> C220LsuLineKey {
        self.key
    }

    pub const fn state(&self) -> C220LsuStoreState {
        self.state
    }

    pub const fn cache_hit(&self) -> bool {
        self.cache_hit
    }

    pub const fn forbidden(&self) -> bool {
        self.forbidden
    }

    pub const fn remaining_ticks(&self) -> u32 {
        self.remaining_ticks
    }

    pub fn requests(&self) -> &[C220LsuRequestId] {
        &self.requests
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn valid_bytes(&self) -> &[bool] {
        &self.valid
    }

    /// Apply only written bytes to an already allocated cache line.
    pub fn write_valid_bytes(&self, line: &mut [u8]) -> Result<(), C220LsuStoreError> {
        if line.len() != self.bytes.len() {
            return Err(C220LsuStoreError::InvalidRange);
        }
        for ((destination, source), valid) in line.iter_mut().zip(&self.bytes).zip(&self.valid) {
            if *valid {
                *destination = *source;
            }
        }
        Ok(())
    }

    pub fn covered_bytes(&self, offset: usize, bytes: usize) -> Result<usize, C220LsuStoreError> {
        let range = checked_range(offset, bytes, self.bytes.len())?;
        Ok(self.valid[range].iter().filter(|valid| **valid).count())
    }

    /// Copy a prefix whose length equals the number of covered request bytes.
    /// The caller initializes the destination from a cache hit or zeroes.
    /// Pair loads have a separate forwarding path.
    pub fn forward_scalar(
        &self,
        offset: usize,
        destination: &mut [u8],
    ) -> Result<usize, C220LsuStoreError> {
        if destination.len() > 8 {
            return Err(C220LsuStoreError::InvalidRange);
        }
        let count = self.covered_bytes(offset, destination.len())?;
        destination[..count].copy_from_slice(&self.bytes[offset..offset + count]);
        Ok(count)
    }

    /// Forward pair-load operands into their pending destination values.
    /// An unsplit pair checks each operand's first validity bit, then copies
    /// that entire operand. Split requests copy their selected operand directly.
    /// Copied values are zero-extended; untouched destinations retain their value.
    /// The returned mask identifies changed operands, not instruction retirement.
    pub fn forward_pair(
        &self,
        offset: usize,
        width: usize,
        part: C220LsuPairPart,
        destinations: &mut [u64; 2],
    ) -> Result<[bool; 2], C220LsuStoreError> {
        if !matches!(width, 1 | 2 | 4 | 8) {
            return Err(C220LsuStoreError::InvalidRange);
        }
        let count = if part == C220LsuPairPart::Both { 2 } else { 1 };
        checked_range(offset, count * width, self.bytes.len())?;
        let mut changed = [false; 2];
        for index in 0..2 {
            let source = match (part, index) {
                (C220LsuPairPart::Both, _) => offset + index * width,
                (C220LsuPairPart::First, 0) | (C220LsuPairPart::Second, 1) => offset,
                _ => continue,
            };
            if part == C220LsuPairPart::Both && !self.valid[source] {
                continue;
            }
            let mut bytes = [0; 8];
            bytes[..width].copy_from_slice(&self.bytes[source..source + width]);
            destinations[index] = u64::from_le_bytes(bytes);
            changed[index] = true;
        }
        Ok(changed)
    }
}

/// Line-coalescing storage and flush eligibility. Cache allocation, lower-level
/// transactions and instruction retirement are owned by the LSU controller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220LsuStoreBuffer {
    config: C220LsuStoreConfig,
    entries: Vec<C220LsuStoreEntry>,
}

impl C220LsuStoreBuffer {
    pub const fn line_bytes(&self) -> usize {
        self.config.line_bytes
    }

    pub fn new(config: C220LsuStoreConfig) -> Result<Self, C220LsuStoreError> {
        if config.line_bytes == 0 || config.main_entries == 0 || config.sub_entries == 0 {
            return Err(C220LsuStoreError::InvalidConfig);
        }
        Ok(Self {
            config,
            entries: Vec::new(),
        })
    }

    pub fn entries(&self) -> &[C220LsuStoreEntry] {
        &self.entries
    }

    pub fn entry(&self, key: C220LsuLineKey) -> Option<&C220LsuStoreEntry> {
        self.entries.iter().find(|entry| entry.key == key)
    }

    /// A matching line consumes a sub-entry, even when all main entries are used.
    pub fn full(&self, key: C220LsuLineKey) -> bool {
        self.entry(key).map_or_else(
            || self.entries.len() >= self.config.main_entries,
            |entry| entry.requests.len() >= self.config.sub_entries,
        )
    }

    pub fn store(
        &mut self,
        key: C220LsuLineKey,
        request: C220LsuRequestId,
        offset: usize,
        bytes: &[u8],
        cache_hit: bool,
    ) -> Result<(), C220LsuStoreError> {
        self.insert(key, request, offset, bytes, cache_hit, false)
    }

    /// Atomic stores may merge into a forbidden entry without changing its
    /// ownership, state, initial hit result, or timeout.
    pub fn store_atomic(
        &mut self,
        key: C220LsuLineKey,
        request: C220LsuRequestId,
        offset: usize,
        bytes: &[u8],
        cache_hit: bool,
    ) -> Result<(), C220LsuStoreError> {
        self.insert(key, request, offset, bytes, cache_hit, true)
    }

    fn insert(
        &mut self,
        key: C220LsuLineKey,
        request: C220LsuRequestId,
        offset: usize,
        bytes: &[u8],
        cache_hit: bool,
        allow_forbidden: bool,
    ) -> Result<(), C220LsuStoreError> {
        if !key.address.is_multiple_of(self.config.line_bytes as u64) {
            return Err(C220LsuStoreError::UnalignedLine);
        }
        let range = checked_range(offset, bytes.len(), self.config.line_bytes)?;
        if self.full(key)
            || (!allow_forbidden && self.entry(key).is_some_and(|entry| entry.forbidden))
        {
            return Err(C220LsuStoreError::Blocked);
        }
        let index = match self.entries.iter().position(|entry| entry.key == key) {
            Some(index) => index,
            None => {
                self.entries.push(C220LsuStoreEntry {
                    key,
                    state: C220LsuStoreState::Idle,
                    cache_hit,
                    forbidden: false,
                    remaining_ticks: self.config.timeout_ticks,
                    requests: Vec::new(),
                    bytes: vec![0; self.config.line_bytes],
                    valid: vec![false; self.config.line_bytes],
                });
                self.entries.len() - 1
            }
        };
        let entry = &mut self.entries[index];
        entry.bytes[range.clone()].copy_from_slice(bytes);
        entry.valid[range].fill(true);
        entry.requests.push(request);
        Ok(())
    }

    /// Called once at the start of each store-buffer processing tick.
    /// Coalescing another store does not restart an entry's timeout.
    pub fn advance_timeouts(&mut self) {
        for entry in &mut self.entries {
            entry.remaining_ticks = entry.remaining_ticks.saturating_sub(1);
        }
    }

    pub fn ready_to_flush(&self) -> impl Iterator<Item = &C220LsuStoreEntry> {
        self.entries.iter().filter(|entry| {
            entry.state == C220LsuStoreState::Idle
                && (entry.remaining_ticks == 0
                    || entry.requests.len() >= self.config.sub_entries
                    || entry.valid.iter().all(|valid| *valid))
        })
    }

    pub fn set_state(
        &mut self,
        key: C220LsuLineKey,
        state: C220LsuStoreState,
    ) -> Result<(), C220LsuStoreError> {
        self.entry_mut(key)?.state = state;
        Ok(())
    }

    pub fn set_forbidden(
        &mut self,
        key: C220LsuLineKey,
        forbidden: bool,
    ) -> Result<(), C220LsuStoreError> {
        self.entry_mut(key)?.forbidden = forbidden;
        Ok(())
    }

    /// Fill unwritten bytes from a returned line; newer stores take precedence.
    /// Data merging does not itself deliver completions or release the entry.
    pub fn merge_line(
        &mut self,
        key: C220LsuLineKey,
        bytes: &[u8],
    ) -> Result<(), C220LsuStoreError> {
        if bytes.len() != self.config.line_bytes {
            return Err(C220LsuStoreError::InvalidRange);
        }
        let entry = self.entry_mut(key)?;
        for (index, value) in bytes.iter().enumerate() {
            if !entry.valid[index] {
                entry.bytes[index] = *value;
                entry.valid[index] = true;
            }
        }
        Ok(())
    }

    pub fn remove(&mut self, key: C220LsuLineKey) -> Option<C220LsuStoreEntry> {
        let index = self.entries.iter().position(|entry| entry.key == key)?;
        Some(self.entries.remove(index))
    }

    /// Handle a UB write acknowledgement using the live backing-memory line.
    /// No completion becomes visible before the backing write. This operation
    /// does not advance time or model the response transport's latency.
    pub fn complete_ub_write(
        &mut self,
        key: C220LsuLineKey,
        write_allocate: bool,
        backing_line: &mut [u8],
        misses: &mut C220LsuMissBuffer,
    ) -> Result<C220LsuWriteCompletion, C220LsuStoreError> {
        if backing_line.len() != self.config.line_bytes
            || misses.line_bytes() != self.config.line_bytes
        {
            return Err(C220LsuStoreError::InvalidRange);
        }
        let bytes = self.prepare_ub_write(key, write_allocate, backing_line)?;
        self.merge_line(key, &bytes)?;
        let entry = self.entry(key).expect("validated pending store");
        backing_line.copy_from_slice(entry.bytes());
        let mut notifications: Vec<_> = entry
            .requests()
            .iter()
            .copied()
            .map(C220LsuCompletion::Store)
            .collect();
        let line = entry.bytes().to_vec();
        if misses.entry(key).is_some() {
            misses
                .receive_line(key, &line)
                .expect("validated matching miss-buffer geometry");
            notifications.extend(
                misses
                    .entry(key)
                    .expect("linked miss exists")
                    .requests()
                    .iter()
                    .copied()
                    .map(C220LsuCompletion::Load),
            );
        }
        self.remove(key);
        misses.remove(key);
        Ok(C220LsuWriteCompletion {
            key,
            line,
            notifications,
        })
    }

    pub(super) fn prepare_ub_write(
        &self,
        key: C220LsuLineKey,
        write_allocate: bool,
        backing_line: &[u8],
    ) -> Result<Vec<u8>, C220LsuStoreError> {
        if backing_line.len() != self.config.line_bytes {
            return Err(C220LsuStoreError::InvalidRange);
        }
        let entry = self.entry(key).ok_or(C220LsuStoreError::MissingEntry)?;
        if key.memory != C220LsuMemory::Ub || entry.state == C220LsuStoreState::Idle {
            return Err(C220LsuStoreError::InvalidWriteResponse);
        }
        if write_allocate && entry.valid.iter().any(|valid| !valid) {
            return Err(C220LsuStoreError::IncompleteLine);
        }
        let mut bytes = backing_line.to_vec();
        entry.write_valid_bytes(&mut bytes)?;
        Ok(bytes)
    }

    fn entry_mut(
        &mut self,
        key: C220LsuLineKey,
    ) -> Result<&mut C220LsuStoreEntry, C220LsuStoreError> {
        self.entries
            .iter_mut()
            .find(|entry| entry.key == key)
            .ok_or(C220LsuStoreError::MissingEntry)
    }
}

fn checked_range(
    offset: usize,
    bytes: usize,
    line_bytes: usize,
) -> Result<std::ops::Range<usize>, C220LsuStoreError> {
    let end = offset
        .checked_add(bytes)
        .filter(|end| *end <= line_bytes)
        .ok_or(C220LsuStoreError::InvalidRange)?;
    Ok(offset..end)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::c220::scalar::lsu::C220LsuRequestPipeline;
    use crate::sim::c220::scalar::lsu::miss_buffer::{C220LsuMissConfig, C220LsuMissReadAction};

    #[test]
    fn pair_forwarding_uses_operand_start_bits_and_split_destination() {
        let mut pipeline = C220LsuRequestPipeline::new(4).unwrap();
        let request = pipeline.admit(0, false).unwrap().unwrap().0;
        let mut stores = C220LsuStoreBuffer::new(C220LsuStoreConfig {
            line_bytes: 64,
            main_entries: 1,
            sub_entries: 4,
            timeout_ticks: 4,
        })
        .unwrap();
        let key = C220LsuLineKey {
            address: 0,
            memory: C220LsuMemory::Ub,
        };
        stores.store(key, request, 4, &[0x80], false).unwrap();
        stores.store(key, request, 9, &[0xee], false).unwrap();
        let entry = stores.entry(key).unwrap();
        let mut values = [u64::MAX; 2];
        assert_eq!(
            entry
                .forward_pair(4, 4, C220LsuPairPart::Both, &mut values)
                .unwrap(),
            [true, false]
        );
        assert_eq!(values, [0x80, u64::MAX]);
        assert_eq!(
            entry
                .forward_pair(8, 4, C220LsuPairPart::Second, &mut values)
                .unwrap(),
            [false, true]
        );
        assert_eq!(values, [0x80, 0xee00]);
        assert_eq!(
            entry
                .forward_pair(8, 4, C220LsuPairPart::First, &mut values)
                .unwrap(),
            [true, false]
        );
        assert_eq!(values, [0xee00, 0xee00]);
        let before = values;
        assert_eq!(
            entry.forward_pair(60, 4, C220LsuPairPart::Both, &mut values),
            Err(C220LsuStoreError::InvalidRange)
        );
        assert_eq!(values, before);
        assert!(
            entry
                .forward_pair(60, 4, C220LsuPairPart::First, &mut values)
                .is_ok()
        );
    }

    #[test]
    fn ub_write_response_updates_memory_before_releasing_linked_requests() {
        for write_allocate in [false, true] {
            let mut pipeline = C220LsuRequestPipeline::new(4).unwrap();
            let (store, load) = pipeline.admit(0, true).unwrap().unwrap();
            let load = load.unwrap();
            let mut stores = C220LsuStoreBuffer::new(C220LsuStoreConfig {
                line_bytes: 64,
                main_entries: 1,
                sub_entries: 4,
                timeout_ticks: 4,
            })
            .unwrap();
            let mut misses = C220LsuMissBuffer::new(C220LsuMissConfig {
                line_bytes: 64,
                main_entries: 1,
                sub_entries: 4,
            })
            .unwrap();
            let key = C220LsuLineKey {
                address: 0x1000,
                memory: C220LsuMemory::Ub,
            };
            stores.store(key, store, 7, &[0xab, 0xcd], false).unwrap();
            stores.set_state(key, C220LsuStoreState::Fetching).unwrap();
            if write_allocate {
                stores.merge_line(key, &[0x11; 64]).unwrap();
                stores.set_state(key, C220LsuStoreState::Ready).unwrap();
                stores.set_forbidden(key, true).unwrap();
            }
            assert_eq!(
                misses.push(key, load, &mut stores).unwrap(),
                if write_allocate {
                    C220LsuMissReadAction::WaitingForStore
                } else {
                    C220LsuMissReadAction::JoinedRead
                }
            );
            let before = (stores.clone(), misses.clone());
            assert_eq!(
                stores.complete_ub_write(key, write_allocate, &mut [0; 8], &mut misses),
                Err(C220LsuStoreError::InvalidRange)
            );
            assert_eq!((&stores, &misses), (&before.0, &before.1));
            let mut backing = [0x22; 64];
            let completed = stores
                .complete_ub_write(key, write_allocate, &mut backing, &mut misses)
                .unwrap();
            assert_eq!(
                completed.notifications,
                vec![
                    C220LsuCompletion::Store(store),
                    C220LsuCompletion::Load(load)
                ]
            );
            assert_eq!(completed.line, backing);
            assert_eq!(&backing[7..9], &[0xab, 0xcd]);
            assert_eq!(backing[0], if write_allocate { 0x11 } else { 0x22 });
            assert!(stores.entries().is_empty());
            assert!(misses.entries().is_empty());
            assert_eq!(
                stores.complete_ub_write(key, write_allocate, &mut backing, &mut misses),
                Err(C220LsuStoreError::MissingEntry)
            );
        }
    }

    #[test]
    fn coalesced_stores_preserve_bytes_capacity_and_flush_state() {
        let mut pipeline = C220LsuRequestPipeline::new(4).unwrap();
        let mut stores = C220LsuStoreBuffer::new(C220LsuStoreConfig {
            line_bytes: 64,
            main_entries: 1,
            sub_entries: 3,
            timeout_ticks: 3,
        })
        .unwrap();
        let key = C220LsuLineKey {
            address: 0x1000,
            memory: C220LsuMemory::External,
        };
        let other = C220LsuLineKey {
            memory: C220LsuMemory::Ub,
            ..key
        };
        let (first, second) = pipeline.admit(0, true).unwrap().unwrap();
        stores.store(key, first, 4, &[1, 2, 3, 4], false).unwrap();
        stores.advance_timeouts();
        stores
            .store(key, second.unwrap(), 6, &[9, 8], true)
            .unwrap();
        assert!(!stores.full(key));
        assert!(stores.full(other));
        assert_eq!(stores.entry(key).unwrap().remaining_ticks(), 2);
        assert!(!stores.entry(key).unwrap().cache_hit());
        assert_eq!(stores.ready_to_flush().count(), 0);
        let mut data = [0xee; 8];
        assert_eq!(
            stores
                .entry(key)
                .unwrap()
                .forward_scalar(2, &mut data)
                .unwrap(),
            4
        );
        assert_eq!(data, [0, 0, 1, 2, 0xee, 0xee, 0xee, 0xee]);
        let before = stores.clone();
        assert_eq!(
            stores.store(key, first, 63, &[1, 2], false),
            Err(C220LsuStoreError::InvalidRange)
        );
        assert_eq!(stores, before);
        stores.set_forbidden(key, true).unwrap();
        assert_eq!(
            stores.store(key, first, 0, &[1], false),
            Err(C220LsuStoreError::Blocked)
        );
        let third = pipeline.admit(0, false).unwrap().unwrap().0;
        stores.store_atomic(key, third, 10, &[7], true).unwrap();
        assert!(stores.entry(key).unwrap().forbidden());
        assert!(!stores.entry(key).unwrap().cache_hit());
        assert_eq!(stores.entry(key).unwrap().remaining_ticks(), 2);
        let full = stores.clone();
        assert_eq!(
            stores.store_atomic(key, third, 11, &[9], true),
            Err(C220LsuStoreError::Blocked)
        );
        assert_eq!(stores, full);
        stores.set_forbidden(key, false).unwrap();
        assert!(stores.full(key));
        assert_eq!(stores.ready_to_flush().count(), 1);
        stores.set_state(key, C220LsuStoreState::Fetching).unwrap();
        stores.advance_timeouts();
        stores.advance_timeouts();
        assert_eq!(stores.ready_to_flush().count(), 0);
        stores.merge_line(key, &[0x55; 64]).unwrap();
        let entry = stores.entry(key).unwrap();
        assert_eq!(&entry.bytes()[4..8], &[1, 2, 9, 8]);
        assert_eq!(entry.bytes()[10], 7);
        assert_eq!(entry.bytes()[0], 0x55);
        assert!(entry.valid_bytes().iter().all(|valid| *valid));
        assert_eq!(
            stores.remove(key).unwrap().requests(),
            &[first, second.unwrap(), third]
        );
        assert!(!stores.full(other));
    }
}
