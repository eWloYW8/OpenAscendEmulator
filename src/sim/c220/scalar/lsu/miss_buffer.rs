use super::C220LsuRequestId;
use super::store_buffer::{C220LsuLineKey, C220LsuStoreBuffer, C220LsuStoreState};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220LsuMissConfig {
    pub line_bytes: usize,
    pub main_entries: usize,
    pub sub_entries: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220LsuMissState {
    Idle,
    Fetching,
    Ready,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220LsuMissReadAction {
    IssueRead,
    JoinedRead,
    WaitingForStore,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum C220LsuMissError {
    #[error("miss-buffer geometry must be nonzero")]
    InvalidConfig,
    #[error("miss-buffer line address is not aligned")]
    UnalignedLine,
    #[error("miss-buffer entry cannot accept a request")]
    Blocked,
    #[error("miss-buffer entry does not exist")]
    MissingEntry,
    #[error("miss-buffer response has the wrong line size")]
    InvalidLineSize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220LsuMissEntry {
    key: C220LsuLineKey,
    state: C220LsuMissState,
    forbidden: bool,
    requests: Vec<C220LsuRequestId>,
    data: Vec<u8>,
}

impl C220LsuMissEntry {
    pub const fn key(&self) -> C220LsuLineKey {
        self.key
    }

    pub const fn state(&self) -> C220LsuMissState {
        self.state
    }

    pub const fn forbidden(&self) -> bool {
        self.forbidden
    }

    pub fn requests(&self) -> &[C220LsuRequestId] {
        &self.requests
    }

    pub fn returned_data(&self) -> Option<&[u8]> {
        (self.state == C220LsuMissState::Ready).then_some(self.data.as_slice())
    }
}

/// Tracks coalesced load misses independently of request-pipeline occupancy.
/// Responses remain present until the controller has delivered completions and
/// serviced any linked stores. Receiving data does not retire an instruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220LsuMissBuffer {
    config: C220LsuMissConfig,
    entries: Vec<C220LsuMissEntry>,
}

impl C220LsuMissBuffer {
    pub const fn line_bytes(&self) -> usize {
        self.config.line_bytes
    }

    pub fn new(config: C220LsuMissConfig) -> Result<Self, C220LsuMissError> {
        if config.line_bytes == 0 || config.main_entries == 0 || config.sub_entries == 0 {
            return Err(C220LsuMissError::InvalidConfig);
        }
        Ok(Self {
            config,
            entries: Vec::new(),
        })
    }

    pub fn entries(&self) -> &[C220LsuMissEntry] {
        &self.entries
    }

    pub fn entry(&self, key: C220LsuLineKey) -> Option<&C220LsuMissEntry> {
        self.entries.iter().find(|entry| entry.key == key)
    }

    pub fn full(&self, key: C220LsuLineKey) -> bool {
        self.entry(key).map_or_else(
            || self.entries.len() >= self.config.main_entries,
            |entry| entry.requests.len() >= self.config.sub_entries,
        )
    }

    pub fn push(
        &mut self,
        key: C220LsuLineKey,
        request: C220LsuRequestId,
        stores: &mut C220LsuStoreBuffer,
    ) -> Result<C220LsuMissReadAction, C220LsuMissError> {
        if !key.address.is_multiple_of(self.config.line_bytes as u64) {
            return Err(C220LsuMissError::UnalignedLine);
        }
        if self.full(key) || self.entry(key).is_some_and(|entry| entry.forbidden) {
            return Err(C220LsuMissError::Blocked);
        }
        let index = match self.entries.iter().position(|entry| entry.key == key) {
            Some(index) => index,
            None => {
                self.entries.push(C220LsuMissEntry {
                    key,
                    state: C220LsuMissState::Idle,
                    forbidden: false,
                    requests: Vec::new(),
                    data: vec![0; self.config.line_bytes],
                });
                self.entries.len() - 1
            }
        };
        let entry = &mut self.entries[index];
        entry.requests.push(request);
        if entry.state != C220LsuMissState::Idle {
            return Ok(C220LsuMissReadAction::JoinedRead);
        }
        match stores.entry(key).map(|entry| entry.state()) {
            Some(C220LsuStoreState::Ready) => Ok(C220LsuMissReadAction::WaitingForStore),
            Some(C220LsuStoreState::Fetching) => {
                entry.state = C220LsuMissState::Fetching;
                Ok(C220LsuMissReadAction::JoinedRead)
            }
            store_state => {
                if store_state.is_some() {
                    stores
                        .set_state(key, C220LsuStoreState::Fetching)
                        .expect("the matching store entry exists");
                }
                entry.state = C220LsuMissState::Fetching;
                Ok(C220LsuMissReadAction::IssueRead)
            }
        }
    }

    pub fn receive_line(
        &mut self,
        key: C220LsuLineKey,
        data: &[u8],
    ) -> Result<(), C220LsuMissError> {
        if data.len() != self.config.line_bytes {
            return Err(C220LsuMissError::InvalidLineSize);
        }
        let entry = self
            .entries
            .iter_mut()
            .find(|entry| entry.key == key)
            .ok_or(C220LsuMissError::MissingEntry)?;
        entry.data.copy_from_slice(data);
        entry.state = C220LsuMissState::Ready;
        entry.forbidden = true;
        Ok(())
    }

    pub fn remove(&mut self, key: C220LsuLineKey) -> Option<C220LsuMissEntry> {
        let index = self.entries.iter().position(|entry| entry.key == key)?;
        Some(self.entries.remove(index))
    }
}

#[cfg(test)]
mod tests {
    use super::super::C220LsuRequestPipeline;
    use super::super::store_buffer::{C220LsuMemory, C220LsuStoreConfig};
    use super::*;

    #[test]
    fn misses_share_store_fetches_and_release_only_after_completion() {
        let mut pipeline = C220LsuRequestPipeline::new(4).unwrap();
        let mut stores = C220LsuStoreBuffer::new(C220LsuStoreConfig {
            line_bytes: 64,
            main_entries: 2,
            sub_entries: 4,
            timeout_ticks: 4,
        })
        .unwrap();
        let mut misses = C220LsuMissBuffer::new(C220LsuMissConfig {
            line_bytes: 64,
            main_entries: 1,
            sub_entries: 3,
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
        let (store, first) = pipeline.admit(0, true).unwrap().unwrap();
        let (second, third) = pipeline.admit(0, true).unwrap().unwrap();
        stores.store(key, store, 0, &[7], false).unwrap();
        assert_eq!(
            misses.push(key, first.unwrap(), &mut stores).unwrap(),
            C220LsuMissReadAction::IssueRead
        );
        assert_eq!(
            stores.entry(key).unwrap().state(),
            C220LsuStoreState::Fetching
        );
        assert!(misses.full(other));
        assert!(!misses.full(key));
        assert_eq!(
            misses.push(key, second, &mut stores).unwrap(),
            C220LsuMissReadAction::JoinedRead
        );
        assert!(misses.entry(key).unwrap().returned_data().is_none());
        let before = misses.clone();
        assert_eq!(
            misses.receive_line(key, &[0; 8]),
            Err(C220LsuMissError::InvalidLineSize)
        );
        assert_eq!(misses, before);
        misses.receive_line(key, &[0x33; 64]).unwrap();
        assert_eq!(
            misses.push(key, third.unwrap(), &mut stores),
            Err(C220LsuMissError::Blocked)
        );
        assert!(misses.full(other));
        let completed = misses.remove(key).unwrap();
        assert_eq!(completed.requests(), &[first.unwrap(), second]);
        assert_eq!(completed.returned_data(), Some([0x33; 64].as_slice()));
        assert!(!misses.full(other));
        assert_eq!(
            misses.push(key, third.unwrap(), &mut stores).unwrap(),
            C220LsuMissReadAction::JoinedRead
        );
        misses.remove(key).unwrap();
        stores.set_state(key, C220LsuStoreState::Ready).unwrap();
        assert_eq!(
            misses.push(key, third.unwrap(), &mut stores).unwrap(),
            C220LsuMissReadAction::WaitingForStore
        );
        assert_eq!(misses.entry(key).unwrap().state(), C220LsuMissState::Idle);
    }
}
