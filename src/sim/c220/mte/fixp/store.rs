use std::collections::BTreeSet;
use std::num::NonZeroU32;

use crate::sim::c220::mte::interface::C220MteOutputFragment;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FixpStoreWrite {
    pub token: NonZeroU32,
    pub fragment: C220MteOutputFragment,
}

/// Core-owned FIX output availability, shared with the Cube BIU source.
/// Taking a token acknowledges source availability, not external write completion.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct C220FixpStoreBuffer {
    sequence: u32,
    available: BTreeSet<NonZeroU32>,
}

impl C220FixpStoreBuffer {
    pub fn tokens(&self) -> impl Iterator<Item = NonZeroU32> + '_ {
        self.available.iter().copied()
    }

    pub fn len(&self) -> usize {
        self.available.len()
    }

    pub fn is_empty(&self) -> bool {
        self.available.is_empty()
    }

    /// Credits track published source tokens, not BIU tags or final responses.
    pub fn below_limit(&self, limit: u32) -> bool {
        self.available.len() < limit as usize
    }

    pub(super) fn publish(&mut self, fragment: C220MteOutputFragment) -> C220FixpStoreWrite {
        self.sequence = self.sequence.wrapping_add(1).max(1);
        let token = NonZeroU32::new(self.sequence).expect("nonzero sequence");
        self.available.insert(token);
        C220FixpStoreWrite { token, fragment }
    }

    fn consume(&mut self, token: NonZeroU32) -> bool {
        self.available.remove(&token)
    }
}

/// Cube-source completion probe. The owning BIU egress calls this once per
/// eligible cycle after DBID/ingress; it does not perform a UB memory read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FixpStoreRead {
    token: NonZeroU32,
    bytes: NonZeroU32,
    remaining: u32,
    complete: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220FixpStoreProbe {
    Transferring { remaining: u32 },
    WaitingForToken { token: NonZeroU32 },
    Complete,
}

impl C220FixpStoreRead {
    pub fn new(token: NonZeroU32, bytes: NonZeroU32) -> Self {
        Self {
            token,
            bytes,
            remaining: 0,
            complete: false,
        }
    }

    pub fn remaining(&self) -> u32 {
        self.remaining
    }

    pub fn is_complete(&self) -> bool {
        self.complete
    }

    pub fn probe(&mut self, stores: &mut C220FixpStoreBuffer) -> bool {
        self.probe_progress(stores) == C220FixpStoreProbe::Complete
    }

    pub fn probe_progress(&mut self, stores: &mut C220FixpStoreBuffer) -> C220FixpStoreProbe {
        if self.complete {
            return C220FixpStoreProbe::Complete;
        }
        self.remaining = if self.remaining == 0 {
            (self.bytes.get() - 1) / 128
        } else {
            self.remaining - 1
        };
        if self.remaining != 0 {
            return C220FixpStoreProbe::Transferring {
                remaining: self.remaining,
            };
        }
        self.complete = stores.consume(self.token);
        if self.complete {
            C220FixpStoreProbe::Complete
        } else {
            C220FixpStoreProbe::WaitingForToken { token: self.token }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cube_dbid_queues_feed_shared_data_port_without_early_release() {
        use crate::sim::c220::mte::interface::biu_write::{
            C220BiuWriteSourceRequest,
            cube::{C220BiuCubeWriteSource, C220BiuCubeWriteState},
            data::C220BiuWriteDataPort,
        };
        let mut stores = C220FixpStoreBuffer::default();
        let write = stores.publish(C220MteOutputFragment {
            instruction_id: 11,
            request_id: 9,
            destination_address: 8192,
            bytes: 257,
            last_in_uop: true,
            last_in_instruction: true,
        });
        let tag = NonZeroU32::new(3).unwrap();
        assert!(!stores.below_limit(0));
        assert!(!stores.below_limit(1));
        assert!(stores.below_limit(2));
        let mut source = C220BiuCubeWriteSource::default();
        source
            .register(
                C220BiuWriteSourceRequest {
                    tag,
                    instruction_id: 11,
                    source_address: 0,
                    bytes: 257,
                    gather_stride: None,
                    last_in_instruction: false,
                },
                write.token,
            )
            .unwrap();
        assert_eq!(source.state(tag), Some(C220BiuCubeWriteState::AwaitingDbid));
        assert!(source.release_response(0, tag).is_err());
        assert!(source.egress(0, &mut stores).unwrap().is_none());
        source.receive_dbid(4, tag).unwrap();
        assert!(source.receive_dbid(4, tag).is_err());
        assert!(
            source
                .ingress(4, |_| panic!("not yet eligible"))
                .unwrap()
                .is_none()
        );
        assert_eq!(
            source
                .ingress(5, |started| {
                    assert_eq!(started, tag);
                    true
                })
                .unwrap(),
            Some(tag)
        );
        for tick in 5..8 {
            assert!(source.egress(tick, &mut stores).unwrap().is_none());
            assert!(!stores.below_limit(1));
        }
        let ready = source.egress(8, &mut stores).unwrap().unwrap();
        assert_eq!(ready.ready_tick, 9);
        assert!(ready.request.last_in_instruction);
        assert!(stores.is_empty());
        assert!(!source.is_idle());
        assert!(stores.below_limit(1));
        assert!(!stores.below_limit(0));
        assert!(source.take_data_ready(8).unwrap().is_none());
        let mut port = C220BiuWriteDataPort::default();
        assert!(
            port.send(9, [Some(ready), None, None])
                .unwrap()
                .sent
                .is_some()
        );
        assert_eq!(source.take_data_ready(9).unwrap(), Some(ready));
        assert_eq!(source.state(tag), Some(C220BiuCubeWriteState::Sent));
        assert!(port.receive_response(9, tag).is_err());
        assert!(port.take_request(9).unwrap().is_none());
        assert!(port.take_request(10).unwrap().is_some());
        let response = port.receive_response(15, tag).unwrap();
        assert_eq!(response.retired_instruction(), Some(11));
        source.release_response(15, tag).unwrap();
        assert!(source.is_idle() && port.is_idle());
        assert!(source.release_response(15, tag).is_err());
    }

    #[test]
    fn cube_source_consumes_token_only_after_full_beat_count_and_retries() {
        let mut stores = C220FixpStoreBuffer::default();
        let mut read =
            C220FixpStoreRead::new(NonZeroU32::new(1).unwrap(), NonZeroU32::new(257).unwrap());
        assert_eq!(
            read.probe_progress(&mut stores),
            C220FixpStoreProbe::Transferring { remaining: 2 }
        );
        assert_eq!(read.remaining(), 2);
        assert_eq!(
            read.probe_progress(&mut stores),
            C220FixpStoreProbe::Transferring { remaining: 1 }
        );
        assert_eq!(
            read.probe_progress(&mut stores),
            C220FixpStoreProbe::WaitingForToken {
                token: NonZeroU32::new(1).unwrap()
            }
        );
        let write = stores.publish(C220MteOutputFragment {
            instruction_id: 7,
            request_id: 3,
            destination_address: 4096,
            bytes: 257,
            last_in_uop: true,
            last_in_instruction: true,
        });
        assert_eq!(write.token.get(), 1);
        assert!(!read.probe(&mut stores));
        assert!(!read.probe(&mut stores));
        assert_eq!(stores.len(), 1);
        assert!(read.probe(&mut stores));
        assert!(stores.is_empty());
        assert!(read.probe(&mut stores));
        stores.sequence = u32::MAX;
        assert_eq!(stores.publish(write.fragment).token.get(), 1);
        assert!(
            C220FixpStoreRead::new(write.token, NonZeroU32::new(128).unwrap()).probe(&mut stores)
        );
    }
}
