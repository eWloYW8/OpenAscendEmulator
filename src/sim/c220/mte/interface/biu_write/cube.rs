use std::collections::{BTreeMap, VecDeque};
use std::num::NonZeroU32;

use super::{C220BiuWriteDataReady, C220BiuWriteSourceRequest};
use crate::sim::c220::mte::fixp::{C220FixpStoreBuffer, C220FixpStoreProbe, C220FixpStoreRead};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220BiuCubeWriteProbe {
    pub tick: u64,
    pub request: C220BiuWriteSourceRequest,
    pub progress: C220FixpStoreProbe,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220BiuCubeWriteState {
    AwaitingDbid,
    Ingress { ready_tick: u64 },
    Egress { ready_tick: u64, remaining: u32 },
    Ready { ready_tick: u64 },
    Sent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Pending {
    request: C220BiuWriteSourceRequest,
    read: C220FixpStoreRead,
    state: C220BiuCubeWriteState,
}

#[derive(Debug, thiserror::Error)]
pub enum C220BiuCubeWriteError {
    #[error("Cube BIU tag {0} is already active")]
    DuplicateTag(NonZeroU32),
    #[error("Cube BIU tag {0} is not awaiting DBID")]
    UnexpectedDbid(NonZeroU32),
    #[error("Cube BIU tag {0} has no sent data awaiting completion")]
    UnexpectedCompletion(NonZeroU32),
    #[error("Cube BIU source requires positive bytes and no UB gather stride")]
    InvalidRequest,
    #[error("Cube BIU source time reversed from {previous} to {requested}")]
    TimeReversed { previous: u64, requested: u64 },
    #[error("Cube BIU source {phase} callback repeated at tick {tick}")]
    RepeatedCallback { phase: &'static str, tick: u64 },
    #[error("Cube BIU source time overflow")]
    TimeOverflow,
}

/// DBID-gated Cube source path. Queue order follows returned DBIDs, while
/// instruction-tail resolution belongs to the shared command issuer.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct C220BiuCubeWriteSource {
    pending: BTreeMap<NonZeroU32, Pending>,
    ingress: VecDeque<NonZeroU32>,
    egress: VecDeque<NonZeroU32>,
    ready: VecDeque<C220BiuWriteDataReady>,
    observed: Option<u64>,
    callbacks: [Option<u64>; 2],
    last_probe: Option<C220BiuCubeWriteProbe>,
}

impl C220BiuCubeWriteSource {
    pub fn last_probe(&self) -> Option<C220BiuCubeWriteProbe> {
        self.last_probe
    }
    pub fn is_idle(&self) -> bool {
        self.pending.is_empty()
    }

    pub fn state(&self, tag: NonZeroU32) -> Option<C220BiuCubeWriteState> {
        self.pending.get(&tag).map(|p| p.state)
    }

    pub fn data_ready(&self) -> &VecDeque<C220BiuWriteDataReady> {
        &self.ready
    }

    pub fn register(
        &mut self,
        request: C220BiuWriteSourceRequest,
        token: NonZeroU32,
    ) -> Result<(), C220BiuCubeWriteError> {
        if self.pending.contains_key(&request.tag) {
            return Err(C220BiuCubeWriteError::DuplicateTag(request.tag));
        }
        let bytes = NonZeroU32::new(request.bytes)
            .filter(|_| request.gather_stride.is_none())
            .ok_or(C220BiuCubeWriteError::InvalidRequest)?;
        self.pending.insert(
            request.tag,
            Pending {
                request,
                read: C220FixpStoreRead::new(token, bytes),
                state: C220BiuCubeWriteState::AwaitingDbid,
            },
        );
        Ok(())
    }

    pub fn receive_dbid(
        &mut self,
        tick: u64,
        tag: NonZeroU32,
    ) -> Result<(), C220BiuCubeWriteError> {
        self.observe(tick)?;
        let ready_tick = tick
            .checked_add(1)
            .ok_or(C220BiuCubeWriteError::TimeOverflow)?;
        let pending = self
            .pending
            .get_mut(&tag)
            .filter(|p| p.state == C220BiuCubeWriteState::AwaitingDbid)
            .ok_or(C220BiuCubeWriteError::UnexpectedDbid(tag))?;
        pending.state = C220BiuCubeWriteState::Ingress { ready_tick };
        self.ingress.push_back(tag);
        Ok(())
    }

    pub fn ingress(
        &mut self,
        tick: u64,
        resolve_tail: impl FnOnce(NonZeroU32) -> bool,
    ) -> Result<Option<NonZeroU32>, C220BiuCubeWriteError> {
        self.begin(tick, 0, "ingress")?;
        let Some(&tag) = self.ingress.front() else {
            return Ok(None);
        };
        let pending = self.pending.get_mut(&tag).expect("registered ingress");
        let C220BiuCubeWriteState::Ingress { ready_tick } = pending.state else {
            unreachable!("ingress state")
        };
        if ready_tick > tick {
            return Ok(None);
        }
        let ready_tick = tick
            .checked_add(1)
            .ok_or(C220BiuCubeWriteError::TimeOverflow)?;
        pending.request.last_in_instruction = resolve_tail(tag);
        pending.state = C220BiuCubeWriteState::Egress {
            ready_tick,
            remaining: 0,
        };
        self.ingress.pop_front();
        self.egress.push_back(tag);
        Ok(Some(tag))
    }

    pub fn egress(
        &mut self,
        tick: u64,
        stores: &mut C220FixpStoreBuffer,
    ) -> Result<Option<C220BiuWriteDataReady>, C220BiuCubeWriteError> {
        self.begin(tick, 1, "egress")?;
        self.last_probe = None;
        let Some(&tag) = self.egress.front() else {
            return Ok(None);
        };
        let pending = self.pending.get_mut(&tag).expect("registered egress");
        let C220BiuCubeWriteState::Egress { ready_tick, .. } = pending.state else {
            unreachable!("egress state")
        };
        if ready_tick > tick {
            return Ok(None);
        }
        let next_tick = tick
            .checked_add(1)
            .ok_or(C220BiuCubeWriteError::TimeOverflow)?;
        let progress = pending.read.probe_progress(stores);
        self.last_probe = Some(C220BiuCubeWriteProbe {
            tick,
            request: pending.request,
            progress,
        });
        if progress != C220FixpStoreProbe::Complete {
            pending.state = C220BiuCubeWriteState::Egress {
                ready_tick,
                remaining: pending.read.remaining(),
            };
            return Ok(None);
        }
        let data = C220BiuWriteDataReady {
            ready_tick: next_tick,
            request: pending.request,
        };
        pending.state = C220BiuCubeWriteState::Ready {
            ready_tick: next_tick,
        };
        self.egress.pop_front();
        self.ready.push_back(data);
        Ok(Some(data))
    }

    /// Call only after the shared write-data port accepts this source's head.
    pub fn take_data_ready(
        &mut self,
        tick: u64,
    ) -> Result<Option<C220BiuWriteDataReady>, C220BiuCubeWriteError> {
        self.observe(tick)?;
        let data = self.ready.pop_front_if(|head| head.ready_tick <= tick);
        if let Some(data) = data {
            self.pending
                .get_mut(&data.request.tag)
                .expect("registered data")
                .state = C220BiuCubeWriteState::Sent;
        }
        Ok(data)
    }

    /// Release source bookkeeping only after the data port validates the final
    /// response. BIU tag ownership remains with the command issuer.
    pub fn release_response(
        &mut self,
        tick: u64,
        tag: NonZeroU32,
    ) -> Result<C220BiuWriteSourceRequest, C220BiuCubeWriteError> {
        self.observe(tick)?;
        if !self
            .pending
            .get(&tag)
            .is_some_and(|p| p.state == C220BiuCubeWriteState::Sent)
        {
            return Err(C220BiuCubeWriteError::UnexpectedCompletion(tag));
        }
        Ok(self
            .pending
            .remove(&tag)
            .expect("validated completion")
            .request)
    }

    fn observe(&mut self, tick: u64) -> Result<(), C220BiuCubeWriteError> {
        if let Some(previous) = self.observed
            && tick < previous
        {
            return Err(C220BiuCubeWriteError::TimeReversed {
                previous,
                requested: tick,
            });
        }
        self.observed = Some(tick);
        Ok(())
    }

    fn begin(
        &mut self,
        tick: u64,
        index: usize,
        phase: &'static str,
    ) -> Result<(), C220BiuCubeWriteError> {
        self.observe(tick)?;
        if self.callbacks[index] == Some(tick) {
            return Err(C220BiuCubeWriteError::RepeatedCallback { phase, tick });
        }
        self.callbacks[index] = Some(tick);
        Ok(())
    }
}
