use std::collections::{BTreeSet, VecDeque};

use thiserror::Error;

use crate::sim::c220::cube::{C220CubeL0cAccess, C220CubeL0cRequest};
use crate::sim::c220::memory::buffer::{C220LocalBuffer, C220LocalBufferError};

pub const C220_L0C_FRAGMENT_BYTES: u32 = 512;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220L0cMaster {
    Mte,
    Cube,
}

impl C220L0cMaster {
    const fn index(self) -> usize {
        match self {
            Self::Mte => 0,
            Self::Cube => 1,
        }
    }

    const fn other(self) -> Self {
        match self {
            Self::Mte => Self::Cube,
            Self::Cube => Self::Mte,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220L0cFragmentRequest {
    pub address: u64,
    pub bytes: u32,
    pub access: C220CubeL0cAccess,
    pub check_unit_flags: bool,
    pub update_unit_flags: bool,
}

impl From<C220CubeL0cRequest> for C220L0cFragmentRequest {
    fn from(request: C220CubeL0cRequest) -> Self {
        Self {
            address: request.address,
            bytes: u32::from(request.bytes),
            access: request.access,
            check_unit_flags: false,
            update_unit_flags: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220L0cUnitFlagBlock {
    WriterOccupied { fragment: u32 },
    ReaderUnavailable { fragment: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum C220L0cError {
    #[error("L0C capacity must contain at least one 512-byte fragment")]
    InvalidCapacity,
    #[error("L0C unit-flag ready tick overflowed")]
    TimeOverflow,
    #[error(transparent)]
    Buffer(#[from] C220LocalBufferError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PendingReadFlags {
    ready_tick: u64,
    start_fragment: u64,
    fragment_count: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220L0cScoreboard {
    fragment_count: u32,
    unit_flag_read_latency: u32,
    writer_flags: BTreeSet<u32>,
    reader_flags: BTreeSet<u32>,
    pending_reader_flags: VecDeque<PendingReadFlags>,
}

impl C220L0cScoreboard {
    pub fn new(total_bytes: u64, unit_flag_read_latency: u32) -> Result<Self, C220L0cError> {
        let fragment_count = total_bytes / u64::from(C220_L0C_FRAGMENT_BYTES);
        let fragment_count =
            u32::try_from(fragment_count).map_err(|_| C220L0cError::InvalidCapacity)?;
        if fragment_count == 0 {
            return Err(C220L0cError::InvalidCapacity);
        }
        Ok(Self {
            fragment_count,
            unit_flag_read_latency,
            writer_flags: BTreeSet::new(),
            reader_flags: BTreeSet::new(),
            pending_reader_flags: VecDeque::new(),
        })
    }

    pub const fn fragment_count(&self) -> u32 {
        self.fragment_count
    }

    pub const fn unit_flag_read_latency(&self) -> u32 {
        self.unit_flag_read_latency
    }

    pub fn writer_flag_count(&self) -> usize {
        self.writer_flags.len()
    }

    pub fn reader_flag_count(&self) -> usize {
        self.reader_flags.len()
    }

    pub fn advance_to(&mut self, tick: u64) {
        while self
            .pending_reader_flags
            .front()
            .is_some_and(|pending| pending.ready_tick <= tick)
        {
            let pending = self
                .pending_reader_flags
                .pop_front()
                .expect("front pending flag exists");
            let fragments = self
                .fragments(pending.start_fragment, pending.fragment_count)
                .collect::<Vec<_>>();
            for fragment in fragments {
                self.reader_flags.insert(fragment);
            }
        }
    }

    pub fn blocked_by(&self, request: C220L0cFragmentRequest) -> Option<C220L0cUnitFlagBlock> {
        if !request.check_unit_flags {
            return None;
        }
        let (start, count) = self.request_fragments(request);
        match request.access {
            C220CubeL0cAccess::Write => self
                .fragments(start, count)
                .find(|fragment| self.writer_flags.contains(fragment))
                .map(|fragment| C220L0cUnitFlagBlock::WriterOccupied { fragment }),
            C220CubeL0cAccess::Read => self
                .fragments(start, count)
                .find(|fragment| !self.reader_flags.contains(fragment))
                .map(|fragment| C220L0cUnitFlagBlock::ReaderUnavailable { fragment }),
        }
    }

    pub fn admit(
        &mut self,
        request: C220L0cFragmentRequest,
        tick: u64,
    ) -> Result<Result<(), C220L0cUnitFlagBlock>, C220L0cError> {
        if let Some(blocked) = self.blocked_by(request) {
            return Ok(Err(blocked));
        }
        if !request.check_unit_flags || !request.update_unit_flags {
            return Ok(Ok(()));
        }
        let (start_fragment, fragment_count) = self.request_fragments(request);
        match request.access {
            C220CubeL0cAccess::Write => {
                let fragments = self
                    .fragments(start_fragment, fragment_count)
                    .collect::<Vec<_>>();
                for fragment in fragments {
                    self.writer_flags.insert(fragment);
                }
                let ready_tick = tick
                    .checked_add(u64::from(self.unit_flag_read_latency))
                    .ok_or(C220L0cError::TimeOverflow)?;
                self.pending_reader_flags.push_back(PendingReadFlags {
                    ready_tick,
                    start_fragment,
                    fragment_count,
                });
            }
            C220CubeL0cAccess::Read => {
                let fragments = self
                    .fragments(start_fragment, fragment_count)
                    .collect::<Vec<_>>();
                for fragment in fragments {
                    self.reader_flags.remove(&fragment);
                    self.writer_flags.remove(&fragment);
                }
            }
        }
        Ok(Ok(()))
    }

    fn request_fragments(&self, request: C220L0cFragmentRequest) -> (u64, u32) {
        (
            request.address / u64::from(C220_L0C_FRAGMENT_BYTES),
            request.bytes / C220_L0C_FRAGMENT_BYTES,
        )
    }

    fn fragments(&self, start: u64, count: u32) -> impl Iterator<Item = u32> + '_ {
        (0..count).map(move |offset| {
            ((start + u64::from(offset)) % u64::from(self.fragment_count)) as u32
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct C220L0cWriteArbiter {
    busy: [u32; 2],
    waiting: [VecDeque<u64>; 2],
}

impl C220L0cWriteArbiter {
    pub fn enqueue(&mut self, master: C220L0cMaster, tick: u64) {
        self.waiting[master.index()].push_back(tick);
    }

    pub fn queued(&self, master: C220L0cMaster) -> usize {
        self.waiting[master.index()].len()
    }

    pub const fn busy(&self, master: C220L0cMaster) -> u32 {
        self.busy[master.index()]
    }

    pub fn can_grant(&self, master: C220L0cMaster) -> bool {
        if self.busy[master.other().index()] != 0 {
            return false;
        }
        let own = self.waiting[master.index()].front().copied();
        let other = self.waiting[master.other().index()].front().copied();
        match (master, own, other) {
            (_, _, None) => true,
            (C220L0cMaster::Cube, Some(own), Some(other)) => own <= other,
            (C220L0cMaster::Cube, None, Some(_)) => false,
            (C220L0cMaster::Mte, Some(own), Some(other)) => own < other,
            (C220L0cMaster::Mte, None, Some(_)) => true,
        }
    }

    pub fn grant(&mut self, master: C220L0cMaster) -> bool {
        if !self.can_grant(master) {
            return false;
        }
        self.waiting[master.index()].pop_front();
        self.busy[master.index()] = self.busy[master.index()].saturating_add(1);
        true
    }

    pub fn complete(&mut self, master: C220L0cMaster) {
        self.busy[master.index()] = self.busy[master.index()].saturating_sub(1);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220L0c {
    buffer: C220LocalBuffer,
    scoreboard: C220L0cScoreboard,
    write_arbiter: C220L0cWriteArbiter,
}

impl C220L0c {
    pub fn new(total_bytes: u64, unit_flag_read_latency: u32) -> Result<Self, C220L0cError> {
        Ok(Self {
            buffer: C220LocalBuffer::new(total_bytes),
            scoreboard: C220L0cScoreboard::new(total_bytes, unit_flag_read_latency)?,
            write_arbiter: C220L0cWriteArbiter::default(),
        })
    }

    pub const fn buffer(&self) -> &C220LocalBuffer {
        &self.buffer
    }

    pub fn buffer_mut(&mut self) -> &mut C220LocalBuffer {
        &mut self.buffer
    }

    pub const fn scoreboard(&self) -> &C220L0cScoreboard {
        &self.scoreboard
    }

    pub fn scoreboard_mut(&mut self) -> &mut C220L0cScoreboard {
        &mut self.scoreboard
    }

    pub const fn write_arbiter(&self) -> &C220L0cWriteArbiter {
        &self.write_arbiter
    }

    pub fn write_arbiter_mut(&mut self) -> &mut C220L0cWriteArbiter {
        &mut self.write_arbiter
    }
}
