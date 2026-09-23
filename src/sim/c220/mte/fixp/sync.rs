#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220FixpSyncPoint {
    ReadWait,
    ConversionSet,
    WriteWait,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FixpSyncRequest {
    pub tick: u64,
    pub instruction_id: u64,
    pub point: C220FixpSyncPoint,
}

/// Called only when the stage has capacity and its queue head is ready.
/// Implementations may consume individual ready events even when other events
/// still block the command; consumed events must stay consumed on retry.
pub trait C220FixpSync {
    fn blocked(&mut self, request: C220FixpSyncRequest) -> bool;
}

impl<T: C220FixpSync + ?Sized> C220FixpSync for &mut T {
    fn blocked(&mut self, request: C220FixpSyncRequest) -> bool {
        (**self).blocked(request)
    }
}

impl C220FixpSync for bool {
    fn blocked(&mut self, _request: C220FixpSyncRequest) -> bool {
        *self
    }
}
