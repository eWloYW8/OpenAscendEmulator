mod arbiter;
mod service;
mod transport;

pub use arbiter::{C220L1Access, C220L1Arbiter, C220L1Decision, C220L1Geometry};
pub use service::C220L1Pipeline;
pub use transport::{
    C220_L1_TRANSPORT_CAPACITY, C220_L1_TRANSPORT_TICKS, C220L1Transit, C220L1Transport,
    C220L1TransportError,
};

use thiserror::Error;

pub const C220_L1_READ_RESPONSE_TICKS: u64 = 8;
pub const C220_L1_WRITE_RESPONSE_TICKS: u64 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum C220L1Port {
    FixpWrite = 0,
    MteWrite = 1,
    MteRead = 2,
}

impl C220L1Port {
    pub(super) const ALL: [Self; 3] = [Self::FixpWrite, Self::MteWrite, Self::MteRead];

    pub const fn response_ticks(self) -> u64 {
        match self {
            Self::FixpWrite | Self::MteWrite => C220_L1_WRITE_RESPONSE_TICKS,
            Self::MteRead => C220_L1_READ_RESPONSE_TICKS,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220L1Request {
    pub id: u64,
    pub access: C220L1Access,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220L1Response {
    pub request: C220L1Request,
    pub accepted_tick: u64,
    pub ready_tick: u64,
    pub bank_mask: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220L1Cycle {
    pub tick: u64,
    pub decisions: [Option<C220L1Decision>; 3],
    /// A granted request is consumed only when its receiver was notified.
    pub accepted: [bool; 3],
    pub responses: [Option<C220L1Response>; 3],
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum C220L1Error {
    #[error("L1 geometry exceeds the supported bank-mask layout")]
    UnsupportedGeometry,
    #[error("L1 cycle {requested} must follow the last cycle {previous}")]
    NonIncreasingTick { previous: u64, requested: u64 },
    #[error("L1 response tick overflowed")]
    TimeOverflow,
}

#[cfg(test)]
mod tests;
