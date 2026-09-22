pub(crate) const PARTIAL_WRITE_TICKS: u64 = 6;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum C220UbBlockProgress {
    #[default]
    Waiting,
    ReadModifyWriteDelay {
        first_grant: u64,
    },
    AwaitingWriteGrant {
        first_grant: u64,
    },
    Complete {
        first_grant: u64,
        final_grant: u64,
    },
}

impl C220UbBlockProgress {
    /// Whether this block needs a bank grant, and whether it is the second pass.
    pub const fn pending_grant(self) -> Option<bool> {
        match self {
            Self::Waiting => Some(false),
            Self::AwaitingWriteGrant { .. } => Some(true),
            _ => None,
        }
    }

    pub const fn completion_tick(self) -> Option<u64> {
        match self {
            Self::Complete { final_grant, .. } => Some(final_grant),
            _ => None,
        }
    }

    pub(crate) fn grant(&mut self, tick: u64, partial_write: bool) {
        *self = match *self {
            Self::Waiting if partial_write => Self::ReadModifyWriteDelay { first_grant: tick },
            Self::Waiting => Self::Complete {
                first_grant: tick,
                final_grant: tick,
            },
            Self::AwaitingWriteGrant { first_grant } => Self::Complete {
                first_grant,
                final_grant: tick,
            },
            _ => return,
        };
    }

    /// Runs after arbitration. Expiring a delay does not grant the second pass.
    pub(crate) fn finish_tick(&mut self, tick: u64) {
        if let Self::ReadModifyWriteDelay { first_grant } = *self
            && tick.saturating_sub(first_grant) >= PARTIAL_WRITE_TICKS
        {
            *self = Self::AwaitingWriteGrant { first_grant };
        }
    }
}
