use super::{C220Core, C220CoreError};

/// Ordinary single-task termination. Early task handoff is a separate protocol.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum C220Termination {
    #[default]
    Running,
    Draining {
        end_tick: u64,
        next_check_tick: u64,
    },
    Complete {
        end_tick: u64,
        completion_tick: u64,
    },
}

impl C220Termination {
    pub fn is_complete(self) -> bool {
        matches!(self, Self::Complete { .. })
    }

    pub(super) fn next_tick(self) -> Option<u64> {
        match self {
            Self::Draining {
                next_check_tick, ..
            } => Some(next_check_tick),
            _ => None,
        }
    }
}

impl C220Core {
    pub fn termination(&self) -> C220Termination {
        self.termination
    }

    pub(super) fn begin_termination(&mut self, tick: u64) -> Result<(), C220CoreError> {
        self.termination = C220Termination::Draining {
            end_tick: tick,
            next_check_tick: tick.checked_add(1).ok_or(C220CoreError::TimeOverflow)?,
        };
        Ok(())
    }

    pub(super) fn advance_termination(&mut self, tick: u64) -> Result<(), C220CoreError> {
        let C220Termination::Draining {
            end_tick,
            next_check_tick,
        } = self.termination
        else {
            return Ok(());
        };
        if tick < next_check_tick {
            return Ok(());
        }
        self.termination = if self.activity().is_idle() {
            C220Termination::Complete {
                end_tick,
                completion_tick: tick,
            }
        } else {
            C220Termination::Draining {
                end_tick,
                next_check_tick: tick.checked_add(100).ok_or(C220CoreError::TimeOverflow)?,
            }
        };
        Ok(())
    }
}
