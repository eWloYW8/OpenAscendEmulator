use std::collections::VecDeque;

use super::{C220L0WritePort, C220MteOutputFragment, C220MteOutputPlan};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220MteL1OutputDestination {
    /// Read-only timing sink; index responses do not generate output fragments.
    SparseIndex,
    /// Whole-transaction acknowledgment to the external-output engine.
    External,
    Bt,
    Fb,
    Smask,
    L0a(C220L0WritePort),
    L0b(C220L0WritePort),
}

impl C220MteL1OutputDestination {
    fn local_retirement_delay(self) -> Option<u64> {
        match self {
            Self::Bt | Self::Smask => Some(5),
            Self::Fb => Some(0),
            Self::SparseIndex | Self::External | Self::L0a(_) | Self::L0b(_) => None,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct C220MteL1OutputCredits {
    pub external: bool,
    pub l0a: [bool; 3],
    pub l0b: [bool; 3],
}

impl C220MteL1OutputCredits {
    fn permits(self, destination: C220MteL1OutputDestination) -> bool {
        match destination {
            C220MteL1OutputDestination::SparseIndex => false,
            C220MteL1OutputDestination::External => self.external,
            C220MteL1OutputDestination::Bt
            | C220MteL1OutputDestination::Fb
            | C220MteL1OutputDestination::Smask => true,
            C220MteL1OutputDestination::L0a(port) => self.l0a[port as usize],
            C220MteL1OutputDestination::L0b(port) => self.l0b[port as usize],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MteL1OutputTransfer<T> {
    pub destination: C220MteL1OutputDestination,
    pub fragment: C220MteOutputFragment,
    pub payload: T,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct C220MteL1OutputQueues {
    pub acknowledged: usize,
    pub output_fragments: usize,
    pub awaiting_retirement: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220MteL1OutputCycle<T> {
    pub tick: u64,
    pub sent: Option<C220MteL1OutputTransfer<T>>,
    pub blocked: Option<C220MteL1OutputDestination>,
    /// Local completion. L0 destinations retire through their write interface.
    pub retired: Option<C220MteL1OutputTransfer<T>>,
    pub queues: C220MteL1OutputQueues,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220MteL1OutputSend<T> {
    pub tick: u64,
    pub sent: Option<C220MteL1OutputTransfer<T>>,
    pub blocked: Option<C220MteL1OutputDestination>,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum C220MteL1OutputError {
    #[error("L1 output time reversed from {previous} to {requested}")]
    TimeReversed { previous: u64, requested: u64 },
    #[error("L1 output {phase} callback already ran at tick {tick}")]
    RepeatedCallback { phase: &'static str, tick: u64 },
    #[error("L1 output time overflowed")]
    TimeOverflow,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Acknowledgment<T> {
    ready_tick: u64,
    destination: C220MteL1OutputDestination,
    fragments: C220MteOutputPlan,
    expanded: bool,
    payload: T,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Retirement<T> {
    ready_tick: u64,
    transfer: C220MteL1OutputTransfer<T>,
}

/// One shared L1 output interface, not one lane per destination. Completing
/// responses enter a FIFO with one tick of visibility delay. Each cycle can
/// forward only one fragment; a blocked target holds up all later responses.
/// The payload preserves the caller's logical request without imposing a
/// particular instruction representation on the interface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220MteL1Output<T> {
    acknowledged: VecDeque<Acknowledgment<T>>,
    retiring: VecDeque<Retirement<T>>,
    observed_tick: Option<u64>,
    send_tick: Option<u64>,
    retire_tick: Option<u64>,
}

impl<T> Default for C220MteL1Output<T> {
    fn default() -> Self {
        Self {
            acknowledged: VecDeque::new(),
            retiring: VecDeque::new(),
            observed_tick: None,
            send_tick: None,
            retire_tick: None,
        }
    }
}

impl<T: Copy> C220MteL1Output<T> {
    pub fn is_idle(&self) -> bool {
        self.acknowledged.is_empty() && self.retiring.is_empty()
    }

    pub fn queue_state(&self) -> C220MteL1OutputQueues {
        C220MteL1OutputQueues {
            acknowledged: self.acknowledged.len(),
            output_fragments: self.acknowledged.front().map_or(0, |head| {
                if head.expanded && head.destination != C220MteL1OutputDestination::External {
                    head.fragments.len()
                } else {
                    0
                }
            }),
            awaiting_retirement: self.retiring.len(),
        }
    }

    /// Accept only completing logical responses. Non-completing physical reads
    /// are discarded by the request owner. Receive may precede or follow step
    /// in the same cycle; neither ordering permits same-cycle output.
    pub fn receive(
        &mut self,
        tick: u64,
        destination: C220MteL1OutputDestination,
        fragments: C220MteOutputPlan,
        payload: T,
    ) -> Result<(), C220MteL1OutputError> {
        self.validate_receive(tick)?;
        self.acknowledged.push_back(Acknowledgment {
            ready_tick: tick + 1,
            destination,
            fragments,
            expanded: false,
            payload,
        });
        self.observed_tick = Some(tick);
        Ok(())
    }

    pub(in crate::sim::c220::mte) fn validate_receive(
        &self,
        tick: u64,
    ) -> Result<(), C220MteL1OutputError> {
        self.check_time(tick)?;
        tick.checked_add(1)
            .ok_or(C220MteL1OutputError::TimeOverflow)?;
        Ok(())
    }

    /// Validate before a composing unit mutates its other queues.
    pub(in crate::sim::c220::mte) fn validate_step(
        &self,
        tick: u64,
        credits: C220MteL1OutputCredits,
    ) -> Result<(), C220MteL1OutputError> {
        self.validate_retire(tick)?;
        self.validate_send(tick, credits)
    }

    pub(in crate::sim::c220::mte) fn validate_retire(
        &self,
        tick: u64,
    ) -> Result<(), C220MteL1OutputError> {
        self.check_callback(tick, self.retire_tick, "retire")
    }

    pub(in crate::sim::c220::mte) fn validate_send(
        &self,
        tick: u64,
        credits: C220MteL1OutputCredits,
    ) -> Result<(), C220MteL1OutputError> {
        self.check_callback(tick, self.send_tick, "send")?;
        if let Some(head) = self.ready_head(tick)
            && let Some(delay) = head.destination.local_retirement_delay()
            && credits.permits(head.destination)
            && head
                .fragments
                .front()
                .is_some_and(|fragment| fragment.last_in_uop)
        {
            tick.checked_add(delay)
                .ok_or(C220MteL1OutputError::TimeOverflow)?;
        }
        Ok(())
    }

    /// Credits refer to the actual downstream input queues. The caller must
    /// enqueue a returned L0 fragment into the selected target and port.
    /// Empty active output plans remain pending instead of inventing retirement.
    pub fn step(
        &mut self,
        tick: u64,
        credits: C220MteL1OutputCredits,
    ) -> Result<C220MteL1OutputCycle<T>, C220MteL1OutputError> {
        self.validate_step(tick, credits)?;
        let retired = self.retire(tick)?;
        let sent = self.send(tick, credits)?;
        Ok(C220MteL1OutputCycle {
            tick,
            sent: sent.sent,
            blocked: sent.blocked,
            retired,
            queues: self.queue_state(),
        })
    }

    pub fn acknowledgment_ready_tick(&self) -> Option<u64> {
        self.acknowledged.front().map(|entry| entry.ready_tick)
    }

    /// Only the shared acknowledgment head may be offered to the external
    /// receiver. A blocked external response also holds up local outputs.
    pub fn external_head(&self, tick: u64) -> Option<T> {
        self.ready_head(tick)
            .filter(|head| head.destination == C220MteL1OutputDestination::External)
            .map(|head| head.payload)
    }

    pub fn retirement_ready_tick(&self) -> Option<u64> {
        self.retiring.front().map(|entry| entry.ready_tick)
    }

    pub fn retire(
        &mut self,
        tick: u64,
    ) -> Result<Option<C220MteL1OutputTransfer<T>>, C220MteL1OutputError> {
        self.validate_retire(tick)?;
        let retired = self
            .retiring
            .pop_front_if(|entry| entry.ready_tick <= tick)
            .map(|entry| entry.transfer);
        self.observed_tick = Some(tick);
        self.retire_tick = Some(tick);
        Ok(retired)
    }

    pub fn send(
        &mut self,
        tick: u64,
        credits: C220MteL1OutputCredits,
    ) -> Result<C220MteL1OutputSend<T>, C220MteL1OutputError> {
        self.validate_send(tick, credits)?;
        let mut sent = None;
        let mut blocked = None;
        if let Some(head) = self
            .acknowledged
            .front_mut()
            .filter(|head| head.ready_tick <= tick)
        {
            head.expanded = true;
            if head.fragments.front().is_some() {
                if credits.permits(head.destination) {
                    let fragment = head.fragments.next().expect("nonempty output");
                    let transfer = C220MteL1OutputTransfer {
                        destination: head.destination,
                        fragment,
                        payload: head.payload,
                    };
                    sent = Some(transfer);
                    if fragment.last_in_uop {
                        if let Some(delay) = head.destination.local_retirement_delay() {
                            self.retiring.push_back(Retirement {
                                ready_tick: tick + delay,
                                transfer,
                            });
                        }
                        self.acknowledged.pop_front();
                    }
                } else {
                    blocked = Some(head.destination);
                }
            }
        }
        self.observed_tick = Some(tick);
        self.send_tick = Some(tick);
        Ok(C220MteL1OutputSend {
            tick,
            sent,
            blocked,
        })
    }

    fn ready_head(&self, tick: u64) -> Option<&Acknowledgment<T>> {
        self.acknowledged
            .front()
            .filter(|head| head.ready_tick <= tick)
    }

    fn check_time(&self, tick: u64) -> Result<(), C220MteL1OutputError> {
        if let Some(previous) = self.observed_tick
            && tick < previous
        {
            return Err(C220MteL1OutputError::TimeReversed {
                previous,
                requested: tick,
            });
        }
        Ok(())
    }

    fn check_callback(
        &self,
        tick: u64,
        previous: Option<u64>,
        phase: &'static str,
    ) -> Result<(), C220MteL1OutputError> {
        self.check_time(tick)?;
        if previous == Some(tick) {
            return Err(C220MteL1OutputError::RepeatedCallback { phase, tick });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
