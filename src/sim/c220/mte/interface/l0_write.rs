use super::C220MteOutputFragment;
use std::collections::VecDeque;

mod events;
pub use events::{C220L0WriteCallback, C220L0WriteEventOutcome, C220L0WriteEvents};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum C220L0WritePort {
    Port0 = 0,
    Port1 = 1,
    Port2 = 2,
}

impl C220L0WritePort {
    pub const fn input_ticks(self) -> u64 {
        match self {
            Self::Port0 => 2,
            Self::Port1 => 4,
            Self::Port2 => 10,
        }
    }

    pub const fn capacity(self) -> usize {
        self.input_ticks() as usize + 1
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220L0WriteEntry {
    pub ready_tick: u64,
    pub fragment: C220MteOutputFragment,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220L0WriteCycle {
    pub tick: u64,
    pub selected_port: Option<C220L0WritePort>,
    pub sent: Option<C220MteOutputFragment>,
    pub acknowledged: Option<C220MteOutputFragment>,
    pub retired_instruction: Option<u64>,
    pub queue_lengths: [usize; 3],
    pub pending_acknowledgments: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220L0WriteSend {
    pub tick: u64,
    pub selected_port: Option<C220L0WritePort>,
    pub sent: Option<C220MteOutputFragment>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220L0WriteAcknowledgment {
    pub tick: u64,
    pub fragment: C220MteOutputFragment,
}

impl C220L0WriteAcknowledgment {
    pub const fn retired_instruction(self) -> Option<u64> {
        if self.fragment.last_in_instruction {
            Some(self.fragment.instruction_id)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum C220L0WriteError {
    #[error("L0 write time reversed from {previous} to {requested}")]
    TimeReversed { previous: u64, requested: u64 },
    #[error("L0 write {phase} callback already ran at tick {tick}")]
    RepeatedCallback { phase: &'static str, tick: u64 },
    #[error("L0 write time overflowed")]
    TimeOverflow,
}

/// One L0A or L0B write interface. Instantiate separately for the two targets.
/// Ports 0 and 1 share round-robin priority; port 2 runs only when neither is
/// ready. Sending confirms locally after one tick, without a memory response.
/// Sending and retirement are independent callbacks, each once per tick.
/// The event owner decides their order relative to producers and consumers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220L0WritePipeline {
    inputs: [VecDeque<C220L0WriteEntry>; 3],
    acknowledgments: VecDeque<C220L0WriteEntry>,
    last_primary: C220L0WritePort,
    observed_tick: Option<u64>,
    send_tick: Option<u64>,
    retire_tick: Option<u64>,
}

impl Default for C220L0WritePipeline {
    fn default() -> Self {
        Self {
            inputs: std::array::from_fn(|_| VecDeque::new()),
            acknowledgments: VecDeque::new(),
            last_primary: C220L0WritePort::Port1,
            observed_tick: None,
            send_tick: None,
            retire_tick: None,
        }
    }
}

impl C220L0WritePipeline {
    pub fn is_idle(&self) -> bool {
        self.inputs.iter().all(VecDeque::is_empty) && self.acknowledgments.is_empty()
    }

    pub fn queue(&self, port: C220L0WritePort) -> &VecDeque<C220L0WriteEntry> {
        &self.inputs[port as usize]
    }

    pub fn acknowledgments(&self) -> &VecDeque<C220L0WriteEntry> {
        &self.acknowledgments
    }

    pub fn can_push(&self, port: C220L0WritePort) -> bool {
        self.queue(port).len() < port.capacity()
    }

    /// Returns false without consuming the fragment when its input is full.
    pub fn push(
        &mut self,
        tick: u64,
        port: C220L0WritePort,
        fragment: C220MteOutputFragment,
    ) -> Result<bool, C220L0WriteError> {
        self.check_time(tick)?;
        if !self.can_push(port) {
            self.observed_tick = Some(tick);
            return Ok(false);
        }
        let ready_tick = tick
            .checked_add(port.input_ticks())
            .ok_or(C220L0WriteError::TimeOverflow)?;
        self.inputs[port as usize].push_back(C220L0WriteEntry {
            ready_tick,
            fragment,
        });
        self.observed_tick = Some(tick);
        Ok(true)
    }

    pub fn selected_port(&self, tick: u64) -> Option<C220L0WritePort> {
        use C220L0WritePort::{Port0, Port1, Port2};
        let order = if self.last_primary == Port1 {
            [Port0, Port1, Port2]
        } else {
            [Port1, Port0, Port2]
        };
        order.into_iter().find(|port| {
            self.queue(*port)
                .front()
                .is_some_and(|head| head.ready_tick <= tick)
        })
    }

    pub fn send(&mut self, tick: u64) -> Result<C220L0WriteSend, C220L0WriteError> {
        self.check_callback(tick, self.send_tick, "send")?;
        let selected_port = self.selected_port(tick);
        let ready_tick = if selected_port.is_some() {
            tick.checked_add(1).ok_or(C220L0WriteError::TimeOverflow)?
        } else {
            tick
        };
        let sent = selected_port.map(|port| {
            if port != C220L0WritePort::Port2 {
                self.last_primary = port;
            }
            let fragment = self.inputs[port as usize]
                .pop_front()
                .expect("selected queue head")
                .fragment;
            self.acknowledgments.push_back(C220L0WriteEntry {
                ready_tick,
                fragment,
            });
            fragment
        });
        self.send_tick = Some(tick);
        self.observed_tick = Some(tick);
        Ok(C220L0WriteSend {
            tick,
            selected_port,
            sent,
        })
    }

    pub fn retire(
        &mut self,
        tick: u64,
    ) -> Result<Option<C220L0WriteAcknowledgment>, C220L0WriteError> {
        self.check_callback(tick, self.retire_tick, "retire")?;
        let acknowledgment = self
            .acknowledgments
            .pop_front_if(|head| head.ready_tick <= tick)
            .map(|head| C220L0WriteAcknowledgment {
                tick,
                fragment: head.fragment,
            });
        self.retire_tick = Some(tick);
        self.observed_tick = Some(tick);
        Ok(acknowledgment)
    }

    /// Convenience for owners that explicitly choose retire-before-send.
    /// Event-driven owners call the two phases separately when notified.
    pub fn step(&mut self, tick: u64) -> Result<C220L0WriteCycle, C220L0WriteError> {
        self.check_callback(tick, self.send_tick, "send")?;
        self.check_callback(tick, self.retire_tick, "retire")?;
        if self.selected_port(tick).is_some() && tick == u64::MAX {
            return Err(C220L0WriteError::TimeOverflow);
        }
        let acknowledgment = self.retire(tick)?;
        let send = self.send(tick)?;
        Ok(C220L0WriteCycle {
            tick,
            selected_port: send.selected_port,
            sent: send.sent,
            acknowledged: acknowledgment.map(|ack| ack.fragment),
            retired_instruction: acknowledgment.and_then(|ack| ack.retired_instruction()),
            queue_lengths: std::array::from_fn(|index| self.inputs[index].len()),
            pending_acknowledgments: self.acknowledgments.len(),
        })
    }

    fn check_time(&self, tick: u64) -> Result<(), C220L0WriteError> {
        if let Some(previous) = self.observed_tick
            && tick < previous
        {
            return Err(C220L0WriteError::TimeReversed {
                previous,
                requested: tick,
            });
        }
        Ok(())
    }

    fn check_callback(
        &self,
        tick: u64,
        last: Option<u64>,
        phase: &'static str,
    ) -> Result<(), C220L0WriteError> {
        self.check_time(tick)?;
        if last == Some(tick) {
            return Err(C220L0WriteError::RepeatedCallback { phase, tick });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
