use std::collections::{BTreeMap, VecDeque};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EventId(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProcessId(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProcessState {
    pub enabled: bool,
    pub repeat_in_tick: bool,
    pub last_run: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Invocation<T> {
    pub tick: u64,
    pub event: EventId,
    pub process: ProcessId,
    pub callback: T,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EventError {
    #[error("event time reversed from {previous} to {requested}")]
    TimeReversed { previous: u64, requested: u64 },
    #[error("events at tick {tick} must drain before advancing time")]
    PendingEvents { tick: u64 },
    #[error("event time overflowed")]
    TimeOverflow,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Event {
    enabled: bool,
    triggered_at: Option<u64>,
    subscribers: Vec<ProcessId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Process<T> {
    state: ProcessState,
    callback: T,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Delivery {
    event: EventId,
    next_subscriber: usize,
    subscriber_count: usize,
}

/// Serial event evaluation with FIFO notifications and ordered subscribers.
/// Repeated notifications are retained; suppression belongs to each process.
/// Callbacks execute outside the dispatcher and may append same-tick events.
/// Drain `next_callback` before advancing time or committing deferred state.
/// Handles belong to the dispatcher that created them. Topology is configured
/// before execution; no subscriber removal or recursive evaluation is needed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EventDispatcher<T> {
    tick: u64,
    events: Vec<Event>,
    processes: Vec<Process<T>>,
    pending: BTreeMap<u64, VecDeque<EventId>>,
    delivery: Option<Delivery>,
}

impl<T: Copy> EventDispatcher<T> {
    pub fn new(tick: u64) -> Self {
        Self {
            tick,
            events: Vec::new(),
            processes: Vec::new(),
            pending: BTreeMap::new(),
            delivery: None,
        }
    }

    pub fn tick(&self) -> u64 {
        self.tick
    }

    pub fn add_event(&mut self) -> EventId {
        let id = EventId(self.events.len());
        self.events.push(Event {
            enabled: true,
            triggered_at: None,
            subscribers: Vec::new(),
        });
        id
    }

    pub fn add_process(&mut self, callback: T, repeat_in_tick: bool) -> ProcessId {
        let id = ProcessId(self.processes.len());
        self.processes.push(Process {
            state: ProcessState {
                enabled: true,
                repeat_in_tick,
                last_run: None,
            },
            callback,
        });
        id
    }

    pub fn subscribe(&mut self, event: EventId, process: ProcessId) {
        assert!(process.0 < self.processes.len(), "unknown process");
        let subscribers = &mut self.events[event.0].subscribers;
        if !subscribers.contains(&process) {
            subscribers.push(process);
        }
    }

    pub fn process_state(&self, process: ProcessId) -> ProcessState {
        self.processes[process.0].state
    }

    pub fn set_process_enabled(&mut self, process: ProcessId, enabled: bool) {
        self.processes[process.0].state.enabled = enabled;
    }

    pub fn set_event_enabled(&mut self, event: EventId, enabled: bool) {
        self.events[event.0].enabled = enabled;
    }

    pub fn triggered(&self, event: EventId) -> bool {
        self.events[event.0].triggered_at == Some(self.tick)
    }

    /// Notifications into the past are ignored. Notifying a disabled event
    /// still queues it; its enabled state is sampled when delivery begins.
    pub fn notify_at(&mut self, event: EventId, tick: u64) -> bool {
        assert!(event.0 < self.events.len(), "unknown event");
        if tick < self.tick {
            return false;
        }
        self.pending.entry(tick).or_default().push_back(event);
        true
    }

    pub fn notify_after(&mut self, event: EventId, delay: u64) -> Result<(), EventError> {
        let tick = self
            .tick
            .checked_add(delay)
            .ok_or(EventError::TimeOverflow)?;
        self.notify_at(event, tick);
        Ok(())
    }

    /// Remove all queued occurrences, preserving the order of other events.
    /// A delivery already in progress finishes its subscribers.
    pub fn cancel(&mut self, event: EventId) -> usize {
        let mut removed = 0;
        self.pending.retain(|_, queue| {
            let before = queue.len();
            queue.retain(|candidate| *candidate != event);
            removed += before - queue.len();
            !queue.is_empty()
        });
        removed
    }

    pub fn pending_notifications(&self) -> impl Iterator<Item = (u64, EventId)> + '_ {
        self.pending
            .iter()
            .flat_map(|(&tick, events)| events.iter().map(move |&event| (tick, event)))
    }

    pub fn next_event_tick(&self) -> Option<u64> {
        self.delivery
            .map(|_| self.tick)
            .or_else(|| self.pending.first_key_value().map(|(&tick, _)| tick))
    }

    /// Advance across idle ticks only. Owners with clocked processes must
    /// enqueue their clock events; they cannot skip those evaluations.
    pub fn advance_to(&mut self, tick: u64) -> Result<(), EventError> {
        self.check_advance_to(tick)?;
        self.tick = tick;
        Ok(())
    }

    pub fn check_advance_to(&self, tick: u64) -> Result<(), EventError> {
        if tick < self.tick {
            return Err(EventError::TimeReversed {
                previous: self.tick,
                requested: tick,
            });
        }
        if let Some(pending) = self.next_event_tick()
            && pending < tick
        {
            return Err(EventError::PendingEvents { tick: pending });
        }
        Ok(())
    }

    /// Execute the returned callback before asking for another one. A normal
    /// process is dispatched at most once at this tick, across all its events.
    pub fn next_callback(&mut self) -> Option<Invocation<T>> {
        loop {
            if let Some(delivery) = &mut self.delivery {
                if delivery.next_subscriber < delivery.subscriber_count {
                    let id = self.events[delivery.event.0].subscribers[delivery.next_subscriber];
                    delivery.next_subscriber += 1;
                    let process = &mut self.processes[id.0];
                    if !process.state.enabled
                        || (!process.state.repeat_in_tick
                            && process.state.last_run == Some(self.tick))
                    {
                        continue;
                    }
                    process.state.last_run = Some(self.tick);
                    return Some(Invocation {
                        tick: self.tick,
                        event: delivery.event,
                        process: id,
                        callback: process.callback,
                    });
                }
                self.delivery = None;
            }
            let queue = self.pending.get_mut(&self.tick)?;
            let id = queue.pop_front().expect("nonempty event queue");
            if queue.is_empty() {
                self.pending.remove(&self.tick);
            }
            let event = &mut self.events[id.0];
            if event.enabled {
                event.triggered_at = Some(self.tick);
                self.delivery = Some(Delivery {
                    event: id,
                    next_subscriber: 0,
                    subscriber_count: event.subscribers.len(),
                });
            }
        }
    }
}

#[cfg(test)]
mod tests;
