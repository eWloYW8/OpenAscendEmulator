use super::{C220BiuReadOutput, C220BiuReadReturns, C220BiuReturnError, C220BiuRobBeat, Tag};
use crate::sim::c220::mte::interface::biu_read::C220BiuSubcore;
use crate::sim::c220::mte::interface::biu_read::write::C220BiuWriteSend;
use crate::sim::common::event::{EventDispatcher, EventError, EventId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220BiuReturnCallback {
    Probe,
    Ingress(usize),
    Select,
    Read,
    Egress(C220BiuSubcore),
    Send(C220BiuSubcore),
    Nd2NzPush,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum C220BiuReturnEvent {
    Readiness,
    Ingress { port: usize, tag: Option<Tag> },
    Selected([Option<Tag>; 3]),
    Read(Vec<C220BiuRobBeat>),
    Egress(Option<C220BiuReadOutput>),
    Send(C220BiuWriteSend),
    Nd2Nz(super::C220BiuNd2NzProgress),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220BiuReturnEvents {
    ingress: [EventId; 2],
    selection: EventId,
    read: EventId,
    egress: [EventId; 3],
    send: [EventId; 3],
    nd2nz: EventId,
}

impl C220BiuReturnEvents {
    pub fn register<T: Copy>(
        events: &mut EventDispatcher<T>,
        clock: EventId,
        tag: impl Fn(C220BiuReturnCallback) -> T,
    ) -> Self {
        let probe = events.add_process(tag(C220BiuReturnCallback::Probe), false);
        events.subscribe(clock, probe);
        let ingress = std::array::from_fn(|port| {
            let event = events.add_event();
            let process = events.add_process(tag(C220BiuReturnCallback::Ingress(port)), false);
            events.subscribe(event, process);
            event
        });
        let selection = events.add_event();
        let process = events.add_process(tag(C220BiuReturnCallback::Select), false);
        events.subscribe(selection, process);
        let read = events.add_event();
        let process = events.add_process(tag(C220BiuReturnCallback::Read), false);
        events.subscribe(read, process);
        let egress = C220BiuSubcore::ALL.map(|core| {
            let event = events.add_event();
            let process = events.add_process(tag(C220BiuReturnCallback::Egress(core)), false);
            events.subscribe(event, process);
            event
        });
        let send = C220BiuSubcore::ALL.map(|core| {
            let event = events.add_event();
            let process = events.add_process(tag(C220BiuReturnCallback::Send(core)), false);
            events.subscribe(event, process);
            event
        });
        let nd2nz = events.add_event();
        let process = events.add_process(tag(C220BiuReturnCallback::Nd2NzPush), false);
        events.subscribe(nd2nz, process);
        Self {
            ingress,
            selection,
            read,
            egress,
            send,
            nd2nz,
        }
    }

    pub fn handle<T: Copy>(
        &self,
        callback: C220BiuReturnCallback,
        events: &mut EventDispatcher<T>,
        returns: &mut C220BiuReadReturns,
        destination_ready: [bool; 3],
    ) -> Result<C220BiuReturnEvent, C220BiuReturnError> {
        let tick = events.tick();
        match callback {
            C220BiuReturnCallback::Nd2NzPush => Err(C220BiuReturnError::DedicatedEgressRequired),
            C220BiuReturnCallback::Probe => {
                if returns
                    .nd2nz_rows
                    .iter()
                    .flatten()
                    .any(|head| head.ready_tick <= tick)
                {
                    events.notify_at(self.nd2nz, tick);
                }
                for (port, queue) in returns.ingress.iter().enumerate() {
                    if queue.front().is_some_and(|head| head.ready_tick <= tick) {
                        events.notify_at(self.ingress[port], tick);
                    }
                }
                if returns
                    .order
                    .iter()
                    .flatten()
                    .any(|queue| queue.front().is_some_and(|head| head.ready_tick <= tick))
                {
                    events.notify_at(self.selection, tick);
                }
                for (core, queue) in returns.egress.iter().enumerate() {
                    if queue.front().is_some_and(|head| head.ready_tick <= tick) {
                        events.notify_at(self.egress[core], tick);
                    }
                }
                for (core, queue) in returns.adapters.iter().enumerate() {
                    if queue.front().is_some_and(|head| head.ready_tick <= tick) {
                        events.notify_at(self.send[core], tick);
                    }
                }
                Ok(C220BiuReturnEvent::Readiness)
            }
            C220BiuReturnCallback::Ingress(port) => Ok(C220BiuReturnEvent::Ingress {
                port,
                tag: returns.ingress(tick, port)?,
            }),
            C220BiuReturnCallback::Select => {
                let selected = returns.select(tick)?;
                if selected.iter().any(Option::is_some) {
                    self.schedule_read(events)?;
                }
                Ok(C220BiuReturnEvent::Selected(selected))
            }
            C220BiuReturnCallback::Read => {
                let read = returns.read(tick)?;
                if returns.has_active_tags() {
                    self.schedule_read(events)?;
                }
                Ok(C220BiuReturnEvent::Read(read))
            }
            C220BiuReturnCallback::Egress(core) => {
                returns.egress(tick, core).map(C220BiuReturnEvent::Egress)
            }
            C220BiuReturnCallback::Send(core) => returns
                .send_output(tick, core, destination_ready[core as usize])
                .map(C220BiuReturnEvent::Send),
        }
    }

    fn schedule_read<T: Copy>(
        &self,
        events: &mut EventDispatcher<T>,
    ) -> Result<(), C220BiuReturnError> {
        events
            .notify_after(self.read, 1)
            .map_err(|error| match error {
                EventError::TimeOverflow => C220BiuReturnError::TimeOverflow,
                _ => unreachable!("scheduling a future event cannot reverse time"),
            })
    }
}
