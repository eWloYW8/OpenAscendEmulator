use super::*;
use crate::sim::c220::mte::{
    generator::{C220MteGeneratorCallback, GeneratorEvents},
    interface::{C220MteL1WriteInterface, biu_read::C220BiuReadFrontend},
};
use crate::sim::common::event::{EventDispatcher, EventId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220Nd2NzCallback {
    Generator(C220MteGeneratorCallback),
    Probe,
    Write,
    Small,
    Lane(u8),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum C220Nd2NzEvent {
    Readiness,
    Generated(bool),
    Read(Option<C220Nd2NzIssuedRead>),
    Write(Option<C220MteOutputFragment>),
    Small(u32),
    Lane { lane: u8, row: Option<u32> },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Nd2NzEvents {
    queues: GeneratorEvents,
    write: EventId,
    small: EventId,
    lanes: [EventId; 4],
}

impl C220Nd2NzEvents {
    pub fn register<T: Copy>(
        events: &mut EventDispatcher<T>,
        clock: EventId,
        tag: impl Fn(C220Nd2NzCallback) -> T,
    ) -> Self {
        let queues = GeneratorEvents::register(events, clock, |callback| {
            tag(C220Nd2NzCallback::Generator(callback))
        });
        let probe = events.add_process(tag(C220Nd2NzCallback::Probe), false);
        events.subscribe(clock, probe);
        let mut bind = |callback| {
            let event = events.add_event();
            let process = events.add_process(tag(callback), false);
            events.subscribe(event, process);
            event
        };
        let write = bind(C220Nd2NzCallback::Write);
        let small = bind(C220Nd2NzCallback::Small);
        let lanes = std::array::from_fn(|lane| bind(C220Nd2NzCallback::Lane(lane as u8)));
        Self {
            queues,
            write,
            small,
            lanes,
        }
    }

    pub fn arm<T: Copy>(&self, events: &mut EventDispatcher<T>) {
        self.queues.arm_instruction(events);
    }

    pub fn handle<T: Copy>(
        &self,
        callback: C220Nd2NzCallback,
        events: &mut EventDispatcher<T>,
        engine: &mut C220Nd2NzEngine,
        biu: &mut C220BiuReadFrontend,
        l1: &mut C220MteL1WriteInterface,
        hardware_sync_blocked: bool,
    ) -> Result<C220Nd2NzEvent, C220Nd2NzEngineError> {
        let tick = events.tick();
        Ok(match callback {
            C220Nd2NzCallback::Generator(phase) => match phase {
                C220MteGeneratorCallback::InstructionReady => {
                    self.queues
                        .probe_instruction(events, engine.source.as_ref().map(|s| s.ready_tick));
                    C220Nd2NzEvent::Readiness
                }
                C220MteGeneratorCallback::GeneratedReady => {
                    self.queues
                        .probe_generated(events, engine.generated.front().map(|s| s.ready_tick));
                    C220Nd2NzEvent::Readiness
                }
                C220MteGeneratorCallback::Generate => {
                    let generated = engine.generate(tick)?;
                    if generated {
                        self.queues.arm_generated(events);
                    }
                    C220Nd2NzEvent::Generated(generated)
                }
                C220MteGeneratorCallback::Send => {
                    C220Nd2NzEvent::Read(engine.send_biu(tick, biu, hardware_sync_blocked)?)
                }
            },
            C220Nd2NzCallback::Probe => {
                if engine
                    .commands
                    .front()
                    .is_some_and(|c| c.ready_tick <= tick)
                {
                    events.notify_at(self.write, tick);
                }
                if engine
                    .staging
                    .small_ready_tick()
                    .is_some_and(|ready| ready <= tick)
                {
                    events.notify_at(self.small, tick);
                }
                for lane in 0..4 {
                    if engine
                        .staging
                        .lane_ready_tick(lane)
                        .is_some_and(|ready| ready <= tick)
                    {
                        events.notify_at(self.lanes[lane], tick);
                    }
                }
                C220Nd2NzEvent::Readiness
            }
            C220Nd2NzCallback::Write => C220Nd2NzEvent::Write(engine.send_l1(tick, l1)?),
            C220Nd2NzCallback::Small => C220Nd2NzEvent::Small(engine.stage_small(tick)?),
            C220Nd2NzCallback::Lane(lane) => C220Nd2NzEvent::Lane {
                lane,
                row: engine.stage_lane(tick, u32::from(lane))?,
            },
        })
    }
}
