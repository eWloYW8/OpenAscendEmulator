use super::*;
use crate::isa::c220::mte::set2d::C220Set2dInstruction;
use crate::sim::c220::memory::l1::{C220L1Geometry, C220L1Port, C220L1Transport};
use crate::sim::c220::mte::C220MteGeneratorCallback;
use crate::sim::c220::mte::interface::{
    C220L0WriteCallback, C220L0WriteEventOutcome, C220L0WriteEvents,
};
use crate::sim::c220::mte::set2d::{C220Set2dEventOutcome, C220Set2dEvents};
use crate::sim::common::event::EventDispatcher;
use std::num::NonZeroU32;

fn bandwidths() -> C220Set2dBandwidths {
    C220Set2dBandwidths {
        l0a: NonZeroU32::new(64).unwrap(),
        l0b: NonZeroU32::new(64).unwrap(),
        l1: NonZeroU32::new(8).unwrap(),
    }
}

fn fill(destination: u32, repeats: u64) -> C220Set2dFill {
    let instruction =
        C220Set2dInstruction::decode((3 << 29) | (1 << 22) | (1 << 7) | destination).unwrap();
    let mut registers = [0; 32];
    registers[1] = repeats | (1 << 16) | (1 << 32);
    instruction.capture(&registers, 0x1234)
}

#[derive(Clone, Copy)]
enum Callback {
    L0a(C220L0WriteCallback),
    L0b(C220L0WriteCallback),
    Set2d(C220MteGeneratorCallback),
}

#[test]
fn three_routes_preserve_uops_through_distinct_gates_and_shared_outputs() {
    for destination in 0..3 {
        let fill = fill(destination, 2);
        let expected = C220Set2dUops::new(fill, bandwidths()).collect::<Vec<_>>();
        let mut frontend = C220Set2dFrontend::new(bandwidths());
        let mut events = EventDispatcher::new(0);
        let clock = events.add_event();
        let l0a_events = C220L0WriteEvents::register(&mut events, clock, Callback::L0a);
        let l0b_events = C220L0WriteEvents::register(&mut events, clock, Callback::L0b);
        let binding = C220Set2dEvents::register(&mut events, clock, Callback::Set2d);
        let mut l0a = C220L0WritePipeline::default();
        let mut l0b = C220L0WritePipeline::default();
        let mut l1 = C220MteL1WriteInterface::default();
        let mut memory = C220L1Transport::new(C220L1Geometry::new(32, 4, 1, 0).unwrap());
        let issue = binding.issue(&mut events, &mut frontend, 42, fill).unwrap();
        assert!(!issue.completion_ready);
        assert_eq!(issue.uop_count, expected.len() as u64);
        let mut sent = Vec::new();
        let mut completions = Vec::new();
        let mut full = false;
        let mut output_blocked = false;
        for tick in 0..150 {
            events.advance_to(tick).unwrap();
            events.notify_at(clock, tick);
            // Repeated notifications must not double-run either frontend phase.
            events.notify_at(clock, tick);
            let gates = C220Set2dGates {
                hardware_sync_blocked: destination == 2 || tick < 10,
                l1_prefetch_blocked: destination != 2 || tick < 10,
            };
            let mut callbacks = Vec::new();
            while let Some(invocation) = events.next_callback() {
                let callback = match invocation.callback {
                    Callback::Set2d(callback) => callback,
                    Callback::L0a(callback) | Callback::L0b(callback) => {
                        let (binding, pipeline) = if matches!(invocation.callback, Callback::L0a(_))
                        {
                            (&l0a_events, &mut l0a)
                        } else {
                            (&l0b_events, &mut l0b)
                        };
                        if let C220L0WriteEventOutcome::Acknowledged(Some(ack)) =
                            binding.handle(callback, &mut events, pipeline).unwrap()
                            && let Some(id) = ack.retired_instruction()
                        {
                            completions.push((tick, id));
                        }
                        continue;
                    }
                };
                callbacks.push(callback);
                let outcome = binding
                    .handle(
                        callback,
                        &mut events,
                        &mut frontend,
                        gates,
                        if destination == 2 {
                            C220Set2dOutputs::L1(&mut l1)
                        } else {
                            C220Set2dOutputs::L0 {
                                l0a: &mut l0a,
                                l0b: &mut l0b,
                            }
                        },
                    )
                    .unwrap();
                match outcome {
                    C220Set2dEventOutcome::Sent(cycle) => {
                        output_blocked |= cycle.stall == Some(C220Set2dStall::OutputFull);
                        if tick == 4 {
                            assert_eq!(
                                cycle.stall,
                                Some(if destination == 2 {
                                    C220Set2dStall::Prefetch
                                } else {
                                    C220Set2dStall::HardwareFlag
                                })
                            );
                        }
                        if let Some(entry) = cycle.sent {
                            if sent.is_empty() {
                                assert_eq!(tick, 10);
                            }
                            assert_eq!(entry.uop_index, sent.len() as u64);
                            assert_eq!(entry.instruction_id, 42);
                            sent.push(entry.uop);
                        }
                    }
                    C220Set2dEventOutcome::Generated(Some(entry)) if tick == 1 => {
                        assert_eq!(entry.ready_tick, 4);
                    }
                    _ => {}
                }
                full |= frontend.queue_state().generated == GENERATED_CAPACITY;
            }
            if tick == 10 {
                assert_eq!(
                    callbacks,
                    [
                        C220MteGeneratorCallback::GeneratedReady,
                        C220MteGeneratorCallback::InstructionReady,
                        C220MteGeneratorCallback::Send,
                        C220MteGeneratorCallback::Generate,
                    ]
                );
            }
            // The L1 transport remains an explicit downstream phase in this fixture.
            if let Some(ack) = l1.retire(tick).unwrap()
                && let Some(id) = ack.retired_instruction()
            {
                completions.push((tick, id));
            }
            if let Some(response) = memory.receive_response(tick, C220L1Port::MteWrite).unwrap() {
                l1.receive_response(tick, response.request.id).unwrap();
            }
            if let Some(request) = l1
                .send(
                    tick,
                    tick >= 25 && memory.request_ready(C220L1Port::MteWrite),
                )
                .unwrap()
                .sent
            {
                assert!(
                    memory
                        .send_request(tick, C220L1Port::MteWrite, request.l1_request())
                        .unwrap()
                );
            }
            memory.advance(tick).unwrap();
            if frontend.is_idle() && l0a.is_idle() && l0b.is_idle() && l1.is_idle() {
                break;
            }
        }
        assert!(full);
        assert_eq!(sent, expected);
        assert_eq!(completions.len(), 1);
        assert_eq!(completions[0].1, 42);
        assert_eq!(output_blocked, destination == 2);
        assert!(
            frontend.is_idle()
                && l0a.is_idle()
                && l0b.is_idle()
                && l1.is_idle()
                && memory.is_idle()
        );
    }
}

#[test]
fn admission_and_callback_order_do_not_confuse_generation_with_retirement() {
    let bandwidth = NonZeroU32::new(512).unwrap();
    let mut frontend = C220Set2dFrontend::new(C220Set2dBandwidths {
        l0a: bandwidth,
        l0b: bandwidth,
        l1: bandwidth,
    });
    assert!(frontend.issue(0, 0, fill(0, 0)).unwrap().completion_ready);
    assert!(frontend.is_idle());
    frontend.issue(0, 1, fill(0, 1)).unwrap();
    assert!(frontend.generate(0).unwrap().is_none());
    assert_eq!(
        frontend.issue(0, 2, fill(1, 1)),
        Err(C220Set2dFrontendError::CommandBusy)
    );
    frontend.generate(1).unwrap();
    assert!(frontend.can_issue() && !frontend.is_idle());
    // A new command may follow generation, before the previous output drains.
    frontend.issue(1, 2, fill(1, 1)).unwrap();
    frontend.generate(2).unwrap();
    assert_eq!(
        frontend
            .generated()
            .iter()
            .map(|entry| entry.instruction_id)
            .collect::<Vec<_>>(),
        [1, 2]
    );
    let before = frontend.clone();
    assert!(matches!(
        frontend.generate(2),
        Err(C220Set2dFrontendError::RepeatedCallback { .. })
    ));
    assert_eq!(frontend, before);
    assert!(matches!(
        frontend.generate(1),
        Err(C220Set2dFrontendError::TimeReversed { .. })
    ));
    assert_eq!(frontend, before);
    assert_eq!(
        frontend.issue(u64::MAX, 3, fill(0, 1)),
        Err(C220Set2dFrontendError::TimeOverflow)
    );
    assert_eq!(frontend, before);
}
