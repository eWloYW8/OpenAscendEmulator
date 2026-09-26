use super::*;
use crate::isa::c220::mte::bias::C220MovL1ToBtInstruction;
use crate::isa::c220::mte::load2d::C220Load2dInstruction;
use crate::sim::c220::memory::l1::{
    C220L1Callback, C220L1Events, C220L1Geometry, C220L1Port, C220L1Request, C220L1Transport,
};
use crate::sim::c220::mte::C220MteGeneratorCallback;
use crate::sim::c220::mte::interface::{
    C220L0WriteCallback, C220L0WriteEventOutcome, C220L0WriteEvents, C220L0WritePipeline,
    C220MteL1Callback, C220MteL1CycleInputs, C220MteL1EventOutcome, C220MteL1Events,
    C220MteL1OutputCredits,
};
use crate::sim::c220::mte::mte1::bias::c220_bt_uops;
use crate::sim::common::event::EventDispatcher;
use std::collections::{BTreeMap, BTreeSet};

fn nz(value: u32) -> NonZeroU32 {
    NonZeroU32::new(value).unwrap()
}

fn bandwidths(bt: u32) -> C220Mte1ReadBandwidths {
    C220Mte1ReadBandwidths {
        bt: nz(bt),
        l0a: nz(256),
        l0b: nz(128),
    }
}

fn bt_transfer(convert: bool) -> C220BtTransfer {
    let word = (3 << 29) | (2 << 27) | (4 << 23) | (1 << 17) | (2 << 12) | (3 << 7) | (5 << 3);
    let mut registers = [0; 32];
    registers[1] = 1024;
    registers[2] = 31;
    registers[3] = (2 << 4) | (2 << 16) | (1 << 32) | (1 << 48) | if convert { 8 } else { 0 };
    C220MovL1ToBtInstruction::decode(word)
        .unwrap()
        .capture(&registers)
}

#[test]
fn expanded_output_stalls_reads_until_the_following_cycle() {
    let mut frontend = C220Mte1ReadFrontend::new(C220Mte1ReadKind::Bt, nz(32), bandwidths(32));
    let mut interface = C220MteL1Interface::default();
    frontend
        .issue(0, 7, C220Mte1ReadTransfer::Bt(bt_transfer(true)))
        .unwrap();
    let mut response = None;
    let mut gated_cycles = 0;
    let mut output_bytes = 0;
    for tick in 0..100 {
        let before = interface.queue_state();
        let cycle = interface
            .step(
                tick,
                C220MteL1CycleInputs {
                    request_ready: true,
                    response: response.take(),
                    ..Default::default()
                },
            )
            .unwrap();
        frontend.step(tick, false, &mut interface).unwrap();
        if before.output.output_fragments != 0 && before.inputs[0] != 0 {
            gated_cycles += 1;
            assert!(cycle.sent.is_none() && cycle.decision.selected.is_none());
        }
        response = cycle.sent.map(|request| request.id);
        if let Some(output) = cycle.output.sent {
            output_bytes += u64::from(output.fragment.bytes);
        }
        if frontend.is_idle() && interface.is_idle() {
            break;
        }
    }
    assert!(gated_cycles > 0);
    assert!(frontend.is_idle() && interface.is_idle());
    assert_eq!(output_bytes, bt_transfer(true).output_bytes());
}

#[test]
fn bt_frontend_preserves_physical_requests_backpressure_and_retirement() {
    for convert in [false, true] {
        let transfer = bt_transfer(convert);
        let plan = C220BtRequestPlan::new(transfer, nz(96), nz(32));
        assert_eq!(plan.remaining_requests(), 8);
        let requests = plan.collect::<Vec<_>>();
        assert_eq!(requests.iter().map(|r| r.input_bytes).sum::<u32>(), 256);
        assert_eq!(
            requests.iter().filter(|r| r.completes_logical_uop).count(),
            4
        );
        assert_eq!(requests.iter().filter(|r| r.last_in_instruction).count(), 1);
        assert!(requests.last().unwrap().last_in_instruction);
        assert_eq!(
            requests[..3]
                .iter()
                .map(|r| r.source_address)
                .collect::<Vec<_>>(),
            [31, 63, 95]
        );
        assert_eq!(requests[0].logical, requests[2].logical);
        assert_eq!(requests[3].logical.destination_address, 1120);
        let mut frontend = C220Mte1ReadFrontend::new(C220Mte1ReadKind::Bt, nz(32), bandwidths(96));
        let mut interface = C220MteL1Interface::default();
        let issue = frontend
            .issue(0, 7, C220Mte1ReadTransfer::Bt(transfer))
            .unwrap();
        assert!(!issue.completion_ready && issue.request_count == 8);
        let interface_before = interface.clone();
        assert!(matches!(
            interface.step(
                0,
                C220MteL1CycleInputs {
                    response: Some(999),
                    ..Default::default()
                }
            ),
            Err(C220MteL1Error::UnknownResponse(999))
        ));
        assert_eq!(interface, interface_before);
        assert_eq!(
            frontend.issue(0, 8, C220Mte1ReadTransfer::Bt(transfer)),
            Err(C220Mte1ReadFrontendError::CommandBusy)
        );

        let mut l1 = C220L1Transport::new(C220L1Geometry::new(32, 4, 1, 0).unwrap());
        let mut physical = Vec::new();
        let mut output = Vec::new();
        let mut tails = BTreeMap::new();
        let mut retired = 0;
        let mut blocked_read = false;
        let mut write_injected = false;
        let mut generated_full = false;
        let mut completed = false;
        for tick in 0..200 {
            let response = l1
                .receive_response(tick, C220L1Port::MteRead)
                .unwrap()
                .map(|r| r.request.id);
            l1.receive_response(tick, C220L1Port::FixpWrite).unwrap();
            let cycle = interface
                .step(
                    tick,
                    C220MteL1CycleInputs {
                        request_ready: l1.request_ready(C220L1Port::MteRead),
                        response,
                        ..Default::default()
                    },
                )
                .unwrap();
            let generated = frontend.step(tick, tick < 13, &mut interface).unwrap();
            if tick == 0 {
                let before = frontend.clone();
                assert!(matches!(
                    frontend.step(tick, false, &mut interface),
                    Err(C220Mte1ReadFrontendError::RepeatedCallback { .. })
                ));
                assert_eq!(frontend, before);
            }
            assert!(generated.queues.generated <= 4);
            assert!(interface.queue_state().inputs[0] <= 5);
            generated_full |= generated.queues.generated == 4;
            if let Some(request) = cycle.sent {
                if physical.is_empty() {
                    assert_eq!(tick, 17);
                }
                let C220Mte1ReadUop::Bt(uop) = request.operation.payload else {
                    panic!("BT request")
                };
                physical.push(uop);
                assert!(
                    l1.send_request(tick, C220L1Port::MteRead, request.l1_request())
                        .unwrap()
                );
                if !write_injected {
                    assert!(
                        l1.send_request(
                            tick,
                            C220L1Port::FixpWrite,
                            C220L1Request {
                                id: 999,
                                access: request.operation.access
                            }
                        )
                        .unwrap()
                    );
                    write_injected = true;
                }
            }
            if let Some(chunk) = cycle.output.sent {
                assert!(chunk.payload.operation.completes_logical_uop);
                output.push((chunk.fragment.destination_address, chunk.fragment.bytes));
                if chunk.fragment.last_in_uop {
                    tails.insert(chunk.fragment.request_id, tick);
                }
            }
            if let Some(event) = cycle.output.retired {
                assert_eq!(tick, tails[&event.fragment.request_id] + 5);
                retired += 1;
                if event.fragment.last_in_instruction {
                    assert_eq!(event.fragment.instruction_id, 7);
                    assert!(!completed);
                    completed = true;
                }
            }
            let cycle = l1.advance(tick).unwrap();
            if let Some(decision) = cycle.decisions[2]
                && !decision.granted
            {
                assert!(cycle.decisions[0].is_some());
                blocked_read = true;
            }
            if frontend.is_idle() && interface.is_idle() {
                break;
            }
        }
        assert!(
            frontend.is_idle()
                && interface.is_idle()
                && completed
                && blocked_read
                && generated_full
        );
        assert!(l1.is_idle());
        assert_eq!(physical, requests);
        assert_eq!(retired, 4);
        let expected = c220_bt_uops(transfer, nz(96))
            .flat_map(|uop| {
                (0..uop.output_bytes).step_by(96).map(move |offset| {
                    (
                        uop.destination_address + u64::from(offset),
                        (uop.output_bytes - offset).min(96),
                    )
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(output, expected);
        assert_eq!(
            output
                .iter()
                .map(|&(_, bytes)| u64::from(bytes))
                .sum::<u64>(),
            transfer.output_bytes()
        );
        let mut empty = transfer;
        empty.descriptor.burst_blocks = 0;
        let issue = frontend
            .issue(200, 8, C220Mte1ReadTransfer::Bt(empty))
            .unwrap();
        assert!(issue.completion_ready && issue.request_count == 0 && frontend.is_idle());
        let before = frontend.clone();
        assert_eq!(
            frontend.issue(u64::MAX, 9, C220Mte1ReadTransfer::Bt(transfer)),
            Err(C220Mte1ReadFrontendError::TimeOverflow)
        );
        assert_eq!(frontend, before);
    }
    let mut large = bt_transfer(true);
    large.descriptor.burst_count = 4095;
    large.descriptor.burst_blocks = u16::MAX;
    large.source_base = u64::MAX;
    let mut plan = C220BtRequestPlan::new(large, nz(64), NonZeroU32::MIN);
    assert_eq!(plan.remaining_requests(), large.input_bytes());
    assert_eq!(plan.next().unwrap().source_address, u64::MAX);
    assert_eq!(plan.next().unwrap().source_address, 0);
    assert_eq!(plan.remaining_requests(), large.input_bytes() - 2);
}

#[test]
fn load2d_and_bt_share_input_capacity_ids_and_output_with_independent_generators() {
    #[derive(Clone, Copy)]
    enum Callback {
        Generator(C220Mte1ReadKind, C220MteGeneratorCallback),
        Memory(C220L1Callback),
        L1(C220MteL1Callback),
        L0(bool, C220L0WriteCallback),
    }
    for word in [0x6000_2180, 0x6000_2181, 0x6000_218d, 0x6000_21a0] {
        let mut registers = [0; 32];
        registers[0] = 0x800;
        registers[2] = 31;
        registers[3] = (2 << 16) | (3 << 24) | (1 << 44);
        let transfer = C220Load2dInstruction::decode(word)
            .unwrap()
            .capture(&registers)
            .unwrap();
        let expected = C220Load2dRequestPlan::new(transfer, nz(96))
            .unwrap()
            .collect::<Vec<_>>();
        let mut load = C220Mte1ReadFrontend::new(C220Mte1ReadKind::Load2d, nz(96), bandwidths(96));
        let mut bt = C220Mte1ReadFrontend::new(C220Mte1ReadKind::Bt, nz(96), bandwidths(96));
        let mut interface = C220MteL1Interface::default();
        let mut l1 = C220L1Transport::new(C220L1Geometry::new(32, 4, 1, 0).unwrap());
        let mut l0a = C220L0WritePipeline::default();
        let mut l0b = C220L0WritePipeline::default();
        let mut events = EventDispatcher::new(0);
        let clock = events.add_event();
        let memory_events = C220L1Events::register(&mut events, clock, Callback::Memory);
        let l1_events = C220MteL1Events::register(&mut events, clock, Callback::L1);
        let l0a_events =
            C220L0WriteEvents::register(&mut events, clock, |phase| Callback::L0(false, phase));
        let l0b_events =
            C220L0WriteEvents::register(&mut events, clock, |phase| Callback::L0(true, phase));
        let load_events = C220Mte1ReadEvents::register(&mut events, clock, |phase| {
            Callback::Generator(C220Mte1ReadKind::Load2d, phase)
        });
        let bt_events = C220Mte1ReadEvents::register(&mut events, clock, |phase| {
            Callback::Generator(C220Mte1ReadKind::Bt, phase)
        });
        load_events
            .issue(
                &mut events,
                &mut load,
                1,
                C220Mte1ReadTransfer::Load2d(transfer),
            )
            .unwrap();
        bt_events
            .issue(
                &mut events,
                &mut bt,
                2,
                C220Mte1ReadTransfer::Bt(bt_transfer(true)),
            )
            .unwrap();
        let mut ids = BTreeSet::new();
        let mut reads = Vec::new();
        let mut retired = BTreeSet::new();
        let mut output_bytes = [0_u32; 2];
        let mut shared_full = false;
        let mut idle_before_retirement = false;
        let mut bt_tails = BTreeMap::new();
        for tick in 0..300 {
            events.advance_to(tick).unwrap();
            events.notify_at(clock, tick);
            events.notify_at(clock, tick);
            let mut phases = [Vec::new(), Vec::new()];
            let mut l1_phases = Vec::new();
            while let Some(invocation) = events.next_callback() {
                let (kind, phase) = match invocation.callback {
                    Callback::Memory(phase) => {
                        memory_events.handle(phase, &mut events, &mut l1).unwrap();
                        continue;
                    }
                    Callback::Generator(kind, phase) => (kind, phase),
                    Callback::L0(is_b, phase) => {
                        let (binding, pipeline) = if is_b {
                            (&l0b_events, &mut l0b)
                        } else {
                            (&l0a_events, &mut l0a)
                        };
                        if let C220L0WriteEventOutcome::Acknowledged(Some(ack)) =
                            binding.handle(phase, &mut events, pipeline).unwrap()
                            && let Some(id) = ack.retired_instruction()
                        {
                            assert!(retired.insert(id));
                        }
                        continue;
                    }
                    Callback::L1(phase) => {
                        assert!(!l1_phases.contains(&phase));
                        l1_phases.push(phase);
                        let response = l1
                            .responses(C220L1Port::MteRead)
                            .front()
                            .filter(|head| head.ready_tick <= tick)
                            .map(|head| head.payload.request.id);
                        let inputs = C220MteL1CycleInputs {
                            request_ready: tick >= 20 && l1.request_ready(C220L1Port::MteRead),
                            response,
                            output_credits: C220MteL1OutputCredits {
                                l0a: [l0a.can_push(C220L0WritePort::Port0), false, false],
                                l0b: [l0b.can_push(C220L0WritePort::Port0), false, false],
                            },
                        };
                        match l1_events
                            .handle(phase, &mut events, &mut interface, inputs)
                            .unwrap()
                        {
                            C220MteL1EventOutcome::Request(send) => {
                                if let Some(request) = send.sent {
                                    assert!(ids.insert(request.id));
                                    if let C220Mte1ReadUop::Load2d(uop) = request.operation.payload
                                    {
                                        reads.push(uop);
                                    }
                                    assert!(
                                        l1.send_request(
                                            tick,
                                            C220L1Port::MteRead,
                                            request.l1_request()
                                        )
                                        .unwrap()
                                    );
                                }
                            }
                            C220MteL1EventOutcome::Response(Some(request)) => {
                                let received = l1
                                    .receive_response(tick, C220L1Port::MteRead)
                                    .unwrap()
                                    .unwrap();
                                assert_eq!(received.request.id, request.id);
                            }
                            C220MteL1EventOutcome::Output(output) => {
                                if let Some(sent) = output.sent {
                                    output_bytes[(sent.fragment.instruction_id - 1) as usize] +=
                                        sent.fragment.bytes;
                                    match sent.destination {
                                        C220MteL1OutputDestination::Fb => {
                                            panic!("MTE1 workload does not load factors")
                                        }
                                        C220MteL1OutputDestination::Smask => {
                                            panic!("workload does not load sparse masks")
                                        }
                                        C220MteL1OutputDestination::SparseIndex => {
                                            panic!("index responses must not emit output")
                                        }
                                        C220MteL1OutputDestination::L0a(port) => {
                                            assert!(l0a.push(tick, port, sent.fragment).unwrap())
                                        }
                                        C220MteL1OutputDestination::L0b(port) => {
                                            assert!(l0b.push(tick, port, sent.fragment).unwrap())
                                        }
                                        C220MteL1OutputDestination::Bt => {
                                            if sent.fragment.last_in_uop {
                                                bt_tails.insert(sent.fragment.request_id, tick);
                                            }
                                        }
                                    }
                                }
                            }
                            C220MteL1EventOutcome::Retired(Some(event)) => {
                                assert_eq!(tick, bt_tails[&event.fragment.request_id] + 5);
                                if event.fragment.last_in_instruction {
                                    assert!(retired.insert(event.fragment.instruction_id));
                                }
                            }
                            _ => {}
                        }
                        continue;
                    }
                };
                let (binding, frontend, index) = match kind {
                    C220Mte1ReadKind::Load3dv2 => unreachable!("not registered in this test"),
                    C220Mte1ReadKind::Load2d => (&load_events, &mut load, 0),
                    C220Mte1ReadKind::Bt => (&bt_events, &mut bt, 1),
                };
                if matches!(
                    phase,
                    C220MteGeneratorCallback::Send | C220MteGeneratorCallback::Generate
                ) {
                    phases[index].push(phase);
                }
                binding
                    .handle(phase, &mut events, frontend, false, &mut interface)
                    .unwrap();
            }
            for phases in phases {
                assert!(phases.len() <= 2);
                if phases.len() == 2 {
                    assert_eq!(
                        phases,
                        [
                            C220MteGeneratorCallback::Send,
                            C220MteGeneratorCallback::Generate
                        ]
                    );
                }
            }
            idle_before_retirement |= load.is_idle() && bt.is_idle() && retired.len() < 2;
            let inputs = interface.queue_state().inputs;
            assert!(inputs[0] <= 5 && inputs[1] == 0 && inputs[2] == 0);
            shared_full |= inputs[0] == 5;
            if load.is_idle()
                && bt.is_idle()
                && interface.is_idle()
                && l0a.is_idle()
                && l0b.is_idle()
            {
                break;
            }
        }
        assert!(shared_full && idle_before_retirement);
        assert_eq!(reads, expected);
        assert_eq!(ids.len(), expected.len() + 4);
        assert_eq!(output_bytes, [1024, 512]);
        assert_eq!(retired, BTreeSet::from([1, 2]));
        assert!(
            load.is_idle()
                && bt.is_idle()
                && interface.is_idle()
                && l1.is_idle()
                && l0a.is_idle()
                && l0b.is_idle()
        );
    }
}
