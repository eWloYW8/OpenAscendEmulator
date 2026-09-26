use super::*;
use crate::isa::c220::mte::{C220MovInstruction, CAPTURED_C220_MOV_OUT_TO_UB_X_WORD};
use crate::memory::{
    mapped::MappedMemory, region::MemoryRegion, sparse::SparseMemory, ub::UbMemory,
};
use crate::sim::c220::core::C220CoreTimingRules;
use crate::sim::c220::memory::l1::C220L1Geometry;
use crate::sim::c220::mte::C220MtePipelineConfig;
use crate::sim::c220::mte::mte2::{C220Mte2Completion, C220Mte2TimingRules};
use crate::sim::c220::mte::mte3::C220Mte3TimingRules;
use crate::sim::c220::mte::read::C220MteReadBandwidths;
use crate::sim::c220::mte::set2d::C220Set2dBandwidths;
use crate::sim::c220::state::C220State;
use crate::sim::c220::vector::pipeline::C220VectorTimingRules;
use crate::sim::common::scalar::{ScalarMachine, ScalarStepper};
use std::num::{NonZeroU32, NonZeroU64};

fn configured_dma_core() -> C220Core {
    let memory = MappedMemory::bind(
        SparseMemory::new(
            vec![MemoryRegion::new(4096, vec![7; 4096]).unwrap()],
            4096,
            4096,
        ),
        &[0x2000],
    )
    .unwrap();
    let word = CAPTURED_C220_MOV_OUT_TO_UB_X_WORD;
    let operands = C220MovInstruction::decode(word).unwrap();
    let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
    machine.set_xreg(operands.source_register, 0x2000).unwrap();
    machine.set_xreg(operands.destination_register, 0).unwrap();
    machine
        .set_xreg(operands.descriptor_register, (128 << 16) | (1 << 4))
        .unwrap();
    let one = NonZeroU64::new(1).unwrap();
    let mut core = C220Core::new(
        C220State::new(ScalarStepper::new(machine, 0), UbMemory::new(8192, 8192)),
        memory,
        C220CoreTimingRules {
            mte2: C220Mte2TimingRules {
                issue_interval: one,
                startup_ticks: u64::MAX,
                bytes_per_tick: one,
                retire_ticks: u64::MAX,
            },
            mte3: C220Mte3TimingRules {
                issue_interval: one,
                startup_ticks: 0,
                bytes_per_tick: one,
                retire_ticks: 0,
            },
            vector: C220VectorTimingRules {
                dispatch_ticks: 0,
                uop_issue_interval: one,
                ub_response_ticks: 1,
            },
        },
    )
    .unwrap();
    let width = NonZeroU32::new(32).unwrap();
    core.configure_mte_pipeline(C220MtePipelineConfig {
        core_kind: crate::sim::c220::device::C220CoreKind::Vector0,
        l1: C220L1Geometry::new(32, 4, 1, 0).unwrap(),
        read_width: width,
        output_bandwidths: C220MteReadBandwidths {
            l0a: width,
            l0b: width,
            bt: width,
            smask: width,
        },
        set2d_bandwidths: C220Set2dBandwidths {
            l0a: width,
            l0b: width,
            l1: width,
        },
    })
    .unwrap();
    core
}

#[test]
fn nd2nz_runs_through_core_events_and_retires_after_l1_ack() {
    use crate::isa::c220::mte::nd2nz::C220Nd2NzInstruction;
    use crate::sim::c220::mte::{
        C220MtePipelineEvent,
        interface::biu_read::write::C220BiuWriteBandwidths,
        interface::{C220MteL1WriteEventOutcome, C220MteL1WritePort},
        mte2::C220Mte2Result,
        nd2nz::{C220Nd2NzStagingConfig, execute_c220_nd2nz},
    };
    use std::collections::VecDeque;
    for columns in [35_u64, 64] {
        let mut core = configured_dma_core();
        let width = NonZeroU32::new(32).unwrap();
        core.connect_mte2_biu(
            C220BiuReadConfig {
                outstanding: NonZeroU32::new(4).unwrap(),
                weights: [1; 3],
                group_vector_returns: false,
                write_bandwidths: C220BiuWriteBandwidths {
                    l1: width,
                    l0a: width,
                    l0b: width,
                    ub: width,
                },
            },
            C220BiuSubcore::Cube,
        )
        .unwrap();
        core.configure_nd2nz(
            C220Nd2NzStagingConfig {
                rows: NonZeroU32::new(8).unwrap(),
                alignment_depth: 256,
                small_data_capacity: NonZeroU32::new(256).unwrap(),
                receive_bandwidth: width,
            },
            width,
        )
        .unwrap();
        let word = (3 << 29) | (1 << 27) | (12 << 22) | (1 << 17) | (2 << 12) | (3 << 7) | (4 << 2);
        let machine = core.state.scalar_mut().machine_mut();
        for (reg, value) in [
            (1, 512),
            (2, 0x2000),
            (3, (1 << 4) | (3 << 16) | (columns << 32)),
            (4, 64 | (3 << 16) | (1 << 32)),
        ] {
            machine.set_xreg(reg, value).unwrap();
        }
        let transfer = C220Nd2NzInstruction::decode(word)
            .unwrap()
            .capture(machine.xregs());
        let mut expected = crate::sim::c220::memory::C220LocalBuffer::new(4096);
        let result = execute_c220_nd2nz(transfer, &core.memory, &mut expected).unwrap();
        core.step_word_at(0, word).unwrap();
        for reg in 1..=4 {
            core.state
                .scalar_mut()
                .machine_mut()
                .set_xreg(reg, 0)
                .unwrap();
        }
        let mut responses = VecDeque::new();
        let mut ack_tick = None;
        let mut retired = None;
        for tick in 1..300 {
            core.advance_to(tick).unwrap();
            for event in core.mte_pipeline().unwrap().last_events() {
                if let C220MtePipelineEvent::L1Write(C220MteL1WriteEventOutcome::Acknowledged(
                    Some(ack),
                )) = event
                    && ack.retired_instruction().is_some()
                {
                    assert_eq!(ack.request.port, C220MteL1WritePort::Port1);
                    ack_tick = Some(tick);
                }
            }
            if let Some(request) = core.take_mte2_biu_request() {
                responses.extend(
                    (0..request.input.generated.request.bytes.div_ceil(128)).map(
                        |transaction_id| C220BiuReadBeat {
                            tag: request.tag,
                            transaction_id,
                        },
                    ),
                );
            }
            if let Some(outcome) = core.mte2.last_outcomes().first() {
                retired = Some(outcome.clone());
                break;
            }
            if core
                .receive_mte2_biu_at(tick, [responses.front().copied(), None])
                .unwrap()[0]
            {
                responses.pop_front();
            }
        }
        let retired = retired.expect("ND2NZ core retirement");
        assert_eq!(retired.retire_tick, ack_tick.unwrap() + 1);
        assert_eq!(retired.result, C220Mte2Result::Nd2Nz(result));
        for segment in transfer.segments() {
            assert_eq!(
                core.local_memory
                    .l1()
                    .read_states_linear(segment.destination_address, segment.output_bytes as usize)
                    .unwrap(),
                expected
                    .read_states_linear(segment.destination_address, segment.output_bytes as usize)
                    .unwrap()
            );
        }
        assert!(core.mte_pipeline().unwrap().is_idle());
        assert!(!core.mte2_is_busy());
    }
}

#[test]
fn mov_pad_input_counts_padding_traffic_and_retires_after_ub_ack() {
    use crate::memory::sparse::MemoryByteState;
    use crate::sim::c220::mte::interface::biu_read::write::C220BiuWriteBandwidths;
    use crate::sim::c220::mte::mte2::C220Mte2Result;
    use crate::sim::c220::mte::{C220MtePipelineEvent, dma::C220DmaEventOutcome};
    use std::collections::VecDeque;

    for (length, padding, gap, expected_written) in [
        (17, 0, 0, 64),
        (128, 0, 0, 256),
        (513, 1, 1, 1088),
        (64, 0, 1, 128),
    ] {
        let mut core = configured_dma_core();
        let width = NonZeroU32::new(32).unwrap();
        core.connect_mte2_biu(
            C220BiuReadConfig {
                outstanding: NonZeroU32::new(2).unwrap(),
                weights: [1; 3],
                group_vector_returns: true,
                write_bandwidths: C220BiuWriteBandwidths {
                    l1: width,
                    l0a: width,
                    l0b: width,
                    ub: width,
                },
            },
            C220BiuSubcore::Vector0,
        )
        .unwrap();
        core.state.set_isa_instance_index(1);
        let word =
            (3 << 29) | (1 << 27) | (14 << 22) | (1 << 17) | (2 << 12) | (3 << 7) | (4 << 2) | 2;
        let machine = core.state.scalar_mut().machine_mut();
        machine.set_xreg(1, 0).unwrap();
        machine.set_xreg(2, 0x2000).unwrap();
        machine
            .set_xreg(
                3,
                9 | (2 << 4) | (length << 16) | (padding << 48) | (padding << 54),
            )
            .unwrap();
        machine.set_xreg(4, gap << 32).unwrap();
        machine.set_spr_value(70, 0x44332211).unwrap();
        machine.set_spr_value(93, 5).unwrap();
        machine.set_spr_value(94, 7).unwrap();
        let captured =
            crate::sim::c220::mte::mov_pad::C220MovPadCommand::capture(machine, 0, word, 1)
                .unwrap();
        core.step_word_at(0, word).unwrap();
        let machine = core.state.scalar_mut().machine_mut();
        machine.set_xreg(3, 0).unwrap();
        machine.set_xreg(4, u64::MAX).unwrap();
        machine.set_spr_value(70, 0).unwrap();
        machine.set_spr_value(93, 0).unwrap();
        let mut responses = VecDeque::new();
        let mut traffic = 0;
        let mut last_response = None;
        let mut acknowledged = None;
        let mut retired = None;
        for tick in 1..500 {
            core.advance_to(tick).unwrap();
            for event in core.mte_pipeline().unwrap().last_events() {
                match event {
                    C220MtePipelineEvent::Dma(C220DmaEventOutcome::Generated(Some(request))) => {
                        assert_eq!(request.ready_tick, tick + 3);
                        assert_eq!(request.sid, Some(9));
                    }
                    C220MtePipelineEvent::UbRequest(_, request) => {
                        traffic += request.fragment.bytes
                    }
                    C220MtePipelineEvent::UbResponse(_, request)
                        if request.fragment.last_in_instruction =>
                    {
                        last_response = Some(tick)
                    }
                    C220MtePipelineEvent::UbWrite(
                        _,
                        crate::sim::c220::mte::interface::ub_write::C220UbWriteEvent::Acknowledged(
                            Some(ack),
                        ),
                    ) if ack.retired_instruction().is_some() => acknowledged = Some(tick),
                    _ => {}
                }
            }
            if let Some(request) = core.take_mte2_biu_request() {
                assert_eq!(request.input.generated.sid, Some(9));
                assert!(request.input.generated.request.bytes <= 128);
                responses.push_back(C220BiuReadBeat {
                    tag: request.tag,
                    transaction_id: 0,
                });
            }
            if let Some(outcome) = core.mte2.last_outcomes().first() {
                retired = Some(outcome.clone());
                break;
            }
            assert_eq!(core.state.ub.tracked_bytes(), 0);
            if core
                .receive_mte2_biu_at(tick, [responses.front().copied(), None])
                .unwrap()[0]
            {
                responses.pop_front();
            }
        }
        let retired = retired.expect("MOV_PAD input retires");
        assert!(retired.retire_tick > last_response.unwrap());
        assert_eq!(retired.retire_tick, acknowledged.unwrap() + 1);
        assert_eq!(traffic, expected_written);
        let C220Mte2Result::MovPad(result) = retired.result else {
            panic!("MOV_PAD result")
        };
        assert_eq!(result.bytes, captured.transfer.output_bytes() as usize * 2);
        for segment in captured.transfer.segments() {
            let bytes = core
                .state
                .ub
                .read_known(segment.destination_address, segment.output_bytes as usize)
                .unwrap();
            let left = padding as usize * 4;
            assert_eq!(
                &bytes[left..left + length as usize],
                vec![7; length as usize]
            );
            if padding != 0 {
                assert_eq!(&bytes[..4], &[0x11, 0x22, 0x33, 0x44]);
            }
            if gap != 0 {
                assert_eq!(
                    core.state
                        .ub
                        .read_states(
                            segment.destination_address + u64::from(segment.output_bytes),
                            1
                        )
                        .unwrap(),
                    [MemoryByteState::Unknown]
                );
            }
        }
        assert!(core.mte_pipeline().unwrap().is_idle());
        assert!(!core.mte2_is_busy());
        let tracked = core.state.ub.tracked_bytes();
        core.step_word_at(retired.retire_tick + 1, word).unwrap();
        core.advance_to(retired.retire_tick + 15).unwrap();
        assert!(!core.mte2_is_busy());
        assert!(core.mte_pipeline().unwrap().is_idle());
        assert_eq!(core.state.ub.tracked_bytes(), tracked);
    }
}

#[test]
fn external_load2d_retires_only_after_its_selected_local_destination() {
    use crate::sim::c220::mte::interface::biu_read::write::{
        C220BiuWriteBandwidths, C220BiuWriteDestination,
    };
    use crate::sim::c220::mte::interface::{C220L0WriteEventOutcome, C220MteL1WriteEventOutcome};
    use crate::sim::c220::mte::{C220MtePipelineEvent, dma::C220DmaEventOutcome};
    use std::collections::VecDeque;

    for destination in 0..=2 {
        for offset in [0, 7] {
            let mut core = configured_dma_core();
            let width = NonZeroU32::new(32).unwrap();
            core.connect_mte2_biu(
                C220BiuReadConfig {
                    outstanding: NonZeroU32::new(2).unwrap(),
                    weights: [1; 3],
                    group_vector_returns: true,
                    write_bandwidths: C220BiuWriteBandwidths {
                        l1: width,
                        l0a: width,
                        l0b: width,
                        ub: width,
                    },
                },
                C220BiuSubcore::Cube,
            )
            .unwrap();
            core.state.isa_instance_index = 1;
            let machine = core.state.scalar_mut().machine_mut();
            machine.set_spr_value(93, 5).unwrap();
            machine.set_xreg(1, 1024).unwrap();
            machine.set_xreg(2, 0x2000 + offset).unwrap();
            machine
                .set_xreg(
                    3,
                    (2 << 16)
                        | (1 << 24)
                        | (11 << 40)
                        | (1 << 44)
                        | if offset == 0 { 0 } else { 7 << 61 },
                )
                .unwrap();
            let transpose = destination != 2;
            let word = (3 << 29)
                | (1 << 17)
                | (2 << 12)
                | (3 << 7)
                | 16
                | if offset == 0 { 0 } else { 64 }
                | destination
                | (u32::from(transpose) << 2);
            core.step_word_at(0, word).unwrap();
            for register in 1..=3 {
                core.state
                    .scalar_mut()
                    .machine_mut()
                    .set_xreg(register, 0)
                    .unwrap();
            }
            let input: Vec<u8> = (0..1024).map(|i| (i % 251) as u8).collect();
            core.memory.write_known_at(0x2000 + offset, &input).unwrap();
            let mut responses = VecDeque::new();
            let mut generated = 0;
            let mut acknowledged = None;
            let mut retired = None;
            for tick in 1..300 {
                core.advance_to(tick).unwrap();
                assert!(core.mte_pipeline().unwrap().mte1_completions().is_empty());
                for event in core.mte_pipeline().unwrap().last_events() {
                    match event {
                        C220MtePipelineEvent::ExternalLoad2d(C220DmaEventOutcome::Generated(
                            Some(request),
                        )) => {
                            assert_eq!(request.ready_tick, tick + 2);
                            assert_eq!(request.sid, Some(11));
                            generated += 1;
                        }
                        C220MtePipelineEvent::L0a(C220L0WriteEventOutcome::Acknowledged(Some(
                            ack,
                        )))
                        | C220MtePipelineEvent::L0b(C220L0WriteEventOutcome::Acknowledged(Some(
                            ack,
                        ))) if ack.retired_instruction().is_some() => acknowledged = Some(tick),
                        C220MtePipelineEvent::L1Write(
                            C220MteL1WriteEventOutcome::Acknowledged(Some(ack)),
                        ) if ack.retired_instruction().is_some() => acknowledged = Some(tick),
                        _ => {}
                    }
                }
                if let Some(request) = core.take_mte2_biu_request() {
                    assert_eq!(request.input.generated.sid, Some(11));
                    assert_eq!(
                        request.input.destination,
                        [
                            C220BiuWriteDestination::L0A,
                            C220BiuWriteDestination::L0B,
                            C220BiuWriteDestination::L1
                        ][destination as usize]
                    );
                    responses.extend(
                        (0..request.input.generated.request.bytes.div_ceil(128))
                            .rev()
                            .map(|transaction_id| C220BiuReadBeat {
                                tag: request.tag,
                                transaction_id,
                            }),
                    );
                }
                if let Some(outcome) = core.mte2.last_outcomes().first() {
                    retired = Some(outcome.clone());
                    break;
                }
                if core
                    .receive_mte2_biu_at(tick, [responses.front().copied(), None])
                    .unwrap()[0]
                {
                    responses.pop_front();
                }
            }
            let retired = retired.expect("LOAD2D completed");
            assert_eq!(retired.retire_tick, acknowledged.unwrap() + 1);
            assert_eq!(generated, if offset == 0 { 8 } else { 2 });
            let target = match destination {
                0 => core.local_memory.l0a(),
                1 => core.local_memory.l0b(),
                _ => core.local_memory.l1(),
            };
            for block in 0..2 {
                let actual = target
                    .read_initialized_linear(1024 + block * 1024, 512)
                    .unwrap();
                for (index, byte) in input[block as usize * 512..(block as usize + 1) * 512]
                    .iter()
                    .enumerate()
                {
                    let out = if transpose {
                        (index / 32 + ((index / 2) % 16) * 16) * 2 + index % 2
                    } else {
                        index
                    };
                    assert_eq!(actual[out], *byte);
                }
            }
            assert!(!core.mte2_is_busy());
            assert!(core.mte_pipeline().unwrap().is_idle());
        }
    }
}

#[test]
fn external_smask_uses_default_generation_and_ordered_local_retirement() {
    use crate::sim::c220::mte::C220MtePipelineEvent;
    use crate::sim::c220::mte::interface::C220MteL1EventOutcome;
    use crate::sim::c220::mte::mte2::C220Mte2Result;
    use crate::sim::c220::mte::read::{C220MteReadEventOutcome, C220MteReadKind};

    let mut core = configured_dma_core();
    core.advance_to(300).unwrap();
    let machine = core.state.scalar_mut().machine_mut();
    machine.set_xreg(10, 511).unwrap();
    machine.set_xreg(11, 0x2000).unwrap();
    machine.set_xreg(12, 65).unwrap();
    let word = (3 << 29) | (17 << 22) | (10 << 17) | (11 << 12) | (12 << 2);
    assert!(matches!(
        core.step_word_at(301, word).unwrap(),
        C220CoreStep::Executed {
            instruction: C220CoreInstruction::Mte2Queued(_),
            ..
        }
    ));
    for register in 10..=12 {
        core.state
            .scalar_mut()
            .machine_mut()
            .set_xreg(register, 0)
            .unwrap();
    }
    let input: Vec<u8> = (0..130).collect();
    core.memory.write_known_at(0x2000, &input).unwrap();
    let mut generated = 0;
    let mut tail = None;
    let mut completion = None;
    let mut retired = None;
    for tick in 302..400 {
        core.advance_to(tick).unwrap();
        let pipeline = core.mte_pipeline().unwrap();
        assert!(pipeline.mte1_completions().is_empty());
        assert!(pipeline.dma_output().is_none());
        for event in pipeline.last_events() {
            match event {
                C220MtePipelineEvent::Generator(
                    C220MteReadKind::Default,
                    C220MteReadEventOutcome::Generated(Some(request)),
                ) => {
                    assert_eq!(request.ready_tick, tick + 6);
                    generated += 1;
                }
                C220MtePipelineEvent::Interface(C220MteL1EventOutcome::Output(output))
                    if output
                        .sent
                        .is_some_and(|sent| sent.fragment.last_in_instruction) =>
                {
                    tail = Some(tick);
                }
                _ => {}
            }
        }
        if !pipeline.mte2_read_completions().is_empty() {
            completion = Some(tick);
        }
        if let Some(outcome) = core.mte2.last_outcomes().first() {
            retired = Some(outcome.clone());
            break;
        }
        assert_eq!(core.local_memory.smask().read_byte(512), 0);
    }
    let retired = retired.expect("external SMASK retired");
    assert_eq!(generated, 5);
    assert_eq!(completion.unwrap(), tail.unwrap() + 5);
    assert_eq!(retired.retire_tick, completion.unwrap() + 1);
    let C220Mte2Result::MovOutToSmask(result) = retired.result else {
        panic!("SMASK result")
    };
    assert_eq!(result.bytes, 130);
    let mut first = [0; 2];
    core.local_memory
        .smask()
        .read_into(511, &mut first)
        .unwrap();
    assert_eq!(first, input[..2]);
    let mut rest = [0; 128];
    core.local_memory.smask().read_into(1, &mut rest).unwrap();
    assert_eq!(rest, input[2..]);
    assert!(!core.mte2_is_busy());
    assert!(core.mte_pipeline().unwrap().is_idle());

    let tick = retired.retire_tick + 1;
    core.step_word_at(tick, word).unwrap();
    core.advance_to(tick + 10).unwrap();
    assert!(!core.mte2_is_busy());
    assert!(core.mte_pipeline().unwrap().is_idle());
    core.local_memory.smask().read_into(1, &mut rest).unwrap();
    assert_eq!(rest, input[2..]);
}

#[test]
fn shared_spr_write_requires_both_queues_and_retires_on_both_pipes() {
    use crate::sim::c220::core::{C220Mte1Operation, C220Mte2Operation};
    use crate::sim::c220::mte::mte2::{C220Mte2Command, C220Mte2Result};
    for spr in [13, 15] {
        let mut core = configured_dma_core();
        core.mte2_frontend.config.vector_issue_queue_depth = NonZeroU32::new(1).unwrap();
        core.state
            .scalar_mut()
            .machine_mut()
            .set_xreg(14, u64::MAX)
            .unwrap();
        // An unmatched WAIT holds the MTE2 issue queue without a running DMA.
        let wait = (2 << 29) | (6 << 21) | (3 << 10) | (4 << 7) | 7;
        core.step_word_at(0, wait).unwrap();
        let pc = core.state.scalar().pc();
        let prior = core.state.scalar().machine().spr_value(spr);
        let word = (2 << 24) | (u32::from(spr) << 17) | (14 << 12) | (18 << 7);
        let blocked = core.step_word_at(1, word).unwrap();
        assert!(
            matches!(
                blocked,
                C220CoreStep::Stalled(C220Stall {
                    cause: C220StallCause::Mte2IssueQueueFull,
                    ..
                })
            ),
            "{blocked:?}"
        );
        assert_eq!(core.state.scalar().pc(), pc);
        assert_eq!(core.state.scalar().machine().spr_value(spr), prior);
        assert_eq!(core.queued_mte1_instructions().count(), 0);

        let signal = FlagInstruction::decode(Architecture::Dav2201, wait - (1 << 21))
            .unwrap()
            .resolve(pc, core.state.scalar().machine().xregs());
        core.pipeline_events.set(100, signal, None, 1);
        let C220CoreStep::Executed {
            instruction: C220CoreInstruction::MteSprQueued { mte1, mte2 },
            ..
        } = core.step_word_at(2, word).unwrap()
        else {
            panic!("shared SPR issue");
        };
        assert_eq!(mte1.instruction_id, mte2.instruction_id);
        assert_eq!(core.state.scalar().pc(), pc + 4);
        assert!(matches!(mte1.operation, C220Mte1Operation::Command(_)));
        let C220Mte2Operation::Command(C220Mte2Command::WriteSpr(step)) = mte2.operation else {
            panic!("MTE2 SPR command");
        };
        assert_eq!(step.value, u64::from(u32::MAX));
        core.advance_to(6).unwrap();
        assert!(core.mte2.last_outcomes().is_empty());
        core.advance_to(7).unwrap();
        let outcome = core.mte2.last_outcomes().last().unwrap();
        assert_eq!(outcome.command.instruction_id, mte1.instruction_id);
        assert_eq!(outcome.result, C220Mte2Result::WriteSpr(step));
        assert_eq!(outcome.retire_tick, 7);
        assert!(core.activity().is_idle());
    }
}

#[test]
fn mte2_issue_queue_preserves_operands_and_releases_mte1_at_retirement() {
    let mut core = configured_dma_core();
    core.mte2_frontend.config.vector_issue_queue_depth = NonZeroU32::new(2).unwrap();
    core.mte2_frontend.config.outstanding_limit = NonZeroU32::new(1).unwrap();
    let flag = |op: u32, source: u32, dest: u32| {
        (2 << 29) | (op << 21) | (1 << 17) | (source << 10) | (dest << 7) | (12 << 2)
    };
    let machine = core.state.scalar_mut().machine_mut();
    machine.set_xreg(1, 0).unwrap();
    machine.set_xreg(3, 1 | (2 << 16)).unwrap();
    machine.set_xreg(12, 0x1234_5678).unwrap();
    machine.set_spr_value(15, 0x4321_4321).unwrap();
    let fill = (3 << 29) | (1 << 22) | (1 << 17) | (3 << 7) | 6;
    core.step_word_at(0, flag(6, 2, 4)).unwrap();
    core.step_word_at(1, fill).unwrap();
    assert!(matches!(core.step_word_at(2, flag(5, 4, 3)).unwrap(),
        C220CoreStep::Stalled(stall) if stall.cause == C220StallCause::Mte2IssueQueueFull));
    assert_eq!(core.queued_mte2_instructions().len(), 2);
    assert_eq!(core.outstanding_mte2_commands(), 0);
    assert!(matches!(
        core.step_word_at(3, 0x4140_0000).unwrap(),
        C220CoreStep::Executed { .. }
    ));
    core.step_word_at(4, flag(5, 2, 4)).unwrap();
    core.step_word_at(6, flag(5, 4, 3)).unwrap();
    core.step_word_at(8, fill).unwrap();
    let identity = (core.state.scalar().pc(), core.next_instruction_id);
    assert!(matches!(core.step_word_at(9, flag(5, 4, 3)).unwrap(),
        C220CoreStep::Stalled(stall) if stall.cause == C220StallCause::Mte2IssueQueueFull));
    assert_eq!(
        (core.state.scalar().pc(), core.next_instruction_id),
        identity
    );
    assert_eq!(core.outstanding_mte2_commands(), 1);
    assert!(
        core.mte2_frontend_outcomes()
            .iter()
            .any(|step| matches!(step,
        C220CoreStep::Stalled(stall) if stall.cause == C220StallCause::Mte2OutstandingLimit))
    );
    core.step_word_at(10, flag(6, 4, 3)).unwrap();
    let machine = core.state.scalar_mut().machine_mut();
    machine.set_xreg(12, 0).unwrap();
    machine.set_xreg(3, 0).unwrap();
    machine.set_spr_value(15, 0).unwrap();
    core.advance_to(150).unwrap();
    assert!(!core.mte2_is_busy());
    assert_eq!(core.mte2.last_outcomes().len(), 2);
    assert_eq!(core.queued_mte1_instructions().count(), 0);
    let event = core.pipeline_events().last_consumptions().last().unwrap();
    assert_eq!(event.step.flag_id, 0x1234_5678);
    assert_eq!(
        event.event.published_tick,
        core.mte2.last_outcomes()[0].retire_tick
    );
    assert!(event.tick >= event.event.published_tick);
    assert_eq!(
        core.local_memory.l1().read_known(0, 64).unwrap(),
        0x4321_4321_u32.to_le_bytes().repeat(16)
    );
}

#[test]
fn l1_dma_routes_cube_returns_and_commits_after_l1_acknowledgment() {
    use crate::isa::c220::mte::out_to_l1::{C220L1DmaDescriptor, C220L1DmaLayout};
    use crate::sim::c220::mte::C220MtePipelineEvent;
    use crate::sim::c220::mte::interface::biu_read::returns::C220BiuReturnEvent;
    use crate::sim::c220::mte::interface::biu_read::write::C220BiuWriteBandwidths;
    use crate::sim::c220::mte::interface::{C220MteL1WriteEventOutcome, C220MteL1WritePort};
    use std::collections::{BTreeMap, VecDeque};

    for (layout, xm, source_offset) in [
        (C220L1DmaLayout::Copy32, (64 << 16) | (1 << 4), 0),
        (C220L1DmaLayout::Copy32, (60 << 16) | (1 << 4), 1),
        (
            C220L1DmaLayout::Copy32,
            (1_u64 << 48) | (2 << 16) | (4 << 4),
            32,
        ),
        (
            C220L1DmaLayout::Copy32,
            (2_u64 << 48) | (1 << 32) | (2 << 16) | (4 << 4),
            0,
        ),
        (C220L1DmaLayout::Take4, (1 << 16) | (3 << 4), 0),
        (
            C220L1DmaLayout::Take8,
            (1_u64 << 48) | (2 << 32) | (5 << 16) | (3 << 4),
            1,
        ),
        (C220L1DmaLayout::Take16, (4 << 16) | (2 << 4), 0),
    ]
    .into_iter()
    .chain(
        [
            C220L1DmaLayout::Pad1,
            C220L1DmaLayout::Pad2,
            C220L1DmaLayout::Pad4,
            C220L1DmaLayout::Pad8,
            C220L1DmaLayout::Pad16,
        ]
        .into_iter()
        .flat_map(|layout| {
            [0, 1].map(move |offset| {
                (
                    layout,
                    (1_u64 << 48) | (1 << 32) | (2 << 16) | (9 << 4),
                    offset,
                )
            })
        }),
    ) {
        let mut core = configured_dma_core();
        let width = NonZeroU32::new(32).unwrap();
        core.connect_mte2_biu(
            C220BiuReadConfig {
                outstanding: NonZeroU32::new(2).unwrap(),
                weights: [1; 3],
                group_vector_returns: true,
                write_bandwidths: C220BiuWriteBandwidths {
                    l1: width,
                    l0a: width,
                    l0b: width,
                    ub: width,
                },
            },
            C220BiuSubcore::Cube,
        )
        .unwrap();
        let word = (3 << 29) | (2 << 27) | (2 << 23) | (1 << 17) | (2 << 12) | (3 << 7) | (4 << 3);
        let source = 0x2000 + source_offset;
        let machine = core.state.scalar_mut().machine_mut();
        machine.set_xreg(1, 0).unwrap();
        machine.set_xreg(2, source).unwrap();
        machine.set_xreg(3, xm).unwrap();
        machine.set_spr_value(13, 0xbbaa).unwrap();
        assert!(core.step_word_at(0, word | (1 << 22) | 1).is_err());
        assert!(!core.mte2_is_busy());
        let mode = layout as u32;
        let word = word | (mode & 7) | ((mode & 8) << 19);
        core.step_word_at(0, word).unwrap();
        core.state
            .scalar_mut()
            .machine_mut()
            .set_spr_value(13, 0x4321)
            .unwrap();
        let initial = core.local_memory.l1().read_states(0, 4096).unwrap();
        let mut beats = VecDeque::new();
        let mut offered = BTreeMap::new();
        let mut responses = BTreeMap::new();
        let mut acknowledged = None;
        let mut sent_bytes = 0;
        let mut truncated_bytes = 0;
        for tick in 1..400 {
            core.advance_to(tick).unwrap();
            if let Some(request) = core.take_mte2_biu_request() {
                assert_eq!(request.input.subcore, C220BiuSubcore::Cube);
                let request_data = request.input.generated.request;
                truncated_bytes += (request_data.bytes / 32) * layout.destination_bytes();
                core.memory
                    .write_known_at(
                        request_data.source_address,
                        &vec![9; request_data.bytes as usize],
                    )
                    .unwrap();
                beats.extend(
                    (0..request_data.bytes.div_ceil(128))
                        .rev()
                        .map(|transaction_id| C220BiuReadBeat {
                            tag: request.tag,
                            transaction_id,
                        }),
                );
            }
            for event in core.mte_pipeline().unwrap().last_events() {
                match event {
                    C220MtePipelineEvent::BiuReturn(C220BiuReturnEvent::Send(send)) => {
                        if let Some(fragment) = send.sent() {
                            offered.insert(fragment.destination_address, tick);
                        }
                    }
                    C220MtePipelineEvent::L1Write(C220MteL1WriteEventOutcome::Sent(send)) => {
                        if let Some(request) = send.sent {
                            assert_eq!(request.port, C220MteL1WritePort::Port0);
                            assert!(tick >= offered[&request.fragment.destination_address] + 9);
                            sent_bytes += request.fragment.bytes;
                        }
                    }
                    C220MtePipelineEvent::L1Write(C220MteL1WriteEventOutcome::Response(Some(
                        request,
                    ))) => {
                        responses.insert(request.id, tick);
                    }
                    C220MtePipelineEvent::L1Write(C220MteL1WriteEventOutcome::Acknowledged(
                        Some(ack),
                    )) => {
                        assert_eq!(tick, responses[&ack.request.id] + 1);
                        if ack.retired_instruction().is_some() {
                            acknowledged = Some(tick);
                        }
                    }
                    _ => {}
                }
            }
            assert!(
                core.mte_pipeline()
                    .unwrap()
                    .l1_fill_completions()
                    .is_empty()
            );
            if !core.mte2_is_busy() {
                assert_eq!(Some(tick - 1), acknowledged);
                break;
            }
            assert_eq!(
                core.local_memory.l1().read_states(0, 4096).unwrap(),
                initial
            );
            if core
                .receive_mte2_biu_at(tick, [beats.front().copied(), None])
                .unwrap()[0]
            {
                beats.pop_front();
            }
        }
        assert!(!core.mte2_is_busy());
        assert!(core.mte_pipeline().unwrap().is_idle());
        assert!(beats.is_empty());
        let descriptor = C220L1DmaDescriptor { xm, layout };
        assert_eq!(
            sent_bytes as usize,
            if layout.source_bytes() < 32 {
                let read_bytes =
                    usize::from(descriptor.burst_count()) * layout.source_bytes() as usize;
                if source_offset == 0 {
                    read_bytes.div_ceil(32) * 32
                } else {
                    usize::from(descriptor.burst_count()) * 32
                }
            } else if layout == C220L1DmaLayout::Copy32 {
                descriptor.segments(source, 0).len() * 32
            } else {
                truncated_bytes as usize
            }
        );
        for segment in descriptor.segments(source, 0) {
            use crate::memory::sparse::MemoryByteState;
            let mut expected: Vec<_> = (0..segment.destination_bytes)
                .map(|index| {
                    MemoryByteState::Known(if layout == C220L1DmaLayout::Pad1 || index % 2 == 0 {
                        0xaa
                    } else {
                        0xbb
                    })
                })
                .collect();
            let copied = segment.source_bytes.min(segment.destination_bytes) as usize;
            expected[..copied].copy_from_slice(
                &core
                    .memory
                    .read_states_at(segment.source_address, copied)
                    .unwrap(),
            );
            assert_eq!(
                core.local_memory
                    .l1()
                    .read_states(
                        segment.destination_address,
                        segment.destination_bytes as usize
                    )
                    .unwrap(),
                expected
            );
        }
    }
}

#[test]
fn native_memory_drives_mte2_through_bus_rob_and_ub_retirement() {
    for (descriptor, offset) in [
        ((128 << 16) | (1 << 4), 0),
        ((120 << 16) | (1 << 4), 1),
        ((1_u64 << 48) | (2 << 16) | (4 << 4), 32),
        ((2_u64 << 48) | (1 << 32) | (2 << 16) | (4 << 4), 0),
    ] {
        run_native_memory_retirement(descriptor, offset);
    }
}

fn run_native_memory_retirement(descriptor: u64, source_offset: u64) {
    use crate::memory::sparse::MemoryByteState;
    use crate::sim::c220::memory::timed_memory::{
        C220MemoryCredits, C220MemoryLatency, C220MemoryRegionTiming, C220TimedMemoryConfig,
    };
    use crate::sim::c220::mte::interface::biu_read::write::C220BiuWriteBandwidths;
    let mut core = configured_dma_core();
    let word = CAPTURED_C220_MOV_OUT_TO_UB_X_WORD;
    let operands = C220MovInstruction::decode(word).unwrap();
    let source = 0x2000 + source_offset;
    let machine = core.state.scalar_mut().machine_mut();
    machine.set_xreg(operands.source_register, source).unwrap();
    machine
        .set_xreg(operands.descriptor_register, descriptor)
        .unwrap();
    let plan = decode_mte2_transfer(core.state.scalar().machine(), 0, word, 0).unwrap();
    let width = NonZeroU32::new(128).unwrap();
    core.connect_mte2_bus(
        C220BiuReadConfig {
            outstanding: NonZeroU32::new(2).unwrap(),
            weights: [1; 3],
            group_vector_returns: true,
            write_bandwidths: C220BiuWriteBandwidths {
                l1: width,
                l0a: width,
                l0b: width,
                ub: width,
            },
        },
        C220BiuSubcore::Vector0,
        NonZeroU32::new(1).unwrap(),
    )
    .unwrap();
    let timing = C220MemoryRegionTiming {
        read: C220MemoryLatency {
            minimum: 1,
            spread: 5,
        },
        dbid: C220MemoryLatency::fixed(2),
        completion: C220MemoryLatency::fixed(4),
    };
    let credits = C220MemoryCredits {
        limit: 1025,
        refill: 128,
    };
    core.configure_timed_memory(C220TimedMemoryConfig {
        input_capacity: NonZeroU32::new(2).unwrap(),
        pending_limit: NonZeroU32::new(1).unwrap(),
        credit_period: NonZeroU64::new(4).unwrap(),
        ddr_credits: credits,
        l2_read_credits: credits,
        l2_write_credits: credits,
        ddr: timing,
        l2: timing,
        l2_start: 0,
        l2_bytes: 0,
    })
    .unwrap();
    let id = core.next_instruction_id();
    let initial_ub = core.state.ub().read_states(0, 8192).unwrap();
    use crate::sim::c220::memory::biu_read::C220BiuReadCacheConfig;
    use crate::sim::c220::memory::timed_memory::C220MemoryReadId;
    use crate::sim::c220::scalar::lsu::{cache::*, miss_buffer::*, scheduler::*, store_buffer::*};
    let mut lsu = C220LsuRequestScheduler::new(
        4,
        2,
        2,
        2,
        C220LsuMissBuffer::new(C220LsuMissConfig {
            line_bytes: 64,
            main_entries: 2,
            sub_entries: 2,
        })
        .unwrap(),
        C220LsuStoreBuffer::new(C220LsuStoreConfig {
            line_bytes: 64,
            main_entries: 2,
            sub_entries: 2,
            timeout_ticks: 4,
        })
        .unwrap(),
    )
    .unwrap();
    let line = C220LsuLineKey {
        address: 0x2fc0,
        memory: C220LsuMemory::External,
    };
    use crate::sim::c220::scalar::lsu::C220LsuStage;
    use crate::sim::c220::scalar::lsu::commit::{
        C220LoadCommitMode, C220LoadId, C220LsuCommitLane, C220LsuRetirement,
    };
    use crate::sim::c220::scalar::{C220LoadOperands, C220ScalarMappedAddress};
    let mut load_machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
    load_machine.set_xreg(5, line.address + 8).unwrap();
    load_machine.set_xreg(7, u64::MAX).unwrap();
    let operands = C220LoadOperands::capture(&load_machine, 0x4000, 0x03ce_5000).unwrap();
    let mapped = C220ScalarMappedAddress {
        address: operands.effective_address,
        memory: line.memory,
        stack: false,
    };
    let (load, second) = lsu.admit_load(0, operands, mapped, false).unwrap().unwrap();
    assert_eq!(second, None);
    load_machine.set_xreg(5, 0).unwrap();
    assert_eq!(lsu.pending_load(load), Some(&operands));
    let mut read = None;
    let mut cache = C220DataCache::new(
        C220CacheAddressLayout::new(6, 0, 6, u64::MAX).unwrap(),
        64,
        vec![
            C220CacheSet::new(
                vec![C220CacheTag {
                    atomic: false,
                    valid: false,
                    dirty: false,
                    age: 0,
                    memory: C220LsuMemory::External,
                    tag: 0,
                }],
                None,
            )
            .unwrap(),
        ],
    )
    .unwrap();
    let cache_port = core
        .connect_cache_read_port(C220BiuReadCacheConfig {
            request_capacity: NonZeroU32::new(2).unwrap(),
            response_capacity: NonZeroU32::new(1).unwrap(),
            request_latency: 1,
            response_latency: 1,
        })
        .unwrap();
    let mut cache_tag = None;
    let mut cache_completed = false;
    assert!(
        matches!(core.step_word_at(0, word).unwrap(), C220CoreStep::Executed {
        instruction: C220CoreInstruction::Mte2Queued(issue), ..
    } if issue.instruction_id == id)
    );
    core.memory.write_unknown_at(source, 1).unwrap();
    core.memory.write_known_at(source + 1, &[5]).unwrap();
    let mut admitted = std::collections::BTreeSet::new();
    let mut admitted_bytes = 0;
    assert!(core.receive_mte2_biu_at(0, [None; 2]).is_err());
    for tick in 1..250 {
        core.advance_to(tick).unwrap();
        core.advance_to(tick).unwrap();
        let external = C220LsuExternalHazards {
            maintenance_active: false,
            maintenance_draining: false,
        };
        for stage in [C220LsuStage::M2, C220LsuStage::M1, C220LsuStage::M0] {
            lsu.advance_with_cache(stage, tick, external, &mut cache)
                .unwrap();
        }
        if let Some(request) = lsu
            .reads
            .dispatch_clock(tick, false, true)
            .unwrap()
            .first()
            .copied()
        {
            assert_eq!(tick, 4);
            read = Some(request.id);
            cache_tag = Some(C220MemoryReadId::DataCache {
                port: cache_port,
                transaction: request.id.sequence(),
            });
            assert!(core.send_cache_read_at(tick, cache_port, request).unwrap());
            core.memory
                .write_known_at(line.address, &[0x31; 64])
                .unwrap();
        }
        if let Some((completed, result, refill)) = core
            .commit_cache_read_at(tick, cache_port, &mut lsu, &mut cache)
            .unwrap()
        {
            assert_eq!(Some(completed), read);
            assert_eq!(result.notifications, [C220LsuCompletion::Load(load)]);
            assert_eq!(
                result.load_line,
                core.memory.read_known_at(line.address, 64).unwrap()
            );
            assert_eq!(cache.line(refill.location).unwrap(), result.load_line);
            assert_eq!(lsu.reads.outstanding(), 0);
            assert!(lsu.misses.entries().is_empty());
            let completed_lsu = lsu.clone();
            let values = lsu.take_load_values();
            assert_eq!(values.len(), 1);
            assert_eq!(values[0].request, load);
            assert_eq!(values[0].operands, operands);
            assert_eq!(values[0].tick, tick);
            assert_eq!(values[0].path, C220LsuLoadPath::Refill);
            assert_eq!(
                values[0].value,
                u64::from_le_bytes(result.load_line[8..16].try_into().unwrap())
            );
            assert_eq!(load_machine.xregs()[7], u64::MAX);
            for mode in [
                C220LoadCommitMode::DataBypass,
                C220LoadCommitMode::Retirement,
            ] {
                let mut machine = load_machine.clone();
                let mut commits = C220LsuCommitLane::new(mode);
                let mut completed_lsu = completed_lsu.clone();
                assert!(
                    completed_lsu
                        .deliver_values(tick, &mut commits, &mut machine)
                        .is_err()
                );
                commits
                    .issue(0, C220LoadId(7), operands, &mut machine)
                    .unwrap();
                commits.admit(0, C220LoadId(7), load, None).unwrap();
                assert_eq!(
                    completed_lsu
                        .deliver_values(tick, &mut commits, &mut machine)
                        .unwrap(),
                    1
                );
                assert_eq!(
                    completed_lsu
                        .deliver_values(tick, &mut commits, &mut machine)
                        .unwrap(),
                    0
                );
                assert_eq!(commits.next_retirement_tick(), Some(tick + 1));
                assert_eq!(
                    machine.xregs()[7],
                    if mode == C220LoadCommitMode::DataBypass {
                        values[0].value
                    } else {
                        u64::MAX
                    }
                );
                assert!(
                    commits
                        .retire_next_at(tick, &mut machine)
                        .unwrap()
                        .is_none()
                );
                let C220LsuRetirement::Load(retired) = commits
                    .retire_next_at(tick + 1, &mut machine)
                    .unwrap()
                    .unwrap()
                else {
                    panic!("load retirement");
                };
                assert_eq!(retired.register_value, values[0].value);
                assert_eq!(
                    retired.writeback_tick,
                    Some(if mode == C220LoadCommitMode::DataBypass {
                        tick
                    } else {
                        tick + 1
                    })
                );
                assert_eq!(commits.pending_destination(7), None);
                assert_eq!(commits.pending_count(), 0);
            }
            assert!(!cache_completed);
            cache_completed = true;
        }
        for admission in core
            .mte_pipeline
            .as_ref()
            .unwrap()
            .timed_memory()
            .unwrap()
            .read_admissions()
        {
            let request = admission.request;
            if Some(request.tag) == cache_tag {
                continue;
            }
            assert!(matches!(
                request.tag,
                crate::sim::c220::memory::timed_memory::C220MemoryReadId::Mte(_)
            ));
            assert!(admission.tick <= tick);
            if admitted.insert((admission.tick, request.tag)) {
                admitted_bytes += request.bytes as usize;
                core.memory
                    .write_known_at(request.address, &vec![9; request.bytes as usize])
                    .unwrap();
                core.memory.write_unknown_at(source + 2, 1).unwrap();
            }
        }
        assert!(core.take_mte2_biu_request().is_none());
        if !core.mte2_is_busy() {
            break;
        }
        assert_eq!(core.state.ub().read_states(0, 8192).unwrap(), initial_ub);
    }
    assert!(!core.mte2_is_busy());
    assert!(core.mte_pipeline().unwrap().is_idle());
    assert_eq!(
        core.mte_pipeline()
            .unwrap()
            .biu_bus_reads()
            .unwrap()
            .outstanding(),
        0
    );
    assert_eq!(admitted_bytes, plan.bytes);
    assert!(cache_completed);
    for path in [
        C220LsuLoadPath::Cache,
        C220LsuLoadPath::CacheAndStore,
        C220LsuLoadPath::StoreForward,
    ] {
        let tick = lsu.reads.tick() + 1;
        let mut mapped = mapped;
        if path == C220LsuLoadPath::StoreForward {
            mapped.address += 64;
        }
        load_machine.set_xreg(7, mapped.address).unwrap();
        let operands = C220LoadOperands::capture(
            &load_machine,
            0x4004,
            (19 << 24) | (3 << 22) | (7 << 17) | (7 << 12) | 8,
        )
        .unwrap();
        let key = C220LsuLineKey {
            address: mapped.address / 64 * 64,
            memory: mapped.memory,
        };
        let (request, second) = lsu
            .admit_load(tick, operands, mapped, false)
            .unwrap()
            .unwrap();
        assert_eq!(second, None);
        let mut expected = if path == C220LsuLoadPath::StoreForward {
            [0xbb; 8]
        } else {
            let location = cache.find_way(line.address, line.memory).unwrap();
            cache.line(location).unwrap()[8..16].try_into().unwrap()
        };
        if path == C220LsuLoadPath::CacheAndStore {
            lsu.stores.store(key, load, 11, &[0xaa], true).unwrap();
            expected[0] = 0;
        } else if path == C220LsuLoadPath::StoreForward {
            lsu.stores.store(key, load, 8, &[0xbb; 8], false).unwrap();
        }
        let external = C220LsuExternalHazards {
            maintenance_active: false,
            maintenance_draining: false,
        };
        for delta in 1..=3 {
            for stage in [C220LsuStage::M2, C220LsuStage::M1, C220LsuStage::M0] {
                lsu.advance_with_cache(stage, tick + delta, external, &mut cache)
                    .unwrap();
            }
            if delta < 3 {
                assert!(lsu.take_load_values().is_empty());
            }
        }
        let values = lsu.take_load_values();
        assert_eq!(values.len(), 1);
        assert_eq!(values[0].request, request);
        assert_eq!(values[0].path, path);
        assert_eq!(values[0].tick, tick + 3);
        assert_eq!(values[0].value, u64::from_le_bytes(expected));
        assert_eq!(lsu.reads.requests().count(), 0);
        for mode in [
            C220LoadCommitMode::DataBypass,
            C220LoadCommitMode::Retirement,
        ] {
            for (suppressed, second_register) in [
                (false, None),
                (true, None),
                (false, Some(8)),
                (true, Some(8)),
                (false, Some(7)),
                (true, Some(7)),
            ] {
                let mut machine = load_machine.clone();
                let prior_second = machine.xregs()[usize::from(second_register.unwrap_or(8))];
                let operands = C220LoadOperands {
                    second_destination: second_register.map(|register| (register, prior_second)),
                    ..operands
                };
                let data = crate::sim::c220::scalar::lsu::scheduler::C220LsuLoadValue {
                    operands,
                    second_value: second_register.map(|_| 0x5678),
                    ..values[0]
                };
                let mut commits = C220LsuCommitLane::new(mode);
                commits
                    .issue(tick, C220LoadId(7), operands, &mut machine)
                    .unwrap();
                assert_eq!(machine.xregs()[7], mapped.address + 8);
                assert_eq!(commits.pending_destination(7), Some(C220LoadId(7)));
                if suppressed && mode == C220LoadCommitMode::DataBypass {
                    commits.supersede(7);
                    machine.set_xreg(7, 0x1234).unwrap();
                }
                commits
                    .admit(tick + 1, C220LoadId(7), request, None)
                    .unwrap();
                assert!(
                    commits
                        .admit(tick + 1, C220LoadId(7), request, None)
                        .is_err()
                );
                commits
                    .complete_data_at(tick + 3, data, &mut machine)
                    .unwrap();
                assert!(
                    commits
                        .complete_data_at(tick + 3, data, &mut machine)
                        .is_err()
                );
                if suppressed && mode == C220LoadCommitMode::Retirement {
                    commits.supersede(7);
                    machine.set_xreg(7, 0x1234).unwrap();
                }
                assert!(
                    commits
                        .retire_next_at(tick + 3, &mut machine)
                        .unwrap()
                        .is_none()
                );
                let C220LsuRetirement::Load(retired) = commits
                    .retire_next_at(tick + 4, &mut machine)
                    .unwrap()
                    .unwrap()
                else {
                    panic!("load retirement");
                };
                assert_eq!(retired.data, data);
                assert_eq!(retired.instruction, C220LoadId(7));
                assert_eq!(retired.admission_tick, tick + 1);
                assert_eq!(retired.suppressed, suppressed);
                assert_eq!(
                    retired.register_value,
                    if suppressed { 0x1234 } else { values[0].value }
                );
                assert_eq!(retired.writeback_tick.is_none(), suppressed);
                assert_eq!(commits.pending_count(), 0);
                assert_eq!(commits.pending_destination(7), None);
                if let Some(register) = second_register {
                    assert_eq!(
                        retired.second_register_value,
                        Some(if register == 7 {
                            retired.register_value
                        } else if suppressed {
                            prior_second
                        } else {
                            0x5678
                        })
                    );
                    assert_eq!(
                        commits.pending_destination(register),
                        (suppressed && register != 7).then_some(C220LoadId(7))
                    );
                    commits.supersede(register);
                    assert_eq!(commits.pending_destination(register), None);
                }
            }
        }
    }
    for segment in plan.descriptor_segments().unwrap() {
        let expected: Vec<_> = (0..segment.bytes)
            .map(
                |offset| match segment.source_hbm + u64::from(offset) - source {
                    2 => MemoryByteState::Unknown,
                    _ => MemoryByteState::Known(9),
                },
            )
            .collect();
        assert_eq!(
            core.state
                .ub()
                .read_states(segment.destination_local, segment.bytes as usize)
                .unwrap(),
            expected
        );
    }
}

#[test]
fn biu_tags_backpressure_dma_without_retiring_on_request_delivery() {
    for bus_connected in [false, true] {
        run_biu_dma(bus_connected);
    }
}

fn run_biu_dma(bus_connected: bool) {
    let mut core = configured_dma_core();
    let operands = C220MovInstruction::decode(CAPTURED_C220_MOV_OUT_TO_UB_X_WORD).unwrap();
    core.state
        .scalar_mut()
        .machine_mut()
        .set_xreg(operands.descriptor_register, (128 << 16) | (1 << 4) | 13)
        .unwrap();
    core.connect_mte2_biu(
        C220BiuReadConfig {
            outstanding: NonZeroU32::new(2).unwrap(),
            weights: [1; 3],
            group_vector_returns: true,
            write_bandwidths:
                crate::sim::c220::mte::interface::biu_read::write::C220BiuWriteBandwidths {
                    l1: NonZeroU32::new(128).unwrap(),
                    l0a: NonZeroU32::new(128).unwrap(),
                    l0b: NonZeroU32::new(128).unwrap(),
                    ub: NonZeroU32::new(128).unwrap(),
                },
        },
        C220BiuSubcore::Vector0,
    )
    .unwrap();
    if bus_connected {
        core.mte_pipeline
            .as_mut()
            .unwrap()
            .connect_biu_bus_reads(NonZeroU32::new(1).unwrap())
            .unwrap();
    }
    let C220CoreStep::Executed {
        instruction: C220CoreInstruction::Mte2Queued(issue),
        ..
    } = core
        .step_word_at(0, CAPTURED_C220_MOV_OUT_TO_UB_X_WORD)
        .unwrap()
    else {
        panic!("DMA issue expected");
    };
    core.advance_to(11).unwrap();
    assert!(core.take_mte2_biu_request().is_none());
    core.advance_to(12).unwrap();
    assert!(
        core.receive_mte2_biu_at(
            12,
            [
                Some(C220BiuReadBeat {
                    tag: NonZeroU32::new(1).unwrap(),
                    transaction_id: 0
                }),
                None
            ]
        )
        .is_err()
    );
    let first_tick = if bus_connected { 14 } else { 12 };
    core.advance_to(first_tick).unwrap();
    let first = core.take_mte2_biu_request().unwrap();
    assert_eq!(first.input.generated.sid, Some(13));
    core.advance_to(first_tick + 1).unwrap();
    let second = core.take_mte2_biu_request().unwrap();
    assert_eq!(second.input.generated.sid, Some(13));
    core.advance_to(20).unwrap();
    assert!(core.take_mte2_biu_request().is_none());
    assert!(core.take_mte2_dma_request().is_none());
    assert_eq!(
        core.mte_pipeline()
            .unwrap()
            .biu_read()
            .unwrap()
            .free_tag_count(),
        0
    );
    assert!(core.complete_mte2_dma_at(20, issue.instruction_id).is_err());
    let mut requests = vec![first, second];
    let mut beats: std::collections::VecDeque<_> = requests
        .iter()
        .flat_map(|request| {
            (0..4).rev().map(move |transaction_id| C220BiuReadBeat {
                tag: request.tag,
                transaction_id,
            })
        })
        .collect();
    let mut outputs = Vec::new();
    let mut completed_at = None;
    for tick in 21..250 {
        core.advance_to(tick).unwrap();
        if let Some(request) = core.take_mte2_biu_request() {
            requests.push(request);
            assert_eq!(request.input.generated.sid, Some(13));
            beats.extend((0..4).rev().map(|transaction_id| C220BiuReadBeat {
                tag: request.tag,
                transaction_id,
            }));
        }
        let heads = [beats.front().copied(), None];
        if core.receive_mte2_biu_at(tick, heads).unwrap()[0] {
            beats.pop_front();
        }
        for event in core.mte_pipeline().unwrap().last_events() {
            use crate::sim::c220::mte::C220MtePipelineEvent;
            match event {
                C220MtePipelineEvent::UbRequest(C220BiuSubcore::Vector0, request) => {
                    assert_eq!(request.ready_tick, request.sent_tick + 1);
                    assert!(request.ready_tick <= tick);
                    outputs.push(request.fragment);
                }
                C220MtePipelineEvent::UbResponse(C220BiuSubcore::Vector0, request)
                    if request.fragment.last_in_instruction =>
                {
                    completed_at = Some(tick);
                }
                _ => {}
            }
        }
        if completed_at.is_some() {
            assert!(beats.is_empty());
            assert_eq!(outputs.len(), 32);
            assert!(core.state.ub().read_known(0, 32).is_err());
            break;
        }
    }
    assert_eq!(requests.len(), 8);
    assert_eq!(
        outputs
            .iter()
            .map(|output| u64::from(output.logical_bytes))
            .sum::<u64>(),
        4096
    );
    for (index, fragment) in outputs.iter().enumerate() {
        assert_eq!(fragment.destination_address, index as u64 * 128);
        assert_eq!(fragment.bytes, 128);
    }
    assert_eq!(
        outputs
            .iter()
            .filter(|output| output.last_in_instruction)
            .count(),
        1
    );
    assert!(outputs.last().unwrap().last_in_instruction);
    for (index, request) in requests.iter().enumerate() {
        let generated = request.input.generated;
        assert_eq!(generated.instruction_id, issue.instruction_id);
        assert_eq!(
            generated.request.source_address,
            0x2000 + index as u64 * 512
        );
        assert_eq!(generated.request.bytes, 512);
        assert_eq!(request.byte_offset, 0);
    }
    core.advance_to(completed_at.unwrap() + 1).unwrap();
    assert!(core.state.ub().read_known(0, 32).is_err());
    assert!(core.mte2_is_busy());
    // Reentering the same tick must not apply the acknowledgment twice.
    core.advance_to(completed_at.unwrap() + 1).unwrap();
    core.advance_to(completed_at.unwrap() + 2).unwrap();
    assert_eq!(core.state.ub().read_known(0, 4096).unwrap(), vec![7; 4096]);
    assert!(!core.mte2_is_busy());
    assert!(core.mte_pipeline().unwrap().is_idle());
}

#[test]
fn mte3_queue_and_barriers_order_shared_events_after_destination_acknowledgment() {
    use crate::isa::c220::mte::CAPTURED_C220_MOV_UB_TO_OUT_WORD;
    use crate::memory::sparse::MemoryByteState;

    for source in [1, 2] {
        let mut core = configured_dma_core();
        core.connect_mte3_dma().unwrap();
        core.mte3_issue_queue.config.vector_depth = NonZeroU32::new(2).unwrap();
        core.state
            .ub
            .write_states(0, &vec![MemoryByteState::Known(9); 128])
            .unwrap();
        let word = CAPTURED_C220_MOV_UB_TO_OUT_WORD;
        let operands = C220MovInstruction::decode(word).unwrap();
        let machine = core.state.scalar_mut().machine_mut();
        machine.set_xreg(operands.source_register, 0).unwrap();
        machine
            .set_xreg(operands.destination_register, 0x2000)
            .unwrap();
        machine
            .set_xreg(operands.descriptor_register, (4 << 16) | (1 << 4))
            .unwrap();
        machine.set_xreg(12, 0x1234_5678).unwrap();
        let flag = |op: u32, src: u32, dest: u32| {
            (2 << 29) | (op << 21) | (1 << 17) | (src << 10) | (dest << 7) | (12 << 2)
        };
        core.step_word_at(0, flag(6, source, 5)).unwrap();
        let dma_id = core.next_instruction_id();
        core.step_word_at(1, word).unwrap();
        let pc = core.state.scalar().pc();
        assert!(matches!(core.step_word_at(2, 0x40e0_1400).unwrap(),
            C220CoreStep::Stalled(stall) if stall.cause == C220StallCause::Mte3IssueQueueFull));
        assert_eq!(core.state.scalar().pc(), pc);
        assert!(core.connect_mte3_dma().is_err());
        assert!(matches!(
            core.step_word_at(3, 0x4140_0000).unwrap(),
            C220CoreStep::Executed { .. }
        ));
        core.step_word_at(4, flag(5, source, 5)).unwrap();
        let machine = core.state.scalar_mut().machine_mut();
        machine.set_xreg(operands.destination_register, 0).unwrap();
        machine.set_xreg(operands.descriptor_register, 0).unwrap();
        assert!(matches!(
            core.step_word_at(7, 0x40e0_1400).unwrap(),
            C220CoreStep::Executed {
                instruction: C220CoreInstruction::Mte3Barrier {
                    completed_tick: None,
                    ..
                },
                ..
            }
        ));
        core.step_word_at(8, flag(5, 5, 4)).unwrap();
        core.step_word_at(9, flag(6, 5, 4)).unwrap();
        let machine = core.state.scalar_mut().machine_mut();
        machine.set_xreg(1, 0).unwrap();
        machine.set_xreg(3, 1 | (2 << 16)).unwrap();
        machine.set_spr_value(15, 0x4321_4321).unwrap();
        let fill = (3 << 29) | (1 << 22) | (1 << 17) | (3 << 7) | 6;
        core.step_word_at(10, fill).unwrap();
        let mut requests = Vec::new();
        for tick in 11..60 {
            core.advance_to(tick).unwrap();
            if let Some(request) = core.take_mte3_dma_request() {
                requests.push(request);
            }
        }
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].instruction_id, dma_id);
        assert!(requests[0].last_in_instruction);
        assert_eq!(core.queued_mte3_instructions().len(), 1);
        assert_eq!(core.queued_mte2_instructions().len(), 2);
        assert_eq!(core.pending_mte3_barriers().len(), 1);
        assert_eq!(
            core.memory().read_known_at(0x2000, 128).unwrap(),
            vec![7; 128]
        );
        core.acknowledge_mte3_dma_at(60, dma_id, requests[0].uop_index)
            .unwrap();
        core.advance_to(61).unwrap();
        assert_eq!(core.pending_mte3_barriers().len(), 0);
        assert_eq!(
            core.memory().read_known_at(0x2000, 128).unwrap(),
            vec![9; 128]
        );
        core.advance_to(100).unwrap();
        assert!(!core.mte3_is_busy());
        assert!(!core.mte2_is_busy());
        assert_eq!(
            core.local_memory.l1().read_known(0, 64).unwrap(),
            0x4321_4321_u32.to_le_bytes().repeat(16)
        );
        assert!(matches!(
            core.step_word_at(100, 0x40e0_1400).unwrap(),
            C220CoreStep::Executed {
                instruction: C220CoreInstruction::Mte3Barrier {
                    completed_tick: Some(100),
                    ..
                },
                ..
            }
        ));
    }
}

#[test]
fn mte2_barrier_releases_on_destination_retirement_without_blocking_scalar() {
    for flag_predecessor in [false, true] {
        let mut core = configured_dma_core();
        core.connect_mte2_dma().unwrap();
        if flag_predecessor {
            core.mte2_frontend.config.outstanding_limit = NonZeroU32::new(1).unwrap();
        }
        let dma_id = core.next_instruction_id();
        core.step_word_at(0, CAPTURED_C220_MOV_OUT_TO_UB_X_WORD)
            .unwrap();
        if flag_predecessor {
            let set = (2 << 29) | (5 << 21) | (4 << 10) | (3 << 7) | 3;
            core.step_word_at(1, set).unwrap();
        }
        for tick in 2..4 {
            assert!(matches!(
                core.step_word_at(tick, 0x40e0_1000).unwrap(),
                C220CoreStep::Executed {
                    instruction: C220CoreInstruction::Mte2Barrier {
                        completed_tick: None,
                        ..
                    },
                    ..
                }
            ));
        }
        assert_eq!(core.pending_mte2_barriers().len(), 2);
        assert_eq!(core.outstanding_mte2_commands(), 1);
        assert_eq!(
            core.pending_mte2_barriers().next().unwrap().requires_idle,
            flag_predecessor
        );
        let machine = core.state.scalar_mut().machine_mut();
        machine.set_xreg(1, 0).unwrap();
        machine.set_xreg(3, 1 | (2 << 16)).unwrap();
        machine.set_spr_value(15, 0x1234_5678).unwrap();
        let fill = (3 << 29) | (1 << 22) | (1 << 17) | (3 << 7) | 6;
        core.step_word_at(4, fill).unwrap();
        assert!(matches!(
            core.step_word_at(5, 0x4140_0000).unwrap(),
            C220CoreStep::Executed { .. }
        ));
        let mut tail_delivered = false;
        for tick in 6..60 {
            core.advance_to(tick).unwrap();
            if let Some(request) = core.take_mte2_dma_request() {
                tail_delivered |= request.last_in_instruction;
            }
        }
        assert!(tail_delivered);
        assert_eq!(core.pending_mte2_barriers().len(), 2);
        assert_eq!(core.outstanding_mte2_commands(), 1);
        assert_eq!(core.queued_mte2_commands().len(), 0);
        assert!(core.local_memory.l1().read_known(0, 64).is_err());
        core.complete_mte2_dma_at(60, dma_id).unwrap();
        assert_eq!(core.pending_mte2_barriers().len(), 2);
        core.advance_to(61).unwrap();
        assert_eq!(core.pending_mte2_barriers().len(), 0);
        assert_eq!(
            core.mte2_frontend_outcomes()
                .iter()
                .filter(|outcome| matches!(
                    outcome,
                    C220CoreStep::Executed {
                        instruction: C220CoreInstruction::Mte2Barrier {
                            completed_tick: Some(61),
                            ..
                        },
                        ..
                    }
                ))
                .count(),
            2
        );
        core.advance_to(100).unwrap();
        assert!(!core.mte2_is_busy());
        assert_eq!(core.state.ub().read_known(0, 4096).unwrap(), vec![7; 4096]);
        assert_eq!(
            core.local_memory.l1().read_known(0, 64).unwrap(),
            0x1234_5678_u32.to_le_bytes().repeat(16)
        );
        assert!(matches!(
            core.step_word_at(100, 0x40e0_1000).unwrap(),
            C220CoreStep::Executed {
                instruction: C220CoreInstruction::Mte2Barrier {
                    completed_tick: Some(100),
                    ..
                },
                ..
            }
        ));
    }
}

#[test]
fn dma_credit_stalls_generation_but_destination_response_controls_retirement() {
    let mut core = configured_dma_core();
    let word = CAPTURED_C220_MOV_OUT_TO_UB_X_WORD;
    core.connect_mte2_dma().unwrap();
    core.set_mte2_dma_hardware_sync_blocked(true).unwrap();
    let C220CoreStep::Executed {
        instruction: C220CoreInstruction::Mte2Queued(issue),
        ..
    } = core.step_word_at(0, word).unwrap()
    else {
        panic!("DMA issue expected");
    };
    assert_eq!(issue.ready_tick, 1);
    assert!(core.connect_mte2_dma().is_err());
    core.advance_to(12).unwrap();
    let generator = core.mte_pipeline().unwrap().dma_generator();
    assert_eq!(generator.generated().len(), 4);
    assert_eq!(generator.generated().front().unwrap().ready_tick, 8);
    assert_eq!(generator.pending_instruction(), Some(issue.instruction_id));
    assert!(core.take_mte2_dma_request().is_none());
    assert!(core.complete_mte2_dma_at(12, issue.instruction_id).is_err());
    core.set_mte2_dma_hardware_sync_blocked(false).unwrap();
    core.advance_to(13).unwrap();
    let offered = core.mte_pipeline().unwrap().dma_output().unwrap();
    core.advance_to(20).unwrap();
    assert_eq!(core.mte_pipeline().unwrap().dma_output(), Some(offered));
    assert_eq!(
        core.mte_pipeline()
            .unwrap()
            .dma_generator()
            .generated()
            .len(),
        4
    );

    let fill_word = (3 << 29) | (1 << 22) | (1 << 17) | (3 << 7) | 6;
    let machine = core.state.scalar_mut().machine_mut();
    machine.set_xreg(1, 0).unwrap();
    machine.set_xreg(3, 1 | (2 << 16)).unwrap();
    machine.set_spr_value(15, 0x1234_5678).unwrap();
    let pc = core.state.scalar().pc();
    assert!(matches!(
        core.step_word_at(20, fill_word).unwrap(),
        C220CoreStep::Executed { .. }
    ));
    assert_eq!(core.state.scalar().pc(), pc + 4);

    let mut requests = Vec::new();
    let mut tail_tick = None;
    for tick in 20..40 {
        core.advance_to(tick).unwrap();
        if let Some(request) = core.take_mte2_dma_request() {
            requests.push(request);
            if request.last_in_instruction {
                tail_tick = Some(tick);
                break;
            }
        }
    }
    let tail_tick = tail_tick.expect("request tail");
    assert_eq!(requests.len(), 8);
    for (index, request) in requests.iter().enumerate() {
        assert_eq!(request.instruction_id, issue.instruction_id);
        assert_eq!(request.uop_index, index as u64);
        assert_eq!(request.request.source_address, 0x2000 + index as u64 * 512);
        assert_eq!(request.request.bytes, 512);
    }
    assert!(core.mte_pipeline().unwrap().dma_generator().is_idle());
    assert_eq!(
        core.mte2.pending_commands().next().unwrap().completion,
        C220Mte2Completion::AwaitingDma {
            tail_delivered: true
        }
    );
    assert!(core.state.ub().read_known(0, 32).is_err());
    let set = (2 << 29) | (5 << 21) | (4 << 10) | 3;
    let wait = (set & !(15 << 21)) | (6 << 21);
    core.step_word_at(tail_tick + 2, set).unwrap();
    core.advance_to(60).unwrap();
    assert!(matches!(
        core.step_word_at(60, wait).unwrap(),
        C220CoreStep::Stalled(_)
    ));
    assert!(core.state.ub().read_known(0, 32).is_err());
    assert!(core.local_memory.l1().read_known(0, 32).is_err());
    assert_eq!(core.mte2.pending_commands().count(), 2);
    core.complete_mte2_dma_at(60, issue.instruction_id).unwrap();
    assert!(core.complete_mte2_dma_at(60, issue.instruction_id).is_err());
    core.advance_to(62).unwrap();
    let outcomes = core.mte2.last_outcomes();
    assert_eq!(outcomes.len(), 2);
    assert_eq!(outcomes[0].retire_tick, 61);
    assert_eq!(outcomes[1].retire_tick, 62);
    assert_eq!(core.state.ub().read_known(0, 4096).unwrap(), vec![7; 4096]);
    assert_eq!(
        core.local_memory.l1().read_known(0, 64).unwrap(),
        0x1234_5678_u32.to_le_bytes().repeat(16)
    );
    assert!(matches!(
        core.step_word_at(62, wait).unwrap(),
        C220CoreStep::Executed { .. }
    ));
    assert!(!core.mte2_is_busy());
}
