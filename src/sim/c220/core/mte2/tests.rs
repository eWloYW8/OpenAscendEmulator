use super::*;
use crate::isa::c220::mte::{C220MovInstruction, CAPTURED_C220_MOV_OUT_TO_UB_X_WORD};
use crate::memory::{
    mapped::MappedMemory, region::MemoryRegion, sparse::SparseMemory, ub::UbMemory,
};
use crate::sim::c220::core::C220CoreTimingRules;
use crate::sim::c220::memory::l1::C220L1Geometry;
use crate::sim::c220::mte::C220MtePipelineConfig;
use crate::sim::c220::mte::mte1::frontend::C220Mte1ReadBandwidths;
use crate::sim::c220::mte::mte2::{C220Mte2Completion, C220Mte2IssueTiming, C220Mte2TimingRules};
use crate::sim::c220::mte::mte3::C220Mte3TimingRules;
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
        l1: C220L1Geometry::new(32, 4, 1, 0).unwrap(),
        read_width: width,
        output_bandwidths: C220Mte1ReadBandwidths {
            l0a: width,
            l0b: width,
            bt: width,
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
    core.step_word_at(0, word).unwrap();
    core.memory.write_unknown_at(source, 1).unwrap();
    core.memory.write_known_at(source + 1, &[5]).unwrap();
    let mut admitted = std::collections::BTreeSet::new();
    let mut admitted_bytes = 0;
    assert!(core.receive_mte2_biu_at(0, [None; 2]).is_err());
    for tick in 1..250 {
        core.advance_to(tick).unwrap();
        core.advance_to(tick).unwrap();
        for admission in core
            .mte_pipeline
            .as_ref()
            .unwrap()
            .timed_memory()
            .unwrap()
            .read_admissions()
        {
            let request = admission.request;
            assert_eq!(request.input.generated.instruction_id, id);
            assert!(admission.tick <= tick);
            if admitted.insert((request.input.generated.uop_index, request.byte_offset)) {
                let request = request.input.generated.request;
                admitted_bytes += request.bytes as usize;
                core.memory
                    .write_known_at(request.source_address, &vec![9; request.bytes as usize])
                    .unwrap();
                core.memory.write_unknown_at(source + 2, 1).unwrap();
            }
        }
        assert!(core.take_mte2_biu_request().is_none());
        if !core.mte2.is_busy() {
            break;
        }
        assert_eq!(core.state.ub().read_states(0, 8192).unwrap(), initial_ub);
    }
    assert!(!core.mte2.is_busy());
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
        instruction: C220CoreInstruction::Mte2(issue),
        ..
    } = core
        .step_word_at(0, CAPTURED_C220_MOV_OUT_TO_UB_X_WORD)
        .unwrap()
    else {
        panic!("DMA issue expected");
    };
    core.advance_to(7).unwrap();
    assert!(core.take_mte2_biu_request().is_none());
    core.advance_to(8).unwrap();
    assert!(
        core.receive_mte2_biu_at(
            8,
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
    let first_tick = if bus_connected { 10 } else { 8 };
    core.advance_to(first_tick).unwrap();
    let first = core.take_mte2_biu_request().unwrap();
    core.advance_to(first_tick + 1).unwrap();
    let second = core.take_mte2_biu_request().unwrap();
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
    assert!(core.mte2.is_busy());
    // Reentering the same tick must not apply the acknowledgment twice.
    core.advance_to(completed_at.unwrap() + 1).unwrap();
    core.advance_to(completed_at.unwrap() + 2).unwrap();
    assert_eq!(core.state.ub().read_known(0, 4096).unwrap(), vec![7; 4096]);
    assert!(!core.mte2.is_busy());
    assert!(core.mte_pipeline().unwrap().is_idle());
}

#[test]
fn dma_credit_stalls_generation_but_destination_response_controls_retirement() {
    let mut core = configured_dma_core();
    let word = CAPTURED_C220_MOV_OUT_TO_UB_X_WORD;
    core.connect_mte2_dma().unwrap();
    core.set_mte2_dma_hardware_sync_blocked(true).unwrap();
    let C220CoreStep::Executed {
        instruction: C220CoreInstruction::Mte2(issue),
        ..
    } = core.step_word_at(0, word).unwrap()
    else {
        panic!("DMA issue expected");
    };
    assert!(matches!(issue.timing, C220Mte2IssueTiming::Dma(_)));
    assert!(core.connect_mte2_dma().is_err());
    core.advance_to(8).unwrap();
    let generator = core.mte_pipeline().unwrap().dma_generator();
    assert_eq!(generator.generated().len(), 4);
    assert_eq!(generator.generated().front().unwrap().ready_tick, 4);
    assert_eq!(generator.pending_instruction(), Some(issue.instruction_id));
    assert!(core.take_mte2_dma_request().is_none());
    assert!(core.complete_mte2_dma_at(8, issue.instruction_id).is_err());
    core.set_mte2_dma_hardware_sync_blocked(false).unwrap();
    core.advance_to(9).unwrap();
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
        C220CoreStep::Stalled(_)
    ));
    assert_eq!(core.state.scalar().pc(), pc);

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
    assert!(matches!(
        core.step_word_at(tail_tick + 1, fill_word).unwrap(),
        C220CoreStep::Executed { .. }
    ));
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
    assert!(!core.mte2.is_busy());
}
