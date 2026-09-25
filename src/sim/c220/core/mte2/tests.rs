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
        core_kind: crate::sim::c220::device::C220CoreKind::Vector0,
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
        assert!(!core.mte2.is_busy());
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
            if !core.mte2.is_busy() {
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
        assert!(!core.mte2.is_busy());
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
    use crate::sim::c220::scalar::lsu::load_commit::{C220LoadCommitLane, C220LoadCommitMode};
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
    let load = lsu.admit_load(0, operands, mapped, false).unwrap().unwrap();
    load_machine.set_xreg(5, 0).unwrap();
    assert_eq!(lsu.pending_load(load), Some(&operands));
    let mut read = None;
    let mut cache = C220DataCache::new(
        C220CacheAddressLayout::new(6, 0, 6, u64::MAX).unwrap(),
        64,
        vec![
            C220CacheSet::new(
                vec![C220CacheTag {
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
        instruction: C220CoreInstruction::Mte2(issue), ..
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
                let mut commits = C220LoadCommitLane::new(mode);
                let mut completed_lsu = completed_lsu.clone();
                assert!(
                    completed_lsu
                        .deliver_load_values(tick, &mut commits, &mut machine)
                        .is_err()
                );
                commits.issue(0, load, operands, &mut machine).unwrap();
                assert_eq!(
                    completed_lsu
                        .deliver_load_values(tick, &mut commits, &mut machine)
                        .unwrap(),
                    1
                );
                assert_eq!(
                    completed_lsu
                        .deliver_load_values(tick, &mut commits, &mut machine)
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
                let retired = commits
                    .retire_next_at(tick + 1, &mut machine)
                    .unwrap()
                    .unwrap();
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
        let request = lsu
            .admit_load(tick, operands, mapped, false)
            .unwrap()
            .unwrap();
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
            for suppressed in [false, true] {
                let mut machine = load_machine.clone();
                let mut commits = C220LoadCommitLane::new(mode);
                commits
                    .issue(tick, request, operands, &mut machine)
                    .unwrap();
                assert_eq!(machine.xregs()[7], mapped.address + 8);
                assert_eq!(commits.pending_destination(7), Some(request));
                if suppressed && mode == C220LoadCommitMode::DataBypass {
                    commits.supersede(7);
                    machine.set_xreg(7, 0x1234).unwrap();
                }
                commits
                    .complete_data_at(tick + 3, values[0], &mut machine)
                    .unwrap();
                assert!(
                    commits
                        .complete_data_at(tick + 3, values[0], &mut machine)
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
                let retired = commits
                    .retire_next_at(tick + 4, &mut machine)
                    .unwrap()
                    .unwrap();
                assert_eq!(retired.data, values[0]);
                assert_eq!(retired.suppressed, suppressed);
                assert_eq!(
                    retired.register_value,
                    if suppressed { 0x1234 } else { values[0].value }
                );
                assert_eq!(retired.writeback_tick.is_none(), suppressed);
                assert_eq!(commits.pending_count(), 0);
                assert_eq!(commits.pending_destination(7), None);
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
