use super::*;
use crate::sim::c220::vector::C220VectorInstruction;

use crate::architecture::Architecture;
use crate::memory::mapped::MappedMemory;
use crate::sim::c220::mte::mte2::C220Mte2TimingRules;
use crate::sim::c220::mte::mte3::C220Mte3TimingRules;
use crate::sim::c220::schedule::{C220Stall, C220StallCause};
use crate::sim::c220::state::C220State;
use crate::sim::c220::vector::pipeline::{C220VectorPipelineError, C220VectorTimingRules};
use crate::sim::c220::vector::timing::C220VectorUopKind;
use std::num::NonZeroU64;

use crate::isa::c220::mte::CAPTURED_C220_MOV_UB_TO_OUT_WORD;
use crate::memory::region::MemoryRegion;
use crate::memory::sparse::{MemoryByteState, SparseMemory};
use crate::memory::ub::UbMemory;
use crate::sim::c220::numeric::fp16::C220Fp16Mode;
use crate::sim::c220::vector::{
    C220_CAPTURED_MOVEV_CONTROL, C220_CAPTURED_MOVEV_WORD, C220_CAPTURED_VADD_CONTROL,
    C220_CAPTURED_VADD_WORD,
};
use crate::sim::common::scalar::ScalarMachine;
use crate::sim::common::scalar::ScalarStepper;

const C220_VECTOR_TO_MTE3_SET_FLAG_WORD: u32 = 0x40a2_06b8;
const C220_VECTOR_TO_MTE3_WAIT_FLAG_WORD: u32 = 0x40c2_06b4;
const C220_MTE3_TO_VECTOR_SET_FLAG_WORD: u32 = 0x40a2_14a8;
const C220_MTE3_TO_VECTOR_WAIT_FLAG_WORD: u32 = 0x40c2_14cc;

fn connect_test_read_bus(core: &mut C220Core) {
    use crate::sim::c220::mte::interface::biu_read::write::C220BiuWriteBandwidths;
    use crate::sim::c220::mte::interface::biu_read::{C220BiuReadConfig, C220BiuSubcore};
    use std::num::NonZeroU32;
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
        NonZeroU32::new(2).unwrap(),
    )
    .unwrap();
}

fn run_core_loads(core: &mut C220Core, mut tick: u64, bypass: bool) {
    use crate::sim::c220::memory::biu_read::C220BiuReadCacheConfig;
    use crate::sim::c220::scalar::lsu::cache::*;
    use crate::sim::c220::scalar::lsu::commit::C220LoadCommitMode;
    use crate::sim::c220::scalar::lsu::store_buffer::C220LsuMemory;
    use std::num::NonZeroU32;
    let cache = C220DataCache::new(
        C220CacheAddressLayout::new(6, 3, 8, 0xffffffffff).unwrap(),
        64,
        (0..4)
            .map(|_| {
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
                .unwrap()
            })
            .collect(),
    )
    .unwrap();
    core.configure_lsu_cache(
        cache,
        if bypass {
            C220LoadCommitMode::DataBypass
        } else {
            C220LoadCommitMode::Retirement
        },
        C220BiuReadCacheConfig {
            request_capacity: NonZeroU32::new(2).unwrap(),
            response_capacity: NonZeroU32::new(2).unwrap(),
            request_latency: 1,
            response_latency: 1,
        },
    )
    .unwrap();
    let machine = core.state.scalar_mut().machine_mut();
    machine.set_xreg(5, 0x2080).unwrap();
    machine.set_xreg(7, u64::MAX).unwrap();
    let expected_second = u64::from_le_bytes(
        core.memory
            .read_known_at(0x2088, 8)
            .unwrap()
            .try_into()
            .unwrap(),
    );
    let pc = core.state.scalar().pc();
    let C220CoreStep::Executed {
        instruction: C220CoreInstruction::Load(issue),
        ..
    } = core
        .step_word_at(
            tick,
            (9 << 24) | (3 << 22) | (7 << 17) | (5 << 12) | (9 << 7),
        )
        .unwrap()
    else {
        panic!("timed load issue");
    };
    assert_eq!(core.state.scalar().pc(), pc + 4);
    assert_eq!(core.state.scalar().machine().xregs()[7], u64::MAX);
    assert_eq!(core.pending_load_instructions().count(), 1);
    assert_eq!(core.lsu_ingress_occupancy(), 1);
    assert!(core.take_lsu_admissions().is_empty());
    let read = 0x0200_0800 | (8 << 17) | (7 << 12);
    assert!(matches!(
        core.step_word_at(tick + 1, read).unwrap(),
        C220CoreStep::Stalled(_)
    ));
    let mut read_tick = None;
    let completion = (tick + 2..tick + 100)
        .find_map(|now| {
            if read_tick.is_none() {
                match core.step_word_at(now, read).unwrap() {
                    C220CoreStep::Executed { .. } => read_tick = Some(now),
                    C220CoreStep::Stalled(_) => assert_eq!(core.state.scalar().pc(), pc + 4),
                }
            } else {
                core.advance_to(now).unwrap();
            }
            core.take_load_completions().into_iter().next()
        })
        .expect("automatic load completion");
    assert_eq!(completion.issue, issue);
    assert_eq!(completion.retirement.admission_tick, issue.tick + 1);
    let admissions = core.take_lsu_admissions();
    assert_eq!(admissions.len(), 1);
    assert_eq!(admissions[0].instruction_id, issue.instruction_id);
    assert_eq!(admissions[0].request, completion.retirement.data.request);
    assert_eq!(completion.retirement.data.value, 0xab00_0000);
    assert_eq!(
        completion.retirement.data.second_value,
        Some(expected_second)
    );
    assert_eq!(core.state.scalar().machine().xregs()[9], expected_second);
    assert_eq!(
        completion.retirement.data.path,
        crate::sim::c220::scalar::lsu::scheduler::C220LsuLoadPath::Refill
    );
    assert_eq!(core.state.scalar().machine().xregs()[8], 0xab00_0000);
    assert_eq!(read_tick, completion.retirement.writeback_tick);
    assert_eq!(
        completion.retirement.retire_tick - read_tick.unwrap(),
        u64::from(bypass)
    );
    assert_eq!(core.pending_load_instructions().count(), 0);
    tick = completion.retirement.retire_tick + 2;
    for suppressed in [false, true] {
        core.state
            .scalar_mut()
            .machine_mut()
            .set_xreg(5, 0x2080)
            .unwrap();
        let word = (19 << 24) | (3 << 22) | (5 << 17) | (5 << 12) | 8;
        assert!(matches!(
            core.step_word_at(tick, word).unwrap(),
            C220CoreStep::Executed {
                instruction: C220CoreInstruction::Load(_),
                ..
            }
        ));
        assert_eq!(core.state.scalar().machine().xregs()[5], 0x2088);
        if suppressed {
            let add = (5 << 17) | (1 << 12) | (2 << 7) | 1;
            assert!(matches!(
                core.step_word_at(tick + 1, add).unwrap(),
                C220CoreStep::Executed { .. }
            ));
        }
        let prior = core.state.scalar().machine().xregs()[5];
        core.advance_to(tick + 5).unwrap();
        let completions = core.take_load_completions();
        assert_eq!(completions.len(), 1);
        assert_eq!(completions[0].retirement.suppressed, suppressed);
        assert_eq!(completions[0].retirement.data.tick, tick + 4);
        assert_eq!(
            core.state.scalar().machine().xregs()[5],
            if suppressed { prior } else { 0xab00_0000 }
        );
        tick += 6;
    }
    core.take_lsu_admissions();
    core.state
        .scalar_mut()
        .machine_mut()
        .set_xreg(5, 0x2080)
        .unwrap();
    let mut completed = Vec::new();
    let mut stalled = false;
    for _ in 0..12 {
        let pc = core.state.scalar().pc();
        loop {
            let step = core.step_word_at(tick, 0x03ce_5000).unwrap();
            completed.extend(core.take_load_completions());
            assert!(core.lsu_ingress_occupancy() <= 2);
            tick += 1;
            match step {
                C220CoreStep::Executed { .. } => break,
                C220CoreStep::Stalled(stall) => {
                    assert_eq!(stall.cause, C220StallCause::LsuDependency);
                    assert_eq!(core.lsu_ingress_occupancy(), 2);
                    assert_eq!(core.state.scalar().pc(), pc);
                    stalled = true;
                }
            }
        }
    }
    assert!(stalled);
    core.state
        .scalar_mut()
        .machine_mut()
        .set_xreg(5, u64::MAX)
        .unwrap();
    core.advance_to(tick + 100).unwrap();
    completed.extend(core.take_load_completions());
    assert_eq!(completed.len(), 12);
    assert!(
        completed
            .iter()
            .all(|done| done.retirement.data.value == 0xab00_0000)
    );
    let admissions = core.take_lsu_admissions();
    assert_eq!(admissions.len(), 12);
    assert!(
        admissions
            .windows(2)
            .all(|pair| pair[0].tick < pair[1].tick)
    );
    assert!(
        completed
            .iter()
            .any(|done| done.retirement.admission_tick > done.issue.tick + 1)
    );
    assert_eq!(core.pending_load_instructions().count(), 0);
    assert_eq!(core.lsu_ingress_occupancy(), 0);
    run_core_stores(core, tick + 102);
}

fn run_core_stores(core: &mut C220Core, mut tick: u64) {
    use crate::sim::c220::scalar::lsu::scheduler::C220LsuStorePath;
    use crate::sim::c220::scalar::lsu::store_buffer::C220LsuMemory;
    for (address, path) in [
        (0x2080, C220LsuStorePath::Cache),
        (0x20c0, C220LsuStorePath::Refill),
    ] {
        if path == C220LsuStorePath::Refill {
            core.memory.write_known_at(address, &[0x55; 64]).unwrap();
        }
        let backing = core.memory.read_known_at(address, 64).unwrap();
        let machine = core.state.scalar_mut().machine_mut();
        machine.set_xreg(5, address).unwrap();
        machine.set_xreg(6, address).unwrap();
        machine.set_xreg(9, 0x1122_3344).unwrap();
        machine.set_xreg(10, 0xaabb_ccdd).unwrap();
        let (first_store, first_value) = if path == C220LsuStorePath::Refill {
            ((15 << 24) | (3 << 22) | (5 << 12) | (8 << 5) | 4 | 1, 1_u64)
        } else {
            (
                (20 << 24) | (3 << 22) | (9 << 17) | (5 << 12) | 8,
                0x1122_3344,
            )
        };
        let words = [
            first_store,
            (9 << 24) | (3 << 22) | (10 << 17) | (5 << 12) | (9 << 7) | 1,
            (9 << 24) | (3 << 22) | (7 << 17) | (6 << 12) | (8 << 7),
        ];
        let mut issues = Vec::new();
        let mut stores = Vec::new();
        let mut loads = Vec::new();
        for word in words {
            loop {
                let step = core.step_word_at(tick, word).unwrap();
                stores.extend(core.take_store_completions());
                loads.extend(core.take_load_completions());
                tick += 1;
                if let C220CoreStep::Executed { instruction, .. } = step {
                    if let C220CoreInstruction::Store(issue) = instruction {
                        issues.push(issue);
                        assert_eq!(core.state.scalar().machine().xregs()[5], address + 8);
                    }
                    break;
                }
            }
        }
        assert_eq!(issues.len(), 2);
        let machine = core.state.scalar_mut().machine_mut();
        machine.set_xreg(9, 0).unwrap();
        machine.set_xreg(10, 0).unwrap();
        assert!(matches!(
            core.step_word_at(tick, 0x40e0_1800).unwrap(),
            C220CoreStep::Stalled(_)
        ));
        for now in tick + 1..tick + 150 {
            core.advance_to(now).unwrap();
            stores.extend(core.take_store_completions());
            loads.extend(core.take_load_completions());
            if stores.len() == 2 && loads.len() == 1 {
                tick = now + 2;
                break;
            }
        }
        assert_eq!(stores.len(), 2);
        assert_eq!(loads.len(), 1);
        assert_eq!(stores[0].issue, issues[0]);
        assert_eq!(stores[1].issue, issues[1]);
        assert!(
            stores
                .iter()
                .all(|store| store.data.path == path && store.retire_tick > store.data.tick)
        );
        assert_eq!(stores[0].data.tick, stores[1].data.tick);
        assert_eq!(stores[1].retire_tick, stores[0].retire_tick + 1);
        assert_ne!(loads[0].retirement.retire_tick, stores[0].retire_tick);
        assert_ne!(loads[0].retirement.retire_tick, stores[1].retire_tick);
        assert_eq!(core.state.scalar().machine().xregs()[7], first_value);
        assert_eq!(core.state.scalar().machine().xregs()[8], 0xaabb_ccdd);
        assert_eq!(loads[0].retirement.data.second_value, Some(0xaabb_ccdd));
        assert_eq!(loads[0].retirement.second_register_value, Some(0xaabb_ccdd));
        let cache = core.data_cache().unwrap();
        let location = cache.find_way(address, C220LsuMemory::External).unwrap();
        let bytes = cache.line(location).unwrap();
        assert_eq!(&bytes[..8], &first_value.to_le_bytes());
        assert_eq!(&bytes[8..16], &0xaabb_ccdd_u64.to_le_bytes());
        assert_eq!(&bytes[16..24], &0x1122_3344_u64.to_le_bytes());
        assert_eq!(core.memory.read_known_at(address, 64).unwrap(), backing);
        assert_eq!(core.pending_store_instructions().count(), 0);
        assert_eq!(core.lsu_retirement_occupancy(), 0);
        assert!(matches!(
            core.step_word_at(tick, 0x40e0_1800).unwrap(),
            C220CoreStep::Executed { .. }
        ));
        tick += 2;
        for (offset, dtype) in [(56, 3), (61, 1)] {
            core.state
                .scalar_mut()
                .machine_mut()
                .set_xreg(5, address + offset)
                .unwrap();
            let pc = core.state.scalar().pc();
            let word = (9 << 24) | (dtype << 22) | (10 << 17) | (5 << 12) | (9 << 7) | 1;
            assert!(matches!(
                core.step_word_at(tick, word),
                Err(C220CoreError::UnsupportedTimedLsuAccess)
            ));
            assert_eq!(core.state.scalar().pc(), pc);
            assert_eq!(core.pending_store_instructions().count(), 0);
            tick += 1;
        }
    }
}

#[test]
fn native_mte3_waits_for_responses_and_reads_ub_at_retirement() {
    for mode in 0..4 {
        native_mte3_write_path(mode);
    }
}

fn native_mte3_write_path(mode: u8) {
    let bus = mode != 0;
    use crate::isa::c220::mte::C220MovInstruction;
    use crate::sim::c220::memory::biu_write::C220BiuWriteReturnKind::{Completion, Dbid};
    use crate::sim::c220::memory::l1::C220L1Geometry;
    use crate::sim::c220::mte::mte1::frontend::C220Mte1ReadBandwidths;
    use crate::sim::c220::mte::set2d::C220Set2dBandwidths;
    use std::num::NonZeroU32;

    let word = CAPTURED_C220_MOV_UB_TO_OUT_WORD;
    let operands = C220MovInstruction::decode(word).unwrap();
    let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
    machine.set_xreg(operands.source_register, 0).unwrap();
    machine
        .set_xreg(operands.destination_register, 0x2000)
        .unwrap();
    machine
        .set_xreg(operands.descriptor_register, (4 << 16) | (1 << 4))
        .unwrap();
    let one = NonZeroU64::new(1).unwrap();
    let mut core = C220Core::new(
        C220State::new(ScalarStepper::new(machine, 0), UbMemory::new(256, 256)),
        MappedMemory::bind(
            SparseMemory::new(vec![MemoryRegion::unknown(256)], 256, 256),
            &[0x2000],
        )
        .unwrap(),
        C220CoreTimingRules {
            mte2: C220Mte2TimingRules {
                issue_interval: one,
                startup_ticks: 0,
                bytes_per_tick: one,
                retire_ticks: 0,
            },
            mte3: C220Mte3TimingRules {
                issue_interval: one,
                startup_ticks: u64::MAX,
                bytes_per_tick: one,
                retire_ticks: u64::MAX,
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
    let config = crate::sim::c220::mte::interface::biu_write::command::C220BiuWriteConfig {
        outstanding: NonZeroU32::new(2).unwrap(),
        weights: [1; 3],
        source_bandwidth: width,
    };
    if bus {
        core.connect_mte3_bus(config, NonZeroU32::new(2).unwrap())
            .unwrap();
    } else {
        core.connect_mte3_biu(config).unwrap();
    }
    if mode >= 2 {
        connect_test_read_bus(&mut core);
        use crate::sim::c220::memory::timed_memory::{
            C220MemoryCredits, C220MemoryLatency, C220MemoryRegionTiming, C220TimedMemoryConfig,
        };
        let region = C220MemoryRegionTiming {
            read: C220MemoryLatency::fixed(5),
            dbid: C220MemoryLatency {
                minimum: 2,
                spread: 3,
            },
            completion: C220MemoryLatency {
                minimum: 3,
                spread: 4,
            },
        };
        core.configure_timed_memory(C220TimedMemoryConfig {
            input_capacity: NonZeroU32::new(2).unwrap(),
            pending_limit: NonZeroU32::new(4).unwrap(),
            credit_period: NonZeroU64::new(4).unwrap(),
            ddr_credits: C220MemoryCredits {
                limit: 257,
                refill: 64,
            },
            l2_read_credits: C220MemoryCredits {
                limit: 257,
                refill: 64,
            },
            l2_write_credits: C220MemoryCredits {
                limit: 257,
                refill: 64,
            },
            ddr: region,
            l2: region,
            l2_start: 0,
            l2_bytes: 0,
        })
        .unwrap();
    }
    if mode >= 2 {
        use crate::sim::c220::scalar::lsu::cache::C220CacheAddressLayout;
        use crate::sim::c220::scalar::lsu::miss_buffer::C220LsuMissConfig;
        use crate::sim::c220::scalar::lsu::store_buffer::C220LsuStoreConfig;
        core.configure_lsu(C220CoreLsuConfig {
            request_capacity: 2,
            read_capacity: 2,
            write_capacity: 2,
            direct_store_capacity: 2,
            misses: C220LsuMissConfig {
                line_bytes: 64,
                main_entries: 2,
                sub_entries: 2,
            },
            stores: C220LsuStoreConfig {
                line_bytes: 64,
                main_entries: 2,
                sub_entries: 2,
                timeout_ticks: 4,
            },
            layout: C220CacheAddressLayout::new(6, 3, 8, 0xffffffffff).unwrap(),
            partition_stack: false,
        })
        .unwrap();
    }
    assert!(matches!(
        core.step_word_at(0, word).unwrap(),
        C220CoreStep::Executed {
            instruction: C220CoreInstruction::Mte3Dma { .. },
            ..
        }
    ));
    core.state
        .scalar_mut()
        .machine_mut()
        .set_xreg(10, 0)
        .unwrap();
    core.state
        .scalar_mut()
        .machine_mut()
        .set_xreg(19, 0)
        .unwrap();
    core.step_word_at(1, C220_MTE3_TO_VECTOR_SET_FLAG_WORD)
        .unwrap();
    core.advance_to(6).unwrap();
    assert!(core.take_mte3_dma_request().is_none());
    core.advance_to(7).unwrap();
    assert!(core.take_mte3_dma_request().is_none());
    if mode >= 2 {
        assert!(core.take_biu_write_command_at(7).is_err());
        let machine = core.state.scalar_mut().machine_mut();
        machine.set_xreg(1, 0x2086).unwrap();
        machine.set_xreg(2, 0x12ab).unwrap();
        machine.set_spr_value(67, 0x2000000).unwrap();
        machine.set_spr_value(68, 0x2000000).unwrap();
        let pc = core.state.scalar().pc();
        let C220CoreStep::Executed {
            instruction: C220CoreInstruction::DirectStore(issue),
            ..
        } = core
            .step_word_at(7, 0x1800_0000 | (2 << 17) | (1 << 12) | 0xffd)
            .unwrap()
        else {
            panic!("direct store must enter the LSU");
        };
        assert_eq!(core.state.scalar().pc(), pc + 4);
        assert_eq!(core.pending_lsu_instructions().count(), 1);
        assert!(core.take_lsu_completions().is_empty());
        core.state
            .scalar_mut()
            .machine_mut()
            .set_xreg(2, 0)
            .unwrap();
        core.advance_to(11).unwrap();
        let admissions = core.take_lsu_admissions();
        assert_eq!(admissions.len(), 1);
        assert_eq!(admissions[0].instruction_id, issue.instruction_id);
        assert_eq!(admissions[0].tick, issue.tick + 1);
        let lsu = core.lsu_scheduler().unwrap();
        assert!(lsu.pending_direct_store(admissions[0].request).is_none());
        assert_eq!(lsu.pipeline().queued_requests(), 0);
        assert_eq!(lsu.direct_stores().entries().len(), 1);
        core.advance_to(11).unwrap();
        assert_eq!(
            core.lsu_scheduler()
                .unwrap()
                .direct_stores()
                .entries()
                .len(),
            1
        );
        assert!(core.memory().read_known_at(0x2080, 64).is_err());
        assert!(matches!(
            core.step_word_at(11, 0x40e0_1800).unwrap(),
            C220CoreStep::Stalled(_)
        ));
        assert_eq!(core.state.scalar().pc(), pc + 4);
        core.state
            .ub
            .write_states(0, &vec![MemoryByteState::Known(9); 128])
            .unwrap();
        let mut cache_done = false;
        let mut mte_done = false;
        let retired = (12..160)
            .find(|&tick| {
                core.advance_to(tick).unwrap();
                mte_done |= !core.last_mte3_dma_outcomes().is_empty();
                for completion in core.take_lsu_completions() {
                    assert_eq!(completion.issue, issue);
                    assert_eq!(completion.tick, tick);
                    assert_eq!(completion.tick, completion.response_tick + 1);
                    let mut expected = vec![0; 64];
                    expected[3] = 0xab;
                    assert_eq!(core.memory().read_known_at(0x2080, 64).unwrap(), expected);
                    assert!(
                        core.lsu_scheduler()
                            .unwrap()
                            .direct_stores()
                            .entries()
                            .is_empty()
                    );
                    assert_eq!(core.lsu_scheduler().unwrap().writes.outstanding(), 0);
                    assert_eq!(core.pending_lsu_instructions().count(), 0);
                    cache_done = true;
                }
                mte_done && cache_done
            })
            .expect("native memory service completes without externally injected responses");
        assert_eq!(
            core.memory().read_known_at(0x2000, 128).unwrap(),
            vec![9; 128]
        );
        assert!(matches!(
            core.step_word_at(retired + 1, C220_MTE3_TO_VECTOR_WAIT_FLAG_WORD)
                .unwrap(),
            C220CoreStep::Executed { .. }
        ));
        assert!(core.mte_pipeline().unwrap().is_idle());
        assert_eq!(
            core.mte_pipeline()
                .unwrap()
                .biu_bus_writes()
                .unwrap()
                .outstanding(),
            0
        );
        assert_eq!(
            core.mte_pipeline()
                .unwrap()
                .timed_memory()
                .unwrap()
                .pending_completions(),
            0
        );
        run_core_loads(&mut core, retired + 2, mode == 3);
        return;
    }
    let transfer = (8..30)
        .find_map(|tick| core.take_biu_write_command_at(tick).unwrap())
        .expect("MTE3 automatically enters BIU command transport");
    let request = transfer.command.input.generated;
    let tag = transfer.command.tag;
    let command_tick = transfer.ready_tick;
    assert_eq!(command_tick, if bus { 13 } else { 12 });
    assert!(request.last_in_instruction);
    assert!(
        core.register_mte3_biu_write_at(command_tick, transfer.command.source_request())
            .is_err()
    );
    assert!(
        core.receive_mte3_biu_write_response_at(command_tick, tag)
            .is_err()
    );
    if bus {
        assert!(core.receive_mte3_biu_dbid_at(command_tick, tag).is_err());
        assert!(
            core.receive_mte3_bus_return_at(command_tick, Completion, tag)
                .is_err()
        );
        assert!(
            core.receive_mte3_bus_return_at(command_tick, Dbid, tag)
                .unwrap()
        );
        assert!(
            core.receive_mte3_bus_return_at(command_tick, Dbid, tag)
                .is_err()
        );
        assert_eq!(
            core.mte_pipeline()
                .unwrap()
                .biu_bus_writes()
                .unwrap()
                .outstanding(),
            1
        );
    } else {
        core.receive_mte3_biu_dbid_at(command_tick, tag).unwrap();
    }
    assert!(
        core.acknowledge_mte3_dma_at(command_tick, request.instruction_id, request.uop_index)
            .is_err()
    );
    assert!(core.last_mte3_dma_outcomes().is_empty());
    assert!(matches!(
        core.step_word_at(command_tick + 1, C220_MTE3_TO_VECTOR_WAIT_FLAG_WORD)
            .unwrap(),
        C220CoreStep::Stalled(_)
    ));
    let data = (command_tick + 2..60)
        .find_map(|tick| core.take_biu_write_data_at(tick).unwrap())
        .expect("source packets reach the shared data port");
    assert_eq!(data.source.request.tag, tag);
    assert!(core.memory().read_known_at(0x2000, 128).is_err());
    assert!(core.last_mte3_dma_outcomes().is_empty());
    core.state
        .ub
        .write_states(0, &vec![MemoryByteState::Known(9); 128])
        .unwrap();
    let response_tick = data.ready_tick + 3;
    let retirement_tick = if bus {
        assert!(
            core.receive_mte3_bus_return_at(response_tick, Completion, tag)
                .unwrap()
        );
        assert_eq!(
            core.mte_pipeline()
                .unwrap()
                .biu_bus_writes()
                .unwrap()
                .outstanding(),
            1
        );
        core.advance_to(response_tick + 1).unwrap();
        assert_eq!(
            core.mte_pipeline()
                .unwrap()
                .biu_bus_writes()
                .unwrap()
                .outstanding(),
            0
        );
        assert!(core.memory().read_known_at(0x2000, 128).is_err());
        response_tick + 2
    } else {
        let response = core
            .receive_mte3_biu_write_response_at(response_tick, tag)
            .unwrap();
        assert_eq!(response.retired_instruction(), Some(request.instruction_id));
        response_tick + 1
    };
    core.advance_to(retirement_tick).unwrap();
    assert_eq!(
        core.memory().read_known_at(0x2000, 128).unwrap(),
        vec![9; 128]
    );
    assert_eq!(core.last_mte3_dma_outcomes().len(), 1);
    assert_eq!(core.last_mte3_dma_outcomes()[0].tick, retirement_tick);
    assert!(matches!(
        core.step_word_at(retirement_tick + 1, C220_MTE3_TO_VECTOR_WAIT_FLAG_WORD)
            .unwrap(),
        C220CoreStep::Executed { .. }
    ));
    assert!(core.mte_pipeline().unwrap().is_idle());
    assert_eq!(
        core.mte_pipeline()
            .unwrap()
            .biu_write_commands()
            .unwrap()
            .free_tag_count(),
        2
    );
}

#[test]
fn reduction_state_waits_for_both_repeats() {
    let memory = SparseMemory::new(vec![MemoryRegion::unknown(256)], 512, 512);
    let memory = MappedMemory::bind(memory, &[0x2000]).unwrap();
    let mut ub = UbMemory::new(4096, 256);
    for (address, value) in [(0x100, 1.0_f32), (0x200, 2.0_f32)] {
        ub.write_states(
            address,
            &value
                .to_le_bytes()
                .repeat(64)
                .into_iter()
                .map(MemoryByteState::Known)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    }
    let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
    machine.set_xreg(2, 0x100).unwrap();
    machine.set_xreg(3, 0x800).unwrap();
    machine
        .set_xreg(4, (2_u64 << 56) | 1 | (1 << 16) | (1 << 32) | (8 << 40))
        .unwrap();
    machine.set_spr_value(3, 0).unwrap();
    machine.set_spr_value(100, u64::MAX).unwrap();
    machine.set_spr_value(101, 0).unwrap();
    let execution = C220State::new(ScalarStepper::new(machine, 0x4000), ub);
    let rate = NonZeroU64::new(32).unwrap();
    let mut core = C220Core::new(
        execution,
        memory,
        C220CoreTimingRules {
            mte2: C220Mte2TimingRules {
                issue_interval: NonZeroU64::new(1).unwrap(),
                startup_ticks: 0,
                bytes_per_tick: rate,
                retire_ticks: 0,
            },
            mte3: C220Mte3TimingRules {
                issue_interval: NonZeroU64::new(1).unwrap(),
                startup_ticks: 0,
                bytes_per_tick: rate,
                retire_ticks: 0,
            },
            vector: C220VectorTimingRules {
                dispatch_ticks: 0,
                uop_issue_interval: NonZeroU64::new(1).unwrap(),
                ub_response_ticks: 1,
            },
        },
    )
    .unwrap();
    assert!(matches!(
        core.step_word_at(0, 0x83c6_2392).unwrap(),
        C220CoreStep::Executed {
            instruction: C220CoreInstruction::Vector(C220VectorInstruction::Reduction(_)),
            ..
        }
    ));
    assert!(matches!(
        core.step_word_at(1, 0x40a0_0400).unwrap(),
        C220CoreStep::Executed {
            instruction: C220CoreInstruction::VectorToScalarFlag(_),
            ..
        }
    ));
    let C220CoreStep::Stalled(stall) = core.step_word_at(2, 0x40c0_0400).unwrap() else {
        panic!("scalar wait should observe pending vector work");
    };
    assert!(matches!(
        core.step_word_at(stall.resume_tick, 0x40c0_0400).unwrap(),
        C220CoreStep::Executed {
            instruction: C220CoreInstruction::VectorToScalarFlag(_),
            ..
        }
    ));
    assert_eq!(
        core.state().scalar().machine().spr_value(87),
        Some(192.0_f32.to_bits().into())
    );
    let max_tick = stall.resume_tick + 1;
    assert!(matches!(
        core.step_word_at(max_tick, 0x83c6_2410).unwrap(),
        C220CoreStep::Executed {
            instruction: C220CoreInstruction::Vector(C220VectorInstruction::Reduction(_)),
            ..
        }
    ));
    core.advance_to(max_tick + 64).unwrap();
    assert_eq!(
        core.state().scalar().machine().spr_value(63),
        Some(u64::from(2.0_f32.to_bits()) | (127_u64 << 32))
    );
    assert_eq!(
        core.state().ub().read_known(0x820, 8).unwrap(),
        [2.0_f32.to_le_bytes(), 63_u32.to_le_bytes()].concat()
    );
}

#[test]
fn moveva_updates_only_its_selected_pair_and_retires_as_vector_work() {
    let memory = SparseMemory::new(vec![MemoryRegion::unknown(256)], 512, 512);
    let memory = MappedMemory::bind(memory, &[0x2000]).unwrap();
    let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
    machine.set_xreg(6, 0x120).unwrap();
    machine.set_xreg(7, 0x160).unwrap();
    let execution = C220State::new(ScalarStepper::new(machine, 0x4000), UbMemory::new(512, 256));
    let rate = NonZeroU64::new(32).unwrap();
    let mut core = C220Core::new(
        execution,
        memory,
        C220CoreTimingRules {
            mte2: C220Mte2TimingRules {
                issue_interval: NonZeroU64::new(1).unwrap(),
                startup_ticks: 0,
                bytes_per_tick: rate,
                retire_ticks: 0,
            },
            mte3: C220Mte3TimingRules {
                issue_interval: NonZeroU64::new(1).unwrap(),
                startup_ticks: 0,
                bytes_per_tick: rate,
                retire_ticks: 0,
            },
            vector: C220VectorTimingRules {
                dispatch_ticks: 0,
                uop_issue_interval: NonZeroU64::new(1).unwrap(),
                ub_response_ticks: 1,
            },
        },
    )
    .unwrap();
    let C220CoreStep::Executed {
        instruction: first @ C220CoreInstruction::Vector(C220VectorInstruction::MoveAddress { .. }),
        ..
    } = core.step_word_at(0, 0x8000_6380).unwrap()
    else {
        panic!("MOVEVA should issue");
    };
    assert_eq!(
        first.as_vector().unwrap().uops().unwrap()[0]
            .stages
            .execute_ticks,
        1
    );
    assert_eq!(core.va_registers().entry(0, 0), Some(9));
    assert_eq!(core.va_registers().entry(0, 1), Some(11));
    assert_eq!(core.va_registers().entry(0, 2), None);
    assert!(matches!(
        core.step_word_at(1, 0x8000_6390).unwrap(),
        C220CoreStep::Executed {
            instruction: C220CoreInstruction::Vector(C220VectorInstruction::MoveAddress { .. }),
            ..
        }
    ));
    assert_eq!(core.va_registers().entry(0, 2), Some(9));
    assert_eq!(core.va_registers().entry(0, 3), Some(11));
    core.advance_to(30).unwrap();
    assert_eq!(core.last_vector_releases().len(), 2);
    assert!(core.vector_pipeline().last_read_samples().is_empty());
}

#[test]
fn loadva_commits_at_execute_and_high_half_observes_the_ldvad_hazard() {
    let memory = SparseMemory::new(vec![MemoryRegion::unknown(256)], 512, 512);
    let memory = MappedMemory::bind(memory, &[0x2000]).unwrap();
    let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
    machine.set_xreg(1, 0x100).unwrap();
    machine.set_xreg(3, 0x1234_5678).unwrap();
    let mut ub = UbMemory::new(512, 256);
    let source = (0_u16..16)
        .flat_map(u16::to_le_bytes)
        .map(MemoryByteState::Known)
        .collect::<Vec<_>>();
    ub.write_states(0x100, &source).unwrap();
    let execution = C220State::new(ScalarStepper::new(machine, 0x4000), ub);
    let rate = NonZeroU64::new(32).unwrap();
    let mut core = C220Core::new(
        execution,
        memory,
        C220CoreTimingRules {
            mte2: C220Mte2TimingRules {
                issue_interval: NonZeroU64::new(1).unwrap(),
                startup_ticks: 0,
                bytes_per_tick: rate,
                retire_ticks: 0,
            },
            mte3: C220Mte3TimingRules {
                issue_interval: NonZeroU64::new(1).unwrap(),
                startup_ticks: 0,
                bytes_per_tick: rate,
                retire_ticks: 0,
            },
            vector: C220VectorTimingRules {
                dispatch_ticks: 0,
                uop_issue_interval: NonZeroU64::new(1).unwrap(),
                ub_response_ticks: 1,
            },
        },
    )
    .unwrap();

    assert!(matches!(
        core.step_word_at(0, 0x8080_1000).unwrap(),
        C220CoreStep::Executed {
            instruction: C220CoreInstruction::Vector(C220VectorInstruction::LoadAddress(_)),
            ..
        }
    ));
    assert_eq!(core.va_registers().entry(0, 0), None);
    let C220CoreStep::Stalled(stall) = core.step_word_at(1, 0x8082_1002).unwrap() else {
        panic!("high-half LD_VAD should wait for an in-flight LD_VAD");
    };
    assert!(matches!(
        core.step_word_at(stall.resume_tick, 0x8082_1002).unwrap(),
        C220CoreStep::Executed {
            instruction: C220CoreInstruction::Vector(C220VectorInstruction::LoadAddress(_)),
            ..
        }
    ));
    assert_eq!(core.va_registers().entry(0, 0), Some(0));
    assert_eq!(core.va_registers().entry(0, 7), Some(7));

    assert!(matches!(
        core.step_word_at(stall.resume_tick + 1, 0x8040_000c)
            .unwrap(),
        C220CoreStep::Executed {
            instruction: C220CoreInstruction::Vector(C220VectorInstruction::Movemask(_)),
            ..
        }
    ));
    assert_eq!(
        core.state().scalar().machine().spr_value(100),
        Some(0x1234_5678)
    );
    core.advance_to(stall.resume_tick + 16).unwrap();
    assert_eq!(core.va_registers().entry(1, 0), Some(8));
    assert_eq!(core.va_registers().entry(1, 7), Some(15));
}

#[test]
fn nchw_uses_va_rows_and_two_timed_uops_for_each_element_width() {
    for (opcode, width, source_high, destination_high) in [
        (0x8200_0680_u32, 1_usize, true, true),
        (0x8240_0680, 2, false, false),
        (0x8280_0680, 4, false, false),
    ] {
        let memory = SparseMemory::new(vec![MemoryRegion::unknown(256)], 512, 512);
        let memory = MappedMemory::bind(memory, &[0x2000]).unwrap();
        let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
        machine.set_xreg(5, (1_u64 << 56) | (1 << 16) | 1).unwrap();
        let mut ub = UbMemory::new(4096, 256);
        for row in 0..16 {
            let mut source = [0_u8; 32];
            for (column, chunk) in source.chunks_exact_mut(width).enumerate() {
                let value = (row * 100 + column + 1) as u32;
                chunk.copy_from_slice(&value.to_le_bytes()[..width]);
            }
            ub.write_states(
                0x200 + (row * 32) as u64,
                &source.map(MemoryByteState::Known),
            )
            .unwrap();
            ub.write_states(
                0x600 + (row * 32) as u64,
                &[MemoryByteState::Known(0xaa); 32],
            )
            .unwrap();
        }
        let execution = C220State::new(ScalarStepper::new(machine, 0x4000), ub);
        let rate = NonZeroU64::new(32).unwrap();
        let mut core = C220Core::new(
            execution,
            memory,
            C220CoreTimingRules {
                mte2: C220Mte2TimingRules {
                    issue_interval: NonZeroU64::new(1).unwrap(),
                    startup_ticks: 0,
                    bytes_per_tick: rate,
                    retire_ticks: 0,
                },
                mte3: C220Mte3TimingRules {
                    issue_interval: NonZeroU64::new(1).unwrap(),
                    startup_ticks: 0,
                    bytes_per_tick: rate,
                    retire_ticks: 0,
                },
                vector: C220VectorTimingRules {
                    dispatch_ticks: 0,
                    uop_issue_interval: NonZeroU64::new(1).unwrap(),
                    ub_response_ticks: 1,
                },
            },
        )
        .unwrap();
        let mut tick = 0;
        for (base_va, base_address) in [(0_u32, 0x600_u64), (2, 0x200)] {
            for half in 0..2_u64 {
                let va = base_va + half as u32;
                for pair in 0..4_u64 {
                    let row = half * 8 + pair * 2;
                    let source_0 = base_address - 32 + row * 32;
                    core.state
                        .scalar_mut()
                        .machine_mut()
                        .set_xreg(6, source_0)
                        .unwrap();
                    core.state
                        .scalar_mut()
                        .machine_mut()
                        .set_xreg(7, source_0 + 32)
                        .unwrap();
                    let word =
                        0x8000_0000 | (va << 17) | ((pair as u32 * 2) << 3) | (6 << 12) | (7 << 7);
                    core.step_word_at(tick, word).unwrap();
                    tick += 1;
                }
            }
        }
        let word = opcode
            | (2 << 12)
            | (5 << 2)
            | u32::from(destination_high)
            | (u32::from(source_high) << 1);
        let C220CoreStep::Executed {
            instruction: C220CoreInstruction::Vector(C220VectorInstruction::Nchw(issue)),
            ..
        } = core.step_word_at(tick, word).unwrap()
        else {
            panic!("VNCHWCONV should issue");
        };
        let uops = C220CoreInstruction::Vector(C220VectorInstruction::Nchw(issue))
            .as_vector()
            .unwrap()
            .uops()
            .unwrap();
        assert_eq!(uops.len(), 2);
        assert!(uops.iter().all(|uop| uop.stages.execute_ticks == 1));
        core.advance_to(300).unwrap();
        assert_eq!(
            core.last_vector_releases()
                .iter()
                .filter(|release| release.pc == 0x4040)
                .count(),
            2
        );
        let output_rows = if width == 4 { 8 } else { 16 };
        for row in 0..output_rows {
            for column in 0..16 {
                let source_offset = (column * 32
                    + if width == 1 {
                        usize::from(source_high) * 16 + row
                    } else {
                        row * width
                    }) as u64;
                let destination_offset = if width == 1 {
                    (row * 32 + usize::from(destination_high) * 16 + column) as u64
                } else {
                    (row * 16 * width + column * width) as u64
                };
                assert_eq!(
                    core.state()
                        .ub()
                        .read_known(0x600 + destination_offset, width)
                        .unwrap(),
                    core.state()
                        .ub()
                        .read_known(0x200 + source_offset, width)
                        .unwrap()
                );
            }
        }
    }
}

#[test]
fn transpose_reads_full_matrix_at_functional_completion() {
    for in_place in [false, true] {
        let word = 0x8240_0c00 | (3 << 17) | (4 << 12);
        let memory = SparseMemory::new(vec![MemoryRegion::unknown(256)], 512, 512);
        let memory = MappedMemory::bind(memory, &[0x2000]).unwrap();
        let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
        let source_address = if in_place { 0x400 } else { 0 };
        machine.set_xreg(3, 0x400).unwrap();
        machine.set_xreg(4, source_address).unwrap();
        let mut ub = UbMemory::new(2048, 256);
        for block in 0..16 {
            let bytes = (0..16)
                .flat_map(|lane| ((block * 16 + lane) as u16).to_le_bytes())
                .map(MemoryByteState::Known)
                .collect::<Vec<_>>();
            ub.write_states(source_address + (block * 32) as u64, &bytes)
                .unwrap();
            if !in_place {
                ub.write_states(
                    0x400 + (block * 32) as u64,
                    &[MemoryByteState::Known(0xaa); 32],
                )
                .unwrap();
            }
        }
        let execution = C220State::new(ScalarStepper::new(machine, 0x4000), ub);
        let rate = NonZeroU64::new(32).unwrap();
        let mut core = C220Core::new(
            execution,
            memory,
            C220CoreTimingRules {
                mte2: C220Mte2TimingRules {
                    issue_interval: NonZeroU64::new(1).unwrap(),
                    startup_ticks: 0,
                    bytes_per_tick: rate,
                    retire_ticks: 0,
                },
                mte3: C220Mte3TimingRules {
                    issue_interval: NonZeroU64::new(1).unwrap(),
                    startup_ticks: 0,
                    bytes_per_tick: rate,
                    retire_ticks: 0,
                },
                vector: C220VectorTimingRules {
                    dispatch_ticks: 0,
                    uop_issue_interval: NonZeroU64::new(1).unwrap(),
                    ub_response_ticks: 1,
                },
            },
        )
        .unwrap();
        let C220CoreStep::Executed {
            instruction: C220CoreInstruction::Vector(C220VectorInstruction::Transpose(issue)),
            ..
        } = core.step_word_at(0, word).unwrap()
        else {
            panic!("transpose should issue");
        };
        let uops = C220CoreInstruction::Vector(C220VectorInstruction::Transpose(issue))
            .as_vector()
            .unwrap()
            .uops()
            .unwrap();
        assert_eq!(uops.len(), 2);
        assert!(
            uops.iter()
                .all(|uop| uop.lane_group.is_none() && uop.stages.execute_ticks == 1)
        );
        let mut releases = 0;
        let mut timing_reads = 0;
        for tick in 0..100 {
            core.advance_to(tick).unwrap();
            releases += core.last_vector_releases().len();
            for sample in core.vector_pipeline().last_read_samples() {
                timing_reads += 1;
                assert_eq!(sample.accesses.len(), 16);
                assert!(sample.lanes.is_empty());
            }
            assert!(core.vector_pipeline().last_functional_samples().is_empty());
            if releases == 2 {
                break;
            }
        }
        assert_eq!(releases, 2);
        assert_eq!(timing_reads, 1);
        let original = if in_place { 0_u16 } else { 0xaaaa };
        assert_eq!(
            core.state().ub().read_known(0x400, 2).unwrap(),
            original.to_le_bytes()
        );
        let updated = (256..512_u16)
            .flat_map(u16::to_le_bytes)
            .map(MemoryByteState::Known)
            .collect::<Vec<_>>();
        for (index, block) in updated.chunks_exact(32).enumerate() {
            core.state
                .ub_mut()
                .write_states(source_address + (index * 32) as u64, block)
                .unwrap();
        }
        core.advance_to(100).unwrap();
        let samples = core.vector_pipeline().last_functional_samples();
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].source_0_bytes.len(), 512);
        assert_eq!(samples[0].lanes.len(), 256);
        for row in 0..16 {
            for column in 0..16 {
                let destination = 0x400 + (row * 16 + column) as u64 * 2;
                let expected = (256 + column * 16 + row) as u16;
                assert_eq!(
                    core.state().ub().read_known(destination, 2).unwrap(),
                    expected.to_le_bytes()
                );
            }
        }
    }
}

#[test]
fn broadcast_issues_one_full_tile_uop_per_repeat() {
    for (opcode, width, destination_pipe) in [
        (0x8000_0044_u32, 2_usize, 5),
        (0x8000_004c, 4, 5),
        (0x8000_0044, 2, 0),
        (0x8000_004c, 4, 0),
    ] {
        let word = opcode | (3 << 17) | (4 << 12) | (5 << 7);
        let memory = SparseMemory::new(vec![MemoryRegion::unknown(256)], 512, 512);
        let memory = MappedMemory::bind(memory, &[0x2000]).unwrap();
        let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
        machine.set_xreg(3, 0x200).unwrap();
        machine.set_xreg(4, 0).unwrap();
        machine
            .set_xreg(5, (2_u64 << 56) | (1 << 52) | (20 << 32) | 2)
            .unwrap();
        let mut ub = UbMemory::new(2048, 256);
        let mut source = vec![0_u8; 16 * width];
        for (index, chunk) in source.chunks_exact_mut(width).enumerate() {
            chunk.copy_from_slice(&((index + 1) as u32).to_le_bytes()[..width]);
        }
        ub.write_states(
            0,
            &source
                .iter()
                .copied()
                .map(MemoryByteState::Known)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        for repeat in 0..2 {
            for block in 0..8 {
                let address = 0x200_u64 + 32 * (276 * repeat + 2 * block);
                ub.write_states(address, &[MemoryByteState::Known(0xaa); 32])
                    .unwrap();
            }
        }
        let execution = C220State::new(ScalarStepper::new(machine, 0x4000), ub);
        let rate = NonZeroU64::new(32).unwrap();
        let mut core = C220Core::new(
            execution,
            memory,
            C220CoreTimingRules {
                mte2: C220Mte2TimingRules {
                    issue_interval: NonZeroU64::new(1).unwrap(),
                    startup_ticks: 0,
                    bytes_per_tick: rate,
                    retire_ticks: 0,
                },
                mte3: C220Mte3TimingRules {
                    issue_interval: NonZeroU64::new(1).unwrap(),
                    startup_ticks: 0,
                    bytes_per_tick: rate,
                    retire_ticks: 0,
                },
                vector: C220VectorTimingRules {
                    dispatch_ticks: 0,
                    uop_issue_interval: NonZeroU64::new(1).unwrap(),
                    ub_response_ticks: 1,
                },
            },
        )
        .unwrap();
        let C220CoreStep::Executed {
            instruction: C220CoreInstruction::Vector(C220VectorInstruction::Broadcast(issue)),
            ..
        } = core.step_word_at(0, word).unwrap()
        else {
            panic!("broadcast should issue");
        };
        let uops = C220CoreInstruction::Vector(C220VectorInstruction::Broadcast(issue))
            .as_vector()
            .unwrap()
            .uops()
            .unwrap();
        assert_eq!(uops.len(), 2);
        assert!(
            uops.iter()
                .all(|uop| uop.lane_group.is_none() && uop.stages.execute_ticks == 2)
        );
        assert_eq!(
            core.state().ub().read_known(0x200, width).unwrap(),
            vec![0xaa; width]
        );
        core.advance_to(100).unwrap();
        let ub = core.state().ub();
        for repeat in 0..2 {
            for block in 0..8 {
                let address = 0x200_u64 + 32 * (276 * repeat + 2 * block);
                let element = (repeat * 8 + block) as usize;
                let expected = &source[element * width..(element + 1) * width];
                assert_eq!(
                    ub.read_known(address, 32).unwrap(),
                    expected.repeat(32 / width)
                );
            }
        }
        assert_eq!(core.vector_pipeline().last_read_samples().len(), 2);
        assert!(
            core.vector_pipeline()
                .last_read_samples()
                .iter()
                .all(|sample| sample.lane_group.is_none() && sample.lanes.is_empty())
        );
        let samples = core.vector_pipeline().last_functional_samples();
        assert_eq!(samples.len(), 2);
        assert!(
            samples
                .iter()
                .all(|sample| sample.accesses[0].bytes as usize == 8 * width)
        );
        core.state
            .scalar_mut()
            .machine_mut()
            .set_xreg(3, (8 * width) as u64)
            .unwrap();
        core.state
            .scalar_mut()
            .machine_mut()
            .set_xreg(5, (2_u64 << 56) | (8 << 32) | 1)
            .unwrap();
        let set = 0x40a0_0000 | (1 << 10) | (destination_pipe << 7);
        core.step_word_at(101, set).unwrap();
        assert!(matches!(
            core.step_word_at(102, word).unwrap(),
            C220CoreStep::Executed { .. }
        ));
        assert!(core.vector_pipeline().pending_retirement_tick().unwrap() > 103);
        core.step_word_at(103, set | 1).unwrap();
        assert!(matches!(
            core.step_word_at(104, word).unwrap(),
            C220CoreStep::Executed { .. }
        ));
        assert!(matches!(
            core.step_word_at(105, set + (1 << 21)).unwrap(),
            C220CoreStep::Executed { .. }
        ));
        let mut consumed = false;
        for tick in 106..200 {
            if matches!(
                core.step_word_at(tick, (set | 1) + (1 << 21)).unwrap(),
                C220CoreStep::Executed { .. }
            ) {
                assert!(core.vector_pipeline().pending_retirement_tick().unwrap() > tick);
                consumed = true;
                break;
            }
        }
        assert!(consumed);
        core.advance_to(200).unwrap();
        let expected = 1_u32.to_le_bytes()[..width].repeat(32 / width);
        for block in 0..8 {
            assert_eq!(
                core.state()
                    .ub()
                    .read_known((8 * width + 256 + block * 32) as u64, 32)
                    .unwrap(),
                expected
            );
        }
        assert!(matches!(
            core.step_word_at(201, 0x40e0_1800).unwrap(),
            C220CoreStep::Executed {
                instruction: C220CoreInstruction::Barrier(_),
                ..
            }
        ));
        for tick in [202, 203] {
            assert!(matches!(
                core.step_word_at(tick, set | 7).unwrap(),
                C220CoreStep::Executed { .. }
            ));
        }
        for tick in [204, 205] {
            assert!(matches!(
                core.step_word_at(tick, (set | 7) + (1 << 21)).unwrap(),
                C220CoreStep::Executed { .. }
            ));
        }
        let pc = core.state().scalar().pc();
        assert!(matches!(
            core.step_word_at(206, (set | 7) + (1 << 21)).unwrap(),
            C220CoreStep::Stalled(C220Stall {
                resume_tick: 207,
                cause: C220StallCause::VectorDependency,
                ..
            })
        ));
        assert_eq!(core.state().scalar().pc(), pc);
    }
}

#[test]
fn vector_scalar_s32_and_f32_capture_scalar_and_delay_writeback() {
    for (opcode, source_bits, scalar_bits, result_bits, execute_ticks, saturating) in [
        (
            0x92c0_0000,
            (-4.0_f32).to_bits(),
            0.5_f32.to_bits(),
            (-2.0_f32).to_bits(),
            8,
            false,
        ),
        (
            0x92c0_0000,
            (-0.0_f32).to_bits(),
            0.5_f32.to_bits(),
            (-0.0_f32).to_bits(),
            8,
            false,
        ),
        (0x9600_0000, u32::MAX, 2, 2, 5, false),
        (0x9600_0001, u32::MAX, 2, u32::MAX, 5, false),
        (0x9700_0000, u32::MAX, 2, 1, 5, false),
        (0x9700_0001, u32::MAX, 2, u32::MAX - 1, 6, false),
        (0x9700_0000, i32::MAX as u32, 1, i32::MAX as u32, 5, true),
        (0x9700_0001, i32::MIN as u32, 2, i32::MIN as u32, 6, true),
        (
            0x96c0_0000,
            1.5_f32.to_bits(),
            2.0_f32.to_bits(),
            2.0_f32.to_bits(),
            5,
            false,
        ),
        (
            0x96c0_0001,
            1.5_f32.to_bits(),
            2.0_f32.to_bits(),
            1.5_f32.to_bits(),
            5,
            false,
        ),
        (
            0x96c0_0000,
            (-0.0_f32).to_bits(),
            0.0_f32.to_bits(),
            0.0_f32.to_bits(),
            5,
            false,
        ),
        (
            0x96c0_0001,
            (-0.0_f32).to_bits(),
            0.0_f32.to_bits(),
            (-0.0_f32).to_bits(),
            5,
            false,
        ),
        (
            0x97c0_0000,
            1.5_f32.to_bits(),
            2.0_f32.to_bits(),
            3.5_f32.to_bits(),
            7,
            false,
        ),
        (
            0x97c0_0001,
            1.5_f32.to_bits(),
            2.0_f32.to_bits(),
            3.0_f32.to_bits(),
            8,
            false,
        ),
    ] {
        let word = opcode | (3 << 17) | (4 << 12) | (6 << 7) | (5 << 2);
        let memory = SparseMemory::new(vec![MemoryRegion::unknown(256)], 512, 512);
        let memory = MappedMemory::bind(memory, &[0x2000]).unwrap();
        let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
        machine.set_xreg(3, 0x100).unwrap();
        machine.set_xreg(4, 0).unwrap();
        machine.set_xreg(5, (1_u64 << 56) | (1 << 16) | 1).unwrap();
        machine.set_xreg(6, u64::from(scalar_bits)).unwrap();
        let control_spr = (1_u64 << 56) | (u64::from(saturating) << 53);
        machine.set_spr_value(3, control_spr).unwrap();
        machine.set_spr_value(100, 1).unwrap();
        machine.set_spr_value(101, 0).unwrap();
        let mut ub = UbMemory::new(512, 256);
        let mut source = [0; 32];
        source[..4].copy_from_slice(&source_bits.to_le_bytes());
        ub.write_states(0, &source.map(MemoryByteState::Known))
            .unwrap();
        ub.write_states(0x100, &[MemoryByteState::Known(0xaa); 32])
            .unwrap();
        let execution = C220State::new(ScalarStepper::new(machine, 0x4000), ub);
        let rate = NonZeroU64::new(32).unwrap();
        let mut core = C220Core::new(
            execution,
            memory,
            C220CoreTimingRules {
                mte2: C220Mte2TimingRules {
                    issue_interval: NonZeroU64::new(1).unwrap(),
                    startup_ticks: 0,
                    bytes_per_tick: rate,
                    retire_ticks: 0,
                },
                mte3: C220Mte3TimingRules {
                    issue_interval: NonZeroU64::new(1).unwrap(),
                    startup_ticks: 0,
                    bytes_per_tick: rate,
                    retire_ticks: 0,
                },
                vector: C220VectorTimingRules {
                    dispatch_ticks: 0,
                    uop_issue_interval: NonZeroU64::new(1).unwrap(),
                    ub_response_ticks: 1,
                },
            },
        )
        .unwrap();
        let C220CoreStep::Executed {
            instruction: C220CoreInstruction::Vector(C220VectorInstruction::Scalar(issue)),
            ..
        } = core.step_word_at(0, word).unwrap()
        else {
            panic!("vector-scalar instruction should issue");
        };
        assert_eq!(issue.scalar.bits, scalar_bits);
        assert_eq!(issue.scalar.integer_saturating, saturating);
        if saturating {
            core.state
                .scalar_mut()
                .machine_mut()
                .set_spr_value(3, 1 << 56)
                .unwrap();
        }
        assert_eq!(
            C220CoreInstruction::Vector(C220VectorInstruction::Scalar(issue))
                .as_vector()
                .unwrap()
                .uops()
                .unwrap()[0]
                .stages
                .execute_ticks,
            execute_ticks
        );
        assert_eq!(core.state().ub().read_known(0x100, 4).unwrap(), [0xaa; 4]);
        core.state
            .scalar_mut()
            .machine_mut()
            .set_xreg(6, 0x8000_0000)
            .unwrap();
        core.advance_to(100).unwrap();
        assert_eq!(
            core.state().ub().read_known(0x100, 4).unwrap(),
            result_bits.to_le_bytes()
        );
        assert_eq!(core.state().ub().read_known(0x104, 4).unwrap(), [0xaa; 4]);
        assert!(
            core.vector_pipeline().last_read_samples()[0]
                .read1_grants
                .is_empty()
        );
        assert_eq!(
            core.vector_pipeline().last_functional_samples()[0].lanes[0]
                .fp32_status
                .is_some(),
            opcode & 0x00c0_0000 == 0x00c0_0000
        );
    }
}

#[test]
fn vector_scalar_16_bit_forms_use_native_uop_widths_and_preserve_inactive_tail() {
    for (opcode, source_bits, scalar_bits, expected, execute_ticks, is_f16, fp_mode, sat) in [
        (0x9240_0000, 0xc400, 0x3800, 0xc000, 8, true, false, false),
        (0x9240_0000, 0x8000, 0x3800, 0x8000, 8, true, false, false),
        (0x9240_0000, 0x7e00, 0x3800, 0, 8, true, false, false),
        (
            0x9640_0000,
            0x3c00_u16,
            0x4000_u16,
            0x4000_u16,
            5,
            true,
            false,
            false,
        ),
        (0x9640_0001, 0x3c00, 0x4000, 0x3c00, 5, true, false, false),
        (0x9740_0000, 0x3c00, 0x4000, 0x4200, 7, true, false, false),
        (0x9740_0001, 0x3c00, 0x4000, 0x4000, 8, true, false, false),
        (0x9740_0000, 0x7c00, 0x3c00, 0x7bff, 7, true, false, false),
        (0x9740_0000, 0x7c00, 0x3c00, 0x7c00, 7, true, true, false),
        (0x9680_0000, u16::MAX, 2, 2, 5, false, false, false),
        (0x9680_0001, u16::MAX, 2, u16::MAX, 5, false, false, false),
        (0x9780_0000, u16::MAX, 2, 1, 5, false, false, false),
        (
            0x9780_0001,
            u16::MAX,
            2,
            u16::MAX - 1,
            6,
            false,
            false,
            false,
        ),
        (
            0x9780_0000,
            i16::MAX as u16,
            1,
            i16::MAX as u16,
            5,
            false,
            false,
            true,
        ),
        (
            0x9780_0001,
            i16::MIN as u16,
            2,
            i16::MIN as u16,
            6,
            false,
            false,
            true,
        ),
    ] {
        let word = opcode | (3 << 17) | (4 << 12) | (6 << 7) | (5 << 2);
        let memory = SparseMemory::new(vec![MemoryRegion::unknown(256)], 512, 512);
        let memory = MappedMemory::bind(memory, &[0x2000]).unwrap();
        let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
        machine.set_xreg(3, 0x200).unwrap();
        machine.set_xreg(4, 0).unwrap();
        machine.set_xreg(5, (1_u64 << 56) | (1 << 16) | 1).unwrap();
        machine.set_xreg(6, u64::from(scalar_bits)).unwrap();
        let control_spr = (1_u64 << 56) | (u64::from(fp_mode) << 48) | (u64::from(sat) << 53);
        machine.set_spr_value(3, control_spr).unwrap();
        machine.set_spr_value(100, 65).unwrap();
        machine.set_spr_value(101, 0).unwrap();
        let mut ub = UbMemory::new(1024, 256);
        let mut source = [0; 256];
        source[..2].copy_from_slice(&source_bits.to_le_bytes());
        source[128..130].copy_from_slice(&source_bits.to_le_bytes());
        ub.write_states(0, &source.map(MemoryByteState::Known))
            .unwrap();
        ub.write_states(0x200, &[MemoryByteState::Known(0xaa); 256])
            .unwrap();
        let execution = C220State::new(ScalarStepper::new(machine, 0x4000), ub);
        let rate = NonZeroU64::new(32).unwrap();
        let mut core = C220Core::new(
            execution,
            memory,
            C220CoreTimingRules {
                mte2: C220Mte2TimingRules {
                    issue_interval: NonZeroU64::new(1).unwrap(),
                    startup_ticks: 0,
                    bytes_per_tick: rate,
                    retire_ticks: 0,
                },
                mte3: C220Mte3TimingRules {
                    issue_interval: NonZeroU64::new(1).unwrap(),
                    startup_ticks: 0,
                    bytes_per_tick: rate,
                    retire_ticks: 0,
                },
                vector: C220VectorTimingRules {
                    dispatch_ticks: 0,
                    uop_issue_interval: NonZeroU64::new(1).unwrap(),
                    ub_response_ticks: 1,
                },
            },
        )
        .unwrap();
        let C220CoreStep::Executed {
            instruction: C220CoreInstruction::Vector(C220VectorInstruction::Scalar(issue)),
            ..
        } = core.step_word_at(0, word).unwrap()
        else {
            panic!("s16 vector-scalar instruction should issue");
        };
        assert_eq!(
            issue.scalar.fp16_mode,
            C220Fp16Mode::from_control_spr(control_spr)
        );
        assert_eq!(issue.scalar.integer_saturating, sat);
        if sat {
            core.state
                .scalar_mut()
                .machine_mut()
                .set_spr_value(3, 1 << 56)
                .unwrap();
        }
        if source_bits == 0x7c00 {
            core.state
                .scalar_mut()
                .machine_mut()
                .set_spr_value(3, control_spr ^ (1 << 48))
                .unwrap();
        }
        let expected_uops = usize::from(
            issue.instruction.operation
                == crate::isa::c220::vector::scalar::C220VectorScalarOperation::Multiply
                && issue.instruction.dtype
                    == crate::isa::c220::vector::scalar::C220VectorScalarType::S16,
        ) + 1;
        let uops = C220CoreInstruction::Vector(C220VectorInstruction::Scalar(issue))
            .as_vector()
            .unwrap()
            .uops()
            .unwrap();
        assert_eq!(uops.len(), expected_uops);
        assert_eq!(uops[0].stages.execute_ticks, execute_ticks);
        core.advance_to(200).unwrap();
        assert_eq!(
            core.state().ub().read_known(0x200, 2).unwrap(),
            expected.to_le_bytes()
        );
        assert_eq!(
            core.state().ub().read_known(0x280, 2).unwrap(),
            expected.to_le_bytes()
        );
        assert_eq!(core.state().ub().read_known(0x282, 2).unwrap(), [0xaa; 2]);
        assert_eq!(core.vector_pipeline().pending_uops(), 0);
        assert_eq!(
            core.vector_pipeline().last_functional_samples()[0].lanes[0]
                .fp16_status
                .is_some(),
            is_f16
        );
    }
}

#[test]
fn vector_s32_binary_operations_use_delayed_reads_and_captured_saturation() {
    for (opcode, first, second, expected, execute_ticks, saturating) in [
        (
            0x8500_0000,
            i32::MAX as u32,
            1_u32,
            i32::MIN as u32,
            5,
            false,
        ),
        (0x8500_0000, i32::MAX as u32, 1, i32::MAX as u32, 5, true),
        (0x8500_0001, i32::MIN as u32, 1, i32::MAX as u32, 5, false),
        (0x8500_0001, i32::MIN as u32, 1, i32::MIN as u32, 5, true),
        (0x8900_0000, i32::MAX as u32, 2, u32::MAX - 1, 6, false),
        (0x8900_0000, i32::MAX as u32, 2, i32::MAX as u32, 6, true),
        (0x8700_0000, u32::MAX, 2, 2, 5, false),
        (0x8700_0001, u32::MAX, 2, u32::MAX, 5, false),
    ] {
        let word = opcode | (3 << 17) | (4 << 12) | (5 << 7) | (6 << 2);
        let memory = SparseMemory::new(vec![MemoryRegion::unknown(256)], 512, 512);
        let memory = MappedMemory::bind(memory, &[0x2000]).unwrap();
        let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
        machine.set_xreg(3, 0x200).unwrap();
        machine.set_xreg(4, 0).unwrap();
        machine.set_xreg(5, 0x100).unwrap();
        machine.set_xreg(6, 0x0100_0808_0801_0101).unwrap();
        let control_spr = (1_u64 << 56) | (u64::from(saturating) << 53);
        machine.set_spr_value(3, control_spr).unwrap();
        machine.set_spr_value(100, 1).unwrap();
        machine.set_spr_value(101, 0).unwrap();
        let mut ub = UbMemory::new(1024, 256);
        ub.write_states(0, &[MemoryByteState::Known(0); 32])
            .unwrap();
        ub.write_states(0x100, &[MemoryByteState::Known(0); 32])
            .unwrap();
        ub.write_states(0, &first.to_le_bytes().map(MemoryByteState::Known))
            .unwrap();
        ub.write_states(0x100, &second.to_le_bytes().map(MemoryByteState::Known))
            .unwrap();
        ub.write_states(0x200, &[MemoryByteState::Known(0xaa); 32])
            .unwrap();
        let execution = C220State::new(ScalarStepper::new(machine, 0x4000), ub);
        let rate = NonZeroU64::new(32).unwrap();
        let mut core = C220Core::new(
            execution,
            memory,
            C220CoreTimingRules {
                mte2: C220Mte2TimingRules {
                    issue_interval: NonZeroU64::new(1).unwrap(),
                    startup_ticks: 0,
                    bytes_per_tick: rate,
                    retire_ticks: 0,
                },
                mte3: C220Mte3TimingRules {
                    issue_interval: NonZeroU64::new(1).unwrap(),
                    startup_ticks: 0,
                    bytes_per_tick: rate,
                    retire_ticks: 0,
                },
                vector: C220VectorTimingRules {
                    dispatch_ticks: 0,
                    uop_issue_interval: NonZeroU64::new(1).unwrap(),
                    ub_response_ticks: 1,
                },
            },
        )
        .unwrap();
        let C220CoreStep::Executed {
            instruction: C220CoreInstruction::Vector(C220VectorInstruction::Arithmetic(issue)),
            ..
        } = core.step_word_at(0, word).unwrap()
        else {
            panic!("S32 vector instruction should issue");
        };
        assert_eq!(issue.modes.integer_saturating, saturating);
        assert!(issue.hint.has_s32_value_path());
        assert_eq!(
            C220CoreInstruction::Vector(C220VectorInstruction::Arithmetic(issue))
                .as_vector()
                .unwrap()
                .uops()
                .unwrap()[0]
                .stages
                .execute_ticks,
            execute_ticks
        );
        assert_eq!(core.state().ub().read_known(0x200, 4).unwrap(), [0xaa; 4]);
        core.state
            .scalar_mut()
            .machine_mut()
            .set_spr_value(3, control_spr ^ (1 << 53))
            .unwrap();
        core.advance_to(100).unwrap();
        assert_eq!(
            core.state().ub().read_known(0x200, 4).unwrap(),
            expected.to_le_bytes()
        );
        assert!(
            core.vector_pipeline().last_functional_samples()[0].lanes[0]
                .fp32_status
                .is_none()
        );
    }
}

#[test]
fn vector_s16_binary_operations_use_native_uop_widths() {
    for (opcode, first, second, expected, saturating, widen_bit, execute_ticks) in [
        (0x9480_0000, 3, -5, 0, false, false, 5),
        (0x9480_0000, i16::MAX, 1, 0, false, false, 5),
        (0x9480_0000, i16::MAX, 1, i16::MAX, true, false, 5),
        (0x9480_0001, 3, 5, 0, false, true, 5),
        (0x8580_0000, i16::MAX, 1_i16, i16::MIN, false, false, 5),
        (0x8580_0000, i16::MAX, 1, i16::MAX, true, false, 5),
        (0x8580_0001, i16::MIN, 1, i16::MAX, false, false, 5),
        (0x8580_0001, i16::MIN, 1, i16::MIN, true, false, 5),
        (0x8980_0000, i16::MAX, 2, -2, false, false, 6),
        (0x8980_0000, i16::MAX, 2, i16::MAX, true, false, 6),
        (0x8780_0000, -1, 2, 2, false, false, 5),
        (0x8780_0000, -1, 2, 2, false, true, 5),
        (0x8780_0001, -1, 2, -1, false, false, 5),
        (0x9a40_0000, 0x0f0f, 0x3333, 0x3f3f, false, true, 1),
        (0x9a40_0001, 0x0f0f, 0x3333, 0x0303, false, true, 1),
    ] {
        let word = opcode | (3 << 17) | (4 << 12) | (5 << 7) | (6 << 2);
        let memory = SparseMemory::new(vec![MemoryRegion::unknown(256)], 512, 512);
        let memory = MappedMemory::bind(memory, &[0x2000]).unwrap();
        let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
        machine.set_xreg(3, 0x200).unwrap();
        machine.set_xreg(4, 0).unwrap();
        machine.set_xreg(5, 0x100).unwrap();
        machine.set_xreg(6, 0x0100_0808_0801_0101).unwrap();
        let control_spr = (u64::from(saturating) << 53) | (u64::from(widen_bit) << 52);
        machine.set_spr_value(3, control_spr).unwrap();
        machine.set_spr_value(100, 1).unwrap();
        machine.set_spr_value(101, 1).unwrap();
        let mut ub = UbMemory::new(1024, 256);
        for address in [0, 0x80, 0x100, 0x180] {
            ub.write_states(address, &[MemoryByteState::Known(0); 32])
                .unwrap();
        }
        for offset in [0, 0x80] {
            ub.write_states(offset, &first.to_le_bytes().map(MemoryByteState::Known))
                .unwrap();
            ub.write_states(
                offset + 0x100,
                &second.to_le_bytes().map(MemoryByteState::Known),
            )
            .unwrap();
            ub.write_states(offset + 0x200, &[MemoryByteState::Known(0xaa); 32])
                .unwrap();
        }
        let execution = C220State::new(ScalarStepper::new(machine, 0x4000), ub);
        let rate = NonZeroU64::new(32).unwrap();
        let mut core = C220Core::new(
            execution,
            memory,
            C220CoreTimingRules {
                mte2: C220Mte2TimingRules {
                    issue_interval: NonZeroU64::new(1).unwrap(),
                    startup_ticks: 0,
                    bytes_per_tick: rate,
                    retire_ticks: 0,
                },
                mte3: C220Mte3TimingRules {
                    issue_interval: NonZeroU64::new(1).unwrap(),
                    startup_ticks: 0,
                    bytes_per_tick: rate,
                    retire_ticks: 0,
                },
                vector: C220VectorTimingRules {
                    dispatch_ticks: 0,
                    uop_issue_interval: NonZeroU64::new(1).unwrap(),
                    ub_response_ticks: 1,
                },
            },
        )
        .unwrap();
        let C220CoreStep::Executed {
            instruction: C220CoreInstruction::Vector(C220VectorInstruction::Arithmetic(issue)),
            ..
        } = core.step_word_at(0, word).unwrap()
        else {
            panic!("S16 vector instruction should issue");
        };
        assert_eq!(issue.result_element_bytes, 2);
        assert_eq!(issue.modes.integer_saturating, saturating);
        assert_eq!(issue.modes.widen_s16, widen_bit);
        let operation = issue.hint.operation;
        let uops = C220CoreInstruction::Vector(C220VectorInstruction::Arithmetic(issue))
            .as_vector()
            .unwrap()
            .uops()
            .unwrap();
        let expected_uops = usize::from(
            operation == crate::isa::c220::vector::C220VecArithmeticOperation::Multiply,
        ) + 1;
        assert_eq!(uops.len(), expected_uops);
        assert!(
            uops.iter()
                .all(|uop| uop.stages.execute_ticks == execute_ticks)
        );
        core.state
            .scalar_mut()
            .machine_mut()
            .set_spr_value(3, control_spr ^ (1 << 53))
            .unwrap();
        core.advance_to(100).unwrap();
        for address in [0x200, 0x280] {
            assert_eq!(
                core.state().ub().read_known(address, 2).unwrap(),
                expected.to_le_bytes()
            );
        }
        assert_eq!(
            core.vector_pipeline()
                .last_read_samples()
                .iter()
                .map(|sample| sample.lane_group)
                .collect::<Vec<_>>(),
            if expected_uops == 2 {
                vec![Some(0), Some(1)]
            } else {
                vec![Some(0)]
            }
        );
    }
}

#[test]
fn scalar_conversion_retires_after_two_ticks_and_blocks_dependent_conversion() {
    let memory = SparseMemory::new(vec![MemoryRegion::unknown(64)], 128, 128);
    let memory = MappedMemory::bind(memory, &[0x2000]).unwrap();
    let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
    machine.set_xreg(8, u64::from(4.75_f32.to_bits())).unwrap();
    machine.set_xreg(11, u64::from(6.5_f32.to_bits())).unwrap();
    let execution = C220State::new(ScalarStepper::new(machine, 0x4000), UbMemory::new(512, 256));
    let rate = NonZeroU64::new(32).unwrap();
    let mut core = C220Core::new(
        execution,
        memory,
        C220CoreTimingRules {
            mte2: C220Mte2TimingRules {
                issue_interval: NonZeroU64::new(1).unwrap(),
                startup_ticks: 0,
                bytes_per_tick: rate,
                retire_ticks: 0,
            },
            mte3: C220Mte3TimingRules {
                issue_interval: NonZeroU64::new(1).unwrap(),
                startup_ticks: 0,
                bytes_per_tick: rate,
                retire_ticks: 0,
            },
            vector: C220VectorTimingRules {
                dispatch_ticks: 0,
                uop_issue_interval: NonZeroU64::new(1).unwrap(),
                ub_response_ticks: 1,
            },
        },
    )
    .unwrap();

    let C220CoreStep::Executed {
        instruction:
            C220CoreInstruction::Scalar {
                timing: Some(ticket),
                ..
            },
        ..
    } = core.step_word_at(10, 0x0210_8583).unwrap()
    else {
        panic!("scalar conversion should issue");
    };
    assert_eq!((ticket.issue_tick, ticket.retire_tick), (10, 12));
    assert_eq!(ticket.execution_stage, 2);
    assert_eq!(core.scalar_timing().pending_xreg_retirement(8), Some(12));
    for word in [
        (2 << 29) | (8 << 12),
        (2 << 29) | (1 << 21) | (8 << 12),
        0x4800_0000 | (8 << 5) | 8,
    ] {
        assert_eq!(core.scalar_timing().dependency_tick(word, 11), None);
    }

    let C220CoreStep::Stalled(stall) = core.step_word_at(11, 0x0210_8583).unwrap() else {
        panic!("dependent conversion should wait");
    };
    assert_eq!(stall.cause, C220StallCause::ScalarDependency);
    assert_eq!(stall.resume_tick, 12);
    assert_eq!(core.state().scalar().pc(), 0x4004);

    let C220CoreStep::Stalled(move_stall) = core.step_word_at(11, 0x0202_8800).unwrap() else {
        panic!("scalar register read should wait");
    };
    assert_eq!(move_stall.cause, C220StallCause::ScalarDependency);
    assert_eq!(move_stall.resume_tick, 12);

    for word in [
        (3 << 29) | (2 << 27) | (2 << 23) | (1 << 17) | (8 << 12) | (3 << 7) | 8,
        (3 << 29) | (1 << 17) | (2 << 12) | (8 << 7),
        (2 << 29) | (5 << 21) | (1 << 17) | (1 << 10) | (8 << 2),
        (2 << 29) | (15 << 21) | (3 << 15) | (10 << 10) | (2 << 7) | (2 << 5) | 8,
        (2 << 29) | (1 << 17) | (8 << 12),
        (2 << 29) | (1 << 21) | (1 << 17) | (8 << 12),
        0x4800_0000 | (4 << 21),
        0x4800_0000 | (1 << 16) | 8,
        0x4800_0000 | (1 << 17) | (8 << 5),
        (2 << 29) | (1 << 21) | (3 << 18) | (8 << 12),
        (2 << 29) | (1 << 21) | (3 << 18) | (1 << 17) | (8 << 12),
        0x1000_0000 | (8 << 17) | 7,
    ] {
        let C220CoreStep::Stalled(stall) = core.step_word_at(11, word).unwrap() else {
            panic!("pending scalar operand must block dispatch");
        };
        assert_eq!(stall.cause, C220StallCause::ScalarDependency);
        assert_eq!(stall.resume_tick, 12);
        assert_eq!(core.state().scalar().pc(), 0x4004);
        assert_eq!(core.queued_fixp_commands(), 0);
    }

    assert!(matches!(
        core.step_word_at(11, 0x0216_b583).unwrap(),
        C220CoreStep::Executed {
            instruction: C220CoreInstruction::Scalar { .. },
            ..
        }
    ));
    assert_eq!(core.scalar_timing().pending_xreg_retirement(11), Some(13));
    assert!(matches!(
        core.step_word_at(12, 0x0202_8800).unwrap(),
        C220CoreStep::Executed { .. }
    ));
    assert_eq!(core.scalar_timing().pending_xreg_retirement(8), None);
    assert_eq!(core.state().scalar().machine().xregs()[1], 4);
    assert!(matches!(
        core.step_word_at(13, 0x0210_8583).unwrap(),
        C220CoreStep::Executed { .. }
    ));
    assert_eq!(core.scalar_timing().pending_xreg_retirement(8), Some(15));
    let branch_pc = core.state().scalar().pc();
    let offset = core.state().scalar().machine().xregs()[8];
    assert!(matches!(
        core.step_word_at(15, (2 << 29) | (1 << 17) | (8 << 12))
            .unwrap(),
        C220CoreStep::Executed { .. }
    ));
    assert_eq!(core.state().scalar().pc(), branch_pc + offset * 4);
    let multiply_add = (6 << 17) | (1 << 12) | (2 << 7) | 4;
    let C220CoreStep::Executed {
        instruction:
            C220CoreInstruction::Scalar {
                timing: Some(ticket),
                ..
            },
        ..
    } = core.step_word_at(16, multiply_add).unwrap()
    else {
        panic!("multiply-add should issue with fixed timing");
    };
    assert_eq!((ticket.retire_tick, ticket.execution_stage), (19, 3));
    let add = (6 << 17) | (1 << 12) | (2 << 7) | 1;
    assert!(matches!(
        core.step_word_at(17, add).unwrap(),
        C220CoreStep::Executed { .. }
    ));
    assert_eq!(core.scalar_timing().pending_xreg_retirement(6), Some(18));
    assert_eq!(core.scalar_timing().pending_drain_tick(), Some(19));
    let read_result = 0x0200_0800 | (7 << 17) | (6 << 12);
    assert!(matches!(
        core.step_word_at(18, read_result).unwrap(),
        C220CoreStep::Executed { .. }
    ));
    assert_eq!(core.scalar_timing().pending_xreg_retirement(7), Some(19));
    assert_eq!(
        core.state().scalar().machine().xregs()[7],
        core.state().scalar().machine().xregs()[6]
    );
    let divide = (9 << 17) | (1 << 12) | (1 << 7) | 5;
    assert!(matches!(
        core.step_word_at(20, divide).unwrap(),
        C220CoreStep::Executed { .. }
    ));
    assert_eq!(core.state().scalar().machine().xregs()[9], 1);
    assert_eq!(core.scalar_timing().pending_xreg_retirement(9), Some(40));
    let sqrt = 0x0200_0000 | (10 << 17) | (1 << 12);
    assert!(matches!(
        core.step_word_at(21, sqrt).unwrap(),
        C220CoreStep::Executed { .. }
    ));
    assert_eq!(core.scalar_timing().pending_xreg_retirement(9), Some(36));
    assert_eq!(core.scalar_timing().pending_xreg_retirement(10), Some(40));
    assert_eq!(core.scalar_timing().variable_retirements().count(), 2);
    let read_divide = 0x0200_0800 | (12 << 17) | (9 << 12);
    let C220CoreStep::Stalled(stall) = core.step_word_at(35, read_divide).unwrap() else {
        panic!("division must await a retirement event");
    };
    assert_eq!(stall.resume_tick, 36);
    assert!(matches!(
        core.step_word_at(36, read_divide).unwrap(),
        C220CoreStep::Executed { .. }
    ));
    assert_eq!(core.state().scalar().machine().xregs()[12], 1);
    assert_eq!(core.scalar_timing().variable_retirements().count(), 1);
    let read_sqrt = 0x0200_0800 | (13 << 17) | (10 << 12);
    let C220CoreStep::Stalled(stall) = core.step_word_at(39, read_sqrt).unwrap() else {
        panic!("second FIFO entry must await the remaining event");
    };
    assert_eq!(stall.resume_tick, 40);
    assert!(matches!(
        core.step_word_at(40, read_sqrt).unwrap(),
        C220CoreStep::Executed { .. }
    ));
    assert_eq!(core.scalar_timing().variable_retirements().count(), 0);
    let write_condition = 0x0200_0900 | (11 << 17) | (1 << 12);
    let condition_step = core.step_word_at(41, write_condition).unwrap();
    let C220CoreStep::Executed {
        instruction:
            C220CoreInstruction::Scalar {
                spr_timing: Some(ticket),
                ..
            },
        ..
    } = condition_step
    else {
        panic!("ordinary scalar SPR write needs a retirement ticket: {condition_step:?}");
    };
    assert_eq!((ticket.destination_spr, ticket.retire_tick), (11, 42));
    assert_eq!(core.state().scalar().machine().spr_value(11), Some(0));
    let read_condition = 0x0200_0880 | (14 << 17) | (11 << 12);
    for word in [write_condition, read_condition] {
        assert_eq!(core.scalar_timing().dependency_tick(word, 41), Some(42));
        let C220CoreStep::Stalled(stall) = core.step_word_at(41, word).unwrap() else {
            panic!("SPR read and overwrite must await its commit");
        };
        assert_eq!(stall.resume_tick, 42);
    }
    assert!(matches!(
        core.step_word_at(42, read_condition).unwrap(),
        C220CoreStep::Executed { .. }
    ));
    assert_eq!(core.state().scalar().machine().xregs()[14], 0);
    assert_eq!(core.scalar_timing().pending_xreg_retirement(14), Some(43));
    assert_eq!(core.scalar_timing().pending_spr_retirement(11), None);
    let write_control = 0x0200_0900 | (3 << 17) | (1 << 12);
    assert!(matches!(
        core.step_word_at(43, write_control).unwrap(),
        C220CoreStep::Executed { .. }
    ));
    assert_eq!(core.scalar_timing().pending_spr_retirement(3), None);
    for (tick, register, immediate, value) in [(44, 11, 0xffff, 1), (45, 90, 0xabcd, 0xcd)] {
        let word = 0x1200_0000 | (register << 17) | immediate;
        let C220CoreStep::Executed {
            instruction:
                C220CoreInstruction::Scalar {
                    step,
                    spr_timing: Some(ticket),
                    ..
                },
            ..
        } = core.step_word_at(tick, word).unwrap()
        else {
            panic!("immediate SPR write should issue");
        };
        let crate::sim::common::scalar::ScalarInstructionStep::SprWrite(write) = step.instruction
        else {
            panic!("expected SPR write result");
        };
        assert_eq!(write.source_register, None);
        assert_eq!(write.source_value, u64::from(immediate));
        assert_eq!(write.value, value);
        assert_eq!(ticket.retire_tick, tick + 1);
        assert_eq!(
            core.state().scalar().machine().spr_value(register as u16),
            Some(value)
        );
        assert_eq!(
            core.scalar_timing().dependency_tick(word, tick),
            Some(tick + 1)
        );
    }
    let machine = core.state.scalar_mut().machine_mut();
    machine.set_xreg(1, u64::from(1.5_f32.to_bits())).unwrap();
    machine.set_xreg(2, u64::from(2.0_f32.to_bits())).unwrap();
    for (tick, opcode, expected, latency, stage) in [
        (46, 1, 3.5_f32, 5, 3),
        (51, 2, 1.5, 5, 3),
        (56, 3, 3.0, 5, 3),
        (61, 4, 9.0, 5, 3),
        (66, 5, 4.5, 14, 14),
        (80, 8, 2.0, 1, 1),
        (81, 7, 2.0, 1, 1),
    ] {
        let word = 0x0082_1100 | opcode;
        let C220CoreStep::Executed {
            instruction:
                C220CoreInstruction::Scalar {
                    timing: Some(ticket),
                    ..
                },
            ..
        } = core.step_word_at(tick, word).unwrap()
        else {
            panic!("FP32 operation should issue");
        };
        assert_eq!(ticket.retire_tick, tick + latency);
        assert_eq!(ticket.execution_stage, stage);
        assert_eq!(
            ticket.class,
            if opcode == 5 {
                crate::sim::c220::scalar::timing::C220ScalarTimingClass::Variable
            } else {
                crate::sim::c220::scalar::timing::C220ScalarTimingClass::Fixed
            }
        );
        assert_eq!(
            core.state().scalar().machine().xregs()[1],
            u64::from(expected.to_bits())
        );
        let before = core.state().scalar().machine().clone();
        if latency == 1 {
            assert_eq!(
                core.scalar_timing().pending_xreg_retirement(1),
                Some(tick + 1)
            );
            continue;
        }
        let C220CoreStep::Stalled(stall) = core.step_word_at(tick + 1, word).unwrap() else {
            panic!("dependent FP32 operation must await retirement");
        };
        assert_eq!(stall.resume_tick, tick + latency);
        assert_eq!(core.state().scalar().machine(), &before);
    }
    core.state
        .scalar_mut()
        .machine_mut()
        .set_xreg(1, u64::from(4.0_f32.to_bits()))
        .unwrap();
    for (tick, opcode, expected, latency) in [
        (82, 0, 2.0_f32, 18),
        (100, 0x80, -2.0, 1),
        (101, 0x100, 2.0, 1),
    ] {
        let word = 0x0282_1000 | opcode;
        let C220CoreStep::Executed {
            instruction:
                C220CoreInstruction::Scalar {
                    timing: Some(ticket),
                    ..
                },
            ..
        } = core.step_word_at(tick, word).unwrap()
        else {
            panic!("unary FP32 operation should execute");
        };
        assert_eq!(ticket.retire_tick, tick + latency);
        assert_eq!(ticket.execution_stage, latency as u8);
        assert_eq!(
            core.state().scalar().machine().xregs()[1],
            u64::from(expected.to_bits())
        );
        if opcode == 0 {
            let C220CoreStep::Stalled(stall) = core.step_word_at(83, 0x0282_1080).unwrap() else {
                panic!("negate should wait for square root");
            };
            assert_eq!(stall.resume_tick, 100);
        }
    }
    assert!(matches!(
        core.step_word_at(102, 0x0086_110f).unwrap(),
        C220CoreStep::Executed { .. }
    ));
    assert_eq!(core.state().scalar().machine().xregs()[3], 1);
    assert_eq!(core.scalar_timing().pending_xreg_retirement(3), Some(103));
    assert!(matches!(
        core.step_word_at(103, 0x0080_111e).unwrap(),
        C220CoreStep::Executed { .. }
    ));
    assert_eq!(core.scalar_timing().pending_drain_tick(), Some(104));
    assert_eq!(core.scalar_timing().pending_spr_retirement(11), None);
    assert_eq!(core.state().scalar().machine().spr_value(11), Some(0));
    assert!(matches!(
        core.step_word_at(104, 0x0088_1109).unwrap(),
        C220CoreStep::Executed { .. }
    ));
    assert_eq!(
        core.state().scalar().machine().xregs()[4],
        u64::from(2.0_f32.to_bits())
    );
}

#[test]
fn vabs_uses_modeled_five_tick_execution_stage() {
    let word = 0x83c0_0300 | (3 << 17) | (4 << 12) | (5 << 2);
    let make_core = || {
        let memory = SparseMemory::new(vec![MemoryRegion::unknown(256)], 512, 512);
        let memory = MappedMemory::bind(memory, &[0x2000]).unwrap();
        let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
        machine.set_xreg(3, 0x100).unwrap();
        machine.set_xreg(4, 0).unwrap();
        machine.set_xreg(5, (1_u64 << 56) | (1 << 16) | 1).unwrap();
        machine.set_spr_value(3, 1 << 56).unwrap();
        machine.set_spr_value(100, 1).unwrap();
        machine.set_spr_value(101, 0).unwrap();
        let mut ub = UbMemory::new(512, 256);
        let mut source = [0; 32];
        source[..4].copy_from_slice(&(-1.0_f32).to_le_bytes());
        ub.write_states(0, &source.map(MemoryByteState::Known))
            .unwrap();
        let execution = C220State::new(ScalarStepper::new(machine, 0x4000), ub);
        let rate = NonZeroU64::new(32).unwrap();
        C220Core::new(
            execution,
            memory,
            C220CoreTimingRules {
                mte2: C220Mte2TimingRules {
                    issue_interval: NonZeroU64::new(1).unwrap(),
                    startup_ticks: 0,
                    bytes_per_tick: rate,
                    retire_ticks: 0,
                },
                mte3: C220Mte3TimingRules {
                    issue_interval: NonZeroU64::new(1).unwrap(),
                    startup_ticks: 0,
                    bytes_per_tick: rate,
                    retire_ticks: 0,
                },
                vector: C220VectorTimingRules {
                    dispatch_ticks: 0,
                    uop_issue_interval: NonZeroU64::new(1).unwrap(),
                    ub_response_ticks: 1,
                },
            },
        )
        .unwrap()
    };
    let mut timed = make_core();
    let C220CoreStep::Executed {
        instruction: C220CoreInstruction::Vector(C220VectorInstruction::Arithmetic(issue)),
        ..
    } = timed.step_word_at(0, word).unwrap()
    else {
        panic!("VABS should issue to the vector pipeline");
    };
    assert_eq!(
        C220CoreInstruction::Vector(C220VectorInstruction::Arithmetic(issue))
            .as_vector()
            .unwrap()
            .uops()
            .unwrap()[0]
            .stages
            .execute_ticks,
        5
    );
    let visible = timed.vector_pipeline().pending_visibility_tick().unwrap();
    timed.advance_to(visible).unwrap();
    assert_eq!(
        timed.state().ub().read_known(0x100, 4).unwrap(),
        1.0_f32.to_le_bytes()
    );
    assert!(
        timed.vector_pipeline().last_read_samples()[0]
            .read1_grants
            .is_empty()
    );
}

#[test]
fn vnot_b16_reads_one_source_and_preserves_inactive_ub_lanes() {
    let word = 0x8240_0800 | (3 << 17) | (4 << 12) | (5 << 2);
    let memory = SparseMemory::new(vec![MemoryRegion::unknown(256)], 512, 512);
    let memory = MappedMemory::bind(memory, &[0x2000]).unwrap();
    let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
    machine.set_xreg(3, 0x100).unwrap();
    machine.set_xreg(4, 0).unwrap();
    machine.set_xreg(5, (1_u64 << 56) | (1 << 16) | 1).unwrap();
    machine.set_spr_value(3, 1 << 52).unwrap();
    machine.set_spr_value(100, 1).unwrap();
    machine.set_spr_value(101, 0).unwrap();
    let mut ub = UbMemory::new(512, 256);
    let mut source = [0; 32];
    source[..2].copy_from_slice(&0x00f0_u16.to_le_bytes());
    ub.write_states(0, &source.map(MemoryByteState::Known))
        .unwrap();
    ub.write_states(0x100, &[MemoryByteState::Known(0xaa); 32])
        .unwrap();
    let execution = C220State::new(ScalarStepper::new(machine, 0x4000), ub);
    let rate = NonZeroU64::new(32).unwrap();
    let mut core = C220Core::new(
        execution,
        memory,
        C220CoreTimingRules {
            mte2: C220Mte2TimingRules {
                issue_interval: NonZeroU64::new(1).unwrap(),
                startup_ticks: 0,
                bytes_per_tick: rate,
                retire_ticks: 0,
            },
            mte3: C220Mte3TimingRules {
                issue_interval: NonZeroU64::new(1).unwrap(),
                startup_ticks: 0,
                bytes_per_tick: rate,
                retire_ticks: 0,
            },
            vector: C220VectorTimingRules {
                dispatch_ticks: 0,
                uop_issue_interval: NonZeroU64::new(1).unwrap(),
                ub_response_ticks: 1,
            },
        },
    )
    .unwrap();
    let C220CoreStep::Executed {
        instruction: C220CoreInstruction::Vector(C220VectorInstruction::Arithmetic(issue)),
        ..
    } = core.step_word_at(0, word).unwrap()
    else {
        panic!("VNOT should issue to the vector pipeline");
    };
    assert_eq!(issue.hint.source_1_register, None);
    assert_eq!(issue.result_element_bytes, 2);
    let uops = C220CoreInstruction::Vector(C220VectorInstruction::Arithmetic(issue))
        .as_vector()
        .unwrap()
        .uops()
        .unwrap();
    assert_eq!(uops[0].stages.execute_ticks, 1);
    core.advance_to(100).unwrap();
    let ub = core.state().ub();
    assert_eq!(ub.read_known(0x100, 2).unwrap(), 0xff0f_u16.to_le_bytes());
    assert_eq!(ub.read_known(0x102, 2).unwrap(), [0xaa; 2]);
    assert!(
        core.vector_pipeline().last_read_samples()[0]
            .read1_grants
            .is_empty()
    );
}

#[test]
fn vector_shifts_capture_scalar_and_follow_masked_pipeline() {
    for (opcode, source, shift, expected, width) in [
        (0x9c80_0003_u32, 0x8001_u32, 1_u64, 2_u32, 2_usize),
        (0x9cc0_0003, 0x8000_0001, 32, 0, 4),
        (0x9b00_0001, 0x8001, 1, 0x4000, 2),
        (0x9b40_0000, 0xfffd, 1, 0xfffe, 2),
        (0x9b40_0001, 0xfffd, 1, 0xffff, 2),
        (0x9b40_0000, 0xfffd, 17, 0xffff, 2),
        (0x9b40_0001, 0xfffd, 17, 0, 2),
        (0x9b80_0000, 0x8000_0001, 64, 0x8000_0001, 4),
        (0x9bc0_0001, 0xffff_fffd, 1, 0xffff_ffff, 4),
    ] {
        let word = opcode | (3 << 17) | (4 << 12) | (6 << 7) | (5 << 2);
        let memory = SparseMemory::new(vec![MemoryRegion::unknown(256)], 512, 512);
        let memory = MappedMemory::bind(memory, &[0x2000]).unwrap();
        let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
        machine.set_xreg(3, 0x100).unwrap();
        machine.set_xreg(4, 0).unwrap();
        machine.set_xreg(5, (1_u64 << 56) | (1 << 16) | 1).unwrap();
        machine.set_xreg(6, shift).unwrap();
        machine.set_spr_value(3, 0).unwrap();
        machine.set_spr_value(100, 1).unwrap();
        machine.set_spr_value(101, 0).unwrap();
        let mut ub = UbMemory::new(512, 256);
        let mut tile = [0; 32];
        tile[..width].copy_from_slice(&source.to_le_bytes()[..width]);
        ub.write_states(0, &tile.map(MemoryByteState::Known))
            .unwrap();
        ub.write_states(0x100, &[MemoryByteState::Known(0xaa); 32])
            .unwrap();
        let execution = C220State::new(ScalarStepper::new(machine, 0x4000), ub);
        let rate = NonZeroU64::new(32).unwrap();
        let mut core = C220Core::new(
            execution,
            memory,
            C220CoreTimingRules {
                mte2: C220Mte2TimingRules {
                    issue_interval: NonZeroU64::new(1).unwrap(),
                    startup_ticks: 0,
                    bytes_per_tick: rate,
                    retire_ticks: 0,
                },
                mte3: C220Mte3TimingRules {
                    issue_interval: NonZeroU64::new(1).unwrap(),
                    startup_ticks: 0,
                    bytes_per_tick: rate,
                    retire_ticks: 0,
                },
                vector: C220VectorTimingRules {
                    dispatch_ticks: 0,
                    uop_issue_interval: NonZeroU64::new(1).unwrap(),
                    ub_response_ticks: 1,
                },
            },
        )
        .unwrap();
        let C220CoreStep::Executed {
            instruction: C220CoreInstruction::Vector(C220VectorInstruction::Shift(issue)),
            ..
        } = core.step_word_at(0, word).unwrap()
        else {
            panic!("shift should issue to the vector pipeline");
        };
        assert_eq!(issue.shift, shift as u32);
        assert_eq!(
            C220CoreInstruction::Vector(C220VectorInstruction::Shift(issue))
                .as_vector()
                .unwrap()
                .uops()
                .unwrap()[0]
                .stages
                .execute_ticks,
            6
        );
        assert_eq!(
            core.state().ub().read_known(0x100, width).unwrap(),
            vec![0xaa; width]
        );
        core.advance_to(100).unwrap();
        let ub = core.state().ub();
        assert_eq!(
            ub.read_known(0x100, width).unwrap(),
            expected.to_le_bytes()[..width]
        );
        assert_eq!(ub.read_known(0x100 + width as u64, 2).unwrap(), [0xaa; 2]);
        assert!(
            core.vector_pipeline().last_read_samples()[0]
                .read1_grants
                .is_empty()
        );
    }
}

#[test]
fn vector_copy_uses_both_lane_groups_and_preserves_masked_destinations() {
    for (opcode, element_bytes, selected_lane) in [
        (0x8240_0700_u32, 2_usize, 64_usize),
        (0x8280_0700_u32, 4_usize, 32_usize),
    ] {
        let word = opcode | (3 << 17) | (4 << 12) | (5 << 2);
        let memory = SparseMemory::new(vec![MemoryRegion::unknown(256)], 512, 512);
        let memory = MappedMemory::bind(memory, &[0x2000]).unwrap();
        let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
        machine.set_xreg(3, 0x200).unwrap();
        machine.set_xreg(4, 0).unwrap();
        machine.set_xreg(5, (1_u64 << 56) | (8 << 32) | 1).unwrap();
        machine.set_spr_value(3, 0).unwrap();
        machine.set_spr_value(100, 1).unwrap();
        machine
            .set_spr_value(101, u64::from(element_bytes == 2))
            .unwrap();
        if element_bytes == 4 {
            machine.set_spr_value(100, 1 | (1 << 32)).unwrap();
        }
        let mut ub = UbMemory::new(1024, 256);
        let mut source = [0_u8; 256];
        source[..element_bytes].copy_from_slice(&0x1234_5678_u32.to_le_bytes()[..element_bytes]);
        source[128..128 + element_bytes]
            .copy_from_slice(&0xabcd_ef12_u32.to_le_bytes()[..element_bytes]);
        ub.write_states(0, &source.map(MemoryByteState::Known))
            .unwrap();
        ub.write_states(0x200, &[MemoryByteState::Known(0xaa); 256])
            .unwrap();
        let execution = C220State::new(ScalarStepper::new(machine, 0x4000), ub);
        let rate = NonZeroU64::new(32).unwrap();
        let mut core = C220Core::new(
            execution,
            memory,
            C220CoreTimingRules {
                mte2: C220Mte2TimingRules {
                    issue_interval: NonZeroU64::new(1).unwrap(),
                    startup_ticks: 0,
                    bytes_per_tick: rate,
                    retire_ticks: 0,
                },
                mte3: C220Mte3TimingRules {
                    issue_interval: NonZeroU64::new(1).unwrap(),
                    startup_ticks: 0,
                    bytes_per_tick: rate,
                    retire_ticks: 0,
                },
                vector: C220VectorTimingRules {
                    dispatch_ticks: 0,
                    uop_issue_interval: NonZeroU64::new(1).unwrap(),
                    ub_response_ticks: 1,
                },
            },
        )
        .unwrap();
        let C220CoreStep::Executed {
            instruction: C220CoreInstruction::Vector(C220VectorInstruction::Copy(issue)),
            ..
        } = core.step_word_at(0, word).unwrap()
        else {
            panic!("copy should issue to the vector pipeline");
        };
        assert_eq!(issue.control.source_0_block_stride, 1);
        assert_eq!(
            C220CoreInstruction::Vector(C220VectorInstruction::Copy(issue))
                .as_vector()
                .unwrap()
                .uops()
                .unwrap()[0]
                .stages
                .execute_ticks,
            1
        );
        assert_eq!(
            core.state().ub().read_known(0x200, element_bytes).unwrap(),
            vec![0xaa; element_bytes]
        );
        core.advance_to(100).unwrap();
        let ub = core.state().ub();
        assert_eq!(
            ub.read_known(0x200, element_bytes).unwrap(),
            source[..element_bytes]
        );
        let selected_offset = (selected_lane * element_bytes) as u64;
        assert_eq!(
            ub.read_known(0x200 + selected_offset, element_bytes)
                .unwrap(),
            source[128..128 + element_bytes]
        );
        assert_eq!(
            ub.read_known(0x200 + element_bytes as u64, 2).unwrap(),
            [0xaa; 2]
        );
    }
}

#[test]
fn vector_read_samples_ub_after_issue_without_an_implicit_raw_wait() {
    let memory = SparseMemory::new(vec![MemoryRegion::unknown(256)], 512, 512);
    let memory = MappedMemory::bind(memory, &[0x2000]).unwrap();
    let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
    machine.set_xreg(5, C220_CAPTURED_MOVEV_CONTROL).unwrap();
    machine.set_xreg(6, 0x4000_0000).unwrap();
    machine.set_xreg(8, C220_CAPTURED_VADD_CONTROL).unwrap();
    machine.set_xreg(13, 0).unwrap();
    machine.set_xreg(14, 0x200).unwrap();
    machine.set_xreg(16, 0).unwrap();
    machine.set_spr_value(3, 1 << 56).unwrap();
    machine.set_spr_value(100, 32).unwrap();
    machine.set_spr_value(101, 0).unwrap();
    let mut ub = UbMemory::new(1024, 256);
    let ones = 0x3f80_0000_u32
        .to_le_bytes()
        .repeat(64)
        .into_iter()
        .map(MemoryByteState::Known)
        .collect::<Vec<_>>();
    ub.write_states(0, &ones).unwrap();
    ub.write_states(0x200, &ones).unwrap();
    let execution = C220State::new(ScalarStepper::new(machine, 0x4000), ub);
    let rate = NonZeroU64::new(32).unwrap();
    let mut core = C220Core::new(
        execution,
        memory,
        C220CoreTimingRules {
            mte2: C220Mte2TimingRules {
                issue_interval: NonZeroU64::new(1).unwrap(),
                startup_ticks: 0,
                bytes_per_tick: rate,
                retire_ticks: 0,
            },
            mte3: C220Mte3TimingRules {
                issue_interval: NonZeroU64::new(1).unwrap(),
                startup_ticks: 0,
                bytes_per_tick: rate,
                retire_ticks: 0,
            },
            vector: C220VectorTimingRules {
                dispatch_ticks: 1,
                uop_issue_interval: NonZeroU64::new(20).unwrap(),
                ub_response_ticks: 2,
            },
        },
    )
    .unwrap();
    assert!(matches!(
        core.step_word_at(0, C220_CAPTURED_MOVEV_WORD).unwrap(),
        C220CoreStep::Executed { .. }
    ));
    let movev_visible = core.vector_pipeline().pending_visibility_tick().unwrap();
    assert!(movev_visible < 21);
    core.state
        .scalar_mut()
        .machine_mut()
        .set_xreg(16, 0x400)
        .unwrap();
    assert!(matches!(
        core.step_word_at(1, C220_CAPTURED_VADD_WORD).unwrap(),
        C220CoreStep::Executed {
            instruction: C220CoreInstruction::Vector(C220VectorInstruction::Arithmetic(_)),
            ..
        }
    ));
    assert!(core.vector_pipeline().last_read_samples().is_empty());
    assert!(core.state().ub().read_known(0x400, 4).is_err());
    core.advance_to(30).unwrap();
    assert_eq!(
        core.state().ub().read_known(0, 4).unwrap(),
        0x4000_0000_u32.to_le_bytes()
    );
    let sample = &core.vector_pipeline().last_read_samples()[0];
    let last_grant = sample
        .read0_grants
        .iter()
        .chain(&sample.read1_grants)
        .flatten()
        .copied()
        .max()
        .unwrap();
    assert_eq!(sample.tick, last_grant + 6);
    assert_eq!(sample.accesses.len(), 8);
    assert!(sample.accesses.iter().all(|access| access.block_index < 4));
    assert!(sample.source_0_bytes.is_empty());
    assert!(sample.lanes.is_empty());
    assert!(core.vector_pipeline().last_functional_samples().is_empty());
    assert!(core.state().ub().read_known(0x400, 4).is_err());
    let arithmetic_visible = core.vector_pipeline().pending_visibility_tick().unwrap();
    core.advance_to(arithmetic_visible).unwrap();
    let sample = &core.vector_pipeline().last_functional_samples()[0];
    assert_eq!(sample.tick, arithmetic_visible);
    assert_eq!(&sample.source_0_bytes[..4], &0x4000_0000_u32.to_le_bytes());
    assert_eq!(&sample.source_1_bytes[..4], &0x3f80_0000_u32.to_le_bytes());
    assert_eq!(sample.lanes[0].bits, 0x4040_0000);
    assert_eq!(
        core.state().ub().read_known(0x400, 4).unwrap(),
        0x4040_0000_u32.to_le_bytes()
    );
}

#[test]
fn halfword_movev_uses_one_native_128_lane_uop() {
    let memory = SparseMemory::new(vec![MemoryRegion::unknown(256)], 512, 512);
    let memory = MappedMemory::bind(memory, &[0x2000]).unwrap();
    let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
    machine.set_xreg(5, C220_CAPTURED_MOVEV_CONTROL).unwrap();
    machine.set_xreg(6, 0x3c00).unwrap();
    machine.set_xreg(16, 0).unwrap();
    machine.set_spr_value(3, 1 << 56).unwrap();
    machine.set_spr_value(100, 65).unwrap();
    machine.set_spr_value(101, 0).unwrap();
    let execution = C220State::new(ScalarStepper::new(machine, 0x4000), UbMemory::new(256, 256));
    let rate = NonZeroU64::new(32).unwrap();
    let mut core = C220Core::new(
        execution,
        memory,
        C220CoreTimingRules {
            mte2: C220Mte2TimingRules {
                issue_interval: NonZeroU64::new(1).unwrap(),
                startup_ticks: 0,
                bytes_per_tick: rate,
                retire_ticks: 0,
            },
            mte3: C220Mte3TimingRules {
                issue_interval: NonZeroU64::new(1).unwrap(),
                startup_ticks: 0,
                bytes_per_tick: rate,
                retire_ticks: 0,
            },
            vector: C220VectorTimingRules {
                dispatch_ticks: 1,
                uop_issue_interval: NonZeroU64::new(1).unwrap(),
                ub_response_ticks: 2,
            },
        },
    )
    .unwrap();
    let halfword_word = (C220_CAPTURED_MOVEV_WORD & !(7 << 22)) | (1 << 22);
    let step = core.step_word_at(0, halfword_word).unwrap();
    let C220CoreStep::Executed {
        instruction: C220CoreInstruction::Vector(C220VectorInstruction::Move(step)),
        ..
    } = step
    else {
        panic!("expected MOVEV");
    };
    let uops = C220CoreInstruction::Vector(C220VectorInstruction::Move(step))
        .as_vector()
        .unwrap()
        .uops()
        .unwrap();
    assert_eq!(uops.len(), 1);
    assert!(matches!(
        uops[0].kind,
        C220VectorUopKind::LaneSlice {
            first_lane: 0,
            lane_count: 128
        }
    ));
    assert_eq!(core.vector_pipeline().pending_ub_responses(), 1);
    assert!(core.state().ub().read_known(0, 2).is_err());
    let final_visibility = core.vector_pipeline().pending_visibility_tick().unwrap();
    core.advance_to(final_visibility - 1).unwrap();
    assert!(core.state().ub().read_known(0, 2).is_err());
    assert!(core.state().ub().read_known(128, 2).is_err());
    core.advance_to(final_visibility).unwrap();
    assert_eq!(
        core.state().ub().read_known(0, 2).unwrap(),
        0x3c00_u16.to_le_bytes()
    );
    assert_eq!(
        core.state().ub().read_known(128, 2).unwrap(),
        0x3c00_u16.to_le_bytes()
    );
}

#[test]
fn vector_issue_failure_keeps_execution_state_uncommitted() {
    let memory = SparseMemory::new(vec![MemoryRegion::unknown(256)], 512, 512);
    let memory = MappedMemory::bind(memory, &[0x2000]).unwrap();
    let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
    machine.set_xreg(5, C220_CAPTURED_MOVEV_CONTROL).unwrap();
    machine.set_xreg(6, 0x3c00).unwrap();
    machine.set_xreg(16, 0).unwrap();
    machine.set_spr_value(3, 1 << 56).unwrap();
    machine.set_spr_value(100, 1).unwrap();
    machine.set_spr_value(101, 0).unwrap();
    let execution = C220State::new(ScalarStepper::new(machine, 0x4000), UbMemory::new(256, 256));
    let before = execution.clone();
    let rate = NonZeroU64::new(32).unwrap();
    let mut core = C220Core::new(
        execution,
        memory,
        C220CoreTimingRules {
            mte2: C220Mte2TimingRules {
                issue_interval: NonZeroU64::new(1).unwrap(),
                startup_ticks: 0,
                bytes_per_tick: rate,
                retire_ticks: 0,
            },
            mte3: C220Mte3TimingRules {
                issue_interval: NonZeroU64::new(1).unwrap(),
                startup_ticks: 0,
                bytes_per_tick: rate,
                retire_ticks: 0,
            },
            vector: C220VectorTimingRules {
                dispatch_ticks: u64::MAX,
                uop_issue_interval: NonZeroU64::new(1).unwrap(),
                ub_response_ticks: 1,
            },
        },
    )
    .unwrap();
    assert!(matches!(
        core.step_word_at(1, C220_CAPTURED_MOVEV_WORD),
        Err(C220CoreError::VectorRuntime(
            crate::sim::c220::vector::C220VectorRuntimeError::Issue(
                C220VectorPipelineError::TimeOverflow
            )
        ))
    ));
    assert_eq!(core.state(), &before);
    assert_eq!(core.vector_pipeline().pending_uops(), 0);
}

#[test]
fn disabled_dma_retires_without_memory_access_or_bandwidth_delay() {
    use crate::isa::c220::mte::{C220MovInstruction, CAPTURED_C220_MOV_OUT_TO_UB_X_WORD};
    use crate::sim::c220::mte::mte2::{C220Mte2IssueTiming, C220Mte2Result};

    for xm in [0, 0x10, 0x10000] {
        let memory = MappedMemory::bind(
            SparseMemory::new(vec![MemoryRegion::unknown(32)], 32, 32),
            &[0x2000],
        )
        .unwrap();
        let machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
        let state = C220State::new(ScalarStepper::new(machine, 0x4000), UbMemory::new(32, 32));
        let interval = NonZeroU64::new(u64::MAX).unwrap();
        let mut core = C220Core::new(
            state,
            memory,
            C220CoreTimingRules {
                mte2: C220Mte2TimingRules {
                    issue_interval: interval,
                    startup_ticks: u64::MAX,
                    bytes_per_tick: interval,
                    retire_ticks: u64::MAX,
                },
                mte3: C220Mte3TimingRules {
                    issue_interval: interval,
                    startup_ticks: u64::MAX,
                    bytes_per_tick: interval,
                    retire_ticks: u64::MAX,
                },
                vector: C220VectorTimingRules {
                    dispatch_ticks: 0,
                    uop_issue_interval: NonZeroU64::new(1).unwrap(),
                    ub_response_ticks: 1,
                },
            },
        )
        .unwrap();
        for (tick, word) in [
            CAPTURED_C220_MOV_OUT_TO_UB_X_WORD,
            CAPTURED_C220_MOV_UB_TO_OUT_WORD,
        ]
        .into_iter()
        .enumerate()
        {
            let operands = C220MovInstruction::decode(word).unwrap();
            let machine = core.state.scalar_mut().machine_mut();
            machine
                .set_xreg(operands.source_register, u64::MAX)
                .unwrap();
            machine
                .set_xreg(operands.destination_register, u64::MAX)
                .unwrap();
            machine.set_xreg(operands.descriptor_register, xm).unwrap();
            let C220CoreStep::Executed { instruction, .. } =
                core.step_word_at(tick as u64, word).unwrap()
            else {
                panic!("disabled command should issue without a data dependency");
            };
            match instruction {
                C220CoreInstruction::Mte2(issue) => {
                    assert_eq!(issue.timing, C220Mte2IssueTiming::Disabled)
                }
                C220CoreInstruction::Mte3 {
                    ticket: Some(ticket),
                    ..
                } => {
                    assert_eq!(ticket.requests().unwrap().next(), None);
                    assert_eq!(ticket.uop_count, 0);
                    assert_eq!(ticket.modeled_service_ticks, 0);
                    assert_eq!(ticket.retire_tick, 2);
                }
                _ => panic!("expected a DMA command"),
            }
        }
        let outcomes = core.mte2_pipeline().last_outcomes();
        assert_eq!(outcomes.len(), 1);
        assert!(
            matches!(outcomes[0].result, C220Mte2Result::MovOutToUb(result) if result.bytes == 0 && result.segment_count == 0)
        );
        assert_eq!(core.mte2_pipeline().next_mte2_issue_tick(), 0);
        assert_eq!(core.mte3.timing.next_issue_tick(), 0);
        core.state
            .scalar_mut()
            .machine_mut()
            .set_xreg(10, 0)
            .unwrap();
        core.step_word_at(2, C220_MTE3_TO_VECTOR_SET_FLAG_WORD)
            .unwrap();
        core.state
            .scalar_mut()
            .machine_mut()
            .set_xreg(19, 0)
            .unwrap();
        core.step_word_at(3, C220_MTE3_TO_VECTOR_WAIT_FLAG_WORD)
            .unwrap();
        assert_eq!(core.pending_output_retirement_tick(), None);
        assert!(core.memory().read_known_at(0x2000, 32).is_err());
        assert!(core.state().ub().read_known(0, 32).is_err());
        let mut tick = 4;
        for (source_pipe, destination_pipe) in [(1_u32, 5_u32), (1, 4), (5, 1)] {
            let set = 0x40a0_0000 | (source_pipe << 10) | (destination_pipe << 7) | (1 << 18) | 3;
            let wait = set + (1 << 21);
            for _ in 0..2 {
                assert!(matches!(
                    core.step_word_at(tick, set).unwrap(),
                    C220CoreStep::Executed { .. }
                ));
                tick += 1;
            }
            let events = core.state().pending_output_events().collect::<Vec<_>>();
            assert_eq!(events.len(), 2);
            assert!(
                events
                    .iter()
                    .all(|event| event.flag_id == 7 && event.source_pipe == source_pipe as u8)
            );
            for remaining in [1, 0] {
                assert!(matches!(
                    core.step_word_at(tick, wait).unwrap(),
                    C220CoreStep::Executed { .. }
                ));
                tick += 1;
                assert_eq!(core.state().pending_output_events().count(), remaining);
            }
            let pc = core.state().scalar().pc();
            assert!(matches!(
                core.step_word_at(tick, wait).unwrap(),
                C220CoreStep::Stalled(_)
            ));
            assert_eq!(core.state().scalar().pc(), pc);
            tick += 1;
        }
    }
}

#[test]
fn mte3_completion_wait_uses_the_scheduled_request_service() {
    let regions = vec![
        MemoryRegion::unknown(128),
        MemoryRegion::new(8, 0x2000_u64.to_le_bytes().to_vec()).unwrap(),
    ];
    let memory = SparseMemory::new(regions, 256, 256);
    let memory = MappedMemory::bind(memory, &[0x2000, 0x1000]).unwrap();
    let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
    machine.set_xreg(5, C220_CAPTURED_MOVEV_CONTROL).unwrap();
    machine.set_xreg(6, 0x3f80_0000).unwrap();
    machine.set_xreg(16, 0).unwrap();
    machine.set_spr_value(3, 1 << 56).unwrap();
    machine.set_spr_value(100, 32).unwrap();
    machine.set_spr_value(101, 0).unwrap();
    let execution = C220State::new(ScalarStepper::new(machine, 0x4000), UbMemory::new(512, 256));
    let rate = NonZeroU64::new(32).unwrap();
    let mut core = C220Core::new(
        execution,
        memory,
        C220CoreTimingRules {
            mte2: C220Mte2TimingRules {
                issue_interval: NonZeroU64::new(1).unwrap(),
                startup_ticks: 0,
                bytes_per_tick: rate,
                retire_ticks: 0,
            },
            mte3: C220Mte3TimingRules {
                issue_interval: NonZeroU64::new(1).unwrap(),
                startup_ticks: 2,
                bytes_per_tick: rate,
                retire_ticks: 1,
            },
            vector: C220VectorTimingRules {
                dispatch_ticks: 2,
                uop_issue_interval: NonZeroU64::new(1).unwrap(),
                ub_response_ticks: 3,
            },
        },
    )
    .unwrap();
    core.step_word_at(0, C220_CAPTURED_MOVEV_WORD).unwrap();
    core.state
        .scalar_mut()
        .machine_mut()
        .set_xreg(16, 0x80)
        .unwrap();
    core.step_word_at(2, C220_CAPTURED_MOVEV_WORD).unwrap();
    let machine = core.state.scalar_mut().machine_mut();
    machine.set_xreg(16, 0x100).unwrap();
    core.step_word_at(4, C220_CAPTURED_MOVEV_WORD).unwrap();
    let read_ready_tick = core.vector_pipeline().pending_visibility_tick().unwrap();
    core.state
        .scalar_mut()
        .machine_mut()
        .set_xreg(16, 0x180)
        .unwrap();
    core.step_word_at(6, C220_CAPTURED_MOVEV_WORD).unwrap();
    let machine = core.state.scalar_mut().machine_mut();
    machine.set_xreg(8, C220_CAPTURED_VADD_CONTROL).unwrap();
    machine.set_xreg(13, 0).unwrap();
    machine.set_xreg(14, 0x80).unwrap();
    let movev_visible = core.vector_pipeline().pending_visibility_tick().unwrap();
    assert!(read_ready_tick < movev_visible);
    let last_release = movev_visible - 3;
    core.advance_to(last_release).unwrap();
    assert_eq!(core.last_vector_releases().len(), 4);
    assert!(core.state().ub().read_known(0x180, 4).is_err());
    core.advance_to(read_ready_tick).unwrap();
    core.step_word_at(read_ready_tick, C220_CAPTURED_VADD_WORD)
        .unwrap();
    core.state
        .scalar_mut()
        .machine_mut()
        .set_xreg(14, 0)
        .unwrap();
    let set_tick = read_ready_tick + 1;
    core.step_word_at(set_tick, C220_VECTOR_TO_MTE3_SET_FLAG_WORD)
        .unwrap();
    core.state
        .scalar_mut()
        .machine_mut()
        .set_xreg(13, 0)
        .unwrap();
    let vector_retired = core.vector_pipeline().pending_retirement_tick().unwrap();
    assert!(matches!(
        core.step_word_at(set_tick + 1, C220_VECTOR_TO_MTE3_WAIT_FLAG_WORD)
            .unwrap(),
        C220CoreStep::Stalled(C220Stall {
            resume_tick,
            cause: C220StallCause::VectorDependency,
            ..
        }) if resume_tick == vector_retired
    ));
    core.step_word_at(vector_retired, C220_VECTOR_TO_MTE3_WAIT_FLAG_WORD)
        .unwrap();
    let machine = core.state.scalar_mut().machine_mut();
    machine.set_xreg(14, 0x180).unwrap();
    machine.set_xreg(10, 0x2000).unwrap();
    machine.set_xreg(3, 0x40010).unwrap();
    let issue_tick = vector_retired + 1;
    let issued = core
        .step_word_at(issue_tick, CAPTURED_C220_MOV_UB_TO_OUT_WORD)
        .unwrap();
    let C220CoreStep::Executed {
        instruction:
            C220CoreInstruction::Mte3 {
                ticket: Some(ticket),
                ..
            },
        ..
    } = issued
    else {
        panic!("expected a timed MTE3 transfer");
    };
    assert_eq!(ticket.issue_tick, issue_tick);
    assert_eq!(ticket.data_ready_tick, issue_tick + 6);
    assert_eq!(ticket.retire_tick, issue_tick + 7);
    assert_eq!(ticket.uop_count, 1);
    assert!(core.memory().read_known_at(0x2000, 128).is_err());
    assert!(matches!(
        core.step_word_at(issue_tick + 1, 0x40e0_1800).unwrap(),
        C220CoreStep::Stalled(C220Stall {
            resume_tick,
            cause: C220StallCause::Mte3Dependency,
            ..
        }) if resume_tick == ticket.retire_tick
    ));
    core.state
        .scalar_mut()
        .machine_mut()
        .set_xreg(10, 0)
        .unwrap();
    core.step_word_at(issue_tick + 1, C220_MTE3_TO_VECTOR_SET_FLAG_WORD)
        .unwrap();
    core.state
        .scalar_mut()
        .machine_mut()
        .set_xreg(19, 0)
        .unwrap();
    assert!(matches!(
        core.step_word_at(issue_tick + 2, C220_MTE3_TO_VECTOR_WAIT_FLAG_WORD)
            .unwrap(),
        C220CoreStep::Stalled(C220Stall {
            resume_tick,
            cause: C220StallCause::Mte3Dependency,
            ..
        }) if resume_tick == ticket.retire_tick
    ));
    assert!(core.memory().read_known_at(0x2000, 128).is_err());
    assert_eq!(
        core.pending_output_retirement_tick(),
        Some(ticket.retire_tick)
    );
    assert!(core.advance_to(ticket.data_ready_tick).unwrap().is_none());
    assert!(core.last_mte3_outcomes().is_empty());
    assert!(core.memory().read_known_at(0x2000, 128).is_err());
    let updated = 3.0_f32.to_le_bytes().repeat(32);
    for (index, block) in updated.chunks_exact(32).enumerate() {
        core.state
            .ub_mut()
            .write_states(
                0x180 + (index * 32) as u64,
                &block
                    .iter()
                    .copied()
                    .map(MemoryByteState::Known)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
    }
    assert!(matches!(
        core.step_word_at(ticket.data_ready_tick, C220_MTE3_TO_VECTOR_WAIT_FLAG_WORD)
            .unwrap(),
        C220CoreStep::Stalled(C220Stall {
            resume_tick,
            cause: C220StallCause::Mte3Dependency,
            ..
        }) if resume_tick == ticket.retire_tick
    ));
    assert!(core.memory().read_known_at(0x2000, 128).is_err());
    assert!(matches!(
        core.step_word_at(ticket.retire_tick, 0x40e0_1800).unwrap(),
        C220CoreStep::Executed {
            instruction: C220CoreInstruction::Barrier(_),
            ..
        }
    ));
    assert_eq!(core.pending_output_retirement_tick(), None);
    let outcomes = core.last_mte3_outcomes();
    assert_eq!(outcomes.len(), 1);
    assert_eq!(outcomes[0].tick, ticket.retire_tick);
    assert_eq!(outcomes[0].word, CAPTURED_C220_MOV_UB_TO_OUT_WORD);
    assert_eq!(outcomes[0].ticket, ticket);
    assert_eq!(outcomes[0].result.known_bytes, 128);
    assert_eq!(outcomes[0].result.unknown_bytes, 0);
    assert_eq!(core.memory().read_known_at(0x2000, 128).unwrap(), updated);
    core.step_word_at(ticket.retire_tick + 1, C220_MTE3_TO_VECTOR_WAIT_FLAG_WORD)
        .unwrap();
    assert_eq!(core.memory().read_known_at(0x2000, 128).unwrap(), updated);
    core.state
        .scalar_mut()
        .machine_mut()
        .set_xreg(10, 0x2000)
        .unwrap();
    let C220CoreStep::Executed {
        instruction: C220CoreInstruction::Mte3 {
            ticket: Some(next), ..
        },
        ..
    } = core
        .step_word_at(ticket.retire_tick + 2, CAPTURED_C220_MOV_UB_TO_OUT_WORD)
        .unwrap()
    else {
        panic!("a transfer does not require a fresh event");
    };
    let C220CoreStep::Executed {
        instruction:
            C220CoreInstruction::Mte3 {
                ticket: Some(queued),
                ..
            },
        ..
    } = core
        .step_word_at(next.issue_tick + 1, CAPTURED_C220_MOV_UB_TO_OUT_WORD)
        .unwrap()
    else {
        panic!("the next transfer can issue before its predecessor retires");
    };
    assert!(queued.issue_tick < next.retire_tick);
    assert!(queued.retire_tick > next.retire_tick);
    core.state
        .scalar_mut()
        .machine_mut()
        .set_xreg(3, 0)
        .unwrap();
    let C220CoreStep::Executed {
        instruction:
            C220CoreInstruction::Mte3 {
                ticket: Some(disabled),
                ..
            },
        ..
    } = core
        .step_word_at(next.issue_tick + 2, CAPTURED_C220_MOV_UB_TO_OUT_WORD)
        .unwrap()
    else {
        panic!("disabled DMA still enters the ordered retirement queue");
    };
    assert!(disabled.data_ready_tick < next.retire_tick);
    assert_eq!(disabled.retire_tick, queued.retire_tick + 1);
    assert_eq!(core.pending_mte3_commands().count(), 3);
    core.advance_to(next.retire_tick).unwrap();
    assert_eq!(core.last_mte3_outcomes().len(), 1);
    assert_eq!(core.last_mte3_outcomes()[0].ticket, next);
    assert_eq!(core.pending_mte3_commands().count(), 2);
    assert!(matches!(
        core.step_word_at(next.retire_tick, 0x40e0_1800).unwrap(),
        C220CoreStep::Stalled(C220Stall { resume_tick, .. }) if resume_tick == disabled.retire_tick
    ));
    core.advance_to(disabled.retire_tick).unwrap();
    let outcomes = core.last_mte3_outcomes();
    assert_eq!(outcomes.len(), 2);
    assert_eq!(outcomes[0].ticket, queued);
    assert_eq!(outcomes[1].ticket, disabled);
    assert_eq!(outcomes[1].result.bytes, 0);
    assert_eq!(core.pending_mte3_commands().count(), 0);
    assert!(matches!(
        core.step_word_at(disabled.retire_tick, 0x40e0_1800)
            .unwrap(),
        C220CoreStep::Executed {
            instruction: C220CoreInstruction::Barrier(_),
            ..
        }
    ));
}

#[test]
fn vms4v2_merges_four_lists_through_the_vmsu_pipeline() {
    let memory = SparseMemory::new(vec![MemoryRegion::unknown(256)], 512, 512);
    let memory = MappedMemory::bind(memory, &[0x2000]).unwrap();
    let mut ub = UbMemory::new(1024, 512);
    for (list, keys) in [[10.0_f32, 6.0], [10.0, 5.0], [9.0, 4.0], [8.0, 3.0]]
        .into_iter()
        .enumerate()
    {
        for (index, key) in keys.into_iter().enumerate() {
            let mut record = key.to_le_bytes().to_vec();
            record.extend_from_slice(&((list * 10 + index) as u32).to_le_bytes());
            ub.write_states(
                (list * 16 + index * 8) as u64,
                &record
                    .into_iter()
                    .map(MemoryByteState::Known)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        }
    }
    let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
    machine.set_xreg(1, 0x100).unwrap();
    machine
        .set_xreg(2, (2_u64 << 16) | (4_u64 << 32) | (6_u64 << 48))
        .unwrap();
    machine
        .set_xreg(3, 2 | (2_u64 << 16) | (2_u64 << 32) | (2_u64 << 48))
        .unwrap();
    machine.set_xreg(4, 1 | (0xf << 8)).unwrap();
    let execution = C220State::new(ScalarStepper::new(machine, 0x4000), ub);
    let rate = NonZeroU64::new(32).unwrap();
    let mut core = C220Core::new(
        execution,
        memory,
        C220CoreTimingRules {
            mte2: C220Mte2TimingRules {
                issue_interval: NonZeroU64::new(1).unwrap(),
                startup_ticks: 0,
                bytes_per_tick: rate,
                retire_ticks: 0,
            },
            mte3: C220Mte3TimingRules {
                issue_interval: NonZeroU64::new(1).unwrap(),
                startup_ticks: 0,
                bytes_per_tick: rate,
                retire_ticks: 0,
            },
            vector: C220VectorTimingRules {
                dispatch_ticks: 1,
                uop_issue_interval: NonZeroU64::new(1).unwrap(),
                ub_response_ticks: 2,
            },
        },
    )
    .unwrap();
    let word = 0x85c0_0003 | (1 << 17) | (2 << 12) | (3 << 7) | (4 << 2);
    assert!(matches!(
        core.step_word_at(0, word).unwrap(),
        C220CoreStep::Executed {
            instruction: C220CoreInstruction::Vector(C220VectorInstruction::Merge(_)),
            ..
        }
    ));
    let visibility = core.vmsu_pipeline().pending_visibility_tick().unwrap();
    let retirement = core.vmsu_pipeline().pending_drain_tick().unwrap();
    assert!(retirement > visibility);
    let initial_trace = &core.vmsu_pipeline().trace().unwrap().repeats[0];
    assert_eq!(initial_trace.completion_tick, None);
    assert!(initial_trace.ub_cycles.is_empty());
    assert!(initial_trace.comparisons.is_empty());
    assert!(matches!(
        core.step_word_at(1, word).unwrap(),
        C220CoreStep::Stalled(C220Stall {
            resume_tick,
            cause: C220StallCause::VectorDependency,
            ..
        }) if resume_tick == retirement
    ));
    let mut standalone = core.vmsu_pipeline().clone();
    let mut standalone_state = core.state().clone();
    standalone
        .advance_to(retirement, &mut standalone_state)
        .unwrap();
    core.advance_to(3).unwrap();
    let partial_trace = &core.vmsu_pipeline().trace().unwrap().repeats[0];
    assert_eq!(partial_trace.completion_tick, None);
    assert!(!partial_trace.ub_cycles.is_empty());
    assert!(partial_trace.ub_cycles.iter().all(|cycle| cycle.tick <= 3));
    assert!(partial_trace.write_groups.is_empty());
    core.advance_to(visibility).unwrap();
    let output = core.state().ub().read_known(0x100, 64).unwrap();
    let payloads = output
        .chunks_exact(8)
        .map(|record| u32::from_le_bytes(record[4..8].try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(payloads, [0, 10, 20, 30, 1, 11, 21, 31]);
    assert!(matches!(
        core.step_word_at(visibility, 0x8040_0000).unwrap(),
        C220CoreStep::Executed {
            instruction: C220CoreInstruction::Vector(C220VectorInstruction::Movemask(_)),
            ..
        }
    ));
    core.advance_to(retirement).unwrap();
    assert!(!core.vmsu_pipeline().is_active());
    assert_eq!(core.vmsu_pipeline().trace(), standalone.trace());
    assert_eq!(core.state().ub(), standalone_state.ub());
    assert_eq!(core.state().scalar().machine().spr_value(17), Some(0));
}
