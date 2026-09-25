use super::super::cache::C220DataCache;
use super::super::cache::{C220CacheAddressLayout, C220CacheSet, C220CacheTag};
use super::super::miss_buffer::C220LsuMissConfig;
use super::super::store_buffer::{C220LsuCompletion, C220LsuStoreState};
use super::super::store_buffer::{C220LsuMemory, C220LsuStoreConfig};
use super::*;

#[test]
fn maintenance_write_samples_live_data_and_retries_before_resetting_tags() {
    let mut scheduler = C220LsuRequestScheduler::new(
        2,
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
            timeout_ticks: 1,
        })
        .unwrap(),
    )
    .unwrap();
    let mut cache = C220DataCache::new(
        C220CacheAddressLayout::new(6, 0, 6, u64::MAX).unwrap(),
        64,
        vec![
            C220CacheSet::new(
                vec![C220CacheTag {
                    atomic: true,
                    valid: true,
                    dirty: true,
                    age: 7,
                    memory: C220LsuMemory::External,
                    tag: 4,
                }],
                None,
            )
            .unwrap(),
        ],
    )
    .unwrap();
    let location = super::super::cache::C220CacheLocation { index: 0, way: 0 };
    use super::super::cache::C220CacheMaintenanceTarget;
    for (target, cleaned) in [
        (C220CacheMaintenanceTarget::External, false),
        (C220CacheMaintenanceTarget::Atomic, true),
        (C220CacheMaintenanceTarget::All, true),
    ] {
        let mut staged = scheduler.clone();
        let mut ram = cache.clone();
        staged
            .admit_maintenance(0, C220LsuMaintenanceScope::All { target })
            .unwrap()
            .unwrap();
        for tick in 1..5 {
            for stage in [C220LsuStage::M2, C220LsuStage::M1, C220LsuStage::M0] {
                staged
                    .advance_with_cache(
                        stage,
                        tick,
                        C220LsuExternalHazards {
                            maintenance_active: false,
                            maintenance_draining: false,
                        },
                        &mut ram,
                    )
                    .unwrap();
            }
        }
        let completed = staged.take_maintenance_completions();
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].invalidated, [location]);
        assert_eq!(completed[0].writes.len(), usize::from(cleaned));
        assert!(!ram.tag(location).unwrap().valid);
        assert!(ram.tag(location).unwrap().atomic);
    }
    let pending = scheduler
        .invalidate_external_line(0, &mut cache, location)
        .unwrap()
        .unwrap();
    assert_eq!(pending.line.address, 0x100);
    assert!(!cache.tag(location).unwrap().valid);
    assert!(cache.tag(location).unwrap().dirty);
    assert!(
        scheduler
            .invalidate_external_line(0, &mut cache, location)
            .unwrap()
            .is_none()
    );
    scheduler.writes.dispatch_clock(1, false, true).unwrap();
    assert_eq!(scheduler.writes.outstanding(), 1);
    assert!(
        scheduler
            .apply_maintenance_write_response::<C220LsuSchedulerError>(
                pending.write,
                &mut cache,
                |_, _| Err(C220LsuSchedulerError::MissingWriteData),
            )
            .is_err()
    );
    assert_eq!(scheduler.writes.outstanding(), 0);
    assert_eq!(scheduler.maintenance_writes().count(), 1);
    assert!(cache.tag(location).unwrap().dirty);
    cache.line_mut(location).unwrap().fill(0x99);
    scheduler
        .apply_maintenance_write_response::<C220LsuSchedulerError>(
            pending.write,
            &mut cache,
            |key, bytes| {
                assert_eq!(key, pending.line);
                assert_eq!(bytes, &[0x99; 64]);
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(
        cache.tag(location).unwrap(),
        C220CacheTag {
            atomic: false,
            valid: true,
            dirty: false,
            age: 0,
            memory: C220LsuMemory::External,
            tag: 0,
        }
    );
    assert_eq!(scheduler.maintenance_writes().count(), 0);
    assert!(scheduler.writes.requests().next().is_none());
    assert!(
        scheduler
            .invalidate_external_line(2, &mut cache, location)
            .unwrap()
            .is_none()
    );
    assert!(!cache.tag(location).unwrap().valid);
}

#[test]
fn captured_stores_coalesce_and_complete_through_cache_or_refill() {
    use crate::architecture::Architecture;
    use crate::sim::c220::scalar::{C220ScalarMappedAddress, C220StoreOperands};
    use crate::sim::common::scalar::ScalarMachine;
    for (hit, pair) in [(false, false), (true, false), (false, true), (true, true)] {
        let mut scheduler = C220LsuRequestScheduler::new(
            4,
            2,
            2,
            2,
            C220LsuMissBuffer::new(C220LsuMissConfig {
                line_bytes: 64,
                main_entries: 2,
                sub_entries: 4,
            })
            .unwrap(),
            C220LsuStoreBuffer::new(C220LsuStoreConfig {
                line_bytes: 64,
                main_entries: 2,
                sub_entries: 4,
                timeout_ticks: 3,
            })
            .unwrap(),
        )
        .unwrap();
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
        if hit {
            cache
                .refill(0x100, 0x100, C220LsuMemory::External, &[0x55; 64], false)
                .unwrap();
        }
        let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
        machine.set_xreg(5, 0x108).unwrap();
        machine.set_xreg(6, 1).unwrap();
        let first = C220StoreOperands::capture(
            &machine,
            0,
            if pair {
                (9 << 24) | (3 << 22) | (5 << 17) | (5 << 12) | (6 << 7) | 1
            } else {
                (20 << 24) | (3 << 22) | (5 << 17) | (5 << 12) | 8
            },
        )
        .unwrap();
        assert_eq!(first.updated_base, (!pair).then_some(0x110));
        assert_eq!(first.second_source_operand, pair.then_some((6, 1)));
        assert_eq!(first.source_operand, Some((5, 0x108)));
        assert_eq!(machine.xregs()[5], 0x108);
        let second =
            C220StoreOperands::capture(&machine, 4, (14 << 24) | (5 << 12) | (6 << 7) | 2).unwrap();
        assert_eq!(second.effective_address, 0x109);
        let mapped = |operands: C220StoreOperands| C220ScalarMappedAddress {
            address: operands.effective_address,
            memory: C220LsuMemory::External,
            stack: false,
        };
        let (a, a_second) = scheduler
            .admit_store(0, first, mapped(first), false)
            .unwrap()
            .unwrap();
        let (b, b_second) = scheduler
            .admit_store(0, second, mapped(second), false)
            .unwrap()
            .unwrap();
        assert_eq!((a_second, b_second), (None, None));
        machine.set_xreg(5, u64::MAX).unwrap();
        machine.set_xreg(6, u64::MAX).unwrap();
        let hazards = C220LsuExternalHazards {
            maintenance_active: false,
            maintenance_draining: false,
        };
        for tick in 1..=6 {
            scheduler.process_stores(tick, &mut cache, true).unwrap();
            let before = scheduler.clone();
            scheduler.process_stores(tick, &mut cache, true).unwrap();
            assert_eq!(scheduler, before);
            for stage in [C220LsuStage::M2, C220LsuStage::M1, C220LsuStage::M0] {
                scheduler
                    .advance_with_cache(stage, tick, hazards, &mut cache)
                    .unwrap();
            }
            if tick < 6 {
                assert!(scheduler.take_store_values().is_empty());
            }
            if tick == 4 {
                assert_eq!(scheduler.stores.entries()[0].remaining_ticks(), 2);
            }
        }
        let completion_tick = if hit {
            6
        } else {
            assert!(
                scheduler
                    .reads
                    .dispatch_clock(6, false, true)
                    .unwrap()
                    .is_empty()
            );
            let read = scheduler.reads.dispatch_clock(7, false, true).unwrap()[0];
            scheduler
                .apply_read_response::<C220LsuSchedulerError>(
                    read.id,
                    Some(&mut cache),
                    |_, size| Ok(vec![0x55; size]),
                )
                .unwrap();
            7
        };
        let values = scheduler.take_store_values();
        assert_eq!(
            values.iter().map(|value| value.request).collect::<Vec<_>>(),
            [a, b]
        );
        assert!(values.iter().all(|value| value.tick == completion_tick
            && value.path
                == if hit {
                    C220LsuStorePath::Cache
                } else {
                    C220LsuStorePath::Refill
                }));
        let location = cache.find_way(0x100, C220LsuMemory::External).unwrap();
        let mut expected = [0x55; 64];
        expected[8..16].copy_from_slice(&0x108_u64.to_le_bytes());
        if pair {
            expected[16..24].copy_from_slice(&1_u64.to_le_bytes());
        }
        expected[9] = 0xff;
        assert_eq!(cache.line(location).unwrap(), expected);
        assert!(cache.sets()[0].ways()[0].dirty);
        assert!(scheduler.stores.entries().is_empty());
        assert!(scheduler.writes.requests().next().is_none());
    }
}

#[test]
fn response_ownership_controls_linked_completion_order_and_data() {
    for memory in [C220LsuMemory::Ub, C220LsuMemory::External] {
        let mut scheduler = C220LsuRequestScheduler::new(
            4,
            2,
            2,
            2,
            C220LsuMissBuffer::new(C220LsuMissConfig {
                line_bytes: 64,
                main_entries: 2,
                sub_entries: 4,
            })
            .unwrap(),
            C220LsuStoreBuffer::new(C220LsuStoreConfig {
                line_bytes: 64,
                main_entries: 2,
                sub_entries: 4,
                timeout_ticks: 4,
            })
            .unwrap(),
        )
        .unwrap();
        let line = C220LsuLineKey { address: 0, memory };
        let (store, load) = scheduler
            .admit(
                0,
                C220LsuRequest {
                    line,
                    access: C220LsuAccess::Store,
                },
                Some(C220LsuRequest {
                    line,
                    access: C220LsuAccess::Load,
                }),
            )
            .unwrap()
            .unwrap();
        let load = load.unwrap();
        scheduler
            .stores
            .store(line, store, 7, &[0xaa], false)
            .unwrap();
        let read = scheduler
            .enqueue_load_miss(line, load, line.address)
            .unwrap()
            .unwrap();
        assert!(
            scheduler
                .reads
                .dispatch_clock(0, true, true)
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            scheduler.reads.dispatch_clock(1, true, true).unwrap()[0].id,
            read
        );
        assert_eq!(scheduler.reads.outstanding(), 1);
        let before = scheduler.clone();
        assert!(scheduler.complete_miss_read(line, &[0; 8], None).is_err());
        assert_eq!(scheduler, before);
        let mut cache_ram = C220DataCache::new(
            C220CacheAddressLayout::new(6, 0, 6, u64::MAX).unwrap(),
            64,
            vec![
                C220CacheSet::new(
                    vec![C220CacheTag {
                        atomic: false,
                        valid: false,
                        dirty: false,
                        age: 0,
                        memory,
                        tag: 0,
                    }],
                    None,
                )
                .unwrap(),
            ],
        )
        .unwrap();
        cache_ram
            .refill(128, 128, C220LsuMemory::External, &[0x77; 64], true)
            .unwrap();
        let cache_before = cache_ram.clone();
        assert!(
            scheduler
                .complete_cached_miss_read(line, line.address, &[0; 8], &mut cache_ram)
                .is_err()
        );
        assert_eq!(cache_ram, cache_before);
        assert_eq!(scheduler, before);
        assert!(
            scheduler
                .apply_read_response::<C220LsuSchedulerError>(read, Some(&mut cache_ram), |_, _| {
                    Err(C220LsuSchedulerError::MissingCacheLine)
                })
                .is_err()
        );
        assert_eq!(scheduler.reads.outstanding(), 0);
        assert!(scheduler.reads.request(read).is_some());
        let (completion, refill) = scheduler
            .apply_read_response::<C220LsuSchedulerError>(
                read,
                Some(&mut cache_ram),
                |key, size| {
                    assert_eq!(key, line);
                    assert_eq!(size, 64);
                    Ok(vec![0x33; size])
                },
            )
            .unwrap();
        assert!(scheduler.reads.request(read).is_none());
        let refill = refill.unwrap();
        let evicted = refill.writeback.unwrap();
        assert_eq!(evicted.address, 128);
        assert_eq!(evicted.bytes, [0x77; 64]);
        assert_eq!(evicted.memory, C220LsuMemory::External);
        let mut cache: [u8; 64] = cache_ram.line(refill.location).unwrap().try_into().unwrap();
        assert_eq!(
            cache_ram.sets()[0].ways()[0].dirty,
            memory == C220LsuMemory::External
        );
        assert_eq!(
            cache_ram.lookup(line.address, memory),
            Some(refill.location)
        );
        assert_eq!(completion.load_line, [0x33; 64]);
        assert_eq!(cache[7], 0xaa);
        assert_eq!(cache[6], 0x33);
        assert!(scheduler.misses.entries().is_empty());
        assert_eq!(scheduler.evicted_line(128), Some([0x77; 64].as_slice()));
        let sent = scheduler.writes.dispatch_clock(1, true, true).unwrap();
        let eviction = sent
            .iter()
            .find(|request| request.line.memory == C220LsuMemory::External)
            .unwrap();
        let mut old_backing = [0; 64];
        scheduler
            .complete_eviction_write(eviction.id, &mut old_backing)
            .unwrap();
        assert_eq!(old_backing, [0x77; 64]);
        assert!(scheduler.evicted_line(128).is_none());
        if memory == C220LsuMemory::External {
            assert_eq!(
                completion.notifications,
                vec![
                    C220LsuCompletion::Load(load),
                    C220LsuCompletion::Store(store)
                ]
            );
            assert!(completion.mark_cache_dirty);
            assert!(completion.pending_ub_write.is_none());
            assert!(scheduler.stores.entries().is_empty());
        } else {
            assert_eq!(
                completion.notifications,
                vec![C220LsuCompletion::Load(load)]
            );
            assert!(!completion.mark_cache_dirty);
            assert_eq!(
                completion.pending_ub_write.as_deref(),
                Some(cache.as_slice())
            );
            assert_eq!(
                scheduler.stores.entry(line).unwrap().state(),
                C220LsuStoreState::Ready
            );
            let mut ub = [0x44; 64];
            let ub_id = sent
                .iter()
                .find(|request| request.line.memory == C220LsuMemory::Ub)
                .unwrap()
                .id;
            let written = scheduler
                .complete_ub_store_write(ub_id, true, &mut ub)
                .unwrap();
            assert_eq!(written.notifications, vec![C220LsuCompletion::Store(store)]);
            assert_eq!(ub, cache);
        }
        let (store, load) = scheduler
            .admit(
                1,
                C220LsuRequest {
                    line,
                    access: C220LsuAccess::Store,
                },
                Some(C220LsuRequest {
                    line,
                    access: C220LsuAccess::Load,
                }),
            )
            .unwrap()
            .unwrap();
        let load = load.unwrap();
        scheduler
            .stores
            .store(line, store, 9, &[0xbb], false)
            .unwrap();
        let read = scheduler.enqueue_store_read(line, line.address).unwrap();
        assert_eq!(
            scheduler
                .enqueue_load_miss(line, load, line.address)
                .unwrap(),
            None
        );
        assert_eq!(scheduler.reads.requests().count(), 1);
        assert_eq!(
            scheduler.reads.dispatch_clock(2, true, true).unwrap()[0].id,
            read
        );
        let mut buffer_only = scheduler.clone();
        let expected_completion = buffer_only
            .complete_store_read(
                line,
                &[0x55; 64],
                if memory == C220LsuMemory::External {
                    Some(&mut cache)
                } else {
                    None
                },
            )
            .unwrap();
        let before = scheduler.clone();
        let cache_before = cache_ram.clone();
        assert!(
            scheduler
                .complete_cached_store_read(line, line.address, &[0; 8], &mut cache_ram)
                .is_err()
        );
        assert_eq!(scheduler, before);
        assert_eq!(cache_ram, cache_before);
        let (completion, refill) = scheduler
            .apply_read_response::<C220LsuSchedulerError>(read, Some(&mut cache_ram), |_, size| {
                Ok(vec![0x55; size])
            })
            .unwrap();
        assert_eq!(completion, expected_completion);
        let refill = refill.unwrap();
        assert_eq!(scheduler.misses, buffer_only.misses);
        assert_eq!(scheduler.stores, buffer_only.stores);
        assert_eq!(
            cache_ram.line(refill.location).unwrap(),
            completion.load_line
        );
        assert_eq!(
            cache_ram.sets()[0].ways()[0].dirty,
            memory == C220LsuMemory::External
        );
        assert_eq!(
            refill.writeback.is_some(),
            memory == C220LsuMemory::External
        );
        let expected = vec![
            C220LsuCompletion::Store(store),
            C220LsuCompletion::Load(load),
        ];
        assert_eq!(completion.load_line[9], 0xbb);
        assert_eq!(completion.load_line[8], 0x55);
        let sent = scheduler.writes.dispatch_clock(2, true, true).unwrap();
        assert_eq!(sent.len(), 1);
        if memory == C220LsuMemory::External {
            assert_eq!(completion.notifications, expected);
            assert_eq!(completion.load_line, cache);
            assert!(completion.mark_cache_dirty);
            let mut backing = [0; 64];
            scheduler
                .complete_eviction_write(sent[0].id, &mut backing)
                .unwrap();
            assert_eq!(backing[7], 0xaa);
            assert_eq!(backing[9], 0x33);
        } else {
            assert!(completion.notifications.is_empty());
            assert!(scheduler.stores.entry(line).unwrap().forbidden());
            assert!(scheduler.misses.entry(line).is_some());
            let mut ub = [0x66; 64];
            let completed = scheduler
                .complete_ub_store_write(sent[0].id, true, &mut ub)
                .unwrap();
            assert_eq!(completed.notifications, expected);
            assert_eq!(completed.line, completion.load_line);
        }
        assert!(scheduler.misses.entries().is_empty());
        assert!(scheduler.stores.entries().is_empty());
        assert_eq!(scheduler.writes.outstanding(), 0);
        assert_eq!(scheduler.writes.requests().count(), 0);
        if memory == C220LsuMemory::Ub {
            let write = scheduler.writes.enqueue(line).unwrap();
            scheduler.writes.dispatch_clock(3, true, false).unwrap();
            cache_ram.line_mut(refill.location).unwrap()[9] = 0xcc;
            let mut backing = [0; 64];
            assert_eq!(
                scheduler
                    .complete_ub_write_response(write, true, &mut backing, &cache_ram)
                    .unwrap(),
                None
            );
            assert_eq!(backing[9], 0xcc);
            assert!(!scheduler.writes.has_hazard(line));
        }
    }
}

#[test]
fn live_miss_hazards_replay_stores_until_the_line_is_released() {
    let mut scheduler = C220LsuRequestScheduler::new(
        4,
        2,
        2,
        2,
        C220LsuMissBuffer::new(C220LsuMissConfig {
            line_bytes: 64,
            main_entries: 2,
            sub_entries: 4,
        })
        .unwrap(),
        C220LsuStoreBuffer::new(C220LsuStoreConfig {
            line_bytes: 64,
            main_entries: 2,
            sub_entries: 4,
            timeout_ticks: 4,
        })
        .unwrap(),
    )
    .unwrap();
    let external = C220LsuExternalHazards {
        maintenance_active: false,
        maintenance_draining: false,
    };
    let line = C220LsuLineKey {
        address: 0,
        memory: C220LsuMemory::Ub,
    };
    let load = C220LsuRequest {
        line,
        access: C220LsuAccess::Load,
    };
    let store = C220LsuRequest {
        line,
        access: C220LsuAccess::Store,
    };
    let (load_id, store_id) = scheduler.admit(0, load, Some(store)).unwrap().unwrap();
    let store_id = store_id.unwrap();
    scheduler.advance(C220LsuStage::M0, 1, external).unwrap();
    scheduler.advance(C220LsuStage::M1, 2, external).unwrap();
    scheduler.advance(C220LsuStage::M0, 2, external).unwrap();
    assert_eq!(
        scheduler
            .advance(C220LsuStage::M2, 3, external)
            .unwrap()
            .consumed,
        Some(load)
    );
    scheduler
        .misses
        .push(line, load_id, &mut scheduler.stores)
        .unwrap();
    let outcome = scheduler.advance(C220LsuStage::M1, 3, external).unwrap();
    assert_eq!(
        outcome.progress,
        C220LsuStageProgress::ReplayScheduled(store_id)
    );
    assert_eq!(outcome.stall, Some(C220LsuStall::PendingLoad));
    assert_eq!(outcome.consumed, None);
    assert_eq!(scheduler.pipeline().queued_requests(), 1);
    assert_eq!(scheduler.request(store_id), Some(&store));
    scheduler.misses.receive_line(line, &[0; 64]).unwrap();
    assert_eq!(
        scheduler.hazard(load, external),
        Some(C220LsuStall::MissForbidden)
    );
    assert_eq!(
        scheduler
            .advance(C220LsuStage::M0, 4, external)
            .unwrap()
            .progress,
        C220LsuStageProgress::Blocked(store_id)
    );
    scheduler.misses.remove(line).unwrap();
    assert_eq!(
        scheduler
            .advance(C220LsuStage::M0, 5, external)
            .unwrap()
            .progress,
        C220LsuStageProgress::Advanced(store_id)
    );
    scheduler.advance(C220LsuStage::M1, 6, external).unwrap();
    assert_eq!(
        scheduler
            .advance(C220LsuStage::M2, 7, external)
            .unwrap()
            .consumed,
        Some(store)
    );
    assert_eq!(scheduler.pipeline().queued_requests(), 0);
    assert!(scheduler.request(store_id).is_none());
    let external_line = C220LsuLineKey {
        memory: C220LsuMemory::External,
        ..line
    };
    let external_write = scheduler.writes.enqueue(external_line).unwrap();
    let next_write = scheduler.writes.enqueue(external_line).unwrap();
    assert_eq!(scheduler.writes.next_ready_tick(), Some(8));
    assert!(
        scheduler
            .writes
            .dispatch_clock(7, true, true)
            .unwrap()
            .is_empty()
    );
    scheduler.writes.advance_to(8).unwrap();
    let ub_write = scheduler.writes.enqueue(line).unwrap();
    assert_eq!(
        scheduler.hazard(load, external),
        Some(C220LsuStall::Eviction)
    );
    assert_eq!(scheduler.writes.outstanding(), 0);
    let sent = scheduler.writes.dispatch_clock(8, true, true).unwrap();
    assert_eq!(
        sent.iter().map(|r| r.id).collect::<Vec<_>>(),
        vec![ub_write, external_write]
    );
    assert_eq!(scheduler.writes.outstanding(), 2);
    assert!(
        scheduler
            .writes
            .dispatch_clock(8, true, true)
            .unwrap()
            .is_empty()
    );
    assert!(scheduler.writes.begin_response(next_write).is_err());
    scheduler.writes.begin_response(ub_write).unwrap();
    assert_eq!(scheduler.writes.outstanding(), 1);
    assert_eq!(
        scheduler.hazard(load, external),
        Some(C220LsuStall::Eviction)
    );
    scheduler.writes.finish_response(ub_write).unwrap();
    assert_eq!(scheduler.hazard(load, external), None);
    assert!(scheduler.writes.has_hazard(external_line));
    assert!(
        scheduler
            .writes
            .dispatch_clock(8, false, true)
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        scheduler.writes.dispatch_clock(9, false, true).unwrap()[0].id,
        next_write
    );
    scheduler.writes.begin_response(external_write).unwrap();
    scheduler.writes.finish_response(external_write).unwrap();
    assert!(scheduler.writes.has_hazard(external_line));
    scheduler.writes.begin_response(next_write).unwrap();
    scheduler.writes.finish_response(next_write).unwrap();
    assert!(!scheduler.writes.has_hazard(external_line));
    assert_eq!(scheduler.writes.outstanding(), 0);
    let layout = C220CacheAddressLayout::new(6, 3, 8, 0xffffffffff).unwrap();
    let mut machine = crate::sim::common::scalar::ScalarMachine::from_pem_initial_state(
        crate::architecture::Architecture::Dav2201,
    );
    machine.set_xreg(1, 0xffff_0000_0000_0006).unwrap();
    machine.set_xreg(2, 0x12ab).unwrap();
    let operands = crate::sim::c220::scalar::C220DirectStoreOperands::capture(
        &machine,
        0x100,
        0x1800_0000 | (2 << 17) | (1 << 12) | 0xffd,
    )
    .unwrap();
    machine.set_xreg(2, 0).unwrap();
    assert_eq!(operands.bytes(), &[0xab]);
    assert_eq!(operands.effective_address, 0xffff_0000_0000_0003);
    assert_eq!(machine.xregs()[1], operands.base_value);
    scheduler
        .execute_direct_store(9, store_id, &operands, layout)
        .unwrap();
    scheduler
        .push_direct_store(9, 5, load_id, &[0xcd], layout)
        .unwrap();
    assert_eq!(scheduler.next_direct_store_tick(), Some(10));
    assert!(scheduler.process_direct_stores(9).unwrap().is_empty());
    assert!(scheduler.direct_stores.full());
    assert_eq!(
        scheduler.hazard(store, external),
        Some(C220LsuStall::DirectStoreCapacity)
    );
    let generated = scheduler.process_direct_stores(10).unwrap();
    assert_eq!(generated.len(), 2);
    assert!(scheduler.process_direct_stores(10).unwrap().is_empty());
    assert_eq!(scheduler.next_direct_store_tick(), None);
    let first = generated[0];
    let second = generated[1];
    assert!(
        scheduler
            .writes
            .dispatch_clock(10, false, true)
            .unwrap()
            .is_empty()
    );
    let sent = scheduler.writes.dispatch_clock(11, false, true).unwrap();
    assert_eq!(sent[0].byte_len, 64);
    assert_eq!(sent[0].line.address, 0);
    let mut backing = [0xff; 64];
    assert_eq!(
        scheduler.apply_external_write_response::<C220LsuSchedulerError>(first, |_, _| {
            Err(C220LsuSchedulerError::MissingCacheLine)
        }),
        Err(C220LsuSchedulerError::MissingCacheLine)
    );
    assert_eq!(scheduler.writes.outstanding(), 0);
    assert_eq!(scheduler.direct_stores.entries().len(), 2);
    assert!(scheduler.writes.has_hazard(external_line));
    assert_eq!(
        scheduler
            .complete_external_write_response(first, &mut backing)
            .unwrap(),
        Some(C220LsuCompletion::Store(store_id))
    );
    assert_eq!(backing[3], 0xab);
    assert_eq!(backing[5], 0);
    assert!(!scheduler.direct_stores.full());
    assert_eq!(scheduler.direct_stores.entries().len(), 1);
    scheduler.writes.dispatch_clock(12, false, true).unwrap();
    assert_eq!(
        scheduler
            .complete_external_write_response(second, &mut backing)
            .unwrap(),
        Some(C220LsuCompletion::Store(load_id))
    );
    assert_eq!(backing[3], 0);
    assert_eq!(backing[5], 0xcd);
    assert!(scheduler.direct_stores.entries().is_empty());
    assert_eq!(scheduler.writes.outstanding(), 0);
    scheduler
        .push_direct_store(12, 64, store_id, &[1], layout)
        .unwrap();
    let before = scheduler.clone();
    assert!(scheduler.advance(C220LsuStage::M0, 14, external).is_err());
    assert_eq!(scheduler, before);
    assert_eq!(scheduler.process_direct_stores(13).unwrap().len(), 1);
    scheduler
        .push_direct_store(13, 128, load_id, &[2], layout)
        .unwrap();
    let repeated = scheduler.process_direct_stores(14).unwrap();
    assert_eq!(repeated.len(), 2);
    assert_eq!(
        scheduler.writes.request(repeated[0]).unwrap().line.address,
        64
    );
    assert_eq!(
        scheduler.writes.request(repeated[1]).unwrap().line.address,
        128
    );
}
