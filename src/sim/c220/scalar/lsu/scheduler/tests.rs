use super::super::miss_buffer::C220LsuMissConfig;
use super::super::store_buffer::{C220LsuMemory, C220LsuStoreConfig};
use super::*;

#[test]
fn response_ownership_controls_linked_completion_order_and_data() {
    for memory in [C220LsuMemory::Ub, C220LsuMemory::External] {
        let mut scheduler = C220LsuRequestScheduler::new(
            4,
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
        scheduler
            .misses
            .push(line, load, &mut scheduler.stores)
            .unwrap();
        let before = scheduler.clone();
        assert!(scheduler.complete_miss_read(line, &[0; 8], None).is_err());
        assert_eq!(scheduler, before);
        let mut cache = [0x33; 64];
        let completion = scheduler
            .complete_miss_read(line, &[0x33; 64], Some(&mut cache))
            .unwrap();
        assert_eq!(completion.load_line, [0x33; 64]);
        assert_eq!(cache[7], 0xaa);
        assert_eq!(cache[6], 0x33);
        assert!(scheduler.misses.entries().is_empty());
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
            let written = scheduler
                .stores
                .complete_ub_write(line, true, &mut ub, &mut scheduler.misses)
                .unwrap();
            assert_eq!(written.notifications, vec![C220LsuCompletion::Store(store)]);
            assert_eq!(ub, cache);
        }
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
            .store(line, store, 9, &[0xbb], false)
            .unwrap();
        scheduler
            .stores
            .set_state(line, C220LsuStoreState::Fetching)
            .unwrap();
        scheduler
            .misses
            .push(line, load, &mut scheduler.stores)
            .unwrap();
        let completion = scheduler
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
        let expected = vec![
            C220LsuCompletion::Store(store),
            C220LsuCompletion::Load(load),
        ];
        assert_eq!(completion.load_line[9], 0xbb);
        assert_eq!(completion.load_line[8], 0x55);
        if memory == C220LsuMemory::External {
            assert_eq!(completion.notifications, expected);
            assert_eq!(completion.load_line, cache);
            assert!(completion.mark_cache_dirty);
        } else {
            assert!(completion.notifications.is_empty());
            assert!(scheduler.stores.entry(line).unwrap().forbidden());
            assert!(scheduler.misses.entry(line).is_some());
            let mut ub = [0x66; 64];
            let completed = scheduler
                .stores
                .complete_ub_write(line, true, &mut ub, &mut scheduler.misses)
                .unwrap();
            assert_eq!(completed.notifications, expected);
            assert_eq!(completed.line, completion.load_line);
        }
        assert!(scheduler.misses.entries().is_empty());
        assert!(scheduler.stores.entries().is_empty());
    }
}

#[test]
fn live_miss_hazards_replay_stores_until_the_line_is_released() {
    let mut scheduler = C220LsuRequestScheduler::new(
        4,
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
        eviction_conflict: false,
        direct_store_full: false,
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
}
