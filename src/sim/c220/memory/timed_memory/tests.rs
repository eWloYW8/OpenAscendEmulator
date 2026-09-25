use super::*;

#[test]
fn reads_share_ddr_credits_and_pending_slots_with_writes() {
    for (slots, limit) in [(1, 1025), (8, 513)] {
        let timing = C220MemoryRegionTiming {
            read: C220MemoryLatency {
                minimum: 1,
                spread: 5,
            },
            dbid: C220MemoryLatency::fixed(2),
            completion: C220MemoryLatency {
                minimum: 2,
                spread: 2,
            },
        };
        let credits = C220MemoryCredits { limit, refill: 128 };
        let mut memory = C220TimedMemory::new(
            C220TimedMemoryConfig {
                input_capacity: NonZeroU32::new(2).unwrap(),
                pending_limit: NonZeroU32::new(slots).unwrap(),
                credit_period: NonZeroU64::new(4).unwrap(),
                ddr_credits: credits,
                l2_write_credits: credits,
                l2_read_credits: credits,
                ddr: timing,
                l2: timing,
                l2_start: 4096,
                l2_bytes: 4096,
            },
            0,
        )
        .unwrap();
        let tag = NonZeroU32::new(1).unwrap();
        let mut shared_reads = memory.clone();
        let sources = [
            C220MemoryReadId::Mte(tag),
            C220MemoryReadId::DataCache {
                port: 0,
                transaction: 1,
            },
            C220MemoryReadId::InstructionCache {
                port: 0,
                transaction: 1,
            },
        ];
        let mut received = Vec::new();
        let mut sent = 0;
        for tick in 0..20 {
            if sent < sources.len() && shared_reads.can_push_read() {
                shared_reads
                    .push_read(
                        tick,
                        C220MemoryReadCommand {
                            ready_tick: tick,
                            tag: sources[sent],
                            address: 0,
                            bytes: 64,
                        },
                    )
                    .unwrap();
                sent += 1;
            }
            shared_reads.advance(tick).unwrap();
            while let Some(beat) = shared_reads.read_front(tick) {
                assert_eq!(beat.transaction_id, 0);
                received.push(beat.tag);
                shared_reads.pop_read();
            }
        }
        assert_eq!(received, sources);
        assert!(shared_reads.is_idle());
        memory.transactions.insert(
            C220MemoryWriteId::Mte(tag),
            Transaction {
                region: 0,
                bytes: 128,
            },
        );
        memory
            .push_data(
                0,
                C220MemoryWriteTransfer {
                    ready_tick: 0,
                    tag: C220MemoryWriteId::Mte(tag),
                },
            )
            .unwrap();
        let read = C220MemoryReadCommand {
            ready_tick: 0,
            tag: C220MemoryReadId::Mte(tag),
            address: 0,
            bytes: 512,
        };
        memory.push_read(0, read).unwrap();
        memory.advance(1).unwrap();
        assert_eq!(memory.pending_completions(), 4);
        assert_eq!(memory.input_occupancy(), [0, 1, 0]);
        assert_eq!(memory.credits(), [limit - 512, limit, limit]);
        memory.advance(4).unwrap();
        assert_eq!(memory.pending_completions(), 1);
        assert_eq!(memory.input_occupancy(), [0; 3]);
        assert_eq!(memory.ready_reads().len(), 4);
        assert_eq!(memory.read_front(4), None);
        for id in 0..4 {
            assert_eq!(
                memory.read_front(5),
                Some(C220MemoryReadBeat {
                    tag: read.tag,
                    transaction_id: id
                })
            );
            memory.pop_read();
        }
        memory.advance(7).unwrap();
        assert_eq!(memory.pending_completions(), 0);
        memory.pop(C220BiuWriteReturnKind::Completion);
        assert!(memory.is_idle());
        let mut l2_read = read;
        l2_read.address = 4096;
        memory.push_read(7, l2_read).unwrap();
        memory.advance(8).unwrap();
        assert_eq!(memory.credits(), [limit - 384, limit, limit - 512]);
    }
}

#[test]
fn exact_credit_boundary_and_service_slot_release() {
    assert_eq!(C220MemoryLatency::fixed(0).deterministic_ticks(), 0);
    assert_eq!(
        C220MemoryLatency {
            minimum: u32::MAX,
            spread: 2
        }
        .deterministic_ticks(),
        0
    );
    let region = C220MemoryRegionTiming {
        read: C220MemoryLatency::fixed(3),
        dbid: C220MemoryLatency::fixed(2),
        completion: C220MemoryLatency {
            minimum: 0,
            spread: 7,
        },
    };
    let mut memory = C220TimedMemory::new(
        C220TimedMemoryConfig {
            input_capacity: NonZeroU32::new(2).unwrap(),
            pending_limit: NonZeroU32::new(1).unwrap(),
            credit_period: NonZeroU64::new(4).unwrap(),
            ddr_credits: C220MemoryCredits {
                limit: 129,
                refill: 1,
            },
            l2_read_credits: C220MemoryCredits {
                limit: 129,
                refill: 1,
            },
            l2_write_credits: C220MemoryCredits {
                limit: 129,
                refill: 1,
            },
            ddr: region,
            l2: region,
            l2_start: 0,
            l2_bytes: 0,
        },
        0,
    )
    .unwrap();
    let tag = C220MemoryWriteId::Cache {
        port: 0,
        transaction: 1,
    };
    memory.transactions.insert(
        tag,
        Transaction {
            region: 0,
            bytes: 128,
        },
    );
    memory.credits[0] = 128;
    memory
        .push_data(0, C220MemoryWriteTransfer { ready_tick: 0, tag })
        .unwrap();
    memory.advance(1).unwrap();
    assert_eq!(memory.input_occupancy(), [0, 1, 0]);
    memory.advance(4).unwrap();
    assert_eq!(memory.input_occupancy(), [0, 0, 0]);
    assert_eq!(memory.credits(), [1, 129, 129]);
    assert_eq!(memory.pending_completions(), 1);
    memory.advance(7).unwrap();
    assert_eq!(memory.pending_completions(), 0);
    let kind = C220BiuWriteReturnKind::Completion;
    assert_eq!(memory.front(7, kind), None);
    assert_eq!(memory.front(8, kind), Some(tag));
    assert!(!memory.is_idle());
    memory.pop(kind);
    assert!(memory.is_idle());
    let tags = [tag, C220MemoryWriteId::Mte(NonZeroU32::new(1).unwrap())];
    for tag in tags {
        memory
            .push_command(
                8,
                C220MemoryWriteCommand {
                    ready_tick: 8,
                    tag,
                    address: 0,
                    bytes: 1,
                },
            )
            .unwrap();
    }
    memory.advance(9).unwrap();
    memory.advance(11).unwrap();
    for tag in tags {
        assert_eq!(memory.front(12, C220BiuWriteReturnKind::Dbid), Some(tag));
        memory.pop(C220BiuWriteReturnKind::Dbid);
        memory
            .push_data(
                12,
                C220MemoryWriteTransfer {
                    ready_tick: 12,
                    tag,
                },
            )
            .unwrap();
    }
    let mut completed = Vec::new();
    for tick in 13..=20 {
        memory.advance(tick).unwrap();
        while let Some(tag) = memory.front(tick, kind) {
            completed.push(tag);
            memory.pop(kind);
        }
    }
    assert_eq!(completed, tags);
    assert!(memory.is_idle());
}
