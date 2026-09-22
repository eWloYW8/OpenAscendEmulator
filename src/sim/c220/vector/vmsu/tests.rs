use std::num::NonZeroU64;

use super::*;
use crate::Architecture;
use crate::isa::c220::vector::merge::C220MergeWidth;
use crate::memory::sparse::MemoryByteState;
use crate::sim::c220::memory::C220UbCycle;
use crate::sim::c220::vector::ops::merge::C220MergeRecord;
use crate::sim::c220::vector::ops::merge::plan_c220_merge_issue;
use crate::sim::common::scalar::{ScalarMachine, ScalarStepper};

#[test]
fn write_lane_releases_before_transport_and_transport_applies_backpressure() {
    let mut writeback = writeback::Writeback::default();
    let mut ub = UbMemory::new(1024, 512);
    let mut trace = C220VmsuRepeatTrace {
        repeat_index: 0,
        start_tick: 0,
        completion_tick: None,
        consumed: [0; 4],
        read_batches: Vec::new(),
        comparisons: Vec::new(),
        write_groups: Vec::new(),
        ub_cycles: Vec::new(),
    };
    for tick in 0..32 {
        writeback.commit(tick, &mut ub).unwrap();
        writeback.prepare(tick).unwrap();
        if let Some(request) = writeback.request_mut() {
            trace
                .ub_cycles
                .push(C220UbCycle::arbitrate(tick, Some(request), None, None));
        }
        writeback.finish(tick, &mut trace).unwrap();
        if tick < 3 {
            assert!(writeback.has_room());
            writeback.push(
                tick as usize * 4,
                vec![
                    C220MergeRecord {
                        source_list: 0,
                        source_index: 0,
                        key_bits: 0,
                        bytes: [tick as u8 + 1; 8],
                    };
                    4
                ],
                8 + tick * 32,
                tick,
            );
        }
    }
    let groups = &trace.write_groups;
    assert_eq!(groups.len(), 3);
    assert_eq!(groups[0].submission_tick, 1);
    assert_eq!(groups[1].submission_tick, 2);
    assert!(groups[1].submission_tick < groups[0].grant_tick);
    assert_eq!(groups[2].submission_tick, groups[0].grant_tick + 1);
    for group in groups {
        assert_eq!(group.submission_tick, group.lane_departure_tick);
        assert!(group.grant_tick >= group.submission_tick + 7);
        assert_eq!(
            ub.read_known(group.destination, 32).unwrap(),
            vec![(group.first_output / 4 + 1) as u8; 32]
        );
    }
}

#[test]
fn merge_pipeline_executes_snapshots_repeats_and_exhaustion_in_both_widths() {
    for width in [C220MergeWidth::F16, C220MergeWidth::F32] {
        for (repeats, suspend) in [(1, false), (1, true), (2, false)] {
            for destination in [512, 520] {
                let mut ub = UbMemory::new(1024, 512);
                let mut expected = Vec::new();
                for repeat in 0..repeats {
                    for index in 0..12 {
                        let sequence = repeat * 12 + index;
                        let mut record = [0_u8; 8];
                        match width {
                            C220MergeWidth::F16 => {
                                let key = 0x5400_u16 - sequence as u16 * 0x20;
                                record[..2].copy_from_slice(&key.to_le_bytes());
                                record[2..4].fill(0xaa);
                            }
                            C220MergeWidth::F32 => {
                                record[..4]
                                    .copy_from_slice(&(64.0 - sequence as f32).to_le_bytes());
                            }
                        }
                        record[4..].copy_from_slice(&(sequence as u32).to_le_bytes());
                        ub.write_states(sequence as u64 * 8, &record.map(MemoryByteState::Known))
                            .unwrap();
                        if width == C220MergeWidth::F16 {
                            record[2..4].fill(0);
                        }
                        if !suspend || index < 3 {
                            expected.extend(record);
                        }
                    }
                }
                let mut registers = [0; 32];
                registers[1] = destination;
                registers[2] = (3 << 16) | (6 << 32) | (9 << 48);
                registers[3] = 3 | (3 << 16) | (3 << 32) | (3 << 48);
                registers[4] = repeats as u64 | (0xf << 8) | (u64::from(suspend) << 12);
                let word = match width {
                    C220MergeWidth::F16 => 0x8540_0003,
                    C220MergeWidth::F32 => 0x85c0_0003,
                } | (1 << 17)
                    | (2 << 12)
                    | (3 << 7)
                    | (4 << 2);
                let issue = plan_c220_merge_issue(0, word, &registers, &ub).unwrap();
                let mut pipeline = C220VmsuPipeline::new(C220VectorTimingRules {
                    dispatch_ticks: 1,
                    uop_issue_interval: NonZeroU64::new(1).unwrap(),
                    ub_response_ticks: 2,
                });
                pipeline.issue_at(0, issue, &ub).unwrap();
                ub.write_states(4, &[MemoryByteState::Known(0xff); 4])
                    .unwrap();
                if repeats == 2 {
                    ub.write_states(100, &[MemoryByteState::Known(0xfe); 4])
                        .unwrap();
                    expected[100..104].fill(0xfe);
                }
                let machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
                let mut state = C220State::new(ScalarStepper::new(machine, 0), ub);
                for tick in 0..512 {
                    pipeline.advance_to(tick, &mut state).unwrap();
                    if !pipeline.is_active() {
                        break;
                    }
                }
                assert!(!pipeline.is_active());
                assert_eq!(
                    state.ub().read_known(destination, expected.len()).unwrap(),
                    expected
                );
                assert_eq!(
                    state.scalar().machine().spr_value(17),
                    Some(if suspend { 3 } else { 0 })
                );
                let trace = pipeline.trace().unwrap();
                assert_eq!(trace.repeats.len(), repeats);
                for repeat in &trace.repeats {
                    assert!(repeat.completion_tick.is_some());
                    assert_eq!(repeat.consumed, if suspend { [3, 0, 0, 0] } else { [3; 4] });
                    assert_eq!(repeat.comparisons.len(), if suspend { 3 } else { 12 });
                    for group in &repeat.write_groups {
                        assert!(group.submission_tick > group.creation_tick);
                        assert_eq!(
                            group.lane_departure_tick - group.submission_tick,
                            if group.records == 4 { 0 } else { 5 }
                        );
                        assert_eq!(group.done_tick, group.grant_tick + 1);
                    }
                }
            }
        }
    }
}
