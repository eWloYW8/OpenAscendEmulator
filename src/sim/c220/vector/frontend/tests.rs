use super::*;
use crate::architecture::Architecture;
use crate::memory::ub::UbMemory;
use crate::sim::c220::vector::ops::compare::C220CompareMask;
use crate::sim::c220::vector::pipeline::C220VectorTimingRules;
use crate::sim::common::scalar::{ScalarMachine, ScalarStepper};
use std::num::NonZeroU64;

#[test]
fn flags_use_reception_order_and_barriers_wait_for_running_work() {
    use super::super::barrier::VectorBarrierAdmission;
    use crate::isa::flow::{FlagInstruction, PipelineBarrierStep};
    let mut state = C220State::new(
        ScalarStepper::new(
            ScalarMachine::from_pem_initial_state(Architecture::Dav2201),
            0x1000,
        ),
        UbMemory::new(256, 256),
    );
    state
        .scalar_mut()
        .machine_mut()
        .set_xreg(6, 0x1_abcd)
        .unwrap();
    let mut engine = VectorEngine::new(
        C220VectorTimingRules {
            dispatch_ticks: 3,
            uop_issue_interval: NonZeroU64::new(1).unwrap(),
            ub_response_ticks: 1,
        },
        C220CompareMask::from_bits([0; 2]),
        Default::default(),
    );
    let mut events = C220PipelineEvents::default();
    let flag = |source: u32, destination: u32, wait: bool| {
        let word = (2 << 29)
            | ((if wait { 6 } else { 5 }) << 21)
            | (source << 10)
            | (destination << 7)
            | (1 << 17)
            | (6 << 2);
        let step = FlagInstruction::decode(Architecture::Dav2201, word)
            .unwrap()
            .resolve(0x1000, state.scalar().machine().xregs());
        (word, step)
    };
    let (_, incoming_set) = flag(4, 1, false);
    let (wait_word, incoming_wait) = flag(4, 1, true);
    let (set_word, outgoing_set) = flag(1, 2, false);
    let (_, outgoing_wait) = flag(1, 2, true);
    let request = C220VectorRequest::capture(state.scalar(), 0x8000_6380).unwrap();
    engine.enqueue_at(0, 0, request.clone()).unwrap();
    engine.advance_event(1, &mut state, &mut events).unwrap();
    engine
        .enqueue_flag_at(1, 1, wait_word, incoming_wait)
        .unwrap();
    engine.advance_event(2, &mut state, &mut events).unwrap();
    assert_eq!(engine.queued_instructions().len(), 1);
    assert_eq!(engine.outstanding_instructions(), 1);
    let barrier = PipelineBarrierStep::decode(Architecture::Dav2201, 0x1000, 0x40e0_0400).unwrap();
    let VectorBarrierAdmission::Issued(outcome) = engine.issue_barrier_at(2, 2, barrier).unwrap()
    else {
        panic!("barrier should issue");
    };
    assert!(outcome.barrier.requires_idle);
    engine.enqueue_at(3, 3, request).unwrap();
    events.set(100, incoming_set, None, 3);
    engine.advance_event(3, &mut state, &mut events).unwrap();
    assert_eq!(events.last_consumptions()[0].step.flag_id, 0x1_abcd);
    assert_eq!(engine.pending_barriers().len(), 1);
    for tick in 4..=5 {
        engine.advance_event(tick, &mut state, &mut events).unwrap();
        assert_eq!(engine.queued_instructions().len(), 1);
    }
    engine.advance_event(6, &mut state, &mut events).unwrap();
    assert_eq!(engine.pending_barriers().len(), 0);
    engine
        .enqueue_flag_at(6, 4, set_word, outgoing_set)
        .unwrap();
    engine.advance_event(7, &mut state, &mut events).unwrap();
    assert!(events.ready(1, 2).is_empty());
    assert_eq!(events.pending().next().unwrap().predecessor, 3);
    assert_eq!(engine.outstanding_instructions(), 1);
    for tick in 8..=11 {
        engine.advance_event(tick, &mut state, &mut events).unwrap();
    }
    let event = events.consume(5, outgoing_wait, 11).unwrap();
    assert_eq!(event.instruction_id, 4);
    assert_eq!(event.received_tick, 7);
    assert_eq!(event.published_tick, 11);
    assert_eq!(engine.outstanding_instructions(), 0);
}

#[test]
fn outstanding_credit_is_held_from_reception_through_retirement() {
    let mut state = C220State::new(
        ScalarStepper::new(
            ScalarMachine::from_pem_initial_state(Architecture::Dav2201),
            0x1000,
        ),
        UbMemory::new(256, 256),
    );
    let mut engine = VectorEngine::new(
        C220VectorTimingRules {
            dispatch_ticks: 3,
            uop_issue_interval: NonZeroU64::new(1).unwrap(),
            ub_response_ticks: 1,
        },
        C220CompareMask::from_bits([0; 2]),
        C220VectorFrontendConfig {
            issue_queue_depth: NonZeroU32::new(2).unwrap(),
            outstanding_limit: NonZeroU32::new(1).unwrap(),
        },
    );
    let capture =
        |state: &C220State| C220VectorRequest::capture(state.scalar(), 0x8000_6380).unwrap();
    assert!(matches!(
        engine.enqueue_at(0, 0, capture(&state)).unwrap(),
        VectorAdmission::Queued(_)
    ));
    engine
        .advance_event(1, &mut state, &mut Default::default())
        .unwrap();
    assert_eq!(engine.outstanding_instructions(), 1);
    assert_eq!(engine.received_instructions().next().unwrap().ready_tick, 4);
    for id in 1..=2 {
        assert!(matches!(
            engine.enqueue_at(id, id, capture(&state)).unwrap(),
            VectorAdmission::Queued(_)
        ));
        assert_eq!(engine.frontend.next_tick, Some(id + 1));
        engine
            .advance_event(id + 1, &mut state, &mut Default::default())
            .unwrap();
    }
    assert!(matches!(
        engine.enqueue_at(3, 3, capture(&state)).unwrap(),
        VectorAdmission::Stalled(C220Stall {
            cause: C220StallCause::VectorIssueQueueFull,
            ..
        })
    ));
    let fence = engine.instruction_fence();
    let barrier =
        crate::isa::flow::PipelineBarrierStep::decode(Architecture::Dav2201, 0x1000, 0x40e0_0400)
            .unwrap();
    assert!(matches!(
        engine.issue_barrier_at(3, 3, barrier).unwrap(),
        super::super::barrier::VectorBarrierAdmission::Stalled(C220Stall {
            cause: C220StallCause::VectorIssueQueueFull,
            ..
        })
    ));
    assert_eq!(engine.pending_barriers().len(), 0);
    assert_eq!(fence.instruction_id, Some(2));
    for tick in 4..=5 {
        engine
            .advance_event(tick, &mut state, &mut Default::default())
            .unwrap();
        assert_eq!(engine.queued_instructions().len(), 2);
        assert_eq!(engine.outstanding_instructions(), 1);
    }
    engine
        .advance_event(6, &mut state, &mut Default::default())
        .unwrap();
    assert_eq!(engine.retirements[0].instruction_id, 0);
    assert_eq!(engine.queued_instructions().len(), 1);
    assert_eq!(engine.outstanding_instructions(), 1);
    for tick in 7..=16 {
        assert!(engine.fence_retirement_tick(fence).is_some());
        engine
            .advance_event(tick, &mut state, &mut Default::default())
            .unwrap();
    }
    assert_eq!(engine.fence_retirement_tick(fence), None);
    assert_eq!(engine.outstanding_instructions(), 0);
    assert_eq!(
        engine
            .retirements
            .iter()
            .map(|entry| entry.retirement_tick)
            .collect::<Vec<_>>(),
        [6, 11, 16]
    );
}
