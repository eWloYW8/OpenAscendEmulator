use super::*;
use crate::isa::c220::mte::load2d::C220Load2dInstruction;
use crate::sim::c220::mte::interface::C220L0WritePipeline;
use crate::sim::c220::mte::mte1::load2d::C220Load2dRequestPlan;
use std::num::NonZeroU32;

#[test]
fn blocked_load2d_holds_bt_until_output_tail_and_l0_confirms_separately() {
    let mut registers = [0; 32];
    registers[0] = u64::MAX - 255;
    registers[2] = 32;
    registers[3] = 1 << 16;
    let transfer = C220Load2dInstruction::decode(0x6000_2180)
        .unwrap()
        .capture(&registers)
        .unwrap();
    let requests = C220Load2dRequestPlan::new(transfer, NonZeroU32::new(96).unwrap())
        .unwrap()
        .collect::<Vec<_>>();
    assert_eq!(
        requests.iter().filter(|r| r.completes_logical_uop).count(),
        1
    );
    let output_plan = requests
        .last()
        .unwrap()
        .outputs(7, 10, NonZeroU32::new(256).unwrap());
    let mut output = C220MteL1Output::default();
    let mut l0a = C220L0WritePipeline::default();
    output
        .receive(
            0,
            C220MteL1OutputDestination::L0a(C220L0WritePort::Port0),
            output_plan,
            10,
        )
        .unwrap();
    assert_eq!(output.queue_state().output_fragments, 0);
    let mut sent = Vec::new();
    let mut l0_retirement = None;
    let mut bt_retirement = None;
    for tick in 0..15 {
        if tick == 1 {
            output
                .receive(
                    1,
                    C220MteL1OutputDestination::Bt,
                    C220MteOutputPlan::new(8, 11, 0x1000, 64, true, NonZeroU32::new(128).unwrap()),
                    11,
                )
                .unwrap();
        }
        if tick == 2 {
            let before = output.clone();
            assert!(matches!(
                output.step(4, C220MteL1OutputCredits::default()),
                Err(C220MteL1OutputError::InvalidCycle { .. })
            ));
            assert_eq!(output, before);
        }
        let l0_cycle = l0a.step(tick).unwrap();
        if let Some(id) = l0_cycle.retired_instruction {
            assert_eq!(id, 7);
            assert!(l0_retirement.replace(tick).is_none());
        }
        let credits = C220MteL1OutputCredits {
            l0a: [
                tick >= 4 && l0a.can_push(C220L0WritePort::Port0),
                false,
                false,
            ],
            ..Default::default()
        };
        let cycle = output.step(tick, credits).unwrap();
        if (1..4).contains(&tick) {
            assert_eq!(
                cycle.blocked,
                Some(C220MteL1OutputDestination::L0a(C220L0WritePort::Port0))
            );
            assert_eq!(cycle.queues.output_fragments, 2);
            assert_eq!(cycle.queues.acknowledged, 2);
            assert!(cycle.sent.is_none());
        }
        if let Some(event) = cycle.sent {
            sent.push((tick, event.payload, event.fragment.destination_address));
            if let C220MteL1OutputDestination::L0a(port) = event.destination {
                assert!(l0a.push(tick, port, event.fragment).unwrap());
            }
        }
        if let Some(event) = cycle.retired {
            assert_eq!(event.payload, 11);
            assert_eq!(event.fragment.instruction_id, 8);
            assert!(event.fragment.last_in_instruction);
            assert!(bt_retirement.replace(tick).is_none());
        }
    }
    assert_eq!(sent, [(4, 10, u64::MAX - 255), (5, 10, 0), (6, 11, 0x1000)]);
    assert_eq!(l0_retirement, Some(8));
    assert_eq!(bt_retirement, Some(11));
    assert!(l0a.is_idle() && output.is_idle());
}

#[test]
fn overflow_is_atomic_and_empty_active_output_does_not_retire() {
    let mut output = C220MteL1Output::default();
    let plan = C220MteOutputPlan::new(1, 2, 0, 64, true, NonZeroU32::new(64).unwrap());
    output
        .receive(u64::MAX - 5, C220MteL1OutputDestination::Bt, plan, ())
        .unwrap();
    output
        .step(u64::MAX - 5, C220MteL1OutputCredits::default())
        .unwrap();
    let before = output.clone();
    assert_eq!(
        output.step(u64::MAX - 4, C220MteL1OutputCredits::default()),
        Err(C220MteL1OutputError::TimeOverflow)
    );
    assert_eq!(output, before);

    let mut empty = C220MteL1Output::default();
    empty
        .receive(
            0,
            C220MteL1OutputDestination::Bt,
            C220MteOutputPlan::new(1, 2, 0, 0, true, NonZeroU32::MIN),
            (),
        )
        .unwrap();
    for tick in 0..10 {
        let cycle = empty.step(tick, C220MteL1OutputCredits::default()).unwrap();
        assert!(cycle.sent.is_none() && cycle.retired.is_none());
        assert_eq!(cycle.queues.acknowledged, 1);
    }
    assert!(!empty.is_idle());
}
