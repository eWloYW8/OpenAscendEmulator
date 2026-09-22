use super::*;
use crate::isa::c220::mte::load2d::C220Load2dInstruction;
use crate::sim::c220::mte::mte1::load2d::C220Load2dRequestPlan;
use std::num::NonZeroU32;

fn fragment(id: u64) -> C220MteOutputFragment {
    C220MteOutputFragment {
        instruction_id: id,
        request_id: id,
        destination_address: 0,
        bytes: 32,
        last_in_uop: true,
        last_in_instruction: true,
    }
}

#[test]
fn bounded_queues_prioritize_primary_ports_and_acknowledge_locally() {
    use C220L0WritePort::{Port0, Port1, Port2};
    let mut pipeline = C220L0WritePipeline::default();
    assert!(pipeline.push(0, Port2, fragment(20)).unwrap());
    for tick in 0..10 {
        if tick == 6 {
            for id in 10..15 {
                assert!(pipeline.push(tick, Port1, fragment(id)).unwrap());
            }
            assert!(!pipeline.push(tick, Port1, fragment(99)).unwrap());
        }
        if tick == 8 {
            for id in 0..3 {
                assert!(pipeline.push(tick, Port0, fragment(id)).unwrap());
            }
            assert!(!pipeline.push(tick, Port0, fragment(99)).unwrap());
        }
        assert!(pipeline.step(tick).unwrap().sent.is_none());
    }
    let before = pipeline.clone();
    assert!(pipeline.step(11).is_err());
    assert_eq!(pipeline, before);
    let mut previous = None;
    for (tick, expected) in (10..).zip([0, 10, 1, 11, 2, 12, 13, 14, 20]) {
        let cycle = pipeline.step(tick).unwrap();
        assert_eq!(cycle.sent.unwrap().instruction_id, expected);
        assert_eq!(cycle.retired_instruction, previous);
        previous = Some(expected);
    }
    assert_eq!(pipeline.step(19).unwrap().retired_instruction, Some(20));
    assert!(pipeline.is_idle());
    // A producer may enqueue after this cycle's service phase.
    assert!(pipeline.push(19, Port0, fragment(30)).unwrap());
    let before = pipeline.clone();
    assert!(pipeline.step(19).is_err());
    assert_eq!(pipeline, before);
    assert!(pipeline.step(20).unwrap().sent.is_none());
    assert_eq!(pipeline.step(21).unwrap().sent.unwrap().instruction_id, 30);
    assert_eq!(pipeline.step(22).unwrap().retired_instruction, Some(30));
    let before = pipeline.clone();
    assert_eq!(
        pipeline.push(u64::MAX, Port2, fragment(99)),
        Err(C220L0WriteError::TimeOverflow)
    );
    assert_eq!(pipeline, before);
}

#[test]
fn load2d_completion_expands_to_l0_fragments_and_retires_only_the_tail() {
    let instruction = C220Load2dInstruction::decode(0x6000_2181).unwrap();
    let mut registers = [0; 32];
    registers[0] = u64::MAX - 63;
    registers[2] = 31;
    registers[3] = 2 << 16;
    let transfer = instruction.capture(&registers).unwrap();
    for bandwidth in [96, 128, 256, 1024] {
        let requests = C220Load2dRequestPlan::new(transfer, NonZeroU32::new(96).unwrap()).unwrap();
        let mut outputs = requests
            .enumerate()
            .flat_map(|(id, request)| {
                let plan = request.outputs(7, id as u64, NonZeroU32::new(bandwidth).unwrap());
                assert_eq!(
                    plan.len(),
                    if request.completes_logical_uop {
                        512_u32.div_ceil(bandwidth) as usize
                    } else {
                        0
                    }
                );
                plan
            })
            .peekable();
        let mut pipeline = C220L0WritePipeline::default();
        let mut acknowledgments = Vec::new();
        let mut retirements = Vec::new();
        for tick in 0..100 {
            // One destination interface runs before the shared L1 output callback.
            let cycle = pipeline.step(tick).unwrap();
            if let Some(ack) = cycle.acknowledged {
                acknowledgments.push(ack);
            }
            if let Some(id) = cycle.retired_instruction {
                retirements.push(id);
            }
            if let Some(fragment) = outputs.peek()
                && pipeline
                    .push(tick, C220L0WritePort::Port0, *fragment)
                    .unwrap()
            {
                outputs.next();
            }
            if outputs.peek().is_none() && pipeline.is_idle() {
                break;
            }
        }
        assert!(outputs.peek().is_none() && pipeline.is_idle());
        assert_eq!(retirements, [7]);
        assert_eq!(acknowledgments.iter().map(|f| f.bytes).sum::<u32>(), 1024);
        assert_eq!(acknowledgments.iter().filter(|f| f.last_in_uop).count(), 2);
        assert!(acknowledgments.last().unwrap().last_in_instruction);
        assert_eq!(acknowledgments[0].destination_address, u64::MAX - 63);
        if bandwidth < 512 {
            assert_eq!(
                acknowledgments[1].destination_address,
                u64::from(bandwidth) - 64
            );
        }
    }
}
