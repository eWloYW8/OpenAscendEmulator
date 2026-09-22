use super::*;
use crate::sim::c220::vector::read::C220VectorReadIssue;

use crate::architecture::Architecture;
use crate::memory::sparse::MemoryByteState;
use crate::sim::c220::memory::C220UbBank;
use crate::sim::c220::vector::{
    C220VectorAddresses, C220VectorArithmeticModes, C220VectorControl,
    plan_c220_vector_arithmetic_issue,
};
use crate::sim::common::scalar::ScalarMachine;
use crate::sim::common::scalar::ScalarStepper;

#[test]
fn conflicting_read_ports_delay_visibility_and_keep_granted_bytes() {
    let mut ub = UbMemory::new(4096, 256);
    for (address, value) in [(0, 1.0_f32), (0x10000, 2.0_f32)] {
        let bytes = value
            .to_le_bytes()
            .repeat(8)
            .into_iter()
            .map(MemoryByteState::Known)
            .collect::<Vec<_>>();
        ub.write_states(address, &bytes).unwrap();
    }
    let issue = plan_c220_vector_arithmetic_issue(
        0,
        0x85dc_b618,
        C220VectorControl {
            encoded_repeat_count: 0,
            destination_block_stride: 1,
            source_0_block_stride: 1,
            source_1_block_stride: 1,
            destination_repeat_stride: 1,
            source_0_repeat_stride: 1,
            source_1_repeat_stride: 1,
        },
        C220VectorAddresses {
            source_0: 0,
            source_1: 0x10000,
            destination: 0x200,
        },
        &[[0xff, 0, 0, 0]],
        C220VectorArithmeticModes::from_control_spr(0),
        &ub,
    )
    .unwrap();
    let machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
    let mut core = C220State::new(ScalarStepper::new(machine, 0), ub);
    let mut pipeline = C220VectorPipeline::new(C220VectorTimingRules {
        dispatch_ticks: 0,
        uop_issue_interval: NonZeroU64::new(1).unwrap(),
        ub_response_ticks: 2,
    });
    let write_stores = (0..8)
        .map(|lane| C220VectorStore {
            repeat_index: 0,
            lane_index: lane,
            address: (lane * 4) as u64,
            bank: C220UbBank::from_address((lane * 4) as u64),
            width_bytes: 4,
            data: crate::sim::c220::vector::access::store_data(4.0_f32.to_le_bytes()),
        })
        .collect::<Vec<_>>();
    pipeline
        .issue_at(
            0,
            &[C220VectorUop {
                pc: 0,
                repeat_index: 0,
                lane_group: Some(0),
                kind: crate::sim::c220::vector::timing::C220VectorUopKind::Ordinary,
                stages: crate::sim::c220::vector::timing::C220VectorUopStages {
                    read_ticks: 1,
                    execute_ticks: 0,
                },
                writeback_ticks: 1,
                writes_ub: true,
            }],
            &write_stores,
            None,
        )
        .unwrap();
    pipeline
        .issue_at(
            0,
            &[C220VectorUop {
                pc: 0,
                repeat_index: 0,
                lane_group: Some(0),
                kind: crate::sim::c220::vector::timing::C220VectorUopKind::Ordinary,
                stages: crate::sim::c220::vector::timing::C220VectorUopStages {
                    read_ticks: 6,
                    execute_ticks: 7,
                },
                writeback_ticks: 1,
                writes_ub: true,
            }],
            &issue.write_targets,
            Some(C220VectorReadIssue::Arithmetic(&issue)),
        )
        .unwrap();
    pipeline.advance_to(1, &mut core).unwrap();
    let cycle = &pipeline.last_ub_cycles()[0];
    assert!(
        cycle
            .decisions
            .iter()
            .any(|decision| { decision.port == C220UbPort::VectorWrite && decision.granted })
    );
    assert!(
        cycle
            .decisions
            .iter()
            .any(|decision| { decision.port == C220UbPort::VectorRead0 && !decision.granted })
    );
    pipeline.advance_to(3, &mut core).unwrap();
    assert_eq!(pipeline.pending_visibility_tick(), Some(18));
    pipeline.advance_to(8, &mut core).unwrap();
    let sample = &pipeline.last_read_samples()[0];
    assert_eq!(sample.tick, 8);
    assert_eq!(sample.read0_grants, [Some(2)]);
    assert_eq!(sample.read1_grants, [Some(1)]);
    assert_eq!(core.ub().read_known(0, 4).unwrap(), 4.0_f32.to_le_bytes());
    assert_eq!(&sample.source_0_bytes[..4], &1.0_f32.to_le_bytes());
    assert_eq!(sample.lanes[0].bits, 3.0_f32.to_bits());
    pipeline.advance_to(18, &mut core).unwrap();
    assert_eq!(
        core.ub().read_known(0x200, 4).unwrap(),
        3.0_f32.to_le_bytes()
    );
}
