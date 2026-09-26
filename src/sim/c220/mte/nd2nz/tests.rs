use super::*;
use crate::isa::c220::mte::{nd2nz::C220Nd2NzInstruction, read_register_mask};
use crate::memory::{
    mapped::MappedMemory,
    region::MemoryRegion,
    sparse::{MemoryByteState, SparseMemory},
};
use crate::sim::c220::memory::C220LocalBuffer;
use std::num::NonZeroU32;

#[test]
fn engine_pipelines_reads_gates_response_ownership_and_sends_to_l1_port1() {
    use crate::isa::c220::mte::nd2nz::C220Nd2NzTransfer;
    use crate::sim::c220::mte::{
        interface::{C220MteL1WriteInterface, C220MteL1WritePort, C220MteOutputFragment},
        uop::C220DmaUopMode,
    };
    let mut engine = C220Nd2NzEngine::new(C220Nd2NzStagingConfig {
        rows: NonZeroU32::new(8).unwrap(),
        alignment_depth: 256,
        small_data_capacity: NonZeroU32::new(256).unwrap(),
        receive_bandwidth: NonZeroU32::new(32).unwrap(),
    })
    .unwrap();
    let transfer = C220Nd2NzTransfer {
        instruction: C220Nd2NzInstruction::decode((3 << 29) | (1 << 27) | (12 << 22)).unwrap(),
        source_base: 0x1000,
        destination_base: 256,
        xm: (1 << 4) | (2 << 16) | (32 << 32),
        xt: 32 | (2 << 16) | (1 << 32),
    };
    let route = C220Nd2NzReadRoute::PerRow;
    let mode = C220DmaUopMode::Fixed128;
    assert!(engine.submit(0, 10, transfer, route, mode).unwrap());
    assert!(!engine.generate(0).unwrap());
    assert!(engine.generate(1).unwrap());
    assert!(!engine.submit(1, 11, transfer, route, mode).unwrap());
    assert!(engine.generate(2).unwrap());
    assert!(engine.submit(2, 11, transfer, route, mode).unwrap());
    assert!(engine.generate(3).unwrap());
    assert!(engine.generate(4).unwrap());
    assert!(!engine.can_submit());
    assert!(engine.take_read(6, true).unwrap().is_none());
    assert_eq!(engine.take_read(7, true).unwrap().unwrap().request_id, 0);
    assert_eq!(engine.take_read(8, true).unwrap().unwrap().request_id, 1);
    assert_eq!(
        engine.take_read(9, true).unwrap().unwrap().instruction_id,
        11
    );
    assert_eq!(engine.receive(9, 11, 0).unwrap(), 0);
    assert_eq!(engine.take_read(10, true).unwrap().unwrap().request_id, 1);
    assert_eq!(engine.receive(10, 10, 0).unwrap(), 32);
    assert_eq!(engine.stage_lane(10, 0).unwrap(), None);
    assert_eq!(engine.receive(11, 10, 1).unwrap(), 32);
    assert_eq!(engine.stage_lane(11, 0).unwrap(), Some(0));
    assert_eq!(engine.stage_lane(12, 1).unwrap(), Some(1));
    let mut l1 = C220MteL1WriteInterface::default();
    for request_id in 0..2 {
        assert!(
            l1.push(
                12,
                C220MteL1WritePort::Port1,
                C220MteOutputFragment {
                    instruction_id: 9,
                    request_id,
                    destination_address: 0,
                    bytes: 32,
                    last_in_uop: true,
                    last_in_instruction: false,
                }
            )
            .unwrap()
        );
    }
    assert!(engine.send_l1(12, &mut l1).unwrap().is_none());
    assert_eq!(&engine.staging().alignment_bytes()[..2], [32, 32]);
    assert!(l1.send(13, true).unwrap().sent.is_some());
    let first = engine.send_l1(13, &mut l1).unwrap().unwrap();
    assert_eq!(
        (first.instruction_id, first.destination_address, first.bytes),
        (10, 256, 64)
    );
    assert!(first.last_in_instruction);
    assert_eq!(engine.active_instruction(), Some(11));
    assert_eq!(engine.receive(13, 11, 0).unwrap(), 32);
    assert!(l1.send(14, true).unwrap().sent.is_some());
    assert_eq!(engine.receive(14, 11, 1).unwrap(), 32);
    assert_eq!(engine.stage_lane(14, 0).unwrap(), Some(0));
    assert_eq!(engine.stage_lane(15, 1).unwrap(), Some(1));
    assert_eq!(
        engine.send_l1(15, &mut l1).unwrap().unwrap().instruction_id,
        11
    );
    assert!(engine.is_idle());
    assert!(!l1.is_idle());
    assert_eq!(
        l1.queue(C220MteL1WritePort::Port1)
            .back()
            .unwrap()
            .ready_tick,
        16
    );

    let large = C220Nd2NzTransfer {
        xm: (1 << 4) | (1 << 16) | (2048 << 32),
        ..transfer
    };
    assert!(engine.submit(16, 12, large, route, mode).unwrap());
    for tick in 17..24 {
        assert!(engine.generate(tick).unwrap());
    }
    assert_eq!(engine.generated().len(), 7);
    assert!(!engine.generate(24).unwrap());
    assert!(engine.take_read(24, false).unwrap().is_none());
    assert_eq!(engine.take_read(25, true).unwrap().unwrap().request_id, 0);
    assert!(engine.generate(25).unwrap());
    assert_eq!(engine.generated().back().unwrap().ready_tick, 31);
}

fn staging_request(route: C220Nd2NzReadRoute, bytes: &[u32], padding: u32) -> C220Nd2NzResponse {
    C220Nd2NzResponse::new(&C220Nd2NzReadRequest {
        matrix_index: 0,
        source_address: 0,
        bytes: bytes.iter().sum(),
        padding_bytes: padding,
        elements: bytes
            .iter()
            .enumerate()
            .map(|(row, &bytes)| C220Nd2NzReadElement {
                row_slot: row as u32,
                bytes,
            })
            .collect(),
        route,
        last_in_instruction: true,
    })
    .unwrap()
}

#[test]
fn per_row_responses_wait_one_cycle_and_preserve_fragment_padding() {
    let mut staging = C220Nd2NzStaging::new(C220Nd2NzStagingConfig {
        rows: NonZeroU32::new(2).unwrap(),
        alignment_depth: 32,
        small_data_capacity: NonZeroU32::new(64).unwrap(),
        receive_bandwidth: NonZeroU32::new(32).unwrap(),
    })
    .unwrap();
    let mut response = staging_request(C220Nd2NzReadRoute::PerRow, &[48], 16);
    assert_eq!(staging.receive(0, &mut response).unwrap(), 32);
    assert!(!response.is_complete());
    assert_eq!(staging.stage_lane(0, 0).unwrap(), None);
    assert_eq!(staging.receive(1, &mut response).unwrap(), 16);
    assert!(response.is_complete());
    assert_eq!(staging.stage_lane(1, 0).unwrap(), Some(0));
    assert_eq!(staging.alignment_bytes(), [48, 0]);
    assert_eq!(staging.stage_lane(2, 0).unwrap(), None);
    assert!(!staging.take_write_credit(2, 1, false).unwrap());
    assert_eq!(staging.alignment_bytes(), [48, 0]);
    assert_eq!(staging.pending_row_fragments(0), Some(1));
    assert!(staging.take_write_credit(3, 1, true).unwrap());
    assert_eq!(staging.stage_lane(3, 0).unwrap(), Some(0));
    assert_eq!(staging.alignment_bytes(), [48, 0]);
    assert!(staging.take_write_credit(4, 1, true).unwrap());
    assert_eq!(staging.alignment_bytes(), [16, 0]);
    assert!(!staging.is_idle());
    assert!(matches!(
        staging.stage_lane(3, 0),
        Err(C220Nd2NzStagingError::TimeReversed { .. })
    ));
    assert!(matches!(
        staging.take_write_credit(4, 1, true),
        Err(C220Nd2NzStagingError::RepeatedCallback { .. })
    ));

    let mut lanes = C220Nd2NzStaging::new(C220Nd2NzStagingConfig {
        rows: NonZeroU32::new(8).unwrap(),
        ..staging.config()
    })
    .unwrap();
    let mut lower = staging_request(C220Nd2NzReadRoute::PerRow, &[64], 0);
    let mut upper = C220Nd2NzResponse::new(&C220Nd2NzReadRequest {
        matrix_index: 0,
        source_address: 0,
        bytes: 32,
        padding_bytes: 0,
        elements: vec![C220Nd2NzReadElement {
            row_slot: 4,
            bytes: 32,
        }],
        route: C220Nd2NzReadRoute::PerRow,
        last_in_instruction: true,
    })
    .unwrap();
    lanes.receive(0, &mut lower).unwrap();
    lanes.receive(1, &mut upper).unwrap();
    lanes.receive(2, &mut lower).unwrap();
    assert_eq!(lanes.stage_lane(2, 0).unwrap(), Some(0));
    assert_eq!(lanes.pending_row_fragments(4), Some(1));
    assert_eq!(lanes.stage_lane(3, 0).unwrap(), Some(4));
    assert_eq!(lanes.pending_row_fragments(0), Some(1));
}

#[test]
fn small_data_retains_blocked_batch_heads_and_shared_capacity() {
    let mut staging = C220Nd2NzStaging::new(C220Nd2NzStagingConfig {
        rows: NonZeroU32::new(2).unwrap(),
        alignment_depth: 32,
        small_data_capacity: NonZeroU32::new(64).unwrap(),
        receive_bandwidth: NonZeroU32::new(64).unwrap(),
    })
    .unwrap();
    let mut first = staging_request(C220Nd2NzReadRoute::ContiguousRows, &[32, 32], 0);
    let mut second = first.clone();
    let mut third = first.clone();
    assert_eq!(staging.receive(0, &mut first).unwrap(), 64);
    assert_eq!(staging.stage_small(0).unwrap(), 0);
    assert_eq!(staging.receive(1, &mut second).unwrap(), 0);
    assert_eq!(staging.stage_small(1).unwrap(), 64);
    assert_eq!(staging.receive(2, &mut second).unwrap(), 64);
    assert_eq!(staging.stage_small(3).unwrap(), 0);
    assert!(staging.take_write_credit(3, 1, true).unwrap());
    assert_eq!(staging.stage_small(4).unwrap(), 32);
    assert_eq!(staging.small_data_bytes(), 32);
    assert_eq!(staging.alignment_bytes(), [32, 32]);
    assert_eq!(staging.receive(4, &mut third).unwrap(), 32);
    assert_eq!(third.remaining_bytes(), 32);
    assert_eq!(staging.pending_small_batches(), 2);
    assert!(staging.take_write_credit(4, 2, true).unwrap());
    assert_eq!(staging.stage_small(5).unwrap(), 32);
    assert_eq!(staging.alignment_bytes(), [0, 32]);
    assert_eq!(staging.pending_small_batches(), 1);
    assert_eq!(staging.stage_small(6).unwrap(), 32);
    assert!(staging.take_write_credit(6, 2, true).unwrap());
    assert_eq!(staging.receive(6, &mut third).unwrap(), 32);
    assert!(third.is_complete());
    assert_eq!(staging.stage_small(7).unwrap(), 32);
    assert_eq!(staging.alignment_bytes(), [0, 32]);
    assert_eq!(staging.small_data_bytes(), 0);
}

#[test]
fn read_plans_retain_row_priority_splits_and_response_elements() {
    use crate::isa::c220::mte::nd2nz::C220Nd2NzTransfer;
    use crate::sim::c220::mte::uop::C220DmaUopMode;
    let transfer = C220Nd2NzTransfer {
        instruction: C220Nd2NzInstruction::decode((3 << 29) | (1 << 27) | (12 << 22)).unwrap(),
        source_base: 0x1003,
        destination_base: 0,
        xm: (1 << 4) | (3 << 16) | (140 << 32),
        xt: 160 | (3 << 16) | (1 << 32),
    };
    let requests: Vec<_> = C220Nd2NzReadPlan::new(
        transfer,
        C220Nd2NzReadRoute::PerRow,
        C220DmaUopMode::Fixed128,
        NonZeroU32::new(2).unwrap(),
    )
    .collect();
    assert_eq!(
        requests.iter().map(|r| r.bytes).collect::<Vec<_>>(),
        [125, 93, 47, 15, 61, 79]
    );
    assert_eq!(
        requests
            .iter()
            .map(|r| r.elements[0].row_slot)
            .collect::<Vec<_>>(),
        [0, 1, 1, 0, 0, 0]
    );
    assert_eq!(
        requests.iter().map(|r| r.padding_bytes).collect::<Vec<_>>(),
        [0, 0, 20, 20, 0, 20]
    );
    assert_eq!(requests.iter().filter(|r| r.last_in_instruction).count(), 1);
    assert!(requests.last().unwrap().last_in_instruction);
    let packed = C220Nd2NzTransfer {
        source_base: 0x1000,
        xm: (1 << 4) | (3 << 16) | (128 << 32),
        xt: 128 | (3 << 16) | (1 << 32),
        ..transfer
    };
    let requests: Vec<_> = C220Nd2NzReadPlan::new(
        packed,
        C220Nd2NzReadRoute::ContiguousRows,
        C220DmaUopMode::Wide512,
        NonZeroU32::new(2).unwrap(),
    )
    .collect();
    assert_eq!(
        requests.iter().map(|r| r.bytes).collect::<Vec<_>>(),
        [256, 128]
    );
    assert_eq!(
        requests[0].elements,
        [
            C220Nd2NzReadElement {
                row_slot: 0,
                bytes: 128
            },
            C220Nd2NzReadElement {
                row_slot: 1,
                bytes: 128
            }
        ]
    );
    assert_eq!(
        requests[1].elements,
        [C220Nd2NzReadElement {
            row_slot: 0,
            bytes: 128
        }]
    );
    assert!(requests[1].last_in_instruction);
}

#[test]
fn formats_capture_strides_preserve_unknowns_and_zero_pad_tails() {
    let mut input = MappedMemory::bind(
        SparseMemory::new(
            vec![MemoryRegion::new(4096, (0..4096).map(|x| x as u8).collect()).unwrap()],
            4096,
            4096,
        ),
        &[0x1000],
    )
    .unwrap();
    input.write_unknown_at(0x1000, 1).unwrap();
    for mode in 0..4 {
        let instruction = C220Nd2NzInstruction::decode(
            (3 << 29) | (1 << 27) | (12 << 22) | (1 << 17) | (2 << 12) | (3 << 7) | (4 << 2) | mode,
        )
        .unwrap();
        assert_eq!(read_register_mask(instruction.word), Some(0b1_1110));
        let mut registers = [0; 32];
        registers[1] = 128;
        registers[2] = 0x1000;
        registers[3] = 7 | (2 << 4) | (3 << 16) | (35 << 32) | (512 << 48);
        registers[4] = 64 | (16 << 16) | (1 << 32) | (2048 << 48);
        let transfer = instruction.capture(&registers);
        let element = [1, 2, 4, 4][mode as usize];
        let block = if mode == 3 { 64 } else { 32 };
        let blocks = (35_u32 * element).div_ceil(block);
        assert_eq!(transfer.sid(), 7);
        assert_eq!(transfer.segment_count(), u64::from(6 * blocks));
        let second_matrix = transfer.segment(u64::from(3 * blocks)).unwrap();
        assert_eq!(
            second_matrix.source_address,
            0x1000 + u64::from(512 * element)
        );
        assert_eq!(
            second_matrix.destination_address,
            128 + u64::from(2048 * element)
        );
        let mut output = C220LocalBuffer::new(64);
        let result = execute_c220_nd2nz(transfer, &input, &mut output).unwrap();
        assert_eq!(result.segments, u64::from(6 * blocks));
        assert_eq!(result.source_bytes, u64::from(6 * 35 * element));
        assert_eq!(
            result.padding_bytes,
            u64::from(6 * (blocks * block - 35 * element))
        );
        assert_eq!(
            output.read_states_linear(128, 1).unwrap(),
            [MemoryByteState::Unknown]
        );
        let last = transfer.segment(u64::from(blocks - 1)).unwrap();
        assert_eq!(last.source_address, 0x1000 + u64::from((blocks - 1) * 32));
        let tail = output
            .read_states_linear(
                last.destination_address + u64::from(last.input_bytes),
                (block - last.input_bytes) as usize,
            )
            .unwrap();
        assert!(tail.iter().all(|b| *b == MemoryByteState::Known(0)));
        let requests: Vec<_> =
            C220Nd2NzWritePlan::new(transfer, NonZeroU32::new(2).unwrap()).collect();
        assert_eq!(requests.len(), (4 * blocks) as usize);
        assert_eq!(requests[0].bytes, 64);
        assert_eq!(requests[blocks as usize].bytes, 32);
        assert_eq!(requests.iter().filter(|r| r.last_in_instruction).count(), 1);
        assert!(requests.last().unwrap().last_in_instruction);
        if mode == 3 {
            assert_eq!(
                transfer
                    .segments()
                    .take(3)
                    .map(|s| s.destination_address)
                    .collect::<Vec<_>>(),
                [128, 160, 1152]
            );
        }
        for cleared in [4095_u64 << 4, 65535 << 16, 65535 << 32] {
            let empty = crate::isa::c220::mte::nd2nz::C220Nd2NzTransfer {
                xm: transfer.xm & !cleared,
                ..transfer
            };
            assert!(empty.is_disabled());
            assert!(empty.segments().next().is_none());
            assert!(
                C220Nd2NzWritePlan::new(empty, NonZeroU32::new(2).unwrap())
                    .next()
                    .is_none()
            );
            assert_eq!(
                execute_c220_nd2nz(empty, &input, &mut output).unwrap(),
                C220Nd2NzResult::default()
            );
        }
    }
}
