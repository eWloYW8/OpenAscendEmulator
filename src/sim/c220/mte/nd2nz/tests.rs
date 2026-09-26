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
