use super::*;
use crate::isa::c220::mte::set2d::C220Set2dInstruction;
use crate::sim::c220::memory::C220LocalMemoryConfig;
use crate::sim::c220::mte::interface::{C220L0WritePipeline, C220L0WritePort};
use std::num::NonZeroU32;

fn fill(destination: u32, b32: bool, base: u64, descriptor: u64) -> C220Set2dFill {
    let word = (3 << 29) | (1 << 22) | (4 << 17) | (7 << 7) | (u32::from(b32) << 2) | destination;
    let instruction = C220Set2dInstruction::decode(word).unwrap();
    let mut registers = [0; 32];
    registers[4] = base;
    registers[7] = descriptor;
    instruction.capture(&registers, 0x9876_abcd_7fc1_3456)
}

fn bandwidths() -> C220Set2dBandwidths {
    C220Set2dBandwidths {
        l0a: NonZeroU32::new(192).unwrap(),
        l0b: NonZeroU32::new(256).unwrap(),
        l1: NonZeroU32::new(24).unwrap(),
    }
}

#[test]
fn destinations_formats_gaps_and_linear_addresses() {
    for destination in 0..3 {
        for b32 in [false, true] {
            let mut memory = C220LocalMemory::new(C220LocalMemoryConfig {
                l0a_bytes: 1024,
                l0b_bytes: 1024,
                l1_bytes: 1024,
                ..C220LocalMemoryConfig::default()
            })
            .unwrap();
            let fill = fill(destination, b32, 1020, 2 | (1 << 16) | (1 << 32));
            let result = execute_c220_set2d(&mut memory, fill).unwrap();
            assert_eq!(result.bytes, fill.byte_count());
            assert_eq!(result.repetitions, 2);
            assert_eq!(result.skipped_overflow_repetitions, 0);
            let buffer = match destination {
                0 => memory.l0a(),
                1 => memory.l0b(),
                _ => memory.l1(),
            };
            for segment in fill.segments() {
                let data = buffer
                    .read_initialized_linear(segment.destination_address, segment.bytes as usize)
                    .unwrap();
                for element in data.chunks_exact(if b32 { 4 } else { 2 }) {
                    assert_eq!(
                        element,
                        &fill.pattern_register.to_le_bytes()[..element.len()]
                    );
                }
                assert_eq!(
                    buffer
                        .read_initialized_linear(
                            segment.destination_address + u64::from(segment.bytes),
                            4,
                        )
                        .unwrap(),
                    [0; 4]
                );
            }
            assert_eq!(buffer.read_initialized_linear(0, 4).unwrap(), [0; 4]);
        }
    }
    let mut memory = C220LocalMemory::new(C220LocalMemoryConfig::default()).unwrap();
    let result =
        execute_c220_set2d(&mut memory, fill(2, false, u64::MAX - 15, 2 | (1 << 16))).unwrap();
    assert_eq!(
        (
            result.repetitions,
            result.skipped_overflow_repetitions,
            result.bytes
        ),
        (1, 1, 32)
    );
    assert_eq!(
        memory.l1().read_known(16, 4).unwrap(),
        [0x56, 0x34, 0x56, 0x34]
    );
}

#[test]
fn descriptor_masks_empty_fills_and_lazy_uops() {
    let empty = fill(0, false, 0, 0);
    assert!(empty.descriptor.is_disabled());
    assert_eq!(C220Set2dUops::new(empty, bandwidths()).count(), 0);
    let empty = fill(0, false, 0, 3);
    assert!(!empty.descriptor.is_disabled());
    assert!(empty.descriptor.is_empty());
    assert_eq!(C220Set2dUops::new(empty, bandwidths()).count(), 0);
    let maximum = fill(0, true, 0, u64::MAX);
    assert_eq!(maximum.descriptor.repeat_count, 0x7fff);
    assert_eq!(maximum.descriptor.burst_blocks, 0x7fff);
    assert_eq!(maximum.descriptor.destination_gap_blocks, 0x7fff);
    let mut plan = C220Set2dUops::new(maximum, bandwidths());
    let count = plan.remaining();
    assert!(count > 1_000_000);
    assert_eq!(plan.next().unwrap().bytes, 192);
    assert_eq!(plan.remaining(), count - 1);
    assert!(C220Set2dInstruction::decode(maximum.instruction.word | 3).is_none());
    for (destination, lengths) in [
        (0, vec![192, 192, 128]),
        (1, vec![256, 256]),
        (2, vec![24, 8]),
    ] {
        let fill = fill(destination, false, 16, 2 | (1 << 16) | (2 << 32));
        let uops = C220Set2dUops::new(fill, bandwidths()).collect::<Vec<_>>();
        assert_eq!(
            uops.iter().map(|u| u.bytes).collect::<Vec<_>>(),
            lengths.repeat(2)
        );
        assert_eq!(uops.iter().filter(|u| u.last_in_instruction).count(), 1);
        assert!(uops.last().unwrap().last_in_instruction);
        assert_eq!(
            uops[lengths.len()].destination_address,
            16 + fill.destination_stride()
        );
    }
}

#[test]
fn l0_tail_retires_after_port_one_acknowledgment() {
    let fill = fill(0, true, 0, 1 | (1 << 16));
    let uops = C220Set2dUops::new(fill, bandwidths()).collect::<Vec<_>>();
    let mut output = C220L0WritePipeline::default();
    for uop in &uops {
        assert_eq!(uop.route, C220Set2dOutputRoute::L0a(C220L0WritePort::Port1));
    }
    let mut retired = Vec::new();
    for tick in 0..8 {
        if let Some(uop) = uops.get(tick as usize) {
            assert!(
                output
                    .push(tick, C220L0WritePort::Port1, uop.output_fragment(9, tick))
                    .unwrap()
            );
        }
        let cycle = output.step(tick).unwrap();
        if let Some(instruction) = cycle.retired_instruction {
            retired.push((tick, instruction));
        }
    }
    assert_eq!(retired, [(7, 9)]);
    assert!(output.is_idle());
}
