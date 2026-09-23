use super::layout::*;
use super::*;
use crate::isa::c220::cube::{C220CubeInstruction, C220CubeRegisterValues};
use crate::isa::c220::mte::bias::C220MovL1ToBtInstruction;
use crate::sim::c220::cube::{C220CubeConfig, C220CubePipeline, C220CubeTimingControl};
use crate::sim::c220::memory::{C220LocalMemory, C220LocalMemoryConfig};
use crate::sim::c220::mte::mte1::bias::prepare_c220_mov_l1_to_bt;

fn s4_issue(m: u16, k: u16, n: u16, clear: bool) -> C220CubeIssue {
    let word = (7 << 29) | (6 << 22);
    let instruction = C220CubeInstruction::decode(word).unwrap();
    let registers = C220CubeRegisterValues {
        xd: 7168,
        xn: 3584,
        xm: 3072,
        xt: u64::from(m) | (u64::from(k) << 12) | (u64::from(n) << 24) | (u64::from(clear) << 63),
    };
    let parameters = instruction.parameters(registers);
    let ticket = C220CubePipeline::new(C220CubeConfig::default())
        .unwrap()
        .preview_issue(
            0,
            instruction,
            parameters,
            C220CubeTimingControl::from_sprs(0, 0, 0),
        )
        .unwrap();
    C220CubeIssue {
        instruction_id: 0,
        pc: 0,
        word,
        instruction,
        registers,
        parameters,
        ticket,
    }
}

fn integer_memory() -> C220LocalMemory {
    C220LocalMemory::new(C220LocalMemoryConfig {
        l0a_bytes: 4096,
        l0b_bytes: 4096,
        l0c_bytes: 8192,
        ..C220LocalMemoryConfig::default()
    })
    .unwrap()
}

#[test]
fn single_row_mmad_reads_contiguous_a_across_input_blocks() {
    for raw_type in [0, 1, 2, 3, 5, 6, 9, 10] {
        let (k, left, right) = match raw_type {
            6 => (1025, vec![0x22], vec![0x11]),
            0 | 1 | 5 => (513, vec![2], vec![1]),
            2 | 3 => (
                257,
                0x4000_u16.to_le_bytes().to_vec(),
                0x3c00_u16.to_le_bytes().to_vec(),
            ),
            9 => (
                257,
                0x4000_u16.to_le_bytes().to_vec(),
                0x3f80_u16.to_le_bytes().to_vec(),
            ),
            10 => (
                129,
                2.0_f32.to_le_bytes().to_vec(),
                1.0_f32.to_le_bytes().to_vec(),
            ),
            _ => unreachable!(),
        };
        let mut issue = s4_issue(1, k, 1, true);
        issue.word = (7 << 29) | ((raw_type & 7) << 22) | (raw_type >> 3);
        issue.instruction = C220CubeInstruction::decode(issue.word).unwrap();
        issue.registers.xn = 0;
        issue.registers.xm = 0;
        issue.registers.xd = 0;
        issue.parameters = issue.instruction.parameters(issue.registers);
        let geometry = issue.instruction.geometry(issue.parameters);
        let mut memory = C220LocalMemory::new(Default::default()).unwrap();
        memory
            .l0a_mut()
            .write_known(0, &vec![0; usize::from(geometry.k_tiles) * 512])
            .unwrap();
        let elements = if raw_type == 6 { k.div_ceil(2) } else { k };
        memory
            .l0a_mut()
            .write_known(0, &left.repeat(usize::from(elements)))
            .unwrap();
        memory
            .l0b_mut()
            .write_known(
                0,
                &right.repeat(usize::from(geometry.k_tiles) * 512 / right.len()),
            )
            .unwrap();
        issue
            .execute(&mut memory, C220CubeExecutionControl::from_spr3(0))
            .unwrap();
        if raw_type == 2 {
            assert_eq!(read_u16_wrapped(memory.l0c().buffer(), 0).unwrap(), 0x6004);
        } else {
            let expected = if matches!(raw_type, 3 | 9 | 10) {
                (2.0 * f32::from(k)).to_bits()
            } else {
                2 * u32::from(k)
            };
            assert_eq!(
                read_u32_wrapped(memory.l0c().buffer(), 0).unwrap(),
                expected,
                "type {raw_type}"
            );
        }
    }
}

#[test]
fn f32_mmad_pads_a_tile_stride_when_requested() {
    for padded in [false, true] {
        for hf32 in [false, true] {
            let mut issue = s4_issue(17, 8, 1, true);
            issue.word = (7 << 29) | (2 << 22) | 1;
            issue.instruction = C220CubeInstruction::decode(issue.word).unwrap();
            issue.registers.xn = 0;
            issue.registers.xm = 0;
            issue.registers.xd = 0;
            issue.registers.xt |= u64::from(padded) << 58;
            issue.parameters = issue.instruction.parameters(issue.registers);
            let mut memory = C220LocalMemory::new(Default::default()).unwrap();
            for (tile, value) in [1.0_f32, 3.0, 5.0].into_iter().enumerate() {
                let bytes = value.to_le_bytes().repeat(128);
                memory
                    .l0a_mut()
                    .write_known(tile as u64 * 512, &bytes)
                    .unwrap();
            }
            memory
                .l0b_mut()
                .write_known(0, &1.0_f32.to_le_bytes().repeat(128))
                .unwrap();
            issue
                .execute(
                    &mut memory,
                    C220CubeExecutionControl::from_spr3(u64::from(hf32) << 46),
                )
                .unwrap();
            for row in 0..17 {
                let expected: f32 = if row < 16 {
                    8.0
                } else if padded {
                    40.0
                } else {
                    24.0
                };
                assert_eq!(
                    read_u32_wrapped(memory.l0c().buffer(), f32_c_address(0, 2, row, 0)).unwrap(),
                    expected.to_bits()
                );
            }
        }
    }
}

#[test]
fn sparse_byte_mmad_selects_dense_inputs_across_tiles_and_partial_k() {
    for raw_type in [0, 1, 5] {
        for raw_k in [32_u16, 64, 68, 128] {
            let mut issue = s4_issue(17, raw_k, 17, true);
            issue.word = (7 << 29) | (5 << 25) | (raw_type << 22);
            issue.instruction = C220CubeInstruction::decode(issue.word).unwrap();
            issue.registers.xn = 0;
            issue.registers.xm = 16384;
            issue.registers.xd = 0;
            issue.parameters = issue.instruction.parameters(issue.registers);
            let mut memory = C220LocalMemory::new(Default::default()).unwrap();
            let compressed = u64::from(issue.parameters.effective_k);
            let a_tiles =
                2 * compressed.div_ceil(32) - u64::from((1..=16).contains(&(compressed & 31)));
            for row in 0..17 {
                for k in 0..a_tiles * 32 {
                    let value = ((row + k) % 13) as i8 - 6;
                    memory
                        .l0a_mut()
                        .write_known(integer_a_element(a_tiles, 32, row, k), &[value as u8])
                        .unwrap();
                }
            }
            for column in 0..17 {
                for k in 0..compressed {
                    memory
                        .l0b_mut()
                        .write_known(16384 + integer_b_element(2, 32, k, column), &[255])
                        .unwrap();
                }
            }
            memory
                .weight_index_mut()
                .write_known_linear(4096, &vec![0x93; compressed.div_ceil(32) as usize * 32 * 8])
                .unwrap();
            let outcome = issue
                .execute(&mut memory, C220CubeExecutionControl::from_spr3(0))
                .unwrap();
            assert_eq!(outcome.mac_count, 17 * 17 * compressed);
            for row in 0..17 {
                let expected: i32 = (0..compressed)
                    .map(|k| {
                        let selected = (k / 4) * 8 + [3, 1, 5, 7][k as usize % 4];
                        let left = ((row + selected) % 13) as i8 - 6;
                        let left = if raw_type == 1 {
                            i32::from(left)
                        } else {
                            i32::from(left as u8)
                        };
                        let right = if raw_type == 0 { 255 } else { -1 };
                        left * right
                    })
                    .sum();
                for column in 0..17 {
                    assert_eq!(
                        read_u32_wrapped(memory.l0c().buffer(), f32_c_address(0, 2, row, column))
                            .unwrap() as i32,
                        expected
                    );
                }
            }
        }
    }
}

#[test]
fn bias_is_broadcast_per_column_for_all_supported_mmad_formats() {
    for raw_type in [0, 1, 2, 3, 5, 6, 9, 10] {
        for hf32 in [false, true] {
            if hf32 && raw_type != 10 {
                continue;
            }
            let mut issue = s4_issue(2, 1, 17, false);
            issue.word = (7 << 29) | ((raw_type & 7) << 22) | (raw_type >> 3);
            issue.instruction = C220CubeInstruction::decode(issue.word).unwrap();
            let bias_address = u32::MAX - 1;
            issue.registers.xd |= u64::from(bias_address) << 32;
            issue.registers.xt |= 1 << 62;
            issue.parameters = issue.instruction.parameters(issue.registers);
            let spr3 = u64::from(hf32) << 46;
            issue.ticket = C220CubePipeline::new(C220CubeConfig::default())
                .unwrap()
                .preview_issue(
                    0,
                    issue.instruction,
                    issue.parameters,
                    C220CubeTimingControl::from_sprs(spr3, 0, 0),
                )
                .unwrap();
            let mut memory = integer_memory();
            let one = match raw_type {
                2 | 3 => 0x3c00_u16.to_le_bytes().to_vec(),
                9 => 0x3f80_u16.to_le_bytes().to_vec(),
                10 => 1.0_f32.to_le_bytes().to_vec(),
                6 => vec![0x11],
                _ => vec![1],
            };
            memory
                .l0a_mut()
                .write_known(0, &one.repeat(4096 / one.len()))
                .unwrap();
            memory
                .l0b_mut()
                .write_known(0, &one.repeat(4096 / one.len()))
                .unwrap();
            let slots = (0..17)
                .flat_map(|column| {
                    let value = if raw_type == 2 {
                        0xdead_0000 | if column % 2 == 0 { 0x3c00 } else { 0x4000 }
                    } else if matches!(raw_type, 3 | 9 | 10) {
                        (if column % 2 == 0 { 1.0_f32 } else { 2.0_f32 }).to_bits()
                    } else {
                        (if column % 2 == 0 { 5_i32 } else { -3_i32 }) as u32
                    };
                    value.to_le_bytes()
                })
                .collect::<Vec<_>>();
            let convert = matches!(raw_type, 3 | 9 | 10);
            let slots = if convert {
                (0..17)
                    .flat_map(|column| {
                        (if column % 2 == 0 { 0x3c00_u16 } else { 0x4000 }).to_le_bytes()
                    })
                    .collect::<Vec<_>>()
            } else {
                slots
            };
            memory.l1_mut().write_known(0, &slots).unwrap();
            let dma_word = (3 << 29) | (2 << 27) | (4 << 23) | (1 << 12) | (2 << 7) | (5 << 3);
            let mut dma_registers = [0; 32];
            dma_registers[0] = u64::from(bias_address);
            dma_registers[2] = (1 << 4) | if convert { (1 << 16) | 8 } else { 2 << 16 };
            let transfer = C220MovL1ToBtInstruction::decode(dma_word)
                .unwrap()
                .capture(&dma_registers);
            prepare_c220_mov_l1_to_bt(&memory, transfer)
                .unwrap()
                .commit(&mut memory)
                .unwrap();
            let outcome = issue
                .execute(&mut memory, C220CubeExecutionControl::from_spr3(spr3))
                .unwrap();
            assert_eq!(
                outcome.accumulator_source,
                C220CubeAccumulatorSource::Bias {
                    address: bias_address,
                    read_bytes: 256,
                }
            );
            for row in 0..2 {
                for column in 0..17 {
                    if raw_type == 2 {
                        let address = f16_c_address(7168, 1, row, column);
                        assert_eq!(
                            read_u16_wrapped(memory.l0c().buffer(), address).unwrap(),
                            if column % 2 == 0 { 0x4000 } else { 0x4200 },
                            "type {raw_type}, row {row}, column {column}"
                        );
                    } else {
                        let expected = if matches!(raw_type, 3 | 9 | 10) {
                            (if column % 2 == 0 { 2.0_f32 } else { 3.0_f32 }).to_bits()
                        } else {
                            (if column % 2 == 0 { 6_i32 } else { -2_i32 }) as u32
                        };
                        let address = f32_c_address(7168, 1, row, column);
                        assert_eq!(
                            read_u32_wrapped(memory.l0c().buffer(), address).unwrap(),
                            expected
                        );
                    }
                }
            }
            assert_eq!(outcome.written_lanes, 512);
            assert_eq!(outcome.padded_lanes, 478);
            issue.parameters.xt_bit_62 = false;
            let accumulated = issue
                .execute(&mut memory, C220CubeExecutionControl::from_spr3(spr3))
                .unwrap();
            assert_eq!(
                accumulated.accumulator_source,
                C220CubeAccumulatorSource::L0c
            );
            if raw_type == 2 {
                assert_eq!(
                    read_u16_wrapped(memory.l0c().buffer(), 7168).unwrap(),
                    0x4200
                );
            } else {
                let expected = if matches!(raw_type, 3 | 9 | 10) {
                    3.0_f32.to_bits()
                } else {
                    7
                };
                assert_eq!(
                    read_u32_wrapped(memory.l0c().buffer(), 7168).unwrap(),
                    expected
                );
            }
        }
    }
}

#[test]
fn accumulator_controls_select_zero_bias_or_prior_output() {
    for (clear, bias, expected) in [(true, true, 0), (false, true, 7), (false, false, 13)] {
        let mut issue = s4_issue(1, 1, 1, clear);
        issue.registers.xd |= 2048_u64 << 32;
        issue.registers.xt |= u64::from(bias) << 62;
        issue.parameters = issue.instruction.parameters(issue.registers);
        let mut memory = integer_memory();
        memory.l0a_mut().write_known(3584, &[0]).unwrap();
        memory.l0b_mut().write_known(3072, &[0]).unwrap();
        memory.bt_mut().write(2048, &7_i32.to_le_bytes()).unwrap();
        memory
            .l0c_mut()
            .buffer_mut()
            .write_known(7168, &13_i32.to_le_bytes())
            .unwrap();
        issue
            .execute(&mut memory, C220CubeExecutionControl::from_spr3(0))
            .unwrap();
        assert_eq!(
            read_u32_wrapped(memory.l0c().buffer(), 7168).unwrap(),
            expected
        );
    }
    let mut issue = s4_issue(1, 1, 1, false);
    issue.parameters.xt_bit_62 = true;
    let mut memory = integer_memory();
    memory.l0a_mut().write_known(3584, &[0]).unwrap();
    memory.l0b_mut().write_known(3072, &[0]).unwrap();
    issue
        .execute(&mut memory, C220CubeExecutionControl::from_spr3(0))
        .unwrap();
    assert_eq!(read_u32_wrapped(memory.l0c().buffer(), 7168).unwrap(), 0);
}

#[test]
fn s4_mmad_unpacks_signed_tiles_and_wraps_buffers() {
    let mut memory = integer_memory();
    let issue = s4_issue(17, 65, 33, true);
    let mut a = vec![0x88_u8; 2048];
    let mut b = vec![0x88_u8; 3072];
    let a_value = |m: usize, k: usize| ((3 * m + 5 * k) % 16) as i32 - 8;
    let b_value = |k: usize, n: usize| ((7 * k + 3 * n) % 16) as i32 - 8;
    for m in 0..17 {
        for k in 0..65 {
            let offset = (m / 16 * 2 + k / 64) * 512 + (m % 16) * 32 + (k % 64) / 2;
            let shift = 4 * (k % 2);
            a[offset] = (a[offset] & !(0xf << shift)) | ((a_value(m, k) as u8 & 0xf) << shift);
        }
    }
    for k in 0..65 {
        for n in 0..33 {
            let offset = (k / 64 * 3 + n / 16) * 512 + (n % 16) * 32 + (k % 64) / 2;
            let shift = 4 * (k % 2);
            b[offset] = (b[offset] & !(0xf << shift)) | ((b_value(k, n) as u8 & 0xf) << shift);
        }
    }
    memory.l0a_mut().write_known_wrapped(3584, &a).unwrap();
    memory.l0b_mut().write_known_wrapped(3072, &b).unwrap();
    let outcome = issue
        .execute(&mut memory, C220CubeExecutionControl::from_spr3(0))
        .unwrap();
    assert_eq!(outcome.mac_count, 17 * 65 * 33);
    assert_eq!(outcome.written_lanes, 32 * 48);
    assert_eq!(outcome.padded_lanes, 32 * 48 - 17 * 33);
    assert!(!outcome.integer_overflow);
    for m in 0..32 {
        for n in 0..48 {
            let expected: i32 = if m < 17 && n < 33 {
                (0..65).map(|k| a_value(m, k) * b_value(k, n)).sum()
            } else {
                0
            };
            let offset = (n / 16 * 2 + m / 16) * 1024 + ((m % 16) * 16 + n % 16) * 4;
            assert_eq!(
                read_u32_wrapped(memory.l0c().buffer(), 7168 + offset as u64).unwrap() as i32,
                expected
            );
        }
    }
}

#[test]
fn fp16_mmad_writes_complete_halfword_output_tiles() {
    let mut memory = integer_memory();
    let mut issue = s4_issue(1, 1, 1, true);
    issue.word = (7 << 29) | (2 << 22);
    issue.instruction = C220CubeInstruction::decode(issue.word).unwrap();
    issue.ticket = C220CubePipeline::new(C220CubeConfig::default())
        .unwrap()
        .preview_issue(
            0,
            issue.instruction,
            issue.parameters,
            C220CubeTimingControl::from_sprs(0, 0, 0),
        )
        .unwrap();
    write_u16_wrapped(memory.l0a_mut(), 3584, 0x3c00).unwrap();
    write_u16_wrapped(memory.l0b_mut(), 3072, 0x3c00).unwrap();
    let prepared = issue
        .prepare(&memory, C220CubeExecutionControl::from_spr3(0))
        .unwrap();
    assert_eq!(prepared.write_count(), 256);
    assert_eq!(prepared.outcome.padded_lanes, 255);
    prepared.commit(&mut memory).unwrap();
    let mut expected = vec![0; 512];
    expected[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
    assert_eq!(
        memory.l0c().buffer().read_known_wrapped(7168, 512).unwrap(),
        expected
    );
}

#[test]
fn s4_mmad_saturates_each_k_slice_before_accumulating_the_tail() {
    for (left, initial, expected) in [
        (0x77, i32::MAX - 10, i32::MAX - 56),
        (0x88, i32::MIN + 10, i32::MIN + 64),
    ] {
        let mut memory = integer_memory();
        let issue = s4_issue(1, 65, 1, false);
        memory
            .l0a_mut()
            .write_known_wrapped(3584, &[left; 1024])
            .unwrap();
        memory
            .l0b_mut()
            .write_known_wrapped(3072, &[0x77; 512])
            .unwrap();
        memory
            .l0b_mut()
            .write_known_wrapped(3584, &[0x88; 512])
            .unwrap();
        write_u32_wrapped(memory.l0c_mut().buffer_mut(), 7168, initial as u32).unwrap();
        let outcome = issue
            .execute(&mut memory, C220CubeExecutionControl::from_spr3(0))
            .unwrap();
        assert!(outcome.integer_overflow);
        assert_eq!(
            read_u32_wrapped(memory.l0c().buffer(), 7168).unwrap() as i32,
            expected
        );
    }
}
