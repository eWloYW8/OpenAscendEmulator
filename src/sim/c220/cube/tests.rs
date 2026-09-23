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
        execution_control: C220CubeExecutionControl::from_spr3(0),
        ticket,
    }
}

fn integer_memory() -> C220LocalMemory {
    C220LocalMemory::new(C220LocalMemoryConfig {
        l0a_bytes: 4096,
        l0b_bytes: 4096,
        l0c_bytes: 131072,
        ..C220LocalMemoryConfig::default()
    })
    .unwrap()
}

#[test]
fn accumulator_tiles_read_and_write_linear_lanes_at_the_l0c_boundary() {
    for raw_type in [2, 3, 6] {
        for base in [131071, 131072] {
            let mut issue = s4_issue(17, 1, 1, false);
            issue.word = (7 << 29) | (raw_type << 22);
            issue.instruction = C220CubeInstruction::decode(issue.word).unwrap();
            issue.registers.xd = base;
            issue.parameters = issue.instruction.parameters(issue.registers);
            let one = match raw_type {
                2 => 0x3c00_u16.to_le_bytes().to_vec(),
                3 => 1.0_f32.to_le_bytes().to_vec(),
                _ => 1_u32.to_le_bytes().to_vec(),
            };
            let tile_bytes = 256 * one.len();
            let mut memory = C220LocalMemory::new(C220LocalMemoryConfig {
                l0c_bytes: 4096,
                ..Default::default()
            })
            .unwrap();
            memory
                .l0c_mut()
                .buffer_mut()
                .write_known(0, &[0x5a; 64])
                .unwrap();
            for tile in 0..2 {
                let address = (base + tile * tile_bytes as u64) % 131072;
                memory
                    .l0c_mut()
                    .buffer_mut()
                    .write_known_linear(address, &one.repeat(256))
                    .unwrap();
            }
            let outcome = issue
                .execute(&mut memory, C220CubeExecutionControl::from_spr3(0))
                .unwrap();
            assert_eq!(outcome.written_lanes, 512);
            for row in 0..32 {
                let tile = row / 16;
                let tile_address = (base + tile * tile_bytes as u64) % 131072;
                for column in 0..16 {
                    let address = tile_address + (row % 16 * 16 + column) * one.len() as u64;
                    let expected = if row < 17 && column == 0 {
                        one.clone()
                    } else {
                        vec![0; one.len()]
                    };
                    assert_eq!(
                        memory
                            .l0c()
                            .buffer()
                            .read_initialized_linear(address, one.len())
                            .unwrap(),
                        expected
                    );
                }
            }
            if base == 131071 {
                assert_eq!(memory.l0c().buffer().read_known(0, 64).unwrap(), [0x5a; 64]);
            }
        }
    }
}

#[test]
fn input_tiles_normalize_the_base_without_wrapping_individual_lanes() {
    for raw_type in [0, 1, 2, 3, 5, 6, 9, 10] {
        for base in [65535, 65536, 131071] {
            for m in [1, 2] {
                let mut issue = s4_issue(m, 1, 2, true);
                issue.word = (7 << 29) | ((raw_type & 7) << 22) | (raw_type >> 3);
                issue.instruction = C220CubeInstruction::decode(issue.word).unwrap();
                issue.registers.xn = base;
                issue.registers.xm = base;
                issue.parameters = issue.instruction.parameters(issue.registers);
                let (a, b) = match raw_type {
                    2 | 3 => (
                        0x4000_u16.to_le_bytes().to_vec(),
                        0x4200_u16.to_le_bytes().to_vec(),
                    ),
                    9 => (
                        0x4000_u16.to_le_bytes().to_vec(),
                        0x4040_u16.to_le_bytes().to_vec(),
                    ),
                    10 => (
                        2.0_f32.to_le_bytes().to_vec(),
                        3.0_f32.to_le_bytes().to_vec(),
                    ),
                    6 => (vec![0x22], vec![0x33]),
                    _ => (vec![2], vec![3]),
                };
                let mut memory = integer_memory();
                memory
                    .l0a_mut()
                    .write_known_linear(base % 65536, &a.repeat(512 / a.len()))
                    .unwrap();
                memory
                    .l0b_mut()
                    .write_known_linear(base % 65536, &b.repeat(512 / b.len()))
                    .unwrap();
                issue
                    .execute(&mut memory, C220CubeExecutionControl::from_spr3(0))
                    .unwrap();
                for row in 0..u64::from(m) {
                    for column in 0..2 {
                        if raw_type == 2 {
                            assert_eq!(
                                read_u16_wrapped(
                                    memory.l0c().buffer(),
                                    f16_c_address(7168, 1, row, column)
                                )
                                .unwrap(),
                                0x4600
                            );
                        } else {
                            let expected = if matches!(raw_type, 3 | 9 | 10) {
                                6.0_f32.to_bits()
                            } else {
                                6
                            };
                            assert_eq!(
                                read_u32_wrapped(
                                    memory.l0c().buffer(),
                                    f32_c_address(7168, 1, row, column)
                                )
                                .unwrap(),
                                expected,
                                "dtype={raw_type} base={base} row={row} col={column}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn zero_dimension_mmad_does_not_access_or_modify_memory() {
    for (m, k, n) in [(0, 17, 17), (17, 0, 17), (17, 17, 0)] {
        for sparse in [false, true] {
            for controls in [0, 1 << 62, 1 << 63, (127 << 44) | (1 << 58)] {
                let mut issue = s4_issue(m, k, n, false);
                if sparse {
                    issue.word |= 5 << 25;
                    issue.instruction = C220CubeInstruction::decode(issue.word).unwrap();
                }
                issue.registers.xt |= controls;
                issue.registers.xn = u64::MAX;
                issue.registers.xm = u64::MAX;
                issue.registers.xd = u64::MAX;
                issue.parameters = issue.instruction.parameters(issue.registers);
                let mut memory = integer_memory();
                memory
                    .l0c_mut()
                    .buffer_mut()
                    .write_known(0, &[0xa5; 8192])
                    .unwrap();
                let before = memory.clone();
                let outcome = issue
                    .execute(&mut memory, C220CubeExecutionControl::from_spr3(0))
                    .unwrap();
                assert_eq!(outcome.written_lanes, 0);
                assert_eq!(outcome.mac_count, 0);
                assert_eq!(memory, before);
            }
        }
    }
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
fn sparse_int4_mmad_sign_extends_selected_nibbles() {
    for raw_k in [32, 128, 256] {
        let mut issue = s4_issue(8, raw_k, 17, true);
        issue.word = (7 << 29) | (5 << 25) | (6 << 22);
        issue.instruction = C220CubeInstruction::decode(issue.word).unwrap();
        issue.registers.xn = 0;
        issue.registers.xm = 16384;
        issue.registers.xd = 0;
        issue.parameters = issue.instruction.parameters(issue.registers);
        let mut memory = C220LocalMemory::new(Default::default()).unwrap();
        memory.l0a_mut().write_known(0, &[0xf2; 4096]).unwrap();
        memory.l0b_mut().write_known(16384, &[0x21; 4096]).unwrap();
        memory
            .weight_index_mut()
            .write_known_linear(4096, &[0x55; 1024])
            .unwrap();
        let outcome = issue
            .execute(&mut memory, C220CubeExecutionControl::from_spr3(0))
            .unwrap();
        assert!(!outcome.integer_overflow);
        let expected = u32::from(issue.parameters.effective_k) / 2 * 3;
        for row in 0..8 {
            for column in 0..17 {
                assert_eq!(
                    read_u32_wrapped(memory.l0c().buffer(), f32_c_address(0, 1, row, column))
                        .unwrap(),
                    expected
                );
            }
        }
        let mut parameters = issue.parameters;
        parameters.m = 9;
        assert!(matches!(
            super::sparse::read_nibble_pair(parameters, &memory, 8, 0, 0),
            Err(C220CubeExecutionError::SparseUninitializedRow {
                row: 8,
                initialized_rows: 8
            })
        ));
    }
}

#[test]
fn sparse_f32_mmad_reassembles_selected_halfwords() {
    for (raw_k, expected) in [(64, 48.0_f32), (34, 26.0)] {
        for padded in [false, true] {
            for hf32 in [false, true] {
                let mut issue = s4_issue(17, raw_k, 17, true);
                issue.word = (7 << 29) | (5 << 25) | (2 << 22) | 1;
                issue.instruction = C220CubeInstruction::decode(issue.word).unwrap();
                issue.registers.xn = 0;
                issue.registers.xm = 16384;
                issue.registers.xd = 0;
                issue.registers.xt |= u64::from(padded) << 58;
                issue.parameters = issue.instruction.parameters(issue.registers);
                let k = u64::from(issue.parameters.effective_k);
                let tiles = k.div_ceil(8);
                let mut memory = C220LocalMemory::new(Default::default()).unwrap();
                for row in 0..17 {
                    for dense_k in 0..2 * tiles * 8 {
                        let element = integer_a_element(2 * tiles, 8, row, dense_k);
                        let mut address = 4 * element;
                        if padded {
                            address += (address / 512 / tiles) * (2 * k.div_ceil(16) - tiles) * 512;
                        }
                        let value = if dense_k % 2 == 0 { 1.0_f32 } else { 2.0 };
                        memory
                            .l0a_mut()
                            .write_known(address, &value.to_le_bytes())
                            .unwrap();
                    }
                }
                for column in 0..17 {
                    for lane in 0..k {
                        memory
                            .l0b_mut()
                            .write_known(
                                f32_b_address(16384, 2, lane, column),
                                &1.0_f32.to_le_bytes(),
                            )
                            .unwrap();
                    }
                    for slice in 0..tiles {
                        memory
                            .weight_index_mut()
                            .write_known_linear(
                                4096 + 8 * (column + slice * 32),
                                &[if slice % 2 == 0 { 0 } else { 0xaa }; 4],
                            )
                            .unwrap();
                    }
                }
                issue
                    .execute(
                        &mut memory,
                        C220CubeExecutionControl::from_spr3(u64::from(hf32) << 46),
                    )
                    .unwrap();
                for row in 0..17 {
                    for column in 0..17 {
                        assert_eq!(
                            read_u32_wrapped(
                                memory.l0c().buffer(),
                                f32_c_address(0, 2, row, column),
                            )
                            .unwrap(),
                            expected.to_bits()
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn sparse_half_mmad_uses_half_slices_and_compact_input_tiles() {
    for raw_type in [2, 3, 9] {
        for (m, raw_k, expected) in [(17, 64, 40.0_f32), (17, 34, 19.0), (1, 32, 8.0)] {
            let mut issue = s4_issue(m, raw_k, 17, true);
            issue.word = (7 << 29) | (5 << 25) | ((raw_type & 7) << 22) | (raw_type >> 3);
            issue.instruction = C220CubeInstruction::decode(issue.word).unwrap();
            issue.registers.xn = 0;
            issue.registers.xm = 16384;
            issue.registers.xd = 0;
            issue.parameters = issue.instruction.parameters(issue.registers);
            let k = u64::from(issue.parameters.effective_k);
            let tiles = 2 * k.div_ceil(16) - u64::from((1..=16).contains(&(k & 31)));
            let one: u16 = if raw_type == 9 { 0x3f80 } else { 0x3c00 };
            let mut memory = C220LocalMemory::new(Default::default()).unwrap();
            for row in 0..u64::from(m) {
                for dense_k in 0..tiles * 16 {
                    let value = if dense_k % 4 == 3 { 0x4000 } else { one };
                    memory
                        .l0a_mut()
                        .write_known(
                            2 * integer_a_element(tiles, 16, row, dense_k),
                            &value.to_le_bytes(),
                        )
                        .unwrap();
                }
            }
            for column in 0..17 {
                for lane in 0..k {
                    memory
                        .l0b_mut()
                        .write_known(
                            16384 + 2 * integer_b_element(2, 16, lane, column),
                            &one.to_le_bytes(),
                        )
                        .unwrap();
                }
                for slice in 0..k.div_ceil(16) {
                    memory
                        .weight_index_mut()
                        .write_known_linear(
                            4096 + 8 * (column + slice * 32),
                            &[if slice == 0 { 0 } else { 0x77 }; 4],
                        )
                        .unwrap();
                }
            }
            if raw_k == 32 {
                let mut parameters = issue.parameters;
                parameters.m = 17;
                assert!(matches!(
                    super::sparse::read_half_pair(parameters, &memory, 16, 0, 8),
                    Err(C220CubeExecutionError::SparseInputOutsideLoadedTiles {
                        dense_k: 16,
                        loaded_k: 16,
                    })
                ));
            }
            issue
                .execute(&mut memory, C220CubeExecutionControl::from_spr3(0))
                .unwrap();
            for row in 0..u64::from(m) {
                for column in 0..17 {
                    if raw_type == 2 {
                        let bits: u16 = match raw_k {
                            64 => 0x5100,
                            34 => 0x4cc0,
                            _ => 0x4800,
                        };
                        assert_eq!(
                            memory
                                .l0c()
                                .buffer()
                                .read_known(
                                    f16_c_address(0, u64::from(m.div_ceil(16)), row, column),
                                    2,
                                )
                                .unwrap(),
                            bits.to_le_bytes()
                        );
                    } else {
                        assert_eq!(
                            read_u32_wrapped(
                                memory.l0c().buffer(),
                                f32_c_address(0, u64::from(m.div_ceil(16)), row, column),
                            )
                            .unwrap(),
                            expected.to_bits()
                        );
                    }
                }
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
fn inactive_weight_offset_index_does_not_change_mmad() {
    let memory = C220LocalMemory::new(Default::default()).unwrap();
    let control = C220CubeExecutionControl::from_spr3(0);
    for raw_type in [0, 1, 2, 3, 5, 6, 9, 10] {
        for operation in [0, 5] {
            let mut issue = s4_issue(1, 4, 1, true);
            issue.word = (7 << 29) | (operation << 25) | ((raw_type & 7) << 22) | (raw_type >> 3);
            issue.instruction = C220CubeInstruction::decode(issue.word).unwrap();
            issue.parameters = issue.instruction.parameters(issue.registers);
            let expected = issue.prepare(&memory, control).unwrap();
            for index in [1, 63, 127] {
                issue.registers.xt = (issue.registers.xt & !(127 << 44)) | (index << 44);
                issue.parameters = issue.instruction.parameters(issue.registers);
                assert_eq!(issue.prepare(&memory, control).unwrap(), expected);
            }
        }
    }
}

#[test]
fn non_f32_mmad_ignores_input_padding_control() {
    let mut memory = C220LocalMemory::new(Default::default()).unwrap();
    memory.l0a_mut().write_known(0, &[0x11; 8192]).unwrap();
    memory.l0b_mut().write_known(0, &[0x22; 8192]).unwrap();
    let control = C220CubeExecutionControl::from_spr3(0);
    for raw_type in [0, 1, 2, 3, 5, 6, 9] {
        for operation in [0, 5] {
            let mut issue = s4_issue(2, 64, 17, true);
            issue.word = (7 << 29) | (operation << 25) | ((raw_type & 7) << 22) | (raw_type >> 3);
            issue.instruction = C220CubeInstruction::decode(issue.word).unwrap();
            issue.parameters = issue.instruction.parameters(issue.registers);
            let expected = issue.prepare(&memory, control).unwrap();
            issue.registers.xt |= 1 << 58;
            issue.parameters = issue.instruction.parameters(issue.registers);
            assert_eq!(issue.prepare(&memory, control).unwrap(), expected);
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
    let mut memory = C220LocalMemory::new(C220LocalMemoryConfig {
        l0a_bytes: 65536,
        l0b_bytes: 65536,
        l0c_bytes: 131072,
        ..Default::default()
    })
    .unwrap();
    let mut issue = s4_issue(17, 65, 33, true);
    issue.registers.xn = 65024;
    issue.registers.xm = 64512;
    issue.parameters = issue.instruction.parameters(issue.registers);
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
    memory.l0a_mut().write_known_wrapped(65024, &a).unwrap();
    memory.l0b_mut().write_known_wrapped(64512, &b).unwrap();
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
