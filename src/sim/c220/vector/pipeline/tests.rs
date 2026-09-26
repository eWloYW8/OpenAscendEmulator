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
fn sort_functional_completion_observes_live_inputs_and_repeat_feedback() {
    use crate::sim::c220::numeric::fp16::to_f64;
    use crate::sim::c220::vector::C220VectorInstruction;
    use crate::sim::c220::vector::ops::sort::plan_c220_sort_issue;

    for (word, width) in [(0x8540_0002, 2), (0x85c0_0002, 4)] {
        for destination in [0, 256, 1024] {
            let mut bytes = vec![0_u8; 2048];
            for repeat in 0..2 {
                for lane in 0..32 {
                    let value = if width == 2 {
                        u32::from(0x3c00_u16 + lane as u16 * 32)
                    } else {
                        (lane as f32 + 1.0).to_bits()
                    };
                    let offset = repeat * 256 + lane * width;
                    bytes[offset..offset + width].copy_from_slice(&value.to_le_bytes()[..width]);
                    let offset = 768 + repeat * 256 + lane * 4;
                    bytes[offset..offset + 4].copy_from_slice(&(lane as f32 + 100.0).to_le_bytes());
                }
            }
            let mut ub = UbMemory::new(2048, 2048);
            ub.write_states(0, &vec![MemoryByteState::Known(0); 2048])
                .unwrap();
            let instruction = C220VectorInstruction::Sort(
                plan_c220_sort_issue(
                    0,
                    word,
                    2 << 56,
                    C220VectorAddresses {
                        source_0: 0,
                        source_1: 768,
                        destination,
                    },
                    &ub,
                )
                .unwrap(),
            );
            let mut pipeline = C220VectorPipeline::new(C220VectorTimingRules {
                dispatch_ticks: 0,
                uop_issue_interval: NonZeroU64::new(1).unwrap(),
                ub_response_ticks: 2,
            });
            pipeline
                .issue_at(
                    0,
                    &instruction.uops().unwrap(),
                    instruction.stores(),
                    instruction.read_issue(),
                )
                .unwrap();
            ub.write_states(
                0,
                &bytes
                    .iter()
                    .copied()
                    .map(MemoryByteState::Known)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            let machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
            let mut core = C220State::new(ScalarStepper::new(machine, 0), ub);
            for repeat in 0..2 {
                let mut records = (0..32)
                    .map(|lane| {
                        let offset = repeat * 256 + lane * width;
                        let mut record = [0_u8; 8];
                        record[..width].copy_from_slice(&bytes[offset..offset + width]);
                        let offset = 768 + repeat * 256 + lane * 4;
                        record[4..].copy_from_slice(&bytes[offset..offset + 4]);
                        record
                    })
                    .collect::<Vec<_>>();
                let value = |record: &[u8; 8]| {
                    if width == 2 {
                        to_f64(u16::from_le_bytes(record[..2].try_into().unwrap()))
                    } else {
                        f64::from(f32::from_le_bytes(record[..4].try_into().unwrap()))
                    }
                };
                records.sort_by(|left, right| value(right).partial_cmp(&value(left)).unwrap());
                for (lane, record) in records.iter().enumerate() {
                    let offset = destination as usize + repeat * 256 + lane * 8;
                    bytes[offset..offset + 8].copy_from_slice(record);
                }
            }
            let releases = pipeline.advance_to(512, &mut core).unwrap();
            let samples = pipeline.last_functional_samples();
            assert_eq!(samples.len(), 2);
            assert!(
                samples
                    .iter()
                    .all(|sample| sample.tick == releases.last().unwrap().release_tick + 1)
            );
            assert!(
                pipeline
                    .last_read_samples()
                    .iter()
                    .all(|sample| sample.sort_lanes.is_none())
            );
            assert_eq!(core.ub().read_known(0, 2048).unwrap(), bytes);
            assert_eq!(pipeline.pending_uops(), 0);
            assert_eq!(pipeline.pending_ub_responses(), 0);
        }
    }
}

#[test]
fn accumulator_reads_use_two_ports_and_share_aliased_grants() {
    use crate::sim::c220::numeric::fp16::C220Fp16Mode;
    use crate::sim::c220::vector::C220VectorInstruction;
    use crate::sim::c220::vector::ops::axpy::{C220AxpyIssueInputs, plan_c220_axpy_issue};
    use crate::sim::c220::vector::ops::ternary::plan_c220_ternary_issue;

    for word in [0x89c0_0001, 0x93c0_0000, 0x93c0_0001, 0x95c0_0000] {
        for (source_1, destination) in [(512, 1024), (512, 0), (0, 0)] {
            let axpy = word == 0x95c0_0000;
            let mut ub = UbMemory::new(4096, 256);
            let bytes = 2.0_f32.to_le_bytes().repeat(8);
            for address in [0, source_1, destination] {
                ub.write_states(
                    address,
                    &bytes
                        .iter()
                        .copied()
                        .map(MemoryByteState::Known)
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            }
            let control = C220VectorControl::decode_binary(1 << 56);
            let addresses = C220VectorAddresses {
                source_0: 0,
                source_1,
                destination,
            };
            let masks = [[0xff, 0, 0, 0]];
            let instruction = if axpy {
                C220VectorInstruction::Axpy(
                    plan_c220_axpy_issue(
                        C220AxpyIssueInputs {
                            pc: 0,
                            word,
                            scalar_bits: 2.0_f32.to_bits(),
                            control,
                            addresses,
                            iteration_masks: &masks,
                            fp16_mode: C220Fp16Mode::NonSaturating,
                        },
                        &ub,
                    )
                    .unwrap(),
                )
            } else {
                C220VectorInstruction::Ternary(
                    plan_c220_ternary_issue(
                        0,
                        word,
                        control,
                        addresses,
                        &masks,
                        C220Fp16Mode::NonSaturating,
                        &ub,
                    )
                    .unwrap(),
                )
            };
            let machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
            let mut state = C220State::new(ScalarStepper::new(machine, 0), ub);
            let mut pipeline = C220VectorPipeline::new(C220VectorTimingRules {
                dispatch_ticks: 0,
                uop_issue_interval: NonZeroU64::new(1).unwrap(),
                ub_response_ticks: 2,
            });
            pipeline
                .issue_at(
                    0,
                    &instruction.uops().unwrap(),
                    instruction.stores(),
                    instruction.read_issue(),
                )
                .unwrap();
            pipeline.advance_to(128, &mut state).unwrap();
            let sample = &pipeline.last_read_samples()[0];
            let source_requests = if axpy || source_1 == 0 { 1 } else { 2 };
            assert_eq!(sample.read0_grants.len(), source_requests);
            assert_eq!(sample.read1_grants.len(), usize::from(destination != 0));
            assert!(sample.lanes.is_empty());
            let sample = &pipeline.last_functional_samples()[0];
            assert_eq!(&sample.source_0_bytes[..32], &bytes);
            if !axpy {
                assert_eq!(&sample.source_1_bytes[..32], &bytes);
            }
            assert_eq!(&sample.destination_bytes[..32], &bytes);
            assert!(
                pipeline
                    .last_ub_cycles()
                    .iter()
                    .flat_map(|cycle| &cycle.decisions)
                    .all(|decision| decision.port != C220UbPort::VectorReadDestination)
            );
            assert_eq!(
                state.ub().read_known(destination, 32).unwrap(),
                6.0_f32.to_le_bytes().repeat(8)
            );
        }
    }
}

#[test]
fn accumulator_repeats_separate_bypass_traffic_from_numerical_feedback() {
    use crate::sim::c220::numeric::fp16::C220Fp16Mode;
    use crate::sim::c220::vector::C220VectorInstruction;
    use crate::sim::c220::vector::ops::axpy::{C220AxpyIssueInputs, plan_c220_axpy_issue};
    use crate::sim::c220::vector::ops::ternary::plan_c220_ternary_issue;

    for word in [0x89c0_0001, 0x93c0_0000, 0x93c0_0001, 0x95c0_0000] {
        for destination_stride in [0, 8] {
            for destination in [0, 1024] {
                let axpy = word == 0x95c0_0000;
                let control = C220VectorControl {
                    encoded_repeat_count: 3,
                    destination_block_stride: 1,
                    source_0_block_stride: 1,
                    source_1_block_stride: 1,
                    destination_repeat_stride: destination_stride,
                    source_0_repeat_stride: 0,
                    source_1_repeat_stride: 0,
                };
                let mut values = vec![2.0_f32; 1024];
                values[128..192].fill(3.0);
                if destination != 0 {
                    values[256..448].fill(4.0);
                }
                let initial = values.clone();
                let mut ub = UbMemory::new(4096, 4096);
                ub.write_states(
                    0,
                    &values
                        .iter()
                        .flat_map(|value| value.to_le_bytes().map(MemoryByteState::Known))
                        .collect::<Vec<_>>(),
                )
                .unwrap();
                let addresses = C220VectorAddresses {
                    source_0: 0,
                    source_1: 512,
                    destination,
                };
                let masks = [[u64::MAX, 0, 0, 0]; 3];
                let instruction = if axpy {
                    C220VectorInstruction::Axpy(
                        plan_c220_axpy_issue(
                            C220AxpyIssueInputs {
                                pc: 0,
                                word,
                                scalar_bits: 2.0_f32.to_bits(),
                                control,
                                addresses,
                                iteration_masks: &masks,
                                fp16_mode: C220Fp16Mode::NonSaturating,
                            },
                            &ub,
                        )
                        .unwrap(),
                    )
                } else {
                    C220VectorInstruction::Ternary(
                        plan_c220_ternary_issue(
                            0,
                            word,
                            control,
                            addresses,
                            &masks,
                            C220Fp16Mode::NonSaturating,
                            &ub,
                        )
                        .unwrap(),
                    )
                };
                for repeat in 0..3 {
                    let inputs = values.clone();
                    let first =
                        destination as usize / 4 + repeat * usize::from(destination_stride) * 8;
                    for lane in 0..64 {
                        let source = inputs[lane];
                        let other = inputs[128 + lane];
                        let previous = inputs[first + lane];
                        values[first + lane] = match word {
                            0x89c0_0001 => source * other + previous,
                            0x95c0_0000 => source * 2.0 + previous,
                            _ => source * previous + other,
                        };
                    }
                }
                let machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
                let mut core = C220State::new(ScalarStepper::new(machine, 0), ub);
                let mut pipeline = C220VectorPipeline::new(C220VectorTimingRules {
                    dispatch_ticks: 0,
                    uop_issue_interval: NonZeroU64::new(1).unwrap(),
                    ub_response_ticks: 2,
                });
                let uops = instruction.uops().unwrap();
                let bypass = destination_stride == 0;
                assert_eq!(
                    uops.len(),
                    match (axpy, bypass) {
                        (true, true) => 2,
                        (true, false) => 3,
                        (false, true) => 4,
                        (false, false) => 6,
                    }
                );
                assert!(
                    uops.iter()
                        .filter(|uop| uop.repeat_index < 2)
                        .all(|uop| uop.writes_ub != bypass)
                );
                pipeline
                    .issue_at(0, &uops, instruction.stores(), instruction.read_issue())
                    .unwrap();
                pipeline.advance_to(12, &mut core).unwrap();
                assert!(pipeline.last_functional_samples().is_empty());
                assert_eq!(
                    core.ub().read_known(0, 4096).unwrap(),
                    initial
                        .iter()
                        .flat_map(|value| value.to_le_bytes())
                        .collect::<Vec<_>>()
                );
                let releases = pipeline.advance_to(256, &mut core).unwrap();
                assert_eq!(pipeline.last_functional_samples().len(), 3);
                assert_eq!(
                    core.ub().read_known(0, 4096).unwrap(),
                    values
                        .iter()
                        .flat_map(|value| value.to_le_bytes())
                        .collect::<Vec<_>>()
                );
                assert!(
                    pipeline
                        .last_read_samples()
                        .iter()
                        .filter(|sample| sample.repeat_index > 0)
                        .all(|sample| sample.accesses.iter().all(|access| {
                            access.source_index != 0 && (!bypass || access.source_index != 2)
                        }))
                );
                if bypass {
                    for pair in releases.windows(2) {
                        if pair[0].repeat_index != pair[1].repeat_index {
                            assert!(pair[1].conflict_check_tick >= pair[0].conflict_check_tick + 8);
                        }
                    }
                }
                assert!(pipeline.pending_drain_tick().is_none());
            }
        }
    }
}

#[test]
fn arithmetic_accumulator_and_scalar_complete_in_order_with_overlapping_timing() {
    use crate::sim::c220::numeric::fp16::C220Fp16Mode;
    use crate::sim::c220::vector::C220VectorInstruction;
    use crate::sim::c220::vector::ops::scalar::{
        C220VectorScalarOperand, plan_c220_vector_scalar_issue,
    };
    use crate::sim::c220::vector::ops::ternary::plan_c220_ternary_issue;

    let mut ub = UbMemory::new(4096, 256);
    for (address, value) in [(0, 2.0_f32), (512, 3.0), (1024, 4.0)] {
        ub.write_states(
            address,
            &value
                .to_le_bytes()
                .repeat(8)
                .into_iter()
                .map(MemoryByteState::Known)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    }
    let control = C220VectorControl::decode_binary(0x0100_0808_0801_0101);
    let masks = [[0xff, 0, 0, 0]];
    let arithmetic = |pc, source, destination| {
        C220VectorInstruction::Arithmetic(
            plan_c220_vector_arithmetic_issue(
                pc,
                0x85c0_0000,
                control,
                C220VectorAddresses {
                    source_0: source,
                    source_1: 512,
                    destination,
                },
                &masks,
                C220VectorArithmeticModes::from_control_spr(0),
                &ub,
            )
            .unwrap(),
        )
    };
    let instructions = [
        arithmetic(0, 0, 256),
        C220VectorInstruction::Ternary(
            plan_c220_ternary_issue(
                4,
                0x89c0_0001,
                control,
                C220VectorAddresses {
                    source_0: 256,
                    source_1: 512,
                    destination: 1024,
                },
                &masks,
                C220Fp16Mode::NonSaturating,
                &ub,
            )
            .unwrap(),
        ),
        arithmetic(8, 1024, 1536),
        C220VectorInstruction::Scalar(
            plan_c220_vector_scalar_issue(
                12,
                0x97c0_0001,
                C220VectorScalarOperand {
                    bits: 2.0_f32.to_bits(),
                    fp16_mode: C220Fp16Mode::NonSaturating,
                    integer_saturating: false,
                },
                control,
                C220VectorAddresses {
                    source_0: 1536,
                    source_1: 0,
                    destination: 2048,
                },
                &masks,
                &ub,
            )
            .unwrap(),
        ),
    ];
    let machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
    let mut core = C220State::new(ScalarStepper::new(machine, 0), ub);
    let mut pipeline = C220VectorPipeline::new(C220VectorTimingRules {
        dispatch_ticks: 0,
        uop_issue_interval: NonZeroU64::new(1).unwrap(),
        ub_response_ticks: 2,
    });
    for instruction in instructions {
        pipeline
            .issue_at(
                0,
                &instruction.uops().unwrap(),
                instruction.stores(),
                instruction.read_issue(),
            )
            .unwrap();
    }
    pipeline.advance_to(256, &mut core).unwrap();
    for (address, value) in [(256, 5.0_f32), (1024, 19.0), (1536, 22.0), (2048, 44.0)] {
        assert_eq!(
            core.ub().read_known(address, 32).unwrap(),
            value.to_le_bytes().repeat(8)
        );
    }
    let results = pipeline.last_functional_samples();
    assert_eq!(
        results.iter().map(|sample| sample.pc).collect::<Vec<_>>(),
        [0, 4, 8, 12]
    );
    assert!(results.windows(2).all(|pair| pair[0].tick < pair[1].tick));
    let consumer_read = pipeline
        .last_read_samples()
        .iter()
        .find(|sample| sample.pc == 4)
        .unwrap();
    assert!(consumer_read.tick < results[0].tick);
    assert_eq!(&results[1].source_0_bytes[..4], &5.0_f32.to_le_bytes());
}

#[test]
fn ordinary_repeat_bypass_preserves_live_alias_feedback() {
    use crate::sim::c220::numeric::fp16::C220Fp16Mode;
    use crate::sim::c220::vector::C220VectorInstruction;
    use crate::sim::c220::vector::ops::scalar::{
        C220VectorScalarOperand, plan_c220_vector_scalar_issue,
    };

    for (word, repeat_count) in [0x85c0_0000, 0x89c0_0000, 0x97c0_0001]
        .into_iter()
        .flat_map(|word| [1, 3].map(|repeats| (word, repeats)))
    {
        for (source_1, destination) in [(512, 0), (512, 512), (0, 1024), (0, 0)] {
            let scalar = word == 0x97c0_0001;
            let control = C220VectorControl {
                encoded_repeat_count: repeat_count,
                destination_block_stride: 1,
                source_0_block_stride: 1,
                source_1_block_stride: 1,
                destination_repeat_stride: 0,
                source_0_repeat_stride: 0,
                source_1_repeat_stride: 0,
            };
            let mut values = vec![2.0_f32; 1024];
            values[128..192].fill(3.0);
            let mut ub = UbMemory::new(4096, 4096);
            ub.write_states(
                0,
                &values
                    .iter()
                    .flat_map(|value| value.to_le_bytes().map(MemoryByteState::Known))
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            let addresses = C220VectorAddresses {
                source_0: 0,
                source_1,
                destination,
            };
            let masks = vec![[u64::MAX, 0, 0, 0]; usize::from(repeat_count)];
            let instruction = if scalar {
                C220VectorInstruction::Scalar(
                    plan_c220_vector_scalar_issue(
                        0,
                        word,
                        C220VectorScalarOperand {
                            bits: 2.0_f32.to_bits(),
                            fp16_mode: C220Fp16Mode::NonSaturating,
                            integer_saturating: false,
                        },
                        control,
                        addresses,
                        &masks,
                        &ub,
                    )
                    .unwrap(),
                )
            } else {
                C220VectorInstruction::Arithmetic(
                    plan_c220_vector_arithmetic_issue(
                        0,
                        word,
                        control,
                        addresses,
                        &masks,
                        C220VectorArithmeticModes::from_control_spr(0),
                        &ub,
                    )
                    .unwrap(),
                )
            };
            for _ in 0..repeat_count {
                let inputs = values.clone();
                for lane in 0..64 {
                    let first = inputs[lane];
                    let second = if scalar {
                        2.0
                    } else {
                        inputs[source_1 as usize / 4 + lane]
                    };
                    values[destination as usize / 4 + lane] = if word == 0x85c0_0000 {
                        first + second
                    } else {
                        first * second
                    };
                }
            }
            let machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
            let mut core = C220State::new(ScalarStepper::new(machine, 0), ub);
            let mut pipeline = C220VectorPipeline::new(C220VectorTimingRules {
                dispatch_ticks: 0,
                uop_issue_interval: NonZeroU64::new(1).unwrap(),
                ub_response_ticks: 2,
            });
            pipeline
                .issue_at(
                    0,
                    &instruction.uops().unwrap(),
                    instruction.stores(),
                    instruction.read_issue(),
                )
                .unwrap();
            let releases = pipeline.advance_to(256, &mut core).unwrap();
            let feedback = !scalar && source_1 == destination;
            for release in &releases {
                assert_eq!(
                    release.ub_write_requested,
                    !feedback
                        || (release.repeat_index > 0
                            && release.repeat_index + 1 == usize::from(repeat_count))
                );
            }
            if feedback && repeat_count == 3 {
                assert!(releases[2].conflict_check_tick >= releases[1].conflict_check_tick + 4);
                assert!(releases[2].admission_tick < releases[1].conflict_check_tick + 4);
            }
            assert_eq!(
                core.ub().read_known(0, 4096).unwrap(),
                values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>()
            );
            let timing = pipeline.last_read_samples();
            assert_eq!(timing.len(), usize::from(repeat_count));
            for sample in timing {
                let first_read = sample.repeat_index == 0 || (!scalar && source_1 == destination);
                assert_eq!(
                    sample
                        .accesses
                        .iter()
                        .any(|access| access.source_index == 0),
                    first_read
                );
                assert_eq!(
                    sample
                        .accesses
                        .iter()
                        .any(|access| access.source_index == 1),
                    !scalar && source_1 != 0
                );
                assert!(sample.lanes.is_empty());
            }
            assert_eq!(
                pipeline.last_functional_samples().len(),
                usize::from(repeat_count)
            );
            assert!(pipeline.pending_drain_tick().is_none());
        }
    }
}

#[test]
fn fused_conversion_chain_snapshots_full_repeats_before_aliased_writes() {
    use crate::sim::c220::numeric::fp16::C220Fp16Mode;
    use crate::sim::c220::vector::C220VectorInstruction;
    use crate::sim::c220::vector::ops::conversion::{
        C220ConversionIssueInputs, plan_c220_conversion_issue,
    };
    use crate::sim::c220::vector::ops::fused::{C220FusedIssueInputs, plan_c220_fused_issue};

    let mut ub = UbMemory::new(4096, 256);
    for (address, value) in [(0, 0x4000_u16), (1024, 0x4200)] {
        ub.write_states(
            address,
            &value
                .to_le_bytes()
                .repeat(128)
                .into_iter()
                .map(MemoryByteState::Known)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    }
    let masks = [[u64::MAX, u64::MAX, 0, 0]];
    let control = C220VectorControl::decode_binary(0x0100_0808_0801_0101);
    let instructions = [
        C220VectorInstruction::Fused(
            plan_c220_fused_issue(
                C220FusedIssueInputs {
                    pc: 0,
                    word: 0x9e40_0001,
                    control,
                    addresses: C220VectorAddresses {
                        source_0: 0,
                        source_1: 1024,
                        destination: 128,
                    },
                    iteration_masks: &masks,
                    fp16_mode: C220Fp16Mode::NonSaturating,
                    arithmetic_saturating: false,
                    integer_saturating: true,
                    descriptor_address: 0,
                    deq_scale: 0,
                },
                &ub,
            )
            .unwrap(),
        ),
        C220VectorInstruction::Conversion(
            plan_c220_conversion_issue(
                C220ConversionIssueInputs {
                    pc: 4,
                    word: 0x8c00_0000 | (6 << 22) | (1 << 3),
                    control,
                    addresses: C220VectorAddresses {
                        source_0: 128,
                        source_1: 0,
                        destination: 512,
                    },
                    iteration_masks: &masks,
                    fp16_mode: C220Fp16Mode::NonSaturating,
                    integer_saturating: true,
                    deq_scale: 0,
                },
                &ub,
            )
            .unwrap(),
        ),
    ];
    let machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
    let mut core = C220State::new(ScalarStepper::new(machine, 0), ub);
    let mut pipeline = C220VectorPipeline::new(C220VectorTimingRules {
        dispatch_ticks: 0,
        uop_issue_interval: NonZeroU64::new(1).unwrap(),
        ub_response_ticks: 2,
    });
    for instruction in instructions {
        pipeline
            .issue_at(
                0,
                &instruction.uops().unwrap(),
                instruction.stores(),
                instruction.read_issue(),
            )
            .unwrap();
    }
    pipeline.advance_to(256, &mut core).unwrap();
    assert_eq!(core.ub().read_known(128, 128).unwrap(), vec![6; 128]);
    assert_eq!(
        core.ub().read_known(512, 256).unwrap(),
        0x4600_u16.to_le_bytes().repeat(128)
    );
    let samples = pipeline.last_functional_samples();
    assert_eq!(samples.len(), 2);
    assert_eq!(
        samples[0]
            .fused_lanes
            .as_ref()
            .unwrap()
            .iter()
            .filter(|lane| lane.active)
            .count(),
        128
    );
    assert_eq!(
        samples[1]
            .conversion_lanes
            .as_ref()
            .unwrap()
            .iter()
            .filter(|lane| lane.active)
            .count(),
        128
    );
    assert!(samples[0].tick < samples[1].tick);
}

#[test]
fn conversion_uses_live_deqscale_while_fused_keeps_its_captured_descriptor_address() {
    use crate::sim::c220::numeric::fp16::C220Fp16Mode;
    use crate::sim::c220::vector::C220VectorInstruction;
    use crate::sim::c220::vector::ops::conversion::{
        C220ConversionIssueInputs, plan_c220_conversion_issue,
    };
    use crate::sim::c220::vector::ops::fused::{C220FusedIssueInputs, plan_c220_fused_issue};

    let mut ub = UbMemory::new(4096, 256);
    for (address, bytes) in [
        (0, 2_i16.to_le_bytes().repeat(128)),
        (512, 3_i16.to_le_bytes().repeat(128)),
        (1024, u64::from(2.0_f32.to_bits()).to_le_bytes().repeat(16)),
        (2048, u64::from(3.0_f32.to_bits()).to_le_bytes().repeat(16)),
    ] {
        ub.write_states(
            address,
            &bytes
                .into_iter()
                .map(MemoryByteState::Known)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    }
    let masks = [[1, 0, 0, 0]];
    let control = C220VectorControl::decode_binary(0x0100_0808_0801_0101);
    let instructions = [
        C220VectorInstruction::Fused(
            plan_c220_fused_issue(
                C220FusedIssueInputs {
                    pc: 0,
                    word: 0x8ac0_0000,
                    control,
                    addresses: C220VectorAddresses {
                        source_0: 0,
                        source_1: 512,
                        destination: 3072,
                    },
                    iteration_masks: &masks,
                    fp16_mode: C220Fp16Mode::NonSaturating,
                    arithmetic_saturating: false,
                    integer_saturating: true,
                    descriptor_address: 1024,
                    deq_scale: 32,
                },
                &ub,
            )
            .unwrap(),
        ),
        C220VectorInstruction::Conversion(
            plan_c220_conversion_issue(
                C220ConversionIssueInputs {
                    pc: 4,
                    word: 0x8dc0_115f,
                    control,
                    addresses: C220VectorAddresses {
                        source_0: 0,
                        source_1: 1024,
                        destination: 3584,
                    },
                    iteration_masks: &masks,
                    fp16_mode: C220Fp16Mode::NonSaturating,
                    integer_saturating: true,
                    deq_scale: 32,
                },
                &ub,
            )
            .unwrap(),
        ),
    ];
    let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
    machine.set_spr_value(12, 64).unwrap();
    let mut core = C220State::new(ScalarStepper::new(machine, 0), ub);
    let mut pipeline = C220VectorPipeline::new(C220VectorTimingRules {
        dispatch_ticks: 0,
        uop_issue_interval: NonZeroU64::new(1).unwrap(),
        ub_response_ticks: 2,
    });
    for instruction in instructions {
        pipeline
            .issue_at(
                0,
                &instruction.uops().unwrap(),
                instruction.stores(),
                instruction.read_issue(),
            )
            .unwrap();
    }
    pipeline.advance_to(256, &mut core).unwrap();
    assert_eq!(core.ub().read_known(3072, 1).unwrap(), [10]);
    assert_eq!(core.ub().read_known(3584, 1).unwrap(), [6]);
    for sample in pipeline.last_read_samples() {
        let source_count = if sample.pc == 0 { 2 } else { 1 };
        assert!(
            sample
                .accesses
                .iter()
                .all(|access| access.source_index < source_count)
        );
    }
    assert!(
        pipeline
            .last_ub_cycles()
            .iter()
            .flat_map(|cycle| &cycle.decisions)
            .all(|decision| decision.port != C220UbPort::VectorReadDestination)
    );
    let samples = pipeline.last_functional_samples();
    assert!(
        samples[0]
            .accesses
            .iter()
            .any(|access| access.source_index == 2 && access.address == 1024)
    );
    assert!(
        samples[1]
            .accesses
            .iter()
            .any(|access| access.source_index == 1 && access.address == 2048)
    );
}

#[test]
fn masked_write_reacquires_bank_and_delays_mte_until_second_grant() {
    use crate::sim::c220::memory::ub_service::{
        C220UbService, C220UbServicePort, C220UbServiceRequest,
    };
    use crate::sim::c220::vector::timing::{C220VectorUopKind, C220VectorUopStages};

    let machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
    let mut core = C220State::new(ScalarStepper::new(machine, 0), UbMemory::new(256, 256));
    let mut pipeline = C220VectorPipeline::new(C220VectorTimingRules {
        dispatch_ticks: 0,
        uop_issue_interval: NonZeroU64::new(1).unwrap(),
        ub_response_ticks: 2,
    });
    let store = C220VectorStore {
        repeat_index: 0,
        lane_index: 0,
        address: 0,
        bank: C220UbBank::from_address(0),
        width_bytes: 4,
        data: [7; 8],
    };
    pipeline
        .issue_at(
            0,
            &[C220VectorUop {
                pc: 0,
                repeat_index: 0,
                lane_group: Some(0),
                kind: C220VectorUopKind::Ordinary,
                stages: C220VectorUopStages {
                    read_ticks: 0,
                    execute_ticks: 0,
                },
                writeback_ticks: 7,
                writes_ub: true,
            }],
            &[store],
            None,
        )
        .unwrap();
    assert_eq!(pipeline.pending_visibility_tick(), Some(16));
    let full_stores = (0..8)
        .map(|lane| C220VectorStore {
            lane_index: lane,
            address: 32 + lane as u64 * 4,
            bank: C220UbBank::from_address(32 + lane as u64 * 4),
            data: [9; 8],
            ..store
        })
        .collect::<Vec<_>>();
    pipeline
        .issue_at(
            0,
            &[C220VectorUop {
                pc: 4,
                repeat_index: 0,
                lane_group: Some(0),
                kind: C220VectorUopKind::Ordinary,
                stages: C220VectorUopStages {
                    read_ticks: 0,
                    execute_ticks: 0,
                },
                writeback_ticks: 1,
                writes_ub: true,
            }],
            &full_stores,
            None,
        )
        .unwrap();
    assert_eq!(pipeline.pending_visibility_tick(), Some(17));
    assert_eq!(pipeline.pending_drain_tick(), Some(17));
    let mut memory = C220UbService::default();
    for tick in 0..=17 {
        let releases = pipeline.advance_to(tick, &mut core).unwrap();
        if tick == 7 || tick == 8 {
            assert_eq!(releases.len(), 1);
            assert_eq!(releases[0].release_tick, tick);
            assert_eq!(pipeline.pending_uops(), usize::from(tick == 7));
            assert_eq!(pipeline.pending_retirement_tick(), Some(10));
        } else {
            assert!(releases.is_empty());
        }
        if tick == 10 {
            assert_eq!(pipeline.pending_retirement_tick(), None);
            assert_eq!(pipeline.pending_ub_responses(), 2);
            assert_eq!(pipeline.pending_drain_tick(), Some(17));
        }
        let cycles = pipeline.last_ub_cycles();
        let banks = cycles.iter().fold(0, |mask, cycle| mask | cycle.bank_mask);
        if tick == 14 {
            memory
                .receive(
                    tick,
                    C220UbServicePort::MteWrite0,
                    C220UbServiceRequest {
                        id: 1,
                        address: 0,
                        bytes: 32,
                    },
                )
                .unwrap();
        }
        let mte = memory
            .arbitrate(
                tick,
                crate::sim::c220::memory::ub_service::C220UbVectorActivity {
                    bank_mask: banks,
                    triggered: !cycles.is_empty(),
                    ..Default::default()
                },
            )
            .unwrap();
        if tick == 7 || tick == 14 {
            let decision = cycles[0].decisions[0];
            assert!(decision.granted);
            assert_eq!(decision.second_grant, tick == 14);
        } else if tick < 14 {
            assert!(cycles.iter().all(|cycle| cycle.decisions.is_empty()));
        }
        if tick == 14 {
            assert!(!mte.decisions[0].granted);
            assert!(mte.completed.is_empty());
            let completion = &pipeline.last_write_completions()[0];
            assert_eq!(completion.submitted_tick, 7);
            assert_eq!(completion.completion_tick, 14);
            assert_eq!(completion.block_grants, [Some(14)]);
            assert_eq!(pipeline.pending_ub_responses(), 1);
        }
        if tick == 15 {
            assert_eq!(mte.completed.len(), 1);
            let completion = &pipeline.last_write_completions()[0];
            assert_eq!(completion.pc, 4);
            assert_eq!(completion.submitted_tick, 8);
            assert_eq!(completion.completion_tick, 15);
            assert_eq!(pipeline.pending_ub_responses(), 0);
        }
        if tick < 16 {
            assert!(core.ub().read_known(0, 4).is_err());
        }
    }
    assert_eq!(core.ub().read_known(0, 4).unwrap(), [7; 4]);
    assert!(core.ub().read_known(4, 1).is_err());
    assert_eq!(core.ub().read_known(32, 32).unwrap(), [9; 32]);
    assert_eq!(pipeline.pending_drain_tick(), None);
}

#[test]
fn conflicting_read_ports_delay_timing_but_functional_reads_observe_completed_writes() {
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
                    read_ticks: 0,
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
    assert_eq!(pipeline.pending_visibility_tick(), Some(17));
    pipeline.advance_to(8, &mut core).unwrap();
    let sample = &pipeline.last_read_samples()[0];
    assert_eq!(sample.tick, 8);
    assert_eq!(sample.read0_grants, [Some(2)]);
    assert_eq!(sample.read1_grants, [Some(1)]);
    assert_eq!(core.ub().read_known(0, 4).unwrap(), 4.0_f32.to_le_bytes());
    assert!(sample.source_0_bytes.is_empty());
    assert!(sample.lanes.is_empty());
    assert!(pipeline.last_functional_samples().is_empty());
    pipeline.advance_to(17, &mut core).unwrap();
    let sample = &pipeline.last_functional_samples()[0];
    assert_eq!(sample.tick, 17);
    assert_eq!(&sample.source_0_bytes[..4], &4.0_f32.to_le_bytes());
    assert_eq!(sample.lanes[0].bits, 6.0_f32.to_bits());
    assert_eq!(
        core.ub().read_known(0x200, 4).unwrap(),
        6.0_f32.to_le_bytes()
    );
}
