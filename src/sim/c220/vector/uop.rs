use crate::isa::c220::vector::gather::C220GatherKind;
use crate::isa::c220::vector::special::C220SpecialUnaryOperation;
use crate::sim::c220::vector::ops::select::C220SelectMode;
use crate::sim::c220::vector::timing::{
    C220VectorUop, C220VectorUopKind, C220VectorUopStages, C220VectorWritePlan,
    C220VectorWritePlanError,
};
use crate::sim::c220::vector::{C220_VECTOR_BLOCK_BYTES, C220_VECTOR_BLOCK_COUNT, C220VectorStore};

use super::C220VectorUopError;
use super::instruction::C220VectorInstruction;
use super::repeat::{AccumulatorSchedule, OrdinaryRepeatSchedule};

struct VectorUopInputs {
    pc: u64,
    repeat_count: usize,
    lane_groups: u8,
    stages: C220VectorUopStages,
    plan: C220VectorWritePlan,
}

fn synthetic_vector_uop(pc: u64) -> C220VectorUop {
    C220VectorUop {
        pc,
        repeat_index: 0,
        lane_group: None,
        kind: C220VectorUopKind::Ordinary,
        stages: C220VectorUopStages::empty(),
        writeback_ticks: 1,
        writes_ub: false,
    }
}

fn ensure_vector_uop(pc: u64, mut uops: Vec<C220VectorUop>) -> Vec<C220VectorUop> {
    if uops.is_empty() {
        uops.push(synthetic_vector_uop(pc));
    }
    uops
}

fn lane_sliced_uops(
    pc: u64,
    repeat_count: usize,
    lane_count: usize,
    lanes_per_uop: usize,
    stages: C220VectorUopStages,
    plan: &C220VectorWritePlan,
) -> Vec<C220VectorUop> {
    let mut uops = Vec::new();
    for repeat_index in 0..repeat_count {
        for first_lane in (0..lane_count).step_by(lanes_per_uop) {
            let slice_lanes = lanes_per_uop.min(lane_count - first_lane);
            let Some(writeback_ticks) =
                plan.writeback_ticks_for_lane_slice(repeat_index, first_lane, slice_lanes)
            else {
                continue;
            };
            uops.push(C220VectorUop {
                pc,
                repeat_index,
                lane_group: Some((first_lane / 64) as u8),
                kind: C220VectorUopKind::LaneSlice {
                    first_lane: first_lane as u16,
                    lane_count: slice_lanes as u16,
                },
                stages,
                writeback_ticks,
                writes_ub: true,
            });
        }
    }
    ensure_vector_uop(pc, uops)
}

fn whole_repeat_uops(
    pc: u64,
    repeat_count: usize,
    lane_count: usize,
    stages: C220VectorUopStages,
    stores: &[C220VectorStore],
) -> Result<Vec<C220VectorUop>, C220VectorUopError> {
    let mut uops = Vec::new();
    for repeat_index in 0..repeat_count {
        let repeat_stores = stores
            .iter()
            .copied()
            .filter(|store| store.repeat_index == repeat_index)
            .collect::<Vec<_>>();
        if repeat_stores.is_empty() {
            continue;
        }
        let writeback_ticks = C220VectorWritePlan::from_stores(&repeat_stores)?
            .writeback_ticks_for_repeat(repeat_index)
            .ok_or(C220VectorUopError::MissingVectorWriteback { repeat_index })?;
        uops.push(C220VectorUop {
            pc,
            repeat_index,
            lane_group: Some(0),
            kind: C220VectorUopKind::LaneSlice {
                first_lane: 0,
                lane_count: lane_count as u16,
            },
            stages,
            writeback_ticks,
            writes_ub: true,
        });
    }
    Ok(ensure_vector_uop(pc, uops))
}

fn accumulator_uops(
    pc: u64,
    schedule: AccumulatorSchedule,
    stages: C220VectorUopStages,
    plan: &C220VectorWritePlan,
) -> Vec<C220VectorUop> {
    let mut uops = Vec::new();
    for repeat in 0..schedule.repeat_count {
        if !schedule.has_traffic(repeat) {
            continue;
        }
        let lanes = schedule.lanes_per_uop(repeat);
        for first in (0..schedule.lane_count).step_by(lanes) {
            let Some(writeback_ticks) = plan.writeback_ticks_for_lane_slice(repeat, first, lanes)
            else {
                continue;
            };
            let writes_ub = schedule.writes_destination(repeat);
            uops.push(C220VectorUop {
                pc,
                repeat_index: repeat,
                lane_group: Some((first / 64) as u8),
                kind: C220VectorUopKind::LaneSlice {
                    first_lane: first as u16,
                    lane_count: lanes as u16,
                },
                stages,
                writeback_ticks: if writes_ub { writeback_ticks } else { 1 },
                writes_ub,
            });
        }
    }
    ensure_vector_uop(pc, uops)
}

impl C220VectorInstruction {
    pub fn write_plan(&self) -> Result<C220VectorWritePlan, C220VectorWritePlanError> {
        C220VectorWritePlan::from_stores(self.stores())
    }

    fn vector_uop_stages(&self) -> Option<C220VectorUopStages> {
        match self {
            Self::Move(step) => C220VectorUopStages::movev(step.instruction),
            Self::Arithmetic(step) => C220VectorUopStages::vector_arithmetic(step.hint),
            Self::Scalar(step) => Some(C220VectorUopStages::vector_scalar(step.instruction)),
            Self::Shift(_) => Some(C220VectorUopStages::shift()),
            Self::Copy(_) => Some(C220VectorUopStages::copy()),
            Self::Broadcast(_) => Some(C220VectorUopStages::broadcast()),
            Self::Transpose(_) => Some(C220VectorUopStages::transpose()),
            Self::CompareMask(_) => Some(C220VectorUopStages::packed_compare()),
            Self::MoveMask(_) => Some(C220VectorUopStages::select()),
            Self::Select(_) => Some(C220VectorUopStages::select()),
            Self::PackedCompare(_) => Some(C220VectorUopStages::packed_compare()),
            Self::Reduction(step) => Some(C220VectorUopStages::reduction(step.instruction)),
            Self::Sort(_) => Some(C220VectorUopStages::sort()),
            Self::Ternary(step) => Some(C220VectorUopStages::ternary(step.instruction)),
            Self::Axpy(_) => Some(C220VectorUopStages::axpy()),
            Self::SpecialUnary(step) => Some(C220VectorUopStages::special_unary(step.instruction)),
            Self::Conversion(step) => Some(C220VectorUopStages::conversion(step.instruction.kind)),
            Self::Fused(step) => Some(C220VectorUopStages::fused(step.instruction)),
            Self::Gather(step) => Some(C220VectorUopStages::gather(step.instruction.kind)),
            Self::Nchw(_) => Some(C220VectorUopStages::transpose()),
            _ => None,
        }
    }

    /// Describes admitted vector work in 64-lane groups or a complete tile.
    pub fn uops(&self) -> Result<Vec<C220VectorUop>, C220VectorUopError> {
        let mut uops = self.generate_uops()?;
        if let Some(schedule) = self
            .read_issue()
            .and_then(OrdinaryRepeatSchedule::from_issue)
        {
            for uop in &mut uops {
                if !schedule.writes_destination(uop.repeat_index) {
                    uop.writes_ub = false;
                    uop.writeback_ticks = 1;
                }
            }
        }
        Ok(uops)
    }

    fn generate_uops(&self) -> Result<Vec<C220VectorUop>, C220VectorUopError> {
        if let Self::Control { pc, .. } = self {
            return Ok(vec![C220VectorUop {
                pc: *pc,
                repeat_index: 0,
                lane_group: None,
                kind: C220VectorUopKind::Ordinary,
                stages: C220VectorUopStages::control(),
                writeback_ticks: 1,
                writes_ub: false,
            }]);
        }
        if let Self::NoEffect {
            pc,
            repeat_count,
            lane_groups,
            ..
        } = self
        {
            let mut uops = Vec::with_capacity(*repeat_count * usize::from(*lane_groups));
            for repeat_index in 0..*repeat_count {
                for lane_group in 0..*lane_groups {
                    uops.push(C220VectorUop {
                        pc: *pc,
                        repeat_index,
                        lane_group: Some(lane_group),
                        kind: C220VectorUopKind::Ordinary,
                        stages: C220VectorUopStages::no_effect(),
                        writeback_ticks: 1,
                        writes_ub: false,
                    });
                }
            }
            return Ok(ensure_vector_uop(*pc, uops));
        }
        if let Self::Gather(step) = self {
            if step.repeat_count() == 0 {
                return Ok(vec![synthetic_vector_uop(step.pc)]);
            }
            let mut uops = Vec::new();
            match step.instruction.kind {
                C220GatherKind::Elements(_) => {
                    for repeat_index in 0..step.repeat_count() {
                        for group in 0..step.groups_per_repeat() {
                            if group.is_multiple_of(4) {
                                uops.push(C220VectorUop {
                                    pc: step.pc,
                                    repeat_index,
                                    lane_group: None,
                                    kind: C220VectorUopKind::GatherIndex { group: group / 4 },
                                    stages: C220VectorUopStages::gather_index(),
                                    writeback_ticks: 0,
                                    writes_ub: false,
                                });
                            }
                            let stores = step.stores_for_data_uop(repeat_index, group);
                            let writes_ub = !stores.is_empty();
                            let writeback_ticks = if writes_ub {
                                C220VectorWritePlan::from_stores(&stores)?
                                    .writeback_ticks_for_repeat(repeat_index)
                                    .unwrap_or(1)
                            } else {
                                1
                            };
                            uops.push(C220VectorUop {
                                pc: step.pc,
                                repeat_index,
                                lane_group: Some(group),
                                kind: C220VectorUopKind::GatherData { group },
                                stages: C220VectorUopStages::gather(step.instruction.kind),
                                writeback_ticks,
                                writes_ub,
                            });
                        }
                    }
                }
                C220GatherKind::Blocks => {
                    uops.push(C220VectorUop {
                        pc: step.pc,
                        repeat_index: 0,
                        lane_group: None,
                        kind: C220VectorUopKind::GatherIndex { group: 0 },
                        stages: C220VectorUopStages::gather_index(),
                        writeback_ticks: 0,
                        writes_ub: false,
                    });
                    for repeat_index in 0..step.repeat_count() {
                        let stores = step.stores_for_data_uop(repeat_index, 0);
                        let writeback_ticks = C220VectorWritePlan::from_stores(&stores)?
                            .writeback_ticks_for_repeat(repeat_index)
                            .unwrap_or(1);
                        uops.push(C220VectorUop {
                            pc: step.pc,
                            repeat_index,
                            lane_group: Some(0),
                            kind: C220VectorUopKind::GatherData { group: 0 },
                            stages: C220VectorUopStages::gather(step.instruction.kind),
                            writeback_ticks,
                            writes_ub: true,
                        });
                    }
                }
            }
            return Ok(uops);
        }
        if let Self::Conversion(step) = self {
            return whole_repeat_uops(
                step.pc,
                step.iteration_masks.len(),
                step.instruction.lane_count(),
                C220VectorUopStages::conversion(step.instruction.kind),
                &step.write_targets,
            );
        }
        if let Self::Fused(step) = self {
            return whole_repeat_uops(
                step.pc,
                step.iteration_masks.len(),
                step.instruction.lane_count(),
                C220VectorUopStages::fused(step.instruction),
                &step.write_targets,
            );
        }
        if let Self::Move(step) = self {
            let element_bytes = step
                .instruction
                .supported_element_bytes()
                .expect("modeled MOVEV has an element width");
            let lane_count =
                C220_VECTOR_BLOCK_COUNT * C220_VECTOR_BLOCK_BYTES / usize::from(element_bytes);
            let plan = self.write_plan()?;
            return Ok(lane_sliced_uops(
                step.pc,
                step.iteration_masks.len(),
                lane_count,
                lane_count,
                C220VectorUopStages::movev(step.instruction).expect("modeled MOVEV has timing"),
                &plan,
            ));
        }
        if let Self::Scalar(step) = self {
            let element_bytes = step.instruction.dtype.element_bytes();
            let lane_count =
                C220_VECTOR_BLOCK_COUNT * C220_VECTOR_BLOCK_BYTES / usize::from(element_bytes);
            let lanes_per_uop = if element_bytes == 2
                && !(step.instruction.operation
                    == crate::isa::c220::vector::scalar::C220VectorScalarOperation::Multiply
                    && step.instruction.dtype
                        == crate::isa::c220::vector::scalar::C220VectorScalarType::S16)
            {
                128
            } else {
                64
            };
            let plan = self.write_plan()?;
            return Ok(lane_sliced_uops(
                step.pc,
                step.iteration_masks.len(),
                lane_count,
                lanes_per_uop,
                C220VectorUopStages::vector_scalar(step.instruction),
                &plan,
            ));
        }
        if let Self::Shift(step) = self {
            let lane_count = C220_VECTOR_BLOCK_COUNT * C220_VECTOR_BLOCK_BYTES
                / usize::from(step.instruction.element_bytes);
            let plan = self.write_plan()?;
            return Ok(lane_sliced_uops(
                step.pc,
                step.iteration_masks.len(),
                lane_count,
                lane_count.min(128),
                C220VectorUopStages::shift(),
                &plan,
            ));
        }
        if let Self::Copy(step) = self {
            let lane_count = C220_VECTOR_BLOCK_COUNT * C220_VECTOR_BLOCK_BYTES
                / usize::from(step.instruction.element_bytes);
            let plan = self.write_plan()?;
            return Ok(lane_sliced_uops(
                step.pc,
                step.iteration_masks.len(),
                lane_count,
                lane_count.min(128),
                C220VectorUopStages::copy(),
                &plan,
            ));
        }
        if let Self::Ternary(step) = self {
            let plan = self.write_plan()?;
            return Ok(accumulator_uops(
                step.pc,
                AccumulatorSchedule::new(
                    step.control,
                    step.iteration_masks.len(),
                    step.instruction.width.lane_count(),
                    true,
                ),
                C220VectorUopStages::ternary(step.instruction),
                &plan,
            ));
        }
        if let Self::Axpy(step) = self {
            let plan = self.write_plan()?;
            return Ok(accumulator_uops(
                step.pc,
                AccumulatorSchedule::new(
                    step.control,
                    step.iteration_masks.len(),
                    step.instruction.width.lane_count(),
                    false,
                ),
                C220VectorUopStages::axpy(),
                &plan,
            ));
        }
        if let Self::Arithmetic(step) = self
            && !step.modes.widens(step.hint)
        {
            let stages = C220VectorUopStages::vector_arithmetic(step.hint)
                .expect("modeled vector arithmetic has timing");
            let lane_count = C220_VECTOR_BLOCK_COUNT * C220_VECTOR_BLOCK_BYTES
                / usize::from(step.result_element_bytes);
            let lanes_per_uop = if step.hint.operation
                == crate::isa::c220::vector::C220VecArithmeticOperation::Divide
            {
                32
            } else if step.hint.has_f16_value_path()
                || step.hint.has_bitwise_b16_value_path()
                || (step.hint.has_s16_value_path()
                    && step.hint.operation
                        != crate::isa::c220::vector::C220VecArithmeticOperation::Multiply
                    && step.hint.operation
                        != crate::isa::c220::vector::C220VecArithmeticOperation::Absolute)
            {
                128
            } else {
                64
            };
            let plan = self.write_plan()?;
            let mut uops = Vec::new();
            for repeat_index in 0..step.iteration_masks.len() {
                for first_lane in (0..lane_count).step_by(lanes_per_uop) {
                    let Some(writeback_ticks) = plan.writeback_ticks_for_lane_slice(
                        repeat_index,
                        first_lane,
                        lanes_per_uop,
                    ) else {
                        continue;
                    };
                    uops.push(C220VectorUop {
                        pc: step.pc,
                        repeat_index,
                        lane_group: Some((first_lane / 64) as u8),
                        kind: C220VectorUopKind::LaneSlice {
                            first_lane: first_lane as u16,
                            lane_count: lanes_per_uop as u16,
                        },
                        stages,
                        writeback_ticks,
                        writes_ub: true,
                    });
                }
            }
            return Ok(ensure_vector_uop(step.pc, uops));
        }
        if let Self::SpecialUnary(step) = self {
            let stages = C220VectorUopStages::special_unary(step.instruction);
            let lane_count = C220_VECTOR_BLOCK_COUNT * C220_VECTOR_BLOCK_BYTES
                / usize::from(step.instruction.width.element_bytes());
            let lanes_per_uop = match step.instruction.operation {
                C220SpecialUnaryOperation::Exp
                | C220SpecialUnaryOperation::Ln
                | C220SpecialUnaryOperation::Sqrt => 32,
                C220SpecialUnaryOperation::Reciprocal
                | C220SpecialUnaryOperation::ReciprocalSqrt => lane_count,
            };
            let plan = self.write_plan()?;
            let mut uops = Vec::new();
            for repeat_index in 0..step.iteration_masks.len() {
                for first_lane in (0..lane_count).step_by(lanes_per_uop) {
                    let Some(writeback_ticks) = plan.writeback_ticks_for_lane_slice(
                        repeat_index,
                        first_lane,
                        lanes_per_uop,
                    ) else {
                        continue;
                    };
                    uops.push(C220VectorUop {
                        pc: step.pc,
                        repeat_index,
                        lane_group: Some((first_lane / 64) as u8),
                        kind: C220VectorUopKind::LaneSlice {
                            first_lane: first_lane as u16,
                            lane_count: lanes_per_uop as u16,
                        },
                        stages,
                        writeback_ticks,
                        writes_ub: true,
                    });
                }
            }
            return Ok(ensure_vector_uop(step.pc, uops));
        }
        if let Self::MoveAddress { pc, .. } = self {
            return Ok(vec![C220VectorUop {
                pc: *pc,
                repeat_index: 0,
                lane_group: None,
                kind: C220VectorUopKind::MoveVa,
                stages: C220VectorUopStages {
                    read_ticks: 6,
                    execute_ticks: 1,
                },
                writeback_ticks: 1,
                writes_ub: false,
            }]);
        }
        if let Self::LoadAddress(step) = self {
            return Ok(vec![C220VectorUop {
                pc: step.pc,
                repeat_index: 0,
                lane_group: None,
                kind: C220VectorUopKind::Ordinary,
                stages: C220VectorUopStages {
                    read_ticks: 6,
                    execute_ticks: 1,
                },
                writeback_ticks: 0,
                writes_ub: false,
            }]);
        }
        if let Self::Movemask(step) = self {
            return Ok(vec![C220VectorUop {
                pc: step.pc,
                repeat_index: 0,
                lane_group: None,
                kind: C220VectorUopKind::Ordinary,
                stages: C220VectorUopStages {
                    read_ticks: 6,
                    execute_ticks: 1,
                },
                writeback_ticks: 1,
                writes_ub: false,
            }]);
        }
        if let Self::Transpose(step) = self {
            let plan = self.write_plan()?;
            return (0..2)
                .map(|repeat_index| {
                    Ok(C220VectorUop {
                        pc: step.pc,
                        repeat_index,
                        lane_group: None,
                        kind: C220VectorUopKind::Ordinary,
                        stages: C220VectorUopStages::transpose(),
                        writeback_ticks: plan
                            .writeback_ticks_for_repeat(repeat_index)
                            .ok_or(C220VectorUopError::MissingVectorWriteback { repeat_index })?,
                        writes_ub: true,
                    })
                })
                .collect();
        }
        if let Self::Nchw(step) = self {
            let plan = self.write_plan()?;
            let uops = (0..step.rows.len() * 2)
                .map(|repeat_index| {
                    Ok(C220VectorUop {
                        pc: step.pc,
                        repeat_index,
                        lane_group: None,
                        kind: C220VectorUopKind::Ordinary,
                        stages: C220VectorUopStages::transpose(),
                        writeback_ticks: plan
                            .writeback_ticks_for_repeat(repeat_index)
                            .ok_or(C220VectorUopError::MissingVectorWriteback { repeat_index })?,
                        writes_ub: true,
                    })
                })
                .collect::<Result<Vec<_>, C220VectorUopError>>()?;
            return Ok(ensure_vector_uop(step.pc, uops));
        }
        if let Self::CompareMask(step) = self {
            let lane_count = step.instruction.width.lane_count();
            let uops = (0..step.iteration_masks.len())
                .map(|repeat_index| C220VectorUop {
                    pc: step.pc,
                    repeat_index,
                    lane_group: Some(0),
                    kind: C220VectorUopKind::LaneSlice {
                        first_lane: 0,
                        lane_count: lane_count as u16,
                    },
                    stages: C220VectorUopStages::packed_compare(),
                    writeback_ticks: 1,
                    writes_ub: false,
                })
                .collect();
            return Ok(ensure_vector_uop(step.pc, uops));
        }
        if let Self::MoveMask(step) = self {
            let writeback_ticks = if step.write_targets.is_empty() {
                1
            } else {
                self.write_plan()?
                    .writeback_ticks_for_repeat(0)
                    .ok_or(C220VectorUopError::MissingVectorWriteback { repeat_index: 0 })?
            };
            return Ok(vec![C220VectorUop {
                pc: step.pc,
                repeat_index: 0,
                lane_group: None,
                kind: C220VectorUopKind::Ordinary,
                stages: C220VectorUopStages::select(),
                writeback_ticks,
                writes_ub: !step.write_targets.is_empty(),
            }]);
        }
        if let Self::PackedCompare(step) = self {
            let plan = self.write_plan()?;
            let lane_count = step.instruction.width.lane_count();
            return Ok(lane_sliced_uops(
                step.pc,
                usize::from(step.control.encoded_repeat_count),
                lane_count,
                lane_count,
                C220VectorUopStages::packed_compare(),
                &plan,
            ));
        }
        if let Self::Reduction(step) = self {
            let stages = C220VectorUopStages::reduction(step.instruction);
            if step.iteration_masks.is_empty() {
                return Ok(if step.instruction.writes_accumulator() {
                    vec![C220VectorUop {
                        pc: step.pc,
                        repeat_index: 0,
                        lane_group: None,
                        kind: C220VectorUopKind::Ordinary,
                        stages,
                        writeback_ticks: 1,
                        writes_ub: false,
                    }]
                } else {
                    vec![synthetic_vector_uop(step.pc)]
                });
            }
            let plan = self.write_plan()?;
            return (0..step.iteration_masks.len())
                .map(|repeat_index| {
                    let writes_ub = step
                        .write_targets
                        .iter()
                        .any(|store| store.repeat_index == repeat_index);
                    Ok(C220VectorUop {
                        pc: step.pc,
                        repeat_index,
                        lane_group: None,
                        kind: C220VectorUopKind::Ordinary,
                        stages,
                        writeback_ticks: plan.writeback_ticks_for_repeat(repeat_index).unwrap_or(1),
                        writes_ub,
                    })
                })
                .collect();
        }
        if let Self::Sort(step) = self {
            if step.repeat_count == 0 {
                return Ok(vec![synthetic_vector_uop(step.pc)]);
            }
            let plan = self.write_plan()?;
            return (0..usize::from(step.repeat_count))
                .map(|repeat_index| {
                    Ok(C220VectorUop {
                        pc: step.pc,
                        repeat_index,
                        lane_group: None,
                        kind: C220VectorUopKind::Ordinary,
                        stages: C220VectorUopStages::sort(),
                        writeback_ticks: plan
                            .writeback_ticks_for_repeat(repeat_index)
                            .ok_or(C220VectorUopError::MissingVectorWriteback { repeat_index })?,
                        writes_ub: true,
                    })
                })
                .collect();
        }
        if let Self::Select(step) = self {
            let plan = self.write_plan()?;
            let groups = step.instruction.width.groups_per_repeat() as u8;
            let mut uops = Vec::new();
            for repeat_index in 0..step.iteration_masks.len() {
                if matches!(step.mode, C220SelectMode::TensorTensor)
                    && repeat_index.is_multiple_of(step.mask_repeats_per_block())
                {
                    uops.push(C220VectorUop {
                        pc: step.pc,
                        repeat_index,
                        lane_group: None,
                        kind: C220VectorUopKind::Ordinary,
                        stages: C220VectorUopStages::select(),
                        writeback_ticks: 1,
                        writes_ub: false,
                    });
                }
                for lane_group in 0..groups {
                    if let Some(writeback_ticks) =
                        plan.writeback_ticks_for_lane_group(repeat_index, lane_group)
                    {
                        uops.push(C220VectorUop {
                            pc: step.pc,
                            repeat_index,
                            lane_group: Some(lane_group),
                            kind: C220VectorUopKind::Ordinary,
                            stages: C220VectorUopStages::select(),
                            writeback_ticks,
                            writes_ub: true,
                        });
                    }
                }
            }
            if uops.is_empty() {
                uops.push(C220VectorUop {
                    pc: step.pc,
                    repeat_index: 0,
                    lane_group: Some(0),
                    kind: C220VectorUopKind::Ordinary,
                    stages: C220VectorUopStages::empty(),
                    writeback_ticks: 1,
                    writes_ub: false,
                });
            }
            return Ok(uops);
        }
        if let Self::Broadcast(step) = self {
            if step.control.repeat_count == 0 {
                return Ok(vec![synthetic_vector_uop(step.pc)]);
            }
            let plan = self.write_plan()?;
            let uops = (0..usize::from(step.control.repeat_count))
                .map(|repeat_index| {
                    Ok(C220VectorUop {
                        pc: step.pc,
                        repeat_index,
                        lane_group: None,
                        kind: C220VectorUopKind::Ordinary,
                        stages: C220VectorUopStages::broadcast(),
                        writeback_ticks: plan
                            .writeback_ticks_for_repeat(repeat_index)
                            .ok_or(C220VectorUopError::MissingVectorWriteback { repeat_index })?,
                        writes_ub: true,
                    })
                })
                .collect::<Result<Vec<_>, C220VectorUopError>>()?;
            return Ok(uops);
        }
        let Some(inputs) = self.vector_uop_inputs()? else {
            return Ok(Vec::new());
        };
        let mut uops = Vec::new();
        for repeat_index in 0..inputs.repeat_count {
            for lane_group in 0..inputs.lane_groups {
                if let Some(writeback_ticks) = inputs
                    .plan
                    .writeback_ticks_for_lane_group(repeat_index, lane_group)
                {
                    uops.push(C220VectorUop {
                        pc: inputs.pc,
                        repeat_index,
                        lane_group: Some(lane_group),
                        kind: C220VectorUopKind::Ordinary,
                        stages: inputs.stages,
                        writeback_ticks,
                        writes_ub: true,
                    });
                }
            }
        }
        if uops.is_empty() {
            uops.push(C220VectorUop {
                pc: inputs.pc,
                repeat_index: 0,
                lane_group: Some(0),
                kind: C220VectorUopKind::Ordinary,
                stages: C220VectorUopStages::empty(),
                writeback_ticks: 1,
                writes_ub: false,
            });
        }
        Ok(uops)
    }

    fn vector_uop_inputs(&self) -> Result<Option<VectorUopInputs>, C220VectorUopError> {
        let (pc, repeat_count, lane_groups) = match self {
            Self::Move(step) if step.instruction.supported_element_bytes() == Some(2) => {
                (step.pc, step.iteration_masks.len(), 2)
            }
            Self::Move(step) if step.instruction.supported_element_bytes() == Some(4) => {
                (step.pc, step.iteration_masks.len(), 1)
            }
            Self::Arithmetic(step) => (
                step.pc,
                step.iteration_masks.len(),
                4 / step.result_element_bytes,
            ),
            Self::SpecialUnary(step) => (
                step.pc,
                step.iteration_masks.len(),
                step.instruction.width.lane_groups(),
            ),
            Self::Scalar(step) => (
                step.pc,
                step.iteration_masks.len(),
                4 / step.instruction.dtype.element_bytes(),
            ),
            Self::Shift(step) => (
                step.pc,
                step.iteration_masks.len(),
                4 / step.instruction.element_bytes,
            ),
            Self::Copy(step) => (
                step.pc,
                step.iteration_masks.len(),
                4 / step.instruction.element_bytes,
            ),
            _ => return Ok(None),
        };
        let Some(stages) = self.vector_uop_stages() else {
            return Ok(None);
        };
        let plan = self.write_plan()?;
        Ok(Some(VectorUopInputs {
            pc,
            repeat_count,
            lane_groups,
            stages,
            plan,
        }))
    }
}
