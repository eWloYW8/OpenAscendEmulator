

use crate::isa::c220::control::C220VectorControlInstruction;
use crate::isa::c220::gather::C220GatherKind;
use crate::isa::c220::hflag::C220HardwareFlagStep;
use crate::isa::c220::mte1::C220Load2dTransfer;
use crate::isa::c220::no_effect::C220NoEffectVectorInstruction;
use crate::isa::c220::special::C220SpecialUnaryOperation;
use crate::isa::c220::vector::C220MoveVaInstruction;
use crate::isa::flow::FlagStep;
use crate::sim::c220::core::functional::C220OutputStep;
use crate::sim::c220::cube::C220CubeIssue;
use crate::sim::c220::mte::load2d::C220Load2dTransferResult;
use crate::sim::c220::mte::uop::C220DmaUopRequest;
use crate::sim::c220::scalar::C220MovemaskStep;
use crate::sim::c220::timing::mte1::C220Mte1Ticket;
use crate::sim::c220::timing::mte2::C220Mte2Step;
use crate::sim::c220::timing::mte3::C220Mte3Ticket;
use crate::sim::c220::timing::scalar::C220ScalarTimingTicket;
use crate::sim::c220::vector::axpy::C220AxpyIssue;
use crate::sim::c220::vector::broadcast::C220BroadcastIssue;
use crate::sim::c220::vector::compare::{
    C220CompareMaskIssue, C220MoveMaskIssue, C220PackedCompareIssue,
};
use crate::sim::c220::vector::conversion::C220ConversionIssue;
use crate::sim::c220::vector::copy::C220CopyIssue;
use crate::sim::c220::vector::fused::C220FusedIssue;
use crate::sim::c220::vector::gather::C220GatherIssue;
use crate::sim::c220::vector::load_va::C220LoadVaIssue;
use crate::sim::c220::vector::merge::C220MergeIssue;
use crate::sim::c220::vector::nchw::C220NchwIssue;
use crate::sim::c220::vector::pipeline::C220VectorQueueClass;
use crate::sim::c220::vector::reduce::C220ReductionIssue;
use crate::sim::c220::vector::scalar::C220VectorScalarIssue;
use crate::sim::c220::vector::select::{C220SelectIssue, C220SelectMode};
use crate::sim::c220::vector::shift::C220ShiftIssue;
use crate::sim::c220::vector::sort::C220SortIssue;
use crate::sim::c220::vector::special::C220SpecialUnaryIssue;
use crate::sim::c220::vector::ternary::C220TernaryIssue;
use crate::sim::c220::vector::timing::{
    C220VectorUop, C220VectorUopKind, C220VectorUopStages,
    C220VectorWritePlan, C220VectorWritePlanError,
};
use crate::sim::c220::vector::transpose::C220TransposeIssue;
use crate::sim::c220::vector::{
    C220_VECTOR_BLOCK_BYTES, C220_VECTOR_BLOCK_COUNT, C220MovevStep, C220VectorArithmeticIssue, C220VectorStore,
};
use crate::sim::common::scalar::stepper::ScalarProgramStep;

use super::engine::C220CoreError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum C220CoreInstruction {
    Scalar {
        step: ScalarProgramStep,
        timing: Option<C220ScalarTimingTicket>,
    },
    Barrier(ScalarProgramStep),
    Mte1Load2d {
        transfer: C220Load2dTransfer,
        result: C220Load2dTransferResult,
        ticket: Box<C220Mte1Ticket>,
    },
    Mte1Flag(FlagStep),
    HardwareFlag {
        step: C220HardwareFlagStep,
        token_ready_tick: Option<u64>,
    },
    Mte2(C220Mte2Step),
    Cube(C220CubeIssue),
    VectorMoveAddress {
        pc: u64,
        word: u32,
        instruction: C220MoveVaInstruction,
    },
    VectorControl {
        pc: u64,
        word: u32,
        instruction: C220VectorControlInstruction,
    },
    VectorNoEffect {
        pc: u64,
        word: u32,
        instruction: C220NoEffectVectorInstruction,
        repeat_count: usize,
        lane_groups: u8,
    },
    VectorLoadAddress(C220LoadVaIssue),
    VectorMovemask(C220MovemaskStep),
    VectorMove(C220MovevStep),
    VectorArithmetic(C220VectorArithmeticIssue),
    VectorScalar(C220VectorScalarIssue),
    VectorShift(C220ShiftIssue),
    VectorCopy(C220CopyIssue),
    VectorBroadcast(C220BroadcastIssue),
    VectorTranspose(C220TransposeIssue),
    VectorCompareMask(C220CompareMaskIssue),
    VectorMoveMask(C220MoveMaskIssue),
    VectorSelect(C220SelectIssue),
    VectorPackedCompare(C220PackedCompareIssue),
    VectorReduction(C220ReductionIssue),
    VectorSort(C220SortIssue),
    VectorTernary(C220TernaryIssue),
    VectorAxpy(C220AxpyIssue),
    VectorSpecialUnary(C220SpecialUnaryIssue),
    VectorConversion(C220ConversionIssue),
    VectorFused(C220FusedIssue),
    VectorGather(C220GatherIssue),
    VectorNchw(C220NchwIssue),
    VectorMerge(C220MergeIssue),
    VectorToScalarFlag(FlagStep),
    Mte3 {
        step: C220OutputStep,
        requests: Vec<C220DmaUopRequest>,
        ticket: Option<C220Mte3Ticket>,
    },
}

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

fn lane_sliced_vector_uops(
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

fn whole_repeat_vector_uops(
    pc: u64,
    repeat_count: usize,
    lane_count: usize,
    stages: C220VectorUopStages,
    stores: &[C220VectorStore],
) -> Result<Vec<C220VectorUop>, C220CoreError> {
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
            .ok_or(C220CoreError::MissingVectorWriteback { repeat_index })?;
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

impl C220CoreInstruction {
    pub(super) fn vector_queue_class(&self) -> C220VectorQueueClass {
        match self {
            Self::VectorNoEffect { instruction, .. } => match instruction.operation {
                crate::isa::c220::no_effect::C220NoEffectVectorOperation::Vrpac => {
                    C220VectorQueueClass::Vrpac
                }
                crate::isa::c220::no_effect::C220NoEffectVectorOperation::Vms4 => {
                    C220VectorQueueClass::Vms4
                }
                _ => C220VectorQueueClass::Ordinary,
            },
            Self::VectorReduction(issue) => match issue.instruction.kind {
                crate::isa::c220::reduce::C220ReductionKind::WholeAdd { .. } => {
                    C220VectorQueueClass::WholeAdd
                }
                crate::isa::c220::reduce::C220ReductionKind::GroupAdd => {
                    C220VectorQueueClass::GroupAdd
                }
                crate::isa::c220::reduce::C220ReductionKind::WholeExtremum { .. } => {
                    C220VectorQueueClass::WholeExtremum
                }
                crate::isa::c220::reduce::C220ReductionKind::GroupExtremum { .. } => {
                    C220VectorQueueClass::GroupExtremum
                }
                crate::isa::c220::reduce::C220ReductionKind::PairAdd => {
                    C220VectorQueueClass::Ordinary
                }
            },
            Self::VectorTernary(issue)
                if issue.instruction.operation
                    == crate::isa::c220::ternary::C220TernaryOperation::MultiplyAccumulate =>
            {
                C220VectorQueueClass::MultiplyAccumulate
            }
            Self::VectorAxpy(_) => C220VectorQueueClass::Axpy,
            _ => C220VectorQueueClass::Ordinary,
        }
    }

    pub(super) fn vector_stores(&self) -> Option<&[C220VectorStore]> {
        match self {
            Self::VectorMoveAddress { .. }
            | Self::VectorControl { .. }
            | Self::VectorNoEffect { .. }
            | Self::VectorLoadAddress(_)
            | Self::VectorMovemask(_) => Some(&[]),
            Self::VectorMove(step) => Some(&step.stores),
            Self::VectorArithmetic(step) => Some(&step.write_targets),
            Self::VectorScalar(step) => Some(&step.write_targets),
            Self::VectorShift(step) => Some(&step.write_targets),
            Self::VectorCopy(step) => Some(&step.write_targets),
            Self::VectorBroadcast(step) => Some(&step.write_targets),
            Self::VectorTranspose(step) => Some(&step.write_targets),
            Self::VectorCompareMask(_) => Some(&[]),
            Self::VectorMoveMask(step) => Some(&step.write_targets),
            Self::VectorSelect(step) => Some(&step.write_targets),
            Self::VectorPackedCompare(step) => Some(&step.write_targets),
            Self::VectorReduction(step) => Some(&step.write_targets),
            Self::VectorSort(step) => Some(&step.write_targets),
            Self::VectorTernary(step) => Some(&step.write_targets),
            Self::VectorAxpy(step) => Some(&step.write_targets),
            Self::VectorSpecialUnary(step) => Some(&step.write_targets),
            Self::VectorConversion(step) => Some(&step.write_targets),
            Self::VectorFused(step) => Some(&step.write_targets),
            Self::VectorGather(step) => Some(&step.write_targets),
            Self::VectorNchw(step) => Some(&step.write_targets),
            _ => None,
        }
    }

    pub fn vector_write_plan(
        &self,
    ) -> Result<Option<C220VectorWritePlan>, C220VectorWritePlanError> {
        match self {
            Self::VectorMove(step) => C220VectorWritePlan::from_stores(&step.stores).map(Some),
            Self::VectorArithmetic(step) => {
                C220VectorWritePlan::from_stores(&step.write_targets).map(Some)
            }
            Self::VectorScalar(step) => {
                C220VectorWritePlan::from_stores(&step.write_targets).map(Some)
            }
            Self::VectorShift(step) => {
                C220VectorWritePlan::from_stores(&step.write_targets).map(Some)
            }
            Self::VectorCopy(step) => {
                C220VectorWritePlan::from_stores(&step.write_targets).map(Some)
            }
            Self::VectorBroadcast(step) => {
                C220VectorWritePlan::from_stores(&step.write_targets).map(Some)
            }
            Self::VectorTranspose(step) => {
                C220VectorWritePlan::from_stores(&step.write_targets).map(Some)
            }
            Self::VectorSelect(step) => {
                C220VectorWritePlan::from_stores(&step.write_targets).map(Some)
            }
            Self::VectorMoveMask(step) => {
                C220VectorWritePlan::from_stores(&step.write_targets).map(Some)
            }
            Self::VectorPackedCompare(step) => {
                C220VectorWritePlan::from_stores(&step.write_targets).map(Some)
            }
            Self::VectorReduction(step) => {
                C220VectorWritePlan::from_stores(&step.write_targets).map(Some)
            }
            Self::VectorSort(step) => {
                C220VectorWritePlan::from_stores(&step.write_targets).map(Some)
            }
            Self::VectorTernary(step) => {
                C220VectorWritePlan::from_stores(&step.write_targets).map(Some)
            }
            Self::VectorAxpy(step) => {
                C220VectorWritePlan::from_stores(&step.write_targets).map(Some)
            }
            Self::VectorSpecialUnary(step) => {
                C220VectorWritePlan::from_stores(&step.write_targets).map(Some)
            }
            Self::VectorConversion(step) => {
                C220VectorWritePlan::from_stores(&step.write_targets).map(Some)
            }
            Self::VectorFused(step) => {
                C220VectorWritePlan::from_stores(&step.write_targets).map(Some)
            }
            Self::VectorGather(step) => {
                C220VectorWritePlan::from_stores(&step.write_targets).map(Some)
            }
            Self::VectorNchw(step) => {
                C220VectorWritePlan::from_stores(&step.write_targets).map(Some)
            }
            _ => Ok(None),
        }
    }

    fn vector_uop_stages(&self) -> Option<C220VectorUopStages> {
        match self {
            Self::VectorMove(step) => C220VectorUopStages::movev(step.instruction),
            Self::VectorArithmetic(step) => C220VectorUopStages::vector_arithmetic(step.hint),
            Self::VectorScalar(step) => Some(C220VectorUopStages::vector_scalar(step.instruction)),
            Self::VectorShift(_) => Some(C220VectorUopStages::shift()),
            Self::VectorCopy(_) => Some(C220VectorUopStages::copy()),
            Self::VectorBroadcast(_) => Some(C220VectorUopStages::broadcast()),
            Self::VectorTranspose(_) => Some(C220VectorUopStages::transpose()),
            Self::VectorCompareMask(_) => Some(C220VectorUopStages::packed_compare()),
            Self::VectorMoveMask(_) => Some(C220VectorUopStages::select()),
            Self::VectorSelect(_) => Some(C220VectorUopStages::select()),
            Self::VectorPackedCompare(_) => Some(C220VectorUopStages::packed_compare()),
            Self::VectorReduction(step) => Some(C220VectorUopStages::reduction(step.instruction)),
            Self::VectorSort(_) => Some(C220VectorUopStages::sort()),
            Self::VectorTernary(step) => Some(C220VectorUopStages::ternary(step.instruction)),
            Self::VectorAxpy(_) => Some(C220VectorUopStages::axpy()),
            Self::VectorSpecialUnary(step) => {
                Some(C220VectorUopStages::special_unary(step.instruction))
            }
            Self::VectorConversion(step) => {
                Some(C220VectorUopStages::conversion(step.instruction.kind))
            }
            Self::VectorFused(step) => Some(C220VectorUopStages::fused(step.instruction)),
            Self::VectorGather(step) => Some(C220VectorUopStages::gather(step.instruction.kind)),
            Self::VectorNchw(_) => Some(C220VectorUopStages::transpose()),
            _ => None,
        }
    }

    /// Describes admitted vector work in 64-lane groups or a complete tile.
    pub fn vector_uops(&self) -> Result<Vec<C220VectorUop>, C220CoreError> {
        if let Self::VectorControl { pc, .. } = self {
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
        if let Self::VectorNoEffect {
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
        if let Self::VectorGather(step) = self {
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
        if let Self::VectorConversion(step) = self {
            return whole_repeat_vector_uops(
                step.pc,
                step.iteration_masks.len(),
                step.instruction.lane_count(),
                C220VectorUopStages::conversion(step.instruction.kind),
                &step.write_targets,
            );
        }
        if let Self::VectorFused(step) = self {
            return whole_repeat_vector_uops(
                step.pc,
                step.iteration_masks.len(),
                step.instruction.lane_count(),
                C220VectorUopStages::fused(step.instruction),
                &step.write_targets,
            );
        }
        if let Self::VectorMove(step) = self {
            let element_bytes = step
                .instruction
                .supported_element_bytes()
                .expect("modeled MOVEV has an element width");
            let lane_count =
                C220_VECTOR_BLOCK_COUNT * C220_VECTOR_BLOCK_BYTES / usize::from(element_bytes);
            let plan = self.vector_write_plan()?.expect("MOVEV has write plan");
            return Ok(lane_sliced_vector_uops(
                step.pc,
                step.iteration_masks.len(),
                lane_count,
                lane_count,
                C220VectorUopStages::movev(step.instruction).expect("modeled MOVEV has timing"),
                &plan,
            ));
        }
        if let Self::VectorScalar(step) = self {
            let element_bytes = step.instruction.dtype.element_bytes();
            let lane_count =
                C220_VECTOR_BLOCK_COUNT * C220_VECTOR_BLOCK_BYTES / usize::from(element_bytes);
            let lanes_per_uop = if element_bytes == 2
                && !(step.instruction.operation
                    == crate::isa::c220::vector_scalar::C220VectorScalarOperation::Multiply
                    && step.instruction.dtype
                        == crate::isa::c220::vector_scalar::C220VectorScalarType::S16)
            {
                128
            } else {
                64
            };
            let plan = self
                .vector_write_plan()?
                .expect("vector-scalar instruction has write plan");
            return Ok(lane_sliced_vector_uops(
                step.pc,
                step.iteration_masks.len(),
                lane_count,
                lanes_per_uop,
                C220VectorUopStages::vector_scalar(step.instruction),
                &plan,
            ));
        }
        if let Self::VectorShift(step) = self {
            let lane_count = C220_VECTOR_BLOCK_COUNT * C220_VECTOR_BLOCK_BYTES
                / usize::from(step.instruction.element_bytes);
            let plan = self
                .vector_write_plan()?
                .expect("vector shift has write plan");
            return Ok(lane_sliced_vector_uops(
                step.pc,
                step.iteration_masks.len(),
                lane_count,
                lane_count.min(128),
                C220VectorUopStages::shift(),
                &plan,
            ));
        }
        if let Self::VectorCopy(step) = self {
            let lane_count = C220_VECTOR_BLOCK_COUNT * C220_VECTOR_BLOCK_BYTES
                / usize::from(step.instruction.element_bytes);
            let plan = self
                .vector_write_plan()?
                .expect("vector copy has write plan");
            return Ok(lane_sliced_vector_uops(
                step.pc,
                step.iteration_masks.len(),
                lane_count,
                lane_count.min(128),
                C220VectorUopStages::copy(),
                &plan,
            ));
        }
        if let Self::VectorTernary(step) = self {
            let lane_count = step.instruction.width.lane_count();
            let lanes_per_uop = match step.instruction.width {
                crate::isa::c220::ternary::C220TernaryWidth::F16 => 128,
                crate::isa::c220::ternary::C220TernaryWidth::F16ToF32
                | crate::isa::c220::ternary::C220TernaryWidth::F32 => 64,
            };
            let plan = self
                .vector_write_plan()?
                .expect("ternary instruction has write plan");
            return Ok(lane_sliced_vector_uops(
                step.pc,
                step.iteration_masks.len(),
                lane_count,
                lanes_per_uop,
                C220VectorUopStages::ternary(step.instruction),
                &plan,
            ));
        }
        if let Self::VectorAxpy(step) = self {
            let lane_count = step.instruction.width.lane_count();
            let lanes_per_uop = match step.instruction.width {
                crate::isa::c220::axpy::C220AxpyWidth::F16 => 128,
                crate::isa::c220::axpy::C220AxpyWidth::F16ToF32
                | crate::isa::c220::axpy::C220AxpyWidth::F32 => 64,
            };
            let plan = self
                .vector_write_plan()?
                .expect("AXPY instruction has write plan");
            return Ok(lane_sliced_vector_uops(
                step.pc,
                step.iteration_masks.len(),
                lane_count,
                lanes_per_uop,
                C220VectorUopStages::axpy(),
                &plan,
            ));
        }
        if let Self::VectorArithmetic(step) = self
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
            let plan = self
                .vector_write_plan()?
                .expect("vector arithmetic has write plan");
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
        if let Self::VectorSpecialUnary(step) = self {
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
            let plan = self
                .vector_write_plan()?
                .expect("special unary has write plan");
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
        if let Self::VectorMoveAddress { pc, .. } = self {
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
        if let Self::VectorLoadAddress(step) = self {
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
        if let Self::VectorMovemask(step) = self {
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
        if let Self::VectorTranspose(step) = self {
            let plan = self.vector_write_plan()?.expect("transpose has write plan");
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
                            .ok_or(C220CoreError::MissingVectorWriteback { repeat_index })?,
                        writes_ub: true,
                    })
                })
                .collect();
        }
        if let Self::VectorNchw(step) = self {
            let plan = self.vector_write_plan()?.expect("NCHW has write plan");
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
                            .ok_or(C220CoreError::MissingVectorWriteback { repeat_index })?,
                        writes_ub: true,
                    })
                })
                .collect::<Result<Vec<_>, C220CoreError>>()?;
            return Ok(ensure_vector_uop(step.pc, uops));
        }
        if let Self::VectorCompareMask(step) = self {
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
        if let Self::VectorMoveMask(step) = self {
            let writeback_ticks = if step.write_targets.is_empty() {
                1
            } else {
                self.vector_write_plan()?
                    .expect("mask store has write plan")
                    .writeback_ticks_for_repeat(0)
                    .ok_or(C220CoreError::MissingVectorWriteback { repeat_index: 0 })?
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
        if let Self::VectorPackedCompare(step) = self {
            let plan = self
                .vector_write_plan()?
                .expect("packed compare has write plan");
            let lane_count = step.instruction.width.lane_count();
            return Ok(lane_sliced_vector_uops(
                step.pc,
                usize::from(step.control.encoded_repeat_count),
                lane_count,
                lane_count,
                C220VectorUopStages::packed_compare(),
                &plan,
            ));
        }
        if let Self::VectorReduction(step) = self {
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
            let plan = self.vector_write_plan()?.expect("reduction has write plan");
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
        if let Self::VectorSort(step) = self {
            if step.repeat_count == 0 {
                return Ok(vec![synthetic_vector_uop(step.pc)]);
            }
            let plan = self.vector_write_plan()?.expect("sort has write plan");
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
                            .ok_or(C220CoreError::MissingVectorWriteback { repeat_index })?,
                        writes_ub: true,
                    })
                })
                .collect();
        }
        if let Self::VectorSelect(step) = self {
            let plan = self.vector_write_plan()?.expect("select has write plan");
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
        if let Self::VectorBroadcast(step) = self {
            if step.control.repeat_count == 0 {
                return Ok(vec![synthetic_vector_uop(step.pc)]);
            }
            let plan = self.vector_write_plan()?.expect("broadcast has write plan");
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
                            .ok_or(C220CoreError::MissingVectorWriteback { repeat_index })?,
                        writes_ub: true,
                    })
                })
                .collect::<Result<Vec<_>, C220CoreError>>()?;
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

    fn vector_uop_inputs(&self) -> Result<Option<VectorUopInputs>, C220CoreError> {
        let (pc, repeat_count, lane_groups) = match self {
            Self::VectorMove(step) if step.instruction.supported_element_bytes() == Some(2) => {
                (step.pc, step.iteration_masks.len(), 2)
            }
            Self::VectorMove(step) if step.instruction.supported_element_bytes() == Some(4) => {
                (step.pc, step.iteration_masks.len(), 1)
            }
            Self::VectorArithmetic(step) => (
                step.pc,
                step.iteration_masks.len(),
                4 / step.result_element_bytes,
            ),
            Self::VectorSpecialUnary(step) => (
                step.pc,
                step.iteration_masks.len(),
                step.instruction.width.lane_groups(),
            ),
            Self::VectorScalar(step) => (
                step.pc,
                step.iteration_masks.len(),
                4 / step.instruction.dtype.element_bytes(),
            ),
            Self::VectorShift(step) => (
                step.pc,
                step.iteration_masks.len(),
                4 / step.instruction.element_bytes,
            ),
            Self::VectorCopy(step) => (
                step.pc,
                step.iteration_masks.len(),
                4 / step.instruction.element_bytes,
            ),
            _ => return Ok(None),
        };
        let Some(stages) = self.vector_uop_stages() else {
            return Ok(None);
        };
        let Some(plan) = self.vector_write_plan()? else {
            return Ok(None);
        };
        Ok(Some(VectorUopInputs {
            pc,
            repeat_count,
            lane_groups,
            stages,
            plan,
        }))
    }
}
