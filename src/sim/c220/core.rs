use std::collections::BTreeMap;

use thiserror::Error;

use crate::architecture::Architecture;
use crate::image::loader::{DeviceKernelFetchError, LoadedDeviceKernel};
use crate::isa::c220::compare::{
    C220CompareMaskInstruction, C220MoveMaskDirection, C220MoveMaskInstruction,
    C220PackedCompareInstruction,
};
use crate::isa::c220::gather::{C220GatherInstruction, C220GatherKind};
use crate::isa::c220::mte::C220DmaMovDescriptor;
use crate::isa::c220::reduce::C220ReductionInstruction;
use crate::isa::c220::scalar::C220ScalarConversionHint;
use crate::isa::c220::select::C220SelectInstruction;
use crate::isa::c220::ternary::C220TernaryInstruction;
use crate::isa::c220::vector::{
    C220BroadcastInstruction, C220CopyInstruction, C220MoveVaInstruction, C220MovevInstruction,
    C220NchwInstruction, C220ShiftInstruction, C220TransposeInstruction, C220VecArithmeticHint,
};
use crate::isa::c220::vector_scalar::C220VectorScalarInstruction;
use crate::isa::flow::{
    FlagInstruction, FlagOperation, FlagStep, PipelineBarrierScope, PipelineBarrierStep,
};
use crate::memory::hbm_pv_memory::HbmPvMemory;
use crate::memory::mapped::{MappedMemory, MappedMemoryError};
use crate::sim::c220::mte::output::{C220OutputAction, C220OutputStep};
use crate::sim::c220::mte::transfer::C220PreparedOutput;
use crate::sim::c220::mte::uop::C220DmaUopRequest;
use crate::sim::c220::timing::mte2::{
    C220Mte2TimingRules, C220Stall, C220StallCause, C220TimedMte2Core, C220TimedMte2Step,
    C220TimingError, is_mte2_transfer,
};
use crate::sim::c220::timing::mte3::{
    C220Mte3Ticket, C220Mte3TimingError, C220Mte3TimingRules, C220TimedMte3Lane,
};
use crate::sim::c220::timing::scalar::{C220ScalarTimingLane, C220ScalarTimingTicket};
use crate::sim::c220::va::C220VaRegisters;
use crate::sim::c220::vector::broadcast::C220BroadcastIssue;
use crate::sim::c220::vector::compare::{
    C220CompareMask, C220CompareMaskIssue, C220MoveMaskIssue, C220PackedCompareIssue,
    plan_c220_compare_mask_issue, plan_c220_move_mask_issue, plan_c220_packed_compare_issue,
};
use crate::sim::c220::vector::copy::C220CopyIssue;
use crate::sim::c220::vector::gather::C220GatherIssue;
use crate::sim::c220::vector::nchw::{C220NchwIssue, plan_c220_nchw_issue};
use crate::sim::c220::vector::pipeline::{
    C220VectorAdvanceError, C220VectorPipeline, C220VectorPipelineError, C220VectorTimingRules,
};
use crate::sim::c220::vector::read::C220VectorReadIssue;
use crate::sim::c220::vector::reduce::C220ReductionIssue;
use crate::sim::c220::vector::scalar::C220VectorScalarIssue;
use crate::sim::c220::vector::select::{C220SelectIssue, C220SelectMode, plan_c220_select_issue};
use crate::sim::c220::vector::shift::C220ShiftIssue;
use crate::sim::c220::vector::ternary::C220TernaryIssue;
use crate::sim::c220::vector::timing::{
    C220VectorUop, C220VectorUopKind, C220VectorUopRelease, C220VectorUopStages,
    C220VectorWritePlan, C220VectorWritePlanError,
};
use crate::sim::c220::vector::transpose::C220TransposeIssue;
use crate::sim::c220::vector::{
    C220MovevStep, C220VectorArithmeticIssue, C220VectorError, C220VectorMaskState, C220VectorStore,
};
use crate::sim::machine::ScalarInstructionError;
use crate::sim::mte_stepper::{MteCoreStepper, MteStepperError};
use crate::sim::scalar_bus::UbScalarBusError;
use crate::sim::stepper::ScalarProgramStep;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum C220CoreInstruction {
    Scalar {
        step: ScalarProgramStep,
        timing: Option<C220ScalarTimingTicket>,
    },
    Barrier(ScalarProgramStep),
    Mte2(C220TimedMte2Step),
    VectorMoveAddress {
        pc: u64,
        word: u32,
        instruction: C220MoveVaInstruction,
    },
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
    VectorTernary(C220TernaryIssue),
    VectorGather(C220GatherIssue),
    VectorNchw(C220NchwIssue),
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

impl C220CoreInstruction {
    fn vector_stores(&self) -> Option<&[C220VectorStore]> {
        match self {
            Self::VectorMoveAddress { .. } => Some(&[]),
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
            Self::VectorTernary(step) => Some(&step.write_targets),
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
            Self::VectorTernary(step) => {
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
            Self::VectorTernary(step) => Some(C220VectorUopStages::ternary(step.instruction)),
            Self::VectorGather(step) => Some(C220VectorUopStages::gather(step.instruction.kind)),
            Self::VectorNchw(_) => Some(C220VectorUopStages::transpose()),
            _ => None,
        }
    }

    /// Describes admitted vector work in 64-lane groups or a complete tile.
    pub fn vector_uops(&self) -> Result<Vec<C220VectorUop>, C220CoreError> {
        if let Self::VectorGather(step) = self {
            if step.repeat_count() == 0 {
                return Ok(Vec::new());
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
        if let Self::VectorMoveAddress { pc, .. } = self {
            return Ok(vec![C220VectorUop {
                pc: *pc,
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
            return (0..step.rows.len() * 2)
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
        if let Self::VectorCompareMask(step) = self {
            return (0..step.uop_count())
                .map(|uop_index| {
                    let (_, lane_group) = step.split_uop(uop_index)?;
                    Ok(C220VectorUop {
                        pc: step.pc,
                        repeat_index: uop_index,
                        lane_group: Some(lane_group),
                        kind: C220VectorUopKind::Ordinary,
                        stages: C220VectorUopStages::packed_compare(),
                        writeback_ticks: 1,
                        writes_ub: false,
                    })
                })
                .collect();
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
            return (0..step.uop_count())
                .map(|repeat_index| {
                    step.split_uop(repeat_index)?;
                    Ok(C220VectorUop {
                        pc: step.pc,
                        repeat_index,
                        lane_group: None,
                        kind: C220VectorUopKind::Ordinary,
                        stages: C220VectorUopStages::packed_compare(),
                        writeback_ticks: plan
                            .writeback_ticks_for_repeat(repeat_index)
                            .ok_or(C220CoreError::MissingVectorWriteback { repeat_index })?,
                        writes_ub: true,
                    })
                })
                .collect();
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
                    Vec::new()
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
                return Ok(Vec::new());
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
            Self::VectorTernary(step) => (
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220CoreTimingRules {
    pub mte2: C220Mte2TimingRules,
    pub mte3: C220Mte3TimingRules,
    pub vector: C220VectorTimingRules,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum C220CoreStep {
    Executed {
        tick: u64,
        instruction: C220CoreInstruction,
    },
    Stalled(C220Stall),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220RunStop {
    Halted,
    TickBudget,
    EventBudget,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220CoreRun {
    pub start_tick: u64,
    pub next_tick: u64,
    pub next_pc: u64,
    pub events: Vec<C220CoreStep>,
    pub stop: C220RunStop,
}

#[derive(Debug, Error)]
pub enum C220CoreError {
    #[error("loaded kernel is not for dav_2201")]
    ArchitectureMismatch,
    #[error(transparent)]
    Fetch(#[from] DeviceKernelFetchError),
    #[error(transparent)]
    Timing(#[from] C220TimingError),
    #[error(transparent)]
    Mte(#[from] MteStepperError),
    #[error(transparent)]
    Mte3Timing(#[from] C220Mte3TimingError),
    #[error(transparent)]
    VectorPlan(#[from] C220VectorWritePlanError),
    #[error(transparent)]
    Vector(#[from] C220VectorError),
    #[error(transparent)]
    VectorPipeline(#[from] C220VectorPipelineError),
    #[error(transparent)]
    VectorAdvance(#[from] C220VectorAdvanceError),
    #[error(transparent)]
    OutputMemory(#[from] MappedMemoryError),
    #[error(transparent)]
    Scalar(#[from] ScalarInstructionError<UbScalarBusError<MappedMemoryError>>),
    #[error("timed core stalled without a future resume tick")]
    NonprogressingStall,
    #[error("tick counter overflowed")]
    TimeOverflow,
    #[error("vector repeat {repeat_index} has no writeback schedule")]
    MissingVectorWriteback { repeat_index: usize },
    #[error("C220 vector-to-scalar flag {flag_id} is already pending")]
    VectorScalarFlagAlreadySet { flag_id: u32 },
    #[error("C220 vector-to-scalar flag {flag_id} was not set before wait")]
    VectorScalarWaitWithoutFlag { flag_id: u32 },
}

pub struct C220Core {
    execution: C220TimedMte2Core,
    scalar_timing: C220ScalarTimingLane,
    mte3: C220TimedMte3Lane,
    vector: C220VectorPipeline,
    va: C220VaRegisters,
    last_vector_releases: Vec<C220VectorUopRelease>,
    pending_output: Option<C220PendingOutput>,
    vector_to_scalar_flags: BTreeMap<u32, u64>,
    memory: MappedMemory,
}

struct C220PendingOutput {
    data_ready_tick: u64,
    prepared: C220PreparedOutput,
}

impl C220Core {
    pub fn new(
        execution: MteCoreStepper,
        memory: MappedMemory,
        timing: C220CoreTimingRules,
    ) -> Result<Self, C220CoreError> {
        Ok(Self {
            execution: C220TimedMte2Core::new(execution, timing.mte2)?,
            scalar_timing: C220ScalarTimingLane::default(),
            mte3: C220TimedMte3Lane::new(timing.mte3),
            vector: C220VectorPipeline::new(timing.vector),
            va: C220VaRegisters::default(),
            last_vector_releases: Vec::new(),
            pending_output: None,
            vector_to_scalar_flags: BTreeMap::new(),
            memory,
        })
    }

    pub const fn execution(&self) -> &C220TimedMte2Core {
        &self.execution
    }

    pub const fn scalar_timing(&self) -> &C220ScalarTimingLane {
        &self.scalar_timing
    }

    pub const fn memory(&self) -> &MappedMemory {
        &self.memory
    }

    pub const fn vector_pipeline(&self) -> &C220VectorPipeline {
        &self.vector
    }

    pub const fn compare_mask(&self) -> C220CompareMask {
        self.vector.compare_mask()
    }

    pub const fn va_registers(&self) -> &C220VaRegisters {
        &self.va
    }

    pub fn last_vector_releases(&self) -> &[C220VectorUopRelease] {
        &self.last_vector_releases
    }

    pub fn memory_mut(&mut self) -> &mut MappedMemory {
        &mut self.memory
    }

    pub fn pending_output_ready_tick(&self) -> Option<u64> {
        self.pending_output
            .as_ref()
            .map(|pending| pending.data_ready_tick)
    }

    pub fn advance_to(&mut self, tick: u64) -> Result<Option<C220Stall>, C220CoreError> {
        let gate = self.execution.gate_other_at(tick)?;
        self.scalar_timing.advance_to(tick);
        self.last_vector_releases = self.vector.advance_to(tick, self.execution.core_mut())?;
        self.commit_ready_output_at(tick)?;
        Ok(gate)
    }

    pub fn step_loaded_at(
        &mut self,
        tick: u64,
        kernel: &LoadedDeviceKernel,
        code_memory: &mut HbmPvMemory,
    ) -> Result<C220CoreStep, C220CoreError> {
        if kernel.placement().architecture != Architecture::Dav2201 {
            return Err(C220CoreError::ArchitectureMismatch);
        }
        let pc = self.execution.core().scalar().pc();
        let word = kernel.fetch_executable_word(code_memory, pc)?;
        self.step_word_at(tick, word)
    }

    pub fn run_loaded_until(
        &mut self,
        start_tick: u64,
        tick_limit: u64,
        max_events: usize,
        kernel: &LoadedDeviceKernel,
        code_memory: &mut HbmPvMemory,
    ) -> Result<C220CoreRun, C220CoreError> {
        let mut tick = start_tick;
        let mut events = Vec::new();
        loop {
            let stop = if self.execution.core().scalar().is_halted() {
                Some(C220RunStop::Halted)
            } else if tick >= tick_limit {
                Some(C220RunStop::TickBudget)
            } else if events.len() >= max_events {
                Some(C220RunStop::EventBudget)
            } else {
                None
            };
            if let Some(stop) = stop {
                self.advance_to(tick)?;
                return Ok(C220CoreRun {
                    start_tick,
                    next_tick: tick,
                    next_pc: self.execution.core().scalar().pc(),
                    events,
                    stop,
                });
            }
            let event = self.step_loaded_at(tick, kernel, code_memory)?;
            tick = match &event {
                C220CoreStep::Executed { .. } => {
                    tick.checked_add(1).ok_or(C220CoreError::TimeOverflow)?
                }
                C220CoreStep::Stalled(stall) if stall.resume_tick > tick => {
                    stall.resume_tick.min(tick_limit)
                }
                _ => return Err(C220CoreError::NonprogressingStall),
            };
            events.push(event);
        }
    }

    pub fn step_word_at(&mut self, tick: u64, word: u32) -> Result<C220CoreStep, C220CoreError> {
        let gate = self.advance_to(tick)?;
        if let Some(stall) = gate {
            return Ok(C220CoreStep::Stalled(stall));
        }
        if let Some(resume_tick) = self.scalar_timing.dependency_tick(word, tick) {
            return Ok(C220CoreStep::Stalled(C220Stall {
                tick,
                pc: self.execution.core().scalar().pc(),
                resume_tick,
                cause: C220StallCause::ScalarDependency,
            }));
        }
        if is_mte2_word(word) {
            return match self.execution.step_at(tick, word, &self.memory)? {
                C220TimedMte2Step::Stalled(stall) => Ok(C220CoreStep::Stalled(stall)),
                result @ C220TimedMte2Step::Executed { .. } => Ok(C220CoreStep::Executed {
                    tick,
                    instruction: C220CoreInstruction::Mte2(result),
                }),
            };
        }
        let pc = self.execution.core().scalar().pc();
        let instruction = if is_mte3_word(word) {
            if let Some(flag) = FlagInstruction::decode(Architecture::Dav2201, word)
                && flag.source_pipe_code == 1
                && flag.trigger_pipe_code == 5
                && flag.operation == FlagOperation::Wait
                && let Some(resume_tick) = self.vector.pending_visibility_tick()
                && tick < resume_tick
            {
                return Ok(C220CoreStep::Stalled(C220Stall {
                    tick,
                    pc,
                    resume_tick,
                    cause: C220StallCause::VectorDependency,
                }));
            }
            if let Some(flag) = FlagInstruction::decode(Architecture::Dav2201, word)
                && flag.source_pipe_code == 5
                && flag.trigger_pipe_code == 1
                && flag.operation == FlagOperation::Wait
            {
                let flag_id = flag
                    .resolve(pc, self.execution.core().scalar().machine().xregs())
                    .flag_id;
                if let Ok(flag_id) = u8::try_from(flag_id)
                    && let Some(resume_tick) = self.mte3.completion_ready_tick(flag_id)
                    && tick < resume_tick
                {
                    return Ok(C220CoreStep::Stalled(C220Stall {
                        tick,
                        pc,
                        resume_tick,
                        cause: C220StallCause::Mte3Dependency,
                    }));
                }
            }
            let (ticket, requests) = if C220DmaMovDescriptor::is_word(word) {
                if tick < self.mte3.next_issue_tick() {
                    return Ok(C220CoreStep::Stalled(C220Stall {
                        tick,
                        pc,
                        resume_tick: self.mte3.next_issue_tick(),
                        cause: C220StallCause::Mte3IssueRate,
                    }));
                }
                let plan = self.execution.core().preview_c220_mte3_transfer(word)?;
                let (ticket, requests) = self.mte3.preview_issue(tick, plan)?;
                if self.pending_output.is_some() {
                    return Err(C220Mte3TimingError::TicketMismatch.into());
                }
                (Some(ticket), requests)
            } else {
                (None, Vec::new())
            };
            let (step, prepared) = self
                .execution
                .core_mut()
                .step_c220_output_word_deferred(word)?;
            match step.action {
                C220OutputAction::CopyToHbm { .. } => {
                    let ticket = ticket.ok_or(C220Mte3TimingError::TicketMismatch)?;
                    self.mte3.issue(ticket)?;
                    self.pending_output = Some(C220PendingOutput {
                        data_ready_tick: ticket.data_ready_tick,
                        prepared: prepared.ok_or(C220Mte3TimingError::TicketMismatch)?,
                    });
                }
                C220OutputAction::SetMte3CompletionFlag { flag_id, .. } => {
                    self.mte3.set_completion_flag(flag_id)?;
                }
                C220OutputAction::WaitMte3CompletionFlag { flag_id, .. } => {
                    self.mte3.wait_completion_flag(flag_id)?;
                }
                _ => {}
            }
            C220CoreInstruction::Mte3 {
                step,
                requests,
                ticket,
            }
        } else {
            match word {
                _ if FlagInstruction::decode(Architecture::Dav2201, word).is_some_and(|flag| {
                    flag.source_pipe_code == 1 && flag.trigger_pipe_code == 0
                }) =>
                {
                    let flag = FlagInstruction::decode(Architecture::Dav2201, word)
                        .expect("matched vector-to-scalar flag")
                        .resolve(pc, self.execution.core().scalar().machine().xregs());
                    match flag.instruction.operation {
                        FlagOperation::Set => {
                            if self.vector_to_scalar_flags.contains_key(&flag.flag_id) {
                                return Err(C220CoreError::VectorScalarFlagAlreadySet {
                                    flag_id: flag.flag_id,
                                });
                            }
                            let ready_tick = self.vector.pending_drain_tick().unwrap_or(tick);
                            self.vector_to_scalar_flags.insert(flag.flag_id, ready_tick);
                        }
                        FlagOperation::Wait => {
                            let ready_tick = self
                                .vector_to_scalar_flags
                                .get(&flag.flag_id)
                                .copied()
                                .ok_or(C220CoreError::VectorScalarWaitWithoutFlag {
                                    flag_id: flag.flag_id,
                                })?;
                            if tick < ready_tick {
                                return Ok(C220CoreStep::Stalled(C220Stall {
                                    tick,
                                    pc,
                                    resume_tick: ready_tick,
                                    cause: C220StallCause::VectorDependency,
                                }));
                            }
                            self.vector_to_scalar_flags.remove(&flag.flag_id);
                        }
                    }
                    self.execution.core_mut().commit_c220_vector_issue(None);
                    C220CoreInstruction::VectorToScalarFlag(flag)
                }
                _ if C220MoveVaInstruction::decode(word).is_some() => {
                    let decoded = C220MoveVaInstruction::decode(word).expect("matched decode");
                    let instruction = C220CoreInstruction::VectorMoveAddress {
                        pc,
                        word,
                        instruction: decoded,
                    };
                    self.issue_vector_at(tick, &instruction)?;
                    self.va
                        .write_pair(decoded, self.execution.core().scalar().machine().xregs());
                    self.execution.core_mut().commit_c220_vector_issue(None);
                    instruction
                }
                _ if C220MovevInstruction::decode(word).is_some() => {
                    let step = self.execution.core().preview_c220_movev_word(word)?;
                    let instruction = C220CoreInstruction::VectorMove(step);
                    self.issue_vector_at(tick, &instruction)?;
                    self.execution.core_mut().commit_c220_vector_issue(None);
                    instruction
                }
                _ if C220NchwInstruction::decode(word).is_some() => {
                    let decoded = C220NchwInstruction::decode(word).expect("matched decode");
                    let control = self.execution.core().scalar().machine().xregs()
                        [usize::from(decoded.control_register)];
                    let step = plan_c220_nchw_issue(
                        pc,
                        word,
                        control,
                        &self.va,
                        self.execution.core().ub(),
                    )?;
                    let destination = step.rows.first().map(|rows| rows.destination[0]);
                    let instruction = C220CoreInstruction::VectorNchw(step);
                    self.issue_vector_at(tick, &instruction)?;
                    self.execution
                        .core_mut()
                        .commit_c220_vector_issue(destination);
                    instruction
                }
                _ if C220MoveMaskInstruction::decode(word).is_some() => {
                    let step = plan_c220_move_mask_issue(
                        pc,
                        word,
                        self.execution.core().scalar().machine().xregs(),
                        self.execution.core().ub(),
                    )?;
                    let destination =
                        matches!(step.instruction.direction, C220MoveMaskDirection::ToMemory)
                            .then_some(step.address);
                    let instruction = C220CoreInstruction::VectorMoveMask(step);
                    self.issue_vector_at(tick, &instruction)?;
                    self.execution
                        .core_mut()
                        .commit_c220_vector_issue(destination);
                    instruction
                }
                _ if C220CompareMaskInstruction::decode(word).is_some() => {
                    let decoded = C220CompareMaskInstruction::decode(word).expect("matched decode");
                    let machine = self.execution.core().scalar().machine();
                    let registers = machine.xregs();
                    let step = plan_c220_compare_mask_issue(
                        pc,
                        word,
                        registers[usize::from(decoded.control_register)],
                        C220VectorMaskState {
                            control: machine
                                .spr_value(3)
                                .ok_or(C220VectorError::MissingMaskState)?,
                            low: machine
                                .spr_value(100)
                                .ok_or(C220VectorError::MissingMaskState)?,
                            high: machine
                                .spr_value(101)
                                .ok_or(C220VectorError::MissingMaskState)?,
                        },
                        registers,
                        self.execution.core().ub(),
                    )?;
                    let instruction = C220CoreInstruction::VectorCompareMask(step);
                    self.issue_vector_at(tick, &instruction)?;
                    self.execution.core_mut().commit_c220_vector_issue(None);
                    instruction
                }
                _ if C220SelectInstruction::decode(word).is_some() => {
                    let decoded = C220SelectInstruction::decode(word).expect("matched decode");
                    let machine = self.execution.core().scalar().machine();
                    let registers = machine.xregs();
                    let control_value = registers[usize::from(decoded.control_register)];
                    let mode = C220SelectMode::decode(control_value).ok_or(
                        C220VectorError::UnsupportedSelectMode(((control_value >> 48) & 3) as u8),
                    )?;
                    if matches!(mode, C220SelectMode::TensorTensor)
                        && self.vector.has_pending_compare_mask_write()
                    {
                        return Ok(C220CoreStep::Stalled(C220Stall {
                            tick,
                            pc,
                            resume_tick: self.vector.pending_drain_tick().unwrap_or(tick),
                            cause: C220StallCause::VectorDependency,
                        }));
                    }
                    let step = plan_c220_select_issue(
                        pc,
                        word,
                        control_value,
                        C220VectorMaskState {
                            control: machine
                                .spr_value(3)
                                .ok_or(C220VectorError::MissingMaskState)?,
                            low: machine
                                .spr_value(100)
                                .ok_or(C220VectorError::MissingMaskState)?,
                            high: machine
                                .spr_value(101)
                                .ok_or(C220VectorError::MissingMaskState)?,
                        },
                        self.vector.compare_mask(),
                        registers,
                        self.execution.core().ub(),
                    )?;
                    let destination = step.addresses.destination;
                    let instruction = C220CoreInstruction::VectorSelect(step);
                    self.issue_vector_at(tick, &instruction)?;
                    self.execution
                        .core_mut()
                        .commit_c220_vector_issue(Some(destination));
                    instruction
                }
                _ if C220PackedCompareInstruction::decode(word).is_some() => {
                    let decoded =
                        C220PackedCompareInstruction::decode(word).expect("matched decode");
                    let registers = self.execution.core().scalar().machine().xregs();
                    let control = registers[usize::from(decoded.control_register)];
                    let step = plan_c220_packed_compare_issue(
                        pc,
                        word,
                        control,
                        registers,
                        self.execution.core().ub(),
                    )?;
                    let destination = step.addresses.destination;
                    let instruction = C220CoreInstruction::VectorPackedCompare(step);
                    self.issue_vector_at(tick, &instruction)?;
                    self.execution
                        .core_mut()
                        .commit_c220_vector_issue(Some(destination));
                    instruction
                }
                _ if C220ReductionInstruction::decode(word).is_some() => {
                    let step = self.execution.core().preview_c220_reduction_word(word)?;
                    let produces_output =
                        !step.instruction.writes_accumulator() && !step.iteration_masks.is_empty();
                    let destination = step.addresses.destination;
                    let instruction = C220CoreInstruction::VectorReduction(step);
                    self.issue_vector_at(tick, &instruction)?;
                    self.execution
                        .core_mut()
                        .commit_c220_vector_issue(produces_output.then_some(destination));
                    instruction
                }
                _ if C220TernaryInstruction::decode(word).is_some() => {
                    let step = self.execution.core().preview_c220_ternary_word(word)?;
                    let destination = step.addresses.destination;
                    let instruction = C220CoreInstruction::VectorTernary(step);
                    self.issue_vector_at(tick, &instruction)?;
                    self.execution
                        .core_mut()
                        .commit_c220_vector_issue(Some(destination));
                    instruction
                }
                _ if C220GatherInstruction::decode(word).is_some() => {
                    let step = self.execution.core().preview_c220_gather_word(word)?;
                    let destination = step.destination_address;
                    let instruction = C220CoreInstruction::VectorGather(step);
                    self.issue_vector_at(tick, &instruction)?;
                    self.execution
                        .core_mut()
                        .commit_c220_vector_issue(Some(destination));
                    instruction
                }
                _ if C220VecArithmeticHint::from_word(word).is_some() => {
                    let step = self.execution.core().preview_c220_vector_word(word)?;
                    let destination_address = step.addresses.destination;
                    let instruction = C220CoreInstruction::VectorArithmetic(step);
                    self.issue_vector_at(tick, &instruction)?;
                    self.execution
                        .core_mut()
                        .commit_c220_vector_issue(Some(destination_address));
                    instruction
                }
                _ if C220VectorScalarInstruction::decode(word).is_some() => {
                    let step = self
                        .execution
                        .core()
                        .preview_c220_vector_scalar_word(word)?;
                    let destination_address = step.addresses.destination;
                    let instruction = C220CoreInstruction::VectorScalar(step);
                    self.issue_vector_at(tick, &instruction)?;
                    self.execution
                        .core_mut()
                        .commit_c220_vector_issue(Some(destination_address));
                    instruction
                }
                _ if C220ShiftInstruction::decode(word).is_some() => {
                    let step = self.execution.core().preview_c220_shift_word(word)?;
                    let destination_address = step.addresses.destination;
                    let instruction = C220CoreInstruction::VectorShift(step);
                    self.issue_vector_at(tick, &instruction)?;
                    self.execution
                        .core_mut()
                        .commit_c220_vector_issue(Some(destination_address));
                    instruction
                }
                _ if C220CopyInstruction::decode(word).is_some() => {
                    let step = self.execution.core().preview_c220_copy_word(word)?;
                    let destination_address = step.addresses.destination;
                    let instruction = C220CoreInstruction::VectorCopy(step);
                    self.issue_vector_at(tick, &instruction)?;
                    self.execution
                        .core_mut()
                        .commit_c220_vector_issue(Some(destination_address));
                    instruction
                }
                _ if C220BroadcastInstruction::decode(word).is_some() => {
                    let step = self.execution.core().preview_c220_broadcast_word(word)?;
                    let destination_address = step.destination_address;
                    let has_repeats = step.control.repeat_count != 0;
                    let instruction = C220CoreInstruction::VectorBroadcast(step);
                    self.issue_vector_at(tick, &instruction)?;
                    self.execution
                        .core_mut()
                        .commit_c220_vector_issue(has_repeats.then_some(destination_address));
                    instruction
                }
                _ if C220TransposeInstruction::decode(word).is_some() => {
                    let step = self.execution.core().preview_c220_transpose_word(word)?;
                    let destination_address = step.destination_address;
                    let instruction = C220CoreInstruction::VectorTranspose(step);
                    self.issue_vector_at(tick, &instruction)?;
                    self.execution
                        .core_mut()
                        .commit_c220_vector_issue(Some(destination_address));
                    instruction
                }
                _ if matches!(
                    PipelineBarrierStep::decode(Architecture::Dav2201, pc, word),
                    Some(PipelineBarrierStep {
                        scope: PipelineBarrierScope::All,
                        ..
                    })
                ) =>
                {
                    if let Some(resume_tick) = self.vector.pending_drain_tick()
                        && tick < resume_tick
                    {
                        return Ok(C220CoreStep::Stalled(C220Stall {
                            tick,
                            pc,
                            resume_tick,
                            cause: C220StallCause::VectorDependency,
                        }));
                    }
                    C220CoreInstruction::Barrier(self.execution.core_mut().step_barrier_word(word)?)
                }
                _ => {
                    let timing = if let Some(hint) = C220ScalarConversionHint::from_word(word) {
                        Some(
                            C220ScalarTimingTicket::for_conversion(tick, hint)
                                .ok_or(C220CoreError::TimeOverflow)?,
                        )
                    } else {
                        None
                    };
                    let step = self
                        .execution
                        .core_mut()
                        .step_scalar_word_with_ub(word, &mut self.memory)?;
                    if let Some(ticket) = timing {
                        self.scalar_timing.issue(ticket);
                    }
                    C220CoreInstruction::Scalar { step, timing }
                }
            }
        };
        self.execution.finish_other_at(tick)?;
        Ok(C220CoreStep::Executed { tick, instruction })
    }

    fn issue_vector_at(
        &mut self,
        tick: u64,
        instruction: &C220CoreInstruction,
    ) -> Result<(), C220CoreError> {
        let uops = instruction.vector_uops()?;
        let stores = instruction
            .vector_stores()
            .expect("vector instruction has stores");
        let compute = match instruction {
            C220CoreInstruction::VectorArithmetic(issue) => {
                Some(C220VectorReadIssue::Arithmetic(issue))
            }
            C220CoreInstruction::VectorScalar(issue) => {
                Some(C220VectorReadIssue::VectorScalar(issue))
            }
            C220CoreInstruction::VectorShift(issue) => Some(C220VectorReadIssue::Shift(issue)),
            C220CoreInstruction::VectorCopy(issue) => Some(C220VectorReadIssue::Copy(issue)),
            C220CoreInstruction::VectorBroadcast(issue) => {
                Some(C220VectorReadIssue::Broadcast(issue))
            }
            C220CoreInstruction::VectorTranspose(issue) => {
                Some(C220VectorReadIssue::Transpose(issue))
            }
            C220CoreInstruction::VectorCompareMask(issue) => {
                Some(C220VectorReadIssue::CompareMask(issue))
            }
            C220CoreInstruction::VectorMoveMask(issue) => {
                Some(C220VectorReadIssue::MoveMask(issue))
            }
            C220CoreInstruction::VectorSelect(issue) => Some(C220VectorReadIssue::Select(issue)),
            C220CoreInstruction::VectorPackedCompare(issue) => {
                Some(C220VectorReadIssue::PackedCompare(issue))
            }
            C220CoreInstruction::VectorReduction(issue) => {
                Some(C220VectorReadIssue::Reduction(issue))
            }
            C220CoreInstruction::VectorTernary(issue) => Some(C220VectorReadIssue::Ternary(issue)),
            C220CoreInstruction::VectorGather(issue) => Some(C220VectorReadIssue::Gather(issue)),
            C220CoreInstruction::VectorNchw(issue) => Some(C220VectorReadIssue::Nchw(issue)),
            _ => None,
        };
        self.vector.issue_at(tick, &uops, stores, compute)?;
        Ok(())
    }

    fn commit_ready_output_at(&mut self, tick: u64) -> Result<(), C220CoreError> {
        if let Some(pending) = self.pending_output.as_ref()
            && tick >= pending.data_ready_tick
        {
            self.memory.write_segments_at(&pending.prepared.writes)?;
            self.pending_output = None;
        }
        Ok(())
    }
}

fn is_mte2_word(word: u32) -> bool {
    is_mte2_transfer(word)
        || FlagInstruction::decode(Architecture::Dav2201, word).is_some_and(|instruction| {
            instruction.source_pipe_code == 4 && matches!(instruction.trigger_pipe_code, 0 | 1)
        })
}

fn is_mte3_word(word: u32) -> bool {
    C220DmaMovDescriptor::is_word(word)
        || FlagInstruction::decode(Architecture::Dav2201, word).is_some_and(|instruction| {
            matches!(
                (instruction.source_pipe_code, instruction.trigger_pipe_code),
                (1, 5) | (1, 4) | (5, 1)
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroU64;

    use crate::isa::c220::mte::CAPTURED_C220_MOV_UB_TO_OUT_WORD;
    use crate::memory::mapped::MappedMemory;
    use crate::memory::region::MemoryRegion;
    use crate::memory::sparse::{MemoryByteState, SparseMemory};
    use crate::memory::ub::UbMemory;
    use crate::sim::c220::fp16::C220Fp16Mode;
    use crate::sim::c220::vector::{
        C220_CAPTURED_MOVEV_CONTROL, C220_CAPTURED_MOVEV_WORD, C220_CAPTURED_VADD_CONTROL,
        C220_CAPTURED_VADD_WORD,
    };
    use crate::sim::machine::ScalarMachine;
    use crate::sim::mte_stepper::{
        C220_MTE3_TO_VECTOR_SET_FLAG_WORD, C220_MTE3_TO_VECTOR_WAIT_FLAG_WORD,
        C220_VECTOR_TO_MTE3_SET_FLAG_WORD, C220_VECTOR_TO_MTE3_WAIT_FLAG_WORD,
    };
    use crate::sim::stepper::ScalarStepper;

    #[test]
    fn reduction_state_waits_for_both_repeats() {
        let memory = SparseMemory::new(vec![MemoryRegion::unknown(256)], 512, 512);
        let memory = MappedMemory::bind(memory, &[0x2000]).unwrap();
        let mut ub = UbMemory::new(4096, 256);
        for (address, value) in [(0x100, 1.0_f32), (0x200, 2.0_f32)] {
            ub.write_states(
                address,
                &value
                    .to_le_bytes()
                    .repeat(64)
                    .into_iter()
                    .map(MemoryByteState::Known)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        }
        let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
        machine.set_xreg(2, 0x100).unwrap();
        machine.set_xreg(3, 0x800).unwrap();
        machine
            .set_xreg(4, (2_u64 << 56) | 1 | (1 << 16) | (1 << 32) | (8 << 40))
            .unwrap();
        machine.set_spr_value(3, 0).unwrap();
        machine.set_spr_value(100, u64::MAX).unwrap();
        machine.set_spr_value(101, 0).unwrap();
        let execution = MteCoreStepper::new(ScalarStepper::new(machine, 0x4000), ub);
        let rate = NonZeroU64::new(32).unwrap();
        let mut core = C220Core::new(
            execution,
            memory,
            C220CoreTimingRules {
                mte2: C220Mte2TimingRules {
                    issue_interval: NonZeroU64::new(1).unwrap(),
                    startup_ticks: 0,
                    bytes_per_tick: rate,
                    retire_ticks: 0,
                },
                mte3: C220Mte3TimingRules {
                    issue_interval: NonZeroU64::new(1).unwrap(),
                    startup_ticks: 0,
                    bytes_per_tick: rate,
                    retire_ticks: 0,
                },
                vector: C220VectorTimingRules {
                    dispatch_ticks: 0,
                    uop_issue_interval: NonZeroU64::new(1).unwrap(),
                    ub_response_ticks: 1,
                },
            },
        )
        .unwrap();
        assert!(matches!(
            core.step_word_at(0, 0x83c6_2392).unwrap(),
            C220CoreStep::Executed {
                instruction: C220CoreInstruction::VectorReduction(_),
                ..
            }
        ));
        assert!(matches!(
            core.step_word_at(1, 0x40a0_0400).unwrap(),
            C220CoreStep::Executed {
                instruction: C220CoreInstruction::VectorToScalarFlag(_),
                ..
            }
        ));
        let C220CoreStep::Stalled(stall) = core.step_word_at(2, 0x40c0_0400).unwrap() else {
            panic!("scalar wait should observe pending vector work");
        };
        assert!(matches!(
            core.step_word_at(stall.resume_tick, 0x40c0_0400).unwrap(),
            C220CoreStep::Executed {
                instruction: C220CoreInstruction::VectorToScalarFlag(_),
                ..
            }
        ));
        assert_eq!(
            core.execution().core().scalar().machine().spr_value(87),
            Some(192.0_f32.to_bits().into())
        );
        let max_tick = stall.resume_tick + 1;
        assert!(matches!(
            core.step_word_at(max_tick, 0x83c6_2410).unwrap(),
            C220CoreStep::Executed {
                instruction: C220CoreInstruction::VectorReduction(_),
                ..
            }
        ));
        core.advance_to(max_tick + 64).unwrap();
        assert_eq!(
            core.execution().core().scalar().machine().spr_value(63),
            Some(u64::from(2.0_f32.to_bits()) | (127_u64 << 32))
        );
        assert_eq!(
            core.execution().core().ub().read_known(0x820, 8).unwrap(),
            [2.0_f32.to_le_bytes(), 63_u32.to_le_bytes()].concat()
        );
    }

    #[test]
    fn moveva_updates_only_its_selected_pair_and_retires_as_vector_work() {
        let memory = SparseMemory::new(vec![MemoryRegion::unknown(256)], 512, 512);
        let memory = MappedMemory::bind(memory, &[0x2000]).unwrap();
        let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
        machine.set_xreg(6, 0x120).unwrap();
        machine.set_xreg(7, 0x160).unwrap();
        let execution =
            MteCoreStepper::new(ScalarStepper::new(machine, 0x4000), UbMemory::new(512, 256));
        let rate = NonZeroU64::new(32).unwrap();
        let mut core = C220Core::new(
            execution,
            memory,
            C220CoreTimingRules {
                mte2: C220Mte2TimingRules {
                    issue_interval: NonZeroU64::new(1).unwrap(),
                    startup_ticks: 0,
                    bytes_per_tick: rate,
                    retire_ticks: 0,
                },
                mte3: C220Mte3TimingRules {
                    issue_interval: NonZeroU64::new(1).unwrap(),
                    startup_ticks: 0,
                    bytes_per_tick: rate,
                    retire_ticks: 0,
                },
                vector: C220VectorTimingRules {
                    dispatch_ticks: 0,
                    uop_issue_interval: NonZeroU64::new(1).unwrap(),
                    ub_response_ticks: 1,
                },
            },
        )
        .unwrap();
        let C220CoreStep::Executed {
            instruction: first @ C220CoreInstruction::VectorMoveAddress { .. },
            ..
        } = core.step_word_at(0, 0x8000_6380).unwrap()
        else {
            panic!("MOVEVA should issue");
        };
        assert_eq!(first.vector_uops().unwrap()[0].stages.execute_ticks, 1);
        assert_eq!(core.va_registers().entry(0, 0), Some(9));
        assert_eq!(core.va_registers().entry(0, 1), Some(11));
        assert_eq!(core.va_registers().entry(0, 2), None);
        assert!(matches!(
            core.step_word_at(1, 0x8000_6390).unwrap(),
            C220CoreStep::Executed {
                instruction: C220CoreInstruction::VectorMoveAddress { .. },
                ..
            }
        ));
        assert_eq!(core.va_registers().entry(0, 2), Some(9));
        assert_eq!(core.va_registers().entry(0, 3), Some(11));
        core.advance_to(30).unwrap();
        assert_eq!(core.last_vector_releases().len(), 2);
        assert!(core.vector_pipeline().last_read_samples().is_empty());
    }

    #[test]
    fn nchw_uses_va_rows_and_two_timed_uops_for_each_element_width() {
        for (opcode, width, source_high, destination_high) in [
            (0x8200_0680_u32, 1_usize, true, true),
            (0x8240_0680, 2, false, false),
            (0x8280_0680, 4, false, false),
        ] {
            let memory = SparseMemory::new(vec![MemoryRegion::unknown(256)], 512, 512);
            let memory = MappedMemory::bind(memory, &[0x2000]).unwrap();
            let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
            machine.set_xreg(5, (1_u64 << 56) | (1 << 16) | 1).unwrap();
            let mut ub = UbMemory::new(4096, 256);
            for row in 0..16 {
                let mut source = [0_u8; 32];
                for (column, chunk) in source.chunks_exact_mut(width).enumerate() {
                    let value = (row * 100 + column + 1) as u32;
                    chunk.copy_from_slice(&value.to_le_bytes()[..width]);
                }
                ub.write_states(
                    0x200 + (row * 32) as u64,
                    &source.map(MemoryByteState::Known),
                )
                .unwrap();
                ub.write_states(
                    0x600 + (row * 32) as u64,
                    &[MemoryByteState::Known(0xaa); 32],
                )
                .unwrap();
            }
            let execution = MteCoreStepper::new(ScalarStepper::new(machine, 0x4000), ub);
            let rate = NonZeroU64::new(32).unwrap();
            let mut core = C220Core::new(
                execution,
                memory,
                C220CoreTimingRules {
                    mte2: C220Mte2TimingRules {
                        issue_interval: NonZeroU64::new(1).unwrap(),
                        startup_ticks: 0,
                        bytes_per_tick: rate,
                        retire_ticks: 0,
                    },
                    mte3: C220Mte3TimingRules {
                        issue_interval: NonZeroU64::new(1).unwrap(),
                        startup_ticks: 0,
                        bytes_per_tick: rate,
                        retire_ticks: 0,
                    },
                    vector: C220VectorTimingRules {
                        dispatch_ticks: 0,
                        uop_issue_interval: NonZeroU64::new(1).unwrap(),
                        ub_response_ticks: 1,
                    },
                },
            )
            .unwrap();
            let mut tick = 0;
            for (base_va, base_address) in [(0_u32, 0x600_u64), (2, 0x200)] {
                for half in 0..2_u64 {
                    let va = base_va + half as u32;
                    for pair in 0..4_u64 {
                        let row = half * 8 + pair * 2;
                        let source_0 = base_address - 32 + row * 32;
                        core.execution
                            .core_mut()
                            .scalar_mut()
                            .machine_mut()
                            .set_xreg(6, source_0)
                            .unwrap();
                        core.execution
                            .core_mut()
                            .scalar_mut()
                            .machine_mut()
                            .set_xreg(7, source_0 + 32)
                            .unwrap();
                        let word = 0x8000_0000
                            | (va << 17)
                            | ((pair as u32 * 2) << 3)
                            | (6 << 12)
                            | (7 << 7);
                        core.step_word_at(tick, word).unwrap();
                        tick += 1;
                    }
                }
            }
            let word = opcode
                | (2 << 12)
                | (5 << 2)
                | u32::from(destination_high)
                | (u32::from(source_high) << 1);
            let C220CoreStep::Executed {
                instruction: C220CoreInstruction::VectorNchw(issue),
                ..
            } = core.step_word_at(tick, word).unwrap()
            else {
                panic!("VNCHWCONV should issue");
            };
            let uops = C220CoreInstruction::VectorNchw(issue)
                .vector_uops()
                .unwrap();
            assert_eq!(uops.len(), 2);
            assert!(uops.iter().all(|uop| uop.stages.execute_ticks == 1));
            core.advance_to(300).unwrap();
            assert_eq!(
                core.last_vector_releases()
                    .iter()
                    .filter(|release| release.pc == 0x4040)
                    .count(),
                2
            );
            let output_rows = if width == 4 { 8 } else { 16 };
            for row in 0..output_rows {
                for column in 0..16 {
                    let source_offset = (column * 32
                        + if width == 1 {
                            usize::from(source_high) * 16 + row
                        } else {
                            row * width
                        }) as u64;
                    let destination_offset = if width == 1 {
                        (row * 32 + usize::from(destination_high) * 16 + column) as u64
                    } else {
                        (row * 16 * width + column * width) as u64
                    };
                    assert_eq!(
                        core.execution()
                            .core()
                            .ub()
                            .read_known(0x600 + destination_offset, width)
                            .unwrap(),
                        core.execution()
                            .core()
                            .ub()
                            .read_known(0x200 + source_offset, width)
                            .unwrap()
                    );
                }
            }
        }
    }

    #[test]
    fn transpose_reads_full_matrix_before_two_half_tile_writebacks() {
        for in_place in [false, true] {
            let word = 0x8240_0c00 | (3 << 17) | (4 << 12);
            let memory = SparseMemory::new(vec![MemoryRegion::unknown(256)], 512, 512);
            let memory = MappedMemory::bind(memory, &[0x2000]).unwrap();
            let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
            let source_address = if in_place { 0x400 } else { 0 };
            machine.set_xreg(3, 0x400).unwrap();
            machine.set_xreg(4, source_address).unwrap();
            let mut ub = UbMemory::new(2048, 256);
            for block in 0..16 {
                let bytes = (0..16)
                    .flat_map(|lane| ((block * 16 + lane) as u16).to_le_bytes())
                    .map(MemoryByteState::Known)
                    .collect::<Vec<_>>();
                ub.write_states(source_address + (block * 32) as u64, &bytes)
                    .unwrap();
                if !in_place {
                    ub.write_states(
                        0x400 + (block * 32) as u64,
                        &[MemoryByteState::Known(0xaa); 32],
                    )
                    .unwrap();
                }
            }
            let execution = MteCoreStepper::new(ScalarStepper::new(machine, 0x4000), ub);
            let rate = NonZeroU64::new(32).unwrap();
            let mut core = C220Core::new(
                execution,
                memory,
                C220CoreTimingRules {
                    mte2: C220Mte2TimingRules {
                        issue_interval: NonZeroU64::new(1).unwrap(),
                        startup_ticks: 0,
                        bytes_per_tick: rate,
                        retire_ticks: 0,
                    },
                    mte3: C220Mte3TimingRules {
                        issue_interval: NonZeroU64::new(1).unwrap(),
                        startup_ticks: 0,
                        bytes_per_tick: rate,
                        retire_ticks: 0,
                    },
                    vector: C220VectorTimingRules {
                        dispatch_ticks: 0,
                        uop_issue_interval: NonZeroU64::new(1).unwrap(),
                        ub_response_ticks: 1,
                    },
                },
            )
            .unwrap();
            let C220CoreStep::Executed {
                instruction: C220CoreInstruction::VectorTranspose(issue),
                ..
            } = core.step_word_at(0, word).unwrap()
            else {
                panic!("transpose should issue");
            };
            let uops = C220CoreInstruction::VectorTranspose(issue)
                .vector_uops()
                .unwrap();
            assert_eq!(uops.len(), 2);
            assert!(
                uops.iter()
                    .all(|uop| uop.lane_group.is_none() && uop.stages.execute_ticks == 1)
            );
            core.advance_to(100).unwrap();
            assert_eq!(core.last_vector_releases().len(), 2);
            assert_eq!(core.vector_pipeline().last_read_samples().len(), 1);
            assert_eq!(
                core.vector_pipeline().last_read_samples()[0]
                    .source_0_bytes
                    .len(),
                512
            );
            assert_eq!(
                core.vector_pipeline().last_read_samples()[0].accesses.len(),
                16
            );
            for row in 0..16 {
                for column in 0..16 {
                    let destination = 0x400 + (row * 16 + column) as u64 * 2;
                    let expected = (column * 16 + row) as u16;
                    assert_eq!(
                        core.execution()
                            .core()
                            .ub()
                            .read_known(destination, 2)
                            .unwrap(),
                        expected.to_le_bytes()
                    );
                }
            }
        }
    }

    #[test]
    fn broadcast_issues_one_full_tile_uop_per_repeat() {
        for (opcode, width) in [(0x8000_0044_u32, 2_usize), (0x8000_004c, 4)] {
            let word = opcode | (3 << 17) | (4 << 12) | (5 << 7);
            let memory = SparseMemory::new(vec![MemoryRegion::unknown(256)], 512, 512);
            let memory = MappedMemory::bind(memory, &[0x2000]).unwrap();
            let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
            machine.set_xreg(3, 0x200).unwrap();
            machine.set_xreg(4, 0).unwrap();
            machine
                .set_xreg(5, (2_u64 << 56) | (1 << 52) | (20 << 32) | 2)
                .unwrap();
            let mut ub = UbMemory::new(2048, 256);
            let mut source = vec![0_u8; 16 * width];
            for (index, chunk) in source.chunks_exact_mut(width).enumerate() {
                chunk.copy_from_slice(&((index + 1) as u32).to_le_bytes()[..width]);
            }
            ub.write_states(
                0,
                &source
                    .iter()
                    .copied()
                    .map(MemoryByteState::Known)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            for repeat in 0..2 {
                for block in 0..8 {
                    let address = 0x200_u64 + 32 * (276 * repeat + 2 * block);
                    ub.write_states(address, &[MemoryByteState::Known(0xaa); 32])
                        .unwrap();
                }
            }
            let execution = MteCoreStepper::new(ScalarStepper::new(machine, 0x4000), ub);
            let rate = NonZeroU64::new(32).unwrap();
            let mut core = C220Core::new(
                execution,
                memory,
                C220CoreTimingRules {
                    mte2: C220Mte2TimingRules {
                        issue_interval: NonZeroU64::new(1).unwrap(),
                        startup_ticks: 0,
                        bytes_per_tick: rate,
                        retire_ticks: 0,
                    },
                    mte3: C220Mte3TimingRules {
                        issue_interval: NonZeroU64::new(1).unwrap(),
                        startup_ticks: 0,
                        bytes_per_tick: rate,
                        retire_ticks: 0,
                    },
                    vector: C220VectorTimingRules {
                        dispatch_ticks: 0,
                        uop_issue_interval: NonZeroU64::new(1).unwrap(),
                        ub_response_ticks: 1,
                    },
                },
            )
            .unwrap();
            let C220CoreStep::Executed {
                instruction: C220CoreInstruction::VectorBroadcast(issue),
                ..
            } = core.step_word_at(0, word).unwrap()
            else {
                panic!("broadcast should issue");
            };
            let uops = C220CoreInstruction::VectorBroadcast(issue)
                .vector_uops()
                .unwrap();
            assert_eq!(uops.len(), 2);
            assert!(
                uops.iter()
                    .all(|uop| uop.lane_group.is_none() && uop.stages.execute_ticks == 2)
            );
            assert_eq!(
                core.execution()
                    .core()
                    .ub()
                    .read_known(0x200, width)
                    .unwrap(),
                vec![0xaa; width]
            );
            core.advance_to(100).unwrap();
            let ub = core.execution().core().ub();
            for repeat in 0..2 {
                for block in 0..8 {
                    let address = 0x200_u64 + 32 * (276 * repeat + 2 * block);
                    let element = (repeat * 8 + block) as usize;
                    let expected = &source[element * width..(element + 1) * width];
                    assert_eq!(
                        ub.read_known(address, 32).unwrap(),
                        expected.repeat(32 / width)
                    );
                }
            }
            assert_eq!(core.vector_pipeline().last_read_samples().len(), 2);
            assert!(
                core.vector_pipeline()
                    .last_read_samples()
                    .iter()
                    .all(|sample| sample.lane_group.is_none())
            );
        }
    }

    #[test]
    fn vector_scalar_s32_and_f32_capture_scalar_and_delay_writeback() {
        for (opcode, source_bits, scalar_bits, result_bits, execute_ticks, saturating) in [
            (
                0x92c0_0000,
                (-4.0_f32).to_bits(),
                0.5_f32.to_bits(),
                (-2.0_f32).to_bits(),
                8,
                false,
            ),
            (
                0x92c0_0000,
                (-0.0_f32).to_bits(),
                0.5_f32.to_bits(),
                (-0.0_f32).to_bits(),
                8,
                false,
            ),
            (0x9600_0000, u32::MAX, 2, 2, 5, false),
            (0x9600_0001, u32::MAX, 2, u32::MAX, 5, false),
            (0x9700_0000, u32::MAX, 2, 1, 5, false),
            (0x9700_0001, u32::MAX, 2, u32::MAX - 1, 6, false),
            (0x9700_0000, i32::MAX as u32, 1, i32::MAX as u32, 5, true),
            (0x9700_0001, i32::MIN as u32, 2, i32::MIN as u32, 6, true),
            (
                0x96c0_0000,
                1.5_f32.to_bits(),
                2.0_f32.to_bits(),
                2.0_f32.to_bits(),
                5,
                false,
            ),
            (
                0x96c0_0001,
                1.5_f32.to_bits(),
                2.0_f32.to_bits(),
                1.5_f32.to_bits(),
                5,
                false,
            ),
            (
                0x96c0_0000,
                (-0.0_f32).to_bits(),
                0.0_f32.to_bits(),
                0.0_f32.to_bits(),
                5,
                false,
            ),
            (
                0x96c0_0001,
                (-0.0_f32).to_bits(),
                0.0_f32.to_bits(),
                (-0.0_f32).to_bits(),
                5,
                false,
            ),
            (
                0x97c0_0000,
                1.5_f32.to_bits(),
                2.0_f32.to_bits(),
                3.5_f32.to_bits(),
                7,
                false,
            ),
            (
                0x97c0_0001,
                1.5_f32.to_bits(),
                2.0_f32.to_bits(),
                3.0_f32.to_bits(),
                8,
                false,
            ),
        ] {
            let word = opcode | (3 << 17) | (4 << 12) | (6 << 7) | (5 << 2);
            let memory = SparseMemory::new(vec![MemoryRegion::unknown(256)], 512, 512);
            let memory = MappedMemory::bind(memory, &[0x2000]).unwrap();
            let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
            machine.set_xreg(3, 0x100).unwrap();
            machine.set_xreg(4, 0).unwrap();
            machine.set_xreg(5, (1_u64 << 56) | (1 << 16) | 1).unwrap();
            machine.set_xreg(6, u64::from(scalar_bits)).unwrap();
            let control_spr = (1_u64 << 56) | (u64::from(saturating) << 53);
            machine.set_spr_value(3, control_spr).unwrap();
            machine.set_spr_value(100, 1).unwrap();
            machine.set_spr_value(101, 0).unwrap();
            let mut ub = UbMemory::new(512, 256);
            let mut source = [0; 32];
            source[..4].copy_from_slice(&source_bits.to_le_bytes());
            ub.write_states(0, &source.map(MemoryByteState::Known))
                .unwrap();
            ub.write_states(0x100, &[MemoryByteState::Known(0xaa); 32])
                .unwrap();
            let execution = MteCoreStepper::new(ScalarStepper::new(machine, 0x4000), ub);
            let rate = NonZeroU64::new(32).unwrap();
            let mut core = C220Core::new(
                execution,
                memory,
                C220CoreTimingRules {
                    mte2: C220Mte2TimingRules {
                        issue_interval: NonZeroU64::new(1).unwrap(),
                        startup_ticks: 0,
                        bytes_per_tick: rate,
                        retire_ticks: 0,
                    },
                    mte3: C220Mte3TimingRules {
                        issue_interval: NonZeroU64::new(1).unwrap(),
                        startup_ticks: 0,
                        bytes_per_tick: rate,
                        retire_ticks: 0,
                    },
                    vector: C220VectorTimingRules {
                        dispatch_ticks: 0,
                        uop_issue_interval: NonZeroU64::new(1).unwrap(),
                        ub_response_ticks: 1,
                    },
                },
            )
            .unwrap();
            let C220CoreStep::Executed {
                instruction: C220CoreInstruction::VectorScalar(issue),
                ..
            } = core.step_word_at(0, word).unwrap()
            else {
                panic!("vector-scalar instruction should issue");
            };
            assert_eq!(issue.scalar.bits, scalar_bits);
            assert_eq!(issue.scalar.integer_saturating, saturating);
            if saturating {
                core.execution
                    .core_mut()
                    .scalar_mut()
                    .machine_mut()
                    .set_spr_value(3, 1 << 56)
                    .unwrap();
            }
            assert_eq!(
                C220CoreInstruction::VectorScalar(issue)
                    .vector_uops()
                    .unwrap()[0]
                    .stages
                    .execute_ticks,
                execute_ticks
            );
            assert_eq!(
                core.execution().core().ub().read_known(0x100, 4).unwrap(),
                [0xaa; 4]
            );
            core.execution
                .core_mut()
                .scalar_mut()
                .machine_mut()
                .set_xreg(6, 0x8000_0000)
                .unwrap();
            core.advance_to(100).unwrap();
            assert_eq!(
                core.execution().core().ub().read_known(0x100, 4).unwrap(),
                result_bits.to_le_bytes()
            );
            assert_eq!(
                core.execution().core().ub().read_known(0x104, 4).unwrap(),
                [0xaa; 4]
            );
            assert!(
                core.vector_pipeline().last_read_samples()[0]
                    .read1_grants
                    .is_empty()
            );
            assert_eq!(
                core.vector_pipeline().last_read_samples()[0].lanes[0]
                    .fp32_status
                    .is_some(),
                opcode & 0x00c0_0000 == 0x00c0_0000
            );
        }
    }

    #[test]
    fn vector_scalar_16_bit_forms_use_both_lane_groups_and_preserve_inactive_tail() {
        for (opcode, source_bits, scalar_bits, expected, execute_ticks, is_f16, fp_mode, sat) in [
            (0x9240_0000, 0xc400, 0x3800, 0xc000, 8, true, false, false),
            (0x9240_0000, 0x8000, 0x3800, 0x8000, 8, true, false, false),
            (0x9240_0000, 0x7e00, 0x3800, 0, 8, true, false, false),
            (
                0x9640_0000,
                0x3c00_u16,
                0x4000_u16,
                0x4000_u16,
                5,
                true,
                false,
                false,
            ),
            (0x9640_0001, 0x3c00, 0x4000, 0x3c00, 5, true, false, false),
            (0x9740_0000, 0x3c00, 0x4000, 0x4200, 7, true, false, false),
            (0x9740_0001, 0x3c00, 0x4000, 0x4000, 8, true, false, false),
            (0x9740_0000, 0x7c00, 0x3c00, 0x7bff, 7, true, false, false),
            (0x9740_0000, 0x7c00, 0x3c00, 0x7c00, 7, true, true, false),
            (0x9680_0000, u16::MAX, 2, 2, 5, false, false, false),
            (0x9680_0001, u16::MAX, 2, u16::MAX, 5, false, false, false),
            (0x9780_0000, u16::MAX, 2, 1, 5, false, false, false),
            (
                0x9780_0001,
                u16::MAX,
                2,
                u16::MAX - 1,
                6,
                false,
                false,
                false,
            ),
            (
                0x9780_0000,
                i16::MAX as u16,
                1,
                i16::MAX as u16,
                5,
                false,
                false,
                true,
            ),
            (
                0x9780_0001,
                i16::MIN as u16,
                2,
                i16::MIN as u16,
                6,
                false,
                false,
                true,
            ),
        ] {
            let word = opcode | (3 << 17) | (4 << 12) | (6 << 7) | (5 << 2);
            let memory = SparseMemory::new(vec![MemoryRegion::unknown(256)], 512, 512);
            let memory = MappedMemory::bind(memory, &[0x2000]).unwrap();
            let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
            machine.set_xreg(3, 0x200).unwrap();
            machine.set_xreg(4, 0).unwrap();
            machine.set_xreg(5, (1_u64 << 56) | (1 << 16) | 1).unwrap();
            machine.set_xreg(6, u64::from(scalar_bits)).unwrap();
            let control_spr = (1_u64 << 56) | (u64::from(fp_mode) << 48) | (u64::from(sat) << 53);
            machine.set_spr_value(3, control_spr).unwrap();
            machine.set_spr_value(100, 65).unwrap();
            machine.set_spr_value(101, 0).unwrap();
            let mut ub = UbMemory::new(1024, 256);
            let mut source = [0; 256];
            source[..2].copy_from_slice(&source_bits.to_le_bytes());
            source[128..130].copy_from_slice(&source_bits.to_le_bytes());
            ub.write_states(0, &source.map(MemoryByteState::Known))
                .unwrap();
            ub.write_states(0x200, &[MemoryByteState::Known(0xaa); 256])
                .unwrap();
            let execution = MteCoreStepper::new(ScalarStepper::new(machine, 0x4000), ub);
            let rate = NonZeroU64::new(32).unwrap();
            let mut core = C220Core::new(
                execution,
                memory,
                C220CoreTimingRules {
                    mte2: C220Mte2TimingRules {
                        issue_interval: NonZeroU64::new(1).unwrap(),
                        startup_ticks: 0,
                        bytes_per_tick: rate,
                        retire_ticks: 0,
                    },
                    mte3: C220Mte3TimingRules {
                        issue_interval: NonZeroU64::new(1).unwrap(),
                        startup_ticks: 0,
                        bytes_per_tick: rate,
                        retire_ticks: 0,
                    },
                    vector: C220VectorTimingRules {
                        dispatch_ticks: 0,
                        uop_issue_interval: NonZeroU64::new(1).unwrap(),
                        ub_response_ticks: 1,
                    },
                },
            )
            .unwrap();
            let C220CoreStep::Executed {
                instruction: C220CoreInstruction::VectorScalar(issue),
                ..
            } = core.step_word_at(0, word).unwrap()
            else {
                panic!("s16 vector-scalar instruction should issue");
            };
            assert_eq!(
                issue.scalar.fp16_mode,
                C220Fp16Mode::from_control_spr(control_spr)
            );
            assert_eq!(issue.scalar.integer_saturating, sat);
            if sat {
                core.execution
                    .core_mut()
                    .scalar_mut()
                    .machine_mut()
                    .set_spr_value(3, 1 << 56)
                    .unwrap();
            }
            if source_bits == 0x7c00 {
                core.execution
                    .core_mut()
                    .scalar_mut()
                    .machine_mut()
                    .set_spr_value(3, control_spr ^ (1 << 48))
                    .unwrap();
            }
            let uops = C220CoreInstruction::VectorScalar(issue)
                .vector_uops()
                .unwrap();
            assert_eq!(uops.len(), 2);
            assert_eq!(uops[0].stages.execute_ticks, execute_ticks);
            core.advance_to(200).unwrap();
            assert_eq!(
                core.execution().core().ub().read_known(0x200, 2).unwrap(),
                expected.to_le_bytes()
            );
            assert_eq!(
                core.execution().core().ub().read_known(0x280, 2).unwrap(),
                expected.to_le_bytes()
            );
            assert_eq!(
                core.execution().core().ub().read_known(0x282, 2).unwrap(),
                [0xaa; 2]
            );
            assert_eq!(core.vector_pipeline().pending_uops(), 0);
            assert_eq!(
                core.vector_pipeline().last_read_samples()[0].lanes[0]
                    .fp16_status
                    .is_some(),
                is_f16
            );
        }
    }

    #[test]
    fn vector_s32_binary_operations_use_delayed_reads_and_captured_saturation() {
        for (opcode, first, second, expected, execute_ticks, saturating) in [
            (
                0x8500_0000,
                i32::MAX as u32,
                1_u32,
                i32::MIN as u32,
                5,
                false,
            ),
            (0x8500_0000, i32::MAX as u32, 1, i32::MAX as u32, 5, true),
            (0x8500_0001, i32::MIN as u32, 1, i32::MAX as u32, 5, false),
            (0x8500_0001, i32::MIN as u32, 1, i32::MIN as u32, 5, true),
            (0x8900_0000, i32::MAX as u32, 2, u32::MAX - 1, 6, false),
            (0x8900_0000, i32::MAX as u32, 2, i32::MAX as u32, 6, true),
            (0x8700_0000, u32::MAX, 2, 2, 5, false),
            (0x8700_0001, u32::MAX, 2, u32::MAX, 5, false),
        ] {
            let word = opcode | (3 << 17) | (4 << 12) | (5 << 7) | (6 << 2);
            let memory = SparseMemory::new(vec![MemoryRegion::unknown(256)], 512, 512);
            let memory = MappedMemory::bind(memory, &[0x2000]).unwrap();
            let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
            machine.set_xreg(3, 0x200).unwrap();
            machine.set_xreg(4, 0).unwrap();
            machine.set_xreg(5, 0x100).unwrap();
            machine.set_xreg(6, 0x0100_0808_0801_0101).unwrap();
            let control_spr = (1_u64 << 56) | (u64::from(saturating) << 53);
            machine.set_spr_value(3, control_spr).unwrap();
            machine.set_spr_value(100, 1).unwrap();
            machine.set_spr_value(101, 0).unwrap();
            let mut ub = UbMemory::new(1024, 256);
            ub.write_states(0, &[MemoryByteState::Known(0); 32])
                .unwrap();
            ub.write_states(0x100, &[MemoryByteState::Known(0); 32])
                .unwrap();
            ub.write_states(0, &first.to_le_bytes().map(MemoryByteState::Known))
                .unwrap();
            ub.write_states(0x100, &second.to_le_bytes().map(MemoryByteState::Known))
                .unwrap();
            ub.write_states(0x200, &[MemoryByteState::Known(0xaa); 32])
                .unwrap();
            let execution = MteCoreStepper::new(ScalarStepper::new(machine, 0x4000), ub);
            let rate = NonZeroU64::new(32).unwrap();
            let mut core = C220Core::new(
                execution,
                memory,
                C220CoreTimingRules {
                    mte2: C220Mte2TimingRules {
                        issue_interval: NonZeroU64::new(1).unwrap(),
                        startup_ticks: 0,
                        bytes_per_tick: rate,
                        retire_ticks: 0,
                    },
                    mte3: C220Mte3TimingRules {
                        issue_interval: NonZeroU64::new(1).unwrap(),
                        startup_ticks: 0,
                        bytes_per_tick: rate,
                        retire_ticks: 0,
                    },
                    vector: C220VectorTimingRules {
                        dispatch_ticks: 0,
                        uop_issue_interval: NonZeroU64::new(1).unwrap(),
                        ub_response_ticks: 1,
                    },
                },
            )
            .unwrap();
            let C220CoreStep::Executed {
                instruction: C220CoreInstruction::VectorArithmetic(issue),
                ..
            } = core.step_word_at(0, word).unwrap()
            else {
                panic!("S32 vector instruction should issue");
            };
            assert_eq!(issue.modes.integer_saturating, saturating);
            assert!(issue.hint.has_s32_value_path());
            assert_eq!(
                C220CoreInstruction::VectorArithmetic(issue)
                    .vector_uops()
                    .unwrap()[0]
                    .stages
                    .execute_ticks,
                execute_ticks
            );
            assert_eq!(
                core.execution().core().ub().read_known(0x200, 4).unwrap(),
                [0xaa; 4]
            );
            core.execution
                .core_mut()
                .scalar_mut()
                .machine_mut()
                .set_spr_value(3, control_spr ^ (1 << 53))
                .unwrap();
            core.advance_to(100).unwrap();
            assert_eq!(
                core.execution().core().ub().read_known(0x200, 4).unwrap(),
                expected.to_le_bytes()
            );
            assert!(
                core.vector_pipeline().last_read_samples()[0].lanes[0]
                    .fp32_status
                    .is_none()
            );
        }
    }

    #[test]
    fn vector_s16_binary_operations_issue_two_lane_groups() {
        for (opcode, first, second, expected, saturating, widen_bit, execute_ticks) in [
            (0x9480_0000, 3, -5, 0, false, false, 5),
            (0x9480_0000, i16::MAX, 1, 0, false, false, 5),
            (0x9480_0000, i16::MAX, 1, i16::MAX, true, false, 5),
            (0x9480_0001, 3, 5, 0, false, true, 5),
            (0x8580_0000, i16::MAX, 1_i16, i16::MIN, false, false, 5),
            (0x8580_0000, i16::MAX, 1, i16::MAX, true, false, 5),
            (0x8580_0001, i16::MIN, 1, i16::MAX, false, false, 5),
            (0x8580_0001, i16::MIN, 1, i16::MIN, true, false, 5),
            (0x8980_0000, i16::MAX, 2, -2, false, false, 6),
            (0x8980_0000, i16::MAX, 2, i16::MAX, true, false, 6),
            (0x8780_0000, -1, 2, 2, false, false, 5),
            (0x8780_0000, -1, 2, 2, false, true, 5),
            (0x8780_0001, -1, 2, -1, false, false, 5),
            (0x9a40_0000, 0x0f0f, 0x3333, 0x3f3f, false, true, 1),
            (0x9a40_0001, 0x0f0f, 0x3333, 0x0303, false, true, 1),
        ] {
            let word = opcode | (3 << 17) | (4 << 12) | (5 << 7) | (6 << 2);
            let memory = SparseMemory::new(vec![MemoryRegion::unknown(256)], 512, 512);
            let memory = MappedMemory::bind(memory, &[0x2000]).unwrap();
            let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
            machine.set_xreg(3, 0x200).unwrap();
            machine.set_xreg(4, 0).unwrap();
            machine.set_xreg(5, 0x100).unwrap();
            machine.set_xreg(6, 0x0100_0808_0801_0101).unwrap();
            let control_spr = (u64::from(saturating) << 53) | (u64::from(widen_bit) << 52);
            machine.set_spr_value(3, control_spr).unwrap();
            machine.set_spr_value(100, 1).unwrap();
            machine.set_spr_value(101, 1).unwrap();
            let mut ub = UbMemory::new(1024, 256);
            for address in [0, 0x80, 0x100, 0x180] {
                ub.write_states(address, &[MemoryByteState::Known(0); 32])
                    .unwrap();
            }
            for offset in [0, 0x80] {
                ub.write_states(offset, &first.to_le_bytes().map(MemoryByteState::Known))
                    .unwrap();
                ub.write_states(
                    offset + 0x100,
                    &second.to_le_bytes().map(MemoryByteState::Known),
                )
                .unwrap();
                ub.write_states(offset + 0x200, &[MemoryByteState::Known(0xaa); 32])
                    .unwrap();
            }
            let execution = MteCoreStepper::new(ScalarStepper::new(machine, 0x4000), ub);
            let rate = NonZeroU64::new(32).unwrap();
            let mut core = C220Core::new(
                execution,
                memory,
                C220CoreTimingRules {
                    mte2: C220Mte2TimingRules {
                        issue_interval: NonZeroU64::new(1).unwrap(),
                        startup_ticks: 0,
                        bytes_per_tick: rate,
                        retire_ticks: 0,
                    },
                    mte3: C220Mte3TimingRules {
                        issue_interval: NonZeroU64::new(1).unwrap(),
                        startup_ticks: 0,
                        bytes_per_tick: rate,
                        retire_ticks: 0,
                    },
                    vector: C220VectorTimingRules {
                        dispatch_ticks: 0,
                        uop_issue_interval: NonZeroU64::new(1).unwrap(),
                        ub_response_ticks: 1,
                    },
                },
            )
            .unwrap();
            let C220CoreStep::Executed {
                instruction: C220CoreInstruction::VectorArithmetic(issue),
                ..
            } = core.step_word_at(0, word).unwrap()
            else {
                panic!("S16 vector instruction should issue");
            };
            assert_eq!(issue.result_element_bytes, 2);
            assert_eq!(issue.modes.integer_saturating, saturating);
            assert_eq!(issue.modes.widen_s16, widen_bit);
            let uops = C220CoreInstruction::VectorArithmetic(issue)
                .vector_uops()
                .unwrap();
            assert_eq!(uops.len(), 2);
            assert_eq!((uops[0].lane_group, uops[1].lane_group), (Some(0), Some(1)));
            assert!(
                uops.iter()
                    .all(|uop| uop.stages.execute_ticks == execute_ticks)
            );
            core.execution
                .core_mut()
                .scalar_mut()
                .machine_mut()
                .set_spr_value(3, control_spr ^ (1 << 53))
                .unwrap();
            core.advance_to(100).unwrap();
            for address in [0x200, 0x280] {
                assert_eq!(
                    core.execution().core().ub().read_known(address, 2).unwrap(),
                    expected.to_le_bytes()
                );
            }
            assert_eq!(
                core.vector_pipeline()
                    .last_read_samples()
                    .iter()
                    .map(|sample| sample.lane_group)
                    .collect::<Vec<_>>(),
                [Some(0), Some(1)]
            );
        }
    }

    #[test]
    fn scalar_conversion_retires_after_two_ticks_and_blocks_dependent_conversion() {
        let memory = SparseMemory::new(vec![MemoryRegion::unknown(64)], 128, 128);
        let memory = MappedMemory::bind(memory, &[0x2000]).unwrap();
        let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
        machine.set_xreg(8, u64::from(4.75_f32.to_bits())).unwrap();
        machine.set_xreg(11, u64::from(6.5_f32.to_bits())).unwrap();
        let execution =
            MteCoreStepper::new(ScalarStepper::new(machine, 0x4000), UbMemory::new(512, 256));
        let rate = NonZeroU64::new(32).unwrap();
        let mut core = C220Core::new(
            execution,
            memory,
            C220CoreTimingRules {
                mte2: C220Mte2TimingRules {
                    issue_interval: NonZeroU64::new(1).unwrap(),
                    startup_ticks: 0,
                    bytes_per_tick: rate,
                    retire_ticks: 0,
                },
                mte3: C220Mte3TimingRules {
                    issue_interval: NonZeroU64::new(1).unwrap(),
                    startup_ticks: 0,
                    bytes_per_tick: rate,
                    retire_ticks: 0,
                },
                vector: C220VectorTimingRules {
                    dispatch_ticks: 0,
                    uop_issue_interval: NonZeroU64::new(1).unwrap(),
                    ub_response_ticks: 1,
                },
            },
        )
        .unwrap();

        let C220CoreStep::Executed {
            instruction:
                C220CoreInstruction::Scalar {
                    timing: Some(ticket),
                    ..
                },
            ..
        } = core.step_word_at(10, 0x0210_8583).unwrap()
        else {
            panic!("scalar conversion should issue");
        };
        assert_eq!((ticket.issue_tick, ticket.retire_tick), (10, 12));
        assert_eq!(ticket.execution_stage, 2);
        assert_eq!(core.scalar_timing().pending_xreg_retirement(8), Some(12));

        let C220CoreStep::Stalled(stall) = core.step_word_at(11, 0x0210_8583).unwrap() else {
            panic!("dependent conversion should wait");
        };
        assert_eq!(stall.cause, C220StallCause::ScalarDependency);
        assert_eq!(stall.resume_tick, 12);
        assert_eq!(core.execution().core().scalar().pc(), 0x4004);

        let C220CoreStep::Stalled(move_stall) = core.step_word_at(11, 0x0202_8800).unwrap() else {
            panic!("scalar register read should wait");
        };
        assert_eq!(move_stall.cause, C220StallCause::ScalarDependency);
        assert_eq!(move_stall.resume_tick, 12);

        assert!(matches!(
            core.step_word_at(11, 0x0216_b583).unwrap(),
            C220CoreStep::Executed {
                instruction: C220CoreInstruction::Scalar { .. },
                ..
            }
        ));
        assert_eq!(core.scalar_timing().pending_xreg_retirement(11), Some(13));
        assert!(matches!(
            core.step_word_at(12, 0x0202_8800).unwrap(),
            C220CoreStep::Executed { .. }
        ));
        assert_eq!(core.scalar_timing().pending_xreg_retirement(8), None);
        assert_eq!(core.execution().core().scalar().machine().xregs()[1], 4);
        assert!(matches!(
            core.step_word_at(13, 0x0210_8583).unwrap(),
            C220CoreStep::Executed { .. }
        ));
        assert_eq!(core.scalar_timing().pending_xreg_retirement(8), Some(15));
    }

    #[test]
    fn vabs_uses_modeled_fifteen_tick_execution_stage() {
        let word = 0x83c0_0300 | (3 << 17) | (4 << 12) | (5 << 2);
        let make_core = || {
            let memory = SparseMemory::new(vec![MemoryRegion::unknown(256)], 512, 512);
            let memory = MappedMemory::bind(memory, &[0x2000]).unwrap();
            let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
            machine.set_xreg(3, 0x100).unwrap();
            machine.set_xreg(4, 0).unwrap();
            machine.set_xreg(5, (1_u64 << 56) | (1 << 16) | 1).unwrap();
            machine.set_spr_value(3, 1 << 56).unwrap();
            machine.set_spr_value(100, 1).unwrap();
            machine.set_spr_value(101, 0).unwrap();
            let mut ub = UbMemory::new(512, 256);
            let mut source = [0; 32];
            source[..4].copy_from_slice(&(-1.0_f32).to_le_bytes());
            ub.write_states(0, &source.map(MemoryByteState::Known))
                .unwrap();
            let execution = MteCoreStepper::new(ScalarStepper::new(machine, 0x4000), ub);
            let rate = NonZeroU64::new(32).unwrap();
            C220Core::new(
                execution,
                memory,
                C220CoreTimingRules {
                    mte2: C220Mte2TimingRules {
                        issue_interval: NonZeroU64::new(1).unwrap(),
                        startup_ticks: 0,
                        bytes_per_tick: rate,
                        retire_ticks: 0,
                    },
                    mte3: C220Mte3TimingRules {
                        issue_interval: NonZeroU64::new(1).unwrap(),
                        startup_ticks: 0,
                        bytes_per_tick: rate,
                        retire_ticks: 0,
                    },
                    vector: C220VectorTimingRules {
                        dispatch_ticks: 0,
                        uop_issue_interval: NonZeroU64::new(1).unwrap(),
                        ub_response_ticks: 1,
                    },
                },
            )
            .unwrap()
        };
        let mut timed = make_core();
        let C220CoreStep::Executed {
            instruction: C220CoreInstruction::VectorArithmetic(issue),
            ..
        } = timed.step_word_at(0, word).unwrap()
        else {
            panic!("VABS should issue to the vector pipeline");
        };
        assert_eq!(
            C220CoreInstruction::VectorArithmetic(issue)
                .vector_uops()
                .unwrap()[0]
                .stages
                .execute_ticks,
            15
        );
        let visible = timed.vector_pipeline().pending_visibility_tick().unwrap();
        timed.advance_to(visible).unwrap();
        assert_eq!(
            timed.execution().core().ub().read_known(0x100, 4).unwrap(),
            1.0_f32.to_le_bytes()
        );
        assert!(
            timed.vector_pipeline().last_read_samples()[0]
                .read1_grants
                .is_empty()
        );
    }

    #[test]
    fn vnot_b16_reads_one_source_and_preserves_inactive_ub_lanes() {
        let word = 0x8240_0800 | (3 << 17) | (4 << 12) | (5 << 2);
        let memory = SparseMemory::new(vec![MemoryRegion::unknown(256)], 512, 512);
        let memory = MappedMemory::bind(memory, &[0x2000]).unwrap();
        let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
        machine.set_xreg(3, 0x100).unwrap();
        machine.set_xreg(4, 0).unwrap();
        machine.set_xreg(5, (1_u64 << 56) | (1 << 16) | 1).unwrap();
        machine.set_spr_value(3, 1 << 52).unwrap();
        machine.set_spr_value(100, 1).unwrap();
        machine.set_spr_value(101, 0).unwrap();
        let mut ub = UbMemory::new(512, 256);
        let mut source = [0; 32];
        source[..2].copy_from_slice(&0x00f0_u16.to_le_bytes());
        ub.write_states(0, &source.map(MemoryByteState::Known))
            .unwrap();
        ub.write_states(0x100, &[MemoryByteState::Known(0xaa); 32])
            .unwrap();
        let execution = MteCoreStepper::new(ScalarStepper::new(machine, 0x4000), ub);
        let rate = NonZeroU64::new(32).unwrap();
        let mut core = C220Core::new(
            execution,
            memory,
            C220CoreTimingRules {
                mte2: C220Mte2TimingRules {
                    issue_interval: NonZeroU64::new(1).unwrap(),
                    startup_ticks: 0,
                    bytes_per_tick: rate,
                    retire_ticks: 0,
                },
                mte3: C220Mte3TimingRules {
                    issue_interval: NonZeroU64::new(1).unwrap(),
                    startup_ticks: 0,
                    bytes_per_tick: rate,
                    retire_ticks: 0,
                },
                vector: C220VectorTimingRules {
                    dispatch_ticks: 0,
                    uop_issue_interval: NonZeroU64::new(1).unwrap(),
                    ub_response_ticks: 1,
                },
            },
        )
        .unwrap();
        let C220CoreStep::Executed {
            instruction: C220CoreInstruction::VectorArithmetic(issue),
            ..
        } = core.step_word_at(0, word).unwrap()
        else {
            panic!("VNOT should issue to the vector pipeline");
        };
        assert_eq!(issue.hint.source_1_register, None);
        assert_eq!(issue.result_element_bytes, 2);
        let uops = C220CoreInstruction::VectorArithmetic(issue)
            .vector_uops()
            .unwrap();
        assert_eq!(uops[0].stages.execute_ticks, 1);
        core.advance_to(100).unwrap();
        let ub = core.execution().core().ub();
        assert_eq!(ub.read_known(0x100, 2).unwrap(), 0xff0f_u16.to_le_bytes());
        assert_eq!(ub.read_known(0x102, 2).unwrap(), [0xaa; 2]);
        assert!(
            core.vector_pipeline().last_read_samples()[0]
                .read1_grants
                .is_empty()
        );
    }

    #[test]
    fn vector_shifts_capture_scalar_and_follow_masked_pipeline() {
        for (opcode, source, shift, expected, width) in [
            (0x9c80_0003_u32, 0x8001_u32, 1_u64, 2_u32, 2_usize),
            (0x9cc0_0003, 0x8000_0001, 32, 0, 4),
            (0x9b00_0001, 0x8001, 1, 0x4000, 2),
            (0x9b40_0000, 0xfffd, 1, 0xfffe, 2),
            (0x9b40_0001, 0xfffd, 1, 0xffff, 2),
            (0x9b40_0000, 0xfffd, 17, 0xffff, 2),
            (0x9b40_0001, 0xfffd, 17, 0, 2),
            (0x9b80_0000, 0x8000_0001, 64, 0x8000_0001, 4),
            (0x9bc0_0001, 0xffff_fffd, 1, 0xffff_ffff, 4),
        ] {
            let word = opcode | (3 << 17) | (4 << 12) | (6 << 7) | (5 << 2);
            let memory = SparseMemory::new(vec![MemoryRegion::unknown(256)], 512, 512);
            let memory = MappedMemory::bind(memory, &[0x2000]).unwrap();
            let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
            machine.set_xreg(3, 0x100).unwrap();
            machine.set_xreg(4, 0).unwrap();
            machine.set_xreg(5, (1_u64 << 56) | (1 << 16) | 1).unwrap();
            machine.set_xreg(6, shift).unwrap();
            machine.set_spr_value(3, 0).unwrap();
            machine.set_spr_value(100, 1).unwrap();
            machine.set_spr_value(101, 0).unwrap();
            let mut ub = UbMemory::new(512, 256);
            let mut tile = [0; 32];
            tile[..width].copy_from_slice(&source.to_le_bytes()[..width]);
            ub.write_states(0, &tile.map(MemoryByteState::Known))
                .unwrap();
            ub.write_states(0x100, &[MemoryByteState::Known(0xaa); 32])
                .unwrap();
            let execution = MteCoreStepper::new(ScalarStepper::new(machine, 0x4000), ub);
            let rate = NonZeroU64::new(32).unwrap();
            let mut core = C220Core::new(
                execution,
                memory,
                C220CoreTimingRules {
                    mte2: C220Mte2TimingRules {
                        issue_interval: NonZeroU64::new(1).unwrap(),
                        startup_ticks: 0,
                        bytes_per_tick: rate,
                        retire_ticks: 0,
                    },
                    mte3: C220Mte3TimingRules {
                        issue_interval: NonZeroU64::new(1).unwrap(),
                        startup_ticks: 0,
                        bytes_per_tick: rate,
                        retire_ticks: 0,
                    },
                    vector: C220VectorTimingRules {
                        dispatch_ticks: 0,
                        uop_issue_interval: NonZeroU64::new(1).unwrap(),
                        ub_response_ticks: 1,
                    },
                },
            )
            .unwrap();
            let C220CoreStep::Executed {
                instruction: C220CoreInstruction::VectorShift(issue),
                ..
            } = core.step_word_at(0, word).unwrap()
            else {
                panic!("shift should issue to the vector pipeline");
            };
            assert_eq!(issue.shift, shift as u32);
            assert_eq!(
                C220CoreInstruction::VectorShift(issue)
                    .vector_uops()
                    .unwrap()[0]
                    .stages
                    .execute_ticks,
                6
            );
            assert_eq!(
                core.execution()
                    .core()
                    .ub()
                    .read_known(0x100, width)
                    .unwrap(),
                vec![0xaa; width]
            );
            core.advance_to(100).unwrap();
            let ub = core.execution().core().ub();
            assert_eq!(
                ub.read_known(0x100, width).unwrap(),
                expected.to_le_bytes()[..width]
            );
            assert_eq!(ub.read_known(0x100 + width as u64, 2).unwrap(), [0xaa; 2]);
            assert!(
                core.vector_pipeline().last_read_samples()[0]
                    .read1_grants
                    .is_empty()
            );
        }
    }

    #[test]
    fn vector_copy_uses_both_lane_groups_and_preserves_masked_destinations() {
        for (opcode, element_bytes, selected_lane) in [
            (0x8240_0700_u32, 2_usize, 64_usize),
            (0x8280_0700_u32, 4_usize, 32_usize),
        ] {
            let word = opcode | (3 << 17) | (4 << 12) | (5 << 2);
            let memory = SparseMemory::new(vec![MemoryRegion::unknown(256)], 512, 512);
            let memory = MappedMemory::bind(memory, &[0x2000]).unwrap();
            let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
            machine.set_xreg(3, 0x200).unwrap();
            machine.set_xreg(4, 0).unwrap();
            machine.set_xreg(5, (1_u64 << 56) | (8 << 32) | 1).unwrap();
            machine.set_spr_value(3, 0).unwrap();
            machine.set_spr_value(100, 1).unwrap();
            machine
                .set_spr_value(101, u64::from(element_bytes == 2))
                .unwrap();
            if element_bytes == 4 {
                machine.set_spr_value(100, 1 | (1 << 32)).unwrap();
            }
            let mut ub = UbMemory::new(1024, 256);
            let mut source = [0_u8; 256];
            source[..element_bytes]
                .copy_from_slice(&0x1234_5678_u32.to_le_bytes()[..element_bytes]);
            source[128..128 + element_bytes]
                .copy_from_slice(&0xabcd_ef12_u32.to_le_bytes()[..element_bytes]);
            ub.write_states(0, &source.map(MemoryByteState::Known))
                .unwrap();
            ub.write_states(0x200, &[MemoryByteState::Known(0xaa); 256])
                .unwrap();
            let execution = MteCoreStepper::new(ScalarStepper::new(machine, 0x4000), ub);
            let rate = NonZeroU64::new(32).unwrap();
            let mut core = C220Core::new(
                execution,
                memory,
                C220CoreTimingRules {
                    mte2: C220Mte2TimingRules {
                        issue_interval: NonZeroU64::new(1).unwrap(),
                        startup_ticks: 0,
                        bytes_per_tick: rate,
                        retire_ticks: 0,
                    },
                    mte3: C220Mte3TimingRules {
                        issue_interval: NonZeroU64::new(1).unwrap(),
                        startup_ticks: 0,
                        bytes_per_tick: rate,
                        retire_ticks: 0,
                    },
                    vector: C220VectorTimingRules {
                        dispatch_ticks: 0,
                        uop_issue_interval: NonZeroU64::new(1).unwrap(),
                        ub_response_ticks: 1,
                    },
                },
            )
            .unwrap();
            let C220CoreStep::Executed {
                instruction: C220CoreInstruction::VectorCopy(issue),
                ..
            } = core.step_word_at(0, word).unwrap()
            else {
                panic!("copy should issue to the vector pipeline");
            };
            assert_eq!(issue.control.source_0_block_stride, 1);
            assert_eq!(
                C220CoreInstruction::VectorCopy(issue)
                    .vector_uops()
                    .unwrap()[0]
                    .stages
                    .execute_ticks,
                1
            );
            assert_eq!(
                core.execution()
                    .core()
                    .ub()
                    .read_known(0x200, element_bytes)
                    .unwrap(),
                vec![0xaa; element_bytes]
            );
            core.advance_to(100).unwrap();
            let ub = core.execution().core().ub();
            assert_eq!(
                ub.read_known(0x200, element_bytes).unwrap(),
                source[..element_bytes]
            );
            let selected_offset = (selected_lane * element_bytes) as u64;
            assert_eq!(
                ub.read_known(0x200 + selected_offset, element_bytes)
                    .unwrap(),
                source[128..128 + element_bytes]
            );
            assert_eq!(
                ub.read_known(0x200 + element_bytes as u64, 2).unwrap(),
                [0xaa; 2]
            );
        }
    }

    #[test]
    fn vector_read_samples_ub_after_issue_without_an_implicit_raw_wait() {
        let memory = SparseMemory::new(vec![MemoryRegion::unknown(256)], 512, 512);
        let memory = MappedMemory::bind(memory, &[0x2000]).unwrap();
        let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
        machine.set_xreg(5, C220_CAPTURED_MOVEV_CONTROL).unwrap();
        machine.set_xreg(6, 0x4000_0000).unwrap();
        machine.set_xreg(8, C220_CAPTURED_VADD_CONTROL).unwrap();
        machine.set_xreg(13, 0).unwrap();
        machine.set_xreg(14, 0x200).unwrap();
        machine.set_xreg(16, 0).unwrap();
        machine.set_spr_value(3, 1 << 56).unwrap();
        machine.set_spr_value(100, 32).unwrap();
        machine.set_spr_value(101, 0).unwrap();
        let mut ub = UbMemory::new(1024, 256);
        let ones = 0x3f80_0000_u32
            .to_le_bytes()
            .repeat(64)
            .into_iter()
            .map(MemoryByteState::Known)
            .collect::<Vec<_>>();
        ub.write_states(0, &ones).unwrap();
        ub.write_states(0x200, &ones).unwrap();
        let execution = MteCoreStepper::new(ScalarStepper::new(machine, 0x4000), ub);
        let rate = NonZeroU64::new(32).unwrap();
        let mut core = C220Core::new(
            execution,
            memory,
            C220CoreTimingRules {
                mte2: C220Mte2TimingRules {
                    issue_interval: NonZeroU64::new(1).unwrap(),
                    startup_ticks: 0,
                    bytes_per_tick: rate,
                    retire_ticks: 0,
                },
                mte3: C220Mte3TimingRules {
                    issue_interval: NonZeroU64::new(1).unwrap(),
                    startup_ticks: 0,
                    bytes_per_tick: rate,
                    retire_ticks: 0,
                },
                vector: C220VectorTimingRules {
                    dispatch_ticks: 1,
                    uop_issue_interval: NonZeroU64::new(20).unwrap(),
                    ub_response_ticks: 2,
                },
            },
        )
        .unwrap();
        assert!(matches!(
            core.step_word_at(0, C220_CAPTURED_MOVEV_WORD).unwrap(),
            C220CoreStep::Executed { .. }
        ));
        let movev_visible = core.vector_pipeline().pending_visibility_tick().unwrap();
        assert!(movev_visible < 21);
        core.execution
            .core_mut()
            .scalar_mut()
            .machine_mut()
            .set_xreg(16, 0x400)
            .unwrap();
        assert!(matches!(
            core.step_word_at(1, C220_CAPTURED_VADD_WORD).unwrap(),
            C220CoreStep::Executed {
                instruction: C220CoreInstruction::VectorArithmetic(_),
                ..
            }
        ));
        assert!(core.vector_pipeline().last_read_samples().is_empty());
        assert!(core.execution().core().ub().read_known(0x400, 4).is_err());
        core.advance_to(30).unwrap();
        assert_eq!(
            core.execution().core().ub().read_known(0, 4).unwrap(),
            0x4000_0000_u32.to_le_bytes()
        );
        let sample = &core.vector_pipeline().last_read_samples()[0];
        let last_grant = sample
            .read0_grants
            .iter()
            .chain(&sample.read1_grants)
            .flatten()
            .copied()
            .max()
            .unwrap();
        assert_eq!(sample.tick, last_grant + 6);
        assert_eq!(sample.accesses.len(), 8);
        assert!(sample.accesses.iter().all(|access| access.block_index < 4));
        assert_eq!(&sample.source_0_bytes[..4], &0x4000_0000_u32.to_le_bytes());
        assert_eq!(&sample.source_1_bytes[..4], &0x3f80_0000_u32.to_le_bytes());
        assert_eq!(sample.lanes[0].bits, 0x4040_0000);
        assert!(core.execution().core().ub().read_known(0x400, 4).is_err());
        let arithmetic_visible = core.vector_pipeline().pending_visibility_tick().unwrap();
        core.advance_to(arithmetic_visible).unwrap();
        assert_eq!(
            core.execution().core().ub().read_known(0x400, 4).unwrap(),
            0x4040_0000_u32.to_le_bytes()
        );
    }

    #[test]
    fn halfword_movev_commits_its_tail_group_after_the_first_group() {
        let memory = SparseMemory::new(vec![MemoryRegion::unknown(256)], 512, 512);
        let memory = MappedMemory::bind(memory, &[0x2000]).unwrap();
        let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
        machine.set_xreg(5, C220_CAPTURED_MOVEV_CONTROL).unwrap();
        machine.set_xreg(6, 0x3c00).unwrap();
        machine.set_xreg(16, 0).unwrap();
        machine.set_spr_value(3, 1 << 56).unwrap();
        machine.set_spr_value(100, 65).unwrap();
        machine.set_spr_value(101, 0).unwrap();
        let execution =
            MteCoreStepper::new(ScalarStepper::new(machine, 0x4000), UbMemory::new(256, 256));
        let rate = NonZeroU64::new(32).unwrap();
        let mut core = C220Core::new(
            execution,
            memory,
            C220CoreTimingRules {
                mte2: C220Mte2TimingRules {
                    issue_interval: NonZeroU64::new(1).unwrap(),
                    startup_ticks: 0,
                    bytes_per_tick: rate,
                    retire_ticks: 0,
                },
                mte3: C220Mte3TimingRules {
                    issue_interval: NonZeroU64::new(1).unwrap(),
                    startup_ticks: 0,
                    bytes_per_tick: rate,
                    retire_ticks: 0,
                },
                vector: C220VectorTimingRules {
                    dispatch_ticks: 1,
                    uop_issue_interval: NonZeroU64::new(1).unwrap(),
                    ub_response_ticks: 2,
                },
            },
        )
        .unwrap();
        let halfword_word = (C220_CAPTURED_MOVEV_WORD & !(7 << 22)) | (1 << 22);
        let step = core.step_word_at(0, halfword_word).unwrap();
        let C220CoreStep::Executed {
            instruction: C220CoreInstruction::VectorMove(step),
            ..
        } = step
        else {
            panic!("expected MOVEV");
        };
        let uops = C220CoreInstruction::VectorMove(step).vector_uops().unwrap();
        assert_eq!(uops.len(), 2);
        assert_eq!((uops[0].lane_group, uops[1].lane_group), (Some(0), Some(1)));
        assert_eq!(core.vector_pipeline().pending_ub_responses(), 2);
        assert!(core.execution().core().ub().read_known(0, 2).is_err());
        let final_visibility = core.vector_pipeline().pending_visibility_tick().unwrap();
        core.advance_to(final_visibility - 1).unwrap();
        assert_eq!(
            core.execution().core().ub().read_known(0, 2).unwrap(),
            0x3c00_u16.to_le_bytes()
        );
        assert!(core.execution().core().ub().read_known(128, 2).is_err());
        core.advance_to(final_visibility).unwrap();
        assert_eq!(
            core.execution().core().ub().read_known(128, 2).unwrap(),
            0x3c00_u16.to_le_bytes()
        );
    }

    #[test]
    fn vector_issue_failure_keeps_execution_state_uncommitted() {
        let memory = SparseMemory::new(vec![MemoryRegion::unknown(256)], 512, 512);
        let memory = MappedMemory::bind(memory, &[0x2000]).unwrap();
        let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
        machine.set_xreg(5, C220_CAPTURED_MOVEV_CONTROL).unwrap();
        machine.set_xreg(6, 0x3c00).unwrap();
        machine.set_xreg(16, 0).unwrap();
        machine.set_spr_value(3, 1 << 56).unwrap();
        machine.set_spr_value(100, 1).unwrap();
        machine.set_spr_value(101, 0).unwrap();
        let execution =
            MteCoreStepper::new(ScalarStepper::new(machine, 0x4000), UbMemory::new(256, 256));
        let before = execution.clone();
        let rate = NonZeroU64::new(32).unwrap();
        let mut core = C220Core::new(
            execution,
            memory,
            C220CoreTimingRules {
                mte2: C220Mte2TimingRules {
                    issue_interval: NonZeroU64::new(1).unwrap(),
                    startup_ticks: 0,
                    bytes_per_tick: rate,
                    retire_ticks: 0,
                },
                mte3: C220Mte3TimingRules {
                    issue_interval: NonZeroU64::new(1).unwrap(),
                    startup_ticks: 0,
                    bytes_per_tick: rate,
                    retire_ticks: 0,
                },
                vector: C220VectorTimingRules {
                    dispatch_ticks: u64::MAX,
                    uop_issue_interval: NonZeroU64::new(1).unwrap(),
                    ub_response_ticks: 1,
                },
            },
        )
        .unwrap();
        assert!(matches!(
            core.step_word_at(1, C220_CAPTURED_MOVEV_WORD),
            Err(C220CoreError::VectorPipeline(
                C220VectorPipelineError::TimeOverflow
            ))
        ));
        assert_eq!(core.execution().core(), &before);
        assert_eq!(core.vector_pipeline().pending_uops(), 0);
    }

    #[test]
    fn mte3_completion_wait_uses_the_scheduled_request_service() {
        let regions = vec![
            MemoryRegion::unknown(128),
            MemoryRegion::new(8, 0x2000_u64.to_le_bytes().to_vec()).unwrap(),
        ];
        let memory = SparseMemory::new(regions, 256, 256);
        let memory = MappedMemory::bind(memory, &[0x2000, 0x1000]).unwrap();
        let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
        machine.set_xreg(5, C220_CAPTURED_MOVEV_CONTROL).unwrap();
        machine.set_xreg(6, 0x3f80_0000).unwrap();
        machine.set_xreg(16, 0).unwrap();
        machine.set_spr_value(3, 1 << 56).unwrap();
        machine.set_spr_value(100, 32).unwrap();
        machine.set_spr_value(101, 0).unwrap();
        let execution =
            MteCoreStepper::new(ScalarStepper::new(machine, 0x4000), UbMemory::new(512, 256));
        let rate = NonZeroU64::new(32).unwrap();
        let mut core = C220Core::new(
            execution,
            memory,
            C220CoreTimingRules {
                mte2: C220Mte2TimingRules {
                    issue_interval: NonZeroU64::new(1).unwrap(),
                    startup_ticks: 0,
                    bytes_per_tick: rate,
                    retire_ticks: 0,
                },
                mte3: C220Mte3TimingRules {
                    issue_interval: NonZeroU64::new(1).unwrap(),
                    startup_ticks: 2,
                    bytes_per_tick: rate,
                    retire_ticks: 1,
                },
                vector: C220VectorTimingRules {
                    dispatch_ticks: 2,
                    uop_issue_interval: NonZeroU64::new(1).unwrap(),
                    ub_response_ticks: 3,
                },
            },
        )
        .unwrap();
        core.step_word_at(0, C220_CAPTURED_MOVEV_WORD).unwrap();
        core.execution
            .core_mut()
            .scalar_mut()
            .machine_mut()
            .set_xreg(16, 0x80)
            .unwrap();
        core.step_word_at(1, C220_CAPTURED_MOVEV_WORD).unwrap();
        let machine = core.execution.core_mut().scalar_mut().machine_mut();
        machine.set_xreg(16, 0x100).unwrap();
        core.step_word_at(2, C220_CAPTURED_MOVEV_WORD).unwrap();
        let read_ready_tick = core.vector_pipeline().pending_visibility_tick().unwrap();
        core.execution
            .core_mut()
            .scalar_mut()
            .machine_mut()
            .set_xreg(16, 0x180)
            .unwrap();
        core.step_word_at(3, C220_CAPTURED_MOVEV_WORD).unwrap();
        let machine = core.execution.core_mut().scalar_mut().machine_mut();
        machine.set_xreg(8, C220_CAPTURED_VADD_CONTROL).unwrap();
        machine.set_xreg(13, 0).unwrap();
        machine.set_xreg(14, 0x80).unwrap();
        let movev_visible = core.vector_pipeline().pending_visibility_tick().unwrap();
        assert!(read_ready_tick < movev_visible);
        let last_release = movev_visible - 3;
        core.advance_to(last_release).unwrap();
        assert_eq!(core.last_vector_releases().len(), 4);
        assert!(core.execution().core().ub().read_known(0x180, 4).is_err());
        core.advance_to(read_ready_tick).unwrap();
        core.step_word_at(read_ready_tick, C220_CAPTURED_VADD_WORD)
            .unwrap();
        core.execution
            .core_mut()
            .scalar_mut()
            .machine_mut()
            .set_xreg(14, 0)
            .unwrap();
        let set_tick = read_ready_tick + 1;
        core.step_word_at(set_tick, C220_VECTOR_TO_MTE3_SET_FLAG_WORD)
            .unwrap();
        core.execution
            .core_mut()
            .scalar_mut()
            .machine_mut()
            .set_xreg(13, 0)
            .unwrap();
        let vector_visible = core.vector_pipeline().pending_visibility_tick().unwrap();
        assert!(matches!(
            core.step_word_at(set_tick + 1, C220_VECTOR_TO_MTE3_WAIT_FLAG_WORD)
                .unwrap(),
            C220CoreStep::Stalled(C220Stall {
                resume_tick,
                cause: C220StallCause::VectorDependency,
                ..
            }) if resume_tick == vector_visible
        ));
        core.step_word_at(vector_visible, C220_VECTOR_TO_MTE3_WAIT_FLAG_WORD)
            .unwrap();
        let machine = core.execution.core_mut().scalar_mut().machine_mut();
        machine.set_xreg(14, 0x180).unwrap();
        machine.set_xreg(10, 0x2000).unwrap();
        machine.set_xreg(3, 0x40010).unwrap();
        let issue_tick = vector_visible + 1;
        let issued = core
            .step_word_at(issue_tick, CAPTURED_C220_MOV_UB_TO_OUT_WORD)
            .unwrap();
        let C220CoreStep::Executed {
            instruction:
                C220CoreInstruction::Mte3 {
                    ticket: Some(ticket),
                    ..
                },
            ..
        } = issued
        else {
            panic!("expected a timed MTE3 transfer");
        };
        assert_eq!(ticket.issue_tick, issue_tick);
        assert_eq!(ticket.data_ready_tick, issue_tick + 6);
        assert_eq!(ticket.retire_tick, issue_tick + 7);
        assert_eq!(ticket.uop_count, 1);
        assert!(core.memory().read_known_at(0x2000, 128).is_err());
        core.execution
            .core_mut()
            .scalar_mut()
            .machine_mut()
            .set_xreg(10, 0)
            .unwrap();
        core.step_word_at(issue_tick + 1, C220_MTE3_TO_VECTOR_SET_FLAG_WORD)
            .unwrap();
        core.execution
            .core_mut()
            .scalar_mut()
            .machine_mut()
            .set_xreg(19, 0)
            .unwrap();
        assert!(matches!(
            core.step_word_at(issue_tick + 2, C220_MTE3_TO_VECTOR_WAIT_FLAG_WORD)
                .unwrap(),
            C220CoreStep::Stalled(C220Stall {
                resume_tick,
                cause: C220StallCause::Mte3Dependency,
                ..
            }) if resume_tick == ticket.retire_tick
        ));
        assert!(core.memory().read_known_at(0x2000, 128).is_err());
        assert_eq!(
            core.pending_output_ready_tick(),
            Some(ticket.data_ready_tick)
        );
        assert!(core.advance_to(ticket.data_ready_tick).unwrap().is_none());
        assert_eq!(core.pending_output_ready_tick(), None);
        assert_eq!(
            core.memory().read_known_at(0x2000, 128).unwrap(),
            0x4000_0000_u32.to_le_bytes().repeat(32)
        );
        assert!(matches!(
            core.step_word_at(ticket.data_ready_tick, C220_MTE3_TO_VECTOR_WAIT_FLAG_WORD)
                .unwrap(),
            C220CoreStep::Stalled(C220Stall {
                resume_tick,
                cause: C220StallCause::Mte3Dependency,
                ..
            }) if resume_tick == ticket.retire_tick
        ));
        assert_eq!(
            core.memory().read_known_at(0x2000, 128).unwrap(),
            0x4000_0000_u32.to_le_bytes().repeat(32)
        );
        core.step_word_at(ticket.retire_tick, C220_MTE3_TO_VECTOR_WAIT_FLAG_WORD)
            .unwrap();
        assert_eq!(
            core.memory().read_known_at(0x2000, 128).unwrap(),
            0x4000_0000_u32.to_le_bytes().repeat(32)
        );
    }
}
