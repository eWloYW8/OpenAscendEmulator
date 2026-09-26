use crate::architecture::Architecture;
use crate::isa::c220::cube::C220CubeInstruction;
use crate::isa::c220::hflag::C220HardwareFlagInstruction;
use crate::isa::c220::mte::C220DmaMovDescriptor;
use crate::isa::c220::mte::bias::C220MovL1ToBtInstruction;
use crate::isa::c220::mte::load2d::C220Load2dInstruction;
use crate::isa::c220::mte::load2d_sparse::C220Load2dSparseInstruction;
use crate::isa::c220::mte::load2d_transpose::C220Load2dTransposeInstruction;
use crate::isa::c220::mte::set2d::{C220Set2dDestination, C220Set2dInstruction};
use crate::isa::flow::{FlagInstruction, FlagOperation};
use crate::sim::c220::mte::mte2::is_mte2_transfer;
use crate::sim::c220::vector::dispatch::is_vector_word;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum C220DispatchKind {
    Mte1,
    Fixp,
    Factor(crate::isa::c220::mte::factor::C220FactorLoadInstruction),
    Mte2,
    HardwareFlag(C220HardwareFlagInstruction),
    Cube(C220CubeInstruction),
    CubeFlag,
    CubeSpr(crate::isa::c220::cube::spr::C220CubeSprWrite),
    Vector,
    VectorFlag,
    ScalarFlag,
    Mte3,
    Scalar,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct C220DecodedWord {
    pub(super) kind: C220DispatchKind,
    pub(super) flow_flag: Option<FlagInstruction>,
}

impl C220DecodedWord {
    pub(super) fn decode(word: u32) -> Self {
        let flow_flag = FlagInstruction::decode(Architecture::Dav2201, word);
        let shared_flag_pipe = flow_flag
            .filter(|flag| {
                matches!(flag.source_pipe_code, 0..=5 | 10)
                    && matches!(flag.trigger_pipe_code, 0..=5 | 10)
            })
            .map(|flag| match flag.operation {
                FlagOperation::Set => flag.source_pipe_code,
                FlagOperation::Wait => flag.trigger_pipe_code,
            });
        let kind = if let Some(pipe) = shared_flag_pipe {
            match pipe {
                0 => C220DispatchKind::ScalarFlag,
                1 => C220DispatchKind::VectorFlag,
                2 => C220DispatchKind::CubeFlag,
                3 => C220DispatchKind::Mte1,
                4 => C220DispatchKind::Mte2,
                5 => C220DispatchKind::Mte3,
                10 => C220DispatchKind::Fixp,
                _ => unreachable!("shared event route was checked"),
            }
        } else if crate::isa::c220::mte::fixp::C220FixpInstruction::decode(word).is_some() {
            C220DispatchKind::Fixp
        } else if let Some(instruction) =
            crate::isa::c220::mte::factor::C220FactorLoadInstruction::decode(word)
        {
            C220DispatchKind::Factor(instruction)
        } else if C220Load2dInstruction::decode(word)
            .is_some_and(|instruction| instruction.is_mte1())
            || C220Load2dTransposeInstruction::decode(word).is_some()
            || C220Load2dSparseInstruction::decode(word).is_some()
            || crate::isa::c220::mte::load3d::C220Load3dV2Instruction::decode(word).is_some()
            || crate::isa::c220::mte::spr::C220Mte1SprWrite::decode(word).is_some()
            || C220MovL1ToBtInstruction::decode(word).is_some()
            || C220Set2dInstruction::decode(word)
                .is_some_and(|instruction| instruction.destination != C220Set2dDestination::L1)
        {
            C220DispatchKind::Mte1
        } else if is_mte2_transfer(word)
            || C220Set2dInstruction::decode(word)
                .is_some_and(|instruction| instruction.destination == C220Set2dDestination::L1)
        {
            C220DispatchKind::Mte2
        } else if let Some(instruction) = C220HardwareFlagInstruction::decode(word) {
            C220DispatchKind::HardwareFlag(instruction)
        } else if let Some(instruction) =
            crate::isa::c220::cube::spr::C220CubeSprWrite::decode(word)
        {
            C220DispatchKind::CubeSpr(instruction)
        } else if let Some(instruction) = C220CubeInstruction::decode(word) {
            C220DispatchKind::Cube(instruction)
        } else if is_vector_word(word) {
            C220DispatchKind::Vector
        } else if C220DmaMovDescriptor::is_word(word) {
            C220DispatchKind::Mte3
        } else {
            C220DispatchKind::Scalar
        };
        Self { kind, flow_flag }
    }
}
