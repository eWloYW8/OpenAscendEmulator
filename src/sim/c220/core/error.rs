use thiserror::Error;

use crate::image::loader::DeviceKernelFetchError;
use crate::isa::c220::hflag::{
    C220HardwareFlagError, C220HardwareFlagSourcePipe, C220MatrixMemory,
};
use crate::memory::mapped::MappedMemoryError;
use crate::sim::c220::cube::C220CubeTimingError;
use crate::sim::c220::memory::C220L0cError;
use crate::sim::c220::mte::mte1::load2d::C220Load2dTransferError;
use crate::sim::c220::mte::mte3::C220Mte3TimingError;
use crate::sim::c220::scalar::bus::C220ScalarBusError;
use crate::sim::c220::schedule::C220ScheduleError;
use crate::sim::c220::state::C220ExecutionError;
use crate::sim::c220::sync::C220HardwareFlagTimingError;
use crate::sim::common::scalar::ScalarInstructionError;

#[derive(Debug, Error)]
pub enum C220CoreError {
    #[error(transparent)]
    ExternalFixp(#[from] crate::sim::c220::mte::fixp::C220FixpRuntimeError),
    #[error("factor loads require explicit read port and bandwidth configuration")]
    FactorUnconfigured,
    #[error(transparent)]
    Factor(#[from] crate::sim::c220::mte::factor::C220FactorExecutionError),
    #[error("FIX requires explicit engine, factor-buffer and clock-stage configuration")]
    FixpUnconfigured,
    #[error("FIX L0C capacity differs from the core's shared L0C")]
    FixpCapacityMismatch,
    #[error("FIX command requires {required} request identifiers, exceeding the u32 namespace")]
    FixpRequestCapacity { required: u64 },
    #[error(transparent)]
    FixpExecution(#[from] crate::sim::c220::mte::fixp::C220FixpExecutionError),
    #[error(transparent)]
    FixpEngine(#[from] crate::sim::c220::mte::fixp::C220FixpEngineError),
    #[error("MTE requires explicit L1 geometry and bandwidth configuration")]
    MteUnconfigured,
    #[error("MTE cannot be reconfigured while commands or physical transfers are active")]
    MtePipelineBusy,
    #[error(transparent)]
    MtePipeline(#[from] crate::sim::c220::mte::C220MtePipelineError),
    #[error(transparent)]
    CubeRuntime(#[from] crate::sim::c220::cube::C220CubeRuntimeError),
    #[error(transparent)]
    VectorRuntime(#[from] crate::sim::c220::vector::C220VectorRuntimeError),
    #[error("loaded kernel is not for dav_2201")]
    ArchitectureMismatch,
    #[error(transparent)]
    Schedule(#[from] C220ScheduleError),
    #[error(transparent)]
    Fetch(#[from] DeviceKernelFetchError),
    #[error(transparent)]
    Mte2Runtime(#[from] crate::sim::c220::mte::mte2::C220Mte2RuntimeError),
    #[error(transparent)]
    Execution(#[from] C220ExecutionError),
    #[error(transparent)]
    Mte1Runtime(#[from] crate::sim::c220::mte::mte1::C220Mte1RuntimeError),
    #[error(transparent)]
    HardwareFlagDecode(#[from] C220HardwareFlagError),
    #[error(transparent)]
    HardwareFlagTiming(#[from] C220HardwareFlagTimingError),
    #[error(transparent)]
    Load2d(#[from] C220Load2dTransferError),
    #[error(transparent)]
    Mte3Timing(#[from] C220Mte3TimingError),
    #[error(transparent)]
    Mte3Runtime(#[from] crate::sim::c220::mte::mte3::C220Mte3RuntimeError),
    #[error(transparent)]
    Mte3Transfer(#[from] crate::sim::c220::mte::C220TransferError),
    #[error(transparent)]
    CubeTiming(#[from] C220CubeTimingError),
    #[error(transparent)]
    LocalMemory(#[from] C220L0cError),
    #[error(transparent)]
    OutputMemory(#[from] MappedMemoryError),
    #[error(transparent)]
    Scalar(#[from] ScalarInstructionError<C220ScalarBusError<MappedMemoryError>>),
    #[error("timed core stalled without a future resume tick")]
    NonprogressingStall,
    #[error("tick counter overflowed")]
    TimeOverflow,
    #[error("C220 Cube execution requires initialized SPR3 control state")]
    MissingCubeControlSpr,
    #[error("C220 SET_2D requires initialized SPR15 fill pattern")]
    MissingSet2dPatternSpr,
    #[error("C220 Cube timing requires initialized SPR{spr} control state")]
    MissingCubeTimingSpr { spr: u16 },
    #[error(
        "C220 hardware flag source {source_pipe:?} for {memory:?} requires an unimplemented data path"
    )]
    UnsupportedHardwareFlagCheckpoint {
        source_pipe: C220HardwareFlagSourcePipe,
        memory: C220MatrixMemory,
    },
}
