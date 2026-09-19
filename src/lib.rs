pub mod acl_args;
pub mod architecture;
pub mod binary_alloc;
pub mod cli;
pub mod device_elf;
pub mod hbm;
pub mod hbm_pv_memory;
pub mod ipc;
pub mod isa;
pub mod kernel_config;
pub mod kernel_record;
pub mod machine;
pub mod plan;
pub mod pv_memory;
pub mod replay_memory;
pub mod replay_seed;
pub mod runner;
pub mod scalar;
pub mod trace;
pub mod workspace;

pub use acl_args::{AclArgKind, AclArgSlot, AclArgumentPlan, AclArgumentPlanError};
pub use architecture::{Architecture, DeviceProfile, ResolvedTarget};
pub use binary_alloc::{
    BINARY_ALIGNMENT_BYTES, BINARY_FEATURE_42_EXTRA_BYTES, BINARY_POOL_BYTES,
    BinaryAllocationPlanError, BinaryDeviceAllocationPlan,
};
pub use device_elf::{
    DeviceElf, DeviceElfError, DeviceElfHeader, DeviceKernel, DeviceKernelSummary, DeviceLoadImage,
    DeviceLoadImageSummary, ProjectedDeviceKernel,
};
pub use hbm::{
    C310_DRIVER_ALLOCATION_ALIGNMENT, CAMODEL_HBM_BASE, CAMODEL_HBM_BYTES, CamodelHbmAllocator,
    HbmAllocationError, HbmResolveError, HbmResolvedSpan, HbmSpan,
};
pub use hbm_pv_memory::{HbmPvMemory, HbmPvMemoryError};
pub use ipc::{IpcMemoryName, IpcMemoryOperation, IpcMemoryRequest, IpcResponse, IpcWireError};
pub use isa::{
    AicClass, AicDecoderHint, AicFramingError, AicInstructionWord, AicWordFramer,
    ScalarAddressEffect, ScalarDirectBiuRoute, ScalarKey0Operation, ScalarKey7Operation,
    ScalarKey8Operation, ScalarLoadStoreOperation,
};
pub use kernel_config::{
    DecodedKernelConfig, KernelConfigDocument, ReplayRunner, ReplayTilingData,
};
pub use kernel_record::{KernelRecordRequest, KernelRecordResponse, KernelRecordWireError};
pub use machine::{
    SCALAR_X_REGISTER_COUNT, ScalarMachine, ScalarMachineError, ScalarMemoryBus,
    ScalarMemoryExecutionError, ScalarMemoryStep, ScalarStep,
};
pub use plan::{LaunchPlan, SimulatorRequest};
pub use pv_memory::{PV_PAGE_BYTES, PvMemory, PvMemoryError};
pub use replay_memory::{MemoryByteState, ReplayMemory, ReplayMemoryError};
pub use replay_seed::{
    ReplaySeed, ReplaySeedError, ReplaySeedSummary, SeedArgument, SeedReadError, SeedRegion,
    SeedRegionSummary,
};
pub use runner::{ModelConfigLoad, RunOutcome};
pub use scalar::{ScalarIntegerError, ScalarIntegerOutcome, evaluate_scalar_integer_immediate};
pub use trace::{ScalarTraceMismatch, ScalarTraceSummary, verify_scalar_trace};
pub use workspace::{PreparedRun, WorkspaceOptions};
