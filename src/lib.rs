pub mod acl_args;
pub mod addressed_replay;
pub mod architecture;
pub mod binary_alloc;
pub mod cli;
pub mod device_elf;
pub mod fp32_vector;
pub mod hbm;
pub mod hbm_pv_memory;
pub mod ipc;
pub mod isa;
pub mod kernel_config;
pub mod kernel_record;
pub mod machine;
pub mod plan;
pub mod prof_stub_object_verify;
pub mod prof_stub_packet;
pub mod prof_stub_stream;
pub mod prof_stub_trace;
pub mod pv_memory;
pub mod replay_memory;
pub mod replay_seed;
pub mod runner;
pub mod rvec;
pub mod scalar;
pub mod trace;
pub mod vec_c220;
pub mod workspace;

pub use acl_args::{AclArgKind, AclArgSlot, AclArgumentPlan, AclArgumentPlanError};
pub use addressed_replay::{
    AddressedReplayMemory, ReplayAddressBinding, ReplayAddressBindingError, ReplayAddressError,
    ResolvedReplayAddress,
};
pub use architecture::{Architecture, DeviceProfile, ResolvedTarget};
pub use binary_alloc::{
    BINARY_ALIGNMENT_BYTES, BINARY_FEATURE_42_EXTRA_BYTES, BINARY_POOL_BYTES,
    BinaryAllocationPlanError, BinaryDeviceAllocationPlan,
};
pub use device_elf::{
    DeviceElf, DeviceElfError, DeviceElfHeader, DeviceKernel, DeviceKernelSummary, DeviceLoadImage,
    DeviceLoadImageSummary, ProjectedDeviceKernel,
};
pub use fp32_vector::{
    Fp32LaneOutcome, Fp32MaskLayout, Fp32ValueOutcome, Fp32ValueStatus, Fp32VectorError,
    Fp32VectorOperation, Fp32WritebackOutcome, Fp32WritebackPolicy, apply_fp32_writeback,
    evaluate_fp32_value, evaluate_masked_fp32_lanes,
};
pub use hbm::{
    CAMODEL_DRIVER_ALLOCATION_ALIGNMENT, CAMODEL_HBM_BASE, CAMODEL_HBM_BYTES, CamodelHbmAllocator,
    DriverMemStatus, HbmAllocationError, HbmResolveError, HbmResolvedSpan, HbmSpan,
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
    ScalarMemoryExecutionError, ScalarMemoryStep, ScalarSprStep, ScalarStep,
};
pub use plan::{LaunchPlan, SimulatorRequest};
pub use prof_stub_object_verify::{
    ProfStubObjectIssue, ProfStubObjectIssueReason, ProfStubObjectVerification,
    verify_prof_stub_object,
};
pub use prof_stub_packet::{
    LOG_TRANSLATE_ACK_BYTES, LOG_TRANSLATE_KERNEL_NAME_FIELD_BYTES,
    LOG_TRANSLATE_OUTPUT_PATH_FIELD_BYTES, PROF_STUB_PACKET_HEADER_BYTES, ProfStubPacket,
    ProfStubPacketError, decode_log_translate_start, encode_log_translate_start,
    encode_log_translate_stop, max_prof_stub_payload_bytes,
};
pub use prof_stub_stream::{
    ProfStubRecordDetail, ProfStubRecordSummary, ProfStubStreamSummary, inspect_prof_stub_stream,
};
pub use prof_stub_trace::{
    ICACHE_LOG_PAYLOAD_BYTES, ICacheLog, INSTRUCTION_LOG_PAYLOAD_BYTES,
    INSTRUCTION_LOG_TEXT_FIELD_BYTES, InstructionLog, MTE_LOG_PAYLOAD_BYTES, ProfStubCoreKind,
    ProfStubTraceLog,
};
pub use pv_memory::{PV_PAGE_BYTES, PvMemory, PvMemoryError};
pub use replay_memory::{MemoryByteState, ReplayMemory, ReplayMemoryError};
pub use replay_seed::{
    ReplaySeed, ReplaySeedError, ReplaySeedSummary, SeedArgument, SeedReadError, SeedRegion,
    SeedRegionSummary,
};
pub use runner::{ModelConfigLoad, RunOutcome};
pub use rvec::{
    C310_MASK0_SPR_INDEX, C310_MASK1_SPR_INDEX, C310ObservedMovemaskError,
    C310ObservedMovemaskHint, C310ObservedMovemaskStep, C310RvecArithmeticHint,
    C310RvecArithmeticOperation, C310RvecMaskSprState, C310RvecMovpHint, C310RvecMovpStep,
    C310RvecValueError, C310RvecValueMachine, C310RvecValueStep, C310RvecVstiHint,
    C310RvecVstiStore, c310_movp_u32_mask_to_predicate_bytes, c310_normal_u32_masked_store,
    c310_predicate_bytes_to_mask,
};
pub use scalar::{ScalarIntegerError, ScalarIntegerOutcome, evaluate_scalar_integer_immediate};
pub use trace::{ScalarTraceMismatch, ScalarTraceSummary, verify_scalar_trace};
pub use vec_c220::{C220VecArithmeticHint, C220VecArithmeticOperation};
pub use workspace::{PreparedRun, WorkspaceOptions};
