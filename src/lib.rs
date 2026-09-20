pub mod acl_address_space;
pub mod acl_args;
pub mod addressed_replay;
pub mod architecture;
pub mod binary_alloc;
pub mod c220_masked_add;
pub mod c220_sub;
pub mod c310_masked_add;
pub mod c310_sub;
pub mod cli;
pub mod device_elf;
pub mod device_loader;
pub mod device_pool;
pub mod flow;
pub mod flow_trace;
pub mod fp32_vector;
pub mod hbm;
pub mod hbm_pv_memory;
pub mod ipc;
pub mod isa;
pub mod kernel_config;
pub mod kernel_record;
pub mod machine;
pub mod mte_c220;
pub mod mte_c310;
pub mod mte_stepper;
pub mod plan;
pub mod prof_stub_flow;
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
pub mod stepper;
pub mod trace;
pub mod ub_replay;
pub mod vec_c220;
pub mod workspace;

pub use acl_address_space::{
    AclAddressSpaceBuildError, AclAddressSpaceError, AclArgumentImage, AclArgumentImageError,
    AclArgumentImageSummary, AclReplayAddressSpace, MAX_ACL_ARGUMENT_IMAGE_BYTES,
};
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
pub use c220_masked_add::{
    C220_CAPTURED_MASKED_ADD_BYTES, C220_CAPTURED_MASKED_ADD_TILE_BYTES,
    C220_CAPTURED_MASKED_ADD_TILES, C220CapturedByteSpan, C220CapturedMaskedAddError,
    C220CapturedMaskedAddRun, C220CapturedMaskedAddTile, execute_captured_c220_masked_add,
};
pub use c220_sub::{
    C220_CAPTURED_SUB_BYTES, C220_CAPTURED_SUB_TILE_BYTES, C220_CAPTURED_SUB_TILES,
    C220_CAPTURED_SUB_VSUB_WORD, C220CapturedSubError, C220CapturedSubRun, C220CapturedSubTile,
    execute_captured_c220_sub, execute_captured_c220_sub_predecessor_chains,
};
pub use c310_masked_add::{
    C310_CAPTURED_MASKED_ADD_BYTES, C310_CAPTURED_MASKED_ADD_TILE_BYTES,
    C310_CAPTURED_MASKED_ADD_TILES, C310CapturedMaskedAddError, C310CapturedMaskedAddRun,
    C310CapturedMaskedAddTile, execute_captured_c310_masked_add,
};
pub use c310_sub::{
    C310_CAPTURED_SUB_BYTES, C310_CAPTURED_SUB_MTE2_X_WORD, C310_CAPTURED_SUB_MTE2_Y_WORD,
    C310_CAPTURED_SUB_MTE3_WORD, C310_CAPTURED_SUB_P1, C310_CAPTURED_SUB_TILE_BYTES,
    C310_CAPTURED_SUB_TILES, C310_CAPTURED_SUB_VST_WORD, C310_CAPTURED_SUB_VSUB_WORD,
    C310CapturedSubError, C310CapturedSubInputChunk, C310CapturedSubOutputChunk,
    C310CapturedSubRun, C310CapturedSubTile, execute_captured_c310_sub,
    execute_captured_c310_sub_predecessor_chains,
};
pub use device_elf::{
    DeviceElf, DeviceElfError, DeviceElfHeader, DeviceGlobalAddresses, DeviceGlobalPatchSite,
    DeviceGlobalSymbol, DeviceKernel, DeviceKernelSummary, DeviceLoadImage, DeviceLoadImageSummary,
    PreparedDeviceLoadImage, ProjectedDeviceKernel,
};
pub use device_loader::{
    DeviceBinaryAllocation, DeviceBinaryLoad, DeviceBinaryLoadError, DeviceBinaryPlacement,
    DeviceKernelFetchError, DeviceKernelLoadError, DevicePoolAttempt, InstructionFetchWindow,
    LoadedDeviceKernel, copy_prepared_image_at, load_direct_fallback, load_named_kernel,
    load_with_pool_preference,
};
pub use device_pool::{
    DeviceMemoryPoolManager, DevicePoolBlock, DevicePoolError, DevicePoolSummary,
};
pub use flow::{
    ConditionalJump, ConditionalJumpTarget, DcciInstruction, DcciStep, DsbStep, FlowEnd, FlowNop,
    JumpCompare, JumpCompareError, JumpCompareOffset, JumpCompareOperand, JumpCompareTarget,
    JumpOffsetSource, JumpTarget, UnconditionalJump,
};
pub use flow_trace::{JumpTraceIssue, JumpTraceSummary, verify_jump_trace};
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
    ScalarKey8Operation, ScalarLoadStoreOperation, ScalarStoreImmediateValue,
};
pub use kernel_config::{
    DecodedKernelConfig, KernelConfigDocument, ReplayRunner, ReplayTilingData,
};
pub use kernel_record::{KernelRecordRequest, KernelRecordResponse, KernelRecordWireError};
pub use machine::{
    C220MovemaskStep, SCALAR_X_REGISTER_COUNT, ScalarCacheHintStep, ScalarCompareImmediateStep,
    ScalarCompareRegisterStep, ScalarCompareStep, ScalarFlowStep, ScalarImmediateStoreStep,
    ScalarInstructionError, ScalarInstructionStep, ScalarMachine, ScalarMachineError,
    ScalarMemoryBus, ScalarMemoryExecutionError, ScalarMemoryStep, ScalarPairLoadStep,
    ScalarPairStoreStep, ScalarSelectStep, ScalarSprReadSource, ScalarSprReadStep, ScalarSprStep,
    ScalarStep,
};
pub use mte_c220::{
    C220_MOV_UB_TO_OUT_UNIT_BYTES, C220DmaMovDescriptor, C220DmaMovError, C220DmaMovSegment,
    CAPTURED_C220_MOV_UB_TO_OUT_WORD, CAPTURED_C220_SUB_MOV_OUT_TO_UB_X_WORD,
    CAPTURED_C220_SUB_MOV_OUT_TO_UB_Y_WORD, CAPTURED_C220_SUB_MOV_UB_TO_OUT_WORD,
    CAPTURED_C220_TILING_MOV_OUT_TO_UB_WORD, MAX_C220_DMAMOV_SEGMENTS,
};
pub use mte_c310::{
    C310_ADD_MOV_ALIGN_X_WORD, C310_ADD_MOV_ALIGN_Y_WORD, C310_TILING_MOV_ALIGN_WORD,
    C310CapturedMovAlignDecode, C310CapturedMovAlignError, C310CapturedMovAlignRegisters,
    C310MovAlignCoordinate, C310MovAlignCoordinateError, C310MovAlignParameters,
    C310MovAlignRegisterSelectors, C310TilingMovAlignRegisters, MAX_C310_MOV_ALIGN_COORDINATES,
};
pub use mte_stepper::{
    MAX_PENDING_MTE2_TRANSFERS, MTE2_TO_SCALAR_SET_FLAG0_WORD, MTE2_TO_SCALAR_WAIT_FLAG0_WORD,
    MteAction, MteCoreStepper, MteProgramStep, MteStepperError,
};
pub use plan::{LaunchPlan, SimulatorRequest};
pub use prof_stub_flow::{
    ProfStubBranchEdgeIssue, ProfStubBranchEdgeIssueReason, ProfStubBranchEdgeSummary,
    ProfStubBranchKind, inspect_prof_stub_branch_edges,
};
pub use prof_stub_object_verify::{
    ProfStubObjectIssue, ProfStubObjectIssueReason, ProfStubObjectVerification,
    verify_prof_stub_loaded_kernel, verify_prof_stub_object,
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
    C310_CAPTURED_PLT32_WORD, C310_CAPTURED_VDUPS_WORD, C310_CAPTURED_VLD_V0_WORD,
    C310_CAPTURED_VLD_V1_WORD, C310_CAPTURED_VLDI_V0_WORD, C310_CAPTURED_VLDI_V1_WORD,
    C310_CAPTURED_VST_WORD, C310_MASK0_SPR_INDEX, C310_MASK1_SPR_INDEX, C310CapturedPltError,
    C310CapturedPltStep, C310CapturedVdupsStep, C310CapturedVectorError,
    C310CapturedVectorLoadError, C310CapturedVectorLoadHint, C310CapturedVectorLoadStep,
    C310CapturedVldiError, C310CapturedVldiHint, C310CapturedVldiStep, C310CapturedVstStep,
    C310CapturedVstStore, C310ObservedMovemaskError, C310ObservedMovemaskHint,
    C310ObservedMovemaskStep, C310RvecArithmeticHint, C310RvecArithmeticOperation,
    C310RvecMaskSprState, C310RvecMovpHint, C310RvecMovpStep, C310RvecValueError,
    C310RvecValueMachine, C310RvecValueStep, C310RvecVstiHint, C310RvecVstiStore,
    c310_movp_u32_mask_to_predicate_bytes, c310_normal_u32_masked_store,
    c310_predicate_bytes_to_mask,
};
pub use scalar::{ScalarIntegerError, ScalarIntegerOutcome, evaluate_scalar_integer_immediate};
pub use stepper::{ScalarProgramStep, ScalarStepper, ScalarStepperError};
pub use trace::{ScalarTraceMismatch, ScalarTraceSummary, verify_scalar_trace};
pub use ub_replay::{UbReplayError, UbReplayMemory, UbTransferResult};
pub use vec_c220::{C220MovemaskHint, C220VecArithmeticHint, C220VecArithmeticOperation};
pub use workspace::{PreparedRun, WorkspaceOptions};
