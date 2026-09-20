pub mod acl_address_space;
pub mod acl_args;
pub mod addressed_replay;
pub mod architecture;
pub mod binary_alloc;
pub mod buffer_c310;
pub mod c220_masked_add;
pub mod c220_mul;
pub mod c220_scalar_address_space;
pub mod c220_sub;
pub mod c310_masked_add;
pub mod c310_mul;
pub mod c310_scalar_address_space;
pub mod c310_sub;
pub mod captured_replay;
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
pub mod issue_queue_c310;
pub mod kernel_config;
pub mod kernel_record;
pub mod machine;
pub mod mte_c220;
pub mod mte_c310;
pub mod mte_stepper;
pub mod plan;
pub mod predicate_buffer_bus_c310;
pub mod predicate_buffer_c310;
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
pub mod rvec_address_c310;
pub mod rvec_loop_c310;
pub mod rvec_pb_c310;
pub mod rvec_program_c310;
pub mod scalar;
pub mod stepper;
pub mod trace;
pub mod ub_replay;
pub mod vec_c220;
pub mod vec_queue_c310;
pub mod workspace;

pub use buffer_c310::{
    C310BufferCounter, C310BufferCounterError, C310BufferCounters, C310BufferDisposition,
    C310GetBufDispatch, C310GetBufGateResult, C310GetBufIssueTick, C310GetBufIssueTickResult,
    evaluate_c310_get_buf_gate,
};
pub use c220_scalar_address_space::{C220ScalarAddressSpace, C220ScalarAddressSpaceError};
pub use c310_scalar_address_space::{C310ScalarAddressSpace, C310ScalarAddressSpaceError};
pub use issue_queue_c310::{
    C310DequeueOutcome, C310IssueQueue, C310IssueQueueError, C310IssueQueueSnapshot,
    C310IssueQueueTransition,
};
pub use predicate_buffer_bus_c310::{
    C310PredicateBufferBus, C310PredicateBufferBusError, C310PredicateBufferCompletedVfIssue,
    C310PredicateBufferQueuedVfIssue,
};
pub use predicate_buffer_c310::{
    C310_PB_DEFAULT_SLOTS, C310_PB_HALFWORDS_PER_SLOT, C310_PB_PUSH_BYTES, C310_PB_SLOT_BYTES,
    C310PredicateBuffer, C310PredicateBufferError, C310PredicateBufferVfIssue,
    C310PredicateBufferWrite, C310PushPbDisposition, C310PushPbInstruction, C310PushPbStep,
};
pub use rvec_pb_c310::{
    C310PbRvecScalarProjection, C310RvecScalarWrite, project_c310_pb_rvec_scalar_init,
};
pub use rvec_program_c310::{
    C310CapturedRvecEffect, C310CapturedRvecProgram, C310CapturedRvecProgramError,
    C310CapturedRvecProgramStep,
};
pub use vec_queue_c310::{
    C310RvecAdmissionCounters, C310RvecAdmissionError, C310RvecAdmissionStep, C310SimdGateBlockers,
    C310VfQueueDisposition, C310VfQueueInstruction, C310VfQueueStep,
};

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
pub use c220_mul::{
    C220_CAPTURED_MUL_BYTES, C220_CAPTURED_MUL_TILE_BYTES, C220_CAPTURED_MUL_TILES,
    C220_CAPTURED_MUL_VMUL_PC, C220CapturedMulError, C220CapturedMulRun, C220CapturedMulTile,
    execute_captured_c220_mul, execute_captured_c220_mul_predecessor_chains,
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
pub use c310_mul::{
    C310_CAPTURED_MUL_BYTES, C310_CAPTURED_MUL_MTE2_X_WORD, C310_CAPTURED_MUL_MTE2_Y_WORD,
    C310_CAPTURED_MUL_MTE3_WORD, C310_CAPTURED_MUL_PB_SOURCE_VALUES,
    C310_CAPTURED_MUL_PUSH_PB_WORD, C310_CAPTURED_MUL_TILE_BYTES, C310_CAPTURED_MUL_TILES,
    C310_CAPTURED_MUL_VMUL_WORD, C310_CAPTURED_MUL_VST_WORD, C310CapturedMulError,
    C310CapturedMulInputChunk, C310CapturedMulOutputChunk, C310CapturedMulRun, C310CapturedMulTile,
    c310_captured_mul_predicate_buffer_slot, execute_captured_c310_mul,
    execute_captured_c310_mul_predecessor_chains,
};
pub use c310_sub::{
    C310_CAPTURED_SUB_BYTES, C310_CAPTURED_SUB_MTE2_X_WORD, C310_CAPTURED_SUB_MTE2_Y_WORD,
    C310_CAPTURED_SUB_MTE3_WORD, C310_CAPTURED_SUB_TILE_BYTES, C310_CAPTURED_SUB_TILES,
    C310_CAPTURED_SUB_VST_WORD, C310_CAPTURED_SUB_VSUB_WORD, C310CapturedSubError,
    C310CapturedSubInputChunk, C310CapturedSubOutputChunk, C310CapturedSubRun, C310CapturedSubTile,
    execute_captured_c310_sub, execute_captured_c310_sub_predecessor_chains,
};
pub use captured_replay::{
    CapturedReplay, CapturedReplayError, CapturedReplayOperation, execute_captured_replay,
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
    BufferEncoding, BufferIdSource, BufferOperation, C310BufferInstruction, C310BufferStep,
    ConditionalJump, ConditionalJumpTarget, DcciInstruction, DcciStep, DsbStep, FlagIdSource,
    FlagInstruction, FlagOperation, FlagStep, FlowEnd, FlowNop, JumpCompare, JumpCompareError,
    JumpCompareOffset, JumpCompareOperand, JumpCompareTarget, JumpOffsetSource, JumpTarget,
    PipelineBarrierScope, PipelineBarrierStep, UnconditionalJump,
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
    ScalarIndexedImmediateStoreStep, ScalarIndexedLoadStep, ScalarInstructionError,
    ScalarInstructionStep, ScalarMachine, ScalarMachineError, ScalarMemoryBus,
    ScalarMemoryExecutionError, ScalarMemoryStep, ScalarPairLoadStep, ScalarPairStoreStep,
    ScalarSelectStep, ScalarSprReadSource, ScalarSprReadStep, ScalarSprStep, ScalarStep,
};
pub use mte_c220::{
    C220_MOV_UB_TO_OUT_UNIT_BYTES, C220DmaMovDescriptor, C220DmaMovError, C220DmaMovSegment,
    CAPTURED_C220_MOV_UB_TO_OUT_WORD, CAPTURED_C220_SUB_MOV_OUT_TO_UB_X_WORD,
    CAPTURED_C220_SUB_MOV_OUT_TO_UB_Y_WORD, CAPTURED_C220_SUB_MOV_UB_TO_OUT_WORD,
    CAPTURED_C220_SUB_TILING_MOV_OUT_TO_UB_WORD, CAPTURED_C220_TILING_MOV_OUT_TO_UB_WORD,
    MAX_C220_DMAMOV_SEGMENTS,
};
pub use mte_c310::{
    C310_ADD_MOV_ALIGN_X_WORD, C310_ADD_MOV_ALIGN_Y_WORD, C310_SUB_TILING_MOV_ALIGN_WORD,
    C310_TILING_MOV_ALIGN_WORD, C310CapturedMovAlignDecode, C310CapturedMovAlignError,
    C310CapturedMovAlignRegisters, C310MovAlignCoordinate, C310MovAlignCoordinateError,
    C310MovAlignParameters, C310MovAlignRegisterSelectors, C310TilingMovAlignRegisters,
    MAX_C310_MOV_ALIGN_COORDINATES,
};
pub use mte_stepper::{
    C220_MTE3_TO_VECTOR_SET_FLAG_WORD, C220_MTE3_TO_VECTOR_WAIT_FLAG_WORD,
    C220_SUB_MTE2_TO_VECTOR_SET_FLAG0_WORD, C220_SUB_MTE2_TO_VECTOR_SET_FLAG1_WORD,
    C220_SUB_MTE2_TO_VECTOR_WAIT_FLAG0_WORD, C220_SUB_MTE3_TO_VECTOR_SET_FLAG_WORD,
    C220_SUB_MTE3_TO_VECTOR_WAIT_FLAG_WORD, C220_SUB_VECTOR_TO_MTE2_SET_FLAG_WORD,
    C220_SUB_VECTOR_TO_MTE2_WAIT_FLAG0_WORD, C220_SUB_VECTOR_TO_MTE2_WAIT_FLAG1_WORD,
    C220_SUB_VECTOR_TO_MTE3_SET_FLAG_WORD, C220_SUB_VECTOR_TO_MTE3_WAIT_FLAG_WORD,
    C220_VECTOR_TO_MTE2_SET_FLAG_WORD, C220_VECTOR_TO_MTE2_WAIT_DYNAMIC_WORD,
    C220_VECTOR_TO_MTE2_WAIT_FLAG0_WORD, C220_VECTOR_TO_MTE2_WAIT_FLAG1_WORD,
    C220_VECTOR_TO_MTE3_SET_FLAG_WORD, C220_VECTOR_TO_MTE3_WAIT_FLAG_WORD, C220OutputAction,
    C220OutputStep, MAX_PENDING_MTE2_TRANSFERS, MTE2_TO_SCALAR_SET_FLAG0_WORD,
    MTE2_TO_SCALAR_WAIT_FLAG0_WORD, MTE2_TO_VECTOR_SET_FLAG0_WORD, MTE2_TO_VECTOR_SET_FLAG1_WORD,
    MTE2_TO_VECTOR_WAIT_FLAG0_WORD, MTE2_TO_VECTOR_WAIT_FLAG1_WORD, MteAction, MteCoreStepper,
    MteProgramStep, MteStepperError, SCALAR_UB_ALIAS_BASE, SCALAR_UB_ALIAS_BYTES, UbScalarBusError,
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
    DATA_PATH_KERNEL_NAME_FIELD_BYTES, DATA_PATH_REQUEST_PAYLOAD_BYTES, LOG_TRANSLATE_ACK_BYTES,
    LOG_TRANSLATE_KERNEL_NAME_FIELD_BYTES, LOG_TRANSLATE_OUTPUT_PATH_FIELD_BYTES,
    MODEL_CONFIG_RESPONSE_BYTES, PROF_STUB_PACKET_HEADER_BYTES,
    PROF_STUB_RESPONSE_MAX_WAIT_ATTEMPTS, PROF_STUB_RESPONSE_READ_BYTES, ProfStubPacket,
    ProfStubPacketError, ProfStubReply, decode_data_path_request, decode_log_translate_start,
    encode_data_path_request, encode_log_translate_start, encode_log_translate_stop,
    encode_model_config_request, max_prof_stub_payload_bytes,
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
    C310_CAPTURED_PLT32_WORD, C310_CAPTURED_PSET_WORD, C310_CAPTURED_SMOVI32_WORD,
    C310_CAPTURED_VDUPS_WORD, C310_CAPTURED_VLD_V0_WORD, C310_CAPTURED_VLD_V1_WORD,
    C310_CAPTURED_VLDI_V0_WORD, C310_CAPTURED_VLDI_V1_WORD, C310_CAPTURED_VST_WORD,
    C310_MASK0_SPR_INDEX, C310_MASK1_SPR_INDEX, C310CapturedPltError, C310CapturedPltStep,
    C310CapturedPsetError, C310CapturedPsetStep, C310CapturedSmoviError, C310CapturedSmoviStep,
    C310CapturedVdupsStep, C310CapturedVectorError, C310CapturedVectorLoadAddress,
    C310CapturedVectorLoadError, C310CapturedVectorLoadHint, C310CapturedVectorLoadStep,
    C310CapturedVldiError, C310CapturedVldiHint, C310CapturedVldiStep, C310CapturedVstStep,
    C310CapturedVstStore, C310ObservedMovemaskError, C310ObservedMovemaskHint,
    C310ObservedMovemaskStep, C310RvecArithmeticHint, C310RvecArithmeticOperation,
    C310RvecMaskSprState, C310RvecMovpHint, C310RvecMovpStep, C310RvecValueError,
    C310RvecValueMachine, C310RvecValueStep, C310RvecVstiHint, C310RvecVstiStore,
    c310_movp_u32_mask_to_predicate_bytes, c310_normal_u32_masked_store,
    c310_predicate_bytes_to_mask,
};
pub use rvec_address_c310::{
    C310_CAPTURED_VAG_WORD, C310CapturedA0Step, C310CapturedVagDescriptor, C310RvecAddressError,
    C310RvecAddressState,
};
pub use rvec_loop_c310::{
    C310_CAPTURED_LONG_VLOOP_WORD, C310_CAPTURED_SHORT_VLOOP_WORD, C310CapturedLoopError,
    C310CapturedLoopHeader, C310CapturedLoopPcStep, C310RvecLoopController,
};
pub use scalar::{ScalarIntegerError, ScalarIntegerOutcome, evaluate_scalar_integer_immediate};
pub use stepper::{
    ScalarProgramRun, ScalarProgramStep, ScalarProgramStop, ScalarStepper, ScalarStepperError,
};
pub use trace::{ScalarTraceMismatch, ScalarTraceSummary, verify_scalar_trace};
pub use ub_replay::{UbReplayError, UbReplayMemory, UbTransferResult};
pub use vec_c220::{
    C220_CAPTURED_MOVEV_CONTROL, C220_CAPTURED_MOVEV_WORD, C220_CAPTURED_VADD_CONTROL,
    C220_CAPTURED_VADD_WORD, C220_CAPTURED_VMUL_CONTROL, C220_CAPTURED_VMUL_WORD,
    C220_CAPTURED_VSUB_WORD, C220CapturedFp32Step, C220CapturedMovevStep, C220CapturedVectorError,
    C220CapturedVectorStore, C220MovemaskHint, C220VecArithmeticHint, C220VecArithmeticOperation,
    decode_captured_c220_fp32_mask, execute_captured_c220_fp32_to_ub,
    execute_captured_c220_movev_to_ub,
};
pub use workspace::{PreparedRun, WorkspaceOptions};
