use serde::Serialize;
use thiserror::Error;

use crate::acl_address_space::AclReplayAddressSpace;
use crate::architecture::Architecture;
use crate::buffer_c310::C310BufferDisposition;
use crate::flow::{
    C310BufferStep, DcciStep, DsbStep, FlagInstruction, FlagOperation, PipelineBarrierStep,
};
use crate::machine::{ScalarInstructionError, ScalarMemoryBus};
use crate::mte_c220::{
    C220DmaMovDescriptor, C220MovOutToUbDescriptor, C220MovOutToUbError,
    CAPTURED_C220_MOV_OUT_TO_UB_X_WORD, CAPTURED_C220_MOV_OUT_TO_UB_Y_WORD,
    CAPTURED_C220_MOV_UB_TO_OUT_WORD, CAPTURED_C220_SUB_MOV_OUT_TO_UB_X_WORD,
    CAPTURED_C220_SUB_MOV_OUT_TO_UB_Y_WORD, CAPTURED_C220_SUB_MOV_UB_TO_OUT_WORD,
    CAPTURED_C220_SUB_TILING_MOV_OUT_TO_UB_WORD, CAPTURED_C220_TILING_MOV_OUT_TO_UB_WORD,
};
use crate::mte_c310::{
    C310_ADD_MOV_ALIGN_X_WORD, C310_ADD_MOV_ALIGN_Y_WORD, C310_TILING_MOV_ALIGN_WORD,
    C310CapturedMovAlignDecode, C310CapturedMovAlignError, C310MovAlignRegisterSelectors,
    C310TilingMovAlignRegisters,
};
use crate::predicate_buffer_c310::{C310PushPbDisposition, C310PushPbStep};
use crate::replay_memory::MemoryByteState;
use crate::stepper::{ScalarProgramStep, ScalarStepper};
use crate::ub_replay::{UbReplayError, UbReplayMemory, UbTransferResult};
use crate::vec_c220::{
    C220_CAPTURED_MOVEV_CONTROL, C220_CAPTURED_MOVEV_WORD, C220_CAPTURED_VADD_CONTROL,
    C220_CAPTURED_VADD_WORD, C220_CAPTURED_VSUB_WORD, C220CapturedFp32Step, C220CapturedMovevStep,
    C220CapturedVectorError, C220VecArithmeticHint, decode_captured_c220_fp32_mask,
    execute_captured_c220_fp32_to_ub, execute_captured_c220_movev_to_ub,
};
use crate::vec_queue_c310::{C310VfQueueDisposition, C310VfQueueStep};

pub const MTE2_TO_SCALAR_SET_FLAG0_WORD: u32 = 0x40a0_1000;
pub const MTE2_TO_SCALAR_WAIT_FLAG0_WORD: u32 = 0x40c0_1000;
pub const MTE2_TO_VECTOR_SET_FLAG0_WORD: u32 = 0x40a2_10b8;
pub const C220_SUB_MTE2_TO_VECTOR_SET_FLAG0_WORD: u32 = 0x40a2_10b0;
pub const MTE2_TO_VECTOR_SET_FLAG1_WORD: u32 = 0x40a2_10b4;
pub const C220_SUB_MTE2_TO_VECTOR_SET_FLAG1_WORD: u32 = 0x40a2_10ac;
pub const MTE2_TO_VECTOR_WAIT_FLAG0_WORD: u32 = 0x40c2_10b0;
pub const C220_SUB_MTE2_TO_VECTOR_WAIT_FLAG0_WORD: u32 = 0x40c2_10a8;
pub const MTE2_TO_VECTOR_WAIT_FLAG1_WORD: u32 = 0x40c2_10b8;
pub const C220_VECTOR_TO_MTE3_SET_FLAG_WORD: u32 = 0x40a2_06b8;
pub const C220_SUB_VECTOR_TO_MTE3_SET_FLAG_WORD: u32 = 0x40a2_06b0;
pub const C220_VECTOR_TO_MTE3_WAIT_FLAG_WORD: u32 = 0x40c2_06b4;
pub const C220_SUB_VECTOR_TO_MTE3_WAIT_FLAG_WORD: u32 = 0x40c2_06ac;
pub const C220_VECTOR_TO_MTE2_SET_FLAG_WORD: u32 = 0x40a2_0630;
pub const C220_SUB_VECTOR_TO_MTE2_SET_FLAG_WORD: u32 = 0x40a2_0628;
pub const C220_VECTOR_TO_MTE2_WAIT_FLAG0_WORD: u32 = 0x40c2_0628;
pub const C220_SUB_VECTOR_TO_MTE2_WAIT_FLAG0_WORD: u32 = 0x40c2_0620;
pub const C220_VECTOR_TO_MTE2_WAIT_FLAG1_WORD: u32 = 0x40c2_0648;
pub const C220_SUB_VECTOR_TO_MTE2_WAIT_FLAG1_WORD: u32 = 0x40c2_0640;
pub const C220_VECTOR_TO_MTE2_WAIT_DYNAMIC_WORD: u32 = 0x40c2_0608;
pub const C220_MTE3_TO_VECTOR_SET_FLAG_WORD: u32 = 0x40a2_14a8;
pub const C220_SUB_MTE3_TO_VECTOR_SET_FLAG_WORD: u32 = 0x40a2_14a0;
pub const C220_MTE3_TO_VECTOR_WAIT_FLAG_WORD: u32 = 0x40c2_14cc;
pub const C220_SUB_MTE3_TO_VECTOR_WAIT_FLAG_WORD: u32 = 0x40c2_14c4;
pub const MAX_PENDING_MTE2_TRANSFERS: usize = 64;
pub const SCALAR_UB_ALIAS_BASE: u64 = 0x80000;
pub const SCALAR_UB_ALIAS_BYTES: u64 = 0x80000;

#[derive(Debug, Error)]
pub enum UbScalarBusError<E: std::error::Error + 'static> {
    #[error("scalar address range overflows u64")]
    AddressOverflow,
    #[error(
        "scalar address range at {address:#x} with {bytes} bytes crosses the UB alias boundary"
    )]
    AliasBoundary { address: u64, bytes: usize },
    #[error(transparent)]
    Ub(#[from] UbReplayError),
    #[error("scalar memory backend: {0}")]
    Fallback(#[source] E),
}

struct UbScalarBus<'a, B> {
    ub: &'a mut UbReplayMemory,
    fallback: &'a mut B,
}

impl<B: ScalarMemoryBus> UbScalarBus<'_, B> {
    fn alias_offset(
        &self,
        address: u64,
        bytes: usize,
    ) -> Result<Option<u64>, UbScalarBusError<B::Error>> {
        let length = u64::try_from(bytes).map_err(|_| UbScalarBusError::AddressOverflow)?;
        let end = address
            .checked_add(length)
            .ok_or(UbScalarBusError::AddressOverflow)?;
        let alias_end = SCALAR_UB_ALIAS_BASE + SCALAR_UB_ALIAS_BYTES;
        if address < SCALAR_UB_ALIAS_BASE && end > SCALAR_UB_ALIAS_BASE
            || address < alias_end && end > alias_end
        {
            return Err(UbScalarBusError::AliasBoundary { address, bytes });
        }
        Ok((SCALAR_UB_ALIAS_BASE..alias_end)
            .contains(&address)
            .then_some(address - SCALAR_UB_ALIAS_BASE))
    }
}

impl<B: ScalarMemoryBus> ScalarMemoryBus for UbScalarBus<'_, B> {
    type Error = UbScalarBusError<B::Error>;

    fn read(&mut self, address: u64, destination: &mut [u8]) -> Result<(), Self::Error> {
        if let Some(offset) = self.alias_offset(address, destination.len())? {
            destination.copy_from_slice(&self.ub.read_known(offset, destination.len())?);
            Ok(())
        } else {
            self.fallback
                .read(address, destination)
                .map_err(UbScalarBusError::Fallback)
        }
    }

    fn write(&mut self, address: u64, source: &[u8]) -> Result<(), Self::Error> {
        if let Some(offset) = self.alias_offset(address, source.len())? {
            let states = source
                .iter()
                .copied()
                .map(MemoryByteState::Known)
                .collect::<Vec<_>>();
            self.ub.write_states(offset, &states)?;
            Ok(())
        } else {
            self.fallback
                .write(address, source)
                .map_err(UbScalarBusError::Fallback)
        }
    }

    fn maintain_data_cache(&mut self, step: DcciStep) -> Result<bool, Self::Error> {
        self.fallback
            .maintain_data_cache(step)
            .map_err(UbScalarBusError::Fallback)
    }

    fn synchronize_pipeline(&mut self, step: DsbStep) -> Result<bool, Self::Error> {
        self.fallback
            .synchronize_pipeline(step)
            .map_err(UbScalarBusError::Fallback)
    }

    fn synchronize_barrier(&mut self, step: PipelineBarrierStep) -> Result<bool, Self::Error> {
        self.fallback
            .synchronize_barrier(step)
            .map_err(UbScalarBusError::Fallback)
    }

    fn execute_c310_buffer(
        &mut self,
        step: C310BufferStep,
    ) -> Result<C310BufferDisposition, Self::Error> {
        self.fallback
            .execute_c310_buffer(step)
            .map_err(UbScalarBusError::Fallback)
    }

    fn execute_c310_push_pb(
        &mut self,
        step: C310PushPbStep,
    ) -> Result<C310PushPbDisposition, Self::Error> {
        self.fallback
            .execute_c310_push_pb(step)
            .map_err(UbScalarBusError::Fallback)
    }

    fn enqueue_c310_vf(
        &mut self,
        step: C310VfQueueStep,
    ) -> Result<C310VfQueueDisposition, Self::Error> {
        self.fallback
            .enqueue_c310_vf(step)
            .map_err(UbScalarBusError::Fallback)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PendingMte2 {
    C220 {
        descriptor: C220MovOutToUbDescriptor,
        source_address: u64,
        destination_address: u64,
    },
    C310(C310CapturedMovAlignDecode),
}

impl PendingMte2 {
    fn commit(
        self,
        ub: &mut UbReplayMemory,
        source: &AclReplayAddressSpace,
    ) -> Result<UbTransferResult, UbReplayError> {
        match self {
            Self::C220 {
                descriptor,
                source_address,
                destination_address,
            } => {
                ub.copy_c220_mov_out_to_ub(source, descriptor, source_address, destination_address)
            }
            Self::C310(decoded) => ub.copy_c310_mov_align_hbm_to_ub(source, decoded),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MteCoreStepper {
    scalar: ScalarStepper,
    ub: UbReplayMemory,
    pending_mte2: Vec<PendingMte2>,
    flag0_set: bool,
    vector_flags: [Option<PendingMte2>; 2],
    unsignaled_output: Option<C220OutputTile>,
    mte3_flags: [Option<C220OutputTile>; 4],
    mte2_reuse_flags: [bool; 2],
    ready_output: Option<C220OutputTile>,
    copied_output: Option<C220OutputTile>,
    mte3_completion_flags: [Option<C220OutputTile>; 4],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct C220OutputTile {
    source_address: u64,
    bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum C220OutputAction {
    SetMte3Flag {
        flag_id: u8,
        source_address: u64,
    },
    SetMte2ReuseFlag {
        flag_id: u8,
    },
    WaitMte2ReuseFlag {
        flag_id: u8,
    },
    WaitMte3Flag {
        flag_id: u8,
        source_address: u64,
    },
    CopyToHbm {
        source_address: u64,
        destination_address: u64,
        transfer: UbTransferResult,
    },
    SetMte3CompletionFlag {
        flag_id: u8,
        source_address: u64,
    },
    WaitMte3CompletionFlag {
        flag_id: u8,
        source_address: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct C220OutputStep {
    pub pc: u64,
    pub word: u32,
    pub next_pc: u64,
    pub action: C220OutputAction,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum MteAction {
    Issue {
        source_address: u64,
        destination_address: u64,
        planned_bytes: usize,
        pending_count: usize,
    },
    SetFlag {
        pending_count: usize,
    },
    WaitFlag {
        transfers: Vec<UbTransferResult>,
    },
    SetVectorFlag {
        flag_id: u8,
        remaining_pending: usize,
    },
    WaitVectorFlag {
        flag_id: u8,
        transfer: UbTransferResult,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MteProgramStep {
    pub pc: u64,
    pub word: u32,
    pub next_pc: u64,
    pub action: MteAction,
}

#[derive(Debug, Error)]
pub enum MteStepperError {
    #[error("program ended before MTE word at PC {pc:#x}")]
    ProgramEnded { pc: u64 },
    #[error("word {word:#010x} at PC {pc:#x} is not an implemented MTE2 or flag operation")]
    UnsupportedWord { pc: u64, word: u32 },
    #[error("SPR {index} is unavailable for MTE word at PC {pc:#x}")]
    MissingSpr { pc: u64, index: u16 },
    #[error("MTE2 pending count exceeds {MAX_PENDING_MTE2_TRANSFERS}")]
    PendingLimit,
    #[error("MTE2-to-Scalar flag 0 has no pending transfer")]
    SetWithoutTransfer,
    #[error("MTE2-to-Scalar flag 0 is already set")]
    FlagAlreadySet,
    #[error("MTE2-to-Scalar flag 0 was not set before wait")]
    WaitWithoutFlag,
    #[error("MTE2-to-Vector flag {flag_id} has no unassigned transfer")]
    VectorSetWithoutTransfer { flag_id: u8 },
    #[error("MTE2-to-Vector flag {flag_id} is already set")]
    VectorFlagAlreadySet { flag_id: u8 },
    #[error("MTE2-to-Vector flag {flag_id} was not set before wait")]
    VectorWaitWithoutFlag { flag_id: u8 },
    #[error("MTE2-to-Vector flag ID {flag_id} is outside the modeled 0..1 range")]
    UnsupportedVectorFlagId { flag_id: u32 },
    #[error("MTE2 transfer byte count overflows usize")]
    TransferSizeOverflow,
    #[error("C220 output tile was not produced before its flag")]
    OutputNotProduced,
    #[error("C220 output tile is still waiting for a flag")]
    UnsignaledOutputTile,
    #[error("C220 output flag {flag_id} is already set")]
    OutputFlagAlreadySet { flag_id: u8 },
    #[error("C220 output flag {flag_id} was not set before wait")]
    OutputWaitWithoutFlag { flag_id: u8 },
    #[error("C220 output flag ID {flag_id} is outside the modeled range")]
    UnsupportedOutputFlagId { flag_id: u32 },
    #[error("C220 output transfer has no matching completed flag")]
    OutputNotReady,
    #[error("C220 output transfer source or length differs from the signaled tile")]
    OutputTileMismatch,
    #[error("C220 output dependency is already waiting for transfer")]
    OutputDependencyOutstanding,
    #[error("C220 MTE2 reuse flag {flag_id} is already set")]
    ReuseFlagAlreadySet { flag_id: u8 },
    #[error("C220 MTE2 reuse flag {flag_id} was not set before wait")]
    ReuseWaitWithoutFlag { flag_id: u8 },
    #[error("C220 MTE3 completion flag {flag_id} is already set")]
    CompletionFlagAlreadySet { flag_id: u8 },
    #[error("C220 MTE3 completion flag {flag_id} was not set before wait")]
    CompletionWaitWithoutFlag { flag_id: u8 },
    #[error("C220 output was not copied before MTE3 completion flag")]
    OutputNotCopied,
    #[error(transparent)]
    C220(#[from] C220MovOutToUbError),
    #[error(transparent)]
    C310(#[from] C310CapturedMovAlignError),
    #[error(transparent)]
    Ub(#[from] UbReplayError),
    #[error(transparent)]
    Vector(#[from] C220CapturedVectorError),
}

impl MteCoreStepper {
    pub fn new(scalar: ScalarStepper, ub: UbReplayMemory) -> Self {
        Self {
            scalar,
            ub,
            pending_mte2: Vec::new(),
            flag0_set: false,
            vector_flags: [None; 2],
            unsignaled_output: None,
            mte3_flags: [None; 4],
            mte2_reuse_flags: [false; 2],
            ready_output: None,
            copied_output: None,
            mte3_completion_flags: [None; 4],
        }
    }

    pub const fn scalar(&self) -> &ScalarStepper {
        &self.scalar
    }

    pub fn scalar_mut(&mut self) -> &mut ScalarStepper {
        &mut self.scalar
    }

    pub const fn ub(&self) -> &UbReplayMemory {
        &self.ub
    }

    pub const fn pending_mte2_count(&self) -> usize {
        self.pending_mte2.len()
    }

    pub const fn flag0_set(&self) -> bool {
        self.flag0_set
    }

    pub const fn vector_flags_set(&self) -> [bool; 2] {
        [
            self.vector_flags[0].is_some(),
            self.vector_flags[1].is_some(),
        ]
    }

    pub const fn output_flags_set(&self) -> [bool; 4] {
        [
            self.mte3_flags[0].is_some(),
            self.mte3_flags[1].is_some(),
            self.mte3_flags[2].is_some(),
            self.mte3_flags[3].is_some(),
        ]
    }

    pub const fn reuse_flags_set(&self) -> [bool; 2] {
        self.mte2_reuse_flags
    }

    pub const fn completion_flags_set(&self) -> [bool; 4] {
        [
            self.mte3_completion_flags[0].is_some(),
            self.mte3_completion_flags[1].is_some(),
            self.mte3_completion_flags[2].is_some(),
            self.mte3_completion_flags[3].is_some(),
        ]
    }

    pub fn step_scalar_word<B: ScalarMemoryBus>(
        &mut self,
        word: u32,
        bus: &mut B,
    ) -> Result<ScalarProgramStep, ScalarInstructionError<B::Error>> {
        self.scalar.step_word(word, bus)
    }

    pub fn step_scalar_word_with_ub<B: ScalarMemoryBus>(
        &mut self,
        word: u32,
        fallback: &mut B,
    ) -> Result<ScalarProgramStep, ScalarInstructionError<UbScalarBusError<B::Error>>> {
        let mut bus = UbScalarBus {
            ub: &mut self.ub,
            fallback,
        };
        self.scalar.step_word(word, &mut bus)
    }

    pub fn step_c310_vf_words_with_ub<B: ScalarMemoryBus>(
        &mut self,
        first_word: u32,
        second_word: u32,
        fallback: &mut B,
    ) -> Result<ScalarProgramStep, ScalarInstructionError<UbScalarBusError<B::Error>>> {
        let mut bus = UbScalarBus {
            ub: &mut self.ub,
            fallback,
        };
        self.scalar
            .step_c310_vf_words(first_word, second_word, &mut bus)
    }

    pub fn step_mte_word(
        &mut self,
        word: u32,
        source: &AclReplayAddressSpace,
    ) -> Result<MteProgramStep, MteStepperError> {
        let pc = self.scalar.pc();
        if self.scalar.is_halted() {
            return Err(MteStepperError::ProgramEnded { pc });
        }
        let action = match word {
            MTE2_TO_SCALAR_SET_FLAG0_WORD => {
                if self.flag0_set {
                    return Err(MteStepperError::FlagAlreadySet);
                }
                if self.pending_mte2.is_empty() {
                    return Err(MteStepperError::SetWithoutTransfer);
                }
                self.flag0_set = true;
                MteAction::SetFlag {
                    pending_count: self.pending_mte2.len(),
                }
            }
            MTE2_TO_SCALAR_WAIT_FLAG0_WORD => {
                if !self.flag0_set {
                    return Err(MteStepperError::WaitWithoutFlag);
                }
                let mut staged = self.ub.clone();
                let mut transfers = Vec::with_capacity(self.pending_mte2.len());
                for transfer in &self.pending_mte2 {
                    transfers.push(transfer.commit(&mut staged, source)?);
                }
                self.ub = staged;
                self.pending_mte2.clear();
                self.flag0_set = false;
                MteAction::WaitFlag { transfers }
            }
            MTE2_TO_VECTOR_SET_FLAG0_WORD
            | C220_SUB_MTE2_TO_VECTOR_SET_FLAG0_WORD
            | MTE2_TO_VECTOR_SET_FLAG1_WORD
            | C220_SUB_MTE2_TO_VECTOR_SET_FLAG1_WORD
                if self.scalar.machine().architecture() == Architecture::Dav2201 =>
            {
                let flag_id = self.resolve_c220_vector_flag_id(pc, word)?;
                let slot = &mut self.vector_flags[usize::from(flag_id)];
                if slot.is_some() {
                    return Err(MteStepperError::VectorFlagAlreadySet { flag_id });
                }
                if self.pending_mte2.is_empty() {
                    return Err(MteStepperError::VectorSetWithoutTransfer { flag_id });
                }
                *slot = Some(self.pending_mte2.remove(0));
                MteAction::SetVectorFlag {
                    flag_id,
                    remaining_pending: self.pending_mte2.len(),
                }
            }
            MTE2_TO_VECTOR_WAIT_FLAG0_WORD
            | C220_SUB_MTE2_TO_VECTOR_WAIT_FLAG0_WORD
            | MTE2_TO_VECTOR_WAIT_FLAG1_WORD
                if self.scalar.machine().architecture() == Architecture::Dav2201 =>
            {
                let flag_id = self.resolve_c220_vector_flag_id(pc, word)?;
                let transfer = self.vector_flags[usize::from(flag_id)]
                    .ok_or(MteStepperError::VectorWaitWithoutFlag { flag_id })?;
                let mut staged = self.ub.clone();
                let result = transfer.commit(&mut staged, source)?;
                self.ub = staged;
                self.vector_flags[usize::from(flag_id)] = None;
                MteAction::WaitVectorFlag {
                    flag_id,
                    transfer: result,
                }
            }
            _ => {
                if self.pending_mte2.len()
                    + self
                        .vector_flags
                        .iter()
                        .filter(|slot| slot.is_some())
                        .count()
                    >= MAX_PENDING_MTE2_TRANSFERS
                {
                    return Err(MteStepperError::PendingLimit);
                }
                if self.flag0_set {
                    return Err(MteStepperError::FlagAlreadySet);
                }
                let (transfer, source_address, destination_address, planned_bytes) =
                    self.decode_transfer(pc, word)?;
                self.pending_mte2.push(transfer);
                MteAction::Issue {
                    source_address,
                    destination_address,
                    planned_bytes,
                    pending_count: self.pending_mte2.len(),
                }
            }
        };
        self.scalar.advance_sequential();
        Ok(MteProgramStep {
            pc,
            word,
            next_pc: self.scalar.pc(),
            action,
        })
    }

    pub fn step_c220_movev_word(
        &mut self,
        word: u32,
    ) -> Result<C220CapturedMovevStep, MteStepperError> {
        let pc = self.scalar.pc();
        if self.scalar.is_halted() {
            return Err(MteStepperError::ProgramEnded { pc });
        }
        if self.scalar.machine().architecture() != Architecture::Dav2201
            || word != C220_CAPTURED_MOVEV_WORD
        {
            return Err(MteStepperError::UnsupportedWord { pc, word });
        }
        if self.output_buffer_busy() {
            return Err(MteStepperError::OutputDependencyOutstanding);
        }
        let machine = self.scalar.machine();
        let control = machine.xregs()[5];
        if control != C220_CAPTURED_MOVEV_CONTROL {
            return Err(C220CapturedVectorError::UnsupportedMovevControl { control }.into());
        }
        let mask0 = machine.spr_value(100);
        let mask1 = machine.spr_value(101);
        if (mask0, mask1) != (Some(32), Some(0)) {
            return Err(C220CapturedVectorError::UnsupportedMovevMask { mask0, mask1 }.into());
        }
        let destination_address = machine.xregs()[16];
        let scalar_word = machine.xregs()[6] as u32;
        let mut staged = self.ub.clone();
        let step = execute_captured_c220_movev_to_ub(
            pc,
            word,
            destination_address,
            scalar_word,
            &mut staged,
        )?;
        self.ub = staged;
        self.scalar.advance_sequential();
        Ok(step)
    }

    fn resolve_c220_vector_flag_id(&self, pc: u64, word: u32) -> Result<u8, MteStepperError> {
        let instruction = FlagInstruction::decode(Architecture::Dav2201, word)
            .filter(|instruction| {
                instruction.source_pipe_code == 4 && instruction.trigger_pipe_code == 1
            })
            .ok_or(MteStepperError::UnsupportedWord { pc, word })?;
        let flag_id = instruction
            .resolve(pc, self.scalar.machine().xregs())
            .flag_id;
        u8::try_from(flag_id)
            .ok()
            .filter(|id| *id < 2)
            .ok_or(MteStepperError::UnsupportedVectorFlagId { flag_id })
    }

    fn output_buffer_busy(&self) -> bool {
        self.unsignaled_output.is_some()
            || self.mte3_flags.iter().any(Option::is_some)
            || self.ready_output.is_some()
            || self.copied_output.is_some()
            || self.mte3_completion_flags.iter().any(Option::is_some)
    }

    pub fn step_c220_vadd_word(
        &mut self,
        word: u32,
    ) -> Result<C220CapturedFp32Step, MteStepperError> {
        if word != C220_CAPTURED_VADD_WORD {
            return Err(MteStepperError::UnsupportedWord {
                pc: self.scalar.pc(),
                word,
            });
        }
        self.step_c220_fp32_word(word)
    }

    pub fn step_c220_vsub_word(
        &mut self,
        word: u32,
    ) -> Result<C220CapturedFp32Step, MteStepperError> {
        if word != C220_CAPTURED_VSUB_WORD {
            return Err(MteStepperError::UnsupportedWord {
                pc: self.scalar.pc(),
                word,
            });
        }
        self.step_c220_fp32_word(word)
    }

    fn step_c220_fp32_word(&mut self, word: u32) -> Result<C220CapturedFp32Step, MteStepperError> {
        let pc = self.scalar.pc();
        if self.scalar.is_halted() {
            return Err(MteStepperError::ProgramEnded { pc });
        }
        if self.scalar.machine().architecture() != Architecture::Dav2201
            || !matches!(word, C220_CAPTURED_VADD_WORD | C220_CAPTURED_VSUB_WORD)
        {
            return Err(MteStepperError::UnsupportedWord { pc, word });
        }
        if self.output_buffer_busy() {
            return Err(MteStepperError::OutputDependencyOutstanding);
        }
        let hint = C220VecArithmeticHint::from_word(word)
            .ok_or(MteStepperError::UnsupportedWord { pc, word })?;
        let machine = self.scalar.machine();
        let xregs = machine.xregs();
        let control = xregs[usize::from(hint.x_register_index_8)];
        if control != C220_CAPTURED_VADD_CONTROL {
            return Err(C220CapturedVectorError::UnsupportedVaddControl { control }.into());
        }
        let ctrl = machine.spr_value(3);
        let mask0 = machine.spr_value(100);
        let mask1 = machine.spr_value(101);
        match word {
            C220_CAPTURED_VADD_WORD
                if (ctrl, mask0, mask1) != (Some(0), Some(0x5555_5555), Some(0)) =>
            {
                return Err(
                    C220CapturedVectorError::UnsupportedVaddMask { ctrl, mask0, mask1 }.into(),
                );
            }
            C220_CAPTURED_VSUB_WORD
                if (ctrl, mask0, mask1) != (Some(1 << 56), Some(32), Some(0)) =>
            {
                return Err(
                    C220CapturedVectorError::UnsupportedVsubMask { ctrl, mask0, mask1 }.into(),
                );
            }
            _ => {}
        }
        let active_mask =
            decode_captured_c220_fp32_mask(ctrl.unwrap(), mask0.unwrap(), mask1.unwrap())?;
        let mut staged = self.ub.clone();
        let step = execute_captured_c220_fp32_to_ub(
            pc,
            word,
            xregs[usize::from(hint.x_register_index_4)],
            xregs[usize::from(hint.x_register_index_6)],
            xregs[usize::from(hint.x_register_index_0)],
            &active_mask,
            &mut staged,
        )?;
        self.unsignaled_output = Some(C220OutputTile {
            source_address: step.destination_address,
            bytes: 128,
        });
        self.ub = staged;
        self.scalar.advance_sequential();
        Ok(step)
    }

    pub fn step_c220_output_word(
        &mut self,
        word: u32,
        destination: &mut AclReplayAddressSpace,
    ) -> Result<C220OutputStep, MteStepperError> {
        let pc = self.scalar.pc();
        if self.scalar.is_halted() {
            return Err(MteStepperError::ProgramEnded { pc });
        }
        if self.scalar.machine().architecture() != Architecture::Dav2201 {
            return Err(MteStepperError::UnsupportedWord { pc, word });
        }
        let x = self.scalar.machine().xregs();
        let action = match word {
            C220_VECTOR_TO_MTE3_SET_FLAG_WORD | C220_SUB_VECTOR_TO_MTE3_SET_FLAG_WORD => {
                let flag_id = self.resolve_c220_output_flag_id(pc, word, 4)?;
                if self.mte3_flags[usize::from(flag_id)].is_some() {
                    return Err(MteStepperError::OutputFlagAlreadySet { flag_id });
                }
                let tile = self
                    .unsignaled_output
                    .ok_or(MteStepperError::OutputNotProduced)?;
                self.ub.read_known(tile.source_address, tile.bytes)?;
                self.unsignaled_output = None;
                self.mte3_flags[usize::from(flag_id)] = Some(tile);
                C220OutputAction::SetMte3Flag {
                    flag_id,
                    source_address: tile.source_address,
                }
            }
            C220_VECTOR_TO_MTE2_SET_FLAG_WORD | C220_SUB_VECTOR_TO_MTE2_SET_FLAG_WORD => {
                let flag_id = self.resolve_c220_output_flag_id(pc, word, 2)?;
                if self.mte2_reuse_flags[usize::from(flag_id)] {
                    return Err(MteStepperError::ReuseFlagAlreadySet { flag_id });
                }
                if self.unsignaled_output.is_some() {
                    return Err(MteStepperError::UnsignaledOutputTile);
                }
                if self.mte3_flags.iter().all(Option::is_none) && self.ready_output.is_none() {
                    return Err(MteStepperError::OutputNotProduced);
                }
                self.mte2_reuse_flags[usize::from(flag_id)] = true;
                C220OutputAction::SetMte2ReuseFlag { flag_id }
            }
            C220_VECTOR_TO_MTE2_WAIT_FLAG0_WORD
            | C220_SUB_VECTOR_TO_MTE2_WAIT_FLAG0_WORD
            | C220_VECTOR_TO_MTE2_WAIT_FLAG1_WORD
            | C220_SUB_VECTOR_TO_MTE2_WAIT_FLAG1_WORD
            | C220_VECTOR_TO_MTE2_WAIT_DYNAMIC_WORD => {
                let flag_id = self.resolve_c220_output_flag_id(pc, word, 2)?;
                if !self.mte2_reuse_flags[usize::from(flag_id)] {
                    return Err(MteStepperError::ReuseWaitWithoutFlag { flag_id });
                }
                self.mte2_reuse_flags[usize::from(flag_id)] = false;
                C220OutputAction::WaitMte2ReuseFlag { flag_id }
            }
            C220_VECTOR_TO_MTE3_WAIT_FLAG_WORD | C220_SUB_VECTOR_TO_MTE3_WAIT_FLAG_WORD => {
                let flag_id = self.resolve_c220_output_flag_id(pc, word, 4)?;
                if self.ready_output.is_some() {
                    return Err(MteStepperError::OutputDependencyOutstanding);
                }
                let tile = self.mte3_flags[usize::from(flag_id)]
                    .take()
                    .ok_or(MteStepperError::OutputWaitWithoutFlag { flag_id })?;
                self.ready_output = Some(tile);
                C220OutputAction::WaitMte3Flag {
                    flag_id,
                    source_address: tile.source_address,
                }
            }
            CAPTURED_C220_MOV_UB_TO_OUT_WORD | CAPTURED_C220_SUB_MOV_UB_TO_OUT_WORD => {
                let tile = self.ready_output.ok_or(MteStepperError::OutputNotReady)?;
                if self.copied_output.is_some()
                    || self.mte3_completion_flags.iter().any(Option::is_some)
                {
                    return Err(MteStepperError::OutputDependencyOutstanding);
                }
                let (source_address, destination_address, xm) =
                    if word == CAPTURED_C220_MOV_UB_TO_OUT_WORD {
                        (x[14], x[10], x[3])
                    } else {
                        (x[12], x[8], x[4])
                    };
                let descriptor =
                    C220DmaMovDescriptor::decode(word, xm).map_err(UbReplayError::from)?;
                let segments = descriptor
                    .segments(source_address, destination_address)
                    .map_err(UbReplayError::from)?;
                let bytes = segments
                    .len()
                    .checked_mul(32)
                    .ok_or(MteStepperError::TransferSizeOverflow)?;
                if tile.source_address != source_address || tile.bytes != bytes {
                    return Err(MteStepperError::OutputTileMismatch);
                }
                self.ub.read_known(source_address, bytes)?;
                let transfer = self.ub.copy_c220_mov_ub_to_hbm(
                    destination.regions_mut(),
                    descriptor,
                    source_address,
                    destination_address,
                )?;
                self.ready_output = None;
                self.copied_output = Some(tile);
                C220OutputAction::CopyToHbm {
                    source_address,
                    destination_address,
                    transfer,
                }
            }
            C220_MTE3_TO_VECTOR_SET_FLAG_WORD | C220_SUB_MTE3_TO_VECTOR_SET_FLAG_WORD => {
                let flag_id = self.resolve_c220_output_flag_id(pc, word, 4)?;
                if self.mte3_completion_flags[usize::from(flag_id)].is_some() {
                    return Err(MteStepperError::CompletionFlagAlreadySet { flag_id });
                }
                let tile = self.copied_output.ok_or(MteStepperError::OutputNotCopied)?;
                self.copied_output = None;
                self.mte3_completion_flags[usize::from(flag_id)] = Some(tile);
                C220OutputAction::SetMte3CompletionFlag {
                    flag_id,
                    source_address: tile.source_address,
                }
            }
            C220_MTE3_TO_VECTOR_WAIT_FLAG_WORD | C220_SUB_MTE3_TO_VECTOR_WAIT_FLAG_WORD => {
                let flag_id = self.resolve_c220_output_flag_id(pc, word, 4)?;
                let tile = self.mte3_completion_flags[usize::from(flag_id)]
                    .take()
                    .ok_or(MteStepperError::CompletionWaitWithoutFlag { flag_id })?;
                C220OutputAction::WaitMte3CompletionFlag {
                    flag_id,
                    source_address: tile.source_address,
                }
            }
            _ => return Err(MteStepperError::UnsupportedWord { pc, word }),
        };
        self.scalar.advance_sequential();
        Ok(C220OutputStep {
            pc,
            word,
            next_pc: self.scalar.pc(),
            action,
        })
    }

    fn resolve_c220_output_flag_id(
        &self,
        pc: u64,
        word: u32,
        max_id: u8,
    ) -> Result<u8, MteStepperError> {
        let (source_pipe_code, trigger_pipe_code, operation) = match word {
            C220_VECTOR_TO_MTE3_SET_FLAG_WORD | C220_SUB_VECTOR_TO_MTE3_SET_FLAG_WORD => {
                (1, 5, FlagOperation::Set)
            }
            C220_VECTOR_TO_MTE2_SET_FLAG_WORD | C220_SUB_VECTOR_TO_MTE2_SET_FLAG_WORD => {
                (1, 4, FlagOperation::Set)
            }
            C220_VECTOR_TO_MTE3_WAIT_FLAG_WORD | C220_SUB_VECTOR_TO_MTE3_WAIT_FLAG_WORD => {
                (1, 5, FlagOperation::Wait)
            }
            C220_VECTOR_TO_MTE2_WAIT_FLAG0_WORD
            | C220_SUB_VECTOR_TO_MTE2_WAIT_FLAG0_WORD
            | C220_VECTOR_TO_MTE2_WAIT_FLAG1_WORD
            | C220_SUB_VECTOR_TO_MTE2_WAIT_FLAG1_WORD
            | C220_VECTOR_TO_MTE2_WAIT_DYNAMIC_WORD => (1, 4, FlagOperation::Wait),
            C220_MTE3_TO_VECTOR_SET_FLAG_WORD | C220_SUB_MTE3_TO_VECTOR_SET_FLAG_WORD => {
                (5, 1, FlagOperation::Set)
            }
            C220_MTE3_TO_VECTOR_WAIT_FLAG_WORD | C220_SUB_MTE3_TO_VECTOR_WAIT_FLAG_WORD => {
                (5, 1, FlagOperation::Wait)
            }
            _ => return Err(MteStepperError::UnsupportedWord { pc, word }),
        };
        let instruction = FlagInstruction::decode(Architecture::Dav2201, word)
            .filter(|instruction| {
                instruction.source_pipe_code == source_pipe_code
                    && instruction.trigger_pipe_code == trigger_pipe_code
                    && instruction.operation == operation
            })
            .ok_or(MteStepperError::UnsupportedWord { pc, word })?;
        let flag_id = instruction
            .resolve(pc, self.scalar.machine().xregs())
            .flag_id;
        u8::try_from(flag_id)
            .ok()
            .filter(|id| *id < max_id)
            .ok_or(MteStepperError::UnsupportedOutputFlagId { flag_id })
    }

    fn decode_transfer(
        &self,
        pc: u64,
        word: u32,
    ) -> Result<(PendingMte2, u64, u64, usize), MteStepperError> {
        let machine = self.scalar.machine();
        let x = machine.xregs();
        match machine.architecture() {
            Architecture::Dav2201 => {
                let (destination, source, descriptor) = match word {
                    CAPTURED_C220_TILING_MOV_OUT_TO_UB_WORD => (0, 1, 2),
                    CAPTURED_C220_SUB_TILING_MOV_OUT_TO_UB_WORD => (0, 1, 3),
                    CAPTURED_C220_MOV_OUT_TO_UB_X_WORD => (15, 19, 3),
                    CAPTURED_C220_MOV_OUT_TO_UB_Y_WORD => (18, 15, 3),
                    CAPTURED_C220_SUB_MOV_OUT_TO_UB_X_WORD => (13, 17, 4),
                    CAPTURED_C220_SUB_MOV_OUT_TO_UB_Y_WORD => (16, 13, 4),
                    _ => return Err(MteStepperError::UnsupportedWord { pc, word }),
                };
                let descriptor = C220MovOutToUbDescriptor::decode(word, x[descriptor])?;
                let source_address = x[source];
                let destination_address = x[destination];
                let planned_bytes = descriptor
                    .segments(source_address, destination_address)?
                    .len()
                    .checked_mul(32)
                    .ok_or(MteStepperError::TransferSizeOverflow)?;
                Ok((
                    PendingMte2::C220 {
                        descriptor,
                        source_address,
                        destination_address,
                    },
                    source_address,
                    destination_address,
                    planned_bytes,
                ))
            }
            Architecture::Dav3510 => {
                let spr = |index| {
                    machine
                        .spr_value(index)
                        .ok_or(MteStepperError::MissingSpr { pc, index })
                };
                let decoded = match word {
                    C310_TILING_MOV_ALIGN_WORD => C310TilingMovAlignRegisters {
                        source_xreg1: x[1],
                        shape_xreg4: x[4],
                        destination_and_stride_xreg7: x[7],
                        loop_spr105: spr(105)?,
                        inner_stride_spr106: spr(106)?,
                        outer_stride_spr107: spr(107)?,
                    }
                    .decode(word)?,
                    C310_ADD_MOV_ALIGN_X_WORD
                    | C310_ADD_MOV_ALIGN_Y_WORD
                    | 0x74ad_8bae
                    | 0x74b3_6bae => C310MovAlignRegisterSelectors::from_captured_word(word)?
                        .capture(x, spr(105)?, spr(106)?, spr(107)?)
                        .decode_hbm_to_ub_word(word)?,
                    _ => return Err(MteStepperError::UnsupportedWord { pc, word }),
                };
                let coordinates = decoded
                    .parameters
                    .coordinates()
                    .map_err(UbReplayError::from)?;
                let planned_bytes = coordinates
                    .len()
                    .checked_mul(decoded.burst_bytes as usize)
                    .ok_or(MteStepperError::TransferSizeOverflow)?;
                Ok((
                    PendingMte2::C310(decoded),
                    decoded.parameters.source_base,
                    decoded.parameters.destination_base,
                    planned_bytes,
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acl_address_space::AclArgumentImage;
    use crate::acl_args::AclArgumentPlan;
    use crate::addressed_replay::AddressedReplayMemory;
    use crate::kernel_config::KernelConfigDocument;
    use crate::machine::{ScalarInstructionStep, ScalarMachine, ScalarMemoryExecutionError};
    use crate::replay_memory::{MemoryByteState, ReplayMemory};
    use crate::replay_seed::ReplaySeed;
    use crate::rvec::{C310_CAPTURED_VLDI_V0_WORD, C310RvecValueMachine};
    use std::fs;
    use std::io;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

    struct Fixture {
        directory: PathBuf,
        source: AclReplayAddressSpace,
    }

    impl Fixture {
        fn new(bytes: &[u8], tiling: bool) -> Self {
            let id = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
            let directory = std::env::temp_dir().join(format!(
                "open-ascend-mte-stepper-{}-{id}",
                std::process::id()
            ));
            fs::create_dir(&directory).unwrap();
            let path = directory.join("input.bin");
            fs::write(&path, bytes).unwrap();
            let json = if tiling {
                format!(
                    "{{\"old_mode\":\"0\",\"output_name\":\"z.bin\",\"output_size\":\"32\",\"tiling_data_path\":\"{};{}\"}}",
                    path.display(),
                    bytes.len(),
                )
            } else {
                format!(
                    "{{\"old_mode\":\"0\",\"input_path\":\"{}\",\"input_size\":\"{}\",\"output_name\":\"z.bin\",\"output_size\":\"32\"}}",
                    path.display(),
                    bytes.len(),
                )
            };
            let config = KernelConfigDocument::from_slice(json.as_bytes())
                .unwrap()
                .decode()
                .unwrap();
            let plan = AclArgumentPlan::from_config(&config).unwrap();
            let seed = ReplaySeed::load(&config, bytes.len() as u64).unwrap();
            let memory = ReplayMemory::new(seed, 64, 128);
            let pointers: &[u64] = if tiling {
                &[0x2000, 0x3000]
            } else {
                &[0x3000, 0x2000]
            };
            let regions = AddressedReplayMemory::bind(memory, pointers).unwrap();
            let image = AclArgumentImage::new(&plan, 0x1000, pointers, &[]).unwrap();
            let source = AclReplayAddressSpace::new(image, regions).unwrap();
            Self { directory, source }
        }

        fn two_inputs(x: &[u8], y: &[u8]) -> Self {
            let id = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
            let directory = std::env::temp_dir().join(format!(
                "open-ascend-mte-stepper-{}-{id}",
                std::process::id()
            ));
            fs::create_dir(&directory).unwrap();
            let x_path = directory.join("x.bin");
            let y_path = directory.join("y.bin");
            fs::write(&x_path, x).unwrap();
            fs::write(&y_path, y).unwrap();
            let json = format!(
                "{{\"old_mode\":\"0\",\"input_path\":\"{};{}\",\"input_size\":\"{};{}\",\"output_name\":\"z.bin\",\"output_size\":\"128\"}}",
                x_path.display(),
                y_path.display(),
                x.len(),
                y.len(),
            );
            let config = KernelConfigDocument::from_slice(json.as_bytes())
                .unwrap()
                .decode()
                .unwrap();
            let plan = AclArgumentPlan::from_config(&config).unwrap();
            let seed = ReplaySeed::load(&config, 4096).unwrap();
            let memory = ReplayMemory::new(seed, 512, 512);
            let pointers = &[0x3000, 0x4000, 0x2000];
            let regions = AddressedReplayMemory::bind(memory, pointers).unwrap();
            let image = AclArgumentImage::new(&plan, 0x1000, pointers, &[]).unwrap();
            let source = AclReplayAddressSpace::new(image, regions).unwrap();
            Self { directory, source }
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.directory);
        }
    }

    fn stepper(architecture: Architecture, source_address: u64) -> MteCoreStepper {
        let mut machine = ScalarMachine::from_pem_initial_state(architecture);
        machine.set_xreg(1, source_address).unwrap();
        match architecture {
            Architecture::Dav2201 => machine.set_xreg(2, 0x10010).unwrap(),
            Architecture::Dav3510 => machine.set_xreg(4, 0x4000_0010).unwrap(),
        }
        MteCoreStepper::new(
            ScalarStepper::new(machine, 0x1000),
            UbReplayMemory::new(512, 256),
        )
    }

    struct NoMemoryBus;

    impl ScalarMemoryBus for NoMemoryBus {
        type Error = io::Error;

        fn read(&mut self, _address: u64, _destination: &mut [u8]) -> Result<(), Self::Error> {
            Err(io::Error::other("memory read was not expected"))
        }

        fn write(&mut self, _address: u64, _source: &[u8]) -> Result<(), Self::Error> {
            Err(io::Error::other("memory write was not expected"))
        }
    }

    #[test]
    fn scalar_ub_adapter_forwards_c310_buffer_disposition() {
        struct BufferBus {
            last: Option<C310BufferStep>,
        }

        impl ScalarMemoryBus for BufferBus {
            type Error = io::Error;

            fn read(&mut self, _address: u64, _destination: &mut [u8]) -> Result<(), Self::Error> {
                Err(io::Error::other("memory read was not expected"))
            }

            fn write(&mut self, _address: u64, _source: &[u8]) -> Result<(), Self::Error> {
                Err(io::Error::other("memory write was not expected"))
            }

            fn execute_c310_buffer(
                &mut self,
                step: C310BufferStep,
            ) -> Result<C310BufferDisposition, Self::Error> {
                self.last = Some(step);
                Ok(C310BufferDisposition::Accepted)
            }
        }

        let mut core = stepper(Architecture::Dav3510, 0);
        core.scalar_mut().machine_mut().set_xreg(1, 0x25).unwrap();
        let mut bus = BufferBus { last: None };
        let step = core
            .step_scalar_word_with_ub(0x4202_1004, &mut bus)
            .unwrap();
        let ScalarInstructionStep::Buffer(buffer) = step.instruction else {
            panic!("expected buffer instruction");
        };
        assert_eq!(buffer.buffer_id, 5);
        assert_eq!(bus.last, Some(buffer));
        assert_eq!(core.scalar().pc(), 0x1004);
    }

    #[test]
    fn first_mte_transfer_becomes_visible_only_after_matching_flag_wait() {
        let fixture = Fixture::new(&1024_u32.to_le_bytes(), true);
        for (architecture, word) in [
            (
                Architecture::Dav2201,
                CAPTURED_C220_TILING_MOV_OUT_TO_UB_WORD,
            ),
            (Architecture::Dav3510, C310_TILING_MOV_ALIGN_WORD),
        ] {
            let mut core = stepper(architecture, 0x3000);
            let issue = core.step_mte_word(word, &fixture.source).unwrap();
            assert_eq!(issue.pc, 0x1000);
            assert_eq!(issue.next_pc, 0x1004);
            assert!(matches!(
                issue.action,
                MteAction::Issue {
                    source_address: 0x3000,
                    destination_address: 0,
                    planned_bytes: 32,
                    pending_count: 1,
                }
            ));
            assert_eq!(core.ub().tracked_bytes(), 0);
            assert_eq!(core.pending_mte2_count(), 1);

            let signal = core
                .step_mte_word(MTE2_TO_SCALAR_SET_FLAG0_WORD, &fixture.source)
                .unwrap();
            assert_eq!(signal.pc, 0x1004);
            assert!(matches!(
                signal.action,
                MteAction::SetFlag { pending_count: 1 }
            ));
            assert!(core.flag0_set());
            assert_eq!(core.ub().tracked_bytes(), 0);

            let wait = core
                .step_mte_word(MTE2_TO_SCALAR_WAIT_FLAG0_WORD, &fixture.source)
                .unwrap();
            assert_eq!(wait.pc, 0x1008);
            assert_eq!(wait.next_pc, 0x100c);
            assert_eq!(
                wait.action,
                MteAction::WaitFlag {
                    transfers: vec![UbTransferResult {
                        segment_count: 1,
                        bytes: 32,
                        known_bytes: 4,
                        unknown_bytes: 28,
                    }]
                }
            );
            assert_eq!(core.ub().read_known(0, 4).unwrap(), 1024_u32.to_le_bytes());
            assert_eq!(
                core.ub().read_states(4, 28).unwrap(),
                [MemoryByteState::Unknown; 28]
            );
            assert_eq!(core.pending_mte2_count(), 0);
            assert!(!core.flag0_set());
        }
    }

    #[test]
    fn c220_sub_tiling_uses_its_own_descriptor_register() {
        let fixture = Fixture::new(&1024_u32.to_le_bytes(), true);
        let mut core = stepper(Architecture::Dav2201, 0x3000);
        core.scalar_mut().machine_mut().set_xreg(2, 0).unwrap();
        core.scalar_mut()
            .machine_mut()
            .set_xreg(3, 0x10010)
            .unwrap();
        let issue = core
            .step_mte_word(CAPTURED_C220_SUB_TILING_MOV_OUT_TO_UB_WORD, &fixture.source)
            .unwrap();
        assert!(matches!(
            issue.action,
            MteAction::Issue {
                source_address: 0x3000,
                destination_address: 0,
                planned_bytes: 32,
                ..
            }
        ));
        core.step_mte_word(MTE2_TO_SCALAR_SET_FLAG0_WORD, &fixture.source)
            .unwrap();
        core.step_mte_word(MTE2_TO_SCALAR_WAIT_FLAG0_WORD, &fixture.source)
            .unwrap();
        assert_eq!(core.ub().read_known(0, 4).unwrap(), 1024_u32.to_le_bytes());

        let mut wrong = stepper(Architecture::Dav2201, 0x3000);
        wrong.scalar_mut().machine_mut().set_xreg(3, 0).unwrap();
        let before = wrong.clone();
        assert!(matches!(
            wrong.step_mte_word(CAPTURED_C220_SUB_TILING_MOV_OUT_TO_UB_WORD, &fixture.source),
            Err(MteStepperError::C220(C220MovOutToUbError::UnsupportedXm {
                xm: 0
            }))
        ));
        assert_eq!(wrong, before);
    }

    #[test]
    fn scalar_ub_alias_reads_the_same_mte2_result_on_both_architectures() {
        let fixture = Fixture::new(&1024_u32.to_le_bytes(), true);
        for (architecture, mte_word, scalar_word, address_register, result_register) in [
            (
                Architecture::Dav2201,
                CAPTURED_C220_TILING_MOV_OUT_TO_UB_WORD,
                0x0384_1000,
                1,
                2,
            ),
            (
                Architecture::Dav3510,
                C310_TILING_MOV_ALIGN_WORD,
                0x1c88_0000,
                0,
                4,
            ),
        ] {
            let mut core = stepper(architecture, 0x3000);
            core.step_mte_word(mte_word, &fixture.source).unwrap();
            core.scalar_mut()
                .machine_mut()
                .set_xreg(address_register, SCALAR_UB_ALIAS_BASE)
                .unwrap();
            let before = core.clone();
            assert!(matches!(
                core.step_scalar_word_with_ub(scalar_word, &mut NoMemoryBus),
                Err(ScalarInstructionError::Memory(
                    ScalarMemoryExecutionError::Backend(UbScalarBusError::Ub(
                        UbReplayError::UnknownByte { address: 0 }
                    ))
                ))
            ));
            assert_eq!(core, before);

            core.step_mte_word(MTE2_TO_SCALAR_SET_FLAG0_WORD, &fixture.source)
                .unwrap();
            core.step_mte_word(MTE2_TO_SCALAR_WAIT_FLAG0_WORD, &fixture.source)
                .unwrap();
            let ScalarInstructionStep::Memory(read) = core
                .step_scalar_word_with_ub(scalar_word, &mut NoMemoryBus)
                .unwrap()
                .instruction
            else {
                panic!("expected scalar UB read")
            };
            assert_eq!(read.effective_address, SCALAR_UB_ALIAS_BASE);
            assert_eq!(read.data_value, 1024);
            assert_eq!(core.scalar().machine().xregs()[result_register], 1024);
            assert_eq!(core.ub().read_known(0, 4).unwrap(), 1024_u32.to_le_bytes());
        }
    }

    #[test]
    fn scalar_ub_alias_writes_preserve_provenance_and_reject_crossings() {
        let mut ub = UbReplayMemory::new(64, 64);
        let mut fallback = NoMemoryBus;
        let mut bus = UbScalarBus {
            ub: &mut ub,
            fallback: &mut fallback,
        };
        assert!(matches!(
            bus.read(SCALAR_UB_ALIAS_BASE + 4, &mut [0; 4]),
            Err(UbScalarBusError::Ub(UbReplayError::UnknownByte {
                address: 4
            }))
        ));
        bus.write(SCALAR_UB_ALIAS_BASE + 4, &[1, 2, 3, 4]).unwrap();
        let mut bytes = [0; 4];
        bus.read(SCALAR_UB_ALIAS_BASE + 4, &mut bytes).unwrap();
        assert_eq!(bytes, [1, 2, 3, 4]);
        assert_eq!(bus.ub.read_known(4, 4).unwrap(), bytes);
        assert!(matches!(
            bus.read(SCALAR_UB_ALIAS_BASE - 1, &mut [0; 2]),
            Err(UbScalarBusError::AliasBoundary { .. })
        ));
        assert!(matches!(
            bus.write(SCALAR_UB_ALIAS_BASE + SCALAR_UB_ALIAS_BYTES - 1, &[5, 6]),
            Err(UbScalarBusError::AliasBoundary { .. })
        ));
        assert_eq!(bus.ub.read_known(4, 4).unwrap(), bytes);
    }

    #[test]
    fn scalar_words_interleave_with_mte_issue_signal_and_wait() {
        let fixture = Fixture::new(&1024_u32.to_le_bytes(), true);
        for (architecture, word) in [
            (
                Architecture::Dav2201,
                CAPTURED_C220_TILING_MOV_OUT_TO_UB_WORD,
            ),
            (Architecture::Dav3510, C310_TILING_MOV_ALIGN_WORD),
        ] {
            let mut core = stepper(architecture, 0x3000);
            let mut bus = NoMemoryBus;
            core.step_mte_word(word, &fixture.source).unwrap();
            assert_eq!(
                core.step_scalar_word(0x0706_0001, &mut bus).unwrap().pc,
                0x1004
            );
            assert_eq!(core.ub().tracked_bytes(), 0);
            core.step_mte_word(MTE2_TO_SCALAR_SET_FLAG0_WORD, &fixture.source)
                .unwrap();
            assert_eq!(
                core.step_scalar_word(0x0708_0002, &mut bus).unwrap().pc,
                0x100c
            );
            assert_eq!(core.ub().tracked_bytes(), 0);
            let wait = core
                .step_mte_word(MTE2_TO_SCALAR_WAIT_FLAG0_WORD, &fixture.source)
                .unwrap();
            assert_eq!(wait.pc, 0x1010);
            assert_eq!(core.scalar().pc(), 0x1014);
            assert_eq!(core.ub().read_known(0, 4).unwrap(), 1024_u32.to_le_bytes());
        }
    }

    #[test]
    fn invalid_flag_order_and_failed_source_keep_pc_and_ub_unchanged() {
        let fixture = Fixture::new(&[0x5a; 32], false);
        let mut core = stepper(Architecture::Dav2201, 0x4000);
        assert!(matches!(
            core.step_mte_word(MTE2_TO_SCALAR_SET_FLAG0_WORD, &fixture.source),
            Err(MteStepperError::SetWithoutTransfer)
        ));
        assert_eq!(core.scalar().pc(), 0x1000);
        core.step_mte_word(CAPTURED_C220_TILING_MOV_OUT_TO_UB_WORD, &fixture.source)
            .unwrap();
        let before = core.clone();
        assert!(matches!(
            core.step_mte_word(MTE2_TO_SCALAR_WAIT_FLAG0_WORD, &fixture.source),
            Err(MteStepperError::WaitWithoutFlag)
        ));
        assert_eq!(core, before);
        core.step_mte_word(MTE2_TO_SCALAR_SET_FLAG0_WORD, &fixture.source)
            .unwrap();
        let before = core.clone();
        assert!(matches!(
            core.step_mte_word(MTE2_TO_SCALAR_WAIT_FLAG0_WORD, &fixture.source),
            Err(MteStepperError::Ub(UbReplayError::Source(_)))
        ));
        assert_eq!(core, before);
        assert!(matches!(
            core.step_mte_word(MTE2_TO_SCALAR_SET_FLAG0_WORD, &fixture.source),
            Err(MteStepperError::FlagAlreadySet)
        ));
        assert_eq!(core, before);
    }

    #[test]
    fn c220_registers_are_captured_at_issue_for_multiple_transfers() {
        let input: Vec<u8> = (0..128).collect();
        let fixture = Fixture::new(&input, false);
        let mut core = stepper(Architecture::Dav2201, 0x3000);
        core.scalar_mut()
            .machine_mut()
            .set_xreg(3, 0x40010)
            .unwrap();
        core.scalar_mut().machine_mut().set_xreg(15, 0).unwrap();
        core.scalar_mut()
            .machine_mut()
            .set_xreg(19, 0x3000)
            .unwrap();
        core.step_mte_word(CAPTURED_C220_MOV_OUT_TO_UB_X_WORD, &fixture.source)
            .unwrap();
        core.scalar_mut()
            .machine_mut()
            .set_xreg(15, 0x3000)
            .unwrap();
        core.scalar_mut().machine_mut().set_xreg(18, 0x80).unwrap();
        core.step_mte_word(CAPTURED_C220_MOV_OUT_TO_UB_Y_WORD, &fixture.source)
            .unwrap();
        assert_eq!(core.pending_mte2_count(), 2);
        core.step_mte_word(MTE2_TO_SCALAR_SET_FLAG0_WORD, &fixture.source)
            .unwrap();
        let wait = core
            .step_mte_word(MTE2_TO_SCALAR_WAIT_FLAG0_WORD, &fixture.source)
            .unwrap();
        assert!(matches!(
            wait.action,
            MteAction::WaitFlag { transfers } if transfers.len() == 2
                && transfers.iter().all(|transfer| transfer.bytes == 128)
        ));
        assert_eq!(core.ub().read_known(0, 128).unwrap(), input);
        assert_eq!(core.ub().read_known(0x80, 128).unwrap(), input);
    }

    #[test]
    fn c220_vector_flags_commit_each_captured_input_only_at_its_wait() {
        let x: Vec<u8> = (0..128).collect();
        let y: Vec<u8> = (128..=255).collect();
        let fixture = Fixture::two_inputs(&x, &y);
        let mut core = stepper(Architecture::Dav2201, 0x3000);
        core.scalar_mut()
            .machine_mut()
            .set_xreg(3, 0x40010)
            .unwrap();
        core.scalar_mut().machine_mut().set_xreg(15, 0).unwrap();
        core.scalar_mut()
            .machine_mut()
            .set_xreg(19, 0x3000)
            .unwrap();
        core.step_mte_word(CAPTURED_C220_MOV_OUT_TO_UB_X_WORD, &fixture.source)
            .unwrap();
        core.scalar_mut()
            .machine_mut()
            .set_xreg(15, 0x4000)
            .unwrap();
        core.scalar_mut().machine_mut().set_xreg(18, 128).unwrap();
        core.step_mte_word(CAPTURED_C220_MOV_OUT_TO_UB_Y_WORD, &fixture.source)
            .unwrap();
        assert_eq!(core.pending_mte2_count(), 2);
        assert_eq!(core.vector_flags_set(), [false, false]);
        let before = core.clone();
        assert!(matches!(
            core.step_mte_word(MTE2_TO_VECTOR_WAIT_FLAG0_WORD, &fixture.source),
            Err(MteStepperError::VectorWaitWithoutFlag { flag_id: 0 })
        ));
        assert_eq!(core, before);

        let set_x = core
            .step_mte_word(MTE2_TO_VECTOR_SET_FLAG0_WORD, &fixture.source)
            .unwrap();
        assert_eq!(
            set_x.action,
            MteAction::SetVectorFlag {
                flag_id: 0,
                remaining_pending: 1,
            }
        );
        let before = core.clone();
        assert!(matches!(
            core.step_mte_word(MTE2_TO_VECTOR_SET_FLAG0_WORD, &fixture.source),
            Err(MteStepperError::VectorFlagAlreadySet { flag_id: 0 })
        ));
        assert_eq!(core, before);
        core.scalar_mut().machine_mut().set_xreg(13, 1).unwrap();
        core.step_mte_word(MTE2_TO_VECTOR_SET_FLAG1_WORD, &fixture.source)
            .unwrap();
        assert_eq!(core.pending_mte2_count(), 0);
        assert_eq!(core.vector_flags_set(), [true, true]);
        assert!(matches!(
            core.ub().read_known(0, 1),
            Err(UbReplayError::UnknownByte { address: 0 })
        ));
        assert!(matches!(
            core.ub().read_known(128, 1),
            Err(UbReplayError::UnknownByte { address: 128 })
        ));

        let wait_x = core
            .step_mte_word(MTE2_TO_VECTOR_WAIT_FLAG0_WORD, &fixture.source)
            .unwrap();
        assert!(matches!(
            wait_x.action,
            MteAction::WaitVectorFlag {
                flag_id: 0,
                transfer: UbTransferResult {
                    bytes: 128,
                    known_bytes: 128,
                    unknown_bytes: 0,
                    ..
                },
            }
        ));
        assert_eq!(core.vector_flags_set(), [false, true]);
        assert_eq!(core.ub().read_known(0, 128).unwrap(), x);
        assert!(matches!(
            core.ub().read_known(128, 1),
            Err(UbReplayError::UnknownByte { address: 128 })
        ));
        core.scalar_mut().machine_mut().set_xreg(14, 1).unwrap();
        core.step_mte_word(MTE2_TO_VECTOR_WAIT_FLAG1_WORD, &fixture.source)
            .unwrap();
        assert_eq!(core.vector_flags_set(), [false, false]);
        assert_eq!(core.ub().read_known(128, 128).unwrap(), y);
    }

    #[test]
    fn c220_vector_wait_failure_keeps_pending_flag_and_ub_unchanged() {
        let x: Vec<u8> = (0..128).collect();
        let fixture = Fixture::two_inputs(&x, &x);
        let mut core = stepper(Architecture::Dav2201, 0x3000);
        core.scalar_mut()
            .machine_mut()
            .set_xreg(3, 0x40010)
            .unwrap();
        core.scalar_mut().machine_mut().set_xreg(15, 0).unwrap();
        core.scalar_mut()
            .machine_mut()
            .set_xreg(19, 0x5000)
            .unwrap();
        core.step_mte_word(CAPTURED_C220_MOV_OUT_TO_UB_X_WORD, &fixture.source)
            .unwrap();
        core.step_mte_word(MTE2_TO_VECTOR_SET_FLAG0_WORD, &fixture.source)
            .unwrap();
        let before = core.clone();
        assert!(matches!(
            core.step_mte_word(MTE2_TO_VECTOR_WAIT_FLAG0_WORD, &fixture.source),
            Err(MteStepperError::Ub(UbReplayError::Source(_)))
        ));
        assert_eq!(core, before);
        assert_eq!(core.vector_flags_set(), [true, false]);
        assert_eq!(core.ub().tracked_bytes(), 0);
    }

    #[test]
    fn vector_flag_words_reject_wrong_architecture_and_unpaired_order() {
        let fixture = Fixture::new(&[0x5a; 32], false);
        let mut c220 = stepper(Architecture::Dav2201, 0x3000);
        let before = c220.clone();
        assert!(matches!(
            c220.step_mte_word(MTE2_TO_VECTOR_SET_FLAG0_WORD, &fixture.source),
            Err(MteStepperError::VectorSetWithoutTransfer { flag_id: 0 })
        ));
        assert_eq!(c220, before);
        assert!(matches!(
            c220.step_mte_word(C220_SUB_MTE2_TO_VECTOR_SET_FLAG0_WORD, &fixture.source),
            Err(MteStepperError::VectorSetWithoutTransfer { flag_id: 0 })
        ));
        assert_eq!(c220, before);
        c220.scalar_mut().machine_mut().set_xreg(14, 1).unwrap();
        let before = c220.clone();
        assert!(matches!(
            c220.step_mte_word(MTE2_TO_VECTOR_WAIT_FLAG1_WORD, &fixture.source),
            Err(MteStepperError::VectorWaitWithoutFlag { flag_id: 1 })
        ));
        assert_eq!(c220, before);

        let mut c310 = stepper(Architecture::Dav3510, 0x3000);
        let before = c310.clone();
        for word in [
            MTE2_TO_VECTOR_SET_FLAG0_WORD,
            C220_SUB_MTE2_TO_VECTOR_SET_FLAG0_WORD,
            MTE2_TO_VECTOR_SET_FLAG1_WORD,
            C220_SUB_MTE2_TO_VECTOR_SET_FLAG1_WORD,
            MTE2_TO_VECTOR_WAIT_FLAG0_WORD,
            C220_SUB_MTE2_TO_VECTOR_WAIT_FLAG0_WORD,
            MTE2_TO_VECTOR_WAIT_FLAG1_WORD,
        ] {
            assert!(matches!(
                c310.step_mte_word(word, &fixture.source),
                Err(MteStepperError::UnsupportedWord { word: rejected, .. }) if rejected == word
            ));
            assert_eq!(c310, before);
        }
    }

    #[test]
    fn c220_identical_flag_words_resolve_different_live_register_ids() {
        let x = vec![0x11; 128];
        let y = vec![0x22; 128];
        let fixture = Fixture::two_inputs(&x, &y);
        let mut core = stepper(Architecture::Dav2201, 0x3000);
        let machine = core.scalar_mut().machine_mut();
        machine.set_xreg(3, 0x40010).unwrap();
        machine.set_xreg(15, 0).unwrap();
        machine.set_xreg(19, 0x3000).unwrap();
        core.step_mte_word(CAPTURED_C220_MOV_OUT_TO_UB_X_WORD, &fixture.source)
            .unwrap();
        core.scalar_mut()
            .machine_mut()
            .set_xreg(15, 0x4000)
            .unwrap();
        core.scalar_mut().machine_mut().set_xreg(18, 128).unwrap();
        core.step_mte_word(CAPTURED_C220_MOV_OUT_TO_UB_Y_WORD, &fixture.source)
            .unwrap();

        core.scalar_mut().machine_mut().set_xreg(14, 2).unwrap();
        let before = core.clone();
        assert!(matches!(
            core.step_mte_word(MTE2_TO_VECTOR_SET_FLAG0_WORD, &fixture.source),
            Err(MteStepperError::UnsupportedVectorFlagId { flag_id: 2 })
        ));
        assert_eq!(core, before);

        core.scalar_mut().machine_mut().set_xreg(14, 0).unwrap();
        assert!(matches!(
            core.step_mte_word(MTE2_TO_VECTOR_SET_FLAG0_WORD, &fixture.source)
                .unwrap()
                .action,
            MteAction::SetVectorFlag { flag_id: 0, .. }
        ));
        core.scalar_mut().machine_mut().set_xreg(14, 1).unwrap();
        assert!(matches!(
            core.step_mte_word(MTE2_TO_VECTOR_SET_FLAG0_WORD, &fixture.source)
                .unwrap()
                .action,
            MteAction::SetVectorFlag { flag_id: 1, .. }
        ));
        assert_eq!(core.vector_flags_set(), [true, true]);

        core.scalar_mut().machine_mut().set_xreg(12, 1).unwrap();
        assert!(matches!(
            core.step_mte_word(MTE2_TO_VECTOR_WAIT_FLAG0_WORD, &fixture.source)
                .unwrap()
                .action,
            MteAction::WaitVectorFlag { flag_id: 1, .. }
        ));
        assert_eq!(core.ub().read_known(128, 128).unwrap(), y);
        assert!(matches!(
            core.ub().read_known(0, 1),
            Err(UbReplayError::UnknownByte { address: 0 })
        ));
        core.scalar_mut().machine_mut().set_xreg(12, 0).unwrap();
        assert!(matches!(
            core.step_mte_word(MTE2_TO_VECTOR_WAIT_FLAG0_WORD, &fixture.source)
                .unwrap()
                .action,
            MteAction::WaitVectorFlag { flag_id: 0, .. }
        ));
        assert_eq!(core.ub().read_known(0, 128).unwrap(), x);
    }

    #[test]
    fn c220_movev_uses_live_scalar_fill_and_destination_registers() {
        let mut core = stepper(Architecture::Dav2201, 0x3000);
        let machine = core.scalar_mut().machine_mut();
        machine.set_xreg(5, C220_CAPTURED_MOVEV_CONTROL).unwrap();
        machine.set_xreg(6, 0xc2f6_0000).unwrap();
        machine.set_xreg(16, 0x100).unwrap();
        machine.set_spr_value(100, 32).unwrap();
        machine.set_spr_value(101, 0).unwrap();
        let step = core.step_c220_movev_word(C220_CAPTURED_MOVEV_WORD).unwrap();
        assert_eq!(step.pc, 0x1000);
        assert_eq!(step.word, C220_CAPTURED_MOVEV_WORD);
        assert_eq!(step.destination_address, 0x100);
        assert_eq!(step.scalar_word, 0xc2f6_0000);
        assert_eq!(step.stores.len(), 32);
        assert_eq!(core.scalar().pc(), 0x1004);
        assert_eq!(
            core.ub().read_known(0x100, 128).unwrap(),
            0xc2f6_0000_u32.to_le_bytes().repeat(32)
        );
        assert!(matches!(
            core.ub().read_known(0, 1),
            Err(UbReplayError::UnknownByte { address: 0 })
        ));
    }

    #[test]
    fn c220_movev_rejects_unverified_control_mask_and_overflow_atomically() {
        let mut core = stepper(Architecture::Dav2201, 0x3000);
        let machine = core.scalar_mut().machine_mut();
        machine.set_xreg(5, C220_CAPTURED_MOVEV_CONTROL).unwrap();
        machine.set_xreg(6, 0xc2f6_0000).unwrap();
        machine.set_xreg(16, 0x100).unwrap();
        machine.set_spr_value(100, 32).unwrap();
        machine.set_spr_value(101, 0).unwrap();

        core.scalar_mut().machine_mut().set_xreg(5, 0).unwrap();
        let before = core.clone();
        assert!(matches!(
            core.step_c220_movev_word(C220_CAPTURED_MOVEV_WORD),
            Err(MteStepperError::Vector(
                C220CapturedVectorError::UnsupportedMovevControl { control: 0 }
            ))
        ));
        assert_eq!(core, before);

        core.scalar_mut()
            .machine_mut()
            .set_xreg(5, C220_CAPTURED_MOVEV_CONTROL)
            .unwrap();
        core.scalar_mut()
            .machine_mut()
            .set_spr_value(100, 31)
            .unwrap();
        let before = core.clone();
        assert!(matches!(
            core.step_c220_movev_word(C220_CAPTURED_MOVEV_WORD),
            Err(MteStepperError::Vector(
                C220CapturedVectorError::UnsupportedMovevMask { .. }
            ))
        ));
        assert_eq!(core, before);

        core.scalar_mut()
            .machine_mut()
            .set_spr_value(100, 32)
            .unwrap();
        core.ub = UbReplayMemory::new(64, 256);
        let before = core.clone();
        assert!(matches!(
            core.step_c220_movev_word(C220_CAPTURED_MOVEV_WORD),
            Err(MteStepperError::Vector(C220CapturedVectorError::Ub(
                UbReplayError::TrackedLimitExceeded { .. }
            )))
        ));
        assert_eq!(core, before);

        let mut c310 = stepper(Architecture::Dav3510, 0x3000);
        let before = c310.clone();
        assert!(matches!(
            c310.step_c220_movev_word(C220_CAPTURED_MOVEV_WORD),
            Err(MteStepperError::UnsupportedWord { .. })
        ));
        assert_eq!(c310, before);
    }

    #[test]
    fn c220_vadd_uses_live_sources_prior_fill_and_alternating_mask() {
        let mut core = stepper(Architecture::Dav2201, 0x3000);
        let x = (0..32_u32)
            .flat_map(|lane| (lane as f32).to_le_bytes())
            .collect::<Vec<_>>();
        let y = 0.5_f32.to_le_bytes().repeat(32);
        core.ub
            .write_states(
                0,
                &x.iter()
                    .chain(&y)
                    .copied()
                    .map(MemoryByteState::Known)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let machine = core.scalar_mut().machine_mut();
        machine.set_xreg(5, C220_CAPTURED_MOVEV_CONTROL).unwrap();
        machine.set_xreg(6, 0xc2f6_0000).unwrap();
        machine.set_xreg(16, 0x100).unwrap();
        machine.set_spr_value(100, 32).unwrap();
        machine.set_spr_value(101, 0).unwrap();
        core.step_c220_movev_word(C220_CAPTURED_MOVEV_WORD).unwrap();

        let machine = core.scalar_mut().machine_mut();
        machine.set_xreg(8, C220_CAPTURED_VADD_CONTROL).unwrap();
        machine.set_xreg(13, 0).unwrap();
        machine.set_xreg(14, 0x80).unwrap();
        machine.set_spr_value(3, 0).unwrap();
        machine.set_spr_value(100, 0x5555_5555).unwrap();
        let step = core.step_c220_vadd_word(C220_CAPTURED_VADD_WORD).unwrap();
        assert_eq!(step.pc, 0x1004);
        assert_eq!(step.source_0_address, 0);
        assert_eq!(step.source_1_address, 0x80);
        assert_eq!(step.destination_address, 0x100);
        assert_eq!(step.source_0_bytes[..128], x);
        assert_eq!(step.source_1_bytes[..128], y);
        assert_eq!(step.stores.len(), 16);
        assert_eq!(core.scalar().pc(), 0x1008);
        let result = core.ub().read_known(0x100, 128).unwrap();
        for lane in 0..32 {
            let offset = lane * 4;
            let expected = if lane % 2 == 0 {
                ((lane as f32) + 0.5).to_le_bytes()
            } else {
                0xc2f6_0000_u32.to_le_bytes()
            };
            assert_eq!(result[offset..offset + 4], expected);
        }

        let mut fixture = Fixture::two_inputs(&x, &y);
        assert!(fixture.source.regions().read_known_at(0x2000, 128).is_err());
        core.scalar_mut().machine_mut().set_xreg(14, 0).unwrap();
        let set = core
            .step_c220_output_word(C220_VECTOR_TO_MTE3_SET_FLAG_WORD, &mut fixture.source)
            .unwrap();
        assert!(matches!(
            set.action,
            C220OutputAction::SetMte3Flag {
                flag_id: 0,
                source_address: 0x100
            }
        ));
        assert_eq!(core.output_flags_set(), [true, false, false, false]);
        core.scalar_mut().machine_mut().set_xreg(12, 0).unwrap();
        core.step_c220_output_word(C220_VECTOR_TO_MTE2_SET_FLAG_WORD, &mut fixture.source)
            .unwrap();
        core.scalar_mut().machine_mut().set_xreg(12, 1).unwrap();
        core.step_c220_output_word(C220_VECTOR_TO_MTE2_SET_FLAG_WORD, &mut fixture.source)
            .unwrap();
        assert_eq!(core.reuse_flags_set(), [true, true]);
        core.scalar_mut().machine_mut().set_xreg(13, 0).unwrap();
        let waited = core
            .step_c220_output_word(C220_VECTOR_TO_MTE3_WAIT_FLAG_WORD, &mut fixture.source)
            .unwrap();
        assert!(matches!(
            waited.action,
            C220OutputAction::WaitMte3Flag {
                flag_id: 0,
                source_address: 0x100
            }
        ));
        assert!(fixture.source.regions().read_known_at(0x2000, 128).is_err());
        let machine = core.scalar_mut().machine_mut();
        machine.set_xreg(14, 0x100).unwrap();
        machine.set_xreg(10, 0x2000).unwrap();
        machine.set_xreg(3, 0x40010).unwrap();
        let copy = core
            .step_c220_output_word(CAPTURED_C220_MOV_UB_TO_OUT_WORD, &mut fixture.source)
            .unwrap();
        assert!(matches!(
            copy.action,
            C220OutputAction::CopyToHbm {
                source_address: 0x100,
                destination_address: 0x2000,
                transfer: UbTransferResult {
                    bytes: 128,
                    known_bytes: 128,
                    unknown_bytes: 0,
                    ..
                }
            }
        ));
        assert_eq!(
            fixture.source.regions().read_known_at(0x2000, 128).unwrap(),
            result
        );
        let before = core.clone();
        assert!(matches!(
            core.step_c220_movev_word(C220_CAPTURED_MOVEV_WORD),
            Err(MteStepperError::OutputDependencyOutstanding)
        ));
        assert_eq!(core, before);
        core.scalar_mut().machine_mut().set_xreg(10, 0).unwrap();
        let set_completion = core
            .step_c220_output_word(C220_MTE3_TO_VECTOR_SET_FLAG_WORD, &mut fixture.source)
            .unwrap();
        assert!(matches!(
            set_completion.action,
            C220OutputAction::SetMte3CompletionFlag {
                flag_id: 0,
                source_address: 0x100
            }
        ));
        assert_eq!(core.completion_flags_set(), [true, false, false, false]);
        let reuse_zero = core
            .step_c220_output_word(C220_VECTOR_TO_MTE2_WAIT_FLAG0_WORD, &mut fixture.source)
            .unwrap();
        assert!(matches!(
            reuse_zero.action,
            C220OutputAction::WaitMte2ReuseFlag { flag_id: 0 }
        ));
        core.scalar_mut().machine_mut().set_xreg(18, 1).unwrap();
        let reuse_one = core
            .step_c220_output_word(C220_VECTOR_TO_MTE2_WAIT_FLAG1_WORD, &mut fixture.source)
            .unwrap();
        assert!(matches!(
            reuse_one.action,
            C220OutputAction::WaitMte2ReuseFlag { flag_id: 1 }
        ));
        assert_eq!(core.reuse_flags_set(), [false, false]);
        core.scalar_mut().machine_mut().set_xreg(19, 0).unwrap();
        let wait_completion = core
            .step_c220_output_word(C220_MTE3_TO_VECTOR_WAIT_FLAG_WORD, &mut fixture.source)
            .unwrap();
        assert!(matches!(
            wait_completion.action,
            C220OutputAction::WaitMte3CompletionFlag {
                flag_id: 0,
                source_address: 0x100
            }
        ));
        assert_eq!(core.completion_flags_set(), [false; 4]);
    }

    #[test]
    fn c220_vsub_count_mask_writes_and_copies_a_complete_tile() {
        let mut core = stepper(Architecture::Dav2201, 0x3000);
        let x = (0..32_u32)
            .flat_map(|lane| (lane as f32).to_le_bytes())
            .collect::<Vec<_>>();
        let y = 0.5_f32.to_le_bytes().repeat(32);
        core.ub
            .write_states(
                0,
                &x.iter()
                    .chain(&y)
                    .copied()
                    .map(MemoryByteState::Known)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        core.ub
            .write_states(0x100, &[MemoryByteState::Known(0); 128])
            .unwrap();
        let machine = core.scalar_mut().machine_mut();
        machine.set_xreg(6, C220_CAPTURED_VADD_CONTROL).unwrap();
        machine.set_xreg(11, 0).unwrap();
        machine.set_xreg(12, 0x80).unwrap();
        machine.set_xreg(14, 0x100).unwrap();
        machine.set_spr_value(3, 1 << 56).unwrap();
        machine.set_spr_value(100, 31).unwrap();
        machine.set_spr_value(101, 0).unwrap();
        let before = core.clone();
        assert!(matches!(
            core.step_c220_vsub_word(C220_CAPTURED_VSUB_WORD),
            Err(MteStepperError::Vector(
                C220CapturedVectorError::UnsupportedVsubMask { .. }
            ))
        ));
        assert_eq!(core, before);
        core.scalar_mut()
            .machine_mut()
            .set_spr_value(100, 32)
            .unwrap();
        let step = core.step_c220_vsub_word(C220_CAPTURED_VSUB_WORD).unwrap();
        assert_eq!(step.stores.len(), 32);
        assert_eq!(step.source_0_bytes[..128], x);
        assert_eq!(step.source_1_bytes[..128], y);
        let expected = (0..32_u32)
            .flat_map(|lane| (lane as f32 - 0.5).to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(core.ub().read_known(0x100, 128).unwrap(), expected);
        let mut fixture = Fixture::two_inputs(&x, &y);
        core.scalar_mut().machine_mut().set_xreg(12, 0).unwrap();
        core.step_c220_output_word(C220_SUB_VECTOR_TO_MTE3_SET_FLAG_WORD, &mut fixture.source)
            .unwrap();
        core.step_c220_output_word(C220_SUB_VECTOR_TO_MTE3_WAIT_FLAG_WORD, &mut fixture.source)
            .unwrap();
        let machine = core.scalar_mut().machine_mut();
        machine.set_xreg(12, 0x100).unwrap();
        machine.set_xreg(8, 0x2000).unwrap();
        machine.set_xreg(4, 0x40010).unwrap();
        let copied = core
            .step_c220_output_word(CAPTURED_C220_SUB_MOV_UB_TO_OUT_WORD, &mut fixture.source)
            .unwrap();
        assert!(matches!(
            copied.action,
            C220OutputAction::CopyToHbm {
                source_address: 0x100,
                destination_address: 0x2000,
                ..
            }
        ));
        assert_eq!(
            fixture.source.regions().read_known_at(0x2000, 128).unwrap(),
            expected
        );
    }

    #[test]
    fn c220_output_flags_and_transfer_reject_invalid_order_without_mutation() {
        let mut fixture = Fixture::two_inputs(&[0; 128], &[0; 128]);
        let mut core = stepper(Architecture::Dav2201, 0x3000);
        let before = core.clone();
        assert!(matches!(
            core.step_c220_output_word(C220_VECTOR_TO_MTE3_WAIT_FLAG_WORD, &mut fixture.source),
            Err(MteStepperError::OutputWaitWithoutFlag { flag_id: 0 })
        ));
        assert_eq!(core, before);
        assert!(matches!(
            core.step_c220_output_word(C220_VECTOR_TO_MTE3_SET_FLAG_WORD, &mut fixture.source),
            Err(MteStepperError::OutputNotProduced)
        ));
        assert_eq!(core, before);
        assert!(matches!(
            core.step_c220_output_word(CAPTURED_C220_MOV_UB_TO_OUT_WORD, &mut fixture.source),
            Err(MteStepperError::OutputNotReady)
        ));
        assert_eq!(core, before);
        assert!(matches!(
            core.step_c220_output_word(C220_VECTOR_TO_MTE2_SET_FLAG_WORD, &mut fixture.source),
            Err(MteStepperError::OutputNotProduced)
        ));
        assert_eq!(core, before);
        assert!(matches!(
            core.step_c220_output_word(C220_VECTOR_TO_MTE2_WAIT_FLAG0_WORD, &mut fixture.source),
            Err(MteStepperError::ReuseWaitWithoutFlag { flag_id: 0 })
        ));
        assert_eq!(core, before);
        assert!(matches!(
            core.step_c220_output_word(C220_MTE3_TO_VECTOR_SET_FLAG_WORD, &mut fixture.source),
            Err(MteStepperError::OutputNotCopied)
        ));
        assert_eq!(core, before);
        assert!(matches!(
            core.step_c220_output_word(C220_MTE3_TO_VECTOR_WAIT_FLAG_WORD, &mut fixture.source),
            Err(MteStepperError::CompletionWaitWithoutFlag { flag_id: 0 })
        ));
        assert_eq!(core, before);
        core.unsignaled_output = Some(C220OutputTile {
            source_address: 0x100,
            bytes: 128,
        });
        let before = core.clone();
        assert!(matches!(
            core.step_c220_output_word(C220_VECTOR_TO_MTE3_SET_FLAG_WORD, &mut fixture.source),
            Err(MteStepperError::Ub(UbReplayError::UnknownByte {
                address: 0x100
            }))
        ));
        assert_eq!(core, before);
        core.ub
            .write_states(0x100, &[MemoryByteState::Known(7); 128])
            .unwrap();
        core.step_c220_output_word(C220_VECTOR_TO_MTE3_SET_FLAG_WORD, &mut fixture.source)
            .unwrap();
        let before = core.clone();
        assert!(matches!(
            core.step_c220_output_word(C220_VECTOR_TO_MTE3_SET_FLAG_WORD, &mut fixture.source),
            Err(MteStepperError::OutputFlagAlreadySet { flag_id: 0 })
        ));
        assert_eq!(core, before);
        core.step_c220_output_word(C220_VECTOR_TO_MTE3_WAIT_FLAG_WORD, &mut fixture.source)
            .unwrap();
        let machine = core.scalar_mut().machine_mut();
        machine.set_xreg(14, 0x101).unwrap();
        machine.set_xreg(10, 0x2000).unwrap();
        machine.set_xreg(3, 0x40010).unwrap();
        let before = core.clone();
        assert!(matches!(
            core.step_c220_output_word(CAPTURED_C220_MOV_UB_TO_OUT_WORD, &mut fixture.source),
            Err(MteStepperError::OutputTileMismatch)
        ));
        assert_eq!(core, before);
        assert!(fixture.source.regions().read_known_at(0x2000, 128).is_err());
        core.scalar_mut().machine_mut().set_xreg(14, 0x100).unwrap();
        core.scalar_mut()
            .machine_mut()
            .set_xreg(10, 0x2080)
            .unwrap();
        let before = core.clone();
        assert!(matches!(
            core.step_c220_output_word(CAPTURED_C220_MOV_UB_TO_OUT_WORD, &mut fixture.source),
            Err(MteStepperError::Ub(UbReplayError::Destination(
                crate::addressed_replay::ReplayAddressError::Unmapped { .. }
            )))
        ));
        assert_eq!(core, before);
        assert!(fixture.source.regions().read_known_at(0x2000, 128).is_err());
    }

    #[test]
    fn c220_identical_reuse_wait_words_resolve_both_live_ids() {
        let mut fixture = Fixture::two_inputs(&[0; 128], &[0; 128]);
        let mut core = stepper(Architecture::Dav2201, 0x3000);
        core.mte2_reuse_flags = [true, true];
        core.scalar_mut().machine_mut().set_xreg(2, 0).unwrap();
        assert!(matches!(
            core.step_c220_output_word(C220_VECTOR_TO_MTE2_WAIT_DYNAMIC_WORD, &mut fixture.source)
                .unwrap()
                .action,
            C220OutputAction::WaitMte2ReuseFlag { flag_id: 0 }
        ));
        assert_eq!(core.reuse_flags_set(), [false, true]);
        core.scalar_mut().machine_mut().set_xreg(2, 1).unwrap();
        assert!(matches!(
            core.step_c220_output_word(C220_VECTOR_TO_MTE2_WAIT_DYNAMIC_WORD, &mut fixture.source)
                .unwrap()
                .action,
            C220OutputAction::WaitMte2ReuseFlag { flag_id: 1 }
        ));
        assert_eq!(core.reuse_flags_set(), [false, false]);
    }

    #[test]
    fn c220_vadd_rejects_unverified_parameters_and_unknown_input_atomically() {
        let mut core = stepper(Architecture::Dav2201, 0x3000);
        let machine = core.scalar_mut().machine_mut();
        machine.set_xreg(8, C220_CAPTURED_VADD_CONTROL).unwrap();
        machine.set_xreg(13, 0).unwrap();
        machine.set_xreg(14, 0x80).unwrap();
        machine.set_xreg(16, 0x100).unwrap();
        machine.set_spr_value(3, 0).unwrap();
        machine.set_spr_value(100, 0x5555_5555).unwrap();
        machine.set_spr_value(101, 0).unwrap();

        core.scalar_mut().machine_mut().set_xreg(8, 0).unwrap();
        let before = core.clone();
        assert!(matches!(
            core.step_c220_vadd_word(C220_CAPTURED_VADD_WORD),
            Err(MteStepperError::Vector(
                C220CapturedVectorError::UnsupportedVaddControl { control: 0 }
            ))
        ));
        assert_eq!(core, before);

        core.scalar_mut()
            .machine_mut()
            .set_xreg(8, C220_CAPTURED_VADD_CONTROL)
            .unwrap();
        core.scalar_mut()
            .machine_mut()
            .set_spr_value(100, 31)
            .unwrap();
        let before = core.clone();
        assert!(matches!(
            core.step_c220_vadd_word(C220_CAPTURED_VADD_WORD),
            Err(MteStepperError::Vector(
                C220CapturedVectorError::UnsupportedVaddMask { .. }
            ))
        ));
        assert_eq!(core, before);

        core.scalar_mut()
            .machine_mut()
            .set_spr_value(100, 0x5555_5555)
            .unwrap();
        let before = core.clone();
        assert!(matches!(
            core.step_c220_vadd_word(C220_CAPTURED_VADD_WORD),
            Err(MteStepperError::Vector(C220CapturedVectorError::Ub(
                UbReplayError::UnknownByte { address: 0 }
            )))
        ));
        assert_eq!(core, before);

        let mut c310 = stepper(Architecture::Dav3510, 0x3000);
        let before = c310.clone();
        assert!(matches!(
            c310.step_c220_vadd_word(C220_CAPTURED_VADD_WORD),
            Err(MteStepperError::UnsupportedWord { .. })
        ));
        assert_eq!(c310, before);
    }

    #[test]
    fn c310_input_transfers_commit_both_windows_on_one_wait() {
        let input: Vec<u8> = (0..128).collect();
        let fixture = Fixture::new(&input, false);
        let mut core = stepper(Architecture::Dav3510, 0x3000);
        let machine = core.scalar_mut().machine_mut();
        machine.set_xreg(22, 0).unwrap();
        machine.set_xreg(24, 0x3000).unwrap();
        machine.set_xreg(23, 0x0400_0001_0000_0010).unwrap();
        machine.set_xreg(11, 0x0000_8000_0000_0080).unwrap();
        let first = core.step_mte_word(0x74ad_8bae, &fixture.source).unwrap();
        assert!(matches!(
            first.action,
            MteAction::Issue {
                source_address: 0x3000,
                destination_address: 0,
                planned_bytes: 128,
                pending_count: 1,
            }
        ));
        core.scalar_mut()
            .machine_mut()
            .set_xreg(22, 0x3000)
            .unwrap();
        core.scalar_mut().machine_mut().set_xreg(25, 0x80).unwrap();
        let second = core.step_mte_word(0x74b3_6bae, &fixture.source).unwrap();
        assert!(matches!(
            second.action,
            MteAction::Issue {
                source_address: 0x3000,
                destination_address: 0x80,
                planned_bytes: 128,
                pending_count: 2,
            }
        ));
        assert_eq!(core.ub().tracked_bytes(), 0);
        core.step_mte_word(MTE2_TO_SCALAR_SET_FLAG0_WORD, &fixture.source)
            .unwrap();
        core.step_mte_word(MTE2_TO_SCALAR_WAIT_FLAG0_WORD, &fixture.source)
            .unwrap();
        assert_eq!(core.ub().read_known(0, 128).unwrap(), input);
        assert_eq!(core.ub().read_known(0x80, 128).unwrap(), input);
    }

    #[test]
    fn c310_add_input_words_use_distinct_register_selectors() {
        let input: Vec<u8> = (0..128).collect();
        let fixture = Fixture::new(&input, false);
        let mut core = stepper(Architecture::Dav3510, 0x3000);
        let machine = core.scalar_mut().machine_mut();
        machine.set_xreg(25, 0).unwrap();
        machine.set_xreg(0, 0x0400_0001_0000_0010).unwrap();
        machine.set_xreg(8, 0x0000_8000_0000_0080).unwrap();
        let x_issue = core
            .step_mte_word(C310_ADD_MOV_ALIGN_X_WORD, &fixture.source)
            .unwrap();
        assert!(matches!(
            x_issue.action,
            MteAction::Issue {
                source_address: 0x3000,
                destination_address: 0,
                planned_bytes: 128,
                pending_count: 1,
            }
        ));
        let machine = core.scalar_mut().machine_mut();
        machine.set_xreg(0, 0x3000).unwrap();
        machine.set_xreg(1, 0x0400_0001_0000_0010).unwrap();
        machine.set_xreg(2, 0x80).unwrap();
        let y_issue = core
            .step_mte_word(C310_ADD_MOV_ALIGN_Y_WORD, &fixture.source)
            .unwrap();
        assert!(matches!(
            y_issue.action,
            MteAction::Issue {
                source_address: 0x3000,
                destination_address: 0x80,
                planned_bytes: 128,
                pending_count: 2,
            }
        ));
        core.step_mte_word(MTE2_TO_SCALAR_SET_FLAG0_WORD, &fixture.source)
            .unwrap();
        core.step_mte_word(MTE2_TO_SCALAR_WAIT_FLAG0_WORD, &fixture.source)
            .unwrap();
        assert_eq!(core.ub().read_known(0, 128).unwrap(), input);
        assert_eq!(core.ub().read_known(0x80, 128).unwrap(), input);
    }

    #[test]
    fn c310_add_mte2_windows_feed_the_captured_vldi_v0_image() {
        let x = (0..128).collect::<Vec<u8>>();
        let y = (0..128).map(|value| 255 - value).collect::<Vec<u8>>();
        let fixture = Fixture::two_inputs(&x, &y);
        let mut core = stepper(Architecture::Dav3510, 0x3000);
        let machine = core.scalar_mut().machine_mut();
        machine.set_xreg(25, 0).unwrap();
        machine.set_xreg(0, 0x0400_0001_0000_0010).unwrap();
        machine.set_xreg(8, 0x0000_8000_0000_0080).unwrap();
        core.step_mte_word(C310_ADD_MOV_ALIGN_X_WORD, &fixture.source)
            .unwrap();
        let machine = core.scalar_mut().machine_mut();
        machine.set_xreg(0, 0x4000).unwrap();
        machine.set_xreg(1, 0x0400_0001_0000_0010).unwrap();
        machine.set_xreg(2, 0x80).unwrap();
        core.step_mte_word(C310_ADD_MOV_ALIGN_Y_WORD, &fixture.source)
            .unwrap();
        core.step_mte_word(MTE2_TO_SCALAR_SET_FLAG0_WORD, &fixture.source)
            .unwrap();
        core.step_mte_word(MTE2_TO_SCALAR_WAIT_FLAG0_WORD, &fixture.source)
            .unwrap();
        assert_eq!(core.ub().read_known(0, 128).unwrap(), x);
        assert_eq!(core.ub().read_known(0x80, 128).unwrap(), y);

        let mut vldi_xregs = [0_u64; 32];
        vldi_xregs[4] = 0;
        let mut rvec = C310RvecValueMachine::from_vector_words(vec![vec![0; 64]; 2]).unwrap();
        let step = rvec
            .execute_captured_vldi_word(
                0x10d0_db00,
                C310_CAPTURED_VLDI_V0_WORD,
                &vldi_xregs,
                core.ub(),
            )
            .unwrap();
        assert_eq!(&step.loaded_bytes[..128], x);
        assert_eq!(&step.loaded_bytes[128..], y);
    }

    #[test]
    fn unknown_mte_words_and_cross_architecture_words_do_not_advance() {
        let fixture = Fixture::new(&[0x5a; 32], false);
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut core = stepper(architecture, 0x3000);
            let before = core.clone();
            assert!(matches!(
                core.step_mte_word(0x7000_0000, &fixture.source),
                Err(MteStepperError::UnsupportedWord {
                    pc: 0x1000,
                    word: 0x7000_0000,
                })
            ));
            assert_eq!(core, before);
            let wrong_arch_word = if architecture == Architecture::Dav2201 {
                C310_TILING_MOV_ALIGN_WORD
            } else {
                CAPTURED_C220_TILING_MOV_OUT_TO_UB_WORD
            };
            assert!(matches!(
                core.step_mte_word(wrong_arch_word, &fixture.source),
                Err(MteStepperError::UnsupportedWord { .. })
            ));
            assert_eq!(core, before);
        }
    }
}
