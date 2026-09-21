use serde::Serialize;
use thiserror::Error;

use crate::device::architecture::Architecture;
use crate::execution::buffer_c310::C310BufferDisposition;
use crate::execution::machine::{ScalarInstructionError, ScalarInstructionStep, ScalarMemoryBus};
use crate::execution::predicate_buffer_c310::{C310PushPbDisposition, C310PushPbStep};
use crate::execution::stepper::{ScalarProgramStep, ScalarStepper};
use crate::execution::vec_queue_c310::{C310VfQueueDisposition, C310VfQueueStep};
use crate::instruction::flow::{
    C310BufferStep, DcciStep, DsbStep, FlagInstruction, FlagOperation, PipelineBarrierScope,
    PipelineBarrierStep,
};
use crate::instruction::mte_c220::{
    C220DmaMovDescriptor, C220MovInstruction, C220MovOutToUbDescriptor, C220MovOutToUbError,
    C220MovOutToUbSegment,
};
use crate::instruction::mte_c310::{
    C310CapturedMovAlignDecode, C310CapturedMovAlignError, C310MovAlignRegisterSelectors,
};
use crate::instruction::vec_c220::{
    C220_VECTOR_TILE_BYTES, C220Fp32Addresses, C220Fp32Step, C220MovevInstruction, C220MovevStep,
    C220VecArithmeticHint, C220VecArithmeticOperation, C220VectorError, decode_c220_fp32_control,
    decode_c220_fp32_mask, decode_c220_movev_control, decode_c220_tile_mask,
    execute_c220_fp32_to_ub, execute_c220_movev_to_ub,
};
use crate::memory::c220_scalar_address_space::{
    C220_UB_BYTES, C220ScalarRoute, classify_c220_scalar_address,
};
use crate::memory::c310_scalar_address_space::{
    C310_UB_ROUTE_BYTES, C310ScalarRoute, classify_c310_scalar_address,
};
use crate::memory::mapped::MappedMemory;
use crate::memory::sparse::MemoryByteState;
use crate::memory::ub::{C220PreparedOutput, UbMemory, UbMemoryError, UbTransferResult};

#[cfg(test)]
mod test_words {
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
}

#[cfg(test)]
pub use test_words::*;
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
    #[error("scalar local-address roots are unavailable")]
    MissingLocalRoots,
    #[error("scalar local address {address:#x} is unsupported")]
    UnsupportedLocal { address: u64 },
    #[error(transparent)]
    Ub(#[from] UbMemoryError),
    #[error("scalar memory backend: {0}")]
    Fallback(#[source] E),
}

struct UbScalarBus<'a, B> {
    ub: &'a mut UbMemory,
    fallback: &'a mut B,
    architecture: Architecture,
    local_roots: Option<(u64, u64)>,
}

enum ScalarBusRoute {
    Fallback(u64),
    Ub(u64),
}

impl<B: ScalarMemoryBus> UbScalarBus<'_, B> {
    fn route(
        &self,
        address: u64,
        bytes: usize,
    ) -> Result<ScalarBusRoute, UbScalarBusError<B::Error>> {
        if bytes == 0 {
            return Ok(ScalarBusRoute::Fallback(address));
        }
        let length = u64::try_from(bytes).map_err(|_| UbScalarBusError::AddressOverflow)?;
        let end = address
            .checked_add(length)
            .ok_or(UbScalarBusError::AddressOverflow)?;
        let (spr67, spr68) = self
            .local_roots
            .ok_or(UbScalarBusError::MissingLocalRoots)?;
        match self.architecture {
            Architecture::Dav2201 => {
                let start = classify_c220_scalar_address(address, spr67, spr68);
                let last = classify_c220_scalar_address(end - 1, spr67, spr68);
                match (start, last) {
                    (C220ScalarRoute::Hbm, C220ScalarRoute::Hbm) => {
                        Ok(ScalarBusRoute::Fallback(address))
                    }
                    (C220ScalarRoute::Ub(offset), C220ScalarRoute::Ub(_))
                        if length <= C220_UB_BYTES.saturating_sub(offset) =>
                    {
                        Ok(ScalarBusRoute::Ub(offset))
                    }
                    (C220ScalarRoute::Unsupported, _) => {
                        Err(UbScalarBusError::UnsupportedLocal { address })
                    }
                    _ => Err(UbScalarBusError::AliasBoundary { address, bytes }),
                }
            }
            Architecture::Dav3510 => {
                let start = classify_c310_scalar_address(address, spr67, spr68);
                let last = classify_c310_scalar_address(end - 1, spr67, spr68);
                match (start, last) {
                    (C310ScalarRoute::Hbm(mapped), C310ScalarRoute::Hbm(last_mapped))
                        if mapped.checked_add(length - 1) == Some(last_mapped) =>
                    {
                        Ok(ScalarBusRoute::Fallback(mapped))
                    }
                    (C310ScalarRoute::Ub(offset), C310ScalarRoute::Ub(_))
                        if length <= C310_UB_ROUTE_BYTES.saturating_sub(offset) =>
                    {
                        Ok(ScalarBusRoute::Ub(offset))
                    }
                    (C310ScalarRoute::Unsupported, _) => {
                        Err(UbScalarBusError::UnsupportedLocal { address })
                    }
                    _ => Err(UbScalarBusError::AliasBoundary { address, bytes }),
                }
            }
        }
    }
}

impl<B: ScalarMemoryBus> ScalarMemoryBus for UbScalarBus<'_, B> {
    type Error = UbScalarBusError<B::Error>;

    fn read(&mut self, address: u64, destination: &mut [u8]) -> Result<(), Self::Error> {
        match self.route(address, destination.len())? {
            ScalarBusRoute::Ub(offset) => {
                destination.copy_from_slice(&self.ub.read_known(offset, destination.len())?);
                Ok(())
            }
            ScalarBusRoute::Fallback(mapped) => self
                .fallback
                .read(mapped, destination)
                .map_err(UbScalarBusError::Fallback),
        }
    }

    fn write(&mut self, address: u64, source: &[u8]) -> Result<(), Self::Error> {
        match self.route(address, source.len())? {
            ScalarBusRoute::Ub(offset) => {
                let states = source
                    .iter()
                    .copied()
                    .map(MemoryByteState::Known)
                    .collect::<Vec<_>>();
                self.ub.write_states(offset, &states)?;
                Ok(())
            }
            ScalarBusRoute::Fallback(mapped) => self
                .fallback
                .write(mapped, source)
                .map_err(UbScalarBusError::Fallback),
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
    C220(C220Mte2TransferPlan),
    C310(C310CapturedMovAlignDecode),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct C220Mte2TransferPlan {
    pub descriptor: C220MovOutToUbDescriptor,
    pub source_address: u64,
    pub destination_address: u64,
    pub bytes: usize,
    pub dma_mode_word: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct C220Mte3TransferPlan {
    pub descriptor: C220DmaMovDescriptor,
    pub source_address: u64,
    pub destination_address: u64,
    pub bytes: usize,
    pub dma_mode_word: u64,
}

impl C220Mte2TransferPlan {
    pub fn descriptor_segments(self) -> Result<Vec<C220MovOutToUbSegment>, C220MovOutToUbError> {
        self.descriptor
            .segments(self.source_address, self.destination_address)
    }
}

impl PendingMte2 {
    fn commit(
        self,
        ub: &mut UbMemory,
        source: &MappedMemory,
    ) -> Result<UbTransferResult, UbMemoryError> {
        match self {
            Self::C220(plan) => ub.copy_c220_mov_out_to_ub(
                source,
                plan.descriptor,
                plan.source_address,
                plan.destination_address,
            ),
            Self::C310(decoded) => ub.copy_c310_mov_align_hbm_to_ub(source, decoded),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MteCoreStepper {
    scalar: ScalarStepper,
    ub: UbMemory,
    c220_isa_instance_index: u32,
    pending_mte2: Vec<PendingMte2>,
    flag0_set: bool,
    vector_flags: [Option<PendingMte2>; 2],
    unsignaled_output: Option<C220OutputToken>,
    mte3_flags: [Option<C220OutputToken>; 4],
    mte2_reuse_flags: [bool; 2],
    ready_output: Option<C220OutputToken>,
    copied_output: Option<C220OutputToken>,
    mte3_completion_flags: [Option<C220OutputToken>; 4],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct C220OutputToken {
    source_address: u64,
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
    #[error("pipeline barrier at PC {pc:#x} has outstanding modeled work")]
    BarrierBusy { pc: u64 },
    #[error("C220 output was not produced before its flag")]
    OutputNotProduced,
    #[error("C220 output is still waiting for a flag")]
    UnsignaledOutput,
    #[error("C220 output flag {flag_id} is already set")]
    OutputFlagAlreadySet { flag_id: u8 },
    #[error("C220 output flag {flag_id} was not set before wait")]
    OutputWaitWithoutFlag { flag_id: u8 },
    #[error("C220 output flag ID {flag_id} is outside the modeled range")]
    UnsupportedOutputFlagId { flag_id: u32 },
    #[error("C220 output transfer has no matching completed flag")]
    OutputNotReady,
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
    Ub(#[from] UbMemoryError),
    #[error(transparent)]
    Vector(#[from] C220VectorError),
}

impl MteCoreStepper {
    pub fn new(scalar: ScalarStepper, ub: UbMemory) -> Self {
        Self {
            scalar,
            ub,
            c220_isa_instance_index: 0,
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

    pub fn set_c220_isa_instance_index(&mut self, index: u32) {
        self.c220_isa_instance_index = index;
    }

    pub fn scalar_mut(&mut self) -> &mut ScalarStepper {
        &mut self.scalar
    }

    pub const fn ub(&self) -> &UbMemory {
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

    pub fn step_barrier_word(&mut self, word: u32) -> Result<ScalarProgramStep, MteStepperError> {
        let pc = self.scalar.pc();
        if self.scalar.is_halted() {
            return Err(MteStepperError::ProgramEnded { pc });
        }
        let architecture = self.scalar.machine().architecture();
        let barrier = PipelineBarrierStep::decode(architecture, pc, word)
            .filter(|step| step.scope == PipelineBarrierScope::All)
            .ok_or(MteStepperError::UnsupportedWord { pc, word })?;
        if !self.pending_mte2.is_empty()
            || self.flag0_set
            || self.vector_flags.iter().any(Option::is_some)
            || self.output_buffer_busy()
            || self.mte2_reuse_flags.iter().any(|set| *set)
        {
            return Err(MteStepperError::BarrierBusy { pc });
        }
        self.scalar.advance_sequential();
        Ok(ScalarProgramStep {
            pc,
            word,
            next_pc: self.scalar.pc(),
            instruction: ScalarInstructionStep::Barrier(barrier),
            halted_after: false,
        })
    }

    pub fn step_scalar_word_with_ub<B: ScalarMemoryBus>(
        &mut self,
        word: u32,
        fallback: &mut B,
    ) -> Result<ScalarProgramStep, ScalarInstructionError<UbScalarBusError<B::Error>>> {
        let machine = self.scalar.machine();
        let mut bus = UbScalarBus {
            ub: &mut self.ub,
            fallback,
            architecture: machine.architecture(),
            local_roots: machine.spr_value(67).zip(machine.spr_value(68)),
        };
        self.scalar.step_word(word, &mut bus)
    }

    pub fn step_c310_vf_words_with_ub<B: ScalarMemoryBus>(
        &mut self,
        first_word: u32,
        second_word: u32,
        fallback: &mut B,
    ) -> Result<ScalarProgramStep, ScalarInstructionError<UbScalarBusError<B::Error>>> {
        let machine = self.scalar.machine();
        let mut bus = UbScalarBus {
            ub: &mut self.ub,
            fallback,
            architecture: machine.architecture(),
            local_roots: machine.spr_value(67).zip(machine.spr_value(68)),
        };
        self.scalar
            .step_c310_vf_words(first_word, second_word, &mut bus)
    }

    pub fn step_mte_word(
        &mut self,
        word: u32,
        source: &MappedMemory,
    ) -> Result<MteProgramStep, MteStepperError> {
        let pc = self.scalar.pc();
        if self.scalar.is_halted() {
            return Err(MteStepperError::ProgramEnded { pc });
        }
        let flag = FlagInstruction::decode(self.scalar.machine().architecture(), word);
        let route = flag.map(|instruction| {
            (
                instruction.source_pipe_code,
                instruction.trigger_pipe_code,
                instruction.operation,
            )
        });
        let action = match route {
            Some((4, 0, FlagOperation::Set))
                if flag
                    .unwrap()
                    .resolve(pc, self.scalar.machine().xregs())
                    .flag_id
                    == 0 =>
            {
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
            Some((4, 0, FlagOperation::Wait))
                if flag
                    .unwrap()
                    .resolve(pc, self.scalar.machine().xregs())
                    .flag_id
                    == 0 =>
            {
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
            Some((4, 1, FlagOperation::Set))
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
            Some((4, 1, FlagOperation::Wait))
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

    pub fn step_c220_movev_word(&mut self, word: u32) -> Result<C220MovevStep, MteStepperError> {
        let pc = self.scalar.pc();
        if self.scalar.is_halted() {
            return Err(MteStepperError::ProgramEnded { pc });
        }
        if self.scalar.machine().architecture() != Architecture::Dav2201 {
            return Err(MteStepperError::UnsupportedWord { pc, word });
        }
        let instruction = C220MovevInstruction::decode(word)
            .ok_or(MteStepperError::UnsupportedWord { pc, word })?;
        let element_bytes = instruction
            .supported_element_bytes()
            .ok_or(MteStepperError::UnsupportedWord { pc, word })?;
        if self.output_buffer_busy() {
            return Err(MteStepperError::OutputDependencyOutstanding);
        }
        let machine = self.scalar.machine();
        let xregs = machine.xregs();
        let control = xregs[usize::from(instruction.control_register)];
        let control = decode_c220_movev_control(control)?;
        let mask_control = machine
            .spr_value(3)
            .ok_or(C220VectorError::MissingMaskState)?;
        let mask0 = machine
            .spr_value(100)
            .ok_or(C220VectorError::MissingMaskState)?;
        let mask1 = machine
            .spr_value(101)
            .ok_or(C220VectorError::MissingMaskState)?;
        let lane_count = C220_VECTOR_TILE_BYTES / usize::from(element_bytes);
        let active_mask = decode_c220_tile_mask(mask_control, mask0, mask1, lane_count)?;
        let destination_address = xregs[usize::from(instruction.destination_register)];
        let scalar_word = xregs[usize::from(instruction.source_register)] as u32;
        let step = execute_c220_movev_to_ub(
            pc,
            word,
            control,
            destination_address,
            scalar_word,
            &active_mask,
            &mut self.ub,
        )?;
        self.scalar.advance_sequential();
        Ok(step)
    }

    pub(crate) fn resolve_c220_vector_flag_id(
        &self,
        pc: u64,
        word: u32,
    ) -> Result<u8, MteStepperError> {
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

    pub fn step_c220_vadd_word(&mut self, word: u32) -> Result<C220Fp32Step, MteStepperError> {
        self.step_c220_fp32_word(word, Some(C220VecArithmeticOperation::Add))
    }

    pub fn step_c220_vsub_word(&mut self, word: u32) -> Result<C220Fp32Step, MteStepperError> {
        self.step_c220_fp32_word(word, Some(C220VecArithmeticOperation::Subtract))
    }

    pub fn step_c220_vmul_word(&mut self, word: u32) -> Result<C220Fp32Step, MteStepperError> {
        self.step_c220_fp32_word(word, Some(C220VecArithmeticOperation::Multiply))
    }

    pub fn step_c220_vector_word(&mut self, word: u32) -> Result<C220Fp32Step, MteStepperError> {
        self.step_c220_fp32_word(word, None)
    }

    fn step_c220_fp32_word(
        &mut self,
        word: u32,
        expected_operation: Option<C220VecArithmeticOperation>,
    ) -> Result<C220Fp32Step, MteStepperError> {
        let pc = self.scalar.pc();
        if self.scalar.is_halted() {
            return Err(MteStepperError::ProgramEnded { pc });
        }
        if self.scalar.machine().architecture() != Architecture::Dav2201 {
            return Err(MteStepperError::UnsupportedWord { pc, word });
        }
        if self.output_buffer_busy() {
            return Err(MteStepperError::OutputDependencyOutstanding);
        }
        let hint = C220VecArithmeticHint::from_word(word)
            .filter(|hint| hint.has_fp32_value_path())
            .filter(|hint| expected_operation.is_none_or(|operation| hint.operation == operation))
            .ok_or(MteStepperError::UnsupportedWord { pc, word })?;
        let machine = self.scalar.machine();
        let xregs = machine.xregs();
        let control = xregs[usize::from(hint.x_register_index_8)];
        let control = decode_c220_fp32_control(control)?;
        let ctrl = machine.spr_value(3);
        let mask0 = machine.spr_value(100);
        let mask1 = machine.spr_value(101);
        let active_mask = decode_c220_fp32_mask(
            ctrl.ok_or(C220VectorError::MissingMaskState)?,
            mask0.ok_or(C220VectorError::MissingMaskState)?,
            mask1.ok_or(C220VectorError::MissingMaskState)?,
        )?;
        let step = execute_c220_fp32_to_ub(
            pc,
            word,
            control,
            C220Fp32Addresses {
                source_0: xregs[usize::from(hint.x_register_index_4)],
                source_1: xregs[usize::from(hint.x_register_index_6)],
                destination: xregs[usize::from(hint.x_register_index_0)],
            },
            &active_mask,
            &mut self.ub,
        )?;
        self.unsignaled_output = Some(C220OutputToken {
            source_address: step.destination_address,
        });
        self.scalar.advance_sequential();
        Ok(step)
    }

    pub fn step_c220_output_word(
        &mut self,
        word: u32,
        destination: &mut MappedMemory,
    ) -> Result<C220OutputStep, MteStepperError> {
        self.step_c220_output_word_impl(word, Some(destination))
            .map(|(step, _)| step)
    }

    pub(crate) fn step_c220_output_word_deferred(
        &mut self,
        word: u32,
    ) -> Result<(C220OutputStep, Option<C220PreparedOutput>), MteStepperError> {
        self.step_c220_output_word_impl(word, None)
    }

    fn step_c220_output_word_impl(
        &mut self,
        word: u32,
        destination: Option<&mut MappedMemory>,
    ) -> Result<(C220OutputStep, Option<C220PreparedOutput>), MteStepperError> {
        let pc = self.scalar.pc();
        if self.scalar.is_halted() {
            return Err(MteStepperError::ProgramEnded { pc });
        }
        if self.scalar.machine().architecture() != Architecture::Dav2201 {
            return Err(MteStepperError::UnsupportedWord { pc, word });
        }
        let route = FlagInstruction::decode(Architecture::Dav2201, word).map(|instruction| {
            (
                instruction.source_pipe_code,
                instruction.trigger_pipe_code,
                instruction.operation,
            )
        });
        let mut prepared_output = None;
        let action = match route {
            Some((1, 5, FlagOperation::Set)) => {
                let flag_id = self.resolve_c220_output_flag_id(pc, word, 4)?;
                if self.mte3_flags[usize::from(flag_id)].is_some() {
                    return Err(MteStepperError::OutputFlagAlreadySet { flag_id });
                }
                let token = self
                    .unsignaled_output
                    .ok_or(MteStepperError::OutputNotProduced)?;
                self.unsignaled_output = None;
                self.mte3_flags[usize::from(flag_id)] = Some(token);
                C220OutputAction::SetMte3Flag {
                    flag_id,
                    source_address: token.source_address,
                }
            }
            Some((1, 4, FlagOperation::Set)) => {
                let flag_id = self.resolve_c220_output_flag_id(pc, word, 2)?;
                if self.mte2_reuse_flags[usize::from(flag_id)] {
                    return Err(MteStepperError::ReuseFlagAlreadySet { flag_id });
                }
                if self.unsignaled_output.is_some() {
                    return Err(MteStepperError::UnsignaledOutput);
                }
                if self.mte3_flags.iter().all(Option::is_none) && self.ready_output.is_none() {
                    return Err(MteStepperError::OutputNotProduced);
                }
                self.mte2_reuse_flags[usize::from(flag_id)] = true;
                C220OutputAction::SetMte2ReuseFlag { flag_id }
            }
            Some((1, 4, FlagOperation::Wait)) => {
                let flag_id = self.resolve_c220_output_flag_id(pc, word, 2)?;
                if !self.mte2_reuse_flags[usize::from(flag_id)] {
                    return Err(MteStepperError::ReuseWaitWithoutFlag { flag_id });
                }
                self.mte2_reuse_flags[usize::from(flag_id)] = false;
                C220OutputAction::WaitMte2ReuseFlag { flag_id }
            }
            Some((1, 5, FlagOperation::Wait)) => {
                let flag_id = self.resolve_c220_output_flag_id(pc, word, 4)?;
                if self.ready_output.is_some() {
                    return Err(MteStepperError::OutputDependencyOutstanding);
                }
                let token = self.mte3_flags[usize::from(flag_id)]
                    .take()
                    .ok_or(MteStepperError::OutputWaitWithoutFlag { flag_id })?;
                self.ready_output = Some(token);
                C220OutputAction::WaitMte3Flag {
                    flag_id,
                    source_address: token.source_address,
                }
            }
            None if C220DmaMovDescriptor::is_word(word) => {
                self.ready_output.ok_or(MteStepperError::OutputNotReady)?;
                if self.copied_output.is_some()
                    || self.mte3_completion_flags.iter().any(Option::is_some)
                {
                    return Err(MteStepperError::OutputDependencyOutstanding);
                }
                let plan = self.preview_c220_mte3_transfer(word)?;
                let prepared = self.ub.prepare_c220_mov_ub_to_hbm(
                    plan.descriptor,
                    plan.source_address,
                    plan.destination_address,
                )?;
                if let Some(destination) = destination {
                    destination
                        .write_segments_at(&prepared.writes)
                        .map_err(UbMemoryError::from)?;
                }
                let transfer = prepared.result;
                prepared_output = Some(prepared);
                self.ready_output = None;
                self.copied_output = Some(C220OutputToken {
                    source_address: plan.source_address,
                });
                C220OutputAction::CopyToHbm {
                    source_address: plan.source_address,
                    destination_address: plan.destination_address,
                    transfer,
                }
            }
            Some((5, 1, FlagOperation::Set)) => {
                let flag_id = self.resolve_c220_output_flag_id(pc, word, 4)?;
                if self.mte3_completion_flags[usize::from(flag_id)].is_some() {
                    return Err(MteStepperError::CompletionFlagAlreadySet { flag_id });
                }
                let token = self.copied_output.ok_or(MteStepperError::OutputNotCopied)?;
                self.copied_output = None;
                self.mte3_completion_flags[usize::from(flag_id)] = Some(token);
                C220OutputAction::SetMte3CompletionFlag {
                    flag_id,
                    source_address: token.source_address,
                }
            }
            Some((5, 1, FlagOperation::Wait)) => {
                let flag_id = self.resolve_c220_output_flag_id(pc, word, 4)?;
                let token = self.mte3_completion_flags[usize::from(flag_id)]
                    .take()
                    .ok_or(MteStepperError::CompletionWaitWithoutFlag { flag_id })?;
                C220OutputAction::WaitMte3CompletionFlag {
                    flag_id,
                    source_address: token.source_address,
                }
            }
            _ => return Err(MteStepperError::UnsupportedWord { pc, word }),
        };
        self.scalar.advance_sequential();
        Ok((
            C220OutputStep {
                pc,
                word,
                next_pc: self.scalar.pc(),
                action,
            },
            prepared_output,
        ))
    }

    fn resolve_c220_output_flag_id(
        &self,
        pc: u64,
        word: u32,
        max_id: u8,
    ) -> Result<u8, MteStepperError> {
        let instruction = FlagInstruction::decode(Architecture::Dav2201, word)
            .filter(|instruction| {
                matches!(
                    (instruction.source_pipe_code, instruction.trigger_pipe_code),
                    (1, 5) | (1, 4) | (5, 1)
                )
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
                if !C220MovOutToUbDescriptor::is_word(word) {
                    return Err(MteStepperError::UnsupportedWord { pc, word });
                }
                let selectors = C220MovInstruction::decode(word)
                    .ok_or(MteStepperError::UnsupportedWord { pc, word })?;
                let destination_address = x[usize::from(selectors.destination_register)];
                let source_address = x[usize::from(selectors.source_register)];
                let descriptor = C220MovOutToUbDescriptor::decode(
                    word,
                    x[usize::from(selectors.descriptor_register)],
                )?;
                let planned_bytes = descriptor
                    .segments(source_address, destination_address)?
                    .len()
                    .checked_mul(32)
                    .ok_or(MteStepperError::TransferSizeOverflow)?;
                let dma_mode_word = if self.c220_isa_instance_index == 0 {
                    0
                } else {
                    machine
                        .spr_value(93)
                        .ok_or(MteStepperError::MissingSpr { pc, index: 93 })?
                };
                Ok((
                    PendingMte2::C220(C220Mte2TransferPlan {
                        descriptor,
                        source_address,
                        destination_address,
                        bytes: planned_bytes,
                        dma_mode_word,
                    }),
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
                let decoded = C310MovAlignRegisterSelectors::from_hbm_to_ub_word(word)
                    .map_err(|_| MteStepperError::UnsupportedWord { pc, word })?
                    .capture(x, spr(105)?, spr(106)?, spr(107)?)
                    .decode_hbm_to_ub(word)?;
                let coordinates = decoded
                    .parameters
                    .coordinates()
                    .map_err(UbMemoryError::from)?;
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

    pub(crate) fn preview_c220_mte2_transfer(
        &self,
        word: u32,
    ) -> Result<C220Mte2TransferPlan, MteStepperError> {
        let pc = self.scalar.pc();
        let (pending, _, _, _) = self.decode_transfer(pc, word)?;
        let PendingMte2::C220(plan) = pending else {
            return Err(MteStepperError::UnsupportedWord { pc, word });
        };
        Ok(plan)
    }

    pub(crate) fn preview_c220_mte3_transfer(
        &self,
        word: u32,
    ) -> Result<C220Mte3TransferPlan, MteStepperError> {
        let pc = self.scalar.pc();
        let machine = self.scalar.machine();
        if machine.architecture() != Architecture::Dav2201 {
            return Err(MteStepperError::UnsupportedWord { pc, word });
        }
        let selectors = C220MovInstruction::decode(word)
            .filter(|_| C220DmaMovDescriptor::is_word(word))
            .ok_or(MteStepperError::UnsupportedWord { pc, word })?;
        let x = machine.xregs();
        let source_address = x[usize::from(selectors.source_register)];
        let destination_address = x[usize::from(selectors.destination_register)];
        let descriptor =
            C220DmaMovDescriptor::decode(word, x[usize::from(selectors.descriptor_register)])
                .map_err(UbMemoryError::from)?;
        let bytes = usize::from(descriptor.burst_count) * usize::from(descriptor.burst_length) * 32;
        let dma_mode_word = if self.c220_isa_instance_index == 0 {
            0
        } else {
            machine
                .spr_value(94)
                .ok_or(MteStepperError::MissingSpr { pc, index: 94 })?
        };
        Ok(C220Mte3TransferPlan {
            descriptor,
            source_address,
            destination_address,
            bytes,
            dma_mode_word,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::machine::{
        ScalarInstructionStep, ScalarMachine, ScalarMemoryExecutionError,
    };
    use crate::instruction::mte_c220::{
        CAPTURED_C220_MOV_OUT_TO_UB_X_WORD, CAPTURED_C220_MOV_OUT_TO_UB_Y_WORD,
        CAPTURED_C220_MOV_UB_TO_OUT_WORD, CAPTURED_C220_SUB_MOV_UB_TO_OUT_WORD,
        CAPTURED_C220_SUB_TILING_MOV_OUT_TO_UB_WORD, CAPTURED_C220_TILING_MOV_OUT_TO_UB_WORD,
    };
    use crate::instruction::mte_c310::{
        C310_ADD_MOV_ALIGN_X_WORD, C310_ADD_MOV_ALIGN_Y_WORD, C310_SUB_TILING_MOV_ALIGN_WORD,
        C310_TILING_MOV_ALIGN_WORD,
    };
    use crate::instruction::rvec::{C310_CAPTURED_VLDI_V0_WORD, C310RvecValueMachine};
    use crate::instruction::vec_c220::{
        C220_CAPTURED_MOVEV_CONTROL, C220_CAPTURED_MOVEV_WORD, C220_CAPTURED_VADD_CONTROL,
        C220_CAPTURED_VADD_WORD, C220_CAPTURED_VMUL_CONTROL, C220_CAPTURED_VMUL_WORD,
        C220_CAPTURED_VSUB_WORD,
    };
    use crate::memory::mapped::MappedMemory;
    use crate::memory::region::MemoryRegion;
    use crate::memory::sparse::{MemoryByteState, SparseMemory};
    use std::io;

    struct Fixture {
        source: MappedMemory,
    }

    impl Fixture {
        fn new(bytes: &[u8], tiling: bool) -> Self {
            let mut regions = if tiling {
                vec![
                    MemoryRegion::unknown(32),
                    MemoryRegion::new(64, bytes.to_vec()).unwrap(),
                ]
            } else {
                vec![
                    MemoryRegion::new(bytes.len() as u64, bytes.to_vec()).unwrap(),
                    MemoryRegion::unknown(32),
                ]
            };
            let pointers: &[u64] = if tiling {
                &[0x2000, 0x3000]
            } else {
                &[0x3000, 0x2000]
            };
            let image: Vec<u8> = pointers.iter().flat_map(|p| p.to_le_bytes()).collect();
            regions.push(MemoryRegion::new(image.len() as u64, image).unwrap());
            let memory = SparseMemory::new(regions, 64, 128);
            let source = MappedMemory::bind(memory, &[pointers[0], pointers[1], 0x1000]).unwrap();
            Self { source }
        }

        fn two_inputs(x: &[u8], y: &[u8]) -> Self {
            let pointers = &[0x3000_u64, 0x4000, 0x2000];
            let image: Vec<u8> = pointers.iter().flat_map(|p| p.to_le_bytes()).collect();
            let regions = vec![
                MemoryRegion::new(x.len() as u64, x.to_vec()).unwrap(),
                MemoryRegion::new(y.len() as u64, y.to_vec()).unwrap(),
                MemoryRegion::unknown(128),
                MemoryRegion::new(image.len() as u64, image).unwrap(),
            ];
            let memory = SparseMemory::new(regions, 512, 512);
            let source = MappedMemory::bind(memory, &[0x3000, 0x4000, 0x2000, 0x1000]).unwrap();
            Self { source }
        }
    }

    fn stepper(architecture: Architecture, source_address: u64) -> MteCoreStepper {
        let mut machine = ScalarMachine::from_pem_initial_state(architecture);
        machine.set_xreg(1, source_address).unwrap();
        match architecture {
            Architecture::Dav2201 => machine.set_xreg(2, 0x10010).unwrap(),
            Architecture::Dav3510 => {
                machine.set_xreg(4, 0x4000_0010).unwrap();
                machine.set_spr_value(67, 0).unwrap();
                machine.set_spr_value(68, 0).unwrap();
            }
        }
        MteCoreStepper::new(ScalarStepper::new(machine, 0x1000), UbMemory::new(512, 256))
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
    fn c310_sub_tiling_transfer_uses_selected_registers_and_wait_visibility() {
        let fixture = Fixture::new(&1024_u32.to_le_bytes(), true);
        let mut core = stepper(Architecture::Dav3510, 0);
        core.scalar_mut().machine_mut().set_xreg(2, 0x3000).unwrap();
        core.scalar_mut()
            .machine_mut()
            .set_xreg(3, 0x4000_0010)
            .unwrap();
        let issue = core
            .step_mte_word(C310_SUB_TILING_MOV_ALIGN_WORD, &fixture.source)
            .unwrap();
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
        core.step_mte_word(MTE2_TO_SCALAR_SET_FLAG0_WORD, &fixture.source)
            .unwrap();
        core.step_mte_word(MTE2_TO_SCALAR_WAIT_FLAG0_WORD, &fixture.source)
            .unwrap();
        assert_eq!(core.ub().read_known(0, 4).unwrap(), 1024_u32.to_le_bytes());
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
            Err(MteStepperError::C220(C220MovOutToUbError::EmptyDescriptor))
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
                        UbMemoryError::UnknownByte { address: 0 }
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
    fn c220_canonical_scalar_store_and_mte_share_ub_state() {
        let mut core = stepper(Architecture::Dav2201, 0);
        let pointer = 0x1000_2200_u64;
        core.scalar_mut()
            .machine_mut()
            .set_xreg(3, pointer)
            .unwrap();
        core.scalar_mut()
            .machine_mut()
            .set_xreg(7, 0x100000)
            .unwrap();
        let step = core
            .step_scalar_word_with_ub(0x04c6_7000, &mut NoMemoryBus)
            .unwrap();
        assert_eq!(step.pc, 0x1000);
        assert_eq!(core.ub().read_known(0, 8).unwrap(), pointer.to_le_bytes());
        let mut alias_bytes = [0; 8];
        let mut bus = UbScalarBus {
            ub: &mut core.ub,
            fallback: &mut NoMemoryBus,
            architecture: Architecture::Dav2201,
            local_roots: Some((0, 0)),
        };
        bus.read(SCALAR_UB_ALIAS_BASE, &mut alias_bytes).unwrap();
        assert_eq!(alias_bytes, pointer.to_le_bytes());
        assert!(matches!(
            bus.write(0x12_ffff, &[1, 2]),
            Err(UbScalarBusError::AliasBoundary { .. })
        ));
        assert_eq!(bus.ub.read_known(0, 8).unwrap(), pointer.to_le_bytes());
    }

    #[test]
    fn c310_canonical_scalar_store_and_alias_share_ub_state() {
        let mut core = stepper(Architecture::Dav3510, 0);
        core.scalar_mut()
            .machine_mut()
            .set_xreg(11, 0x107f40)
            .unwrap();
        core.scalar_mut()
            .machine_mut()
            .set_xreg(12, 0x3344)
            .unwrap();
        let step = core
            .step_scalar_word_with_ub(0x0358_b000, &mut NoMemoryBus)
            .unwrap();
        assert_eq!(step.pc, 0x1000);
        assert_eq!(core.ub().read_known(0x7f40, 2).unwrap(), [0x44, 0x33]);
        let mut fallback = NoMemoryBus;
        let mut bus = UbScalarBus {
            ub: &mut core.ub,
            fallback: &mut fallback,
            architecture: Architecture::Dav3510,
            local_roots: Some((0, 0)),
        };
        let mut actual = [0; 2];
        bus.read(0x87f40, &mut actual).unwrap();
        assert_eq!(actual, [0x44, 0x33]);
        assert!(matches!(
            bus.write(0x13ffff, &[1, 2]),
            Err(UbScalarBusError::AliasBoundary { .. })
        ));
    }

    #[test]
    fn all_pipeline_barrier_requires_modeled_work_to_be_idle() {
        let fixture = Fixture::new(&1024_u32.to_le_bytes(), true);
        for (architecture, transfer_word) in [
            (
                Architecture::Dav2201,
                CAPTURED_C220_TILING_MOV_OUT_TO_UB_WORD,
            ),
            (Architecture::Dav3510, C310_TILING_MOV_ALIGN_WORD),
        ] {
            let mut core = stepper(architecture, 0x3000);
            core.step_mte_word(transfer_word, &fixture.source).unwrap();
            let before = core.clone();
            assert!(matches!(
                core.step_barrier_word(0x40e0_1800),
                Err(MteStepperError::BarrierBusy { pc: 0x1004 })
            ));
            assert_eq!(core, before);
            core.step_mte_word(MTE2_TO_SCALAR_SET_FLAG0_WORD, &fixture.source)
                .unwrap();
            core.step_mte_word(MTE2_TO_SCALAR_WAIT_FLAG0_WORD, &fixture.source)
                .unwrap();
            let barrier = core.step_barrier_word(0x40e0_1800).unwrap();
            assert_eq!(barrier.pc, 0x100c);
            assert_eq!(barrier.next_pc, 0x1010);
            assert!(matches!(
                barrier.instruction,
                ScalarInstructionStep::Barrier(PipelineBarrierStep {
                    scope: PipelineBarrierScope::All,
                    ..
                })
            ));
        }
    }

    #[test]
    fn scalar_ub_alias_writes_preserve_provenance_and_reject_crossings() {
        let mut ub = UbMemory::new(64, 64);
        let mut fallback = NoMemoryBus;
        let mut bus = UbScalarBus {
            ub: &mut ub,
            fallback: &mut fallback,
            architecture: Architecture::Dav3510,
            local_roots: Some((0, 0)),
        };
        assert!(matches!(
            bus.read(SCALAR_UB_ALIAS_BASE + 4, &mut [0; 4]),
            Err(UbScalarBusError::Ub(UbMemoryError::UnknownByte {
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
            Err(UbScalarBusError::UnsupportedLocal { .. })
        ));
        assert!(matches!(
            bus.write(SCALAR_UB_ALIAS_BASE + C310_UB_ROUTE_BYTES - 1, &[5, 6]),
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
            Err(MteStepperError::Ub(UbMemoryError::Source(_)))
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
        core.scalar_mut()
            .machine_mut()
            .set_spr_value(93, 5)
            .unwrap();
        assert_eq!(
            core.preview_c220_mte2_transfer(CAPTURED_C220_MOV_OUT_TO_UB_X_WORD)
                .unwrap()
                .dma_mode_word,
            0
        );
        core.set_c220_isa_instance_index(1);
        assert_eq!(
            core.preview_c220_mte2_transfer(CAPTURED_C220_MOV_OUT_TO_UB_X_WORD)
                .unwrap()
                .dma_mode_word,
            5
        );
        core.step_mte_word(CAPTURED_C220_MOV_OUT_TO_UB_X_WORD, &fixture.source)
            .unwrap();
        core.scalar_mut()
            .machine_mut()
            .set_spr_value(93, 0)
            .unwrap();
        assert!(matches!(
            core.pending_mte2[0],
            PendingMte2::C220(C220Mte2TransferPlan {
                dma_mode_word: 5,
                ..
            })
        ));
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
            Err(UbMemoryError::UnknownByte { address: 0 })
        ));
        assert!(matches!(
            core.ub().read_known(128, 1),
            Err(UbMemoryError::UnknownByte { address: 128 })
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
            Err(UbMemoryError::UnknownByte { address: 128 })
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
            Err(MteStepperError::Ub(UbMemoryError::Source(_)))
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
            Err(UbMemoryError::UnknownByte { address: 0 })
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
        machine.set_spr_value(3, 1 << 56).unwrap();
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
            Err(UbMemoryError::UnknownByte { address: 0 })
        ));
    }

    #[test]
    fn c220_movev_decodes_registers_and_preserves_masked_lanes() {
        let mut core = stepper(Architecture::Dav2201, 0x3000);
        let word = (C220_CAPTURED_MOVEV_WORD & !((31 << 17) | (31 << 12) | (31 << 2)))
            | (3 << 17)
            | (9 << 12)
            | (7 << 2);
        let machine = core.scalar_mut().machine_mut();
        machine.set_xreg(3, 0x80).unwrap();
        machine.set_xreg(7, C220_CAPTURED_MOVEV_CONTROL).unwrap();
        machine.set_xreg(9, 0x1234_5678_9abc_def0).unwrap();
        machine.set_spr_value(3, 0).unwrap();
        machine.set_spr_value(100, 0b1010).unwrap();
        machine.set_spr_value(101, 0).unwrap();

        let step = core.step_c220_movev_word(word).unwrap();
        assert_eq!(step.instruction.destination_register, 3);
        assert_eq!(step.instruction.source_register, 9);
        assert_eq!(step.instruction.control_register, 7);
        assert_eq!(step.stores.len(), 2);
        assert_eq!(step.stores[0].lane_index, 1);
        assert_eq!(step.stores[1].lane_index, 3);
        assert_eq!(
            core.ub().read_known(0x84, 4).unwrap(),
            0x9abc_def0_u32.to_le_bytes()
        );
        assert!(matches!(
            core.ub().read_known(0x80, 4),
            Err(UbMemoryError::UnknownByte { address: 0x80 })
        ));
    }

    #[test]
    fn c220_movev_writes_full_i16_tile_with_count_mask() {
        let mut core = stepper(Architecture::Dav2201, 0x3000);
        let word = (C220_CAPTURED_MOVEV_WORD & !(7 << 22)) | (1 << 22);
        let machine = core.scalar_mut().machine_mut();
        machine.set_xreg(5, C220_CAPTURED_MOVEV_CONTROL).unwrap();
        machine.set_xreg(6, 0x1234_5678).unwrap();
        machine.set_xreg(16, 0x100).unwrap();
        machine.set_spr_value(3, 1 << 56).unwrap();
        machine.set_spr_value(100, 128).unwrap();
        machine.set_spr_value(101, 0).unwrap();
        let step = core.step_c220_movev_word(word).unwrap();
        assert_eq!(step.instruction.supported_element_bytes(), Some(2));
        assert_eq!(step.stores.len(), 128);
        assert_eq!(step.stores[127].address, 0x1fe);
        assert_eq!(step.stores[127].width_bytes, 2);
        assert_eq!(
            core.ub().read_known(0x100, 256).unwrap(),
            [0x78, 0x56].repeat(128)
        );
    }

    #[test]
    fn c220_movev_rejects_unverified_control_mask_and_overflow_atomically() {
        let mut core = stepper(Architecture::Dav2201, 0x3000);
        let machine = core.scalar_mut().machine_mut();
        machine.set_xreg(5, C220_CAPTURED_MOVEV_CONTROL).unwrap();
        machine.set_xreg(6, 0xc2f6_0000).unwrap();
        machine.set_xreg(16, 0x100).unwrap();
        machine.set_spr_value(3, 1 << 56).unwrap();
        machine.set_spr_value(100, 32).unwrap();
        machine.set_spr_value(101, 0).unwrap();

        core.scalar_mut().machine_mut().set_xreg(5, 0).unwrap();
        let before = core.clone();
        assert!(matches!(
            core.step_c220_movev_word(C220_CAPTURED_MOVEV_WORD),
            Err(MteStepperError::Vector(
                C220VectorError::UnsupportedMovevControl { control: 0 }
            ))
        ));
        assert_eq!(core, before);

        core.scalar_mut()
            .machine_mut()
            .set_xreg(5, C220_CAPTURED_MOVEV_CONTROL)
            .unwrap();
        core.scalar_mut()
            .machine_mut()
            .set_spr_value(100, 65)
            .unwrap();
        let before = core.clone();
        assert!(matches!(
            core.step_c220_movev_word(C220_CAPTURED_MOVEV_WORD),
            Err(MteStepperError::Vector(
                C220VectorError::CountMaskExceedsTile { .. }
            ))
        ));
        assert_eq!(core, before);

        core.scalar_mut()
            .machine_mut()
            .set_spr_value(100, 32)
            .unwrap();
        core.ub = UbMemory::new(64, 256);
        let before = core.clone();
        assert!(matches!(
            core.step_c220_movev_word(C220_CAPTURED_MOVEV_WORD),
            Err(MteStepperError::Vector(C220VectorError::Ub(
                UbMemoryError::TrackedLimitExceeded { .. }
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
        machine.set_spr_value(3, 1 << 56).unwrap();
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
        assert!(fixture.source.read_known_at(0x2000, 128).is_err());
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
        assert!(fixture.source.read_known_at(0x2000, 128).is_err());
        let machine = core.scalar_mut().machine_mut();
        machine.set_xreg(14, 0x100).unwrap();
        machine.set_xreg(10, 0x2000).unwrap();
        machine.set_xreg(3, 0x40010).unwrap();
        machine.set_spr_value(94, 5).unwrap();
        assert_eq!(
            core.preview_c220_mte3_transfer(CAPTURED_C220_MOV_UB_TO_OUT_WORD)
                .unwrap()
                .dma_mode_word,
            0
        );
        core.set_c220_isa_instance_index(1);
        assert_eq!(
            core.preview_c220_mte3_transfer(CAPTURED_C220_MOV_UB_TO_OUT_WORD)
                .unwrap()
                .dma_mode_word,
            5
        );
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
        assert_eq!(fixture.source.read_known_at(0x2000, 128).unwrap(), result);
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
        let mut partial = core.clone();
        let partial_step = partial
            .step_c220_vsub_word(C220_CAPTURED_VSUB_WORD)
            .unwrap();
        assert_eq!(partial_step.stores.len(), 31);
        assert_eq!(partial.ub().read_known(0x17c, 4).unwrap(), [0; 4]);
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
        assert_eq!(fixture.source.read_known_at(0x2000, 128).unwrap(), expected);
    }

    #[test]
    fn c220_vmul_uses_live_control_mask_addresses_and_ub_words() {
        let mut core = stepper(Architecture::Dav2201, 0x3000);
        let mut x = std::array::from_fn::<_, 32, _>(|lane| (lane as f32).to_bits());
        let mut y = [0.5_f32.to_bits(); 32];
        x[0] = 0;
        y[0] = f32::INFINITY.to_bits();
        x[1] = (-0.0_f32).to_bits();
        y[1] = 2.0_f32.to_bits();
        let input = x
            .iter()
            .chain(&y)
            .flat_map(|word| word.to_le_bytes().map(MemoryByteState::Known))
            .collect::<Vec<_>>();
        core.ub.write_states(0, &input).unwrap();
        core.ub
            .write_states(0x100, &[MemoryByteState::Known(0); 128])
            .unwrap();
        let machine = core.scalar_mut().machine_mut();
        machine.set_xreg(6, C220_CAPTURED_VMUL_CONTROL).unwrap();
        machine.set_xreg(11, 0).unwrap();
        machine.set_xreg(12, 0x80).unwrap();
        machine.set_xreg(14, 0x100).unwrap();
        machine.set_spr_value(3, 1 << 56).unwrap();
        machine.set_spr_value(100, 31).unwrap();
        machine.set_spr_value(101, 0).unwrap();
        core.scalar_mut().machine_mut().set_xreg(6, 0).unwrap();
        let before_control = core.clone();
        assert!(matches!(
            core.step_c220_vmul_word(C220_CAPTURED_VMUL_WORD),
            Err(MteStepperError::Vector(
                C220VectorError::UnsupportedFp32Control { control: 0 }
            ))
        ));
        assert_eq!(core, before_control);
        core.scalar_mut()
            .machine_mut()
            .set_xreg(6, C220_CAPTURED_VMUL_CONTROL)
            .unwrap();
        let mut partial = core.clone();
        let partial_step = partial
            .step_c220_vmul_word(C220_CAPTURED_VMUL_WORD)
            .unwrap();
        assert_eq!(partial_step.stores.len(), 31);
        assert_eq!(partial.ub().read_known(0x17c, 4).unwrap(), [0; 4]);
        core.scalar_mut()
            .machine_mut()
            .set_spr_value(100, 32)
            .unwrap();
        let step = core.step_c220_vmul_word(C220_CAPTURED_VMUL_WORD).unwrap();
        assert_eq!(step.stores.len(), 32);
        assert_eq!(step.source_0_address, 0);
        assert_eq!(step.source_1_address, 0x80);
        assert_eq!(step.destination_address, 0x100);
        assert_eq!(step.lanes[0].bits, 0x7fff_ffff);
        assert_eq!(step.lanes[1].bits, 0x8000_0000);
        let output = core.ub().read_known(0x100, 128).unwrap();
        assert_eq!(&output[..4], &0x7fff_ffff_u32.to_le_bytes());
        assert_eq!(&output[4..8], &0x8000_0000_u32.to_le_bytes());
        for lane in 2..32 {
            let expected = ((lane as f32) * 0.5).to_le_bytes();
            assert_eq!(&output[lane * 4..lane * 4 + 4], &expected);
        }
        let x_bytes = x
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>();
        let y_bytes = y
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>();
        let mut fixture = Fixture::two_inputs(&x_bytes, &y_bytes);
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
                transfer: UbTransferResult {
                    bytes: 128,
                    known_bytes: 128,
                    unknown_bytes: 0,
                    ..
                }
            }
        ));
        assert_eq!(fixture.source.read_known_at(0x2000, 128).unwrap(), output);
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
        core.unsignaled_output = Some(C220OutputToken {
            source_address: 0x100,
        });
        core.step_c220_output_word(C220_VECTOR_TO_MTE3_SET_FLAG_WORD, &mut fixture.source)
            .unwrap();
        core.ub
            .write_states(0x100, &[MemoryByteState::Known(7); 128])
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
        machine.set_xreg(14, 0x100).unwrap();
        machine.set_xreg(10, 0x2000).unwrap();
        machine.set_xreg(3, 0x40010).unwrap();
        core.scalar_mut()
            .machine_mut()
            .set_xreg(10, 0x2080)
            .unwrap();
        let before = core.clone();
        assert!(matches!(
            core.step_c220_output_word(CAPTURED_C220_MOV_UB_TO_OUT_WORD, &mut fixture.source),
            Err(MteStepperError::Ub(UbMemoryError::Destination(
                crate::memory::mapped::MappedMemoryError::Unmapped { .. }
            )))
        ));
        assert_eq!(core, before);
        assert!(fixture.source.read_known_at(0x2000, 128).is_err());

        core.scalar_mut().machine_mut().set_xreg(14, 0x120).unwrap();
        core.scalar_mut()
            .machine_mut()
            .set_xreg(10, 0x2000)
            .unwrap();
        let step = core
            .step_c220_output_word(CAPTURED_C220_MOV_UB_TO_OUT_WORD, &mut fixture.source)
            .unwrap();
        assert!(matches!(
            step.action,
            C220OutputAction::CopyToHbm {
                source_address: 0x120,
                transfer: UbTransferResult {
                    known_bytes: 96,
                    unknown_bytes: 32,
                    ..
                },
                ..
            }
        ));
        assert_eq!(fixture.source.read_known_at(0x2000, 96).unwrap(), [7; 96]);
        assert_eq!(
            fixture.source.read_states_at(0x2060, 32).unwrap(),
            [MemoryByteState::Unknown; 32]
        );
    }

    #[test]
    fn c220_output_transfer_uses_descriptor_segments_after_flag() {
        let mut fixture = Fixture::two_inputs(&[0; 128], &[0; 128]);
        let mut core = stepper(Architecture::Dav2201, 0x3000);
        core.unsignaled_output = Some(C220OutputToken {
            source_address: 0x100,
        });
        core.ub
            .write_states(0x100, &[MemoryByteState::Known(7); 32])
            .unwrap();
        core.ub
            .write_states(0x140, &[MemoryByteState::Known(9); 32])
            .unwrap();
        core.step_c220_output_word(C220_VECTOR_TO_MTE3_SET_FLAG_WORD, &mut fixture.source)
            .unwrap();
        core.step_c220_output_word(C220_VECTOR_TO_MTE3_WAIT_FLAG_WORD, &mut fixture.source)
            .unwrap();
        let machine = core.scalar_mut().machine_mut();
        machine.set_xreg(14, 0x100).unwrap();
        machine.set_xreg(10, 0x2000).unwrap();
        machine
            .set_xreg(3, (1_u64 << 48) | (1 << 32) | (1 << 16) | (2 << 4))
            .unwrap();
        let step = core
            .step_c220_output_word(CAPTURED_C220_MOV_UB_TO_OUT_WORD, &mut fixture.source)
            .unwrap();
        assert!(matches!(
            step.action,
            C220OutputAction::CopyToHbm {
                transfer: UbTransferResult {
                    segment_count: 2,
                    bytes: 64,
                    known_bytes: 64,
                    unknown_bytes: 0
                },
                ..
            }
        ));
        assert_eq!(fixture.source.read_known_at(0x2000, 32).unwrap(), [7; 32]);
        assert_eq!(fixture.source.read_known_at(0x2040, 32).unwrap(), [9; 32]);
        assert!(fixture.source.read_known_at(0x2020, 32).is_err());
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
    fn c220_arithmetic_decodes_registers_and_uses_live_bitset_mask() {
        let mut core = stepper(Architecture::Dav2201, 0);
        let word = (C220_CAPTURED_VADD_WORD & !((31 << 17) | (31 << 12) | (31 << 7) | (31 << 2)))
            | (3 << 17)
            | (9 << 12)
            | (10 << 7)
            | (7 << 2);
        let mut bytes = vec![MemoryByteState::Known(0x5a); 384];
        for lane in 0..32 {
            bytes[lane * 4..lane * 4 + 4]
                .copy_from_slice(&1.0_f32.to_le_bytes().map(MemoryByteState::Known));
            bytes[128 + lane * 4..128 + lane * 4 + 4]
                .copy_from_slice(&2.0_f32.to_le_bytes().map(MemoryByteState::Known));
        }
        core.ub.write_states(0, &bytes[..256]).unwrap();
        core.ub.write_states(256, &bytes[256..]).unwrap();
        let machine = core.scalar_mut().machine_mut();
        machine.set_xreg(3, 0x100).unwrap();
        machine.set_xreg(9, 0).unwrap();
        machine.set_xreg(10, 0x80).unwrap();
        machine.set_xreg(7, C220_CAPTURED_VADD_CONTROL).unwrap();
        machine.set_spr_value(3, 0).unwrap();
        machine.set_spr_value(100, 0b101).unwrap();
        machine.set_spr_value(101, 0).unwrap();

        let before = core.clone();
        assert!(matches!(
            core.step_c220_vsub_word(word),
            Err(MteStepperError::UnsupportedWord { .. })
        ));
        assert_eq!(core, before);

        let step = core.step_c220_vadd_word(word).unwrap();
        assert_eq!(step.hint.operation, C220VecArithmeticOperation::Add);
        assert_eq!(step.hint.x_register_index_0, 3);
        assert_eq!(step.hint.x_register_index_4, 9);
        assert_eq!(step.hint.x_register_index_6, 10);
        assert_eq!(step.hint.x_register_index_8, 7);
        assert_eq!(step.stores.len(), 2);
        assert_eq!(step.stores[0].lane_index, 0);
        assert_eq!(step.stores[1].lane_index, 2);
        assert_eq!(
            core.ub().read_known(0x100, 4).unwrap(),
            3.0_f32.to_le_bytes()
        );
        assert_eq!(core.ub().read_known(0x104, 4).unwrap(), [0x5a; 4]);
        assert_eq!(
            core.ub().read_known(0x108, 4).unwrap(),
            3.0_f32.to_le_bytes()
        );
    }

    #[test]
    fn c220_vadd_rejects_unverified_control_and_unknown_input_atomically() {
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
                C220VectorError::UnsupportedFp32Control { control: 0 }
            ))
        ));
        assert_eq!(core, before);

        core.scalar_mut()
            .machine_mut()
            .set_xreg(8, C220_CAPTURED_VADD_CONTROL)
            .unwrap();
        let before = core.clone();
        assert!(matches!(
            core.step_c220_vadd_word(C220_CAPTURED_VADD_WORD),
            Err(MteStepperError::Vector(C220VectorError::Ub(
                UbMemoryError::UnknownByte { address: 0 }
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
            .execute_captured_vector_load_word(
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
