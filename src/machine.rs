use crate::architecture::Architecture;
use crate::buffer_c310::C310BufferDisposition;
use crate::flow::{
    C310BufferInstruction, C310BufferStep, ConditionalJump, ConditionalJumpTarget, DcciInstruction,
    DcciStep, DsbStep, FlowEnd, FlowNop, JumpCompare, JumpCompareError, JumpCompareTarget,
    JumpTarget, PipelineBarrierStep, UnconditionalJump, compare_values,
};
use crate::isa::{
    AicClass, AicDecoderHint, ScalarKey0Operation, ScalarKey7Operation, ScalarKey8Operation,
    ScalarLoadStoreOperation, ScalarStoreImmediateValue,
};
use crate::predicate_buffer_c310::{C310PushPbDisposition, C310PushPbInstruction, C310PushPbStep};
use crate::rvec::{C310ObservedMovemaskHint, C310ObservedMovemaskStep};
use crate::scalar::{
    ScalarIntegerError, evaluate_scalar_integer_immediate, update_neg_overflow_spr2,
    update_overflow_spr2,
};
use crate::vec_c220::C220MovemaskHint;
use crate::vec_queue_c310::{C310VfQueueDisposition, C310VfQueueInstruction, C310VfQueueStep};
use serde::Serialize;
use thiserror::Error;

pub const SCALAR_X_REGISTER_COUNT: usize = 32;
const SCALAR_SPR_SNAPSHOT_CAPACITY: usize = 243;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScalarMachine {
    architecture: Architecture,
    xregs: [u64; SCALAR_X_REGISTER_COUNT],
    spr2: u64,
    spr_values: [Option<u64>; SCALAR_SPR_SNAPSHOT_CAPACITY],
    model_time: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ScalarStep {
    pub pc: u64,
    pub word: u32,
    pub destination_register: u8,
    pub prior_destination_value: u64,
    pub source_register: Option<u8>,
    pub source_value: Option<u64>,
    pub second_source_register: Option<u8>,
    pub second_source_value: Option<u64>,
    pub value: u64,
    pub signed_overflow: bool,
    pub prior_spr2: u64,
    pub spr2: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ScalarSprStep {
    pub pc: u64,
    pub word: u32,
    pub destination_spr: u16,
    pub prior_destination_value: Option<u64>,
    pub source_register: u8,
    pub source_value: u64,
    pub value: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ScalarSprReadSource {
    RegisterSnapshot,
    ProgramCounter,
    ModelTime,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ScalarSprReadStep {
    pub pc: u64,
    pub word: u32,
    pub destination_register: u8,
    pub prior_destination_value: u64,
    pub source_spr: u16,
    pub source: ScalarSprReadSource,
    pub source_value: u64,
    pub value: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ScalarInstructionStep {
    Register(ScalarStep),
    SprRead(ScalarSprReadStep),
    SprWrite(ScalarSprStep),
    Memory(ScalarMemoryStep),
    ImmediateStore(ScalarImmediateStoreStep),
    PairLoad(ScalarPairLoadStep),
    PairStore(ScalarPairStoreStep),
    IndexedLoad(ScalarIndexedLoadStep),
    IndexedImmediateStore(ScalarIndexedImmediateStoreStep),
    Flow(ScalarFlowStep),
    CacheHint(ScalarCacheHintStep),
    C220Movemask(C220MovemaskStep),
    C310Movemask(C310ObservedMovemaskStep),
    Compare(ScalarCompareStep),
    CompareRegister(ScalarCompareRegisterStep),
    CompareImmediate(ScalarCompareImmediateStep),
    Select(ScalarSelectStep),
    Dcci(DcciStep),
    Dsb(DsbStep),
    Barrier(PipelineBarrierStep),
    Buffer(C310BufferStep),
    PushPb(C310PushPbStep),
    VfQueue(C310VfQueueStep),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct C220MovemaskStep {
    pub pc: u64,
    pub word: u32,
    pub source_register: u8,
    pub source_value: u64,
    pub destination_spr: u16,
    pub prior_destination_value: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ScalarCacheHintStep {
    pub pc: u64,
    pub word: u32,
    pub source_register: u8,
    pub source_value: u64,
    pub encoded_immediate: u16,
    pub effective_address: u64,
    pub requested_bytes: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ScalarFlowStep {
    Nop(FlowNop),
    End(FlowEnd),
    Jump(JumpTarget),
    Conditional(ConditionalJumpTarget),
    Compare(JumpCompareTarget),
}

impl ScalarFlowStep {
    pub const fn target_pc(self) -> u64 {
        match self {
            Self::Nop(step) => step.target_pc,
            Self::End(step) => step.sequential_pc,
            Self::Jump(step) => step.target_pc,
            Self::Conditional(step) => step.target_pc,
            Self::Compare(step) => step.target_pc,
        }
    }
}

pub trait ScalarMemoryBus {
    type Error: std::error::Error + 'static;

    fn read(&mut self, effective_address: u64, destination: &mut [u8]) -> Result<(), Self::Error>;
    fn write(&mut self, effective_address: u64, source: &[u8]) -> Result<(), Self::Error>;

    fn maintain_data_cache(&mut self, _step: DcciStep) -> Result<bool, Self::Error> {
        Ok(false)
    }

    fn synchronize_pipeline(&mut self, _step: DsbStep) -> Result<bool, Self::Error> {
        Ok(false)
    }

    fn synchronize_barrier(&mut self, _step: PipelineBarrierStep) -> Result<bool, Self::Error> {
        Ok(false)
    }

    fn execute_c310_buffer(
        &mut self,
        _step: C310BufferStep,
    ) -> Result<C310BufferDisposition, Self::Error> {
        Ok(C310BufferDisposition::Unsupported)
    }

    fn execute_c310_push_pb(
        &mut self,
        _step: C310PushPbStep,
    ) -> Result<C310PushPbDisposition, Self::Error> {
        Ok(C310PushPbDisposition::Unsupported)
    }

    fn enqueue_c310_vf(
        &mut self,
        _step: C310VfQueueStep,
    ) -> Result<C310VfQueueDisposition, Self::Error> {
        Ok(C310VfQueueDisposition::Unsupported)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ScalarMemoryStep {
    pub pc: u64,
    pub word: u32,
    pub operation: ScalarLoadStoreOperation,
    pub effective_address: u64,
    pub width_bytes: u8,
    pub data_register: u8,
    pub prior_data_value: u64,
    pub data_value: u64,
    pub base_register: u8,
    pub prior_base_value: u64,
    pub updated_base: Option<u64>,
    pub bytes: [u8; 8],
    pub sign_extension_requested: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ScalarIndexedLoadStep {
    pub pc: u64,
    pub word: u32,
    pub effective_address: u64,
    pub width_bytes: u8,
    pub destination_register: u8,
    pub prior_destination_value: u64,
    pub value: u64,
    pub base_register: u8,
    pub base_value: u64,
    pub offset_register: u8,
    pub offset_value: u64,
    pub bytes: [u8; 8],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ScalarIndexedImmediateStoreStep {
    pub pc: u64,
    pub word: u32,
    pub effective_address: u64,
    pub width_bytes: u8,
    pub base_register: u8,
    pub base_value: u64,
    pub offset_register: u8,
    pub offset_value: u64,
    pub value: ScalarStoreImmediateValue,
    pub bytes: [u8; 8],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ScalarImmediateStoreStep {
    pub pc: u64,
    pub word: u32,
    pub effective_address: u64,
    pub width_bytes: u8,
    pub base_register: u8,
    pub prior_base_value: u64,
    pub value: ScalarStoreImmediateValue,
    pub bytes: [u8; 8],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ScalarCompareStep {
    pub pc: u64,
    pub word: u32,
    pub dtype_field: u8,
    pub condition_field: u8,
    pub first_source_register: u8,
    pub first_source_value: u64,
    pub second_source_register: u8,
    pub second_source_value: u64,
    pub prior_spr11: Option<u64>,
    pub spr11: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ScalarCompareRegisterStep {
    pub pc: u64,
    pub word: u32,
    pub dtype_field: u8,
    pub condition_field: u8,
    pub destination_register: u8,
    pub prior_destination_value: u64,
    pub first_source_register: u8,
    pub first_source_value: u64,
    pub second_source_register: u8,
    pub second_source_value: u64,
    pub value: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ScalarCompareImmediateStep {
    pub pc: u64,
    pub word: u32,
    pub vendor_isa_name: u16,
    pub condition_field: u8,
    pub source_register: u8,
    pub source_value: u64,
    pub encoded_immediate: u16,
    pub signed_immediate: i16,
    pub prior_spr11: Option<u64>,
    pub spr11: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ScalarSelectStep {
    pub pc: u64,
    pub word: u32,
    pub vendor_isa_name: u16,
    pub dtype_field: u8,
    pub condition_flag: u64,
    pub destination_register: u8,
    pub prior_destination_value: u64,
    pub first_source_register: u8,
    pub first_source_value: u64,
    pub second_source_register: u8,
    pub second_source_value: u64,
    pub value: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ScalarPairLoadStep {
    pub pc: u64,
    pub word: u32,
    pub first_address: u64,
    pub second_address: u64,
    pub width_bytes: u8,
    pub base_register: u8,
    pub base_value: u64,
    pub first_destination_register: u8,
    pub first_prior_value: u64,
    pub first_value: u64,
    pub first_bytes: [u8; 8],
    pub second_destination_register: u8,
    pub second_prior_value: u64,
    pub second_value: u64,
    pub second_bytes: [u8; 8],
    pub sign_extension_requested: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ScalarPairStoreStep {
    pub pc: u64,
    pub word: u32,
    pub first_address: u64,
    pub second_address: u64,
    pub width_bytes: u8,
    pub base_register: u8,
    pub base_value: u64,
    pub first_source_register: u8,
    pub first_value: u64,
    pub first_bytes: [u8; 8],
    pub second_source_register: u8,
    pub second_value: u64,
    pub second_bytes: [u8; 8],
}

#[derive(Debug, Error)]
pub enum ScalarMemoryExecutionError<E: std::error::Error + 'static> {
    #[error("word {word:#010x} at PC {pc:#x} is not an implemented scalar LD/ST instruction")]
    UnsupportedWord { pc: u64, word: u32 },
    #[error(
        "post-indexed word {word:#010x} at PC {pc:#x} aliases data and base register X{register}"
    )]
    AliasedPostIndex { pc: u64, word: u32, register: u8 },
    #[error("scalar memory backend failed: {0}")]
    Backend(#[source] E),
}

#[derive(Debug, Error)]
pub enum ScalarInstructionError<E: std::error::Error + 'static> {
    #[error(transparent)]
    Scalar(#[from] ScalarMachineError),
    #[error(transparent)]
    Memory(#[from] ScalarMemoryExecutionError<E>),
    #[error(
        "data-cache maintenance for word {word:#010x} at PC {pc:#x} is not supported by this memory backend"
    )]
    CacheMaintenanceUnsupported { pc: u64, word: u32 },
    #[error("data-cache maintenance backend failed: {0}")]
    CacheBackend(#[source] E),
    #[error(
        "pipeline synchronization for word {word:#010x} at PC {pc:#x} is not supported by this backend"
    )]
    SynchronizationUnsupported { pc: u64, word: u32 },
    #[error("pipeline synchronization backend failed: {0}")]
    SynchronizationBackend(#[source] E),
    #[error("buffer instruction {word:#010x} at PC {pc:#x} is not supported by this backend")]
    BufferUnsupported { pc: u64, word: u32 },
    #[error("buffer instruction {word:#010x} at PC {pc:#x} stalled")]
    BufferStalled { pc: u64, word: u32 },
    #[error("buffer instruction backend failed: {0}")]
    BufferBackend(#[source] E),
    #[error("predicate-buffer push {word:#010x} at PC {pc:#x} is not supported by this backend")]
    PushPbUnsupported { pc: u64, word: u32 },
    #[error("predicate-buffer push {word:#010x} at PC {pc:#x} stalled")]
    PushPbStalled { pc: u64, word: u32 },
    #[error("predicate-buffer backend failed: {0}")]
    PushPbBackend(#[source] E),
    #[error(
        "vector queue words {first_word:#010x}/{second_word:#010x} at PC {pc:#x} are not supported by this backend"
    )]
    VfQueueUnsupported {
        pc: u64,
        first_word: u32,
        second_word: u32,
    },
    #[error("vector queue words {first_word:#010x}/{second_word:#010x} at PC {pc:#x} stalled")]
    VfQueueStalled {
        pc: u64,
        first_word: u32,
        second_word: u32,
    },
    #[error("vector queue backend failed: {0}")]
    VfQueueBackend(#[source] E),
    #[error("scalar program already ended before PC {pc:#x}")]
    ProgramEnded { pc: u64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum ScalarMachineError {
    #[error("X-register index {0} is outside X0..X31")]
    RegisterOutOfRange(u8),
    #[error("SPR index {0} is outside the modeled architecture's register range")]
    SprRegisterOutOfRange(u16),
    #[error("SPR {spr} has no supplied value at PC {pc:#x}")]
    SprValueUnavailable { pc: u64, spr: u16 },
    #[error("model time has no supplied value at PC {pc:#x}")]
    ModelTimeUnavailable { pc: u64 },
    #[error("word {word:#010x} at PC {pc:#x} is not an implemented scalar instruction")]
    UnsupportedWord { pc: u64, word: u32 },
    #[error(transparent)]
    Arithmetic(#[from] ScalarIntegerError),
    #[error(transparent)]
    Compare(#[from] JumpCompareError),
}

impl ScalarMachine {
    pub fn enqueue_c310_vf<B: ScalarMemoryBus>(
        &mut self,
        pc: u64,
        first_word: u32,
        second_word: u32,
        bus: &mut B,
    ) -> Result<ScalarInstructionStep, ScalarInstructionError<B::Error>> {
        let instruction =
            C310VfQueueInstruction::decode(self.architecture, first_word, second_word).ok_or(
                ScalarMachineError::UnsupportedWord {
                    pc,
                    word: first_word,
                },
            )?;
        let step = instruction.resolve(pc, &self.xregs);
        match bus
            .enqueue_c310_vf(step)
            .map_err(ScalarInstructionError::VfQueueBackend)?
        {
            C310VfQueueDisposition::Accepted => Ok(ScalarInstructionStep::VfQueue(step)),
            C310VfQueueDisposition::Stalled => Err(ScalarInstructionError::VfQueueStalled {
                pc,
                first_word,
                second_word,
            }),
            C310VfQueueDisposition::Unsupported => {
                Err(ScalarInstructionError::VfQueueUnsupported {
                    pc,
                    first_word,
                    second_word,
                })
            }
        }
    }

    pub const fn new(
        architecture: Architecture,
        xregs: [u64; SCALAR_X_REGISTER_COUNT],
        spr2: u64,
    ) -> Self {
        Self {
            architecture,
            xregs,
            spr2,
            spr_values: [None; SCALAR_SPR_SNAPSHOT_CAPACITY],
            model_time: None,
        }
    }

    pub fn from_pem_initial_state(architecture: Architecture) -> Self {
        let mut machine = Self::new(architecture, [0; SCALAR_X_REGISTER_COUNT], 0);
        let sprs = match architecture {
            Architecture::Dav2201 => &mut machine.spr_values[..187],
            Architecture::Dav3510 => &mut machine.spr_values[..SCALAR_SPR_SNAPSHOT_CAPACITY],
        };
        sprs.fill(Some(0));
        match architecture {
            Architecture::Dav2201 => {
                for (index, slot) in sprs[111..175].iter_mut().enumerate() {
                    *slot = Some(index as u64);
                }
                for index in [100, 101, 104, 105, 181, 182, 183, 184, 185, 186] {
                    sprs[index] = Some(u64::MAX);
                }
                sprs[55] = Some(15360);
                sprs[58] = Some(0x10000);
                sprs[107] = None;
                sprs[108] = None;
            }
            Architecture::Dav3510 => {
                for (index, slot) in sprs[164..228].iter_mut().enumerate() {
                    *slot = Some(index as u64);
                }
                for index in [152, 153, 156, 157, 234, 235, 236, 237, 238, 239] {
                    sprs[index] = Some(u64::MAX);
                }
                for index in [105, 109, 112, 115] {
                    sprs[index] = Some(0x200001);
                }
                sprs[3] = Some(0x1000000000000008);
                sprs[20] = Some(784);
                sprs[58] = Some(0x10000);
                sprs[90] = Some(36);
                sprs[104] = Some(0x10000);
                sprs[241] = Some(0x2000);
                sprs[242] = Some(4);
                for index in [67, 68, 73] {
                    sprs[index] = None;
                }
            }
        }
        machine
    }

    pub const fn architecture(&self) -> Architecture {
        self.architecture
    }

    pub const fn xregs(&self) -> &[u64; SCALAR_X_REGISTER_COUNT] {
        &self.xregs
    }

    pub const fn spr2(&self) -> u64 {
        self.spr2
    }

    pub fn spr_value(&self, index: u16) -> Option<u64> {
        if index == 2 {
            return Some(self.spr2);
        }
        self.spr_values.get(usize::from(index)).copied().flatten()
    }

    pub const fn model_time(&self) -> Option<u64> {
        self.model_time
    }

    pub fn set_model_time(&mut self, ticks: u64) {
        self.model_time = Some(ticks);
    }

    pub fn set_spr_value(&mut self, index: u16, value: u64) -> Result<(), ScalarMachineError> {
        let limit = match self.architecture {
            Architecture::Dav2201 => 187,
            Architecture::Dav3510 => SCALAR_SPR_SNAPSHOT_CAPACITY,
        };
        let slot = self
            .spr_values
            .get_mut(usize::from(index))
            .filter(|_| usize::from(index) < limit)
            .ok_or(ScalarMachineError::SprRegisterOutOfRange(index))?;
        *slot = Some(value);
        if index == 2 {
            self.spr2 = value;
        }
        Ok(())
    }

    pub fn execute_c220_movemask_word(
        &mut self,
        pc: u64,
        word: u32,
    ) -> Result<C220MovemaskStep, ScalarMachineError> {
        if self.architecture != Architecture::Dav2201 {
            return Err(ScalarMachineError::UnsupportedWord { pc, word });
        }
        let hint = C220MovemaskHint::from_word(word)
            .ok_or(ScalarMachineError::UnsupportedWord { pc, word })?;
        let source_value = self.xregs[usize::from(hint.source_register)];
        let prior_destination_value = self.spr_value(hint.destination_spr);
        self.set_spr_value(hint.destination_spr, source_value)?;
        Ok(C220MovemaskStep {
            pc,
            word,
            source_register: hint.source_register,
            source_value,
            destination_spr: hint.destination_spr,
            prior_destination_value,
        })
    }

    pub fn execute_c310_movemask_word(
        &mut self,
        pc: u64,
        word: u32,
    ) -> Result<C310ObservedMovemaskStep, ScalarMachineError> {
        if self.architecture != Architecture::Dav3510 {
            return Err(ScalarMachineError::UnsupportedWord { pc, word });
        }
        let hint = C310ObservedMovemaskHint::from_word(word)
            .ok_or(ScalarMachineError::UnsupportedWord { pc, word })?;
        let prior_value = self.spr_value(hint.destination_spr).ok_or(
            ScalarMachineError::SprValueUnavailable {
                pc,
                spr: hint.destination_spr,
            },
        )?;
        let value = self.xregs[usize::from(hint.source_x_register)];
        self.set_spr_value(hint.destination_spr, value)?;
        Ok(C310ObservedMovemaskStep {
            pc,
            word,
            hint,
            prior_value,
            value,
        })
    }

    pub fn execute_compare_word(
        &mut self,
        pc: u64,
        word: u32,
    ) -> Result<ScalarCompareStep, ScalarMachineError> {
        let Some(AicDecoderHint::ScalarCompare {
            dtype_field,
            condition_field,
            first_source_register,
            second_source_register,
        }) = AicDecoderHint::from_word(self.architecture, word)
        else {
            return Err(ScalarMachineError::UnsupportedWord { pc, word });
        };
        let first_source_value = self.xregs[usize::from(first_source_register)];
        let second_source_value = self.xregs[usize::from(second_source_register)];
        let result = evaluate_integer_compare(
            dtype_field,
            condition_field,
            first_source_value,
            second_source_value,
        )
        .ok_or(ScalarMachineError::UnsupportedWord { pc, word })?;
        let prior_spr11 = self.spr_value(11);
        let spr11 = u64::from(result);
        self.set_spr_value(11, spr11)?;
        Ok(ScalarCompareStep {
            pc,
            word,
            dtype_field,
            condition_field,
            first_source_register,
            first_source_value,
            second_source_register,
            second_source_value,
            prior_spr11,
            spr11,
        })
    }

    pub fn execute_compare_register_word(
        &mut self,
        pc: u64,
        word: u32,
    ) -> Result<ScalarCompareRegisterStep, ScalarMachineError> {
        let Some(AicDecoderHint::ScalarCompareRegister {
            dtype_field,
            condition_field,
            destination_register,
            first_source_register,
            second_source_register,
        }) = AicDecoderHint::from_word(self.architecture, word)
        else {
            return Err(ScalarMachineError::UnsupportedWord { pc, word });
        };
        let first_source_value = self.xregs[usize::from(first_source_register)];
        let second_source_value = self.xregs[usize::from(second_source_register)];
        let value = u64::from(
            evaluate_integer_compare(
                dtype_field,
                condition_field,
                first_source_value,
                second_source_value,
            )
            .ok_or(ScalarMachineError::UnsupportedWord { pc, word })?,
        );
        let prior_destination_value = self.xregs[usize::from(destination_register)];
        self.xregs[usize::from(destination_register)] = value;
        Ok(ScalarCompareRegisterStep {
            pc,
            word,
            dtype_field,
            condition_field,
            destination_register,
            prior_destination_value,
            first_source_register,
            first_source_value,
            second_source_register,
            second_source_value,
            value,
        })
    }

    pub fn execute_compare_immediate_word(
        &mut self,
        pc: u64,
        word: u32,
    ) -> Result<ScalarCompareImmediateStep, ScalarMachineError> {
        let Some(AicDecoderHint::ScalarCompareImmediate {
            vendor_isa_name,
            condition_field,
            source_register,
            encoded_immediate,
        }) = AicDecoderHint::from_word(self.architecture, word)
        else {
            return Err(ScalarMachineError::UnsupportedWord { pc, word });
        };
        if condition_field > 5 {
            return Err(ScalarMachineError::UnsupportedWord { pc, word });
        }
        let source_value = self.xregs[usize::from(source_register)];
        let signed_immediate = if encoded_immediate & 0x800 != 0 {
            (encoded_immediate as i16) - 4096
        } else {
            encoded_immediate as i16
        };
        let spr11 = u64::from(compare_values(
            source_value as i64,
            i64::from(signed_immediate),
            condition_field,
        ));
        let prior_spr11 = self.spr_value(11);
        self.set_spr_value(11, spr11)?;
        Ok(ScalarCompareImmediateStep {
            pc,
            word,
            vendor_isa_name,
            condition_field,
            source_register,
            source_value,
            encoded_immediate,
            signed_immediate,
            prior_spr11,
            spr11,
        })
    }

    pub fn execute_select_word(
        &mut self,
        pc: u64,
        word: u32,
    ) -> Result<ScalarSelectStep, ScalarMachineError> {
        let Some(AicDecoderHint::ScalarSelect {
            vendor_isa_name,
            dtype_field,
            destination_register,
            first_source_register,
            second_source_register,
        }) = AicDecoderHint::from_word(self.architecture, word)
        else {
            return Err(ScalarMachineError::UnsupportedWord { pc, word });
        };
        let condition_flag = self
            .spr_value(11)
            .ok_or(ScalarMachineError::SprValueUnavailable { pc, spr: 11 })?;
        let first_source_value = self.xregs[usize::from(first_source_register)];
        let second_source_value = self.xregs[usize::from(second_source_register)];
        let prior_destination_value = self.xregs[usize::from(destination_register)];
        let value = if condition_flag != 0 {
            first_source_value
        } else {
            second_source_value
        };
        self.xregs[usize::from(destination_register)] = value;
        Ok(ScalarSelectStep {
            pc,
            word,
            vendor_isa_name,
            dtype_field,
            condition_flag,
            destination_register,
            prior_destination_value,
            first_source_register,
            first_source_value,
            second_source_register,
            second_source_value,
            value,
        })
    }

    pub fn execute_spr_read_word(
        &mut self,
        pc: u64,
        word: u32,
    ) -> Result<ScalarSprReadStep, ScalarMachineError> {
        let Some(AicDecoderHint::ScalarKey2MoveFromSpr {
            destination_register,
            encoded_source_spr,
            ..
        }) = AicDecoderHint::from_word(self.architecture, word)
        else {
            return Err(ScalarMachineError::UnsupportedWord { pc, word });
        };
        let limit = match self.architecture {
            Architecture::Dav2201 => 187,
            Architecture::Dav3510 => SCALAR_SPR_SNAPSHOT_CAPACITY,
        };
        if usize::from(encoded_source_spr) >= limit {
            return Err(ScalarMachineError::SprRegisterOutOfRange(
                encoded_source_spr,
            ));
        }
        let (source, source_value) = match (self.architecture, encoded_source_spr) {
            (_, 0) => (ScalarSprReadSource::ProgramCounter, pc),
            (Architecture::Dav3510, 72) => (
                ScalarSprReadSource::ModelTime,
                self.model_time
                    .ok_or(ScalarMachineError::ModelTimeUnavailable { pc })?,
            ),
            _ => (
                ScalarSprReadSource::RegisterSnapshot,
                self.spr_value(encoded_source_spr).ok_or(
                    ScalarMachineError::SprValueUnavailable {
                        pc,
                        spr: encoded_source_spr,
                    },
                )?,
            ),
        };
        let value = if self.architecture == Architecture::Dav2201 && encoded_source_spr == 7 {
            source_value & !1
        } else {
            source_value
        };
        let destination = &mut self.xregs[usize::from(destination_register)];
        let prior_destination_value = *destination;
        *destination = value;
        Ok(ScalarSprReadStep {
            pc,
            word,
            destination_register,
            prior_destination_value,
            source_spr: encoded_source_spr,
            source,
            source_value,
            value,
        })
    }

    pub fn execute_spr_word(
        &mut self,
        pc: u64,
        word: u32,
    ) -> Result<ScalarSprStep, ScalarMachineError> {
        let Some(AicDecoderHint::ScalarKey2MoveToSpr {
            encoded_destination_spr,
            source_register,
            ..
        }) = AicDecoderHint::from_word(self.architecture, word)
        else {
            return Err(ScalarMachineError::UnsupportedWord { pc, word });
        };
        let mask = match (self.architecture, encoded_destination_spr) {
            (Architecture::Dav2201, 3) | (Architecture::Dav3510, 3 | 105 | 112) => u64::MAX,
            (Architecture::Dav3510, 11) => 1,
            (Architecture::Dav3510, 90) => 0xff,
            _ => return Err(ScalarMachineError::UnsupportedWord { pc, word }),
        };
        let Some(&source_value) = self.xregs.get(usize::from(source_register)) else {
            return Err(ScalarMachineError::UnsupportedWord { pc, word });
        };
        let slot = &mut self.spr_values[usize::from(encoded_destination_spr)];
        let prior_destination_value = *slot;
        let value = source_value & mask;
        *slot = Some(value);
        Ok(ScalarSprStep {
            pc,
            word,
            destination_spr: encoded_destination_spr,
            prior_destination_value,
            source_register,
            source_value,
            value,
        })
    }

    pub fn set_xreg(&mut self, register: u8, value: u64) -> Result<(), ScalarMachineError> {
        let slot = self
            .xregs
            .get_mut(usize::from(register))
            .ok_or(ScalarMachineError::RegisterOutOfRange(register))?;
        *slot = value;
        Ok(())
    }

    pub fn execute_jump_compare_word(
        &mut self,
        pc: u64,
        word: u32,
    ) -> Result<JumpCompareTarget, ScalarMachineError> {
        let jump = JumpCompare::decode(self.architecture, word)
            .ok_or(ScalarMachineError::UnsupportedWord { pc, word })?;
        let target = jump.evaluate(pc, &self.xregs)?;
        self.set_spr_value(11, target.spr11_value)?;
        Ok(target)
    }

    pub fn execute_cache_hint_word(
        &self,
        pc: u64,
        word: u32,
    ) -> Result<ScalarCacheHintStep, ScalarMachineError> {
        let Some(AicDecoderHint::ScalarKey8 {
            operation: ScalarKey8Operation::DcPreload,
            destination_register: None,
            source_register,
            encoded_immediate,
            ..
        }) = AicDecoderHint::from_word(self.architecture, word)
        else {
            return Err(ScalarMachineError::UnsupportedWord { pc, word });
        };
        let source_value = self.xregs[usize::from(source_register)];
        Ok(ScalarCacheHintStep {
            pc,
            word,
            source_register,
            source_value,
            encoded_immediate,
            effective_address: source_value.wrapping_add(u64::from(encoded_immediate)),
            requested_bytes: 64,
        })
    }

    pub fn execute_flow_word(
        &mut self,
        pc: u64,
        word: u32,
    ) -> Result<ScalarFlowStep, ScalarMachineError> {
        if let Some(end) = FlowEnd::decode(pc, word) {
            return Ok(ScalarFlowStep::End(end));
        }
        if let Some(nop) = FlowNop::decode(self.architecture, pc, word) {
            return Ok(ScalarFlowStep::Nop(nop));
        }
        if let Some(jump) = UnconditionalJump::decode(self.architecture, word) {
            return Ok(ScalarFlowStep::Jump(jump.resolve(pc, &self.xregs)));
        }
        if let Some(jump) = ConditionalJump::decode(self.architecture, word) {
            let condition = self.spr_values[11]
                .ok_or(ScalarMachineError::SprValueUnavailable { pc, spr: 11 })?;
            return Ok(ScalarFlowStep::Conditional(jump.resolve(
                pc,
                &self.xregs,
                condition,
            )));
        }
        if JumpCompare::decode(self.architecture, word).is_some() {
            return Ok(ScalarFlowStep::Compare(
                self.execute_jump_compare_word(pc, word)?,
            ));
        }
        Err(ScalarMachineError::UnsupportedWord { pc, word })
    }

    pub fn execute_word(&mut self, pc: u64, word: u32) -> Result<ScalarStep, ScalarMachineError> {
        let hint = AicDecoderHint::from_word(self.architecture, word)
            .ok_or(ScalarMachineError::UnsupportedWord { pc, word })?;
        let (
            destination_register,
            source_register,
            source_value,
            second_source_register,
            second_source_value,
            value,
            signed_overflow,
            spr2,
        ) = match hint {
            AicDecoderHint::ScalarMoveX8Immediate {
                destination_register,
                encoded_immediate,
                ..
            } => (
                destination_register,
                None,
                None,
                None,
                None,
                u64::from(encoded_immediate) * 8,
                false,
                self.spr2,
            ),
            AicDecoderHint::ScalarKey0 {
                operation,
                dtype_field,
                destination_register,
                first_source_register,
                second_source_register,
                ..
            } => {
                let Some(&first_source_value) = self.xregs.get(usize::from(first_source_register))
                else {
                    return Err(ScalarMachineError::UnsupportedWord { pc, word });
                };
                let Some(&second_source_value) =
                    self.xregs.get(usize::from(second_source_register))
                else {
                    return Err(ScalarMachineError::UnsupportedWord { pc, word });
                };
                if self.xregs.get(usize::from(destination_register)).is_none() {
                    return Err(ScalarMachineError::UnsupportedWord { pc, word });
                }
                let (value, signed_overflow) = match (operation, dtype_field) {
                    (ScalarKey0Operation::Add, 0) => {
                        let (value, overflow) =
                            (first_source_value as i64).overflowing_add(second_source_value as i64);
                        (value as u64, overflow)
                    }
                    (ScalarKey0Operation::Subtract, 0) => {
                        let (value, overflow) =
                            (first_source_value as i64).overflowing_sub(second_source_value as i64);
                        (value as u64, overflow)
                    }
                    (ScalarKey0Operation::Multiply, 0) => {
                        let (value, overflow) =
                            (first_source_value as i64).overflowing_mul(second_source_value as i64);
                        (value as u64, overflow)
                    }
                    (ScalarKey0Operation::MultiplyAdd, 0) => {
                        let (product, multiply_overflow) =
                            (first_source_value as i64).overflowing_mul(second_source_value as i64);
                        let (value, add_overflow) = (self.xregs[usize::from(destination_register)]
                            as i64)
                            .overflowing_add(product);
                        (value as u64, multiply_overflow || add_overflow)
                    }
                    (ScalarKey0Operation::And, 3) => {
                        (first_source_value & second_source_value, false)
                    }
                    (ScalarKey0Operation::Or, 3) => {
                        (first_source_value | second_source_value, false)
                    }
                    _ => return Err(ScalarMachineError::UnsupportedWord { pc, word }),
                };
                (
                    destination_register,
                    Some(first_source_register),
                    Some(first_source_value),
                    Some(second_source_register),
                    Some(second_source_value),
                    value,
                    signed_overflow,
                    update_overflow_spr2(self.spr2, pc, signed_overflow),
                )
            }
            AicDecoderHint::ScalarKey2ShiftLeft {
                dtype_field,
                destination_register,
                count_register,
                encoded_immediate,
                ..
            } => {
                if dtype_field != 3 {
                    return Err(ScalarMachineError::UnsupportedWord { pc, word });
                }
                let prior = self.xregs[usize::from(destination_register)];
                let (second_source_register, second_source_value, shift) =
                    if let Some(register) = count_register {
                        let count_value = self.xregs[usize::from(register)];
                        (
                            Some(register),
                            Some(count_value),
                            (count_value & 0x3f) as u32,
                        )
                    } else {
                        (None, None, u32::from(encoded_immediate))
                    };
                (
                    destination_register,
                    Some(destination_register),
                    Some(prior),
                    second_source_register,
                    second_source_value,
                    prior << shift,
                    false,
                    self.spr2,
                )
            }
            AicDecoderHint::ScalarKey2ShiftRight {
                dtype_field,
                destination_register,
                count_register,
                encoded_immediate,
                ..
            } => {
                if !matches!(dtype_field, 0 | 1) {
                    return Err(ScalarMachineError::UnsupportedWord { pc, word });
                }
                let prior = self.xregs[usize::from(destination_register)];
                let (second_source_register, second_source_value, shift) =
                    if let Some(register) = count_register {
                        let count_value = self.xregs[usize::from(register)];
                        (
                            Some(register),
                            Some(count_value),
                            (count_value & 0x3f) as u32,
                        )
                    } else {
                        (None, None, u32::from(encoded_immediate))
                    };
                let value = if dtype_field == 0 {
                    ((prior as i64) >> shift) as u64
                } else {
                    prior >> shift
                };
                (
                    destination_register,
                    Some(destination_register),
                    Some(prior),
                    second_source_register,
                    second_source_value,
                    value,
                    false,
                    self.spr2,
                )
            }
            AicDecoderHint::ScalarKey2FindFirst {
                destination_register,
                source_register,
                find_set,
            } => {
                let source_value = self.xregs[usize::from(source_register)];
                let index = if find_set {
                    source_value.trailing_zeros()
                } else {
                    source_value.trailing_ones()
                };
                let value = if index == u64::BITS {
                    u64::MAX
                } else {
                    u64::from(index)
                };
                (
                    destination_register,
                    Some(source_register),
                    Some(source_value),
                    None,
                    None,
                    value,
                    false,
                    self.spr2,
                )
            }
            AicDecoderHint::ScalarKey2MoveRegister {
                destination_register,
                source_register,
                ..
            } => {
                let Some(&source_value) = self.xregs.get(usize::from(source_register)) else {
                    return Err(ScalarMachineError::UnsupportedWord { pc, word });
                };
                if self.xregs.get(usize::from(destination_register)).is_none() {
                    return Err(ScalarMachineError::UnsupportedWord { pc, word });
                }
                (
                    destination_register,
                    Some(source_register),
                    Some(source_value),
                    None,
                    None,
                    source_value,
                    false,
                    self.spr2,
                )
            }
            AicDecoderHint::ScalarKey2Negate {
                dtype_field,
                destination_register,
                source_register,
                ..
            } => {
                if dtype_field != 0 {
                    return Err(ScalarMachineError::UnsupportedWord { pc, word });
                }
                let Some(&source_value) = self.xregs.get(usize::from(source_register)) else {
                    return Err(ScalarMachineError::UnsupportedWord { pc, word });
                };
                if self.xregs.get(usize::from(destination_register)).is_none() {
                    return Err(ScalarMachineError::UnsupportedWord { pc, word });
                }
                let signed_overflow = source_value == i64::MIN as u64;
                (
                    destination_register,
                    Some(source_register),
                    Some(source_value),
                    None,
                    None,
                    source_value.wrapping_neg(),
                    signed_overflow,
                    update_neg_overflow_spr2(self.architecture, self.spr2, pc, signed_overflow),
                )
            }
            AicDecoderHint::ScalarKey2ZeroExtend {
                width,
                destination_register,
                source_register,
                ..
            } => {
                let Some(&source_value) = self.xregs.get(usize::from(source_register)) else {
                    return Err(ScalarMachineError::UnsupportedWord { pc, word });
                };
                if self.xregs.get(usize::from(destination_register)).is_none() {
                    return Err(ScalarMachineError::UnsupportedWord { pc, word });
                }
                (
                    destination_register,
                    Some(source_register),
                    Some(source_value),
                    None,
                    None,
                    source_value & width.mask(),
                    false,
                    self.spr2,
                )
            }
            AicDecoderHint::ScalarKey2SignExtend {
                width_bits,
                destination_register,
                source_register,
            } => {
                let Some(&source_value) = self.xregs.get(usize::from(source_register)) else {
                    return Err(ScalarMachineError::UnsupportedWord { pc, word });
                };
                if self.xregs.get(usize::from(destination_register)).is_none() {
                    return Err(ScalarMachineError::UnsupportedWord { pc, word });
                }
                let shift = 64 - u32::from(width_bits);
                (
                    destination_register,
                    Some(source_register),
                    Some(source_value),
                    None,
                    None,
                    (((source_value << shift) as i64) >> shift) as u64,
                    false,
                    self.spr2,
                )
            }
            AicDecoderHint::ScalarKey2Insert {
                destination_register,
                source_register,
                least_significant_bit,
                width_bits,
                ..
            } => {
                let Some(&source_value) = self.xregs.get(usize::from(source_register)) else {
                    return Err(ScalarMachineError::UnsupportedWord { pc, word });
                };
                let Some(&prior) = self.xregs.get(usize::from(destination_register)) else {
                    return Err(ScalarMachineError::UnsupportedWord { pc, word });
                };
                if u16::from(least_significant_bit) + u16::from(width_bits) > 64 {
                    return Err(ScalarMachineError::UnsupportedWord { pc, word });
                }
                let mask = ((1_u64 << width_bits) - 1) << least_significant_bit;
                (
                    destination_register,
                    Some(source_register),
                    Some(source_value),
                    None,
                    None,
                    (prior & !mask) | ((source_value << least_significant_bit) & mask),
                    false,
                    self.spr2,
                )
            }
            AicDecoderHint::ScalarKey2InsertImmediate {
                destination_register,
                position,
                immediate,
                extended,
                ..
            } => {
                let Some(&prior) = self.xregs.get(usize::from(destination_register)) else {
                    return Err(ScalarMachineError::UnsupportedWord { pc, word });
                };
                let width = if extended {
                    8
                } else {
                    (u8::BITS - immediate.leading_zeros()).max(1)
                };
                if u32::from(position) + width > 64 {
                    return Err(ScalarMachineError::UnsupportedWord { pc, word });
                }
                let mask = ((1_u64 << width) - 1) << position;
                (
                    destination_register,
                    Some(destination_register),
                    Some(prior),
                    None,
                    None,
                    (prior & !mask) | ((u64::from(immediate) << position) & mask),
                    false,
                    self.spr2,
                )
            }
            AicDecoderHint::ScalarKey2BitSet {
                destination_register,
                source_register,
                set_bit,
                ..
            } => {
                let Some(&source_value) = self.xregs.get(usize::from(source_register)) else {
                    return Err(ScalarMachineError::UnsupportedWord { pc, word });
                };
                let Some(&prior) = self.xregs.get(usize::from(destination_register)) else {
                    return Err(ScalarMachineError::UnsupportedWord { pc, word });
                };
                let mask = 1_u64 << (source_value & 63);
                (
                    destination_register,
                    Some(source_register),
                    Some(source_value),
                    None,
                    None,
                    if set_bit { prior | mask } else { prior & !mask },
                    false,
                    self.spr2,
                )
            }
            AicDecoderHint::ScalarKey7 {
                operation,
                destination_register,
                encoded_immediate,
                halfword_lane,
                ..
            } => {
                let prior = self.xregs[usize::from(destination_register)];
                let value = match operation {
                    ScalarKey7Operation::MoveImmediate => u64::from(encoded_immediate),
                    ScalarKey7Operation::MoveKeep => {
                        let lane = halfword_lane.expect("decoder supplies MOVK lane");
                        let shift = u32::from(lane) * 16;
                        (prior & !(0xffff_u64 << shift)) | (u64::from(encoded_immediate) << shift)
                    }
                };
                let source = (operation == ScalarKey7Operation::MoveKeep)
                    .then_some((destination_register, prior));
                (
                    destination_register,
                    source.map(|(register, _)| register),
                    source.map(|(_, value)| value),
                    None,
                    None,
                    value,
                    false,
                    self.spr2,
                )
            }
            AicDecoderHint::ScalarKey8 {
                source_register,
                destination_register: Some(_),
                ..
            } => {
                let source_value = self.xregs[usize::from(source_register)];
                let result = evaluate_scalar_integer_immediate(hint, source_value, pc, self.spr2)?;
                (
                    result.destination_register,
                    Some(source_register),
                    Some(source_value),
                    None,
                    None,
                    result.value,
                    result.signed_overflow,
                    result.spr2,
                )
            }
            _ => return Err(ScalarMachineError::UnsupportedWord { pc, word }),
        };
        let prior_destination_value = self.xregs[usize::from(destination_register)];
        let prior_spr2 = self.spr2;
        self.xregs[usize::from(destination_register)] = value;
        self.spr2 = spr2;
        Ok(ScalarStep {
            pc,
            word,
            destination_register,
            prior_destination_value,
            source_register,
            source_value,
            second_source_register,
            second_source_value,
            value,
            signed_overflow,
            prior_spr2,
            spr2,
        })
    }

    pub fn execute_instruction<B: ScalarMemoryBus>(
        &mut self,
        pc: u64,
        word: u32,
        bus: &mut B,
    ) -> Result<ScalarInstructionStep, ScalarInstructionError<B::Error>> {
        if let Some(instruction) = C310PushPbInstruction::decode(self.architecture, word) {
            let step = instruction.resolve(pc, &self.xregs);
            return match bus
                .execute_c310_push_pb(step)
                .map_err(ScalarInstructionError::PushPbBackend)?
            {
                C310PushPbDisposition::Accepted => Ok(ScalarInstructionStep::PushPb(step)),
                C310PushPbDisposition::Stalled => {
                    Err(ScalarInstructionError::PushPbStalled { pc, word })
                }
                C310PushPbDisposition::Unsupported => {
                    Err(ScalarInstructionError::PushPbUnsupported { pc, word })
                }
            };
        }
        if let Some(instruction) = C310BufferInstruction::decode(self.architecture, word) {
            let step = instruction.resolve(pc, &self.xregs);
            return match bus
                .execute_c310_buffer(step)
                .map_err(ScalarInstructionError::BufferBackend)?
            {
                C310BufferDisposition::Accepted => Ok(ScalarInstructionStep::Buffer(step)),
                C310BufferDisposition::Stalled => {
                    Err(ScalarInstructionError::BufferStalled { pc, word })
                }
                C310BufferDisposition::Unsupported => {
                    Err(ScalarInstructionError::BufferUnsupported { pc, word })
                }
            };
        }
        if matches!(AicClass::from_word(word), AicClass::FlowControl) {
            if let Some(step) = PipelineBarrierStep::decode(self.architecture, pc, word) {
                if !bus
                    .synchronize_barrier(step)
                    .map_err(ScalarInstructionError::SynchronizationBackend)?
                {
                    return Err(ScalarInstructionError::SynchronizationUnsupported { pc, word });
                }
                return Ok(ScalarInstructionStep::Barrier(step));
            }
            if let Some(instruction) = DcciInstruction::decode(self.architecture, word) {
                let step = instruction.resolve(pc, &self.xregs);
                if !bus
                    .maintain_data_cache(step)
                    .map_err(ScalarInstructionError::CacheBackend)?
                {
                    return Err(ScalarInstructionError::CacheMaintenanceUnsupported { pc, word });
                }
                return Ok(ScalarInstructionStep::Dcci(step));
            }
            if let Some(step) = DsbStep::decode(pc, word) {
                if !bus
                    .synchronize_pipeline(step)
                    .map_err(ScalarInstructionError::SynchronizationBackend)?
                {
                    return Err(ScalarInstructionError::SynchronizationUnsupported { pc, word });
                }
                return Ok(ScalarInstructionStep::Dsb(step));
            }
            return Ok(ScalarInstructionStep::Flow(
                self.execute_flow_word(pc, word)?,
            ));
        }
        if self.architecture == Architecture::Dav2201 && C220MovemaskHint::from_word(word).is_some()
        {
            return Ok(ScalarInstructionStep::C220Movemask(
                self.execute_c220_movemask_word(pc, word)?,
            ));
        }
        if self.architecture == Architecture::Dav3510
            && C310ObservedMovemaskHint::from_word(word).is_some()
        {
            return Ok(ScalarInstructionStep::C310Movemask(
                self.execute_c310_movemask_word(pc, word)?,
            ));
        }
        match AicDecoderHint::from_word(self.architecture, word) {
            Some(AicDecoderHint::ScalarCompare { .. }) => Ok(ScalarInstructionStep::Compare(
                self.execute_compare_word(pc, word)?,
            )),
            Some(AicDecoderHint::ScalarCompareRegister { .. }) => {
                Ok(ScalarInstructionStep::CompareRegister(
                    self.execute_compare_register_word(pc, word)?,
                ))
            }
            Some(AicDecoderHint::ScalarCompareImmediate { .. }) => {
                Ok(ScalarInstructionStep::CompareImmediate(
                    self.execute_compare_immediate_word(pc, word)?,
                ))
            }
            Some(AicDecoderHint::ScalarSelect { .. }) => Ok(ScalarInstructionStep::Select(
                self.execute_select_word(pc, word)?,
            )),
            Some(AicDecoderHint::ScalarPairLoad { .. }) => Ok(ScalarInstructionStep::PairLoad(
                self.execute_pair_load_word(pc, word, bus)?,
            )),
            Some(AicDecoderHint::ScalarPairStore { .. }) => Ok(ScalarInstructionStep::PairStore(
                self.execute_pair_store_word(pc, word, bus)?,
            )),
            Some(AicDecoderHint::ScalarIndexedLoad { .. }) => Ok(
                ScalarInstructionStep::IndexedLoad(self.execute_indexed_load_word(pc, word, bus)?),
            ),
            Some(AicDecoderHint::ScalarIndexedImmediateStore { .. }) => {
                Ok(ScalarInstructionStep::IndexedImmediateStore(
                    self.execute_indexed_immediate_store_word(pc, word, bus)?,
                ))
            }
            Some(AicDecoderHint::ScalarKey8 {
                operation: ScalarKey8Operation::DcPreload,
                ..
            }) => Ok(ScalarInstructionStep::CacheHint(
                self.execute_cache_hint_word(pc, word)?,
            )),
            Some(AicDecoderHint::ScalarLoadStoreImmediate { .. }) => Ok(
                ScalarInstructionStep::Memory(self.execute_memory_word(pc, word, bus)?),
            ),
            Some(AicDecoderHint::ScalarStoreImmediate { .. }) => {
                Ok(ScalarInstructionStep::ImmediateStore(
                    self.execute_immediate_store_word(pc, word, bus)?,
                ))
            }
            Some(AicDecoderHint::ScalarKey2MoveFromSpr { .. }) => Ok(
                ScalarInstructionStep::SprRead(self.execute_spr_read_word(pc, word)?),
            ),
            Some(AicDecoderHint::ScalarKey2MoveToSpr { .. }) => Ok(
                ScalarInstructionStep::SprWrite(self.execute_spr_word(pc, word)?),
            ),
            _ => Ok(ScalarInstructionStep::Register(
                self.execute_word(pc, word)?,
            )),
        }
    }

    pub fn execute_memory_word<B: ScalarMemoryBus>(
        &mut self,
        pc: u64,
        word: u32,
        bus: &mut B,
    ) -> Result<ScalarMemoryStep, ScalarMemoryExecutionError<B::Error>> {
        let Some(
            hint @ AicDecoderHint::ScalarLoadStoreImmediate {
                operation,
                width_bytes,
                data_register,
                base_register,
                post_index,
                sign_extend,
                ..
            },
        ) = AicDecoderHint::from_word(self.architecture, word)
        else {
            return Err(ScalarMemoryExecutionError::UnsupportedWord { pc, word });
        };
        if post_index
            && matches!(operation, ScalarLoadStoreOperation::Load)
            && data_register == base_register
        {
            return Err(ScalarMemoryExecutionError::AliasedPostIndex {
                pc,
                word,
                register: data_register,
            });
        }
        let prior_base_value = self.xregs[usize::from(base_register)];
        let prior_data_value = self.xregs[usize::from(data_register)];
        let effect = hint
            .scalar_address_effect(prior_base_value)
            .expect("the load/store hint has an address effect");
        let width = usize::from(width_bytes);
        let mut bytes = [0_u8; 8];
        let sign_extension_requested = sign_extend == Some(true) && width_bytes < 8;
        let data_value = match operation {
            ScalarLoadStoreOperation::Load => {
                bus.read(effect.effective_address, &mut bytes[..width])
                    .map_err(ScalarMemoryExecutionError::Backend)?;
                let raw = u64::from_le_bytes(bytes);
                if sign_extension_requested {
                    let bits = u32::from(width_bytes) * 8;
                    (((raw << (64 - bits)) as i64) >> (64 - bits)) as u64
                } else {
                    raw
                }
            }
            ScalarLoadStoreOperation::Store => {
                bytes[..width].copy_from_slice(&prior_data_value.to_le_bytes()[..width]);
                bus.write(effect.effective_address, &bytes[..width])
                    .map_err(ScalarMemoryExecutionError::Backend)?;
                prior_data_value
            }
        };
        if matches!(operation, ScalarLoadStoreOperation::Load) {
            self.xregs[usize::from(data_register)] = data_value;
        }
        if let Some(updated_base) = effect.updated_base {
            self.xregs[usize::from(base_register)] = updated_base;
        }
        Ok(ScalarMemoryStep {
            pc,
            word,
            operation,
            effective_address: effect.effective_address,
            width_bytes,
            data_register,
            prior_data_value,
            data_value,
            base_register,
            prior_base_value,
            updated_base: effect.updated_base,
            bytes,
            sign_extension_requested,
        })
    }

    pub fn execute_indexed_load_word<B: ScalarMemoryBus>(
        &mut self,
        pc: u64,
        word: u32,
        bus: &mut B,
    ) -> Result<ScalarIndexedLoadStep, ScalarMemoryExecutionError<B::Error>> {
        let Some(AicDecoderHint::ScalarIndexedLoad {
            width_bytes,
            destination_register,
            base_register,
            offset_register,
        }) = AicDecoderHint::from_word(self.architecture, word)
        else {
            return Err(ScalarMemoryExecutionError::UnsupportedWord { pc, word });
        };
        let base_value = self.xregs[usize::from(base_register)];
        let offset_value = self.xregs[usize::from(offset_register)];
        let effective_address = base_value.wrapping_add(offset_value * u64::from(width_bytes));
        let mut bytes = [0_u8; 8];
        bus.read(effective_address, &mut bytes[..usize::from(width_bytes)])
            .map_err(ScalarMemoryExecutionError::Backend)?;
        let value = u64::from_le_bytes(bytes);
        let prior_destination_value = self.xregs[usize::from(destination_register)];
        self.xregs[usize::from(destination_register)] = value;
        Ok(ScalarIndexedLoadStep {
            pc,
            word,
            effective_address,
            width_bytes,
            destination_register,
            prior_destination_value,
            value,
            base_register,
            base_value,
            offset_register,
            offset_value,
            bytes,
        })
    }

    pub fn execute_indexed_immediate_store_word<B: ScalarMemoryBus>(
        &mut self,
        pc: u64,
        word: u32,
        bus: &mut B,
    ) -> Result<ScalarIndexedImmediateStoreStep, ScalarMemoryExecutionError<B::Error>> {
        let Some(AicDecoderHint::ScalarIndexedImmediateStore {
            width_bytes,
            base_register,
            offset_register,
            value,
        }) = AicDecoderHint::from_word(self.architecture, word)
        else {
            return Err(ScalarMemoryExecutionError::UnsupportedWord { pc, word });
        };
        let base_value = self.xregs[usize::from(base_register)];
        let offset_value = self.xregs[usize::from(offset_register)];
        let effective_address = base_value.wrapping_add(offset_value * u64::from(width_bytes));
        let mut bytes = [0_u8; 8];
        match value {
            ScalarStoreImmediateValue::Zero => {}
            ScalarStoreImmediateValue::One => bytes[0] = 1,
            ScalarStoreImmediateValue::Ones => bytes.fill(0xff),
        }
        bus.write(effective_address, &bytes[..usize::from(width_bytes)])
            .map_err(ScalarMemoryExecutionError::Backend)?;
        Ok(ScalarIndexedImmediateStoreStep {
            pc,
            word,
            effective_address,
            width_bytes,
            base_register,
            base_value,
            offset_register,
            offset_value,
            value,
            bytes,
        })
    }

    pub fn execute_immediate_store_word<B: ScalarMemoryBus>(
        &mut self,
        pc: u64,
        word: u32,
        bus: &mut B,
    ) -> Result<ScalarImmediateStoreStep, ScalarMemoryExecutionError<B::Error>> {
        let Some(AicDecoderHint::ScalarStoreImmediate {
            width_bytes,
            base_register,
            signed_offset,
            post_index: false,
            value,
            ..
        }) = AicDecoderHint::from_word(self.architecture, word)
        else {
            return Err(ScalarMemoryExecutionError::UnsupportedWord { pc, word });
        };
        let prior_base_value = self.xregs[usize::from(base_register)];
        let effective_address = prior_base_value.wrapping_add(signed_offset as i64 as u64);
        let mut bytes = [0_u8; 8];
        match value {
            ScalarStoreImmediateValue::Zero => {}
            ScalarStoreImmediateValue::One => bytes[0] = 1,
            ScalarStoreImmediateValue::Ones => bytes.fill(0xff),
        }
        bus.write(effective_address, &bytes[..usize::from(width_bytes)])
            .map_err(ScalarMemoryExecutionError::Backend)?;
        Ok(ScalarImmediateStoreStep {
            pc,
            word,
            effective_address,
            width_bytes,
            base_register,
            prior_base_value,
            value,
            bytes,
        })
    }

    pub fn execute_pair_load_word<B: ScalarMemoryBus>(
        &mut self,
        pc: u64,
        word: u32,
        bus: &mut B,
    ) -> Result<ScalarPairLoadStep, ScalarMemoryExecutionError<B::Error>> {
        let Some(AicDecoderHint::ScalarPairLoad {
            width_bytes,
            first_destination_register,
            second_destination_register,
            base_register,
            signed_offset,
            sign_extend,
            ..
        }) = AicDecoderHint::from_word(self.architecture, word)
        else {
            return Err(ScalarMemoryExecutionError::UnsupportedWord { pc, word });
        };
        let base_value = self.xregs[usize::from(base_register)];
        let first_address = base_value.wrapping_add(signed_offset as i64 as u64);
        let second_address = first_address.wrapping_add(u64::from(width_bytes));
        let width = usize::from(width_bytes);
        let mut first_bytes = [0_u8; 8];
        let mut second_bytes = [0_u8; 8];
        bus.read(first_address, &mut first_bytes[..width])
            .map_err(ScalarMemoryExecutionError::Backend)?;
        bus.read(second_address, &mut second_bytes[..width])
            .map_err(ScalarMemoryExecutionError::Backend)?;
        let decode = |bytes: [u8; 8]| {
            let raw = u64::from_le_bytes(bytes);
            if sign_extend && width_bytes < 8 {
                let bits = u32::from(width_bytes) * 8;
                (((raw << (64 - bits)) as i64) >> (64 - bits)) as u64
            } else {
                raw
            }
        };
        let first_value = decode(first_bytes);
        let second_value = decode(second_bytes);
        let first_prior_value = self.xregs[usize::from(first_destination_register)];
        let second_prior_value = self.xregs[usize::from(second_destination_register)];
        self.xregs[usize::from(first_destination_register)] = first_value;
        self.xregs[usize::from(second_destination_register)] = second_value;
        Ok(ScalarPairLoadStep {
            pc,
            word,
            first_address,
            second_address,
            width_bytes,
            base_register,
            base_value,
            first_destination_register,
            first_prior_value,
            first_value,
            first_bytes,
            second_destination_register,
            second_prior_value,
            second_value,
            second_bytes,
            sign_extension_requested: sign_extend && width_bytes < 8,
        })
    }

    pub fn execute_pair_store_word<B: ScalarMemoryBus>(
        &mut self,
        pc: u64,
        word: u32,
        bus: &mut B,
    ) -> Result<ScalarPairStoreStep, ScalarMemoryExecutionError<B::Error>> {
        let Some(AicDecoderHint::ScalarPairStore {
            width_bytes,
            first_source_register,
            second_source_register,
            base_register,
            signed_offset,
            ..
        }) = AicDecoderHint::from_word(self.architecture, word)
        else {
            return Err(ScalarMemoryExecutionError::UnsupportedWord { pc, word });
        };
        let base_value = self.xregs[usize::from(base_register)];
        let first_address = base_value.wrapping_add(signed_offset as i64 as u64);
        let second_address = first_address.wrapping_add(u64::from(width_bytes));
        let first_value = self.xregs[usize::from(first_source_register)];
        let second_value = self.xregs[usize::from(second_source_register)];
        let width = usize::from(width_bytes);
        let mut first_bytes = [0_u8; 8];
        let mut second_bytes = [0_u8; 8];
        first_bytes[..width].copy_from_slice(&first_value.to_le_bytes()[..width]);
        second_bytes[..width].copy_from_slice(&second_value.to_le_bytes()[..width]);
        bus.write(first_address, &first_bytes[..width])
            .map_err(ScalarMemoryExecutionError::Backend)?;
        bus.write(second_address, &second_bytes[..width])
            .map_err(ScalarMemoryExecutionError::Backend)?;
        Ok(ScalarPairStoreStep {
            pc,
            word,
            first_address,
            second_address,
            width_bytes,
            base_register,
            base_value,
            first_source_register,
            first_value,
            first_bytes,
            second_source_register,
            second_value,
            second_bytes,
        })
    }
}

fn evaluate_integer_compare(
    dtype_field: u8,
    condition_field: u8,
    first: u64,
    second: u64,
) -> Option<bool> {
    if condition_field > 5 {
        return None;
    }
    match dtype_field {
        0 => Some(compare_values(first as i64, second as i64, condition_field)),
        1 => Some(compare_values(first, second, condition_field)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captured_indexed_loads_scale_offset_and_preserve_machine_on_read_failure() {
        for (architecture, pc, word, destination, base, offset, base_value) in [
            (
                Architecture::Dav2201,
                0x1131_23c0,
                0x011c_b600,
                14,
                11,
                12,
                0x001c_79c8,
            ),
            (
                Architecture::Dav3510,
                0x10d0_d660,
                0x0103_2980,
                1,
                18,
                19,
                0x0010_7940,
            ),
            (
                Architecture::Dav2201,
                0x1131_243c,
                0x0124_a880,
                18,
                10,
                17,
                0x001c_79d8,
            ),
            (
                Architecture::Dav2201,
                0x1131_25d4,
                0x0126_f800,
                19,
                15,
                16,
                0x001c_79e8,
            ),
        ] {
            let mut machine = ScalarMachine::new(architecture, [0; 32], 0);
            machine.set_xreg(destination, 0x55).unwrap();
            machine.set_xreg(base, base_value).unwrap();
            machine.set_xreg(offset, 3).unwrap();
            let before = machine.clone();
            let mut bus = TestBus::new(base_value);
            bus.bytes[3] = 0xa7;
            bus.fail = true;
            assert!(matches!(
                machine.execute_instruction(pc, word, &mut bus),
                Err(ScalarInstructionError::Memory(
                    ScalarMemoryExecutionError::Backend(_)
                ))
            ));
            assert_eq!(machine, before);
            assert_eq!(bus.accesses, 0);

            bus.fail = false;
            let ScalarInstructionStep::IndexedLoad(step) =
                machine.execute_instruction(pc, word, &mut bus).unwrap()
            else {
                panic!("expected indexed scalar load")
            };
            assert_eq!(step.effective_address, base_value + 3);
            assert_eq!(step.width_bytes, 1);
            assert_eq!(step.destination_register, destination);
            assert_eq!(step.prior_destination_value, 0x55);
            assert_eq!(step.value, 0xa7);
            assert_eq!(step.base_register, base);
            assert_eq!(step.base_value, base_value);
            assert_eq!(step.offset_register, offset);
            assert_eq!(step.offset_value, 3);
            assert_eq!(step.bytes[0], 0xa7);
            assert_eq!(machine.xregs()[usize::from(destination)], 0xa7);
            assert_eq!(machine.xregs()[usize::from(base)], base_value);
            assert_eq!(machine.xregs()[usize::from(offset)], 3);
            assert_eq!(bus.accesses, 1);
        }
    }

    #[test]
    fn c310_indexed_loads_scale_offsets_by_width() {
        for (dtype, width_bytes) in [(0, 1_u8), (1, 2), (2, 4), (3, 8)] {
            let word = 0x0103_4b00 | (dtype << 22);
            let mut machine = ScalarMachine::new(Architecture::Dav3510, [0; 32], 0);
            machine.set_xreg(1, 0x55).unwrap();
            machine.set_xreg(20, 0x1000).unwrap();
            machine.set_xreg(22, 2).unwrap();
            let mut bus = TestBus::new(0x1000);
            for (index, byte) in bus.bytes.iter_mut().enumerate() {
                *byte = index as u8;
            }
            let start = usize::from(width_bytes) * 2;
            let width = usize::from(width_bytes);
            let mut expected_bytes = [0_u8; 8];
            expected_bytes[..width].copy_from_slice(&bus.bytes[start..start + width]);
            let ScalarInstructionStep::IndexedLoad(step) = machine
                .execute_instruction(0x10d0_d6b8, word, &mut bus)
                .unwrap()
            else {
                panic!("expected indexed scalar load");
            };
            assert_eq!(step.effective_address, 0x1000 + start as u64);
            assert_eq!(step.width_bytes, width_bytes);
            assert_eq!(step.value, u64::from_le_bytes(expected_bytes));
            assert_eq!(step.prior_destination_value, 0x55);
            assert_eq!(machine.xregs()[1], step.value);
            assert_eq!(machine.xregs()[20], 0x1000);
            assert_eq!(machine.xregs()[22], 2);
            assert_eq!(bus.accesses, 1);
        }
    }

    #[test]
    fn captured_indexed_immediate_stores_write_one_byte_and_preserve_machine() {
        for (architecture, pc, word, base, offset, base_value) in [
            (
                Architecture::Dav2201,
                0x1131_23cc,
                0x0e00_b601,
                11,
                12,
                0x001c_79c8,
            ),
            (
                Architecture::Dav3510,
                0x10d0_d66c,
                0x0e01_2981,
                18,
                19,
                0x0010_7940,
            ),
            (
                Architecture::Dav2201,
                0x1131_2448,
                0x0e00_a881,
                10,
                17,
                0x001c_79d8,
            ),
            (
                Architecture::Dav2201,
                0x1131_25e0,
                0x0e00_f801,
                15,
                16,
                0x001c_79e8,
            ),
        ] {
            let mut machine = ScalarMachine::new(architecture, [0; 32], 0);
            machine.set_xreg(base, base_value).unwrap();
            machine.set_xreg(offset, 3).unwrap();
            let before = machine.clone();
            let mut bus = TestBus::new(base_value);
            bus.bytes.fill(0xa5);
            bus.fail = true;
            assert!(matches!(
                machine.execute_instruction(pc, word, &mut bus),
                Err(ScalarInstructionError::Memory(
                    ScalarMemoryExecutionError::Backend(_)
                ))
            ));
            assert_eq!(machine, before);
            assert_eq!(bus.bytes, [0xa5; 32]);
            assert_eq!(bus.accesses, 0);

            bus.fail = false;
            let ScalarInstructionStep::IndexedImmediateStore(step) =
                machine.execute_instruction(pc, word, &mut bus).unwrap()
            else {
                panic!("expected indexed immediate store")
            };
            assert_eq!(step.effective_address, base_value + 3);
            assert_eq!(step.width_bytes, 1);
            assert_eq!(step.base_register, base);
            assert_eq!(step.base_value, base_value);
            assert_eq!(step.offset_register, offset);
            assert_eq!(step.offset_value, 3);
            assert_eq!(step.value, ScalarStoreImmediateValue::One);
            assert_eq!(step.bytes[0], 1);
            assert_eq!(bus.bytes[3], 1);
            assert!(bus.bytes[..3].iter().all(|byte| *byte == 0xa5));
            assert!(bus.bytes[4..].iter().all(|byte| *byte == 0xa5));
            assert_eq!(bus.accesses, 1);
            assert_eq!(machine, before);
        }
    }

    #[test]
    fn c310_indexed_immediate_stores_scale_offsets_and_write_selected_width() {
        for (dtype, width_bytes) in [(0, 1_u8), (1, 2), (2, 4), (3, 8)] {
            for value_bits in 0..=2 {
                let word = 0x0e01_4b00 | (dtype << 22) | value_bits;
                let mut machine = ScalarMachine::new(Architecture::Dav3510, [0; 32], 0);
                machine.set_xreg(20, 0x1000).unwrap();
                machine.set_xreg(22, 2).unwrap();
                let before = machine.clone();
                let mut bus = TestBus::new(0x1000);
                bus.bytes.fill(0xa5);
                let ScalarInstructionStep::IndexedImmediateStore(step) = machine
                    .execute_instruction(0x10d0_d6c8, word, &mut bus)
                    .unwrap()
                else {
                    panic!("expected indexed immediate store");
                };
                let start = usize::from(width_bytes) * 2;
                assert_eq!(step.effective_address, 0x1000 + start as u64);
                assert_eq!(step.width_bytes, width_bytes);
                let mut expected_bytes = [0_u8; 8];
                match value_bits {
                    1 => expected_bytes[0] = 1,
                    2 => expected_bytes.fill(0xff),
                    _ => {}
                }
                assert_eq!(
                    &bus.bytes[start..start + usize::from(width_bytes)],
                    &expected_bytes[..usize::from(width_bytes)]
                );
                assert_eq!(bus.bytes[start - 1], 0xa5);
                assert_eq!(bus.bytes[start + usize::from(width_bytes)], 0xa5);
                assert_eq!(machine, before);
                assert_eq!(bus.accesses, 1);
            }
        }
    }

    struct TestBus {
        base: u64,
        bytes: [u8; 32],
        fail: bool,
        accesses: usize,
        supports_cache_maintenance: bool,
        cache_events: Vec<DcciStep>,
        supports_synchronization: bool,
        synchronization_events: Vec<DsbStep>,
        supports_barrier: bool,
        barrier_events: Vec<PipelineBarrierStep>,
    }

    impl TestBus {
        fn new(base: u64) -> Self {
            Self {
                base,
                bytes: [0; 32],
                fail: false,
                accesses: 0,
                supports_cache_maintenance: false,
                cache_events: Vec::new(),
                supports_synchronization: false,
                synchronization_events: Vec::new(),
                supports_barrier: false,
                barrier_events: Vec::new(),
            }
        }

        fn range(&self, address: u64, len: usize) -> std::io::Result<std::ops::Range<usize>> {
            let start = usize::try_from(
                address
                    .checked_sub(self.base)
                    .ok_or_else(|| std::io::Error::from(std::io::ErrorKind::InvalidInput))?,
            )
            .map_err(|_| std::io::Error::from(std::io::ErrorKind::InvalidInput))?;
            let end = start
                .checked_add(len)
                .filter(|end| *end <= self.bytes.len())
                .ok_or_else(|| std::io::Error::from(std::io::ErrorKind::InvalidInput))?;
            Ok(start..end)
        }
    }

    impl ScalarMemoryBus for TestBus {
        type Error = std::io::Error;

        fn read(&mut self, address: u64, destination: &mut [u8]) -> std::io::Result<()> {
            if self.fail {
                return Err(std::io::Error::from(std::io::ErrorKind::BrokenPipe));
            }
            let range = self.range(address, destination.len())?;
            destination.copy_from_slice(&self.bytes[range]);
            self.accesses += 1;
            Ok(())
        }

        fn write(&mut self, address: u64, source: &[u8]) -> std::io::Result<()> {
            if self.fail {
                return Err(std::io::Error::from(std::io::ErrorKind::BrokenPipe));
            }
            let range = self.range(address, source.len())?;
            self.bytes[range].copy_from_slice(source);
            self.accesses += 1;
            Ok(())
        }

        fn maintain_data_cache(&mut self, step: DcciStep) -> std::io::Result<bool> {
            if self.fail {
                return Err(std::io::Error::from(std::io::ErrorKind::BrokenPipe));
            }
            if self.supports_cache_maintenance {
                self.cache_events.push(step);
            }
            Ok(self.supports_cache_maintenance)
        }

        fn synchronize_pipeline(&mut self, step: DsbStep) -> std::io::Result<bool> {
            if self.fail {
                return Err(std::io::Error::from(std::io::ErrorKind::BrokenPipe));
            }
            if self.supports_synchronization {
                self.synchronization_events.push(step);
            }
            Ok(self.supports_synchronization)
        }

        fn synchronize_barrier(&mut self, step: PipelineBarrierStep) -> std::io::Result<bool> {
            if self.fail {
                return Err(std::io::Error::from(std::io::ErrorKind::BrokenPipe));
            }
            if self.supports_barrier {
                self.barrier_events.push(step);
            }
            Ok(self.supports_barrier)
        }
    }

    #[test]
    fn barrier_requires_explicit_pipeline_completion() {
        for (architecture, pc, word) in [
            (Architecture::Dav2201, 0x1131_2090, 0x40e0_1800),
            (Architecture::Dav2201, 0x1131_2638, 0x40e0_0400),
            (Architecture::Dav3510, 0x10d0_d090, 0x40e0_1800),
        ] {
            let mut machine = ScalarMachine::from_pem_initial_state(architecture);
            let before = machine.clone();
            let mut bus = TestBus::new(0);
            assert!(matches!(
                machine.execute_instruction(pc, word, &mut bus),
                Err(ScalarInstructionError::SynchronizationUnsupported { .. })
            ));
            assert_eq!(machine, before);
            assert!(bus.barrier_events.is_empty());

            bus.fail = true;
            assert!(matches!(
                machine.execute_instruction(pc, word, &mut bus),
                Err(ScalarInstructionError::SynchronizationBackend(_))
            ));
            assert_eq!(machine, before);
            assert!(bus.barrier_events.is_empty());

            bus.fail = false;
            bus.supports_barrier = true;
            let ScalarInstructionStep::Barrier(step) =
                machine.execute_instruction(pc, word, &mut bus).unwrap()
            else {
                panic!("expected barrier step")
            };
            assert_eq!(step.pc, pc);
            assert_eq!(step.word, word);
            assert_eq!(bus.barrier_events, vec![step]);
            assert_eq!(bus.accesses, 0);
            assert_eq!(machine, before);
        }
    }

    #[test]
    fn dsb_requires_an_explicit_synchronization_backend() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut machine = ScalarMachine::from_pem_initial_state(architecture);
            let before = machine.clone();
            let mut bus = TestBus::new(0);

            assert!(matches!(
                machine.execute_instruction(0x1131_2170, 0x41c1_0000, &mut bus),
                Err(ScalarInstructionError::SynchronizationUnsupported { .. })
            ));
            assert_eq!(machine, before);
            assert!(bus.synchronization_events.is_empty());

            bus.fail = true;
            assert!(matches!(
                machine.execute_instruction(0x1131_2170, 0x41c1_0000, &mut bus),
                Err(ScalarInstructionError::SynchronizationBackend(_))
            ));
            assert_eq!(machine, before);
            assert!(bus.synchronization_events.is_empty());

            bus.fail = false;
            bus.supports_synchronization = true;
            let ScalarInstructionStep::Dsb(step) = machine
                .execute_instruction(0x1131_2170, 0x41c1_0000, &mut bus)
                .unwrap()
            else {
                panic!("expected DSB step")
            };
            assert_eq!(step.scope_field, 1);
            assert_eq!(bus.synchronization_events, vec![step]);
            assert_eq!(bus.accesses, 0);
            assert_eq!(machine, before);
        }
    }

    #[test]
    fn dcci_requires_an_explicit_cache_backend_and_preserves_machine_on_failure() {
        for (architecture, pc, word, source_register, address) in [
            (
                Architecture::Dav2201,
                0x1131_216c,
                0x402c_8000,
                8,
                0x1131_3000,
            ),
            (Architecture::Dav3510, 0x10d0_d8b4, 0x402e_f000, 15, 0),
        ] {
            let mut machine = ScalarMachine::from_pem_initial_state(architecture);
            machine.set_xreg(source_register, 0x1131_3000).unwrap();
            let before = machine.clone();
            let mut bus = TestBus::new(0);

            assert!(matches!(
                machine.execute_instruction(pc, word, &mut bus),
                Err(ScalarInstructionError::CacheMaintenanceUnsupported { .. })
            ));
            assert_eq!(machine, before);
            assert!(bus.cache_events.is_empty());
            assert_eq!(bus.accesses, 0);

            bus.fail = true;
            assert!(matches!(
                machine.execute_instruction(pc, word, &mut bus),
                Err(ScalarInstructionError::CacheBackend(_))
            ));
            assert_eq!(machine, before);
            assert!(bus.cache_events.is_empty());

            bus.fail = false;
            bus.supports_cache_maintenance = true;
            let ScalarInstructionStep::Dcci(step) =
                machine.execute_instruction(pc, word, &mut bus).unwrap()
            else {
                panic!("expected DCCI step")
            };
            assert_eq!(step.effective_address, address);
            assert_eq!(step.source_register, source_register);
            assert_eq!(step.source_value, 0x1131_3000);
            assert_eq!(step.operation_field, 0);
            assert_eq!(bus.cache_events, vec![step]);
            assert_eq!(bus.accesses, 0);
            assert_eq!(machine, before);
        }
    }

    #[test]
    fn unified_scalar_dispatch_preserves_spr_memory_dependency_on_both_architectures() {
        for (architecture, load_word) in [
            (Architecture::Dav2201, 0x03c2_0008),
            (Architecture::Dav3510, 0x1cc2_0008),
        ] {
            let mut machine = ScalarMachine::new(architecture, [0; 32], 0x55);
            machine.set_spr_value(4, 0x1000).unwrap();
            let mut bus = TestBus::new(0x1000);
            let expected = 0x1234_5678_9abc_def0_u64;
            bus.bytes[8..16].copy_from_slice(&expected.to_le_bytes());

            let read = machine
                .execute_instruction(0x200, 0x0200_4880, &mut bus)
                .unwrap();
            let ScalarInstructionStep::SprRead(read) = read else {
                panic!("expected SPR read");
            };
            assert_eq!(read.value, 0x1000);
            assert_eq!(machine.xregs()[0], 0x1000);

            bus.fail = true;
            let before = machine.clone();
            assert!(matches!(
                machine.execute_instruction(0x204, load_word, &mut bus),
                Err(ScalarInstructionError::Memory(
                    ScalarMemoryExecutionError::Backend(_)
                ))
            ));
            assert_eq!(machine, before);
            bus.fail = false;

            let load = machine
                .execute_instruction(0x204, load_word, &mut bus)
                .unwrap();
            let ScalarInstructionStep::Memory(load) = load else {
                panic!("expected memory load");
            };
            assert_eq!(load.effective_address, 0x1008);
            assert_eq!(load.data_value, expected);
            assert_eq!(machine.xregs()[1], expected);
            assert_eq!(bus.accesses, 1);

            let write = machine
                .execute_instruction(0x208, 0x0206_1900, &mut bus)
                .unwrap();
            let ScalarInstructionStep::SprWrite(write) = write else {
                panic!("expected SPR write");
            };
            assert_eq!(write.destination_spr, 3);
            assert_eq!(write.value, expected);
            assert_eq!(machine.spr_value(3), Some(expected));
            assert_eq!(machine.spr2(), 0x55);

            let register = machine
                .execute_instruction(0x20c, 0x0706_0001, &mut bus)
                .unwrap();
            assert!(matches!(register, ScalarInstructionStep::Register(_)));
            assert_eq!(machine.xregs()[3], 1);
            let before = machine.clone();
            assert!(matches!(
                machine.execute_instruction(0x210, 0x6000_0000, &mut bus),
                Err(ScalarInstructionError::Scalar(
                    ScalarMachineError::UnsupportedWord { .. }
                ))
            ));
            assert_eq!(machine, before);
        }
    }

    #[test]
    fn c220_narrow_load_zero_extends_and_negative_offset_selects_access_address() {
        let mut machine = ScalarMachine::new(Architecture::Dav2201, [0; 32], 0x55);
        machine.set_xreg(5, 0x1000).unwrap();
        machine.set_xreg(7, u64::MAX).unwrap();
        let mut bus = TestBus::new(0x1000 - 584);
        bus.bytes[0] = 0x80;
        let word = 0x03ce_5db8 & !(3 << 22);
        let step = machine.execute_memory_word(0x100, word, &mut bus).unwrap();
        assert_eq!(step.operation, ScalarLoadStoreOperation::Load);
        assert_eq!(step.effective_address, 0x1000 - 584);
        assert_eq!(step.width_bytes, 1);
        assert_eq!(step.prior_data_value, u64::MAX);
        assert_eq!(step.data_value, 0x80);
        assert_eq!(machine.xregs()[7], 0x80);
        assert_eq!(machine.xregs()[5], 0x1000);
        assert_eq!(machine.spr2(), 0x55);
    }

    #[test]
    fn pair_load_reads_adjacent_values_and_updates_both_registers() {
        for (architecture, word) in [
            (Architecture::Dav2201, 0x09ca_0190),
            (Architecture::Dav3510, 0x0cca_0190),
        ] {
            let mut machine = ScalarMachine::new(architecture, [0; 32], 0x55);
            machine.set_xreg(0, 0x1000).unwrap();
            machine.set_xreg(5, u64::MAX).unwrap();
            machine.set_xreg(3, u64::MAX).unwrap();
            let mut bus = TestBus::new(0x1000);
            let first = 0x0123_4567_89ab_cdef_u64;
            let second = 0xfedc_ba98_7654_3210_u64;
            bus.bytes[8..16].copy_from_slice(&first.to_le_bytes());
            bus.bytes[16..24].copy_from_slice(&second.to_le_bytes());

            let ScalarInstructionStep::PairLoad(step) =
                machine.execute_instruction(0x200, word, &mut bus).unwrap()
            else {
                panic!("expected pair load");
            };
            assert_eq!(step.first_address, 0x1008);
            assert_eq!(step.second_address, 0x1010);
            assert_eq!(step.first_prior_value, u64::MAX);
            assert_eq!(step.second_prior_value, u64::MAX);
            assert_eq!(step.first_value, first);
            assert_eq!(step.second_value, second);
            assert_eq!(machine.xregs()[5], first);
            assert_eq!(machine.xregs()[3], second);
            assert_eq!(machine.xregs()[0], 0x1000);
            assert_eq!(machine.spr2(), 0x55);
            assert_eq!(bus.accesses, 2);
        }
    }

    #[test]
    fn pair_load_sign_extension_and_negative_offset_are_architecture_specific() {
        for (architecture, key, expected) in [
            (Architecture::Dav2201, 9_u32, 0x80_u64),
            (Architecture::Dav3510, 13_u32, (-128_i64) as u64),
        ] {
            let mut machine = ScalarMachine::new(architecture, [0; 32], 0);
            machine.set_xreg(0, 0x1001).unwrap();
            let mut bus = TestBus::new(0x1000);
            bus.bytes[..2].copy_from_slice(&[0x80, 0x81]);
            let word =
                (0x0cca_0190 & !((0x1f << 24) | (3 << 22) | (0x3f << 1))) | (key << 24) | (63 << 1);
            let step = machine
                .execute_pair_load_word(0x100, word, &mut bus)
                .unwrap();
            assert_eq!(step.first_address, 0x1000);
            assert_eq!(step.second_address, 0x1001);
            assert_eq!(step.first_value, expected);
            assert_eq!(machine.xregs()[5], expected);
        }
    }

    #[test]
    fn pair_load_second_read_failure_preserves_registers() {
        let mut machine = ScalarMachine::new(Architecture::Dav3510, [0; 32], 0);
        machine.set_xreg(0, 0x1010).unwrap();
        machine.set_xreg(5, 7).unwrap();
        machine.set_xreg(3, 9).unwrap();
        let before = machine.clone();
        let mut bus = TestBus::new(0x1000);
        let word = 0x0cca_0190;
        assert!(matches!(
            machine.execute_pair_load_word(0x100, word, &mut bus),
            Err(ScalarMemoryExecutionError::Backend(_))
        ));
        assert_eq!(bus.accesses, 1);
        assert_eq!(machine, before);
    }

    #[test]
    fn pair_store_writes_adjacent_low_bytes_without_changing_registers() {
        for (architecture, word) in [
            (Architecture::Dav2201, 0x09ca_0191),
            (Architecture::Dav3510, 0x0cca_0191),
        ] {
            let mut machine = ScalarMachine::new(architecture, [0; 32], 0x55);
            machine.set_xreg(0, 0x1000).unwrap();
            machine.set_xreg(5, 0x0123_4567_89ab_cdef).unwrap();
            machine.set_xreg(3, 0xfedc_ba98_7654_3210).unwrap();
            let before = machine.clone();
            let mut bus = TestBus::new(0x1000);
            let ScalarInstructionStep::PairStore(step) =
                machine.execute_instruction(0x204, word, &mut bus).unwrap()
            else {
                panic!("expected pair store");
            };
            assert_eq!(step.first_address, 0x1008);
            assert_eq!(step.second_address, 0x1010);
            assert_eq!(bus.bytes[8..16], step.first_bytes);
            assert_eq!(bus.bytes[16..24], step.second_bytes);
            assert_eq!(bus.accesses, 2);
            assert_eq!(machine, before);
        }
    }

    #[test]
    fn pair_store_failure_after_first_write_keeps_registers() {
        let mut machine = ScalarMachine::new(Architecture::Dav3510, [0; 32], 0);
        machine.set_xreg(0, 0x1010).unwrap();
        machine.set_xreg(5, 0x0123_4567_89ab_cdef).unwrap();
        let before = machine.clone();
        let mut bus = TestBus::new(0x1000);
        assert!(matches!(
            machine.execute_pair_store_word(0x100, 0x0cca_0191, &mut bus),
            Err(ScalarMemoryExecutionError::Backend(_))
        ));
        assert_eq!(bus.accesses, 1);
        assert_eq!(bus.bytes[24..32], 0x0123_4567_89ab_cdef_u64.to_le_bytes());
        assert_eq!(machine, before);
    }

    #[test]
    fn pair_store_uses_signed_offset_and_only_selected_width() {
        for (architecture, key) in [
            (Architecture::Dav2201, 9_u32),
            (Architecture::Dav3510, 12_u32),
        ] {
            let mut machine = ScalarMachine::new(architecture, [0; 32], 0);
            machine.set_xreg(0, 0x1001).unwrap();
            machine.set_xreg(5, 0x1234_5678_9abc_def0).unwrap();
            machine.set_xreg(3, 0x1234_5678_9abc_de80).unwrap();
            let mut bus = TestBus::new(0x1000);
            let word = (0x0cca_0190 & !((0x1f << 24) | (3 << 22) | (0x3f << 1)))
                | (key << 24)
                | (63 << 1)
                | 1;
            let step = machine
                .execute_pair_store_word(0x100, word, &mut bus)
                .unwrap();
            assert_eq!(step.first_address, 0x1000);
            assert_eq!(step.second_address, 0x1001);
            assert_eq!(step.first_bytes, [0xf0, 0, 0, 0, 0, 0, 0, 0]);
            assert_eq!(step.second_bytes, [0x80, 0, 0, 0, 0, 0, 0, 0]);
            assert_eq!(bus.bytes[..2], [0xf0, 0x80]);
        }
    }

    #[test]
    fn c310_signed_post_index_load_sign_extends_and_updates_base() {
        let mut machine = ScalarMachine::new(Architecture::Dav3510, [0; 32], 0);
        machine.set_xreg(5, 0x1000).unwrap();
        let mut bus = TestBus::new(0x1000);
        bus.bytes[0] = 0x80;
        let word = 0x1fce_5db8 & !(3 << 22);
        let step = machine.execute_memory_word(0x104, word, &mut bus).unwrap();
        assert_eq!(step.effective_address, 0x1000);
        assert_eq!(step.updated_base, Some(0x1000 - 584));
        assert!(step.sign_extension_requested);
        assert_eq!(step.data_value, (-128_i64) as u64);
        assert_eq!(machine.xregs()[7], (-128_i64) as u64);
        assert_eq!(machine.xregs()[5], 0x1000 - 584);
    }

    #[test]
    fn c310_sign_extension_applies_to_one_two_and_four_byte_loads() {
        let mut bus = TestBus::new(0x1000);
        bus.bytes[..8].copy_from_slice(&[0x78, 0x80, 0x34, 0x80, 0, 0, 0, 0x80]);
        for (dtype_bits, expected) in [
            (0_u32, 0x78_u64),
            (1_u32, 0xffff_ffff_ffff_8078_u64),
            (2_u32, 0xffff_ffff_8034_8078_u64),
            (3_u32, 0x8000_0000_8034_8078_u64),
        ] {
            let mut machine = ScalarMachine::new(Architecture::Dav3510, [0; 32], 0);
            machine.set_xreg(5, 0x1000).unwrap();
            let word = (0x1fce_5db8 & !(3 << 22)) | (dtype_bits << 22);
            let step = machine.execute_memory_word(0, word, &mut bus).unwrap();
            assert_eq!(step.data_value, expected);
            assert_eq!(machine.xregs()[7], expected);
        }
    }

    #[test]
    fn both_architectures_store_low_bytes_in_little_endian_order() {
        for (architecture, word) in [
            (Architecture::Dav2201, 0x04c6_6000),
            (Architecture::Dav3510, 0x03c6_6000),
        ] {
            let mut machine = ScalarMachine::new(architecture, [0; 32], 0);
            machine.set_xreg(6, 0x2000).unwrap();
            machine.set_xreg(3, 0x0123_4567_89ab_cdef).unwrap();
            let mut bus = TestBus::new(0x2000);
            let step = machine.execute_memory_word(0x108, word, &mut bus).unwrap();
            assert_eq!(step.operation, ScalarLoadStoreOperation::Store);
            assert_eq!(step.width_bytes, 8);
            assert_eq!(step.bytes, [0xef, 0xcd, 0xab, 0x89, 0x67, 0x45, 0x23, 0x01]);
            assert_eq!(bus.bytes[..8], step.bytes);
            assert_eq!(machine.xregs()[3], 0x0123_4567_89ab_cdef);
        }
    }

    #[test]
    fn c310_post_index_store_can_alias_its_data_register() {
        let mut machine = ScalarMachine::new(Architecture::Dav3510, [0; 32], 0);
        machine.set_xreg(5, 0x3000).unwrap();
        let mut bus = TestBus::new(0x3000);
        let aliased = (0x13ce_5db8 & !(0x1f << 17)) | (5 << 17);
        let step = machine
            .execute_memory_word(0x10c, aliased, &mut bus)
            .unwrap();
        assert_eq!(step.data_value, 0x3000);
        assert_eq!(step.effective_address, 0x3000);
        assert_eq!(bus.bytes[..8], 0x3000_u64.to_le_bytes());
        assert_eq!(machine.xregs()[5], 0x3000 - 584);
    }

    #[test]
    fn memory_backend_error_and_aliased_post_index_do_not_change_registers() {
        let mut machine = ScalarMachine::new(Architecture::Dav3510, [7; 32], 0x55);
        let before = machine.clone();
        let mut bus = TestBus::new(7);
        bus.fail = true;
        assert!(matches!(
            machine.execute_memory_word(0x100, 0x1fce_5db8, &mut bus),
            Err(ScalarMemoryExecutionError::Backend(_))
        ));
        assert_eq!(machine, before);
        assert_eq!(bus.accesses, 0);
        let aliased = (0x1fce_5db8 & !(0x1f << 17)) | (5 << 17);
        assert!(matches!(
            machine.execute_memory_word(0x100, aliased, &mut bus),
            Err(ScalarMemoryExecutionError::AliasedPostIndex { register: 5, .. })
        ));
        assert_eq!(machine, before);
        assert_eq!(bus.accesses, 0);
    }

    #[test]
    fn observed_greater_move_and_arithmetic_chain_matches_register_values() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut machine = ScalarMachine::new(architecture, [0; 32], 0);
            assert_eq!(
                machine.execute_word(0x112f5000, 0x073a7f80).unwrap().value,
                0x7f80
            );
            assert_eq!(
                machine.execute_word(0x112f5004, 0x077b0010).unwrap().value,
                0x107f80
            );
            machine.set_xreg(29, 0x1c7f80).unwrap();
            assert_eq!(
                machine.execute_word(0x112f5034, 0x083dd0a8).unwrap().value,
                0x1c8028
            );
            let step = machine.execute_word(0x112f503c, 0x0893e790).unwrap();
            assert_eq!(step.source_register, Some(30));
            assert_eq!(step.source_value, Some(0x1c8028));
            assert_eq!(step.value, 0x1c7898);
            assert_eq!(machine.xregs()[9], 0x1c7898);

            assert_eq!(
                machine.execute_word(0x112f5138, 0x070c_a000).unwrap().value,
                0xa000
            );
            assert_eq!(
                machine.execute_word(0x112f513c, 0x078d_ffff).unwrap().value,
                0xffff_0000_a000
            );
            assert_eq!(
                machine.execute_word(0x112f5140, 0x07cd_ffff).unwrap().value,
                0xffff_ffff_0000_a000
            );
        }
    }

    #[test]
    fn movk_preserves_three_other_halfwords_and_supports_all_lanes() {
        let mut machine = ScalarMachine::new(Architecture::Dav2201, [0; 32], 0x55);
        machine.set_xreg(6, 0x0123_4567_89ab_cdef).unwrap();
        for (word, expected) in [
            (0x070d_ffff, 0x0123_4567_89ab_ffff),
            (0x074d_0001, 0x0123_4567_0001_ffff),
            (0x078d_0002, 0x0123_0002_0001_ffff),
            (0x07cd_0003, 0x0003_0002_0001_ffff),
        ] {
            assert_eq!(machine.execute_word(0, word).unwrap().value, expected);
            assert_eq!(machine.spr2(), 0x55);
        }
    }

    #[test]
    fn overflow_updates_spr2_and_unsupported_words_do_not_mutate_state() {
        let mut machine = ScalarMachine::new(Architecture::Dav3510, [0; 32], 0x1234);
        machine.set_xreg(1, i64::MAX as u64).unwrap();
        let step = machine.execute_word(0x100, 0x0802_1001).unwrap();
        assert!(step.signed_overflow);
        assert_eq!(machine.xregs()[1], i64::MIN as u64);
        assert_eq!(machine.spr2(), 0x401234);
        let snapshot = machine.clone();
        assert!(matches!(
            machine.execute_word(0x104, 0x08c0_0000),
            Err(ScalarMachineError::UnsupportedWord { .. })
        ));
        assert_eq!(machine, snapshot);
        assert_eq!(
            machine.set_xreg(32, 0),
            Err(ScalarMachineError::RegisterOutOfRange(32))
        );
    }

    #[test]
    fn decoded_memory_words_do_not_mutate_the_register_only_machine() {
        for (architecture, words) in [
            (Architecture::Dav2201, [0x03ce_5db8, 0x04c6_6000]),
            (Architecture::Dav3510, [0x1cce_5db8, 0x03ce_5db8]),
        ] {
            let mut machine = ScalarMachine::new(architecture, [7; 32], 0x55);
            let snapshot = machine.clone();
            for word in words {
                assert!(matches!(
                    machine.execute_word(0x100, word),
                    Err(ScalarMachineError::UnsupportedWord { .. })
                ));
                assert_eq!(machine, snapshot);
            }
        }
    }

    #[test]
    fn zero_extend_masks_valid_widths_and_rejects_unmodeled_high_registers() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut machine = ScalarMachine::new(architecture, [0; 32], 0x55);
            machine.set_xreg(3, 0x1234_5678_9abc_def0).unwrap();
            for (word, value) in [
                (0x0208_3a00, 0xf0),
                (0x0248_3a00, 0xdef0),
                (0x0288_3a00, 0x9abc_def0),
            ] {
                let step = machine.execute_word(0x100, word).unwrap();
                assert_eq!(step.source_register, Some(3));
                assert_eq!(step.source_value, Some(0x1234_5678_9abc_def0));
                assert_eq!(step.value, value);
                assert_eq!(machine.xregs()[4], value);
                assert_eq!(machine.spr2(), 0x55);
            }
        }
        let mut machine = ScalarMachine::new(Architecture::Dav2201, [0; 32], 0);
        let before = machine.clone();
        assert!(matches!(
            machine.execute_word(0x104, 0x0208_3a60),
            Err(ScalarMachineError::UnsupportedWord { .. })
        ));
        assert_eq!(machine, before);
    }

    #[test]
    fn sign_extend_preserves_signed_values_at_eight_sixteen_and_thirty_two_bits() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut machine = ScalarMachine::new(architecture, [0; 32], 0x55);
            for (word, source, expected) in [
                (0x021c_f980, 0x1234_5678_0000_0080, (-128_i64) as u64),
                (0x025c_f980, 0x1234_5678_0000_8000, (-32768_i64) as u64),
                (0x029c_f980, 0x1234_5678_8000_0000, (-2147483648_i64) as u64),
                (0x021c_f980, 0xffff_ffff_ffff_007f, 127),
            ] {
                machine.set_xreg(15, source).unwrap();
                let step = machine.execute_word(0x1131_24fc, word).unwrap();
                assert_eq!(step.destination_register, 14);
                assert_eq!(step.source_register, Some(15));
                assert_eq!(step.source_value, Some(source));
                assert_eq!(step.value, expected);
                assert_eq!(machine.xregs()[14], expected);
                assert_eq!(machine.xregs()[15], source);
                assert_eq!(machine.spr2(), 0x55);
            }
            let before = machine.clone();
            assert!(matches!(
                machine.execute_word(0x1131_24fc, 0x02dc_f980),
                Err(ScalarMachineError::UnsupportedWord { .. })
            ));
            assert_eq!(machine, before);
        }
    }

    #[test]
    fn register_move_copies_all_bits_without_modifying_spr2() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut machine = ScalarMachine::new(architecture, [0; 32], 0x55);
            machine.set_xreg(30, 0xfedc_ba98_7654_3210).unwrap();
            let step = machine.execute_word(0x100, 0x0209_e800).unwrap();
            assert_eq!(step.source_register, Some(30));
            assert_eq!(step.source_value, Some(0xfedc_ba98_7654_3210));
            assert_eq!(step.value, 0xfedc_ba98_7654_3210);
            assert_eq!(machine.xregs()[4], 0xfedc_ba98_7654_3210);
            assert_eq!(machine.spr2(), 0x55);
        }
        let mut machine = ScalarMachine::new(Architecture::Dav2201, [0; 32], 0);
        let before = machine.clone();
        assert!(matches!(
            machine.execute_word(0x104, 0x0209_e860),
            Err(ScalarMachineError::UnsupportedWord { .. })
        ));
        assert_eq!(machine, before);
    }

    #[test]
    fn register_binary_forms_use_two_sources_and_update_only_verified_state() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut machine = ScalarMachine::new(architecture, [0; 32], 0x55);
            machine.set_xreg(29, 0x0010_7fa0).unwrap();
            machine.set_xreg(15, 0).unwrap();
            let add = machine.execute_word(0x126e_cb04, 0x003b_d781).unwrap();
            assert_eq!(add.value, 0x0010_7fa0);
            assert_eq!(add.second_source_register, Some(15));
            assert_eq!(add.second_source_value, Some(0));
            machine.set_xreg(8, 0x209).unwrap();
            machine.set_xreg(7, 1).unwrap();
            let mul = machine.execute_word(0x126e_d004, 0x000e_8383).unwrap();
            assert_eq!(mul.value, 0x209);
            assert_eq!(machine.xregs()[7], 0x209);
            machine.set_xreg(15, 0x18).unwrap();
            machine.set_xreg(16, 0x7fff).unwrap();
            assert_eq!(
                machine
                    .execute_word(0x126e_cb14, 0x00de_f80a)
                    .unwrap()
                    .value,
                0x18
            );
            machine.set_xreg(1, 0xc000_0000_0000_0000).unwrap();
            machine.set_xreg(0, 0x0808_0801_0101).unwrap();
            let or = machine.execute_word(0x126e_d29c, 0x00c0_100b).unwrap();
            assert_eq!(or.value, 0xc000_0808_0801_0101);
            assert_eq!(or.prior_destination_value, 0x0808_0801_0101);
            assert_eq!(or.second_source_value, Some(0x0808_0801_0101));
            assert_eq!(machine.spr2(), 0x55);
        }
    }

    #[test]
    fn register_add_and_multiply_overflow_update_spr2_and_reject_unmodeled_forms() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut machine = ScalarMachine::new(architecture, [0; 32], 0);
            machine.set_xreg(1, i64::MAX as u64).unwrap();
            machine.set_xreg(2, 1).unwrap();
            let add = machine.execute_word(0x100, 0x0002_1101).unwrap();
            assert_eq!(add.value, i64::MIN as u64);
            assert!(add.signed_overflow);
            assert_eq!(machine.spr2(), 0x400010);
            machine.set_xreg(2, u64::MAX).unwrap();
            let mul = machine.execute_word(0x104, 0x0002_1103).unwrap();
            assert_eq!(mul.value, i64::MIN as u64);
            assert!(mul.signed_overflow);
            assert_eq!(machine.spr2(), 0x410010);

            let before = machine.clone();
            assert!(matches!(
                machine.execute_word(0x108, 0x0082_1101),
                Err(ScalarMachineError::UnsupportedWord { .. })
            ));
            assert_eq!(machine, before);
        }
        let mut machine = ScalarMachine::new(Architecture::Dav2201, [0; 32], 0);
        let before = machine.clone();
        assert!(matches!(
            machine.execute_word(0x100, 0x003b_d7f1),
            Err(ScalarMachineError::UnsupportedWord { .. })
        ));
        assert_eq!(machine, before);
    }

    #[test]
    fn register_subtract_wraps_and_reports_signed_overflow() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut machine = ScalarMachine::new(architecture, [0; 32], 0);
            machine.set_xreg(1, 0).unwrap();
            machine.set_xreg(2, 1).unwrap();
            let step = machine.execute_word(0x100, 0x0002_1102).unwrap();
            assert_eq!(step.value, u64::MAX);
            assert_eq!(step.prior_destination_value, 0);
            assert!(!step.signed_overflow);
            assert_eq!(machine.spr2(), 0);

            machine.set_xreg(1, i64::MIN as u64).unwrap();
            let overflow = machine.execute_word(0x104, 0x0002_1102).unwrap();
            assert_eq!(overflow.value, i64::MAX as u64);
            assert!(overflow.signed_overflow);
            assert_eq!(machine.spr2(), 0x410010);
        }
    }

    #[test]
    fn multiply_add_uses_prior_destination_and_preserves_overflow_state() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut machine = ScalarMachine::new(architecture, [0; 32], 0x24);
            machine.set_xreg(29, 0x107f80).unwrap();
            machine.set_xreg(15, 0x18).unwrap();
            machine.set_xreg(17, 0x8000).unwrap();
            let step = machine.execute_word(0x112f5028, 0x003a_f884).unwrap();
            assert_eq!(step.prior_destination_value, 0x107f80);
            assert_eq!(step.source_register, Some(15));
            assert_eq!(step.second_source_register, Some(17));
            assert_eq!(step.value, 0x1c7f80);
            assert!(!step.signed_overflow);
            assert_eq!(step.spr2, 0x24);

            machine.set_xreg(29, 0).unwrap();
            machine.set_xreg(15, i64::MIN as u64).unwrap();
            machine.set_xreg(17, u64::MAX).unwrap();
            let product_overflow = machine.execute_word(0x100, 0x003a_f884).unwrap();
            assert_eq!(product_overflow.value, i64::MIN as u64);
            assert!(product_overflow.signed_overflow);
            assert_eq!(product_overflow.spr2, 0x400034);

            machine.set_xreg(29, i64::MAX as u64).unwrap();
            machine.set_xreg(15, 1).unwrap();
            machine.set_xreg(17, 1).unwrap();
            let add_overflow = machine.execute_word(0x104, 0x003a_f884).unwrap();
            assert_eq!(add_overflow.value, i64::MIN as u64);
            assert!(add_overflow.signed_overflow);
            assert_eq!(add_overflow.spr2, 0x410034);

            let before = machine.clone();
            assert!(matches!(
                machine.execute_word(0x108, 0x00ba_f884),
                Err(ScalarMachineError::UnsupportedWord { .. })
            ));
            assert_eq!(machine, before);
        }
    }

    #[test]
    fn negate_handles_distinct_registers_and_architecture_specific_overflow_bits() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut machine = ScalarMachine::new(architecture, [0; 32], 0x00f5_00ff);
            machine.set_xreg(2, 0x4120).unwrap();
            let step = machine.execute_word(0x100, 0x0202_2080).unwrap();
            assert_eq!(step.destination_register, 1);
            assert_eq!(step.source_register, Some(2));
            assert_eq!(step.value, 0xffff_ffff_ffff_bee0);
            assert_eq!(step.spr2, 0x00f5_00ff);

            machine.set_xreg(2, i64::MIN as u64).unwrap();
            let overflow = machine.execute_word(0x100, 0x0202_2080).unwrap();
            assert_eq!(overflow.value, i64::MIN as u64);
            assert!(overflow.signed_overflow);
            assert_eq!(
                overflow.spr2,
                match architecture {
                    Architecture::Dav2201 => 0x0040_00ff,
                    Architecture::Dav3510 => 0x00f5_000f,
                }
            );

            let before = machine.clone();
            assert!(matches!(
                machine.execute_word(0x104, 0x0242_2080),
                Err(ScalarMachineError::UnsupportedWord { .. })
            ));
            assert_eq!(machine, before);
        }
    }

    #[test]
    fn shift_left_uses_old_destination_and_masks_register_count() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut machine = ScalarMachine::new(architecture, [0; 32], 0x1234);
            machine.set_xreg(4, 0x38).unwrap();
            let immediate = machine.execute_word(0x100, 0x02c8_020f).unwrap();
            assert_eq!(immediate.value, 0x1c_0000);
            assert_eq!(immediate.source_register, Some(4));
            assert_eq!(immediate.source_value, Some(0x38));
            assert_eq!(immediate.second_source_register, None);

            machine.set_xreg(10, 0xffff_ffff).unwrap();
            machine.set_xreg(22, 0x60).unwrap();
            let register = machine.execute_word(0x104, 0x02d5_6240).unwrap();
            assert_eq!(register.value, 0xffff_ffff_0000_0000);
            assert_eq!(register.source_register, Some(10));
            assert_eq!(register.second_source_register, Some(22));
            assert_eq!(register.second_source_value, Some(0x60));
            assert_eq!(machine.spr2(), 0x1234);

            let before = machine.clone();
            assert!(matches!(
                machine.execute_word(0x108, 0x0288_020f),
                Err(ScalarMachineError::UnsupportedWord { .. })
            ));
            assert_eq!(machine, before);
        }
    }

    #[test]
    fn shift_right_supports_logical_and_arithmetic_modes() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut machine = ScalarMachine::new(architecture, [0; 32], 0x55);
            machine.set_xreg(9, 0x1c7898).unwrap();
            let logical = machine.execute_word(0x112f5044, 0x0252_028f).unwrap();
            assert_eq!(logical.value, 0x38);
            assert_eq!(logical.source_register, Some(9));
            assert_eq!(logical.second_source_register, None);

            machine.set_xreg(9, i64::MIN as u64).unwrap();
            machine.set_xreg(6, 0x41).unwrap();
            let arithmetic = machine.execute_word(0x104, 0x0212_62c0).unwrap();
            assert_eq!(arithmetic.value, 0xc000_0000_0000_0000);
            assert_eq!(arithmetic.second_source_register, Some(6));
            assert_eq!(arithmetic.second_source_value, Some(0x41));
            assert_eq!(machine.spr2(), 0x55);

            let before = machine.clone();
            assert!(matches!(
                machine.execute_word(0x108, 0x02d2_028f),
                Err(ScalarMachineError::UnsupportedWord { .. })
            ));
            assert_eq!(machine, before);
        }
    }

    #[test]
    fn observed_c220_mov_spr_xn_writes_full_width_ctrl_value() {
        let mut machine = ScalarMachine::new(Architecture::Dav2201, [0; 32], 0x55);
        machine.set_xreg(19, 0x0100_0000_0000_0000).unwrap();
        let first = machine.execute_spr_word(0x1131_2644, 0x0207_3900).unwrap();
        assert_eq!(first.destination_spr, 3);
        assert_eq!(first.source_register, 19);
        assert_eq!(first.source_value, 0x0100_0000_0000_0000);
        assert_eq!(first.prior_destination_value, None);
        assert_eq!(machine.spr_value(3), Some(0x0100_0000_0000_0000));

        machine.set_xreg(19, 0).unwrap();
        let second = machine.execute_spr_word(0x1131_2654, 0x0207_3900).unwrap();
        assert_eq!(second.prior_destination_value, Some(first.value));
        assert_eq!(machine.spr_value(3), Some(0));
        assert_eq!(machine.spr2(), 0x55);
    }

    #[test]
    fn observed_c310_mov_spr_xn_applies_register_masks() {
        let mut machine = ScalarMachine::new(Architecture::Dav3510, [0; 32], 0x55);
        machine.set_xreg(0, 0x1234_5678_9abc_def0).unwrap();
        for (word, destination, expected) in [
            (0x0206_0900, 3, 0x1234_5678_9abc_def0),
            (0x0216_0900, 11, 0),
            (0x02b4_0900, 90, 0xf0),
            (0x02d2_0900, 105, 0x1234_5678_9abc_def0),
            (0x02e0_0900, 112, 0x1234_5678_9abc_def0),
        ] {
            let step = machine.execute_spr_word(0x10d0_d140, word).unwrap();
            assert_eq!(step.destination_spr, destination);
            assert_eq!(step.value, expected);
            assert_eq!(machine.spr_value(destination), Some(expected));
        }
        machine.set_xreg(0, 0x1234_5678_9abc_def1).unwrap();
        let predicate = machine.execute_spr_word(0x10d0_d144, 0x0216_0900).unwrap();
        assert_eq!(predicate.source_value, 0x1234_5678_9abc_def1);
        assert_eq!(predicate.value, 1);
        assert_eq!(machine.spr_value(11), Some(1));
        assert_eq!(machine.spr2(), 0x55);
    }

    #[test]
    fn c220_spr_reads_feed_the_captured_scalar_prefix() {
        let mut machine = ScalarMachine::new(Architecture::Dav2201, [0; 32], 0x55);
        machine.set_spr_value(67, 0).unwrap();
        machine.set_spr_value(16, 0).unwrap();
        machine.set_spr_value(4, 0x1022_be00).unwrap();
        machine.execute_word(0x10d0_d000, 0x073a_7fa0).unwrap();
        machine.execute_word(0x10d0_d004, 0x077b_0010).unwrap();
        let read = machine
            .execute_spr_read_word(0x10d0_d008, 0x029e_3880)
            .unwrap();
        assert_eq!(read.source_spr, 67);
        assert_eq!(read.source, ScalarSprReadSource::RegisterSnapshot);
        assert_eq!(read.destination_register, 15);
        assert_eq!(read.value, 0);
        assert_eq!(machine.xregs()[15], 0);
        assert_eq!(
            machine
                .execute_word(0x10d0_d00c, 0x003b_d781)
                .unwrap()
                .value,
            0x107fa0
        );
        assert_eq!(
            machine
                .execute_spr_read_word(0x10d0_d010, 0x021f_0880)
                .unwrap()
                .source_spr,
            16
        );
        let parameter_base = machine
            .execute_spr_read_word(0x10d0_d028, 0x0200_4880)
            .unwrap();
        assert_eq!(parameter_base.value, 0x1022_be00);
        assert_eq!(machine.xregs()[0], 0x1022_be00);
        assert_eq!(machine.spr2(), 0x55);
    }

    #[test]
    fn c220_block_and_subblock_register_reads_use_independent_spr_values() {
        let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
        machine.set_spr_value(1, 7).unwrap();
        machine.set_spr_value(95, 1).unwrap();
        machine.set_spr_value(96, 0).unwrap();
        for (word, destination, source_spr, value) in [
            (0x028f_f880, 7, 95, 1),
            (0x020c_1880, 6, 1, 7),
            (0x02d0_0880, 8, 96, 0),
        ] {
            let step = machine.execute_spr_read_word(0x1131_212c, word).unwrap();
            assert_eq!(step.destination_register, destination);
            assert_eq!(step.source_spr, source_spr);
            assert_eq!(step.value, value);
            assert_eq!(machine.xregs()[usize::from(destination)], value);
        }
    }

    #[test]
    fn spr_read_special_sources_and_missing_values_are_explicit() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut machine = ScalarMachine::new(architecture, [0; 32], 0);
            let pc = 0x10d0_d000;
            let read = machine.execute_spr_read_word(pc, 0x0200_0880).unwrap();
            assert_eq!(read.source_spr, 0);
            assert_eq!(read.source, ScalarSprReadSource::ProgramCounter);
            assert_eq!(read.value, pc);
            assert_eq!(machine.xregs()[0], pc);

            let before = machine.clone();
            assert_eq!(
                machine.execute_spr_read_word(pc + 4, 0x0200_4880),
                Err(ScalarMachineError::SprValueUnavailable { pc: pc + 4, spr: 4 })
            );
            assert_eq!(machine, before);
            machine.set_spr_value(7, 0x1235).unwrap();
            let read = machine.execute_spr_read_word(pc + 8, 0x0200_7880).unwrap();
            assert_eq!(read.source_value, 0x1235);
            assert_eq!(
                read.value,
                if architecture == Architecture::Dav2201 {
                    0x1234
                } else {
                    0x1235
                }
            );
            assert_eq!(machine.spr2(), 0);
        }

        let mut c310 = ScalarMachine::new(Architecture::Dav3510, [0; 32], 0);
        let pc = 0x100;
        assert_eq!(
            c310.execute_spr_read_word(pc, 0x0280_8880),
            Err(ScalarMachineError::ModelTimeUnavailable { pc })
        );
        c310.set_model_time(0x5678);
        let read = c310.execute_spr_read_word(pc, 0x0280_8880).unwrap();
        assert_eq!(read.source_spr, 72);
        assert_eq!(read.source, ScalarSprReadSource::ModelTime);
        assert_eq!(read.value, 0x5678);
        assert_eq!(c310.xregs()[0], 0x5678);
        assert_eq!(c310.model_time(), Some(0x5678));
        assert_eq!(
            c310.set_spr_value(243, 1),
            Err(ScalarMachineError::SprRegisterOutOfRange(243))
        );
        let mut c220 = ScalarMachine::new(Architecture::Dav2201, [0; 32], 0);
        assert_eq!(
            c220.set_spr_value(187, 1),
            Err(ScalarMachineError::SprRegisterOutOfRange(187))
        );
    }

    #[test]
    fn compare_jump_writes_spr11_for_following_scalar_read() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut machine = ScalarMachine::new(architecture, [0; 32], 0);
            machine.set_xreg(8, 1).unwrap();
            let not_taken = machine
                .execute_jump_compare_word(0x1000, 0x4884_1521)
                .unwrap();
            assert!(!not_taken.branch_taken);
            assert_eq!(machine.spr_value(11), Some(0));

            machine.set_xreg(8, 2).unwrap();
            let taken = machine
                .execute_jump_compare_word(0x1000, 0x4884_1521)
                .unwrap();
            assert!(taken.branch_taken);
            assert_eq!(machine.spr_value(11), Some(1));
            let read = machine.execute_spr_read_word(0x1004, 0x0200_b880).unwrap();
            assert_eq!(read.source_spr, 11);
            assert_eq!(read.value, 1);
            assert_eq!(machine.xregs()[0], 1);

            let before = machine.clone();
            assert_eq!(
                machine.execute_jump_compare_word(0x1008, 0x4e84_1521),
                Err(ScalarMachineError::Compare(
                    JumpCompareError::UnsupportedDtype(3)
                ))
            );
            assert_eq!(machine, before);
        }
    }

    #[test]
    fn flow_words_chain_compare_flag_into_conditional_target() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut machine = ScalarMachine::new(architecture, [0; 32], 0);
            let before = machine.clone();
            assert_eq!(
                machine.execute_flow_word(0x1004, 0x4020_0002),
                Err(ScalarMachineError::SprValueUnavailable {
                    pc: 0x1004,
                    spr: 11
                })
            );
            assert_eq!(machine, before);

            machine.set_xreg(8, 1).unwrap();
            let compare = machine.execute_flow_word(0x1000, 0x4884_1521).unwrap();
            assert!(matches!(compare, ScalarFlowStep::Compare(_)));
            assert_eq!(machine.spr_value(11), Some(0));
            let fallthrough = machine.execute_flow_word(0x1004, 0x4020_0002).unwrap();
            assert_eq!(fallthrough.target_pc(), 0x1008);
            assert!(matches!(fallthrough, ScalarFlowStep::Conditional(_)));

            machine.set_xreg(8, 2).unwrap();
            machine.execute_flow_word(0x1000, 0x4884_1521).unwrap();
            assert_eq!(machine.spr_value(11), Some(1));
            let mut bus = TestBus::new(0);
            let ScalarInstructionStep::Flow(taken) = machine
                .execute_instruction(0x1004, 0x4020_0002, &mut bus)
                .unwrap()
            else {
                panic!("expected flow step")
            };
            assert_eq!(taken.target_pc(), 0x100c);
            assert_eq!(bus.accesses, 0);
            let jump = machine.execute_flow_word(0x100c, 0x4000_0004).unwrap();
            assert!(matches!(jump, ScalarFlowStep::Jump(_)));
            assert_eq!(jump.target_pc(), 0x101c);

            let before = machine.clone();
            assert_eq!(
                machine.execute_flow_word(0x101c, 0x6000_0000),
                Err(ScalarMachineError::UnsupportedWord {
                    pc: 0x101c,
                    word: 0x6000_0000
                })
            );
            assert_eq!(machine, before);
        }
    }

    #[test]
    fn dc_preload_exposes_request_without_changing_scalar_value_state() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut machine = ScalarMachine::new(architecture, [0; 32], 0);
            machine.set_xreg(0, 0x1234).unwrap();
            let before = machine.clone();
            let mut bus = TestBus::new(0);
            let step = machine
                .execute_instruction(0x1008, 0x08c0_0000, &mut bus)
                .unwrap();
            let ScalarInstructionStep::CacheHint(hint) = step else {
                panic!("expected cache hint")
            };
            assert_eq!(hint.pc, 0x1008);
            assert_eq!(hint.source_register, 0);
            assert_eq!(hint.source_value, 0x1234);
            assert_eq!(hint.encoded_immediate, 0);
            assert_eq!(hint.effective_address, 0x1234);
            assert_eq!(hint.requested_bytes, 64);
            assert_eq!(bus.accesses, 0);
            assert_eq!(machine, before);

            machine.set_xreg(0, u64::MAX).unwrap();
            let wrap = machine
                .execute_cache_hint_word(0x100c, 0x08c0_0001)
                .unwrap();
            assert_eq!(wrap.encoded_immediate, 1);
            assert_eq!(wrap.effective_address, 0);
        }
    }

    #[test]
    fn insert_replaces_only_the_selected_destination_bit_field() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut machine = ScalarMachine::new(architecture, [0; 32], 0x55);
            machine.set_xreg(1, 0x1234_5678_9abc_def0).unwrap();
            machine.set_xreg(7, 5).unwrap();
            let step = machine.execute_word(0x100, 0x0202_7cc2).unwrap();
            let mask = 0b111_u64 << 6;
            let expected = (0x1234_5678_9abc_def0 & !mask) | (5 << 6);
            assert_eq!(step.destination_register, 1);
            assert_eq!(step.prior_destination_value, 0x1234_5678_9abc_def0);
            assert_eq!(step.source_register, Some(7));
            assert_eq!(step.source_value, Some(5));
            assert_eq!(step.value, expected);
            assert_eq!(step.spr2, 0x55);
            assert_eq!(machine.xregs()[1], expected);

            let overflow_word =
                (0x0202_7cc2 & !((3 << 22) | (0xf << 5) | 0x1f)) | (3 << 22) | (0xf << 5) | 0x1f;
            assert!(matches!(
                AicDecoderHint::from_word(architecture, overflow_word),
                Some(AicDecoderHint::ScalarKey2Insert {
                    least_significant_bit: 63,
                    width_bits: 32,
                    ..
                })
            ));
            let before = machine.clone();
            assert_eq!(
                machine.execute_word(0x104, overflow_word),
                Err(ScalarMachineError::UnsupportedWord {
                    pc: 0x104,
                    word: overflow_word,
                })
            );
            assert_eq!(machine, before);
        }
    }

    #[test]
    fn immediate_insert_and_register_indexed_bitset_preserve_unselected_bits() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut machine = ScalarMachine::new(architecture, [0; 32], 0x55);
            machine.set_xreg(1, 0xffff_ffff_ffff_ffff).unwrap();
            let insert = machine.execute_word(0x100, 0x0243_8b00).unwrap();
            assert_eq!(insert.value, !(1_u64 << 56));
            assert_eq!(insert.source_register, Some(1));
            assert_eq!(insert.source_value, Some(u64::MAX));
            assert_eq!(machine.xregs()[1], insert.value);

            machine.set_xreg(0, u64::MAX).unwrap();
            machine.set_xreg(1, 70).unwrap();
            let clear = machine.execute_word(0x104, 0x02c0_1400).unwrap();
            assert_eq!(clear.value, !(1_u64 << 6));
            assert_eq!(clear.source_register, Some(1));
            assert_eq!(clear.source_value, Some(70));
            let set = machine.execute_word(0x108, 0x02c0_1440).unwrap();
            assert_eq!(set.value, u64::MAX);
            assert_eq!(machine.spr2(), 0x55);

            let extended = 0x0243_8b00 | 0x80_0000 | 0xff;
            machine.set_xreg(1, 0x1234_5678_9abc_def0).unwrap();
            let step = machine.execute_word(0x10c, extended).unwrap();
            assert_eq!(step.value, 0xff34_5678_9abc_def0);

            let overflow_word = 0x0243_8b00 | 0x80_0000 | 0x7000;
            let before = machine.clone();
            assert_eq!(
                machine.execute_word(0x110, overflow_word),
                Err(ScalarMachineError::UnsupportedWord {
                    pc: 0x110,
                    word: overflow_word,
                })
            );
            assert_eq!(machine, before);
        }
    }

    #[test]
    fn unverified_spr_destinations_and_out_of_range_sources_fail_closed() {
        let mut c310 = ScalarMachine::new(Architecture::Dav3510, [0; 32], 0x55);
        let before = c310.clone();
        assert!(matches!(
            c310.execute_spr_word(0x100, 0x0230_0901),
            Err(ScalarMachineError::UnsupportedWord { .. })
        ));
        assert_eq!(c310, before);
        assert!(matches!(
            c310.execute_word(0x100, 0x0206_0900),
            Err(ScalarMachineError::UnsupportedWord { .. })
        ));
        assert_eq!(c310, before);

        let mut c220 = ScalarMachine::new(Architecture::Dav2201, [0; 32], 0x55);
        let before = c220.clone();
        assert!(matches!(
            c220.execute_spr_word(0x100, 0x0206_1920),
            Err(ScalarMachineError::UnsupportedWord { .. })
        ));
        assert_eq!(c220, before);
    }

    #[test]
    fn scalar_immediate_store_writes_only_the_selected_bytes() {
        const WORD: u32 = 0x0f35_e580;
        const EFFECTIVE_ADDRESS: u64 = 0x1c7df4;
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            for (value_bits, expected_byte) in [(0, 0), (1, 1), (2, 0xff)] {
                for (dtype, width) in [(0, 1_usize), (1, 2), (2, 4), (3, 8)] {
                    let word = (WORD & !(3 << 22)) | (dtype << 22) | value_bits;
                    let mut machine = ScalarMachine::new(architecture, [0; 32], 0x55);
                    machine.set_xreg(30, 0x1c80c8).unwrap();
                    let before = machine.clone();
                    let mut bus = TestBus::new(EFFECTIVE_ADDRESS - 8);
                    bus.bytes.fill(0xa5);

                    let ScalarInstructionStep::ImmediateStore(step) = machine
                        .execute_instruction(0x113120bc, word, &mut bus)
                        .unwrap()
                    else {
                        panic!("expected immediate store")
                    };
                    assert_eq!(step.effective_address, EFFECTIVE_ADDRESS);
                    assert_eq!(usize::from(step.width_bytes), width);
                    assert_eq!(step.base_register, 30);
                    assert_eq!(step.prior_base_value, 0x1c80c8);
                    assert_eq!(bus.accesses, 1);
                    assert_eq!(&bus.bytes[..8], &[0xa5; 8]);
                    assert_eq!(&bus.bytes[8 + width..], &[0xa5; 32][8 + width..]);
                    for (index, byte) in bus.bytes[8..8 + width].iter().enumerate() {
                        let expected = if value_bits == 1 && index != 0 {
                            0
                        } else {
                            expected_byte
                        };
                        assert_eq!(*byte, expected);
                        assert_eq!(step.bytes[index], expected);
                    }
                    assert_eq!(machine, before);
                }
            }
        }
    }

    #[test]
    fn scalar_immediate_store_rejects_unsupported_forms_and_backend_errors_atomically() {
        const WORD: u32 = 0x0f35_e580;
        let mut machine = ScalarMachine::new(Architecture::Dav2201, [0; 32], 0x55);
        machine.set_xreg(30, 0x1c80c8).unwrap();
        let before = machine.clone();
        let mut bus = TestBus::new(0x1c7df4);
        bus.bytes.fill(0xa5);

        for word in [WORD | 3, WORD | 4] {
            assert!(matches!(
                machine.execute_immediate_store_word(0x113120bc, word, &mut bus),
                Err(ScalarMemoryExecutionError::UnsupportedWord { .. })
            ));
            assert_eq!(machine, before);
            assert_eq!(bus.bytes, [0xa5; 32]);
            assert_eq!(bus.accesses, 0);
        }

        bus.fail = true;
        assert!(matches!(
            machine.execute_instruction(0x113120bc, WORD, &mut bus),
            Err(ScalarInstructionError::Memory(
                ScalarMemoryExecutionError::Backend(_)
            ))
        ));
        assert_eq!(machine, before);
        assert_eq!(bus.bytes, [0xa5; 32]);
        assert_eq!(bus.accesses, 0);

        let mut c310 = ScalarMachine::new(Architecture::Dav3510, [0; 32], 0x55);
        c310.set_xreg(30, 0x208028).unwrap();
        let before = c310.clone();
        let mut c310_bus = TestBus::new(0x207d70);
        c310_bus.bytes.fill(0xa5);
        let step = c310
            .execute_immediate_store_word(0x10d0d158, 0x0f35_e900, &mut c310_bus)
            .unwrap();
        assert_eq!(step.effective_address, 0x207d70);
        assert_eq!(step.width_bytes, 1);
        assert_eq!(c310_bus.bytes[0], 0);
        assert_eq!(&c310_bus.bytes[1..], &[0xa5; 32][1..]);
        assert_eq!(c310, before);
    }

    #[test]
    fn pem_initial_spr_values_keep_config_dependent_slots_unknown() {
        let mut c220 = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
        assert_eq!(c220.spr_value(1), Some(0));
        assert_eq!(c220.spr_value(95), Some(0));
        assert_eq!(c220.spr_value(96), Some(0));
        assert_eq!(c220.spr_value(55), Some(15360));
        assert_eq!(c220.spr_value(58), Some(0x10000));
        assert_eq!(c220.spr_value(100), Some(u64::MAX));
        assert_eq!(c220.spr_value(111), Some(0));
        assert_eq!(c220.spr_value(174), Some(63));
        assert_eq!(c220.spr_value(181), Some(u64::MAX));
        assert_eq!(c220.spr_value(107), None);
        assert_eq!(c220.spr_value(108), None);
        assert_eq!(c220.spr_value(187), None);
        assert_eq!(
            c220.execute_spr_read_word(0x1131212c, 0x028f_f880)
                .unwrap()
                .value,
            0
        );

        let mut c310 = ScalarMachine::from_pem_initial_state(Architecture::Dav3510);
        assert_eq!(c310.spr_value(1), Some(0));
        assert_eq!(c310.spr_value(3), Some(0x1000000000000008));
        assert_eq!(c310.spr_value(20), Some(784));
        assert_eq!(c310.spr_value(105), Some(0x200001));
        assert_eq!(c310.spr_value(164), Some(0));
        assert_eq!(c310.spr_value(227), Some(63));
        assert_eq!(c310.spr_value(234), Some(u64::MAX));
        assert_eq!(c310.spr_value(67), None);
        assert_eq!(c310.spr_value(68), None);
        assert_eq!(c310.spr_value(73), None);
        assert_eq!(
            c310.execute_spr_read_word(0x10d0d110, 0x0212_1880)
                .unwrap()
                .value,
            0
        );
    }

    #[test]
    fn flow_nop_preserves_registers_and_advances_pc_on_both_architectures() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut machine = ScalarMachine::from_pem_initial_state(architecture);
            machine.set_xreg(8, 0x1234).unwrap();
            let before = machine.clone();
            let mut bus = TestBus::new(0);
            let ScalarInstructionStep::Flow(ScalarFlowStep::Nop(step)) = machine
                .execute_instruction(0x11312160, 0x4140_0000, &mut bus)
                .unwrap()
            else {
                panic!("expected flow NOP")
            };
            assert_eq!(step.target_pc, 0x11312164);
            assert_eq!(bus.accesses, 0);
            assert_eq!(machine, before);
        }
    }

    #[test]
    fn c220_scalar_words_supply_vector_fill_and_mask_values() {
        for (fill_halfword, mask_halfword, expected_fill, expected_mask) in [
            (0x074d_c2f6, 0x074f_5555, 0xc2f6_0000, 0x5555_5555),
            (0x074d_c2f7, 0x074f_5554, 0xc2f7_0000, 0x5554_5555),
        ] {
            let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
            let mut bus = TestBus::new(0);
            for (pc, word) in [
                (0x1131_2018, 0x0700_0001),
                (0x1131_2024, 0x0200_0080),
                (0x1131_202c, 0x8040_0000),
                (0x1131_2030, 0x8040_0080),
                (0x1131_2060, 0x0700_0000),
                (0x1131_2314, 0x070c_0000),
                (0x1131_2318, 0x070e_5555),
                (0x1131_2330, fill_halfword),
                (0x1131_2334, mask_halfword),
                (0x1131_2340, 0x8040_0080),
                (0x1131_265c, 0x8040_001c),
            ] {
                machine.execute_instruction(pc, word, &mut bus).unwrap();
            }
            assert_eq!(machine.xregs()[6], expected_fill);
            assert_eq!(machine.xregs()[7], expected_mask);
            assert_eq!(machine.spr_value(100), Some(expected_mask));
            assert_eq!(machine.spr_value(101), Some(0));
            assert_eq!(bus.accesses, 0);
        }
    }

    #[test]
    fn c220_scalar_words_select_count_mask_mode() {
        for (count_word, expected_count, expected_mask) in [
            (0x070a_0020, 32, 0xffff_ffff),
            (0x070a_001f, 31, 0x7fff_ffff),
        ] {
            let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
            machine.set_xreg(17, 0x100_0000).unwrap();
            let mut bus = TestBus::new(0);
            for (pc, word) in [
                (0x1131_2060, 0x0700_0000),
                (0x1131_2264, 0x0704_0001),
                (0x1131_22c0, 0x0204_2080),
                (0x1131_22cc, 0x8040_0088),
                (0x1131_22d4, 0x8040_0008),
                (0x1131_2314, count_word),
                (0x1131_2604, 0x8040_0080),
                (0x1131_2618, 0x0222_3880),
                (0x1131_261c, 0x0263_8b01),
                (0x1131_2620, 0x0207_1900),
                (0x1131_2624, 0x8040_0014),
            ] {
                machine.execute_instruction(pc, word, &mut bus).unwrap();
            }
            assert_eq!(machine.spr_value(3), Some(1 << 56));
            assert_eq!(machine.spr_value(100), Some(expected_count));
            assert_eq!(machine.spr_value(101), Some(0));
            assert_eq!(
                crate::vec_c220::decode_captured_c220_fp32_mask(
                    machine.spr_value(3).unwrap(),
                    machine.spr_value(100).unwrap(),
                    machine.spr_value(101).unwrap(),
                )
                .unwrap(),
                [expected_mask, 0, 0, 0]
            );
            assert_eq!(bus.accesses, 0);
        }
    }

    #[test]
    fn spr_two_reads_follow_the_live_overflow_register() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut machine = ScalarMachine::new(architecture, [0; 32], 0x1234);
            assert_eq!(machine.spr_value(2), Some(0x1234));
            let first = machine.execute_spr_read_word(0x100, 0x0200_2880).unwrap();
            assert_eq!(first.source_spr, 2);
            assert_eq!(first.value, 0x1234);

            machine.set_xreg(1, i64::MAX as u64).unwrap();
            assert!(
                machine
                    .execute_word(0x104, 0x0802_1001)
                    .unwrap()
                    .signed_overflow
            );
            assert_eq!(machine.spr_value(2), Some(0x411234));
            let after_overflow = machine.execute_spr_read_word(0x108, 0x0200_2880).unwrap();
            assert_eq!(after_overflow.value, 0x411234);

            machine.set_spr_value(2, 0x55).unwrap();
            assert_eq!(machine.spr2(), 0x55);
            assert_eq!(machine.spr_value(2), Some(0x55));
            assert_eq!(
                machine
                    .execute_spr_read_word(0x10c, 0x0200_2880)
                    .unwrap()
                    .value,
                0x55
            );
        }
    }

    #[test]
    fn c310_observed_movemask_updates_the_selected_live_spr() {
        let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav3510);
        machine.set_xreg(0, 0).unwrap();
        machine.set_xreg(13, 0x5555_5555).unwrap();
        let mut bus = TestBus::new(0);

        let ScalarInstructionStep::C310Movemask(first) = machine
            .execute_instruction(0x10d0d130, 0x15c0_0033, &mut bus)
            .unwrap()
        else {
            panic!("expected MOVEMASK step")
        };
        assert_eq!(first.hint.vendor_isa_name, 180);
        assert_eq!(first.hint.source_x_register, 0);
        assert_eq!(first.hint.destination_spr, 153);
        assert_eq!(first.prior_value, u64::MAX);
        assert_eq!(first.value, 0);
        assert_eq!(machine.spr_value(152), Some(u64::MAX));
        assert_eq!(machine.spr_value(153), Some(0));

        let ScalarInstructionStep::C310Movemask(second) = machine
            .execute_instruction(0x10d0d55c, 0x15cd_0013, &mut bus)
            .unwrap()
        else {
            panic!("expected MOVEMASK step")
        };
        assert_eq!(second.hint.source_x_register, 13);
        assert_eq!(second.hint.destination_spr, 152);
        assert_eq!(second.prior_value, u64::MAX);
        assert_eq!(second.value, 0x5555_5555);
        assert_eq!(machine.spr_value(152), Some(0x5555_5555));
        assert_eq!(machine.spr_value(153), Some(0));
        assert_eq!(bus.accesses, 0);

        let before = machine.clone();
        assert!(matches!(
            machine.execute_c310_movemask_word(0x10d0d560, 0x15ce_0013),
            Err(ScalarMachineError::UnsupportedWord { .. })
        ));
        assert_eq!(machine, before);
    }

    #[test]
    fn c310_movemask_requires_a_known_prior_mask_and_the_right_architecture() {
        let mut c310 = ScalarMachine::new(Architecture::Dav3510, [0; 32], 0);
        let before = c310.clone();
        assert_eq!(
            c310.execute_c310_movemask_word(0x10d0d130, 0x15c0_0033),
            Err(ScalarMachineError::SprValueUnavailable {
                pc: 0x10d0d130,
                spr: 153,
            })
        );
        assert_eq!(c310, before);

        let mut c220 = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
        let before = c220.clone();
        assert!(matches!(
            c220.execute_c310_movemask_word(0x10d0d130, 0x15c0_0033),
            Err(ScalarMachineError::UnsupportedWord { .. })
        ));
        assert_eq!(c220, before);
    }

    #[test]
    fn c310_integer_compare_updates_only_the_condition_spr() {
        for (condition, expected) in [(0, 0), (1, 1), (2, 1), (3, 0), (4, 0), (5, 1)] {
            let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav3510);
            machine.set_xreg(0, 0).unwrap();
            machine.set_xreg(2, 1).unwrap();
            let before_xregs = *machine.xregs();
            let word = 0x0000_010e | (u32::from(condition) << 4);
            let step = machine.execute_compare_word(0x10d0d210, word).unwrap();
            assert_eq!(step.dtype_field, 0);
            assert_eq!(step.condition_field, condition);
            assert_eq!(step.first_source_register, 0);
            assert_eq!(step.first_source_value, 0);
            assert_eq!(step.second_source_register, 2);
            assert_eq!(step.second_source_value, 1);
            assert_eq!(step.prior_spr11, Some(0));
            assert_eq!(step.spr11, expected);
            assert_eq!(machine.spr_value(11), Some(expected));
            assert_eq!(*machine.xregs(), before_xregs);
            assert_eq!(machine.spr2(), 0);
        }

        let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav3510);
        machine.set_xreg(0, u64::MAX).unwrap();
        machine.set_xreg(2, 1).unwrap();
        let signed = machine.execute_compare_word(0x100, 0x0000_012e).unwrap();
        assert_eq!(signed.dtype_field, 0);
        assert_eq!(signed.condition_field, 2);
        assert_eq!(signed.spr11, 1);
        let unsigned = machine.execute_compare_word(0x104, 0x0040_012e).unwrap();
        assert_eq!(unsigned.dtype_field, 1);
        assert_eq!(unsigned.prior_spr11, Some(1));
        assert_eq!(unsigned.spr11, 0);

        let before = machine.clone();
        for word in [0x0080_011e, 0x0000_016e] {
            assert!(matches!(
                machine.execute_compare_word(0x108, word),
                Err(ScalarMachineError::UnsupportedWord { .. })
            ));
            assert_eq!(machine, before);
        }
    }

    #[test]
    fn c220_integer_compare_uses_the_same_signedness_mapping() {
        let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
        machine.set_xreg(2, 0x400).unwrap();
        machine.set_xreg(4, 0x100).unwrap();
        let mut bus = TestBus::new(0);
        let ScalarInstructionStep::Compare(step) = machine
            .execute_instruction(0x1131_22d0, 0x0040_222e, &mut bus)
            .unwrap()
        else {
            panic!("expected compare step")
        };
        assert_eq!(step.dtype_field, 1);
        assert_eq!(step.condition_field, 2);
        assert_eq!(step.first_source_register, 2);
        assert_eq!(step.first_source_value, 0x400);
        assert_eq!(step.second_source_register, 4);
        assert_eq!(step.second_source_value, 0x100);
        assert_eq!(step.spr11, 0);
        assert_eq!(machine.spr_value(11), Some(0));
        assert_eq!(bus.accesses, 0);

        machine.set_xreg(0, u64::MAX).unwrap();
        machine.set_xreg(7, 1).unwrap();
        let signed = machine
            .execute_compare_register_word(0x100, 0x0000_03af)
            .unwrap();
        assert_eq!(signed.value, 1);
        let unsigned = machine
            .execute_compare_register_word(0x104, 0x0040_03af)
            .unwrap();
        assert_eq!(unsigned.value, 0);
    }

    #[test]
    fn c310_integer_compare_register_writes_result_after_reading_both_sources() {
        let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav3510);
        machine.set_xreg(0, 0).unwrap();
        machine.set_xreg(7, 0x4000_0000_0000_0000).unwrap();
        let mut bus = TestBus::new(0);
        let ScalarInstructionStep::CompareRegister(step) = machine
            .execute_instruction(0x10d0d24c, 0x0000_039f, &mut bus)
            .unwrap()
        else {
            panic!("expected compare-register step")
        };
        assert_eq!(step.dtype_field, 0);
        assert_eq!(step.condition_field, 1);
        assert_eq!(step.destination_register, 0);
        assert_eq!(step.prior_destination_value, 0);
        assert_eq!(step.first_source_register, 0);
        assert_eq!(step.first_source_value, 0);
        assert_eq!(step.second_source_register, 7);
        assert_eq!(step.second_source_value, 0x4000_0000_0000_0000);
        assert_eq!(step.value, 1);
        assert_eq!(machine.xregs()[0], 1);
        assert_eq!(machine.spr_value(11), Some(0));
        assert_eq!(bus.accesses, 0);

        machine.set_xreg(0, u64::MAX).unwrap();
        machine.set_xreg(7, 1).unwrap();
        let signed = machine
            .execute_compare_register_word(0x100, 0x0000_03af)
            .unwrap();
        assert_eq!(signed.value, 1);
        let unsigned = machine
            .execute_compare_register_word(0x104, 0x0040_03af)
            .unwrap();
        assert_eq!(unsigned.value, 0);

        let before = machine.clone();
        for word in [0x0080_039f, 0x0000_03ef] {
            assert!(matches!(
                machine.execute_compare_register_word(0x108, word),
                Err(ScalarMachineError::UnsupportedWord { .. })
            ));
            assert_eq!(machine, before);
        }
    }

    #[test]
    fn c310_find_first_scans_from_low_bit_and_reports_no_match() {
        for (source, find_set, expected) in [
            (0_u64, false, 0_u64),
            (0, true, u64::MAX),
            (u64::MAX, false, u64::MAX),
            (u64::MAX, true, 0),
            (0b1110, false, 0),
            (0b1110, true, 1),
            (0b1111, false, 4),
        ] {
            let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav3510);
            machine.set_xreg(0, source).unwrap();
            machine.set_xreg(11, 99).unwrap();
            let word = 0x02d6_0380 | if find_set { 0x40 } else { 0 };
            let step = machine.execute_word(0x10d0d3dc, word).unwrap();
            assert_eq!(step.destination_register, 11);
            assert_eq!(step.prior_destination_value, 99);
            assert_eq!(step.source_register, Some(0));
            assert_eq!(step.source_value, Some(source));
            assert_eq!(step.value, expected);
            assert_eq!(machine.xregs()[11], expected);
            assert_eq!(machine.xregs()[0], source);
            assert_eq!(machine.spr2(), 0);
        }

        let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav3510);
        machine.set_xreg(11, 0b1110).unwrap();
        let word = 0x02d6_b3c0;
        let step = machine.execute_word(0x10d0d3dc, word).unwrap();
        assert_eq!(step.source_register, Some(11));
        assert_eq!(step.source_value, Some(0b1110));
        assert_eq!(step.value, 1);
        assert_eq!(machine.xregs()[11], 1);
    }

    #[test]
    fn c220_captured_find_first_zero_matches_scalar_register_state() {
        for (word, source_register, destination_register) in [
            (0x02de_d380, 13, 15),
            (0x02de_b380, 11, 15),
            (0x02da_a380, 10, 13),
            (0x02da_b380, 11, 13),
            (0x02dc_d380, 13, 14),
            (0x02e0_f380, 15, 16),
        ] {
            for (source, expected) in [(0_u64, 0), (0b111, 3), (u64::MAX, u64::MAX)] {
                let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
                machine.set_xreg(source_register, source).unwrap();
                machine.set_xreg(destination_register, 99).unwrap();
                let step = machine.execute_word(0x1131_24dc, word).unwrap();
                assert_eq!(step.destination_register, destination_register);
                assert_eq!(step.prior_destination_value, 99);
                assert_eq!(step.source_register, Some(source_register));
                assert_eq!(step.source_value, Some(source));
                assert_eq!(step.value, expected);
                assert_eq!(machine.xregs()[usize::from(destination_register)], expected);
                assert_eq!(machine.xregs()[usize::from(source_register)], source);
            }
        }
    }

    #[test]
    fn compare_immediate_sign_extends_and_updates_condition_on_both_architectures() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut machine = ScalarMachine::from_pem_initial_state(architecture);
            machine.set_xreg(9, 1).unwrap();
            let mut bus = TestBus::new(0);
            let ScalarInstructionStep::CompareImmediate(first) = machine
                .execute_instruction(0x1131_2184, 0x0a00_9000, &mut bus)
                .unwrap()
            else {
                panic!("expected immediate compare")
            };
            assert_eq!(first.vendor_isa_name, 50);
            assert_eq!(first.source_register, 9);
            assert_eq!(first.source_value, 1);
            assert_eq!(first.signed_immediate, 0);
            assert_eq!(first.prior_spr11, Some(0));
            assert_eq!(first.spr11, 0);
            assert_eq!(machine.spr_value(11), Some(0));
            assert_eq!(bus.accesses, 0);

            machine.set_xreg(9, u64::MAX).unwrap();
            let equal_negative = machine
                .execute_compare_immediate_word(0x100, 0x0a00_9fff)
                .unwrap();
            assert_eq!(equal_negative.encoded_immediate, 0xfff);
            assert_eq!(equal_negative.signed_immediate, -1);
            assert_eq!(equal_negative.spr11, 1);

            machine.set_xreg(9, (-2_i64) as u64).unwrap();
            let less_negative = machine
                .execute_compare_immediate_word(0x104, 0x0a80_9fff)
                .unwrap();
            assert_eq!(less_negative.condition_field, 2);
            assert_eq!(less_negative.prior_spr11, Some(1));
            assert_eq!(less_negative.spr11, 1);

            let before = machine.clone();
            assert!(matches!(
                machine.execute_compare_immediate_word(0x108, 0x0b80_9fff),
                Err(ScalarMachineError::UnsupportedWord { .. })
            ));
            assert_eq!(machine, before);
        }
    }

    #[test]
    fn scalar_select_uses_condition_flag_and_reads_aliased_source_first() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut machine = ScalarMachine::from_pem_initial_state(architecture);
            machine.set_xreg(10, 11).unwrap();
            machine.set_xreg(5, 22).unwrap();
            let mut bus = TestBus::new(0);

            let ScalarInstructionStep::Select(false_step) = machine
                .execute_instruction(0x1131_218c, 0x00d4_a289, &mut bus)
                .unwrap()
            else {
                panic!("expected select step")
            };
            assert_eq!(false_step.vendor_isa_name, 8);
            assert_eq!(false_step.condition_flag, 0);
            assert_eq!(false_step.destination_register, 10);
            assert_eq!(false_step.prior_destination_value, 11);
            assert_eq!(false_step.first_source_register, 10);
            assert_eq!(false_step.first_source_value, 11);
            assert_eq!(false_step.second_source_register, 5);
            assert_eq!(false_step.second_source_value, 22);
            assert_eq!(false_step.value, 22);
            assert_eq!(machine.xregs()[10], 22);

            machine.set_xreg(10, 11).unwrap();
            machine.set_spr_value(11, 2).unwrap();
            let true_step = machine
                .execute_select_word(0x1131_218c, 0x00d4_a289)
                .unwrap();
            assert_eq!(true_step.condition_flag, 2);
            assert_eq!(true_step.value, 11);
            assert_eq!(machine.xregs()[10], 11);
            assert_eq!(bus.accesses, 0);

            let mut unknown = ScalarMachine::new(architecture, [0; 32], 0);
            let before = unknown.clone();
            assert_eq!(
                unknown.execute_select_word(0x1131_218c, 0x00d4_a289),
                Err(ScalarMachineError::SprValueUnavailable {
                    pc: 0x1131_218c,
                    spr: 11,
                })
            );
            assert_eq!(unknown, before);
        }
    }
}
