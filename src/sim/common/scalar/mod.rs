use crate::architecture::Architecture;
use crate::isa::class::AicClass;
use crate::isa::flow::{
    ConditionalJump, ConditionalJumpTarget, DcciInstruction, DcciStep, DsbStep, FlowEnd, FlowNop,
    JumpCompare, JumpCompareError, JumpCompareTarget, JumpTarget, PipelineBarrierStep,
    UnconditionalJump, compare_values,
};
use crate::isa::scalar::{
    ScalarInstruction, ScalarKey0Operation, ScalarKey7Operation, ScalarKey8Operation,
    ScalarLoadStoreOperation, ScalarStoreImmediateValue,
};
use crate::sim::common::scalar::alu::{
    evaluate_scalar_integer_immediate, update_neg_overflow_spr2, update_overflow_spr2,
};
use thiserror::Error;

mod alu;
mod bus;
mod integer;
mod memory;
mod stepper;

pub use alu::ScalarIntegerError;
pub use stepper::{ScalarProgramStep, ScalarStepper, ScalarStepperError};

pub const SCALAR_X_REGISTER_COUNT: usize = 32;
const SCALAR_SPR_SNAPSHOT_CAPACITY: usize = 243;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScalarMachine {
    architecture: Architecture,
    xregs: [u64; SCALAR_X_REGISTER_COUNT],
    c220_xreg32: u64,
    spr2: u64,
    spr_values: [Option<u64>; SCALAR_SPR_SNAPSHOT_CAPACITY],
    model_time: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScalarSprStep {
    pub pc: u64,
    pub word: u32,
    pub destination_spr: u16,
    pub prior_destination_value: Option<u64>,
    pub source_register: Option<u8>,
    pub source_value: u64,
    pub value: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarSprReadSource {
    RegisterSnapshot,
    ProgramCounter,
    ModelTime,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarInstructionStep {
    Register(ScalarStep),
    SprRead(ScalarSprReadStep),
    SprWrite(ScalarSprStep),
    Memory(ScalarMemoryStep),
    ImmediateStore(ScalarImmediateStoreStep),
    PairLoad(ScalarPairLoadStep),
    PairStore(ScalarPairStoreStep),
    IndexedLoad(ScalarIndexedLoadStep),
    IndexedStore(ScalarIndexedStoreStep),
    IndexedImmediateStore(ScalarIndexedImmediateStoreStep),
    Flow(ScalarFlowStep),
    CacheHint(ScalarCacheHintStep),
    Compare(ScalarCompareStep),
    CompareRegister(ScalarCompareRegisterStep),
    CompareImmediate(ScalarCompareImmediateStep),
    Select(ScalarSelectStep),
    Dcci(DcciStep),
    Dsb(DsbStep),
    Barrier(PipelineBarrierStep),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScalarCacheHintStep {
    pub pc: u64,
    pub word: u32,
    pub source_register: u8,
    pub source_value: u64,
    pub encoded_immediate: u16,
    pub effective_address: u64,
    pub requested_bytes: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScalarIndexedStoreStep {
    pub pc: u64,
    pub word: u32,
    pub effective_address: u64,
    pub width_bytes: u8,
    pub source_register: u8,
    pub value: u64,
    pub base_register: u8,
    pub base_value: u64,
    pub updated_base: Option<u64>,
    pub offset_register: u8,
    pub offset_value: u64,
    pub bytes: [u8; 8],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    pub updated_base: Option<u64>,
    pub offset_register: u8,
    pub offset_value: u64,
    pub bytes: [u8; 8],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScalarIndexedImmediateStoreStep {
    pub pc: u64,
    pub word: u32,
    pub effective_address: u64,
    pub width_bytes: u8,
    pub base_register: u8,
    pub base_value: u64,
    pub updated_base: Option<u64>,
    pub offset_register: u8,
    pub offset_value: u64,
    pub value: ScalarStoreImmediateValue,
    pub bytes: [u8; 8],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScalarImmediateStoreStep {
    pub pc: u64,
    pub word: u32,
    pub effective_address: u64,
    pub width_bytes: u8,
    pub base_register: u8,
    pub prior_base_value: u64,
    pub updated_base: Option<u64>,
    pub value: ScalarStoreImmediateValue,
    pub bytes: [u8; 8],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScalarCompareImmediateStep {
    pub pc: u64,
    pub word: u32,
    pub condition_field: u8,
    pub source_register: u8,
    pub source_value: u64,
    pub encoded_immediate: u16,
    pub signed_immediate: i16,
    pub prior_spr11: Option<u64>,
    pub spr11: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScalarSelectStep {
    pub pc: u64,
    pub word: u32,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    pub const fn new(
        architecture: Architecture,
        xregs: [u64; SCALAR_X_REGISTER_COUNT],
        spr2: u64,
    ) -> Self {
        Self {
            architecture,
            xregs,
            c220_xreg32: 0,
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

    /// Read a numbered register without conflating an absent register with zero.
    pub fn xreg_value(&self, register: u8) -> Option<u64> {
        if register == 32 && self.architecture == Architecture::Dav2201 {
            Some(self.c220_xreg32)
        } else {
            self.xregs.get(usize::from(register)).copied()
        }
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

    pub fn execute_compare_word(
        &mut self,
        pc: u64,
        word: u32,
    ) -> Result<ScalarCompareStep, ScalarMachineError> {
        let Some(ScalarInstruction::ScalarCompare {
            dtype_field,
            condition_field,
            first_source_register,
            second_source_register,
        }) = ScalarInstruction::from_word(self.architecture, word)
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
        let Some(ScalarInstruction::ScalarCompareRegister {
            dtype_field,
            condition_field,
            destination_register,
            first_source_register,
            second_source_register,
        }) = ScalarInstruction::from_word(self.architecture, word)
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
        let Some(ScalarInstruction::ScalarCompareImmediate {
            condition_field,
            source_register,
            encoded_immediate,
        }) = ScalarInstruction::from_word(self.architecture, word)
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
        let Some(ScalarInstruction::ScalarSelect {
            dtype_field,
            destination_register,
            first_source_register,
            second_source_register,
        }) = ScalarInstruction::from_word(self.architecture, word)
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
        let Some(ScalarInstruction::ScalarKey2MoveFromSpr {
            destination_register,
            encoded_source_spr,
            ..
        }) = ScalarInstruction::from_word(self.architecture, word)
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
        let Some(ScalarInstruction::ScalarKey2MoveToSpr {
            encoded_destination_spr,
            source_register,
            ..
        }) = ScalarInstruction::from_word(self.architecture, word)
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
            source_register: Some(source_register),
            source_value,
            value,
        })
    }

    pub fn set_xreg(&mut self, register: u8, value: u64) -> Result<(), ScalarMachineError> {
        if register == 32 && self.architecture == Architecture::Dav2201 {
            self.c220_xreg32 = value;
            return Ok(());
        }
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
        let Some(ScalarInstruction::ScalarKey8 {
            operation: ScalarKey8Operation::DcPreload,
            destination_register: None,
            source_register,
            encoded_immediate,
            ..
        }) = ScalarInstruction::from_word(self.architecture, word)
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

    pub fn execute_instruction<B: ScalarMemoryBus>(
        &mut self,
        pc: u64,
        word: u32,
        bus: &mut B,
    ) -> Result<ScalarInstructionStep, ScalarInstructionError<B::Error>> {
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
        match ScalarInstruction::from_word(self.architecture, word) {
            Some(ScalarInstruction::ScalarCompare { .. }) => Ok(ScalarInstructionStep::Compare(
                self.execute_compare_word(pc, word)?,
            )),
            Some(ScalarInstruction::ScalarCompareRegister { .. }) => {
                Ok(ScalarInstructionStep::CompareRegister(
                    self.execute_compare_register_word(pc, word)?,
                ))
            }
            Some(ScalarInstruction::ScalarCompareImmediate { .. }) => {
                Ok(ScalarInstructionStep::CompareImmediate(
                    self.execute_compare_immediate_word(pc, word)?,
                ))
            }
            Some(ScalarInstruction::ScalarSelect { .. }) => Ok(ScalarInstructionStep::Select(
                self.execute_select_word(pc, word)?,
            )),
            Some(ScalarInstruction::ScalarPairLoad { .. }) => Ok(ScalarInstructionStep::PairLoad(
                self.execute_pair_load_word(pc, word, bus)?,
            )),
            Some(ScalarInstruction::ScalarPairStore { .. }) => Ok(
                ScalarInstructionStep::PairStore(self.execute_pair_store_word(pc, word, bus)?),
            ),
            Some(ScalarInstruction::ScalarIndexedLoad { .. }) => Ok(
                ScalarInstructionStep::IndexedLoad(self.execute_indexed_load_word(pc, word, bus)?),
            ),
            Some(ScalarInstruction::ScalarIndexedStore { .. }) => {
                Ok(ScalarInstructionStep::IndexedStore(
                    self.execute_indexed_store_word(pc, word, bus)?,
                ))
            }
            Some(ScalarInstruction::ScalarIndexedImmediateStore { .. }) => {
                Ok(ScalarInstructionStep::IndexedImmediateStore(
                    self.execute_indexed_immediate_store_word(pc, word, bus)?,
                ))
            }
            Some(ScalarInstruction::ScalarKey8 {
                operation: ScalarKey8Operation::DcPreload,
                ..
            }) => Ok(ScalarInstructionStep::CacheHint(
                self.execute_cache_hint_word(pc, word)?,
            )),
            Some(ScalarInstruction::ScalarLoadStoreImmediate { .. }) => Ok(
                ScalarInstructionStep::Memory(self.execute_memory_word(pc, word, bus)?),
            ),
            Some(ScalarInstruction::ScalarStoreImmediate { .. }) => {
                Ok(ScalarInstructionStep::ImmediateStore(
                    self.execute_immediate_store_word(pc, word, bus)?,
                ))
            }
            Some(ScalarInstruction::ScalarKey2MoveFromSpr { .. }) => Ok(
                ScalarInstructionStep::SprRead(self.execute_spr_read_word(pc, word)?),
            ),
            Some(ScalarInstruction::ScalarKey2MoveToSpr { .. }) => Ok(
                ScalarInstructionStep::SprWrite(self.execute_spr_word(pc, word)?),
            ),
            _ => Ok(ScalarInstructionStep::Register(
                self.execute_word(pc, word)?,
            )),
        }
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
mod tests;
