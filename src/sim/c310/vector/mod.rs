mod arithmetic;
mod predicate;
mod transfer;
pub use arithmetic::{
    C310CapturedVdupsStep, C310CapturedVectorError, C310RvecValueStep, evaluate_c310_fp32_lanes,
};
pub use predicate::{
    C310CapturedPltError, C310CapturedPltStep, C310CapturedPsetError, C310CapturedPsetStep,
    C310CapturedSmoviError, C310CapturedSmoviStep, C310ObservedMovemaskError,
    C310ObservedMovemaskStep, C310RvecMaskSprState, C310RvecMovpStep,
    c310_movp_u32_mask_to_predicate_bytes, c310_predicate_bytes_to_mask,
};
pub use transfer::{
    C310CapturedVectorLoadAddress, C310CapturedVectorLoadError, C310CapturedVectorLoadStep,
    C310CapturedVstStep, C310CapturedVstStore, C310RvecVstiStore, c310_normal_u32_masked_store,
};

use crate::isa::c310::layout::C310_PB_SLOT_BYTES;
use crate::memory::ub::UbMemoryError;
use crate::numeric::fp32::Fp32VectorError;
use crate::sim::c310::predicate::{C310PbRvecScalarProjection, project_c310_pb_rvec_scalar_init};

use thiserror::Error;

const MAX_ENCODED_V_REGISTERS: usize = 32;
const MAX_ENCODED_P_REGISTERS: usize = 32;
const MAX_FP32_WORDS_PER_REGISTER: usize = 64;
const MAX_PREDICATE_BYTES: usize = 32;
const C310_SCALAR_REGISTER_COUNT: usize = 96;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C310RvecValueMachine {
    vector_registers: Vec<Vec<u32>>,
    predicate_registers: Option<Vec<Vec<u8>>>,
    scalar_registers: Vec<Option<u32>>,
    words_per_register: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum C310RvecValueError {
    #[error("C310 RVec Add/Sub word is unsupported")]
    UnsupportedWord,
    #[error("C310 RVec MOVP word is unsupported")]
    UnsupportedMovpWord,
    #[error("C310 RVec MOVP word does not select the u32 value path")]
    UnsupportedMovpDtype,
    #[error("C310 RVec VSTI word is unsupported")]
    UnsupportedVstiWord,
    #[error("C310 RVec VSTI word does not select the normal-u32 store path")]
    UnsupportedVstiDtype,
    #[error("C310 VSTI buffer address overflows at lane {lane} from base {base:#x}")]
    VstiAddressOverflow { base: u64, lane: usize },
    #[error("C310 RVec word does not select the FP32 value path")]
    UnsupportedDtype,
    #[error("V-register count {count} is outside 1..={MAX_ENCODED_V_REGISTERS}")]
    RegisterCount { count: usize },
    #[error("FP32 V-register width {words} is outside 1..={MAX_FP32_WORDS_PER_REGISTER} words")]
    RegisterWidth { words: usize },
    #[error("V-register {index} has {actual} words, expected {expected}")]
    UnequalRegisterWidth {
        index: usize,
        actual: usize,
        expected: usize,
    },
    #[error("encoded V-register {index} is outside the caller-supplied bank of {count}")]
    RegisterIndex { index: usize, count: usize },
    #[error("P-register count {count} is outside 1..={MAX_ENCODED_P_REGISTERS}")]
    PredicateRegisterCount { count: usize },
    #[error("P-register byte width {bytes} exceeds {MAX_PREDICATE_BYTES}")]
    PredicateWidth { bytes: usize },
    #[error("MOVP P-register byte width {actual} differs from the modeled {expected}")]
    MovpPredicateWidth { expected: usize, actual: usize },
    #[error("no caller-supplied P-register bank is attached")]
    MissingPredicateBank,
    #[error("encoded P-register {index} is outside the caller-supplied bank of {count}")]
    PredicateRegisterIndex { index: usize, count: usize },
    #[error(transparent)]
    Fp32(#[from] Fp32VectorError),
    #[error(transparent)]
    Ub(#[from] UbMemoryError),
}

impl C310RvecValueMachine {
    pub fn from_vector_words(vector_registers: Vec<Vec<u32>>) -> Result<Self, C310RvecValueError> {
        if vector_registers.is_empty() || vector_registers.len() > MAX_ENCODED_V_REGISTERS {
            return Err(C310RvecValueError::RegisterCount {
                count: vector_registers.len(),
            });
        }
        let words_per_register = vector_registers[0].len();
        if words_per_register == 0 || words_per_register > MAX_FP32_WORDS_PER_REGISTER {
            return Err(C310RvecValueError::RegisterWidth {
                words: words_per_register,
            });
        }
        for (index, register) in vector_registers.iter().enumerate().skip(1) {
            if register.len() != words_per_register {
                return Err(C310RvecValueError::UnequalRegisterWidth {
                    index,
                    actual: register.len(),
                    expected: words_per_register,
                });
            }
        }
        Ok(Self {
            vector_registers,
            predicate_registers: None,
            scalar_registers: vec![None; C310_SCALAR_REGISTER_COUNT],
            words_per_register,
        })
    }

    pub fn from_vector_and_predicate_bytes(
        vector_registers: Vec<Vec<u32>>,
        predicate_registers: Vec<Vec<u8>>,
    ) -> Result<Self, C310RvecValueError> {
        let mut machine = Self::from_vector_words(vector_registers)?;
        if predicate_registers.is_empty() || predicate_registers.len() > MAX_ENCODED_P_REGISTERS {
            return Err(C310RvecValueError::PredicateRegisterCount {
                count: predicate_registers.len(),
            });
        }
        for register in &predicate_registers {
            c310_predicate_bytes_to_mask(register)?;
        }
        machine.predicate_registers = Some(predicate_registers);
        Ok(machine)
    }

    pub fn words_per_register(&self) -> usize {
        self.words_per_register
    }

    pub fn vector_register(&self, index: usize) -> Option<&[u32]> {
        self.vector_registers.get(index).map(Vec::as_slice)
    }

    pub fn predicate_register(&self, index: usize) -> Option<&[u8]> {
        self.predicate_registers
            .as_ref()?
            .get(index)
            .map(Vec::as_slice)
    }

    pub fn captured_s65(&self) -> Option<u32> {
        self.scalar_register(65)
    }

    pub fn scalar_register(&self, index: usize) -> Option<u32> {
        self.scalar_registers.get(index).copied().flatten()
    }

    pub fn apply_pb_scalar_init(
        &mut self,
        slot: &[u8; C310_PB_SLOT_BYTES],
    ) -> C310PbRvecScalarProjection {
        let projection = project_c310_pb_rvec_scalar_init(slot);
        for write in &projection.writes {
            self.scalar_registers[usize::from(write.register_index)] = Some(write.value);
        }
        projection
    }
}

#[cfg(test)]
mod tests;
