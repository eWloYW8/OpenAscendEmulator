use super::{C310RvecValueError, C310RvecValueMachine, c310_predicate_bytes_to_mask};
use crate::isa::c310::vector::{
    C310_CAPTURED_VDUPS_WORD, C310_CAPTURED_VECTOR_LOAD_BYTES, C310RvecArithmeticHint,
    C310RvecArithmeticOperation,
};

use crate::memory::ub::UbMemoryError;
use crate::numeric::fp32::{
    Fp32LaneOutcome, Fp32MaskLayout, Fp32VectorError, Fp32VectorOperation, Fp32WritebackOutcome,
    Fp32WritebackPolicy, apply_fp32_writeback, evaluate_masked_fp32_lanes,
};

use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C310CapturedVdupsStep {
    pub pc: u64,
    pub word: u32,
    pub destination_v_register: u8,
    pub scalar_word: u32,
    pub written_lanes: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum C310CapturedVectorError {
    #[error("C310 vector word {word:#010x} at PC {pc:#x} is outside the captured path")]
    UnsupportedWord { pc: u64, word: u32 },
    #[error("C310 captured vector path requires a 256-byte register, got {actual} bytes")]
    RegisterWidth { actual: usize },
    #[error("C310 V-register {index} is outside the bank of {count}")]
    RegisterIndex { index: usize, count: usize },
    #[error("C310 VST address overflows at lane {lane} from base {base:#x}")]
    AddressOverflow { base: u64, lane: usize },
    #[error("cannot reserve {lanes} vector lanes")]
    HostAllocationFailed { lanes: usize },
    #[error(transparent)]
    Ub(#[from] UbMemoryError),
}

pub fn evaluate_c310_fp32_lanes(
    hint: C310RvecArithmeticHint,
    first: &[u32],
    second: &[u32],
    active_mask: &[u64; 4],
) -> Result<Vec<Fp32LaneOutcome>, Fp32VectorError> {
    if !hint.has_fp32_value_path() {
        return Err(Fp32VectorError::UnsupportedInstruction);
    }
    let operation = match hint.operation {
        C310RvecArithmeticOperation::Add => Fp32VectorOperation::Add,
        C310RvecArithmeticOperation::Subtract => Fp32VectorOperation::Subtract,
        C310RvecArithmeticOperation::Multiply => Fp32VectorOperation::Multiply,
    };
    evaluate_masked_fp32_lanes(
        operation,
        Fp32MaskLayout::C310ByteStart,
        first,
        second,
        active_mask,
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C310RvecValueStep {
    pub hint: C310RvecArithmeticHint,
    pub active_mask: [u64; 4],
    pub first_source: Vec<u32>,
    pub second_source: Vec<u32>,
    pub lanes: Vec<Fp32LaneOutcome>,
    pub writeback: Fp32WritebackOutcome,
}

impl C310RvecValueMachine {
    pub fn execute_captured_vdups_word(
        &mut self,
        pc: u64,
        word: u32,
        scalar_word: u32,
        active_mask: &[u64; 4],
    ) -> Result<C310CapturedVdupsStep, C310CapturedVectorError> {
        if word != C310_CAPTURED_VDUPS_WORD {
            return Err(C310CapturedVectorError::UnsupportedWord { pc, word });
        }
        let actual = self.words_per_register * 4;
        if actual != C310_CAPTURED_VECTOR_LOAD_BYTES {
            return Err(C310CapturedVectorError::RegisterWidth { actual });
        }
        let destination_v_register = ((word >> 25) & 0x1f) as u8;
        let index = usize::from(destination_v_register);
        let count = self.vector_registers.len();
        let destination = self
            .vector_registers
            .get_mut(index)
            .ok_or(C310CapturedVectorError::RegisterIndex { index, count })?;
        let mut written_lanes = Vec::new();
        written_lanes
            .try_reserve_exact(destination.len())
            .map_err(|_| C310CapturedVectorError::HostAllocationFailed {
                lanes: destination.len(),
            })?;
        for (lane, value) in destination.iter_mut().enumerate() {
            let byte_bit = lane * 4;
            if (active_mask[byte_bit / 64] >> (byte_bit % 64)) & 1 != 0 {
                *value = scalar_word;
                written_lanes.push(lane);
            }
        }
        Ok(C310CapturedVdupsStep {
            pc,
            word,
            destination_v_register,
            scalar_word,
            written_lanes,
        })
    }

    pub fn execute_fp32_word_from_predicate_registers(
        &mut self,
        word: u32,
    ) -> Result<C310RvecValueStep, C310RvecValueError> {
        let hint =
            C310RvecArithmeticHint::from_word(word).ok_or(C310RvecValueError::UnsupportedWord)?;
        if !hint.has_fp32_value_path() {
            return Err(C310RvecValueError::UnsupportedDtype);
        }
        let predicate_registers = self
            .predicate_registers
            .as_ref()
            .ok_or(C310RvecValueError::MissingPredicateBank)?;
        let index = usize::from(hint.predicate_register);
        let bytes =
            predicate_registers
                .get(index)
                .ok_or(C310RvecValueError::PredicateRegisterIndex {
                    index,
                    count: predicate_registers.len(),
                })?;
        let active_mask = c310_predicate_bytes_to_mask(bytes)?;
        self.execute_fp32_word(word, &active_mask)
    }

    pub fn execute_fp32_word(
        &mut self,
        word: u32,
        active_mask: &[u64; 4],
    ) -> Result<C310RvecValueStep, C310RvecValueError> {
        let hint =
            C310RvecArithmeticHint::from_word(word).ok_or(C310RvecValueError::UnsupportedWord)?;
        if !hint.has_fp32_value_path() {
            return Err(C310RvecValueError::UnsupportedDtype);
        }
        let count = self.vector_registers.len();
        for index in [
            hint.destination_v_register,
            hint.first_source_v_register,
            hint.second_source_v_register,
        ] {
            if usize::from(index) >= count {
                return Err(C310RvecValueError::RegisterIndex {
                    index: usize::from(index),
                    count,
                });
            }
        }
        let first_source = self.vector_registers[usize::from(hint.first_source_v_register)].clone();
        let second_source =
            self.vector_registers[usize::from(hint.second_source_v_register)].clone();
        let lanes = evaluate_c310_fp32_lanes(hint, &first_source, &second_source, active_mask)?;
        let destination = usize::from(hint.destination_v_register);
        let writeback = apply_fp32_writeback(
            Fp32WritebackPolicy::C310WholeResult,
            &self.vector_registers[destination],
            &lanes,
        )?;
        self.vector_registers[destination].clone_from(&writeback.words);
        Ok(C310RvecValueStep {
            hint,
            active_mask: *active_mask,
            first_source,
            second_source,
            lanes,
            writeback,
        })
    }
}
