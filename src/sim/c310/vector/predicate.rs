use super::{C310RvecValueError, C310RvecValueMachine, MAX_PREDICATE_BYTES};
use crate::isa::c310::vector::{
    C310_CAPTURED_PLT32_WORD, C310_CAPTURED_PSET_WORD, C310_CAPTURED_SMOVI32_WORD,
    C310_CAPTURED_VECTOR_LOAD_BYTES, C310_MASK0_SPR_INDEX, C310ObservedMovemaskHint,
    C310RvecMovpHint,
};

use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C310CapturedSmoviStep {
    pub pc: u64,
    pub word: u32,
    pub destination_s_register: u8,
    pub prior_value: Option<u32>,
    pub value: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum C310CapturedSmoviError {
    #[error("C310 SMOVI word {word:#010x} at PC {pc:#x} is outside the captured path")]
    UnsupportedWord { pc: u64, word: u32 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C310CapturedPltStep {
    pub pc: u64,
    pub word: u32,
    pub destination_p_register: u8,
    pub lane_limit: usize,
    pub remaining_scalar_value: u32,
    pub predicate_bytes: [u8; MAX_PREDICATE_BYTES],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C310CapturedPsetStep {
    pub pc: u64,
    pub word: u32,
    pub destination_p_register: u8,
    pub predicate_bytes: [u8; MAX_PREDICATE_BYTES],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum C310CapturedPsetError {
    #[error("C310 PSET word {word:#010x} at PC {pc:#x} is outside the captured path")]
    UnsupportedWord { pc: u64, word: u32 },
    #[error("C310 captured PSET requires a P-register bank")]
    MissingPredicateBank,
    #[error("C310 captured PSET requires P1 in the bank of {count}")]
    MissingP1 { count: usize },
    #[error("C310 captured PSET requires a 32-byte P1, got {actual} bytes")]
    PredicateWidth { actual: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum C310CapturedPltError {
    #[error("C310 PLT word {word:#010x} at PC {pc:#x} is outside the captured B32 path")]
    UnsupportedWord { pc: u64, word: u32 },
    #[error("C310 captured PLT requires a 256-byte V-register, got {actual} bytes")]
    RegisterWidth { actual: usize },
    #[error("C310 captured PLT requires a P-register bank")]
    MissingPredicateBank,
    #[error("C310 captured PLT requires P1 in the bank of {count}")]
    MissingP1 { count: usize },
    #[error("C310 captured PLT requires a 32-byte P1, got {actual} bytes")]
    PredicateWidth { actual: usize },
    #[error("C310 captured PLT requires a preceding S65 value")]
    MissingScalarLimit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C310RvecMaskSprState {
    pub mask0: u64,
    pub mask1: u64,
}

impl Default for C310RvecMaskSprState {
    fn default() -> Self {
        Self {
            mask0: u64::MAX,
            mask1: u64::MAX,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C310ObservedMovemaskStep {
    pub pc: u64,
    pub word: u32,
    pub hint: C310ObservedMovemaskHint,
    pub prior_value: u64,
    pub value: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum C310ObservedMovemaskError {
    #[error("C310 MOVEMASK word {word:#010x} at PC {pc:#x} is not in the verified set")]
    UnsupportedWord { pc: u64, word: u32 },
}

impl C310RvecMaskSprState {
    pub fn execute_observed_movemask_word(
        &mut self,
        pc: u64,
        word: u32,
        xregs: &[u64; 32],
    ) -> Result<C310ObservedMovemaskStep, C310ObservedMovemaskError> {
        let hint = C310ObservedMovemaskHint::from_word(word)
            .ok_or(C310ObservedMovemaskError::UnsupportedWord { pc, word })?;
        let destination = if hint.destination_spr == C310_MASK0_SPR_INDEX {
            &mut self.mask0
        } else {
            &mut self.mask1
        };
        let prior_value = *destination;
        let value = xregs[usize::from(hint.source_x_register)];
        *destination = value;
        Ok(C310ObservedMovemaskStep {
            pc,
            word,
            hint,
            prior_value,
            value,
        })
    }
}

pub fn c310_predicate_bytes_to_mask(bytes: &[u8]) -> Result<[u64; 4], C310RvecValueError> {
    if bytes.len() > MAX_PREDICATE_BYTES {
        return Err(C310RvecValueError::PredicateWidth { bytes: bytes.len() });
    }
    let mut mask = [0_u64; 4];
    for (byte_index, &byte) in bytes.iter().enumerate() {
        mask[byte_index / 8] |= u64::from(byte) << (8 * (byte_index % 8));
    }
    Ok(mask)
}

pub fn c310_movp_u32_mask_to_predicate_bytes(scalar_mask: u64) -> [u8; MAX_PREDICATE_BYTES] {
    let mut bytes = [0_u8; MAX_PREDICATE_BYTES];
    for lane in 0..64 {
        if (scalar_mask >> lane) & 1 != 0 {
            bytes[lane / 2] |= 0x0f << (4 * (lane % 2));
        }
    }
    bytes
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C310RvecMovpStep {
    pub hint: C310RvecMovpHint,
    pub scalar_mask: u64,
    pub predicate_bytes: [u8; MAX_PREDICATE_BYTES],
}

impl C310RvecValueMachine {
    pub fn execute_captured_smovi_word(
        &mut self,
        pc: u64,
        word: u32,
    ) -> Result<C310CapturedSmoviStep, C310CapturedSmoviError> {
        if word != C310_CAPTURED_SMOVI32_WORD {
            return Err(C310CapturedSmoviError::UnsupportedWord { pc, word });
        }
        let value = (word >> 9) & 0xffff;
        let destination_s_register = 64 + ((word >> 25) & 0x1f) as u8;
        let prior_value = self.scalar_registers[usize::from(destination_s_register)].replace(value);
        Ok(C310CapturedSmoviStep {
            pc,
            word,
            destination_s_register,
            prior_value,
            value,
        })
    }

    pub fn execute_captured_plt32_word(
        &mut self,
        pc: u64,
        word: u32,
    ) -> Result<C310CapturedPltStep, C310CapturedPltError> {
        if word != C310_CAPTURED_PLT32_WORD {
            return Err(C310CapturedPltError::UnsupportedWord { pc, word });
        }
        let actual = self.words_per_register * 4;
        if actual != C310_CAPTURED_VECTOR_LOAD_BYTES {
            return Err(C310CapturedPltError::RegisterWidth { actual });
        }
        let predicates = self
            .predicate_registers
            .as_mut()
            .ok_or(C310CapturedPltError::MissingPredicateBank)?;
        let count = predicates.len();
        let destination = predicates
            .get_mut(1)
            .ok_or(C310CapturedPltError::MissingP1 { count })?;
        if destination.len() != MAX_PREDICATE_BYTES {
            return Err(C310CapturedPltError::PredicateWidth {
                actual: destination.len(),
            });
        }
        let scalar_limit =
            self.scalar_registers[65].ok_or(C310CapturedPltError::MissingScalarLimit)?;
        let lane_limit = usize::try_from(scalar_limit)
            .unwrap_or(usize::MAX)
            .min(self.words_per_register);
        let mut predicate_bytes = [0_u8; MAX_PREDICATE_BYTES];
        for lane in 0..lane_limit {
            predicate_bytes[lane / 2] |= 1 << (4 * (lane % 2));
        }
        destination.copy_from_slice(&predicate_bytes);
        let remaining_scalar_value = scalar_limit.saturating_sub(self.words_per_register as u32);
        self.scalar_registers[65] = Some(remaining_scalar_value);
        Ok(C310CapturedPltStep {
            pc,
            word,
            destination_p_register: 1,
            lane_limit,
            remaining_scalar_value,
            predicate_bytes,
        })
    }

    pub fn execute_captured_pset_word(
        &mut self,
        pc: u64,
        word: u32,
    ) -> Result<C310CapturedPsetStep, C310CapturedPsetError> {
        if word != C310_CAPTURED_PSET_WORD {
            return Err(C310CapturedPsetError::UnsupportedWord { pc, word });
        }
        let predicates = self
            .predicate_registers
            .as_mut()
            .ok_or(C310CapturedPsetError::MissingPredicateBank)?;
        let count = predicates.len();
        let destination = predicates
            .get_mut(1)
            .ok_or(C310CapturedPsetError::MissingP1 { count })?;
        if destination.len() != MAX_PREDICATE_BYTES {
            return Err(C310CapturedPsetError::PredicateWidth {
                actual: destination.len(),
            });
        }
        let predicate_bytes = [0x11; MAX_PREDICATE_BYTES];
        destination.copy_from_slice(&predicate_bytes);
        Ok(C310CapturedPsetStep {
            pc,
            word,
            destination_p_register: 1,
            predicate_bytes,
        })
    }

    pub fn execute_movp_u32_word(
        &mut self,
        word: u32,
        scalar_mask: u64,
    ) -> Result<C310RvecMovpStep, C310RvecValueError> {
        let hint =
            C310RvecMovpHint::from_word(word).ok_or(C310RvecValueError::UnsupportedMovpWord)?;
        if !hint.has_u32_value_path() {
            return Err(C310RvecValueError::UnsupportedMovpDtype);
        }
        let predicate_registers = self
            .predicate_registers
            .as_mut()
            .ok_or(C310RvecValueError::MissingPredicateBank)?;
        let index = usize::from(hint.destination_p_register);
        let count = predicate_registers.len();
        let destination = predicate_registers
            .get_mut(index)
            .ok_or(C310RvecValueError::PredicateRegisterIndex { index, count })?;
        if destination.len() != MAX_PREDICATE_BYTES {
            return Err(C310RvecValueError::MovpPredicateWidth {
                expected: MAX_PREDICATE_BYTES,
                actual: destination.len(),
            });
        }
        let predicate_bytes = c310_movp_u32_mask_to_predicate_bytes(scalar_mask);
        destination.copy_from_slice(&predicate_bytes);
        Ok(C310RvecMovpStep {
            hint,
            scalar_mask,
            predicate_bytes,
        })
    }

    pub fn execute_movp_u32_from_mask_sprs(
        &mut self,
        word: u32,
        mask_sprs: &C310RvecMaskSprState,
    ) -> Result<C310RvecMovpStep, C310RvecValueError> {
        self.execute_movp_u32_word(word, mask_sprs.mask0)
    }
}
