use super::{
    C310CapturedVectorError, C310RvecValueError, C310RvecValueMachine, MAX_FP32_WORDS_PER_REGISTER,
    c310_predicate_bytes_to_mask,
};
use crate::isa::c310::vector::{
    C310_CAPTURED_SUB_VST_WORD, C310_CAPTURED_VECTOR_LOAD_BYTES, C310_CAPTURED_VST_WORD,
    C310CapturedVectorLoadHint, C310RvecVstiHint,
};

use crate::memory::sparse::MemoryByteState;
use crate::memory::ub::{UbMemory, UbMemoryError};

use crate::numeric::fp32::{Fp32VectorError, Fp32WritebackOutcome};

use crate::sim::c310::address::C310RvecAddressState;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C310CapturedVstStep {
    pub pc: u64,
    pub word: u32,
    pub source_v_register: u8,
    pub destination_address: u64,
    pub stores: Vec<C310CapturedVstStore>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C310CapturedVstStore {
    pub lane_index: usize,
    pub buffer_address: u64,
    pub data: [u8; 4],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C310CapturedVectorLoadStep {
    pub pc: u64,
    pub word: u32,
    pub hint: C310CapturedVectorLoadHint,
    pub source_address: u64,
    pub loaded_bytes: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C310CapturedVectorLoadAddress {
    pub source_s_register: u8,
    pub source_scalar_low: u32,
    pub source_scalar_high: u32,
    pub address_a0: Option<u32>,
    pub effective_address: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum C310CapturedVectorLoadError {
    #[error(
        "C310 vector load word {word:#010x} at PC {pc:#x} is outside the captured normal-u8 path"
    )]
    UnsupportedWord { pc: u64, word: u32 },
    #[error(
        "C310 vector load requires a {C310_CAPTURED_VECTOR_LOAD_BYTES}-byte V-register, got {actual} bytes"
    )]
    RegisterWidth { actual: usize },
    #[error("C310 vector load destination V-register {index} is outside the bank of {count}")]
    RegisterIndex { index: usize, count: usize },
    #[error("C310 vector load requires scalar register S{index}")]
    MissingScalar { index: u8 },
    #[error("C310 vector load requires address register A0")]
    MissingAddressA0,
    #[error(transparent)]
    Ub(#[from] UbMemoryError),
}

pub fn c310_normal_u32_masked_store(
    previous_memory: &[u32],
    source_words: &[u32],
    predicate_bytes: &[u8],
) -> Result<Fp32WritebackOutcome, C310RvecValueError> {
    if previous_memory.len() > MAX_FP32_WORDS_PER_REGISTER {
        return Err(Fp32VectorError::TooManyLanes {
            lanes: previous_memory.len(),
        }
        .into());
    }
    if source_words.len() > MAX_FP32_WORDS_PER_REGISTER {
        return Err(Fp32VectorError::TooManyLanes {
            lanes: source_words.len(),
        }
        .into());
    }
    if previous_memory.len() < source_words.len() {
        return Err(Fp32VectorError::DestinationTooSmall {
            destination: previous_memory.len(),
            results: source_words.len(),
        }
        .into());
    }
    let active_mask = c310_predicate_bytes_to_mask(predicate_bytes)?;
    let mut words = Vec::new();
    words
        .try_reserve_exact(previous_memory.len())
        .map_err(|_| Fp32VectorError::HostAllocationFailed {
            lanes: previous_memory.len(),
        })?;
    words.extend_from_slice(previous_memory);
    let mut written = Vec::new();
    written
        .try_reserve_exact(previous_memory.len())
        .map_err(|_| Fp32VectorError::HostAllocationFailed {
            lanes: previous_memory.len(),
        })?;
    written.resize(previous_memory.len(), false);
    for (lane, &source) in source_words.iter().enumerate() {
        let bit = lane * 4;
        if (active_mask[bit / 64] >> (bit % 64)) & 1 != 0 {
            words[lane] = source;
            written[lane] = true;
        }
    }
    Ok(Fp32WritebackOutcome { words, written })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C310RvecVstiStore {
    pub lane_index: usize,
    pub buffer_address: u64,
    pub data: [u8; 4],
}

impl C310RvecValueMachine {
    pub fn execute_captured_vst_word(
        &self,
        pc: u64,
        word: u32,
        destination_address: u64,
        active_mask: &[u64; 4],
        ub: &mut UbMemory,
    ) -> Result<C310CapturedVstStep, C310CapturedVectorError> {
        if !matches!(word, C310_CAPTURED_VST_WORD | C310_CAPTURED_SUB_VST_WORD) {
            return Err(C310CapturedVectorError::UnsupportedWord { pc, word });
        }
        let actual = self.words_per_register * 4;
        if actual != C310_CAPTURED_VECTOR_LOAD_BYTES {
            return Err(C310CapturedVectorError::RegisterWidth { actual });
        }
        let source_v_register = ((word >> 25) & 0x1f) as u8;
        let index = usize::from(source_v_register);
        let source =
            self.vector_registers
                .get(index)
                .ok_or(C310CapturedVectorError::RegisterIndex {
                    index,
                    count: self.vector_registers.len(),
                })?;
        let mut stores = Vec::new();
        stores.try_reserve_exact(source.len()).map_err(|_| {
            C310CapturedVectorError::HostAllocationFailed {
                lanes: source.len(),
            }
        })?;
        for (lane_index, &value) in source.iter().enumerate() {
            let byte_bit = lane_index * 4;
            if (active_mask[byte_bit / 64] >> (byte_bit % 64)) & 1 == 0 {
                continue;
            }
            let buffer_address = destination_address
                .checked_add((lane_index * 4) as u64)
                .ok_or(C310CapturedVectorError::AddressOverflow {
                    base: destination_address,
                    lane: lane_index,
                })?;
            stores.push(C310CapturedVstStore {
                lane_index,
                buffer_address,
                data: value.to_le_bytes(),
            });
        }
        let mut staged = ub.clone();
        for store in &stores {
            staged.write_states(
                store.buffer_address,
                &store.data.map(MemoryByteState::Known),
            )?;
        }
        *ub = staged;
        Ok(C310CapturedVstStep {
            pc,
            word,
            source_v_register,
            destination_address,
            stores,
        })
    }

    pub fn execute_captured_vector_load_word(
        &mut self,
        pc: u64,
        word: u32,
        caller_addresses: &[u64; 32],
        ub: &UbMemory,
    ) -> Result<C310CapturedVectorLoadStep, C310CapturedVectorLoadError> {
        let hint = C310CapturedVectorLoadHint::from_word(word)
            .ok_or(C310CapturedVectorLoadError::UnsupportedWord { pc, word })?;
        let selector = usize::from(hint.source_s_register) * 2;
        let source_address = caller_addresses[selector];
        self.load_captured_vector_at_address(pc, word, hint, source_address, ub)
    }

    pub fn resolve_captured_vector_load_address(
        &self,
        pc: u64,
        word: u32,
        address_state: &C310RvecAddressState,
    ) -> Result<C310CapturedVectorLoadAddress, C310CapturedVectorLoadError> {
        let hint = C310CapturedVectorLoadHint::from_word(word)
            .ok_or(C310CapturedVectorLoadError::UnsupportedWord { pc, word })?;
        let low_index = hint.source_s_register;
        let high_index = low_index + 1;
        let low = self
            .scalar_register(usize::from(low_index))
            .ok_or(C310CapturedVectorLoadError::MissingScalar { index: low_index })?;
        let high = self
            .scalar_register(usize::from(high_index))
            .ok_or(C310CapturedVectorLoadError::MissingScalar { index: high_index })?;
        let address_a0 = if hint.source_a_register.is_some() {
            Some(
                address_state
                    .address_a0()
                    .ok_or(C310CapturedVectorLoadError::MissingAddressA0)?,
            )
        } else {
            None
        };
        let base = low | high.wrapping_shl(16);
        let effective_address = u64::from(base.wrapping_add(address_a0.unwrap_or(0)));
        Ok(C310CapturedVectorLoadAddress {
            source_s_register: low_index,
            source_scalar_low: low,
            source_scalar_high: high,
            address_a0,
            effective_address,
        })
    }

    pub fn execute_captured_vector_load_from_scalar_state(
        &mut self,
        pc: u64,
        word: u32,
        address_state: &C310RvecAddressState,
        ub: &UbMemory,
    ) -> Result<C310CapturedVectorLoadStep, C310CapturedVectorLoadError> {
        let resolved = self.resolve_captured_vector_load_address(pc, word, address_state)?;
        let hint = C310CapturedVectorLoadHint::from_word(word)
            .ok_or(C310CapturedVectorLoadError::UnsupportedWord { pc, word })?;
        self.load_captured_vector_at_address(pc, word, hint, resolved.effective_address, ub)
    }

    fn load_captured_vector_at_address(
        &mut self,
        pc: u64,
        word: u32,
        hint: C310CapturedVectorLoadHint,
        source_address: u64,
        ub: &UbMemory,
    ) -> Result<C310CapturedVectorLoadStep, C310CapturedVectorLoadError> {
        let actual = self.words_per_register * 4;
        if actual != hint.byte_count {
            return Err(C310CapturedVectorLoadError::RegisterWidth { actual });
        }
        let destination = usize::from(hint.destination_v_register);
        if destination >= self.vector_registers.len() {
            return Err(C310CapturedVectorLoadError::RegisterIndex {
                index: destination,
                count: self.vector_registers.len(),
            });
        }
        let loaded_bytes = ub.read_known(source_address, hint.byte_count)?;
        for (word, bytes) in self.vector_registers[destination]
            .iter_mut()
            .zip(loaded_bytes.chunks_exact(4))
        {
            *word = u32::from_le_bytes(bytes.try_into().expect("four bytes"));
        }
        Ok(C310CapturedVectorLoadStep {
            pc,
            word,
            hint,
            source_address,
            loaded_bytes,
        })
    }

    pub fn plan_normal_u32_vsti(
        &self,
        word: u32,
        buffer_start: u64,
    ) -> Result<Vec<C310RvecVstiStore>, C310RvecValueError> {
        let hint =
            C310RvecVstiHint::from_word(word).ok_or(C310RvecValueError::UnsupportedVstiWord)?;
        if !hint.has_normal_u32_store_path() {
            return Err(C310RvecValueError::UnsupportedVstiDtype);
        }
        let source_index = usize::from(hint.source_v_register);
        let source =
            self.vector_registers
                .get(source_index)
                .ok_or(C310RvecValueError::RegisterIndex {
                    index: source_index,
                    count: self.vector_registers.len(),
                })?;
        let predicates = self
            .predicate_registers
            .as_ref()
            .ok_or(C310RvecValueError::MissingPredicateBank)?;
        let predicate_index = usize::from(hint.predicate_register);
        let predicate =
            predicates
                .get(predicate_index)
                .ok_or(C310RvecValueError::PredicateRegisterIndex {
                    index: predicate_index,
                    count: predicates.len(),
                })?;
        let active_mask = c310_predicate_bytes_to_mask(predicate)?;
        let mut stores = Vec::new();
        stores.try_reserve_exact(source.len()).map_err(|_| {
            Fp32VectorError::HostAllocationFailed {
                lanes: source.len(),
            }
        })?;
        for (lane_index, &word) in source.iter().enumerate() {
            let bit_index = lane_index * 4;
            if (active_mask[bit_index / 64] >> (bit_index % 64)) & 1 == 0 {
                continue;
            }
            let buffer_address = buffer_start.checked_add((lane_index as u64) * 4).ok_or(
                C310RvecValueError::VstiAddressOverflow {
                    base: buffer_start,
                    lane: lane_index,
                },
            )?;
            stores.push(C310RvecVstiStore {
                lane_index,
                buffer_address,
                data: word.to_le_bytes(),
            });
        }
        Ok(stores)
    }

    pub fn execute_normal_u32_vsti_to_ub(
        &self,
        word: u32,
        buffer_start: u64,
        ub: &mut UbMemory,
    ) -> Result<Vec<C310RvecVstiStore>, C310RvecValueError> {
        let stores = self.plan_normal_u32_vsti(word, buffer_start)?;
        let mut staged = ub.clone();
        for store in &stores {
            staged.write_states(
                store.buffer_address,
                &store.data.map(MemoryByteState::Known),
            )?;
        }
        *ub = staged;
        Ok(stores)
    }
}
