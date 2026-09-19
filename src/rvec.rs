
use crate::fp32_vector::{
    Fp32LaneOutcome, Fp32MaskLayout, Fp32VectorError, Fp32VectorOperation, Fp32WritebackOutcome,
    Fp32WritebackPolicy, apply_fp32_writeback, evaluate_masked_fp32_lanes,
};
use serde::Serialize;
use thiserror::Error;

const MAX_ENCODED_V_REGISTERS: usize = 32;
const MAX_ENCODED_P_REGISTERS: usize = 32;
const MAX_FP32_WORDS_PER_REGISTER: usize = 64;
const MAX_PREDICATE_BYTES: usize = 32;

pub const C310_MASK0_SPR_INDEX: u16 = 152;
pub const C310_MASK1_SPR_INDEX: u16 = 153;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct C310ObservedMovemaskHint {
    pub vendor_isa_name: u16,
    pub source_x_register: u8,
    pub destination_spr: u16,
}

impl C310ObservedMovemaskHint {
    pub const fn from_word(word: u32) -> Option<Self> {
        let (source_x_register, destination_spr) = match word {
            0x15c0_0033 => (0, C310_MASK1_SPR_INDEX),
            0x15c0_0013 => (0, C310_MASK0_SPR_INDEX),
            0x15c3_0033 => (3, C310_MASK1_SPR_INDEX),
            0x15cd_0013 => (13, C310_MASK0_SPR_INDEX),
            _ => return None,
        };
        Some(Self {
            vendor_isa_name: 180,
            source_x_register,
            destination_spr,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct C310RvecMovpHint {
    pub vendor_isa_name: u16,
    pub destination_p_register: u8,
    pub btype: u8,
}

impl C310RvecMovpHint {
    pub const fn from_word(word: u32) -> Option<Self> {
        if word >> 30 != 2
            || (word >> 20) & 0x1f != 0
            || (word >> 9) & 0x7ff != 0x200
            || word & 0x7f != 0x56
        {
            return None;
        }
        Some(Self {
            vendor_isa_name: 374,
            destination_p_register: ((word >> 25) & 0x1f) as u8,
            btype: ((word >> 7) & 3) as u8,
        })
    }

    pub const fn has_u32_value_path(self) -> bool {
        self.btype == 2
    }

    pub const fn u32_source_spr_index(self) -> Option<u16> {
        if self.has_u32_value_path() {
            Some(C310_MASK0_SPR_INDEX)
        } else {
            None
        }
    }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct C310RvecVstiHint {
    pub vendor_isa_name: u16,
    pub source_v_register: u8,
    pub scalar_register: u8,
    pub offset: u8,
    pub predicate_register: u8,
    pub p: bool,
    pub distance: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct C310RvecVstiStore {
    pub lane_index: usize,
    pub buffer_address: u64,
    pub data: [u8; 4],
}

impl C310RvecVstiHint {
    pub const fn from_word(word: u32) -> Option<Self> {
        if word >> 30 != 1 || (word >> 6) & 1 != 0 || word & 3 != 2 {
            return None;
        }
        Some(Self {
            vendor_isa_name: 299,
            source_v_register: ((word >> 25) & 0x1f) as u8,
            scalar_register: ((word >> 19) & 0x3f) as u8,
            offset: ((word >> 11) & 0xff) as u8,
            predicate_register: ((word >> 8) & 7) as u8,
            p: (word >> 7) & 1 != 0,
            distance: ((word >> 2) & 0xf) as u8,
        })
    }

    pub const fn dtype_code(self) -> u8 {
        self.distance + 24
    }

    pub const fn has_normal_u32_store_path(self) -> bool {
        self.dtype_code() == 26
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum C310RvecArithmeticOperation {
    Add,
    Subtract,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct C310RvecArithmeticHint {
    pub operation: C310RvecArithmeticOperation,
    pub vendor_isa_name: u16,
    pub destination_v_register: u8,
    pub first_source_v_register: u8,
    pub second_source_v_register: u8,
    pub predicate_register: u8,
    pub dtype_selector: u8,
}

impl C310RvecArithmeticHint {
    pub const fn from_word(word: u32) -> Option<Self> {
        if (word >> 30) != 2 || (word >> 19) & 1 != 1 || (word >> 6) & 1 != 0 {
            return None;
        }
        let (operation, vendor_isa_name) = match word & 0x3f {
            0 => (C310RvecArithmeticOperation::Add, 319),
            1 => (C310RvecArithmeticOperation::Subtract, 320),
            _ => return None,
        };
        Some(Self {
            operation,
            vendor_isa_name,
            destination_v_register: ((word >> 25) & 0x1f) as u8,
            first_source_v_register: ((word >> 20) & 0x1f) as u8,
            second_source_v_register: ((word >> 13) & 0x1f) as u8,
            predicate_register: ((word >> 10) & 7) as u8,
            dtype_selector: ((((word >> 18) & 1) << 3) | ((word >> 7) & 7)) as u8,
        })
    }

    pub const fn has_fp32_value_path(self) -> bool {
        self.dtype_selector == 7
    }

    pub fn evaluate_fp32_lanes(
        self,
        first: &[u32],
        second: &[u32],
        active_mask: &[u64; 4],
    ) -> Result<Vec<Fp32LaneOutcome>, Fp32VectorError> {
        if !self.has_fp32_value_path() {
            return Err(Fp32VectorError::UnsupportedInstruction);
        }
        let operation = match self.operation {
            C310RvecArithmeticOperation::Add => Fp32VectorOperation::Add,
            C310RvecArithmeticOperation::Subtract => Fp32VectorOperation::Subtract,
        };
        evaluate_masked_fp32_lanes(
            operation,
            Fp32MaskLayout::C310ByteStart,
            first,
            second,
            active_mask,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct C310RvecValueMachine {
    vector_registers: Vec<Vec<u32>>,
    predicate_registers: Option<Vec<Vec<u8>>>,
    words_per_register: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct C310RvecValueStep {
    pub hint: C310RvecArithmeticHint,
    pub active_mask: [u64; 4],
    pub first_source: Vec<u32>,
    pub second_source: Vec<u32>,
    pub lanes: Vec<Fp32LaneOutcome>,
    pub writeback: Fp32WritebackOutcome,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct C310RvecMovpStep {
    pub hint: C310RvecMovpHint,
    pub scalar_mask: u64,
    pub predicate_bytes: [u8; MAX_PREDICATE_BYTES],
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
        let lanes = hint.evaluate_fp32_lanes(&first_source, &second_source, active_mask)?;
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Architecture, ScalarMachine};

    #[test]
    fn observed_c310_movemask_words_transfer_live_source_x_values_to_mask_sprs() {
        let mut xregs = [0_u64; 32];
        xregs[0] = u64::MAX;
        xregs[13] = 0x5555_5555;
        let mut sprs = C310RvecMaskSprState::default();
        for (pc, word, source, destination, value) in [
            (0x10d0_d130, 0x15c0_0033, 0, 153, u64::MAX),
            (0x10d0_d134, 0x15c0_0013, 0, 152, u64::MAX),
            (0x10d0_d54c, 0x15c3_0033, 3, 153, 0),
            (0x10d0_d55c, 0x15cd_0013, 13, 152, 0x5555_5555),
        ] {
            let step = sprs
                .execute_observed_movemask_word(pc, word, &xregs)
                .unwrap();
            assert_eq!(step.hint.vendor_isa_name, 180);
            assert_eq!(step.hint.source_x_register, source);
            assert_eq!(step.hint.destination_spr, destination);
            assert_eq!(step.value, value);
        }
        assert_eq!(sprs.mask0, 0x5555_5555);
        assert_eq!(sprs.mask1, 0);
        let before = sprs;
        assert_eq!(
            sprs.execute_observed_movemask_word(0x10d0_d55c, 0x15ce_0013, &xregs),
            Err(C310ObservedMovemaskError::UnsupportedWord {
                pc: 0x10d0_d55c,
                word: 0x15ce_0013,
            })
        );
        assert_eq!(sprs, before);
    }

    #[test]
    fn captured_scalar_movemask_movp_chain_matches_live_p1_readback() {
        let mut scalar = ScalarMachine::new(Architecture::Dav3510, [0; 32], 0);
        let mut mask_sprs = C310RvecMaskSprState::default();
        scalar.execute_word(0x10d0_d548, 0x071a_5555).unwrap();
        assert_eq!(scalar.xregs()[13], 0x5555);
        mask_sprs
            .execute_observed_movemask_word(0x10d0_d54c, 0x15c3_0033, scalar.xregs())
            .unwrap();
        scalar.execute_word(0x10d0_d550, 0x075b_5555).unwrap();
        assert_eq!(scalar.xregs()[13], 0x5555_5555);
        mask_sprs
            .execute_observed_movemask_word(0x10d0_d55c, 0x15cd_0013, scalar.xregs())
            .unwrap();
        assert_eq!(mask_sprs.mask0, 0x5555_5555);
        assert_eq!(mask_sprs.mask1, 0);
        let mut rvec = C310RvecValueMachine::from_vector_and_predicate_bytes(
            vec![vec![0; 32], vec![0; 32]],
            vec![vec![0; 32], vec![0; 32]],
        )
        .unwrap();
        let movp = rvec
            .execute_movp_u32_from_mask_sprs(0x8204_0156, &mask_sprs)
            .unwrap();
        assert_eq!(movp.scalar_mask, 0x5555_5555);
        assert_eq!(&movp.predicate_bytes[..16], &[0x0f; 16]);
        assert_eq!(&movp.predicate_bytes[16..], &[0; 16]);
    }

    #[test]
    fn captured_vsti_plans_only_predicated_four_byte_buffer_writes() {
        let mut predicate = vec![0_u8; 32];
        predicate[..16].fill(0x0f);
        let machine = C310RvecValueMachine::from_vector_and_predicate_bytes(
            vec![(0..64_u32).collect()],
            vec![vec![0; 32], predicate],
        )
        .unwrap();
        let word = 0x4018_010a;
        let stores = machine.plan_normal_u32_vsti(word, 0x100).unwrap();
        assert_eq!(stores.len(), 16);
        assert_eq!(
            stores[0],
            C310RvecVstiStore {
                lane_index: 0,
                buffer_address: 0x100,
                data: 0_u32.to_le_bytes(),
            }
        );
        assert_eq!(stores[1].lane_index, 2);
        assert_eq!(stores[1].buffer_address, 0x108);
        assert_eq!(stores[1].data, 2_u32.to_le_bytes());
        assert_eq!(stores[15].lane_index, 30);
        assert_eq!(stores[15].buffer_address, 0x178);
        assert_eq!(stores[15].data, 30_u32.to_le_bytes());
        assert_eq!(
            machine.plan_normal_u32_vsti(word ^ 3, 0x100),
            Err(C310RvecValueError::UnsupportedVstiWord)
        );
        assert_eq!(
            machine.plan_normal_u32_vsti(word ^ (1 << 2), 0x100),
            Err(C310RvecValueError::UnsupportedVstiDtype)
        );
        assert_eq!(
            machine.plan_normal_u32_vsti(word, u64::MAX - 3),
            Err(C310RvecValueError::VstiAddressOverflow {
                base: u64::MAX - 3,
                lane: 2,
            })
        );
    }

    #[test]
    fn captured_movp_word_selects_p1_and_u32_value_path() {
        let hint = C310RvecMovpHint::from_word(0x8204_0156).unwrap();
        assert_eq!(hint.vendor_isa_name, 374);
        assert_eq!(hint.destination_p_register, 1);
        assert_eq!(hint.btype, 2);
        assert!(hint.has_u32_value_path());
        assert_eq!(hint.u32_source_spr_index(), Some(C310_MASK0_SPR_INDEX));
        assert_eq!(C310_MASK1_SPR_INDEX, 153);
        let word = 0x8204_0156;
        for wrong in [
            word ^ (1 << 30),
            word ^ (1 << 20),
            word ^ (1 << 9),
            word ^ 1,
        ] {
            assert_eq!(C310RvecMovpHint::from_word(wrong), None);
        }
        let other_btype = C310RvecMovpHint::from_word(word ^ (1 << 7)).unwrap();
        assert!(!other_btype.has_u32_value_path());
        assert_eq!(other_btype.u32_source_spr_index(), None);
    }

    #[test]
    fn movp_reads_caller_mask_spr_snapshot_with_vendor_defaults() {
        let mut machine = C310RvecValueMachine::from_vector_and_predicate_bytes(
            vec![vec![0; 4]],
            vec![vec![0; 32], vec![0; 32]],
        )
        .unwrap();
        let mut mask_sprs = C310RvecMaskSprState::default();
        assert_eq!(mask_sprs.mask0, u64::MAX);
        assert_eq!(mask_sprs.mask1, u64::MAX);
        let default = machine
            .execute_movp_u32_from_mask_sprs(0x8204_0156, &mask_sprs)
            .unwrap();
        assert_eq!(default.predicate_bytes, [0xff; 32]);

        mask_sprs.mask0 = 0x5555_5555;
        let overridden = machine
            .execute_movp_u32_from_mask_sprs(0x8204_0156, &mask_sprs)
            .unwrap();
        assert_eq!(&overridden.predicate_bytes[..16], &[0x0f; 16]);
        assert_eq!(&overridden.predicate_bytes[16..], &[0; 16]);
        assert_eq!(
            machine.predicate_register(1),
            Some(&overridden.predicate_bytes[..])
        );
    }

    #[test]
    fn movp_updates_encoded_p_destination_before_vadd_and_vsti_store() {
        let old = (-123.0_f32).to_bits();
        let mut machine = C310RvecValueMachine::from_vector_and_predicate_bytes(
            vec![
                vec![
                    1.0_f32.to_bits(),
                    2.0_f32.to_bits(),
                    3.0_f32.to_bits(),
                    4.0_f32.to_bits(),
                ],
                vec![
                    5.0_f32.to_bits(),
                    6.0_f32.to_bits(),
                    7.0_f32.to_bits(),
                    8.0_f32.to_bits(),
                ],
            ],
            vec![vec![0; 32], vec![0; 32]],
        )
        .unwrap();
        let movp = machine.execute_movp_u32_word(0x8204_0156, 0b0101).unwrap();
        assert_eq!(movp.hint.destination_p_register, 1);
        assert_eq!(&movp.predicate_bytes[..2], &[0x0f, 0x0f]);
        assert_eq!(
            machine.predicate_register(1),
            Some(movp.predicate_bytes.as_slice())
        );
        let add = machine
            .execute_fp32_word_from_predicate_registers(0x8008_2780)
            .unwrap();
        assert_eq!(
            add.writeback.words,
            [6.0_f32.to_bits(), 0, 10.0_f32.to_bits(), 0]
        );
        let store_hint = C310RvecVstiHint::from_word(0x4018_010a).unwrap();
        assert!(store_hint.has_normal_u32_store_path());
        assert_eq!(
            store_hint.predicate_register,
            movp.hint.destination_p_register
        );
        let store = c310_normal_u32_masked_store(
            &[old; 4],
            &add.writeback.words,
            machine
                .predicate_register(usize::from(store_hint.predicate_register))
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            store.words,
            [6.0_f32.to_bits(), old, 10.0_f32.to_bits(), old]
        );
    }

    #[test]
    fn movp_rejects_missing_or_short_predicate_bank_without_mutation() {
        let mut missing = C310RvecValueMachine::from_vector_words(vec![vec![0]]).unwrap();
        assert_eq!(
            missing.execute_movp_u32_word(0x8204_0156, 1),
            Err(C310RvecValueError::MissingPredicateBank)
        );
        let mut short = C310RvecValueMachine::from_vector_and_predicate_bytes(
            vec![vec![0]],
            vec![vec![0; 32], vec![0; 16]],
        )
        .unwrap();
        assert_eq!(
            short.execute_movp_u32_word(0x8204_0156, 1),
            Err(C310RvecValueError::MovpPredicateWidth {
                expected: 32,
                actual: 16,
            })
        );
        assert_eq!(short.predicate_register(1), Some([0; 16].as_slice()));
    }

    #[test]
    fn captured_vsti_word_decodes_to_normal_u32_store_path() {
        let hint = C310RvecVstiHint::from_word(0x4018_010a).unwrap();
        assert_eq!(hint.vendor_isa_name, 299);
        assert_eq!(hint.source_v_register, 0);
        assert_eq!(hint.scalar_register, 3);
        assert_eq!(hint.offset, 0);
        assert_eq!(hint.predicate_register, 1);
        assert!(!hint.p);
        assert_eq!(hint.distance, 2);
        assert_eq!(hint.dtype_code(), 26);
        assert!(hint.has_normal_u32_store_path());
    }

    #[test]
    fn vsti_decoder_rejects_neighboring_fixed_leaves() {
        let word = 0x4018_010a;
        for wrong in [word ^ (1 << 30), word ^ (1 << 6), word ^ 1] {
            assert_eq!(C310RvecVstiHint::from_word(wrong), None);
        }
        let other_dist = C310RvecVstiHint::from_word(word ^ (1 << 2)).unwrap();
        assert_eq!(other_dist.dtype_code(), 27);
        assert!(!other_dist.has_normal_u32_store_path());
    }

    #[test]
    fn add_and_sub_vendor_trace_words_select_distinct_registered_leaves() {
        let add = C310RvecArithmeticHint::from_word(0x8008_2780).unwrap();
        let subtract = C310RvecArithmeticHint::from_word(0x8008_2781).unwrap();
        assert_eq!(add.operation, C310RvecArithmeticOperation::Add);
        assert_eq!(subtract.operation, C310RvecArithmeticOperation::Subtract);
        assert_eq!(add.vendor_isa_name, 319);
        assert_eq!(subtract.vendor_isa_name, 320);
        assert_eq!(add.destination_v_register, 0);
        assert_eq!(add.first_source_v_register, 0);
        assert_eq!(add.second_source_v_register, 1);
        assert_eq!(add.predicate_register, 1);
        assert_eq!(add.dtype_selector, 7);
        assert!(add.has_fp32_value_path());
        assert!(subtract.has_fp32_value_path());
    }

    #[test]
    fn only_registered_fixed_opcode_fields_are_accepted() {
        let add = 0x8008_2780;
        for wrong in [add ^ (1 << 30), add ^ (1 << 19), add ^ (1 << 6), add | 2] {
            assert_eq!(C310RvecArithmeticHint::from_word(wrong), None);
        }
    }

    #[test]
    fn variable_register_and_dtype_fields_do_not_change_opcode_leaf() {
        let word = 0x8008_2780 | (3 << 25) | (4 << 20) | (5 << 13) | (2 << 10) | (1 << 18);
        let hint = C310RvecArithmeticHint::from_word(word).unwrap();
        assert_eq!(hint.operation, C310RvecArithmeticOperation::Add);
        assert_eq!(hint.destination_v_register, 3);
        assert_eq!(hint.first_source_v_register, 4);
        assert_eq!(hint.second_source_v_register, 5);
        assert_eq!(hint.predicate_register, 3);
        assert_eq!(hint.dtype_selector, 15);
        assert!(!hint.has_fp32_value_path());
        assert_eq!(
            hint.evaluate_fp32_lanes(&[], &[], &[0; 4]),
            Err(Fp32VectorError::UnsupportedInstruction)
        );
    }

    #[test]
    fn captured_fp32_words_reach_the_masked_value_stage() {
        let first = [1.0_f32.to_bits(), 2.0_f32.to_bits()];
        let second = [3.0_f32.to_bits(), 4.0_f32.to_bits()];
        let mask = [1, 0, 0, 0];
        let add = C310RvecArithmeticHint::from_word(0x8008_2780).unwrap();
        let sub = C310RvecArithmeticHint::from_word(0x8008_2781).unwrap();
        let added = add.evaluate_fp32_lanes(&first, &second, &mask).unwrap();
        let subtracted = sub.evaluate_fp32_lanes(&first, &second, &mask).unwrap();
        assert_eq!(added[0].bits, 4.0_f32.to_bits());
        assert_eq!(subtracted[0].bits, (-2.0_f32).to_bits());
        assert_eq!(added[1].bits, 0);
        assert_eq!(subtracted[1].bits, 0);
    }

    #[test]
    fn predicate_bytes_fill_the_256_bit_mask_low_bit_first() {
        let mut bytes = [0_u8; 32];
        bytes[0] = 0b0001_0001;
        bytes[1] = 0b1000_0000;
        bytes[8] = 1;
        bytes[31] = 0b1000_0000;
        assert_eq!(
            c310_predicate_bytes_to_mask(&bytes).unwrap(),
            [0x8011, 1, 0, 1_u64 << 63]
        );
        assert_eq!(c310_predicate_bytes_to_mask(&[]).unwrap(), [0; 4]);
        assert_eq!(
            c310_predicate_bytes_to_mask(&[0; 33]),
            Err(C310RvecValueError::PredicateWidth { bytes: 33 })
        );
    }

    #[test]
    fn predicate_bank_selects_encoded_p_register_before_add() {
        let mut machine = C310RvecValueMachine::from_vector_and_predicate_bytes(
            vec![
                vec![1.0_f32.to_bits(), 2.0_f32.to_bits()],
                vec![3.0_f32.to_bits(), 4.0_f32.to_bits()],
            ],
            vec![vec![0; 32], vec![0x01, 0, 0, 0]],
        )
        .unwrap();
        assert_eq!(machine.predicate_register(1), Some(&[1, 0, 0, 0][..]));
        let step = machine
            .execute_fp32_word_from_predicate_registers(0x8008_2780)
            .unwrap();
        assert_eq!(step.hint.predicate_register, 1);
        assert_eq!(step.active_mask, [1, 0, 0, 0]);
        assert_eq!(step.writeback.words, [4.0_f32.to_bits(), 0]);

        let masked_by_p0 = machine
            .execute_fp32_word_from_predicate_registers(0x8008_2380)
            .unwrap();
        assert_eq!(masked_by_p0.hint.predicate_register, 0);
        assert_eq!(masked_by_p0.active_mask, [0; 4]);
        assert_eq!(masked_by_p0.writeback.words, [0, 0]);
    }

    #[test]
    fn register_value_machine_handles_aliasing_and_two_dependent_words() {
        let x = vec![1.0_f32.to_bits(), 2.0_f32.to_bits()];
        let y = vec![3.0_f32.to_bits(), 4.0_f32.to_bits()];
        let mut machine = C310RvecValueMachine::from_vector_words(vec![x.clone(), y]).unwrap();
        let mask = [0x11, 0, 0, 0];
        let add = machine.execute_fp32_word(0x8008_2780, &mask).unwrap();
        assert_eq!(add.first_source, x);
        assert_eq!(add.writeback.words, [4.0_f32.to_bits(), 6.0_f32.to_bits()]);
        let sub = machine.execute_fp32_word(0x8008_2781, &mask).unwrap();
        assert_eq!(sub.first_source, add.writeback.words);
        assert_eq!(machine.vector_register(0).unwrap(), x);
        assert_eq!(
            machine.vector_register(1).unwrap(),
            [3.0_f32.to_bits(), 4.0_f32.to_bits()]
        );
        assert_eq!(machine.words_per_register(), 2);
    }

    #[test]
    fn register_value_machine_zeroes_inactive_lanes_and_rejects_unsupported_words() {
        let mut machine = C310RvecValueMachine::from_vector_words(vec![
            vec![1.0_f32.to_bits(), 2.0_f32.to_bits()],
            vec![3.0_f32.to_bits(), 4.0_f32.to_bits()],
        ])
        .unwrap();
        let step = machine
            .execute_fp32_word(0x8008_2780, &[1, 0, 0, 0])
            .unwrap();
        assert_eq!(step.writeback.words, [4.0_f32.to_bits(), 0]);
        assert_eq!(step.writeback.written, [true, true]);
        let before = machine.clone();
        assert_eq!(
            machine.execute_fp32_word(0x8008_2780 | (1 << 18), &[0; 4]),
            Err(C310RvecValueError::UnsupportedDtype)
        );
        assert_eq!(
            machine.execute_fp32_word(0, &[0; 4]),
            Err(C310RvecValueError::UnsupportedWord)
        );
        assert_eq!(machine, before);
    }

    #[test]
    fn register_bank_shape_and_indices_fail_closed() {
        assert_eq!(
            C310RvecValueMachine::from_vector_words(vec![]),
            Err(C310RvecValueError::RegisterCount { count: 0 })
        );
        assert_eq!(
            C310RvecValueMachine::from_vector_words(vec![vec![]]),
            Err(C310RvecValueError::RegisterWidth { words: 0 })
        );
        assert_eq!(
            C310RvecValueMachine::from_vector_words(vec![vec![0], vec![0, 0]]),
            Err(C310RvecValueError::UnequalRegisterWidth {
                index: 1,
                actual: 2,
                expected: 1
            })
        );
        let mut machine = C310RvecValueMachine::from_vector_words(vec![vec![0]]).unwrap();
        assert_eq!(
            machine.execute_fp32_word(0x8008_2780, &[1, 0, 0, 0]),
            Err(C310RvecValueError::RegisterIndex { index: 1, count: 1 })
        );
        assert_eq!(
            machine.execute_fp32_word_from_predicate_registers(0x8008_2780),
            Err(C310RvecValueError::MissingPredicateBank)
        );
        assert_eq!(
            C310RvecValueMachine::from_vector_and_predicate_bytes(vec![vec![0]], vec![]),
            Err(C310RvecValueError::PredicateRegisterCount { count: 0 })
        );
        assert_eq!(
            C310RvecValueMachine::from_vector_and_predicate_bytes(vec![vec![0]], vec![vec![0; 33]]),
            Err(C310RvecValueError::PredicateWidth { bytes: 33 })
        );
        let mut short_p_bank = C310RvecValueMachine::from_vector_and_predicate_bytes(
            vec![vec![0], vec![0]],
            vec![vec![0; 32]],
        )
        .unwrap();
        assert_eq!(
            short_p_bank.execute_fp32_word_from_predicate_registers(0x8008_2780),
            Err(C310RvecValueError::PredicateRegisterIndex { index: 1, count: 1 })
        );
    }

    #[test]
    fn movp_expands_each_scalar_bit_to_one_four_bit_lane_predicate() {
        let even = c310_movp_u32_mask_to_predicate_bytes(0x5555_5555);
        assert_eq!(&even[..16], &[0x0f; 16]);
        assert_eq!(&even[16..], &[0; 16]);
        let odd = c310_movp_u32_mask_to_predicate_bytes(0xaaaa_aaaa);
        assert_eq!(&odd[..16], &[0xf0; 16]);
        assert_eq!(&odd[16..], &[0; 16]);
        assert_eq!(c310_movp_u32_mask_to_predicate_bytes(u64::MAX), [0xff; 32]);
    }

    #[test]
    fn normal_u32_masked_store_preserves_inactive_memory_after_zeroed_vadd_lanes() {
        let predicate = c310_movp_u32_mask_to_predicate_bytes(0b0101);
        let previous = [(-123.0_f32).to_bits(); 5];
        let source = [4.0_f32.to_bits(), 0, 8.0_f32.to_bits(), 0];
        let stored = c310_normal_u32_masked_store(&previous, &source, &predicate).unwrap();
        assert_eq!(
            stored.words,
            [source[0], previous[1], source[2], previous[3], previous[4]]
        );
        assert_eq!(stored.written, [true, false, true, false, false]);
        assert_eq!(
            c310_normal_u32_masked_store(&previous[..2], &source, &predicate),
            Err(Fp32VectorError::DestinationTooSmall {
                destination: 2,
                results: 4
            }
            .into())
        );
        assert_eq!(
            c310_normal_u32_masked_store(&previous, &source, &[0; 33]),
            Err(C310RvecValueError::PredicateWidth { bytes: 33 })
        );
        assert_eq!(
            c310_normal_u32_masked_store(&previous, &[0; 65], &predicate),
            Err(Fp32VectorError::TooManyLanes { lanes: 65 }.into())
        );
    }
}
