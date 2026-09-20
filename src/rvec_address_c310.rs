use serde::Serialize;
use thiserror::Error;

use crate::rvec::C310RvecValueMachine;

pub const C310_CAPTURED_VAG_WORD: u32 = 0xc200_001d;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct C310CapturedVagDescriptor {
    pub pc: u64,
    pub word: u32,
    pub vendor_isa_name: u16,
    pub destination_a_register: u8,
    pub source_s_registers: [u8; 4],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct C310CapturedA0Step {
    pub iteration_i1: u64,
    pub stride: u32,
    pub address_a0: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum C310RvecAddressError {
    #[error("C310 VAG word {word:#010x} at PC {pc:#x} is outside the captured path")]
    UnsupportedVagWord { pc: u64, word: u32 },
    #[error("C310 A0 update requires a configured VAG")]
    MissingVag,
    #[error("C310 A0 update requires scalar register S{index}")]
    MissingScalar { index: u8 },
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct C310RvecAddressState {
    vag: Option<C310CapturedVagDescriptor>,
    iteration_i1: Option<u64>,
    address_a0: Option<u32>,
}

impl C310RvecAddressState {
    pub fn vag(&self) -> Option<C310CapturedVagDescriptor> {
        self.vag
    }

    pub fn iteration_i1(&self) -> Option<u64> {
        self.iteration_i1
    }

    pub fn address_a0(&self) -> Option<u32> {
        self.address_a0
    }

    pub fn configure_vag_word(
        &mut self,
        pc: u64,
        word: u32,
    ) -> Result<C310CapturedVagDescriptor, C310RvecAddressError> {
        if word != C310_CAPTURED_VAG_WORD {
            return Err(C310RvecAddressError::UnsupportedVagWord { pc, word });
        }
        let descriptor = C310CapturedVagDescriptor {
            pc,
            word,
            vendor_isa_name: 280,
            destination_a_register: 0,
            source_s_registers: [2, 0, 0, 0],
        };
        self.vag = Some(descriptor);
        self.iteration_i1 = None;
        self.address_a0 = Some(0);
        Ok(descriptor)
    }

    pub fn start_vloop_i1(
        &mut self,
        scalar_registers: &C310RvecValueMachine,
    ) -> Result<C310CapturedA0Step, C310RvecAddressError> {
        self.update_i1(0, scalar_registers)
    }

    pub fn update_i1(
        &mut self,
        value: u64,
        scalar_registers: &C310RvecValueMachine,
    ) -> Result<C310CapturedA0Step, C310RvecAddressError> {
        self.vag.ok_or(C310RvecAddressError::MissingVag)?;
        let low = scalar_registers
            .scalar_register(2)
            .ok_or(C310RvecAddressError::MissingScalar { index: 2 })?;
        let high = scalar_registers
            .scalar_register(3)
            .ok_or(C310RvecAddressError::MissingScalar { index: 3 })?;
        let stride = low | high.wrapping_shl(16);
        let address_a0 = (value as u32).wrapping_mul(stride);
        self.iteration_i1 = Some(value);
        self.address_a0 = Some(address_a0);
        Ok(C310CapturedA0Step {
            iteration_i1: value,
            stride,
            address_a0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::predicate_buffer_c310::C310_PB_SLOT_BYTES;

    fn scalar_bank(stride: u32) -> C310RvecValueMachine {
        let mut machine = C310RvecValueMachine::from_vector_words(vec![vec![0; 64]]).unwrap();
        let mut slot = [0_u8; C310_PB_SLOT_BYTES];
        slot[..4].copy_from_slice(&(1_u32 << 1).to_le_bytes());
        slot[4..8].copy_from_slice(&stride.to_le_bytes());
        machine.apply_pb_scalar_init(&slot);
        machine
    }

    #[test]
    fn captured_vag_configures_a0_and_its_scalar_sources() {
        let mut state = C310RvecAddressState::default();
        let descriptor = state
            .configure_vag_word(0x10d0_d900, C310_CAPTURED_VAG_WORD)
            .unwrap();
        assert_eq!(descriptor.destination_a_register, 0);
        assert_eq!(descriptor.source_s_registers, [2, 0, 0, 0]);
        assert_eq!(state.address_a0(), Some(0));
        assert_eq!(state.iteration_i1(), None);
    }

    #[test]
    fn loop_start_and_iteration_update_recompute_a0() {
        let mut state = C310RvecAddressState::default();
        state
            .configure_vag_word(0x10d0_d900, C310_CAPTURED_VAG_WORD)
            .unwrap();
        let bank = scalar_bank(0x100);
        assert_eq!(state.start_vloop_i1(&bank).unwrap().address_a0, 0);
        assert_eq!(state.update_i1(3, &bank).unwrap().address_a0, 0x300);
        assert_eq!(state.iteration_i1(), Some(3));
        assert_eq!(
            state.update_i1(0x1_0000_0001, &bank).unwrap().address_a0,
            0x100
        );
    }

    #[test]
    fn stride_and_address_use_32_bit_wrapping_arithmetic() {
        let mut state = C310RvecAddressState::default();
        state
            .configure_vag_word(0x10d0_d900, C310_CAPTURED_VAG_WORD)
            .unwrap();
        let step = state.update_i1(2, &scalar_bank(u32::MAX)).unwrap();
        assert_eq!(step.stride, u32::MAX);
        assert_eq!(step.address_a0, u32::MAX - 1);
    }

    #[test]
    fn unsupported_word_and_missing_inputs_leave_state_unchanged() {
        let mut state = C310RvecAddressState::default();
        let bank = scalar_bank(0x100);
        assert_eq!(
            state.update_i1(1, &bank),
            Err(C310RvecAddressError::MissingVag)
        );
        assert!(matches!(
            state.configure_vag_word(0x10d0_d900, C310_CAPTURED_VAG_WORD ^ 1),
            Err(C310RvecAddressError::UnsupportedVagWord { .. })
        ));
        assert_eq!(state, C310RvecAddressState::default());
        state
            .configure_vag_word(0x10d0_d900, C310_CAPTURED_VAG_WORD)
            .unwrap();
        let empty = C310RvecValueMachine::from_vector_words(vec![vec![0; 64]]).unwrap();
        assert_eq!(
            state.update_i1(1, &empty),
            Err(C310RvecAddressError::MissingScalar { index: 2 })
        );
        assert_eq!(state.address_a0(), Some(0));
        assert_eq!(state.iteration_i1(), None);
    }
}
