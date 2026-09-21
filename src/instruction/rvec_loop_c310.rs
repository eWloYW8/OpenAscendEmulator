use serde::Serialize;
use thiserror::Error;

use crate::device::device_elf::{DeviceElfError, ProjectedDeviceKernel};
use crate::instruction::rvec::C310RvecValueMachine;
use crate::instruction::rvec_address_c310::{C310RvecAddressError, C310RvecAddressState};

pub const C310_CAPTURED_SHORT_VLOOP_WORD: u32 = 0xc280_0b1e;
pub const C310_CAPTURED_LONG_VLOOP_WORD: u32 = 0xc200_171e;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct C310CapturedLoopHeader {
    pub loop_pc: u64,
    pub loop_word: u32,
    pub source_s_register: u8,
    pub source_scalar_low: u32,
    pub source_scalar_high: u32,
    pub iteration_count: u16,
    pub body_instruction_count: u8,
    pub first_body_pc: u64,
    pub last_body_pc: u64,
    pub exit_pc: u64,
    pub initial_a0: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct C310CapturedLoopPcStep {
    pub executed_pc: u64,
    pub executed_word: u32,
    pub next_pc: u64,
    pub iteration_before: u16,
    pub iteration_after: u16,
    pub address_a0: u32,
    pub finished: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum C310CapturedLoopError {
    #[error("C310 VLOOP word {word:#010x} at PC {pc:#x} is outside the captured path")]
    UnsupportedWord { pc: u64, word: u32 },
    #[error("C310 VLOOP requires scalar register S{index}")]
    MissingScalar { index: u8 },
    #[error("C310 VLOOP at PC {pc:#x} has zero iterations")]
    ZeroIterations { pc: u64 },
    #[error("C310 VLOOP instruction address overflows")]
    AddressOverflow,
    #[error("C310 VLOOP controller already has an active loop")]
    ActiveLoop,
    #[error("C310 VLOOP controller has no active loop")]
    MissingLoop,
    #[error("C310 VLOOP expected body PC {expected:#x}, got {actual:#x}")]
    UnexpectedBodyPc { expected: u64, actual: u64 },
    #[error(transparent)]
    Object(#[from] DeviceElfError),
    #[error(transparent)]
    Address(#[from] C310RvecAddressError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ActiveLoop {
    header: C310CapturedLoopHeader,
    current_pc: u64,
    iteration: u16,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct C310RvecLoopController {
    active: Option<ActiveLoop>,
}

impl C310RvecLoopController {
    pub fn active_loop(&self) -> Option<C310CapturedLoopHeader> {
        self.active.map(|loop_state| loop_state.header)
    }

    pub fn current_pc(&self) -> Option<u64> {
        self.active.map(|loop_state| loop_state.current_pc)
    }

    pub fn start_from_object(
        &mut self,
        kernel: &ProjectedDeviceKernel<'_>,
        loop_pc: u64,
        scalar_registers: &C310RvecValueMachine,
        address_state: &mut C310RvecAddressState,
    ) -> Result<C310CapturedLoopHeader, C310CapturedLoopError> {
        let word = kernel.fetch_executable_word(loop_pc)?;
        let header = self.prepare_header(loop_pc, word, scalar_registers)?;
        for index in 0..=u64::from(header.body_instruction_count) {
            let pc = header
                .first_body_pc
                .checked_add(index * 4)
                .ok_or(C310CapturedLoopError::AddressOverflow)?;
            kernel.fetch_executable_word(pc)?;
        }
        self.commit_start(header, scalar_registers, address_state)
    }

    pub fn step_from_object(
        &mut self,
        kernel: &ProjectedDeviceKernel<'_>,
        scalar_registers: &C310RvecValueMachine,
        address_state: &mut C310RvecAddressState,
    ) -> Result<C310CapturedLoopPcStep, C310CapturedLoopError> {
        let pc = self
            .current_pc()
            .ok_or(C310CapturedLoopError::MissingLoop)?;
        let word = kernel.fetch_executable_word(pc)?;
        self.step_word(pc, word, scalar_registers, address_state)
    }

    fn prepare_header(
        &self,
        loop_pc: u64,
        word: u32,
        scalar_registers: &C310RvecValueMachine,
    ) -> Result<C310CapturedLoopHeader, C310CapturedLoopError> {
        if self.active.is_some() {
            return Err(C310CapturedLoopError::ActiveLoop);
        }
        if !matches!(
            word,
            C310_CAPTURED_SHORT_VLOOP_WORD | C310_CAPTURED_LONG_VLOOP_WORD
        ) {
            return Err(C310CapturedLoopError::UnsupportedWord { pc: loop_pc, word });
        }
        let source_s_register = ((word >> 23) & 0x1f) as u8;
        let low = scalar_registers
            .scalar_register(usize::from(source_s_register))
            .ok_or(C310CapturedLoopError::MissingScalar {
                index: source_s_register,
            })?;
        let high_index = source_s_register + 1;
        let high = scalar_registers
            .scalar_register(usize::from(high_index))
            .ok_or(C310CapturedLoopError::MissingScalar { index: high_index })?;
        let iteration_count = (low | high.wrapping_shl(16)) as u16;
        if iteration_count == 0 {
            return Err(C310CapturedLoopError::ZeroIterations { pc: loop_pc });
        }
        let body_instruction_count = ((word >> 10) & 0x3f) as u8;
        let first_body_pc = loop_pc
            .checked_add(4)
            .ok_or(C310CapturedLoopError::AddressOverflow)?;
        let last_body_pc = first_body_pc
            .checked_add(u64::from(body_instruction_count - 1) * 4)
            .ok_or(C310CapturedLoopError::AddressOverflow)?;
        let exit_pc = last_body_pc
            .checked_add(4)
            .ok_or(C310CapturedLoopError::AddressOverflow)?;
        Ok(C310CapturedLoopHeader {
            loop_pc,
            loop_word: word,
            source_s_register,
            source_scalar_low: low,
            source_scalar_high: high,
            iteration_count,
            body_instruction_count,
            first_body_pc,
            last_body_pc,
            exit_pc,
            initial_a0: 0,
        })
    }

    fn commit_start(
        &mut self,
        mut header: C310CapturedLoopHeader,
        scalar_registers: &C310RvecValueMachine,
        address_state: &mut C310RvecAddressState,
    ) -> Result<C310CapturedLoopHeader, C310CapturedLoopError> {
        let mut next_address = *address_state;
        header.initial_a0 = next_address.start_vloop_i1(scalar_registers)?.address_a0;
        *address_state = next_address;
        self.active = Some(ActiveLoop {
            current_pc: header.first_body_pc,
            header,
            iteration: 0,
        });
        Ok(header)
    }

    fn step_word(
        &mut self,
        pc: u64,
        word: u32,
        scalar_registers: &C310RvecValueMachine,
        address_state: &mut C310RvecAddressState,
    ) -> Result<C310CapturedLoopPcStep, C310CapturedLoopError> {
        let mut active = self.active.ok_or(C310CapturedLoopError::MissingLoop)?;
        if pc != active.current_pc {
            return Err(C310CapturedLoopError::UnexpectedBodyPc {
                expected: active.current_pc,
                actual: pc,
            });
        }
        let iteration_before = active.iteration;
        let (next_pc, finished) = if pc != active.header.last_body_pc {
            (pc + 4, false)
        } else if active.iteration + 1 < active.header.iteration_count {
            let next_iteration = active.iteration + 1;
            let mut next_address = *address_state;
            next_address.update_i1(u64::from(next_iteration), scalar_registers)?;
            *address_state = next_address;
            active.iteration = next_iteration;
            (active.header.first_body_pc, false)
        } else {
            (active.header.exit_pc, true)
        };
        let address_a0 = address_state
            .address_a0()
            .ok_or(C310RvecAddressError::MissingVag)?;
        let iteration_after = active.iteration;
        if finished {
            self.active = None;
        } else {
            active.current_pc = next_pc;
            self.active = Some(active);
        }
        Ok(C310CapturedLoopPcStep {
            executed_pc: pc,
            executed_word: word,
            next_pc,
            iteration_before,
            iteration_after,
            address_a0,
            finished,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::predicate_buffer_c310::C310_PB_SLOT_BYTES;

    fn scalar_bank(count: u16, source_s_register: u8) -> C310RvecValueMachine {
        let mut machine = C310RvecValueMachine::from_vector_words(vec![vec![0; 64]]).unwrap();
        let mut slot = [0_u8; C310_PB_SLOT_BYTES];
        let flags = (1_u32 << 1) | (1_u32 << 2) | (1_u32 << 3);
        slot[..4].copy_from_slice(&flags.to_le_bytes());
        slot[4..8].copy_from_slice(&0x100_u32.to_le_bytes());
        if source_s_register == 4 {
            slot[8..12].copy_from_slice(&u32::from(count).to_le_bytes());
        } else {
            slot[8..12].copy_from_slice(&(u32::from(count) << 16).to_le_bytes());
        }
        machine.apply_pb_scalar_init(&slot);
        machine
    }

    #[test]
    fn short_loop_returns_to_body_once_then_exits() {
        let machine = scalar_bank(2, 5);
        let mut address = C310RvecAddressState::default();
        address
            .configure_vag_word(
                0x10d0_d900,
                crate::instruction::rvec_address_c310::C310_CAPTURED_VAG_WORD,
            )
            .unwrap();
        let mut controller = C310RvecLoopController::default();
        let header = controller
            .prepare_header(0x10d0_d910, C310_CAPTURED_SHORT_VLOOP_WORD, &machine)
            .unwrap();
        assert_eq!(header.source_s_register, 5);
        assert_eq!(header.iteration_count, 2);
        assert_eq!(header.body_instruction_count, 2);
        assert_eq!(
            (header.first_body_pc, header.last_body_pc, header.exit_pc),
            (0x10d0_d914, 0x10d0_d918, 0x10d0_d91c)
        );
        controller
            .commit_start(header, &machine, &mut address)
            .unwrap();
        let active_before = controller.clone();
        let address_before = address;
        assert_eq!(
            controller.step_word(0x10d0_d918, 0, &machine, &mut address),
            Err(C310CapturedLoopError::UnexpectedBodyPc {
                expected: 0x10d0_d914,
                actual: 0x10d0_d918,
            })
        );
        assert_eq!(controller, active_before);
        assert_eq!(address, address_before);
        let sequence = [
            (0x10d0_d914, 0x10d0_d918, 0),
            (0x10d0_d918, 0x10d0_d914, 0x100),
            (0x10d0_d914, 0x10d0_d918, 0x100),
            (0x10d0_d918, 0x10d0_d91c, 0x100),
        ];
        for (pc, next, a0) in sequence {
            let step = controller
                .step_word(pc, 0x1234_5678, &machine, &mut address)
                .unwrap();
            assert_eq!(step.next_pc, next);
            assert_eq!(step.address_a0, a0);
        }
        assert_eq!(controller.current_pc(), None);
        assert_eq!(address.iteration_i1(), Some(1));
    }

    #[test]
    fn long_loop_exits_after_five_body_words_at_one_iteration() {
        let machine = scalar_bank(1, 4);
        let mut address = C310RvecAddressState::default();
        address
            .configure_vag_word(
                0x10d0_d900,
                crate::instruction::rvec_address_c310::C310_CAPTURED_VAG_WORD,
            )
            .unwrap();
        let mut controller = C310RvecLoopController::default();
        let header = controller
            .prepare_header(0x10d0_d908, C310_CAPTURED_LONG_VLOOP_WORD, &machine)
            .unwrap();
        assert_eq!(header.source_s_register, 4);
        assert_eq!(header.body_instruction_count, 5);
        assert_eq!(header.iteration_count, 1);
        assert_eq!(
            (header.first_body_pc, header.last_body_pc, header.exit_pc),
            (0x10d0_d90c, 0x10d0_d91c, 0x10d0_d920)
        );
        controller
            .commit_start(header, &machine, &mut address)
            .unwrap();
        for pc in (0x10d0_d90c..=0x10d0_d91c).step_by(4) {
            let step = controller.step_word(pc, 0, &machine, &mut address).unwrap();
            assert_eq!(step.next_pc, pc + 4);
            assert_eq!(step.address_a0, 0);
        }
        assert_eq!(controller.current_pc(), None);
        assert_eq!(address.iteration_i1(), Some(0));
    }

    #[test]
    fn invalid_start_does_not_change_address_or_loop_state() {
        let machine = scalar_bank(0, 5);
        let mut address = C310RvecAddressState::default();
        address
            .configure_vag_word(
                0x10d0_d900,
                crate::instruction::rvec_address_c310::C310_CAPTURED_VAG_WORD,
            )
            .unwrap();
        let controller = C310RvecLoopController::default();
        let before = address;
        assert_eq!(
            controller.prepare_header(0x10d0_d910, C310_CAPTURED_SHORT_VLOOP_WORD, &machine),
            Err(C310CapturedLoopError::ZeroIterations { pc: 0x10d0_d910 })
        );
        assert_eq!(address, before);
        assert_eq!(controller.active_loop(), None);
    }

    #[test]
    fn missing_vag_cannot_commit_a_loop() {
        let machine = scalar_bank(2, 5);
        let mut address = C310RvecAddressState::default();
        let mut controller = C310RvecLoopController::default();
        let header = controller
            .prepare_header(0x10d0_d910, C310_CAPTURED_SHORT_VLOOP_WORD, &machine)
            .unwrap();
        assert_eq!(
            controller.commit_start(header, &machine, &mut address),
            Err(C310CapturedLoopError::Address(
                C310RvecAddressError::MissingVag
            ))
        );
        assert_eq!(controller, C310RvecLoopController::default());
        assert_eq!(address, C310RvecAddressState::default());
    }
}
