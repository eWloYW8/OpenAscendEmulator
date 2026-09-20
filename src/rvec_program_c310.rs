use serde::Serialize;
use thiserror::Error;

use crate::device_elf::{DeviceElfError, ProjectedDeviceKernel};
use crate::rvec::{
    C310_CAPTURED_PLT32_WORD, C310_CAPTURED_PSET_WORD, C310_CAPTURED_SMOVI32_WORD,
    C310_CAPTURED_SUB_VST_WORD, C310_CAPTURED_VDUPS_WORD, C310_CAPTURED_VLD_V0_WORD,
    C310_CAPTURED_VLD_V1_WORD, C310_CAPTURED_VLDI_V0_WORD, C310_CAPTURED_VLDI_V1_WORD,
    C310_CAPTURED_VST_WORD, C310CapturedPltError, C310CapturedPltStep, C310CapturedPsetError,
    C310CapturedPsetStep, C310CapturedSmoviError, C310CapturedSmoviStep, C310CapturedVdupsStep,
    C310CapturedVectorError, C310CapturedVectorLoadError, C310CapturedVectorLoadStep,
    C310CapturedVstStep, C310RvecArithmeticHint, C310RvecMaskSprState, C310RvecMovpHint,
    C310RvecMovpStep, C310RvecValueError, C310RvecValueMachine, C310RvecValueStep,
    C310RvecVstiStore, c310_predicate_bytes_to_mask,
};
use crate::rvec_address_c310::{
    C310_CAPTURED_VAG_WORD, C310CapturedVagDescriptor, C310RvecAddressError, C310RvecAddressState,
};
use crate::rvec_loop_c310::{
    C310_CAPTURED_LONG_VLOOP_WORD, C310_CAPTURED_SHORT_VLOOP_WORD, C310CapturedLoopError,
    C310CapturedLoopHeader, C310CapturedLoopPcStep, C310RvecLoopController,
};
use crate::ub_replay::UbReplayMemory;

const C310_CAPTURED_NORMAL_U32_VSTI_WORD: u32 = 0x4018_010a;
const C310_CAPTURED_VECTOR_END_WORD: u32 = 0xc000_0017;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum C310CapturedRvecEffect {
    Vag(C310CapturedVagDescriptor),
    Smovi(C310CapturedSmoviStep),
    Vloop(C310CapturedLoopHeader),
    Load(C310CapturedVectorLoadStep),
    PredicateSet(C310CapturedPsetStep),
    PredicateLessThan(C310CapturedPltStep),
    Duplicate(C310CapturedVdupsStep),
    PredicateMove(C310RvecMovpStep),
    Arithmetic(C310RvecValueStep),
    Store(C310CapturedVstStep),
    StoreNormalU32(Vec<C310RvecVstiStore>),
    End,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct C310CapturedRvecProgramStep {
    pub pc: u64,
    pub word: u32,
    pub next_pc: u64,
    pub effect: C310CapturedRvecEffect,
    pub loop_step: Option<C310CapturedLoopPcStep>,
    pub halted_after: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum C310CapturedRvecProgramError {
    #[error("C310 vector program is halted at PC {pc:#x}")]
    Halted { pc: u64 },
    #[error("C310 vector word {word:#010x} at PC {pc:#x} is unsupported")]
    UnsupportedWord { pc: u64, word: u32 },
    #[error("C310 vector program PC overflows")]
    PcOverflow,
    #[error("C310 vector program requires scalar register S{index}")]
    MissingScalar { index: u8 },
    #[error("C310 vector program requires address register A0")]
    MissingAddressA0,
    #[error("C310 vector program requires predicate register P{index}")]
    MissingPredicate { index: u8 },
    #[error("C310 vector loop expected PC {expected:#x}, got {actual:#x}")]
    LoopPcMismatch { expected: u64, actual: u64 },
    #[error("C310 vector program did not halt within {limit} steps")]
    StepLimit { limit: usize },
    #[error(transparent)]
    Object(#[from] DeviceElfError),
    #[error(transparent)]
    Address(#[from] C310RvecAddressError),
    #[error(transparent)]
    Loop(#[from] C310CapturedLoopError),
    #[error(transparent)]
    Smovi(#[from] C310CapturedSmoviError),
    #[error(transparent)]
    PredicateSet(#[from] C310CapturedPsetError),
    #[error(transparent)]
    PredicateLessThan(#[from] C310CapturedPltError),
    #[error(transparent)]
    Load(#[from] C310CapturedVectorLoadError),
    #[error(transparent)]
    Value(#[from] C310RvecValueError),
    #[error(transparent)]
    Store(#[from] C310CapturedVectorError),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C310CapturedRvecProgram {
    pc: u64,
    halted: bool,
    registers: C310RvecValueMachine,
    address: C310RvecAddressState,
    loops: C310RvecLoopController,
    mask_sprs: C310RvecMaskSprState,
    ub: UbReplayMemory,
}

impl C310CapturedRvecProgram {
    pub fn new(
        entry_pc: u64,
        registers: C310RvecValueMachine,
        mask_sprs: C310RvecMaskSprState,
        ub: UbReplayMemory,
    ) -> Self {
        Self {
            pc: entry_pc,
            halted: false,
            registers,
            address: C310RvecAddressState::default(),
            loops: C310RvecLoopController::default(),
            mask_sprs,
            ub,
        }
    }

    pub const fn pc(&self) -> u64 {
        self.pc
    }

    pub const fn is_halted(&self) -> bool {
        self.halted
    }

    pub const fn registers(&self) -> &C310RvecValueMachine {
        &self.registers
    }

    pub const fn address(&self) -> &C310RvecAddressState {
        &self.address
    }

    pub const fn ub(&self) -> &UbReplayMemory {
        &self.ub
    }

    pub fn step_object(
        &mut self,
        kernel: &ProjectedDeviceKernel<'_>,
    ) -> Result<C310CapturedRvecProgramStep, C310CapturedRvecProgramError> {
        if self.halted {
            return Err(C310CapturedRvecProgramError::Halted { pc: self.pc });
        }
        let mut next = self.clone();
        let step = next.execute_object_step(kernel)?;
        *self = next;
        Ok(step)
    }

    pub fn run_object(
        &mut self,
        kernel: &ProjectedDeviceKernel<'_>,
        max_steps: usize,
    ) -> Result<Vec<C310CapturedRvecProgramStep>, C310CapturedRvecProgramError> {
        let mut steps = Vec::new();
        for _ in 0..max_steps {
            if self.halted {
                return Ok(steps);
            }
            steps.push(self.step_object(kernel)?);
        }
        if self.halted {
            Ok(steps)
        } else {
            Err(C310CapturedRvecProgramError::StepLimit { limit: max_steps })
        }
    }

    fn execute_object_step(
        &mut self,
        kernel: &ProjectedDeviceKernel<'_>,
    ) -> Result<C310CapturedRvecProgramStep, C310CapturedRvecProgramError> {
        let pc = self.pc;
        let word = kernel.fetch_executable_word(pc)?;
        let effect = match word {
            C310_CAPTURED_VAG_WORD => {
                C310CapturedRvecEffect::Vag(self.address.configure_vag_word(pc, word)?)
            }
            C310_CAPTURED_SMOVI32_WORD => {
                C310CapturedRvecEffect::Smovi(self.registers.execute_captured_smovi_word(pc, word)?)
            }
            C310_CAPTURED_SHORT_VLOOP_WORD | C310_CAPTURED_LONG_VLOOP_WORD => {
                let header =
                    self.loops
                        .start_from_object(kernel, pc, &self.registers, &mut self.address)?;
                self.pc = header.first_body_pc;
                return Ok(C310CapturedRvecProgramStep {
                    pc,
                    word,
                    next_pc: self.pc,
                    effect: C310CapturedRvecEffect::Vloop(header),
                    loop_step: None,
                    halted_after: false,
                });
            }
            C310_CAPTURED_VLD_V0_WORD
            | C310_CAPTURED_VLD_V1_WORD
            | C310_CAPTURED_VLDI_V0_WORD
            | C310_CAPTURED_VLDI_V1_WORD => C310CapturedRvecEffect::Load(
                self.registers
                    .execute_captured_vector_load_from_scalar_state(
                        pc,
                        word,
                        &self.address,
                        &self.ub,
                    )?,
            ),
            C310_CAPTURED_PLT32_WORD => C310CapturedRvecEffect::PredicateLessThan(
                self.registers.execute_captured_plt32_word(pc, word)?,
            ),
            C310_CAPTURED_PSET_WORD => C310CapturedRvecEffect::PredicateSet(
                self.registers.execute_captured_pset_word(pc, word)?,
            ),
            C310_CAPTURED_VDUPS_WORD => {
                let scalar_word = self.scalar_pair_value(word)?;
                let predicate = self.predicate_mask(1)?;
                C310CapturedRvecEffect::Duplicate(self.registers.execute_captured_vdups_word(
                    pc,
                    word,
                    scalar_word,
                    &predicate,
                )?)
            }
            C310_CAPTURED_VST_WORD | C310_CAPTURED_SUB_VST_WORD => {
                let address = self.scalar_pair_address(word, true)?;
                let predicate = self.predicate_mask(1)?;
                C310CapturedRvecEffect::Store(self.registers.execute_captured_vst_word(
                    pc,
                    word,
                    address,
                    &predicate,
                    &mut self.ub,
                )?)
            }
            C310_CAPTURED_NORMAL_U32_VSTI_WORD => {
                let address = self.scalar_pair_address(word, false)?;
                C310CapturedRvecEffect::StoreNormalU32(
                    self.registers
                        .execute_normal_u32_vsti_to_ub(word, address, &mut self.ub)?,
                )
            }
            C310_CAPTURED_VECTOR_END_WORD => {
                if self.loops.active_loop().is_some() {
                    return Err(C310CapturedRvecProgramError::UnsupportedWord { pc, word });
                }
                self.halted = true;
                C310CapturedRvecEffect::End
            }
            _ if C310RvecMovpHint::from_word(word).is_some() => {
                C310CapturedRvecEffect::PredicateMove(
                    self.registers
                        .execute_movp_u32_from_mask_sprs(word, &self.mask_sprs)?,
                )
            }
            _ if C310RvecArithmeticHint::from_word(word).is_some() => {
                C310CapturedRvecEffect::Arithmetic(
                    self.registers
                        .execute_fp32_word_from_predicate_registers(word)?,
                )
            }
            _ => return Err(C310CapturedRvecProgramError::UnsupportedWord { pc, word }),
        };
        let loop_step = if self.loops.active_loop().is_some() {
            let step = self
                .loops
                .step_from_object(kernel, &self.registers, &mut self.address)?;
            if step.executed_pc != pc {
                return Err(C310CapturedRvecProgramError::LoopPcMismatch {
                    expected: step.executed_pc,
                    actual: pc,
                });
            }
            Some(step)
        } else {
            None
        };
        self.pc = match loop_step {
            Some(step) => step.next_pc,
            None => pc
                .checked_add(4)
                .ok_or(C310CapturedRvecProgramError::PcOverflow)?,
        };
        Ok(C310CapturedRvecProgramStep {
            pc,
            word,
            next_pc: self.pc,
            effect,
            loop_step,
            halted_after: self.halted,
        })
    }

    fn predicate_mask(&self, index: u8) -> Result<[u64; 4], C310CapturedRvecProgramError> {
        let bytes = self
            .registers
            .predicate_register(usize::from(index))
            .ok_or(C310CapturedRvecProgramError::MissingPredicate { index })?;
        Ok(c310_predicate_bytes_to_mask(bytes)?)
    }

    fn scalar_pair_address(
        &self,
        word: u32,
        include_a0: bool,
    ) -> Result<u64, C310CapturedRvecProgramError> {
        let value = self.scalar_pair_value(word)?;
        let offset = if include_a0 {
            self.address
                .address_a0()
                .ok_or(C310CapturedRvecProgramError::MissingAddressA0)?
        } else {
            0
        };
        Ok(u64::from(value.wrapping_add(offset)))
    }

    fn scalar_pair_value(&self, word: u32) -> Result<u32, C310CapturedRvecProgramError> {
        let low_index = (((word >> 19) & 0x3f) * 2) as u8;
        let high_index = low_index + 1;
        let low = self
            .registers
            .scalar_register(usize::from(low_index))
            .ok_or(C310CapturedRvecProgramError::MissingScalar { index: low_index })?;
        let high = self
            .registers
            .scalar_register(usize::from(high_index))
            .ok_or(C310CapturedRvecProgramError::MissingScalar { index: high_index })?;
        Ok(low | high.wrapping_shl(16))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::predicate_buffer_c310::C310_PB_SLOT_BYTES;
    use crate::replay_memory::MemoryByteState;

    fn kernel_bytes(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|word| word.to_le_bytes()).collect()
    }

    fn registers(payload_words: &[u32]) -> C310RvecValueMachine {
        let mut machine = C310RvecValueMachine::from_vector_and_predicate_bytes(
            vec![vec![0; 64], vec![0; 64]],
            vec![vec![0; 32], vec![0; 32]],
        )
        .unwrap();
        let mut slot = [0_u8; C310_PB_SLOT_BYTES];
        let flags = (1_u32 << payload_words.len()) * 2 - 2;
        slot[..4].copy_from_slice(&flags.to_le_bytes());
        for (index, word) in payload_words.iter().enumerate() {
            slot[4 + index * 4..8 + index * 4].copy_from_slice(&word.to_le_bytes());
        }
        machine.apply_pb_scalar_init(&slot);
        machine
    }

    fn ub_with_inputs_and_prior(x: &[u32; 32], y: &[u32; 32], prior: &[u32; 32]) -> UbReplayMemory {
        let mut ub = UbReplayMemory::new(384, 384);
        ub.write_states(0, &[MemoryByteState::Known(0); 384])
            .unwrap();
        for (base, words) in [(0, x), (0x80, y), (0x100, prior)] {
            let bytes: Vec<_> = words
                .iter()
                .flat_map(|word| word.to_le_bytes().map(MemoryByteState::Known))
                .collect();
            ub.write_states(base, &bytes).unwrap();
        }
        ub
    }

    fn output_words(ub: &UbReplayMemory) -> Vec<u32> {
        ub.read_known(0x100, 128)
            .unwrap()
            .chunks_exact(4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn object_fetched_duplicate_program_initializes_predicate_and_local_output() {
        let words = [
            C310_CAPTURED_VAG_WORD,
            C310_CAPTURED_PSET_WORD,
            C310_CAPTURED_VDUPS_WORD,
            C310_CAPTURED_SMOVI32_WORD,
            C310_CAPTURED_SHORT_VLOOP_WORD,
            C310_CAPTURED_PLT32_WORD,
            C310_CAPTURED_VST_WORD,
            C310_CAPTURED_VECTOR_END_WORD,
        ];
        let bytes = kernel_bytes(&words);
        let kernel = ProjectedDeviceKernel::from_test_code(&bytes, 0x1000);
        let old = (-123.0_f32).to_bits();
        let mut program = C310CapturedRvecProgram::new(
            0x1000,
            registers(&[0x100, 0x0001_0000, old, 0x100]),
            C310RvecMaskSprState::default(),
            ub_with_inputs_and_prior(&[0; 32], &[0; 32], &[0; 32]),
        );
        let steps = program.run_object(&kernel, words.len()).unwrap();
        assert_eq!(steps.len(), words.len());
        assert!(program.is_halted());
        assert_eq!(program.registers().scalar_register(6), Some(0));
        assert_eq!(program.registers().scalar_register(7), Some(0xc2f6));
        assert_eq!(steps[4].next_pc, 0x1014);
        let C310CapturedRvecEffect::PredicateSet(predicate_set) = &steps[1].effect else {
            panic!("expected predicate set effect");
        };
        assert_eq!(predicate_set.predicate_bytes, [0x11; 32]);
        let C310CapturedRvecEffect::Duplicate(duplicate) = &steps[2].effect else {
            panic!("expected duplicate effect");
        };
        assert_eq!(duplicate.scalar_word, old);
        assert_eq!(duplicate.written_lanes, (0..64).collect::<Vec<_>>());
        assert_eq!(output_words(program.ub()), vec![old; 32]);
        let mut expected_p1 = [0; 32];
        expected_p1[..16].fill(0x11);
        assert_eq!(
            program.registers().predicate_register(1).unwrap(),
            expected_p1
        );
    }

    #[test]
    fn object_fetched_subtract_runs_loop_and_writes_predicated_ub() {
        let words = [
            C310_CAPTURED_VAG_WORD,
            C310_CAPTURED_SMOVI32_WORD,
            C310_CAPTURED_LONG_VLOOP_WORD,
            C310_CAPTURED_VLD_V0_WORD,
            C310_CAPTURED_VLD_V1_WORD,
            C310_CAPTURED_PLT32_WORD,
            0x8008_2781,
            C310_CAPTURED_SUB_VST_WORD,
            C310_CAPTURED_VECTOR_END_WORD,
        ];
        let bytes = kernel_bytes(&words);
        let kernel = ProjectedDeviceKernel::from_test_code(&bytes, 0x1000);
        let x = std::array::from_fn(|index| ((index + 1) as f32).to_bits());
        let y = std::array::from_fn(|index| ((index as f32) * 0.25).to_bits());
        let prior = [0xdead_beef; 32];
        let mut program = C310CapturedRvecProgram::new(
            0x1000,
            registers(&[0x100, 1, 0, 0x80, 0x100]),
            C310RvecMaskSprState::default(),
            ub_with_inputs_and_prior(&x, &y, &prior),
        );
        let steps = program.run_object(&kernel, 9).unwrap();
        assert_eq!(steps.len(), 9);
        assert!(program.is_halted());
        assert_eq!(steps[2].next_pc, 0x100c);
        assert_eq!(steps[7].loop_step.unwrap().next_pc, 0x1020);
        assert!(steps[7].loop_step.unwrap().finished);
        let expected: Vec<_> = (0..32)
            .map(|index| ((index + 1) as f32 - index as f32 * 0.25).to_bits())
            .collect();
        assert_eq!(output_words(program.ub()), expected);
    }

    #[test]
    fn object_fetched_multiply_runs_loop_and_writes_predicated_ub() {
        let words = [
            C310_CAPTURED_VAG_WORD,
            C310_CAPTURED_SMOVI32_WORD,
            C310_CAPTURED_LONG_VLOOP_WORD,
            C310_CAPTURED_VLD_V0_WORD,
            C310_CAPTURED_VLD_V1_WORD,
            C310_CAPTURED_PLT32_WORD,
            0x8000_27c0,
            C310_CAPTURED_SUB_VST_WORD,
            C310_CAPTURED_VECTOR_END_WORD,
        ];
        let bytes = kernel_bytes(&words);
        let kernel = ProjectedDeviceKernel::from_test_code(&bytes, 0x1000);
        let mut x = std::array::from_fn(|index| ((index + 1) as f32).to_bits());
        let mut y = [0.5_f32.to_bits(); 32];
        x[0] = 0;
        y[0] = f32::INFINITY.to_bits();
        x[1] = (-0.0_f32).to_bits();
        y[1] = 2.0_f32.to_bits();
        let mut program = C310CapturedRvecProgram::new(
            0x1000,
            registers(&[0x100, 1, 0, 0x80, 0x100]),
            C310RvecMaskSprState::default(),
            ub_with_inputs_and_prior(&x, &y, &[0xdead_beef; 32]),
        );
        let steps = program.run_object(&kernel, words.len()).unwrap();
        assert_eq!(steps.len(), words.len());
        assert!(program.is_halted());
        assert_eq!(steps[6].word, 0x8000_27c0);
        assert!(matches!(
            steps[6].effect,
            C310CapturedRvecEffect::Arithmetic(_)
        ));
        assert_eq!(steps[7].loop_step.unwrap().next_pc, 0x1020);
        let output = output_words(program.ub());
        assert_eq!(output[0], 0x7fff_ffff);
        assert_eq!(output[1], 0x8000_0000);
        for (index, word) in output.iter().enumerate().skip(2) {
            assert_eq!(*word, ((index + 1) as f32 * 0.5).to_bits());
        }
    }

    #[test]
    fn object_fetched_add_uses_scalar_mask_and_preserves_inactive_ub() {
        let words = [
            C310_CAPTURED_VLDI_V0_WORD,
            C310_CAPTURED_VLDI_V1_WORD,
            0x8204_0156,
            0x8008_2780,
            C310_CAPTURED_NORMAL_U32_VSTI_WORD,
            C310_CAPTURED_VECTOR_END_WORD,
        ];
        let bytes = kernel_bytes(&words);
        let kernel = ProjectedDeviceKernel::from_test_code(&bytes, 0x2000);
        let x = std::array::from_fn(|index| ((index as f32) * 0.5).to_bits());
        let y = std::array::from_fn(|index| ((index as f32) * 0.25).to_bits());
        let prior = [0x1234_5678; 32];
        let mut program = C310CapturedRvecProgram::new(
            0x2000,
            registers(&[0, 0x80, 0x100]),
            C310RvecMaskSprState {
                mask0: 0x5555_5555,
                mask1: 0,
            },
            ub_with_inputs_and_prior(&x, &y, &prior),
        );
        let steps = program.run_object(&kernel, 6).unwrap();
        assert_eq!(steps.len(), 6);
        assert!(program.is_halted());
        let expected: Vec<_> = (0..32)
            .map(|index| {
                if index % 2 == 0 {
                    (index as f32 * 0.75).to_bits()
                } else {
                    prior[index]
                }
            })
            .collect();
        assert_eq!(output_words(program.ub()), expected);
    }

    #[test]
    fn unsupported_and_unknown_input_steps_are_atomic() {
        let words = [C310_CAPTURED_VLDI_V0_WORD, 0xdead_beef];
        let bytes = kernel_bytes(&words);
        let kernel = ProjectedDeviceKernel::from_test_code(&bytes, 0x3000);
        let mut program = C310CapturedRvecProgram::new(
            0x3000,
            registers(&[0, 0x80, 0x100]),
            C310RvecMaskSprState::default(),
            UbReplayMemory::new(384, 384),
        );
        let before = program.clone();
        assert!(matches!(
            program.step_object(&kernel),
            Err(C310CapturedRvecProgramError::Load(_))
        ));
        assert_eq!(program, before);

        let unknown = kernel_bytes(&[0xdead_beef]);
        let kernel = ProjectedDeviceKernel::from_test_code(&unknown, 0x3000);
        assert_eq!(
            program.step_object(&kernel),
            Err(C310CapturedRvecProgramError::UnsupportedWord {
                pc: 0x3000,
                word: 0xdead_beef,
            })
        );
        assert_eq!(program, before);
    }

    #[test]
    fn duplicate_requires_pb_scalar_pair_and_preserves_state_on_failure() {
        let bytes = kernel_bytes(&[C310_CAPTURED_PSET_WORD, C310_CAPTURED_VDUPS_WORD]);
        let kernel = ProjectedDeviceKernel::from_test_code(&bytes, 0x4000);
        let mut program = C310CapturedRvecProgram::new(
            0x4000,
            registers(&[]),
            C310RvecMaskSprState::default(),
            UbReplayMemory::new(384, 384),
        );
        program.step_object(&kernel).unwrap();
        let after_pset = program.clone();
        assert_eq!(
            program.step_object(&kernel),
            Err(C310CapturedRvecProgramError::MissingScalar { index: 6 })
        );
        assert_eq!(program, after_pset);
    }
}
