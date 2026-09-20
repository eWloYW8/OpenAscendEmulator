use crate::device_loader::{DeviceKernelFetchError, LoadedDeviceKernel};
use crate::hbm_pv_memory::HbmPvMemory;
use crate::machine::{
    ScalarFlowStep, ScalarInstructionError, ScalarInstructionStep, ScalarMachine, ScalarMemoryBus,
};
use serde::Serialize;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScalarStepper {
    machine: ScalarMachine,
    pc: u64,
    halted: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ScalarProgramStep {
    pub pc: u64,
    pub word: u32,
    pub next_pc: u64,
    pub instruction: ScalarInstructionStep,
    pub halted_after: bool,
}

#[derive(Debug, Error)]
pub enum ScalarStepperError<E: std::error::Error + 'static> {
    #[error("loaded kernel and scalar machine target different architectures")]
    ArchitectureMismatch,
    #[error(transparent)]
    Fetch(#[from] DeviceKernelFetchError),
    #[error(transparent)]
    Execute(#[from] ScalarInstructionError<E>),
}

impl ScalarStepper {
    pub const fn new(machine: ScalarMachine, entry_pc: u64) -> Self {
        Self {
            machine,
            pc: entry_pc,
            halted: false,
        }
    }

    pub const fn pc(&self) -> u64 {
        self.pc
    }

    pub const fn is_halted(&self) -> bool {
        self.halted
    }

    pub const fn machine(&self) -> &ScalarMachine {
        &self.machine
    }

    pub fn machine_mut(&mut self) -> &mut ScalarMachine {
        &mut self.machine
    }

    pub(crate) fn advance_sequential(&mut self) {
        self.pc = self.pc.wrapping_add(4);
    }

    pub fn step_word<B: ScalarMemoryBus>(
        &mut self,
        word: u32,
        bus: &mut B,
    ) -> Result<ScalarProgramStep, ScalarInstructionError<B::Error>> {
        let pc = self.pc;
        if self.halted {
            return Err(ScalarInstructionError::ProgramEnded { pc });
        }
        let instruction = self.machine.execute_instruction(pc, word, bus)?;
        let halted_after = matches!(
            instruction,
            ScalarInstructionStep::Flow(ScalarFlowStep::End(_))
        );
        let next_pc = match instruction {
            ScalarInstructionStep::Flow(flow) => flow.target_pc(),
            _ => pc.wrapping_add(4),
        };
        self.pc = next_pc;
        self.halted = halted_after;
        Ok(ScalarProgramStep {
            pc,
            word,
            next_pc,
            instruction,
            halted_after,
        })
    }

    pub fn step_loaded<B: ScalarMemoryBus>(
        &mut self,
        kernel: &LoadedDeviceKernel,
        code_memory: &mut HbmPvMemory,
        data_bus: &mut B,
    ) -> Result<ScalarProgramStep, ScalarStepperError<B::Error>> {
        if self.machine.architecture() != kernel.load().placement.architecture {
            return Err(ScalarStepperError::ArchitectureMismatch);
        }
        if self.halted {
            return Err(ScalarStepperError::Execute(
                ScalarInstructionError::ProgramEnded { pc: self.pc },
            ));
        }
        let word = kernel.fetch_executable_word(code_memory, self.pc)?;
        self.step_word(word, data_bus).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::architecture::Architecture;
    use crate::machine::ScalarMachineError;
    use std::io;

    #[derive(Default)]
    struct NoMemoryBus;

    impl ScalarMemoryBus for NoMemoryBus {
        type Error = io::Error;

        fn read(&mut self, _address: u64, _destination: &mut [u8]) -> Result<(), Self::Error> {
            Err(io::Error::other("memory read was not expected"))
        }

        fn write(&mut self, _address: u64, _source: &[u8]) -> Result<(), Self::Error> {
            Err(io::Error::other("memory write was not expected"))
        }
    }

    #[test]
    fn end_halts_the_stepper_before_any_further_instruction() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut stepper = ScalarStepper::new(
                ScalarMachine::from_pem_initial_state(architecture),
                0x10d0_d8b8,
            );
            let mut bus = NoMemoryBus;
            let step = stepper.step_word(0x4160_0000, &mut bus).unwrap();
            assert!(matches!(
                step.instruction,
                ScalarInstructionStep::Flow(ScalarFlowStep::End(_))
            ));
            assert!(step.halted_after);
            assert_eq!(step.next_pc, 0x10d0_d8bc);
            assert!(stepper.is_halted());

            let before = stepper.clone();
            assert!(matches!(
                stepper.step_word(0x0706_0001, &mut bus),
                Err(ScalarInstructionError::ProgramEnded { pc: 0x10d0_d8bc })
            ));
            assert_eq!(stepper, before);
        }
    }

    #[test]
    fn steps_scalar_and_branch_words_without_losing_pc_on_failure() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let machine = ScalarMachine::new(architecture, [0; 32], 0);
            let mut stepper = ScalarStepper::new(machine, 0x1000);
            let mut bus = NoMemoryBus;
            let move_step = stepper.step_word(0x0706_0001, &mut bus).unwrap();
            assert_eq!(move_step.pc, 0x1000);
            assert_eq!(move_step.next_pc, 0x1004);
            assert_eq!(stepper.pc(), 0x1004);

            let preload = stepper.step_word(0x08c0_0000, &mut bus).unwrap();
            assert!(matches!(
                preload.instruction,
                ScalarInstructionStep::CacheHint(_)
            ));
            assert_eq!(preload.next_pc, 0x1008);

            let branch = stepper.step_word(0x4000_0004, &mut bus).unwrap();
            assert_eq!(branch.next_pc, 0x1018);
            assert_eq!(stepper.pc(), 0x1018);

            let before = stepper.clone();
            assert!(matches!(
                stepper.step_word(0x6000_0000, &mut bus),
                Err(ScalarInstructionError::Scalar(
                    ScalarMachineError::UnsupportedWord { .. }
                ))
            ));
            assert_eq!(stepper, before);
        }
    }

    #[test]
    fn c220_movemask_updates_sprs_and_advances_pc() {
        let mut machine = ScalarMachine::new(Architecture::Dav2201, [0; 32], 0);
        machine.set_xreg(3, 0x5555_aaaa_f0f0_0f0f).unwrap();
        let mut stepper = ScalarStepper::new(machine, 0x2000);
        let mut bus = NoMemoryBus;

        let mask0 = stepper.step_word(0x8040_000c, &mut bus).unwrap();
        assert!(matches!(
            mask0.instruction,
            ScalarInstructionStep::C220Movemask(step)
                if step.source_register == 3
                    && step.destination_spr == 100
                    && step.source_value == 0x5555_aaaa_f0f0_0f0f
                    && step.prior_destination_value.is_none()
        ));
        assert_eq!(mask0.next_pc, 0x2004);
        assert_eq!(
            stepper.machine().spr_value(100),
            Some(0x5555_aaaa_f0f0_0f0f)
        );
        assert_eq!(stepper.machine().spr_value(101), None);

        let mask1 = stepper.step_word(0x8040_008c, &mut bus).unwrap();
        assert!(matches!(
            mask1.instruction,
            ScalarInstructionStep::C220Movemask(step)
                if step.destination_spr == 101 && step.source_register == 3
        ));
        assert_eq!(
            stepper.machine().spr_value(101),
            Some(0x5555_aaaa_f0f0_0f0f)
        );
        assert_eq!(stepper.pc(), 0x2008);

        let before = stepper.clone();
        assert!(matches!(
            stepper.step_word(0x8240_008c, &mut bus),
            Err(ScalarInstructionError::Scalar(
                ScalarMachineError::UnsupportedWord { .. }
            ))
        ));
        assert_eq!(stepper, before);
    }

    #[test]
    fn c310_does_not_apply_c220_movemask() {
        let mut stepper = ScalarStepper::new(
            ScalarMachine::new(Architecture::Dav3510, [0; 32], 0),
            0x3000,
        );
        let mut bus = NoMemoryBus;
        assert!(matches!(
            stepper.step_word(0x8040_0000, &mut bus),
            Err(ScalarInstructionError::Scalar(
                ScalarMachineError::UnsupportedWord { .. }
            ))
        ));
        assert_eq!(stepper.pc(), 0x3000);
        assert_eq!(stepper.machine().spr_value(100), None);
    }

    #[test]
    fn movx8_immediate_replaces_destination_on_both_architectures() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut machine = ScalarMachine::new(architecture, [0; 32], 0);
            machine.set_xreg(2, 0x1151_5200).unwrap();
            let mut stepper = ScalarStepper::new(machine, 0x4000);
            let mut bus = NoMemoryBus;

            let step = stepper.step_word(0x1004_2002, &mut bus).unwrap();
            assert!(matches!(
                step.instruction,
                ScalarInstructionStep::Register(register)
                    if register.destination_register == 2
                        && register.prior_destination_value == 0x1151_5200
                        && register.value == 0x10010
                        && register.source_register.is_none()
            ));
            assert_eq!(stepper.machine().xregs()[2], 0x10010);
            assert_eq!(stepper.pc(), 0x4004);
        }
    }
}
