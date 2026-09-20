use crate::device_loader::{DeviceKernelFetchError, LoadedDeviceKernel};
use crate::hbm_pv_memory::HbmPvMemory;
use crate::machine::{
    ScalarFlowStep, ScalarInstructionError, ScalarInstructionStep, ScalarMachine, ScalarMemoryBus,
};
use crate::vec_queue_c310::C310VfQueueInstruction;
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

    pub fn step_c310_vf_words<B: ScalarMemoryBus>(
        &mut self,
        first_word: u32,
        second_word: u32,
        bus: &mut B,
    ) -> Result<ScalarProgramStep, ScalarInstructionError<B::Error>> {
        let pc = self.pc;
        if self.halted {
            return Err(ScalarInstructionError::ProgramEnded { pc });
        }
        let instruction = self
            .machine
            .enqueue_c310_vf(pc, first_word, second_word, bus)?;
        let next_pc = pc.wrapping_add(8);
        self.pc = next_pc;
        Ok(ScalarProgramStep {
            pc,
            word: first_word,
            next_pc,
            instruction,
            halted_after: false,
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
        if C310VfQueueInstruction::is_prefix(self.machine.architecture(), word) {
            let second_word = kernel.fetch_executable_word(code_memory, self.pc.wrapping_add(4))?;
            self.step_c310_vf_words(word, second_word, data_bus)
                .map_err(Into::into)
        } else {
            self.step_word(word, data_bus).map_err(Into::into)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::architecture::Architecture;
    use crate::buffer_c310::C310BufferDisposition;
    use crate::flow::{BufferEncoding, BufferOperation, C310BufferStep};
    use crate::machine::ScalarMachineError;
    use crate::predicate_buffer_c310::{C310PushPbDisposition, C310PushPbStep};
    use crate::vec_queue_c310::{C310VfQueueDisposition, C310VfQueueStep};
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

    struct BufferBus {
        disposition: C310BufferDisposition,
        fail: bool,
        steps: Vec<C310BufferStep>,
    }

    impl ScalarMemoryBus for BufferBus {
        type Error = io::Error;

        fn read(&mut self, _address: u64, _destination: &mut [u8]) -> Result<(), Self::Error> {
            Err(io::Error::other("memory read was not expected"))
        }

        fn write(&mut self, _address: u64, _source: &[u8]) -> Result<(), Self::Error> {
            Err(io::Error::other("memory write was not expected"))
        }

        fn execute_c310_buffer(
            &mut self,
            step: C310BufferStep,
        ) -> Result<C310BufferDisposition, Self::Error> {
            self.steps.push(step);
            if self.fail {
                Err(io::Error::other("buffer backend failed"))
            } else {
                Ok(self.disposition)
            }
        }
    }

    struct PushPbBus {
        disposition: C310PushPbDisposition,
        fail: bool,
        steps: Vec<C310PushPbStep>,
    }

    struct VfQueueBus {
        disposition: C310VfQueueDisposition,
        fail: bool,
        steps: Vec<C310VfQueueStep>,
    }

    impl ScalarMemoryBus for VfQueueBus {
        type Error = io::Error;

        fn read(&mut self, _address: u64, _destination: &mut [u8]) -> Result<(), Self::Error> {
            Err(io::Error::other("memory read was not expected"))
        }

        fn write(&mut self, _address: u64, _source: &[u8]) -> Result<(), Self::Error> {
            Err(io::Error::other("memory write was not expected"))
        }

        fn enqueue_c310_vf(
            &mut self,
            step: C310VfQueueStep,
        ) -> Result<C310VfQueueDisposition, Self::Error> {
            self.steps.push(step);
            if self.fail {
                Err(io::Error::other("vector queue backend failed"))
            } else {
                Ok(self.disposition)
            }
        }
    }

    #[test]
    fn c310_vf_queue_requires_backend_acceptance_before_eight_byte_advance() {
        let mut machine = ScalarMachine::new(Architecture::Dav3510, [0; 32], 0);
        machine.set_xreg(13, 0x10d0_d900).unwrap();
        let mut stepper = ScalarStepper::new(machine, 0x10d0_d7fc);
        let mut bus = VfQueueBus {
            disposition: C310VfQueueDisposition::Unsupported,
            fail: false,
            steps: Vec::new(),
        };
        let before = stepper.clone();
        let words = [0x154d_0000, 0x15e0_0105];
        assert!(matches!(
            stepper.step_c310_vf_words(words[0], words[1], &mut bus),
            Err(ScalarInstructionError::VfQueueUnsupported {
                pc: 0x10d0_d7fc,
                first_word: 0x154d_0000,
                second_word: 0x15e0_0105,
            })
        ));
        assert_eq!(stepper, before);
        assert_eq!(bus.steps[0].vector_pc, 0x10d0_d900);

        bus.disposition = C310VfQueueDisposition::Stalled;
        assert!(matches!(
            stepper.step_c310_vf_words(words[0], words[1], &mut bus),
            Err(ScalarInstructionError::VfQueueStalled {
                pc: 0x10d0_d7fc,
                first_word: 0x154d_0000,
                second_word: 0x15e0_0105,
            })
        ));
        assert_eq!(stepper, before);

        bus.fail = true;
        assert!(matches!(
            stepper.step_c310_vf_words(words[0], words[1], &mut bus),
            Err(ScalarInstructionError::VfQueueBackend(_))
        ));
        assert_eq!(stepper, before);

        bus.fail = false;
        bus.disposition = C310VfQueueDisposition::Accepted;
        let accepted = stepper
            .step_c310_vf_words(words[0], words[1], &mut bus)
            .unwrap();
        assert_eq!(accepted.next_pc, 0x10d0_d804);
        assert!(matches!(
            accepted.instruction,
            ScalarInstructionStep::VfQueue(_)
        ));
        assert_eq!(stepper.pc(), 0x10d0_d804);
    }

    impl ScalarMemoryBus for PushPbBus {
        type Error = io::Error;

        fn read(&mut self, _address: u64, _destination: &mut [u8]) -> Result<(), Self::Error> {
            Err(io::Error::other("memory read was not expected"))
        }

        fn write(&mut self, _address: u64, _source: &[u8]) -> Result<(), Self::Error> {
            Err(io::Error::other("memory write was not expected"))
        }

        fn execute_c310_push_pb(
            &mut self,
            step: C310PushPbStep,
        ) -> Result<C310PushPbDisposition, Self::Error> {
            self.steps.push(step);
            if self.fail {
                Err(io::Error::other("predicate-buffer backend failed"))
            } else {
                Ok(self.disposition)
            }
        }
    }

    #[test]
    fn c310_push_pb_requires_backend_acceptance_before_advancing() {
        let mut machine = ScalarMachine::new(Architecture::Dav3510, [0; 32], 0);
        machine.set_xreg(10, 0x0102_0304_0506_0708).unwrap();
        machine.set_xreg(11, 0x1112_1314_1516_1718).unwrap();
        machine.set_xreg(2, 0x2122_2324_2526_2728).unwrap();
        let mut stepper = ScalarStepper::new(machine, 0x1000);
        let mut bus = PushPbBus {
            disposition: C310PushPbDisposition::Unsupported,
            fail: false,
            steps: Vec::new(),
        };
        let word = 0x4314_b108;
        let before = stepper.clone();
        assert!(matches!(
            stepper.step_word(word, &mut bus),
            Err(ScalarInstructionError::PushPbUnsupported {
                pc: 0x1000,
                word: 0x4314_b108
            })
        ));
        assert_eq!(stepper, before);
        assert_eq!(
            bus.steps[0].source_values,
            [
                0x0102_0304_0506_0708,
                0x1112_1314_1516_1718,
                0x2122_2324_2526_2728,
                0x2122_2324_2526_2728
            ]
        );

        bus.disposition = C310PushPbDisposition::Stalled;
        assert!(matches!(
            stepper.step_word(word, &mut bus),
            Err(ScalarInstructionError::PushPbStalled {
                pc: 0x1000,
                word: 0x4314_b108
            })
        ));
        assert_eq!(stepper, before);

        bus.fail = true;
        assert!(matches!(
            stepper.step_word(word, &mut bus),
            Err(ScalarInstructionError::PushPbBackend(_))
        ));
        assert_eq!(stepper, before);

        bus.fail = false;
        bus.disposition = C310PushPbDisposition::Accepted;
        let accepted = stepper.step_word(word, &mut bus).unwrap();
        assert!(matches!(
            accepted.instruction,
            ScalarInstructionStep::PushPb(_)
        ));
        assert_eq!(stepper.pc(), 0x1004);
    }

    #[test]
    fn c310_buffer_words_require_backend_acceptance_before_advancing() {
        let mut machine = ScalarMachine::new(Architecture::Dav3510, [0; 32], 0);
        machine.set_xreg(1, 0x25).unwrap();
        machine.set_xreg(0, 0x5f).unwrap();
        let mut stepper = ScalarStepper::new(machine, 0x1000);
        let mut bus = BufferBus {
            disposition: C310BufferDisposition::Accepted,
            fail: false,
            steps: Vec::new(),
        };

        for (word, encoding, operation, buffer_id) in [
            (
                0x4202_1004,
                BufferEncoding::FlowControl,
                BufferOperation::Get,
                5,
            ),
            (
                0x4222_1004,
                BufferEncoding::FlowControl,
                BufferOperation::Release,
                5,
            ),
            (
                0x15c0_8001,
                BufferEncoding::PushQueue,
                BufferOperation::Get,
                31,
            ),
            (
                0x15c0_0001,
                BufferEncoding::PushQueue,
                BufferOperation::Release,
                31,
            ),
        ] {
            let pc = stepper.pc();
            let step = stepper.step_word(word, &mut bus).unwrap();
            let ScalarInstructionStep::Buffer(buffer) = step.instruction else {
                panic!("expected buffer instruction");
            };
            assert_eq!(buffer.pc, pc);
            assert_eq!(buffer.buffer_id, buffer_id);
            assert_eq!(buffer.instruction.encoding, encoding);
            assert_eq!(buffer.instruction.operation, operation);
            assert_eq!(step.next_pc, pc + 4);
        }
        assert_eq!(bus.steps.len(), 4);

        for disposition in [
            C310BufferDisposition::Stalled,
            C310BufferDisposition::Unsupported,
        ] {
            bus.disposition = disposition;
            let before = stepper.clone();
            let result = stepper.step_word(0x4202_1004, &mut bus);
            assert!(matches!(
                (disposition, result),
                (
                    C310BufferDisposition::Stalled,
                    Err(ScalarInstructionError::BufferStalled { .. })
                ) | (
                    C310BufferDisposition::Unsupported,
                    Err(ScalarInstructionError::BufferUnsupported { .. })
                )
            ));
            assert_eq!(stepper, before);
        }

        bus.fail = true;
        let before = stepper.clone();
        assert!(matches!(
            stepper.step_word(0x15c0_8001, &mut bus),
            Err(ScalarInstructionError::BufferBackend(_))
        ));
        assert_eq!(stepper, before);
    }

    #[test]
    fn c310_buffer_words_default_to_unsupported_and_do_not_run_on_c220() {
        let mut c310 = ScalarStepper::new(
            ScalarMachine::new(Architecture::Dav3510, [0; 32], 0),
            0x1000,
        );
        let before = c310.clone();
        assert!(matches!(
            c310.step_word(0x4202_1004, &mut NoMemoryBus),
            Err(ScalarInstructionError::BufferUnsupported { .. })
        ));
        assert_eq!(c310, before);

        let mut c220 = ScalarStepper::new(
            ScalarMachine::new(Architecture::Dav2201, [0; 32], 0),
            0x1000,
        );
        let mut bus = BufferBus {
            disposition: C310BufferDisposition::Accepted,
            fail: false,
            steps: Vec::new(),
        };
        assert!(c220.step_word(0x4202_1004, &mut bus).is_err());
        assert!(bus.steps.is_empty());
        assert_eq!(c220.pc(), 0x1000);
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
    fn indexed_memory_backend_failure_preserves_pc_and_registers() {
        for (architecture, pc, word) in [
            (Architecture::Dav2201, 0x1131_23c0, 0x011c_b600),
            (Architecture::Dav3510, 0x10d0_d660, 0x0103_2980),
            (Architecture::Dav2201, 0x1131_243c, 0x0124_a880),
            (Architecture::Dav2201, 0x1131_25d4, 0x0126_f800),
            (Architecture::Dav2201, 0x1131_23cc, 0x0e00_b601),
            (Architecture::Dav2201, 0x1131_2448, 0x0e00_a881),
            (Architecture::Dav2201, 0x1131_25e0, 0x0e00_f801),
            (Architecture::Dav3510, 0x10d0_d66c, 0x0e01_2981),
        ] {
            let machine = ScalarMachine::new(architecture, [0; 32], 0);
            let mut stepper = ScalarStepper::new(machine, pc);
            let before = stepper.clone();
            assert!(matches!(
                stepper.step_word(word, &mut NoMemoryBus),
                Err(ScalarInstructionError::Memory(
                    crate::machine::ScalarMemoryExecutionError::Backend(_)
                ))
            ));
            assert_eq!(stepper, before);
        }
    }

    #[test]
    fn unsupported_barrier_keeps_pc_and_machine_unchanged() {
        for (architecture, pc, word) in [
            (Architecture::Dav2201, 0x1131_2090, 0x40e0_1800),
            (Architecture::Dav2201, 0x1131_2638, 0x40e0_0400),
            (Architecture::Dav3510, 0x10d0_d090, 0x40e0_1800),
        ] {
            let mut stepper =
                ScalarStepper::new(ScalarMachine::from_pem_initial_state(architecture), pc);
            let before = stepper.clone();
            assert!(matches!(
                stepper.step_word(word, &mut NoMemoryBus),
                Err(ScalarInstructionError::SynchronizationUnsupported { .. })
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
