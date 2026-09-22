use thiserror::Error;

use crate::architecture::Architecture;
use crate::image::loader::{DeviceKernelFetchError, LoadedDeviceKernel};
use crate::isa::c310::buffer::{C310BufferInstruction, C310BufferStep};
use crate::isa::c310::vector::C310ObservedMovemaskHint;
use crate::memory::hbm_pv_memory::HbmPvMemory;
use crate::memory::mapped::MappedMemory;
use crate::sim::c310::buffer::C310BufferDisposition;
use crate::sim::c310::predicate_buffer::{
    C310PushPbDisposition, C310PushPbInstruction, C310PushPbStep,
};
use crate::sim::c310::vector::C310ObservedMovemaskStep;
use crate::sim::c310::vector_queue::{
    C310VfQueueDisposition, C310VfQueueInstruction, C310VfQueueStep,
};
use crate::sim::common::scalar::{
    ScalarFlowStep, ScalarInstructionError, ScalarInstructionStep, ScalarMachine,
    ScalarMachineError, ScalarMemoryBus,
};

pub trait C310ScalarBus: ScalarMemoryBus {
    fn execute_buffer(
        &mut self,
        _step: C310BufferStep,
    ) -> Result<C310BufferDisposition, Self::Error> {
        Ok(C310BufferDisposition::Unsupported)
    }

    fn execute_push_pb(
        &mut self,
        _step: C310PushPbStep,
    ) -> Result<C310PushPbDisposition, Self::Error> {
        Ok(C310PushPbDisposition::Unsupported)
    }

    fn enqueue_vf(
        &mut self,
        _step: C310VfQueueStep,
    ) -> Result<C310VfQueueDisposition, Self::Error> {
        Ok(C310VfQueueDisposition::Unsupported)
    }
}

impl C310ScalarBus for MappedMemory {}
impl C310ScalarBus for HbmPvMemory {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C310ScalarInstructionStep {
    Common(ScalarInstructionStep),
    Movemask(C310ObservedMovemaskStep),
    Buffer(C310BufferStep),
    PushPb(C310PushPbStep),
    VfQueue(C310VfQueueStep),
}

#[derive(Debug, Error)]
pub enum C310ScalarExecutionError<E: std::error::Error + 'static> {
    #[error(transparent)]
    Common(#[from] ScalarInstructionError<E>),
    #[error(transparent)]
    Scalar(#[from] ScalarMachineError),
    #[error("buffer instruction {word:#010x} at PC {pc:#x} is not supported by this backend")]
    BufferUnsupported { pc: u64, word: u32 },
    #[error("buffer instruction {word:#010x} at PC {pc:#x} stalled")]
    BufferStalled { pc: u64, word: u32 },
    #[error("buffer instruction backend failed: {0}")]
    BufferBackend(#[source] E),
    #[error("predicate-buffer push {word:#010x} at PC {pc:#x} is not supported by this backend")]
    PushPbUnsupported { pc: u64, word: u32 },
    #[error("predicate-buffer push {word:#010x} at PC {pc:#x} stalled")]
    PushPbStalled { pc: u64, word: u32 },
    #[error("predicate-buffer backend failed: {0}")]
    PushPbBackend(#[source] E),
    #[error(
        "vector queue words {first_word:#010x}/{second_word:#010x} at PC {pc:#x} are not supported by this backend"
    )]
    VfQueueUnsupported {
        pc: u64,
        first_word: u32,
        second_word: u32,
    },
    #[error("vector queue words {first_word:#010x}/{second_word:#010x} at PC {pc:#x} stalled")]
    VfQueueStalled {
        pc: u64,
        first_word: u32,
        second_word: u32,
    },
    #[error("vector queue backend failed: {0}")]
    VfQueueBackend(#[source] E),
    #[error("C310 scalar program already ended before PC {pc:#x}")]
    ProgramEnded { pc: u64 },
}

pub fn execute_instruction<B: C310ScalarBus>(
    machine: &mut ScalarMachine,
    pc: u64,
    word: u32,
    bus: &mut B,
) -> Result<C310ScalarInstructionStep, C310ScalarExecutionError<B::Error>> {
    if machine.architecture() != Architecture::Dav3510 {
        return Err(ScalarMachineError::UnsupportedWord { pc, word }.into());
    }
    if let Some(instruction) = C310PushPbInstruction::decode(Architecture::Dav3510, word) {
        let step = instruction.resolve(pc, machine.xregs());
        return match bus
            .execute_push_pb(step)
            .map_err(C310ScalarExecutionError::PushPbBackend)?
        {
            C310PushPbDisposition::Accepted => Ok(C310ScalarInstructionStep::PushPb(step)),
            C310PushPbDisposition::Stalled => {
                Err(C310ScalarExecutionError::PushPbStalled { pc, word })
            }
            C310PushPbDisposition::Unsupported => {
                Err(C310ScalarExecutionError::PushPbUnsupported { pc, word })
            }
        };
    }
    if let Some(instruction) = C310BufferInstruction::decode(word) {
        let step = instruction.resolve(pc, machine.xregs());
        return match bus
            .execute_buffer(step)
            .map_err(C310ScalarExecutionError::BufferBackend)?
        {
            C310BufferDisposition::Accepted => Ok(C310ScalarInstructionStep::Buffer(step)),
            C310BufferDisposition::Stalled => {
                Err(C310ScalarExecutionError::BufferStalled { pc, word })
            }
            C310BufferDisposition::Unsupported => {
                Err(C310ScalarExecutionError::BufferUnsupported { pc, word })
            }
        };
    }
    if C310ObservedMovemaskHint::from_word(word).is_some() {
        return Ok(C310ScalarInstructionStep::Movemask(execute_movemask(
            machine, pc, word,
        )?));
    }
    machine
        .execute_instruction(pc, word, bus)
        .map(C310ScalarInstructionStep::Common)
        .map_err(Into::into)
}

pub fn execute_movemask(
    machine: &mut ScalarMachine,
    pc: u64,
    word: u32,
) -> Result<C310ObservedMovemaskStep, ScalarMachineError> {
    if machine.architecture() != Architecture::Dav3510 {
        return Err(ScalarMachineError::UnsupportedWord { pc, word });
    }
    let hint = C310ObservedMovemaskHint::from_word(word)
        .ok_or(ScalarMachineError::UnsupportedWord { pc, word })?;
    let prior_value =
        machine
            .spr_value(hint.destination_spr)
            .ok_or(ScalarMachineError::SprValueUnavailable {
                pc,
                spr: hint.destination_spr,
            })?;
    let value = machine.xregs()[usize::from(hint.source_x_register)];
    machine.set_spr_value(hint.destination_spr, value)?;
    Ok(C310ObservedMovemaskStep {
        pc,
        word,
        hint,
        prior_value,
        value,
    })
}

pub fn enqueue_vf<B: C310ScalarBus>(
    machine: &ScalarMachine,
    pc: u64,
    first_word: u32,
    second_word: u32,
    bus: &mut B,
) -> Result<C310ScalarInstructionStep, C310ScalarExecutionError<B::Error>> {
    let instruction =
        C310VfQueueInstruction::decode(machine.architecture(), first_word, second_word).ok_or(
            ScalarMachineError::UnsupportedWord {
                pc,
                word: first_word,
            },
        )?;
    let step = instruction.resolve(pc, machine.xregs());
    match bus
        .enqueue_vf(step)
        .map_err(C310ScalarExecutionError::VfQueueBackend)?
    {
        C310VfQueueDisposition::Accepted => Ok(C310ScalarInstructionStep::VfQueue(step)),
        C310VfQueueDisposition::Stalled => Err(C310ScalarExecutionError::VfQueueStalled {
            pc,
            first_word,
            second_word,
        }),
        C310VfQueueDisposition::Unsupported => Err(C310ScalarExecutionError::VfQueueUnsupported {
            pc,
            first_word,
            second_word,
        }),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C310ScalarStepper {
    machine: ScalarMachine,
    pc: u64,
    halted: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C310ScalarProgramStep {
    pub pc: u64,
    pub word: u32,
    pub next_pc: u64,
    pub instruction: C310ScalarInstructionStep,
    pub halted_after: bool,
}

#[derive(Debug, Error)]
pub enum C310ScalarStepperError<E: std::error::Error + 'static> {
    #[error("loaded kernel is not a C310 image")]
    ArchitectureMismatch,
    #[error(transparent)]
    Fetch(#[from] DeviceKernelFetchError),
    #[error(transparent)]
    Execute(#[from] C310ScalarExecutionError<E>),
}

impl C310ScalarStepper {
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

    pub fn step_word<B: C310ScalarBus>(
        &mut self,
        word: u32,
        bus: &mut B,
    ) -> Result<C310ScalarProgramStep, C310ScalarExecutionError<B::Error>> {
        let pc = self.pc;
        if self.halted {
            return Err(C310ScalarExecutionError::ProgramEnded { pc });
        }
        let instruction = execute_instruction(&mut self.machine, pc, word, bus)?;
        let halted_after = matches!(
            instruction,
            C310ScalarInstructionStep::Common(ScalarInstructionStep::Flow(ScalarFlowStep::End(_)))
        );
        let next_pc = match instruction {
            C310ScalarInstructionStep::Common(ScalarInstructionStep::Flow(flow)) => {
                flow.target_pc()
            }
            _ => pc.wrapping_add(4),
        };
        self.pc = next_pc;
        self.halted = halted_after;
        Ok(C310ScalarProgramStep {
            pc,
            word,
            next_pc,
            instruction,
            halted_after,
        })
    }

    pub fn step_vf_words<B: C310ScalarBus>(
        &mut self,
        first_word: u32,
        second_word: u32,
        bus: &mut B,
    ) -> Result<C310ScalarProgramStep, C310ScalarExecutionError<B::Error>> {
        let pc = self.pc;
        if self.halted {
            return Err(C310ScalarExecutionError::ProgramEnded { pc });
        }
        let instruction = enqueue_vf(&self.machine, pc, first_word, second_word, bus)?;
        let next_pc = pc.wrapping_add(8);
        self.pc = next_pc;
        Ok(C310ScalarProgramStep {
            pc,
            word: first_word,
            next_pc,
            instruction,
            halted_after: false,
        })
    }

    pub fn step_loaded<B: C310ScalarBus>(
        &mut self,
        kernel: &LoadedDeviceKernel,
        code_memory: &mut HbmPvMemory,
        data_bus: &mut B,
    ) -> Result<C310ScalarProgramStep, C310ScalarStepperError<B::Error>> {
        if self.machine.architecture() != Architecture::Dav3510
            || kernel.placement().architecture != Architecture::Dav3510
        {
            return Err(C310ScalarStepperError::ArchitectureMismatch);
        }
        if self.halted {
            return Err(C310ScalarExecutionError::ProgramEnded { pc: self.pc }.into());
        }
        let word = kernel.fetch_executable_word(code_memory, self.pc)?;
        if C310VfQueueInstruction::is_prefix(Architecture::Dav3510, word) {
            let second_word = kernel.fetch_executable_word(code_memory, self.pc.wrapping_add(4))?;
            self.step_vf_words(word, second_word, data_bus)
                .map_err(Into::into)
        } else {
            self.step_word(word, data_bus).map_err(Into::into)
        }
    }
}
