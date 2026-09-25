use crate::architecture::Architecture;
use crate::isa::c220::scalar::C220ScalarDirectStore;
use crate::sim::common::scalar::{ScalarMachine, ScalarMachineError};

mod cached;
pub use cached::C220StoreOperands;

/// Operands captured after register dependencies clear, before LSU execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220DirectStoreOperands {
    pub pc: u64,
    pub word: u32,
    instruction: C220ScalarDirectStore,
    pub base_value: u64,
    pub source_value: u64,
    pub effective_address: u64,
    bytes: [u8; 8],
}

impl C220DirectStoreOperands {
    pub fn capture(
        machine: &ScalarMachine,
        pc: u64,
        word: u32,
    ) -> Result<Self, ScalarMachineError> {
        let instruction = C220ScalarDirectStore::decode(word)
            .filter(|_| machine.architecture() == Architecture::Dav2201)
            .ok_or(ScalarMachineError::UnsupportedWord { pc, word })?;
        let base_value = machine.xregs()[usize::from(instruction.base_register)];
        let source_value = machine.xregs()[usize::from(instruction.source_register)];
        Ok(Self {
            pc,
            word,
            instruction,
            base_value,
            source_value,
            effective_address: instruction.effective_address(base_value),
            bytes: source_value.to_le_bytes(),
        })
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes[..usize::from(self.instruction.width_bytes)]
    }

    pub fn instruction(&self) -> C220ScalarDirectStore {
        self.instruction
    }
}
