use crate::architecture::Architecture;
use crate::isa::c220::scalar::{
    C220AtomicStoreOffset, C220ScalarAtomicStore, C220ScalarDirectStore,
};
use crate::sim::common::scalar::{ScalarMachine, ScalarMachineError};

mod cached;
pub use cached::C220StoreOperands;

mod atomic;
pub use atomic::{C220AtomicStoreError, C220AtomicStoreOperands, C220AtomicStoreResult};

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atomic_capture_separates_transfer_width_type_and_address_update() {
        let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
        machine.set_xreg(1, 0x8877_6655_4433_2211).unwrap();
        machine.set_xreg(2, 0x0100_0000).unwrap();
        machine.set_xreg(3, u64::MAX).unwrap();
        machine.set_spr_value(90, 0x15).unwrap();
        for width in 0..4 {
            for post_index in [false, true] {
                for register_offset in [false, true] {
                    let word = if register_offset {
                        0x0100_0070 | (3 << 7) | (u32::from(post_index) << 3)
                    } else {
                        0x1600_0fff | (u32::from(post_index) << 24)
                    } | (width << 22)
                        | (1 << 17)
                        | (2 << 12);
                    let captured = C220AtomicStoreOperands::capture(&machine, 0x40, word).unwrap();
                    let lane = crate::sim::c220::scalar::timing::C220ScalarTimingLane::default();
                    for register in 0..4 {
                        assert_eq!(
                            lane.dependency_tick_with_loads(word, 5, |source| source == register),
                            (register == 1 || register == 2 || register == 3 && register_offset)
                                .then_some(6),
                        );
                    }
                    let size = 1usize << width;
                    let adjusted = 0x0100_0000 - if register_offset { size as u64 } else { 1 };
                    assert_eq!(
                        captured.bytes(),
                        &0x8877_6655_4433_2211u64.to_le_bytes()[..size]
                    );
                    assert_eq!(captured.data_type(), 5);
                    assert_eq!(captured.updated_base, post_index.then_some(adjusted));
                    assert_eq!(
                        captured.effective_address,
                        if post_index { 0x0100_0000 } else { adjusted }
                    );
                    assert_eq!(captured.is_external(), post_index);
                    assert_eq!(machine.xregs()[2], 0x0100_0000);
                }
            }
        }
        assert!(C220ScalarAtomicStore::decode(0x0100_0020).is_none());
        assert!(C220ScalarAtomicStore::decode(0x3600_0000).is_none());
    }
}
