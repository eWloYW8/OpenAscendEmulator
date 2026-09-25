use super::*;
use crate::memory::mapped::{MappedMemory, MappedMemoryError};
use crate::sim::c220::numeric::atomic::combine_atomic;
use crate::sim::c220::numeric::fp16::C220Fp16AddRounding;

#[derive(Debug, thiserror::Error)]
pub enum C220AtomicStoreError {
    #[error("atomic store requires an external address, got {0:#x}")]
    LocalAddress(u64),
    #[error("atomic input size differs from the instruction transfer width")]
    InputSize,
    #[error(transparent)]
    Memory(#[from] MappedMemoryError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::region::MemoryRegion;
    use crate::memory::sparse::SparseMemory;

    #[test]
    fn atomic_store_executes_all_type_codes_without_mutating_captured_inputs() {
        let cases: [(u32, u32, u32); 8] = [
            (0x1234, 5, 0x1234),
            (0x3f80_0000, 0x4000_0000, 0x4040_0000),
            (0x3c00_3c00, 0x4000_4000, 0x4200_4200),
            (0xffff_7fff, 0x0001_0001, 0x0000_8000),
            (0xffff_ffff, 1, 0),
            (0xffff_ff7f, 0x0101_0101, 0x0000_0080),
            (0x3f80_3f80, 0x4000_4000, 0x4040_4040),
            (0x1234, 5, 0x1234),
        ];
        let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
        machine.set_xreg(2, 0x0100_0000).unwrap();
        for (dtype, (input, previous, expected)) in cases.into_iter().enumerate() {
            machine.set_xreg(1, u64::from(input)).unwrap();
            machine.set_spr_value(90, dtype as u64).unwrap();
            let operands = C220AtomicStoreOperands::capture(&machine, 0, 0x1782_2004).unwrap();
            let mut memory = MappedMemory::bind(
                SparseMemory::new(
                    vec![MemoryRegion::new(8, previous.to_le_bytes().repeat(2)).unwrap()],
                    64,
                    64,
                ),
                &[0x0100_0000],
            )
            .unwrap();
            let result = operands
                .execute(&mut memory, C220Fp16AddRounding::NearestEven)
                .unwrap();
            assert_eq!(result.previous_bytes(), previous.to_le_bytes());
            assert_eq!(result.stored_bytes(), expected.to_le_bytes());
            assert_eq!(
                memory.read_known_at(result.address, 4).unwrap(),
                expected.to_le_bytes()
            );
            assert_eq!(
                memory.read_known_at(result.address + 4, 4).unwrap(),
                previous.to_le_bytes()
            );
            assert_eq!(operands.bytes(), input.to_le_bytes());
            assert_eq!(machine.xregs()[2], 0x0100_0000);
            assert_eq!(result.operands.updated_base, Some(0x0100_0004));
        }
        machine.set_spr_value(90, 1).unwrap();
        let byte_store = C220AtomicStoreOperands::capture(&machine, 0, 0x1602_2000).unwrap();
        assert_eq!(
            byte_store
                .evaluate(&[0xff], C220Fp16AddRounding::NearestEven)
                .unwrap()
                .stored_bytes(),
            &[0x34]
        );
    }
}

/// Functional data result. It neither retires the request nor updates GPRs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220AtomicStoreResult {
    pub operands: C220AtomicStoreOperands,
    pub address: u64,
    pub previous: [u8; 8],
    pub stored: [u8; 8],
}

impl C220AtomicStoreResult {
    pub fn previous_bytes(&self) -> &[u8] {
        &self.previous[..usize::from(self.operands.instruction.width_bytes)]
    }

    pub fn stored_bytes(&self) -> &[u8] {
        &self.stored[..usize::from(self.operands.instruction.width_bytes)]
    }
}

/// Atomic store inputs, captured before requests enter the LSU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220AtomicStoreOperands {
    pub pc: u64,
    pub word: u32,
    instruction: C220ScalarAtomicStore,
    pub base_value: u64,
    pub source_value: u64,
    pub offset_value: u64,
    pub effective_address: u64,
    pub updated_base: Option<u64>,
    pub control: u64,
    pub atomic_control: u64,
    pub local_root: u64,
    bytes: [u8; 8],
}

impl C220AtomicStoreOperands {
    /// Combine complete lanes only. Type controls are independent of transfer
    /// width; an incomplete lane retains the source register's bytes.
    pub fn evaluate(
        self,
        previous: &[u8],
        rounding: C220Fp16AddRounding,
    ) -> Result<C220AtomicStoreResult, C220AtomicStoreError> {
        if !self.is_external() {
            return Err(C220AtomicStoreError::LocalAddress(self.effective_address));
        }
        let width = usize::from(self.instruction.width_bytes);
        if previous.len() != width {
            return Err(C220AtomicStoreError::InputSize);
        }
        let mut result = C220AtomicStoreResult {
            operands: self,
            address: self.effective_address & 0x0000_ffff_ffff_ffff,
            previous: [0; 8],
            stored: [0; 8],
        };
        result.previous[..width].copy_from_slice(previous);
        result.stored[..width].copy_from_slice(self.bytes());
        combine_atomic(
            &mut result.stored[..width],
            previous,
            self.data_type(),
            0,
            self.control,
            rounding,
        );
        Ok(result)
    }

    /// The exclusive memory borrow serializes the functional read-modify-write.
    /// Cache data actions and timing completion are separate from this write.
    pub fn execute(
        self,
        memory: &mut MappedMemory,
        rounding: C220Fp16AddRounding,
    ) -> Result<C220AtomicStoreResult, C220AtomicStoreError> {
        if !self.is_external() {
            return Err(C220AtomicStoreError::LocalAddress(self.effective_address));
        }
        let address = self.effective_address & 0x0000_ffff_ffff_ffff;
        let previous = memory.read_known_at(address, usize::from(self.instruction.width_bytes))?;
        let result = self.evaluate(&previous, rounding)?;
        memory.write_known_at(address, result.stored_bytes())?;
        Ok(result)
    }

    pub fn capture(
        machine: &ScalarMachine,
        pc: u64,
        word: u32,
    ) -> Result<Self, ScalarMachineError> {
        let instruction = C220ScalarAtomicStore::decode(word)
            .filter(|_| machine.architecture() == Architecture::Dav2201)
            .ok_or(ScalarMachineError::UnsupportedWord { pc, word })?;
        let base_value = machine.xregs()[usize::from(instruction.base_register)];
        let source_value = machine.xregs()[usize::from(instruction.source_register)];
        let offset_value = match instruction.offset {
            C220AtomicStoreOffset::Immediate(value) => value as i64 as u64,
            C220AtomicStoreOffset::Register(register) => machine.xregs()[usize::from(register)]
                .wrapping_mul(u64::from(instruction.width_bytes)),
        };
        let adjusted = base_value.wrapping_add(offset_value);
        let spr = |spr| {
            machine
                .spr_value(spr)
                .ok_or(ScalarMachineError::SprValueUnavailable { pc, spr })
        };
        Ok(Self {
            pc,
            word,
            instruction,
            base_value,
            source_value,
            offset_value,
            effective_address: if instruction.post_index {
                base_value
            } else {
                adjusted
            },
            updated_base: instruction.post_index.then_some(adjusted),
            control: spr(3)?,
            atomic_control: spr(90)?,
            local_root: spr(67)?,
            bytes: source_value.to_le_bytes(),
        })
    }

    pub fn instruction(&self) -> C220ScalarAtomicStore {
        self.instruction
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes[..usize::from(self.instruction.width_bytes)]
    }

    pub const fn data_type(&self) -> u8 {
        (self.atomic_control & 7) as u8
    }

    pub const fn is_external(&self) -> bool {
        self.effective_address & (1 << 24) != 0
            || (self.effective_address & 0x0001_ffff_fe00_0000)
                != (self.local_root & 0x0001_ffff_fe00_0000)
    }
}
