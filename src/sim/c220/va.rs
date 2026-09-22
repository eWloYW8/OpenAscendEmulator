use crate::isa::c220::vector::C220MoveVaInstruction;
use crate::sim::c220::vector::load_va::C220VaUpdate;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct C220VaRegisters {
    entries: [[Option<u16>; 8]; 8],
}

impl C220VaRegisters {
    pub fn entry(&self, register: u8, index: u8) -> Option<u16> {
        self.entries
            .get(usize::from(register))?
            .get(usize::from(index))
            .copied()
            .flatten()
    }

    pub fn write_pair(&mut self, instruction: C220MoveVaInstruction, xregs: &[u64; 32]) {
        let destination = &mut self.entries[usize::from(instruction.destination_va)];
        let index = usize::from(instruction.first_entry);
        destination[index] =
            Some(((xregs[usize::from(instruction.source_0_register)] >> 5) & 0x1fff) as u16);
        destination[index + 1] =
            Some(((xregs[usize::from(instruction.source_1_register)] >> 5) & 0x1fff) as u16);
    }

    pub fn apply(&mut self, update: C220VaUpdate) {
        self.entries[usize::from(update.destination_va)] = update.entries.map(Some);
    }
}
