#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MovSmaskInstruction {
    pub word: u32,
    pub source_mode: u8,
    pub destination_register: u8,
    pub source_register: u8,
    pub descriptor_register: u8,
}

impl C220MovSmaskInstruction {
    pub const fn decode(word: u32) -> Option<Self> {
        if word >> 29 != 3 || (word >> 27) & 3 != 0 || (word >> 22) & 31 != 17 {
            return None;
        }
        Some(Self {
            word,
            source_mode: (word & 3) as u8,
            destination_register: ((word >> 17) & 31) as u8,
            source_register: ((word >> 12) & 31) as u8,
            descriptor_register: ((word >> 2) & 31) as u8,
        })
    }

    pub const fn capture(self, registers: &[u64; 32]) -> C220SmaskTransfer {
        C220SmaskTransfer {
            instruction: self,
            source_base: registers[self.source_register as usize],
            destination_base: registers[self.destination_register as usize],
            descriptor: C220SmaskDescriptor::decode(registers[self.descriptor_register as usize]),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220SmaskDescriptor {
    pub raw: u64,
    pub count: u8,
    pub control_bits: u8,
}

impl C220SmaskDescriptor {
    pub const fn decode(raw: u64) -> Self {
        Self {
            raw,
            count: ((raw & 127) | ((raw >> 4) & 128)) as u8,
            control_bits: ((raw >> 7) & 15) as u8,
        }
    }

    pub const fn is_empty(self) -> bool {
        self.count == 0
    }

    pub const fn bytes(self) -> u32 {
        self.count as u32 * 2
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220SmaskTransfer {
    pub instruction: C220MovSmaskInstruction,
    pub source_base: u64,
    pub destination_base: u64,
    pub descriptor: C220SmaskDescriptor,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_uses_xt_and_separates_count_from_control_bits() {
        for mode in 0..4 {
            let word = (3 << 29) | (17 << 22) | (5 << 17) | (7 << 12) | (9 << 7) | (11 << 2) | mode;
            let instruction = C220MovSmaskInstruction::decode(word).unwrap();
            assert_eq!(instruction.source_mode, mode as u8);
            assert_eq!(
                super::super::read_register_mask(word),
                Some((1 << 5) | (1 << 7) | (1 << 11))
            );
            let mut registers = [0; 32];
            registers[5] = 511;
            registers[7] = 4096;
            registers[9] = u64::MAX;
            for count in [0_u64, 1, 127, 128, 255] {
                registers[11] = (count & 127) | ((count & 128) << 4) | (9 << 7);
                let transfer = instruction.capture(&registers);
                assert_eq!(transfer.source_base, 4096);
                assert_eq!(transfer.destination_base, 511);
                assert_eq!(transfer.descriptor.count, count as u8);
                assert_eq!(transfer.descriptor.control_bits, 9);
                assert_eq!(transfer.descriptor.bytes(), count as u32 * 2);
                assert_eq!(transfer.descriptor.is_empty(), count == 0);
            }
            assert!(C220MovSmaskInstruction::decode(word ^ (1 << 22)).is_none());
        }
    }
}
