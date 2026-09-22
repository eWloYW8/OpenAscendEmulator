use super::layout::C310_PB_PUSH_BYTES;
use crate::architecture::Architecture;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C310PushPbInstruction {
    pub word: u32,
    pub source_registers: [u8; 4],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C310PushPbStep {
    pub pc: u64,
    pub instruction: C310PushPbInstruction,
    pub source_values: [u64; 4],
    pub bytes: [u8; C310_PB_PUSH_BYTES],
}

impl C310PushPbInstruction {
    pub const fn decode(architecture: Architecture, word: u32) -> Option<Self> {
        if !matches!(architecture, Architecture::Dav3510)
            || !matches!(word, 0x4314_b108 | 0x4338_2108 | 0x4319_7108)
        {
            return None;
        }
        Some(Self {
            word,
            source_registers: [
                ((word >> 17) & 0x1f) as u8,
                ((word >> 12) & 0x1f) as u8,
                ((word >> 7) & 0x1f) as u8,
                ((word >> 2) & 0x1f) as u8,
            ],
        })
    }

    pub fn from_source_values(self, pc: u64, source_values: [u64; 4]) -> C310PushPbStep {
        let mut bytes = [0; C310_PB_PUSH_BYTES];
        for (chunk, value) in bytes.chunks_exact_mut(8).zip(source_values) {
            chunk.copy_from_slice(&value.to_le_bytes());
        }
        C310PushPbStep {
            pc,
            instruction: self,
            source_values,
            bytes,
        }
    }

    pub fn resolve(self, pc: u64, xregs: &[u64; 32]) -> C310PushPbStep {
        self.from_source_values(
            pc,
            self.source_registers.map(|index| xregs[usize::from(index)]),
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C310VfQueueInstruction {
    pub words: [u32; 2],
    pub vector_pc_register: u8,
    pub encoded_field: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C310VfQueueStep {
    pub pc: u64,
    pub instruction: C310VfQueueInstruction,
    pub vector_pc: u64,
}

impl C310VfQueueInstruction {
    pub const fn is_prefix(architecture: Architecture, first_word: u32) -> bool {
        matches!(architecture, Architecture::Dav3510) && first_word & 0xff80_0000 == 0x1500_0000
    }

    pub const fn decode(
        architecture: Architecture,
        first_word: u32,
        second_word: u32,
    ) -> Option<Self> {
        if !Self::is_prefix(architecture, first_word) || second_word & 0x1f != 5 {
            return None;
        }
        Some(Self {
            words: [first_word, second_word],
            vector_pc_register: ((first_word >> 16) & 0x1f) as u8,
            encoded_field: ((second_word >> 5) & 0xffff) as u16,
        })
    }

    pub fn resolve(self, pc: u64, xregs: &[u64; 32]) -> C310VfQueueStep {
        C310VfQueueStep {
            pc,
            instruction: self,
            vector_pc: xregs[usize::from(self.vector_pc_register)],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vf_queue_decodes_two_words_and_resolves_vector_pc() {
        let first_word = 0x154d_0000;
        let second_word = 0x15e0_0105;
        let instruction =
            C310VfQueueInstruction::decode(Architecture::Dav3510, first_word, second_word).unwrap();
        assert_eq!(instruction.words, [first_word, second_word]);
        assert_eq!(instruction.vector_pc_register, 13);
        assert_eq!(instruction.encoded_field, 8);
        let mut xregs = [0; 32];
        xregs[13] = 0x10d0_d900;
        assert_eq!(
            instruction.resolve(0x10d0_d7fc, &xregs).vector_pc,
            0x10d0_d900
        );
    }

    #[test]
    fn vf_queue_resolves_distinct_registers_and_fields() {
        let mut xregs = [0; 32];
        xregs[1] = 0x10d0_db00;
        xregs[2] = 0x10d0_d900;
        for (pc, words, register, field, vector_pc) in [
            (0x10d0_d57c, [0x1541_0000, 0x15e0_00c5], 1, 6, xregs[1]),
            (0x10d0_d718, [0x1542_0000, 0x15e0_0125], 2, 9, xregs[2]),
        ] {
            let instruction =
                C310VfQueueInstruction::decode(Architecture::Dav3510, words[0], words[1]).unwrap();
            assert_eq!(instruction.words, words);
            assert_eq!(instruction.vector_pc_register, register);
            assert_eq!(instruction.encoded_field, field);
            assert_eq!(instruction.resolve(pc, &xregs).vector_pc, vector_pc);
        }
    }

    #[test]
    fn vf_queue_rejects_other_architecture_or_instruction_form() {
        let first_word = 0x154d_0000;
        let second_word = 0x15e0_0105;
        assert!(C310VfQueueInstruction::is_prefix(
            Architecture::Dav3510,
            first_word
        ));
        assert!(
            C310VfQueueInstruction::decode(Architecture::Dav2201, first_word, second_word)
                .is_none()
        );
        assert!(
            C310VfQueueInstruction::decode(
                Architecture::Dav3510,
                first_word | 0x800000,
                second_word
            )
            .is_none()
        );
        assert!(
            C310VfQueueInstruction::decode(Architecture::Dav3510, first_word, second_word ^ 1)
                .is_none()
        );
    }
}
