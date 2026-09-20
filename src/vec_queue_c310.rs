use crate::architecture::Architecture;
use serde::Serialize;
use thiserror::Error;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct C310RvecAdmissionCounters {
    pub simd_issues: u32,
    pub simt_issues: u32,
    pub simd_executions: u32,
    pub simt_executions: u32,
    pub sfu_executions: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct C310SimdGateBlockers {
    pub simd_issue: bool,
    pub simt_issue: bool,
    pub simt_execution: bool,
    pub sfu_execution: bool,
}

impl C310SimdGateBlockers {
    pub const fn any(self) -> bool {
        self.simd_issue || self.simt_issue || self.simt_execution || self.sfu_execution
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct C310RvecAdmissionStep {
    pub before: C310RvecAdmissionCounters,
    pub after: C310RvecAdmissionCounters,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum C310RvecAdmissionError {
    #[error("SIMD admission gate is blocked")]
    GateBlocked(C310SimdGateBlockers),
    #[error("SIMD execution count would overflow")]
    ExecutionCountOverflow,
    #[error("SIMD issue completion has no outstanding issue")]
    IssueUnderflow,
    #[error("SIMD execution completion has no outstanding execution")]
    ExecutionUnderflow,
}

impl C310RvecAdmissionCounters {
    pub const fn simd_gate_blockers(self) -> C310SimdGateBlockers {
        C310SimdGateBlockers {
            simd_issue: self.simd_issues != 0,
            simt_issue: self.simt_issues != 0,
            simt_execution: self.simt_executions != 0,
            sfu_execution: self.sfu_executions != 0,
        }
    }

    pub const fn can_attempt_simd_push(self) -> bool {
        !self.simd_gate_blockers().any()
    }

    pub fn accept_simd_push(&mut self) -> Result<C310RvecAdmissionStep, C310RvecAdmissionError> {
        let blockers = self.simd_gate_blockers();
        if blockers.any() {
            return Err(C310RvecAdmissionError::GateBlocked(blockers));
        }
        let executions = self
            .simd_executions
            .checked_add(1)
            .ok_or(C310RvecAdmissionError::ExecutionCountOverflow)?;
        let before = *self;
        self.simd_issues = 1;
        self.simd_executions = executions;
        Ok(C310RvecAdmissionStep {
            before,
            after: *self,
        })
    }

    pub fn complete_simd_issue(&mut self) -> Result<C310RvecAdmissionStep, C310RvecAdmissionError> {
        let issues = self
            .simd_issues
            .checked_sub(1)
            .ok_or(C310RvecAdmissionError::IssueUnderflow)?;
        let before = *self;
        self.simd_issues = issues;
        Ok(C310RvecAdmissionStep {
            before,
            after: *self,
        })
    }

    pub fn complete_simd_execution(
        &mut self,
    ) -> Result<C310RvecAdmissionStep, C310RvecAdmissionError> {
        let executions = self
            .simd_executions
            .checked_sub(1)
            .ok_or(C310RvecAdmissionError::ExecutionUnderflow)?;
        let before = *self;
        self.simd_executions = executions;
        Ok(C310RvecAdmissionStep {
            before,
            after: *self,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum C310VfQueueDisposition {
    Accepted,
    Stalled,
    Unsupported,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct C310VfQueueInstruction {
    pub words: [u32; 2],
    pub vendor_isa_name: u16,
    pub vector_pc_register: u8,
    pub encoded_field: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
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
            vendor_isa_name: 262,
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
    fn simd_gate_distinguishes_issue_execution_and_sfu_activity() {
        let mut counters = C310RvecAdmissionCounters::default();
        assert!(counters.can_attempt_simd_push());
        let accepted = counters.accept_simd_push().unwrap();
        assert_eq!(accepted.before, C310RvecAdmissionCounters::default());
        assert_eq!(accepted.after.simd_issues, 1);
        assert_eq!(accepted.after.simd_executions, 1);
        assert!(counters.simd_gate_blockers().simd_issue);
        counters.complete_simd_issue().unwrap();
        assert!(counters.can_attempt_simd_push());
        assert_eq!(counters.simd_executions, 1);
        counters.sfu_executions = 1;
        assert_eq!(
            counters.accept_simd_push(),
            Err(C310RvecAdmissionError::GateBlocked(C310SimdGateBlockers {
                simd_issue: false,
                simt_issue: false,
                simt_execution: false,
                sfu_execution: true,
            }))
        );
        assert_eq!(counters.simd_executions, 1);
        counters.sfu_executions = 0;
        counters.complete_simd_execution().unwrap();
        assert_eq!(counters, C310RvecAdmissionCounters::default());
    }

    #[test]
    fn simd_gate_rejects_other_active_paths_without_mutation() {
        for counters in [
            C310RvecAdmissionCounters {
                simd_issues: 1,
                ..Default::default()
            },
            C310RvecAdmissionCounters {
                simt_issues: 1,
                ..Default::default()
            },
            C310RvecAdmissionCounters {
                simt_executions: 1,
                ..Default::default()
            },
        ] {
            let mut state = counters;
            assert!(matches!(
                state.accept_simd_push(),
                Err(C310RvecAdmissionError::GateBlocked(_))
            ));
            assert_eq!(state, counters);
        }
    }

    #[test]
    fn simd_counter_errors_are_atomic() {
        let mut counters = C310RvecAdmissionCounters {
            simd_executions: u32::MAX,
            ..Default::default()
        };
        assert_eq!(
            counters.accept_simd_push(),
            Err(C310RvecAdmissionError::ExecutionCountOverflow)
        );
        assert_eq!(counters.simd_issues, 0);
        assert_eq!(counters.simd_executions, u32::MAX);
        counters.simd_executions = 0;
        assert_eq!(
            counters.complete_simd_issue(),
            Err(C310RvecAdmissionError::IssueUnderflow)
        );
        assert_eq!(
            counters.complete_simd_execution(),
            Err(C310RvecAdmissionError::ExecutionUnderflow)
        );
        assert_eq!(counters, C310RvecAdmissionCounters::default());
    }

    #[test]
    fn vf_queue_decodes_two_words_and_resolves_vector_pc() {
        let first_word = 0x154d_0000;
        let second_word = 0x15e0_0105;
        let instruction =
            C310VfQueueInstruction::decode(Architecture::Dav3510, first_word, second_word).unwrap();
        assert_eq!(instruction.words, [first_word, second_word]);
        assert_eq!(instruction.vendor_isa_name, 262);
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
