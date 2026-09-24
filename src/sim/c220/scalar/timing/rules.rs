use super::{
    C220ScalarTimingClass, C220ScalarTimingTicket, SCALAR_CONVERSION_EXECUTION_STAGE,
    SCALAR_CONVERSION_LATENCY_TICKS,
};
use crate::architecture::Architecture;
use crate::isa::c220::scalar::C220ScalarConversionHint;
use crate::isa::scalar::{ScalarInstruction, ScalarKey0Operation, ScalarKey8Operation};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220ScalarTimingRule {
    pub class: C220ScalarTimingClass,
    pub latency_ticks: u64,
    pub execution_stage: u8,
    pub source_register: u8,
    pub destination_register: u8,
}

impl C220ScalarTimingRule {
    pub fn decode(word: u32) -> Option<Self> {
        if let Some(hint) = C220ScalarConversionHint::from_word(word) {
            return Some(Self {
                class: C220ScalarTimingClass::Fixed,
                latency_ticks: SCALAR_CONVERSION_LATENCY_TICKS,
                execution_stage: SCALAR_CONVERSION_EXECUTION_STAGE,
                source_register: hint.source_register,
                destination_register: hint.destination_register,
            });
        }
        let instruction = ScalarInstruction::from_word(Architecture::Dav2201, word)?;
        if let ScalarInstruction::ScalarKey2IntegerSqrt {
            dtype_field,
            source_register,
            destination_register,
        } = instruction
        {
            return Some(Self {
                class: C220ScalarTimingClass::Variable,
                latency_ticks: if dtype_field == 2 { 18 } else { 15 },
                execution_stage: if dtype_field == 0 { 10 } else { 18 },
                source_register,
                destination_register,
            });
        }
        if let Some(rule) = Self::register_operation(instruction) {
            return Some(rule);
        }
        let ScalarInstruction::ScalarKey0 {
            operation,
            dtype_field,
            first_source_register,
            destination_register,
            ..
        } = instruction
        else {
            return None;
        };
        let (latency_ticks, execution_stage) = match operation {
            ScalarKey0Operation::Add
            | ScalarKey0Operation::Subtract
            | ScalarKey0Operation::Multiply
            | ScalarKey0Operation::MultiplyAdd => {
                let latency = match (operation, dtype_field) {
                    (ScalarKey0Operation::Multiply, 0) => 2,
                    (ScalarKey0Operation::MultiplyAdd, 0) => 3,
                    (_, 2) => 5,
                    _ => 1,
                };
                let stage = if dtype_field == 0 && operation != ScalarKey0Operation::MultiplyAdd {
                    1
                } else {
                    3
                };
                (latency, stage)
            }
            ScalarKey0Operation::Minimum
            | ScalarKey0Operation::Maximum
            | ScalarKey0Operation::And
            | ScalarKey0Operation::Or
            | ScalarKey0Operation::Xor => (1, 1),
            ScalarKey0Operation::Divide | ScalarKey0Operation::Remainder => (
                if dtype_field == 2 { 14 } else { 20 },
                if operation == ScalarKey0Operation::Divide && dtype_field != 0 {
                    14
                } else {
                    20
                },
            ),
        };
        Some(Self {
            class: if matches!(
                operation,
                ScalarKey0Operation::Divide | ScalarKey0Operation::Remainder
            ) {
                C220ScalarTimingClass::Variable
            } else {
                C220ScalarTimingClass::Fixed
            },
            latency_ticks,
            execution_stage,
            source_register: first_source_register,
            destination_register,
        })
    }

    fn register_operation(instruction: ScalarInstruction) -> Option<Self> {
        let (source_register, destination_register, execution_stage) = match instruction {
            ScalarInstruction::ScalarKey2MoveRegister {
                source_register,
                destination_register,
                ..
            }
            | ScalarInstruction::ScalarKey2Negate {
                source_register,
                destination_register,
                ..
            }
            | ScalarInstruction::ScalarKey2Absolute {
                source_register,
                destination_register,
                ..
            }
            | ScalarInstruction::ScalarKey2BitwiseNot {
                source_register,
                destination_register,
                ..
            }
            | ScalarInstruction::ScalarKey2ZeroExtend {
                source_register,
                destination_register,
                ..
            }
            | ScalarInstruction::ScalarKey2SignExtend {
                source_register,
                destination_register,
                ..
            }
            | ScalarInstruction::ScalarKey2FindFirst {
                source_register,
                destination_register,
                ..
            }
            | ScalarInstruction::ScalarKey2Insert {
                source_register,
                destination_register,
                ..
            }
            | ScalarInstruction::ScalarKey2BitSet {
                source_register,
                destination_register,
                ..
            } => (source_register, destination_register, 1),
            ScalarInstruction::ScalarKey2ShiftLeft {
                destination_register,
                ..
            }
            | ScalarInstruction::ScalarKey2ShiftRight {
                destination_register,
                ..
            }
            | ScalarInstruction::ScalarKey2InsertImmediate {
                destination_register,
                ..
            } => (destination_register, destination_register, 1),
            ScalarInstruction::ScalarKey8 {
                operation,
                source_register,
                destination_register: Some(destination_register),
                ..
            } => (
                source_register,
                destination_register,
                if operation == ScalarKey8Operation::MultiplyImmediate {
                    2
                } else {
                    1
                },
            ),
            _ => return None,
        };
        Some(Self {
            class: C220ScalarTimingClass::Fixed,
            latency_ticks: 1,
            execution_stage,
            source_register,
            destination_register,
        })
    }

    pub fn ticket(self, issue_tick: u64) -> Option<C220ScalarTimingTicket> {
        Some(C220ScalarTimingTicket {
            class: self.class,
            issue_tick,
            retire_tick: issue_tick.checked_add(self.latency_ticks)?,
            execution_stage: self.execution_stage,
            source_register: self.source_register,
            destination_register: self.destination_register,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_timing_distinguishes_retirement_from_execution_stage() {
        for (opcode, stage) in [
            (0x0200_0800, 1),
            (0x0200_0080, 1),
            (0x0200_0100, 1),
            (0x02c0_0180, 1),
            (0x0800_0000, 1),
            (0x0840_0000, 2),
            (0x0880_0000, 1),
        ] {
            let word = opcode | (1 << 17) | (2 << 12);
            let rule = C220ScalarTimingRule::decode(word).unwrap();
            assert_eq!((rule.source_register, rule.destination_register), (2, 1));
            let ticket = rule.ticket(10).unwrap();
            assert_eq!((ticket.retire_tick, ticket.execution_stage), (11, stage));
            assert_eq!(rule.ticket(u64::MAX), None);
        }
        assert_eq!(C220ScalarTimingRule::decode(0x08c0_0000), None);
    }
}
