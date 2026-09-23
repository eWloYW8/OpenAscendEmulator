use crate::architecture::Architecture;
use crate::isa::c220::cube::C220CubeInstruction;
use crate::isa::c220::scalar::C220ScalarConversionHint;
use crate::isa::c220::vector::scalar::C220VectorScalarInstruction;
use crate::isa::c220::vector::{
    C220BroadcastInstruction, C220CopyInstruction, C220MovevInstruction, C220ShiftInstruction,
    C220TransposeInstruction, C220VecArithmeticHint,
};
use crate::isa::scalar::{
    ScalarInstruction, ScalarKey0Operation, ScalarKey7Operation, ScalarLoadStoreOperation,
};
use crate::sim::common::scalar::SCALAR_X_REGISTER_COUNT;

pub const SCALAR_CONVERSION_LATENCY_TICKS: u64 = 2;
pub const SCALAR_CONVERSION_EXECUTION_STAGE: u8 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220ScalarTimingTicket {
    pub issue_tick: u64,
    pub retire_tick: u64,
    pub execution_stage: u8,
    pub source_register: u8,
    pub destination_register: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct C220ScalarTimingLane {
    pending_xreg_retirement: [Option<u64>; SCALAR_X_REGISTER_COUNT],
}

impl C220ScalarTimingLane {
    pub fn advance_to(&mut self, tick: u64) {
        for retirement in &mut self.pending_xreg_retirement {
            if retirement.is_some_and(|retire_tick| retire_tick <= tick) {
                *retirement = None;
            }
        }
    }

    pub fn dependency_tick(&self, word: u32, tick: u64) -> Option<u64> {
        let mut resume_tick = None;
        let mut include = |register: u8| {
            if let Some(retire_tick) = self
                .pending_xreg_retirement(register)
                .filter(|retire_tick| tick < *retire_tick)
            {
                resume_tick =
                    Some(resume_tick.map_or(retire_tick, |prior: u64| prior.max(retire_tick)));
            }
        };
        if let Some(instruction) =
            crate::isa::c220::mte::factor::C220FactorLoadInstruction::decode(word)
        {
            include(instruction.destination_register);
            include(instruction.source_register);
            include(instruction.descriptor_register);
            return resume_tick;
        }
        if let Some(instruction) = crate::isa::c220::mte::fixp::C220FixpInstruction::decode(word) {
            include(instruction.destination_register);
            include(instruction.source_register);
            include(instruction.shape_register);
            include(instruction.control_register);
            return resume_tick;
        }
        if let Some(instruction) = C220CubeInstruction::decode(word) {
            include(instruction.xd);
            include(instruction.xn);
            include(instruction.xm);
            include(instruction.xt);
            return resume_tick;
        }
        if let Some(instruction) = C220VectorScalarInstruction::decode(word) {
            include(instruction.destination_register);
            include(instruction.source_register);
            include(instruction.scalar_register);
            include(instruction.control_register);
            return resume_tick;
        }
        if let Some(instruction) = C220MovevInstruction::decode(word) {
            include(instruction.destination_register);
            include(instruction.source_register);
            include(instruction.control_register);
            return resume_tick;
        }
        if let Some(instruction) = C220ShiftInstruction::decode(word) {
            include(instruction.destination_register);
            include(instruction.source_register);
            include(instruction.shift_register);
            include(instruction.control_register);
            return resume_tick;
        }
        if let Some(instruction) = C220CopyInstruction::decode(word) {
            include(instruction.destination_register);
            include(instruction.source_register);
            include(instruction.control_register);
            return resume_tick;
        }
        if let Some(instruction) = C220BroadcastInstruction::decode(word) {
            include(instruction.destination_register);
            include(instruction.source_register);
            include(instruction.control_register);
            return resume_tick;
        }
        if let Some(instruction) = C220TransposeInstruction::decode(word) {
            include(instruction.destination_register);
            include(instruction.source_register);
            return resume_tick;
        }
        if let Some(instruction) = C220VecArithmeticHint::from_word(word) {
            include(instruction.destination_register);
            include(instruction.source_0_register);
            if let Some(register) = instruction.source_1_register {
                include(register);
            }
            include(instruction.control_register);
            return resume_tick;
        }
        if let Some(instruction) = C220ScalarConversionHint::from_word(word) {
            include(instruction.source_register);
            return resume_tick;
        }
        let hint = ScalarInstruction::from_word(Architecture::Dav2201, word)?;
        match hint {
            ScalarInstruction::ScalarIndexedLoad {
                base_register,
                offset_register,
                ..
            }
            | ScalarInstruction::ScalarIndexedImmediateStore {
                base_register,
                offset_register,
                ..
            } => {
                include(base_register);
                include(offset_register);
            }
            ScalarInstruction::ScalarPairLoad { base_register, .. }
            | ScalarInstruction::ScalarStoreImmediate { base_register, .. } => {
                include(base_register);
            }
            ScalarInstruction::ScalarPairStore {
                first_source_register,
                second_source_register,
                base_register,
                ..
            } => {
                include(first_source_register);
                include(second_source_register);
                include(base_register);
            }
            ScalarInstruction::ScalarLoadStoreImmediate {
                operation,
                data_register,
                base_register,
                ..
            } => {
                include(base_register);
                if operation == ScalarLoadStoreOperation::Store {
                    include(data_register);
                }
            }
            ScalarInstruction::ScalarKey0 {
                operation,
                destination_register,
                first_source_register,
                second_source_register,
                ..
            } => {
                include(first_source_register);
                include(second_source_register);
                if operation == ScalarKey0Operation::MultiplyAdd {
                    include(destination_register);
                }
            }
            ScalarInstruction::ScalarCompare {
                first_source_register,
                second_source_register,
                ..
            }
            | ScalarInstruction::ScalarCompareRegister {
                first_source_register,
                second_source_register,
                ..
            }
            | ScalarInstruction::ScalarSelect {
                first_source_register,
                second_source_register,
                ..
            } => {
                include(first_source_register);
                include(second_source_register);
            }
            ScalarInstruction::ScalarCompareImmediate {
                source_register, ..
            }
            | ScalarInstruction::ScalarKey2MoveRegister {
                source_register, ..
            }
            | ScalarInstruction::ScalarKey2Negate {
                source_register, ..
            }
            | ScalarInstruction::ScalarKey2Absolute {
                source_register, ..
            }
            | ScalarInstruction::ScalarKey2IntegerSqrt {
                source_register, ..
            }
            | ScalarInstruction::ScalarKey2BitwiseNot {
                source_register, ..
            }
            | ScalarInstruction::ScalarKey2MoveToSpr {
                source_register, ..
            }
            | ScalarInstruction::ScalarKey2ZeroExtend {
                source_register, ..
            }
            | ScalarInstruction::ScalarKey2SignExtend {
                source_register, ..
            }
            | ScalarInstruction::ScalarKey2FindFirst {
                source_register, ..
            }
            | ScalarInstruction::ScalarKey8 {
                source_register, ..
            } => include(source_register),
            ScalarInstruction::ScalarKey2ShiftLeft {
                destination_register,
                count_register,
                ..
            }
            | ScalarInstruction::ScalarKey2ShiftRight {
                destination_register,
                count_register,
                ..
            } => {
                include(destination_register);
                if let Some(register) = count_register {
                    include(register);
                }
            }
            ScalarInstruction::ScalarKey2Insert {
                destination_register,
                source_register,
                ..
            }
            | ScalarInstruction::ScalarKey2BitSet {
                destination_register,
                source_register,
                ..
            } => {
                include(destination_register);
                include(source_register);
            }
            ScalarInstruction::ScalarKey2InsertImmediate {
                destination_register,
                ..
            } => include(destination_register),
            ScalarInstruction::ScalarKey7 {
                operation: ScalarKey7Operation::MoveKeep,
                destination_register,
                ..
            } => include(destination_register),
            ScalarInstruction::ScalarMoveX8Immediate { .. }
            | ScalarInstruction::ScalarKey2MoveFromSpr { .. }
            | ScalarInstruction::ScalarKey7 { .. } => {}
        }
        resume_tick
    }

    pub(crate) fn issue(&mut self, ticket: C220ScalarTimingTicket) {
        self.pending_xreg_retirement[usize::from(ticket.destination_register)] =
            Some(ticket.retire_tick);
    }

    pub fn pending_xreg_retirement(&self, register: u8) -> Option<u64> {
        self.pending_xreg_retirement
            .get(usize::from(register))
            .copied()
            .flatten()
    }

    pub fn pending_drain_tick(&self) -> Option<u64> {
        self.pending_xreg_retirement.iter().flatten().copied().max()
    }
}

impl C220ScalarTimingTicket {
    pub fn for_conversion(issue_tick: u64, hint: C220ScalarConversionHint) -> Option<Self> {
        Some(Self {
            issue_tick,
            retire_tick: issue_tick.checked_add(SCALAR_CONVERSION_LATENCY_TICKS)?,
            execution_stage: SCALAR_CONVERSION_EXECUTION_STAGE,
            source_register: hint.source_register,
            destination_register: hint.destination_register,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_issue_waits_for_scalar_register_retirement() {
        let mut lane = C220ScalarTimingLane::default();
        lane.issue(C220ScalarTimingTicket {
            issue_tick: 0,
            retire_tick: 4,
            execution_stage: 2,
            source_register: 1,
            destination_register: 6,
        });
        for opcode in [0x9700_0000, 0x9700_0001] {
            let word = opcode | (3 << 17) | (4 << 12) | (6 << 7) | (5 << 2);
            assert_eq!(lane.dependency_tick(word, 1), Some(4));
            assert_eq!(lane.dependency_tick(word, 4), None);
        }
        for word in [
            (6 << 29) | (3 << 17) | (4 << 12) | (6 << 7),
            (6 << 29) | (3 << 24) | (3 << 17) | (4 << 12) | (5 << 7) | (6 << 2),
            0x9c80_0003 | (3 << 17) | (4 << 12) | (6 << 7) | (5 << 2),
            0x8240_0700 | (3 << 17) | (6 << 12) | (5 << 2),
        ] {
            assert_eq!(lane.dependency_tick(word, 1), Some(4));
            assert_eq!(lane.dependency_tick(word, 4), None);
        }
    }
}
