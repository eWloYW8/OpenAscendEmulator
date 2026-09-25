use crate::architecture::Architecture;
use crate::isa::c220::cube::C220CubeInstruction;
use crate::isa::c220::scalar::{
    C220AtomicStoreOffset, C220PreloadOffset, C220ScalarAtomicStore, C220ScalarConversionHint,
    C220ScalarDirectStore, C220ScalarPreload,
};
use crate::isa::c220::vector::scalar::C220VectorScalarInstruction;
use crate::isa::c220::vector::{
    C220BroadcastInstruction, C220CopyInstruction, C220MovevInstruction, C220ShiftInstruction,
    C220TransposeInstruction, C220VecArithmeticHint,
};
use crate::isa::flow::{
    ConditionalJump, DcciInstruction, JumpCompare, JumpCompareOffset, JumpCompareOperand,
    JumpOffsetSource, UnconditionalJump,
};
use crate::isa::scalar::{
    ScalarInstruction, ScalarKey0Operation, ScalarKey7Operation, ScalarLoadStoreOperation,
};

mod rules;
use super::spr::C220ScalarSprTimingTicket;
pub use rules::C220ScalarTimingRule;
use std::collections::{BTreeMap, VecDeque};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220ScalarTimingClass {
    Fixed,
    Variable,
}

pub const SCALAR_CONVERSION_LATENCY_TICKS: u64 = 2;
pub const SCALAR_CONVERSION_EXECUTION_STAGE: u8 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220ScalarTimingTicket {
    pub class: C220ScalarTimingClass,
    pub issue_tick: u64,
    /// Scheduled notification time. Variable-class notifications retire the FIFO head.
    pub retire_tick: u64,
    pub execution_stage: u8,
    pub source_register: Option<u8>,
    pub destination_register: Option<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct C220ScalarTimingLane {
    pending_xreg_retirement: BTreeMap<u8, u64>,
    fixed_drain_tick: Option<u64>,
    variable_instructions: VecDeque<PendingVariable>,
    variable_events: Vec<u64>,
    pending_spr_retirement: BTreeMap<u16, u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PendingVariable {
    ticket: C220ScalarTimingTicket,
    destination_live: bool,
}

impl C220ScalarTimingLane {
    pub fn advance_to(&mut self, tick: u64) {
        if self
            .fixed_drain_tick
            .is_some_and(|retirement| retirement <= tick)
        {
            self.fixed_drain_tick = None;
        }
        self.pending_spr_retirement
            .retain(|_, retirement| *retirement > tick);
        let ready = self.variable_events.partition_point(|event| *event <= tick);
        self.variable_events.drain(..ready);
        self.variable_instructions.drain(..ready);
        self.pending_xreg_retirement
            .retain(|_, retirement| *retirement > tick);
    }

    pub fn dependency_tick(&self, word: u32, tick: u64) -> Option<u64> {
        self.dependency_tick_with_loads(word, tick, |_| false)
    }

    pub(crate) fn dependency_tick_with_loads(
        &self,
        word: u32,
        tick: u64,
        mut pending_load: impl FnMut(u8) -> bool,
    ) -> Option<u64> {
        let spr = match ScalarInstruction::from_word(Architecture::Dav2201, word) {
            Some(ScalarInstruction::ScalarKey2MoveFromSpr {
                encoded_source_spr, ..
            }) => Some(encoded_source_spr),
            Some(ScalarInstruction::ScalarKey2MoveToSpr {
                encoded_destination_spr,
                ..
            }) => Some(encoded_destination_spr),
            _ => None,
        };
        let mut resume_tick = spr
            .and_then(|register| self.pending_spr_retirement(register))
            .filter(|retirement| *retirement > tick);
        if let Some(instruction) = crate::isa::c220::scalar::C220ScalarSprImmediate::decode(word) {
            return self
                .pending_spr_retirement(instruction.destination_spr)
                .filter(|retirement| *retirement > tick);
        }
        if let Some(rule) = C220ScalarTimingRule::decode(word)
            && rule.class == C220ScalarTimingClass::Variable
            && let Some(destination) = rule.destination_register
            && let Some(ready) = self
                .pending_xreg_retirement(destination)
                .filter(|ready| *ready > tick)
        {
            resume_tick = Some(resume_tick.map_or(ready, |prior| prior.max(ready)));
        }
        let mut include = |register: u8| {
            if let Some(retire_tick) = self
                .pending_xreg_retirement(register)
                .filter(|retire_tick| tick < *retire_tick)
            {
                resume_tick =
                    Some(resume_tick.map_or(retire_tick, |prior: u64| prior.max(retire_tick)));
            }
            if pending_load(register) {
                let retry = tick.saturating_add(1);
                resume_tick = Some(resume_tick.map_or(retry, |prior| prior.max(retry)));
            }
        };
        if let Some(mut mask) = crate::isa::c220::mte::read_register_mask(word) {
            while mask != 0 {
                include(mask.trailing_zeros() as u8);
                mask &= mask - 1;
            }
            return resume_tick;
        }
        if let Some(instruction) =
            crate::isa::c220::control::C220SetCrossCoreInstruction::decode(word)
        {
            include(instruction.source_register);
            return resume_tick;
        }
        if let Some(instruction) =
            crate::isa::c220::hflag::C220HardwareFlagInstruction::decode(word)
        {
            if let crate::isa::c220::hflag::C220HardwareEventSource::Register(register) =
                instruction.event_source
            {
                include(register);
            }
            return resume_tick;
        }
        if let Some(instruction) =
            crate::isa::flow::FlagInstruction::decode(Architecture::Dav2201, word)
        {
            if let crate::isa::flow::FlagIdSource::Register(register) = instruction.id_source {
                include(register);
            }
            return resume_tick;
        }
        if let Some(offset) = UnconditionalJump::decode(Architecture::Dav2201, word)
            .map(|jump| jump.offset_source)
            .or_else(|| {
                ConditionalJump::decode(Architecture::Dav2201, word).map(|jump| jump.offset_source)
            })
        {
            if let JumpOffsetSource::Register { index } = offset {
                include(index);
            }
            return resume_tick;
        }
        if let Some(instruction) = JumpCompare::decode(Architecture::Dav2201, word) {
            include(instruction.first_source_register);
            if let JumpCompareOperand::Register { index } = instruction.second_operand {
                include(index);
            }
            if let JumpCompareOffset::Register { index } = instruction.offset_source {
                include(index);
            }
            return resume_tick;
        }
        if let Some(instruction) = DcciInstruction::decode(Architecture::Dav2201, word) {
            include(instruction.source_register);
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
        if let Some(instruction) = C220ScalarPreload::decode(word) {
            include(instruction.base_register);
            if let C220PreloadOffset::Register(register) = instruction.offset {
                include(register);
            }
            return resume_tick;
        }
        if let Some(instruction) = C220ScalarAtomicStore::decode(word) {
            include(instruction.source_register);
            include(instruction.base_register);
            if let C220AtomicStoreOffset::Register(register) = instruction.offset {
                include(register);
            }
            return resume_tick;
        }
        if let Some(instruction) = C220ScalarDirectStore::decode(word) {
            include(instruction.source_register);
            include(instruction.base_register);
            return resume_tick;
        }
        let hint = ScalarInstruction::from_word(Architecture::Dav2201, word)?;
        match hint {
            ScalarInstruction::ScalarIndexedStore {
                source_register,
                base_register,
                offset_register,
                ..
            } => {
                include(source_register);
                include(base_register);
                include(offset_register);
            }
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
            | ScalarInstruction::ScalarKey2BitCount {
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
            }
            | ScalarInstruction::ScalarMoveX8Immediate {
                destination_register,
                ..
            } => include(destination_register),
            ScalarInstruction::ScalarKey7 {
                operation: ScalarKey7Operation::MoveKeep,
                destination_register,
                ..
            } => include(destination_register),
            ScalarInstruction::ScalarKey2MoveFromSpr { .. }
            | ScalarInstruction::ScalarKey7 { .. } => {}
        }
        resume_tick
    }

    pub(crate) fn issue(&mut self, ticket: C220ScalarTimingTicket) {
        // Issue is called only after the dependency gate accepts the writer.
        // Replaced results still retire, but no longer block register readers.
        if let Some(destination) = ticket.destination_register {
            self.supersede_destination(destination);
        }
        if ticket.class == C220ScalarTimingClass::Variable {
            let index = self
                .variable_events
                .partition_point(|event| *event <= ticket.retire_tick);
            self.variable_events.insert(index, ticket.retire_tick);
            self.variable_instructions.push_back(PendingVariable {
                ticket,
                destination_live: true,
            });
            return;
        }
        if let Some(destination) = ticket.destination_register {
            self.pending_xreg_retirement
                .insert(destination, ticket.retire_tick);
        }
        self.fixed_drain_tick = Some(
            self.fixed_drain_tick
                .map_or(ticket.retire_tick, |prior| prior.max(ticket.retire_tick)),
        );
    }

    pub(crate) fn supersede_destination(&mut self, destination: u8) {
        self.pending_xreg_retirement.remove(&destination);
        for pending in &mut self.variable_instructions {
            if pending.ticket.destination_register == Some(destination) {
                pending.destination_live = false;
            }
        }
    }

    pub(crate) fn load_waw_tick(&self, destination: u8) -> Option<u64> {
        self.pending_xreg_retirement.get(&destination).copied()
    }

    pub(crate) fn issue_spr(&mut self, ticket: C220ScalarSprTimingTicket) {
        self.pending_spr_retirement
            .insert(ticket.destination_spr, ticket.retire_tick);
    }

    pub fn pending_spr_retirement(&self, register: u16) -> Option<u64> {
        self.pending_spr_retirement.get(&register).copied()
    }

    pub fn pending_xreg_retirement(&self, register: u8) -> Option<u64> {
        self.pending_xreg_retirement
            .get(&register)
            .copied()
            .into_iter()
            .chain(
                self.variable_instructions
                    .iter()
                    .zip(self.variable_events.iter().copied())
                    .filter_map(|(pending, tick)| {
                        (pending.destination_live
                            && pending.ticket.destination_register == Some(register))
                        .then_some(tick)
                    }),
            )
            .max()
    }

    /// Projected FIFO retirements from notifications already scheduled.
    pub fn variable_retirements(&self) -> impl Iterator<Item = (&C220ScalarTimingTicket, u64)> {
        self.variable_instructions
            .iter()
            .map(|pending| &pending.ticket)
            .zip(self.variable_events.iter().copied())
    }

    pub fn pending_drain_tick(&self) -> Option<u64> {
        self.pending_xreg_retirement
            .values()
            .copied()
            .chain(self.variable_events.iter().copied())
            .chain(self.pending_spr_retirement.values().copied())
            .chain(self.fixed_drain_tick)
            .max()
    }
}

impl C220ScalarTimingTicket {
    pub fn for_conversion(issue_tick: u64, hint: C220ScalarConversionHint) -> Option<Self> {
        Some(Self {
            class: C220ScalarTimingClass::Fixed,
            issue_tick,
            retire_tick: issue_tick.checked_add(SCALAR_CONVERSION_LATENCY_TICKS)?,
            execution_stage: SCALAR_CONVERSION_EXECUTION_STAGE,
            source_register: Some(hint.source_register),
            destination_register: Some(hint.destination_register),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variable_notifications_preserve_duplicates_and_destination_hazards() {
        let mut lane = C220ScalarTimingLane::default();
        lane.issue(C220ScalarTimingTicket {
            class: C220ScalarTimingClass::Fixed,
            issue_tick: 0,
            retire_tick: 2,
            execution_stage: 1,
            source_register: None,
            destination_register: Some(32),
        });
        assert_eq!(lane.dependency_tick(0x08c2_003f, 1), Some(2));
        assert_eq!(lane.load_waw_tick(32), Some(2));
        lane.advance_to(2);
        assert_eq!(lane.pending_xreg_retirement(32), None);
        let divide = (5 << 17) | (1 << 12) | (2 << 7) | 5;
        let sqrt = 0x0200_0000 | (6 << 17) | (1 << 12);
        lane.issue(
            C220ScalarTimingRule::decode(divide)
                .unwrap()
                .ticket(10)
                .unwrap(),
        );
        lane.issue(
            C220ScalarTimingRule::decode(sqrt)
                .unwrap()
                .ticket(15)
                .unwrap(),
        );
        assert_eq!(
            lane.variable_retirements()
                .map(|(_, tick)| tick)
                .collect::<Vec<_>>(),
            [30, 30]
        );
        assert_eq!(lane.dependency_tick(divide, 16), Some(30));
        let independent_write = (5 << 17) | (1 << 12) | (2 << 7) | 1;
        assert_eq!(lane.dependency_tick(independent_write, 16), None);
        lane.issue(
            C220ScalarTimingRule::decode(independent_write)
                .unwrap()
                .ticket(16)
                .unwrap(),
        );
        assert_eq!(lane.pending_xreg_retirement(5), Some(17));
        assert_eq!(lane.pending_drain_tick(), Some(30));
        assert_eq!(lane.variable_retirements().count(), 2);
        lane.advance_to(17);
        assert_eq!(lane.pending_xreg_retirement(5), None);
        assert_eq!(lane.pending_xreg_retirement(6), Some(30));
        assert_eq!(lane.dependency_tick(divide, 17), None);
        lane.advance_to(30);
        assert_eq!(lane.variable_retirements().count(), 0);
        assert_eq!(lane.pending_drain_tick(), None);
        lane.issue(
            C220ScalarTimingRule::decode(independent_write)
                .unwrap()
                .ticket(31)
                .unwrap(),
        );
        assert_eq!(lane.dependency_tick(divide, 31), Some(32));
        lane.advance_to(32);
        lane.issue(C220ScalarTimingTicket {
            class: C220ScalarTimingClass::Fixed,
            issue_tick: 40,
            retire_tick: 50,
            execution_stage: 2,
            source_register: None,
            destination_register: Some(5),
        });
        lane.issue(
            C220ScalarTimingRule::decode(independent_write)
                .unwrap()
                .ticket(41)
                .unwrap(),
        );
        assert_eq!(lane.pending_xreg_retirement(5), Some(42));
        lane.advance_to(42);
        assert_eq!(lane.pending_xreg_retirement(5), None);
        assert_eq!(lane.pending_drain_tick(), Some(50));
        lane.advance_to(50);
        assert_eq!(lane.pending_drain_tick(), None);
    }

    #[test]
    fn vector_issue_waits_for_scalar_register_retirement() {
        let mut lane = C220ScalarTimingLane::default();
        lane.issue(C220ScalarTimingTicket {
            class: C220ScalarTimingClass::Fixed,
            issue_tick: 0,
            retire_tick: 4,
            execution_stage: 2,
            source_register: Some(1),
            destination_register: Some(6),
        });
        for opcode in [0x9700_0000, 0x9700_0001] {
            let word = opcode | (3 << 17) | (4 << 12) | (6 << 7) | (5 << 2);
            assert_eq!(lane.dependency_tick(word, 1), Some(4));
            assert_eq!(lane.dependency_tick(word, 4), None);
        }
        for word in [
            0x1800_0000 | (6 << 17) | (4 << 12),
            0x1900_0000 | (3 << 17) | (6 << 12),
            (6 << 29) | (3 << 17) | (4 << 12) | (6 << 7),
            (6 << 29) | (3 << 24) | (3 << 17) | (4 << 12) | (5 << 7) | (6 << 2),
            0x9c80_0003 | (3 << 17) | (4 << 12) | (6 << 7) | (5 << 2),
            0x8240_0700 | (3 << 17) | (6 << 12) | (5 << 2),
        ] {
            assert_eq!(lane.dependency_tick(word, 1), Some(4));
            assert_eq!(lane.dependency_tick(word, 4), None);
        }
        let three = (1 << 1) | (1 << 2) | (1 << 3);
        for (opcode, expected) in [
            (3 << 29, three),
            ((3 << 29) | (1 << 22), (1 << 1) | (1 << 3)),
            ((3 << 29) | (1 << 27) | (24 << 22), three),
            ((3 << 29) | (2 << 27) | (4 << 23) | (5 << 3), three),
            ((3 << 29) | (2 << 27) | (2 << 23) | 8, three),
            ((3 << 29) | (2 << 27) | (1 << 23) | 16, three),
            ((3 << 29) | (2 << 27) | (2 << 23) | 32, three),
            (
                (3 << 29) | (1 << 27) | (5 << 24) | (4 << 2),
                three | (1 << 4),
            ),
            (6 << 29, three),
            ((6 << 29) | (3 << 24) | (4 << 2), three | (1 << 4)),
        ] {
            let word = opcode | (1 << 17) | (2 << 12) | (3 << 7);
            assert_eq!(
                crate::isa::c220::mte::read_register_mask(word),
                Some(expected)
            );
            for register in 0..32 {
                let mut lane = C220ScalarTimingLane::default();
                lane.issue(C220ScalarTimingTicket {
                    class: C220ScalarTimingClass::Fixed,
                    issue_tick: 0,
                    retire_tick: 4,
                    execution_stage: 2,
                    source_register: Some(0),
                    destination_register: Some(register),
                });
                assert_eq!(
                    lane.dependency_tick(word, 1),
                    (expected & (1 << register) != 0).then_some(4)
                );
            }
        }
        for flag in [
            (2 << 29) | (5 << 21) | (1 << 10),
            (2 << 29) | (6 << 21) | (1 << 10),
        ] {
            assert_eq!(
                lane.dependency_tick(flag | (1 << 17) | (6 << 2), 1),
                Some(4)
            );
            assert_eq!(lane.dependency_tick(flag | 2, 1), None);
        }
        for form in 0..4 {
            let flag = (2 << 29) | (15 << 21) | (3 << 15) | (10 << 10) | (2 << 7) | (form << 5);
            assert_eq!(lane.dependency_tick(flag | 6, 1), (form >= 2).then_some(4));
            for special in 0..8 {
                let word = flag | (special << 18) | 6;
                assert_eq!(
                    crate::isa::c220::hflag::C220HardwareFlagInstruction::decode(word).is_some(),
                    matches!(special, 0 | 2 | 6 | 7)
                );
                assert!(
                    crate::isa::c220::hflag::C220HardwareFlagInstruction::decode(word | (1 << 27))
                        .is_none()
                );
            }
        }
        for special in 0..8 {
            for register in 0..32 {
                for pipe in 0..16 {
                    let word =
                        (2 << 29) | (15 << 21) | (special << 18) | (pipe << 10) | (register << 2);
                    let instruction =
                        crate::isa::c220::control::C220SetCrossCoreInstruction::decode(word);
                    assert_eq!(instruction.is_some(), matches!(special, 4 | 5));
                    if let Some(instruction) = instruction {
                        assert_eq!(instruction.source_register, register as u8);
                        assert_eq!(instruction.pipe_code, pipe as u8);
                        assert_eq!(lane.dependency_tick(word, 1), (register == 6).then_some(4));
                        assert_eq!(lane.dependency_tick(word, 4), None);
                        assert_eq!(
                            lane.dependency_tick_with_loads(word, 1, |r| r == register as u8),
                            Some(if register == 6 { 4 } else { 2 })
                        );
                    }
                    assert!(
                        crate::isa::c220::control::C220SetCrossCoreInstruction::decode(
                            word | (1 << 27)
                        )
                        .is_none()
                    );
                }
            }
        }
    }
}
