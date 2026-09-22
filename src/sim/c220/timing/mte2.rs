use std::collections::VecDeque;
use std::num::NonZeroU64;

use thiserror::Error;

use crate::architecture::Architecture;
use crate::isa::flow::{FlagInstruction, FlagOperation};
use crate::memory::mapped::MappedMemory;
use crate::sim::c220::core::functional::{
    C220FunctionalCore, C220FunctionalError, C220MteAction, C220MteProgramStep,
};
use crate::sim::c220::mte::transfer::C220Mte2TransferPlan;
use crate::sim::c220::mte::uop::{C220DmaUopError, C220DmaUopRequest, mte2_requests};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Mte2TimingRules {
    pub issue_interval: NonZeroU64,
    pub startup_ticks: u64,
    pub bytes_per_tick: NonZeroU64,
    pub retire_ticks: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Mte2Ticket {
    pub issue_tick: u64,
    pub data_ready_tick: u64,
    pub retire_tick: u64,
    pub transfer: C220Mte2TransferPlan,
    pub uop_count: usize,
    pub modeled_service_ticks: u64,
}

impl C220Mte2Ticket {
    pub fn requests(self) -> Result<Vec<C220DmaUopRequest>, C220DmaUopError> {
        mte2_requests(self.transfer)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220StallCause {
    InstructionRate,
    ScalarDependency,
    Mte1IssueRate,
    Mte1Dependency,
    HardwareFlagDependency,
    Mte2IssueRate,
    Mte2Dependency,
    Mte3IssueRate,
    Mte3Dependency,
    VectorDependency,
    CubeDependency,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Stall {
    pub tick: u64,
    pub pc: u64,
    pub resume_tick: u64,
    pub cause: C220StallCause,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum C220Mte2Step {
    Executed {
        tick: u64,
        step: C220MteProgramStep,
        ticket: Option<C220Mte2Ticket>,
    },
    Stalled(C220Stall),
}

#[derive(Debug, Error)]
pub enum C220TimingError {
    #[error("timed C220 engine requires a dav_2201 core")]
    ArchitectureMismatch,
    #[error("tick {requested} precedes the previously observed tick {previous}")]
    TimeReversed { requested: u64, previous: u64 },
    #[error("MTE2 timing computation overflowed")]
    TimeOverflow,
    #[error("MTE2 transfer count and timing tickets diverged")]
    TicketMismatch,
    #[error(transparent)]
    Uop(#[from] C220DmaUopError),
    #[error(transparent)]
    Execute(#[from] C220FunctionalError),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220Mte2Pipeline {
    rules: C220Mte2TimingRules,
    last_tick: Option<u64>,
    next_instruction_tick: u64,
    next_mte2_issue_tick: u64,
    mte2_data_port_tick: u64,
    unsignaled: VecDeque<C220Mte2Ticket>,
    vector_flags: [Option<C220Mte2Ticket>; 2],
}

impl C220Mte2Pipeline {
    pub const fn new(rules: C220Mte2TimingRules) -> Self {
        Self {
            rules,
            last_tick: None,
            next_instruction_tick: 0,
            next_mte2_issue_tick: 0,
            mte2_data_port_tick: 0,
            unsignaled: VecDeque::new(),
            vector_flags: [None; 2],
        }
    }

    pub(crate) fn gate_other_at(
        &mut self,
        tick: u64,
        pc: u64,
    ) -> Result<Option<C220Stall>, C220TimingError> {
        if let Some(previous) = self.last_tick
            && tick < previous
        {
            return Err(C220TimingError::TimeReversed {
                requested: tick,
                previous,
            });
        }
        self.last_tick = Some(tick);
        if tick < self.next_instruction_tick {
            return Ok(Some(C220Stall {
                tick,
                pc,
                resume_tick: self.next_instruction_tick,
                cause: C220StallCause::InstructionRate,
            }));
        }
        Ok(None)
    }

    pub(crate) fn finish_other_at(&mut self, tick: u64) -> Result<(), C220TimingError> {
        self.next_instruction_tick = tick.checked_add(1).ok_or(C220TimingError::TimeOverflow)?;
        Ok(())
    }

    pub const fn rules(&self) -> C220Mte2TimingRules {
        self.rules
    }

    pub const fn next_instruction_tick(&self) -> u64 {
        self.next_instruction_tick
    }

    pub const fn next_mte2_issue_tick(&self) -> u64 {
        self.next_mte2_issue_tick
    }

    pub fn outstanding(&self) -> impl Iterator<Item = C220Mte2Ticket> + '_ {
        self.unsignaled
            .iter()
            .copied()
            .chain(self.vector_flags.iter().filter_map(|ticket| *ticket))
    }

    pub fn step_at(
        &mut self,
        tick: u64,
        core: &mut C220FunctionalCore,
        word: u32,
        source: &MappedMemory,
    ) -> Result<C220Mte2Step, C220TimingError> {
        if let Some(previous) = self.last_tick
            && tick < previous
        {
            return Err(C220TimingError::TimeReversed {
                requested: tick,
                previous,
            });
        }
        self.last_tick = Some(tick);
        let pc = core.scalar().pc();
        if tick < self.next_instruction_tick {
            return Ok(C220Mte2Step::Stalled(C220Stall {
                tick,
                pc,
                resume_tick: self.next_instruction_tick,
                cause: C220StallCause::InstructionRate,
            }));
        }

        let route = FlagInstruction::decode(Architecture::Dav2201, word).map(|instruction| {
            (
                instruction.source_pipe_code,
                instruction.trigger_pipe_code,
                instruction.operation,
            )
        });
        let wait_ready_tick = match route {
            Some((4, 0, FlagOperation::Wait)) if core.flag0_set() => self
                .unsignaled
                .iter()
                .map(|ticket| ticket.retire_tick)
                .max(),
            Some((4, 1, FlagOperation::Wait)) => {
                let flag_id = core.resolve_c220_vector_flag_id(pc, word)?;
                self.vector_flags[usize::from(flag_id)].map(|ticket| ticket.retire_tick)
            }
            _ => None,
        };
        if let Some(resume_tick) = wait_ready_tick
            && tick < resume_tick
        {
            return Ok(C220Mte2Step::Stalled(C220Stall {
                tick,
                pc,
                resume_tick,
                cause: C220StallCause::Mte2Dependency,
            }));
        }

        let transfer = if is_mte2_transfer(word) {
            if tick < self.next_mte2_issue_tick {
                return Ok(C220Mte2Step::Stalled(C220Stall {
                    tick,
                    pc,
                    resume_tick: self.next_mte2_issue_tick,
                    cause: C220StallCause::Mte2IssueRate,
                }));
            }
            Some(core.preview_c220_mte2_transfer(word)?)
        } else {
            None
        };
        let ticket = transfer
            .map(|plan| self.preview_ticket(tick, plan))
            .transpose()?;
        let next_instruction_tick = tick.checked_add(1).ok_or(C220TimingError::TimeOverflow)?;
        let step = core.step_mte_word_with_plan(word, source, transfer)?;
        match &step.action {
            C220MteAction::Issue {
                source_address,
                destination_address,
                planned_bytes,
                ..
            } => {
                let issued = ticket.ok_or(C220TimingError::TicketMismatch)?;
                if issued.transfer.bytes != *planned_bytes
                    || issued.transfer.source_address != *source_address
                    || issued.transfer.destination_address != *destination_address
                {
                    return Err(C220TimingError::TicketMismatch);
                }
                self.mte2_data_port_tick = issued.data_ready_tick;
                self.next_mte2_issue_tick = tick
                    .checked_add(self.rules.issue_interval.get())
                    .ok_or(C220TimingError::TimeOverflow)?;
                self.unsignaled.push_back(issued);
            }
            C220MteAction::SetVectorFlag { flag_id, .. } => {
                let issued = self
                    .unsignaled
                    .pop_front()
                    .ok_or(C220TimingError::TicketMismatch)?;
                self.vector_flags[usize::from(*flag_id)] = Some(issued);
            }
            C220MteAction::WaitVectorFlag { flag_id, .. } => {
                self.vector_flags[usize::from(*flag_id)] = None;
            }
            C220MteAction::WaitFlag { .. } => self.unsignaled.clear(),
            C220MteAction::SetFlag { .. } => {}
        }
        self.next_instruction_tick = next_instruction_tick;
        Ok(C220Mte2Step::Executed { tick, step, ticket })
    }

    fn preview_ticket(
        &self,
        issue_tick: u64,
        transfer: C220Mte2TransferPlan,
    ) -> Result<C220Mte2Ticket, C220TimingError> {
        let requests = mte2_requests(transfer)?;
        let uop_count = requests.len();
        let rate = self.rules.bytes_per_tick.get();
        let modeled_service_ticks = requests.iter().try_fold(0_u64, |total, request| {
            let bytes = u64::from(request.bytes);
            let ticks = bytes / rate + u64::from(bytes % rate != 0);
            total
                .checked_add(ticks)
                .ok_or(C220TimingError::TimeOverflow)
        })?;
        let start = issue_tick
            .checked_add(self.rules.startup_ticks)
            .ok_or(C220TimingError::TimeOverflow)?
            .max(self.mte2_data_port_tick);
        let data_ready_tick = start
            .checked_add(modeled_service_ticks)
            .ok_or(C220TimingError::TimeOverflow)?;
        let retire_tick = data_ready_tick
            .checked_add(self.rules.retire_ticks)
            .ok_or(C220TimingError::TimeOverflow)?;
        issue_tick
            .checked_add(self.rules.issue_interval.get())
            .ok_or(C220TimingError::TimeOverflow)?;
        Ok(C220Mte2Ticket {
            issue_tick,
            data_ready_tick,
            retire_tick,
            transfer,
            uop_count,
            modeled_service_ticks,
        })
    }
}

pub(crate) fn is_mte2_transfer(word: u32) -> bool {
    crate::isa::c220::mte::C220MovOutToUbDescriptor::is_word(word)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::c220::mte::{C220MovOutToUbDescriptor, CAPTURED_C220_MOV_OUT_TO_UB_X_WORD};

    fn transfer(burst_length: u16) -> C220Mte2TransferPlan {
        let descriptor = C220MovOutToUbDescriptor::decode(
            CAPTURED_C220_MOV_OUT_TO_UB_X_WORD,
            (u64::from(burst_length) << 16) | (1 << 4),
        )
        .unwrap();
        C220Mte2TransferPlan {
            descriptor,
            source_address: 0x1000,
            destination_address: 0x200,
            bytes: usize::from(burst_length) * 32,
            dma_mode_word: 0,
        }
    }

    #[test]
    fn c220_timeline_separates_issue_service_and_retirement() {
        let rules = C220Mte2TimingRules {
            issue_interval: NonZeroU64::new(2).unwrap(),
            startup_ticks: 3,
            bytes_per_tick: NonZeroU64::new(32).unwrap(),
            retire_ticks: 1,
        };
        let mut timed = C220Mte2Pipeline::new(rules);
        let first = timed.preview_ticket(5, transfer(2)).unwrap();
        assert_eq!((first.data_ready_tick, first.retire_tick), (10, 11));
        assert_eq!(first.transfer.descriptor_segments().unwrap().len(), 2);
        assert_eq!(first.uop_count, 1);
        assert_eq!(first.modeled_service_ticks, 2);
        assert_eq!(first.requests().unwrap()[0].bytes, 64);
        timed.mte2_data_port_tick = first.data_ready_tick;
        let second = timed.preview_ticket(7, transfer(4)).unwrap();
        assert_eq!((second.data_ready_tick, second.retire_tick), (14, 15));
        assert!(matches!(
            timed.preview_ticket(u64::MAX, transfer(1)),
            Err(C220TimingError::TimeOverflow)
        ));
    }

    #[test]
    fn split_requests_each_consume_a_service_quantum() {
        let rules = C220Mte2TimingRules {
            issue_interval: NonZeroU64::new(1).unwrap(),
            startup_ticks: 0,
            bytes_per_tick: NonZeroU64::new(64).unwrap(),
            retire_ticks: 0,
        };
        let timed = C220Mte2Pipeline::new(rules);
        let mut plan = transfer(1);
        plan.descriptor = C220MovOutToUbDescriptor::decode(
            CAPTURED_C220_MOV_OUT_TO_UB_X_WORD,
            (1_u64 << 32) | (1 << 16) | (2 << 4),
        )
        .unwrap();
        plan.bytes = 64;
        let ticket = timed.preview_ticket(5, plan).unwrap();
        assert_eq!(ticket.uop_count, 2);
        assert_eq!(ticket.modeled_service_ticks, 2);
        assert_eq!(ticket.data_ready_tick, 7);
    }
}
