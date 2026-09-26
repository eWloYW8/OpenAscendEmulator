use super::frontend::C220Mte3Record;
use super::{C220Mte3Ticket, C220Mte3TimingError, C220Mte3TimingRules, C220TimedMte3Lane};
use crate::memory::mapped::MappedMemory;
use crate::memory::ub::{UbMemory, UbTransferResult};
use crate::sim::c220::mte::C220TransferError;
use crate::sim::c220::mte::pipeline::{C220MtePipeline, C220MtePipelineError};
use std::collections::{BTreeMap, VecDeque};

pub const C220_MTE3_OUTSTANDING_LIMIT: usize = 31;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Mte3Step {
    pub pc: u64,
    pub word: u32,
    pub next_pc: u64,
    pub transfer: super::C220Mte3TransferPlan,
}

#[derive(Debug, thiserror::Error)]
pub enum C220Mte3RuntimeError {
    #[error("MTE3 retirement has no functional command for instruction {0}")]
    UnknownCommand(u64),
    #[error(transparent)]
    Pipeline(#[from] C220MtePipelineError),
    #[error(transparent)]
    Transfer(#[from] C220TransferError),
    #[error(transparent)]
    MovPad(#[from] crate::sim::c220::mte::mov_pad::C220MovPadError),
    #[error(transparent)]
    L1Output(#[from] crate::sim::c220::mte::l1_to_out::C220L1OutputError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Mte3Outcome {
    pub instruction_id: u64,
    pub tick: u64,
    pub pc: u64,
    pub word: u32,
    pub ticket: C220Mte3Ticket,
    pub result: UbTransferResult,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Mte3DmaOutcome {
    pub tick: u64,
    pub pc: u64,
    pub word: u32,
    pub record: C220Mte3Record,
    pub result: UbTransferResult,
}

pub(in crate::sim::c220) struct Mte3Engine {
    pub(in crate::sim::c220) timing: C220TimedMte3Lane,
    pending: VecDeque<C220Mte3CommandState>,
    pub(in crate::sim::c220) outcomes: Vec<C220Mte3Outcome>,
    pub(in crate::sim::c220) physical: bool,
    pub(in crate::sim::c220) native_commands: BTreeMap<u64, (u64, u32)>,
    pub(in crate::sim::c220) dma_outcomes: Vec<C220Mte3DmaOutcome>,
    pub(in crate::sim::c220) cross_core_outcomes:
        Vec<crate::sim::c220::sync::C220CrossCoreReception>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Mte3CommandState {
    pub instruction_id: u64,
    pub pc: u64,
    pub word: u32,
    pub ticket: C220Mte3Ticket,
}

impl Mte3Engine {
    pub(in crate::sim::c220) fn issue(
        &mut self,
        instruction_id: u64,
        pc: u64,
        word: u32,
        ticket: C220Mte3Ticket,
    ) -> Result<(), C220Mte3TimingError> {
        if self.is_full() {
            return Err(C220Mte3TimingError::QueueFull);
        }
        self.timing.issue(ticket)?;
        self.pending.push_back(C220Mte3CommandState {
            instruction_id,
            pc,
            word,
            ticket,
        });
        Ok(())
    }

    pub(in crate::sim::c220) fn new(rules: C220Mte3TimingRules) -> Self {
        Self {
            timing: C220TimedMte3Lane::new(rules),
            pending: VecDeque::new(),
            outcomes: Vec::new(),
            physical: false,
            native_commands: BTreeMap::new(),
            dma_outcomes: Vec::new(),
            cross_core_outcomes: Vec::new(),
        }
    }

    pub(in crate::sim::c220) fn begin_advance(&mut self) {
        self.outcomes.clear();
        self.dma_outcomes.clear();
        self.cross_core_outcomes.clear();
    }

    pub(in crate::sim::c220) fn commit_native_at(
        &mut self,
        tick: u64,
        pipeline: &mut C220MtePipeline,
        ub: &UbMemory,
        memory: &mut MappedMemory,
        atomics: crate::sim::c220::mte::atomic::C220AtomicConfig,
        l1_output: Option<(
            &crate::sim::c220::memory::C220LocalBuffer,
            &mut crate::sim::c220::mte::l1_to_out::C220L1OutputEngine,
        )>,
    ) -> Result<Option<u64>, C220Mte3RuntimeError> {
        let Some(record) = pipeline.mte3_retirement_candidate() else {
            return Ok(None);
        };
        let &(pc, word) = self
            .native_commands
            .get(&record.instruction_id)
            .ok_or(C220Mte3RuntimeError::UnknownCommand(record.instruction_id))?;
        let result = match record.command {
            super::frontend::C220Mte3Command::L1Output(command) => {
                let (l1, engine) =
                    l1_output.ok_or(C220Mte3RuntimeError::UnknownCommand(record.instruction_id))?;
                let result = crate::sim::c220::mte::l1_to_out::execute_c220_mov_l1_to_out(
                    l1,
                    memory,
                    command.transfer,
                    command.control,
                    atomics,
                )?;
                if !command.transfer.is_disabled() {
                    pipeline.retire_l1_output(engine, record.instruction_id)?;
                }
                result
            }
            super::frontend::C220Mte3Command::Dma(plan) => plan.execute(ub, memory, atomics)?,
            super::frontend::C220Mte3Command::MovPad(command) => {
                crate::sim::c220::mte::mov_pad::prepare_c220_mov_pad(
                    command.transfer,
                    memory,
                    ub,
                    command.padding,
                )?
                .commit_to_external_with_atomics(
                    memory,
                    command.control,
                    atomics,
                )?
            }
            super::frontend::C220Mte3Command::CrossCore {
                instruction,
                payload,
            } => {
                pipeline.retire_mte3(record.instruction_id)?;
                self.native_commands.remove(&record.instruction_id);
                self.cross_core_outcomes
                    .push(crate::sim::c220::sync::C220CrossCoreReception {
                        instruction_id: record.instruction_id,
                        pc,
                        tick,
                        instruction,
                        payload,
                    });
                return Ok(Some(record.instruction_id));
            }
        };
        pipeline.retire_mte3(record.instruction_id)?;
        self.native_commands.remove(&record.instruction_id);
        self.dma_outcomes.push(C220Mte3DmaOutcome {
            tick,
            pc,
            word,
            record,
            result,
        });
        Ok(Some(record.instruction_id))
    }

    pub(in crate::sim::c220) fn is_full(&self) -> bool {
        self.pending.len() >= C220_MTE3_OUTSTANDING_LIMIT
    }

    pub(in crate::sim::c220) fn pending_commands(
        &self,
    ) -> impl Iterator<Item = C220Mte3CommandState> + '_ {
        self.pending.iter().copied()
    }

    pub(in crate::sim::c220) fn pending_retirement_tick(&self) -> Option<u64> {
        self.pending
            .front()
            .map(|pending| pending.ticket.retire_tick)
    }

    pub(in crate::sim::c220) fn commit_ready_at(
        &mut self,
        tick: u64,
        ub: &UbMemory,
        memory: &mut MappedMemory,
        atomics: crate::sim::c220::mte::atomic::C220AtomicConfig,
    ) -> Result<(), C220TransferError> {
        if let Some(pending) = self.pending.front()
            && tick >= pending.ticket.retire_tick
        {
            let plan = pending.ticket.transfer;
            let result = plan.execute(ub, memory, atomics)?;
            self.outcomes.push(C220Mte3Outcome {
                instruction_id: pending.instruction_id,
                tick,
                pc: pending.pc,
                word: pending.word,
                ticket: pending.ticket,
                result,
            });
            self.pending.pop_front();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::C220Mte3TransferPlan;
    use super::*;
    use crate::isa::c220::mte::{C220DmaMovDescriptor, CAPTURED_C220_MOV_UB_TO_OUT_WORD};
    use crate::memory::region::MemoryRegion;
    use crate::memory::sparse::SparseMemory;
    use std::num::NonZeroU64;

    #[test]
    fn outstanding_slots_are_released_only_after_functional_retirement() {
        let mut engine = Mte3Engine::new(C220Mte3TimingRules {
            issue_interval: NonZeroU64::new(1).unwrap(),
            startup_ticks: 100,
            bytes_per_tick: NonZeroU64::new(32).unwrap(),
            retire_ticks: 1,
        });
        let word = CAPTURED_C220_MOV_UB_TO_OUT_WORD;
        let plan = C220Mte3TransferPlan {
            control: 0,
            descriptor: C220DmaMovDescriptor::decode(word, 0x40010).unwrap(),
            source_address: 0,
            destination_address: 0x2000,
            bytes: 128,
            dma_mode_word: 0,
            biu_mode_word: 0,
        };
        for tick in 0..C220_MTE3_OUTSTANDING_LIMIT as u64 {
            let ticket = engine.timing.preview_issue(tick, plan).unwrap();
            engine.issue(tick, tick * 4, word, ticket).unwrap();
        }
        assert!(engine.is_full());
        let rejected = engine.timing.preview_issue(31, plan).unwrap();
        let previous_retirement = engine.timing.latest_retirement_tick();
        assert!(matches!(
            engine.issue(31, 124, word, rejected),
            Err(C220Mte3TimingError::QueueFull)
        ));
        assert_eq!(engine.timing.latest_retirement_tick(), previous_retirement);
        let head = engine.pending.front().unwrap().ticket;
        let mut memory = MappedMemory::bind(
            SparseMemory::new(vec![MemoryRegion::unknown(128)], 256, 256),
            &[0x2000],
        )
        .unwrap();
        let ub = UbMemory::new(256, 256);
        engine
            .commit_ready_at(head.data_ready_tick, &ub, &mut memory, Default::default())
            .unwrap();
        assert!(engine.is_full());
        engine
            .commit_ready_at(head.retire_tick, &ub, &mut memory, Default::default())
            .unwrap();
        assert!(!engine.is_full());
        assert_eq!(engine.outcomes.len(), 1);
        assert_eq!(engine.outcomes[0].result.unknown_bytes, 128);
        let ticket = engine.timing.preview_issue(head.retire_tick, plan).unwrap();
        engine.issue(31, 124, word, ticket).unwrap();
        assert!(engine.is_full());
    }
}
