use super::{C220Mte1Ticket, C220Mte1TimingError, C220TimedMte1Lane};
use crate::isa::c220::hflag::C220MatrixMemory;
use crate::isa::c220::mte::load2d::{C220Load2dDestination, C220Load2dTransfer};
use crate::sim::c220::memory::C220LocalMemory;
use crate::sim::c220::mte::mte1::load2d::{
    C220Load2dTransferError, C220Load2dTransferResult, prepare_c220_load2d,
};
use crate::sim::c220::sync::{
    C220HardwareFlagEvent, C220HardwareFlagState, C220HardwareFlagTimingError,
};
use std::collections::VecDeque;

#[derive(Debug, thiserror::Error)]
pub enum C220Mte1RuntimeError {
    #[error(transparent)]
    Transfer(#[from] C220Load2dTransferError),
    #[error(transparent)]
    Synchronization(#[from] C220HardwareFlagTimingError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Mte1Load2dOutcome {
    pub instruction_id: u64,
    pub pc: u64,
    pub retire_tick: u64,
    pub transfer: C220Load2dTransfer,
    pub result: C220Load2dTransferResult,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Mte1CommandState {
    pub instruction_id: u64,
    pub pc: u64,
    /// Earliest retirement attempt under the configured aggregate timing rules.
    pub eligible_tick: u64,
    pub pending_hardware_sets: usize,
    pub hardware_flag_stall: Option<C220HardwareFlagTimingError>,
}

#[derive(Default)]
pub(in crate::sim::c220) struct Mte1Engine {
    pub(in crate::sim::c220) timing: C220TimedMte1Lane,
    pending: VecDeque<PendingLoad2d>,
    pub(in crate::sim::c220) outcomes: Vec<C220Mte1Load2dOutcome>,
}

struct PendingLoad2d {
    instruction_id: u64,
    pc: u64,
    retire_tick: u64,
    transfer: C220Load2dTransfer,
    sets: VecDeque<C220HardwareFlagEvent>,
    hardware_flag_stall: Option<C220HardwareFlagTimingError>,
}

impl Mte1Engine {
    pub(in crate::sim::c220) fn issue(
        &mut self,
        instruction_id: u64,
        pc: u64,
        ticket: &C220Mte1Ticket,
        flags: &mut C220HardwareFlagState,
    ) -> Result<(), C220Mte1TimingError> {
        self.timing.issue(ticket)?;
        let memory = match ticket.transfer.instruction.destination {
            C220Load2dDestination::L0a => C220MatrixMemory::L0a,
            C220Load2dDestination::L0b => C220MatrixMemory::L0b,
            _ => unreachable!("MTE1 timing admitted a validated LOAD2D route"),
        };
        let sets = flags.take_mte_sets(instruction_id, memory);
        self.pending.push_back(PendingLoad2d {
            instruction_id,
            pc,
            retire_tick: ticket.retire_tick,
            transfer: ticket.transfer,
            sets,
            hardware_flag_stall: None,
        });
        Ok(())
    }

    pub(in crate::sim::c220) fn begin_advance(&mut self) {
        self.outcomes.clear();
        self.timing.begin_retirements();
    }

    pub(in crate::sim::c220) fn pending_commands(
        &self,
    ) -> impl Iterator<Item = C220Mte1CommandState> + '_ {
        self.pending.iter().map(|pending| C220Mte1CommandState {
            instruction_id: pending.instruction_id,
            pc: pending.pc,
            eligible_tick: pending.retire_tick,
            pending_hardware_sets: pending.sets.len(),
            hardware_flag_stall: pending.hardware_flag_stall,
        })
    }

    pub(in crate::sim::c220) fn next_retire_tick(&self) -> Option<u64> {
        self.pending.front().map(|pending| pending.retire_tick)
    }

    pub(in crate::sim::c220) fn commit_ready_at(
        &mut self,
        tick: u64,
        memory: &mut C220LocalMemory,
        flags: &mut C220HardwareFlagState,
    ) -> Result<(), C220Mte1RuntimeError> {
        flags.advance_to(tick)?;
        while let Some(pending) = self.pending.front_mut() {
            if pending.retire_tick > tick {
                break;
            }
            pending.hardware_flag_stall = None;
            let mut index = 0;
            while let Some(&event) = pending.sets.get(index) {
                match flags.schedule_event(event, tick) {
                    Ok(_) => {
                        pending.sets.remove(index);
                    }
                    Err(error @ C220HardwareFlagTimingError::AlmostFull { .. }) => {
                        pending.hardware_flag_stall.get_or_insert(error);
                        index += 1;
                    }
                    Err(error) => return Err(error.into()),
                }
            }
            if let Some(error) = pending.hardware_flag_stall {
                return Err(error.into());
            }
            let prepared = prepare_c220_load2d(memory, pending.transfer)?;
            let outcome = C220Mte1Load2dOutcome {
                instruction_id: pending.instruction_id,
                pc: pending.pc,
                retire_tick: tick,
                transfer: pending.transfer,
                result: prepared.result,
            };
            prepared.commit(memory)?;
            assert!(
                self.timing.retire_ready_front(tick),
                "completed command owns a timing slot"
            );
            self.pending.pop_front();
            self.outcomes.push(outcome);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::c220::hflag::C220HardwareFlagInstruction;
    use crate::isa::c220::mte::load2d::{C220Load2dDestination, C220Load2dInstruction};

    #[test]
    fn load2d_samples_memory_at_ordered_retirement() {
        let mut engine = Mte1Engine::default();
        let mut flags = C220HardwareFlagState::default();
        let mut memory = C220LocalMemory::new(Default::default()).unwrap();
        let mut registers = [0; 32];
        registers[3] = (3 << 16) | (1 << 24);
        let slow = C220Load2dInstruction::decode(0x6000_2181)
            .unwrap()
            .capture(&registers)
            .unwrap();
        let first = engine.timing.preview_issue(0, slow).unwrap();
        engine.issue(10, 0x1000, &first, &mut flags).unwrap();
        let flag_word = (2 << 29) | (15 << 21) | (3 << 10) | (2 << 7);
        let set_a = C220HardwareFlagInstruction::decode(flag_word | (1 << 15))
            .unwrap()
            .resolve(0x1004, &registers)
            .unwrap();
        let set_b = C220HardwareFlagInstruction::decode(flag_word | (2 << 15))
            .unwrap()
            .resolve(0x1008, &registers)
            .unwrap();
        flags.enqueue_mte_set(11, set_b, 1).unwrap();
        flags.enqueue_mte_set(12, set_b, 1).unwrap();
        flags.enqueue_mte_set(13, set_a, 1).unwrap();
        assert_eq!(flags.pending_mte_set_count(), 2);
        registers[3] = (1 << 16) | (1 << 24);
        let fast = C220Load2dInstruction::decode(0x6000_2180)
            .unwrap()
            .capture(&registers)
            .unwrap();
        let second = engine.timing.preview_issue(1, fast).unwrap();
        assert!(second.data_ready_tick < first.data_ready_tick);
        assert_eq!(second.retire_tick, first.retire_tick + 1);
        engine.issue(14, 0x1010, &second, &mut flags).unwrap();
        assert_eq!(flags.pending_mte_set_count(), 1);
        assert_eq!(flags.count(2, C220MatrixMemory::L0a, 0), 0);
        assert_eq!(
            engine
                .timing
                .pending_visibility_tick(C220Load2dDestination::L0a),
            Some(second.retire_tick)
        );

        engine
            .commit_ready_at(first.data_ready_tick, &mut memory, &mut flags)
            .unwrap();
        assert!(engine.outcomes.is_empty());
        assert_eq!(memory.l0a().tracked_bytes(), 0);
        assert_eq!(memory.l0b().tracked_bytes(), 0);
        memory.l1_mut().write_known(0, &vec![7; 1536]).unwrap();
        engine
            .commit_ready_at(first.retire_tick, &mut memory, &mut flags)
            .unwrap();
        assert_eq!(memory.l0b().read_known(0, 1536).unwrap(), vec![7; 1536]);
        assert_eq!(engine.outcomes[0].instruction_id, 10);
        assert_eq!(engine.outcomes[0].pc, 0x1000);
        assert_eq!(engine.outcomes[0].result.known_bytes, 1536);
        assert_eq!(flags.count(2, C220MatrixMemory::L0b, 0), 0);
        assert_eq!(flags.pending_mte_set_count(), 1);
        assert_eq!(engine.next_retire_tick(), Some(second.retire_tick));

        engine.begin_advance();
        memory.l1_mut().write_known(0, &[9; 512]).unwrap();
        engine
            .commit_ready_at(second.retire_tick, &mut memory, &mut flags)
            .unwrap();
        assert_eq!(memory.l0a().read_known(0, 512).unwrap(), vec![9; 512]);
        assert_eq!(engine.outcomes.len(), 1);
        assert_eq!(engine.outcomes[0].instruction_id, 14);
        assert_eq!(engine.outcomes[0].result.known_bytes, 512);
        assert_eq!(engine.next_retire_tick(), None);
        assert_eq!(flags.count(2, C220MatrixMemory::L0a, 0), 0);
        flags.advance_to(second.retire_tick + 1).unwrap();
        assert_eq!(flags.count(2, C220MatrixMemory::L0a, 0), 1);
        assert_eq!(flags.count(2, C220MatrixMemory::L0b, 0), 0);
        let next_issue_tick = second.retire_tick + 1;
        assert_eq!(engine.timing.last_retirements(), &[second]);
        assert_eq!(engine.timing.pending_retirement_count(), 0);
        let third = engine.timing.preview_issue(next_issue_tick, slow).unwrap();
        engine.issue(15, 0x1014, &third, &mut flags).unwrap();
        assert_eq!(flags.pending_mte_set_count(), 0);
        engine
            .commit_ready_at(third.retire_tick, &mut memory, &mut flags)
            .unwrap();
        flags.advance_to(third.retire_tick + 1).unwrap();
        assert_eq!(flags.count(2, C220MatrixMemory::L0b, 0), 1);
    }

    #[test]
    fn blocked_retirement_keeps_its_slot_and_releases_other_attached_sets() {
        let mut engine = Mte1Engine::default();
        let mut flags = C220HardwareFlagState::default();
        let mut memory = C220LocalMemory::new(Default::default()).unwrap();
        memory.l1_mut().write_known(0, &[7; 512]).unwrap();
        let mut registers = [0; 32];
        registers[3] = (1 << 16) | (1 << 24);
        let load = C220Load2dInstruction::decode(0x6000_2180).unwrap();
        let first = engine
            .timing
            .preview_issue(0, load.capture(&registers).unwrap())
            .unwrap();
        engine.issue(1, 0, &first, &mut flags).unwrap();
        let flag = (2 << 29) | (15 << 21) | (1 << 15) | (3 << 10) | (2 << 7);
        let set0 = C220HardwareFlagInstruction::decode(flag)
            .unwrap()
            .resolve(4, &registers)
            .unwrap();
        let set1 = C220HardwareFlagInstruction::decode(flag | 1)
            .unwrap()
            .resolve(8, &registers)
            .unwrap();
        for _ in 0..crate::sim::c220::sync::C220_HARDWARE_FLAG_ALMOST_FULL {
            flags.schedule_set(set0, 0).unwrap();
        }
        flags.enqueue_mte_set(2, set0, 1).unwrap();
        flags.enqueue_mte_set(3, set1, 1).unwrap();
        registers[0] = 512;
        let second = engine
            .timing
            .preview_issue(1, load.capture(&registers).unwrap())
            .unwrap();
        engine.issue(4, 12, &second, &mut flags).unwrap();
        engine.begin_advance();
        engine
            .commit_ready_at(first.retire_tick, &mut memory, &mut flags)
            .unwrap();
        assert_eq!(engine.timing.last_retirements(), &[first]);
        assert_eq!(engine.timing.pending_retirement_count(), 1);
        engine.begin_advance();
        assert!(matches!(
            engine.commit_ready_at(second.retire_tick, &mut memory, &mut flags),
            Err(C220Mte1RuntimeError::Synchronization(
                C220HardwareFlagTimingError::AlmostFull {
                    event_id: 0,
                    count: 32
                }
            ))
        ));
        assert_eq!(engine.timing.pending_retirement_count(), 1);
        assert!(engine.timing.last_retirements().is_empty());
        assert!(engine.outcomes.is_empty());
        assert_eq!(memory.l0a().tracked_bytes(), 512);
        let pending = engine.pending_commands().next().unwrap();
        assert_eq!(pending.instruction_id, 4);
        assert_eq!(pending.pending_hardware_sets, 1);
        assert!(pending.hardware_flag_stall.is_some());
        assert_eq!(
            flags.counter_snapshot(2, C220MatrixMemory::L0a, 1).pending,
            1
        );
    }
}
