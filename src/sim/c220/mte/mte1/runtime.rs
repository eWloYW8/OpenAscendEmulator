use super::bias::{C220BtTransferError, C220BtTransferResult, prepare_c220_mov_l1_to_bt};
use super::frontend::C220Mte1ReadTransfer;
use super::load2d::prepare_c220_load2d_transpose;
use super::load2d::{C220Load2dTransferError, C220Load2dTransferResult, prepare_c220_load2d};
use super::sparse::{C220SparseTransferResult, prepare_c220_load2d_sparse};
use super::{C220Mte1Command, C220Mte1Issue};
use crate::isa::c220::hflag::C220MatrixMemory;
use crate::isa::c220::mte::load2d::C220Load2dDestination;
use crate::isa::c220::mte::set2d::C220Set2dDestination;
use crate::sim::c220::memory::{C220LocalBufferError, C220LocalMemory};
use crate::sim::c220::mte::set2d::{C220Set2dResult, execute_c220_set2d};
use crate::sim::c220::mte::{C220MtePipeline, C220MtePipelineError};
use crate::sim::c220::sync::{
    C220HardwareFlagEvent, C220HardwareFlagState, C220HardwareFlagTimingError,
};
use std::collections::{BTreeMap, VecDeque};

/// Maximum outstanding instructions, including completed commands awaiting
/// retirement. This is separate from the physical generator queue capacity.
pub const C220_MTE1_OUTSTANDING_LIMIT: usize = 31;

#[derive(Debug, thiserror::Error)]
pub enum C220Mte1RuntimeError {
    #[error(transparent)]
    Load3dv2(#[from] crate::sim::c220::mte::load3d::C220Load3dExecutionError),
    #[error(transparent)]
    Transfer(#[from] C220Load2dTransferError),
    #[error(transparent)]
    Bt(#[from] C220BtTransferError),
    #[error(transparent)]
    Set2d(#[from] C220LocalBufferError),
    #[error(transparent)]
    Synchronization(#[from] C220HardwareFlagTimingError),
    #[error(transparent)]
    Pipeline(#[from] C220MtePipelineError),
    #[error("MTE1 cannot accept while busy")]
    Busy,
    #[error("MTE1 time overflowed")]
    TimeOverflow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220Mte1TransferResult {
    HardwareFlag(crate::isa::c220::hflag::C220HardwareFlagStep),
    WriteSpr(crate::sim::common::scalar::ScalarSprStep),
    Load3dv2(crate::sim::c220::mte::load3d::C220Load3dExecutionReport),
    CrossCore(crate::sim::c220::sync::C220DeviceSync),
    Load2dSparse(C220SparseTransferResult),
    Load2d(C220Load2dTransferResult),
    Load2dTranspose(C220Load2dTransferResult),
    Bt(C220BtTransferResult),
    Set2d(C220Set2dResult),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Mte1Outcome {
    pub instruction_id: u64,
    pub pc: u64,
    pub retire_tick: u64,
    pub hardware_flag_stall_ticks: u64,
    pub command: C220Mte1Command,
    pub result: C220Mte1TransferResult,
}

impl C220Mte1Outcome {
    pub fn cross_core_reception(&self) -> Option<crate::sim::c220::sync::C220CrossCoreReception> {
        let C220Mte1Command::CrossCore { instruction, .. } = self.command else {
            return None;
        };
        let C220Mte1TransferResult::CrossCore(payload) = self.result else {
            return None;
        };
        Some(crate::sim::c220::sync::C220CrossCoreReception {
            instruction_id: self.instruction_id,
            pc: self.pc,
            tick: self.retire_tick,
            instruction,
            payload,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Mte1CommandState {
    pub instruction_id: u64,
    pub pc: u64,
    pub issue_tick: u64,
    /// Absent until the final destination acknowledgment is observed.
    pub completion_tick: Option<u64>,
    pub pending_hardware_sets: usize,
    pub hardware_flag_stall: Option<C220HardwareFlagTimingError>,
    /// Retirement clocks blocked by attached HSET capacity, counted once per tick.
    pub hardware_flag_stall_ticks: u64,
}

pub(in crate::sim::c220) struct Mte1Engine {
    pending: VecDeque<PendingCommand>,
    pub(in crate::sim::c220) outcomes: Vec<C220Mte1Outcome>,
    now: u64,
    advanced: Option<u64>,
    events: Vec<u32>,
    deferred_events: BTreeMap<u64, Vec<u32>>,
    outstanding_limit: std::num::NonZeroU32,
}

impl Default for Mte1Engine {
    fn default() -> Self {
        Self::with_outstanding_limit(
            std::num::NonZeroU32::new(C220_MTE1_OUTSTANDING_LIMIT as u32).unwrap(),
        )
    }
}

struct PendingCommand {
    state: C220Mte1CommandState,
    command: C220Mte1Command,
    sets: VecDeque<C220HardwareFlagEvent>,
}

impl Mte1Engine {
    pub(in crate::sim::c220) fn with_outstanding_limit(
        outstanding_limit: std::num::NonZeroU32,
    ) -> Self {
        Self {
            pending: VecDeque::new(),
            outcomes: Vec::new(),
            now: 0,
            advanced: None,
            events: Vec::new(),
            deferred_events: BTreeMap::new(),
            outstanding_limit,
        }
    }
    pub(in crate::sim::c220) fn tick(&self) -> u64 {
        self.now
    }

    pub(in crate::sim::c220) fn can_issue(
        &self,
        pipeline: &C220MtePipeline,
        command: C220Mte1Command,
    ) -> bool {
        pipeline.can_issue_mte1(command)
            && self.pending.len() < self.outstanding_limit.get() as usize
    }

    fn synchronization_memory(
        command: C220Mte1Command,
    ) -> Result<Option<C220MatrixMemory>, C220Mte1RuntimeError> {
        let memory = match command {
            C220Mte1Command::Read(C220Mte1ReadTransfer::Load3dv2(command)) => {
                Some(match command.operands.instruction.destination {
                    crate::isa::c220::mte::load3d::C220Load3dDestination::L0a => {
                        C220MatrixMemory::L0a
                    }
                    crate::isa::c220::mte::load3d::C220Load3dDestination::L0b => {
                        C220MatrixMemory::L0b
                    }
                })
            }
            C220Mte1Command::CrossCore { .. }
            | C220Mte1Command::WriteSpr(_)
            | C220Mte1Command::HardwareFlag(_) => None,
            C220Mte1Command::Set2d(fill) => match fill.instruction.destination {
                C220Set2dDestination::L0a => Some(C220MatrixMemory::L0a),
                C220Set2dDestination::L0b => Some(C220MatrixMemory::L0b),
                C220Set2dDestination::L1 => {
                    return Err(C220MtePipelineError::WrongCommandLane.into());
                }
            },
            C220Mte1Command::Read(C220Mte1ReadTransfer::Bt(_)) => Some(C220MatrixMemory::BiasTable),
            C220Mte1Command::Read(C220Mte1ReadTransfer::Load2dSparse(_)) => {
                Some(C220MatrixMemory::L0b)
            }
            C220Mte1Command::Read(C220Mte1ReadTransfer::Load2dTranspose(transfer)) => {
                match transfer.instruction.destination {
                    C220Load2dDestination::L0a => Some(C220MatrixMemory::L0a),
                    C220Load2dDestination::L0b => Some(C220MatrixMemory::L0b),
                    destination => {
                        return Err(
                            C220Load2dTransferError::UnsupportedDestination(destination).into()
                        );
                    }
                }
            }
            C220Mte1Command::Read(C220Mte1ReadTransfer::Load2d(transfer)) => match transfer
                .instruction
                .destination
            {
                C220Load2dDestination::L0a => Some(C220MatrixMemory::L0a),
                C220Load2dDestination::L0b => Some(C220MatrixMemory::L0b),
                destination => {
                    return Err(C220Load2dTransferError::UnsupportedDestination(destination).into());
                }
            },
        };
        Ok(memory)
    }

    pub(in crate::sim::c220) fn prepare_flags(
        &self,
        pipeline: &mut C220MtePipeline,
        instruction_id: u64,
        command: C220Mte1Command,
        flags: &mut C220HardwareFlagState,
    ) -> Result<bool, C220Mte1RuntimeError> {
        if let Some(memory) = Self::synchronization_memory(command)? {
            pipeline.capture_mte1_flags(instruction_id, memory, flags);
        }
        Ok(
            !command.is_disabled()
                || !pipeline.disabled_mte1_sync_blocked(instruction_id, flags)?,
        )
    }

    pub(in crate::sim::c220) fn issue(
        &mut self,
        pipeline: &mut C220MtePipeline,
        instruction_id: u64,
        pc: u64,
        command: C220Mte1Command,
        flags: &mut C220HardwareFlagState,
    ) -> Result<C220Mte1Issue, C220Mte1RuntimeError> {
        if !self.can_issue(pipeline, command) {
            return Err(C220Mte1RuntimeError::Busy);
        }
        self.now
            .checked_add(1)
            .ok_or(C220Mte1RuntimeError::TimeOverflow)?;
        if !self.prepare_flags(pipeline, instruction_id, command, flags)? {
            return Err(C220Mte1RuntimeError::Busy);
        }
        let issue = pipeline.issue_mte1(instruction_id, command)?;
        let sets = pipeline.take_mte1_sets(instruction_id);
        self.pending.push_back(PendingCommand {
            state: C220Mte1CommandState {
                instruction_id,
                pc,
                issue_tick: self.now,
                completion_tick: issue.completion_ready.then_some(self.now),
                pending_hardware_sets: sets.len(),
                hardware_flag_stall: None,
                hardware_flag_stall_ticks: 0,
            },
            command,
            sets,
        });
        Ok(issue)
    }

    pub(in crate::sim::c220) fn begin_advance(&mut self) {
        self.outcomes.clear();
    }
    pub(in crate::sim::c220) fn pending_commands(
        &self,
    ) -> impl Iterator<Item = C220Mte1CommandState> + '_ {
        self.pending.iter().map(|p| p.state)
    }
    pub(in crate::sim::c220) fn next_event_tick(&self) -> Option<u64> {
        (!self.pending.is_empty()).then(|| self.now.saturating_add(1))
    }

    pub(in crate::sim::c220) fn set_event(&mut self, event_id: u32, queued_tail: Option<u64>) {
        let dependency =
            queued_tail.or_else(|| self.pending.back().map(|p| p.state.instruction_id));
        if let Some(dependency) = dependency {
            self.deferred_events
                .entry(dependency)
                .or_default()
                .push(event_id);
        } else {
            self.events.push(event_id);
        }
    }

    pub(in crate::sim::c220) fn wait_event(&mut self, event_id: u32) -> bool {
        let Some(index) = self.events.iter().position(|&id| id == event_id) else {
            return false;
        };
        self.events.remove(index);
        true
    }

    pub(in crate::sim::c220) fn ready_events(&self) -> &[u32] {
        &self.events
    }

    pub(in crate::sim::c220) fn deferred_events(&self) -> impl Iterator<Item = (u64, u32)> + '_ {
        self.deferred_events
            .iter()
            .flat_map(|(&predecessor, events)| {
                events.iter().map(move |&event| (predecessor, event))
            })
    }

    pub(in crate::sim::c220) fn commit_ready_at(
        &mut self,
        tick: u64,
        memory: &mut C220LocalMemory,
        flags: &mut C220HardwareFlagState,
    ) -> Result<(), C220Mte1RuntimeError> {
        if self.advanced == Some(tick) {
            return Ok(());
        }
        flags.advance_to(tick)?;
        self.try_retire_head(tick, memory, flags)?;
        self.now = tick;
        self.advanced = Some(tick);
        Ok(())
    }

    fn try_retire_head(
        &mut self,
        tick: u64,
        memory: &mut C220LocalMemory,
        flags: &mut C220HardwareFlagState,
    ) -> Result<(), C220Mte1RuntimeError> {
        // Command retirement precedes downstream queue consumers in this tick.
        // A completion observed later becomes eligible on the next clock.
        if let Some(pending) = self.pending.front_mut()
            && pending
                .state
                .completion_tick
                .is_some_and(|done| done < tick)
        {
            pending.state.hardware_flag_stall = None;
            let mut index = 0;
            while let Some(&event) = pending.sets.get(index) {
                match flags.schedule_event(event, tick) {
                    Ok(_) => {
                        pending.sets.remove(index);
                    }
                    Err(error @ C220HardwareFlagTimingError::AlmostFull { .. }) => {
                        pending.state.hardware_flag_stall.get_or_insert(error);
                        index += 1;
                    }
                    Err(error) => return Err(error.into()),
                }
            }
            pending.state.pending_hardware_sets = pending.sets.len();
            if pending.state.hardware_flag_stall.is_some() {
                pending.state.hardware_flag_stall_ticks =
                    pending.state.hardware_flag_stall_ticks.saturating_add(1);
                // Keep the command and retry on the next retirement clock.
                // Other engines must continue so consumers can release credit.
                return Ok(());
            }
            let result = match pending.command {
                C220Mte1Command::WriteSpr(step) => C220Mte1TransferResult::WriteSpr(step),
                C220Mte1Command::HardwareFlag(step) => C220Mte1TransferResult::HardwareFlag(step),
                C220Mte1Command::Read(C220Mte1ReadTransfer::Load3dv2(command)) => {
                    C220Mte1TransferResult::Load3dv2(command.execute(memory)?)
                }
                C220Mte1Command::CrossCore { payload, .. } => {
                    C220Mte1TransferResult::CrossCore(payload)
                }
                C220Mte1Command::Set2d(fill) => {
                    C220Mte1TransferResult::Set2d(execute_c220_set2d(memory, fill)?)
                }
                C220Mte1Command::Read(C220Mte1ReadTransfer::Load2d(transfer)) => {
                    let prepared = prepare_c220_load2d(memory, transfer)?;
                    let result = prepared.result;
                    prepared.commit(memory)?;
                    C220Mte1TransferResult::Load2d(result)
                }
                C220Mte1Command::Read(C220Mte1ReadTransfer::Bt(transfer)) => {
                    C220Mte1TransferResult::Bt(
                        prepare_c220_mov_l1_to_bt(memory, transfer)?.commit(memory)?,
                    )
                }
                C220Mte1Command::Read(C220Mte1ReadTransfer::Load2dTranspose(transfer)) => {
                    let prepared = prepare_c220_load2d_transpose(memory, transfer)?;
                    let result = prepared.result;
                    prepared.commit(memory)?;
                    C220Mte1TransferResult::Load2dTranspose(result)
                }
                C220Mte1Command::Read(C220Mte1ReadTransfer::Load2dSparse(transfer)) => {
                    let prepared = prepare_c220_load2d_sparse(memory.l1(), transfer)?;
                    let (l0b, indices) = memory.sparse_weight_buffers_mut();
                    C220Mte1TransferResult::Load2dSparse(prepared.commit(l0b, indices)?)
                }
            };
            let outcome = C220Mte1Outcome {
                instruction_id: pending.state.instruction_id,
                pc: pending.state.pc,
                retire_tick: tick,
                hardware_flag_stall_ticks: pending.state.hardware_flag_stall_ticks,
                command: pending.command,
                result,
            };
            self.pending.pop_front();
            if let Some(events) = self.deferred_events.remove(&outcome.instruction_id) {
                self.events.extend(events);
            }
            self.outcomes.push(outcome);
        }
        Ok(())
    }

    pub(in crate::sim::c220) fn observe_completions(&mut self, tick: u64, ids: &[u64]) {
        for id in ids {
            let pending = self
                .pending
                .iter_mut()
                .find(|p| p.state.instruction_id == *id)
                .expect("completion owns a pending command");
            pending.state.completion_tick = Some(tick);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::c220::hflag::C220HardwareFlagInstruction;
    use crate::isa::c220::mte::load2d::C220Load2dInstruction;
    use crate::sim::c220::memory::l1::C220L1Geometry;
    use crate::sim::c220::mte::C220MtePipelineConfig;
    use crate::sim::c220::mte::mte1::frontend::C220Mte1ReadBandwidths;
    use std::num::NonZeroU32;

    fn advance(
        engine: &mut Mte1Engine,
        pipeline: &mut C220MtePipeline,
        tick: u64,
        memory: &mut C220LocalMemory,
        flags: &mut C220HardwareFlagState,
    ) -> Result<(), C220Mte1RuntimeError> {
        engine.commit_ready_at(tick, memory, flags)?;
        pipeline.advance(tick)?;
        engine.observe_completions(tick, pipeline.mte1_completions());
        Ok(())
    }

    #[test]
    fn destination_completion_and_flag_credit_gate_commit_and_event_visibility() {
        let width = NonZeroU32::new(256).unwrap();
        let mut engine = Mte1Engine::default();
        let mut memory = C220LocalMemory::new(Default::default()).unwrap();
        let mut flags = C220HardwareFlagState::default();
        let mut pipeline = C220MtePipeline::new(
            0,
            C220MtePipelineConfig {
                core_kind: crate::sim::c220::device::C220CoreKind::Cube,
                l1: C220L1Geometry::new(32, 4, 1, 0).unwrap(),
                read_width: width,
                output_bandwidths: C220Mte1ReadBandwidths {
                    l0a: width,
                    l0b: width,
                    bt: width,
                },
                set2d_bandwidths: crate::sim::c220::mte::set2d::C220Set2dBandwidths {
                    l0a: width,
                    l0b: width,
                    l1: width,
                },
            },
        );
        let mut registers = [0; 32];
        registers[3] = (1 << 16) | (1 << 24);
        let transfer = C220Load2dInstruction::decode(0x6000_2180)
            .unwrap()
            .capture(&registers)
            .unwrap();
        let flag_word = (2 << 29) | (15 << 21) | (1 << 15) | (3 << 10) | (2 << 7);
        let set = C220HardwareFlagInstruction::decode(flag_word)
            .unwrap()
            .resolve(0, &registers)
            .unwrap();
        let wait = C220HardwareFlagInstruction::decode(flag_word | (1 << 5))
            .unwrap()
            .resolve(0, &registers)
            .unwrap();
        for _ in 0..crate::sim::c220::sync::C220_HARDWARE_FLAG_ALMOST_FULL {
            flags.schedule_set(set, 0).unwrap();
        }
        flags.enqueue_mte_flag(1, set, 0).unwrap();
        let independent_set = C220HardwareFlagInstruction::decode(flag_word | 1)
            .unwrap()
            .resolve(0, &registers)
            .unwrap();
        flags.enqueue_mte_flag(1, independent_set, 0).unwrap();
        advance(&mut engine, &mut pipeline, 0, &mut memory, &mut flags).unwrap();
        let command = C220Mte1Command::Read(C220Mte1ReadTransfer::Load2d(transfer));
        let issue = engine
            .issue(&mut pipeline, 2, 4, command, &mut flags)
            .unwrap();
        assert_eq!(issue.uop_count, 2);
        assert!(!engine.can_issue(&pipeline, command));
        engine.set_event(7, None);
        engine.set_event(7, None);
        engine.set_event(8, None);
        assert!(!engine.wait_event(7));
        assert!(!engine.wait_event(8));
        memory.l1_mut().write_known(0, &[7; 512]).unwrap();
        let completed = (1..100)
            .find(|&tick| {
                advance(&mut engine, &mut pipeline, tick, &mut memory, &mut flags).unwrap();
                assert!(engine.outcomes.is_empty());
                engine
                    .pending_commands()
                    .next()
                    .unwrap()
                    .completion_tick
                    .is_some()
            })
            .expect("destination completion");
        assert!(pipeline.selected_generator_idle());
        assert_eq!(memory.l0a().tracked_bytes(), 0);
        assert!(!engine.wait_event(7));
        for tick in completed + 1..=completed + 3 {
            advance(&mut engine, &mut pipeline, tick, &mut memory, &mut flags).unwrap();
            assert!(engine.outcomes.is_empty());
            assert_eq!(engine.tick(), tick);
            assert!(matches!(
                engine
                    .pending_commands()
                    .next()
                    .unwrap()
                    .hardware_flag_stall,
                Some(C220HardwareFlagTimingError::AlmostFull { .. })
            ));
        }
        assert_eq!(flags.count(2, C220MatrixMemory::L0a, 1), 1);
        assert_eq!(
            engine
                .pending_commands()
                .next()
                .unwrap()
                .hardware_flag_stall_ticks,
            3
        );
        assert_eq!(
            engine
                .pending_commands()
                .next()
                .unwrap()
                .pending_hardware_sets,
            1
        );
        assert_eq!(memory.l0a().tracked_bytes(), 0);
        flags.consume_wait(wait).unwrap();
        memory.l1_mut().write_known(0, &[9; 512]).unwrap();
        advance(
            &mut engine,
            &mut pipeline,
            completed + 3,
            &mut memory,
            &mut flags,
        )
        .unwrap();
        assert!(engine.outcomes.is_empty());
        advance(
            &mut engine,
            &mut pipeline,
            completed + 4,
            &mut memory,
            &mut flags,
        )
        .unwrap();
        assert_eq!(engine.outcomes[0].retire_tick, completed + 4);
        assert_eq!(engine.outcomes[0].hardware_flag_stall_ticks, 3);
        assert_eq!(memory.l0a().read_known(0, 512).unwrap(), vec![9; 512]);
        assert!(engine.pending_commands().next().is_none());
        assert!(engine.wait_event(7));
        assert!(engine.wait_event(7));
        assert!(engine.wait_event(8));
        assert!(!engine.wait_event(7));
        flags.advance_to(completed + 5).unwrap();
        assert_eq!(flags.count(2, C220MatrixMemory::L0a, 0), 32);
        assert_eq!(flags.count(2, C220MatrixMemory::L0a, 1), 1);

        registers[0] = u64::MAX;
        registers[2] = u64::MAX;
        registers[3] = !(0xff << 16);
        let empty = C220Load2dInstruction::decode(0x6000_2180)
            .unwrap()
            .capture(&registers)
            .unwrap();
        let before = memory.clone();
        let issue = engine
            .issue(
                &mut pipeline,
                3,
                8,
                C220Mte1Command::Read(C220Mte1ReadTransfer::Load2d(empty)),
                &mut flags,
            )
            .unwrap();
        assert!(issue.completion_ready);
        assert_eq!(issue.uop_count, 0);
        engine.set_event(9, None);
        assert!(!engine.wait_event(9));
        advance(
            &mut engine,
            &mut pipeline,
            completed + 5,
            &mut memory,
            &mut flags,
        )
        .unwrap();
        assert!(engine.wait_event(9));
        let outcome = engine.outcomes.last().unwrap();
        assert_eq!(outcome.instruction_id, 3);
        assert!(
            matches!(outcome.result, C220Mte1TransferResult::Load2d(result) if result.bytes == 0 && result.segment_count == 0)
        );
        assert_eq!(memory, before);
        assert!(engine.pending_commands().next().is_none());
    }
}
