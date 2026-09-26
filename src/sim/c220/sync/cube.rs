use super::{C220HardwareFlagState, C220HardwareFlagTimingError, C220QueuedCubeFlag};
use crate::isa::c220::hflag::{C220HardwareFlagOperation, C220HardwareFlagStep, C220MatrixMemory};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220CubeFlagStages {
    pub read_buffers: u32,
    pub bias: u32,
    pub write_buffer: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::c220::hflag::C220HardwareFlagInstruction;

    #[test]
    fn checkpoints_preserve_instruction_order_and_retry_full_counters() {
        let word = (2 << 29) | (15 << 21) | (3 << 15) | (1 << 14) | (2 << 10) | (2 << 7);
        let set = C220HardwareFlagInstruction::decode(word)
            .unwrap()
            .resolve(0, &[0; 32])
            .unwrap();
        let wait = C220HardwareFlagInstruction::decode(word | (1 << 5))
            .unwrap()
            .resolve(0, &[0; 32])
            .unwrap();
        let mut state = C220HardwareFlagState::default();
        for _ in 0..32 {
            state.schedule_set(set, 0).unwrap();
        }
        state.advance_to(2).unwrap();
        state.enqueue_cube_set(5, set).unwrap();
        state
            .enqueue_cube_checkpoint(C220CubeFlagCheckpoint::WriteBuffer, 5, 2, 14)
            .unwrap();
        state.advance_to(16).unwrap();
        assert_eq!(state.pending_cube_set_count(), 1);
        assert_eq!(state.pending_cube_checkpoints().count(), 0);
        state.check_cube_unit_flag_set(6, 16).unwrap();
        assert_eq!(state.next_notification_tick(), Some(17));
        state.advance_to(20).unwrap();
        assert_eq!(state.pending_cube_set_count(), 1);
        state.consume_wait(wait).unwrap();
        state.advance_to(21).unwrap();
        assert_eq!(state.pending_cube_set_count(), 0);
        assert_eq!(state.pending_cube_checkpoints().count(), 0);
        assert_eq!(state.count(10, C220MatrixMemory::L0c, 0), 31);
        state.advance_to(23).unwrap();
        assert_eq!(state.count(10, C220MatrixMemory::L0c, 0), 32);
    }
}

impl Default for C220CubeFlagStages {
    fn default() -> Self {
        Self {
            read_buffers: 3,
            bias: 3,
            write_buffer: 14,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220CubeFlagCheckpoint {
    ReadBuffers,
    Bias,
    WriteBuffer,
    UnitFlagRetry,
}

impl C220CubeFlagCheckpoint {
    const fn queue(self) -> usize {
        match self {
            Self::ReadBuffers => 0,
            Self::Bias => 1,
            Self::WriteBuffer => 2,
            Self::UnitFlagRetry => 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Checkpoint {
    pub tick: u64,
    instruction_id: u64,
}

impl C220HardwareFlagState {
    pub fn pending_cube_set_count(&self) -> usize {
        self.cube_sets.iter().map(|queue| queue.len()).sum()
    }

    pub fn pending_cube_checkpoints(
        &self,
    ) -> impl Iterator<Item = (C220CubeFlagCheckpoint, u64, u64)> + '_ {
        [
            C220CubeFlagCheckpoint::ReadBuffers,
            C220CubeFlagCheckpoint::Bias,
            C220CubeFlagCheckpoint::WriteBuffer,
            C220CubeFlagCheckpoint::UnitFlagRetry,
        ]
        .into_iter()
        .zip(&self.cube_checkpoints)
        .flat_map(|(kind, queue)| {
            queue
                .iter()
                .map(move |point| (kind, point.instruction_id, point.tick))
        })
    }

    pub(in crate::sim::c220) fn enqueue_cube_set(
        &mut self,
        instruction_id: u64,
        step: C220HardwareFlagStep,
    ) -> Result<(), C220HardwareFlagTimingError> {
        if step.instruction.operation != C220HardwareFlagOperation::Set || step.instruction.trigger
        {
            return Err(C220HardwareFlagTimingError::OperationMismatch);
        }
        let queue = if step.instruction.destination_pipe_code == 10 {
            2
        } else if step.instruction.memory == C220MatrixMemory::BiasTable {
            1
        } else {
            0
        };
        if self.cube_sets[queue]
            .back()
            .is_some_and(|tail| tail.instruction_id >= instruction_id)
        {
            return Err(C220HardwareFlagTimingError::InstructionOrder);
        }
        self.cube_sets[queue].push_back(C220QueuedCubeFlag {
            instruction_id,
            step,
        });
        Ok(())
    }

    pub(in crate::sim::c220) fn enqueue_cube_checkpoint(
        &mut self,
        kind: C220CubeFlagCheckpoint,
        instruction_id: u64,
        tick: u64,
        delay: u32,
    ) -> Result<(), C220HardwareFlagTimingError> {
        let tick = tick
            .checked_add(u64::from(delay))
            .ok_or(C220HardwareFlagTimingError::TimeOverflow)?;
        self.cube_checkpoints[kind.queue()].push_back(Checkpoint {
            tick,
            instruction_id,
        });
        Ok(())
    }

    pub(in crate::sim::c220) fn check_cube_unit_flag_set(
        &mut self,
        instruction_id: u64,
        tick: u64,
    ) -> Result<(), C220HardwareFlagTimingError> {
        if !self.release_cube_sets(2, instruction_id, tick)? {
            self.enqueue_cube_checkpoint(
                C220CubeFlagCheckpoint::UnitFlagRetry,
                instruction_id,
                tick,
                1,
            )?;
        }
        Ok(())
    }

    fn release_cube_sets(
        &mut self,
        queue: usize,
        instruction_id: u64,
        tick: u64,
    ) -> Result<bool, C220HardwareFlagTimingError> {
        while let Some(pending) = self.cube_sets[queue].front().copied()
            && pending.instruction_id < instruction_id
        {
            match self.schedule_set(pending.step, tick) {
                Ok(_) => {
                    self.cube_sets[queue].pop_front();
                }
                Err(C220HardwareFlagTimingError::AlmostFull { .. }) => return Ok(false),
                Err(error) => return Err(error),
            }
        }
        Ok(true)
    }

    pub(super) fn advance_cube_checkpoints(
        &mut self,
        tick: u64,
    ) -> Result<(), C220HardwareFlagTimingError> {
        for queue in 0..4 {
            let Some(head) = self.cube_checkpoints[queue]
                .front()
                .copied()
                .filter(|head| head.tick <= tick)
            else {
                continue;
            };
            if self.release_cube_sets(queue.min(2), head.instruction_id, tick)? {
                self.cube_checkpoints[queue].pop_front();
            }
            if let Some(head) = self.cube_checkpoints[queue].front_mut() {
                head.tick = head.tick.max(
                    tick.checked_add(1)
                        .ok_or(C220HardwareFlagTimingError::TimeOverflow)?,
                );
            }
        }
        Ok(())
    }
}
