use super::{C220Core, C220CoreError, C220CoreInstruction, C220CoreStep};
use crate::isa::c220::control::C220SetCrossCoreInstruction;
use crate::sim::c220::schedule::{C220Stall, C220StallCause};
use crate::sim::c220::sync::{C220CrossCoreReception, C220DeviceSync};

impl C220Core {
    pub(super) fn step_cross_core_at(
        &mut self,
        tick: u64,
        pc: u64,
        instruction: C220SetCrossCoreInstruction,
    ) -> Result<C220CoreStep, C220CoreError> {
        if self.state.scalar().is_halted() {
            return Err(crate::sim::c220::state::C220ExecutionError::ProgramEnded { pc }.into());
        }
        if instruction.pipe_code == 3 {
            return self.step_mte1_at(tick, pc, instruction.word);
        }
        let (pending, cause) = match instruction.pipe_code {
            1 => (
                self.vector.pending_drain_tick(),
                C220StallCause::VectorDependency,
            ),
            2 => (
                self.cube.pipeline.pending_drain_tick(),
                C220StallCause::CubeDependency,
            ),
            pipe => return Err(C220CoreError::UnsupportedCrossCorePipe { pc, pipe }),
        };
        if let Some(ready) = pending {
            return Ok(C220CoreStep::Stalled(C220Stall {
                tick,
                pc,
                resume_tick: ready.max(tick.checked_add(1).ok_or(C220CoreError::TimeOverflow)?),
                cause,
            }));
        }
        let value = self.state.scalar().machine().xregs()[usize::from(instruction.source_register)];
        let reception = C220CrossCoreReception {
            instruction_id: self.next_instruction_id,
            pc,
            tick,
            instruction,
            payload: C220DeviceSync::from_value(value),
        };
        self.state.commit_c220_sequential_issue();
        Ok(C220CoreStep::Executed {
            tick,
            instruction: C220CoreInstruction::CrossCore(reception),
        })
    }
}
