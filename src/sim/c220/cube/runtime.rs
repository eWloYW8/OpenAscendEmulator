use crate::sim::c220::cube::{
    C220CubeConfig, C220CubeExecutionControl, C220CubeExecutionOutcome, C220CubeIssue,
    C220CubePipeline, C220CubeTicket, C220CubeTimingError, C220CubeUopRelease,
    update_cube_status_spr2,
};
use crate::sim::c220::memory::{C220LocalBuffer, C220LocalMemory};
use crate::sim::c220::sync::C220HardwareFlagState;
use crate::sim::common::scalar::ScalarMachine;

pub(in crate::sim::c220) struct CubeEngine {
    pub(in crate::sim::c220) pipeline: C220CubePipeline,
    pending: Vec<PendingCube>,
    pub(in crate::sim::c220) outcomes: Vec<C220CubeExecutionOutcome>,
}

struct PendingCube {
    issue: C220CubeIssue,
    control: C220CubeExecutionControl,
    prepared: Option<PreparedCube>,
}

struct PreparedCube {
    outcome: C220CubeExecutionOutcome,
    result_buffer: C220LocalBuffer,
}

impl CubeEngine {
    pub(in crate::sim::c220) fn new(config: C220CubeConfig) -> Result<Self, C220CubeTimingError> {
        Ok(Self {
            pipeline: C220CubePipeline::new(config)?,
            pending: Vec::new(),
            outcomes: Vec::new(),
        })
    }

    pub(in crate::sim::c220) fn issue(
        &mut self,
        issue: C220CubeIssue,
        control: C220CubeExecutionControl,
        memory: &mut C220LocalMemory,
    ) -> Result<(), C220CubeTimingError> {
        self.pipeline.issue(
            issue.ticket,
            issue.uops(),
            issue.instruction_id,
            memory.l0c_mut(),
        )?;
        self.pending.push(PendingCube {
            issue,
            control,
            prepared: None,
        });
        Ok(())
    }

    pub(in crate::sim::c220) fn begin_advance(&mut self) {
        self.pipeline.begin_advance();
        self.outcomes.clear();
    }

    pub(in crate::sim::c220) fn advance_event(
        &mut self,
        tick: u64,
        memory: &mut C220LocalMemory,
        flags: &mut C220HardwareFlagState,
        machine: &mut ScalarMachine,
    ) -> Result<(), C220CubeRuntimeError> {
        let release_start = self.pipeline.last_uop_releases().len();
        let retirement_start = self.pipeline.last_retirements().len();
        self.pipeline
            .advance_in_batch_to(tick, memory.l0c_mut(), flags)?;
        let releases = self.pipeline.last_uop_releases()[release_start..].to_vec();
        let retired = self.pipeline.last_retirements()[retirement_start..].to_vec();
        self.execute_released(&releases, memory, machine)?;
        self.commit_retired(&retired, memory, machine)
    }

    fn execute_released(
        &mut self,
        releases: &[C220CubeUopRelease],
        memory: &mut C220LocalMemory,
        machine: &mut ScalarMachine,
    ) -> Result<(), C220CubeRuntimeError> {
        for release in releases {
            let pending = self
                .pending
                .iter_mut()
                .find(|pending| pending.issue.instruction_id == release.instruction_id)
                .ok_or(C220CubeTimingError::TicketMismatch)?;
            if pending.prepared.is_none() {
                let prepared = pending.issue.prepare(memory, pending.control)?;
                let mut result_buffer = memory.l0c().buffer().clone();
                prepared.commit_to_buffer(&mut result_buffer)?;
                let spr2 =
                    update_cube_status_spr2(machine.spr2(), pending.issue.pc, prepared.outcome);
                machine.set_spr_value(2, spr2)?;
                pending.prepared = Some(PreparedCube {
                    outcome: prepared.outcome,
                    result_buffer,
                });
            }
            if let Some(request) = release.uop.l0c_write {
                let bytes = pending
                    .prepared
                    .as_ref()
                    .expect("prepared Cube result")
                    .result_buffer
                    .read_initialized_states_linear(request.address, usize::from(request.bytes))
                    .map_err(super::C220CubeExecutionError::from)?;
                memory
                    .l0c_mut()
                    .buffer_mut()
                    .write_states_linear(request.address, &bytes)
                    .map_err(super::C220CubeExecutionError::from)?;
            }
        }
        Ok(())
    }

    fn commit_retired(
        &mut self,
        retired: &[C220CubeTicket],
        memory: &mut C220LocalMemory,
        machine: &mut ScalarMachine,
    ) -> Result<(), C220CubeRuntimeError> {
        for ticket in retired {
            let index = self
                .pending
                .iter()
                .position(|pending| pending.issue.ticket.accept_tick == ticket.accept_tick)
                .ok_or(C220CubeTimingError::TicketMismatch)?;
            let pending = self.pending.remove(index);
            let outcome = match pending.prepared {
                Some(prepared) => prepared.outcome,
                None => {
                    let prepared = pending.issue.prepare(memory, pending.control)?;
                    let outcome = prepared.outcome;
                    prepared.commit(memory)?;
                    let spr2 = update_cube_status_spr2(machine.spr2(), pending.issue.pc, outcome);
                    machine.set_spr_value(2, spr2)?;
                    outcome
                }
            };
            self.outcomes.push(outcome);
        }
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum C220CubeRuntimeError {
    #[error(transparent)]
    Timing(#[from] super::C220CubeTimingError),
    #[error(transparent)]
    Execution(#[from] super::C220CubeExecutionError),
    #[error(transparent)]
    Scalar(#[from] crate::sim::common::scalar::ScalarMachineError),
}
