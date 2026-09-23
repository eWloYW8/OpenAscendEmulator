use std::collections::{BTreeMap, VecDeque};

use super::super::*;
use crate::memory::mapped::MappedMemory;
use crate::sim::c220::memory::{C220L0c, C220LocalBuffer};
use crate::sim::c220::mte::interface::{
    C220MteL0cReadInterface, C220MteL0cReadResponse, C220MteL0cReadSend, biu_read::C220BiuSubcore,
    biu_write::data::C220BiuWriteResponse,
};
use crate::sim::c220::mte::pipeline::{C220MtePipeline, C220MtePipelineError};
use crate::sim::c220::sync::C220HardwareFlagState;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FixpNz2ndCommandState {
    pub operands: C220FixpExternalCommand,
    pub lifecycle: C220FixpCommandState,
}

#[cfg(test)]
#[path = "runtime_tests.rs"]
mod tests;
#[derive(Debug, thiserror::Error)]
pub enum C220FixpNz2ndEngineError {
    #[error(transparent)]
    Engine(#[from] C220FixpEngineError),
    #[error(transparent)]
    Plan(#[from] C220FixpNz2ndPlanError),
    #[error(transparent)]
    Staging(#[from] C220FixpNz2ndStagingError),
    #[error(transparent)]
    Output(#[from] C220FixpNz2ndOutputError),
    #[error(transparent)]
    Pipeline(#[from] C220MtePipelineError),
    #[error("NZ2ND retirement requires a Cube write response")]
    ResponseSource,
}

/// NZ2ND external-write execution. Each method is a distinct scheduler
/// callback; callers own cross-engine ordering and shared BIU service.
/// Numerical writes occur on final L0C read acceptance, whereas retirement
/// requires the final external write response.
#[derive(Debug, Clone)]
pub struct C220FixpNz2ndEngine {
    config: C220FixpEngineConfig,
    main_slots: u32,
    commands: BTreeMap<u64, C220FixpNz2ndCommandState>,
    instruction_fifo: VecDeque<u64>,
    retirement_fifo: VecDeque<u64>,
    read: C220FixpReadPipeline,
    input: C220MteL0cReadInterface,
    functional: C220FixpFunctionalState,
    conversion: C220FixpConversionPipeline,
    staging: C220FixpNz2ndStaging,
    output: C220FixpNz2ndOutput,
    write: C220FixpBiuWritePipeline,
}

impl C220FixpNz2ndEngine {
    pub fn new(
        config: C220FixpEngineConfig,
        main_slots: u32,
        total_slots: usize,
    ) -> Result<Self, C220FixpNz2ndEngineError> {
        Ok(Self {
            config,
            main_slots,
            commands: BTreeMap::new(),
            instruction_fifo: VecDeque::new(),
            retirement_fifo: VecDeque::new(),
            read: C220FixpReadPipeline::default(),
            input: C220MteL0cReadInterface::new(config.read_bank_count, config.read_data_latency)
                .map_err(C220FixpEngineError::from)?,
            functional: C220FixpFunctionalState::new(config.l0c_capacity),
            conversion: C220FixpConversionPipeline::default(),
            staging: C220FixpNz2ndStaging::new(total_slots)?,
            output: C220FixpNz2ndOutput::default(),
            write: C220FixpBiuWritePipeline::default(),
        })
    }

    pub fn commands(&self) -> &BTreeMap<u64, C220FixpNz2ndCommandState> {
        &self.commands
    }
    pub fn instruction_fifo(&self) -> &VecDeque<u64> {
        &self.instruction_fifo
    }
    pub fn retirement_fifo(&self) -> &VecDeque<u64> {
        &self.retirement_fifo
    }
    pub fn read_pipeline(&self) -> &C220FixpReadPipeline {
        &self.read
    }
    pub fn read_interface(&self) -> &C220MteL0cReadInterface {
        &self.input
    }
    pub fn functional(&self) -> &C220FixpFunctionalState {
        &self.functional
    }
    pub fn conversion(&self) -> &C220FixpConversionPipeline {
        &self.conversion
    }
    pub fn staging(&self) -> &C220FixpNz2ndStaging {
        &self.staging
    }
    pub fn output(&self) -> &C220FixpNz2ndOutput {
        &self.output
    }
    pub fn write_pipeline(&self) -> &C220FixpBiuWritePipeline {
        &self.write
    }

    pub fn is_idle(&self) -> bool {
        self.commands.is_empty()
            && self.read.is_idle()
            && self.input.is_idle()
            && self.conversion.entries().is_empty()
            && self.staging.is_idle()
            && self.output.bursts().is_empty()
            && self.write.is_idle()
    }

    /// Attached flags must be captured and disabled-command synchronization
    /// resolved by the frontend before calling this method.
    pub fn admit(
        &mut self,
        tick: u64,
        id: u64,
        first_read: u32,
        first_write: u32,
        operands: C220FixpExternalCommand,
    ) -> Result<C220FixpAdmission, C220FixpNz2ndEngineError> {
        if self.commands.contains_key(&id) {
            return Err(C220FixpEngineError::DuplicateCommand(id).into());
        }
        let command = operands.command;
        if command.descriptor.is_disabled() {
            return Ok(C220FixpAdmission::DisabledReady);
        }
        if self.retirement_fifo.back().is_some_and(|id| {
            let previous = self.commands[id].operands.command;
            previous.descriptor.conversion_mode() != command.descriptor.conversion_mode()
                || previous.descriptor.activation_mode() != command.descriptor.activation_mode()
                || (previous.control ^ command.control) & (1 << 48) != 0
        }) {
            return Ok(C220FixpAdmission::ResourceConflict);
        }
        if self.read.generated_batches() != 0 {
            return Ok(C220FixpAdmission::ReadGenerationBusy);
        }
        if self.instruction_fifo.len() >= self.config.instruction_fifo_depth as usize {
            return Ok(C220FixpAdmission::InstructionFifoFull);
        }
        command
            .validate_activation()
            .map_err(C220FixpEngineError::from)?;
        let plan = operands.plan_nz2nd(
            id,
            first_read,
            first_write,
            self.config.read_bandwidth,
            self.main_slots,
        )?;
        if !self
            .read
            .submit(tick, plan.reads)
            .map_err(C220FixpEngineError::from)?
        {
            return Ok(C220FixpAdmission::DisabledReady);
        }
        self.staging.submit(plan.writes);
        self.commands.insert(
            id,
            C220FixpNz2ndCommandState {
                operands,
                lifecycle: C220FixpCommandState {
                    command,
                    admitted_tick: tick,
                    executed_tick: None,
                    write_dispatched_tick: None,
                },
            },
        );
        self.instruction_fifo.push_back(id);
        self.retirement_fifo.push_back(id);
        Ok(C220FixpAdmission::Active)
    }

    /// Capture attached flags before admission, including on retries.
    pub fn admit_with_flags(
        &mut self,
        tick: u64,
        id: u64,
        first_requests: (u32, u32),
        command: C220FixpExternalCommand,
        bindings: &mut C220FixpSyncBindings,
        flags: &mut C220HardwareFlagState,
    ) -> Result<C220FixpAdmission, C220FixpNz2ndEngineError> {
        if self.commands.contains_key(&id) {
            return Err(C220FixpEngineError::DuplicateCommand(id).into());
        }
        bindings.capture_pending(id, flags);
        if command.command.descriptor.is_disabled() {
            return Ok(
                if bindings
                    .disabled_blocked(tick, id, flags)
                    .map_err(C220FixpEngineError::from)?
                {
                    C220FixpAdmission::HardwareSync
                } else {
                    C220FixpAdmission::DisabledReady
                },
            );
        }
        self.admit(tick, id, first_requests.0, first_requests.1, command)
    }

    pub fn generate_read(
        &mut self,
        tick: u64,
    ) -> Result<C220FixpReadProgress, C220FixpNz2ndEngineError> {
        Ok(self
            .read
            .generate(tick)
            .map_err(C220FixpEngineError::from)?)
    }
    pub fn send_read(
        &mut self,
        tick: u64,
        sync: impl C220FixpSync,
    ) -> Result<C220FixpReadProgress, C220FixpNz2ndEngineError> {
        Ok(self
            .read
            .send(tick, &mut self.input, sync)
            .map_err(C220FixpEngineError::from)?)
    }
    pub fn send_l0c(
        &mut self,
        tick: u64,
        l0c: &mut C220L0c,
    ) -> Result<C220MteL0cReadSend, C220FixpNz2ndEngineError> {
        Ok(self
            .input
            .send_queued(tick, l0c)
            .map_err(C220FixpEngineError::from)?)
    }
    pub fn receive_l0c(
        &mut self,
        tick: u64,
        l0c: &mut C220L0c,
        slopes: &C220LocalBuffer,
        memory: &mut MappedMemory,
        atomics: C220FixpAtomicConfig,
        observe: impl FnMut(&C220FixpSliceResult, &[u8]),
    ) -> Result<(C220MteL0cReadResponse, Option<C220FixpFunctionalEvent>), C220FixpNz2ndEngineError>
    {
        let response = self
            .input
            .receive(tick, l0c)
            .map_err(C220FixpEngineError::from)?;
        let functional = if let C220MteL0cReadResponse::Accepted(ack) = response {
            let id = ack.operation.instruction_id;
            let state = self
                .commands
                .get_mut(&id)
                .ok_or(C220FixpEngineError::UnknownCommand(id))?;
            let event = self
                .functional
                .accept_read_with(ack, l0c.buffer(), |snapshot| {
                    state
                        .operands
                        .execute_to_external(snapshot, slopes, memory, atomics, observe)
                })
                .map_err(C220FixpEngineError::from)?;
            if event.executed {
                state.lifecycle.executed_tick = Some(tick);
            }
            Some(event)
        } else {
            None
        };
        Ok((response, functional))
    }
    pub fn convert(
        &mut self,
        tick: u64,
        sync: impl C220FixpSync,
    ) -> Result<C220FixpConversionReceive, C220FixpNz2ndEngineError> {
        Ok(self
            .conversion
            .receive(tick, &mut self.input, sync)
            .map_err(C220FixpEngineError::from)?)
    }
    pub fn slice(
        &mut self,
        tick: u64,
    ) -> Result<Option<C220FixpConversionEntry>, C220FixpNz2ndEngineError> {
        Ok(self.staging.receive(tick, &mut self.conversion)?)
    }
    pub fn transpose(
        &mut self,
        tick: u64,
    ) -> Result<C220FixpTransposeProgress, C220FixpNz2ndEngineError> {
        Ok(self.staging.stage(tick)?)
    }
    pub fn align(
        &mut self,
        tick: u64,
    ) -> Result<Option<C220FixpNz2ndStagingEntry>, C220FixpNz2ndEngineError> {
        let Some(head) = self.staging.alignment().front() else {
            return Ok(None);
        };
        let id = head.operation.instruction_id;
        let state = self
            .commands
            .get(&id)
            .ok_or(C220FixpEngineError::UnknownCommand(id))?;
        let policy = state
            .operands
            .nz2nd_output_policy()
            .map_err(C220FixpEngineError::from)?;
        Ok(self.output.receive(tick, &mut self.staging, policy)?)
    }
    pub fn packetize(
        &mut self,
        pipeline: &mut C220MtePipeline,
    ) -> Result<Option<C220FixpBiuWrite>, C220FixpNz2ndEngineError> {
        let Some(head) = self.output.bursts().front() else {
            return Ok(None);
        };
        let id = head.instruction_id;
        let mode = self
            .commands
            .get(&id)
            .ok_or(C220FixpEngineError::UnknownCommand(id))?
            .operands
            .biu_mode();
        Ok(pipeline.packetize_fixp_biu_output(&mut self.output, &mut self.write, mode)?)
    }
    pub fn generate_write(
        &mut self,
        tick: u64,
    ) -> Result<C220FixpWriteProgress<C220FixpBiuWrite>, C220FixpNz2ndEngineError> {
        Ok(self
            .write
            .generate(tick)
            .map_err(C220FixpEngineError::from)?)
    }
    pub fn send_write(
        &mut self,
        pipeline: &mut C220MtePipeline,
    ) -> Result<C220FixpWriteProgress<C220FixpBiuWrite>, C220FixpNz2ndEngineError> {
        let result = pipeline.send_fixp_biu_output(&mut self.write)?;
        if let C220FixpWriteProgress::Advanced(packet) = result
            && packet.write.fragment.last_in_instruction
        {
            let id = packet.write.fragment.instruction_id;
            if self.instruction_fifo.front() != Some(&id) {
                return Err(C220FixpEngineError::CommandOrder(id).into());
            }
            self.commands
                .get_mut(&id)
                .ok_or(C220FixpEngineError::UnknownCommand(id))?
                .lifecycle
                .write_dispatched_tick = Some(pipeline.tick());
            self.instruction_fifo.pop_front();
        }
        Ok(result)
    }
    /// Only pass responses accepted by the shared BIU response handler.
    pub fn retire_response(
        &mut self,
        response: C220BiuWriteResponse,
    ) -> Result<Option<C220FixpNz2ndCommandState>, C220FixpNz2ndEngineError> {
        if response.data.subcore != C220BiuSubcore::Cube {
            return Err(C220FixpNz2ndEngineError::ResponseSource);
        }
        let Some(id) = response.retired_instruction() else {
            return Ok(None);
        };
        self.retire_completed_write(id).map(Some)
    }

    pub(crate) fn retire_completed_write(
        &mut self,
        id: u64,
    ) -> Result<C220FixpNz2ndCommandState, C220FixpNz2ndEngineError> {
        let state = self
            .commands
            .get(&id)
            .ok_or(C220FixpEngineError::UnknownCommand(id))?;
        if state.lifecycle.executed_tick.is_none() {
            return Err(C220FixpEngineError::NotExecuted(id).into());
        }
        if state.lifecycle.write_dispatched_tick.is_none() {
            return Err(C220FixpEngineError::NotDispatched(id).into());
        }
        if self.retirement_fifo.front() != Some(&id) {
            return Err(C220FixpEngineError::RetirementOrder(id).into());
        }
        self.retirement_fifo.pop_front();
        Ok(self.commands.remove(&id).expect("validated command"))
    }
}
