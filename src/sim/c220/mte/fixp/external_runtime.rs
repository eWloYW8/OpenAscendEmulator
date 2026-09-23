use std::collections::{BTreeMap, VecDeque};

use super::*;
use crate::memory::mapped::MappedMemory;
use crate::sim::c220::memory::{C220L0c, C220LocalBuffer};
use crate::sim::c220::mte::interface::{
    C220MteL0cReadInterface, C220MteL0cReadResponse, C220MteL0cReadSend, biu_read::C220BiuSubcore,
    biu_write::data::C220BiuWriteResponse,
};
use crate::sim::c220::mte::pipeline::{C220MtePipeline, C220MtePipelineError};
use crate::sim::c220::sync::C220HardwareFlagState;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FixpExternalCommandState {
    pub operands: C220FixpExternalCommand,
    pub lifecycle: C220FixpCommandState,
}

#[cfg(test)]
#[path = "external_runtime_tests.rs"]
mod tests;
#[derive(Debug, thiserror::Error)]
pub enum C220FixpExternalEngineError {
    #[error(transparent)]
    Engine(#[from] C220FixpEngineError),
    #[error(transparent)]
    Plan(#[from] C220FixpNz2ndPlanError),
    #[error(transparent)]
    Staging(#[from] C220FixpNz2ndStagingError),
    #[error(transparent)]
    Output(#[from] C220FixpExternalOutputError),
    #[error(transparent)]
    Pipeline(#[from] C220MtePipelineError),
    #[error("external FIX retirement requires a Cube write response")]
    ResponseSource,
}

/// Ordinary and NZ2ND external-write execution share all transport resources.
/// Each method is a distinct scheduler
/// callback; callers own cross-engine ordering and shared BIU service.
/// Numerical writes occur on final L0C read acceptance, whereas retirement
/// requires the final external write response.
#[derive(Debug, Clone)]
pub struct C220FixpExternalEngine {
    config: C220FixpEngineConfig,
    main_slots: u32,
    commands: BTreeMap<u64, C220FixpExternalCommandState>,
    instruction_fifo: VecDeque<u64>,
    retirement_fifo: VecDeque<u64>,
    write_completions: BTreeMap<u64, u64>,
    last_retirement_tick: Option<u64>,
    read: C220FixpReadPipeline,
    input: C220MteL0cReadInterface,
    functional: C220FixpFunctionalState,
    conversion: C220FixpConversionPipeline,
    staging: C220FixpNz2ndStaging,
    output: C220FixpExternalOutput,
    write: C220FixpDispatchPipeline,
}

impl C220FixpExternalEngine {
    pub fn new(
        config: C220FixpEngineConfig,
        main_slots: u32,
        total_slots: usize,
    ) -> Result<Self, C220FixpExternalEngineError> {
        Ok(Self {
            config,
            main_slots,
            commands: BTreeMap::new(),
            instruction_fifo: VecDeque::new(),
            retirement_fifo: VecDeque::new(),
            write_completions: BTreeMap::new(),
            last_retirement_tick: None,
            read: C220FixpReadPipeline::default(),
            input: C220MteL0cReadInterface::new(config.read_bank_count, config.read_data_latency)
                .map_err(C220FixpEngineError::from)?,
            functional: C220FixpFunctionalState::new(config.l0c_capacity),
            conversion: C220FixpConversionPipeline::default(),
            staging: C220FixpNz2ndStaging::new(total_slots)?,
            output: C220FixpExternalOutput::default(),
            write: C220FixpDispatchPipeline::default(),
        })
    }

    pub fn commands(&self) -> &BTreeMap<u64, C220FixpExternalCommandState> {
        &self.commands
    }
    pub fn instruction_fifo(&self) -> &VecDeque<u64> {
        &self.instruction_fifo
    }
    pub fn retirement_fifo(&self) -> &VecDeque<u64> {
        &self.retirement_fifo
    }
    pub fn write_completion_tick(&self, id: u64) -> Option<u64> {
        self.write_completions.get(&id).copied()
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
    pub fn output(&self) -> &C220FixpExternalOutput {
        &self.output
    }
    pub fn write_pipeline(&self) -> &C220FixpDispatchPipeline {
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
    pub(crate) fn admit(
        &mut self,
        tick: u64,
        id: u64,
        first_read: u32,
        first_write: u32,
        operands: C220FixpExternalCommand,
    ) -> Result<C220FixpAdmission, C220FixpExternalEngineError> {
        if self.commands.contains_key(&id) {
            return Err(C220FixpEngineError::DuplicateCommand(id).into());
        }
        let command = operands.command;
        if command.descriptor.is_disabled() {
            self.commands.insert(
                id,
                C220FixpExternalCommandState {
                    operands,
                    lifecycle: C220FixpCommandState {
                        command,
                        admitted_tick: tick,
                        executed_tick: None,
                        write_dispatched_tick: None,
                    },
                },
            );
            self.retirement_fifo.push_back(id);
            self.write_completions.insert(id, tick);
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
        let (reads, writes) = if command.descriptor.nz_to_nd() {
            let plan = operands.plan_nz2nd(
                id,
                first_read,
                first_write,
                self.config.read_bandwidth,
                self.main_slots,
            )?;
            (C220FixpReadStream::from(plan.reads), Some(plan.writes))
        } else {
            (
                C220FixpReadStream::from(
                    C220FixpReadGenerator::new(command, id, first_read, self.config.read_bandwidth)
                        .map_err(C220FixpEngineError::from)?,
                ),
                None,
            )
        };
        if !self
            .read
            .submit(tick, reads)
            .map_err(C220FixpEngineError::from)?
        {
            return Ok(C220FixpAdmission::DisabledReady);
        }
        if let Some(writes) = writes {
            self.staging.submit(writes);
        }
        self.commands.insert(
            id,
            C220FixpExternalCommandState {
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
    pub(crate) fn admit_with_flags(
        &mut self,
        tick: u64,
        id: u64,
        first_requests: (u32, u32),
        command: C220FixpExternalCommand,
        bindings: &mut C220FixpSyncBindings,
        flags: &mut C220HardwareFlagState,
    ) -> Result<C220FixpAdmission, C220FixpExternalEngineError> {
        if self.commands.contains_key(&id) {
            return Err(C220FixpEngineError::DuplicateCommand(id).into());
        }
        bindings.capture_pending(id, flags);
        if command.command.descriptor.is_disabled()
            && bindings
                .disabled_blocked(tick, id, flags)
                .map_err(C220FixpEngineError::from)?
        {
            return Ok(C220FixpAdmission::HardwareSync);
        }
        self.admit(tick, id, first_requests.0, first_requests.1, command)
    }

    pub fn generate_read(
        &mut self,
        tick: u64,
    ) -> Result<C220FixpReadProgress, C220FixpExternalEngineError> {
        Ok(self
            .read
            .generate(tick)
            .map_err(C220FixpEngineError::from)?)
    }
    pub fn send_read(
        &mut self,
        tick: u64,
        sync: impl C220FixpSync,
    ) -> Result<C220FixpReadProgress, C220FixpExternalEngineError> {
        Ok(self
            .read
            .send(tick, &mut self.input, sync)
            .map_err(C220FixpEngineError::from)?)
    }
    pub fn send_l0c(
        &mut self,
        tick: u64,
        l0c: &mut C220L0c,
    ) -> Result<C220MteL0cReadSend, C220FixpExternalEngineError> {
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
    ) -> Result<
        (C220MteL0cReadResponse, Option<C220FixpFunctionalEvent>),
        C220FixpExternalEngineError,
    > {
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
    ) -> Result<C220FixpConversionReceive, C220FixpExternalEngineError> {
        Ok(self
            .conversion
            .receive(tick, &mut self.input, sync)
            .map_err(C220FixpEngineError::from)?)
    }
    pub fn slice(
        &mut self,
        tick: u64,
    ) -> Result<Option<C220FixpConversionEntry>, C220FixpExternalEngineError> {
        let Some(head) = self.conversion.entries().front() else {
            return Ok(None);
        };
        let id = head.acknowledgment.operation.instruction_id;
        let operands = self
            .commands
            .get(&id)
            .ok_or(C220FixpEngineError::UnknownCommand(id))?
            .operands;
        if operands.command.descriptor.nz_to_nd() {
            Ok(self.staging.receive(tick, &mut self.conversion)?)
        } else {
            let policy = operands
                .output_policy()
                .map_err(C220FixpEngineError::from)?;
            Ok(self
                .output
                .receive_columns(tick, &mut self.conversion, policy)?)
        }
    }
    pub fn transpose(
        &mut self,
        tick: u64,
    ) -> Result<C220FixpTransposeProgress, C220FixpExternalEngineError> {
        Ok(self.staging.stage(tick)?)
    }
    pub fn align(
        &mut self,
        tick: u64,
    ) -> Result<Option<C220FixpNz2ndStagingEntry>, C220FixpExternalEngineError> {
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
            .output_policy()
            .map_err(C220FixpEngineError::from)?;
        Ok(self.output.receive(tick, &mut self.staging, policy)?)
    }
    pub fn packetize(
        &mut self,
        pipeline: &mut C220MtePipeline,
    ) -> Result<Option<C220FixpBiuWrite>, C220FixpExternalEngineError> {
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
    ) -> Result<C220FixpWriteProgress<C220FixpDispatchPacket>, C220FixpExternalEngineError> {
        Ok(self
            .write
            .generate_shared(tick)
            .map_err(C220FixpEngineError::from)?)
    }
    pub fn send_write(
        &mut self,
        pipeline: &mut C220MtePipeline,
    ) -> Result<C220FixpWriteProgress<C220FixpDispatchPacket>, C220FixpExternalEngineError> {
        let result = pipeline.send_fixp_biu_output(&mut self.write)?;
        if let C220FixpWriteProgress::Advanced(C220FixpDispatchPacket::External(packet)) = result
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
    pub fn complete_response(
        &mut self,
        response: C220BiuWriteResponse,
    ) -> Result<Option<u64>, C220FixpExternalEngineError> {
        if response.data.subcore != C220BiuSubcore::Cube {
            return Err(C220FixpExternalEngineError::ResponseSource);
        }
        let Some(id) = response.retired_instruction() else {
            return Ok(None);
        };
        self.complete_write_transport(response.tick, id)?;
        Ok(Some(id))
    }

    pub(crate) fn complete_write_transport(
        &mut self,
        tick: u64,
        id: u64,
    ) -> Result<(), C220FixpExternalEngineError> {
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
        self.write_completions.entry(id).or_insert(tick);
        Ok(())
    }

    /// Consume at most one completed head at the command-retirement phase.
    /// Transport callbacks record completion without freeing command state.
    pub fn retire_ready_write(
        &mut self,
        tick: u64,
    ) -> Result<Option<(u64, C220FixpExternalCommandState)>, C220FixpExternalEngineError> {
        let Some(&id) = self.retirement_fifo.front() else {
            return Ok(None);
        };
        if self
            .last_retirement_tick
            .is_some_and(|previous| previous >= tick)
            || self
                .write_completions
                .get(&id)
                .is_none_or(|&done| done >= tick)
        {
            return Ok(None);
        }
        let state = self
            .commands
            .get(&id)
            .ok_or(C220FixpEngineError::UnknownCommand(id))?;
        if !state.operands.command.descriptor.is_disabled() {
            if state.lifecycle.executed_tick.is_none() {
                return Err(C220FixpEngineError::NotExecuted(id).into());
            }
            if state.lifecycle.write_dispatched_tick.is_none() {
                return Err(C220FixpEngineError::NotDispatched(id).into());
            }
        }
        self.retirement_fifo.pop_front();
        self.write_completions.remove(&id);
        self.last_retirement_tick = Some(tick);
        Ok(Some((
            id,
            self.commands.remove(&id).expect("validated command"),
        )))
    }
}
