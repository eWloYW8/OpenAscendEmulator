use std::collections::{BTreeMap, VecDeque};

use super::*;
use crate::isa::c220::mte::fixp::C220FixpDestination;
use crate::sim::c220::memory::C220L0c;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FixpRetiredCommand {
    pub instruction_id: u64,
    pub lifecycle: C220FixpCommandState,
    /// BIU transport operands, including for L1-destination NZ2ND commands.
    pub external: Option<C220FixpExternalCommand>,
}

#[cfg(test)]
#[path = "runtime_tests.rs"]
mod tests;
#[derive(Debug, thiserror::Error)]
pub enum C220FixpRuntimeError {
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

/// L1, external and factor commands share one engine and output burst queue.
/// Each method is a distinct scheduler callback. Numerical FIX writes occur
/// on final L0C read acceptance; retirement waits for destination completion.
#[derive(Debug, Clone)]
pub struct C220FixpRuntime {
    shared: C220FixpEngine,
    main_slots: u32,
    operands: BTreeMap<u64, C220FixpExternalCommand>,
    staging: C220FixpNz2ndStaging,
    output: C220FixpExternalOutput,
}

impl C220FixpRuntime {
    pub fn new(
        config: C220FixpEngineConfig,
        main_slots: u32,
        total_slots: usize,
    ) -> Result<Self, C220FixpRuntimeError> {
        Ok(Self {
            shared: C220FixpEngine::new(config)?,
            main_slots,
            operands: BTreeMap::new(),
            staging: C220FixpNz2ndStaging::new(total_slots)?,
            output: C220FixpExternalOutput::default(),
        })
    }

    pub fn commands(&self) -> &BTreeMap<u64, C220FixpCommandState> {
        self.shared.commands()
    }
    pub fn external_operands(&self) -> &BTreeMap<u64, C220FixpExternalCommand> {
        &self.operands
    }
    pub fn instruction_fifo(&self) -> &VecDeque<u64> {
        self.shared.instruction_fifo()
    }
    pub fn retirement_fifo(&self) -> &VecDeque<u64> {
        self.shared.retirement_fifo()
    }
    pub fn shared_engine(&self) -> &C220FixpEngine {
        &self.shared
    }
    pub(crate) fn shared_engine_mut(&mut self) -> &mut C220FixpEngine {
        &mut self.shared
    }
    pub fn write_completion_tick(&self, id: u64) -> Option<u64> {
        self.shared.write_completion_tick(id)
    }
    pub fn active_resource_key(&self) -> Option<C220FixpResourceKey> {
        self.shared.active_resource_key()
    }
    pub fn resource_conflict(&self, command: C220FixpExternalCommand) -> bool {
        self.shared.resource_conflict_for(
            command.command,
            crate::isa::c220::mte::fixp::C220FixpDestination::External,
        )
    }
    pub fn read_pipeline(&self) -> &C220FixpReadPipeline {
        &self.shared.datapath.read
    }
    pub fn read_interface(&self) -> &C220MteL0cReadInterface {
        &self.shared.datapath.input
    }
    pub fn functional(&self) -> &C220FixpFunctionalState {
        &self.shared.datapath.functional
    }
    pub fn conversion(&self) -> &C220FixpConversionPipeline {
        &self.shared.datapath.conversion
    }
    pub fn staging(&self) -> &C220FixpNz2ndStaging {
        &self.staging
    }
    pub fn output(&self) -> &C220FixpExternalOutput {
        &self.output
    }
    pub fn write_pipeline(&self) -> &C220FixpDispatchPipeline {
        &self.shared.datapath.write
    }

    pub fn is_idle(&self) -> bool {
        self.shared.is_idle() && self.staging.is_idle() && self.output.bursts().is_empty()
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
    ) -> Result<C220FixpAdmission, C220FixpRuntimeError> {
        self.admit_routed(
            tick,
            id,
            (first_read, first_write),
            (C220FixpDestination::External, operands),
        )
    }

    fn admit_routed(
        &mut self,
        tick: u64,
        id: u64,
        first_requests: (u32, u32),
        (destination, operands): (C220FixpDestination, C220FixpExternalCommand),
    ) -> Result<C220FixpAdmission, C220FixpRuntimeError> {
        let (first_read, first_write) = first_requests;
        if self.shared.contains_command(id) {
            return Err(C220FixpEngineError::DuplicateCommand(id).into());
        }
        let command = operands.command;
        if command.descriptor.is_disabled() {
            self.shared.record_command(tick, id, command, destination);
            self.operands.insert(id, operands);
            return Ok(C220FixpAdmission::DisabledReady);
        }
        if self.shared.resource_conflict_for(command, destination) {
            return Ok(C220FixpAdmission::ResourceConflict);
        }
        if let Some(blocked) = self.shared.admission_backpressure() {
            return Ok(blocked);
        }
        command
            .validate_activation()
            .map_err(C220FixpEngineError::from)?;
        let (reads, writes) = if command.descriptor.nz_to_nd() {
            let plan = operands.plan_nz2nd(
                id,
                first_read,
                first_write,
                self.shared.config.read_bandwidth,
                self.main_slots,
            )?;
            (C220FixpReadStream::from(plan.reads), Some(plan.writes))
        } else {
            (
                C220FixpReadStream::from(
                    C220FixpReadGenerator::new(
                        command,
                        id,
                        first_read,
                        self.shared.config.read_bandwidth,
                    )
                    .map_err(C220FixpEngineError::from)?,
                ),
                None,
            )
        };
        if !self
            .shared
            .datapath
            .read
            .submit(tick, reads)
            .map_err(C220FixpEngineError::from)?
        {
            return Ok(C220FixpAdmission::DisabledReady);
        }
        if let Some(writes) = writes {
            self.staging.submit(writes);
        }
        self.shared.record_command(tick, id, command, destination);
        self.operands.insert(id, operands);
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
    ) -> Result<C220FixpAdmission, C220FixpRuntimeError> {
        self.admit_routed_with_flags(
            tick,
            id,
            first_requests,
            (C220FixpDestination::External, command),
            bindings,
            flags,
        )
    }

    pub(crate) fn admit_l1_nz2nd_with_flags(
        &mut self,
        tick: u64,
        id: u64,
        first_requests: (u32, u32),
        command: C220FixpExternalCommand,
        bindings: &mut C220FixpSyncBindings,
        flags: &mut C220HardwareFlagState,
    ) -> Result<C220FixpAdmission, C220FixpRuntimeError> {
        self.admit_routed_with_flags(
            tick,
            id,
            first_requests,
            (C220FixpDestination::L1, command),
            bindings,
            flags,
        )
    }

    fn admit_routed_with_flags(
        &mut self,
        tick: u64,
        id: u64,
        first_requests: (u32, u32),
        routed: (C220FixpDestination, C220FixpExternalCommand),
        bindings: &mut C220FixpSyncBindings,
        flags: &mut C220HardwareFlagState,
    ) -> Result<C220FixpAdmission, C220FixpRuntimeError> {
        let command = routed.1;
        if self.shared.commands().contains_key(&id)
            || self.shared.factor_commands().contains_key(&id)
        {
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
        self.admit_routed(tick, id, first_requests, routed)
    }

    pub fn generate_read(
        &mut self,
        tick: u64,
    ) -> Result<C220FixpReadProgress, C220FixpRuntimeError> {
        Ok(self.shared.generate_read(tick)?)
    }
    pub fn send_read(
        &mut self,
        tick: u64,
        sync: impl C220FixpSync,
    ) -> Result<C220FixpReadProgress, C220FixpRuntimeError> {
        Ok(self.shared.send_read(tick, sync)?)
    }
    pub fn send_l0c(
        &mut self,
        tick: u64,
        l0c: &mut C220L0c,
    ) -> Result<C220MteL0cReadSend, C220FixpRuntimeError> {
        Ok(self.shared.send_l0c(tick, l0c)?)
    }
    pub fn receive_l0c(
        &mut self,
        tick: u64,
        memory: &mut C220FixpRuntimeMemory<'_>,
        mut observe: impl FnMut(&C220FixpSliceResult, Option<&[u8]>),
    ) -> Result<(C220MteL0cReadResponse, Option<C220FixpFunctionalEvent>), C220FixpRuntimeError>
    {
        let response = self
            .shared
            .datapath
            .input
            .receive(tick, memory.l0c)
            .map_err(C220FixpEngineError::from)?;
        let functional = if let C220MteL0cReadResponse::Accepted(ack) = response {
            let id = ack.operation.instruction_id;
            let state = self
                .shared
                .commands
                .get_mut(&id)
                .ok_or(C220FixpEngineError::UnknownCommand(id))?;
            let event = if state.destination == C220FixpDestination::External {
                let operands = self
                    .operands
                    .get(&id)
                    .ok_or(C220FixpEngineError::UnknownCommand(id))?;
                self.shared.datapath.functional.accept_read_with(
                    ack,
                    memory.l0c.buffer(),
                    |snapshot| {
                        operands.execute_to_external(
                            snapshot,
                            memory.slopes,
                            memory.external,
                            memory.atomics,
                            |slice, bytes| observe(slice, Some(bytes)),
                        )
                    },
                )
            } else {
                self.shared.datapath.functional.accept_read(
                    ack,
                    state.command,
                    memory.l0c.buffer(),
                    memory.slopes,
                    memory.l1,
                    |slice| observe(slice, None),
                )
            }
            .map_err(C220FixpEngineError::from)?;
            if event.executed {
                state.executed_tick = Some(tick);
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
    ) -> Result<C220FixpConversionReceive, C220FixpRuntimeError> {
        Ok(self.shared.convert(tick, sync)?)
    }
    pub fn slice(
        &mut self,
        tick: u64,
    ) -> Result<Option<C220FixpConversionEntry>, C220FixpRuntimeError> {
        let Some(head) = self.shared.datapath.conversion.entries().front() else {
            return Ok(None);
        };
        let id = head.acknowledgment.operation.instruction_id;
        if !self.shared.commands.contains_key(&id) {
            return Err(C220FixpEngineError::UnknownCommand(id).into());
        }
        if let Some(operands) = self.operands.get(&id) {
            if operands.command.descriptor.nz_to_nd() {
                return Ok(self
                    .staging
                    .receive(tick, &mut self.shared.datapath.conversion)?);
            }
            let policy = operands
                .output_policy()
                .map_err(C220FixpEngineError::from)?;
            Ok(self
                .output
                .receive_columns(tick, &mut self.shared.datapath.conversion, policy)?)
        } else {
            Ok(self.output.receive_columns(
                tick,
                &mut self.shared.datapath.conversion,
                C220FixpExternalOutputPolicy::new(0, 0),
            )?)
        }
    }
    pub fn transpose(
        &mut self,
        tick: u64,
    ) -> Result<C220FixpTransposeProgress, C220FixpRuntimeError> {
        Ok(self.staging.stage(tick)?)
    }
    pub fn align(
        &mut self,
        tick: u64,
    ) -> Result<Option<C220FixpNz2ndStagingEntry>, C220FixpRuntimeError> {
        let Some(head) = self.staging.alignment().front() else {
            return Ok(None);
        };
        let id = head.operation.instruction_id;
        let state = self
            .operands
            .get(&id)
            .ok_or(C220FixpEngineError::UnknownCommand(id))?;
        let policy = state.output_policy().map_err(C220FixpEngineError::from)?;
        Ok(self.output.receive(tick, &mut self.staging, policy)?)
    }
    pub fn packetize(
        &mut self,
        pipeline: &mut C220MtePipeline,
    ) -> Result<Option<C220FixpDispatchPacket>, C220FixpRuntimeError> {
        let Some(head) = self.output.bursts().front() else {
            return Ok(None);
        };
        let id = head.instruction_id;
        if let Some(operands) = self.operands.get(&id) {
            Ok(pipeline
                .packetize_fixp_biu_output(
                    &mut self.output,
                    &mut self.shared.datapath.write,
                    operands.biu_mode(),
                )?
                .map(C220FixpDispatchPacket::External))
        } else {
            if !self.shared.commands.contains_key(&id) {
                return Err(C220FixpEngineError::UnknownCommand(id).into());
            }
            Ok(self
                .shared
                .datapath
                .write
                .packetize_l1_shared(pipeline.tick(), &mut self.output)
                .map_err(C220FixpEngineError::from)?
                .map(C220FixpDispatchPacket::Write))
        }
    }
    pub fn generate_write(
        &mut self,
        tick: u64,
    ) -> Result<C220FixpWriteProgress<C220FixpDispatchPacket>, C220FixpRuntimeError> {
        Ok(self.shared.generate_write(tick)?)
    }
    pub fn send_write(
        &mut self,
        pipeline: &mut C220MtePipeline,
    ) -> Result<C220FixpWriteProgress<C220FixpDispatchPacket>, C220FixpRuntimeError> {
        let result = pipeline.send_fixp_biu_output(&mut self.shared.datapath.write)?;
        self.shared
            .observe_write_dispatch(pipeline.tick(), result)?;
        Ok(result)
    }
    /// Only pass responses accepted by the shared BIU response handler.
    pub fn complete_response(
        &mut self,
        response: C220BiuWriteResponse,
    ) -> Result<Option<u64>, C220FixpRuntimeError> {
        if response.data.subcore != C220BiuSubcore::Cube {
            return Err(C220FixpRuntimeError::ResponseSource);
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
    ) -> Result<(), C220FixpRuntimeError> {
        let state = self
            .shared
            .commands()
            .get(&id)
            .ok_or(C220FixpEngineError::UnknownCommand(id))?;
        if state.executed_tick.is_none() {
            return Err(C220FixpEngineError::NotExecuted(id).into());
        }
        Ok(self.shared.complete_write_transport(tick, id)?)
    }

    /// Consume at most one completed head at the command-retirement phase.
    /// Transport callbacks record completion without freeing command state.
    pub fn retire_ready_write(
        &mut self,
        tick: u64,
    ) -> Result<Option<C220FixpRetiredCommand>, C220FixpRuntimeError> {
        let Some((instruction_id, lifecycle)) = self.shared.take_ready_write(tick)? else {
            return Ok(None);
        };
        Ok(Some(C220FixpRetiredCommand {
            instruction_id,
            lifecycle,
            external: self.operands.remove(&instruction_id),
        }))
    }
}
