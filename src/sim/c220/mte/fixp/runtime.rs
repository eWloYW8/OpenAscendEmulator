use crate::sim::c220::sync::{C220HardwareFlagState, C220HardwareFlagTimingError};
use std::collections::{BTreeMap, VecDeque};

use super::*;
use crate::sim::c220::memory::{C220L0c, C220LocalBuffer};
use crate::sim::c220::mte::interface::{
    C220MteL0cReadError, C220MteL0cReadInterface, C220MteL0cReadResponse, C220MteL0cReadSend,
    C220MteOutputFragment,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FixpEngineConfig {
    pub instruction_fifo_depth: u32,
    pub read_bandwidth: u32,
    pub read_bank_count: u8,
    pub read_data_latency: u32,
    pub l0c_capacity: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FixpCommandState {
    pub command: C220FixpCommand,
    pub admitted_tick: u64,
    pub executed_tick: Option<u64>,
    pub write_dispatched_tick: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FactorCommandState {
    pub load: crate::isa::c220::mte::factor::C220FactorLoad,
    pub admitted_tick: u64,
    pub dispatched_tick: Option<u64>,
    pub completed_tick: Option<u64>,
}

#[derive(Debug, thiserror::Error)]
pub enum C220FixpEngineError {
    #[error("FIX write command {0} is not the instruction FIFO head")]
    CommandOrder(u64),
    #[error("FIX command {0} is not the retirement FIFO head")]
    RetirementOrder(u64),
    #[error(transparent)]
    Sync(#[from] C220HardwareFlagTimingError),
    #[error("FIX command identity {0} is already active")]
    DuplicateCommand(u64),
    #[error("FIX command identity {0} is not active")]
    UnknownCommand(u64),
    #[error("FIX command {0} cannot retire before functional execution")]
    NotExecuted(u64),
    #[error("FIX command {0} cannot retire before its final write dispatch")]
    NotDispatched(u64),
    #[error("factor command {0} cannot retire before transport completion")]
    FactorIncomplete(u64),
    #[error(transparent)]
    Generator(#[from] C220FixpReadGeneratorError),
    #[error(transparent)]
    ReadPipeline(#[from] C220FixpReadPipelineError),
    #[error(transparent)]
    ReadInterface(#[from] C220MteL0cReadError),
    #[error(transparent)]
    Functional(#[from] C220FixpExecutionError),
    #[error(transparent)]
    Conversion(#[from] C220FixpConversionError),
    #[error(transparent)]
    Output(#[from] C220FixpL1OutputError),
    #[error(transparent)]
    WritePipeline(#[from] C220FixpWritePipelineError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220FixpAdmission {
    Mte3RetirementPending,
    ResourceConflict,
    ReadGenerationBusy,
    InstructionFifoFull,
    Active,
    DisabledReady,
    HardwareSync,
}

/// Ordinary 32-bit-source FIX-to-L1 execution from admitted command to write
/// handoff. The owner supplies scheduler callbacks, hardware synchronization,
/// shared memories and write acknowledgments. Frontend queue timing and global
/// phase order are external. NZ2ND uses a separate engine; unsupported source
/// and conversion combinations fail during command validation.
#[derive(Debug, Clone)]
pub struct C220FixpEngine {
    config: C220FixpEngineConfig,
    commands: BTreeMap<u64, C220FixpCommandState>,
    factor_commands: BTreeMap<u64, C220FactorCommandState>,
    instruction_fifo: VecDeque<u64>,
    retirement_fifo: VecDeque<u64>,
    command_retirement: VecDeque<u64>,
    write_completions: BTreeMap<u64, u64>,
    last_retirement_tick: Option<u64>,
    read: C220FixpReadPipeline,
    input: C220MteL0cReadInterface,
    functional: C220FixpFunctionalState,
    conversion: C220FixpConversionPipeline,
    output: C220FixpL1Output,
    write: C220FixpDispatchPipeline,
}

impl C220FixpEngine {
    /// Called at the frontend's decode/dispatch boundary after its queue
    /// admission checks. DisabledReady still needs ordered frontend retirement.
    pub fn admit_with_flags(
        &mut self,
        tick: u64,
        id: u64,
        first_request: u32,
        command: C220FixpCommand,
        bindings: &mut C220FixpSyncBindings,
        flags: &mut C220HardwareFlagState,
    ) -> Result<C220FixpAdmission, C220FixpEngineError> {
        if self.commands.contains_key(&id) || self.factor_commands.contains_key(&id) {
            return Err(C220FixpEngineError::DuplicateCommand(id));
        }
        bindings.capture_pending(id, flags);
        if command.descriptor.is_disabled() && bindings.disabled_blocked(tick, id, flags)? {
            return Ok(C220FixpAdmission::HardwareSync);
        }
        self.admit(tick, id, first_request, command)
    }

    pub fn new(config: C220FixpEngineConfig) -> Result<Self, C220FixpEngineError> {
        if config.read_bandwidth == 0 {
            return Err(C220FixpReadGeneratorError::ZeroBandwidth.into());
        }
        Ok(Self {
            config,
            commands: BTreeMap::new(),
            factor_commands: BTreeMap::new(),
            instruction_fifo: VecDeque::new(),
            retirement_fifo: VecDeque::new(),
            command_retirement: VecDeque::new(),
            write_completions: BTreeMap::new(),
            last_retirement_tick: None,
            read: C220FixpReadPipeline::default(),
            input: C220MteL0cReadInterface::new(config.read_bank_count, config.read_data_latency)?,
            functional: C220FixpFunctionalState::new(config.l0c_capacity),
            conversion: C220FixpConversionPipeline::default(),
            output: C220FixpL1Output::default(),
            write: C220FixpDispatchPipeline::default(),
        })
    }

    pub fn commands(&self) -> &BTreeMap<u64, C220FixpCommandState> {
        &self.commands
    }
    pub fn factor_commands(&self) -> &BTreeMap<u64, C220FactorCommandState> {
        &self.factor_commands
    }
    pub fn command_retirement_head(&self) -> Option<u64> {
        self.command_retirement.front().copied()
    }

    pub fn command_retirement_queue(&self) -> &VecDeque<u64> {
        &self.command_retirement
    }

    pub fn write_completion_tick(&self, id: u64) -> Option<u64> {
        self.write_completions.get(&id).copied()
    }

    pub fn can_retire_at(&self, tick: u64, id: u64) -> bool {
        self.command_retirement_head() == Some(id)
            && self
                .last_retirement_tick
                .is_none_or(|previous| previous < tick)
    }
    pub fn instruction_fifo(&self) -> &VecDeque<u64> {
        &self.instruction_fifo
    }

    pub fn retirement_fifo(&self) -> &VecDeque<u64> {
        &self.retirement_fifo
    }

    /// Configuration changes wait for retirement, not just write dispatch.
    /// Addresses, shape and slope values do not select a different resource.
    pub fn resource_conflict(&self, command: C220FixpCommand) -> bool {
        !command.descriptor.is_disabled()
            && self.retirement_fifo.back().is_some_and(|id| {
                let previous = self.commands[id].command;
                previous.descriptor.conversion_mode() != command.descriptor.conversion_mode()
                    || previous.descriptor.activation_mode() != command.descriptor.activation_mode()
                    || (previous.control ^ command.control) & (1 << 48) != 0
            })
    }

    pub fn admission_backpressure(&self) -> Option<C220FixpAdmission> {
        if self.read.generated_batches() != 0 {
            Some(C220FixpAdmission::ReadGenerationBusy)
        } else if self.instruction_fifo.len() >= self.config.instruction_fifo_depth as usize {
            Some(C220FixpAdmission::InstructionFifoFull)
        } else {
            None
        }
    }
    pub fn read_pipeline(&self) -> &C220FixpReadPipeline {
        &self.read
    }
    pub fn read_interface(&self) -> &C220MteL0cReadInterface {
        &self.input
    }
    pub fn conversion(&self) -> &C220FixpConversionPipeline {
        &self.conversion
    }
    pub fn output(&self) -> &C220FixpL1Output {
        &self.output
    }
    pub fn write_pipeline(&self) -> &C220FixpDispatchPipeline {
        &self.write
    }
    pub fn functional(&self) -> &C220FixpFunctionalState {
        &self.functional
    }

    pub fn stage_ready_tick(&self, stage: C220FixpStage, tick: u64) -> Option<u64> {
        use C220FixpStage::*;
        match stage {
            GenerateRead => self.read.generated_ready_tick(),
            SendRead => self
                .read
                .dispatch_queue()
                .front()
                .map(|head| head.ready_tick),
            SendL0c => self.input.input().front().map(|head| head.ready_tick),
            ReceiveL0c => self.input.pending().front().map(|_| tick),
            Convert => self
                .input
                .acknowledgments()
                .front()
                .map(|head| head.ready_tick()),
            Slice => self
                .conversion
                .entries()
                .front()
                .map(|head| head.ready_tick),
            Packetize => self.output.bursts().front().map(|head| head.ready_tick),
            GenerateWrite => self.write.packets().front().map(|head| head.ready_tick),
            SendWrite => self
                .write
                .dispatch_queue()
                .front()
                .map(|head| head.ready_tick),
        }
    }

    pub fn is_idle(&self) -> bool {
        self.commands.is_empty()
            && self.factor_commands.is_empty()
            && self.read.is_idle()
            && self.input.is_idle()
            && self.conversion.entries().is_empty()
            && self.output.bursts().is_empty()
            && self.write.is_idle()
    }

    /// Request IDs must be unique across outstanding commands. Callers that
    /// have not resolved attached flags must use `admit_with_flags`.
    pub fn admit(
        &mut self,
        tick: u64,
        id: u64,
        first_request: u32,
        command: C220FixpCommand,
    ) -> Result<C220FixpAdmission, C220FixpEngineError> {
        if self.commands.contains_key(&id) || self.factor_commands.contains_key(&id) {
            return Err(C220FixpEngineError::DuplicateCommand(id));
        }
        if command.descriptor.is_disabled() {
            self.commands.insert(
                id,
                C220FixpCommandState {
                    command,
                    admitted_tick: tick,
                    executed_tick: None,
                    write_dispatched_tick: None,
                },
            );
            self.retirement_fifo.push_back(id);
            self.command_retirement.push_back(id);
            self.write_completions.insert(id, tick);
            return Ok(C220FixpAdmission::DisabledReady);
        }
        if self.resource_conflict(command) {
            return Ok(C220FixpAdmission::ResourceConflict);
        }
        if let Some(blocked) = self.admission_backpressure() {
            return Ok(blocked);
        }
        command.validate_activation()?;
        let packets =
            C220FixpReadGenerator::new(command, id, first_request, self.config.read_bandwidth)?;
        if !self.read.submit(tick, packets)? {
            return Ok(C220FixpAdmission::DisabledReady);
        }
        self.commands.insert(
            id,
            C220FixpCommandState {
                command,
                admitted_tick: tick,
                executed_tick: None,
                write_dispatched_tick: None,
            },
        );
        self.instruction_fifo.push_back(id);
        self.retirement_fifo.push_back(id);
        self.command_retirement.push_back(id);
        Ok(C220FixpAdmission::Active)
    }

    pub fn generate_read(
        &mut self,
        tick: u64,
    ) -> Result<C220FixpReadProgress, C220FixpEngineError> {
        Ok(self.read.generate(tick)?)
    }

    pub fn send_read(
        &mut self,
        tick: u64,
        sync: impl C220FixpSync,
    ) -> Result<C220FixpReadProgress, C220FixpEngineError> {
        Ok(self.read.send(tick, &mut self.input, sync)?)
    }

    pub fn send_l0c(
        &mut self,
        tick: u64,
        l0c: &mut C220L0c,
    ) -> Result<C220MteL0cReadSend, C220FixpEngineError> {
        Ok(self.input.send_queued(tick, l0c)?)
    }

    /// Snapshot and numerical execution occur in this callback, before delayed
    /// conversion admission. Numerical faults terminate execution; a consumed
    /// memory acknowledgment must not be retried as a fresh read.
    pub fn receive_l0c(
        &mut self,
        tick: u64,
        l0c: &mut C220L0c,
        slopes: &C220LocalBuffer,
        l1: &mut C220LocalBuffer,
        observe: impl FnMut(&C220FixpSliceResult),
    ) -> Result<(C220MteL0cReadResponse, Option<C220FixpFunctionalEvent>), C220FixpEngineError>
    {
        let response = self.input.receive(tick, l0c)?;
        let functional = if let C220MteL0cReadResponse::Accepted(ack) = response {
            let id = ack.operation.instruction_id;
            let state = self
                .commands
                .get_mut(&id)
                .ok_or(C220FixpEngineError::UnknownCommand(id))?;
            let event = self.functional.accept_read(
                ack,
                state.command,
                l0c.buffer(),
                slopes,
                l1,
                observe,
            )?;
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
    ) -> Result<C220FixpConversionReceive, C220FixpEngineError> {
        Ok(self.conversion.receive(tick, &mut self.input, sync)?)
    }

    pub fn slice(
        &mut self,
        tick: u64,
    ) -> Result<Option<C220FixpConversionEntry>, C220FixpEngineError> {
        Ok(self.output.receive(tick, &mut self.conversion)?)
    }

    pub fn packetize(
        &mut self,
        tick: u64,
    ) -> Result<Option<C220MteOutputFragment>, C220FixpEngineError> {
        Ok(self.write.packetize_output(tick, &mut self.output)?)
    }

    pub fn generate_write(
        &mut self,
        tick: u64,
    ) -> Result<C220FixpWriteProgress<C220FixpDispatchPacket>, C220FixpEngineError> {
        Ok(self.write.generate_shared(tick)?)
    }

    pub fn send_write(
        &mut self,
        tick: u64,
        interface: &mut C220FixpL1WriteInterface,
        reader: &mut crate::sim::c220::mte::interface::C220MteL1Interface<
            crate::sim::c220::mte::C220MteReadPayload,
        >,
        biu: Option<
            &mut crate::sim::c220::mte::interface::biu_write::command::C220BiuWriteCommands,
        >,
    ) -> Result<C220FixpWriteProgress<C220FixpDispatchPacket>, C220FixpEngineError> {
        let result = self.write.send_shared(tick, interface, reader, biu)?;
        if let C220FixpWriteProgress::Advanced(C220FixpDispatchPacket::FactorRead {
            operation, ..
        }) = result
            && operation.last_in_instruction
        {
            let id = operation.instruction_id;
            if self.instruction_fifo.front() != Some(&id) {
                return Err(C220FixpEngineError::CommandOrder(id));
            }
            self.factor_commands
                .get_mut(&id)
                .ok_or(C220FixpEngineError::UnknownCommand(id))?
                .dispatched_tick = Some(tick);
            self.instruction_fifo.pop_front();
        }
        if let C220FixpWriteProgress::Advanced(C220FixpDispatchPacket::Write(fragment)) = result
            && fragment.last_in_instruction
        {
            let id = fragment.instruction_id;
            if self.instruction_fifo.front() != Some(&id) {
                return Err(C220FixpEngineError::CommandOrder(id));
            }
            let state = self
                .commands
                .get_mut(&id)
                .ok_or(C220FixpEngineError::UnknownCommand(id))?;
            state.write_dispatched_tick = Some(tick);
            self.instruction_fifo.pop_front();
        }
        Ok(result)
    }

    /// Queues one captured factor command without materializing its requests.
    /// Functional effects, attached flags and ordered command retirement remain
    /// the frontend's responsibility; transport completion does not free state.
    pub fn admit_factor_batch(
        &mut self,
        tick: u64,
        port: crate::sim::c220::mte::interface::C220MteL1ReadPort,
        cursor: crate::sim::c220::mte::factor::C220FactorRequestCursor,
    ) -> Result<C220FixpAdmission, C220FixpEngineError> {
        let id = cursor.instruction_id();
        if self.commands.contains_key(&id) || self.factor_commands.contains_key(&id) {
            return Err(C220FixpEngineError::DuplicateCommand(id));
        }
        if cursor.remaining() == 0 {
            self.factor_commands.insert(
                id,
                C220FactorCommandState {
                    load: cursor.load(),
                    admitted_tick: tick,
                    dispatched_tick: None,
                    completed_tick: Some(tick),
                },
            );
            self.command_retirement.push_back(id);
            return Ok(C220FixpAdmission::DisabledReady);
        }
        if !self.retirement_fifo.is_empty() {
            return Ok(C220FixpAdmission::ResourceConflict);
        }
        if let Some(blocked) = self.admission_backpressure() {
            return Ok(blocked);
        }
        self.write.enqueue_factor_batch(tick, port, cursor)?;
        self.factor_commands.insert(
            id,
            C220FactorCommandState {
                load: cursor.load(),
                admitted_tick: tick,
                dispatched_tick: None,
                completed_tick: None,
            },
        );
        self.instruction_fifo.push_back(id);
        self.command_retirement.push_back(id);
        Ok(C220FixpAdmission::Active)
    }

    pub(crate) fn complete_factor_transport(
        &mut self,
        tick: u64,
        id: u64,
    ) -> Result<(), C220FixpEngineError> {
        let state = self
            .factor_commands
            .get_mut(&id)
            .ok_or(C220FixpEngineError::UnknownCommand(id))?;
        if state.dispatched_tick.is_none() {
            return Err(C220FixpEngineError::NotDispatched(id));
        }
        state.completed_tick.get_or_insert(tick);
        Ok(())
    }

    /// Called after the frontend has applied functional effects and satisfied
    /// its ordered retirement and synchronization checks.
    pub fn retire_factor(
        &mut self,
        tick: u64,
        id: u64,
    ) -> Result<C220FactorCommandState, C220FixpEngineError> {
        let state = self
            .factor_commands
            .get(&id)
            .ok_or(C220FixpEngineError::UnknownCommand(id))?;
        if state.completed_tick.is_none() {
            return Err(C220FixpEngineError::FactorIncomplete(id));
        }
        if !self.can_retire_at(tick, id) || state.completed_tick.is_some_and(|done| done >= tick) {
            return Err(C220FixpEngineError::RetirementOrder(id));
        }
        self.command_retirement.pop_front();
        self.last_retirement_tick = Some(tick);
        Ok(self
            .factor_commands
            .remove(&id)
            .expect("validated factor command"))
    }

    pub(crate) fn complete_write_transport(
        &mut self,
        tick: u64,
        id: u64,
    ) -> Result<(), C220FixpEngineError> {
        let state = self
            .commands
            .get(&id)
            .ok_or(C220FixpEngineError::UnknownCommand(id))?;
        if state.write_dispatched_tick.is_none() {
            return Err(C220FixpEngineError::NotDispatched(id));
        }
        self.write_completions.entry(id).or_insert(tick);
        Ok(())
    }

    /// Runs the ordinary-write retirement checkpoint once per core clock.
    /// A factor at the head is committed by the functional-memory owner.
    pub(crate) fn retire_ready_write(&mut self, tick: u64) -> Result<(), C220FixpEngineError> {
        let Some(id) = self.command_retirement_head() else {
            return Ok(());
        };
        if self.can_retire_at(tick, id)
            && self
                .write_completions
                .get(&id)
                .is_some_and(|&done| done < tick)
        {
            self.retire(id)?;
            self.write_completions.remove(&id);
            self.last_retirement_tick = Some(tick);
        }
        Ok(())
    }

    fn retire(&mut self, id: u64) -> Result<C220FixpCommandState, C220FixpEngineError> {
        let state = self
            .commands
            .get(&id)
            .ok_or(C220FixpEngineError::UnknownCommand(id))?;
        if !state.command.descriptor.is_disabled() && state.executed_tick.is_none() {
            return Err(C220FixpEngineError::NotExecuted(id));
        }
        if !state.command.descriptor.is_disabled() && state.write_dispatched_tick.is_none() {
            return Err(C220FixpEngineError::NotDispatched(id));
        }
        if self.retirement_fifo.front() != Some(&id) || self.command_retirement_head() != Some(id) {
            return Err(C220FixpEngineError::RetirementOrder(id));
        }
        self.retirement_fifo.pop_front();
        self.command_retirement.pop_front();
        Ok(self.commands.remove(&id).expect("validated command"))
    }
}
