use std::collections::{BTreeMap, VecDeque};
use std::num::NonZeroU32;

use super::C220L1OutputReadPlan;
use crate::isa::c220::mte::l1_to_out::C220MovL1ToOutTransfer;
use crate::sim::c220::mte::fixp::*;
use crate::sim::c220::mte::interface::{
    C220MteL1Interface, C220MteL1OutputDestination, C220MteL1ReadPort, C220MteL1ReadRequest,
    C220MteOutputFragment,
    biu_read::C220BiuSubcore,
    biu_write::command::{C220BiuWriteCommands, C220BiuWriteInput},
    biu_write::data::C220BiuWriteResponse,
};
use crate::sim::c220::mte::uop::C220DmaUopMode;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220L1OutputEngineConfig {
    pub read_bandwidth: NonZeroU32,
    pub instruction_fifo_depth: u32,
    pub write_outstanding_limit: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220L1OutputCommand {
    pub transfer: C220MovL1ToOutTransfer,
    pub control: u64,
    pub mode: C220DmaUopMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220L1OutputCommandState {
    pub command: C220L1OutputCommand,
    pub admitted_tick: u64,
    pub source_completed_tick: Option<u64>,
    pub write_dispatched_tick: Option<u64>,
    pub response_tick: Option<u64>,
}

#[derive(Debug, thiserror::Error)]
pub enum C220L1OutputEngineError {
    #[error("L1 output command {0} is already active")]
    DuplicateCommand(u64),
    #[error("L1 output command {0} is not active")]
    UnknownCommand(u64),
    #[error("L1 output command {0} completed out of order")]
    CommandOrder(u64),
    #[error("L1 output command {0} has not completed its write transport")]
    Incomplete(u64),
    #[error("L1 output received a response from the wrong interface")]
    ResponseSource,
    #[error(transparent)]
    Read(#[from] C220FixpReadPipelineError),
    #[error(transparent)]
    Output(#[from] C220FixpExternalOutputError),
    #[error(transparent)]
    Write(#[from] C220FixpWritePipelineError),
}

/// Independent L1-source engine. The surrounding MTE3 scheduler owns issue,
/// generator switching, functional commit and ordered retirement. Physical L1
/// service, BIU transport and output tokens are supplied by the core.
#[derive(Debug, Clone)]
pub struct C220L1OutputEngine {
    config: C220L1OutputEngineConfig,
    read: C220FixpReadPipeline,
    output: C220FixpExternalOutput,
    write: C220FixpDispatchPipeline,
    commands: BTreeMap<u64, C220L1OutputCommandState>,
    instruction_fifo: VecDeque<u64>,
}

impl C220L1OutputEngine {
    pub fn new(config: C220L1OutputEngineConfig) -> Self {
        Self {
            config,
            read: C220FixpReadPipeline::default(),
            output: C220FixpExternalOutput::default(),
            write: C220FixpDispatchPipeline::default(),
            commands: BTreeMap::new(),
            instruction_fifo: VecDeque::new(),
        }
    }

    pub fn commands(&self) -> &BTreeMap<u64, C220L1OutputCommandState> {
        &self.commands
    }
    pub fn instruction_fifo(&self) -> &VecDeque<u64> {
        &self.instruction_fifo
    }
    pub fn read_pipeline(&self) -> &C220FixpReadPipeline {
        &self.read
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
            && self.output.bursts().is_empty()
            && self.write.is_idle()
    }

    /// Called after scheduler-level admission and synchronization. A zero-read
    /// command is returned to the scheduler for its disabled-command handling.
    pub fn admit(
        &mut self,
        tick: u64,
        id: u64,
        command: C220L1OutputCommand,
    ) -> Result<C220FixpAdmission, C220L1OutputEngineError> {
        if self.commands.contains_key(&id) {
            return Err(C220L1OutputEngineError::DuplicateCommand(id));
        }
        if self.read.generated_batches() != 0 {
            return Ok(C220FixpAdmission::ReadGenerationBusy);
        }
        if self.instruction_fifo.len() >= self.config.instruction_fifo_depth as usize {
            return Ok(C220FixpAdmission::InstructionFifoFull);
        }
        let reads =
            C220L1OutputReadPlan::new(command.transfer, command.mode, self.config.read_bandwidth);
        if !self.read.submit(
            tick,
            C220FixpReadStream::L1Output {
                instruction_id: id,
                reads,
            },
        )? {
            return Ok(C220FixpAdmission::DisabledReady);
        }
        self.commands.insert(
            id,
            C220L1OutputCommandState {
                command,
                admitted_tick: tick,
                source_completed_tick: None,
                write_dispatched_tick: None,
                response_tick: None,
            },
        );
        self.instruction_fifo.push_back(id);
        Ok(C220FixpAdmission::Active)
    }

    pub fn generate_read(
        &mut self,
        tick: u64,
    ) -> Result<C220FixpReadProgress, C220L1OutputEngineError> {
        Ok(self.read.generate(tick)?)
    }

    pub fn send_read<T: Copy>(
        &mut self,
        tick: u64,
        stores: &C220FixpStoreBuffer,
        interface: &mut C220MteL1Interface<T>,
        map: impl FnOnce(super::C220L1OutputRead) -> T,
    ) -> Result<C220FixpReadProgress, C220L1OutputEngineError> {
        let ready = stores.below_limit(self.config.write_outstanding_limit);
        Ok(self.read.send_with(tick, |packet| {
            if !ready {
                return Ok(C220FixpReadProgress::DestinationBackpressure);
            }
            let C220FixpReadPacket::L1(operation) = packet else {
                unreachable!("L1 engine only submits L1 reads")
            };
            if interface
                .push(tick, C220MteL1ReadPort::Port1, operation.map_payload(map))?
                .is_none()
            {
                return Ok(C220FixpReadProgress::QueueFull);
            }
            Ok(C220FixpReadProgress::Advanced(packet))
        })?)
    }

    /// Offer only the eligible acknowledgment head. The interface owner removes
    /// it iff accepted; source completion is not final write retirement.
    pub fn receive_source<T: Copy>(
        &mut self,
        tick: u64,
        stores: &C220FixpStoreBuffer,
        request: C220MteL1ReadRequest<T>,
    ) -> Result<bool, C220L1OutputEngineError> {
        let op = request.operation;
        if op.destination != C220MteL1OutputDestination::External || !op.completes_logical_uop {
            return Err(C220L1OutputEngineError::ResponseSource);
        }
        let state = self
            .commands
            .get_mut(&op.instruction_id)
            .ok_or(C220L1OutputEngineError::UnknownCommand(op.instruction_id))?;
        let accepted = self.output.receive_l1_source(
            tick,
            tick,
            C220MteOutputFragment {
                instruction_id: op.instruction_id,
                request_id: request.id,
                destination_address: op.output_address,
                bytes: op.output_bytes,
                last_in_uop: true,
                last_in_instruction: op.last_in_instruction,
            },
            stores.below_limit(self.config.write_outstanding_limit),
        )?;
        if accepted && op.last_in_instruction {
            state.source_completed_tick = Some(tick);
        }
        Ok(accepted)
    }

    pub fn packetize(
        &mut self,
        tick: u64,
        stores: &mut C220FixpStoreBuffer,
    ) -> Result<Option<C220FixpBiuWrite>, C220L1OutputEngineError> {
        let Some(head) = self.output.bursts().front() else {
            return Ok(None);
        };
        let state = self
            .commands
            .get(&head.instruction_id)
            .ok_or(C220L1OutputEngineError::UnknownCommand(head.instruction_id))?;
        Ok(self
            .write
            .packetize_l1_source(tick, &mut self.output, stores, state.command.mode)?)
    }

    pub fn generate_write(
        &mut self,
        tick: u64,
    ) -> Result<C220FixpWriteProgress<C220FixpDispatchPacket>, C220L1OutputEngineError> {
        Ok(self.write.generate_shared(tick)?)
    }

    pub fn send_write(
        &mut self,
        tick: u64,
        biu: &mut C220BiuWriteCommands,
    ) -> Result<C220FixpWriteProgress<C220FixpDispatchPacket>, C220L1OutputEngineError> {
        let progress = self.write.send_with(tick, |packet| {
            let C220FixpDispatchPacket::External(packet) = packet else {
                unreachable!("L1 engine only emits external writes")
            };
            if !biu.can_push(C220BiuSubcore::Cube) {
                return Ok(false);
            }
            Ok(biu.push(
                tick,
                C220BiuWriteInput::from_fixp(packet.write, packet.mode, tick),
            )?)
        })?;
        if let C220FixpWriteProgress::Advanced(C220FixpDispatchPacket::External(packet)) = progress
            && packet.write.fragment.last_in_instruction
        {
            let id = packet.write.fragment.instruction_id;
            if self.instruction_fifo.front() != Some(&id) {
                return Err(C220L1OutputEngineError::CommandOrder(id));
            }
            self.commands
                .get_mut(&id)
                .ok_or(C220L1OutputEngineError::UnknownCommand(id))?
                .write_dispatched_tick = Some(tick);
            self.instruction_fifo.pop_front();
        }
        Ok(progress)
    }

    /// The caller must first accept this response through the real BIU owner.
    pub fn complete_response(
        &mut self,
        response: C220BiuWriteResponse,
    ) -> Result<Option<u64>, C220L1OutputEngineError> {
        if response.data.subcore != C220BiuSubcore::Cube {
            return Err(C220L1OutputEngineError::ResponseSource);
        }
        let Some(id) = response.retired_instruction() else {
            return Ok(None);
        };
        let state = self
            .commands
            .get_mut(&id)
            .ok_or(C220L1OutputEngineError::UnknownCommand(id))?;
        if state.source_completed_tick.is_none()
            || state
                .write_dispatched_tick
                .is_none_or(|tick| tick > response.tick)
            || state.response_tick.is_some()
        {
            return Err(C220L1OutputEngineError::Incomplete(id));
        }
        state.response_tick = Some(response.tick);
        Ok(Some(id))
    }

    /// Called after scheduler retirement checks and successful functional commit.
    pub fn retire(&mut self, id: u64) -> Result<C220L1OutputCommandState, C220L1OutputEngineError> {
        let state = self
            .commands
            .get(&id)
            .ok_or(C220L1OutputEngineError::UnknownCommand(id))?;
        if state.response_tick.is_none() {
            return Err(C220L1OutputEngineError::Incomplete(id));
        }
        Ok(self.commands.remove(&id).expect("validated command"))
    }
}
