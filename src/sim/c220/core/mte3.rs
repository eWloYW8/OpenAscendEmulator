use crate::isa::flow::{FlagInstruction, FlagOperation};
use crate::sim::c220::mte::mte3::{C220Mte3Step, decode_mte3_transfer};
use crate::sim::c220::schedule::{C220Stall, C220StallCause};

use super::{C220Core, C220CoreError, C220CoreInstruction, C220CoreStep};
use crate::sim::c220::mte::interface::biu_write::C220BiuWriteSourceRequest;
use crate::sim::c220::mte::interface::biu_write::command::{
    C220BiuWriteCommandTransfer, C220BiuWriteConfig,
};
use crate::sim::c220::mte::interface::biu_write::data::{C220BiuWriteData, C220BiuWriteResponse};
use crate::sim::c220::mte::interface::ub_read::{C220UbReadAcknowledgment, C220UbReadFragment};
use std::num::NonZeroU32;

impl C220Core {
    /// Connects MTE3 through the core BIU write route. The caller supplies a
    /// downstream bus/memory endpoint, not direct MTE acknowledgments.
    pub fn connect_mte3_bus(
        &mut self,
        config: C220BiuWriteConfig,
        bus_outstanding: NonZeroU32,
    ) -> Result<(), C220CoreError> {
        self.connect_mte3_biu(config)?;
        self.mte_pipeline
            .as_mut()
            .expect("connected pipeline")
            .connect_biu_bus_writes(bus_outstanding)?;
        Ok(())
    }

    pub fn receive_mte3_bus_return_at(
        &mut self,
        tick: u64,
        kind: crate::sim::c220::memory::biu_write::C220BiuWriteReturnKind,
        tag: NonZeroU32,
    ) -> Result<bool, C220CoreError> {
        self.advance_to(tick)?;
        Ok(self
            .mte_pipeline
            .as_mut()
            .ok_or(C220CoreError::MteUnconfigured)?
            .receive_biu_bus_write_return(kind, tag)?)
    }

    pub fn connect_mte3_biu(&mut self, config: C220BiuWriteConfig) -> Result<(), C220CoreError> {
        if self.mte3_is_busy() {
            return Err(C220CoreError::MtePipelineBusy);
        }
        self.mte_pipeline
            .as_mut()
            .ok_or(C220CoreError::MteUnconfigured)?
            .connect_mte3_biu(config)?;
        self.mte3.physical = true;
        Ok(())
    }

    pub fn take_biu_write_command_at(
        &mut self,
        tick: u64,
    ) -> Result<Option<C220BiuWriteCommandTransfer>, C220CoreError> {
        self.advance_to(tick)?;
        Ok(self
            .mte_pipeline
            .as_mut()
            .ok_or(C220CoreError::MteUnconfigured)?
            .take_biu_write_command()?)
    }

    pub fn configure_mte3_biu_source(
        &mut self,
        bandwidth: NonZeroU32,
    ) -> Result<(), C220CoreError> {
        if self.lsu.is_some() {
            return Err(C220CoreError::LsuAlreadyConfigured);
        }
        let pipeline = self
            .mte_pipeline
            .as_mut()
            .ok_or(C220CoreError::MteUnconfigured)?;
        pipeline.configure_biu_write_source(pipeline.ub_vector_subcore(), bandwidth)?;
        pipeline.connect_mte3_biu_retirement()?;
        Ok(())
    }

    pub fn receive_mte3_biu_write_response_at(
        &mut self,
        tick: u64,
        tag: NonZeroU32,
    ) -> Result<C220BiuWriteResponse, C220CoreError> {
        self.advance_to(tick)?;
        let pipeline = self
            .mte_pipeline
            .as_mut()
            .ok_or(C220CoreError::MteUnconfigured)?;
        Ok(pipeline.receive_biu_write_response(tag)?)
    }

    pub fn register_mte3_biu_write_at(
        &mut self,
        tick: u64,
        request: C220BiuWriteSourceRequest,
    ) -> Result<(), C220CoreError> {
        self.advance_to(tick)?;
        let pipeline = self
            .mte_pipeline
            .as_mut()
            .ok_or(C220CoreError::MteUnconfigured)?;
        pipeline.register_biu_write_source(pipeline.ub_vector_subcore(), request)?;
        Ok(())
    }

    pub fn receive_mte3_biu_dbid_at(
        &mut self,
        tick: u64,
        tag: NonZeroU32,
    ) -> Result<(), C220CoreError> {
        self.advance_to(tick)?;
        let pipeline = self
            .mte_pipeline
            .as_mut()
            .ok_or(C220CoreError::MteUnconfigured)?;
        pipeline.receive_biu_write_dbid(pipeline.ub_vector_subcore(), tag)?;
        Ok(())
    }

    pub fn take_biu_write_data_at(
        &mut self,
        tick: u64,
    ) -> Result<Option<C220BiuWriteData>, C220CoreError> {
        self.advance_to(tick)?;
        let pipeline = self
            .mte_pipeline
            .as_mut()
            .ok_or(C220CoreError::MteUnconfigured)?;
        Ok(pipeline.take_biu_write_data()?)
    }

    /// Supplies a source packet from the BIU write transport to this core's UB.
    pub fn push_mte3_ub_read_at(
        &mut self,
        tick: u64,
        fragment: C220UbReadFragment,
    ) -> Result<bool, C220CoreError> {
        self.advance_to(tick)?;
        let pipeline = self
            .mte_pipeline
            .as_mut()
            .ok_or(C220CoreError::MteUnconfigured)?;
        Ok(pipeline.push_ub_read(pipeline.ub_vector_subcore(), fragment)?)
    }

    pub fn take_mte3_ub_read_completion_at(
        &mut self,
        tick: u64,
        tag: u32,
    ) -> Result<Option<C220UbReadAcknowledgment>, C220CoreError> {
        self.advance_to(tick)?;
        let pipeline = self
            .mte_pipeline
            .as_mut()
            .ok_or(C220CoreError::MteUnconfigured)?;
        Ok(pipeline.take_ub_read_completion(pipeline.ub_vector_subcore(), tag)?)
    }

    /// Use the native command/generator path. The transport must drain and
    /// acknowledge each request; no aggregate completion estimate is used.
    pub fn connect_mte3_dma(&mut self) -> Result<(), C220CoreError> {
        if self.mte3_is_busy() {
            return Err(C220CoreError::MtePipelineBusy);
        }
        self.mte_pipeline
            .as_ref()
            .ok_or(C220CoreError::MteUnconfigured)?;
        self.mte3.physical = true;
        Ok(())
    }

    pub fn take_mte3_dma_request(
        &mut self,
    ) -> Option<crate::sim::c220::mte::dma::C220DmaGenerated> {
        self.mte_pipeline.as_mut()?.take_mte3_dma_output()
    }

    pub fn set_mte3_dma_hardware_sync_blocked(
        &mut self,
        blocked: bool,
    ) -> Result<(), C220CoreError> {
        self.mte_pipeline
            .as_mut()
            .ok_or(C220CoreError::MteUnconfigured)?
            .set_mte3_hardware_sync_blocked(blocked);
        Ok(())
    }

    pub fn acknowledge_mte3_dma_at(
        &mut self,
        tick: u64,
        instruction_id: u64,
        uop_index: u64,
    ) -> Result<(), C220CoreError> {
        self.advance_to(tick)?;
        self.mte_pipeline
            .as_mut()
            .ok_or(C220CoreError::MteUnconfigured)?
            .acknowledge_mte3_dma(instruction_id, uop_index)?;
        Ok(())
    }

    pub fn last_mte3_dma_outcomes(&self) -> &[crate::sim::c220::mte::mte3::C220Mte3DmaOutcome] {
        &self.mte3.dma_outcomes
    }

    pub(super) fn step_mte3_at(
        &mut self,
        tick: u64,
        pc: u64,
        word: u32,
        flag: Option<FlagInstruction>,
    ) -> Result<C220CoreStep, C220CoreError> {
        if self.mte3.physical {
            if let Some(instruction) = flag {
                let step = instruction.resolve(pc, self.state.scalar().machine().xregs());
                return self.enqueue_mte3_at(tick, pc, word, super::C220Mte3Operation::Flag(step));
            } else {
                let plan = decode_mte3_transfer(
                    self.state.scalar.machine(),
                    pc,
                    word,
                    self.state.isa_instance_index,
                )?;
                return self.enqueue_mte3_at(
                    tick,
                    pc,
                    word,
                    super::C220Mte3Operation::Command(
                        crate::sim::c220::mte::mte3::frontend::C220Mte3Command::Dma(plan),
                    ),
                );
            }
        }
        if let Some(instruction) = flag {
            let step = instruction.resolve(pc, self.state.scalar().machine().xregs());
            match instruction.operation {
                FlagOperation::Set => {
                    let predecessor = self
                        .mte3
                        .pending_commands()
                        .last()
                        .map(|command| command.instruction_id);
                    self.pipeline_events
                        .set(self.next_instruction_id, step, predecessor, tick);
                }
                FlagOperation::Wait => {
                    if self
                        .pipeline_events
                        .consume(self.next_instruction_id, step, tick)
                        .is_none()
                    {
                        return self.mte3_stall(tick, pc, C220StallCause::PipelineEventDependency);
                    }
                }
            }
            self.state.commit_c220_sequential_issue();
            return Ok(C220CoreStep::Executed {
                tick,
                instruction: C220CoreInstruction::Mte3Flag(step),
            });
        }
        let (plan, ticket) = {
            let plan = decode_mte3_transfer(
                self.state.scalar.machine(),
                pc,
                word,
                self.state.isa_instance_index,
            )?;
            if self.mte3.is_full() {
                return Ok(C220CoreStep::Stalled(C220Stall {
                    tick,
                    pc,
                    resume_tick: self
                        .mte3
                        .pending_retirement_tick()
                        .expect("full command queue has a head"),
                    cause: C220StallCause::Mte3QueueFull,
                }));
            }
            let ready_tick = if plan.descriptor.is_disabled() {
                tick
            } else {
                self.mte3.timing.next_issue_tick()
            };
            if tick < ready_tick {
                return Ok(C220CoreStep::Stalled(C220Stall {
                    tick,
                    pc,
                    resume_tick: ready_tick,
                    cause: C220StallCause::Mte3IssueRate,
                }));
            }
            let ticket = self.mte3.timing.preview_issue(tick, plan)?;
            self.mte3
                .issue(self.next_instruction_id, pc, word, ticket)?;
            (plan, ticket)
        };
        self.state.commit_c220_sequential_issue();
        let step = C220Mte3Step {
            pc,
            word,
            next_pc: self.state.scalar().pc(),
            transfer: plan,
        };
        Ok(C220CoreStep::Executed {
            tick,
            instruction: C220CoreInstruction::Mte3 { step, ticket },
        })
    }
}
