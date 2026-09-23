use crate::isa::flow::{FlagInstruction, FlagOperation};
use crate::sim::c220::mte::mte3::{
    C220OutputAction, C220OutputDependency, C220OutputStep, decode_mte3_transfer,
};
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
        if self.mte3.pending_commands().next().is_some() || !self.mte3.dma_commands.is_empty() {
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
        if self.mte3.pending_commands().next().is_some() || !self.mte3.dma_commands.is_empty() {
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
        let (action, ticket) = if let Some(instruction) = flag {
            let flag = instruction.resolve(pc, self.state.scalar().machine().xregs());
            let cause = if instruction.source_pipe_code == 5 {
                C220StallCause::Mte3Dependency
            } else {
                C220StallCause::VectorDependency
            };
            let dependency = match instruction.operation {
                FlagOperation::Set => {
                    let dependency = if instruction.source_pipe_code == 5 && self.mte3.physical {
                        C220OutputDependency::Mte3(
                            self.mte3.dma_commands.keys().next_back().copied(),
                        )
                    } else if instruction.source_pipe_code == 5 {
                        C220OutputDependency::NotBefore(
                            self.mte3
                                .timing
                                .latest_retirement_tick()
                                .unwrap_or(tick)
                                .max(tick),
                        )
                    } else {
                        C220OutputDependency::Vector(self.vector.instruction_fence())
                    };
                    self.state.output.signal(flag, dependency);
                    dependency
                }
                FlagOperation::Wait => {
                    let ready =
                        self.state.output.dependency(flag).and_then(
                            |dependency| match dependency {
                                C220OutputDependency::NotBefore(ready) => Some(ready),
                                C220OutputDependency::Mte3(fence) => {
                                    if fence.is_some_and(|id| {
                                        self.mte3.dma_commands.range(..=id).next().is_some()
                                    }) {
                                        None
                                    } else {
                                        Some(tick)
                                    }
                                }
                                C220OutputDependency::Vector(fence) => {
                                    Some(self.vector.fence_retirement_tick(fence).unwrap_or(tick))
                                }
                            },
                        );
                    if ready.is_none_or(|ready| tick < ready) {
                        return Ok(C220CoreStep::Stalled(C220Stall {
                            tick,
                            pc,
                            resume_tick: ready
                                .unwrap_or(tick.checked_add(1).ok_or(C220CoreError::TimeOverflow)?),
                            cause,
                        }));
                    }
                    self.state
                        .output
                        .consume(flag)
                        .expect("ready event remains queued")
                }
            };
            (C220OutputAction::Event { flag, dependency }, None)
        } else {
            let plan = decode_mte3_transfer(
                self.state.scalar.machine(),
                pc,
                word,
                self.state.isa_instance_index,
            )?;
            if self.mte3.physical {
                let pipeline = self
                    .mte_pipeline
                    .as_mut()
                    .ok_or(C220CoreError::MteUnconfigured)?;
                if !pipeline.mte3_frontend().can_issue() {
                    return Ok(C220CoreStep::Stalled(C220Stall {
                        tick,
                        pc,
                        resume_tick: tick.checked_add(1).ok_or(C220CoreError::TimeOverflow)?,
                        cause: C220StallCause::Mte3QueueFull,
                    }));
                }
                let record = pipeline.issue_mte3_dma(self.next_instruction_id, plan)?;
                self.mte3
                    .dma_commands
                    .insert(self.next_instruction_id, (pc, word));
                self.state.commit_c220_sequential_issue();
                return Ok(C220CoreStep::Executed {
                    tick,
                    instruction: C220CoreInstruction::Mte3Dma {
                        step: C220OutputStep {
                            pc,
                            word,
                            next_pc: self.state.scalar().pc(),
                            action: C220OutputAction::CopyToHbm { transfer: plan },
                        },
                        record,
                    },
                });
            }
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
            let action = C220OutputAction::CopyToHbm { transfer: plan };
            self.mte3.issue(pc, word, ticket)?;
            (action, Some(ticket))
        };
        self.state.commit_c220_sequential_issue();
        let step = C220OutputStep {
            pc,
            word,
            next_pc: self.state.scalar().pc(),
            action,
        };
        Ok(C220CoreStep::Executed {
            tick,
            instruction: C220CoreInstruction::Mte3 { step, ticket },
        })
    }
}
