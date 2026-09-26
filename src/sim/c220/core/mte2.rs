use super::{C220Core, C220CoreError, C220CoreInstruction, C220CoreStep};
use crate::architecture::Architecture;
use crate::isa::c220::mte::out_to_l1::C220MovOutToL1Instruction;
use crate::isa::c220::mte::set2d::C220Set2dInstruction;
use crate::isa::flow::{FlagInstruction, FlagOperation};
use crate::sim::c220::mte::dma::C220DmaGenerated;
use crate::sim::c220::mte::interface::biu_read::returns::C220BiuReadBeat;
use crate::sim::c220::mte::interface::biu_read::{
    C220BiuReadConfig, C220BiuReadRequest, C220BiuSubcore,
};
use crate::sim::c220::mte::mte2::{C220Mte2L1TransferPlan, decode_mte2_transfer};
use crate::sim::c220::schedule::{C220Stall, C220StallCause};
use crate::sim::c220::state::C220ExecutionError;

#[cfg(test)]
mod tests;

impl C220Core {
    pub fn connect_mte2_bus(
        &mut self,
        config: C220BiuReadConfig,
        subcore: C220BiuSubcore,
        outstanding: std::num::NonZeroU32,
    ) -> Result<(), C220CoreError> {
        self.connect_mte2_biu(config, subcore)?;
        self.mte_pipeline
            .as_mut()
            .expect("configured pipeline")
            .connect_biu_bus_reads(outstanding)?;
        Ok(())
    }

    pub fn connect_mte2_biu(
        &mut self,
        config: C220BiuReadConfig,
        subcore: C220BiuSubcore,
    ) -> Result<(), C220CoreError> {
        if self.mte2_is_busy() {
            return Err(C220CoreError::MtePipelineBusy);
        }
        self.mte_pipeline
            .as_mut()
            .ok_or(C220CoreError::MteUnconfigured)?
            .connect_mte2_biu(config, subcore)?;
        Ok(())
    }

    pub fn take_mte2_biu_request(&mut self) -> Option<C220BiuReadRequest> {
        let request = self.mte_pipeline.as_mut()?.take_biu_read_request()?;
        if request.input.generated.last_in_instruction {
            self.mte2
                .observe_dma_tail(request.input.generated.instruction_id);
        }
        Some(request)
    }

    pub fn receive_mte2_biu_at(
        &mut self,
        tick: u64,
        heads: [Option<C220BiuReadBeat>; 2],
    ) -> Result<[bool; 2], C220CoreError> {
        self.advance_to(tick)?;
        Ok(self
            .mte_pipeline
            .as_mut()
            .ok_or(C220CoreError::MteUnconfigured)?
            .receive_biu_read(heads)?)
    }

    /// Selects explicit DMA generation and destination completion instead of
    /// aggregate timing. A transport consumer must drain requests and return
    /// completion only after the destination has accepted the entire command.
    pub fn connect_mte2_dma(&mut self) -> Result<(), C220CoreError> {
        if self.mte2_is_busy() {
            return Err(C220CoreError::MtePipelineBusy);
        }
        self.mte_pipeline
            .as_mut()
            .ok_or(C220CoreError::MteUnconfigured)?
            .connect_mte2_dma()?;
        Ok(())
    }

    pub fn take_mte2_dma_request(&mut self) -> Option<C220DmaGenerated> {
        let request = self.mte_pipeline.as_mut()?.take_dma_output()?;
        if request.last_in_instruction {
            self.mte2.observe_dma_tail(request.instruction_id);
        }
        Some(request)
    }

    pub fn set_mte2_dma_hardware_sync_blocked(
        &mut self,
        blocked: bool,
    ) -> Result<(), C220CoreError> {
        self.mte_pipeline
            .as_mut()
            .ok_or(C220CoreError::MteUnconfigured)?
            .set_dma_hardware_sync_blocked(blocked);
        Ok(())
    }

    pub fn complete_mte2_dma_at(
        &mut self,
        tick: u64,
        instruction_id: u64,
    ) -> Result<(), C220CoreError> {
        self.advance_to(tick)?;
        if let Some(pipeline) = &self.mte_pipeline {
            pipeline.check_external_dma_completion()?;
        }
        self.mte2.complete_dma(instruction_id)?;
        Ok(())
    }

    pub(super) fn capture_mte2_operation(
        &self,
        pc: u64,
        word: u32,
    ) -> Result<super::C220Mte2Operation, C220CoreError> {
        use super::C220Mte2Operation;
        use crate::sim::c220::mte::mte2::C220Mte2Command;
        let machine = self.state.scalar().machine();
        if let Some(flag) = FlagInstruction::decode(Architecture::Dav2201, word) {
            return Ok(C220Mte2Operation::Flag(flag.resolve(pc, machine.xregs())));
        }
        let command = if let Some(instruction) =
            crate::isa::c220::control::C220SetCrossCoreInstruction::decode(word)
        {
            C220Mte2Command::CrossCore {
                instruction,
                payload: crate::sim::c220::sync::C220DeviceSync::from_value(
                    machine.xregs()[usize::from(instruction.source_register)],
                ),
            }
        } else if let Some(decoded) = C220Set2dInstruction::decode(word) {
            let pattern = machine
                .spr_value(15)
                .ok_or(C220CoreError::MissingSet2dPatternSpr)?;
            C220Mte2Command::Set2d(decoded.capture(machine.xregs(), pattern))
        } else if C220MovOutToL1Instruction::decode(word).is_some() {
            C220Mte2Command::MovOutToL1(C220Mte2L1TransferPlan::decode(
                machine,
                pc,
                word,
                self.state.isa_instance_index,
            )?)
        } else {
            C220Mte2Command::MovOutToUb(decode_mte2_transfer(
                machine,
                pc,
                word,
                self.state.isa_instance_index,
            )?)
        };
        Ok(C220Mte2Operation::Command(command))
    }

    pub(super) fn step_mte2_at(
        &mut self,
        tick: u64,
        pc: u64,
        word: u32,
    ) -> Result<C220CoreStep, C220CoreError> {
        if self.state.scalar().is_halted() {
            return Err(C220ExecutionError::ProgramEnded { pc }.into());
        }
        let flag = FlagInstruction::decode(Architecture::Dav2201, word);
        let inline_wait = flag.is_some_and(|flag| {
            flag.operation == FlagOperation::Wait && flag.trigger_pipe_code == 0
        });
        if self.mte_pipeline.is_some() && !inline_wait {
            if let Some(cause) = self.mte2_accept_blocker() {
                return self.mte2_stall(tick, pc, cause);
            }
            let operation = self.capture_mte2_operation(pc, word)?;
            return self.enqueue_mte2_issue_at(tick, pc, word, operation);
        }
        let operation = self.capture_mte2_operation(pc, word)?;
        let instruction = match operation {
            super::C220Mte2Operation::Flag(step) => {
                match step.instruction.operation {
                    FlagOperation::Set => {
                        let predecessor = self
                            .mte2
                            .pending_commands()
                            .last()
                            .map(|c| c.instruction_id);
                        self.pipeline_events
                            .set(self.next_instruction_id, step, predecessor, tick);
                    }
                    FlagOperation::Wait => {
                        if self
                            .pipeline_events
                            .consume(self.next_instruction_id, step, tick)
                            .is_none()
                        {
                            return self.mte2_stall(tick, pc, C220StallCause::Mte2Dependency);
                        }
                    }
                }
                C220CoreInstruction::Mte2Flag(step)
            }
            super::C220Mte2Operation::Command(command) => {
                if !self.can_dispatch_mte2(command, tick)? {
                    return self.mte2_stall(tick, pc, C220StallCause::Mte2IssueRate);
                }
                C220CoreInstruction::Mte2(self.dispatch_mte2_command(
                    self.next_instruction_id,
                    pc,
                    command,
                )?)
            }
        };
        self.state.commit_c220_sequential_issue();
        Ok(C220CoreStep::Executed { tick, instruction })
    }

    pub(super) fn mte2_stall(
        &self,
        tick: u64,
        pc: u64,
        cause: C220StallCause,
    ) -> Result<C220CoreStep, C220CoreError> {
        Ok(C220CoreStep::Stalled(C220Stall {
            tick,
            pc,
            cause,
            resume_tick: tick.checked_add(1).ok_or(C220CoreError::TimeOverflow)?,
        }))
    }

    pub(super) fn can_dispatch_mte2(
        &self,
        command: crate::sim::c220::mte::mte2::C220Mte2Command,
        tick: u64,
    ) -> Result<bool, C220CoreError> {
        use crate::sim::c220::mte::mte2::C220Mte2Command;
        Ok(match command {
            C220Mte2Command::Set2d(fill) => self.mte2.can_issue_l1_fill(
                self.mte_pipeline
                    .as_ref()
                    .ok_or(C220CoreError::MteUnconfigured)?,
                fill,
            )?,
            C220Mte2Command::MovOutToL1(transfer) => {
                transfer.descriptor.is_disabled()
                    || self
                        .mte_pipeline
                        .as_ref()
                        .ok_or(C220CoreError::MteUnconfigured)?
                        .can_issue_mte2_dma()
            }
            C220Mte2Command::MovOutToUb(transfer) => {
                transfer.descriptor.is_disabled()
                    || match &self.mte_pipeline {
                        Some(pipeline) if pipeline.mte2_dma_connected() => {
                            pipeline.can_issue_mte2_dma()
                        }
                        pipeline => {
                            pipeline
                                .as_ref()
                                .is_none_or(|p| p.l1_fill_generator().is_idle())
                                && tick >= self.mte2.next_mte2_issue_tick()
                        }
                    }
            }
            C220Mte2Command::CrossCore { .. } => {
                !self.mte2.is_busy()
                    && self
                        .mte_pipeline
                        .as_ref()
                        .ok_or(C220CoreError::MteUnconfigured)?
                        .can_issue_mte2_cross_core()
            }
        })
    }

    pub(super) fn dispatch_mte2_command(
        &mut self,
        id: u64,
        pc: u64,
        command: crate::sim::c220::mte::mte2::C220Mte2Command,
    ) -> Result<crate::sim::c220::mte::mte2::C220Mte2Issue, C220CoreError> {
        use crate::sim::c220::mte::mte2::C220Mte2Command;
        Ok(match command {
            C220Mte2Command::MovOutToUb(transfer) => {
                self.mte2
                    .issue_dma(self.mte_pipeline.as_mut(), id, pc, transfer)?
            }
            C220Mte2Command::MovOutToL1(transfer) => self.mte2.issue_l1_dma(
                self.mte_pipeline
                    .as_mut()
                    .ok_or(C220CoreError::MteUnconfigured)?,
                id,
                pc,
                transfer,
            )?,
            C220Mte2Command::Set2d(fill) => self.mte2.issue_l1_fill(
                self.mte_pipeline
                    .as_mut()
                    .ok_or(C220CoreError::MteUnconfigured)?,
                id,
                pc,
                fill,
            )?,
            C220Mte2Command::CrossCore {
                instruction,
                payload,
            } => self.mte2.issue_cross_core(
                self.mte_pipeline
                    .as_mut()
                    .ok_or(C220CoreError::MteUnconfigured)?,
                id,
                pc,
                instruction,
                payload,
            )?,
        })
    }
}
