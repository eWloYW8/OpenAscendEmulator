use super::{C220Core, C220CoreError, C220CoreInstruction, C220CoreStep};
use crate::architecture::Architecture;
use crate::isa::c220::mte::set2d::C220Set2dInstruction;
use crate::isa::flow::{FlagInstruction, FlagOperation};
use crate::sim::c220::mte::dma::C220DmaGenerated;
use crate::sim::c220::mte::interface::biu_read::returns::C220BiuReadBeat;
use crate::sim::c220::mte::interface::biu_read::{
    C220BiuReadConfig, C220BiuReadRequest, C220BiuSubcore,
};
use crate::sim::c220::mte::mte2::decode_mte2_transfer;
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
        if self.mte2.is_busy() {
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
        if self.mte2.is_busy() {
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

    pub(super) fn step_mte2_at(
        &mut self,
        tick: u64,
        pc: u64,
        word: u32,
    ) -> Result<C220CoreStep, C220CoreError> {
        if self.state.scalar().is_halted() {
            return Err(C220ExecutionError::ProgramEnded { pc }.into());
        }
        let stall = |resume_tick, cause| {
            C220CoreStep::Stalled(C220Stall {
                tick,
                pc,
                resume_tick,
                cause,
            })
        };
        let instruction = if let Some(decoded) = C220Set2dInstruction::decode(word) {
            let machine = self.state.scalar().machine();
            let pattern = machine
                .spr_value(15)
                .ok_or(C220CoreError::MissingSet2dPatternSpr)?;
            let fill = decoded.capture(machine.xregs(), pattern);
            let pipeline = self
                .mte_pipeline
                .as_mut()
                .ok_or(C220CoreError::MteUnconfigured)?;
            if !self.mte2.can_issue_l1_fill(pipeline, fill)? {
                return Ok(stall(
                    tick.checked_add(1).ok_or(C220CoreError::TimeOverflow)?,
                    C220StallCause::Mte2IssueRate,
                ));
            }
            C220CoreInstruction::Mte2(self.mte2.issue_l1_fill(
                pipeline,
                self.next_instruction_id,
                pc,
                fill,
            )?)
        } else if let Some(flag) = FlagInstruction::decode(Architecture::Dav2201, word) {
            let flag = flag.resolve(pc, self.state.scalar().machine().xregs());
            let destination = flag.instruction.trigger_pipe_code;
            match flag.instruction.operation {
                FlagOperation::Set => self.mte2.set_event(destination, flag.flag_id),
                FlagOperation::Wait => {
                    if !self.mte2.wait_event(destination, flag.flag_id) {
                        return Ok(stall(
                            tick.checked_add(1).ok_or(C220CoreError::TimeOverflow)?,
                            C220StallCause::Mte2Dependency,
                        ));
                    }
                }
            }
            C220CoreInstruction::Mte2Flag(flag)
        } else {
            let transfer = decode_mte2_transfer(
                self.state.scalar().machine(),
                pc,
                word,
                self.state.isa_instance_index,
            )?;
            let physical = self
                .mte_pipeline
                .as_ref()
                .filter(|p| p.mte2_dma_connected());
            let generator_ready = match physical {
                Some(pipeline) => pipeline.can_issue_mte2_dma(),
                None => self
                    .mte_pipeline
                    .as_ref()
                    .is_none_or(|p| p.l1_fill_generator().is_idle()),
            };
            if !transfer.descriptor.is_disabled() && !generator_ready {
                return Ok(stall(
                    tick.checked_add(1).ok_or(C220CoreError::TimeOverflow)?,
                    C220StallCause::Mte2IssueRate,
                ));
            }
            if !transfer.descriptor.is_disabled()
                && physical.is_none()
                && tick < self.mte2.next_mte2_issue_tick()
            {
                return Ok(stall(
                    self.mte2.next_mte2_issue_tick(),
                    C220StallCause::Mte2IssueRate,
                ));
            }
            C220CoreInstruction::Mte2(self.mte2.issue_dma(
                self.mte_pipeline.as_mut(),
                self.next_instruction_id,
                pc,
                transfer,
            )?)
        };
        self.state.commit_c220_sequential_issue();
        Ok(C220CoreStep::Executed { tick, instruction })
    }
}
