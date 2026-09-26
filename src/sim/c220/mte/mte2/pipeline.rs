use std::collections::VecDeque;

use super::C220Mte2L1TransferPlan;
use super::timing::C220Mte2DmaTiming;
use super::{
    C220Mte2Command, C220Mte2CommandState, C220Mte2Completion, C220Mte2Issue, C220Mte2IssueTiming,
    C220Mte2Outcome, C220Mte2Result, C220Mte2TimingError, C220Mte2TimingRules,
    C220Mte2TransferPlan, copy_c220_mov_out_to_ub,
};
use crate::isa::c220::mte::set2d::C220Set2dFill;
use crate::isa::c220::mte::smask::C220SmaskTransfer;
use crate::memory::mapped::MappedMemory;
use crate::memory::ub::UbMemory;
use crate::sim::c220::memory::{C220LocalBufferError, C220LocalMemory};
use crate::sim::c220::mte::out_to_l1::{C220L1DmaError, execute_c220_mov_out_to_l1};
use crate::sim::c220::mte::set2d::execute_c220_set2d;
use crate::sim::c220::mte::{C220MtePipeline, C220MtePipelineError, C220TransferError};

#[derive(Debug, thiserror::Error)]
pub enum C220Mte2RuntimeError {
    #[error(transparent)]
    Load2d(#[from] crate::sim::c220::mte::load2d::C220Load2dTransferError),
    #[error(
        "DMA completion for command {instruction_id} precedes its request tail or does not match an active transfer"
    )]
    InvalidDmaCompletion { instruction_id: u64 },
    #[error("MTE2 command generator is busy")]
    Busy,
    #[error("switching from an outstanding aggregate DMA requires a physical DMA generator model")]
    UnmodeledDmaGeneratorSwitch,
    #[error("MTE2 time reversed from {previous} to {requested}")]
    TimeReversed { previous: u64, requested: u64 },
    #[error("MTE2 command cannot retire beyond the maximum tick")]
    TimeOverflow,
    #[error(transparent)]
    Timing(#[from] C220Mte2TimingError),
    #[error(transparent)]
    Pipeline(#[from] C220MtePipelineError),
    #[error(transparent)]
    Transfer(#[from] C220TransferError),
    #[error(transparent)]
    LocalMemory(#[from] C220LocalBufferError),
    #[error(transparent)]
    L1Transfer(#[from] C220L1DmaError),
    #[error(transparent)]
    Smask(#[from] crate::sim::c220::mte::smask::C220SmaskTransferError),
}

/// MTE2 command ownership and ordered functional retirement. The physical L1
/// path is core-owned; this lane only consumes its completion notifications.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220Mte2Pipeline {
    dma: C220Mte2DmaTiming,
    pending: VecDeque<C220Mte2CommandState>,
    outcomes: Vec<C220Mte2Outcome>,
    now: u64,
    advanced: Option<u64>,
}

impl C220Mte2Pipeline {
    pub fn new(rules: C220Mte2TimingRules) -> Self {
        Self {
            dma: C220Mte2DmaTiming::new(rules),
            pending: VecDeque::new(),
            outcomes: Vec::new(),
            now: 0,
            advanced: None,
        }
    }

    pub fn pending_commands(&self) -> impl Iterator<Item = C220Mte2CommandState> + '_ {
        self.pending.iter().copied()
    }

    pub fn is_busy(&self) -> bool {
        !self.pending.is_empty()
    }

    pub fn last_outcomes(&self) -> &[C220Mte2Outcome] {
        &self.outcomes
    }

    pub const fn rules(&self) -> C220Mte2TimingRules {
        self.dma.rules
    }

    pub const fn next_mte2_issue_tick(&self) -> u64 {
        self.dma.next_issue_tick
    }

    pub fn next_event_tick(&self) -> Option<u64> {
        let head = self.pending.front()?;
        let next = self.now.saturating_add(1);
        Some(match head.completion {
            C220Mte2Completion::Estimated { retire_tick } => next.max(retire_tick),
            C220Mte2Completion::Observed { tick } => next.max(tick.saturating_add(1)),
            C220Mte2Completion::AwaitingDestination | C220Mte2Completion::AwaitingDma { .. } => {
                next
            }
        })
    }

    pub(crate) fn can_issue_smask(
        &self,
        pipeline: &C220MtePipeline,
        transfer: C220SmaskTransfer,
    ) -> Result<bool, C220Mte2RuntimeError> {
        if !transfer.descriptor.is_empty() {
            self.check_physical_generator_switch()?;
        }
        Ok(pipeline.can_issue_external_smask(transfer))
    }

    pub(crate) fn issue_smask(
        &mut self,
        pipeline: &mut C220MtePipeline,
        instruction_id: u64,
        pc: u64,
        transfer: C220SmaskTransfer,
    ) -> Result<C220Mte2Issue, C220Mte2RuntimeError> {
        if !self.can_issue_smask(pipeline, transfer)? {
            return Err(C220Mte2RuntimeError::Busy);
        }
        self.now
            .checked_add(1)
            .ok_or(C220Mte2RuntimeError::TimeOverflow)?;
        let timing = pipeline.issue_external_smask(instruction_id, transfer)?;
        let command = C220Mte2Command::MovOutToSmask(transfer);
        self.pending.push_back(C220Mte2CommandState {
            instruction_id,
            pc,
            issue_tick: timing.tick,
            command,
            completion: if timing.completion_ready {
                C220Mte2Completion::Observed { tick: timing.tick }
            } else {
                C220Mte2Completion::AwaitingDestination
            },
        });
        Ok(C220Mte2Issue {
            instruction_id,
            pc,
            command,
            timing: C220Mte2IssueTiming::Read(timing),
        })
    }

    pub(crate) fn check_physical_generator_switch(&self) -> Result<(), C220Mte2RuntimeError> {
        if self
            .pending
            .iter()
            .any(|p| matches!(p.completion, C220Mte2Completion::Estimated { .. }))
        {
            return Err(C220Mte2RuntimeError::UnmodeledDmaGeneratorSwitch);
        }
        Ok(())
    }

    pub(crate) fn can_issue_l1_fill(
        &self,
        pipeline: &C220MtePipeline,
        fill: C220Set2dFill,
    ) -> Result<bool, C220Mte2RuntimeError> {
        if !fill.descriptor.is_disabled() {
            self.check_physical_generator_switch()?;
        }
        Ok(pipeline.can_issue_l1_fill(fill))
    }

    pub(crate) fn issue_spr(
        &mut self,
        pipeline: &C220MtePipeline,
        instruction_id: u64,
        pc: u64,
        step: crate::sim::common::scalar::ScalarSprStep,
    ) -> Result<C220Mte2Issue, C220Mte2RuntimeError> {
        if !pipeline.mte2_generator_idle() {
            return Err(C220Mte2RuntimeError::Busy);
        }
        self.now
            .checked_add(1)
            .ok_or(C220Mte2RuntimeError::TimeOverflow)?;
        let command = C220Mte2Command::WriteSpr(step);
        self.pending.push_back(C220Mte2CommandState {
            instruction_id,
            pc,
            issue_tick: self.now,
            command,
            completion: C220Mte2Completion::Observed { tick: self.now },
        });
        Ok(C220Mte2Issue {
            instruction_id,
            pc,
            command,
            timing: C220Mte2IssueTiming::WriteSpr {
                dispatch_tick: self.now,
            },
        })
    }

    pub(crate) fn issue_cross_core(
        &mut self,
        pipeline: &mut C220MtePipeline,
        instruction_id: u64,
        pc: u64,
        instruction: crate::isa::c220::control::C220SetCrossCoreInstruction,
        payload: crate::sim::c220::sync::C220DeviceSync,
    ) -> Result<C220Mte2Issue, C220Mte2RuntimeError> {
        if self.is_busy() {
            return Err(C220Mte2RuntimeError::Busy);
        }
        self.now
            .checked_add(1)
            .ok_or(C220Mte2RuntimeError::TimeOverflow)?;
        let dispatch_tick = pipeline.issue_mte2_cross_core()?;
        let command = C220Mte2Command::CrossCore {
            instruction,
            payload,
        };
        self.pending.push_back(C220Mte2CommandState {
            instruction_id,
            pc,
            issue_tick: dispatch_tick,
            command,
            completion: C220Mte2Completion::Observed {
                tick: dispatch_tick,
            },
        });
        Ok(C220Mte2Issue {
            instruction_id,
            pc,
            command,
            timing: C220Mte2IssueTiming::CrossCore { dispatch_tick },
        })
    }

    pub(crate) fn issue_l1_fill(
        &mut self,
        pipeline: &mut C220MtePipeline,
        instruction_id: u64,
        pc: u64,
        fill: C220Set2dFill,
    ) -> Result<C220Mte2Issue, C220Mte2RuntimeError> {
        if !self.can_issue_l1_fill(pipeline, fill)? {
            return Err(C220Mte2RuntimeError::Busy);
        }
        self.now
            .checked_add(1)
            .ok_or(C220Mte2RuntimeError::TimeOverflow)?;
        let timing = pipeline.issue_l1_fill(instruction_id, fill)?;
        let command = C220Mte2Command::Set2d(fill);
        self.pending.push_back(C220Mte2CommandState {
            instruction_id,
            pc,
            issue_tick: timing.tick,
            command,
            completion: if timing.completion_ready {
                C220Mte2Completion::Observed { tick: timing.tick }
            } else {
                C220Mte2Completion::AwaitingDestination
            },
        });
        Ok(C220Mte2Issue {
            instruction_id,
            pc,
            command,
            timing: C220Mte2IssueTiming::L1(timing),
        })
    }

    pub(crate) fn issue_l1_dma(
        &mut self,
        pipeline: &mut C220MtePipeline,
        instruction_id: u64,
        pc: u64,
        transfer: C220Mte2L1TransferPlan,
    ) -> Result<C220Mte2Issue, C220Mte2RuntimeError> {
        self.now
            .checked_add(1)
            .ok_or(C220Mte2RuntimeError::TimeOverflow)?;
        let timing = pipeline.issue_mte2_l1_dma(
            instruction_id,
            transfer.descriptor,
            transfer.source_address,
            transfer.destination_address,
            transfer.dma_mode_word,
        )?;
        let command = C220Mte2Command::MovOutToL1(transfer);
        self.pending.push_back(C220Mte2CommandState {
            instruction_id,
            pc,
            issue_tick: self.now,
            command,
            completion: if timing.completion_ready {
                C220Mte2Completion::Observed { tick: self.now }
            } else {
                C220Mte2Completion::AwaitingDma {
                    tail_delivered: false,
                }
            },
        });
        Ok(C220Mte2Issue {
            instruction_id,
            pc,
            command,
            timing: C220Mte2IssueTiming::Dma(timing),
        })
    }

    pub(crate) fn issue_load2d(
        &mut self,
        pipeline: &mut C220MtePipeline,
        instruction_id: u64,
        pc: u64,
        transfer: crate::isa::c220::mte::load2d::C220Load2dTransfer,
        mode: crate::sim::c220::mte::uop::C220DmaUopMode,
    ) -> Result<C220Mte2Issue, C220Mte2RuntimeError> {
        if transfer.descriptor.repeat_count != 0 {
            self.check_physical_generator_switch()?;
        }
        self.now
            .checked_add(1)
            .ok_or(C220Mte2RuntimeError::TimeOverflow)?;
        let timing = pipeline.issue_external_load2d(instruction_id, transfer, mode)?;
        let command = C220Mte2Command::Load2d { transfer, mode };
        self.pending.push_back(C220Mte2CommandState {
            instruction_id,
            pc,
            issue_tick: timing.tick,
            command,
            completion: if timing.completion_ready {
                C220Mte2Completion::Observed { tick: timing.tick }
            } else {
                C220Mte2Completion::AwaitingDma {
                    tail_delivered: false,
                }
            },
        });
        Ok(C220Mte2Issue {
            instruction_id,
            pc,
            command,
            timing: C220Mte2IssueTiming::Dma(timing),
        })
    }

    pub(crate) fn issue_dma(
        &mut self,
        pipeline: Option<&mut C220MtePipeline>,
        instruction_id: u64,
        pc: u64,
        transfer: C220Mte2TransferPlan,
    ) -> Result<C220Mte2Issue, C220Mte2RuntimeError> {
        if transfer.descriptor.is_disabled() {
            let _ = super::super::uop::mte2_uops(transfer).map_err(C220Mte2TimingError::from)?;
            self.now
                .checked_add(1)
                .ok_or(C220Mte2RuntimeError::TimeOverflow)?;
            let command = C220Mte2Command::MovOutToUb(transfer);
            self.pending.push_back(C220Mte2CommandState {
                instruction_id,
                pc,
                issue_tick: self.now,
                command,
                completion: C220Mte2Completion::Observed { tick: self.now },
            });
            return Ok(C220Mte2Issue {
                instruction_id,
                pc,
                command,
                timing: C220Mte2IssueTiming::Disabled,
            });
        }
        if let Some(pipeline) = pipeline.filter(|p| p.mte2_dma_connected()) {
            self.now
                .checked_add(1)
                .ok_or(C220Mte2RuntimeError::TimeOverflow)?;
            let timing = pipeline.issue_mte2_dma(instruction_id, transfer)?;
            let command = C220Mte2Command::MovOutToUb(transfer);
            self.pending.push_back(C220Mte2CommandState {
                instruction_id,
                pc,
                issue_tick: self.now,
                command,
                completion: if timing.completion_ready {
                    C220Mte2Completion::Observed { tick: self.now }
                } else {
                    C220Mte2Completion::AwaitingDma {
                        tail_delivered: false,
                    }
                },
            });
            return Ok(C220Mte2Issue {
                instruction_id,
                pc,
                command,
                timing: C220Mte2IssueTiming::Dma(timing),
            });
        }
        if self.now < self.dma.next_issue_tick {
            return Err(C220Mte2RuntimeError::Busy);
        }
        let ticket = self.dma.preview_ticket(self.now, transfer)?;
        let command = C220Mte2Command::MovOutToUb(transfer);
        self.dma.accept(ticket);
        self.pending.push_back(C220Mte2CommandState {
            instruction_id,
            pc,
            issue_tick: self.now,
            command,
            completion: C220Mte2Completion::Estimated {
                retire_tick: ticket.retire_tick,
            },
        });
        Ok(C220Mte2Issue {
            instruction_id,
            pc,
            command,
            timing: C220Mte2IssueTiming::AggregateDma(ticket),
        })
    }

    pub(crate) fn begin_advance(&mut self) {
        self.outcomes.clear();
    }

    pub(crate) fn observe_dma_tail(&mut self, instruction_id: u64) {
        let command = self
            .pending
            .iter_mut()
            .find(|p| p.instruction_id == instruction_id)
            .expect("DMA request belongs to a pending command");
        let C220Mte2Completion::AwaitingDma { tail_delivered } = &mut command.completion else {
            unreachable!("DMA request belongs to an active physical transfer");
        };
        *tail_delivered = true;
    }

    pub(crate) fn complete_dma(&mut self, instruction_id: u64) -> Result<(), C220Mte2RuntimeError> {
        let command = self
            .pending
            .iter_mut()
            .find(|p| {
                p.instruction_id == instruction_id
                    && matches!(
                        p.completion,
                        C220Mte2Completion::AwaitingDma {
                            tail_delivered: true
                        }
                    )
            })
            .ok_or(C220Mte2RuntimeError::InvalidDmaCompletion { instruction_id })?;
        self.now
            .checked_add(1)
            .ok_or(C220Mte2RuntimeError::TimeOverflow)?;
        command.completion = C220Mte2Completion::Observed { tick: self.now };
        Ok(())
    }

    pub(crate) fn commit_ready_at(
        &mut self,
        tick: u64,
        local: &mut C220LocalMemory,
        ub: &mut UbMemory,
        source: &MappedMemory,
    ) -> Result<(), C220Mte2RuntimeError> {
        if tick < self.now {
            return Err(C220Mte2RuntimeError::TimeReversed {
                previous: self.now,
                requested: tick,
            });
        }
        if self.advanced == Some(tick) {
            return Ok(());
        }
        if let Some(&command) = self.pending.front()
            && command.issue_tick < tick
            && match command.completion {
                C220Mte2Completion::AwaitingDestination
                | C220Mte2Completion::AwaitingDma { .. } => false,
                C220Mte2Completion::Observed { tick: done } => done < tick,
                C220Mte2Completion::Estimated { retire_tick } => retire_tick <= tick,
            }
        {
            let result = match command.command {
                C220Mte2Command::Load2d { transfer, .. } => {
                    let prepared = crate::sim::c220::mte::load2d::prepare_c220_external_load2d(
                        source, transfer,
                    )?;
                    let result = prepared.result;
                    prepared.commit(local)?;
                    C220Mte2Result::Load2d(result)
                }
                C220Mte2Command::MovOutToSmask(transfer) => C220Mte2Result::MovOutToSmask(
                    crate::sim::c220::mte::smask::execute_c220_mov_out_to_smask(
                        local, source, transfer,
                    )?,
                ),
                C220Mte2Command::WriteSpr(step) => C220Mte2Result::WriteSpr(step),
                C220Mte2Command::CrossCore { payload, .. } => C220Mte2Result::CrossCore(payload),
                C220Mte2Command::MovOutToL1(plan) => {
                    C220Mte2Result::MovOutToL1(execute_c220_mov_out_to_l1(
                        source,
                        local.l1_mut(),
                        plan.descriptor,
                        plan.source_address,
                        plan.destination_address,
                        plan.padding,
                    )?)
                }
                C220Mte2Command::Set2d(fill) => {
                    C220Mte2Result::Set2d(execute_c220_set2d(local, fill)?)
                }
                C220Mte2Command::MovOutToUb(plan) => {
                    C220Mte2Result::MovOutToUb(copy_c220_mov_out_to_ub(
                        ub,
                        source,
                        plan.descriptor,
                        plan.source_address,
                        plan.destination_address,
                    )?)
                }
            };
            self.pending.pop_front();
            self.outcomes.push(C220Mte2Outcome {
                command,
                retire_tick: tick,
                result,
            });
        }
        self.now = tick;
        self.advanced = Some(tick);
        Ok(())
    }

    pub(crate) fn observe_l1_completions(&mut self, tick: u64, ids: &[u64]) {
        for id in ids {
            let command = self
                .pending
                .iter_mut()
                .find(|p| p.instruction_id == *id)
                .expect("L1 completion owns a pending MTE2 command");
            command.completion = C220Mte2Completion::Observed { tick };
        }
    }
}

pub(crate) fn is_mte2_transfer(word: u32) -> bool {
    crate::isa::c220::mte::C220MovOutToUbDescriptor::is_word(word)
        || crate::isa::c220::mte::out_to_l1::C220MovOutToL1Instruction::decode(word).is_some()
        || crate::isa::c220::mte::smask::C220MovSmaskInstruction::decode(word)
            .is_some_and(|instruction| instruction.source_mode == 0)
        || crate::isa::c220::mte::load2d::C220Load2dInstruction::decode(word)
            .is_some_and(|instruction| instruction.is_external())
}
