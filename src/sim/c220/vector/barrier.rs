use crate::isa::flow::PipelineBarrierStep;
use crate::sim::c220::schedule::{C220Stall, C220StallCause};

use super::C220VectorFrontendEvent;
use super::pipeline::C220VectorPipelineError;
use super::runtime::{C220VectorRuntimeError, VectorEngine};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220VectorBarrier {
    pub instruction_id: u64,
    pub step: PipelineBarrierStep,
    pub predecessor: Option<u64>,
    pub issued_tick: u64,
    pub requires_idle: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220VectorBarrierOutcome {
    pub barrier: C220VectorBarrier,
    pub completed_tick: Option<u64>,
}

pub(in crate::sim::c220) enum VectorBarrierAdmission {
    Issued(C220VectorBarrierOutcome),
    Stalled(C220Stall),
}

impl VectorEngine {
    pub(in crate::sim::c220) fn pending_barriers(
        &self,
    ) -> impl ExactSizeIterator<Item = &C220VectorBarrier> {
        self.frontend.barriers.iter()
    }

    fn instruction_pending(&self, id: u64) -> bool {
        self.queued_instructions()
            .any(|entry| entry.instruction_id == id)
            || self
                .received_instructions()
                .any(|entry| entry.instruction.instruction_id == id)
            || self
                .pending_instructions
                .iter()
                .any(|entry| entry.instruction_id == id)
    }

    pub(in crate::sim::c220) fn issue_barrier_at(
        &mut self,
        tick: u64,
        instruction_id: u64,
        step: PipelineBarrierStep,
    ) -> Result<VectorBarrierAdmission, C220VectorRuntimeError> {
        if self.frontend.is_full() {
            return Ok(VectorBarrierAdmission::Stalled(C220Stall {
                tick,
                pc: step.pc,
                resume_tick: tick
                    .checked_add(1)
                    .ok_or(C220VectorPipelineError::TimeOverflow)?,
                cause: C220StallCause::VectorIssueQueueFull,
            }));
        }
        let barrier = C220VectorBarrier {
            instruction_id,
            step,
            predecessor: self.frontend.last_accepted,
            issued_tick: tick,
            requires_idle: self.frontend.last_queued_is_flag(),
        };
        let pending = barrier
            .predecessor
            .is_some_and(|id| self.instruction_pending(id));
        if pending {
            self.frontend.barriers.push_back(barrier);
        }
        Ok(VectorBarrierAdmission::Issued(C220VectorBarrierOutcome {
            barrier,
            completed_tick: (!pending).then_some(tick),
        }))
    }

    pub(super) fn release_barriers_at(&mut self, tick: u64) {
        while let Some(&barrier) = self.frontend.barriers.front() {
            if barrier
                .predecessor
                .is_some_and(|id| self.instruction_pending(id))
                || (barrier.requires_idle && self.outstanding_instructions() != 0)
            {
                break;
            }
            self.frontend.barriers.pop_front();
            self.frontend
                .events
                .push(C220VectorFrontendEvent::Barrier(C220VectorBarrierOutcome {
                    barrier,
                    completed_tick: Some(tick),
                }));
        }
    }
}
