use crate::isa::c220::mte::C220DmaMovDescriptor;
use crate::isa::flow::{FlagInstruction, FlagOperation};
use crate::sim::c220::mte::mte3::{C220OutputAction, decode_mte3_transfer};

use crate::sim::c220::mte::mte3::C220Mte3TimingError;
use crate::sim::c220::schedule::{C220Stall, C220StallCause};

use super::{C220Core, C220CoreError, C220CoreInstruction, C220CoreStep};

impl C220Core {
    pub(super) fn step_mte3_at(
        &mut self,
        tick: u64,
        pc: u64,
        word: u32,
        flag: Option<FlagInstruction>,
    ) -> Result<C220CoreStep, C220CoreError> {
        if flag.is_some_and(|flag| {
            flag.source_pipe_code == 1
                && flag.trigger_pipe_code == 5
                && flag.operation == FlagOperation::Wait
        }) && let Some(resume_tick) = self.vector.pending_visibility_tick()
            && tick < resume_tick
        {
            return Ok(C220CoreStep::Stalled(C220Stall {
                tick,
                pc,
                resume_tick,
                cause: C220StallCause::VectorDependency,
            }));
        }
        if let Some(flag) = flag.filter(|flag| {
            flag.source_pipe_code == 5
                && flag.trigger_pipe_code == 1
                && flag.operation == FlagOperation::Wait
        }) {
            let flag_id = flag
                .resolve(pc, self.state.scalar().machine().xregs())
                .flag_id;
            if let Ok(flag_id) = u8::try_from(flag_id)
                && let Some(resume_tick) = self.mte3.timing.completion_ready_tick(flag_id)
                && tick < resume_tick
            {
                return Ok(C220CoreStep::Stalled(C220Stall {
                    tick,
                    pc,
                    resume_tick,
                    cause: C220StallCause::Mte3Dependency,
                }));
            }
        }
        let (ticket, requests, transfer) = if C220DmaMovDescriptor::is_word(word) {
            if tick < self.mte3.timing.next_issue_tick() {
                return Ok(C220CoreStep::Stalled(C220Stall {
                    tick,
                    pc,
                    resume_tick: self.mte3.timing.next_issue_tick(),
                    cause: C220StallCause::Mte3IssueRate,
                }));
            }
            let plan = decode_mte3_transfer(
                self.state.scalar.machine(),
                pc,
                word,
                self.state.isa_instance_index,
            )?;
            let (ticket, requests) = self.mte3.timing.preview_issue(tick, plan)?;
            if self.mte3.has_pending() {
                return Err(C220Mte3TimingError::TicketMismatch.into());
            }
            (Some(ticket), requests, Some(plan))
        } else {
            (None, Vec::new(), None)
        };
        let (step, prepared) =
            self.state
                .output
                .step_word(&mut self.state.scalar, &self.state.ub, word, transfer)?;
        match step.action {
            C220OutputAction::CopyToHbm { .. } => {
                let ticket = ticket.ok_or(C220Mte3TimingError::TicketMismatch)?;
                self.mte3
                    .issue(ticket, prepared.ok_or(C220Mte3TimingError::TicketMismatch)?)?;
            }
            C220OutputAction::SetMte3CompletionFlag { flag_id, .. } => {
                self.mte3.timing.set_completion_flag(flag_id)?;
            }
            C220OutputAction::WaitMte3CompletionFlag { flag_id, .. } => {
                self.mte3.timing.wait_completion_flag(flag_id)?;
            }
            _ => {}
        }
        let instruction = C220CoreInstruction::Mte3 {
            step,
            requests,
            ticket,
        };
        Ok(C220CoreStep::Executed { tick, instruction })
    }
}
