use crate::architecture::Architecture;
use crate::isa::c220::mte::load2d::C220Load2dInstruction;
use crate::isa::flow::{FlagInstruction, FlagOperation};
use crate::sim::c220::mte::mte1::load2d::{C220Load2dTransferError, prepare_c220_load2d};
use crate::sim::c220::mte::mte1::{C220Mte1TimingError, C220Mte1TimingRules};

use crate::sim::c220::schedule::{C220Stall, C220StallCause};

use super::{C220Core, C220CoreError, C220CoreInstruction, C220CoreStep};

impl C220Core {
    pub fn configure_mte1_timing(
        &mut self,
        rules: C220Mte1TimingRules,
    ) -> Result<(), C220CoreError> {
        self.mte1.timing.configure(rules)?;
        Ok(())
    }

    pub(super) fn step_mte1_at(
        &mut self,
        tick: u64,
        pc: u64,
        word: u32,
    ) -> Result<C220CoreStep, C220CoreError> {
        let instruction = if let Some(decoded) =
            C220Load2dInstruction::decode(word).filter(|instruction| instruction.is_mte1())
        {
            let next_accept_tick = self.mte1.timing.next_accept_tick();
            if tick < next_accept_tick {
                return Ok(C220CoreStep::Stalled(C220Stall {
                    tick,
                    pc,
                    resume_tick: next_accept_tick,
                    cause: C220StallCause::Mte1IssueRate,
                }));
            }
            let transfer = decoded
                .capture(self.state.scalar().machine().xregs())
                .map_err(C220Load2dTransferError::from)?;
            let ticket = self.mte1.timing.preview_issue(tick, transfer)?;
            let prepared = prepare_c220_load2d(&self.local_memory, transfer)?;
            let result = prepared.result;
            self.mte1.issue(&ticket, prepared)?;
            self.state.commit_c220_sequential_issue();
            C220CoreInstruction::Mte1Load2d {
                transfer,
                result,
                ticket: Box::new(ticket),
            }
        } else {
            let flag = FlagInstruction::decode(Architecture::Dav2201, word)
                .expect("matched C220 MTE1 flag")
                .resolve(pc, self.state.scalar().machine().xregs());
            match flag.instruction.operation {
                FlagOperation::Set => self.mte1.timing.set_event(tick, flag.flag_id),
                FlagOperation::Wait => {
                    let ready_tick = self.mte1.timing.event_ready_tick(flag.flag_id).ok_or(
                        C220Mte1TimingError::MissingEvent {
                            event_id: flag.flag_id,
                        },
                    )?;
                    if tick < ready_tick {
                        return Ok(C220CoreStep::Stalled(C220Stall {
                            tick,
                            pc,
                            resume_tick: ready_tick,
                            cause: C220StallCause::Mte1Dependency,
                        }));
                    }
                    self.mte1.timing.wait_event(tick, flag.flag_id)?;
                }
            }
            self.state.commit_c220_sequential_issue();
            C220CoreInstruction::Mte1Flag(flag)
        };
        Ok(C220CoreStep::Executed { tick, instruction })
    }
}
