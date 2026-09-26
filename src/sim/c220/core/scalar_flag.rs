use super::{C220Core, C220CoreError, C220CoreInstruction, C220CoreStep};
use crate::isa::flow::{FlagInstruction, FlagOperation};
use crate::sim::c220::schedule::{C220Stall, C220StallCause};

impl C220Core {
    pub(super) fn step_scalar_flag_at(
        &mut self,
        tick: u64,
        pc: u64,
        instruction: FlagInstruction,
    ) -> Result<C220CoreStep, C220CoreError> {
        let step = instruction.resolve(pc, self.state.scalar().machine().xregs());
        let cause = match instruction.operation {
            FlagOperation::Set => {
                let scalar_busy = self.scalar_timing.pending_drain_tick().is_some();
                let lsu_busy = self.lsu.as_ref().is_some_and(|lsu| !lsu.is_idle());
                if scalar_busy || lsu_busy {
                    if let Some(lsu) = &mut self.lsu {
                        lsu.trigger_set_flag_flush();
                    }
                    Some(if lsu_busy {
                        C220StallCause::LsuDependency
                    } else {
                        C220StallCause::ScalarDependency
                    })
                } else {
                    self.pipeline_events
                        .set(self.next_instruction_id, step, None, tick);
                    None
                }
            }
            FlagOperation::Wait => self
                .pipeline_events
                .consume(self.next_instruction_id, step, tick)
                .is_none()
                .then_some(C220StallCause::PipelineEventDependency),
        };
        if let Some(cause) = cause {
            return Ok(C220CoreStep::Stalled(C220Stall {
                tick,
                pc,
                resume_tick: tick.checked_add(1).ok_or(C220CoreError::TimeOverflow)?,
                cause,
            }));
        }
        self.state.commit_c220_sequential_issue();
        Ok(C220CoreStep::Executed {
            tick,
            instruction: C220CoreInstruction::ScalarFlag(step),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::architecture::Architecture;
    use crate::memory::{mapped::MappedMemory, sparse::SparseMemory, ub::UbMemory};
    use crate::sim::c220::{
        core::C220CoreTimingRules,
        mte::{mte2::C220Mte2TimingRules, mte3::C220Mte3TimingRules},
        scalar::timing::{C220ScalarTimingClass, C220ScalarTimingTicket},
        state::C220State,
        vector::pipeline::C220VectorTimingRules,
    };
    use crate::sim::common::scalar::{ScalarMachine, ScalarStepper};
    use std::num::NonZeroU64;

    #[test]
    fn scalar_set_waits_for_retirement_and_publishes_to_the_receiving_pipe() {
        for destination in [0_u32, 1, 4, 5] {
            let one = NonZeroU64::new(1).unwrap();
            let mut core = C220Core::new(
                C220State::new(
                    ScalarStepper::new(
                        ScalarMachine::from_pem_initial_state(Architecture::Dav2201),
                        0,
                    ),
                    UbMemory::new(256, 256),
                ),
                MappedMemory::bind(SparseMemory::new(vec![], 256, 256), &[]).unwrap(),
                C220CoreTimingRules {
                    mte2: C220Mte2TimingRules {
                        issue_interval: one,
                        startup_ticks: 0,
                        bytes_per_tick: one,
                        retire_ticks: 0,
                    },
                    mte3: C220Mte3TimingRules {
                        issue_interval: one,
                        startup_ticks: 0,
                        bytes_per_tick: one,
                        retire_ticks: 0,
                    },
                    vector: C220VectorTimingRules {
                        dispatch_ticks: 0,
                        uop_issue_interval: one,
                        ub_response_ticks: 1,
                    },
                },
            )
            .unwrap();
            core.scalar_timing.issue(C220ScalarTimingTicket {
                class: C220ScalarTimingClass::Fixed,
                issue_tick: 0,
                retire_tick: 5,
                execution_stage: 1,
                source_register: None,
                destination_register: None,
            });
            let set = 0x40a0_0000 | (destination << 7) | (1 << 18) | 3;
            assert!(matches!(
                core.step_word_at(1, set).unwrap(),
                C220CoreStep::Stalled(C220Stall {
                    cause: C220StallCause::ScalarDependency,
                    ..
                })
            ));
            assert_eq!(core.state().scalar().pc(), 0);
            assert!(core.pipeline_events.ready(0, destination as u8).is_empty());
            assert!(matches!(
                core.step_word_at(5, set).unwrap(),
                C220CoreStep::Executed {
                    instruction: C220CoreInstruction::ScalarFlag(_),
                    ..
                }
            ));
            let published = core.pipeline_events.ready(0, destination as u8);
            assert_eq!(published.len(), 1);
            assert_eq!(published[0].published_tick, 5);
            assert_eq!(published[0].step.flag_id, 7);
            assert!(matches!(
                core.step_word_at(6, set + (1 << 21)).unwrap(),
                C220CoreStep::Executed { .. }
            ));
            core.advance_to(7).unwrap();
            assert!(core.pipeline_events.ready(0, destination as u8).is_empty());
        }
    }
}
