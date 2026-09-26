use super::decode::{C220DecodedWord, C220DispatchKind};
use super::{C220Core, C220CoreError, C220CoreInstruction, C220CoreStep};
use crate::architecture::Architecture;
use crate::isa::flow::{FlagOperation, PipelineBarrierScope, PipelineBarrierStep};
use crate::sim::c220::scalar::timing::C220ScalarTimingRule;
use crate::sim::c220::schedule::{C220Stall, C220StallCause};
use crate::sim::c220::state::C220ExecutionError;
use crate::sim::c220::vector::dispatch::VectorStep;
use crate::sim::common::scalar::{ScalarInstructionStep, ScalarProgramStep};

impl C220Core {
    pub(super) fn spr_dispatch_dependency(&self, word: u32) -> Option<(u64, C220StallCause)> {
        use crate::isa::scalar::ScalarInstruction;
        let instruction = ScalarInstruction::from_word(Architecture::Dav2201, word)?;
        if matches!(
            instruction,
            ScalarInstruction::ScalarKey2MoveToSpr {
                encoded_destination_spr: 2 | 48..=51,
                ..
            }
        ) {
            return self.pending_compute_drain();
        }
        let ScalarInstruction::ScalarKey2MoveFromSpr {
            encoded_source_spr, ..
        } = instruction
        else {
            return None;
        };
        match encoded_source_spr {
            17 | 19 | 57 | 63 | 74 | 87 => self
                .vector
                .pending_drain_tick()
                .map(|tick| (tick, C220StallCause::VectorDependency)),
            54 => self
                .pending_mte1_tick()
                .map(|tick| (tick, C220StallCause::Mte1Dependency)),
            _ => None,
        }
    }

    fn step_barrier_word(&mut self, word: u32) -> Result<ScalarProgramStep, C220ExecutionError> {
        let pc = self.state.scalar.pc();
        if self.state.scalar.is_halted() {
            return Err(C220ExecutionError::ProgramEnded { pc });
        }
        let architecture = self.state.scalar.machine().architecture();
        let barrier = PipelineBarrierStep::decode(architecture, pc, word)
            .filter(|step| step.scope == PipelineBarrierScope::All)
            .ok_or(C220ExecutionError::UnsupportedWord { pc, word })?;
        self.state.scalar.advance_sequential();
        Ok(ScalarProgramStep {
            pc,
            word,
            next_pc: self.state.scalar.pc(),
            instruction: ScalarInstructionStep::Barrier(barrier),
            halted_after: false,
        })
    }

    pub(super) fn dispatch_word_at(
        &mut self,
        tick: u64,
        word: u32,
    ) -> Result<C220CoreStep, C220CoreError> {
        let gate = self.advance_to(tick)?;
        if let Some(stall) = gate {
            return Ok(C220CoreStep::Stalled(stall));
        }
        if self.device_flags.is_blocked() {
            return Ok(C220CoreStep::Stalled(C220Stall {
                tick,
                pc: self.state.scalar().pc(),
                resume_tick: tick.checked_add(1).ok_or(C220CoreError::TimeOverflow)?,
                cause: C220StallCause::DeviceFlagDependency,
            }));
        }
        if let Some(resume_tick) = self.load_dependency_tick(word, tick) {
            return Ok(C220CoreStep::Stalled(C220Stall {
                tick,
                pc: self.state.scalar().pc(),
                resume_tick,
                cause: C220StallCause::ScalarDependency,
            }));
        }
        let pc = self.state.scalar().pc();
        if let Some((ready, cause)) = self.spr_dispatch_dependency(word) {
            return Ok(C220CoreStep::Stalled(C220Stall {
                tick,
                pc,
                resume_tick: ready.max(tick.checked_add(1).ok_or(C220CoreError::TimeOverflow)?),
                cause,
            }));
        }
        if let Some(instruction) =
            crate::isa::c220::control::C220WaitDeviceFlagInstruction::decode(word)
        {
            return self.step_wait_device_flag_at(tick, pc, instruction);
        }
        if let Some(instruction) =
            crate::isa::c220::control::C220SetCrossCoreInstruction::decode(word)
        {
            return self.step_cross_core_at(tick, pc, instruction);
        }
        if crate::isa::c220::scalar::C220ScalarPreload::decode(word).is_some() {
            return self.step_preload_at(tick, word);
        }
        if crate::isa::c220::scalar::C220ScalarAtomicStore::decode(word).is_some() {
            return self.step_atomic_store_at(tick, word);
        }
        if crate::isa::c220::scalar::C220ScalarDirectStore::decode(word).is_some() {
            return self.step_direct_store_at(tick, word);
        }
        if self.lsu.is_some() {
            if let Some(instruction) =
                crate::isa::flow::DcciInstruction::decode(Architecture::Dav2201, word)
            {
                return self.step_maintenance_at(tick, instruction);
            }
            use crate::isa::scalar::{ScalarInstruction, ScalarLoadStoreOperation};
            match ScalarInstruction::from_word(Architecture::Dav2201, word) {
                Some(
                    ScalarInstruction::ScalarLoadStoreImmediate {
                        operation: ScalarLoadStoreOperation::Load,
                        ..
                    }
                    | ScalarInstruction::ScalarIndexedLoad { .. }
                    | ScalarInstruction::ScalarPairLoad { .. },
                ) => return self.step_load_at(tick, word),
                Some(
                    ScalarInstruction::ScalarIndexedStore { .. }
                    | ScalarInstruction::ScalarIndexedImmediateStore { .. }
                    | ScalarInstruction::ScalarStoreImmediate { .. }
                    | ScalarInstruction::ScalarPairStore { .. }
                    | ScalarInstruction::ScalarLoadStoreImmediate {
                        operation: ScalarLoadStoreOperation::Store,
                        ..
                    },
                ) => return self.step_store_at(tick, word),
                _ => {}
            }
        }
        if let Some(barrier) = PipelineBarrierStep::decode(Architecture::Dav2201, pc, word)
            .filter(|barrier| barrier.scope == PipelineBarrierScope::Fix)
        {
            return self.step_fixp_barrier_at(tick, barrier);
        }
        let decoded_word = C220DecodedWord::decode(word);
        match decoded_word.kind {
            C220DispatchKind::Factor(_) | C220DispatchKind::Fixp => {
                return self.step_fixp_at(tick, pc, word);
            }
            C220DispatchKind::Vector => {
                return Ok(match self.vector.step_at(tick, word, &mut self.state)? {
                    VectorStep::Issued(instruction) => C220CoreStep::Executed {
                        tick,
                        instruction: C220CoreInstruction::Vector(instruction),
                    },
                    VectorStep::Stalled(stall) => C220CoreStep::Stalled(stall),
                });
            }
            C220DispatchKind::Mte1 => return self.step_mte1_at(tick, pc, word),
            C220DispatchKind::Mte2 => return self.step_mte2_at(tick, pc, word),
            C220DispatchKind::HardwareFlag(instruction) => {
                return if instruction.source_pipe
                    == crate::isa::c220::hflag::C220HardwareFlagSourcePipe::Fix
                {
                    self.step_fixp_at(tick, pc, word)
                } else {
                    self.step_mte1_at(tick, pc, word)
                };
            }
            _ => {}
        }
        let cube_instruction = match decoded_word.kind {
            C220DispatchKind::Cube(instruction) => Some(instruction),
            _ => None,
        };
        if cube_instruction.is_some() && tick < self.cube.pipeline.next_accept_tick() {
            return Ok(C220CoreStep::Stalled(C220Stall {
                tick,
                pc,
                resume_tick: self.cube.pipeline.next_accept_tick(),
                cause: C220StallCause::CubeDependency,
            }));
        }
        if decoded_word.kind == C220DispatchKind::Mte3 {
            return self.step_mte3_at(tick, pc, word, decoded_word.flow_flag);
        }
        let instruction = match word {
            _ if cube_instruction.is_some() => {
                let decoded = cube_instruction.expect("matched Cube decode");
                self.issue_cube_at(tick, pc, word, decoded)?
            }
            _ if decoded_word
                .flow_flag
                .is_some_and(|flag| flag.source_pipe_code == 1 && flag.trigger_pipe_code == 0) =>
            {
                let flag = decoded_word
                    .flow_flag
                    .expect("matched vector-to-scalar flag")
                    .resolve(pc, self.state.scalar().machine().xregs());
                match flag.instruction.operation {
                    FlagOperation::Set => {
                        self.vector.signal_scalar(flag.flag_id);
                    }
                    FlagOperation::Wait => {
                        let ready_tick = self.vector.scalar_event_ready_tick(flag.flag_id, tick);
                        if ready_tick.is_none_or(|ready| tick < ready) {
                            return Ok(C220CoreStep::Stalled(C220Stall {
                                tick,
                                pc,
                                resume_tick: ready_tick.unwrap_or(
                                    tick.checked_add(1).ok_or(C220CoreError::TimeOverflow)?,
                                ),
                                cause: C220StallCause::VectorDependency,
                            }));
                        }
                        self.vector.consume_scalar_event(flag.flag_id);
                    }
                }
                self.state.commit_c220_sequential_issue();
                C220CoreInstruction::VectorToScalarFlag(flag)
            }
            _ if matches!(
                PipelineBarrierStep::decode(Architecture::Dav2201, pc, word),
                Some(PipelineBarrierStep {
                    scope: PipelineBarrierScope::All,
                    ..
                })
            ) =>
            {
                if let Some((resume_tick, cause)) = self.pending_compute_drain()
                    && tick < resume_tick
                {
                    return Ok(C220CoreStep::Stalled(C220Stall {
                        tick,
                        pc,
                        resume_tick,
                        cause,
                    }));
                }
                C220CoreInstruction::Barrier(self.step_barrier_word(word)?)
            }
            _ => {
                let spr_timing = crate::sim::c220::scalar::spr::write_destination(word)
                    .filter(|register| *register != 3)
                    .map(|destination_spr| {
                        Ok::<_, C220CoreError>(
                            crate::sim::c220::scalar::spr::C220ScalarSprTimingTicket {
                                destination_spr,
                                issue_tick: tick,
                                retire_tick: tick
                                    .checked_add(1)
                                    .ok_or(C220CoreError::TimeOverflow)?,
                            },
                        )
                    })
                    .transpose()?;
                let timing = if let Some(rule) = C220ScalarTimingRule::decode(word) {
                    Some(rule.ticket(tick).ok_or(C220CoreError::TimeOverflow)?)
                } else {
                    None
                };
                let step = self
                    .state
                    .step_scalar_word_with_ub(word, &mut self.memory)?;
                if let Some(ticket) = timing {
                    if let Some(destination) = ticket.destination_register {
                        self.supersede_load_destination(destination);
                    }
                    self.scalar_timing.issue(ticket);
                }
                if let Some(ticket) = spr_timing {
                    self.scalar_timing.issue_spr(ticket);
                }
                C220CoreInstruction::Scalar {
                    step,
                    timing,
                    spr_timing,
                }
            }
        };
        Ok(C220CoreStep::Executed { tick, instruction })
    }
}
