use super::{C220Core, C220CoreError, C220CoreInstruction, C220CoreStep};
use crate::sim::c220::memory::C220LocalBuffer;
use crate::sim::c220::mte::fixp::*;
use crate::sim::c220::mte::interface::biu_write::command::C220BiuWriteConfig;
use crate::sim::c220::schedule::{C220Stall, C220StallCause};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220CoreFixpConfig {
    pub engine: C220FixpEngineConfig,
    pub main_transpose_slots: u32,
    pub total_transpose_slots: usize,
    pub atomics: C220FixpAtomicConfig,
}

pub(super) struct CoreExternalFixp {
    pub engine: C220FixpRuntime,
    pub factors: C220LocalBuffer,
    pub bindings: C220FixpSyncBindings,
    pub atomics: C220FixpAtomicConfig,
    pub factor_reads: Option<super::factor::C220FactorReadConfig>,
    pub factor_outcomes: Vec<super::factor::C220FactorOutcome>,
    pub(super) next_request: u64,
}

impl C220Core {
    /// Configure L1, external and factor FIX execution with one shared engine.
    /// Replaces selecting the L1-only configuration; do not configure both.
    pub fn configure_fixp_runtime(
        &mut self,
        config: C220CoreFixpConfig,
        factors: C220LocalBuffer,
        stages: &[C220FixpRuntimeStage],
        biu: C220BiuWriteConfig,
    ) -> Result<(), C220CoreError> {
        if self.fixp.is_some() || self.external_fixp.is_some() {
            return Err(C220CoreError::MtePipelineBusy);
        }
        if config.engine.l0c_capacity != self.local_memory.l0c().buffer().capacity() {
            return Err(C220CoreError::FixpCapacityMismatch);
        }
        let engine = C220FixpRuntime::new(
            config.engine,
            config.main_transpose_slots,
            config.total_transpose_slots,
        )?;
        let pipeline = self
            .mte_pipeline
            .as_mut()
            .ok_or(C220CoreError::MteUnconfigured)?;
        pipeline.configure_fixp_runtime(stages, biu)?;
        self.external_fixp = Some(CoreExternalFixp {
            engine,
            factors,
            bindings: C220FixpSyncBindings::default(),
            atomics: config.atomics,
            factor_reads: None,
            factor_outcomes: Vec::new(),
            next_request: 0,
        });
        Ok(())
    }

    pub fn fixp_runtime(&self) -> Option<&C220FixpRuntime> {
        self.external_fixp.as_ref().map(|fixp| &fixp.engine)
    }

    pub fn receive_external_fixp_dbid_at(
        &mut self,
        tick: u64,
        tag: std::num::NonZeroU32,
    ) -> Result<(), C220CoreError> {
        self.advance_to(tick)?;
        if self.external_fixp.is_none() {
            return Err(C220CoreError::FixpUnconfigured);
        }
        self.mte_pipeline
            .as_mut()
            .ok_or(C220CoreError::MteUnconfigured)?
            .receive_biu_write_dbid(
                crate::sim::c220::mte::interface::biu_read::C220BiuSubcore::Cube,
                tag,
            )?;
        Ok(())
    }

    pub fn receive_external_fixp_response_at(
        &mut self,
        tick: u64,
        tag: std::num::NonZeroU32,
    ) -> Result<
        crate::sim::c220::mte::interface::biu_write::data::C220BiuWriteResponse,
        C220CoreError,
    > {
        self.advance_to(tick)?;
        let fixp = self
            .external_fixp
            .as_mut()
            .ok_or(C220CoreError::FixpUnconfigured)?;
        let response = self
            .mte_pipeline
            .as_mut()
            .ok_or(C220CoreError::MteUnconfigured)?
            .receive_biu_write_response(tag)?;
        fixp.engine.complete_response(response)?;
        Ok(response)
    }

    pub(super) fn step_external_fixp_at(
        &mut self,
        tick: u64,
        pc: u64,
        word: u32,
    ) -> Result<C220CoreStep, C220CoreError> {
        use crate::isa::c220::mte::fixp::{C220FixpDestination, C220FixpInstruction};
        let destination = C220FixpInstruction::decode(word)
            .ok_or(C220FixpExecutionError::Instruction(word))?
            .destination;
        let machine = self.state.scalar().machine();
        let command = C220FixpExternalCommand::capture_destination(
            word,
            destination,
            machine
                .spr_value(3)
                .ok_or(C220FixpExecutionError::MissingSpr(3))?,
            self.state.isa_instance_index,
            |register| machine.xregs().get(usize::from(register)).copied(),
            |register| machine.spr_value(u16::from(register)),
        )?;
        let fixp = self
            .external_fixp
            .as_mut()
            .ok_or(C220CoreError::FixpUnconfigured)?;
        let d = command.command.descriptor;
        let nd = if d.nz_to_nd() {
            u64::from(d.nd_count())
        } else {
            1
        };
        let reservation = if d.is_disabled() {
            0
        } else {
            nd * u64::from(d.rows()) * u64::from(d.columns()).div_ceil(16) * 128 + 1
        };
        if reservation > u64::from(u32::MAX) {
            return Err(C220CoreError::FixpRequestCapacity {
                required: reservation,
            });
        }
        if fixp.engine.is_idle() {
            fixp.next_request = 0;
        }
        let pending_mte3 =
            self.mte3.pending_commands().next().is_some() || !self.mte3.dma_commands.is_empty();
        let admission =
            if destination == C220FixpDestination::External && !d.is_disabled() && pending_mte3 {
                fixp.bindings
                    .capture_pending(self.next_instruction_id, &mut self.hardware_flags);
                C220FixpAdmission::Mte3RetirementPending
            } else if fixp.next_request + reservation > u64::from(u32::MAX) {
                C220FixpAdmission::ReadGenerationBusy
            } else if destination == C220FixpDestination::L1 {
                fixp.engine.admit_l1_nz2nd_with_flags(
                    tick,
                    self.next_instruction_id,
                    (fixp.next_request as u32, fixp.next_request as u32),
                    command,
                    &mut fixp.bindings,
                    &mut self.hardware_flags,
                )?
            } else {
                self.mte_pipeline
                    .as_mut()
                    .ok_or(C220CoreError::MteUnconfigured)?
                    .admit_external_fixp_with_flags(
                        &mut fixp.engine,
                        self.next_instruction_id,
                        (fixp.next_request as u32, fixp.next_request as u32),
                        command,
                        &mut fixp.bindings,
                        &mut self.hardware_flags,
                    )?
            };
        if !matches!(
            admission,
            C220FixpAdmission::Active | C220FixpAdmission::DisabledReady
        ) {
            return Ok(C220CoreStep::Stalled(C220Stall {
                tick,
                pc,
                resume_tick: tick.checked_add(1).ok_or(C220CoreError::TimeOverflow)?,
                cause: C220StallCause::FixpDependency,
            }));
        }
        fixp.next_request += reservation;
        self.state.commit_c220_sequential_issue();
        Ok(C220CoreStep::Executed {
            tick,
            instruction: C220CoreInstruction::FixpExternal {
                instruction_id: self.next_instruction_id,
                pc,
                word,
                command,
                destination,
                admission,
            },
        })
    }
}
