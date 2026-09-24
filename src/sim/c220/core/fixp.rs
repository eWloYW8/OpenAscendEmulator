use super::{C220Core, C220CoreError, C220CoreInstruction, C220CoreStep};
use crate::sim::c220::memory::C220LocalBuffer;
use crate::sim::c220::mte::fixp::{
    C220FixpAdmission, C220FixpCommand, C220FixpEngine, C220FixpEngineConfig, C220FixpStage,
    C220FixpSyncBindings,
};
use crate::sim::c220::schedule::{C220Stall, C220StallCause};

pub(super) struct CoreFixp {
    pub engine: C220FixpEngine,
    pub factors: C220LocalBuffer,
    pub bindings: C220FixpSyncBindings,
    pub factor_reads: Option<super::factor::C220FactorReadConfig>,
    pub factor_outcomes: Vec<super::factor::C220FactorOutcome>,
    next_request: u64,
}

impl C220Core {
    /// Connect ordinary L0C-to-L1 FIX to the shared memory clock. Stage order
    /// and buffer geometry are explicit inputs, not inferred timing defaults.
    pub fn configure_fixp_l1(
        &mut self,
        config: C220FixpEngineConfig,
        frontend: super::C220FixpFrontendConfig,
        factors: C220LocalBuffer,
        stages: &[C220FixpStage],
    ) -> Result<(), C220CoreError> {
        if self.fixp.is_some() || self.external_fixp.is_some() {
            return Err(C220CoreError::MtePipelineBusy);
        }
        if config.l0c_capacity != self.local_memory.l0c().buffer().capacity() {
            return Err(C220CoreError::FixpCapacityMismatch);
        }
        let engine = C220FixpEngine::new(config)?;
        self.mte_pipeline
            .as_mut()
            .ok_or(C220CoreError::MteUnconfigured)?
            .bind_fixp_stages(stages)?;
        self.fixp = Some(CoreFixp {
            engine,
            factors,
            bindings: C220FixpSyncBindings::default(),
            factor_reads: None,
            factor_outcomes: Vec::new(),
            next_request: 0,
        });
        self.fixp_frontend.config = Some(frontend);
        Ok(())
    }

    pub fn fixp_engine(&self) -> Option<&C220FixpEngine> {
        self.fixp.as_ref().map(|fixp| &fixp.engine).or_else(|| {
            self.external_fixp
                .as_ref()
                .map(|fixp| fixp.engine.shared_engine())
        })
    }

    pub(super) fn fixp_engine_mut(&mut self) -> Option<&mut C220FixpEngine> {
        self.fixp.as_mut().map(|fixp| &mut fixp.engine).or_else(|| {
            self.external_fixp
                .as_mut()
                .map(|fixp| fixp.engine.shared_engine_mut())
        })
    }

    pub fn fixp_sync_bindings(&self) -> Option<&C220FixpSyncBindings> {
        self.fixp
            .as_ref()
            .map(|fixp| &fixp.bindings)
            .or_else(|| self.external_fixp.as_ref().map(|fixp| &fixp.bindings))
    }

    pub fn fixp_factors(&self) -> Option<&C220LocalBuffer> {
        self.fixp
            .as_ref()
            .map(|fixp| &fixp.factors)
            .or_else(|| self.external_fixp.as_ref().map(|fixp| &fixp.factors))
    }

    pub fn fixp_factors_mut(&mut self) -> Option<&mut C220LocalBuffer> {
        self.fixp
            .as_mut()
            .map(|fixp| &mut fixp.factors)
            .or_else(|| self.external_fixp.as_mut().map(|fixp| &mut fixp.factors))
    }

    pub(super) fn dispatch_fixp_l1_at(
        &mut self,
        tick: u64,
        issue: super::fixp_frontend::FixpIssue,
        command: C220FixpCommand,
    ) -> Result<C220CoreStep, C220CoreError> {
        let super::fixp_frontend::FixpIssue {
            instruction_id,
            pc,
            word,
        } = issue;
        let (engine, bindings, next_request) = if let Some(fixp) = &mut self.fixp {
            (&mut fixp.engine, &mut fixp.bindings, &mut fixp.next_request)
        } else if let Some(fixp) = &mut self.external_fixp {
            (
                fixp.engine.shared_engine_mut(),
                &mut fixp.bindings,
                &mut fixp.next_request,
            )
        } else {
            return Err(C220CoreError::FixpUnconfigured);
        };
        let reservation = if command.descriptor.is_disabled() {
            0
        } else {
            u64::from(command.descriptor.columns()).div_ceil(16)
                * (u64::from(command.descriptor.rows()) * 64 + 1)
        };
        if reservation > u64::from(u32::MAX) {
            return Err(C220CoreError::FixpRequestCapacity {
                required: reservation,
            });
        }
        if engine.is_idle() {
            *next_request = 0;
        }
        if *next_request + reservation > u64::from(u32::MAX) {
            return Ok(C220CoreStep::Stalled(C220Stall {
                tick,
                pc,
                resume_tick: tick.checked_add(1).ok_or(C220CoreError::TimeOverflow)?,
                cause: C220StallCause::FixpDependency,
            }));
        }
        let admission = engine.admit_with_flags(
            tick,
            instruction_id,
            *next_request as u32,
            command,
            bindings,
            &mut self.hardware_flags,
        )?;
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
        *next_request += reservation;
        Ok(C220CoreStep::Executed {
            tick,
            instruction: C220CoreInstruction::Fixp {
                instruction_id,
                pc,
                word,
                command,
                admission,
            },
        })
    }
}
