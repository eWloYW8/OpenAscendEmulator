use super::{C220Core, C220CoreError, C220CoreInstruction, C220CoreStep};
use crate::sim::c220::memory::C220LocalBuffer;
use crate::sim::c220::mte::fixp::{
    C220FixpAdmission, C220FixpCommand, C220FixpEngine, C220FixpEngineConfig,
    C220FixpExecutionError, C220FixpStage, C220FixpSyncBindings,
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
        Ok(())
    }

    pub fn fixp_engine(&self) -> Option<&C220FixpEngine> {
        self.fixp.as_ref().map(|fixp| &fixp.engine)
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

    pub(super) fn step_fixp_at(
        &mut self,
        tick: u64,
        pc: u64,
        word: u32,
    ) -> Result<C220CoreStep, C220CoreError> {
        if crate::isa::c220::mte::fixp::C220FixpInstruction::decode(word).is_some_and(
            |instruction| {
                instruction.destination
                    == crate::isa::c220::mte::fixp::C220FixpDestination::External
            },
        ) {
            return self.step_external_fixp_at(tick, pc, word);
        }
        let machine = self.state.scalar().machine();
        let control = machine
            .spr_value(3)
            .ok_or(C220FixpExecutionError::MissingSpr(3))?;
        let command = C220FixpCommand::capture_l1(
            word,
            control,
            |register| machine.xregs().get(usize::from(register)).copied(),
            |register| machine.spr_value(u16::from(register)),
        )?;
        let fixp = self.fixp.as_mut().ok_or(C220CoreError::FixpUnconfigured)?;
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
        if fixp.engine.is_idle() {
            fixp.next_request = 0;
        }
        if fixp.next_request + reservation > u64::from(u32::MAX) {
            return Ok(C220CoreStep::Stalled(C220Stall {
                tick,
                pc,
                resume_tick: tick.checked_add(1).ok_or(C220CoreError::TimeOverflow)?,
                cause: C220StallCause::FixpDependency,
            }));
        }
        let admission = fixp.engine.admit_with_flags(
            tick,
            self.next_instruction_id,
            fixp.next_request as u32,
            command,
            &mut fixp.bindings,
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
        self.state.commit_c220_sequential_issue();
        fixp.next_request += reservation;
        Ok(C220CoreStep::Executed {
            tick,
            instruction: C220CoreInstruction::Fixp {
                instruction_id: self.next_instruction_id,
                pc,
                word,
                command,
                admission,
            },
        })
    }
}
