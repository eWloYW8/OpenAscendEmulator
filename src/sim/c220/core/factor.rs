use std::num::NonZeroU32;

use super::{C220Core, C220CoreError, C220CoreInstruction, C220CoreStep};
use crate::isa::c220::mte::factor::C220FactorLoad;
use crate::sim::c220::memory::C220LocalBuffer;
use crate::sim::c220::mte::factor::{
    C220FactorLoadResult, c220_factor_l1_requests, execute_c220_factor_load_from_memories,
};
use crate::sim::c220::mte::fixp::{C220FixpAdmission, C220FixpEngine};
use crate::sim::c220::mte::interface::C220MteL1ReadPort;
use crate::sim::c220::schedule::{C220Stall, C220StallCause};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FactorReadConfig {
    pub port: C220MteL1ReadPort,
    pub access_width: NonZeroU32,
    pub output_bandwidth: NonZeroU32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FactorOutcome {
    pub instruction_id: u64,
    pub load: C220FactorLoad,
    pub admitted_tick: u64,
    pub completed_tick: u64,
    pub retired_tick: u64,
    pub result: C220FactorLoadResult,
}

impl C220Core {
    pub fn configure_factor_reads(
        &mut self,
        config: C220FactorReadConfig,
    ) -> Result<(), C220CoreError> {
        if self.queued_fixp_commands() != 0 {
            return Err(C220CoreError::MtePipelineBusy);
        }
        let fixp = factor_context(&mut self.fixp, &mut self.external_fixp)
            .ok_or(C220CoreError::FixpUnconfigured)?;
        if !fixp.engine.is_idle() {
            return Err(C220CoreError::MtePipelineBusy);
        }
        *fixp.factor_reads = Some(config);
        Ok(())
    }

    pub fn factor_outcomes(&self) -> &[C220FactorOutcome] {
        self.fixp
            .as_ref()
            .map(|fixp| fixp.factor_outcomes.as_slice())
            .or_else(|| {
                self.external_fixp
                    .as_ref()
                    .map(|fixp| fixp.factor_outcomes.as_slice())
            })
            .unwrap_or(&[])
    }

    pub(super) fn dispatch_factor_at(
        &mut self,
        tick: u64,
        issue: super::fixp_frontend::FixpIssue,
        load: C220FactorLoad,
    ) -> Result<C220CoreStep, C220CoreError> {
        let super::fixp_frontend::FixpIssue {
            instruction_id: id,
            pc,
            word,
        } = issue;
        let fixp = factor_context(&mut self.fixp, &mut self.external_fixp)
            .ok_or(C220CoreError::FixpUnconfigured)?;
        let config = fixp.factor_reads.ok_or(C220CoreError::FactorUnconfigured)?;
        let admission = fixp.engine.admit_factor_batch(
            tick,
            config.port,
            c220_factor_l1_requests(load, id, config.access_width, config.output_bandwidth),
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
        Ok(C220CoreStep::Executed {
            tick,
            instruction: C220CoreInstruction::Factor {
                instruction_id: id,
                pc,
                word,
                load,
                admission,
            },
        })
    }

    pub(super) fn retire_factor_at(&mut self, tick: u64) -> Result<(), C220CoreError> {
        let Some(fixp) = factor_context(&mut self.fixp, &mut self.external_fixp) else {
            return Ok(());
        };
        let Some(id) = fixp.engine.command_retirement_head() else {
            return Ok(());
        };
        let Some(&state) = fixp.engine.factor_commands().get(&id) else {
            return Ok(());
        };
        if !fixp.engine.can_retire_at(tick, id) {
            return Ok(());
        }
        let Some(completed_tick) = state.completed_tick.filter(|&completed| completed < tick)
        else {
            return Ok(());
        };
        let result = execute_c220_factor_load_from_memories(
            self.local_memory.l1(),
            self.state.ub(),
            fixp.factors,
            state.load,
        )?;
        fixp.engine.retire_factor(tick, id)?;
        fixp.factor_outcomes.push(C220FactorOutcome {
            instruction_id: id,
            load: state.load,
            admitted_tick: state.admitted_tick,
            completed_tick,
            retired_tick: tick,
            result,
        });
        Ok(())
    }
}

struct FactorContext<'a> {
    engine: &'a mut C220FixpEngine,
    factors: &'a mut C220LocalBuffer,
    factor_reads: &'a mut Option<C220FactorReadConfig>,
    factor_outcomes: &'a mut Vec<C220FactorOutcome>,
}

fn factor_context<'a>(
    local: &'a mut Option<super::fixp::CoreFixp>,
    external: &'a mut Option<super::external_fixp::CoreExternalFixp>,
) -> Option<FactorContext<'a>> {
    if let Some(fixp) = local {
        Some(FactorContext {
            engine: &mut fixp.engine,
            factors: &mut fixp.factors,
            factor_reads: &mut fixp.factor_reads,
            factor_outcomes: &mut fixp.factor_outcomes,
        })
    } else {
        external.as_mut().map(|fixp| FactorContext {
            engine: fixp.engine.shared_engine_mut(),
            factors: &mut fixp.factors,
            factor_reads: &mut fixp.factor_reads,
            factor_outcomes: &mut fixp.factor_outcomes,
        })
    }
}
