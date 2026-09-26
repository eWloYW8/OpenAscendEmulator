use super::*;
use crate::sim::c220::mte::fixp::C220FixpRuntimeError;
use crate::sim::c220::mte::fixp::{
    C220FixpAdmission, C220FixpExternalCommand, C220FixpSyncBindings,
};
use crate::sim::c220::sync::C220HardwareFlagState;

impl From<C220FixpRuntimeError> for C220MtePipelineError {
    fn from(error: C220FixpRuntimeError) -> Self {
        Self::FixpExternal(Box::new(error))
    }
}

impl C220MtePipeline {
    pub fn configure_fixp_runtime(
        &mut self,
        stages: &[C220FixpRuntimeStage],
        biu: crate::sim::c220::mte::interface::biu_write::command::C220BiuWriteConfig,
    ) -> Result<(), C220MtePipelineError> {
        if self.timed_memory.is_some() || self.biu_write_commands.is_some() {
            return Err(C220MtePipelineError::BiuOwnedSource);
        }
        self.bind_external_fixp_stages(stages)?;
        self.connect_fixp_biu(biu)
    }
    pub fn external_fixp_admission_blocked(&self, command: C220FixpExternalCommand) -> bool {
        !command.command.descriptor.is_disabled() && self.mte3.retirement_pending()
    }

    /// The frontend resolves attached synchronization before using this entry.
    /// Dispatched MTE3 records block even after their transport queues drain.
    pub fn admit_external_fixp(
        &mut self,
        engine: &mut C220FixpRuntime,
        id: u64,
        first_requests: (u32, u32),
        command: C220FixpExternalCommand,
    ) -> Result<C220FixpAdmission, C220MtePipelineError> {
        if self.external_fixp_admission_blocked(command) {
            return Ok(C220FixpAdmission::Mte3RetirementPending);
        }
        Ok(engine.admit(self.tick(), id, first_requests.0, first_requests.1, command)?)
    }

    pub fn admit_external_fixp_with_flags(
        &mut self,
        engine: &mut C220FixpRuntime,
        id: u64,
        first_requests: (u32, u32),
        command: C220FixpExternalCommand,
        bindings: &mut C220FixpSyncBindings,
        flags: &mut C220HardwareFlagState,
    ) -> Result<C220FixpAdmission, C220MtePipelineError> {
        bindings.capture_pending(id, flags);
        if self.external_fixp_admission_blocked(command) {
            return Ok(C220FixpAdmission::Mte3RetirementPending);
        }
        Ok(engine.admit_with_flags(self.tick(), id, first_requests, command, bindings, flags)?)
    }

    /// Register all stages in the explicit order selected by the core owner.
    /// A bound engine must not also receive direct per-cycle stage calls.
    pub fn bind_external_fixp_stages(
        &mut self,
        stages: &[C220FixpRuntimeStage],
    ) -> Result<(), C220MtePipelineError> {
        if !self.fixp_events.is_empty() || !self.external_fixp_events.is_empty() || !self.is_idle()
        {
            return Err(C220MtePipelineError::FixpBindingBusy);
        }
        if stages.len() != 11
            || stages
                .iter()
                .enumerate()
                .any(|(index, stage)| stages[..index].contains(stage))
        {
            return Err(C220MtePipelineError::InvalidExternalFixpStages);
        }
        for (index, &stage) in stages.iter().enumerate() {
            self.external_fixp_events
                .push(C220FixpStageEvents::register(
                    &mut self.events,
                    self.clock,
                    stage,
                    |phase| Callback::FixpExternal(index, phase),
                ));
        }
        Ok(())
    }

    pub fn next_external_fixp_event_tick(&self, engine: &C220FixpRuntime) -> Option<u64> {
        (!self.is_idle() || !engine.is_idle()).then(|| self.events.tick().saturating_add(1))
    }

    pub fn advance_external_fixp(
        &mut self,
        tick: u64,
        engine: &mut C220FixpRuntime,
        memory: C220FixpRuntimeMemory<'_>,
        mut gates: impl C220FixpSync,
    ) -> Result<(), C220MtePipelineError> {
        self.advance_inner(tick, None, Some((engine, memory, &mut gates)), None)
    }

    pub(super) fn handle_external_fixp(
        &mut self,
        index: usize,
        phase: C220FixpCallback,
        engine: &mut C220FixpRuntime,
        memory: &mut C220FixpRuntimeMemory<'_>,
        gates: &mut dyn C220FixpSync,
    ) -> Result<(), C220MtePipelineError> {
        let binding = self.external_fixp_events[index];
        let stage = binding.stage();
        let tick = self.events.tick();
        if phase == C220FixpCallback::Probe {
            binding.probe(&mut self.events, engine.stage_ready_tick(stage, tick));
            return Ok(());
        }
        use C220FixpRuntimeStage::*;
        let outcome = match stage {
            GenerateRead => C220FixpRuntimeEvent::Read(C220FixpEvent::GeneratedRead(
                engine.generate_read(tick)?,
            )),
            SendRead => C220FixpRuntimeEvent::Read(C220FixpEvent::SentRead(engine.send_read(
                tick,
                &self.fixp_stores,
                gates,
            )?)),
            SendL0c => C220FixpRuntimeEvent::Read(C220FixpEvent::SentL0c(
                engine.send_l0c(tick, memory.l0c)?,
            )),
            ReceiveL0c => {
                let (response, functional) = engine.receive_l0c(tick, memory, |slice, bytes| {
                    self.trace.push(match bytes {
                        Some(bytes) => C220MtePipelineEvent::FixpExternalStored {
                            slice: slice.clone(),
                            bytes: bytes.to_vec(),
                        },
                        None => C220MtePipelineEvent::FixpSlice(slice.clone()),
                    });
                })?;
                C220FixpRuntimeEvent::Read(C220FixpEvent::ReceivedL0c {
                    response,
                    functional,
                })
            }
            Convert => {
                C220FixpRuntimeEvent::Read(C220FixpEvent::Converted(engine.convert(tick, gates)?))
            }
            Slice => C220FixpRuntimeEvent::Read(C220FixpEvent::Sliced(
                engine.slice(tick, &self.fixp_stores)?,
            )),
            Transpose => C220FixpRuntimeEvent::Transposed(engine.transpose(tick)?),
            Align => C220FixpRuntimeEvent::Aligned(engine.align(tick, &self.fixp_stores)?),
            Packetize => C220FixpRuntimeEvent::Packetized(engine.packetize(self)?),
            GenerateWrite => C220FixpRuntimeEvent::GeneratedWrite(engine.generate_write(tick)?),
            SendWrite => C220FixpRuntimeEvent::SentWrite(engine.send_write(self)?),
        };
        self.trace.push(C220MtePipelineEvent::FixpExternal(outcome));
        Ok(())
    }
}
