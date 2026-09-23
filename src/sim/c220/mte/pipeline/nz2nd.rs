use super::*;
use crate::sim::c220::mte::fixp::C220FixpNz2ndEngineError;

impl From<C220FixpNz2ndEngineError> for C220MtePipelineError {
    fn from(error: C220FixpNz2ndEngineError) -> Self {
        Self::Nz2nd(Box::new(error))
    }
}

impl C220MtePipeline {
    /// Register all stages in the explicit order selected by the core owner.
    /// A bound engine must not also receive direct per-cycle stage calls.
    pub fn bind_nz2nd_stages(
        &mut self,
        stages: &[C220FixpNz2ndStage],
    ) -> Result<(), C220MtePipelineError> {
        if !self.fixp_events.is_empty() || !self.nz2nd_events.is_empty() || !self.is_idle() {
            return Err(C220MtePipelineError::FixpBindingBusy);
        }
        if stages.len() != 11
            || stages
                .iter()
                .enumerate()
                .any(|(index, stage)| stages[..index].contains(stage))
        {
            return Err(C220MtePipelineError::InvalidNz2ndStages);
        }
        for (index, &stage) in stages.iter().enumerate() {
            self.nz2nd_events.push(C220FixpStageEvents::register(
                &mut self.events,
                self.clock,
                stage,
                |phase| Callback::Nz2nd(index, phase),
            ));
        }
        Ok(())
    }

    pub fn next_nz2nd_event_tick(&self, engine: &C220FixpNz2ndEngine) -> Option<u64> {
        (!self.is_idle() || !engine.is_idle()).then(|| self.events.tick().saturating_add(1))
    }

    pub fn advance_nz2nd(
        &mut self,
        tick: u64,
        engine: &mut C220FixpNz2ndEngine,
        memory: C220FixpNz2ndMemory<'_>,
        mut gates: impl C220FixpSync,
    ) -> Result<(), C220MtePipelineError> {
        self.advance_inner(tick, None, Some((engine, memory, &mut gates)))
    }

    pub(super) fn handle_nz2nd(
        &mut self,
        index: usize,
        phase: C220FixpCallback,
        engine: &mut C220FixpNz2ndEngine,
        memory: &mut C220FixpNz2ndMemory<'_>,
        gates: &mut dyn C220FixpSync,
    ) -> Result<(), C220MtePipelineError> {
        let binding = self.nz2nd_events[index];
        let stage = binding.stage();
        let tick = self.events.tick();
        if phase == C220FixpCallback::Probe {
            binding.probe(&mut self.events, engine.stage_ready_tick(stage, tick));
            return Ok(());
        }
        use C220FixpNz2ndStage::*;
        let outcome = match stage {
            GenerateRead => {
                C220FixpNz2ndEvent::Read(C220FixpEvent::GeneratedRead(engine.generate_read(tick)?))
            }
            SendRead => {
                C220FixpNz2ndEvent::Read(C220FixpEvent::SentRead(engine.send_read(tick, gates)?))
            }
            SendL0c => {
                C220FixpNz2ndEvent::Read(C220FixpEvent::SentL0c(engine.send_l0c(tick, memory.l0c)?))
            }
            ReceiveL0c => {
                let (response, functional) = engine.receive_l0c(
                    tick,
                    memory.l0c,
                    memory.slopes,
                    memory.external,
                    memory.atomics,
                    |slice, bytes| {
                        self.trace.push(C220MtePipelineEvent::Nz2ndStored {
                            slice: slice.clone(),
                            bytes: bytes.to_vec(),
                        });
                    },
                )?;
                C220FixpNz2ndEvent::Read(C220FixpEvent::ReceivedL0c {
                    response,
                    functional,
                })
            }
            Convert => {
                C220FixpNz2ndEvent::Read(C220FixpEvent::Converted(engine.convert(tick, gates)?))
            }
            Slice => C220FixpNz2ndEvent::Read(C220FixpEvent::Sliced(engine.slice(tick)?)),
            Transpose => C220FixpNz2ndEvent::Transposed(engine.transpose(tick)?),
            Align => C220FixpNz2ndEvent::Aligned(engine.align(tick)?),
            Packetize => C220FixpNz2ndEvent::Packetized(engine.packetize(self)?),
            GenerateWrite => C220FixpNz2ndEvent::GeneratedWrite(engine.generate_write(tick)?),
            SendWrite => C220FixpNz2ndEvent::SentWrite(engine.send_write(self)?),
        };
        self.trace.push(C220MtePipelineEvent::Nz2nd(outcome));
        Ok(())
    }
}
