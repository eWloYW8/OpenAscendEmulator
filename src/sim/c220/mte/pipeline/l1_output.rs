use super::*;
use crate::sim::c220::mte::fixp::C220FixpAdmission;
use crate::sim::c220::mte::l1_to_out::{
    C220L1OutputCommand, C220L1OutputCommandState, C220L1OutputEngine, C220L1OutputEngineError,
    C220L1OutputEvent, C220L1OutputStage,
};

impl C220MtePipeline {
    pub fn issue_mte3_l1_output(
        &mut self,
        id: u64,
        command: C220L1OutputCommand,
    ) -> Result<C220Mte3Record, C220MtePipelineError> {
        if self.l1_output_events.is_empty() {
            return Err(C220MtePipelineError::L1OutputContextMismatch);
        }
        Ok(self.mte3_events.issue_command(
            &mut self.events,
            &mut self.mte3,
            id,
            super::super::mte3::frontend::C220Mte3Command::L1Output(command),
        )?)
    }

    /// Register stages in the core owner's clock order. The shared L1 interface
    /// delivers source completions; the shared BIU delivers write responses.
    pub fn bind_l1_output_stages(
        &mut self,
        stages: &[C220L1OutputStage],
    ) -> Result<(), C220MtePipelineError> {
        if !self.is_idle()
            || !self.l1_output_events.is_empty()
            || self.core_kind != crate::sim::c220::device::C220CoreKind::Cube
            || self.biu_write_commands.is_none()
            || stages.len() != 5
            || stages
                .iter()
                .enumerate()
                .any(|(i, stage)| stages[..i].contains(stage))
        {
            return Err(C220MtePipelineError::InvalidL1OutputBinding);
        }
        for (index, &stage) in stages.iter().enumerate() {
            self.l1_output_events.push(C220FixpStageEvents::register(
                &mut self.events,
                self.clock,
                stage,
                |phase| Callback::L1Output(index, phase),
            ));
        }
        Ok(())
    }

    /// Physical admission after the owning MTE3 scheduler resolves dispatch
    /// ordering and synchronization. IDs are unique across the shared core.
    pub fn admit_l1_output(
        &mut self,
        engine: &mut C220L1OutputEngine,
        id: u64,
        command: C220L1OutputCommand,
    ) -> Result<C220FixpAdmission, C220MtePipelineError> {
        if self.l1_output_events.is_empty() {
            return Err(C220MtePipelineError::L1OutputContextMismatch);
        }
        if self.l1_output_active.contains(&id) {
            return Err(C220L1OutputEngineError::DuplicateCommand(id).into());
        }
        let result = engine.admit(self.tick(), id, command)?;
        if result == C220FixpAdmission::Active {
            self.l1_output_active.insert(id);
        }
        Ok(result)
    }

    /// Remove physical ownership only after the scheduler's functional commit.
    pub fn retire_l1_output(
        &mut self,
        engine: &mut C220L1OutputEngine,
        id: u64,
    ) -> Result<C220L1OutputCommandState, C220MtePipelineError> {
        if !self.l1_output_active.contains(&id) {
            return Err(C220L1OutputEngineError::UnknownCommand(id).into());
        }
        let state = engine.retire(id)?;
        self.l1_output_active.remove(&id);
        Ok(state)
    }

    pub fn advance_l1_output(
        &mut self,
        tick: u64,
        engine: &mut C220L1OutputEngine,
    ) -> Result<(), C220MtePipelineError> {
        self.advance_inner(tick, None, None, Some(engine))
    }

    pub fn advance_external_fixp_with_l1_output(
        &mut self,
        tick: u64,
        fixp: &mut C220FixpRuntime,
        memory: C220FixpRuntimeMemory<'_>,
        mut gates: impl C220FixpSync,
        l1_output: &mut C220L1OutputEngine,
    ) -> Result<(), C220MtePipelineError> {
        self.advance_inner(
            tick,
            None,
            Some((fixp, memory, &mut gates)),
            Some(l1_output),
        )
    }

    pub(super) fn handle_l1_output(
        &mut self,
        index: usize,
        phase: C220FixpCallback,
        engine: &mut C220L1OutputEngine,
    ) -> Result<(), C220MtePipelineError> {
        let binding = self.l1_output_events[index];
        let stage = binding.stage();
        if phase == C220FixpCallback::Probe {
            binding.probe(&mut self.events, engine.stage_ready_tick(stage));
            return Ok(());
        }
        let tick = self.tick();
        let event = match stage {
            C220L1OutputStage::GenerateRead => {
                C220L1OutputEvent::GeneratedRead(engine.generate_read(tick)?)
            }
            C220L1OutputStage::SendRead => C220L1OutputEvent::SentRead(engine.send_read(
                tick,
                &self.fixp_stores,
                &mut self.interface,
                C220MteReadPayload::L1Output,
            )?),
            C220L1OutputStage::Packetize => {
                C220L1OutputEvent::Packetized(engine.packetize(tick, &mut self.fixp_stores)?)
            }
            C220L1OutputStage::GenerateWrite => {
                C220L1OutputEvent::GeneratedWrite(engine.generate_write(tick)?)
            }
            C220L1OutputStage::SendWrite => C220L1OutputEvent::SentWrite(engine.send_write(
                tick,
                self.biu_write_commands.as_mut().expect("bound Cube BIU"),
            )?),
        };
        self.trace.push(C220MtePipelineEvent::L1Output(event));
        if let C220L1OutputEvent::SentWrite(super::super::fixp::C220FixpWriteProgress::Advanced(
            super::super::fixp::C220FixpDispatchPacket::External(packet),
        )) = event
            && packet.write.fragment.last_in_instruction
        {
            self.mte3
                .notify_l1_output_dispatched(packet.write.fragment.instruction_id);
        }
        Ok(())
    }

    pub(super) fn forward_l1_output_source(
        &mut self,
        tick: u64,
        engine: Option<&mut C220L1OutputEngine>,
    ) -> Result<bool, C220MtePipelineError> {
        let Some(head) = self.interface.external_head(tick) else {
            return Ok(false);
        };
        let engine = engine.ok_or(C220MtePipelineError::L1OutputContextMismatch)?;
        if !self
            .l1_output_active
            .contains(&head.operation.instruction_id)
            || !matches!(head.operation.payload, C220MteReadPayload::L1Output(_))
        {
            return Err(C220L1OutputEngineError::ResponseSource.into());
        }
        let accepted = engine.receive_source(tick, &self.fixp_stores, head)?;
        if accepted && head.operation.last_in_instruction {
            self.trace.push(C220MtePipelineEvent::L1Output(
                C220L1OutputEvent::SourceCompleted {
                    tick,
                    instruction_id: head.operation.instruction_id,
                },
            ));
        }
        Ok(accepted)
    }

    pub(super) fn complete_l1_output_responses(
        &mut self,
        engine: &mut C220L1OutputEngine,
    ) -> Result<(), C220MtePipelineError> {
        while let Some(response) = self.l1_output_responses.front().copied() {
            if let Some(instruction_id) = engine.complete_response(response)? {
                if self.mte3.owns_l1_output(instruction_id) {
                    self.mte3.notify_biu_retirement(instruction_id)?;
                }
                self.trace.push(C220MtePipelineEvent::L1Output(
                    C220L1OutputEvent::WriteCompleted {
                        tick: response.tick,
                        instruction_id,
                    },
                ));
            }
            self.l1_output_responses.pop_front();
        }
        Ok(())
    }
}
