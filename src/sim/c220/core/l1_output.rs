use super::{C220Core, C220CoreError, C220CoreStep, C220Mte3Operation};
use crate::isa::c220::mte::l1_to_out::C220MovL1ToOutInstruction;
use crate::sim::c220::mte::interface::biu_write::command::C220BiuWriteConfig;
use crate::sim::c220::mte::l1_to_out::{
    C220L1OutputCommand, C220L1OutputEngine, C220L1OutputEngineConfig, C220L1OutputStage,
};
use crate::sim::c220::mte::mte3::frontend::C220Mte3Command;
use crate::sim::c220::mte::uop::C220DmaUopMode;
use crate::sim::c220::state::C220ExecutionError;

impl C220Core {
    /// Connect a shared Cube output route when no FIX runtime owns one yet.
    pub fn connect_cube_output_biu(
        &mut self,
        config: C220BiuWriteConfig,
    ) -> Result<(), C220CoreError> {
        let pipeline = self
            .mte_pipeline
            .as_mut()
            .ok_or(C220CoreError::MteUnconfigured)?;
        if pipeline.core_kind() != crate::sim::c220::device::C220CoreKind::Cube {
            return Err(C220CoreError::L1OutputUnconfigured);
        }
        pipeline.connect_fixp_biu(config)?;
        Ok(())
    }

    /// Requires the shared Cube BIU route. Stage order and queue limits are
    /// supplied explicitly, independently from the L0C/FIX engine settings.
    pub fn configure_l1_output(
        &mut self,
        config: C220L1OutputEngineConfig,
        stages: &[C220L1OutputStage],
    ) -> Result<(), C220CoreError> {
        if self.fixp.is_some() || self.l1_output.is_some() || self.mte3_is_busy() {
            return Err(C220CoreError::MtePipelineBusy);
        }
        self.mte_pipeline
            .as_mut()
            .ok_or(C220CoreError::MteUnconfigured)?
            .bind_l1_output_stages(stages)?;
        self.l1_output = Some(C220L1OutputEngine::new(config));
        self.mte3.physical = true;
        Ok(())
    }

    pub fn l1_output_engine(&self) -> Option<&C220L1OutputEngine> {
        self.l1_output.as_ref()
    }

    pub(super) fn enqueue_l1_output_at(
        &mut self,
        tick: u64,
        pc: u64,
        instruction: C220MovL1ToOutInstruction,
    ) -> Result<C220CoreStep, C220CoreError> {
        if self.l1_output.is_none() {
            return Err(C220CoreError::L1OutputUnconfigured);
        }
        let machine = self.state.scalar().machine();
        let spr = |index| {
            machine
                .spr_value(index)
                .ok_or(C220ExecutionError::MissingSpr { pc, index })
        };
        let command = C220L1OutputCommand {
            transfer: instruction.capture(machine.xregs()),
            control: spr(3)?,
            mode: C220DmaUopMode::from_mode_word(if self.state.isa_instance_index == 0 {
                0
            } else {
                spr(94)?
            }),
        };
        self.enqueue_mte3_at(
            tick,
            pc,
            instruction.word,
            C220Mte3Operation::Command(C220Mte3Command::L1Output(command)),
        )
    }
}
