use crate::isa::c220::scalar::C220ScalarConversionHint;
use crate::sim::c220::scalar::bus::{C220ScalarBus, C220ScalarBusError};
use crate::sim::c220::state::C220State;
use crate::sim::common::scalar::ScalarProgramStep;
use crate::sim::common::scalar::{ScalarInstructionError, ScalarInstructionStep, ScalarMemoryBus};

impl C220State {
    pub(crate) fn step_scalar_word_with_ub<B: ScalarMemoryBus>(
        &mut self,
        word: u32,
        fallback: &mut B,
    ) -> Result<ScalarProgramStep, ScalarInstructionError<C220ScalarBusError<B::Error>>> {
        if super::float::supports(word) {
            let pc = self.scalar.pc();
            let result = super::execute_fp32_word(self.scalar.machine_mut(), pc, word)?;
            self.scalar.advance_sequential();
            return Ok(ScalarProgramStep {
                pc,
                word,
                next_pc: self.scalar.pc(),
                instruction: ScalarInstructionStep::Register(result.step),
                halted_after: false,
            });
        }
        if super::spr::write_destination(word).is_some() {
            let pc = self.scalar.pc();
            let step = super::spr::execute_write(self.scalar.machine_mut(), pc, word)?;
            self.scalar.advance_sequential();
            return Ok(ScalarProgramStep {
                pc,
                word,
                next_pc: self.scalar.pc(),
                instruction: ScalarInstructionStep::SprWrite(step),
                halted_after: false,
            });
        }
        if C220ScalarConversionHint::from_word(word).is_some() {
            let pc = self.scalar.pc();
            let step = crate::sim::c220::scalar::execute_conversion_word(
                self.scalar.machine_mut(),
                pc,
                word,
            )?;
            self.scalar.advance_sequential();
            return Ok(ScalarProgramStep {
                pc,
                word,
                next_pc: self.scalar.pc(),
                instruction: ScalarInstructionStep::Register(step),
                halted_after: false,
            });
        }
        let machine = self.scalar.machine();
        let mut bus = C220ScalarBus::new(
            &mut self.ub,
            fallback,
            machine.spr_value(67).zip(machine.spr_value(68)),
        );
        self.scalar.step_word(word, &mut bus)
    }

    pub(crate) fn commit_c220_sequential_issue(&mut self) {
        self.scalar.advance_sequential();
    }
}
