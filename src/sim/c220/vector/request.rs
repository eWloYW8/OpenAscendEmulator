use crate::architecture::Architecture;
use crate::sim::c220::state::C220ExecutionError;
use crate::sim::common::scalar::{ScalarMachine, ScalarStepper};

/// CCU register snapshot; Vector-local SPR inputs are resolved at dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220VectorRequest {
    pub pc: u64,
    pub word: u32,
    pub(super) registers: ScalarMachine,
}

impl C220VectorRequest {
    pub fn capture(scalar: &ScalarStepper, word: u32) -> Result<Self, C220ExecutionError> {
        let pc = scalar.pc();
        if scalar.is_halted() {
            return Err(C220ExecutionError::ProgramEnded { pc });
        }
        if scalar.machine().architecture() != Architecture::Dav2201
            || !super::dispatch::is_vector_word(word)
        {
            return Err(C220ExecutionError::UnsupportedWord { pc, word });
        }
        Ok(Self {
            pc,
            word,
            registers: scalar.machine().clone(),
        })
    }

    pub fn registers(&self) -> &ScalarMachine {
        &self.registers
    }

    pub(super) fn context<'a>(
        &'a self,
        live_registers: &'a ScalarMachine,
        ub: &'a crate::memory::ub::UbMemory,
    ) -> super::issue::C220VectorIssueContext<'a> {
        super::issue::C220VectorIssueContext {
            pc: self.pc,
            machine: C220VectorOperands {
                captured: &self.registers,
                live: live_registers,
            },
            ub,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct C220VectorOperands<'a> {
    captured: &'a ScalarMachine,
    live: &'a ScalarMachine,
}

impl C220VectorOperands<'_> {
    pub(super) fn architecture(&self) -> Architecture {
        self.captured.architecture()
    }

    pub(super) fn xregs(&self) -> &[u64; crate::sim::common::scalar::SCALAR_X_REGISTER_COUNT] {
        self.captured.xregs()
    }

    pub(super) fn spr_value(&self, index: u16) -> Option<u64> {
        match index {
            2 | 7 | 12 | 48..=51 | 100 | 101 | 104 | 105 => self.live.spr_value(index),
            _ => self.captured.spr_value(index),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::c220::vector::C220VecArithmeticHint;
    use crate::memory::{sparse::MemoryByteState, ub::UbMemory};
    use crate::sim::c220::state::C220State;
    use crate::sim::c220::vector::{
        C220_CAPTURED_VADD_CONTROL, C220_CAPTURED_VADD_WORD, C220VectorInstruction,
        dispatch::VectorStep, ops::compare::C220CompareMask, pipeline::C220VectorTimingRules,
        runtime::VectorEngine,
    };
    use std::num::NonZeroU64;

    #[test]
    fn delayed_dispatch_uses_captured_gprs_and_control_but_live_vector_sprs_and_ub() {
        let word = C220_CAPTURED_VADD_WORD;
        let hint = C220VecArithmeticHint::from_word(word).unwrap();
        let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
        machine.set_xreg(hint.source_0_register, 0).unwrap();
        machine
            .set_xreg(hint.source_1_register.unwrap(), 32)
            .unwrap();
        machine.set_xreg(hint.destination_register, 64).unwrap();
        machine
            .set_xreg(hint.control_register, C220_CAPTURED_VADD_CONTROL)
            .unwrap();
        machine.set_spr_value(3, 0).unwrap();
        machine.set_spr_value(100, 0).unwrap();
        machine.set_spr_value(101, 0).unwrap();
        let mut state =
            C220State::new(ScalarStepper::new(machine, 0x1000), UbMemory::new(256, 256));
        let request = C220VectorRequest::capture(state.scalar(), word).unwrap();
        state.scalar_mut().advance_sequential();
        let machine = state.scalar_mut().machine_mut();
        machine.set_xreg(hint.destination_register, 128).unwrap();
        machine.set_xreg(hint.control_register, 0).unwrap();
        machine.set_spr_value(100, 1).unwrap();
        machine.set_spr_value(3, 1 << 56).unwrap();
        for index in [2, 7, 12, 48, 49, 50, 51, 104, 105] {
            machine.set_spr_value(index, u64::from(index) + 17).unwrap();
        }
        let inputs = request.context(state.scalar().machine(), state.ub());
        assert_eq!(inputs.machine.spr_value(3), Some(0));
        assert_eq!(inputs.machine.spr_value(100), Some(1));
        for index in [2, 7, 12, 48, 49, 50, 51, 104, 105] {
            assert_eq!(inputs.machine.spr_value(index), Some(u64::from(index) + 17));
        }
        for (address, value) in [(0, 2_f32), (32, 3_f32)] {
            state
                .ub
                .write_states(
                    address,
                    &value
                        .to_le_bytes()
                        .repeat(8)
                        .into_iter()
                        .map(MemoryByteState::Known)
                        .collect::<Vec<_>>(),
                )
                .unwrap();
        }
        let mut engine = VectorEngine::new(
            C220VectorTimingRules {
                dispatch_ticks: 0,
                uop_issue_interval: NonZeroU64::new(1).unwrap(),
                ub_response_ticks: 1,
            },
            C220CompareMask::from_bits([0; 2]),
            Default::default(),
        );
        assert!(matches!(
            engine.dispatch_at(20, 0, &request, &mut state).unwrap(),
            VectorStep::Issued(C220VectorInstruction::Arithmetic(_))
        ));
        assert_eq!(state.scalar().pc(), 0x1004);
        state
            .ub
            .write_states(0, &4_f32.to_le_bytes().map(MemoryByteState::Known))
            .unwrap();
        for tick in 20..100 {
            engine
                .advance_event(tick, &mut state, &mut Default::default())
                .unwrap();
        }
        assert_eq!(state.ub().read_known(64, 4).unwrap(), 7_f32.to_le_bytes());
        assert_eq!(engine.retirements.len(), 1);
        assert_eq!(engine.retirements[0].instruction_id, 0);
        assert_eq!(engine.retirements[0].pc, 0x1000);
        assert_eq!(engine.retirements[0].dispatch_tick, 20);
        assert!(state.ub().read_known(128, 4).is_err());
        assert_eq!(state.scalar().machine().spr_value(100), Some(1));
        assert_eq!(
            state.scalar().machine().xregs()[usize::from(hint.destination_register)],
            128
        );

        state
            .scalar_mut()
            .machine_mut()
            .set_xreg(6, 0xabcdef)
            .unwrap();
        let write = (2 << 24) | (12 << 17) | (6 << 12) | (18 << 7);
        let request = C220VectorRequest::capture(state.scalar(), write).unwrap();
        state.scalar_mut().machine_mut().set_xreg(6, 0).unwrap();
        state
            .scalar_mut()
            .machine_mut()
            .set_spr_value(12, 99)
            .unwrap();
        let VectorStep::Issued(C220VectorInstruction::WriteSpr(step)) =
            engine.dispatch_at(100, 1, &request, &mut state).unwrap()
        else {
            panic!("captured SPR write");
        };
        assert_eq!(step.prior_destination_value, Some(99));
        assert_eq!(step.value, 0xabcdef);
        assert_eq!(state.scalar().machine().spr_value(12), Some(0xabcdef));
        assert_eq!(state.scalar().pc(), 0x1004);
        engine.begin_advance();
        assert!(matches!(
            engine.dispatch_at(101, 7, &request, &mut state).unwrap(),
            VectorStep::Issued(C220VectorInstruction::WriteSpr(_))
        ));
        let fence = engine.instruction_fence();
        assert_eq!(fence.instruction_id, Some(7));
        engine
            .advance_event(102, &mut state, &mut Default::default())
            .unwrap();
        assert_eq!(engine.retirements.len(), 1);
        assert_eq!(engine.retirements[0].instruction_id, 1);
        assert_eq!(engine.retirements[0].retirement_tick, 102);
        assert_eq!(engine.fence_retirement_tick(fence), Some(103));
        engine
            .advance_event(103, &mut state, &mut Default::default())
            .unwrap();
        assert_eq!(engine.retirements[1].instruction_id, 7);
        assert_eq!(engine.retirements[1].retirement_tick, 103);
        assert_eq!(engine.fence_retirement_tick(fence), None);

        state
            .scalar_mut()
            .machine_mut()
            .set_spr_value(17, 57)
            .unwrap();
        let empty = C220VectorRequest::capture(state.scalar(), 0x85c0_0003).unwrap();
        assert!(matches!(
            engine.dispatch_at(104, 8, &empty, &mut state),
            Err(super::super::runtime::C220VectorRuntimeError::Merge(
                super::super::vmsu::C220VmsuError::UnsupportedEmptySchedule
            ))
        ));
        assert_eq!(state.scalar().machine().spr_value(17), Some(57));
        assert_eq!(state.scalar().pc(), 0x1004);
        assert_eq!(engine.instruction_fence().instruction_id, None);
    }
}
