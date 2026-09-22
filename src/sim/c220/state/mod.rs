mod error;
use crate::memory::ub::UbMemory;
use crate::sim::c220::mte::mte3::C220Mte3State;
use crate::sim::common::scalar::ScalarStepper;
pub use error::C220ExecutionError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220State {
    pub(crate) scalar: ScalarStepper,
    pub(crate) ub: UbMemory,
    pub(crate) isa_instance_index: u32,
    pub(crate) output: C220Mte3State,
}

impl C220State {
    pub fn new(scalar: ScalarStepper, ub: UbMemory) -> Self {
        Self {
            scalar,
            ub,
            isa_instance_index: 0,
            output: C220Mte3State::default(),
        }
    }

    pub const fn scalar(&self) -> &ScalarStepper {
        &self.scalar
    }

    pub fn set_isa_instance_index(&mut self, index: u32) {
        self.isa_instance_index = index;
    }

    pub fn scalar_mut(&mut self) -> &mut ScalarStepper {
        &mut self.scalar
    }

    pub const fn ub(&self) -> &UbMemory {
        &self.ub
    }

    pub(crate) fn ub_mut(&mut self) -> &mut UbMemory {
        &mut self.ub
    }

    pub const fn output_flags_set(&self) -> [bool; 4] {
        self.output.output_flags_set()
    }

    pub const fn reuse_flags_set(&self) -> [bool; 2] {
        self.output.reuse_flags_set()
    }

    pub const fn completion_flags_set(&self) -> [bool; 4] {
        self.output.completion_flags_set()
    }
}
