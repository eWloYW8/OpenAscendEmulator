mod error;
use crate::memory::ub::UbMemory;
use crate::sim::common::scalar::ScalarStepper;
pub use error::C220ExecutionError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220State {
    pub(crate) scalar: ScalarStepper,
    pub(crate) ub: UbMemory,
    pub(crate) isa_instance_index: u32,
}

impl C220State {
    pub fn new(scalar: ScalarStepper, ub: UbMemory) -> Self {
        Self {
            scalar,
            ub,
            isa_instance_index: 0,
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
}
