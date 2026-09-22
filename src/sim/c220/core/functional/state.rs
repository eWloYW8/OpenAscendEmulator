use crate::sim::c220::mte::transfer::C220Mte2TransferPlan;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct C220OutputToken {
    pub source_address: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct C220ExecutionState {
    pub isa_instance_index: u32,
    pub vector_flags: [Option<C220Mte2TransferPlan>; 2],
    pub unsignaled_output: Option<C220OutputToken>,
    pub mte3_flags: [Option<C220OutputToken>; 4],
    pub mte2_reuse_flags: [bool; 2],
    pub ready_output: Option<C220OutputToken>,
    pub copied_output: Option<C220OutputToken>,
    pub mte3_completion_flags: [Option<C220OutputToken>; 4],
}

impl C220ExecutionState {
    pub fn barrier_busy(&self) -> bool {
        self.vector_flags.iter().any(Option::is_some)
            || self.output_buffer_busy()
            || self.mte2_reuse_flags.iter().any(|set| *set)
    }

    pub fn output_buffer_busy(&self) -> bool {
        self.unsignaled_output.is_some()
            || self.mte3_flags.iter().any(Option::is_some)
            || self.ready_output.is_some()
            || self.copied_output.is_some()
            || self.mte3_completion_flags.iter().any(Option::is_some)
    }

    pub const fn vector_flags_set(&self) -> [bool; 2] {
        [
            self.vector_flags[0].is_some(),
            self.vector_flags[1].is_some(),
        ]
    }

    pub const fn output_flags_set(&self) -> [bool; 4] {
        [
            self.mte3_flags[0].is_some(),
            self.mte3_flags[1].is_some(),
            self.mte3_flags[2].is_some(),
            self.mte3_flags[3].is_some(),
        ]
    }

    pub const fn completion_flags_set(&self) -> [bool; 4] {
        [
            self.mte3_completion_flags[0].is_some(),
            self.mte3_completion_flags[1].is_some(),
            self.mte3_completion_flags[2].is_some(),
            self.mte3_completion_flags[3].is_some(),
        ]
    }
}
