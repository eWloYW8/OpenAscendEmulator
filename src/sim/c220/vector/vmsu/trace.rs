use crate::sim::c220::memory::C220UbCycle;
use crate::sim::c220::vector::ops::merge::{C220MergeIssue, C220MergeResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220VmsuReadBatchTrace {
    pub repeat_index: usize,
    pub list: u8,
    pub first_record: u16,
    pub records: u8,
    pub generation_tick: u64,
    pub grant_tick: u64,
    pub ready_tick: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220VmsuCompareTrace {
    pub repeat_index: usize,
    pub output_index: usize,
    pub source_list: u8,
    pub source_index: u16,
    pub cmp0_tick: u64,
    pub cmp1_tick: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220VmsuWriteGroupTrace {
    pub repeat_index: usize,
    pub first_output: usize,
    pub records: u8,
    pub destination: u64,
    pub request_bytes: u32,
    pub creation_tick: u64,
    pub submission_tick: u64,
    pub lane_departure_tick: u64,
    pub request_completion_tick: u64,
    pub done_tick: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220VmsuRepeatTrace {
    pub repeat_index: usize,
    pub start_tick: u64,
    pub completion_tick: Option<u64>,
    pub consumed: [u16; 4],
    pub read_batches: Vec<C220VmsuReadBatchTrace>,
    pub comparisons: Vec<C220VmsuCompareTrace>,
    pub write_groups: Vec<C220VmsuWriteGroupTrace>,
    pub ub_cycles: Vec<C220UbCycle>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220VmsuTrace {
    pub issue: C220MergeIssue,
    pub admission_tick: u64,
    pub repeats: Vec<C220VmsuRepeatTrace>,
    pub functional_tick: Option<u64>,
    pub result: Option<C220MergeResult>,
    pub retirement_tick: Option<u64>,
}
