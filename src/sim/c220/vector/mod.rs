mod access;
pub(super) mod dispatch;
mod error;
mod instruction;
mod issue;
mod lanes;
mod mask;
pub mod ops;
pub mod pipeline;
pub mod read;
mod repeat;
pub(super) mod runtime;
mod spr;
pub mod timing;
mod uop;
pub mod va;
pub mod vmsu;

pub use instruction::C220VectorInstruction;
pub use runtime::{C220VectorFence, C220VectorRuntimeError};

use crate::isa::c220::vector::C220VectorControl;
pub use access::{
    C220VectorAddresses, C220VectorReadAccess, C220VectorStore,
    plan_c220_vector_arithmetic_read_accesses,
};
use access::{
    plan_c220_destination_read_accesses, plan_c220_unary_write_targets,
    plan_c220_vector_read_accesses, store_data, vector_destination_address,
    vector_destination_address_for_width,
};
pub use error::{C220VectorError, C220VectorUopError};
use lanes::fp32::evaluate_c220_fp32_repeat_from_bytes;
pub use lanes::fp32::{C220Fp32Step, evaluate_c220_fp32_lanes, execute_c220_fp32_to_ub};
pub use mask::{C220VectorMaskState, decode_c220_fp32_mask};
use mask::{check_repeat_limit, decode_c220_repeat_masks};
pub use ops::arithmetic::{
    C220VectorArithmeticIssue, C220VectorArithmeticModes, plan_c220_vector_arithmetic_issue,
};

use ops::movev::plan_c220_movev_to_ub;
pub use ops::movev::{C220MovevStep, execute_c220_movev_to_ub};

pub(crate) const C220_VECTOR_BLOCK_BYTES: usize = 32;
pub(crate) const C220_VECTOR_BLOCK_COUNT: usize = 8;
pub(crate) const C220_VECTOR_TILE_BYTES: usize = C220_VECTOR_BLOCK_BYTES * C220_VECTOR_BLOCK_COUNT;
const C220_VECTOR32_LANES: usize = C220_VECTOR_TILE_BYTES / 4;

#[cfg(test)]
mod test_words {
    pub const C220_CAPTURED_MOVEV_WORD: u32 = 0x82a0_6014;
    pub const C220_CAPTURED_MOVEV_CONTROL: u64 = 0x0100_0008_0001_0001;
    pub const C220_CAPTURED_VADD_WORD: u32 = 0x85e0_d720;
    pub const C220_CAPTURED_VADD_CONTROL: u64 = 0x0100_0808_0801_0101;
    pub const C220_CAPTURED_VSUB_WORD: u32 = 0x85dc_b619;
    pub const C220_CAPTURED_VMUL_WORD: u32 = 0x89dc_b618;
    pub const C220_CAPTURED_VMUL_CONTROL: u64 = C220_CAPTURED_VADD_CONTROL;
}

#[cfg(test)]
pub use test_words::*;

#[cfg(test)]
mod tests;
