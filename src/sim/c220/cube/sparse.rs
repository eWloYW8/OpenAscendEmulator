use crate::isa::c220::cube::C220MmadParameters;
use crate::sim::c220::memory::{C220LocalBuffer, C220LocalMemory};

use super::C220CubeExecutionError;
use super::layout::{integer_a_element, integer_b_element, read_input};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220SparseByteLane {
    pub index_address: u64,
    pub selector: u8,
    pub dense_k: u64,
}

pub fn select_byte_lane(
    parameters: C220MmadParameters,
    indices: &C220LocalBuffer,
    column: u64,
    compressed_k: u64,
) -> Result<C220SparseByteLane, C220CubeExecutionError> {
    let columns = u64::from(parameters.n.div_ceil(16)) * 16;
    let slice = compressed_k / 32;
    let lane = compressed_k % 32;
    let index_address =
        u64::from((parameters.xm >> 2) as u32) + 8 * (column + slice * columns) + lane / 4;
    let byte = indices.read_initialized_linear(index_address, 1)?[0];
    let selector = (byte >> (2 * (lane & 3))) & 3;
    Ok(C220SparseByteLane {
        index_address,
        selector,
        dense_k: 2 * compressed_k - (lane & 1) + u64::from(selector),
    })
}

pub(super) fn read_byte_pair(
    parameters: C220MmadParameters,
    memory: &C220LocalMemory,
    row: u64,
    column: u64,
    compressed_k: u64,
) -> Result<(u8, u8), C220CubeExecutionError> {
    let selection = select_byte_lane(parameters, memory.weight_index(), column, compressed_k)?;
    let k_tiles = u64::from(parameters.effective_k.div_ceil(32));
    let partial = parameters.effective_k & 31;
    let a_tiles = 2 * k_tiles - u64::from((1..=16).contains(&partial));
    if selection.dense_k >= a_tiles * 32 {
        return Err(C220CubeExecutionError::SparseInputOutsideLoadedTiles {
            dense_k: selection.dense_k,
            loaded_k: a_tiles * 32,
        });
    }
    let a = integer_a_element(a_tiles, 32, row, selection.dense_k);
    let b = integer_b_element(
        u64::from(parameters.n.div_ceil(16)),
        32,
        compressed_k,
        column,
    );
    Ok((
        read_input::<1>(memory.l0a(), parameters.xn, a)?[0],
        read_input::<1>(memory.l0b(), parameters.xm, b)?[0],
    ))
}
