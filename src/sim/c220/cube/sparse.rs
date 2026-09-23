use crate::isa::c220::cube::C220MmadParameters;
use crate::sim::c220::memory::{C220LocalBuffer, C220LocalMemory};

use super::C220CubeExecutionError;
use super::layout::{integer_a_element, integer_b_element, read_input};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220SparseLane {
    pub index_address: u64,
    pub selector: u8,
    pub dense_k: u64,
}

pub fn select_byte_lane(
    parameters: C220MmadParameters,
    indices: &C220LocalBuffer,
    column: u64,
    compressed_k: u64,
) -> Result<C220SparseLane, C220CubeExecutionError> {
    select_lane(parameters, indices, column, compressed_k, 32)
}

pub fn select_half_lane(
    parameters: C220MmadParameters,
    indices: &C220LocalBuffer,
    column: u64,
    compressed_k: u64,
) -> Result<C220SparseLane, C220CubeExecutionError> {
    select_lane(parameters, indices, column, compressed_k, 16)
}

pub fn select_nibble_lane(
    parameters: C220MmadParameters,
    indices: &C220LocalBuffer,
    column: u64,
    compressed_k: u64,
) -> Result<C220SparseLane, C220CubeExecutionError> {
    select_lane(parameters, indices, column, compressed_k, 64)
}

fn select_lane(
    parameters: C220MmadParameters,
    indices: &C220LocalBuffer,
    column: u64,
    compressed_k: u64,
    width: u64,
) -> Result<C220SparseLane, C220CubeExecutionError> {
    let columns = u64::from(parameters.n.div_ceil(16)) * 16;
    let slice = compressed_k / width;
    let lane = compressed_k % width;
    let index_address =
        u64::from((parameters.xm >> 2) as u32) + 8 * (column + slice * columns) + lane / 4;
    let byte = indices.read_initialized_linear(index_address, 1)?[0];
    let selector = (byte >> (2 * (lane & 3))) & 3;
    Ok(C220SparseLane {
        index_address,
        selector,
        dense_k: 2 * compressed_k - (lane & 1) + u64::from(selector),
    })
}

pub(super) fn read_half_pair(
    parameters: C220MmadParameters,
    memory: &C220LocalMemory,
    row: u64,
    column: u64,
    compressed_k: u64,
) -> Result<(u16, u16), C220CubeExecutionError> {
    let selection = select_half_lane(parameters, memory.weight_index(), column, compressed_k)?;
    let k_tiles = u64::from(parameters.effective_k.div_ceil(16));
    let left = read_a_half(parameters, memory, row, selection, k_tiles, false)?;
    let right = u16::from_le_bytes(read_input(
        memory.l0b(),
        parameters.xm,
        2 * integer_b_element(
            u64::from(parameters.n.div_ceil(16)),
            16,
            compressed_k,
            column,
        ),
    )?);
    Ok((left, right))
}

pub(super) fn read_word_pair(
    parameters: C220MmadParameters,
    memory: &C220LocalMemory,
    row: u64,
    column: u64,
    compressed_k: u64,
) -> Result<(u32, u32), C220CubeExecutionError> {
    let k_tiles = u64::from(parameters.effective_k.div_ceil(8));
    let mut left = 0;
    for half in 0..2 {
        let selection = select_half_lane(
            parameters,
            memory.weight_index(),
            column,
            2 * compressed_k + half,
        )?;
        let bits = read_a_half(
            parameters,
            memory,
            row,
            selection,
            k_tiles,
            parameters.xt_bit_58,
        )?;
        left |= u32::from(bits) << (16 * half);
    }
    let right = u32::from_le_bytes(read_input(
        memory.l0b(),
        parameters.xm,
        4 * integer_b_element(
            u64::from(parameters.n.div_ceil(16)),
            8,
            compressed_k,
            column,
        ),
    )?);
    Ok((left, right))
}

fn read_a_half(
    parameters: C220MmadParameters,
    memory: &C220LocalMemory,
    row: u64,
    selection: C220SparseLane,
    k_tiles: u64,
    padded: bool,
) -> Result<u16, C220CubeExecutionError> {
    let a_tiles = 2 * k_tiles - u64::from((1..=16).contains(&(parameters.effective_k & 31)));
    Ok(if selection.dense_k < a_tiles * 16 {
        let mut offset = 2 * integer_a_element(a_tiles, 16, row, selection.dense_k);
        if padded {
            let tile = offset / 512;
            let extra = 2 * u64::from(parameters.effective_k.div_ceil(16)) - k_tiles;
            offset += (tile / k_tiles) * extra * 512;
        }
        u16::from_le_bytes(read_input(memory.l0a(), parameters.xn, offset)?)
    } else if selection.dense_k < 2 * k_tiles * 16 && row < u64::from(parameters.m.div_ceil(16)) * 8
    {
        0
    } else {
        return Err(C220CubeExecutionError::SparseInputOutsideLoadedTiles {
            dense_k: selection.dense_k,
            loaded_k: a_tiles * 16,
        });
    })
}

pub(super) fn read_nibble_pair(
    parameters: C220MmadParameters,
    memory: &C220LocalMemory,
    row: u64,
    column: u64,
    compressed_k: u64,
) -> Result<(u8, u8), C220CubeExecutionError> {
    let selection = select_nibble_lane(parameters, memory.weight_index(), column, compressed_k)?;
    let initialized_rows = u64::from(parameters.m.div_ceil(16)) * 8;
    if row >= initialized_rows {
        return Err(C220CubeExecutionError::SparseUninitializedRow {
            row,
            initialized_rows,
        });
    }
    let k_tiles = u64::from(parameters.effective_k.div_ceil(64));
    let a_tiles = 2 * k_tiles - u64::from((1..=16).contains(&(parameters.effective_k & 31)));
    let left = if selection.dense_k < a_tiles * 64 {
        read_nibble(
            memory.l0a(),
            parameters.xn,
            integer_a_element(a_tiles, 64, row, selection.dense_k),
        )?
    } else if selection.dense_k < 2 * k_tiles * 64 {
        0
    } else {
        return Err(C220CubeExecutionError::SparseInputOutsideLoadedTiles {
            dense_k: selection.dense_k,
            loaded_k: a_tiles * 64,
        });
    };
    let right = read_nibble(
        memory.l0b(),
        parameters.xm,
        integer_b_element(
            u64::from(parameters.n.div_ceil(16)),
            64,
            compressed_k,
            column,
        ),
    )?;
    Ok((left, right))
}

fn read_nibble(
    buffer: &C220LocalBuffer,
    base: u64,
    element: u64,
) -> Result<u8, C220CubeExecutionError> {
    let [byte] = read_input(buffer, base, element / 2)?;
    let nibble = (byte >> (4 * (element & 1))) & 15;
    Ok(((nibble << 4) as i8 >> 4) as u8)
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
