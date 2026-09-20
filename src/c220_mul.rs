use serde::Serialize;
use thiserror::Error;

use crate::replay_memory::MemoryByteState;
use crate::{
    C220_CAPTURED_VMUL_WORD, C220CapturedFp32Step, C220CapturedVectorError, UbReplayError,
    UbReplayMemory, execute_captured_c220_fp32_to_ub,
};

pub const C220_CAPTURED_MUL_TILE_BYTES: usize = 128;
pub const C220_CAPTURED_MUL_TILES: usize = 32;
pub const C220_CAPTURED_MUL_BYTES: usize = C220_CAPTURED_MUL_TILE_BYTES * C220_CAPTURED_MUL_TILES;
pub const C220_CAPTURED_MUL_VMUL_PC: u64 = 0x1131_2628;

const SOURCE_X_LOCAL: u64 = 0;
const SOURCE_Y_LOCAL: u64 = 0x80;
const DESTINATION_LOCAL: u64 = 0x100;
const LOCAL_BYTES: usize = 384;
const ACTIVE_MASK: [u64; 4] = [u32::MAX as u64, 0, 0, 0];

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct C220CapturedMulTile {
    pub group: usize,
    pub input_offset: u64,
    pub vmul: C220CapturedFp32Step,
    pub output: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct C220CapturedMulRun {
    pub tiles: Vec<C220CapturedMulTile>,
    pub output: Vec<u8>,
}

#[derive(Debug, Error)]
pub enum C220CapturedMulError {
    #[error(
        "captured C220 Mul expects three {C220_CAPTURED_MUL_BYTES}-byte tensors; got X={x}, Y={y}, prior={prior}"
    )]
    TensorLengths { x: usize, y: usize, prior: usize },
    #[error(
        "captured C220 Mul expects two {C220_CAPTURED_MUL_BYTES}-byte inputs; got X={x}, Y={y}"
    )]
    InputLengths { x: usize, y: usize },
    #[error(transparent)]
    Vector(#[from] C220CapturedVectorError),
    #[error(transparent)]
    LocalMemory(#[from] UbReplayError),
}

fn known(bytes: &[u8]) -> Vec<MemoryByteState> {
    bytes.iter().copied().map(MemoryByteState::Known).collect()
}

fn execute_tile(
    group: usize,
    x: &[u8],
    y: &[u8],
    prior_destination: &[u8],
) -> Result<C220CapturedMulTile, C220CapturedMulError> {
    let mut local = UbReplayMemory::new(LOCAL_BYTES, LOCAL_BYTES);
    local.write_states(SOURCE_X_LOCAL, &known(x))?;
    local.write_states(SOURCE_Y_LOCAL, &known(y))?;
    local.write_states(DESTINATION_LOCAL, &known(prior_destination))?;
    let vmul = execute_captured_c220_fp32_to_ub(
        C220_CAPTURED_MUL_VMUL_PC,
        C220_CAPTURED_VMUL_WORD,
        SOURCE_X_LOCAL,
        SOURCE_Y_LOCAL,
        DESTINATION_LOCAL,
        &ACTIVE_MASK,
        &mut local,
    )?;
    let output = local.read_known(DESTINATION_LOCAL, C220_CAPTURED_MUL_TILE_BYTES)?;
    Ok(C220CapturedMulTile {
        group,
        input_offset: (group * C220_CAPTURED_MUL_TILE_BYTES) as u64,
        vmul,
        output,
    })
}

pub fn execute_captured_c220_mul(
    x: &[u8],
    y: &[u8],
    prior_destination: &[u8],
) -> Result<C220CapturedMulRun, C220CapturedMulError> {
    if x.len() != C220_CAPTURED_MUL_BYTES
        || y.len() != C220_CAPTURED_MUL_BYTES
        || prior_destination.len() != C220_CAPTURED_MUL_BYTES
    {
        return Err(C220CapturedMulError::TensorLengths {
            x: x.len(),
            y: y.len(),
            prior: prior_destination.len(),
        });
    }
    execute_tiles(x, y, |range, _| prior_destination[range].to_vec())
}

pub fn execute_captured_c220_mul_predecessor_chains(
    x: &[u8],
    y: &[u8],
) -> Result<C220CapturedMulRun, C220CapturedMulError> {
    if x.len() != C220_CAPTURED_MUL_BYTES || y.len() != C220_CAPTURED_MUL_BYTES {
        return Err(C220CapturedMulError::InputLengths {
            x: x.len(),
            y: y.len(),
        });
    }
    execute_tiles(x, y, |_, prior| prior.to_vec())
}

fn execute_tiles(
    x: &[u8],
    y: &[u8],
    mut prior_for: impl FnMut(std::ops::Range<usize>, &[u8]) -> Vec<u8>,
) -> Result<C220CapturedMulRun, C220CapturedMulError> {
    let mut tiles = Vec::with_capacity(C220_CAPTURED_MUL_TILES);
    let mut output = vec![0; C220_CAPTURED_MUL_BYTES];
    let zero_prior = [0; C220_CAPTURED_MUL_TILE_BYTES];
    for group in 0..C220_CAPTURED_MUL_TILES {
        let range =
            group * C220_CAPTURED_MUL_TILE_BYTES..(group + 1) * C220_CAPTURED_MUL_TILE_BYTES;
        let predecessor = if group.is_multiple_of(4) {
            &zero_prior[..]
        } else {
            &output
                [(group - 1) * C220_CAPTURED_MUL_TILE_BYTES..group * C220_CAPTURED_MUL_TILE_BYTES]
        };
        let prior = prior_for(range.clone(), predecessor);
        let tile = execute_tile(group, &x[range.clone()], &y[range.clone()], &prior)?;
        output[range].copy_from_slice(&tile.output);
        tiles.push(tile);
    }
    Ok(C220CapturedMulRun { tiles, output })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Fp32VectorOperation, evaluate_fp32_value};

    #[test]
    fn multiply_tile_exposes_overlapping_sources_and_exact_fp32_results() {
        let mut x = (0..32_u32)
            .flat_map(|lane| (lane as f32).to_le_bytes())
            .collect::<Vec<_>>();
        let mut y = (0..32)
            .flat_map(|_| 0.5_f32.to_le_bytes())
            .collect::<Vec<_>>();
        x[..4].copy_from_slice(&0_u32.to_le_bytes());
        y[..4].copy_from_slice(&f32::INFINITY.to_bits().to_le_bytes());
        x[4..8].copy_from_slice(&(-0.0_f32).to_bits().to_le_bytes());
        y[4..8].copy_from_slice(&2.0_f32.to_bits().to_le_bytes());
        let prior = vec![0x5a; C220_CAPTURED_MUL_TILE_BYTES];
        let tile = execute_tile(0, &x, &y, &prior).unwrap();
        assert_eq!(tile.vmul.source_0_bytes[..128], x);
        assert_eq!(tile.vmul.source_0_bytes[128..], y);
        assert_eq!(tile.vmul.source_1_bytes[..128], y);
        assert_eq!(tile.vmul.source_1_bytes[128..], prior);
        assert_eq!(tile.vmul.stores.len(), 32);
        for lane in 0..32 {
            let offset = lane * 4;
            let first = u32::from_le_bytes(x[offset..offset + 4].try_into().unwrap());
            let second = u32::from_le_bytes(y[offset..offset + 4].try_into().unwrap());
            let expected = evaluate_fp32_value(Fp32VectorOperation::Multiply, first, second).bits;
            assert_eq!(tile.output[offset..offset + 4], expected.to_le_bytes());
        }
    }

    #[test]
    fn predecessor_state_and_tensor_lengths_are_bounded() {
        let x = vec![0; C220_CAPTURED_MUL_BYTES];
        let y = vec![0; C220_CAPTURED_MUL_BYTES];
        let run = execute_captured_c220_mul_predecessor_chains(&x, &y).unwrap();
        assert_eq!(run.tiles.len(), C220_CAPTURED_MUL_TILES);
        assert_eq!(run.tiles[0].vmul.source_1_bytes[128..], [0; 128]);
        assert_eq!(run.tiles[1].vmul.source_1_bytes[128..], run.tiles[0].output);
        assert_eq!(run.tiles[4].vmul.source_1_bytes[128..], [0; 128]);
        assert!(execute_captured_c220_mul_predecessor_chains(&x[..127], &y).is_err());
        assert!(execute_captured_c220_mul(&x, &y, &y[..127]).is_err());
    }
}
