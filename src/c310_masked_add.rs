use serde::Serialize;
use thiserror::Error;

use crate::{
    Architecture, C310ObservedMovemaskError, C310RvecMaskSprState, C310RvecValueError,
    C310RvecValueMachine, C310RvecVstiStore, PvMemory, PvMemoryError, ScalarMachine,
    ScalarMachineError,
};

pub const C310_CAPTURED_MASKED_ADD_TILE_BYTES: usize = 128;
pub const C310_CAPTURED_MASKED_ADD_TILES: usize = 32;
pub const C310_CAPTURED_MASKED_ADD_BYTES: usize =
    C310_CAPTURED_MASKED_ADD_TILE_BYTES * C310_CAPTURED_MASKED_ADD_TILES;
const LOCAL_DESTINATION: u64 = 0x100;
const LOCAL_BYTES: usize = 0x180;
const VLDI_BYTES: usize = 256;
const ACTIVE_STORES_PER_TILE: usize = 16;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct C310CapturedMaskedAddTile {
    pub group: usize,
    pub vldi_v0: Vec<u8>,
    pub vldi_v1: Vec<u8>,
    pub vadd_source_4: Vec<u8>,
    pub vadd_source_6: Vec<u8>,
    pub movp_p1: Vec<u8>,
    pub vadd_v0: Vec<u8>,
    pub vsti_stores: Vec<C310RvecVstiStore>,
    pub output: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct C310CapturedMaskedAddRun {
    pub tiles: Vec<C310CapturedMaskedAddTile>,
    pub output: Vec<u8>,
}

#[derive(Debug, Error)]
pub enum C310CapturedMaskedAddError {
    #[error(
        "captured C310 masked-Add expects three {C310_CAPTURED_MASKED_ADD_BYTES}-byte tensors; got X={x}, Y={y}, prior={prior}"
    )]
    TensorLengths { x: usize, y: usize, prior: usize },
    #[error("local byte at {offset:#x} was not initialized before VLDI")]
    UninitializedLocal { offset: usize },
    #[error("captured VSTI produced {actual} stores, expected {ACTIVE_STORES_PER_TILE}")]
    StoreCount { actual: usize },
    #[error("captured VSTI store {index} has unexpected address {actual:#x}")]
    StoreAddress { index: usize, actual: u64 },
    #[error(transparent)]
    Memory(#[from] PvMemoryError),
    #[error(transparent)]
    Scalar(#[from] ScalarMachineError),
    #[error(transparent)]
    Mask(#[from] C310ObservedMovemaskError),
    #[error(transparent)]
    Rvec(#[from] C310RvecValueError),
}

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().expect("four bytes")))
        .collect()
}

fn bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

fn execute_tile(
    group: usize,
    x: &[u8],
    y: &[u8],
    prior_destination: &[u8],
) -> Result<C310CapturedMaskedAddTile, C310CapturedMaskedAddError> {
    let mut local = PvMemory::new(Architecture::Dav3510, 0, 1);
    local.write(LOCAL_DESTINATION, prior_destination)?;
    local.write(0, x)?;
    local.write(C310_CAPTURED_MASKED_ADD_TILE_BYTES as u64, y)?;
    for offset in 0..LOCAL_BYTES {
        if local.dirty_byte(offset as u64) != Some(1) {
            return Err(C310CapturedMaskedAddError::UninitializedLocal { offset });
        }
    }

    let mut vldi_v0 = vec![0_u8; VLDI_BYTES];
    let mut vldi_v1 = vec![0_u8; VLDI_BYTES];
    local.read_into(0, &mut vldi_v0)?;
    local.read_into(C310_CAPTURED_MASKED_ADD_TILE_BYTES as u64, &mut vldi_v1)?;

    let mut scalar = ScalarMachine::new(Architecture::Dav3510, [0; 32], 0);
    let mut mask_sprs = C310RvecMaskSprState::default();
    scalar.execute_word(0x10d0_d548, 0x071a_5555)?;
    mask_sprs.execute_observed_movemask_word(0x10d0_d54c, 0x15c3_0033, scalar.xregs())?;
    scalar.execute_word(0x10d0_d550, 0x075b_5555)?;
    mask_sprs.execute_observed_movemask_word(0x10d0_d55c, 0x15cd_0013, scalar.xregs())?;
    let mut rvec = C310RvecValueMachine::from_vector_and_predicate_bytes(
        vec![words(&vldi_v0), words(&vldi_v1)],
        vec![vec![0; 32], vec![0; 32]],
    )?;
    rvec.execute_movp_u32_from_mask_sprs(0x8204_0156, &mask_sprs)?;
    let movp_p1 = rvec
        .predicate_register(1)
        .expect("two predicate registers initialized")
        .to_vec();
    let step = rvec.execute_fp32_word_from_predicate_registers(0x8008_2780)?;
    let vadd_source_4 = bytes(&step.first_source);
    let vadd_source_6 = bytes(&step.second_source);
    let vadd_v0 = bytes(rvec.vector_register(0).expect("V0 initialized"));

    let vsti_stores = rvec.plan_normal_u32_vsti(0x4018_010a, LOCAL_DESTINATION)?;
    if vsti_stores.len() != ACTIVE_STORES_PER_TILE {
        return Err(C310CapturedMaskedAddError::StoreCount {
            actual: vsti_stores.len(),
        });
    }
    for (index, store) in vsti_stores.iter().enumerate() {
        let expected = LOCAL_DESTINATION + (index * 8) as u64;
        if store.buffer_address != expected {
            return Err(C310CapturedMaskedAddError::StoreAddress {
                index,
                actual: store.buffer_address,
            });
        }
        local.write(store.buffer_address, &store.data)?;
    }
    let mut output = vec![0_u8; C310_CAPTURED_MASKED_ADD_TILE_BYTES];
    local.read_into(LOCAL_DESTINATION, &mut output)?;
    Ok(C310CapturedMaskedAddTile {
        group,
        vldi_v0,
        vldi_v1,
        vadd_source_4,
        vadd_source_6,
        movp_p1,
        vadd_v0,
        vsti_stores,
        output,
    })
}

pub fn execute_captured_c310_masked_add(
    x: &[u8],
    y: &[u8],
    prior_destination: &[u8],
) -> Result<C310CapturedMaskedAddRun, C310CapturedMaskedAddError> {
    if x.len() != C310_CAPTURED_MASKED_ADD_BYTES
        || y.len() != C310_CAPTURED_MASKED_ADD_BYTES
        || prior_destination.len() != C310_CAPTURED_MASKED_ADD_BYTES
    {
        return Err(C310CapturedMaskedAddError::TensorLengths {
            x: x.len(),
            y: y.len(),
            prior: prior_destination.len(),
        });
    }
    let mut tiles = Vec::with_capacity(C310_CAPTURED_MASKED_ADD_TILES);
    let mut output = vec![0_u8; C310_CAPTURED_MASKED_ADD_BYTES];
    for group in 0..C310_CAPTURED_MASKED_ADD_TILES {
        let range = group * C310_CAPTURED_MASKED_ADD_TILE_BYTES
            ..(group + 1) * C310_CAPTURED_MASKED_ADD_TILE_BYTES;
        let tile = execute_tile(
            group,
            &x[range.clone()],
            &y[range.clone()],
            &prior_destination[range.clone()],
        )?;
        output[range].copy_from_slice(&tile.output);
        tiles.push(tile);
    }
    Ok(C310CapturedMaskedAddRun { tiles, output })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Fp32VectorOperation, evaluate_fp32_value};

    #[test]
    fn exact_tile_uses_local_windows_and_preserves_inactive_memory() {
        let x = (0..32_u32)
            .flat_map(|lane| (lane as f32).to_le_bytes())
            .collect::<Vec<_>>();
        let y = (0..32)
            .flat_map(|_| 0.5_f32.to_le_bytes())
            .collect::<Vec<_>>();
        let old = (0..32)
            .flat_map(|_| (-123.0_f32).to_le_bytes())
            .collect::<Vec<_>>();
        let tile = execute_tile(0, &x, &y, &old).unwrap();
        assert_eq!(&tile.vldi_v0[..128], x);
        assert_eq!(&tile.vldi_v0[128..], y);
        assert_eq!(&tile.vldi_v1[..128], y);
        assert_eq!(&tile.vldi_v1[128..], old);
        assert_eq!(tile.vadd_source_4, tile.vldi_v0);
        assert_eq!(tile.vadd_source_6, tile.vldi_v1);
        assert_eq!(&tile.movp_p1[..16], &[0x0f; 16]);
        assert_eq!(&tile.movp_p1[16..], &[0; 16]);
        assert_eq!(tile.vsti_stores.len(), 16);
        assert_eq!(&tile.vadd_v0[128..], &[0; 128]);
        for lane in 0..32 {
            let at = lane * 4;
            if lane % 2 == 0 {
                let first = u32::from_le_bytes(x[at..at + 4].try_into().unwrap());
                let second = u32::from_le_bytes(y[at..at + 4].try_into().unwrap());
                let expected = evaluate_fp32_value(Fp32VectorOperation::Add, first, second).bits;
                assert_eq!(tile.output[at..at + 4], expected.to_le_bytes());
            } else {
                assert_eq!(tile.output[at..at + 4], old[at..at + 4]);
            }
        }
    }

    #[test]
    fn tensor_lengths_fail_closed() {
        let input = vec![0; C310_CAPTURED_MASKED_ADD_BYTES];
        assert!(execute_captured_c310_masked_add(&input[..127], &input, &input).is_err());
    }
}
