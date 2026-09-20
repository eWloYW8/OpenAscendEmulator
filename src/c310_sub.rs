use serde::Serialize;
use thiserror::Error;

use crate::{
    Architecture, C310RvecValueError, C310RvecValueMachine, C310RvecVstiStore, PvMemory,
    PvMemoryError, c310_normal_u32_masked_store,
};

pub const C310_CAPTURED_SUB_TILE_BYTES: usize = 128;
pub const C310_CAPTURED_SUB_TILES: usize = 32;
pub const C310_CAPTURED_SUB_BYTES: usize = C310_CAPTURED_SUB_TILE_BYTES * C310_CAPTURED_SUB_TILES;
pub const C310_CAPTURED_SUB_MTE2_X_WORD: u32 = 0x74ad_8bae;
pub const C310_CAPTURED_SUB_MTE2_Y_WORD: u32 = 0x74b3_6bae;
pub const C310_CAPTURED_SUB_VSUB_WORD: u32 = 0x8008_2781;
pub const C310_CAPTURED_SUB_VST_WORD: u32 = crate::rvec::C310_CAPTURED_SUB_VST_WORD;
pub const C310_CAPTURED_SUB_MTE3_WORD: u32 = 0x74e1_192c;
const fn observed_p1() -> [u8; 32] {
    let mut image = [0_u8; 32];
    let mut index = 0;
    while index < 16 {
        image[index] = 0x11;
        index += 1;
    }
    image
}
pub const C310_CAPTURED_SUB_P1: [u8; 32] = observed_p1();
const LOCAL_DESTINATION: u64 = 0x100;
const VLD_BYTES: usize = 256;
const WORDS_PER_TILE: usize = C310_CAPTURED_SUB_TILE_BYTES / 4;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct C310CapturedSubInputChunk {
    pub instruction_word: u32,
    pub source_offset: u64,
    pub destination_local: u64,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct C310CapturedSubOutputChunk {
    pub instruction_word: u32,
    pub source_local: u64,
    pub destination_offset: u64,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct C310CapturedSubTile {
    pub group: usize,
    pub mte2_x: C310CapturedSubInputChunk,
    pub mte2_y: C310CapturedSubInputChunk,
    pub vld_x: Vec<u8>,
    pub vld_y: Vec<u8>,
    pub vsub_source_4: Vec<u8>,
    pub vsub_source_6: Vec<u8>,
    pub vsub_v0: Vec<u8>,
    pub vst_stores: Vec<C310RvecVstiStore>,
    pub mte3_chunk: C310CapturedSubOutputChunk,
    pub output: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct C310CapturedSubRun {
    pub tiles: Vec<C310CapturedSubTile>,
    pub output: Vec<u8>,
}

#[derive(Debug, Error)]
pub enum C310CapturedSubError {
    #[error(
        "captured C310 Sub expects three {C310_CAPTURED_SUB_BYTES}-byte tensors; got X={x}, Y={y}, prior={prior}"
    )]
    TensorLengths { x: usize, y: usize, prior: usize },
    #[error(
        "captured C310 Sub expects two {C310_CAPTURED_SUB_BYTES}-byte inputs; got X={x}, Y={y}"
    )]
    InputLengths { x: usize, y: usize },
    #[error("captured C310 Sub P1 must contain exactly 32 bytes; got {actual}")]
    PredicateLength { actual: usize },
    #[error("captured C310 Sub P1 differs from the verified live image")]
    PredicateImage,
    #[error("captured C310 VST produced {actual} writes, expected {WORDS_PER_TILE}")]
    StoreCount { actual: usize },
    #[error("captured C310 VST active-lane footprint differs from lanes 0..31")]
    StoreFootprint,
    #[error(transparent)]
    Memory(#[from] PvMemoryError),
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

fn validate_p1(p1: &[u8]) -> Result<(), C310CapturedSubError> {
    if p1.len() != 32 {
        return Err(C310CapturedSubError::PredicateLength { actual: p1.len() });
    }
    if p1 != C310_CAPTURED_SUB_P1 {
        return Err(C310CapturedSubError::PredicateImage);
    }
    Ok(())
}

fn execute_tile(
    group: usize,
    x: &[u8],
    y: &[u8],
    prior_destination: &[u8],
    p1: &[u8],
) -> Result<C310CapturedSubTile, C310CapturedSubError> {
    let mut local = PvMemory::new(Architecture::Dav3510, 0, 1);
    let mte2_x = C310CapturedSubInputChunk {
        instruction_word: C310_CAPTURED_SUB_MTE2_X_WORD,
        source_offset: (group * C310_CAPTURED_SUB_TILE_BYTES) as u64,
        destination_local: 0,
        data: x.to_vec(),
    };
    let mte2_y = C310CapturedSubInputChunk {
        instruction_word: C310_CAPTURED_SUB_MTE2_Y_WORD,
        source_offset: (group * C310_CAPTURED_SUB_TILE_BYTES) as u64,
        destination_local: C310_CAPTURED_SUB_TILE_BYTES as u64,
        data: y.to_vec(),
    };
    local.write(mte2_x.destination_local, &mte2_x.data)?;
    local.write(mte2_y.destination_local, &mte2_y.data)?;
    local.write(LOCAL_DESTINATION, prior_destination)?;

    let mut vld_x = vec![0; VLD_BYTES];
    let mut vld_y = vec![0; VLD_BYTES];
    local.read_into(0, &mut vld_x)?;
    local.read_into(C310_CAPTURED_SUB_TILE_BYTES as u64, &mut vld_y)?;

    let mut rvec = C310RvecValueMachine::from_vector_and_predicate_bytes(
        vec![words(&vld_x), words(&vld_y)],
        vec![vec![0; 32], p1.to_vec()],
    )?;
    let step = rvec.execute_fp32_word_from_predicate_registers(C310_CAPTURED_SUB_VSUB_WORD)?;
    let vsub_source_4 = bytes(&step.first_source);
    let vsub_source_6 = bytes(&step.second_source);
    let vsub_v0 = bytes(rvec.vector_register(0).expect("V0 initialized"));

    let mut previous_bytes = [0_u8; VLD_BYTES];
    local.read_into(LOCAL_DESTINATION, &mut previous_bytes)?;
    let stored = c310_normal_u32_masked_store(&words(&previous_bytes), &words(&vsub_v0), p1)?;
    if stored
        .written
        .iter()
        .enumerate()
        .any(|(lane, written)| *written != (lane < WORDS_PER_TILE))
    {
        return Err(C310CapturedSubError::StoreFootprint);
    }
    let mut vst_stores = Vec::with_capacity(WORDS_PER_TILE);
    for (lane_index, written) in stored.written.iter().copied().enumerate() {
        if !written {
            continue;
        }
        let buffer_address = LOCAL_DESTINATION + (lane_index * 4) as u64;
        let data = stored.words[lane_index].to_le_bytes();
        local.write(buffer_address, &data)?;
        vst_stores.push(C310RvecVstiStore {
            lane_index,
            buffer_address,
            data,
        });
    }
    if vst_stores.len() != WORDS_PER_TILE {
        return Err(C310CapturedSubError::StoreCount {
            actual: vst_stores.len(),
        });
    }
    let mut output = vec![0; C310_CAPTURED_SUB_TILE_BYTES];
    local.read_into(LOCAL_DESTINATION, &mut output)?;
    let mte3_chunk = C310CapturedSubOutputChunk {
        instruction_word: C310_CAPTURED_SUB_MTE3_WORD,
        source_local: LOCAL_DESTINATION,
        destination_offset: (group * C310_CAPTURED_SUB_TILE_BYTES) as u64,
        data: output.clone(),
    };
    Ok(C310CapturedSubTile {
        group,
        mte2_x,
        mte2_y,
        vld_x,
        vld_y,
        vsub_source_4,
        vsub_source_6,
        vsub_v0,
        vst_stores,
        mte3_chunk,
        output,
    })
}

pub fn execute_captured_c310_sub(
    x: &[u8],
    y: &[u8],
    prior_destination: &[u8],
    p1: &[u8],
) -> Result<C310CapturedSubRun, C310CapturedSubError> {
    if x.len() != C310_CAPTURED_SUB_BYTES
        || y.len() != C310_CAPTURED_SUB_BYTES
        || prior_destination.len() != C310_CAPTURED_SUB_BYTES
    {
        return Err(C310CapturedSubError::TensorLengths {
            x: x.len(),
            y: y.len(),
            prior: prior_destination.len(),
        });
    }
    validate_p1(p1)?;
    let mut tiles = Vec::with_capacity(C310_CAPTURED_SUB_TILES);
    let mut output = vec![0; C310_CAPTURED_SUB_BYTES];
    for group in 0..C310_CAPTURED_SUB_TILES {
        let range =
            group * C310_CAPTURED_SUB_TILE_BYTES..(group + 1) * C310_CAPTURED_SUB_TILE_BYTES;
        let tile = execute_tile(
            group,
            &x[range.clone()],
            &y[range.clone()],
            &prior_destination[range.clone()],
            p1,
        )?;
        output[range].copy_from_slice(&tile.output);
        tiles.push(tile);
    }
    Ok(C310CapturedSubRun { tiles, output })
}

pub fn execute_captured_c310_sub_predecessor_chains(
    x: &[u8],
    y: &[u8],
    p1: &[u8],
) -> Result<C310CapturedSubRun, C310CapturedSubError> {
    if x.len() != C310_CAPTURED_SUB_BYTES || y.len() != C310_CAPTURED_SUB_BYTES {
        return Err(C310CapturedSubError::InputLengths {
            x: x.len(),
            y: y.len(),
        });
    }
    validate_p1(p1)?;
    let mut tiles = Vec::with_capacity(C310_CAPTURED_SUB_TILES);
    let mut output = vec![0; C310_CAPTURED_SUB_BYTES];
    for group in 0..C310_CAPTURED_SUB_TILES {
        let range =
            group * C310_CAPTURED_SUB_TILE_BYTES..(group + 1) * C310_CAPTURED_SUB_TILE_BYTES;
        let prior = if group.is_multiple_of(4) {
            vec![0; C310_CAPTURED_SUB_TILE_BYTES]
        } else {
            output[(group - 1) * C310_CAPTURED_SUB_TILE_BYTES..group * C310_CAPTURED_SUB_TILE_BYTES]
                .to_vec()
        };
        let tile = execute_tile(group, &x[range.clone()], &y[range.clone()], &prior, p1)?;
        output[range].copy_from_slice(&tile.output);
        tiles.push(tile);
    }
    Ok(C310CapturedSubRun { tiles, output })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Fp32VectorOperation, evaluate_fp32_value};

    #[test]
    fn sub_tile_reads_overlapping_windows_and_writes_32_words() {
        let x = (0..32_u32)
            .flat_map(|lane| (lane as f32).to_le_bytes())
            .collect::<Vec<_>>();
        let y = (0..32)
            .flat_map(|_| 0.5_f32.to_le_bytes())
            .collect::<Vec<_>>();
        let prior = vec![0x5a; 128];
        let mut p1 = [0_u8; 32];
        p1[..16].fill(0x11);
        let tile = execute_tile(0, &x, &y, &prior, &p1).unwrap();
        assert_eq!(tile.mte2_x.instruction_word, C310_CAPTURED_SUB_MTE2_X_WORD);
        assert_eq!(tile.mte2_x.source_offset, 0);
        assert_eq!(tile.mte2_x.destination_local, 0);
        assert_eq!(tile.mte2_x.data, x);
        assert_eq!(tile.mte2_y.instruction_word, C310_CAPTURED_SUB_MTE2_Y_WORD);
        assert_eq!(tile.mte2_y.source_offset, 0);
        assert_eq!(tile.mte2_y.destination_local, 0x80);
        assert_eq!(tile.mte2_y.data, y);
        assert_eq!(&tile.vld_x[..128], x);
        assert_eq!(&tile.vld_x[128..], y);
        assert_eq!(&tile.vld_y[..128], y);
        assert_eq!(&tile.vld_y[128..], prior);
        assert_eq!(tile.vsub_source_4, tile.vld_x);
        assert_eq!(tile.vsub_source_6, tile.vld_y);
        assert_eq!(tile.vst_stores.len(), 32);
        assert_eq!(
            tile.mte3_chunk.instruction_word,
            C310_CAPTURED_SUB_MTE3_WORD
        );
        assert_eq!(tile.mte3_chunk.source_local, 0x100);
        assert_eq!(tile.mte3_chunk.destination_offset, 0);
        assert_eq!(tile.mte3_chunk.data, tile.output);
        for lane in 0..32 {
            let first = u32::from_le_bytes(x[lane * 4..lane * 4 + 4].try_into().unwrap());
            let second = u32::from_le_bytes(y[lane * 4..lane * 4 + 4].try_into().unwrap());
            let expected = evaluate_fp32_value(Fp32VectorOperation::Subtract, first, second).bits;
            assert_eq!(tile.output[lane * 4..lane * 4 + 4], expected.to_le_bytes());
        }
    }

    #[test]
    fn wrong_shape_or_inactive_predicate_fails_closed() {
        let tensor = vec![0; C310_CAPTURED_SUB_BYTES];
        let mut p1 = [0_u8; 32];
        p1[..16].fill(0x11);
        assert!(execute_captured_c310_sub(&tensor[..127], &tensor, &tensor, &p1).is_err());
        assert!(execute_captured_c310_sub(&tensor, &tensor, &tensor, &p1[..16]).is_err());
        p1[0] = 0;
        assert!(execute_captured_c310_sub(&tensor, &tensor, &tensor, &p1).is_err());
        p1[0] = 0x11;
        p1[16] = 0x11;
        assert!(execute_captured_c310_sub(&tensor, &tensor, &tensor, &p1).is_err());
    }

    #[test]
    fn predecessor_chains_feed_only_the_prior_group_in_each_four_tile_chain() {
        let x = (0..1024_u32)
            .flat_map(|value| (value as f32).to_le_bytes())
            .collect::<Vec<_>>();
        let y = vec![0; C310_CAPTURED_SUB_BYTES];
        let mut p1 = [0_u8; 32];
        p1[..16].fill(0x11);
        let run = execute_captured_c310_sub_predecessor_chains(&x, &y, &p1).unwrap();
        assert_eq!(&run.tiles[0].vld_y[128..], &[0; 128]);
        assert_eq!(&run.tiles[1].vld_y[128..], run.tiles[0].output);
        assert_eq!(&run.tiles[4].vld_y[128..], &[0; 128]);
        assert_eq!(&run.tiles[5].vld_y[128..], run.tiles[4].output);
    }
}
