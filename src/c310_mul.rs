use serde::Serialize;
use thiserror::Error;

use crate::rvec::{C310_CAPTURED_PLT32_WORD, C310_CAPTURED_SMOVI32_WORD};
use crate::{
    Architecture, C310CapturedPltError, C310CapturedPltStep, C310CapturedSmoviError,
    C310CapturedSmoviStep, C310PbRvecScalarProjection, C310PushPbInstruction, C310RvecValueError,
    C310RvecValueMachine, C310RvecVstiStore, PvMemory, PvMemoryError, c310_normal_u32_masked_store,
};

pub const C310_CAPTURED_MUL_TILE_BYTES: usize = 128;
pub const C310_CAPTURED_MUL_TILES: usize = 32;
pub const C310_CAPTURED_MUL_BYTES: usize = C310_CAPTURED_MUL_TILE_BYTES * C310_CAPTURED_MUL_TILES;
pub const C310_CAPTURED_MUL_MTE2_X_WORD: u32 = 0x74ad_8bae;
pub const C310_CAPTURED_MUL_MTE2_Y_WORD: u32 = 0x74b3_6bae;
pub const C310_CAPTURED_MUL_PUSH_PB_WORD: u32 = 0x4319_7108;
pub const C310_CAPTURED_MUL_VMUL_WORD: u32 = 0x8000_27c0;
pub const C310_CAPTURED_MUL_VST_WORD: u32 = crate::rvec::C310_CAPTURED_SUB_VST_WORD;
pub const C310_CAPTURED_MUL_MTE3_WORD: u32 = 0x74e1_192c;
pub const C310_CAPTURED_MUL_PB_SOURCE_VALUES: [u64; 4] = [
    0x0000_0100_0000_003e,
    1,
    0x0000_0100_0000_0080,
    0x0000_0100_0000_0080,
];

const LOCAL_DESTINATION: u64 = 0x100;
const VLD_BYTES: usize = 256;
const WORDS_PER_TILE: usize = C310_CAPTURED_MUL_TILE_BYTES / 4;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct C310CapturedMulInputChunk {
    pub instruction_word: u32,
    pub source_offset: u64,
    pub destination_local: u64,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct C310CapturedMulOutputChunk {
    pub instruction_word: u32,
    pub source_local: u64,
    pub destination_offset: u64,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct C310CapturedMulTile {
    pub group: usize,
    pub mte2_x: C310CapturedMulInputChunk,
    pub mte2_y: C310CapturedMulInputChunk,
    pub predicate_buffer_slot: Vec<u8>,
    pub scalar_projection: C310PbRvecScalarProjection,
    pub vld_x: Vec<u8>,
    pub vld_y: Vec<u8>,
    pub smovi: C310CapturedSmoviStep,
    pub plt: C310CapturedPltStep,
    pub vmul_word: u32,
    pub vmul_source_4: Vec<u8>,
    pub vmul_source_5: Vec<u8>,
    pub vmul_v0: Vec<u8>,
    pub vst_instruction_word: u32,
    pub vst_stores: Vec<C310RvecVstiStore>,
    pub mte3_chunk: C310CapturedMulOutputChunk,
    pub output: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct C310CapturedMulRun {
    pub tiles: Vec<C310CapturedMulTile>,
    pub output: Vec<u8>,
}

#[derive(Debug, Error)]
pub enum C310CapturedMulError {
    #[error(
        "captured C310 Mul expects three {C310_CAPTURED_MUL_BYTES}-byte tensors; got X={x}, Y={y}, prior={prior}"
    )]
    TensorLengths { x: usize, y: usize, prior: usize },
    #[error(
        "captured C310 Mul expects two {C310_CAPTURED_MUL_BYTES}-byte inputs; got X={x}, Y={y}"
    )]
    InputLengths { x: usize, y: usize },
    #[error("captured C310 VST produced {actual} writes, expected {WORDS_PER_TILE}")]
    StoreCount { actual: usize },
    #[error("captured C310 VST active-lane footprint differs from lanes 0..31")]
    StoreFootprint,
    #[error(transparent)]
    Memory(#[from] PvMemoryError),
    #[error(transparent)]
    Rvec(#[from] C310RvecValueError),
    #[error(transparent)]
    Smovi(#[from] C310CapturedSmoviError),
    #[error(transparent)]
    Plt(#[from] C310CapturedPltError),
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

pub fn c310_captured_mul_predicate_buffer_slot() -> [u8; crate::C310_PB_SLOT_BYTES] {
    let instruction =
        C310PushPbInstruction::decode(Architecture::Dav3510, C310_CAPTURED_MUL_PUSH_PB_WORD)
            .expect("captured C310 Mul PUSH_PB word is supported");
    let step = instruction.from_source_values(0, C310_CAPTURED_MUL_PB_SOURCE_VALUES);
    let mut slot = [0; crate::C310_PB_SLOT_BYTES];
    slot[..step.bytes.len()].copy_from_slice(&step.bytes);
    slot
}

fn execute_tile(
    group: usize,
    x: &[u8],
    y: &[u8],
    prior_destination: &[u8],
) -> Result<C310CapturedMulTile, C310CapturedMulError> {
    let mut local = PvMemory::new(Architecture::Dav3510, 0, 1);
    let mte2_x = C310CapturedMulInputChunk {
        instruction_word: C310_CAPTURED_MUL_MTE2_X_WORD,
        source_offset: (group * C310_CAPTURED_MUL_TILE_BYTES) as u64,
        destination_local: 0,
        data: x.to_vec(),
    };
    let mte2_y = C310CapturedMulInputChunk {
        instruction_word: C310_CAPTURED_MUL_MTE2_Y_WORD,
        source_offset: (group * C310_CAPTURED_MUL_TILE_BYTES) as u64,
        destination_local: C310_CAPTURED_MUL_TILE_BYTES as u64,
        data: y.to_vec(),
    };
    local.write(mte2_x.destination_local, &mte2_x.data)?;
    local.write(mte2_y.destination_local, &mte2_y.data)?;
    local.write(LOCAL_DESTINATION, prior_destination)?;

    let mut vld_x = vec![0; VLD_BYTES];
    let mut vld_y = vec![0; VLD_BYTES];
    local.read_into(0, &mut vld_x)?;
    local.read_into(C310_CAPTURED_MUL_TILE_BYTES as u64, &mut vld_y)?;

    let predicate_buffer_slot = c310_captured_mul_predicate_buffer_slot();
    let mut rvec = C310RvecValueMachine::from_vector_and_predicate_bytes(
        vec![words(&vld_x), words(&vld_y)],
        vec![vec![0; 32], vec![0; 32]],
    )?;
    let scalar_projection = rvec.apply_pb_scalar_init(&predicate_buffer_slot);
    let smovi = rvec.execute_captured_smovi_word(0x10d0_d904, C310_CAPTURED_SMOVI32_WORD)?;
    let plt = rvec.execute_captured_plt32_word(0x10d0_d914, C310_CAPTURED_PLT32_WORD)?;
    let step = rvec.execute_fp32_word_from_predicate_registers(C310_CAPTURED_MUL_VMUL_WORD)?;
    let vmul_source_4 = bytes(&step.first_source);
    let vmul_source_5 = bytes(&step.second_source);
    let vmul_v0 = bytes(rvec.vector_register(0).expect("V0 initialized"));

    let mut previous_bytes = [0_u8; VLD_BYTES];
    local.read_into(LOCAL_DESTINATION, &mut previous_bytes)?;
    let stored = c310_normal_u32_masked_store(
        &words(&previous_bytes),
        &words(&vmul_v0),
        &plt.predicate_bytes,
    )?;
    if stored
        .written
        .iter()
        .enumerate()
        .any(|(lane, written)| *written != (lane < WORDS_PER_TILE))
    {
        return Err(C310CapturedMulError::StoreFootprint);
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
        return Err(C310CapturedMulError::StoreCount {
            actual: vst_stores.len(),
        });
    }
    let mut output = vec![0; C310_CAPTURED_MUL_TILE_BYTES];
    local.read_into(LOCAL_DESTINATION, &mut output)?;
    let mte3_chunk = C310CapturedMulOutputChunk {
        instruction_word: C310_CAPTURED_MUL_MTE3_WORD,
        source_local: LOCAL_DESTINATION,
        destination_offset: (group * C310_CAPTURED_MUL_TILE_BYTES) as u64,
        data: output.clone(),
    };
    Ok(C310CapturedMulTile {
        group,
        mte2_x,
        mte2_y,
        predicate_buffer_slot: predicate_buffer_slot.to_vec(),
        scalar_projection,
        vld_x,
        vld_y,
        smovi,
        plt,
        vmul_word: C310_CAPTURED_MUL_VMUL_WORD,
        vmul_source_4,
        vmul_source_5,
        vmul_v0,
        vst_instruction_word: C310_CAPTURED_MUL_VST_WORD,
        vst_stores,
        mte3_chunk,
        output,
    })
}

pub fn execute_captured_c310_mul(
    x: &[u8],
    y: &[u8],
    prior_destination: &[u8],
) -> Result<C310CapturedMulRun, C310CapturedMulError> {
    if x.len() != C310_CAPTURED_MUL_BYTES
        || y.len() != C310_CAPTURED_MUL_BYTES
        || prior_destination.len() != C310_CAPTURED_MUL_BYTES
    {
        return Err(C310CapturedMulError::TensorLengths {
            x: x.len(),
            y: y.len(),
            prior: prior_destination.len(),
        });
    }
    execute_tiles(x, y, |range, _| prior_destination[range].to_vec())
}

pub fn execute_captured_c310_mul_predecessor_chains(
    x: &[u8],
    y: &[u8],
) -> Result<C310CapturedMulRun, C310CapturedMulError> {
    if x.len() != C310_CAPTURED_MUL_BYTES || y.len() != C310_CAPTURED_MUL_BYTES {
        return Err(C310CapturedMulError::InputLengths {
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
) -> Result<C310CapturedMulRun, C310CapturedMulError> {
    let mut tiles = Vec::with_capacity(C310_CAPTURED_MUL_TILES);
    let mut output = vec![0; C310_CAPTURED_MUL_BYTES];
    let zero_prior = [0; C310_CAPTURED_MUL_TILE_BYTES];
    for group in 0..C310_CAPTURED_MUL_TILES {
        let range =
            group * C310_CAPTURED_MUL_TILE_BYTES..(group + 1) * C310_CAPTURED_MUL_TILE_BYTES;
        let predecessor = if group.is_multiple_of(4) {
            &zero_prior[..]
        } else {
            &output
                [(group - 1) * C310_CAPTURED_MUL_TILE_BYTES..group * C310_CAPTURED_MUL_TILE_BYTES]
        };
        let prior = prior_for(range.clone(), predecessor);
        let tile = execute_tile(group, &x[range.clone()], &y[range.clone()], &prior)?;
        output[range].copy_from_slice(&tile.output);
        tiles.push(tile);
    }
    Ok(C310CapturedMulRun { tiles, output })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Fp32VectorOperation, evaluate_fp32_value};

    #[test]
    fn multiply_tile_projects_pb_scalars_and_writes_32_words() {
        let mut x = (0..32_u32)
            .flat_map(|lane| (lane as f32).to_le_bytes())
            .collect::<Vec<_>>();
        let mut y = (0..32)
            .flat_map(|_| 0.5_f32.to_le_bytes())
            .collect::<Vec<_>>();
        x[..4].copy_from_slice(&0_u32.to_le_bytes());
        y[..4].copy_from_slice(&f32::INFINITY.to_bits().to_le_bytes());
        let prior = vec![0x5a; C310_CAPTURED_MUL_TILE_BYTES];
        let tile = execute_tile(0, &x, &y, &prior).unwrap();
        assert_eq!(tile.scalar_projection.big_flags, 0x3e);
        assert_eq!(tile.scalar_projection.consumed_payload_words, 5);
        assert_eq!(tile.vld_x[..128], x);
        assert_eq!(tile.vld_x[128..], y);
        assert_eq!(tile.vld_y[..128], y);
        assert_eq!(tile.vld_y[128..], prior);
        assert_eq!(tile.smovi.value, 32);
        assert_eq!(tile.plt.lane_limit, 32);
        assert_eq!(tile.vmul_word, C310_CAPTURED_MUL_VMUL_WORD);
        assert_eq!(tile.vmul_source_4, tile.vld_x);
        assert_eq!(tile.vmul_source_5, tile.vld_y);
        assert_eq!(tile.vst_instruction_word, C310_CAPTURED_MUL_VST_WORD);
        assert_eq!(tile.vst_stores.len(), 32);
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
        let x = vec![0; C310_CAPTURED_MUL_BYTES];
        let y = vec![0; C310_CAPTURED_MUL_BYTES];
        let run = execute_captured_c310_mul_predecessor_chains(&x, &y).unwrap();
        assert_eq!(run.tiles.len(), C310_CAPTURED_MUL_TILES);
        assert_eq!(&run.tiles[0].vld_y[128..], &[0; 128]);
        assert_eq!(&run.tiles[1].vld_y[128..], run.tiles[0].output);
        assert_eq!(&run.tiles[4].vld_y[128..], &[0; 128]);
        assert!(execute_captured_c310_mul_predecessor_chains(&x[..127], &y).is_err());
        assert!(execute_captured_c310_mul(&x, &y, &y[..127]).is_err());
    }
}
