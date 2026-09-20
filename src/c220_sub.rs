use serde::Serialize;
use thiserror::Error;

use crate::c220_masked_add::C220CapturedByteSpan;
use crate::mte_c220::{
    C220MovOutToUbDescriptor, C220MovOutToUbError, CAPTURED_C220_SUB_MOV_OUT_TO_UB_X_WORD,
    CAPTURED_C220_SUB_MOV_OUT_TO_UB_Y_WORD, CAPTURED_C220_SUB_MOV_UB_TO_OUT_WORD,
};
use crate::{
    Architecture, C220DmaMovDescriptor, C220DmaMovError, C220VecArithmeticHint, Fp32VectorError,
    PvMemory, PvMemoryError,
};

pub const C220_CAPTURED_SUB_TILE_BYTES: usize = 128;
pub const C220_CAPTURED_SUB_TILES: usize = 32;
pub const C220_CAPTURED_SUB_BYTES: usize = C220_CAPTURED_SUB_TILE_BYTES * C220_CAPTURED_SUB_TILES;
pub const C220_CAPTURED_SUB_VSUB_WORD: u32 = 0x85dc_b619;
const LOCAL_DESTINATION: u64 = 0x100;
const SOURCE_SEGMENT_BYTES: usize = 32;
const SOURCE_SEGMENTS: usize = 8;
const LANES: usize = C220_CAPTURED_SUB_TILE_BYTES / 4;
const LOW_HALF_ACTIVE_MASK: [u64; 4] = [0xffff_ffff, 0, 0, 0];

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct C220CapturedSubTile {
    pub group: usize,
    pub mte2_x_writes: Vec<C220CapturedByteSpan>,
    pub mte2_y_writes: Vec<C220CapturedByteSpan>,
    pub source_0_reads: Vec<C220CapturedByteSpan>,
    pub source_1_reads: Vec<C220CapturedByteSpan>,
    pub vsub_writes: Vec<C220CapturedByteSpan>,
    pub output_local_reads: Vec<C220CapturedByteSpan>,
    pub output_segments: Vec<C220CapturedByteSpan>,
    pub output: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct C220CapturedSubRun {
    pub tiles: Vec<C220CapturedSubTile>,
    pub output: Vec<u8>,
}

#[derive(Debug, Error)]
pub enum C220CapturedSubError {
    #[error(
        "captured C220 Sub expects three {C220_CAPTURED_SUB_BYTES}-byte tensors; got X={x}, Y={y}, prior={prior}"
    )]
    TensorLengths { x: usize, y: usize, prior: usize },
    #[error(
        "captured C220 Sub expects two {C220_CAPTURED_SUB_BYTES}-byte inputs; got X={x}, Y={y}"
    )]
    InputLengths { x: usize, y: usize },
    #[error("captured C220 VSUB word did not select the FP32 value path")]
    VsubWord,
    #[error("captured C220 VSUB low-half lane {lane} was not active")]
    InactiveLane { lane: usize },
    #[error("captured C220 Sub MTE2 segment {index} has unexpected coordinates or size")]
    Mte2Geometry { index: usize },
    #[error("captured C220 Sub MTE3 segment {index} has unexpected coordinates or size")]
    Mte3Geometry { index: usize },
    #[error(transparent)]
    InputDescriptor(#[from] C220MovOutToUbError),
    #[error(transparent)]
    OutputDescriptor(#[from] C220DmaMovError),
    #[error(transparent)]
    Memory(#[from] PvMemoryError),
    #[error(transparent)]
    Vector(#[from] Fp32VectorError),
}

fn source_words(reads: &[C220CapturedByteSpan]) -> Vec<u32> {
    reads[..4]
        .iter()
        .flat_map(|span| span.data.chunks_exact(4))
        .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("four bytes")))
        .collect()
}

fn execute_tile(
    group: usize,
    x: &[u8],
    y: &[u8],
    prior_destination: &[u8],
) -> Result<C220CapturedSubTile, C220CapturedSubError> {
    let mut local = PvMemory::new(Architecture::Dav2201, 0, 1);
    let input_offset = (group * C220_CAPTURED_SUB_TILE_BYTES) as u64;
    let mut mte2_x_writes = Vec::with_capacity(4);
    let mut mte2_y_writes = Vec::with_capacity(4);
    for (word, local_base, input, writes) in [
        (
            CAPTURED_C220_SUB_MOV_OUT_TO_UB_X_WORD,
            0,
            x,
            &mut mte2_x_writes,
        ),
        (
            CAPTURED_C220_SUB_MOV_OUT_TO_UB_Y_WORD,
            C220_CAPTURED_SUB_TILE_BYTES as u64,
            y,
            &mut mte2_y_writes,
        ),
    ] {
        let descriptor = C220MovOutToUbDescriptor::decode(word, 0x40010)?;
        for (index, segment) in descriptor
            .segments(input_offset, local_base)?
            .into_iter()
            .enumerate()
        {
            let offset = index * SOURCE_SEGMENT_BYTES;
            if segment.source_hbm != input_offset + offset as u64
                || segment.destination_local != local_base + offset as u64
                || segment.bytes != SOURCE_SEGMENT_BYTES as u32
            {
                return Err(C220CapturedSubError::Mte2Geometry { index });
            }
            let data = input
                .get(offset..offset + SOURCE_SEGMENT_BYTES)
                .ok_or(C220CapturedSubError::Mte2Geometry { index })?
                .to_vec();
            local.write(segment.destination_local, &data)?;
            writes.push(C220CapturedByteSpan {
                address: segment.destination_local,
                data,
            });
        }
    }
    local.write(LOCAL_DESTINATION, prior_destination)?;

    let mut source_0_reads = Vec::with_capacity(SOURCE_SEGMENTS);
    let mut source_1_reads = Vec::with_capacity(SOURCE_SEGMENTS);
    for (base, reads) in [
        (0_u64, &mut source_0_reads),
        (C220_CAPTURED_SUB_TILE_BYTES as u64, &mut source_1_reads),
    ] {
        for segment in 0..SOURCE_SEGMENTS {
            let address = base + (segment * SOURCE_SEGMENT_BYTES) as u64;
            let mut data = vec![0; SOURCE_SEGMENT_BYTES];
            local.read_into(address, &mut data)?;
            reads.push(C220CapturedByteSpan { address, data });
        }
    }

    let hint = C220VecArithmeticHint::from_word(C220_CAPTURED_SUB_VSUB_WORD)
        .filter(|hint| hint.has_fp32_value_path())
        .ok_or(C220CapturedSubError::VsubWord)?;
    let lanes = hint.evaluate_fp32_lanes(
        &source_words(&source_0_reads),
        &source_words(&source_1_reads),
        &LOW_HALF_ACTIVE_MASK,
    )?;
    let mut vsub_writes = Vec::with_capacity(LANES);
    for (lane, result) in lanes.into_iter().enumerate() {
        if !result.active {
            return Err(C220CapturedSubError::InactiveLane { lane });
        }
        let address = LOCAL_DESTINATION + (lane * 4) as u64;
        let data = result.bits.to_le_bytes().to_vec();
        local.write(address, &data)?;
        vsub_writes.push(C220CapturedByteSpan { address, data });
    }
    let descriptor = C220DmaMovDescriptor::decode(CAPTURED_C220_SUB_MOV_UB_TO_OUT_WORD, 0x40010)?;
    let segments = descriptor.segments(LOCAL_DESTINATION, input_offset)?;
    if segments.len() != 4 {
        return Err(C220CapturedSubError::Mte3Geometry {
            index: segments.len(),
        });
    }
    let mut output_local_reads = Vec::with_capacity(4);
    let mut output_segments = Vec::with_capacity(4);
    let mut output = vec![0; C220_CAPTURED_SUB_TILE_BYTES];
    for (index, segment) in segments.into_iter().enumerate() {
        let offset = index * SOURCE_SEGMENT_BYTES;
        if segment.source_local != LOCAL_DESTINATION + offset as u64
            || segment.destination_hbm != input_offset + offset as u64
            || segment.bytes != SOURCE_SEGMENT_BYTES as u32
        {
            return Err(C220CapturedSubError::Mte3Geometry { index });
        }
        let mut data = vec![0; SOURCE_SEGMENT_BYTES];
        local.read_into(segment.source_local, &mut data)?;
        output[offset..offset + SOURCE_SEGMENT_BYTES].copy_from_slice(&data);
        output_local_reads.push(C220CapturedByteSpan {
            address: segment.source_local,
            data: data.clone(),
        });
        output_segments.push(C220CapturedByteSpan {
            address: segment.destination_hbm,
            data,
        });
    }
    Ok(C220CapturedSubTile {
        group,
        mte2_x_writes,
        mte2_y_writes,
        source_0_reads,
        source_1_reads,
        vsub_writes,
        output_local_reads,
        output_segments,
        output,
    })
}

pub fn execute_captured_c220_sub(
    x: &[u8],
    y: &[u8],
    prior_destination: &[u8],
) -> Result<C220CapturedSubRun, C220CapturedSubError> {
    if x.len() != C220_CAPTURED_SUB_BYTES
        || y.len() != C220_CAPTURED_SUB_BYTES
        || prior_destination.len() != C220_CAPTURED_SUB_BYTES
    {
        return Err(C220CapturedSubError::TensorLengths {
            x: x.len(),
            y: y.len(),
            prior: prior_destination.len(),
        });
    }
    let mut tiles = Vec::with_capacity(C220_CAPTURED_SUB_TILES);
    let mut output = vec![0; C220_CAPTURED_SUB_BYTES];
    for group in 0..C220_CAPTURED_SUB_TILES {
        let range =
            group * C220_CAPTURED_SUB_TILE_BYTES..(group + 1) * C220_CAPTURED_SUB_TILE_BYTES;
        let tile = execute_tile(
            group,
            &x[range.clone()],
            &y[range.clone()],
            &prior_destination[range.clone()],
        )?;
        output[range].copy_from_slice(&tile.output);
        tiles.push(tile);
    }
    Ok(C220CapturedSubRun { tiles, output })
}

pub fn execute_captured_c220_sub_predecessor_chains(
    x: &[u8],
    y: &[u8],
) -> Result<C220CapturedSubRun, C220CapturedSubError> {
    if x.len() != C220_CAPTURED_SUB_BYTES || y.len() != C220_CAPTURED_SUB_BYTES {
        return Err(C220CapturedSubError::InputLengths {
            x: x.len(),
            y: y.len(),
        });
    }
    let mut tiles = Vec::with_capacity(C220_CAPTURED_SUB_TILES);
    let mut output = vec![0; C220_CAPTURED_SUB_BYTES];
    for group in 0..C220_CAPTURED_SUB_TILES {
        let range =
            group * C220_CAPTURED_SUB_TILE_BYTES..(group + 1) * C220_CAPTURED_SUB_TILE_BYTES;
        let prior = if group.is_multiple_of(4) {
            vec![0; C220_CAPTURED_SUB_TILE_BYTES]
        } else {
            output[(group - 1) * C220_CAPTURED_SUB_TILE_BYTES..group * C220_CAPTURED_SUB_TILE_BYTES]
                .to_vec()
        };
        let tile = execute_tile(group, &x[range.clone()], &y[range.clone()], &prior)?;
        output[range].copy_from_slice(&tile.output);
        tiles.push(tile);
    }
    Ok(C220CapturedSubRun { tiles, output })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Fp32VectorOperation, evaluate_fp32_value};

    #[test]
    fn sub_tile_reads_overlapping_source_windows_and_writes_32_words() {
        let x = (0..32_u32)
            .flat_map(|lane| (lane as f32).to_le_bytes())
            .collect::<Vec<_>>();
        let y = (0..32)
            .flat_map(|_| 0.5_f32.to_le_bytes())
            .collect::<Vec<_>>();
        let prior = vec![0x5a; 128];
        let tile = execute_tile(0, &x, &y, &prior).unwrap();
        assert_eq!(tile.mte2_x_writes.len(), 4);
        assert_eq!(tile.mte2_y_writes.len(), 4);
        assert_eq!(tile.mte2_x_writes[0].data, x[..32]);
        assert_eq!(tile.mte2_y_writes[3].data, y[96..]);
        let first: Vec<_> = tile
            .source_0_reads
            .iter()
            .flat_map(|span| span.data.iter().copied())
            .collect();
        let second: Vec<_> = tile
            .source_1_reads
            .iter()
            .flat_map(|span| span.data.iter().copied())
            .collect();
        assert_eq!(&first[..128], x);
        assert_eq!(&first[128..], y);
        assert_eq!(&second[..128], y);
        assert_eq!(&second[128..], prior);
        assert_eq!(tile.vsub_writes.len(), 32);
        assert_eq!(tile.output_local_reads.len(), 4);
        assert_eq!(tile.output_segments.len(), 4);
        assert_eq!(tile.output_local_reads[0].address, 0x100);
        assert_eq!(tile.output_segments[3].address, 96);
        for lane in 0..32 {
            let offset = lane * 4;
            let first = u32::from_le_bytes(x[offset..offset + 4].try_into().unwrap());
            let second = u32::from_le_bytes(y[offset..offset + 4].try_into().unwrap());
            let expected = evaluate_fp32_value(Fp32VectorOperation::Subtract, first, second).bits;
            assert_eq!(tile.output[offset..offset + 4], expected.to_le_bytes());
        }
    }

    #[test]
    fn predecessor_chains_and_lengths_are_bounded() {
        let x = (0..1024_u32)
            .flat_map(|value| (value as f32).to_le_bytes())
            .collect::<Vec<_>>();
        let y = vec![0; C220_CAPTURED_SUB_BYTES];
        let run = execute_captured_c220_sub_predecessor_chains(&x, &y).unwrap();
        assert_eq!(run.tiles[0].source_1_reads[4].data, [0; 32]);
        assert_eq!(run.tiles[4].source_1_reads[4].data, [0; 32]);
        assert_eq!(
            run.tiles[1].source_1_reads[4].data,
            run.tiles[0].output[..32]
        );
        assert!(execute_captured_c220_sub_predecessor_chains(&x[..127], &y).is_err());
        assert!(execute_captured_c220_sub(&x, &y, &y[..127]).is_err());
    }
}
