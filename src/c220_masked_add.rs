use serde::Serialize;
use thiserror::Error;

use crate::mte_c220::{
    C220MovOutToUbDescriptor, C220MovOutToUbError, CAPTURED_C220_MOV_OUT_TO_UB_X_WORD,
    CAPTURED_C220_MOV_OUT_TO_UB_Y_WORD, CAPTURED_C220_MOV_UB_TO_OUT_WORD,
};
use crate::{
    Architecture, C220DmaMovDescriptor, C220DmaMovError, C220VecArithmeticHint, Fp32VectorError,
    PvMemory, PvMemoryError,
};

pub const C220_CAPTURED_MASKED_ADD_TILE_BYTES: usize = 128;
pub const C220_CAPTURED_MASKED_ADD_TILES: usize = 32;
pub const C220_CAPTURED_MASKED_ADD_BYTES: usize =
    C220_CAPTURED_MASKED_ADD_TILE_BYTES * C220_CAPTURED_MASKED_ADD_TILES;
const SEGMENT_BYTES: usize = 32;
const DESTINATION: u64 = 0x100;
const LOCAL_BYTES: usize = 0x180;
const VADD_WORD: u32 = 0x85e0_d720;
const MASK: [u64; 4] = [0x5555_5555, 0, 0, 0];

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct C220CapturedByteSpan {
    pub address: u64,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct C220CapturedMaskedAddTile {
    pub group: usize,
    pub mte2_x_writes: Vec<C220CapturedByteSpan>,
    pub mte2_y_writes: Vec<C220CapturedByteSpan>,
    pub prior_destination_writes: Vec<C220CapturedByteSpan>,
    pub vadd_source_0_reads: Vec<C220CapturedByteSpan>,
    pub vadd_source_1_reads: Vec<C220CapturedByteSpan>,
    pub vadd_writes: Vec<C220CapturedByteSpan>,
    pub output_local_reads: Vec<C220CapturedByteSpan>,
    pub output_segments: Vec<C220CapturedByteSpan>,
    pub output: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct C220CapturedMaskedAddRun {
    pub tiles: Vec<C220CapturedMaskedAddTile>,
    pub output: Vec<u8>,
}

#[derive(Debug, Error)]
pub enum C220CapturedMaskedAddError {
    #[error(
        "captured C220 masked-Add expects three 4096-byte tensors; got X={x}, Y={y}, prior={prior}"
    )]
    TensorLengths { x: usize, y: usize, prior: usize },
    #[error("captured C220 masked-Add word {VADD_WORD:#010x} did not decode")]
    VaddWord,
    #[error("captured C220 {stage} produced {actual} segments, expected {expected}")]
    SegmentCount {
        stage: &'static str,
        actual: usize,
        expected: usize,
    },
    #[error("captured C220 {stage} segment {index} has unexpected coordinates or size")]
    SegmentGeometry { stage: &'static str, index: usize },
    #[error("captured C220 VADD lane {lane} has an unexpected active-mask bit")]
    ActiveMask { lane: usize },
    #[error("local byte at {offset:#x} was not initialized before source read")]
    UninitializedLocal { offset: usize },
    #[error(transparent)]
    Memory(#[from] PvMemoryError),
    #[error(transparent)]
    InputDescriptor(#[from] C220MovOutToUbError),
    #[error(transparent)]
    OutputDescriptor(#[from] C220DmaMovError),
    #[error(transparent)]
    Vector(#[from] Fp32VectorError),
}

fn words(spans: &[C220CapturedByteSpan]) -> Vec<u32> {
    spans
        .iter()
        .take(4)
        .flat_map(|span| span.data.chunks_exact(4))
        .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("four bytes")))
        .collect()
}

fn execute_tile(
    group: usize,
    x: &[u8],
    y: &[u8],
    prior_destination: &[u8],
) -> Result<C220CapturedMaskedAddTile, C220CapturedMaskedAddError> {
    let mut local = PvMemory::new(Architecture::Dav2201, 0, 1);
    let input_offset = (group * C220_CAPTURED_MASKED_ADD_TILE_BYTES) as u64;
    let mut mte2_x_writes = Vec::with_capacity(4);
    let mut mte2_y_writes = Vec::with_capacity(4);
    for (word, local_base, input, writes) in [
        (CAPTURED_C220_MOV_OUT_TO_UB_X_WORD, 0, x, &mut mte2_x_writes),
        (
            CAPTURED_C220_MOV_OUT_TO_UB_Y_WORD,
            C220_CAPTURED_MASKED_ADD_TILE_BYTES as u64,
            y,
            &mut mte2_y_writes,
        ),
    ] {
        let descriptor = C220MovOutToUbDescriptor::decode(word, 0x40010)?;
        let segments = descriptor.segments(input_offset, local_base)?;
        for (index, segment) in segments.into_iter().enumerate() {
            let offset = index * SEGMENT_BYTES;
            if segment.source_hbm != input_offset + offset as u64
                || segment.destination_local != local_base + offset as u64
                || segment.bytes != SEGMENT_BYTES as u32
            {
                return Err(C220CapturedMaskedAddError::SegmentGeometry {
                    stage: "MTE2",
                    index,
                });
            }
            let data = input
                .get(offset..offset + SEGMENT_BYTES)
                .ok_or(C220CapturedMaskedAddError::SegmentGeometry {
                    stage: "MTE2",
                    index,
                })?
                .to_vec();
            local.write(segment.destination_local, &data)?;
            writes.push(C220CapturedByteSpan {
                address: segment.destination_local,
                data,
            });
        }
    }

    let mut prior_destination_writes = Vec::with_capacity(32);
    for (lane, data) in prior_destination.chunks_exact(4).enumerate() {
        let address = DESTINATION + (lane * 4) as u64;
        local.write(address, data)?;
        prior_destination_writes.push(C220CapturedByteSpan {
            address,
            data: data.to_vec(),
        });
    }
    for offset in 0..LOCAL_BYTES {
        if local.dirty_byte(offset as u64) != Some(1) {
            return Err(C220CapturedMaskedAddError::UninitializedLocal { offset });
        }
    }

    let mut vadd_source_0_reads = Vec::with_capacity(8);
    let mut vadd_source_1_reads = Vec::with_capacity(8);
    for (base, reads) in [
        (0, &mut vadd_source_0_reads),
        (
            C220_CAPTURED_MASKED_ADD_TILE_BYTES as u64,
            &mut vadd_source_1_reads,
        ),
    ] {
        for segment in 0..8 {
            let address = base + (segment * SEGMENT_BYTES) as u64;
            let mut data = vec![0; SEGMENT_BYTES];
            local.read_into(address, &mut data)?;
            reads.push(C220CapturedByteSpan { address, data });
        }
    }
    let first = words(&vadd_source_0_reads);
    let second = words(&vadd_source_1_reads);
    let hint =
        C220VecArithmeticHint::from_word(VADD_WORD).ok_or(C220CapturedMaskedAddError::VaddWord)?;
    let lanes = hint.evaluate_fp32_lanes(&first, &second, &MASK)?;
    let mut vadd_writes = Vec::with_capacity(16);
    for (lane, result) in lanes.iter().enumerate() {
        if result.active != lane.is_multiple_of(2) {
            return Err(C220CapturedMaskedAddError::ActiveMask { lane });
        }
        if result.active {
            let address = DESTINATION + (lane * 4) as u64;
            let data = result.bits.to_le_bytes().to_vec();
            local.write(address, &data)?;
            vadd_writes.push(C220CapturedByteSpan { address, data });
        }
    }
    if vadd_writes.len() != 16 {
        return Err(C220CapturedMaskedAddError::SegmentCount {
            stage: "VADD",
            actual: vadd_writes.len(),
            expected: 16,
        });
    }

    let descriptor = C220DmaMovDescriptor::decode(CAPTURED_C220_MOV_UB_TO_OUT_WORD, 0x40010)?;
    let segments = descriptor.segments(DESTINATION, input_offset)?;
    if segments.len() != 4 {
        return Err(C220CapturedMaskedAddError::SegmentCount {
            stage: "MTE3",
            actual: segments.len(),
            expected: 4,
        });
    }
    let mut output_local_reads = Vec::with_capacity(4);
    let mut output_segments = Vec::with_capacity(4);
    let mut output = vec![0; C220_CAPTURED_MASKED_ADD_TILE_BYTES];
    for (index, segment) in segments.into_iter().enumerate() {
        let offset = index * SEGMENT_BYTES;
        if segment.source_local != DESTINATION + offset as u64
            || segment.destination_hbm != input_offset + offset as u64
            || segment.bytes != SEGMENT_BYTES as u32
        {
            return Err(C220CapturedMaskedAddError::SegmentGeometry {
                stage: "MTE3",
                index,
            });
        }
        let mut data = vec![0; SEGMENT_BYTES];
        local.read_into(segment.source_local, &mut data)?;
        output_local_reads.push(C220CapturedByteSpan {
            address: segment.source_local,
            data: data.clone(),
        });
        output[offset..offset + SEGMENT_BYTES].copy_from_slice(&data);
        output_segments.push(C220CapturedByteSpan {
            address: segment.destination_hbm,
            data,
        });
    }
    Ok(C220CapturedMaskedAddTile {
        group,
        mte2_x_writes,
        mte2_y_writes,
        prior_destination_writes,
        vadd_source_0_reads,
        vadd_source_1_reads,
        vadd_writes,
        output_local_reads,
        output_segments,
        output,
    })
}

pub fn execute_captured_c220_masked_add(
    x: &[u8],
    y: &[u8],
    prior_destination: &[u8],
) -> Result<C220CapturedMaskedAddRun, C220CapturedMaskedAddError> {
    if x.len() != C220_CAPTURED_MASKED_ADD_BYTES
        || y.len() != C220_CAPTURED_MASKED_ADD_BYTES
        || prior_destination.len() != C220_CAPTURED_MASKED_ADD_BYTES
    {
        return Err(C220CapturedMaskedAddError::TensorLengths {
            x: x.len(),
            y: y.len(),
            prior: prior_destination.len(),
        });
    }
    let mut tiles = Vec::with_capacity(C220_CAPTURED_MASKED_ADD_TILES);
    let mut output = vec![0; C220_CAPTURED_MASKED_ADD_BYTES];
    for group in 0..C220_CAPTURED_MASKED_ADD_TILES {
        let range = group * C220_CAPTURED_MASKED_ADD_TILE_BYTES
            ..(group + 1) * C220_CAPTURED_MASKED_ADD_TILE_BYTES;
        let tile = execute_tile(
            group,
            &x[range.clone()],
            &y[range.clone()],
            &prior_destination[range.clone()],
        )?;
        output[range].copy_from_slice(&tile.output);
        tiles.push(tile);
    }
    Ok(C220CapturedMaskedAddRun { tiles, output })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Fp32VectorOperation, evaluate_fp32_value};

    #[test]
    fn exact_tile_reads_local_sources_and_preserves_inactive_words() {
        let x = (0..32_u32)
            .flat_map(|lane| (lane as f32).to_le_bytes())
            .collect::<Vec<_>>();
        let y = (0..32)
            .flat_map(|_| 0.5_f32.to_le_bytes())
            .collect::<Vec<_>>();
        let prior = (0..32)
            .flat_map(|_| (-123.0_f32).to_le_bytes())
            .collect::<Vec<_>>();
        let tile = execute_tile(0, &x, &y, &prior).unwrap();
        assert_eq!(tile.mte2_x_writes.len(), 4);
        assert_eq!(tile.mte2_y_writes.len(), 4);
        assert_eq!(tile.vadd_source_0_reads.len(), 8);
        assert_eq!(tile.vadd_source_1_reads.len(), 8);
        assert_eq!(tile.vadd_writes.len(), 16);
        assert_eq!(tile.output_local_reads.len(), 4);
        assert_eq!(tile.output_segments.len(), 4);
        for lane in 0..32 {
            let at = lane * 4;
            if lane % 2 == 0 {
                let first = u32::from_le_bytes(x[at..at + 4].try_into().unwrap());
                let second = u32::from_le_bytes(y[at..at + 4].try_into().unwrap());
                let expected = evaluate_fp32_value(Fp32VectorOperation::Add, first, second).bits;
                assert_eq!(tile.output[at..at + 4], expected.to_le_bytes());
            } else {
                assert_eq!(tile.output[at..at + 4], prior[at..at + 4]);
            }
        }
    }

    #[test]
    fn tensor_lengths_fail_closed() {
        let input = vec![0; C220_CAPTURED_MASKED_ADD_BYTES];
        assert!(execute_captured_c220_masked_add(&input[..127], &input, &input).is_err());
    }
}
