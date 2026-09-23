use std::iter::FusedIterator;
use std::num::NonZeroU32;

use crate::isa::c220::mte::load2d_sparse::{
    C220_SPARSE_INDEX_BYTES, C220_SPARSE_WEIGHT_BYTES, C220Load2dSparseTransfer,
};
use crate::memory::sparse::MemoryByteState;
use crate::sim::c220::memory::{C220LocalBuffer, C220LocalBufferError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220SparseTransferResult {
    pub repeat_count: usize,
    pub weight_bytes: usize,
    pub index_bytes: usize,
    pub unknown_weight_bytes: usize,
    pub unknown_index_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SparseWrite {
    weight_address: u64,
    index_address: u64,
    weights: Vec<MemoryByteState>,
    indices: Vec<MemoryByteState>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220PreparedSparseTransfer {
    writes: Vec<SparseWrite>,
    pub result: C220SparseTransferResult,
}

impl C220PreparedSparseTransfer {
    pub fn commit(
        self,
        l0b: &mut C220LocalBuffer,
        weight_index: &mut C220LocalBuffer,
    ) -> Result<C220SparseTransferResult, C220LocalBufferError> {
        for write in self.writes {
            l0b.write_states_linear(write.weight_address, &write.weights)?;
            weight_index.write_states_linear(write.index_address, &write.indices)?;
        }
        Ok(self.result)
    }
}

pub fn prepare_c220_load2d_sparse(
    l1: &C220LocalBuffer,
    transfer: C220Load2dSparseTransfer,
) -> Result<C220PreparedSparseTransfer, C220LocalBufferError> {
    let repeat_count = usize::from(transfer.repeat_count());
    let mut writes = Vec::with_capacity(repeat_count);
    let mut result = C220SparseTransferResult {
        repeat_count,
        weight_bytes: repeat_count * C220_SPARSE_WEIGHT_BYTES as usize,
        index_bytes: repeat_count * C220_SPARSE_INDEX_BYTES as usize,
        unknown_weight_bytes: 0,
        unknown_index_bytes: 0,
    };
    for segment in transfer.segments() {
        let weights = l1.read_initialized_states_linear(
            segment.weight_source_address,
            C220_SPARSE_WEIGHT_BYTES as usize,
        )?;
        let indices = l1.read_initialized_states_linear(
            segment.index_source_address,
            C220_SPARSE_INDEX_BYTES as usize,
        )?;
        result.unknown_weight_bytes += weights
            .iter()
            .filter(|b| matches!(b, MemoryByteState::Unknown))
            .count();
        result.unknown_index_bytes += indices
            .iter()
            .filter(|b| matches!(b, MemoryByteState::Unknown))
            .count();
        writes.push(SparseWrite {
            weight_address: segment.weight_destination_address,
            index_address: segment.index_destination_address,
            weights,
            indices,
        });
    }
    Ok(C220PreparedSparseTransfer { writes, result })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220SparseOutput {
    Weight,
    Index,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220SparseReadUop {
    pub repeat_index: u8,
    pub output: C220SparseOutput,
    pub source_address: u64,
    pub destination_address: u64,
    pub input_bytes: u32,
    pub output_bytes: u32,
    pub completes_block: bool,
    pub signals_completion: bool,
    pub last_generated: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220SparseRequestPlan {
    transfer: C220Load2dSparseTransfer,
    access_width: NonZeroU32,
    position: usize,
    weight_parts: usize,
    index_parts: usize,
    remaining: usize,
}

impl C220SparseRequestPlan {
    pub fn new(transfer: C220Load2dSparseTransfer, access_width: NonZeroU32) -> Self {
        let weight_parts = C220_SPARSE_WEIGHT_BYTES.div_ceil(access_width.get()) as usize;
        let index_parts = C220_SPARSE_INDEX_BYTES.div_ceil(access_width.get()) as usize;
        Self {
            transfer,
            access_width,
            position: 0,
            weight_parts,
            index_parts,
            remaining: usize::from(transfer.repeat_count()) * (weight_parts + index_parts),
        }
    }
}

impl Iterator for C220SparseRequestPlan {
    type Item = C220SparseReadUop;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let parts = self.weight_parts + self.index_parts;
        let repeat = self.position / parts;
        let part = self.position % parts;
        let segment = self.transfer.segments().nth(repeat)?;
        let (output, source, destination, bytes, child) = if part < self.weight_parts {
            (
                C220SparseOutput::Weight,
                segment.weight_source_address,
                segment.weight_destination_address,
                C220_SPARSE_WEIGHT_BYTES,
                part,
            )
        } else {
            (
                C220SparseOutput::Index,
                segment.index_source_address,
                segment.index_destination_address,
                C220_SPARSE_INDEX_BYTES,
                part - self.weight_parts,
            )
        };
        let offset = child as u32 * self.access_width.get();
        let input_bytes = (bytes - offset).min(self.access_width.get());
        let completes_block = offset + input_bytes == bytes;
        let request = C220SparseReadUop {
            repeat_index: segment.repeat_index,
            output,
            source_address: source.wrapping_add(u64::from(offset)),
            destination_address: destination,
            input_bytes,
            output_bytes: bytes,
            completes_block,
            signals_completion: output == C220SparseOutput::Weight
                && completes_block
                && repeat + 1 == usize::from(self.transfer.repeat_count()),
            last_generated: self.remaining == 1,
        };
        self.position += 1;
        self.remaining -= 1;
        Some(request)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for C220SparseRequestPlan {}
impl FusedIterator for C220SparseRequestPlan {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::c220::mte::load2d_sparse::C220Load2dSparseInstruction;

    #[test]
    fn packed_sources_and_separate_index_tail_preserve_completion_marker() {
        let word = (3 << 29) | (1 << 27) | (24 << 22) | (1 << 17) | (2 << 12) | (3 << 7);
        let instruction = C220Load2dSparseInstruction::decode(word).unwrap();
        let mut registers = [0; 32];
        registers[1] = u64::MAX - 511;
        registers[2] = (19000 << 32) | 7000;
        registers[3] = 3 | (2 << 16) | (u64::MAX << 24);
        let transfer = instruction.capture(&registers);
        let segments = transfer.segments().collect::<Vec<_>>();
        assert_eq!(segments[0].weight_source_address, 7000 + 3 * 512);
        assert_eq!(segments[1].index_source_address, 19000 + 4 * 128);
        assert_eq!(segments[1].weight_destination_address, 0);
        assert_eq!(
            segments[1].index_destination_address,
            (registers[1] >> 2) + 128
        );
        for width in [64, 128, 256, 300, 1024] {
            let plan = C220SparseRequestPlan::new(transfer, NonZeroU32::new(width).unwrap());
            let expected = 2 * (512_u32.div_ceil(width) + 128_u32.div_ceil(width)) as usize;
            assert_eq!(plan.len(), expected);
            let uops = plan.collect::<Vec<_>>();
            assert_eq!(uops.iter().map(|u| u.input_bytes).sum::<u32>(), 1280);
            let completions = uops
                .iter()
                .filter(|u| u.signals_completion)
                .collect::<Vec<_>>();
            assert_eq!(completions.len(), 1);
            assert_eq!(completions[0].output, C220SparseOutput::Weight);
            assert_eq!(completions[0].repeat_index, 1);
            assert!(!completions[0].last_generated);
            assert_eq!(uops.last().unwrap().output, C220SparseOutput::Index);
            assert!(uops.last().unwrap().last_generated);
        }
        let mut l1 = C220LocalBuffer::new(512);
        let mut l0b = C220LocalBuffer::new(512);
        let mut index = C220LocalBuffer::new(128);
        for segment in transfer.segments() {
            l1.write_known_linear(segment.weight_source_address, &[0x42; 512])
                .unwrap();
            l1.write_known_linear(segment.index_source_address, &[0xa5; 128])
                .unwrap();
        }
        l1.write_states_linear(
            segments[1].index_source_address,
            &[MemoryByteState::Unknown],
        )
        .unwrap();
        let prepared = prepare_c220_load2d_sparse(&l1, transfer).unwrap();
        assert_eq!(prepared.result.unknown_index_bytes, 1);
        assert_eq!(prepared.result.unknown_weight_bytes, 0);
        let result = prepared.commit(&mut l0b, &mut index).unwrap();
        assert_eq!((result.weight_bytes, result.index_bytes), (1024, 256));
        for segment in transfer.segments() {
            assert_eq!(
                l0b.read_initialized_linear(segment.weight_destination_address, 512)
                    .unwrap(),
                vec![0x42; 512]
            );
            assert_eq!(
                index
                    .read_initialized_states_linear(segment.index_destination_address, 128)
                    .unwrap(),
                l1.read_initialized_states_linear(segment.index_source_address, 128)
                    .unwrap()
            );
        }
        registers[3] = !(255 << 16);
        assert_eq!(
            C220SparseRequestPlan::new(
                instruction.capture(&registers),
                NonZeroU32::new(128).unwrap()
            )
            .len(),
            0
        );
    }
}
