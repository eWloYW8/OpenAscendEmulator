use super::C220Load3dV2Command;
use crate::isa::c220::mte::load3d::{C220Load3dDestination, C220Load3dElement};

/// One repeated LOAD3D region, expressed in logical elements before packing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Load3dTile {
    pub outer_index: u32,
    pub inner_index: u32,
    pub outer_count: u32,
    pub inner_count: u32,
    pub k_start: u32,
    pub m_start: u32,
    pub k_length: u32,
    pub m_length: u32,
    pub source_base: u64,
    pub destination_base: u64,
    pub block_stride_bytes: u64,
    pub transpose_stride_bytes: u64,
    pub fp32_transpose_half: bool,
}

#[derive(Debug, Clone)]
pub struct C220Load3dTiles {
    command: C220Load3dV2Command,
    next_index: u32,
    outer_count: u32,
    inner_count: u32,
    aligned_k: u32,
    aligned_m: u32,
    k_alignment: u32,
    storage_bytes: u32,
}

impl C220Load3dTiles {
    pub(super) fn new(command: C220Load3dV2Command) -> Self {
        let (k_alignment, storage_bytes) = match command.operands.instruction.element {
            C220Load3dElement::B4 => (64, 1),
            C220Load3dElement::B8 => (32, 1),
            C220Load3dElement::B16 => (16, 2),
            C220Load3dElement::B32 => (8, 4),
        };
        let extent = command.operands.extent;
        let aligned_k = u32::from(extent.k_length).div_ceil(k_alignment) * k_alignment;
        let aligned_m = u32::from(extent.m_length).div_ceil(16) * 16;
        let (outer_count, inner_count, aligned_m) = if command.disabled.any() {
            (0, 1, aligned_m)
        } else if command.repeat.k_mode {
            (aligned_m / 16, u32::from(command.repeat.count), 16)
        } else {
            (u32::from(command.repeat.count), 1, aligned_m)
        };
        Self {
            command,
            next_index: 0,
            outer_count,
            inner_count,
            aligned_k,
            aligned_m,
            k_alignment,
            storage_bytes,
        }
    }

    fn tile(&self, outer: u32, inner: u32) -> C220Load3dTile {
        let command = self.command;
        let extent = command.operands.extent;
        let instruction = command.operands.instruction;
        let k = self.aligned_k;
        let m = self.aligned_m;
        let bytes = self.storage_bytes;
        let mut block_stride = 0;
        let mut transpose_stride = 0;
        let mut half = false;
        let offset = if instruction.element == C220Load3dElement::B32
            && instruction.destination == C220Load3dDestination::L0b
        {
            let inner_offset = (k.div_ceil(8) * 256).wrapping_mul(inner);
            let outer_offset = (self.inner_count * k)
                .div_ceil(16)
                .wrapping_mul(outer)
                .wrapping_mul(m << 6);
            u64::from(inner_offset) + u64::from(outer_offset)
        } else if instruction.element == C220Load3dElement::B32 && command.transposed() {
            let index = k / self.k_alignment * inner;
            half = index & 1 != 0;
            transpose_stride = u64::from(bytes.wrapping_mul(self.outer_count * 16 * m));
            u64::from((index & 1) << 8)
                + u64::from(((m >> 4) << 10).wrapping_mul(outer))
                + transpose_stride * u64::from(index >> 1)
        } else {
            block_stride = u64::from(m.wrapping_mul(k).wrapping_mul(bytes));
            if instruction.element == C220Load3dElement::B4 {
                block_stride >>= 1;
            }
            if command.transposed() {
                u64::from(bytes.wrapping_mul(outer).wrapping_mul(16 * m))
                    + u64::from(inner * self.outer_count) * block_stride
            } else {
                u64::from(inner + outer * self.inner_count) * block_stride
            }
        };
        let (k_step, m_step, m_length) = if command.repeat.k_mode {
            (
                self.k_alignment * u32::from(command.repeat.stride),
                16,
                if outer + 1 == self.outer_count && extent.m_length & 15 != 0 {
                    u32::from(extent.m_length & 15)
                } else {
                    16
                },
            )
        } else {
            (
                0,
                16 * u32::from(command.repeat.stride),
                u32::from(extent.m_length),
            )
        };
        C220Load3dTile {
            outer_index: outer,
            inner_index: inner,
            outer_count: self.outer_count,
            inner_count: self.inner_count,
            k_start: u32::from(extent.k_start) + inner * k_step,
            m_start: u32::from(extent.m_start) + outer * m_step,
            k_length: u32::from(extent.k_length),
            m_length,
            source_base: command.operands.source_base,
            destination_base: command.operands.destination_base.wrapping_add(offset),
            block_stride_bytes: block_stride,
            transpose_stride_bytes: transpose_stride,
            fp32_transpose_half: half,
        }
    }
}

impl Iterator for C220Load3dTiles {
    type Item = C220Load3dTile;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_index == self.outer_count * self.inner_count {
            return None;
        }
        let index = self.next_index;
        self.next_index += 1;
        Some(self.tile(index / self.inner_count, index % self.inner_count))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = (self.outer_count * self.inner_count - self.next_index) as usize;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for C220Load3dTiles {}
impl std::iter::FusedIterator for C220Load3dTiles {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::c220::mte::load3d::C220Load3dV2Instruction;

    #[test]
    fn repeated_regions_preserve_packing_transpose_and_partial_m() {
        for (opcode, alignment, tile_bytes) in
            [(28, 64, 512), (20, 32, 512), (21, 16, 1024), (29, 8, 1536)]
        {
            let instruction = C220Load3dV2Instruction::decode(
                (3 << 29) | (opcode << 22) | (1 << 17) | (2 << 12) | (3 << 7) | (4 << 2),
            )
            .unwrap();
            let mut registers = [0; 32];
            registers[1] = 4096;
            registers[2] = 8192;
            registers[3] = 17 | (33 << 16) | (5 << 32) | (7 << 48);
            registers[4] = (1 << 12) | (1 << 20) | (64 << 48);
            let mut command = C220Load3dV2Command::capture(instruction, &registers, |r| {
                Some(match r {
                    10 | 92 => 64 | (64 << 16),
                    58 => 3 | (2 << 16) | (1 << 24),
                    _ => 0,
                })
            })
            .unwrap();
            let tiles: Vec<_> = command.tiles().collect();
            assert_eq!(tiles.len(), 6);
            for (index, tile) in tiles.iter().enumerate() {
                assert_eq!(tile.destination_base, 4096 + index as u64 * tile_bytes);
                assert_eq!(tile.k_start, 5 + (index as u32 % 2) * 3 * alignment);
                assert_eq!(tile.m_start, 7 + (index as u32 / 2) * 16);
                assert_eq!(tile.m_length, if index >= 4 { 1 } else { 16 });
            }
            if instruction.element == C220Load3dElement::B32 {
                command.operands.geometry.transpose = true;
                let tiles: Vec<_> = command.tiles().collect();
                assert_eq!(tiles[1].destination_base, 4096 + 256 + 3072);
                assert!(tiles[1].fp32_transpose_half);
                assert_eq!(tiles[2].destination_base, 4096 + 1024);
                command.operands.instruction.destination = C220Load3dDestination::L0b;
                let tiles: Vec<_> = command.tiles().collect();
                assert_eq!(tiles[1].destination_base, 4096 + 768);
                assert_eq!(tiles[2].destination_base, 4096 + 3072);
            }
            command.repeat.k_mode = false;
            command.operands.geometry.transpose = false;
            command.operands.instruction.destination = C220Load3dDestination::L0a;
            let tiles: Vec<_> = command.tiles().collect();
            assert_eq!(tiles.len(), 2);
            assert_eq!(tiles[1].m_start, 55);
            assert_eq!(tiles[1].m_length, 33);
            assert_eq!(tiles[1].destination_base, 4096 + tile_bytes * 3);
            command.disabled.zero_repeat = true;
            let mut tiles = command.tiles();
            assert_eq!(tiles.len(), 0);
            assert_eq!(tiles.next(), None);
            assert_eq!(tiles.next(), None);
        }
    }
}
