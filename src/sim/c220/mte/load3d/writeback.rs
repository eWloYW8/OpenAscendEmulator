use super::{C220Load3dTile, C220Load3dV2Command};
use crate::isa::c220::mte::load3d::{C220Load3dDestination, C220Load3dElement};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220Load3dWrite {
    pub destination: C220Load3dDestination,
    pub address: u64,
    pub bytes: Vec<u8>,
}

impl C220Load3dV2Command {
    /// Locate a completed 512-byte output packet within a repeated region.
    pub fn packet_address(self, tile: C220Load3dTile, packet_index: u32) -> Option<u64> {
        let instruction = self.operands.instruction;
        let packed = instruction.element == C220Load3dElement::B4;
        let storage_bytes = match instruction.element {
            C220Load3dElement::B4 | C220Load3dElement::B8 => 1,
            C220Load3dElement::B16 => 2,
            C220Load3dElement::B32 => 4,
        };
        let length = tile.k_length >> u32::from(packed);
        let k_blocks = (length as f32 / (32 / storage_bytes) as f32).ceil() as u32;
        if self.disabled.any() || k_blocks == 0 {
            return None;
        }
        let k = packet_index % k_blocks;
        let m = packet_index / k_blocks;
        let offset = if self.transposed() {
            if instruction.element == C220Load3dElement::B32 {
                let adjusted_k = k.wrapping_add(u32::from(tile.fp32_transpose_half));
                return Some(
                    tile.destination_base
                        .wrapping_sub(if tile.fp32_transpose_half { 256 } else { 0 })
                        .wrapping_add(u64::from((adjusted_k & 1) << 8))
                        .wrapping_add(
                            u64::from(adjusted_k >> 1).wrapping_mul(tile.transpose_stride_bytes),
                        )
                        .wrapping_add(u64::from(m << 10)),
                );
            }
            let m_start = (tile.m_start as f32 / 16.0) as u32 as u16 as u32;
            let m_end = (tile.m_start.wrapping_add(tile.m_length) as f32 / 16.0).ceil() as u32
                as u16 as u32;
            u64::from(
                m.wrapping_add(
                    m_end
                        .wrapping_sub(m_start)
                        .wrapping_mul(tile.outer_count)
                        .wrapping_mul(k),
                )
                .wrapping_mul(512),
            )
        } else if instruction.element == C220Load3dElement::B32
            && instruction.destination == C220Load3dDestination::L0b
        {
            let rows = ((length as f32 * 0.0625).ceil() * 2.0) as u32;
            u64::from(k.wrapping_mul(512) >> 1) + u64::from(rows.wrapping_mul(512).wrapping_mul(m))
        } else {
            u64::from(k.wrapping_add(k_blocks.wrapping_mul(m)).wrapping_mul(512))
        };
        Some(tile.destination_base.wrapping_add(offset))
    }

    /// Transform an assembled packet into ordered physical destination writes.
    pub fn packet_writes(
        self,
        tile: C220Load3dTile,
        address: u64,
        packet: &[u8; 512],
    ) -> Vec<C220Load3dWrite> {
        let instruction = self.operands.instruction;
        let destination = instruction.destination;
        let transpose = destination == C220Load3dDestination::L0b || self.transposed();
        let base = address as u32;
        if instruction.element == C220Load3dElement::B32 && transpose {
            let stride = if destination == C220Load3dDestination::L0b {
                let elements = tile.k_length.wrapping_mul(tile.inner_count);
                (((elements as f32 * 0.0625).ceil() * 2.0) as u32) << 8
            } else {
                512
            };
            packet
                .chunks_exact(256)
                .enumerate()
                .map(|(index, bytes)| C220Load3dWrite {
                    destination,
                    address: u64::from(base.wrapping_add((index as u32).wrapping_mul(stride))),
                    bytes: transpose_elements(bytes, instruction.element, 8),
                })
                .collect()
        } else {
            vec![C220Load3dWrite {
                destination,
                address: u64::from(base),
                bytes: if transpose {
                    transpose_elements(packet, instruction.element, 16)
                } else {
                    packet.to_vec()
                },
            }]
        }
    }
}

fn transpose_elements(input: &[u8], element: C220Load3dElement, rows: usize) -> Vec<u8> {
    let bits = match element {
        C220Load3dElement::B4 => 4,
        C220Load3dElement::B8 => 8,
        C220Load3dElement::B16 => 16,
        C220Load3dElement::B32 => 32,
    };
    let elements = input.len() * 8 / bits;
    let columns = elements / rows;
    let mut output = vec![0; input.len()];
    for index in 0..elements {
        let target = index / columns + index % columns * rows;
        if bits == 4 {
            let nibble = (input[index / 2] >> ((index % 2) * 4)) & 15;
            output[target / 2] |= nibble << ((target % 2) * 4);
        } else {
            let bytes = bits / 8;
            output[target * bytes..(target + 1) * bytes]
                .copy_from_slice(&input[index * bytes..(index + 1) * bytes]);
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::c220::mte::load3d::C220Load3dV2Instruction;

    #[test]
    fn packed_transpose_and_fp32_split_writes_preserve_layout() {
        let packet: [u8; 512] = std::array::from_fn(|i| (i % 251) as u8);
        let packed = transpose_elements(&packet, C220Load3dElement::B4, 16);
        for row in 0..16 {
            for column in 0..64 {
                let source = row * 64 + column;
                let target = column * 16 + row;
                assert_eq!(
                    (packed[target / 2] >> (4 * (target % 2))) & 15,
                    (packet[source / 2] >> (4 * (source % 2))) & 15,
                );
            }
        }
        let instruction = C220Load3dV2Instruction::decode(
            (3 << 29) | (29 << 22) | (1 << 17) | (2 << 12) | (3 << 7) | (4 << 2),
        )
        .unwrap();
        let mut registers = [0; 32];
        registers[1] = 4096;
        registers[3] = 17 | (33 << 16);
        registers[4] = (1 << 12) | (1 << 20) | (1 << 46) | (64 << 48);
        let mut command = C220Load3dV2Command::capture(instruction, &registers, |r| {
            Some(match r {
                10 | 92 => 64 | (64 << 16),
                58 => 1 | (2 << 16) | (1 << 24),
                _ => 0,
            })
        })
        .unwrap();
        let tile = command.tiles().next().unwrap();
        assert_eq!(command.packet_address(tile, 0), Some(4096));
        assert_eq!(command.packet_address(tile, 1), Some(4352));
        assert_eq!(command.packet_address(tile, 2), Some(7168));
        assert_eq!(command.packet_address(tile, 3), Some(5120));
        let writes = command.packet_writes(tile, 4096, &packet);
        assert_eq!(writes.len(), 2);
        assert_eq!((writes[0].address, writes[1].address), (4096, 4608));
        for (half, write) in writes.iter().enumerate() {
            for row in 0..8 {
                for column in 0..8 {
                    let source = half * 256 + (row * 8 + column) * 4;
                    let target = (column * 8 + row) * 4;
                    assert_eq!(write.bytes[target..target + 4], packet[source..source + 4]);
                }
            }
        }
        command.operands.instruction.destination = C220Load3dDestination::L0b;
        let tile = command.tiles().next().unwrap();
        let writes = command.packet_writes(tile, 4096, &packet);
        assert_eq!(writes[1].address, 4096 + 1536);
        assert_eq!(command.packet_address(tile, 3), Some(4096 + 2048));
        command.operands.instruction.destination = C220Load3dDestination::L0a;
        command.operands.geometry.transpose = false;
        let writes = command.packet_writes(tile, (1 << 32) + 4096, &packet);
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].address, 4096);
        assert_eq!(writes[0].bytes, packet);
    }
}
