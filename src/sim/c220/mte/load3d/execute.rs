use super::{C220Load3dCoordinateError, C220Load3dTile, C220Load3dV2Command};
use crate::isa::c220::mte::load3d::{C220Load3dDestination, C220Load3dElement};
use crate::sim::c220::memory::{C220LocalBufferError, C220LocalMemory};

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum C220Load3dExecutionError {
    #[error(transparent)]
    Coordinates(#[from] C220Load3dCoordinateError),
    #[error(transparent)]
    Memory(#[from] C220LocalBufferError),
    #[error("LOAD3D channel fragment exceeds its packet bounds")]
    FragmentBounds,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::c220::mte::load3d::C220Load3dV2Instruction;
    use crate::sim::c220::memory::C220LocalMemoryConfig;

    #[test]
    fn functional_load_packs_tail_channels_and_preserves_padding() {
        let instruction = C220Load3dV2Instruction::decode(
            (3 << 29) | (20 << 22) | (1 << 17) | (2 << 12) | (3 << 7) | (4 << 2),
        )
        .unwrap();
        let mut registers = [0; 32];
        registers[1] = 1024;
        registers[3] = 40 | (16 << 16);
        registers[4] = (1 << 12) | (1 << 20) | (40 << 48);
        let mut command = C220Load3dV2Command::capture(instruction, &registers, |r| {
            Some(match r {
                10 | 92 => 4 | (4 << 16),
                58 => 1 << 16,
                13 => 0xbbaa,
                _ => 0,
            })
        })
        .unwrap();
        let mut memory = C220LocalMemory::new(C220LocalMemoryConfig::default()).unwrap();
        let input: Vec<u8> = (0..640).map(|index| (index % 251) as u8).collect();
        memory.l1_mut().write_known(0, &input).unwrap();
        let report = command.execute(&mut memory).unwrap();
        assert_eq!(report.spr54, Some(40));
        assert_eq!(
            (report.tiles, report.coordinates, report.output_packets),
            (1, 32, 2)
        );
        assert_eq!((report.read_bytes, report.write_bytes), (640, 1024));
        assert_eq!(memory.l0a().read_known(1024, 512).unwrap(), input[..512]);
        let tail = memory.l0a().read_known(1536, 512).unwrap();
        for point in 0..16 {
            assert_eq!(
                tail[point * 32..point * 32 + 8],
                input[512 + point * 8..520 + point * 8]
            );
            assert!(
                tail[point * 32 + 8..point * 32 + 32]
                    .iter()
                    .all(|&byte| byte == 0)
            );
        }
        command.matrix.pad_left = 1;
        command.execute(&mut memory).unwrap();
        assert_eq!(
            memory.l0a().read_known(1024, 32).unwrap(),
            [0xaa, 0xbb].repeat(16)
        );
        assert_eq!(
            memory.l0a().read_known(1536, 8).unwrap(),
            [0xaa, 0xbb].repeat(4)
        );
        command.disabled.zero_repeat = true;
        assert_eq!(
            command.execute(&mut memory).unwrap(),
            C220Load3dExecutionReport::default()
        );
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct C220Load3dExecutionReport {
    pub tiles: u64,
    pub coordinates: u64,
    pub read_operations: u64,
    pub read_bytes: u64,
    pub output_packets: u64,
    pub write_operations: u64,
    pub write_bytes: u64,
    pub spr54: Option<u64>,
}

impl C220Load3dV2Command {
    /// Functional execution only; the report does not assign issue or retirement ticks.
    pub fn execute(
        self,
        memory: &mut C220LocalMemory,
    ) -> Result<C220Load3dExecutionReport, C220Load3dExecutionError> {
        let mut report = C220Load3dExecutionReport::default();
        for tile in self.tiles() {
            self.execute_tile(tile, memory, &mut report)?;
            report.tiles += 1;
        }
        Ok(report)
    }

    fn execute_tile(
        self,
        tile: C220Load3dTile,
        memory: &mut C220LocalMemory,
        report: &mut C220Load3dExecutionReport,
    ) -> Result<(), C220Load3dExecutionError> {
        let mut coordinates = self.coordinates(tile)?;
        let bytes = match self.operands.instruction.element {
            C220Load3dElement::B4 | C220Load3dElement::B8 => 1,
            C220Load3dElement::B16 => 2,
            C220Load3dElement::B32 => 4,
        };
        let geometry = self.effective_geometry();
        let valid_k = u32::from(geometry.filter_w)
            .wrapping_mul(u32::from(geometry.filter_h))
            .wrapping_mul(u32::from(geometry.channel_size))
            .wrapping_sub(u32::from(self.operands.extent.k_start)) as u16;
        report.spr54 = Some(u64::from(valid_k.min(self.operands.extent.k_length)));
        let initial_skip = coordinates.initial_channel_offset() as usize * bytes;
        let mut first_fragment = true;
        let mut current = [0; 512];
        let mut overflow = [0; 512];
        let mut filled = 0;
        let mut packet_index = 0;
        while let Some(first) = coordinates.next() {
            let count = first.channel_width as usize * bytes;
            let available = if first.last_k {
                first.remaining_bytes as usize
            } else {
                count
            };
            let skip = if first_fragment { initial_skip } else { 0 };
            if count > 32 || available > 32 || skip > available || filled + available - skip > 64 {
                return Err(C220Load3dExecutionError::FragmentBounds);
            }
            let mut scratch = [0; 32];
            for point in 0..16 {
                let coordinate = if point == 0 {
                    first
                } else {
                    coordinates
                        .next()
                        .ok_or(C220Load3dExecutionError::FragmentBounds)?
                };
                report.coordinates += 1;
                if coordinate.invalid_k {
                    scratch.fill(0);
                } else if coordinate.spatial_padding {
                    let pattern = self.padding.to_le_bytes();
                    let pattern_len = if bytes == 4 { 4 } else { 2 };
                    for index in 0..count {
                        scratch[index] = pattern[index % pattern_len];
                    }
                    if count & 1 != 0 && count < 32 {
                        scratch[count] = 0;
                    }
                } else {
                    let address = coordinate.source_address;
                    let prefix = (self.effective_l1_size() - address).min(count as u64) as usize;
                    if prefix != 0 {
                        scratch[..prefix].copy_from_slice(
                            &memory.l1().read_initialized_linear(address, prefix)?,
                        );
                        report.read_operations += 1;
                        report.read_bytes += prefix as u64;
                    }
                    if prefix < count {
                        scratch[prefix..count].copy_from_slice(
                            &memory.l1().read_initialized_linear(0, count - prefix)?,
                        );
                        report.read_operations += 1;
                        report.read_bytes += (count - prefix) as u64;
                    }
                }
                for (index, &value) in scratch[skip..available].iter().enumerate() {
                    let column = filled + index;
                    if column < 32 {
                        current[point * 32 + column] = value;
                    } else {
                        overflow[point * 32 + column - 32] = value;
                    }
                }
            }
            filled += available - skip;
            first_fragment = first.last_k;
            while filled >= 32 {
                self.write_packet(tile, packet_index, &current, memory, report)?;
                packet_index += 1;
                filled -= 32;
                current = overflow;
            }
            if first.last_k && filled != 0 {
                let remainder = first.phase_length as usize % (32 / bytes) * bytes;
                if remainder != 0 {
                    for row in current.chunks_exact_mut(32) {
                        row[remainder..].fill(0);
                    }
                    self.write_packet(tile, packet_index, &current, memory, report)?;
                    packet_index += 1;
                }
                filled = 0;
            }
        }
        Ok(())
    }

    fn write_packet(
        self,
        tile: C220Load3dTile,
        index: u32,
        packet: &[u8; 512],
        memory: &mut C220LocalMemory,
        report: &mut C220Load3dExecutionReport,
    ) -> Result<(), C220Load3dExecutionError> {
        let address = self
            .packet_address(tile, index)
            .ok_or(C220Load3dExecutionError::FragmentBounds)?;
        for write in self.packet_writes(tile, address, packet) {
            let destination = match write.destination {
                C220Load3dDestination::L0a => memory.l0a_mut(),
                C220Load3dDestination::L0b => memory.l0b_mut(),
            };
            destination.write_known_linear(write.address, &write.bytes)?;
            report.write_operations += 1;
            report.write_bytes += write.bytes.len() as u64;
        }
        report.output_packets += 1;
        Ok(())
    }
}
