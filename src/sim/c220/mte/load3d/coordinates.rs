use super::{C220Load3dTile, C220Load3dV2Command};
use crate::isa::c220::mte::load3d::C220Load3dElement;

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum C220Load3dCoordinateError {
    #[error("packed LOAD3D requires an even channel count")]
    OddPackedChannels,
    #[error("LOAD3D has no output columns")]
    EmptyOutputWidth,
    #[error("LOAD3D split region has no tail channels")]
    EmptySplitTail,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Load3dCoordinate {
    pub x: i32,
    pub y: i32,
    pub channel_block: u32,
    pub channel_width: u32,
    pub source_base: u64,
    pub source_address: u64,
    pub destination_base: u64,
    pub k_index: u32,
    pub m_chunk: u32,
    pub m_point: u32,
    pub k_iteration: u32,
    pub remaining_bytes: u32,
    pub phase_length: u32,
    pub last_k: bool,
    pub invalid_k: bool,
    pub spatial_padding: bool,
    pub row_wrap: bool,
    pub address_wrap: bool,
}

#[derive(Debug, Clone, Copy, Default)]
struct ChannelPhase {
    source_base: u64,
    width: u32,
    length: u32,
    offset: u32,
    channel: u32,
    filter_x: u32,
    filter_y: u32,
}

/// Lazy source-coordinate stream; padded M chunks retain all sixteen points.
#[derive(Debug, Clone)]
pub struct C220Load3dCoordinates {
    command: C220Load3dV2Command,
    tile: C220Load3dTile,
    phases: [ChannelPhase; 2],
    phase_count: usize,
    phase_index: usize,
    phase: ChannelPhase,
    storage_bytes: u32,
    block_width: u32,
    total_k: u32,
    m_chunks: u32,
    m_chunk: u32,
    m_point: u32,
    k_index: u32,
    k: i32,
    initial_x: i32,
    initial_y: i32,
    x: i32,
    y: i32,
    right_edge: i32,
    pub output_width: u32,
    pub output_height: u32,
    pub used_empty_tail_fallback: bool,
}

impl C220Load3dCoordinates {
    pub(super) fn initial_channel_offset(&self) -> u32 {
        self.phases[0].offset
    }

    pub(super) fn new(
        command: C220Load3dV2Command,
        tile: C220Load3dTile,
    ) -> Result<Self, C220Load3dCoordinateError> {
        if command.odd_packed_channels && !command.disabled.any() {
            return Err(C220Load3dCoordinateError::OddPackedChannels);
        }
        let geometry = command.effective_geometry();
        let matrix = command.matrix;
        let storage_bytes = match command.operands.instruction.element {
            C220Load3dElement::B4 | C220Load3dElement::B8 => 1,
            C220Load3dElement::B16 => 2,
            C220Load3dElement::B32 => 4,
        };
        let packed = command.operands.instruction.element == C220Load3dElement::B4;
        let block_width = 32 / storage_bytes;
        let channels = u32::from(geometry.channel_size) >> u32::from(packed);
        let length = tile.k_length >> u32::from(packed);
        let start = (tile.k_start as f32 * if packed { 0.5 } else { 1.0 } / block_width as f32)
            as u32 as u16 as u32;
        let end = ((length.wrapping_add(tile.k_start >> u32::from(packed))) as f32
            / block_width as f32)
            .ceil() as u32 as u16 as u32;
        let fw = u32::from(geometry.filter_w);
        let fh = u32::from(geometry.filter_h);
        let full_blocks = channels / block_width * fw * fh;
        let tail_width = channels % block_width;
        let full = ChannelPhase {
            source_base: tile.source_base,
            width: block_width,
            length,
            channel: start / (fw * fh),
            filter_x: start % (fw * fh) % fw,
            filter_y: start % (fw * fh) / fw,
            ..ChannelPhase::default()
        };
        let tail_base = tile.source_base.wrapping_add(u64::from(
            (32 * (full_blocks / fw / fh))
                .wrapping_mul(u32::from(matrix.width))
                .wrapping_mul(u32::from(matrix.height)),
        ));
        let mut phases = [full, ChannelPhase::default()];
        let mut phase_count = 1;
        let mut used_empty_tail_fallback = false;
        if full_blocks < end {
            if full_blocks <= start {
                let width = if tail_width == 0 {
                    used_empty_tail_fallback = true;
                    8
                } else {
                    tail_width
                };
                let delta = (start - full_blocks) * block_width;
                let filter = delta / width;
                phases[0] = ChannelPhase {
                    source_base: tail_base,
                    width,
                    length,
                    offset: delta % width,
                    channel: 0,
                    filter_x: filter % fw,
                    filter_y: filter / fw,
                };
            } else {
                if tail_width == 0 {
                    return Err(C220Load3dCoordinateError::EmptySplitTail);
                }
                phases[0].length = (full_blocks - start) * block_width;
                phases[1] = ChannelPhase {
                    source_base: tail_base,
                    width: tail_width,
                    length: length.wrapping_sub(phases[0].length),
                    ..ChannelPhase::default()
                };
                phase_count = 2;
            }
        }
        let filter_width = (fw - 1) * u32::from(geometry.dilation_w) + 1;
        let filter_height = (fh - 1) * u32::from(geometry.dilation_h) + 1;
        let padded_width =
            u32::from(matrix.width) + u32::from(matrix.pad_left) + u32::from(matrix.pad_right);
        let padded_height =
            u32::from(matrix.height) + u32::from(matrix.pad_top) + u32::from(matrix.pad_bottom);
        let output_width = (padded_width.wrapping_sub(filter_width) as f32
            / f32::from(geometry.stride_w)
            + 1.0) as u32;
        let output_height = (padded_height.wrapping_sub(filter_height) as f32
            / f32::from(geometry.stride_h)
            + 1.0) as u32;
        if output_width == 0 && !command.disabled.any() {
            return Err(C220Load3dCoordinateError::EmptyOutputWidth);
        }
        let initial_x = (tile.m_start % output_width.max(1))
            .wrapping_mul(u32::from(geometry.stride_w))
            .wrapping_sub(u32::from(matrix.pad_left)) as i32;
        let initial_y = -(f32::from(matrix.pad_top)
            - f32::from(geometry.stride_h)
                * (tile.m_start as f32 / output_width.max(1) as f32).trunc());
        let initial_y = initial_y as i32;
        Ok(Self {
            command,
            tile,
            phases,
            phase_count,
            phase_index: 0,
            phase: phases[0],
            storage_bytes,
            block_width,
            total_k: fw.wrapping_mul(fh).wrapping_mul(channels),
            m_chunks: if command.disabled.any() || length == 0 {
                0
            } else {
                tile.m_length.div_ceil(16)
            },
            m_chunk: 0,
            m_point: 0,
            k_index: 0,
            k: -(phases[0].offset as i32),
            initial_x,
            initial_y,
            x: initial_x,
            y: initial_y,
            right_edge: u32::from(matrix.width)
                .wrapping_add(u32::from(matrix.pad_right))
                .wrapping_sub(filter_width) as i32,
            output_width,
            output_height,
            used_empty_tail_fallback,
        })
    }

    fn next_k(&mut self) {
        let geometry = self.command.effective_geometry();
        self.k_index = self.k_index.wrapping_add(1);
        self.k = self.k.wrapping_add(self.phase.width as i32);
        self.phase.filter_x += 1;
        if self.phase.filter_x >= u32::from(geometry.filter_w) {
            self.phase.filter_x = 0;
            self.phase.filter_y += 1;
            if self.phase.filter_y >= u32::from(geometry.filter_h) {
                self.phase.filter_y = 0;
                self.phase.channel += 1;
            }
        }
        if self.k >= self.phase.length as i32 {
            self.phase_index += 1;
            if self.phase_index == self.phase_count {
                self.phase_index = 0;
                self.m_chunk += 1;
                self.initial_x = self.x;
                self.initial_y = self.y;
            }
            self.phase = self.phases[self.phase_index];
            self.k = -(self.phase.offset as i32);
        }
        self.x = self.initial_x;
        self.y = self.initial_y;
        self.m_point = 0;
    }
}

impl Iterator for C220Load3dCoordinates {
    type Item = C220Load3dCoordinate;

    fn next(&mut self) -> Option<Self::Item> {
        if self.m_chunk >= self.m_chunks {
            return None;
        }
        let geometry = self.command.effective_geometry();
        let matrix = self.command.matrix;
        let x = self
            .x
            .wrapping_add((self.phase.filter_x * u32::from(geometry.dilation_w)) as i32);
        let y = self
            .y
            .wrapping_add((self.phase.filter_y * u32::from(geometry.dilation_h)) as i32);
        let linear = (x as u32).wrapping_add(
            (y as u32)
                .wrapping_add(u32::from(matrix.height).wrapping_mul(self.phase.channel))
                .wrapping_mul(u32::from(matrix.width)),
        );
        let source_address = u64::from(
            (self.phase.source_base as u32).wrapping_add(
                linear
                    .wrapping_mul(self.storage_bytes)
                    .wrapping_mul(self.phase.width),
            ),
        ) % self.command.effective_l1_size();
        self.x = self.x.wrapping_add(i32::from(geometry.stride_w));
        let row_wrap = self.x > self.right_edge;
        if row_wrap {
            self.x = -i32::from(matrix.pad_left);
            self.y = self.y.wrapping_add(i32::from(geometry.stride_h));
        }
        let next_k = self.phase.width.wrapping_add(self.k as u32);
        let coordinate = C220Load3dCoordinate {
            x,
            y,
            channel_block: self.phase.channel,
            channel_width: self.phase.width,
            source_base: self.phase.source_base,
            source_address,
            destination_base: self.tile.destination_base,
            k_index: self.k_index,
            m_chunk: self.m_chunk,
            m_point: self.m_point,
            k_iteration: self.k as u32 / self.phase.width,
            remaining_bytes: self
                .phase
                .length
                .wrapping_sub(self.k as u32)
                .wrapping_mul(self.storage_bytes),
            phase_length: self.phase.length,
            last_k: next_k as i32
                >= self
                    .phase
                    .length
                    .wrapping_mul(self.storage_bytes)
                    .wrapping_div(self.storage_bytes) as i32,
            invalid_k: next_k > self.total_k
                || (self.phase.channel > 0 && self.block_width != self.phase.width),
            spatial_padding: x < 0
                || y < 0
                || x >= i32::from(matrix.width)
                || y >= i32::from(matrix.height),
            row_wrap,
            address_wrap: source_address + 32 > self.command.effective_l1_size(),
        };
        self.m_point += 1;
        if self.m_point == 16 {
            self.next_k();
        }
        Some(coordinate)
    }
}

impl std::iter::FusedIterator for C220Load3dCoordinates {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::c220::mte::load3d::C220Load3dV2Instruction;

    #[test]
    fn coordinates_split_channels_and_restart_k_for_each_m_chunk() {
        let instruction = C220Load3dV2Instruction::decode(
            (3 << 29) | (20 << 22) | (1 << 17) | (2 << 12) | (3 << 7) | (4 << 2),
        )
        .unwrap();
        let mut registers = [0; 32];
        registers[1] = 2048;
        registers[3] = 40 | (17 << 16);
        registers[4] = (1 << 12) | (1 << 20) | (40 << 48);
        let command = C220Load3dV2Command::capture(instruction, &registers, |r| {
            Some(match r {
                10 | 92 => 4 | (4 << 16),
                58 => 1 << 16,
                _ => 0,
            })
        })
        .unwrap();
        let tile = command.tiles().next().unwrap();
        let coordinates: Vec<_> = command.coordinates(tile).unwrap().collect();
        assert_eq!(coordinates.len(), 64);
        assert_eq!(
            (
                coordinates[0].x,
                coordinates[0].y,
                coordinates[0].channel_width
            ),
            (0, 0, 32)
        );
        assert_eq!(coordinates[3].source_address, 96);
        assert!(coordinates[3].row_wrap);
        assert_eq!(
            (
                coordinates[16].x,
                coordinates[16].y,
                coordinates[16].channel_width
            ),
            (0, 0, 8)
        );
        assert_eq!(coordinates[16].source_base, 512);
        assert_eq!(coordinates[17].source_address, 520);
        assert_eq!(
            (
                coordinates[32].x,
                coordinates[32].y,
                coordinates[32].k_index
            ),
            (0, 4, 2)
        );
        assert!(coordinates[32].spatial_padding);
        assert_eq!(coordinates[48].source_base, 512);
        assert!(coordinates.iter().all(|c| c.last_k && !c.invalid_k));

        let mut tail = tile;
        tail.k_start = 32;
        tail.k_length = 8;
        tail.m_length = 1;
        let coordinates: Vec<_> = command.coordinates(tail).unwrap().collect();
        assert_eq!(coordinates.len(), 16);
        assert_eq!(coordinates[0].source_address, 512);
        let mut wrap = command;
        wrap.l1_size = 520;
        let first = wrap.coordinates(tail).unwrap().next().unwrap();
        assert!(first.address_wrap);
        let mut padded = command;
        padded.matrix.pad_left = 1;
        let first = padded.coordinates(tile).unwrap().next().unwrap();
        assert_eq!(first.x, -1);
        assert!(first.spatial_padding);
        assert_eq!(first.source_address, (1 << 20) - 32);
    }
}
