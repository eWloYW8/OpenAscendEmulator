use super::{C220FixpOutputFormat, C220FixpSourceFormat};
use crate::isa::c220::mte::fixp::C220FixpDescriptor;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FixpSlice {
    pub nd_index: u16,
    pub row: u16,
    pub column_block: u16,
    pub source_address: u64,
    pub destination_address: u64,
    pub lanes: u8,
    pub output_format: C220FixpOutputFormat,
}

impl C220FixpSlice {
    pub const fn source_bytes(self) -> u32 {
        self.lanes as u32 * 4
    }
    pub const fn destination_bytes(self) -> u32 {
        self.lanes as u32 * self.output_format.lane_bytes()
    }

    /// PReLU reads a full 16-lane slope block, including for a partial ND tail.
    pub const fn slope_address(self, base_block: u8) -> u32 {
        2048 + 64 * (self.column_block as u32 + base_block as u32)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum C220FixpLayoutError {
    #[error("unsupported FIX conversion mode {mode} for {source_format:?} source")]
    ConversionMode {
        source_format: C220FixpSourceFormat,
        mode: u8,
    },
}

/// Lazy coordinate generation for 32-bit-source FIX, independent of transport.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FixpLayout {
    descriptor: C220FixpDescriptor,
    source: u64,
    destination: u64,
    format: C220FixpOutputFormat,
}

impl C220FixpLayout {
    pub fn new(
        descriptor: C220FixpDescriptor,
        source_format: C220FixpSourceFormat,
        source: u64,
        destination: u64,
    ) -> Result<Self, C220FixpLayoutError> {
        let format =
            C220FixpOutputFormat::from_conversion_mode(source_format, descriptor.conversion_mode())
                .ok_or(C220FixpLayoutError::ConversionMode {
                    source_format,
                    mode: descriptor.conversion_mode(),
                })?;
        Ok(Self {
            descriptor,
            source,
            destination,
            format,
        })
    }

    pub fn slices(self) -> impl Iterator<Item = C220FixpSlice> + Clone {
        let d = self.descriptor;
        let lane_bytes = self.format.lane_bytes();
        let split = matches!(
            self.format,
            C220FixpOutputFormat::Fp32 | C220FixpOutputFormat::Int32
        ) && d.channel_split()
            && !d.nz_to_nd();
        let blocks = if split {
            u32::from(d.columns() / 16) * 2 + u32::from(d.columns() % 16 == 8)
        } else {
            u32::from(d.columns()).div_ceil(16)
        };
        let nd_count = if d.is_disabled() {
            0
        } else if d.nz_to_nd() {
            d.nd_count()
        } else {
            1
        };
        (0..nd_count).flat_map(move |nd_index| {
            (0..d.rows()).flat_map(move |row| {
                (0..blocks).map(move |block| {
                    let source_nd = u32::from(nd_index)
                        .wrapping_mul(1024)
                        .wrapping_mul(u32::from(d.source_nd_stride()));
                    let source_block = (if split { block / 2 } else { block })
                        .wrapping_mul(64)
                        .wrapping_mul(u32::from(d.source_stride()))
                        .wrapping_add(if split { (block % 2) * 32 } else { 0 });
                    let source_address = self
                        .source
                        .wrapping_add(if d.nz_to_nd() {
                            u64::from(source_nd)
                        } else {
                            0
                        })
                        .wrapping_add(u64::from(row) * 64)
                        .wrapping_add(u64::from(source_block));
                    let destination_address = if d.nz_to_nd() {
                        self.destination
                            .wrapping_add(u64::from(
                                u32::from(nd_index)
                                    .wrapping_mul(d.destination_nd_stride())
                                    .wrapping_mul(lane_bytes),
                            ))
                            .wrapping_add(u64::from(
                                u32::from(row)
                                    .wrapping_mul(d.destination_stride())
                                    .wrapping_mul(lane_bytes),
                            ))
                            .wrapping_add(u64::from(block) * 16 * u64::from(lane_bytes))
                    } else {
                        self.destination
                            .wrapping_add(
                                u64::from(row)
                                    * if split {
                                        32
                                    } else {
                                        16 * u64::from(lane_bytes)
                                    },
                            )
                            .wrapping_add(
                                u64::from(block)
                                    * u64::from(d.destination_stride().wrapping_mul(32)),
                            )
                    };
                    let remainder = (d.columns() & 15) as u8;
                    let lanes = if split {
                        8
                    } else if d.nz_to_nd() && block + 1 == blocks && remainder != 0 {
                        remainder
                    } else {
                        16
                    };
                    C220FixpSlice {
                        nd_index,
                        row,
                        column_block: block as u16,
                        source_address,
                        destination_address,
                        lanes,
                        output_format: self.format,
                    }
                })
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordinary_padding_and_nd_tail_have_distinct_coordinates() {
        let mut d = C220FixpDescriptor {
            xt: (32 << 32) | (2 << 16) | (17 << 4),
            xm: (1 << 34) | 4,
            nd: (128 << 32) | (3 << 16) | 2,
        };
        let ordinary: Vec<_> = C220FixpLayout::new(d, C220FixpSourceFormat::Fp32, 100, 200)
            .unwrap()
            .slices()
            .collect();
        assert_eq!(ordinary.len(), 4);
        assert_eq!(
            (
                ordinary[1].source_address,
                ordinary[1].destination_address,
                ordinary[1].lanes
            ),
            (356, 1224, 16)
        );
        d.xm |= 1 << 43;
        let nd: Vec<_> = C220FixpLayout::new(d, C220FixpSourceFormat::Fp32, 100, 200)
            .unwrap()
            .slices()
            .collect();
        assert_eq!(nd.len(), 8);
        assert_eq!((nd[1].destination_address, nd[1].lanes), (232, 1));
        assert_eq!(
            (nd[4].source_address, nd[4].destination_address),
            (3172, 456)
        );
        assert_eq!(nd[5].slope_address(2), 2240);
        d.xm &= !(31 << 34);
        let fp32_nd: Vec<_> = C220FixpLayout::new(d, C220FixpSourceFormat::Fp32, 100, 200)
            .unwrap()
            .slices()
            .collect();
        assert_eq!(
            (
                fp32_nd[1].destination_address,
                fp32_nd[1].destination_bytes()
            ),
            (264, 4)
        );
        assert_eq!(fp32_nd[4].destination_address, 712);
        let mut split = d;
        split.xt = (32 << 32) | (2 << 16) | (24 << 4);
        split.xm = (split.xm & !(1 << 43)) | (1 << 42);
        let split: Vec<_> = C220FixpLayout::new(split, C220FixpSourceFormat::Fp32, 100, 200)
            .unwrap()
            .slices()
            .collect();
        assert_eq!(split.len(), 6);
        assert_eq!(
            split
                .iter()
                .map(|s| (s.source_address, s.destination_address, s.lanes))
                .collect::<Vec<_>>(),
            [
                (100, 200, 8),
                (132, 1224, 8),
                (356, 2248, 8),
                (164, 232, 8),
                (196, 1256, 8),
                (420, 2280, 8)
            ]
        );
        d.nd = 0;
        assert_eq!(
            C220FixpLayout::new(d, C220FixpSourceFormat::Fp32, 100, 200)
                .unwrap()
                .slices()
                .count(),
            0
        );
    }
}
