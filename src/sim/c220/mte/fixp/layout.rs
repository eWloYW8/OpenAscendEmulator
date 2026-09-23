use crate::isa::c220::mte::fixp::C220FixpDescriptor;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FixpSlice {
    pub nd_index: u16,
    pub row: u16,
    pub column_block: u16,
    pub source_address: u64,
    pub destination_address: u64,
    pub lanes: u8,
}

impl C220FixpSlice {
    pub const fn source_bytes(self) -> u32 {
        self.lanes as u32 * 4
    }
    pub const fn destination_bytes(self) -> u32 {
        self.lanes as u32 * 2
    }

    /// PReLU reads a full 16-lane slope block, including for a partial ND tail.
    pub const fn slope_address(self, base_block: u8) -> u32 {
        2048 + 64 * (self.column_block as u32 + base_block as u32)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum C220FixpLayoutError {
    #[error("FIX FP16 layout requires conversion mode 1, got {0}")]
    ConversionMode(u8),
}

/// Lazy coordinate generation for FP32-to-FP16 FIX, independent of transport.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FixpFp16Layout {
    descriptor: C220FixpDescriptor,
    source: u64,
    destination: u64,
}

impl C220FixpFp16Layout {
    pub fn new(
        descriptor: C220FixpDescriptor,
        source: u64,
        destination: u64,
    ) -> Result<Self, C220FixpLayoutError> {
        if descriptor.conversion_mode() != 1 {
            return Err(C220FixpLayoutError::ConversionMode(
                descriptor.conversion_mode(),
            ));
        }
        Ok(Self {
            descriptor,
            source,
            destination,
        })
    }

    pub fn slices(self) -> impl Iterator<Item = C220FixpSlice> + Clone {
        let d = self.descriptor;
        let blocks = u32::from(d.columns()).div_ceil(16);
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
                    let source_block = block
                        .wrapping_mul(64)
                        .wrapping_mul(u32::from(d.source_stride()));
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
                                    .wrapping_mul(2),
                            ))
                            .wrapping_add(u64::from(
                                u32::from(row)
                                    .wrapping_mul(d.destination_stride())
                                    .wrapping_mul(2),
                            ))
                            .wrapping_add(u64::from(block) * 32)
                    } else {
                        self.destination
                            .wrapping_add(u64::from(row) * 32)
                            .wrapping_add(
                                u64::from(block)
                                    * u64::from(d.destination_stride().wrapping_mul(32)),
                            )
                    };
                    let remainder = (d.columns() & 15) as u8;
                    let lanes = if d.nz_to_nd() && block + 1 == blocks && remainder != 0 {
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
        let ordinary: Vec<_> = C220FixpFp16Layout::new(d, 100, 200)
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
        let nd: Vec<_> = C220FixpFp16Layout::new(d, 100, 200)
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
        d.nd = 0;
        assert_eq!(
            C220FixpFp16Layout::new(d, 100, 200)
                .unwrap()
                .slices()
                .count(),
            0
        );
    }
}
