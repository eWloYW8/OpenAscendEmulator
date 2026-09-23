use std::num::NonZeroU32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FixpExternalOutputPolicy {
    /// Preferred large, medium, and minimum transaction sizes.
    pub burst_sizes: [NonZeroU32; 3],
    pub burst_control: u64,
    pub row_stride_bytes: u32,
}

impl C220FixpExternalOutputPolicy {
    /// Construct the C220 packet policy from the FIX burst-control word and
    /// the destination row stride. The control word is not the BIU mode word.
    pub const fn new(burst_control: u64, row_stride_bytes: u32) -> Self {
        Self {
            burst_sizes: [
                NonZeroU32::new(512).unwrap(),
                NonZeroU32::new(256).unwrap(),
                NonZeroU32::new(128).unwrap(),
            ],
            burst_control,
            row_stride_bytes,
        }
    }

    fn mode(self) -> u8 {
        if self.burst_control & 1 == 0 {
            0
        } else {
            ((self.burst_control >> 1) & 3) as u8
        }
    }

    pub fn row_packet(self, address: u64, remaining: u32) -> u32 {
        let [large, medium, small] = self.burst_sizes.map(NonZeroU32::get);
        let offset = (address % u64::from(small)) as u32;
        if offset != 0 {
            return remaining.min(small - offset);
        }
        for (size, max_mode) in [(large, 0), (medium, 1), (small, 2)] {
            if self.mode() <= max_mode
                && remaining >= size
                && address.is_multiple_of(u64::from(size))
            {
                return size;
            }
        }
        remaining
    }

    pub fn gathered_packet(self, address: u64, available: u32, closed: bool) -> Option<u32> {
        if closed {
            return Some(self.row_packet(address, available));
        }
        let [large, medium, small] = self.burst_sizes.map(NonZeroU32::get);
        let offset = (address % u64::from(small)) as u32;
        let limit = if offset != 0 {
            (small - offset).min(512)
        } else {
            [(large, 0), (medium, 1), (small, 2)]
                .into_iter()
                .find(|&(size, max_mode)| {
                    size <= 512
                        && self.mode() <= max_mode
                        && address.is_multiple_of(u64::from(size))
                })
                .map_or(512, |(size, _)| size)
        };
        (available >= limit).then_some(limit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closed_bursts_are_not_limited_by_open_burst_capacity() {
        let policy = C220FixpExternalOutputPolicy {
            burst_sizes: [4096, 2048, 1024].map(|n| NonZeroU32::new(n).unwrap()),
            burst_control: 0,
            row_stride_bytes: 0,
        };
        assert_eq!(policy.gathered_packet(64, 960, true), Some(960));
        assert_eq!(policy.gathered_packet(64, 960, false), Some(512));
        assert_eq!(policy.gathered_packet(64, 511, false), None);
        assert_eq!(policy.gathered_packet(64, 511, true), Some(511));
        assert_eq!(policy.gathered_packet(0, 4096, true), Some(4096));
        assert_eq!(policy.gathered_packet(0, 4096, false), Some(512));
    }
}
