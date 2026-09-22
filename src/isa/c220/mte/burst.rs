use super::{C220_MOV_UB_TO_OUT_UNIT_BYTES, MAX_C220_DMAMOV_SEGMENTS};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct BurstLayout {
    pub count: u16,
    pub length: u16,
    pub source_gap: u16,
    pub destination_gap: u16,
}

pub(super) enum BurstSizeError {
    Empty,
    TooMany { requested: u64, limit: u64 },
}

pub(super) struct BurstSegment {
    pub burst_index: u16,
    pub unit_index: u16,
    pub source: u64,
    pub destination: u64,
    pub bytes: u32,
}

impl BurstLayout {
    pub(super) fn decode(xm: u64) -> Result<Self, BurstSizeError> {
        let layout = Self {
            count: ((xm >> 4) & 0xfff) as u16,
            length: (xm >> 16) as u16,
            source_gap: (xm >> 32) as u16,
            destination_gap: (xm >> 48) as u16,
        };
        if layout.count == 0 || layout.length == 0 {
            return Err(BurstSizeError::Empty);
        }
        let requested = u64::from(layout.count) * u64::from(layout.length);
        if requested > MAX_C220_DMAMOV_SEGMENTS {
            return Err(BurstSizeError::TooMany {
                requested,
                limit: MAX_C220_DMAMOV_SEGMENTS,
            });
        }
        Ok(layout)
    }

    pub(super) fn segments(
        self,
        source: u64,
        destination: u64,
    ) -> impl Iterator<Item = Option<BurstSegment>> {
        (0..self.count).flat_map(move |burst_index| {
            (0..self.length).map(move |unit_index| {
                Some(BurstSegment {
                    burst_index,
                    unit_index,
                    source: self.address(source, burst_index, unit_index, self.source_gap)?,
                    destination: self.address(
                        destination,
                        burst_index,
                        unit_index,
                        self.destination_gap,
                    )?,
                    bytes: C220_MOV_UB_TO_OUT_UNIT_BYTES as u32,
                })
            })
        })
    }

    fn address(self, base: u64, burst: u16, unit: u16, gap: u16) -> Option<u64> {
        let offset = (u64::from(burst) * (u64::from(self.length) + u64::from(gap))
            + u64::from(unit))
            * C220_MOV_UB_TO_OUT_UNIT_BYTES;
        let address = base.checked_add(offset)?;
        address.checked_add(C220_MOV_UB_TO_OUT_UNIT_BYTES)?;
        Some(address)
    }
}
