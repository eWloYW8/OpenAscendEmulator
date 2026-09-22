use super::C220_MOV_UB_TO_OUT_UNIT_BYTES;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct BurstLayout {
    pub count: u16,
    pub length: u16,
    pub source_gap: u16,
    pub destination_gap: u16,
}

pub(super) struct BurstSegment {
    pub burst_index: u16,
    pub unit_index: u16,
    pub source: u64,
    pub destination: u64,
    pub bytes: u32,
}

impl BurstLayout {
    pub(super) fn decode(xm: u64) -> Self {
        Self {
            count: ((xm >> 4) & 0xfff) as u16,
            length: (xm >> 16) as u16,
            source_gap: (xm >> 32) as u16,
            destination_gap: (xm >> 48) as u16,
        }
    }

    pub(super) fn segments(
        self,
        source: u64,
        destination: u64,
    ) -> Option<impl ExactSizeIterator<Item = BurstSegment> + Clone> {
        let units = usize::from(self.count) * usize::from(self.length);
        if units != 0 {
            for burst in 0..self.count {
                self.source_address(source, burst, self.length - 1)?;
            }
            self.address(
                destination,
                self.count - 1,
                self.length - 1,
                self.destination_gap,
            )?;
        }
        Some((0..units).map(move |index| {
            let burst_index = (index / usize::from(self.length)) as u16;
            let unit_index = (index % usize::from(self.length)) as u16;
            BurstSegment {
                burst_index,
                unit_index,
                source: self
                    .source_address(source, burst_index, unit_index)
                    .expect("validated extent"),
                destination: self
                    .address(destination, burst_index, unit_index, self.destination_gap)
                    .expect("validated extent"),
                bytes: C220_MOV_UB_TO_OUT_UNIT_BYTES as u32,
            }
        }))
    }

    fn address(self, base: u64, burst: u16, unit: u16, gap: u16) -> Option<u64> {
        let offset = (u64::from(burst) * (u64::from(self.length) + u64::from(gap))
            + u64::from(unit))
            * C220_MOV_UB_TO_OUT_UNIT_BYTES;
        let address = base.checked_add(offset)?;
        address.checked_add(C220_MOV_UB_TO_OUT_UNIT_BYTES)?;
        Some(address)
    }

    fn source_address(self, base: u64, burst: u16, unit: u16) -> Option<u64> {
        let stride = (u32::from(self.length) + u32::from(self.source_gap)) * 32;
        let offset = u64::from(u32::from(burst).wrapping_mul(stride)) + u64::from(unit) * 32;
        let address = base.checked_add(offset)?;
        address.checked_add(C220_MOV_UB_TO_OUT_UNIT_BYTES)?;
        Some(address)
    }
}
