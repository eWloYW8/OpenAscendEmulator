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
        self.segments_with_strides(
            source,
            destination,
            C220_MOV_UB_TO_OUT_UNIT_BYTES as u32,
            (u32::from(self.length) + u32::from(self.source_gap)) * 32,
            (u64::from(self.length) + u64::from(self.destination_gap)) * 32,
        )
    }

    pub(super) fn segments_with_strides(
        self,
        source: u64,
        destination: u64,
        unit_bytes: u32,
        source_stride: u32,
        destination_stride: u64,
    ) -> Option<impl ExactSizeIterator<Item = BurstSegment> + Clone> {
        let units = usize::from(self.count) * usize::from(self.length);
        let source_at = move |burst: u16, unit: u16| {
            let offset = u64::from(u32::from(burst).wrapping_mul(source_stride))
                + u64::from(unit) * u64::from(unit_bytes);
            let address = source.checked_add(offset)?;
            address.checked_add(u64::from(unit_bytes))?;
            Some(address)
        };
        let destination_at = move |burst: u16, unit: u16| {
            let offset =
                u64::from(burst) * destination_stride + u64::from(unit) * u64::from(unit_bytes);
            let address = destination.checked_add(offset)?;
            address.checked_add(u64::from(unit_bytes))?;
            Some(address)
        };
        if units != 0 {
            for burst in 0..self.count {
                source_at(burst, self.length - 1)?;
            }
            destination_at(self.count - 1, self.length - 1)?;
        }
        Some((0..units).map(move |index| {
            let burst_index = (index / usize::from(self.length)) as u16;
            let unit_index = (index % usize::from(self.length)) as u16;
            BurstSegment {
                burst_index,
                unit_index,
                source: source_at(burst_index, unit_index).expect("validated extent"),
                destination: destination_at(burst_index, unit_index).expect("validated extent"),
                bytes: unit_bytes,
            }
        }))
    }
}
