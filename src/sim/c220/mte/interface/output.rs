use std::iter::FusedIterator;
use std::num::NonZeroU32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MteOutputFragment {
    pub instruction_id: u64,
    pub request_id: u64,
    pub destination_address: u64,
    pub bytes: u32,
    pub last_in_uop: bool,
    pub last_in_instruction: bool,
}

/// Lazy output fragments for one completed logical L1 read. Queue readiness,
/// target credits and retirement remain the responsibility of the interface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220MteOutputPlan {
    instruction_id: u64,
    request_id: u64,
    destination_address: u64,
    bytes: u32,
    last_in_instruction: bool,
    bandwidth: NonZeroU32,
    offset: u32,
}

impl C220MteOutputPlan {
    pub fn new(
        instruction_id: u64,
        request_id: u64,
        destination_address: u64,
        bytes: u32,
        last_in_instruction: bool,
        bandwidth: NonZeroU32,
    ) -> Self {
        Self {
            instruction_id,
            request_id,
            destination_address,
            bytes,
            last_in_instruction,
            bandwidth,
            offset: 0,
        }
    }

    /// Inspect the next fragment without consuming downstream credit.
    pub fn front(&self) -> Option<C220MteOutputFragment> {
        let remaining = self.bytes - self.offset;
        if remaining == 0 {
            return None;
        }
        let bytes = remaining.min(self.bandwidth.get());
        let last_in_uop = bytes == remaining;
        Some(C220MteOutputFragment {
            instruction_id: self.instruction_id,
            request_id: self.request_id,
            destination_address: self
                .destination_address
                .wrapping_add(u64::from(self.offset)),
            bytes,
            last_in_uop,
            last_in_instruction: last_in_uop && self.last_in_instruction,
        })
    }
}

impl Iterator for C220MteOutputPlan {
    type Item = C220MteOutputFragment;

    fn next(&mut self) -> Option<Self::Item> {
        let fragment = self.front()?;
        self.offset += fragment.bytes;
        Some(fragment)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = (self.bytes - self.offset).div_ceil(self.bandwidth.get()) as usize;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for C220MteOutputPlan {}
impl FusedIterator for C220MteOutputPlan {}
