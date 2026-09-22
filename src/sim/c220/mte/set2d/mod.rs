mod uop;
pub use uop::{C220Set2dBandwidths, C220Set2dOutputRoute, C220Set2dUop, C220Set2dUops};

use crate::isa::c220::mte::set2d::{C220Set2dDestination, C220Set2dFill};
use crate::sim::c220::memory::{C220LocalBufferError, C220LocalMemory};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct C220Set2dResult {
    pub repetitions: u16,
    pub skipped_overflow_repetitions: u16,
    pub bytes: u64,
    pub elements: u64,
}

/// Execute a captured fill at its functional execution point. The pattern is
/// raw register bits: no floating-point conversion or NaN normalization occurs.
pub fn execute_c220_set2d(
    memory: &mut C220LocalMemory,
    fill: C220Set2dFill,
) -> Result<C220Set2dResult, C220LocalBufferError> {
    if fill.descriptor.is_empty() {
        return Ok(C220Set2dResult::default());
    }
    let destination = match fill.instruction.destination {
        C220Set2dDestination::L0a => memory.l0a_mut(),
        C220Set2dDestination::L0b => memory.l0b_mut(),
        C220Set2dDestination::L1 => memory.l1_mut(),
    };
    let element_bytes = fill.instruction.element_format.bytes();
    let bits = fill.pattern_register.to_le_bytes();
    let mut block = [0; 512];
    for element in block.chunks_exact_mut(element_bytes) {
        element.copy_from_slice(&bits[..element_bytes]);
    }
    let mut result = C220Set2dResult::default();
    for segment in fill.segments() {
        if segment
            .destination_address
            .checked_add(u64::from(segment.bytes - 1))
            .is_none()
        {
            result.skipped_overflow_repetitions += 1;
            continue;
        }
        let mut offset = 0_u32;
        while offset < segment.bytes {
            let bytes = (segment.bytes - offset).min(block.len() as u32);
            destination.write_known_linear(
                segment.destination_address + u64::from(offset),
                &block[..bytes as usize],
            )?;
            offset += bytes;
        }
        result.repetitions += 1;
        result.bytes += u64::from(segment.bytes);
    }
    result.elements = result.bytes / element_bytes as u64;
    Ok(result)
}

#[cfg(test)]
mod tests;
