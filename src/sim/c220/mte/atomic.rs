use crate::memory::mapped::{MappedMemory, MappedMemoryError};
use crate::memory::sparse::MemoryByteState;
use crate::memory::ub::UbTransferResult;
use crate::sim::c220::numeric::atomic::combine_atomic;
use crate::sim::c220::numeric::fp16::C220Fp16AddRounding;

/// Model-level atomic controls, independent of the instruction's captured CTRL.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct C220AtomicConfig {
    pub enabled: bool,
    pub fp16_rounding: C220Fp16AddRounding,
}

/// Publishes nonoverlapping output bursts only after all reads and bounds checks succeed.
pub(crate) fn commit_output(
    writes: &[(u64, Vec<MemoryByteState>)],
    destination: &mut MappedMemory,
    control: u64,
    config: C220AtomicConfig,
    mut result: UbTransferResult,
) -> Result<UbTransferResult, MappedMemoryError> {
    let operation = ((control >> 9) & 3) as u8;
    let data_type = ((control >> 6) & 7) as u8;
    if !config.enabled || operation == 3 {
        destination.write_segments_at(writes)?;
        return Ok(result);
    }
    let width = match data_type {
        1 | 4 => 4,
        2 | 3 | 6 => 2,
        5 => 1,
        _ => 0,
    };
    let mut combined = Vec::with_capacity(writes.len());
    result.known_bytes = 0;
    for (address, states) in writes {
        let previous = destination.read_states_at(*address, states.len())?;
        let mut output = states.clone();
        if width != 0 {
            for (next, old) in output
                .chunks_exact_mut(width)
                .zip(previous.chunks_exact(width))
            {
                let mut next_bytes = [0; 4];
                let mut old_bytes = [0; 4];
                let mut known = true;
                for (index, (&new, &prior)) in next.iter().zip(old).enumerate() {
                    if let (MemoryByteState::Known(new), MemoryByteState::Known(prior)) =
                        (new, prior)
                    {
                        next_bytes[index] = new;
                        old_bytes[index] = prior;
                    } else {
                        known = false;
                    }
                }
                if known {
                    combine_atomic(
                        &mut next_bytes[..width],
                        &old_bytes[..width],
                        data_type,
                        operation,
                        control,
                        config.fp16_rounding,
                    );
                    for (state, byte) in next.iter_mut().zip(next_bytes) {
                        *state = MemoryByteState::Known(byte);
                    }
                } else {
                    next.fill(MemoryByteState::Unknown);
                }
            }
        }
        result.known_bytes += output
            .iter()
            .filter(|state| matches!(state, MemoryByteState::Known(_)))
            .count();
        combined.push((*address, output));
    }
    result.unknown_bytes = result.bytes - result.known_bytes;
    destination.write_segments_at(&combined)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{region::MemoryRegion, sparse::SparseMemory};

    #[test]
    fn signed_lanes_unknowns_partial_tail_and_failed_commit() {
        use MemoryByteState::{Known, Unknown};
        let writes = vec![(
            0x1000,
            vec![Known(0xff), Known(0xff), Known(1), Known(0), Known(9)],
        )];
        let result = UbTransferResult {
            segment_count: 1,
            bytes: 5,
            known_bytes: 5,
            unknown_bytes: 0,
        };
        for enabled in [false, true] {
            for (operation, value) in [(0, 1_i16), (1, 2), (2, -1), (3, -1)] {
                let mut memory = MappedMemory::bind(
                    SparseMemory::new(vec![MemoryRegion::unknown(16)], 16, 16),
                    &[0x1000],
                )
                .unwrap();
                memory.write_known_at(0x1000, &2_i16.to_le_bytes()).unwrap();
                memory.write_known_at(0x1002, &[2]).unwrap();
                let config = C220AtomicConfig {
                    enabled,
                    ..Default::default()
                };
                let control = (3 << 6) | (operation << 9);
                let stored = commit_output(&writes, &mut memory, control, config, result).unwrap();
                let expected = if enabled { value } else { -1 };
                assert_eq!(
                    memory.read_known_at(0x1000, 2).unwrap(),
                    expected.to_le_bytes()
                );
                assert_eq!(memory.read_known_at(0x1004, 1).unwrap(), [9]);
                if enabled && operation != 3 {
                    assert_eq!(memory.read_states_at(0x1002, 2).unwrap(), [Unknown; 2]);
                    assert_eq!(stored.unknown_bytes, 2);
                } else {
                    assert_eq!(memory.read_known_at(0x1002, 2).unwrap(), [1, 0]);
                    assert_eq!(stored.unknown_bytes, 0);
                }
                let before = memory.read_states_at(0x1000, 16).unwrap();
                let invalid = vec![writes[0].clone(), (0x1010, vec![Known(1)])];
                assert!(
                    commit_output(
                        &invalid,
                        &mut memory,
                        control,
                        config,
                        UbTransferResult {
                            segment_count: 2,
                            bytes: 6,
                            known_bytes: 6,
                            unknown_bytes: 0
                        }
                    )
                    .is_err()
                );
                assert_eq!(memory.read_states_at(0x1000, 16).unwrap(), before);
            }
        }
    }
}
