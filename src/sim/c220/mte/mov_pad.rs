use crate::isa::c220::mte::mov_pad::C220MovPadTransfer;
use crate::memory::mapped::{MappedMemory, MappedMemoryError};
use crate::memory::sparse::MemoryByteState;
use crate::memory::ub::{UbMemory, UbMemoryError, UbTransferResult};

#[derive(Debug, thiserror::Error)]
pub enum C220MovPadError {
    #[error(transparent)]
    External(#[from] MappedMemoryError),
    #[error(transparent)]
    Local(#[from] UbMemoryError),
    #[error("MOV_PAD transfer direction does not match destination")]
    DirectionMismatch,
}

pub struct C220PreparedMovPad {
    transfer: C220MovPadTransfer,
    writes: Vec<(u64, Vec<MemoryByteState>)>,
    pub result: UbTransferResult,
}

impl C220PreparedMovPad {
    pub fn commit_to_ub(
        &self,
        destination: &mut UbMemory,
    ) -> Result<UbTransferResult, C220MovPadError> {
        if !self.transfer.is_input() {
            return Err(C220MovPadError::DirectionMismatch);
        }
        destination.write_segments(&self.writes)?;
        Ok(self.result)
    }

    pub fn commit_to_external(
        &self,
        destination: &mut MappedMemory,
    ) -> Result<UbTransferResult, C220MovPadError> {
        if self.transfer.is_input() {
            return Err(C220MovPadError::DirectionMismatch);
        }
        destination.write_segments_at(&self.writes)?;
        Ok(self.result)
    }
}

/// Captures memory effects without publishing them before timing retirement.
pub fn prepare_c220_mov_pad(
    transfer: C220MovPadTransfer,
    external: &MappedMemory,
    ub: &UbMemory,
    padding: u32,
) -> Result<C220PreparedMovPad, C220MovPadError> {
    let mut writes = Vec::new();
    writes
        .try_reserve_exact(transfer.segments().len())
        .map_err(|_| UbMemoryError::HostAllocationFailed {
            requested: transfer.segments().len(),
        })?;
    let mut result = UbTransferResult {
        segment_count: 0,
        bytes: 0,
        known_bytes: 0,
        unknown_bytes: 0,
    };
    let fill = padding.to_le_bytes();
    let element_bytes = usize::from(transfer.instruction.element_bytes);
    for segment in transfer.segments() {
        let input = if transfer.is_input() {
            external.read_states_at(segment.source_address, segment.input_bytes as usize)?
        } else {
            ub.read_states(segment.source_address, segment.input_bytes as usize)?
        };
        let output = if transfer.is_input() {
            let size = segment.output_bytes as usize;
            let left = transfer.left_padding() as usize * element_bytes;
            let padded = transfer.padded_bytes() as usize;
            let explicit_padding = transfer.left_padding() + transfer.right_padding() != 0;
            let mut output = Vec::new();
            output
                .try_reserve_exact(size)
                .map_err(|_| UbMemoryError::HostAllocationFailed { requested: size })?;
            for index in 0..size {
                let state = if index < left {
                    MemoryByteState::Known(fill[index % element_bytes])
                } else if index < left + input.len() {
                    input[index - left]
                } else if index < padded {
                    MemoryByteState::Known(fill[(index - left - input.len()) % element_bytes])
                } else if explicit_padding {
                    MemoryByteState::Known(fill[(index - padded) % element_bytes])
                } else {
                    input[((index - padded) % element_bytes) % input.len()]
                };
                output.push(state);
            }
            output
        } else {
            input
        };
        result.segment_count += 1;
        result.bytes += output.len();
        result.known_bytes += output
            .iter()
            .filter(|state| matches!(state, MemoryByteState::Known(_)))
            .count();
        writes.push((segment.destination_address, output));
    }
    result.unknown_bytes = result.bytes - result.known_bytes;
    Ok(C220PreparedMovPad {
        transfer,
        writes,
        result,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::c220::mte::{mov_pad::C220MovPadInstruction, read_register_mask};
    use crate::memory::{region::MemoryRegion, sparse::SparseMemory};
    use crate::sim::c220::mte::uop::{C220DmaUopMode, C220DmaUopRoute, mov_pad_uops};

    fn captured_transfer(
        format: u32,
        output: bool,
        length: u32,
        left: u64,
        right: u64,
    ) -> C220MovPadTransfer {
        let word = (3 << 29)
            | (1 << 27)
            | (14 << 22)
            | (u32::from(output) << 22)
            | (1 << 17)
            | (2 << 12)
            | (3 << 7)
            | (4 << 2)
            | format;
        let instruction = C220MovPadInstruction::decode(word).unwrap();
        assert_eq!(read_register_mask(word), Some(0b11110));
        let mut registers = [0; 32];
        registers[1] = if output { 0x1000 } else { 0 };
        registers[2] = if output { 0 } else { 0x1000 };
        registers[3] = 13 | (2 << 4) | (u64::from(length) << 16) | (left << 48) | (right << 54);
        registers[4] = (2 << 32) | 1;
        instruction.capture(&registers)
    }

    #[test]
    fn padding_tail_unknown_bytes_and_bidirectional_strides() {
        let data: Vec<_> = (0..512).map(|index| index as u8).collect();
        let mut external = MappedMemory::bind(
            SparseMemory::new(vec![MemoryRegion::new(512, data).unwrap()], 512, 512),
            &[0x1000],
        )
        .unwrap();
        external.write_unknown_at(0x1000, 1).unwrap();
        for format in 0..3 {
            for length in [1, 3, 31, 32, 33] {
                for (left, right) in [(0, 0), (1, 2)] {
                    let transfer = captured_transfer(format, false, length, left, right);
                    let mut ub = UbMemory::new(4096, 4096);
                    let prepared =
                        prepare_c220_mov_pad(transfer, &external, &ub, 0x4433_2211).unwrap();
                    assert_eq!(ub.tracked_bytes(), 0);
                    let result = prepared.commit_to_ub(&mut ub).unwrap();
                    assert_eq!(result.bytes, transfer.output_bytes() as usize * 2);
                    assert!(result.unknown_bytes > 0);
                    let width = usize::from(transfer.instruction.element_bytes);
                    for segment in transfer.segments() {
                        let input = external
                            .read_states_at(segment.source_address, length as usize)
                            .unwrap();
                        let output = ub
                            .read_states(segment.destination_address, segment.output_bytes as usize)
                            .unwrap();
                        let start = left as usize * width;
                        assert_eq!(&output[start..start + length as usize], input);
                        let tail = transfer.padded_bytes() as usize;
                        for (index, state) in output[tail..].iter().enumerate() {
                            let expected = if left + right == 0 {
                                input[(index % width) % input.len()]
                            } else {
                                MemoryByteState::Known([0x11, 0x22, 0x33, 0x44][index % width])
                            };
                            assert_eq!(*state, expected);
                        }
                        assert_eq!(
                            ub.read_states(
                                segment.destination_address + u64::from(segment.output_bytes),
                                1
                            )
                            .unwrap(),
                            [MemoryByteState::Unknown]
                        );
                    }
                    let output_transfer = captured_transfer(format, true, length, 63, 63);
                    assert_eq!(output_transfer.left_padding(), 0);
                    assert_eq!(output_transfer.right_padding(), 0);
                    let prepared =
                        prepare_c220_mov_pad(output_transfer, &external, &ub, 0).unwrap();
                    let mut destination = MappedMemory::bind(
                        SparseMemory::new(
                            vec![MemoryRegion::new(512, vec![0; 512]).unwrap()],
                            512,
                            512,
                        ),
                        &[0x1000],
                    )
                    .unwrap();
                    prepared.commit_to_external(&mut destination).unwrap();
                    for segment in output_transfer.segments() {
                        assert_eq!(
                            destination
                                .read_states_at(segment.destination_address, length as usize)
                                .unwrap(),
                            ub.read_states(segment.source_address, length as usize)
                                .unwrap()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn request_routes_and_descriptor_widths() {
        for (output, length, xt, left, right, route, count) in [
            (false, 17, 0, 0, 0, C220DmaUopRoute::ContiguousBatch, 1),
            (false, 32, 0, 0, 0, C220DmaUopRoute::Ordinary, 2),
            (
                false,
                64,
                1 << 32,
                0,
                0,
                C220DmaUopRoute::DestinationGapCollapse,
                1,
            ),
            (false, 513, 0, 1, 1, C220DmaUopRoute::Ordinary, 2),
            (true, 64, 0, 63, 63, C220DmaUopRoute::ContiguousBatch, 1),
            (true, 64, 1, 0, 0, C220DmaUopRoute::SourceGapGather, 1),
            (true, 33, 0, 0, 0, C220DmaUopRoute::Ordinary, 2),
        ] {
            let mut transfer = captured_transfer(0, output, length, left, right);
            transfer.xt = xt;
            let requests = mov_pad_uops(transfer, C220DmaUopMode::Wide512).unwrap();
            assert_eq!(requests.sid(), 13);
            let requests: Vec<_> = requests.collect();
            assert_eq!(requests.len(), count);
            assert!(requests.iter().all(|request| request.route == route));
            assert_eq!(
                requests
                    .iter()
                    .map(|request| u64::from(request.bytes))
                    .sum::<u64>(),
                u64::from(length) * 2
            );
        }
        let mut wide = captured_transfer(2, false, 0x1f_ffff, 63, 63);
        wide.xt = u64::MAX;
        assert_eq!(wide.burst_bytes(), 0x1f_ffff);
        assert_eq!(wide.source_stride(), 0x1f_fffe);
        assert_eq!(
            wide.destination_stride(),
            u64::from(u32::MAX) * 32 + u64::from(wide.output_bytes())
        );
        wide.xm &= !0xfff0;
        assert_eq!(wide.segments().len(), 0);
        assert!(
            mov_pad_uops(wide, C220DmaUopMode::Wide512)
                .unwrap()
                .next()
                .is_none()
        );
        assert!(C220MovPadInstruction::decode(wide.instruction.word | 3).is_none());
    }
}
