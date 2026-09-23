use crate::isa::c220::mte::factor::C220FactorLoad;
use crate::sim::c220::memory::l1::C220L1Access;
use crate::sim::c220::memory::{C220LocalBuffer, C220LocalBufferError};
use crate::sim::c220::mte::interface::{C220MteL1OutputDestination, C220MteL1ReadOperation};
use crate::sim::c220::numeric::fp16::c220_fp16_to_fp32_bits;
use std::num::NonZeroU32;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct C220FactorLoadResult {
    pub blocks: u32,
    pub input_bytes: u64,
    pub output_bytes: u64,
    pub nan_elements: u64,
    pub infinity_elements: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FactorReadPacket {
    pub source_address: u64,
    pub destination_address: u64,
    pub read_bytes: u32,
    pub data_bytes: u32,
    pub instruction_tail: bool,
}

/// Packet data width describes transport, not the functional write width.
pub fn c220_factor_read_packets(
    load: C220FactorLoad,
) -> impl ExactSizeIterator<Item = C220FactorReadPacket> {
    let count = load.descriptor.burst_count() * load.descriptor.burst_blocks();
    let data_bytes = if load.descriptor.convert() { 64 } else { 128 };
    load.blocks()
        .enumerate()
        .map(move |(index, block)| C220FactorReadPacket {
            source_address: block.source_address,
            destination_address: load
                .destination_address
                .wrapping_add(u64::from(block.burst) * load.descriptor.destination_stride())
                .wrapping_add(u64::from(block.block * data_bytes)),
            read_bytes: 128,
            data_bytes,
            instruction_tail: index + 1 == count as usize,
        })
}

/// Splits transport packets into physical reads. Only the last physical
/// response of each packet releases its full logical output to the FB sink.
pub fn c220_factor_l1_requests(
    load: C220FactorLoad,
    instruction_id: u64,
    access_width: NonZeroU32,
    output_bandwidth: NonZeroU32,
) -> impl Iterator<Item = C220MteL1ReadOperation<C220FactorReadPacket>> {
    c220_factor_read_packets(load).flat_map(move |packet| {
        let width = access_width.get();
        (0..packet.read_bytes.div_ceil(width)).map(move |index| {
            let offset = index * width;
            let bytes = (packet.read_bytes - offset).min(width);
            C220MteL1ReadOperation {
                instruction_id,
                access: C220L1Access {
                    address: packet.source_address.wrapping_add(u64::from(offset)),
                    bytes,
                },
                destination: C220MteL1OutputDestination::Fb,
                output_address: packet.destination_address,
                output_bytes: packet.data_bytes,
                output_bandwidth,
                completes_logical_uop: offset + bytes == packet.read_bytes,
                last_in_instruction: packet.instruction_tail,
                payload: packet,
            }
        })
    })
}

/// Applies functional effects using the selected source buffer. Admission,
/// source readiness and completion timing are managed by the caller.
pub fn execute_c220_factor_load(
    source: &C220LocalBuffer,
    factors: &mut C220LocalBuffer,
    load: C220FactorLoad,
) -> Result<C220FactorLoadResult, C220LocalBufferError> {
    let mut result = C220FactorLoadResult::default();
    for block in load.blocks() {
        let input = source.read_initialized_linear(block.source_address, 128)?;
        let mut converted = [0_u8; 256];
        let output = if load.descriptor.convert() {
            for (input, output) in input.chunks_exact(2).zip(converted.chunks_exact_mut(4)) {
                let bits = u16::from_le_bytes([input[0], input[1]]);
                result.nan_elements += u64::from(bits & 32767 > 31744);
                result.infinity_elements += u64::from(bits & 32767 == 31744);
                output.copy_from_slice(&c220_fp16_to_fp32_bits(bits).to_le_bytes());
            }
            &converted[..]
        } else {
            &input[..]
        };
        factors.write_known_linear(block.destination_address, output)?;
        result.blocks += 1;
        result.input_bytes += 128;
        result.output_bytes += output.len() as u64;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::c220::mte::factor::{C220FactorLoadInstruction, C220FactorSource};

    #[test]
    fn factor_load_separates_conversion_stride_and_transport_width() {
        let word = (6 << 29) | (1 << 17) | (2 << 12) | (3 << 7);
        let mut registers = [0; 32];
        registers[1] = (7 << 16) | 32;
        registers[2] = 16;
        let mut source = C220LocalBuffer::new(2048);
        source
            .write_known_linear(0, &0x3c00_u16.to_le_bytes().repeat(1024))
            .unwrap();
        for convert in [false, true] {
            registers[3] =
                (1 << 48) | (3 << 32) | (2 << 16) | (2 << 4) | if convert { 8 } else { 0 };
            for ub in [false, true] {
                let instruction =
                    C220FactorLoadInstruction::decode(word | if ub { 1 << 22 } else { 0 }).unwrap();
                let load = instruction.capture(&registers);
                assert_eq!(
                    load.source,
                    if ub {
                        C220FactorSource::Ub
                    } else {
                        C220FactorSource::L1
                    }
                );
                assert_eq!(load.destination_address, 2080);
                let output_width = if convert { 256 } else { 128 };
                let blocks: Vec<_> = load.blocks().collect();
                assert_eq!(blocks[2].source_address, 16 + 256 + 96);
                assert_eq!(blocks[2].destination_address, 2080 + 3 * output_width);
                let packets: Vec<_> = c220_factor_read_packets(load).collect();
                assert_eq!(packets.len(), 4);
                assert_eq!(
                    packets[1].destination_address,
                    2080 + if convert { 64 } else { 128 }
                );
                assert!(!packets[2].instruction_tail);
                assert!(packets[3].instruction_tail);
                let requests: Vec<_> = c220_factor_l1_requests(
                    load,
                    17,
                    NonZeroU32::new(96).unwrap(),
                    NonZeroU32::new(32).unwrap(),
                )
                .collect();
                assert_eq!(requests.len(), 8);
                for (packet, pair) in packets.iter().zip(requests.chunks_exact(2)) {
                    assert_eq!(pair[0].access.bytes, 96);
                    assert_eq!(pair[1].access.bytes, 32);
                    assert_eq!(pair[1].access.address, packet.source_address + 96);
                    assert!(!pair[0].completes_logical_uop);
                    assert!(pair[1].completes_logical_uop);
                    for request in pair {
                        assert_eq!(request.output_address, packet.destination_address);
                        assert_eq!(request.output_bytes, packet.data_bytes);
                        assert_eq!(request.last_in_instruction, packet.instruction_tail);
                    }
                }
                let mut factors = C220LocalBuffer::new(4096);
                let result = execute_c220_factor_load(&source, &mut factors, load).unwrap();
                assert_eq!(result.output_bytes, 4 * output_width);
                let expected = if convert {
                    1_f32.to_le_bytes().repeat(64)
                } else {
                    0x3c00_u16.to_le_bytes().repeat(64)
                };
                assert_eq!(factors.read_known(2080, expected.len()).unwrap(), expected);
            }
        }
        registers[3] = 0;
        let empty = C220FactorLoadInstruction::decode(word)
            .unwrap()
            .capture(&registers);
        assert_eq!(empty.blocks().len(), 0);
        assert_eq!(c220_factor_read_packets(empty).len(), 0);
        assert!(C220FactorLoadInstruction::decode(word | (2 << 24)).is_none());
    }
}
