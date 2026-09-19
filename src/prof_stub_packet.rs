
use thiserror::Error;

pub const PROF_STUB_PACKET_HEADER_BYTES: usize = 8;
pub const LOG_TRANSLATE_OUTPUT_PATH_FIELD_BYTES: usize = 4096;
pub const LOG_TRANSLATE_KERNEL_NAME_FIELD_BYTES: usize = 1024;
pub const LOG_TRANSLATE_ACK_BYTES: &[u8; 3] = b"SUC";

pub fn encode_log_translate_start(
    output_path: &[u8],
    kernel_name: &[u8],
) -> Result<Vec<u8>, ProfStubPacketError> {
    if output_path.len() >= LOG_TRANSLATE_OUTPUT_PATH_FIELD_BYTES {
        return Err(ProfStubPacketError::FieldTooLong {
            field: "output_path",
            maximum: LOG_TRANSLATE_OUTPUT_PATH_FIELD_BYTES - 1,
            actual: output_path.len(),
        });
    }
    if kernel_name.len() >= LOG_TRANSLATE_KERNEL_NAME_FIELD_BYTES {
        return Err(ProfStubPacketError::FieldTooLong {
            field: "kernel_name",
            maximum: LOG_TRANSLATE_KERNEL_NAME_FIELD_BYTES - 1,
            actual: kernel_name.len(),
        });
    }
    let mut payload =
        vec![0; LOG_TRANSLATE_OUTPUT_PATH_FIELD_BYTES + LOG_TRANSLATE_KERNEL_NAME_FIELD_BYTES];
    payload[..output_path.len()].copy_from_slice(output_path);
    let name_start = LOG_TRANSLATE_OUTPUT_PATH_FIELD_BYTES;
    payload[name_start..name_start + kernel_name.len()].copy_from_slice(kernel_name);
    Ok(ProfStubPacket::new(4, &payload)?.encode_frame())
}

pub fn decode_log_translate_start<'a>(
    packet: ProfStubPacket<'a>,
) -> Result<(&'a [u8], &'a [u8]), ProfStubPacketError> {
    if packet.packet_type != 4 {
        return Err(ProfStubPacketError::WrongType {
            expected: 4,
            actual: packet.packet_type,
        });
    }
    if packet.payload.len()
        != LOG_TRANSLATE_OUTPUT_PATH_FIELD_BYTES + LOG_TRANSLATE_KERNEL_NAME_FIELD_BYTES
    {
        return Err(ProfStubPacketError::InvalidPayloadLength {
            packet_type: 4,
            expected: LOG_TRANSLATE_OUTPUT_PATH_FIELD_BYTES + LOG_TRANSLATE_KERNEL_NAME_FIELD_BYTES,
            actual: packet.payload.len(),
        });
    }
    Ok(packet
        .payload
        .split_at(LOG_TRANSLATE_OUTPUT_PATH_FIELD_BYTES))
}

pub fn encode_log_translate_stop() -> Vec<u8> {
    ProfStubPacket::new(5, &[])
        .expect("registered zero-length stop packet")
        .encode_frame()
}

pub const fn max_prof_stub_payload_bytes(packet_type: u32) -> Option<usize> {
    match packet_type {
        0 | 5 => Some(0),
        1 => Some(1028),
        2 => Some(8),
        3 => Some(104_857_628),
        4 => Some(5120),
        20 | 21 => Some(424),
        22 => Some(40),
        23 => Some(64),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProfStubPacket<'a> {
    packet_type: u32,
    payload: &'a [u8],
}

impl<'a> ProfStubPacket<'a> {
    pub fn new(packet_type: u32, payload: &'a [u8]) -> Result<Self, ProfStubPacketError> {
        let maximum = max_prof_stub_payload_bytes(packet_type)
            .ok_or(ProfStubPacketError::UnknownType(packet_type))?;
        if payload.len() > maximum {
            return Err(ProfStubPacketError::PayloadTooLarge {
                packet_type,
                maximum,
                actual: payload.len(),
            });
        }
        Ok(Self {
            packet_type,
            payload,
        })
    }

    pub fn packet_type(&self) -> u32 {
        self.packet_type
    }

    pub fn payload(&self) -> &'a [u8] {
        self.payload
    }

    pub fn encode_frame(&self) -> Vec<u8> {
        let mut frame = Vec::with_capacity(PROF_STUB_PACKET_HEADER_BYTES + self.payload.len());
        frame.extend_from_slice(&self.packet_type.to_le_bytes());
        frame.extend_from_slice(&(self.payload.len() as u32).to_le_bytes());
        frame.extend_from_slice(self.payload);
        frame
    }

    pub fn decode_prefix(input: &'a [u8]) -> Result<Option<(Self, usize)>, ProfStubPacketError> {
        if input.len() < PROF_STUB_PACKET_HEADER_BYTES {
            return Ok(None);
        }
        let packet_type = u32::from_le_bytes(input[0..4].try_into().expect("four-byte type"));
        let declared = u32::from_le_bytes(input[4..8].try_into().expect("four-byte length"));
        let maximum = max_prof_stub_payload_bytes(packet_type)
            .ok_or(ProfStubPacketError::UnknownType(packet_type))?;
        let length = declared as usize;
        if length > maximum {
            return Err(ProfStubPacketError::PayloadTooLarge {
                packet_type,
                maximum,
                actual: length,
            });
        }
        let end = PROF_STUB_PACKET_HEADER_BYTES + length;
        if input.len() < end {
            return Ok(None);
        }
        Ok(Some((Self::new(packet_type, &input[8..end])?, end)))
    }

    pub fn decode_exact(input: &'a [u8]) -> Result<Self, ProfStubPacketError> {
        match Self::decode_prefix(input)? {
            Some((packet, consumed)) if consumed == input.len() => Ok(packet),
            Some((_, consumed)) => Err(ProfStubPacketError::TrailingBytes(input.len() - consumed)),
            None => Err(ProfStubPacketError::Incomplete),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum ProfStubPacketError {
    #[error("unregistered ProfStub packet type {0}")]
    UnknownType(u32),
    #[error("ProfStub packet type {packet_type} payload has {actual} bytes, above {maximum}")]
    PayloadTooLarge {
        packet_type: u32,
        maximum: usize,
        actual: usize,
    },
    #[error("incomplete ProfStub packet")]
    Incomplete,
    #[error("ProfStub packet has {0} trailing bytes")]
    TrailingBytes(usize),
    #[error("{field} has {actual} bytes, above {maximum}")]
    FieldTooLong {
        field: &'static str,
        maximum: usize,
        actual: usize,
    },
    #[error("expected ProfStub packet type {expected}, got {actual}")]
    WrongType { expected: u32, actual: u32 },
    #[error("ProfStub packet type {packet_type} requires {expected} payload bytes, got {actual}")]
    InvalidPayloadLength {
        packet_type: u32,
        expected: usize,
        actual: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_limits_match_the_inspected_constructor() {
        assert_eq!(
            [0, 1, 2, 3, 4, 5, 20, 21, 22, 23].map(max_prof_stub_payload_bytes),
            [
                Some(0),
                Some(1028),
                Some(8),
                Some(104_857_628),
                Some(5120),
                Some(0),
                Some(424),
                Some(424),
                Some(40),
                Some(64),
            ]
        );
        assert_eq!(max_prof_stub_payload_bytes(1001), None);
    }

    #[test]
    fn exact_frame_and_fragmented_prefix_round_trip() {
        let packet = ProfStubPacket::new(2, &[0, 0xff, 1]).unwrap();
        let frame = packet.encode_frame();
        assert_eq!(frame, [2, 0, 0, 0, 3, 0, 0, 0, 0, 0xff, 1]);
        for prefix in 0..frame.len() {
            assert_eq!(ProfStubPacket::decode_prefix(&frame[..prefix]), Ok(None));
        }
        assert_eq!(ProfStubPacket::decode_exact(&frame), Ok(packet));
    }

    #[test]
    fn concatenated_stream_consumes_only_first_packet() {
        let first = ProfStubPacket::new(0, &[]).unwrap().encode_frame();
        let second = ProfStubPacket::new(22, &[1, 2]).unwrap().encode_frame();
        let stream = [first, second].concat();
        let (packet, consumed) = ProfStubPacket::decode_prefix(&stream).unwrap().unwrap();
        assert_eq!(packet.packet_type(), 0);
        assert!(packet.payload().is_empty());
        let (packet, second_consumed) = ProfStubPacket::decode_prefix(&stream[consumed..])
            .unwrap()
            .unwrap();
        assert_eq!(packet.packet_type(), 22);
        assert_eq!(packet.payload(), &[1, 2]);
        assert_eq!(consumed + second_consumed, stream.len());
    }

    #[test]
    fn rejects_unknown_type_oversize_and_trailing_bytes() {
        let mut unknown = [0_u8; PROF_STUB_PACKET_HEADER_BYTES];
        unknown[..4].copy_from_slice(&1001_u32.to_le_bytes());
        assert_eq!(
            ProfStubPacket::decode_exact(&unknown),
            Err(ProfStubPacketError::UnknownType(1001))
        );
        assert_eq!(
            ProfStubPacket::decode_exact(&[2, 0, 0, 0, 9, 0, 0, 0]),
            Err(ProfStubPacketError::PayloadTooLarge {
                packet_type: 2,
                maximum: 8,
                actual: 9,
            })
        );
        assert_eq!(
            ProfStubPacket::new(5, &[1]),
            Err(ProfStubPacketError::PayloadTooLarge {
                packet_type: 5,
                maximum: 0,
                actual: 1,
            })
        );
        assert_eq!(
            ProfStubPacket::decode_exact(&[0; 9]),
            Err(ProfStubPacketError::TrailingBytes(1))
        );
    }

    #[test]
    fn log_translate_start_has_two_fixed_zero_padded_fields() {
        let frame = encode_log_translate_start(b"/tmp/prof", b"ClearL2Cache").unwrap();
        assert_eq!(frame.len(), PROF_STUB_PACKET_HEADER_BYTES + 5120);
        let packet = ProfStubPacket::decode_exact(&frame).unwrap();
        let (path, name) = decode_log_translate_start(packet).unwrap();
        assert_eq!(&path[..10], b"/tmp/prof\0");
        assert!(path[10..].iter().all(|byte| *byte == 0));
        assert_eq!(&name[..13], b"ClearL2Cache\0");
        assert!(name[13..].iter().all(|byte| *byte == 0));
        assert_eq!(LOG_TRANSLATE_ACK_BYTES, b"SUC");
        assert_eq!(
            encode_log_translate_start(&vec![b'x'; 4096], b"k"),
            Err(ProfStubPacketError::FieldTooLong {
                field: "output_path",
                maximum: 4095,
                actual: 4096,
            })
        );
    }

    #[test]
    fn log_translate_stop_is_an_empty_type_five_packet() {
        let frame = encode_log_translate_stop();
        assert_eq!(frame, [5, 0, 0, 0, 0, 0, 0, 0]);
        let packet = ProfStubPacket::decode_exact(&frame).unwrap();
        assert_eq!(packet.packet_type(), 5);
        assert!(packet.payload().is_empty());
        assert_eq!(
            decode_log_translate_start(packet),
            Err(ProfStubPacketError::WrongType {
                expected: 4,
                actual: 5,
            })
        );
    }
}
