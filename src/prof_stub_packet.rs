use thiserror::Error;

pub const PROF_STUB_PACKET_HEADER_BYTES: usize = 8;
pub const DATA_PATH_KERNEL_NAME_FIELD_BYTES: usize = 1024;
pub const DATA_PATH_REQUEST_PAYLOAD_BYTES: usize = 1028;
pub const MODEL_CONFIG_RESPONSE_BYTES: usize = 1620;
pub const PROF_STUB_RESPONSE_READ_BYTES: usize = 1024;
pub const PROF_STUB_RESPONSE_MAX_WAIT_ATTEMPTS: usize = 64;
pub const LOG_TRANSLATE_OUTPUT_PATH_FIELD_BYTES: usize = 4096;
pub const LOG_TRANSLATE_KERNEL_NAME_FIELD_BYTES: usize = 1024;
pub const LOG_TRANSLATE_ACK_BYTES: &[u8; 3] = b"SUC";

pub fn encode_model_config_request() -> Vec<u8> {
    ProfStubPacket::new(0, &[])
        .expect("registered zero-length model-config packet")
        .encode_frame()
}

pub fn encode_data_path_request(
    kernel_name: &[u8],
    path_index: u32,
) -> Result<Vec<u8>, ProfStubPacketError> {
    if kernel_name.len() >= DATA_PATH_KERNEL_NAME_FIELD_BYTES {
        return Err(ProfStubPacketError::FieldTooLong {
            field: "kernel_name",
            maximum: DATA_PATH_KERNEL_NAME_FIELD_BYTES - 1,
            actual: kernel_name.len(),
        });
    }
    let mut payload = [0_u8; DATA_PATH_REQUEST_PAYLOAD_BYTES];
    payload[..kernel_name.len()].copy_from_slice(kernel_name);
    payload[DATA_PATH_KERNEL_NAME_FIELD_BYTES..].copy_from_slice(&path_index.to_le_bytes());
    Ok(ProfStubPacket::new(1, &payload)?.encode_frame())
}

pub fn decode_data_path_request(
    packet: ProfStubPacket<'_>,
) -> Result<(&[u8], u32), ProfStubPacketError> {
    if packet.packet_type != 1 {
        return Err(ProfStubPacketError::WrongType {
            expected: 1,
            actual: packet.packet_type,
        });
    }
    if packet.payload.len() != DATA_PATH_REQUEST_PAYLOAD_BYTES {
        return Err(ProfStubPacketError::InvalidPayloadLength {
            packet_type: 1,
            expected: DATA_PATH_REQUEST_PAYLOAD_BYTES,
            actual: packet.payload.len(),
        });
    }
    let (name, index) = packet.payload.split_at(DATA_PATH_KERNEL_NAME_FIELD_BYTES);
    Ok((
        name,
        u32::from_le_bytes(index.try_into().expect("four-byte path index")),
    ))
}

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
pub enum ProfStubReply<'a> {
    None,
    ModelConfig {
        config: &'a [u8; MODEL_CONFIG_RESPONSE_BYTES],
        trailing: &'a [u8],
    },
    DataPath(&'a [u8]),
    ClientTerminationFlag(bool),
    Success,
}

impl<'a> ProfStubReply<'a> {
    pub fn decode_for(packet_type: u32, bytes: &'a [u8]) -> Result<Self, ProfStubPacketError> {
        match packet_type {
            0 => {
                if bytes.len() < MODEL_CONFIG_RESPONSE_BYTES {
                    return Err(ProfStubPacketError::InvalidModelConfigResponseLength {
                        minimum: MODEL_CONFIG_RESPONSE_BYTES,
                        actual: bytes.len(),
                    });
                }
                let (config, trailing) = bytes.split_at(MODEL_CONFIG_RESPONSE_BYTES);
                Ok(Self::ModelConfig {
                    config: config.try_into().expect("fixed-size config prefix"),
                    trailing,
                })
            }
            1 => {
                if bytes.last() != Some(&b'\n') {
                    return Err(ProfStubPacketError::InvalidDataPathResponse {
                        actual: bytes.len(),
                    });
                }
                Ok(Self::DataPath(&bytes[..bytes.len() - 1]))
            }
            2 => match bytes {
                [0] => Ok(Self::ClientTerminationFlag(false)),
                [1] => Ok(Self::ClientTerminationFlag(true)),
                _ => Err(ProfStubPacketError::InvalidClientTerminationFlag {
                    actual: bytes.len(),
                }),
            },
            3 | 20 | 21 | 22 | 23 if bytes.is_empty() => Ok(Self::None),
            3 | 20 | 21 | 22 | 23 => Err(ProfStubPacketError::UnexpectedResponse {
                packet_type,
                actual: bytes.len(),
            }),
            4 | 5 if bytes == LOG_TRANSLATE_ACK_BYTES => Ok(Self::Success),
            4 | 5 => Err(ProfStubPacketError::InvalidSuccessResponse {
                packet_type,
                actual: bytes.len(),
            }),
            _ => Err(ProfStubPacketError::UnknownResponsePolicy(packet_type)),
        }
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
    #[error("model-config response requires at least {minimum} bytes, got {actual}")]
    InvalidModelConfigResponseLength { minimum: usize, actual: usize },
    #[error("data-path response requires a newline-terminated value, got {actual} bytes")]
    InvalidDataPathResponse { actual: usize },
    #[error("client-termination response requires a single 0 or 1 byte, got {actual} bytes")]
    InvalidClientTerminationFlag { actual: usize },
    #[error("ProfStub response to packet type {packet_type} must be empty, got {actual} bytes")]
    UnexpectedResponse { packet_type: u32, actual: usize },
    #[error(
        "ProfStub packet type {packet_type} returned an invalid {actual}-byte success response"
    )]
    InvalidSuccessResponse { packet_type: u32, actual: usize },
    #[error("ProfStub packet type {0} has no established response policy")]
    UnknownResponsePolicy(u32),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_limits_cover_all_packet_types() {
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
    fn model_config_request_is_an_empty_type_zero_packet() {
        let frame = encode_model_config_request();
        assert_eq!(frame, [0; PROF_STUB_PACKET_HEADER_BYTES]);
        let packet = ProfStubPacket::decode_exact(&frame).unwrap();
        assert_eq!(packet.packet_type(), 0);
        assert!(packet.payload().is_empty());
    }

    #[test]
    fn data_path_request_has_a_fixed_name_field_and_index() {
        let frame = encode_data_path_request(b"VectorAdd", 7).unwrap();
        assert_eq!(frame.len(), PROF_STUB_PACKET_HEADER_BYTES + 1028);
        let packet = ProfStubPacket::decode_exact(&frame).unwrap();
        let (name, index) = decode_data_path_request(packet).unwrap();
        assert_eq!(&name[..10], b"VectorAdd\0");
        assert!(name[10..].iter().all(|byte| *byte == 0));
        assert_eq!(index, 7);
        assert_eq!(
            encode_data_path_request(&vec![b'x'; 1024], 0),
            Err(ProfStubPacketError::FieldTooLong {
                field: "kernel_name",
                maximum: 1023,
                actual: 1024,
            })
        );
    }

    #[test]
    fn replies_are_raw_and_request_specific() {
        let config = [0_u8; MODEL_CONFIG_RESPONSE_BYTES];
        assert!(matches!(
            ProfStubReply::decode_for(0, &config),
            Ok(ProfStubReply::ModelConfig { trailing: [], .. })
        ));
        assert_eq!(
            ProfStubReply::decode_for(1, b"/tmp/run/0/dump\n"),
            Ok(ProfStubReply::DataPath(b"/tmp/run/0/dump"))
        );
        assert_eq!(
            ProfStubReply::decode_for(4, b"SUC"),
            Ok(ProfStubReply::Success)
        );
        assert_eq!(ProfStubReply::decode_for(20, b""), Ok(ProfStubReply::None));
        assert_eq!(
            ProfStubReply::decode_for(20, b"x"),
            Err(ProfStubPacketError::UnexpectedResponse {
                packet_type: 20,
                actual: 1,
            })
        );
        assert_eq!(
            ProfStubReply::decode_for(2, b"\x00"),
            Ok(ProfStubReply::ClientTerminationFlag(false))
        );
        assert_eq!(
            ProfStubReply::decode_for(2, b"\x01"),
            Ok(ProfStubReply::ClientTerminationFlag(true))
        );
        assert_eq!(
            ProfStubReply::decode_for(2, b"\x02"),
            Err(ProfStubPacketError::InvalidClientTerminationFlag { actual: 1 })
        );
        assert_eq!(
            ProfStubReply::decode_for(2, b""),
            Err(ProfStubPacketError::InvalidClientTerminationFlag { actual: 0 })
        );
        assert_eq!(
            ProfStubReply::decode_for(2, b"\x00\x01"),
            Err(ProfStubPacketError::InvalidClientTerminationFlag { actual: 2 })
        );
    }

    #[test]
    fn response_validation_rejects_truncated_or_malformed_values() {
        assert_eq!(
            ProfStubReply::decode_for(0, &[0; MODEL_CONFIG_RESPONSE_BYTES - 1]),
            Err(ProfStubPacketError::InvalidModelConfigResponseLength {
                minimum: MODEL_CONFIG_RESPONSE_BYTES,
                actual: MODEL_CONFIG_RESPONSE_BYTES - 1,
            })
        );
        assert_eq!(
            ProfStubReply::decode_for(1, b"/tmp/no-newline"),
            Err(ProfStubPacketError::InvalidDataPathResponse { actual: 15 })
        );
        assert_eq!(
            ProfStubReply::decode_for(5, b"FAIL"),
            Err(ProfStubPacketError::InvalidSuccessResponse {
                packet_type: 5,
                actual: 4,
            })
        );
    }

    #[test]
    fn response_can_span_multiple_reads() {
        let mut config = vec![0; MODEL_CONFIG_RESPONSE_BYTES];
        config.extend_from_slice(&[1, 2, 3]);
        let ProfStubReply::ModelConfig {
            config: prefix,
            trailing,
        } = ProfStubReply::decode_for(0, &config).unwrap()
        else {
            panic!("expected model configuration");
        };
        assert_eq!(prefix, &[0; MODEL_CONFIG_RESPONSE_BYTES]);
        assert_eq!(trailing, &[1, 2, 3]);

        let mut path = vec![b'x'; PROF_STUB_RESPONSE_READ_BYTES + 1];
        path.push(b'\n');
        assert_eq!(
            ProfStubReply::decode_for(1, &path),
            Ok(ProfStubReply::DataPath(&path[..path.len() - 1]))
        );
        let mut long_path = vec![b'x'; 65_536];
        long_path.push(b'\n');
        assert_eq!(
            ProfStubReply::decode_for(1, &long_path),
            Ok(ProfStubReply::DataPath(&long_path[..long_path.len() - 1]))
        );
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
