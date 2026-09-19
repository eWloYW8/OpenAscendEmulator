
use thiserror::Error;

use crate::ipc::{KERNEL_RECORD_REQUEST_PACKET, KERNEL_RECORD_RESPONSE_PACKET};

pub const REQUEST_HEADER_SIZE: usize = 12;
pub const RESPONSE_PREFIX_SIZE: usize = 8;
pub const RESPONSE_MIN_SIZE: usize = 4 + RESPONSE_PREFIX_SIZE;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelRecordRequest {
    pub payload: Vec<u8>,
}

impl KernelRecordRequest {
    pub fn encode_frame(&self) -> Result<Vec<u8>, KernelRecordWireError> {
        let payload_len = u64::try_from(self.payload.len())
            .map_err(|_| KernelRecordWireError::LengthOverflow(self.payload.len()))?;
        let frame_len = REQUEST_HEADER_SIZE
            .checked_add(self.payload.len())
            .ok_or(KernelRecordWireError::LengthOverflow(self.payload.len()))?;
        let mut frame = Vec::with_capacity(frame_len);
        frame.extend_from_slice(&KERNEL_RECORD_REQUEST_PACKET.to_le_bytes());
        frame.extend_from_slice(&payload_len.to_le_bytes());
        frame.extend_from_slice(&self.payload);
        Ok(frame)
    }

    pub fn decode_frame(frame: &[u8]) -> Result<Self, KernelRecordWireError> {
        check_packet_type(frame, KERNEL_RECORD_REQUEST_PACKET)?;
        require_min_len(frame, REQUEST_HEADER_SIZE)?;

        let declared_len = u64::from_le_bytes(frame[4..12].try_into().expect("eight-byte field"));
        let payload = &frame[REQUEST_HEADER_SIZE..];
        if u64::try_from(payload.len()).ok() != Some(declared_len) {
            return Err(KernelRecordWireError::DeclaredLengthMismatch {
                declared: declared_len,
                actual: payload.len(),
            });
        }
        Ok(Self {
            payload: payload.to_vec(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelRecordResponse {
    pub opaque_prefix: [u8; RESPONSE_PREFIX_SIZE],
    pub trailing: Vec<u8>,
}

impl KernelRecordResponse {
    pub fn encode_frame(&self) -> Vec<u8> {
        let mut frame = Vec::with_capacity(RESPONSE_MIN_SIZE + self.trailing.len());
        frame.extend_from_slice(&KERNEL_RECORD_RESPONSE_PACKET.to_le_bytes());
        frame.extend_from_slice(&self.opaque_prefix);
        frame.extend_from_slice(&self.trailing);
        frame
    }

    pub fn decode_frame(frame: &[u8]) -> Result<Self, KernelRecordWireError> {
        check_packet_type(frame, KERNEL_RECORD_RESPONSE_PACKET)?;
        require_min_len(frame, RESPONSE_MIN_SIZE)?;

        Ok(Self {
            opaque_prefix: frame[4..12].try_into().expect("eight-byte field"),
            trailing: frame[12..].to_vec(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum KernelRecordWireError {
    #[error("kernel-record frame has length {actual}, expected at least {minimum}")]
    TooShort { minimum: usize, actual: usize },
    #[error("unexpected packet type {actual}, expected {expected}")]
    UnexpectedPacketType { expected: u32, actual: u32 },
    #[error("kernel-record payload length {declared} does not match {actual} bytes supplied")]
    DeclaredLengthMismatch { declared: u64, actual: usize },
    #[error("kernel-record payload length {0} cannot be encoded")]
    LengthOverflow(usize),
}

fn require_min_len(frame: &[u8], minimum: usize) -> Result<(), KernelRecordWireError> {
    if frame.len() < minimum {
        Err(KernelRecordWireError::TooShort {
            minimum,
            actual: frame.len(),
        })
    } else {
        Ok(())
    }
}

fn check_packet_type(frame: &[u8], expected: u32) -> Result<(), KernelRecordWireError> {
    require_min_len(frame, 4)?;
    let actual = u32::from_le_bytes(frame[0..4].try_into().expect("four-byte field"));
    if actual == expected {
        Ok(())
    } else {
        Err(KernelRecordWireError::UnexpectedPacketType { expected, actual })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_has_exact_outer_frame_and_round_trips_binary_payload() {
        let request = KernelRecordRequest {
            payload: vec![0, 0xff, 0, 1],
        };
        let frame = request.encode_frame().unwrap();
        assert_eq!(
            frame,
            [
                1001_u32.to_le_bytes().as_slice(),
                4_u64.to_le_bytes().as_slice(),
                &[0, 0xff, 0, 1]
            ]
            .concat()
        );
        assert_eq!(KernelRecordRequest::decode_frame(&frame), Ok(request));
    }

    #[test]
    fn empty_payload_has_twelve_byte_frame() {
        let request = KernelRecordRequest { payload: vec![] };
        let frame = request.encode_frame().unwrap();
        assert_eq!(frame.len(), REQUEST_HEADER_SIZE);
        assert_eq!(KernelRecordRequest::decode_frame(&frame), Ok(request));
    }

    #[test]
    fn request_rejects_short_wrong_type_and_mismatched_length() {
        assert_eq!(
            KernelRecordRequest::decode_frame(&[1, 2, 3]),
            Err(KernelRecordWireError::TooShort {
                minimum: 4,
                actual: 3,
            })
        );
        assert_eq!(
            KernelRecordRequest::decode_frame(&1001_u32.to_le_bytes()),
            Err(KernelRecordWireError::TooShort {
                minimum: REQUEST_HEADER_SIZE,
                actual: 4,
            })
        );
        let mut frame = KernelRecordRequest { payload: vec![5] }
            .encode_frame()
            .unwrap();
        frame[0..4].copy_from_slice(&1002_u32.to_le_bytes());
        assert_eq!(
            KernelRecordRequest::decode_frame(&frame),
            Err(KernelRecordWireError::UnexpectedPacketType {
                expected: 1001,
                actual: 1002,
            })
        );
        frame[0..4].copy_from_slice(&1001_u32.to_le_bytes());
        frame[4..12].copy_from_slice(&2_u64.to_le_bytes());
        assert_eq!(
            KernelRecordRequest::decode_frame(&frame),
            Err(KernelRecordWireError::DeclaredLengthMismatch {
                declared: 2,
                actual: 1,
            })
        );
        frame[4..12].copy_from_slice(&0_u64.to_le_bytes());
        assert!(matches!(
            KernelRecordRequest::decode_frame(&frame),
            Err(KernelRecordWireError::DeclaredLengthMismatch {
                declared: 0,
                actual: 1,
            })
        ));
    }

    #[test]
    fn response_preserves_opaque_prefix_and_trailing_bytes() {
        let response = KernelRecordResponse {
            opaque_prefix: [0x10, 0x20, 0x30, 0x40, 0, 0, 0, 0xff],
            trailing: vec![0xab, 0xcd],
        };
        let frame = response.encode_frame();
        assert_eq!(&frame[..4], &3001_u32.to_le_bytes());
        assert_eq!(&frame[4..12], &response.opaque_prefix);
        assert_eq!(&frame[12..], &response.trailing);
        assert_eq!(KernelRecordResponse::decode_frame(&frame), Ok(response));
    }

    #[test]
    fn response_rejects_short_body_and_wrong_type() {
        let mut frame = [0_u8; RESPONSE_MIN_SIZE - 1];
        frame[0..4].copy_from_slice(&3001_u32.to_le_bytes());
        assert_eq!(
            KernelRecordResponse::decode_frame(&frame),
            Err(KernelRecordWireError::TooShort {
                minimum: RESPONSE_MIN_SIZE,
                actual: RESPONSE_MIN_SIZE - 1,
            })
        );
        frame[0..4].copy_from_slice(&3000_u32.to_le_bytes());
        assert_eq!(
            KernelRecordResponse::decode_frame(&frame),
            Err(KernelRecordWireError::UnexpectedPacketType {
                expected: 3001,
                actual: 3000,
            })
        );
    }
}
