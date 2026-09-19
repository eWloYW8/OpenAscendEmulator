use std::fmt;
use thiserror::Error;

pub const PACKET_HEADER_SIZE: usize = 4;
pub const IPC_MEMORY_BODY_SIZE: usize = 88;
pub const IPC_MEMORY_REQUEST_SIZE: usize = PACKET_HEADER_SIZE + IPC_MEMORY_BODY_SIZE;
pub const IPC_RESPONSE_BODY_SIZE: usize = 8;
pub const IPC_RESPONSE_SIZE: usize = PACKET_HEADER_SIZE + IPC_RESPONSE_BODY_SIZE;
pub const IPC_MEMORY_REQUEST_PACKET: u32 = 1002;
pub const IPC_MEMORY_RESPONSE_PACKET: u32 = 3000;
pub const KERNEL_RECORD_REQUEST_PACKET: u32 = 1001;
pub const KERNEL_RECORD_RESPONSE_PACKET: u32 = 3001;

const IPC_MEMORY_NAME_CAPACITY: usize = 64;
const IPC_MEMORY_NAME_MAX_LEN: usize = IPC_MEMORY_NAME_CAPACITY - 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum IpcMemoryOperation {
    SetName = 0,
    DestroyName = 1,
    Open = 2,
    Close = 3,
}

impl TryFrom<u32> for IpcMemoryOperation {
    type Error = IpcWireError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::SetName),
            1 => Ok(Self::DestroyName),
            2 => Ok(Self::Open),
            3 => Ok(Self::Close),
            _ => Err(IpcWireError::UnknownOperation(value)),
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct IpcMemoryName([u8; IPC_MEMORY_NAME_CAPACITY]);

impl IpcMemoryName {
    pub fn from_c_bytes(value: &[u8]) -> Self {
        let source_len = value
            .iter()
            .position(|byte| *byte == 0)
            .unwrap_or(value.len());
        let copy_len = source_len.min(IPC_MEMORY_NAME_MAX_LEN);
        let mut field = [0; IPC_MEMORY_NAME_CAPACITY];
        field[..copy_len].copy_from_slice(&value[..copy_len]);
        Self(field)
    }

    pub fn as_bytes(&self) -> &[u8] {
        let len = self
            .0
            .iter()
            .position(|byte| *byte == 0)
            .unwrap_or(IPC_MEMORY_NAME_CAPACITY);
        &self.0[..len]
    }

    pub fn as_c_field(&self) -> &[u8; IPC_MEMORY_NAME_CAPACITY] {
        &self.0
    }
}

impl From<&str> for IpcMemoryName {
    fn from(value: &str) -> Self {
        Self::from_c_bytes(value.as_bytes())
    }
}

impl fmt::Debug for IpcMemoryName {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("IpcMemoryName")
            .field(&String::from_utf8_lossy(self.as_bytes()))
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IpcMemoryRequest {
    SetName {
        device_address: u64,
        byte_count: u64,
        name: IpcMemoryName,
    },
    DestroyName {
        name: IpcMemoryName,
    },
    Open {
        opened_device_address: u64,
        name: IpcMemoryName,
    },
    Close {
        device_address: u64,
    },
}

impl IpcMemoryRequest {
    pub const fn operation(&self) -> IpcMemoryOperation {
        match self {
            Self::SetName { .. } => IpcMemoryOperation::SetName,
            Self::DestroyName { .. } => IpcMemoryOperation::DestroyName,
            Self::Open { .. } => IpcMemoryOperation::Open,
            Self::Close { .. } => IpcMemoryOperation::Close,
        }
    }

    pub fn encode_body(&self) -> [u8; IPC_MEMORY_BODY_SIZE] {
        let mut body = [0; IPC_MEMORY_BODY_SIZE];
        put_u32(&mut body[0..4], self.operation() as u32);

        match self {
            Self::SetName {
                device_address,
                byte_count,
                name,
            } => {
                put_u64(&mut body[8..16], *device_address);
                put_u64(&mut body[16..24], *byte_count);
                body[24..88].copy_from_slice(name.as_c_field());
            }
            Self::DestroyName { name } => {
                body[8..72].copy_from_slice(name.as_c_field());
            }
            Self::Open {
                opened_device_address,
                name,
            } => {
                put_u64(&mut body[8..16], *opened_device_address);
                body[16..80].copy_from_slice(name.as_c_field());
            }
            Self::Close { device_address } => {
                put_u64(&mut body[8..16], *device_address);
            }
        }

        body
    }

    pub fn encode_frame(&self) -> [u8; IPC_MEMORY_REQUEST_SIZE] {
        let mut frame = [0; IPC_MEMORY_REQUEST_SIZE];
        put_u32(&mut frame[0..4], IPC_MEMORY_REQUEST_PACKET);
        frame[4..].copy_from_slice(&self.encode_body());
        frame
    }

    pub fn decode_body(body: &[u8]) -> Result<Self, IpcWireError> {
        require_len("IPC memory body", body, IPC_MEMORY_BODY_SIZE)?;
        let operation = IpcMemoryOperation::try_from(get_u32(&body[0..4]))?;
        let reserved = get_u32(&body[4..8]);
        if reserved != 0 {
            return Err(IpcWireError::NonZeroReserved(reserved));
        }

        Ok(match operation {
            IpcMemoryOperation::SetName => Self::SetName {
                device_address: get_u64(&body[8..16]),
                byte_count: get_u64(&body[16..24]),
                name: IpcMemoryName::from_c_bytes(&body[24..88]),
            },
            IpcMemoryOperation::DestroyName => Self::DestroyName {
                name: IpcMemoryName::from_c_bytes(&body[8..72]),
            },
            IpcMemoryOperation::Open => Self::Open {
                opened_device_address: get_u64(&body[8..16]),
                name: IpcMemoryName::from_c_bytes(&body[16..80]),
            },
            IpcMemoryOperation::Close => Self::Close {
                device_address: get_u64(&body[8..16]),
            },
        })
    }

    pub fn decode_frame(frame: &[u8]) -> Result<Self, IpcWireError> {
        require_len("IPC memory request frame", frame, IPC_MEMORY_REQUEST_SIZE)?;
        let packet_type = get_u32(&frame[0..4]);
        if packet_type != IPC_MEMORY_REQUEST_PACKET {
            return Err(IpcWireError::UnexpectedPacketType {
                expected: IPC_MEMORY_REQUEST_PACKET,
                actual: packet_type,
            });
        }
        Self::decode_body(&frame[4..])
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IpcResponse {
    pub operation: IpcMemoryOperation,
    pub status: u32,
}

impl IpcResponse {
    pub fn encode_body(self) -> [u8; IPC_RESPONSE_BODY_SIZE] {
        let mut body = [0; IPC_RESPONSE_BODY_SIZE];
        put_u32(&mut body[0..4], self.operation as u32);
        put_u32(&mut body[4..8], self.status);
        body
    }

    pub fn encode_frame(self) -> [u8; IPC_RESPONSE_SIZE] {
        let mut frame = [0; IPC_RESPONSE_SIZE];
        put_u32(&mut frame[0..4], IPC_MEMORY_RESPONSE_PACKET);
        frame[4..].copy_from_slice(&self.encode_body());
        frame
    }

    pub fn decode_body(body: &[u8]) -> Result<Self, IpcWireError> {
        require_len("IPC response body", body, IPC_RESPONSE_BODY_SIZE)?;
        Ok(Self {
            operation: IpcMemoryOperation::try_from(get_u32(&body[0..4]))?,
            status: get_u32(&body[4..8]),
        })
    }

    pub fn decode_frame(frame: &[u8]) -> Result<Self, IpcWireError> {
        require_len("IPC response frame", frame, IPC_RESPONSE_SIZE)?;
        let packet_type = get_u32(&frame[0..4]);
        if packet_type != IPC_MEMORY_RESPONSE_PACKET {
            return Err(IpcWireError::UnexpectedPacketType {
                expected: IPC_MEMORY_RESPONSE_PACKET,
                actual: packet_type,
            });
        }
        Self::decode_body(&frame[4..])
    }

    pub fn check_for(self, request: &IpcMemoryRequest) -> Result<(), IpcWireError> {
        let expected = request.operation();
        if self.operation != expected {
            return Err(IpcWireError::MismatchedOperation {
                expected,
                actual: self.operation,
            });
        }
        if self.status != 0 {
            return Err(IpcWireError::RemoteStatus(self.status));
        }
        Ok(())
    }
}

pub const fn response_packet_type(request_packet_type: u32) -> Option<u32> {
    match request_packet_type {
        KERNEL_RECORD_REQUEST_PACKET => Some(KERNEL_RECORD_RESPONSE_PACKET),
        IPC_MEMORY_REQUEST_PACKET => Some(IPC_MEMORY_RESPONSE_PACKET),
        _ => None,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum IpcWireError {
    #[error("{kind} has length {actual}, expected {expected}")]
    UnexpectedLength {
        kind: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("unknown IPC memory operation: {0}")]
    UnknownOperation(u32),
    #[error("unexpected packet type {actual}, expected {expected}")]
    UnexpectedPacketType { expected: u32, actual: u32 },
    #[error("IPCMemRecord reserved field is non-zero: {0}")]
    NonZeroReserved(u32),
    #[error("response operation {actual:?} does not match request {expected:?}")]
    MismatchedOperation {
        expected: IpcMemoryOperation,
        actual: IpcMemoryOperation,
    },
    #[error("IPC peer returned status {0}")]
    RemoteStatus(u32),
}

fn require_len(kind: &'static str, bytes: &[u8], expected: usize) -> Result<(), IpcWireError> {
    if bytes.len() == expected {
        Ok(())
    } else {
        Err(IpcWireError::UnexpectedLength {
            kind,
            expected,
            actual: bytes.len(),
        })
    }
}

fn get_u32(bytes: &[u8]) -> u32 {
    u32::from_le_bytes(bytes.try_into().expect("four-byte field"))
}

fn get_u64(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(bytes.try_into().expect("eight-byte field"))
}

fn put_u32(target: &mut [u8], value: u32) {
    target.copy_from_slice(&value.to_le_bytes());
}

fn put_u64(target: &mut [u8], value: u64) {
    target.copy_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(request: IpcMemoryRequest) {
        let frame = request.encode_frame();
        assert_eq!(IpcMemoryRequest::decode_frame(&frame).unwrap(), request);
    }

    #[test]
    fn packet_type_map_matches_supported_pairs() {
        assert_eq!(response_packet_type(1001), Some(3001));
        assert_eq!(response_packet_type(1002), Some(3000));
        assert_eq!(response_packet_type(9999), None);
    }

    #[test]
    fn set_name_has_exact_92_byte_wire_layout() {
        let request = IpcMemoryRequest::SetName {
            device_address: 0x0102_0304_0506_0708,
            byte_count: 0x1112_1314_1516_1718,
            name: IpcMemoryName::from("shared"),
        };
        let frame = request.encode_frame();

        assert_eq!(frame.len(), 92);
        assert_eq!(&frame[0..4], &1002_u32.to_le_bytes());
        assert_eq!(&frame[4..8], &0_u32.to_le_bytes());
        assert_eq!(&frame[8..12], &[0; 4]);
        assert_eq!(&frame[12..20], &0x0102_0304_0506_0708_u64.to_le_bytes());
        assert_eq!(&frame[20..28], &0x1112_1314_1516_1718_u64.to_le_bytes());
        assert_eq!(&frame[28..35], b"shared\0");
        assert!(frame[35..].iter().all(|byte| *byte == 0));
        round_trip(request);
    }

    #[test]
    fn all_memory_operations_round_trip() {
        round_trip(IpcMemoryRequest::DestroyName {
            name: IpcMemoryName::from("buffer-name"),
        });
        round_trip(IpcMemoryRequest::Open {
            opened_device_address: 0x1234,
            name: IpcMemoryName::from("buffer-name"),
        });
        round_trip(IpcMemoryRequest::Close {
            device_address: 0x5678,
        });
    }

    #[test]
    fn names_stop_at_nul_and_truncate_to_63_bytes() {
        let nul = IpcMemoryName::from_c_bytes(b"abc\0ignored");
        assert_eq!(nul.as_bytes(), b"abc");

        let long = IpcMemoryName::from_c_bytes(&[b'x'; 80]);
        assert_eq!(long.as_bytes(), &[b'x'; 63]);
        assert_eq!(long.as_c_field()[63], 0);
    }

    #[test]
    fn response_requires_packet_operation_and_success_status() {
        let request = IpcMemoryRequest::Close {
            device_address: 0x1234,
        };
        let response = IpcResponse {
            operation: IpcMemoryOperation::Close,
            status: 0,
        };
        let frame = response.encode_frame();
        assert_eq!(IpcResponse::decode_frame(&frame).unwrap(), response);
        assert_eq!(response.check_for(&request), Ok(()));

        assert_eq!(
            IpcResponse {
                operation: IpcMemoryOperation::Open,
                status: 0,
            }
            .check_for(&request),
            Err(IpcWireError::MismatchedOperation {
                expected: IpcMemoryOperation::Close,
                actual: IpcMemoryOperation::Open,
            })
        );
        assert_eq!(
            IpcResponse {
                operation: IpcMemoryOperation::Close,
                status: 17,
            }
            .check_for(&request),
            Err(IpcWireError::RemoteStatus(17))
        );
    }

    #[test]
    fn malformed_frames_are_rejected() {
        assert!(matches!(
            IpcMemoryRequest::decode_frame(&[0; 91]),
            Err(IpcWireError::UnexpectedLength { .. })
        ));

        let mut wrong_type = IpcMemoryRequest::Close { device_address: 0 }.encode_frame();
        wrong_type[0..4].copy_from_slice(&9999_u32.to_le_bytes());
        assert_eq!(
            IpcMemoryRequest::decode_frame(&wrong_type),
            Err(IpcWireError::UnexpectedPacketType {
                expected: IPC_MEMORY_REQUEST_PACKET,
                actual: 9999,
            })
        );

        let mut reserved = IpcMemoryRequest::Close { device_address: 0 }.encode_frame();
        reserved[8..12].copy_from_slice(&1_u32.to_le_bytes());
        assert_eq!(
            IpcMemoryRequest::decode_frame(&reserved),
            Err(IpcWireError::NonZeroReserved(1))
        );
    }
}
