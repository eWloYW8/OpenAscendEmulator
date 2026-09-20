use crate::prof_stub_packet::{ProfStubPacket, ProfStubPacketError};
use serde::Serialize;

pub const INSTRUCTION_LOG_PAYLOAD_BYTES: usize = 424;
pub const INSTRUCTION_LOG_TEXT_FIELD_BYTES: usize = 200;
pub const ICACHE_LOG_PAYLOAD_BYTES: usize = 40;
pub const MTE_LOG_PAYLOAD_BYTES: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProfStubCoreKind {
    Cube,
    Vector0,
    Vector1,
}

impl ProfStubCoreKind {
    pub const fn from_code(code: u32) -> Option<Self> {
        match code {
            0 => Some(Self::Cube),
            1 => Some(Self::Vector0),
            2 => Some(Self::Vector1),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InstructionLog<'a> {
    packet_type: u32,
    payload: &'a [u8],
}

impl<'a> InstructionLog<'a> {
    pub fn packet_type(&self) -> u32 {
        self.packet_type
    }

    pub fn time_word(&self) -> u64 {
        read_u64(self.payload, 0)
    }

    pub fn pc(&self) -> u64 {
        read_u64(self.payload, 8)
    }

    pub fn core_index(&self) -> u32 {
        read_u32(self.payload, 16)
    }

    pub fn core_kind_code(&self) -> u32 {
        read_u32(self.payload, 20)
    }

    pub fn core_kind(&self) -> Option<ProfStubCoreKind> {
        ProfStubCoreKind::from_code(self.core_kind_code())
    }

    pub fn description_field(&self) -> &'a [u8] {
        &self.payload[24..24 + INSTRUCTION_LOG_TEXT_FIELD_BYTES]
    }

    pub fn detail_field(&self) -> &'a [u8] {
        &self.payload[224..224 + INSTRUCTION_LOG_TEXT_FIELD_BYTES]
    }

    pub fn description_bytes(&self) -> &'a [u8] {
        nul_prefix(self.description_field())
    }

    pub fn detail_bytes(&self) -> &'a [u8] {
        nul_prefix(self.detail_field())
    }

    pub fn raw_payload(&self) -> &'a [u8] {
        self.payload
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ICacheLog<'a> {
    payload: &'a [u8],
}

impl<'a> ICacheLog<'a> {
    pub fn time_word(&self) -> u64 {
        read_u64(self.payload, 0)
    }

    pub fn pc(&self) -> u64 {
        read_u64(self.payload, 8)
    }

    pub fn core_index(&self) -> u32 {
        read_u32(self.payload, 16)
    }

    pub fn core_kind_code(&self) -> u32 {
        read_u32(self.payload, 20)
    }

    pub fn core_kind(&self) -> Option<ProfStubCoreKind> {
        ProfStubCoreKind::from_code(self.core_kind_code())
    }

    pub fn access_size(&self) -> u32 {
        read_u32(self.payload, 24)
    }

    pub fn cache_type(&self) -> u32 {
        read_u32(self.payload, 28)
    }

    pub fn last_byte(&self) -> u8 {
        self.payload[32]
    }

    pub fn uninterpreted_tail(&self) -> &'a [u8] {
        &self.payload[33..]
    }

    pub fn raw_payload(&self) -> &'a [u8] {
        self.payload
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfStubTraceLog<'a> {
    Instruction(InstructionLog<'a>),
    ICache(ICacheLog<'a>),
    MteRaw(&'a [u8]),
}

impl<'a> ProfStubTraceLog<'a> {
    pub fn decode(packet: ProfStubPacket<'a>) -> Result<Option<Self>, ProfStubPacketError> {
        let packet_type = packet.packet_type();
        let payload = packet.payload();
        let expected = match packet_type {
            20 | 21 => INSTRUCTION_LOG_PAYLOAD_BYTES,
            22 => ICACHE_LOG_PAYLOAD_BYTES,
            23 => MTE_LOG_PAYLOAD_BYTES,
            _ => return Ok(None),
        };
        if payload.len() != expected {
            return Err(ProfStubPacketError::InvalidPayloadLength {
                packet_type,
                expected,
                actual: payload.len(),
            });
        }
        Ok(Some(match packet_type {
            20 | 21 => Self::Instruction(InstructionLog {
                packet_type,
                payload,
            }),
            22 => Self::ICache(ICacheLog { payload }),
            23 => Self::MteRaw(payload),
            _ => unreachable!("matched packet type above"),
        }))
    }
}

fn read_u64(payload: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(
        payload[offset..offset + 8]
            .try_into()
            .expect("validated payload"),
    )
}

fn read_u32(payload: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(
        payload[offset..offset + 4]
            .try_into()
            .expect("validated payload"),
    )
}

fn nul_prefix(field: &[u8]) -> &[u8] {
    &field[..field
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(field.len())]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instruction_fields_follow_the_vendor_offsets() {
        let mut payload = [0_u8; INSTRUCTION_LOG_PAYLOAD_BYTES];
        payload[0..8].copy_from_slice(&1290_u64.to_le_bytes());
        payload[8..16].copy_from_slice(&0x10d0d000_u64.to_le_bytes());
        payload[16..20].copy_from_slice(&3_u32.to_le_bytes());
        payload[20..24].copy_from_slice(&1_u32.to_le_bytes());
        payload[24..28].copy_from_slice(b"ADD\0");
        payload[224..228].copy_from_slice(b"X0=1");
        for packet_type in [20, 21] {
            let packet = ProfStubPacket::new(packet_type, &payload).unwrap();
            let Some(ProfStubTraceLog::Instruction(log)) =
                ProfStubTraceLog::decode(packet).unwrap()
            else {
                panic!("expected instruction log");
            };
            assert_eq!(log.packet_type(), packet_type);
            assert_eq!(log.time_word(), 1290);
            assert_eq!(log.pc(), 0x10d0d000);
            assert_eq!(log.core_index(), 3);
            assert_eq!(log.core_kind(), Some(ProfStubCoreKind::Vector0));
            assert_eq!(log.description_bytes(), b"ADD");
            assert_eq!(log.detail_bytes(), b"X0=1");
            assert_eq!(log.raw_payload(), payload);
        }
    }

    #[test]
    fn icache_fields_and_uninterpreted_tail_remain_distinct() {
        let mut payload = [0_u8; ICACHE_LOG_PAYLOAD_BYTES];
        payload[0..8].copy_from_slice(&927_u64.to_le_bytes());
        payload[8..16].copy_from_slice(&0x10d0d080_u64.to_le_bytes());
        payload[24..28].copy_from_slice(&128_u32.to_le_bytes());
        payload[28..32].copy_from_slice(&1_u32.to_le_bytes());
        payload[32] = 1;
        payload[33] = 0xa5;
        let packet = ProfStubPacket::new(22, &payload).unwrap();
        let Some(ProfStubTraceLog::ICache(log)) = ProfStubTraceLog::decode(packet).unwrap() else {
            panic!("expected iCache log");
        };
        assert_eq!(log.time_word(), 927);
        assert_eq!(log.pc(), 0x10d0d080);
        assert_eq!(log.core_kind(), Some(ProfStubCoreKind::Cube));
        assert_eq!(log.access_size(), 128);
        assert_eq!(log.cache_type(), 1);
        assert_eq!(log.last_byte(), 1);
        assert_eq!(log.uninterpreted_tail()[0], 0xa5);
    }

    #[test]
    fn typed_handlers_reject_short_but_registry_accepted_payloads() {
        for (packet_type, payload, expected) in
            [(20, 423, 424), (21, 0, 424), (22, 39, 40), (23, 63, 64)]
        {
            let bytes = vec![0; payload];
            let packet = ProfStubPacket::new(packet_type, &bytes).unwrap();
            assert_eq!(
                ProfStubTraceLog::decode(packet),
                Err(ProfStubPacketError::InvalidPayloadLength {
                    packet_type,
                    expected,
                    actual: payload,
                })
            );
        }
        let other = ProfStubPacket::new(5, &[]).unwrap();
        assert_eq!(ProfStubTraceLog::decode(other), Ok(None));
        let mte = [0x5a; MTE_LOG_PAYLOAD_BYTES];
        let packet = ProfStubPacket::new(23, &mte).unwrap();
        assert_eq!(
            ProfStubTraceLog::decode(packet),
            Ok(Some(ProfStubTraceLog::MteRaw(&mte)))
        );
    }
}
