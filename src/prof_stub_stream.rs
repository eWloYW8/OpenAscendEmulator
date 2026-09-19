
use crate::prof_stub_packet::{ProfStubPacket, ProfStubPacketError, decode_log_translate_start};
use crate::prof_stub_trace::{ProfStubCoreKind, ProfStubTraceLog};
use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProfStubStreamSummary {
    pub stream_bytes: usize,
    pub packet_count: usize,
    pub counts_by_type: BTreeMap<u32, usize>,
    pub records: Vec<ProfStubRecordSummary>,
    pub omitted_records: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProfStubRecordSummary {
    pub offset: usize,
    pub packet_type: u32,
    pub payload_bytes: usize,
    pub detail: ProfStubRecordDetail,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ProfStubRecordDetail {
    LogTranslateStart {
        output_path_bytes: usize,
        kernel_name_bytes: usize,
        kernel_name_utf8: Option<String>,
    },
    LogTranslateStop,
    Instruction {
        time_word: u64,
        pc: u64,
        core_index: u32,
        core_kind_code: u32,
        core_kind: Option<ProfStubCoreKind>,
        description_utf8: Option<String>,
        detail_utf8: Option<String>,
    },
    ICache {
        time_word: u64,
        pc: u64,
        core_index: u32,
        core_kind_code: u32,
        core_kind: Option<ProfStubCoreKind>,
        access_size: u32,
        cache_type: u32,
        last_byte: u8,
        uninterpreted_tail_nonzero_bytes: usize,
    },
    MteRaw,
    Other,
}

pub fn inspect_prof_stub_stream(
    input: &[u8],
    max_records: usize,
) -> Result<ProfStubStreamSummary, ProfStubPacketError> {
    let mut summary = ProfStubStreamSummary {
        stream_bytes: input.len(),
        packet_count: 0,
        counts_by_type: BTreeMap::new(),
        records: Vec::new(),
        omitted_records: 0,
    };
    let mut offset = 0;
    while offset < input.len() {
        let (packet, consumed) = ProfStubPacket::decode_prefix(&input[offset..])?
            .ok_or(ProfStubPacketError::Incomplete)?;
        let detail = record_detail(packet)?;
        *summary
            .counts_by_type
            .entry(packet.packet_type())
            .or_default() += 1;
        summary.packet_count += 1;
        if summary.records.len() < max_records {
            summary.records.push(ProfStubRecordSummary {
                offset,
                packet_type: packet.packet_type(),
                payload_bytes: packet.payload().len(),
                detail,
            });
        } else {
            summary.omitted_records += 1;
        }
        offset += consumed;
    }
    Ok(summary)
}

fn record_detail(packet: ProfStubPacket<'_>) -> Result<ProfStubRecordDetail, ProfStubPacketError> {
    if packet.packet_type() == 4 {
        let (path, name) = decode_log_translate_start(packet)?;
        return Ok(ProfStubRecordDetail::LogTranslateStart {
            output_path_bytes: nul_prefix(path).len(),
            kernel_name_bytes: nul_prefix(name).len(),
            kernel_name_utf8: std::str::from_utf8(nul_prefix(name))
                .ok()
                .map(str::to_owned),
        });
    }
    if packet.packet_type() == 5 {
        return Ok(ProfStubRecordDetail::LogTranslateStop);
    }
    Ok(match ProfStubTraceLog::decode(packet)? {
        Some(ProfStubTraceLog::Instruction(log)) => ProfStubRecordDetail::Instruction {
            time_word: log.time_word(),
            pc: log.pc(),
            core_index: log.core_index(),
            core_kind_code: log.core_kind_code(),
            core_kind: log.core_kind(),
            description_utf8: std::str::from_utf8(log.description_bytes())
                .ok()
                .map(str::to_owned),
            detail_utf8: std::str::from_utf8(log.detail_bytes())
                .ok()
                .map(str::to_owned),
        },
        Some(ProfStubTraceLog::ICache(log)) => ProfStubRecordDetail::ICache {
            time_word: log.time_word(),
            pc: log.pc(),
            core_index: log.core_index(),
            core_kind_code: log.core_kind_code(),
            core_kind: log.core_kind(),
            access_size: log.access_size(),
            cache_type: log.cache_type(),
            last_byte: log.last_byte(),
            uninterpreted_tail_nonzero_bytes: log
                .uninterpreted_tail()
                .iter()
                .filter(|byte| **byte != 0)
                .count(),
        },
        Some(ProfStubTraceLog::MteRaw(_)) => ProfStubRecordDetail::MteRaw,
        None => ProfStubRecordDetail::Other,
    })
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
    use crate::prof_stub_packet::{encode_log_translate_start, encode_log_translate_stop};

    #[test]
    fn every_record_is_validated_even_when_preview_is_bounded() {
        let start = encode_log_translate_start(b"/tmp/output", b"ClearL2Cache").unwrap();
        let mut instruction_payload = [0; 424];
        instruction_payload[8..16].copy_from_slice(&0x10d0d000_u64.to_le_bytes());
        instruction_payload[24..28].copy_from_slice(b"ADD\0");
        let instruction = ProfStubPacket::new(20, &instruction_payload)
            .unwrap()
            .encode_frame();
        let stop = encode_log_translate_stop();
        let bytes = [start, instruction, stop].concat();
        let summary = inspect_prof_stub_stream(&bytes, 1).unwrap();
        assert_eq!(summary.packet_count, 3);
        assert_eq!(summary.counts_by_type.get(&4), Some(&1));
        assert_eq!(summary.counts_by_type.get(&20), Some(&1));
        assert_eq!(summary.counts_by_type.get(&5), Some(&1));
        assert_eq!(summary.records.len(), 1);
        assert_eq!(summary.omitted_records, 2);
        assert!(matches!(
            summary.records[0].detail,
            ProfStubRecordDetail::LogTranslateStart { .. }
        ));
        let mut malformed = bytes;
        malformed[5128 + 4..5128 + 8].copy_from_slice(&423_u32.to_le_bytes());
        assert_eq!(
            inspect_prof_stub_stream(&malformed, 1),
            Err(ProfStubPacketError::InvalidPayloadLength {
                packet_type: 20,
                expected: 424,
                actual: 423,
            })
        );
    }

    #[test]
    fn incomplete_stream_is_rejected() {
        assert_eq!(
            inspect_prof_stub_stream(&[4], 0),
            Err(ProfStubPacketError::Incomplete)
        );
        let frame = encode_log_translate_start(b"x", b"y").unwrap();
        assert_eq!(
            inspect_prof_stub_stream(&frame[..frame.len() - 1], 0),
            Err(ProfStubPacketError::Incomplete)
        );
    }
}
