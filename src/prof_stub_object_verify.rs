
use crate::device_elf::ProjectedDeviceKernel;
use crate::prof_stub_packet::{ProfStubPacket, ProfStubPacketError};
use crate::prof_stub_stream::inspect_prof_stub_stream;
use crate::prof_stub_trace::ProfStubTraceLog;
use serde::Serialize;
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProfStubObjectVerification {
    pub stream_bytes: usize,
    pub instruction_events: usize,
    pub checked_events: usize,
    pub unique_checked_pcs: usize,
    pub missing_text_pc: usize,
    pub missing_binary_word: usize,
    pub text_pc_mismatches: usize,
    pub object_word_mismatches: usize,
    pub pcs_not_fetchable: usize,
    pub all_instruction_words_match_object: bool,
    pub examples: Vec<ProfStubObjectIssue>,
    pub omitted_examples: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProfStubObjectIssue {
    pub packet_offset: usize,
    pub packet_type: u32,
    pub pc: u64,
    pub text_pc: Option<u64>,
    pub text_word: Option<u32>,
    pub object_word: Option<u32>,
    pub reason: ProfStubObjectIssueReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProfStubObjectIssueReason {
    MissingTextPc,
    MissingBinaryWord,
    TextPcMismatch,
    ObjectWordMismatch,
    PcNotFetchable,
}

pub fn verify_prof_stub_object(
    stream: &[u8],
    projected: &ProjectedDeviceKernel<'_>,
    max_examples: usize,
) -> Result<ProfStubObjectVerification, ProfStubPacketError> {
    verify_with_word_source(stream, max_examples, |pc| projected.fetch_word(pc).ok())
}

fn verify_with_word_source(
    stream: &[u8],
    max_examples: usize,
    mut fetch_word: impl FnMut(u64) -> Option<u32>,
) -> Result<ProfStubObjectVerification, ProfStubPacketError> {
    inspect_prof_stub_stream(stream, 0)?;
    let mut result = ProfStubObjectVerification {
        stream_bytes: stream.len(),
        instruction_events: 0,
        checked_events: 0,
        unique_checked_pcs: 0,
        missing_text_pc: 0,
        missing_binary_word: 0,
        text_pc_mismatches: 0,
        object_word_mismatches: 0,
        pcs_not_fetchable: 0,
        all_instruction_words_match_object: false,
        examples: Vec::new(),
        omitted_examples: 0,
    };
    let mut checked_pcs = BTreeSet::new();
    let mut offset = 0;
    while offset < stream.len() {
        let (packet, consumed) = ProfStubPacket::decode_prefix(&stream[offset..])?
            .ok_or(ProfStubPacketError::Incomplete)?;
        if let Some(ProfStubTraceLog::Instruction(log)) = ProfStubTraceLog::decode(packet)? {
            result.instruction_events += 1;
            let description = log
                .description_bytes()
                .split(|byte| *byte == 0)
                .next()
                .unwrap_or_default();
            let text_pc =
                parse_hex_field(description, b"(PC: 0x", 16, false).map(|(value, _)| value);
            let text_word = parse_hex_field(description, b"Binary: 0x", 8, true)
                .and_then(|(value, _)| u32::try_from(value).ok());
            let object_word = fetch_word(log.pc());
            let issue = if text_pc.is_none() {
                result.missing_text_pc += 1;
                Some(ProfStubObjectIssueReason::MissingTextPc)
            } else if text_word.is_none() {
                result.missing_binary_word += 1;
                Some(ProfStubObjectIssueReason::MissingBinaryWord)
            } else if text_pc != Some(log.pc()) {
                result.text_pc_mismatches += 1;
                Some(ProfStubObjectIssueReason::TextPcMismatch)
            } else if object_word.is_none() {
                result.pcs_not_fetchable += 1;
                Some(ProfStubObjectIssueReason::PcNotFetchable)
            } else if text_word != object_word {
                result.object_word_mismatches += 1;
                Some(ProfStubObjectIssueReason::ObjectWordMismatch)
            } else {
                result.checked_events += 1;
                checked_pcs.insert(log.pc());
                None
            };
            if let Some(reason) = issue {
                if result.examples.len() < max_examples {
                    result.examples.push(ProfStubObjectIssue {
                        packet_offset: offset,
                        packet_type: log.packet_type(),
                        pc: log.pc(),
                        text_pc,
                        text_word,
                        object_word,
                        reason,
                    });
                } else {
                    result.omitted_examples += 1;
                }
            }
        }
        offset += consumed;
    }
    result.unique_checked_pcs = checked_pcs.len();
    result.all_instruction_words_match_object =
        result.instruction_events != 0 && result.checked_events == result.instruction_events;
    Ok(result)
}

fn parse_hex_field(
    text: &[u8],
    prefix: &[u8],
    max_digits: usize,
    exact: bool,
) -> Option<(u64, usize)> {
    let start = text
        .windows(prefix.len())
        .position(|window| window == prefix)?
        + prefix.len();
    let digits = text[start..]
        .iter()
        .take_while(|byte| byte.is_ascii_hexdigit())
        .count();
    if digits == 0 || digits > max_digits || (exact && digits != max_digits) {
        return None;
    }
    if text.get(start + digits) != Some(&b')') {
        return None;
    }
    let raw = std::str::from_utf8(&text[start..start + digits]).ok()?;
    Some((u64::from_str_radix(raw, 16).ok()?, digits))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prof_stub_packet::{encode_log_translate_start, encode_log_translate_stop};

    fn one_instruction_stream(description: &[u8], pc: u64) -> Vec<u8> {
        let mut payload = [0_u8; 424];
        payload[8..16].copy_from_slice(&pc.to_le_bytes());
        payload[24..24 + description.len()].copy_from_slice(description);
        [
            encode_log_translate_start(b"/tmp", b"Kernel").unwrap(),
            ProfStubPacket::new(21, &payload).unwrap().encode_frame(),
            encode_log_translate_stop(),
        ]
        .concat()
    }

    #[test]
    fn matches_raw_pc_text_pc_binary_word_and_supplied_object_word() {
        let stream = one_instruction_stream(
            b"(PC: 0x1000) SCALAR: (Binary: 0x073a7fa0) MOV_XD_IMM",
            0x1000,
        );
        let result =
            verify_with_word_source(&stream, 4, |pc| (pc == 0x1000).then_some(0x073a7fa0)).unwrap();
        assert_eq!(result.instruction_events, 1);
        assert_eq!(result.checked_events, 1);
        assert_eq!(result.unique_checked_pcs, 1);
        assert!(result.all_instruction_words_match_object);
        assert!(result.examples.is_empty());
    }

    #[test]
    fn mismatched_object_word_is_not_called_verified() {
        let stream = one_instruction_stream(b"(PC: 0x1000) (Binary: 0x073a7fa0)", 0x1000);
        let result = verify_with_word_source(&stream, 1, |_| Some(0x077b0010)).unwrap();
        assert_eq!(result.object_word_mismatches, 1);
        assert_eq!(
            result.examples[0].reason,
            ProfStubObjectIssueReason::ObjectWordMismatch
        );
        assert!(!result.all_instruction_words_match_object);
    }

    #[test]
    fn mismatched_text_pc_and_missing_text_are_kept_separate() {
        let stream = one_instruction_stream(b"(PC: 0x1004) (Binary: 0x073a7fa0)", 0x1000);
        let result = verify_with_word_source(&stream, 1, |_| Some(0x073a7fa0)).unwrap();
        assert_eq!(result.text_pc_mismatches, 1);
        assert!(!result.all_instruction_words_match_object);
        let stream = one_instruction_stream(b"no parseable PC or binary", 0x1000);
        let result = verify_with_word_source(&stream, 0, |_| Some(0x073a7fa0)).unwrap();
        assert_eq!(result.missing_text_pc, 1);
        assert_eq!(result.omitted_examples, 1);
    }

    #[test]
    fn ignores_text_after_description_nul() {
        let stream =
            one_instruction_stream(b"no PC here\0(PC: 0x1000) (Binary: 0x073a7fa0)", 0x1000);
        let result = verify_with_word_source(&stream, 1, |_| Some(0x073a7fa0)).unwrap();
        assert_eq!(result.missing_text_pc, 1);
        assert!(!result.all_instruction_words_match_object);
    }

    #[test]
    fn missing_word_or_unfetchable_pc_fails_closed() {
        let stream = one_instruction_stream(b"(PC: 0x1000) no word", 0x1000);
        let result = verify_with_word_source(&stream, 1, |_| Some(0x073a7fa0)).unwrap();
        assert_eq!(result.missing_binary_word, 1);
        assert!(!result.all_instruction_words_match_object);

        let stream = one_instruction_stream(b"(PC: 0x1000) (Binary: 0x073a7fa0)", 0x1000);
        let result = verify_with_word_source(&stream, 1, |_| None).unwrap();
        assert_eq!(result.pcs_not_fetchable, 1);
        assert!(!result.all_instruction_words_match_object);
    }

    #[test]
    fn zero_instruction_stream_is_not_a_success() {
        let stream = [
            encode_log_translate_start(b"/tmp", b"Kernel").unwrap(),
            encode_log_translate_stop(),
        ]
        .concat();
        let result = verify_with_word_source(&stream, 1, |_| None).unwrap();
        assert_eq!(result.instruction_events, 0);
        assert!(!result.all_instruction_words_match_object);
    }
}
