use crate::architecture::Architecture;
use crate::flow::{
    ConditionalJump, JumpCompare, JumpCompareOffset, JumpOffsetSource, UnconditionalJump,
};
use crate::prof_stub_packet::{ProfStubPacket, ProfStubPacketError};
use crate::prof_stub_stream::inspect_prof_stub_stream;
use crate::prof_stub_trace::ProfStubTraceLog;
use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProfStubBranchKind {
    Unconditional,
    Conditional,
    Compare,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProfStubBranchEdgeIssueReason {
    InvalidDescription,
    TextPcMismatch,
    UnsupportedWord,
    RegisterOffsetUnavailable,
    OutsideCandidates,
    NoSuccessor,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProfStubBranchEdgeIssue {
    pub reason: ProfStubBranchEdgeIssueReason,
    pub packet_offset: usize,
    pub core_index: u32,
    pub core_kind_code: u32,
    pub branch_kind: ProfStubBranchKind,
    pub pc: u64,
    pub word: Option<u32>,
    pub observed_successor_pc: Option<u64>,
    pub possible_target_pcs: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProfStubBranchEdgeSummary {
    pub architecture: Architecture,
    pub stream_bytes: usize,
    pub type21_instruction_events: usize,
    pub branch_records: usize,
    pub unconditional_records: usize,
    pub conditional_records: usize,
    pub compare_records: usize,
    pub unconditional_matches: usize,
    pub conditional_taken_candidates: usize,
    pub conditional_fallthrough_candidates: usize,
    pub conditional_ambiguous_candidates: usize,
    pub register_offset_unavailable: usize,
    pub outside_candidate_edges: usize,
    pub no_successor: usize,
    pub malformed_branch_records: usize,
    pub issues: Vec<ProfStubBranchEdgeIssue>,
    pub omitted_issues: usize,
}

#[derive(Debug, Clone, Copy)]
struct PendingBranch {
    packet_offset: usize,
    core_index: u32,
    core_kind_code: u32,
    kind: ProfStubBranchKind,
    pc: u64,
    word: Option<u32>,
    taken_pc: Option<u64>,
    fallthrough_pc: Option<u64>,
}

impl PendingBranch {
    fn possible_target_pcs(self) -> Vec<u64> {
        let mut result = Vec::with_capacity(2);
        if let Some(target) = self.taken_pc {
            result.push(target);
        }
        if let Some(fallthrough) = self.fallthrough_pc
            && !result.contains(&fallthrough)
        {
            result.push(fallthrough);
        }
        result
    }
}

pub fn inspect_prof_stub_branch_edges(
    stream: &[u8],
    architecture: Architecture,
    max_issues: usize,
) -> Result<ProfStubBranchEdgeSummary, ProfStubPacketError> {
    inspect_prof_stub_stream(stream, 0)?;
    let mut summary = ProfStubBranchEdgeSummary {
        architecture,
        stream_bytes: stream.len(),
        type21_instruction_events: 0,
        branch_records: 0,
        unconditional_records: 0,
        conditional_records: 0,
        compare_records: 0,
        unconditional_matches: 0,
        conditional_taken_candidates: 0,
        conditional_fallthrough_candidates: 0,
        conditional_ambiguous_candidates: 0,
        register_offset_unavailable: 0,
        outside_candidate_edges: 0,
        no_successor: 0,
        malformed_branch_records: 0,
        issues: Vec::new(),
        omitted_issues: 0,
    };
    let mut pending = BTreeMap::new();
    let mut offset = 0;
    while offset < stream.len() {
        let (packet, consumed) = ProfStubPacket::decode_prefix(&stream[offset..])?
            .ok_or(ProfStubPacketError::Incomplete)?;
        if let Some(ProfStubTraceLog::Instruction(log)) = ProfStubTraceLog::decode(packet)?
            && log.packet_type() == 21
        {
            summary.type21_instruction_events += 1;
            let key = (log.core_index(), log.core_kind_code());
            let description = std::str::from_utf8(log.description_bytes()).ok();
            if log.pc() == 0x1111_1111
                && description.is_some_and(|text| text.ends_with(" END_LABEL"))
            {
                if let Some(branch) = pending.remove(&key) {
                    record_no_successor(&mut summary, branch, max_issues);
                }
                offset += consumed;
                continue;
            }
            if let Some(branch) = pending.remove(&key) {
                assess_successor(&mut summary, branch, log.pc(), max_issues);
            }
            let Some(kind) = description.and_then(branch_kind) else {
                offset += consumed;
                continue;
            };
            summary.branch_records += 1;
            match kind {
                ProfStubBranchKind::Unconditional => summary.unconditional_records += 1,
                ProfStubBranchKind::Conditional => summary.conditional_records += 1,
                ProfStubBranchKind::Compare => summary.compare_records += 1,
            }
            let Some((text_pc, word)) = description.and_then(parse_description_pc_word) else {
                summary.malformed_branch_records += 1;
                record_issue(
                    &mut summary,
                    PendingBranch {
                        packet_offset: offset,
                        core_index: key.0,
                        core_kind_code: key.1,
                        kind,
                        pc: log.pc(),
                        word: None,
                        taken_pc: None,
                        fallthrough_pc: None,
                    },
                    ProfStubBranchEdgeIssueReason::InvalidDescription,
                    None,
                    max_issues,
                );
                offset += consumed;
                continue;
            };
            if text_pc != log.pc() {
                summary.malformed_branch_records += 1;
                record_issue(
                    &mut summary,
                    PendingBranch {
                        packet_offset: offset,
                        core_index: key.0,
                        core_kind_code: key.1,
                        kind,
                        pc: log.pc(),
                        word: Some(word),
                        taken_pc: None,
                        fallthrough_pc: None,
                    },
                    ProfStubBranchEdgeIssueReason::TextPcMismatch,
                    None,
                    max_issues,
                );
                offset += consumed;
                continue;
            }
            let branch = decode_branch(offset, key, kind, log.pc(), word, architecture);
            match branch {
                Some(branch) => {
                    pending.insert(key, branch);
                }
                None => {
                    summary.malformed_branch_records += 1;
                    record_issue(
                        &mut summary,
                        PendingBranch {
                            packet_offset: offset,
                            core_index: key.0,
                            core_kind_code: key.1,
                            kind,
                            pc: log.pc(),
                            word: Some(word),
                            taken_pc: None,
                            fallthrough_pc: None,
                        },
                        ProfStubBranchEdgeIssueReason::UnsupportedWord,
                        None,
                        max_issues,
                    );
                }
            }
        }
        offset += consumed;
    }
    for (_, branch) in pending {
        record_no_successor(&mut summary, branch, max_issues);
    }
    Ok(summary)
}

fn branch_kind(description: &str) -> Option<ProfStubBranchKind> {
    match description.split_whitespace().last()? {
        "JUMP" => Some(ProfStubBranchKind::Unconditional),
        "JUMPC" => Some(ProfStubBranchKind::Conditional),
        "JUMPCMP" => Some(ProfStubBranchKind::Compare),
        _ => None,
    }
}

fn parse_description_pc_word(description: &str) -> Option<(u64, u32)> {
    let pc_text = description.split_once("(PC: 0x")?.1.split_once(')')?.0;
    if pc_text.is_empty()
        || pc_text.len() > 16
        || !pc_text.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return None;
    }
    let word_text = description.split_once("Binary: 0x")?.1.split_once(')')?.0;
    if word_text.len() != 8 || !word_text.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return None;
    }
    Some((
        u64::from_str_radix(pc_text, 16).ok()?,
        u32::from_str_radix(word_text, 16).ok()?,
    ))
}

fn decode_branch(
    packet_offset: usize,
    (core_index, core_kind_code): (u32, u32),
    kind: ProfStubBranchKind,
    pc: u64,
    word: u32,
    architecture: Architecture,
) -> Option<PendingBranch> {
    let zero_registers = [0; 32];
    let (taken_pc, fallthrough_pc) = match kind {
        ProfStubBranchKind::Unconditional => {
            let jump = UnconditionalJump::decode(architecture, word)?;
            let target = matches!(jump.offset_source, JumpOffsetSource::Immediate { .. })
                .then(|| jump.resolve(pc, &zero_registers).target_pc);
            (target, None)
        }
        ProfStubBranchKind::Conditional => {
            let jump = ConditionalJump::decode(architecture, word)?;
            let target = matches!(jump.offset_source, JumpOffsetSource::Immediate { .. })
                .then(|| jump.resolve(pc, &zero_registers, 1).target_pc);
            (target, Some(pc.wrapping_add(4)))
        }
        ProfStubBranchKind::Compare => {
            let jump = JumpCompare::decode(architecture, word)?;
            let target = matches!(jump.offset_source, JumpCompareOffset::Immediate { .. })
                .then(|| jump.taken_target(pc, &zero_registers).target_pc);
            (target, Some(pc.wrapping_add(4)))
        }
    };
    Some(PendingBranch {
        packet_offset,
        core_index,
        core_kind_code,
        kind,
        pc,
        word: Some(word),
        taken_pc,
        fallthrough_pc,
    })
}

fn assess_successor(
    summary: &mut ProfStubBranchEdgeSummary,
    branch: PendingBranch,
    observed_pc: u64,
    max_issues: usize,
) {
    let Some(taken_pc) = branch.taken_pc else {
        summary.register_offset_unavailable += 1;
        record_issue(
            summary,
            branch,
            ProfStubBranchEdgeIssueReason::RegisterOffsetUnavailable,
            Some(observed_pc),
            max_issues,
        );
        return;
    };
    if branch.kind == ProfStubBranchKind::Unconditional {
        if observed_pc == taken_pc {
            summary.unconditional_matches += 1;
        } else {
            summary.outside_candidate_edges += 1;
            record_issue(
                summary,
                branch,
                ProfStubBranchEdgeIssueReason::OutsideCandidates,
                Some(observed_pc),
                max_issues,
            );
        }
        return;
    }
    let fallthrough_pc = branch
        .fallthrough_pc
        .expect("conditional branch has fallthrough");
    match (observed_pc == taken_pc, observed_pc == fallthrough_pc) {
        (true, true) => summary.conditional_ambiguous_candidates += 1,
        (true, false) => summary.conditional_taken_candidates += 1,
        (false, true) => summary.conditional_fallthrough_candidates += 1,
        (false, false) => {
            summary.outside_candidate_edges += 1;
            record_issue(
                summary,
                branch,
                ProfStubBranchEdgeIssueReason::OutsideCandidates,
                Some(observed_pc),
                max_issues,
            );
        }
    }
}

fn record_no_successor(
    summary: &mut ProfStubBranchEdgeSummary,
    branch: PendingBranch,
    max_issues: usize,
) {
    summary.no_successor += 1;
    record_issue(
        summary,
        branch,
        ProfStubBranchEdgeIssueReason::NoSuccessor,
        None,
        max_issues,
    );
}

fn record_issue(
    summary: &mut ProfStubBranchEdgeSummary,
    branch: PendingBranch,
    reason: ProfStubBranchEdgeIssueReason,
    observed_successor_pc: Option<u64>,
    max_issues: usize,
) {
    if summary.issues.len() < max_issues {
        summary.issues.push(ProfStubBranchEdgeIssue {
            reason,
            packet_offset: branch.packet_offset,
            core_index: branch.core_index,
            core_kind_code: branch.core_kind_code,
            branch_kind: branch.kind,
            pc: branch.pc,
            word: branch.word,
            observed_successor_pc,
            possible_target_pcs: branch.possible_target_pcs(),
        });
    } else {
        summary.omitted_issues += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prof_stub_packet::{encode_log_translate_start, encode_log_translate_stop};

    fn instruction(packet_type: u32, core: u32, pc: u64, word: u32, name: &str) -> Vec<u8> {
        let description =
            format!("(PC: {pc:#x}) FLOWCTRL : (Binary: {word:#010x}) (ID: 000001) {name}");
        instruction_with_description(packet_type, core, pc, &description)
    }

    fn instruction_with_description(
        packet_type: u32,
        core: u32,
        pc: u64,
        description: &str,
    ) -> Vec<u8> {
        let mut payload = [0; 424];
        payload[8..16].copy_from_slice(&pc.to_le_bytes());
        payload[16..20].copy_from_slice(&core.to_le_bytes());
        payload[20..24].copy_from_slice(&1_u32.to_le_bytes());
        payload[24..24 + description.len()].copy_from_slice(description.as_bytes());
        ProfStubPacket::new(packet_type, &payload)
            .unwrap()
            .encode_frame()
    }

    fn stream(records: &[Vec<u8>]) -> Vec<u8> {
        let mut bytes = encode_log_translate_start(b"/tmp", b"Kernel").unwrap();
        for record in records {
            bytes.extend_from_slice(record);
        }
        bytes.extend_from_slice(&encode_log_translate_stop());
        bytes
    }

    #[test]
    fn uses_type21_and_separates_cores_and_branch_choices() {
        let bytes = stream(&[
            instruction(21, 0, 0x1000, 0x4000_0004, "JUMP"),
            instruction(21, 1, 0x2000, 0x4020_0002, "JUMPC"),
            instruction(20, 0, 0x1004, 0x0700_0000, "MOV_XD_IMM"),
            instruction(21, 0, 0x1010, 0x0700_0000, "MOV_XD_IMM"),
            instruction(21, 1, 0x2004, 0x0700_0000, "MOV_XD_IMM"),
        ]);
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let result = inspect_prof_stub_branch_edges(&bytes, architecture, 4).unwrap();
            assert_eq!(result.type21_instruction_events, 4);
            assert_eq!(result.branch_records, 2);
            assert_eq!(result.unconditional_matches, 1);
            assert_eq!(result.conditional_fallthrough_candidates, 1);
            assert_eq!(result.outside_candidate_edges, 0);
            assert!(result.issues.is_empty());
        }
    }

    #[test]
    fn outside_candidate_and_unknown_register_offset_are_not_verified() {
        let bytes = stream(&[
            instruction(21, 0, 0x1000, 0x4020_0002, "JUMPC"),
            instruction(21, 0, 0x3000, 0x0700_0000, "MOV_XD_IMM"),
            instruction(21, 0, 0x1000, 0x4002_3000, "JUMP"),
            instruction(21, 0, 0x1004, 0x0700_0000, "MOV_XD_IMM"),
        ]);
        let result = inspect_prof_stub_branch_edges(&bytes, Architecture::Dav3510, 1).unwrap();
        assert_eq!(result.outside_candidate_edges, 1);
        assert_eq!(result.register_offset_unavailable, 1);
        assert_eq!(result.issues.len(), 1);
        assert_eq!(result.omitted_issues, 1);
        assert_eq!(result.issues[0].possible_target_pcs, vec![0x1008, 0x1004]);
    }

    #[test]
    fn compare_branch_uses_its_ten_bit_offset_and_reports_terminal_record() {
        let bytes = stream(&[
            instruction(21, 0, 0x1000, 0x4884_1521, "JUMPCMP"),
            instruction(21, 0, 0x1004, 0x0700_0000, "MOV_XD_IMM"),
            instruction(21, 1, 0x2000, 0x4000_0001, "JUMP"),
        ]);
        let result = inspect_prof_stub_branch_edges(&bytes, Architecture::Dav2201, 4).unwrap();
        assert_eq!(result.compare_records, 1);
        assert_eq!(result.conditional_fallthrough_candidates, 1);
        assert_eq!(result.no_successor, 1);
        assert_eq!(
            result.issues[0].reason,
            ProfStubBranchEdgeIssueReason::NoSuccessor
        );
        assert_eq!(result.issues[0].possible_target_pcs, vec![0x2004]);
    }

    #[test]
    fn invalid_branch_word_is_reported_without_fabricating_an_edge() {
        let bytes = stream(&[
            instruction(21, 0, 0x1000, 0xffff_ffff, "JUMP"),
            instruction(21, 0, 0x1004, 0x0700_0000, "MOV_XD_IMM"),
        ]);
        let result = inspect_prof_stub_branch_edges(&bytes, Architecture::Dav3510, 4).unwrap();
        assert_eq!(result.branch_records, 1);
        assert_eq!(result.malformed_branch_records, 1);
        assert_eq!(result.unconditional_matches, 0);
        assert_eq!(
            result.issues[0].reason,
            ProfStubBranchEdgeIssueReason::UnsupportedWord
        );
    }

    #[test]
    fn malformed_text_and_mismatched_pc_keep_word_presence_distinct() {
        let bytes = stream(&[
            instruction_with_description(21, 0, 0x1000, "(PC: 0x1000) FLOWCTRL : JUMP"),
            instruction_with_description(
                21,
                0,
                0x2000,
                "(PC: 0x2004) FLOWCTRL : (Binary: 0x40000004) (ID: 000001) JUMP",
            ),
        ]);
        let result = inspect_prof_stub_branch_edges(&bytes, Architecture::Dav3510, 2).unwrap();
        assert_eq!(result.branch_records, 2);
        assert_eq!(result.malformed_branch_records, 2);
        assert_eq!(result.no_successor, 0);
        assert_eq!(
            result.issues[0].reason,
            ProfStubBranchEdgeIssueReason::InvalidDescription
        );
        assert_eq!(result.issues[0].word, None);
        assert_eq!(
            result.issues[1].reason,
            ProfStubBranchEdgeIssueReason::TextPcMismatch
        );
        assert_eq!(result.issues[1].word, Some(0x4000_0004));
    }

    #[test]
    fn terminal_marker_does_not_become_a_branch_successor() {
        let bytes = stream(&[
            instruction(21, 0, 0x1000, 0x4000_0004, "JUMP"),
            instruction(21, 0, 0x1111_1111, 0, "END_LABEL"),
            instruction(21, 0, 0x1010, 0x0700_0000, "MOV_XD_IMM"),
        ]);
        let result = inspect_prof_stub_branch_edges(&bytes, Architecture::Dav2201, 2).unwrap();
        assert_eq!(result.type21_instruction_events, 3);
        assert_eq!(result.no_successor, 1);
        assert_eq!(result.unconditional_matches, 0);
        assert_eq!(result.outside_candidate_edges, 0);
        assert_eq!(
            result.issues[0].reason,
            ProfStubBranchEdgeIssueReason::NoSuccessor
        );
        assert_eq!(result.issues[0].observed_successor_pc, None);
    }
}
