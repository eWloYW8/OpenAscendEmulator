use crate::architecture::Architecture;
use crate::flow::{
    ConditionalJump, JumpCompare, JumpCompareOffset, JumpCompareOperand, JumpOffsetSource,
    UnconditionalJump,
};
use serde::Serialize;
use std::io::{self, BufRead};

const MAX_ISSUES: usize = 16;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct JumpTraceIssue {
    pub line_number: u64,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct JumpTraceSummary {
    pub architecture: Architecture,
    pub total_lines: u64,
    pub observed_jumps: u64,
    pub observed_conditional_jumps: u64,
    pub observed_compare_jumps: u64,
    pub conditional_taken: u64,
    pub conditional_not_taken: u64,
    pub compare_taken: u64,
    pub compare_not_taken: u64,
    pub compare_signed: u64,
    pub compare_unsigned: u64,
    pub compare_immediate_operands: u64,
    pub compare_register_operands: u64,
    pub verified_immediate_targets: u64,
    pub verified_register_targets: u64,
    pub unverified_register_targets: u64,
    pub mismatches: u64,
    pub issues: Vec<JumpTraceIssue>,
}

pub fn verify_jump_trace<R: BufRead>(
    reader: R,
    architecture: Architecture,
) -> io::Result<JumpTraceSummary> {
    let mut summary = JumpTraceSummary {
        architecture,
        total_lines: 0,
        observed_jumps: 0,
        observed_conditional_jumps: 0,
        observed_compare_jumps: 0,
        conditional_taken: 0,
        conditional_not_taken: 0,
        compare_taken: 0,
        compare_not_taken: 0,
        compare_signed: 0,
        compare_unsigned: 0,
        compare_immediate_operands: 0,
        compare_register_operands: 0,
        verified_immediate_targets: 0,
        verified_register_targets: 0,
        unverified_register_targets: 0,
        mismatches: 0,
        issues: Vec::new(),
    };
    for line in reader.lines() {
        let line = line?;
        summary.total_lines += 1;
        let kind = if line.split_whitespace().any(|token| token == "JUMPCMP") {
            JumpKind::Compare
        } else if line.split_whitespace().any(|token| token == "JUMPC") {
            JumpKind::Conditional
        } else if line.split_whitespace().any(|token| token == "JUMP") {
            JumpKind::Unconditional
        } else {
            continue;
        };
        summary.observed_jumps += 1;
        let result = match kind {
            JumpKind::Compare => {
                summary.observed_compare_jumps += 1;
                check_compare_line(&line, architecture, &mut summary)
            }
            JumpKind::Conditional => {
                summary.observed_conditional_jumps += 1;
                check_jump_line(&line, architecture, true, &mut summary)
            }
            JumpKind::Unconditional => check_jump_line(&line, architecture, false, &mut summary),
        };
        if let Err(reason) = result {
            summary.mismatches += 1;
            if summary.issues.len() < MAX_ISSUES {
                summary.issues.push(JumpTraceIssue {
                    line_number: summary.total_lines,
                    reason,
                });
            }
        }
    }
    Ok(summary)
}

#[derive(Clone, Copy)]
enum JumpKind {
    Unconditional,
    Conditional,
    Compare,
}

fn check_compare_line(
    line: &str,
    architecture: Architecture,
    summary: &mut JumpTraceSummary,
) -> Result<(), String> {
    if !line.split_whitespace().any(|token| token == "FC") {
        return Err("JUMPCMP record lacks FC marker".to_owned());
    }
    let pc = parse_hex_before(line, "(PC: 0x", ')')?;
    let word = u32::try_from(parse_hex_before(line, "(Binary: 0x", ')')?)
        .map_err(|_| "JUMPCMP instruction exceeds 32 bits".to_owned())?;
    let target_pc = parse_hex_prefix(line, "Target PC:0x")?;
    let jump = JumpCompare::decode(architecture, word)
        .ok_or_else(|| format!("word {word:#010x} does not decode as JUMPCMP"))?;
    let logged_dtype = parse_field(line, "CMP_dtype:")?;
    let expected_dtype = match jump.dtype_field {
        0 => "S64",
        1 => "U64",
        _ => return Err(format!("unsupported JUMPCMP dtype {}", jump.dtype_field)),
    };
    if logged_dtype != expected_dtype {
        return Err(format!(
            "JUMPCMP dtype differs: decoded {expected_dtype}, log {logged_dtype}"
        ));
    }
    let (logged_first_index, first_value) = parse_register(line, "XN:X")?;
    if logged_first_index != jump.first_source_register {
        return Err(format!(
            "JUMPCMP first register differs: decoded X{}, log X{logged_first_index}",
            jump.first_source_register
        ));
    }
    let logged_condition = parse_numeric_prefix(line, "Condition_op:")?;
    if logged_condition != u64::from(jump.condition_field) {
        return Err(format!(
            "JUMPCMP condition differs: decoded {}, log {logged_condition}",
            jump.condition_field
        ));
    }
    let mut xregs = [0; 32];
    xregs[usize::from(jump.first_source_register)] = first_value;

    let fc_type = parse_field(line, "FC_type:")?;
    match jump.offset_source {
        JumpCompareOffset::Immediate { encoded_words } => {
            if fc_type != "IMM" {
                return Err(format!(
                    "JUMPCMP offset type differs: decoded IMM, log {fc_type}"
                ));
            }
            let logged_offset = parse_hex_immediate_before_cmp(line)?;
            if logged_offset != u64::from(encoded_words) {
                return Err(format!(
                    "JUMPCMP offset differs: decoded {encoded_words:#x}, log {logged_offset:#x}"
                ));
            }
        }
        JumpCompareOffset::Register { .. } => {
            summary.unverified_register_targets += 1;
            return Err("JUMPCMP register offset trace format is not verified".to_owned());
        }
    }

    let cmp_type = parse_field(line, "CMP_type:")?;
    let register_operand = match jump.second_operand {
        JumpCompareOperand::Immediate { encoded } => {
            if cmp_type != "IMM" {
                return Err(format!(
                    "JUMPCMP operand type differs: decoded IMM, log {cmp_type}"
                ));
            }
            let logged_operand = parse_numeric_prefix(line, "UIMM:")?;
            if logged_operand != u64::from(encoded) {
                return Err(format!(
                    "JUMPCMP immediate differs: decoded {encoded:#x}, log {logged_operand:#x}"
                ));
            }
            false
        }
        JumpCompareOperand::Register { index } => {
            if cmp_type != "REG" {
                return Err(format!(
                    "JUMPCMP operand type differs: decoded REG, log {cmp_type}"
                ));
            }
            let (logged_index, value) = parse_register(line, "XM:X")?;
            if logged_index != index {
                return Err(format!(
                    "JUMPCMP second register differs: decoded X{index}, log X{logged_index}"
                ));
            }
            if index == jump.first_source_register && value != first_value {
                return Err(format!(
                    "JUMPCMP aliased X{index} has conflicting logged values"
                ));
            }
            xregs[usize::from(index)] = value;
            true
        }
    };
    let result = jump
        .evaluate(pc, &xregs)
        .map_err(|error| error.to_string())?;
    if result.target_pc != target_pc {
        return Err(format!(
            "JUMPCMP target differs at PC {pc:#x}: calculated {:#x}, log {target_pc:#x}",
            result.target_pc
        ));
    }
    summary.verified_immediate_targets += 1;
    if result.branch_taken {
        summary.compare_taken += 1;
    } else {
        summary.compare_not_taken += 1;
    }
    if jump.dtype_field == 0 {
        summary.compare_signed += 1;
    } else {
        summary.compare_unsigned += 1;
    }
    if register_operand {
        summary.compare_register_operands += 1;
    } else {
        summary.compare_immediate_operands += 1;
    }
    Ok(())
}

fn parse_field<'a>(line: &'a str, prefix: &str) -> Result<&'a str, String> {
    let text = line
        .split_once(prefix)
        .ok_or_else(|| format!("missing {prefix}"))?
        .1;
    let value = text.split(',').next().unwrap_or_default().trim();
    if value.is_empty() {
        return Err(format!("missing value after {prefix}"));
    }
    Ok(value)
}

fn parse_hex_immediate_before_cmp(line: &str) -> Result<u64, String> {
    let before_cmp = line
        .split_once("CMP_type:")
        .ok_or_else(|| "missing CMP_type:".to_owned())?
        .0;
    let hex = before_cmp
        .rsplit_once("IMM:0x")
        .ok_or_else(|| "missing IMM:0x".to_owned())?
        .1;
    u64::from_str_radix(hex, 16).map_err(|_| "invalid JUMPCMP offset immediate".to_owned())
}

fn check_jump_line(
    line: &str,
    architecture: Architecture,
    conditional: bool,
    summary: &mut JumpTraceSummary,
) -> Result<(), String> {
    let name = if conditional { "JUMPC" } else { "JUMP" };
    if !line.split_whitespace().any(|token| token == "FC") {
        return Err(format!("{name} record lacks FC marker"));
    }
    let pc = parse_hex_before(line, "(PC: 0x", ')')?;
    let word = u32::try_from(parse_hex_before(line, "(Binary: 0x", ')')?)
        .map_err(|_| "JUMP instruction exceeds 32 bits".to_owned())?;
    let target_pc = parse_hex_prefix(line, "Target PC:0x")?;
    let offset_source = if conditional {
        ConditionalJump::decode(architecture, word)
            .ok_or_else(|| format!("word {word:#010x} does not decode as ordinary JUMPC"))?
            .offset_source
    } else {
        UnconditionalJump::decode(architecture, word)
            .ok_or_else(|| format!("word {word:#010x} does not decode as ordinary JUMP"))?
            .offset_source
    };
    let mut xregs = [0; 32];
    let register_mode = match offset_source {
        JumpOffsetSource::Immediate { .. } => false,
        JumpOffsetSource::Register { index } => {
            let (logged_index, value) = match parse_register(line, "XN:X") {
                Ok(source) => source,
                Err(error) => {
                    summary.unverified_register_targets += 1;
                    return Err(error);
                }
            };
            if logged_index != index {
                return Err(format!(
                    "JUMP source register differs: decoded X{index}, log X{logged_index}"
                ));
            }
            xregs[usize::from(index)] = value;
            true
        }
    };
    let computed_target = if conditional {
        let condition_flag = parse_numeric_prefix(line, "cond_flag:")?;
        let jump = ConditionalJump::decode(architecture, word)
            .expect("the conditional route was checked above");
        let outcome = jump.resolve(pc, &xregs, condition_flag);
        if outcome.branch_taken {
            summary.conditional_taken += 1;
        } else {
            summary.conditional_not_taken += 1;
        }
        outcome.target_pc
    } else {
        UnconditionalJump::decode(architecture, word)
            .expect("the unconditional route was checked above")
            .resolve(pc, &xregs)
            .target_pc
    };
    if computed_target != target_pc {
        return Err(format!(
            "JUMP target differs at PC {pc:#x}: calculated {:#x}, log {target_pc:#x}",
            computed_target
        ));
    }
    if register_mode {
        summary.verified_register_targets += 1;
    } else {
        summary.verified_immediate_targets += 1;
    }
    Ok(())
}

fn parse_numeric_prefix(line: &str, prefix: &str) -> Result<u64, String> {
    let text = line
        .split_once(prefix)
        .ok_or_else(|| format!("missing {prefix}"))?
        .1;
    let token = text
        .split(|character: char| character.is_ascii_whitespace() || character == ',')
        .next()
        .ok_or_else(|| format!("missing value after {prefix}"))?;
    if let Some(hex) = token.strip_prefix("0x") {
        u64::from_str_radix(hex, 16).map_err(|_| format!("invalid value after {prefix}"))
    } else {
        token
            .parse::<u64>()
            .map_err(|_| format!("invalid value after {prefix}"))
    }
}

fn parse_hex_before(line: &str, prefix: &str, terminator: char) -> Result<u64, String> {
    let text = line
        .split_once(prefix)
        .ok_or_else(|| format!("missing {prefix}"))?
        .1
        .split_once(terminator)
        .ok_or_else(|| format!("unterminated {prefix}"))?
        .0;
    u64::from_str_radix(text, 16).map_err(|_| format!("invalid hex after {prefix}"))
}

fn parse_hex_prefix(line: &str, prefix: &str) -> Result<u64, String> {
    let text = line
        .split_once(prefix)
        .ok_or_else(|| format!("missing {prefix}"))?
        .1;
    let hex: String = text.chars().take_while(char::is_ascii_hexdigit).collect();
    if hex.is_empty() {
        return Err(format!("missing hex after {prefix}"));
    }
    if text[hex.len()..]
        .chars()
        .next()
        .is_some_and(|character| !character.is_ascii_whitespace() && character != ',')
    {
        return Err(format!("invalid suffix after {prefix}"));
    }
    u64::from_str_radix(&hex, 16).map_err(|_| format!("invalid hex after {prefix}"))
}

fn parse_register(line: &str, prefix: &str) -> Result<(u8, u64), String> {
    let token = line
        .split(|character: char| character.is_ascii_whitespace() || character == ',')
        .find(|token| token.starts_with(prefix))
        .ok_or_else(|| format!("missing {prefix}"))?;
    let (index, value) = token[prefix.len()..]
        .split_once('=')
        .ok_or_else(|| format!("malformed {prefix}"))?;
    let index = index
        .parse::<u8>()
        .map_err(|_| format!("invalid register after {prefix}"))?;
    let value = value
        .strip_prefix("0x")
        .map(|hex| u64::from_str_radix(hex, 16))
        .unwrap_or_else(|| value.parse::<u64>())
        .map_err(|_| format!("invalid register value after {prefix}"))?;
    Ok((index, value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    const JUMPS: &str = "[info] [1] (PC: 0x112f5348) FC : (Binary: 0x400003e0) JUMP Target PC:0x112f62c8\n\
[info] [2] (PC: 0x126ecfb8) FC : (Binary: 0x4000fff9) JUMP Target PC:0x126ecf9c\n";

    #[test]
    fn compares_forward_and_backward_targets_on_both_architectures() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let summary = verify_jump_trace(Cursor::new(JUMPS), architecture).unwrap();
            assert_eq!(summary.observed_jumps, 2);
            assert_eq!(summary.verified_immediate_targets, 2);
            assert_eq!(summary.mismatches, 0);
            assert!(summary.issues.is_empty());

            let wrong = JUMPS.replace("Target PC:0x126ecf9c", "Target PC:0x126ecfa0");
            let summary = verify_jump_trace(Cursor::new(wrong), architecture).unwrap();
            assert_eq!(summary.verified_immediate_targets, 1);
            assert_eq!(summary.mismatches, 1);
            assert!(summary.issues[0].reason.contains("target differs"));

            let malformed = JUMPS.replace("Target PC:0x126ecf9c", "Target PC:0x126ecf9cz");
            let summary = verify_jump_trace(Cursor::new(malformed), architecture).unwrap();
            assert_eq!(summary.mismatches, 1);
            assert!(summary.issues[0].reason.contains("invalid suffix"));
        }
    }

    #[test]
    fn register_target_requires_source_value_and_matching_index() {
        let line = "[info] [1] (PC: 0x1000) FC : (Binary: 0x40023000) JUMP Target PC:0xfe4, XN:X3=0x3ffffffffff9";
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let summary = verify_jump_trace(Cursor::new(line), architecture).unwrap();
            assert_eq!(summary.verified_register_targets, 1);
            assert_eq!(summary.mismatches, 0, "{:?}", summary.issues);

            let missing = line.replace(", XN:X3=0x3ffffffffff9", "");
            let summary = verify_jump_trace(Cursor::new(missing), architecture).unwrap();
            assert_eq!(summary.mismatches, 1);
            assert_eq!(summary.verified_register_targets, 0);
            assert_eq!(summary.unverified_register_targets, 1);
        }
    }

    #[test]
    fn conditional_jump_compares_taken_and_fallthrough_records() {
        let lines = "[info] [1] (PC: 0x126ecfb4) FC : (Binary: 0x40200002) JUMPC Target PC:0x126ecfb8, cond_flag:0\n\
[info] [2] (PC: 0x126ecfb4) FC : (Binary: 0x40200002) JUMPC Target PC:0x126ecfbc, cond_flag:1\n";
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let summary = verify_jump_trace(Cursor::new(lines), architecture).unwrap();
            assert_eq!(summary.observed_jumps, 2);
            assert_eq!(summary.observed_conditional_jumps, 2);
            assert_eq!(summary.conditional_taken, 1);
            assert_eq!(summary.conditional_not_taken, 1);
            assert_eq!(summary.verified_immediate_targets, 2);
            assert_eq!(summary.mismatches, 0, "{:?}", summary.issues);

            let wrong = lines.replace("cond_flag:1", "cond_flag:0");
            let summary = verify_jump_trace(Cursor::new(wrong), architecture).unwrap();
            assert_eq!(summary.mismatches, 1);
            assert!(summary.issues[0].reason.contains("target differs"));
        }
    }

    #[test]
    fn compare_jump_checks_both_integer_types_and_targets() {
        let lines = "[info] [1] (PC: 0x112f50b8) FC : (Binary: 0x48841521) JUMPCMP Target PC:0x112f50bc, CMP_dtype:S64, XN:X8=0x1, Condition_op:1, FC_type:IMM, IMM:0xa9CMP_type:IMM, UIMM:0x1,\n\
[info] [2] (PC: 0x126ed14c) FC : (Binary: 0x4a0983a2) JUMPCMP Target PC:0x126ed1c0, CMP_dtype:U64, XN:X1=0x209, Condition_op:2, FC_type:IMM, IMM:0x1dCMP_type:REG, XM:X2=0x301,\n";
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let summary = verify_jump_trace(Cursor::new(lines), architecture).unwrap();
            assert_eq!(summary.observed_jumps, 2);
            assert_eq!(summary.observed_compare_jumps, 2);
            assert_eq!(summary.compare_taken, 1);
            assert_eq!(summary.compare_not_taken, 1);
            assert_eq!(summary.compare_signed, 1);
            assert_eq!(summary.compare_unsigned, 1);
            assert_eq!(summary.compare_immediate_operands, 1);
            assert_eq!(summary.compare_register_operands, 1);
            assert_eq!(summary.verified_immediate_targets, 2);
            assert_eq!(summary.mismatches, 0, "{:?}", summary.issues);

            let wrong_target = lines.replace("Target PC:0x126ed1c0", "Target PC:0x126ed1c4");
            let summary = verify_jump_trace(Cursor::new(wrong_target), architecture).unwrap();
            assert_eq!(summary.mismatches, 1);
            assert!(summary.issues[0].reason.contains("target differs"));

            let wrong_condition = lines.replace("Condition_op:2", "Condition_op:3");
            let summary = verify_jump_trace(Cursor::new(wrong_condition), architecture).unwrap();
            assert_eq!(summary.mismatches, 1);
            assert!(summary.issues[0].reason.contains("condition differs"));

            let wrong_operand = lines.replace("XM:X2=0x301", "XM:X3=0x301");
            let summary = verify_jump_trace(Cursor::new(wrong_operand), architecture).unwrap();
            assert_eq!(summary.mismatches, 1);
            assert!(summary.issues[0].reason.contains("second register differs"));
        }
    }

    #[test]
    fn compare_jump_rejects_malformed_concatenated_offset_and_dtype() {
        let line = "[info] (PC: 0x1000) FC : (Binary: 0x48841521) JUMPCMP Target PC:0x1004, CMP_dtype:S64, XN:X8=1, Condition_op:1, FC_type:IMM, IMM:0xa9CMP_type:IMM, UIMM:1,";
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let malformed = line.replace("IMM:0xa9CMP_type:", "IMM:0xa9zCMP_type:");
            let summary = verify_jump_trace(Cursor::new(malformed), architecture).unwrap();
            assert_eq!(summary.mismatches, 1);
            assert!(summary.issues[0].reason.contains("invalid JUMPCMP offset"));

            let wrong_dtype = line.replace("CMP_dtype:S64", "CMP_dtype:U64");
            let summary = verify_jump_trace(Cursor::new(wrong_dtype), architecture).unwrap();
            assert_eq!(summary.mismatches, 1);
            assert!(summary.issues[0].reason.contains("dtype differs"));

            let wrong_offset = line.replace("IMM:0xa9CMP_type:", "IMM:0xaaCMP_type:");
            let summary = verify_jump_trace(Cursor::new(wrong_offset), architecture).unwrap();
            assert_eq!(summary.mismatches, 1);
            assert!(summary.issues[0].reason.contains("offset differs"));

            let alias = "[info] (PC: 0x1000) FC : (Binary: 0x4a0983a1) JUMPCMP Target PC:0x1004, CMP_dtype:U64, XN:X1=0x209, Condition_op:2, FC_type:IMM, IMM:0x1dCMP_type:REG, XM:X1=0x301,";
            let summary = verify_jump_trace(Cursor::new(alias), architecture).unwrap();
            assert_eq!(summary.mismatches, 1);
            assert!(
                summary.issues[0]
                    .reason
                    .contains("conflicting logged values")
            );
        }
    }
}
