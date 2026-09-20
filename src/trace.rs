use crate::architecture::Architecture;
use crate::isa::{
    AicDecoderHint, ScalarKey0Operation, ScalarKey7Operation, ScalarKey8Operation, ZeroExtendWidth,
};
use crate::scalar::evaluate_scalar_integer_immediate;
use serde::Serialize;
use std::io::{self, BufRead};

const MAX_MISMATCH_EXAMPLES: usize = 16;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ScalarTraceSummary {
    pub architecture: Architecture,
    pub total_lines: u64,
    pub checked_s64: u64,
    pub add_immediate: u64,
    pub multiply_immediate: u64,
    pub subtract_immediate: u64,
    pub add_register: u64,
    pub multiply_register: u64,
    pub multiply_add_fields: u64,
    pub multiply_add_value_checked: u64,
    pub multiply_add_unknown_prior: u64,
    pub and_register: u64,
    pub or_register: u64,
    pub shift_left_fields: u64,
    pub shift_left_value_checked: u64,
    pub shift_left_unknown_prior: u64,
    pub checked_move_immediate: u64,
    pub checked_move_keep_lane: u64,
    pub checked_register_move: u64,
    pub zero_extend_u8: u64,
    pub zero_extend_u16: u64,
    pub zero_extend_u32: u64,
    pub skipped_other: u64,
    pub skipped_unsupported_dtype: u64,
    pub mismatches: u64,
    pub mismatch_examples: Vec<ScalarTraceMismatch>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ScalarTraceMismatch {
    pub line_number: u64,
    pub reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ScalarTraceRecord {
    pc: u64,
    word: u32,
    operation: ScalarKey8Operation,
    destination_register: u8,
    destination_value: u64,
    source_register: u8,
    source_value: u64,
    encoded_immediate: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ScalarMoveRecord {
    pc: u64,
    word: u32,
    operation: ScalarKey7Operation,
    destination_register: u8,
    destination_value: u64,
    encoded_immediate: u16,
    halfword_lane: Option<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ScalarZeroExtendRecord {
    pc: u64,
    word: u32,
    width: ZeroExtendWidth,
    destination_register: u8,
    destination_value: u64,
    source_register: u8,
    source_value: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ScalarRegisterMoveRecord {
    pc: u64,
    word: u32,
    destination_register: u8,
    destination_value: u64,
    source_register: u8,
    source_value: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ScalarRegisterBinaryRecord {
    pc: u64,
    word: u32,
    operation: ScalarKey0Operation,
    destination_register: u8,
    destination_value: u64,
    first_source_register: u8,
    first_source_value: u64,
    second_source_register: u8,
    second_source_value: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ScalarShiftLeftRecord {
    pc: u64,
    word: u32,
    destination_register: u8,
    destination_value: u64,
    count_register: Option<(u8, u64)>,
    encoded_immediate: u8,
}

enum ParsedLine {
    Other,
    UnsupportedDtype,
    Record(ScalarTraceRecord),
    MoveRecord(ScalarMoveRecord),
    ZeroExtendRecord(ScalarZeroExtendRecord),
    RegisterMoveRecord(ScalarRegisterMoveRecord),
    RegisterBinaryRecord(ScalarRegisterBinaryRecord),
    ShiftLeftRecord(ScalarShiftLeftRecord),
}

enum TraceOperation {
    Arithmetic(ScalarKey8Operation),
    Move(ScalarKey7Operation),
    ZeroExtend,
    RegisterMove,
    RegisterBinary(ScalarKey0Operation),
    ShiftLeft,
}

pub fn verify_scalar_trace<R: BufRead>(
    reader: R,
    architecture: Architecture,
) -> io::Result<ScalarTraceSummary> {
    let mut summary = ScalarTraceSummary {
        architecture,
        total_lines: 0,
        checked_s64: 0,
        add_immediate: 0,
        multiply_immediate: 0,
        subtract_immediate: 0,
        add_register: 0,
        multiply_register: 0,
        multiply_add_fields: 0,
        multiply_add_value_checked: 0,
        multiply_add_unknown_prior: 0,
        and_register: 0,
        or_register: 0,
        shift_left_fields: 0,
        shift_left_value_checked: 0,
        shift_left_unknown_prior: 0,
        checked_move_immediate: 0,
        checked_move_keep_lane: 0,
        checked_register_move: 0,
        zero_extend_u8: 0,
        zero_extend_u16: 0,
        zero_extend_u32: 0,
        skipped_other: 0,
        skipped_unsupported_dtype: 0,
        mismatches: 0,
        mismatch_examples: Vec::new(),
    };
    let mut known_xregs = [None; 32];
    for line in reader.lines() {
        let line = line?;
        summary.total_lines += 1;
        match parse_line(&line) {
            Ok(ParsedLine::Other) => summary.skipped_other += 1,
            Ok(ParsedLine::UnsupportedDtype) => summary.skipped_unsupported_dtype += 1,
            Ok(ParsedLine::Record(record)) => {
                summary.checked_s64 += 1;
                match record.operation {
                    ScalarKey8Operation::AddImmediate => summary.add_immediate += 1,
                    ScalarKey8Operation::MultiplyImmediate => summary.multiply_immediate += 1,
                    ScalarKey8Operation::SubtractImmediate => summary.subtract_immediate += 1,
                    ScalarKey8Operation::DcPreload => unreachable!("not parsed as arithmetic"),
                }
                if let Err(reason) = check_record(record, architecture) {
                    add_mismatch(&mut summary, reason);
                }
            }
            Ok(ParsedLine::MoveRecord(record)) => {
                match record.operation {
                    ScalarKey7Operation::MoveImmediate => summary.checked_move_immediate += 1,
                    ScalarKey7Operation::MoveKeep => summary.checked_move_keep_lane += 1,
                }
                if let Err(reason) = check_move_record(record, architecture) {
                    add_mismatch(&mut summary, reason);
                }
            }
            Ok(ParsedLine::ZeroExtendRecord(record)) => {
                match record.width {
                    ZeroExtendWidth::U8 => summary.zero_extend_u8 += 1,
                    ZeroExtendWidth::U16 => summary.zero_extend_u16 += 1,
                    ZeroExtendWidth::U32 => summary.zero_extend_u32 += 1,
                }
                if let Err(reason) = check_zero_extend_record(record, architecture) {
                    add_mismatch(&mut summary, reason);
                }
            }
            Ok(ParsedLine::RegisterMoveRecord(record)) => {
                summary.checked_register_move += 1;
                if let Err(reason) = check_register_move_record(record, architecture) {
                    add_mismatch(&mut summary, reason);
                }
            }
            Ok(ParsedLine::RegisterBinaryRecord(record)) => {
                let prior_destination =
                    if record.first_source_register == record.destination_register {
                        Some(record.first_source_value)
                    } else if record.second_source_register == record.destination_register {
                        Some(record.second_source_value)
                    } else {
                        known_xregs
                            .get(usize::from(record.destination_register))
                            .copied()
                            .flatten()
                    };
                match record.operation {
                    ScalarKey0Operation::Add => summary.add_register += 1,
                    ScalarKey0Operation::Multiply => summary.multiply_register += 1,
                    ScalarKey0Operation::MultiplyAdd => {
                        summary.multiply_add_fields += 1;
                        if prior_destination.is_some() {
                            summary.multiply_add_value_checked += 1;
                        } else {
                            summary.multiply_add_unknown_prior += 1;
                        }
                    }
                    ScalarKey0Operation::And => summary.and_register += 1,
                    ScalarKey0Operation::Or => summary.or_register += 1,
                }
                if let Err(reason) =
                    check_register_binary_record(record, architecture, prior_destination)
                {
                    add_mismatch(&mut summary, reason);
                }
            }
            Ok(ParsedLine::ShiftLeftRecord(record)) => {
                summary.shift_left_fields += 1;
                let prior_destination = known_xregs[usize::from(record.destination_register)];
                if prior_destination.is_some() {
                    summary.shift_left_value_checked += 1;
                } else {
                    summary.shift_left_unknown_prior += 1;
                }
                if let Err(reason) =
                    check_shift_left_record(record, architecture, prior_destination)
                {
                    add_mismatch(&mut summary, reason);
                }
            }
            Err(reason) => add_mismatch(&mut summary, reason),
        }
        let is_scalar = line
            .split_whitespace()
            .any(|token| token.trim_end_matches(':') == "SCALAR");
        let is_store = line.split_whitespace().any(|token| {
            token.starts_with("ST_") || token.starts_with("STP_") || token.starts_with("STI_")
        });
        if is_scalar
            && !is_store
            && let Ok((register, value)) = parse_register(&line, "XD:X")
            && let Some(slot) = known_xregs.get_mut(usize::from(register))
        {
            *slot = Some(value);
        }
    }
    Ok(summary)
}

fn add_mismatch(summary: &mut ScalarTraceSummary, reason: String) {
    summary.mismatches += 1;
    if summary.mismatch_examples.len() < MAX_MISMATCH_EXAMPLES {
        summary.mismatch_examples.push(ScalarTraceMismatch {
            line_number: summary.total_lines,
            reason,
        });
    }
}

fn parse_line(line: &str) -> Result<ParsedLine, String> {
    let operation = line.split_whitespace().find_map(|token| match token {
        "ADD_IMM" => Some(TraceOperation::Arithmetic(
            ScalarKey8Operation::AddImmediate,
        )),
        "MUL_IMM" => Some(TraceOperation::Arithmetic(
            ScalarKey8Operation::MultiplyImmediate,
        )),
        "SUB_IMM" => Some(TraceOperation::Arithmetic(
            ScalarKey8Operation::SubtractImmediate,
        )),
        "MOV_XD_IMM" => Some(TraceOperation::Move(ScalarKey7Operation::MoveImmediate)),
        "MOVK" => Some(TraceOperation::Move(ScalarKey7Operation::MoveKeep)),
        "ZEROEXT" => Some(TraceOperation::ZeroExtend),
        "MOV_XD_XN" => Some(TraceOperation::RegisterMove),
        "ADD" => Some(TraceOperation::RegisterBinary(ScalarKey0Operation::Add)),
        "MUL" => Some(TraceOperation::RegisterBinary(
            ScalarKey0Operation::Multiply,
        )),
        "MADD" => Some(TraceOperation::RegisterBinary(
            ScalarKey0Operation::MultiplyAdd,
        )),
        "AND" => Some(TraceOperation::RegisterBinary(ScalarKey0Operation::And)),
        "OR" => Some(TraceOperation::RegisterBinary(ScalarKey0Operation::Or)),
        "SHL" => Some(TraceOperation::ShiftLeft),
        _ => None,
    });
    let Some(operation) = operation else {
        return Ok(ParsedLine::Other);
    };
    if !line
        .split_whitespace()
        .any(|token| token.trim_end_matches(':') == "SCALAR")
    {
        return Err("supported scalar record lacks SCALAR marker".to_owned());
    }
    if matches!(operation, TraceOperation::ShiftLeft) {
        let dtype = line
            .split_whitespace()
            .find(|token| token.starts_with("dtype:"))
            .ok_or_else(|| "SHL record lacks dtype".to_owned())?;
        if dtype.trim_end_matches(',') != "dtype:B64" {
            return Ok(ParsedLine::UnsupportedDtype);
        }
        let pc = parse_hex_between(line, "(PC: 0x", ')')?;
        let word = u32::try_from(parse_hex_between(line, "(Binary: 0x", ')')?)
            .map_err(|_| "instruction word exceeds 32 bits".to_owned())?;
        let (destination_register, destination_value) = parse_register(line, "XD:X")?;
        let data_source = line
            .split_whitespace()
            .find(|token| token.starts_with("DATA_SRC:"))
            .ok_or_else(|| "SHL record lacks DATA_SRC".to_owned())?;
        let count_register = match data_source.trim_end_matches(',') {
            "DATA_SRC:IMM" => None,
            "DATA_SRC:XN" => Some(parse_register(line, "XN:X")?),
            _ => return Ok(ParsedLine::UnsupportedDtype),
        };
        let encoded_immediate = u8::try_from(parse_numeric_token(line, "IMM:")?)
            .map_err(|_| "SHL immediate exceeds 8 bits".to_owned())?;
        return Ok(ParsedLine::ShiftLeftRecord(ScalarShiftLeftRecord {
            pc,
            word,
            destination_register,
            destination_value,
            count_register,
            encoded_immediate,
        }));
    }
    if let TraceOperation::RegisterBinary(operation) = operation {
        let dtype = line
            .split_whitespace()
            .find(|token| token.starts_with("dtype:"))
            .ok_or_else(|| "register-binary record lacks dtype".to_owned())?;
        let expected_dtype = match operation {
            ScalarKey0Operation::Add
            | ScalarKey0Operation::Multiply
            | ScalarKey0Operation::MultiplyAdd => "dtype:S64",
            ScalarKey0Operation::And | ScalarKey0Operation::Or => "dtype:B64",
        };
        if dtype.trim_end_matches(',') != expected_dtype {
            return Ok(ParsedLine::UnsupportedDtype);
        }
        let pc = parse_hex_between(line, "(PC: 0x", ')')?;
        let word = u32::try_from(parse_hex_between(line, "(Binary: 0x", ')')?)
            .map_err(|_| "instruction word exceeds 32 bits".to_owned())?;
        let (destination_register, destination_value) = parse_register(line, "XD:X")?;
        let (first_source_register, first_source_value) = parse_register(line, "XN:X")?;
        let (second_source_register, second_source_value) = parse_register(line, "XM:X")?;
        return Ok(ParsedLine::RegisterBinaryRecord(
            ScalarRegisterBinaryRecord {
                pc,
                word,
                operation,
                destination_register,
                destination_value,
                first_source_register,
                first_source_value,
                second_source_register,
                second_source_value,
            },
        ));
    }
    if matches!(operation, TraceOperation::RegisterMove) {
        let dtype = line
            .split_whitespace()
            .find(|token| token.starts_with("dtype:"))
            .ok_or_else(|| "MOV_XD_XN record lacks dtype".to_owned())?;
        if dtype.trim_end_matches(',') != "dtype:S64" {
            return Ok(ParsedLine::UnsupportedDtype);
        }
        let pc = parse_hex_between(line, "(PC: 0x", ')')?;
        let word = u32::try_from(parse_hex_between(line, "(Binary: 0x", ')')?)
            .map_err(|_| "instruction word exceeds 32 bits".to_owned())?;
        let (destination_register, destination_value) = parse_register(line, "XD:X")?;
        let (source_register, source_value) = parse_register(line, "XN:X")?;
        return Ok(ParsedLine::RegisterMoveRecord(ScalarRegisterMoveRecord {
            pc,
            word,
            destination_register,
            destination_value,
            source_register,
            source_value,
        }));
    }
    if matches!(operation, TraceOperation::ZeroExtend) {
        let dtype = line
            .split_whitespace()
            .find(|token| token.starts_with("dtype:"))
            .ok_or_else(|| "ZEROEXT record lacks dtype".to_owned())?;
        let width = match dtype.trim_end_matches(',') {
            "dtype:U8" => ZeroExtendWidth::U8,
            "dtype:U16" => ZeroExtendWidth::U16,
            "dtype:U32" => ZeroExtendWidth::U32,
            _ => return Ok(ParsedLine::UnsupportedDtype),
        };
        let pc = parse_hex_between(line, "(PC: 0x", ')')?;
        let word = u32::try_from(parse_hex_between(line, "(Binary: 0x", ')')?)
            .map_err(|_| "instruction word exceeds 32 bits".to_owned())?;
        let (destination_register, destination_value) = parse_register(line, "XD:X")?;
        let (source_register, source_value) = parse_register(line, "XN:X")?;
        return Ok(ParsedLine::ZeroExtendRecord(ScalarZeroExtendRecord {
            pc,
            word,
            width,
            destination_register,
            destination_value,
            source_register,
            source_value,
        }));
    }
    if let TraceOperation::Move(operation) = operation {
        let pc = parse_hex_between(line, "(PC: 0x", ')')?;
        let word = u32::try_from(parse_hex_between(line, "(Binary: 0x", ')')?)
            .map_err(|_| "instruction word exceeds 32 bits".to_owned())?;
        let (destination_register, destination_value) = parse_register(line, "XD:X")?;
        let encoded_immediate = u16::try_from(parse_numeric_token(line, "IMM:")?)
            .map_err(|_| "immediate exceeds 16 bits".to_owned())?;
        let halfword_lane = if operation == ScalarKey7Operation::MoveKeep {
            Some(
                u8::try_from(parse_numeric_token(line, "UIMM:")?)
                    .map_err(|_| "halfword lane exceeds 8 bits".to_owned())?,
            )
        } else {
            None
        };
        return Ok(ParsedLine::MoveRecord(ScalarMoveRecord {
            pc,
            word,
            operation,
            destination_register,
            destination_value,
            encoded_immediate,
            halfword_lane,
        }));
    }
    let TraceOperation::Arithmetic(operation) = operation else {
        unreachable!("non-immediate forms returned above")
    };
    let dtype = line
        .split_whitespace()
        .find(|token| token.starts_with("dtype:"))
        .ok_or_else(|| "arithmetic record lacks dtype".to_owned())?;
    if dtype.trim_end_matches(',') != "dtype:S64" {
        return Ok(ParsedLine::UnsupportedDtype);
    }

    let pc = parse_hex_between(line, "(PC: 0x", ')')?;
    let word = u32::try_from(parse_hex_between(line, "(Binary: 0x", ')')?)
        .map_err(|_| "instruction word exceeds 32 bits".to_owned())?;
    let (destination_register, destination_value) = parse_register(line, "XD:X")?;
    let (source_register, source_value) = parse_register(line, "XN:X")?;
    let encoded_immediate = u16::try_from(parse_numeric_token(line, "IMM:")?)
        .map_err(|_| "immediate exceeds 16 bits".to_owned())?;
    Ok(ParsedLine::Record(ScalarTraceRecord {
        pc,
        word,
        operation,
        destination_register,
        destination_value,
        source_register,
        source_value,
        encoded_immediate,
    }))
}

fn check_move_record(record: ScalarMoveRecord, architecture: Architecture) -> Result<(), String> {
    let Some(AicDecoderHint::ScalarKey7 {
        operation,
        destination_register,
        encoded_immediate,
        halfword_lane,
        ..
    }) = AicDecoderHint::from_word(architecture, record.word)
    else {
        return Err(format!(
            "word {:#010x} does not select scalar key 7",
            record.word
        ));
    };
    if operation != record.operation
        || destination_register != record.destination_register
        || encoded_immediate != record.encoded_immediate
        || halfword_lane != record.halfword_lane
    {
        return Err(format!(
            "move fields differ at PC {:#x}: decoded {operation:?} XD={destination_register} IMM={encoded_immediate:#x} lane={halfword_lane:?}, log {:?} XD={} IMM={:#x} lane={:?}",
            record.pc,
            record.operation,
            record.destination_register,
            record.encoded_immediate,
            record.halfword_lane
        ));
    }
    let observed = match operation {
        ScalarKey7Operation::MoveImmediate => record.destination_value,
        ScalarKey7Operation::MoveKeep => {
            let shift = u32::from(halfword_lane.expect("MOVK decoder lane")) * 16;
            (record.destination_value >> shift) & 0xffff
        }
    };
    if observed != u64::from(encoded_immediate) {
        return Err(format!(
            "move value differs at PC {:#x}: observed lane/full value {observed:#x}, immediate {encoded_immediate:#x}",
            record.pc
        ));
    }
    Ok(())
}

fn check_zero_extend_record(
    record: ScalarZeroExtendRecord,
    architecture: Architecture,
) -> Result<(), String> {
    let Some(AicDecoderHint::ScalarKey2ZeroExtend {
        width,
        destination_register,
        source_register,
        ..
    }) = AicDecoderHint::from_word(architecture, record.word)
    else {
        return Err(format!(
            "word {:#010x} does not select valid scalar ZEROEXT",
            record.word
        ));
    };
    if width != record.width
        || destination_register != record.destination_register
        || source_register != record.source_register
    {
        return Err(format!(
            "ZEROEXT fields differ at PC {:#x}: decoded {width:?} XD={destination_register} XN={source_register}, log {:?} XD={} XN={}",
            record.pc, record.width, record.destination_register, record.source_register
        ));
    }
    let expected = record.source_value & width.mask();
    if expected != record.destination_value {
        return Err(format!(
            "ZEROEXT value differs at PC {:#x}: calculated {expected:#x}, log {:#x}",
            record.pc, record.destination_value
        ));
    }
    Ok(())
}

fn check_register_move_record(
    record: ScalarRegisterMoveRecord,
    architecture: Architecture,
) -> Result<(), String> {
    let Some(AicDecoderHint::ScalarKey2MoveRegister {
        dtype_field,
        destination_register,
        source_register,
        ..
    }) = AicDecoderHint::from_word(architecture, record.word)
    else {
        return Err(format!(
            "word {:#010x} does not select scalar MOV_XD_XN",
            record.word
        ));
    };
    if dtype_field != 0
        || destination_register != record.destination_register
        || source_register != record.source_register
    {
        return Err(format!(
            "MOV_XD_XN fields differ at PC {:#x}: decoded dtype={dtype_field} XD={destination_register} XN={source_register}, log S64 XD={} XN={}",
            record.pc, record.destination_register, record.source_register
        ));
    }
    if record.source_value != record.destination_value {
        return Err(format!(
            "MOV_XD_XN value differs at PC {:#x}: source {:#x}, destination {:#x}",
            record.pc, record.source_value, record.destination_value
        ));
    }
    Ok(())
}

fn check_register_binary_record(
    record: ScalarRegisterBinaryRecord,
    architecture: Architecture,
    prior_destination: Option<u64>,
) -> Result<(), String> {
    let Some(AicDecoderHint::ScalarKey0 {
        operation,
        dtype_field,
        destination_register,
        first_source_register,
        second_source_register,
        ..
    }) = AicDecoderHint::from_word(architecture, record.word)
    else {
        return Err(format!(
            "word {:#010x} does not select a supported scalar key 0 operation",
            record.word
        ));
    };
    let expected_dtype_field = match record.operation {
        ScalarKey0Operation::Add
        | ScalarKey0Operation::Multiply
        | ScalarKey0Operation::MultiplyAdd => 0,
        ScalarKey0Operation::And | ScalarKey0Operation::Or => 3,
    };
    if operation != record.operation
        || dtype_field != expected_dtype_field
        || destination_register != record.destination_register
        || first_source_register != record.first_source_register
        || second_source_register != record.second_source_register
    {
        return Err(format!(
            "register-binary fields differ at PC {:#x}: decoded {operation:?} dtype={dtype_field} XD={destination_register} XN={first_source_register} XM={second_source_register}, log {:?} XD={} XN={} XM={}",
            record.pc,
            record.operation,
            record.destination_register,
            record.first_source_register,
            record.second_source_register
        ));
    }
    let expected = match operation {
        ScalarKey0Operation::Add => Some(
            record
                .first_source_value
                .wrapping_add(record.second_source_value),
        ),
        ScalarKey0Operation::Multiply => Some(
            record
                .first_source_value
                .wrapping_mul(record.second_source_value),
        ),
        ScalarKey0Operation::MultiplyAdd => prior_destination.map(|prior| {
            prior.wrapping_add(
                record
                    .first_source_value
                    .wrapping_mul(record.second_source_value),
            )
        }),
        ScalarKey0Operation::And => Some(record.first_source_value & record.second_source_value),
        ScalarKey0Operation::Or => Some(record.first_source_value | record.second_source_value),
    };
    if let Some(expected) = expected
        && expected != record.destination_value
    {
        return Err(format!(
            "register-binary value differs at PC {:#x}: calculated {expected:#x}, log {:#x}",
            record.pc, record.destination_value
        ));
    }
    Ok(())
}

fn check_shift_left_record(
    record: ScalarShiftLeftRecord,
    architecture: Architecture,
    prior_destination: Option<u64>,
) -> Result<(), String> {
    let Some(AicDecoderHint::ScalarKey2ShiftLeft {
        dtype_field,
        destination_register,
        count_register,
        encoded_immediate,
        ..
    }) = AicDecoderHint::from_word(architecture, record.word)
    else {
        return Err(format!(
            "word {:#010x} does not select scalar SHL",
            record.word
        ));
    };
    if dtype_field != 3
        || destination_register != record.destination_register
        || count_register != record.count_register.map(|(register, _)| register)
        || encoded_immediate != record.encoded_immediate
    {
        return Err(format!(
            "SHL fields differ at PC {:#x}: decoded dtype={dtype_field} XD={destination_register} XN={count_register:?} IMM={encoded_immediate:#x}, log XD={} XN={:?} IMM={:#x}",
            record.pc,
            record.destination_register,
            record.count_register.map(|(register, _)| register),
            record.encoded_immediate
        ));
    }
    if let Some(prior) = prior_destination {
        let shift = record
            .count_register
            .map_or(u32::from(encoded_immediate), |(_, value)| {
                (value & 0x3f) as u32
            });
        let expected = prior << shift;
        if expected != record.destination_value {
            return Err(format!(
                "SHL value differs at PC {:#x}: prior XD={prior:#x}, count={shift}, calculated {expected:#x}, log {:#x}",
                record.pc, record.destination_value
            ));
        }
    }
    Ok(())
}

fn parse_hex_between(line: &str, prefix: &str, terminator: char) -> Result<u64, String> {
    let rest = line
        .split_once(prefix)
        .ok_or_else(|| format!("missing {prefix}"))?
        .1;
    let hex = rest
        .split_once(terminator)
        .ok_or_else(|| format!("unterminated {prefix}"))?
        .0;
    u64::from_str_radix(hex, 16).map_err(|_| format!("invalid hex after {prefix}"))
}

fn parse_numeric_token(line: &str, prefix: &str) -> Result<u64, String> {
    let token = line
        .split(|character: char| character.is_ascii_whitespace() || character == ',')
        .find(|token| token.starts_with(prefix))
        .ok_or_else(|| format!("missing {prefix}"))?;
    parse_numeric_value(&token[prefix.len()..])
        .map_err(|_| format!("invalid number after {prefix}"))
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
        .map_err(|_| format!("invalid register index after {prefix}"))?;
    let value =
        parse_numeric_value(value).map_err(|_| format!("invalid register value after {prefix}"))?;
    Ok((index, value))
}

fn parse_numeric_value(value: &str) -> Result<u64, std::num::ParseIntError> {
    if let Some(hex) = value.strip_prefix("0x") {
        u64::from_str_radix(hex, 16)
    } else {
        value.parse::<u64>()
    }
}

fn check_record(record: ScalarTraceRecord, architecture: Architecture) -> Result<(), String> {
    let Some(
        hint @ AicDecoderHint::ScalarKey8 {
            operation,
            destination_register,
            source_register,
            encoded_immediate,
            ..
        },
    ) = AicDecoderHint::from_word(architecture, record.word)
    else {
        return Err(format!(
            "word {:#010x} does not select scalar key 8",
            record.word
        ));
    };
    if operation != record.operation {
        return Err(format!(
            "operation differs: decoded {operation:?}, log {:?}",
            record.operation
        ));
    }
    if destination_register != Some(record.destination_register)
        || source_register != record.source_register
        || encoded_immediate != record.encoded_immediate
    {
        return Err(format!(
            "encoded fields differ: decoded XD={destination_register:?} XN={source_register} IMM={encoded_immediate:#x}, log XD={} XN={} IMM={:#x}",
            record.destination_register, record.source_register, record.encoded_immediate
        ));
    }
    let evaluated = evaluate_scalar_integer_immediate(hint, record.source_value, record.pc, 0)
        .map_err(|error| error.to_string())?;
    if evaluated.value != record.destination_value {
        return Err(format!(
            "value differs at PC {:#x}: calculated {:#x}, log {:#x}",
            record.pc, evaluated.value, record.destination_value
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    const GREATER: &str = "[info] [00035037] (PC: 0x112f5034) SCALAR   : (Binary: 0x083dd0a8) ADD_IMM  dtype:S64, XD:X30=0x1c8028, XN:X29=0x1c7f80, IMM:0xa8, \n\
[info] [00035038] (PC: 0x112f503c) SCALAR   : (Binary: 0x0893e790) SUB_IMM  dtype:S64, XD:X9=0x1c7898, XN:X30=0x1c8028, IMM:0x790, \n\
[info] [00035041] (PC: 0x112f504c) SCALAR   : (Binary: 0x080c9000) ADD_IMM  dtype:S64, XD:X6=0x1c0000, XN:X9=0x1c0000, IMM:0, \n\
[info] [00036016] (PC: 0x112f51e4) SCALAR   : (Binary: 0x084c6411) MUL_IMM  dtype:S64, XD:X6=0x1044, XN:X6=0x4, IMM:0x411, \n\
[info] [00037661] (PC: 0x112f652c) SCALAR   : (Binary: 0x08021001) ADD_IMM  dtype:S64, XD:X1=0x1, XN:X1=0, IMM:0x1, \n";

    const TRANSPOSE: &str = "[info] [00001255] (PC: 0x126ecb2c) SCALAR   : (Binary: 0x08bdd308) SUB_IMM  dtype:S64, XD:X30=0x1c7c98, XN:X29=0x1c7fa0, IMM:0x308, \n\
[info] [00004733] (PC: 0x126ed2ac) SCALAR   : (Binary: 0x0849a0c0) MUL_IMM  dtype:S64, XD:X4=0x18c0, XN:X26=0x21, IMM:0xc0, \n";

    const MOVES: &str = "[info] [00035029] (PC: 0x112f5000) SCALAR   : (Binary: 0x073a7f80) MOV_XD_IMM  XD:X29=0x7f80, IMM:0x7f80, \n\
[info] [00035030] (PC: 0x112f5004) SCALAR   : (Binary: 0x077b0010) MOVK  XD:X29=0x107f80, IMM:0x10, UIMM:0x1, \n\
[info] [00035731] (PC: 0x112f513c) SCALAR   : (Binary: 0x078dffff) MOVK  XD:X6=0xffff0000a000, IMM:0xffff, UIMM:0x2, \n";

    const ZERO_EXTENDS: &str = "[info] [00003480] (PC: 0x126ecf84) SCALAR   : (Binary: 0x02883a00) ZEROEXT  dtype:U32, XD:X4=0x3, XN:X3=0x3, \n\
[info] [00004388] (PC: 0x126ed220) SCALAR   : (Binary: 0x02741a00) ZEROEXT  dtype:U16, XD:X26=0x218, XN:X1=0x218, \n\
[info] [00004389] (PC: 0x126ed224) SCALAR   : (Binary: 0x02083a00) ZEROEXT  dtype:U8, XD:X4=0x34, XN:X3=0x1234, \n";

    const REGISTER_MOVES: &str = "[info] [00003463] (PC: 0x126ecf6c) SCALAR   : (Binary: 0x0209e800) MOV_XD_XN  dtype:S64, XD:X4=0x1c7c98, XN:X30=0x1c7c98, \n\
[info] [00004801] (PC: 0x126ed418) SCALAR   : (Binary: 0x02086800) MOV_XD_XN  dtype:S64, XD:X4=0, XN:X6=0, \n";

    const REGISTER_BINARY: &str = "[info] [00001249] (PC: 0x126ecb04) SCALAR   : (Binary: 0x003bd781) ADD  dtype:S64, XD:X29=0x107fa0, XN:X29=0x107fa0, XM:X15=0, \n\
[info] [00003724] (PC: 0x126ed004) SCALAR   : (Binary: 0x000e8383) MUL  dtype:S64, XD:X7=0x209, XN:X8=0x209, XM:X7=0x1, \n\
[info] [00001251] (PC: 0x126ecb14) SCALAR   : (Binary: 0x00def80a) AND  dtype:B64, XD:X15=0x18, XN:X15=0x18, XM:X16=0x7fff, \n\
[info] [00004731] (PC: 0x126ed29c) SCALAR   : (Binary: 0x00c0100b) OR  dtype:B64, XD:X0=0xc000080808010101, XN:X1=0xc000000000000000, XM:X0=0x80808010101, \n";

    const SHIFTS: &str = "[info] [00000100] (PC: 0x100) SCALAR   : (Binary: 0x02ca0202) SHL  dtype:B64, XD:X5=0x4, DATA_SRC:IMM, IMM:0x2, \n\
[info] [00000101] (PC: 0x104) SCALAR   : (Binary: 0x07080038) MOV_XD_IMM  XD:X4=0x38, IMM:0x38, \n\
[info] [00000102] (PC: 0x108) SCALAR   : (Binary: 0x02c8020f) SHL  dtype:B64, XD:X4=0x1c0000, DATA_SRC:IMM, IMM:0xf, \n\
[info] [00000103] (PC: 0x10c) SCALAR   : (Binary: 0x0714ffff) MOV_XD_IMM  XD:X10=0xffff, IMM:0xffff, \n\
[info] [00000104] (PC: 0x110) SCALAR   : (Binary: 0x072c0020) MOV_XD_IMM  XD:X22=0x20, IMM:0x20, \n\
[info] [00000105] (PC: 0x114) SCALAR   : (Binary: 0x02d56240) SHL  dtype:B64, XD:X10=0xffff00000000, XN:X22=0x20, DATA_SRC:XN, IMM:0, \n";

    #[test]
    fn matches_observed_greater_and_transpose_scalar_values() {
        for log in [GREATER, TRANSPOSE] {
            let summary = verify_scalar_trace(Cursor::new(log), Architecture::Dav2201).unwrap();
            assert_eq!(summary.mismatches, 0, "{:?}", summary.mismatch_examples);
            assert_eq!(summary.checked_s64, log.lines().count() as u64);
            assert_eq!(summary.skipped_unsupported_dtype, 0);
        }
    }

    #[test]
    fn rejects_wrong_result_and_reports_unsupported_dtype() {
        let wrong = GREATER.replace("XD:X30=0x1c8028", "XD:X30=0x1c8029");
        let summary = verify_scalar_trace(Cursor::new(wrong), Architecture::Dav2201).unwrap();
        assert_eq!(summary.mismatches, 1);
        assert_eq!(summary.mismatch_examples[0].line_number, 1);

        let fp = GREATER.replace("dtype:S64,", "dtype:F32,");
        let summary = verify_scalar_trace(Cursor::new(fp), Architecture::Dav2201).unwrap();
        assert_eq!(summary.checked_s64, 0);
        assert_eq!(
            summary.skipped_unsupported_dtype,
            GREATER.lines().count() as u64
        );
    }

    #[test]
    fn rejects_encoded_register_mismatch_and_malformed_line() {
        let wrong = GREATER.replace("XN:X29=0x1c7f80", "XN:X28=0x1c7f80");
        let summary = verify_scalar_trace(Cursor::new(wrong), Architecture::Dav2201).unwrap();
        assert_eq!(summary.mismatches, 1);
        assert!(
            summary.mismatch_examples[0]
                .reason
                .contains("encoded fields differ")
        );

        let malformed = GREATER.replace("IMM:0xa8", "IMM:what");
        let summary = verify_scalar_trace(Cursor::new(malformed), Architecture::Dav2201).unwrap();
        assert_eq!(summary.mismatches, 1);
        assert!(
            summary.mismatch_examples[0]
                .reason
                .contains("invalid number after IMM")
        );

        let wrong_class = GREATER.replace("SCALAR   :", "VECTOR   :");
        let summary = verify_scalar_trace(Cursor::new(wrong_class), Architecture::Dav2201).unwrap();
        assert_eq!(summary.mismatches, GREATER.lines().count() as u64);
    }

    #[test]
    fn checks_observed_moves_and_rejects_bad_halfword() {
        let summary = verify_scalar_trace(Cursor::new(MOVES), Architecture::Dav2201).unwrap();
        assert_eq!(summary.checked_move_immediate, 1);
        assert_eq!(summary.checked_move_keep_lane, 2);
        assert_eq!(summary.mismatches, 0);

        let wrong = MOVES.replace("XD:X29=0x107f80", "XD:X29=0x117f80");
        let summary = verify_scalar_trace(Cursor::new(wrong), Architecture::Dav2201).unwrap();
        assert_eq!(summary.mismatches, 1);
        assert!(
            summary.mismatch_examples[0]
                .reason
                .contains("move value differs")
        );

        let wrong_lane = MOVES.replace("UIMM:0x1", "UIMM:0x2");
        let summary = verify_scalar_trace(Cursor::new(wrong_lane), Architecture::Dav2201).unwrap();
        assert_eq!(summary.mismatches, 1);
        assert!(
            summary.mismatch_examples[0]
                .reason
                .contains("move fields differ")
        );
    }

    #[test]
    fn checks_zero_extend_fields_and_masked_values() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let summary = verify_scalar_trace(Cursor::new(ZERO_EXTENDS), architecture).unwrap();
            assert_eq!(summary.zero_extend_u8, 1);
            assert_eq!(summary.zero_extend_u16, 1);
            assert_eq!(summary.zero_extend_u32, 1);
            assert_eq!(summary.mismatches, 0, "{:?}", summary.mismatch_examples);

            let wrong = ZERO_EXTENDS.replace("XD:X4=0x34", "XD:X4=0x35");
            let summary = verify_scalar_trace(Cursor::new(wrong), architecture).unwrap();
            assert_eq!(summary.mismatches, 1);
            assert!(
                summary.mismatch_examples[0]
                    .reason
                    .contains("ZEROEXT value differs")
            );

            let wrong = ZERO_EXTENDS.replace("dtype:U16", "dtype:U8");
            let summary = verify_scalar_trace(Cursor::new(wrong), architecture).unwrap();
            assert_eq!(summary.mismatches, 1);
            assert!(
                summary.mismatch_examples[0]
                    .reason
                    .contains("ZEROEXT fields differ")
            );
        }
    }

    #[test]
    fn checks_register_move_fields_and_values() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let summary = verify_scalar_trace(Cursor::new(REGISTER_MOVES), architecture).unwrap();
            assert_eq!(summary.checked_register_move, 2);
            assert_eq!(summary.mismatches, 0, "{:?}", summary.mismatch_examples);

            let wrong = REGISTER_MOVES.replace("XD:X4=0x1c7c98", "XD:X4=0x1c7c99");
            let summary = verify_scalar_trace(Cursor::new(wrong), architecture).unwrap();
            assert_eq!(summary.mismatches, 1);
            assert!(
                summary.mismatch_examples[0]
                    .reason
                    .contains("MOV_XD_XN value differs")
            );

            let wrong = REGISTER_MOVES.replace("dtype:S64", "dtype:F32");
            let summary = verify_scalar_trace(Cursor::new(wrong), architecture).unwrap();
            assert_eq!(summary.checked_register_move, 0);
            assert_eq!(summary.skipped_unsupported_dtype, 2);
        }
    }

    #[test]
    fn checks_observed_register_binary_fields_and_values() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let summary = verify_scalar_trace(Cursor::new(REGISTER_BINARY), architecture).unwrap();
            assert_eq!(summary.add_register, 1);
            assert_eq!(summary.multiply_register, 1);
            assert_eq!(summary.and_register, 1);
            assert_eq!(summary.or_register, 1);
            assert_eq!(summary.mismatches, 0, "{:?}", summary.mismatch_examples);

            let wrong = REGISTER_BINARY.replace("XM:X15=0", "XM:X14=0");
            let summary = verify_scalar_trace(Cursor::new(wrong), architecture).unwrap();
            assert_eq!(summary.mismatches, 1);
            assert!(
                summary.mismatch_examples[0]
                    .reason
                    .contains("fields differ")
            );

            let wrong = REGISTER_BINARY.replace("XD:X15=0x18", "XD:X15=0x19");
            let summary = verify_scalar_trace(Cursor::new(wrong), architecture).unwrap();
            assert_eq!(summary.mismatches, 1);
            assert!(
                summary.mismatch_examples[0]
                    .reason
                    .contains("value differs")
            );
        }
    }

    #[test]
    fn multiply_add_checks_prior_value_when_known_and_reports_unknown_prior() {
        let prefix = "[info] [1] (PC: 0x100) SCALAR : (Binary: 0x073a7f80) MOV_XD_IMM XD:X29=0x7f80, IMM:0x7f80, \n\
[info] [2] (PC: 0x104) SCALAR : (Binary: 0x077b0010) MOVK XD:X29=0x107f80, IMM:0x10, UIMM:0x1, \n";
        let madd = "[info] [3] (PC: 0x108) SCALAR : (Binary: 0x003af884) MADD dtype:S64, XD:X29=0x1c7f80, XN:X15=0x18, XM:X17=0x8000, \n";
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let summary =
                verify_scalar_trace(Cursor::new(format!("{prefix}{madd}")), architecture).unwrap();
            assert_eq!(summary.multiply_add_fields, 1);
            assert_eq!(summary.multiply_add_value_checked, 1);
            assert_eq!(summary.multiply_add_unknown_prior, 0);
            assert_eq!(summary.mismatches, 0, "{:?}", summary.mismatch_examples);

            let unknown = verify_scalar_trace(Cursor::new(madd), architecture).unwrap();
            assert_eq!(unknown.multiply_add_fields, 1);
            assert_eq!(unknown.multiply_add_value_checked, 0);
            assert_eq!(unknown.multiply_add_unknown_prior, 1);
            assert_eq!(unknown.mismatches, 0);

            let wrong = madd.replace("XD:X29=0x1c7f80", "XD:X29=0x1c7f81");
            let summary =
                verify_scalar_trace(Cursor::new(format!("{prefix}{wrong}")), architecture).unwrap();
            assert_eq!(summary.mismatches, 1);
            assert!(
                summary.mismatch_examples[0]
                    .reason
                    .contains("value differs")
            );
        }
    }

    #[test]
    fn shift_left_checks_fields_and_values_only_with_known_prior_destination() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let summary = verify_scalar_trace(Cursor::new(SHIFTS), architecture).unwrap();
            assert_eq!(summary.shift_left_fields, 3);
            assert_eq!(summary.shift_left_value_checked, 2);
            assert_eq!(summary.shift_left_unknown_prior, 1);
            assert_eq!(summary.mismatches, 0, "{:?}", summary.mismatch_examples);

            let wrong = SHIFTS.replace("XD:X4=0x1c0000", "XD:X4=0x1c0001");
            let summary = verify_scalar_trace(Cursor::new(wrong), architecture).unwrap();
            assert_eq!(summary.mismatches, 1);
            assert!(
                summary.mismatch_examples[0]
                    .reason
                    .contains("SHL value differs")
            );
        }
    }

    #[test]
    fn store_xd_log_field_does_not_overwrite_known_register_state() {
        let log = "[info] [00005939] (PC: 0x126edaf4) SCALAR   : (Binary: 0x02129080) NEG  dtype:S64, XD:X9=0xffffffffffffffff, XN:X9=0x1, \n\
[info] [00005939] (PC: 0x126edb24) SCALAR   : (Binary: 0x0492a000) ST_XD_XN_IMM  dtype:B32, XD:X9=0, XN:X10=0x9b104, IMM:0, \n\
[info] [00005940] (PC: 0x126edafc) SCALAR   : (Binary: 0x02d2a240) SHL  dtype:B64, XD:X9=0xffffffffffffffff, XN:X10=0, DATA_SRC:XN, IMM:0, \n";
        let summary = verify_scalar_trace(Cursor::new(log), Architecture::Dav2201).unwrap();
        assert_eq!(summary.shift_left_fields, 1);
        assert_eq!(summary.shift_left_value_checked, 1);
        assert_eq!(summary.mismatches, 0, "{:?}", summary.mismatch_examples);
    }
}
