use crate::architecture::Architecture;
use crate::isa::class::AicClass;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JumpOffsetSource {
    Immediate { encoded_words: u16 },
    Register { index: u8 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnconditionalJump {
    pub architecture: Architecture,
    pub word: u32,
    pub offset_source: JumpOffsetSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConditionalJump {
    pub architecture: Architecture,
    pub word: u32,
    pub offset_source: JumpOffsetSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JumpTarget {
    pub pc: u64,
    pub word: u32,
    pub source_register: Option<u8>,
    pub source_value: Option<u64>,
    pub effective_offset_words: i64,
    pub target_pc: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlowNop {
    pub pc: u64,
    pub word: u32,
    pub target_pc: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlowEnd {
    pub pc: u64,
    pub word: u32,
    pub sequential_pc: u64,
}

impl FlowEnd {
    pub const fn decode(pc: u64, word: u32) -> Option<Self> {
        if !matches!(AicClass::from_word(word), AicClass::FlowControl) || ((word >> 21) & 0xf) != 11
        {
            return None;
        }
        Some(Self {
            pc,
            word,
            sequential_pc: pc.wrapping_add(4),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DcciInstruction {
    pub architecture: Architecture,
    pub word: u32,
    pub source_register: u8,
    pub entire_cache: bool,
    pub operation_field: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DcciStep {
    pub pc: u64,
    pub word: u32,
    pub source_register: u8,
    pub source_value: u64,
    pub entire_cache: bool,
    pub operation_field: u8,
    pub effective_address: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DsbStep {
    pub pc: u64,
    pub word: u32,
    pub scope_field: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineBarrierScope {
    Vector,
    Cube,
    Mte1,
    Fix,
    All,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PipelineBarrierStep {
    pub pc: u64,
    pub word: u32,
    pub scope: PipelineBarrierScope,
}

impl PipelineBarrierStep {
    pub const fn decode(architecture: Architecture, pc: u64, word: u32) -> Option<Self> {
        let scope = match (architecture, word) {
            (Architecture::Dav2201, 0x40e0_0400) => PipelineBarrierScope::Vector,
            (Architecture::Dav2201, 0x40e0_0800) => PipelineBarrierScope::Cube,
            (Architecture::Dav2201, 0x40e0_0c00) => PipelineBarrierScope::Mte1,
            (Architecture::Dav2201, 0x40e0_2800) => PipelineBarrierScope::Fix,
            (Architecture::Dav2201 | Architecture::Dav3510, 0x40e0_1800) => {
                PipelineBarrierScope::All
            }
            _ => return None,
        };
        Some(Self { pc, word, scope })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlagOperation {
    Set,
    Wait,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlagIdSource {
    Immediate(u8),
    Register(u8),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlagInstruction {
    pub architecture: Architecture,
    pub word: u32,
    pub operation: FlagOperation,
    pub source_pipe_code: u8,
    pub trigger_pipe_code: u8,
    pub id_source: FlagIdSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlagStep {
    pub pc: u64,
    pub instruction: FlagInstruction,
    pub flag_id: u32,
    pub source_value: Option<u64>,
}

impl FlagInstruction {
    pub const fn decode(architecture: Architecture, word: u32) -> Option<Self> {
        if !matches!(AicClass::from_word(word), AicClass::FlowControl) || word & 0x0200_0000 != 0 {
            return None;
        }
        let operation = match (word >> 21) & 0xf {
            5 => FlagOperation::Set,
            6 => FlagOperation::Wait,
            _ => return None,
        };
        let id_source = if word & 0x0002_0000 != 0 {
            FlagIdSource::Register(((word >> 2) & 0x1f) as u8)
        } else {
            FlagIdSource::Immediate((((word >> 18) & 1) << 2 | (word & 3)) as u8)
        };
        Some(Self {
            architecture,
            word,
            operation,
            source_pipe_code: ((word >> 10) & 0xf) as u8,
            trigger_pipe_code: (((word >> 7) & 7) | (((word >> 14) & 1) << 3)) as u8,
            id_source,
        })
    }

    pub fn resolve(self, pc: u64, xregs: &[u64; 32]) -> FlagStep {
        let (flag_id, source_value) = match self.id_source {
            FlagIdSource::Immediate(id) => (u32::from(id), None),
            FlagIdSource::Register(index) => {
                let value = xregs[usize::from(index)];
                (value as u32, Some(value))
            }
        };
        FlagStep {
            pc,
            instruction: self,
            flag_id,
            source_value,
        }
    }
}

impl DsbStep {
    pub const fn decode(pc: u64, word: u32) -> Option<Self> {
        if !matches!(AicClass::from_word(word), AicClass::FlowControl) || ((word >> 21) & 0xf) != 14
        {
            return None;
        }
        Some(Self {
            pc,
            word,
            scope_field: ((word >> 16) & 7) as u8,
        })
    }
}

impl DcciInstruction {
    pub const fn decode(architecture: Architecture, word: u32) -> Option<Self> {
        if !matches!(AicClass::from_word(word), AicClass::FlowControl)
            || ((word >> 25) & 0xf) == 1
            || ((word >> 21) & 0xf) != 1
            || ((word >> 18) & 7) != 3
        {
            return None;
        }
        Some(Self {
            architecture,
            word,
            source_register: ((word >> 12) & 0x1f) as u8,
            entire_cache: word & 0x2_0000 != 0,
            operation_field: (word & 3) as u8,
        })
    }

    pub fn resolve(self, pc: u64, xregs: &[u64; 32]) -> DcciStep {
        let source_value = xregs[usize::from(self.source_register)];
        DcciStep {
            pc,
            word: self.word,
            source_register: self.source_register,
            source_value,
            entire_cache: self.entire_cache,
            operation_field: self.operation_field,
            effective_address: if self.entire_cache { 0 } else { source_value },
        }
    }
}

impl FlowNop {
    pub const fn decode(_architecture: Architecture, pc: u64, word: u32) -> Option<Self> {
        if !matches!(AicClass::from_word(word), AicClass::FlowControl)
            || ((word >> 27) & 3) == 1
            || ((word >> 21) & 0xf) != 10
        {
            return None;
        }
        Some(Self {
            pc,
            word,
            target_pc: pc.wrapping_add(4),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConditionalJumpTarget {
    pub pc: u64,
    pub word: u32,
    pub condition_flag: u64,
    pub branch_taken: bool,
    pub fallthrough_pc: u64,
    pub taken_target: JumpTarget,
    pub target_pc: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JumpCompareOperand {
    Immediate { encoded: u8 },
    Register { index: u8 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JumpCompareOffset {
    Immediate { encoded_words: u16 },
    Register { index: u8 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JumpCompare {
    pub architecture: Architecture,
    pub word: u32,
    pub dtype_field: u8,
    pub condition_field: u8,
    pub first_source_register: u8,
    pub second_operand: JumpCompareOperand,
    pub offset_source: JumpCompareOffset,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JumpCompareTarget {
    pub pc: u64,
    pub word: u32,
    pub dtype_field: u8,
    pub condition_field: u8,
    pub first_source_register: u8,
    pub first_source_value: u64,
    pub second_source_register: Option<u8>,
    pub second_source_value: u64,
    pub branch_taken: bool,
    pub fallthrough_pc: u64,
    pub taken_target: JumpTarget,
    pub target_pc: u64,
    pub spr11_value: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum JumpCompareError {
    #[error("JUMPCMP dtype field {0} is not implemented")]
    UnsupportedDtype(u8),
    #[error("JUMPCMP condition field {0} is not implemented")]
    UnsupportedCondition(u8),
}

impl UnconditionalJump {
    pub const fn decode(architecture: Architecture, word: u32) -> Option<Self> {
        if !matches!(AicClass::from_word(word), AicClass::FlowControl)
            || ((word >> 20) & 0x1f) != 0
            || ((word >> 25) & 0xf) != 0
        {
            return None;
        }
        Some(Self {
            architecture,
            word,
            offset_source: decode_offset_source(word),
        })
    }

    pub fn resolve(self, pc: u64, xregs: &[u64; 32]) -> JumpTarget {
        resolve_offset(pc, self.word, self.offset_source, xregs)
    }
}

impl ConditionalJump {
    pub const fn decode(architecture: Architecture, word: u32) -> Option<Self> {
        if !matches!(AicClass::from_word(word), AicClass::FlowControl)
            || ((word >> 20) & 0x1f) != 2
            || ((word >> 18) & 7) != 0
            || ((word >> 25) & 0xf) != 0
        {
            return None;
        }
        Some(Self {
            architecture,
            word,
            offset_source: decode_offset_source(word),
        })
    }

    pub fn resolve(self, pc: u64, xregs: &[u64; 32], condition_flag: u64) -> ConditionalJumpTarget {
        let taken_target = resolve_offset(pc, self.word, self.offset_source, xregs);
        let fallthrough_pc = pc.wrapping_add(4);
        let branch_taken = condition_flag != 0;
        ConditionalJumpTarget {
            pc,
            word: self.word,
            condition_flag,
            branch_taken,
            fallthrough_pc,
            taken_target,
            target_pc: if branch_taken {
                taken_target.target_pc
            } else {
                fallthrough_pc
            },
        }
    }
}

impl JumpCompare {
    pub const fn decode(architecture: Architecture, word: u32) -> Option<Self> {
        if (word & 0xf800_0000) != 0x4800_0000 {
            return None;
        }
        let second_operand = if word & 0x1_0000 != 0 {
            JumpCompareOperand::Register {
                index: (word & 0x1f) as u8,
            }
        } else {
            JumpCompareOperand::Immediate {
                encoded: (word & 0x1f) as u8,
            }
        };
        let offset_source = if word & 0x2_0000 != 0 {
            JumpCompareOffset::Register {
                index: ((word >> 5) & 0x1f) as u8,
            }
        } else {
            JumpCompareOffset::Immediate {
                encoded_words: ((word >> 5) & 0x3ff) as u16,
            }
        };
        Some(Self {
            architecture,
            word,
            dtype_field: ((word >> 25) & 3) as u8,
            condition_field: ((word >> 18) & 7) as u8,
            first_source_register: ((((word >> 21) & 0xf) << 1) | ((word >> 15) & 1)) as u8,
            second_operand,
            offset_source,
        })
    }

    pub fn evaluate(
        self,
        pc: u64,
        xregs: &[u64; 32],
    ) -> Result<JumpCompareTarget, JumpCompareError> {
        if !matches!(self.dtype_field, 0..=2) {
            return Err(JumpCompareError::UnsupportedDtype(self.dtype_field));
        }
        if self.condition_field > 5 {
            return Err(JumpCompareError::UnsupportedCondition(self.condition_field));
        }
        let first_source_value = xregs[usize::from(self.first_source_register)];
        let (second_source_register, second_source_value) = match self.second_operand {
            JumpCompareOperand::Immediate { encoded } => {
                let signed = if encoded & 0x10 != 0 {
                    i64::from(encoded) - 32
                } else {
                    i64::from(encoded)
                };
                (None, signed as u64)
            }
            JumpCompareOperand::Register { index } => (Some(index), xregs[usize::from(index)]),
        };
        let branch_taken = match self.dtype_field {
            0 => compare_values(
                first_source_value as i64,
                second_source_value as i64,
                self.condition_field,
            ),
            1 => compare_values(
                first_source_value,
                second_source_value,
                self.condition_field,
            ),
            2 => {
                let first = f32::from_bits(first_source_value as u32);
                let second = f32::from_bits(second_source_value as u32);
                !first.is_nan()
                    && !second.is_nan()
                    && compare_values(first, second, self.condition_field)
            }
            _ => unreachable!("dtype was checked before comparison"),
        };
        let taken_target = self.taken_target(pc, xregs);
        let fallthrough_pc = pc.wrapping_add(4);
        Ok(JumpCompareTarget {
            pc,
            word: self.word,
            dtype_field: self.dtype_field,
            condition_field: self.condition_field,
            first_source_register: self.first_source_register,
            first_source_value,
            second_source_register,
            second_source_value,
            branch_taken,
            fallthrough_pc,
            taken_target,
            target_pc: if branch_taken {
                taken_target.target_pc
            } else {
                fallthrough_pc
            },
            spr11_value: u64::from(branch_taken),
        })
    }

    pub fn taken_target(self, pc: u64, xregs: &[u64; 32]) -> JumpTarget {
        resolve_compare_offset(pc, self.word, self.offset_source, xregs)
    }
}

pub(crate) fn compare_values<T: PartialEq + PartialOrd>(
    first: T,
    second: T,
    condition: u8,
) -> bool {
    match condition {
        0 => first == second,
        1 => first != second,
        2 => first < second,
        3 => first > second,
        4 => first >= second,
        5 => first <= second,
        _ => unreachable!("condition range was checked before evaluation"),
    }
}

fn resolve_compare_offset(
    pc: u64,
    word: u32,
    offset_source: JumpCompareOffset,
    xregs: &[u64; 32],
) -> JumpTarget {
    let (source_register, source_value, effective_offset_words) = match offset_source {
        JumpCompareOffset::Immediate { encoded_words } => {
            let signed = if encoded_words & 0x200 != 0 {
                i64::from(encoded_words) - 1024
            } else {
                i64::from(encoded_words)
            };
            (None, None, signed)
        }
        JumpCompareOffset::Register { index } => {
            let value = xregs[usize::from(index)];
            let extended = if value & (1_u64 << 45) != 0 {
                value | 0xffff_c000_0000_0000
            } else {
                value
            };
            (Some(index), Some(value), extended as i64)
        }
    };
    JumpTarget {
        pc,
        word,
        source_register,
        source_value,
        effective_offset_words,
        target_pc: pc.wrapping_add((effective_offset_words as u64).wrapping_mul(4)),
    }
}

const fn decode_offset_source(word: u32) -> JumpOffsetSource {
    if word & 0x20_000 != 0 {
        JumpOffsetSource::Register {
            index: ((word >> 12) & 0x1f) as u8,
        }
    } else {
        JumpOffsetSource::Immediate {
            encoded_words: word as u16,
        }
    }
}

fn resolve_offset(
    pc: u64,
    word: u32,
    offset_source: JumpOffsetSource,
    xregs: &[u64; 32],
) -> JumpTarget {
    let (source_register, source_value, effective_offset_words) = match offset_source {
        JumpOffsetSource::Immediate { encoded_words } => {
            (None, None, i64::from(encoded_words as i16))
        }
        JumpOffsetSource::Register { index } => {
            let value = xregs[usize::from(index)];
            let extended = if value & (1_u64 << 45) != 0 {
                value | 0xffff_c000_0000_0000
            } else {
                value
            };
            (Some(index), Some(value), extended as i64)
        }
    };
    JumpTarget {
        pc,
        word,
        source_register,
        source_value,
        effective_offset_words,
        target_pc: pc.wrapping_add((effective_offset_words as u64).wrapping_mul(4)),
    }
}

#[cfg(test)]
mod tests;
