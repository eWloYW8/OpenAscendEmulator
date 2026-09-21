use crate::device::architecture::Architecture;
use crate::instruction::isa::AicClass;
use serde::Serialize;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum JumpOffsetSource {
    Immediate { encoded_words: u16 },
    Register { index: u8 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct UnconditionalJump {
    pub architecture: Architecture,
    pub word: u32,
    pub offset_source: JumpOffsetSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ConditionalJump {
    pub architecture: Architecture,
    pub word: u32,
    pub offset_source: JumpOffsetSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct JumpTarget {
    pub pc: u64,
    pub word: u32,
    pub source_register: Option<u8>,
    pub source_value: Option<u64>,
    pub effective_offset_words: i64,
    pub target_pc: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct FlowNop {
    pub pc: u64,
    pub word: u32,
    pub target_pc: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct DcciInstruction {
    pub architecture: Architecture,
    pub word: u32,
    pub source_register: u8,
    pub entire_cache: bool,
    pub operation_field: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct DcciStep {
    pub pc: u64,
    pub word: u32,
    pub source_register: u8,
    pub source_value: u64,
    pub entire_cache: bool,
    pub operation_field: u8,
    pub effective_address: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct DsbStep {
    pub pc: u64,
    pub word: u32,
    pub scope_field: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum PipelineBarrierScope {
    Vector,
    All,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct PipelineBarrierStep {
    pub pc: u64,
    pub word: u32,
    pub scope: PipelineBarrierScope,
}

impl PipelineBarrierStep {
    pub const fn decode(architecture: Architecture, pc: u64, word: u32) -> Option<Self> {
        let scope = match (architecture, word) {
            (Architecture::Dav2201, 0x40e0_0400) => PipelineBarrierScope::Vector,
            (Architecture::Dav2201 | Architecture::Dav3510, 0x40e0_1800) => {
                PipelineBarrierScope::All
            }
            _ => return None,
        };
        Some(Self { pc, word, scope })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum FlagOperation {
    Set,
    Wait,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum FlagIdSource {
    Immediate(u8),
    Register(u8),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct FlagInstruction {
    pub architecture: Architecture,
    pub word: u32,
    pub operation: FlagOperation,
    pub source_pipe_code: u8,
    pub trigger_pipe_code: u8,
    pub id_source: FlagIdSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct FlagStep {
    pub pc: u64,
    pub instruction: FlagInstruction,
    pub flag_id: u32,
    pub source_value: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BufferOperation {
    Get,
    Release,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BufferEncoding {
    FlowControl,
    PushQueue,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BufferIdSource {
    Immediate(u8),
    Register(u8),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct C310BufferInstruction {
    pub word: u32,
    pub encoding: BufferEncoding,
    pub operation: BufferOperation,
    pub pipe_code: u8,
    pub id_source: BufferIdSource,
    pub mode_field: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct C310BufferStep {
    pub pc: u64,
    pub instruction: C310BufferInstruction,
    pub buffer_id: u8,
    pub source_value: Option<u64>,
}

impl C310BufferInstruction {
    pub const fn decode(architecture: Architecture, word: u32) -> Option<Self> {
        if !matches!(architecture, Architecture::Dav3510) {
            return None;
        }
        if word & 0xffc0_3fff == 0x15c0_0001 {
            let operation = if word & 0x8000 != 0 {
                BufferOperation::Get
            } else {
                BufferOperation::Release
            };
            let index = ((word >> 16) & 0x1f) as u8;
            let id_source = if word & 0x0020_0000 != 0 {
                BufferIdSource::Immediate(index)
            } else {
                BufferIdSource::Register(index)
            };
            return Some(Self {
                word,
                encoding: BufferEncoding::PushQueue,
                operation,
                pipe_code: 1,
                id_source,
                mode_field: ((word >> 14) & 1) as u8,
            });
        }
        if !matches!(AicClass::from_word(word), AicClass::FlowControl) {
            return None;
        }
        let operation = match word & 0xffe0_0000 {
            0x4200_0000 => BufferOperation::Get,
            0x4220_0000 => BufferOperation::Release,
            _ => return None,
        };
        let index = ((word >> 2) & 0x1f) as u8;
        let id_source = if word & 0x0002_0000 != 0 {
            BufferIdSource::Register(index)
        } else {
            BufferIdSource::Immediate(index)
        };
        Some(Self {
            word,
            encoding: BufferEncoding::FlowControl,
            operation,
            pipe_code: ((word >> 10) & 0xf) as u8,
            id_source,
            mode_field: (word & 1) as u8,
        })
    }

    pub fn resolve(self, pc: u64, xregs: &[u64; 32]) -> C310BufferStep {
        let (buffer_id, source_value) = match self.id_source {
            BufferIdSource::Immediate(id) => (id, None),
            BufferIdSource::Register(index) => {
                let value = xregs[usize::from(index)];
                ((value & 0x1f) as u8, Some(value))
            }
        };
        C310BufferStep {
            pc,
            instruction: self,
            buffer_id,
            source_value,
        }
    }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ConditionalJumpTarget {
    pub pc: u64,
    pub word: u32,
    pub condition_flag: u64,
    pub branch_taken: bool,
    pub fallthrough_pc: u64,
    pub taken_target: JumpTarget,
    pub target_pc: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum JumpCompareOperand {
    Immediate { encoded: u8 },
    Register { index: u8 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum JumpCompareOffset {
    Immediate { encoded_words: u16 },
    Register { index: u8 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct JumpCompare {
    pub architecture: Architecture,
    pub word: u32,
    pub dtype_field: u8,
    pub condition_field: u8,
    pub first_source_register: u8,
    pub second_operand: JumpCompareOperand,
    pub offset_source: JumpCompareOffset,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
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
mod tests {
    use super::*;

    #[test]
    fn end_identifies_the_terminal_flow_instruction() {
        let step = FlowEnd::decode(0x10d0_d8b8, 0x4160_0000).unwrap();
        assert_eq!(step.sequential_pc, 0x10d0_d8bc);
        assert!(FlowEnd::decode(0x10d0_d8b8, 0x402e_f000).is_none());
        assert!(FlowEnd::decode(0x10d0_d8b8, 0x6160_0000).is_none());
    }

    #[test]
    fn dcci_decodes_addressed_and_entire_cache_forms_on_both_architectures() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut regs = [0_u64; 32];
            regs[8] = 0x1131_3000;
            regs[15] = 0x10d0_d000;

            let addressed = DcciInstruction::decode(architecture, 0x402c_8000).unwrap();
            assert_eq!(addressed.source_register, 8);
            assert!(!addressed.entire_cache);
            assert_eq!(addressed.operation_field, 0);
            let step = addressed.resolve(0x1131_216c, &regs);
            assert_eq!(step.source_value, 0x1131_3000);
            assert_eq!(step.effective_address, step.source_value);

            let entire = DcciInstruction::decode(architecture, 0x402e_f000).unwrap();
            assert_eq!(entire.source_register, 15);
            assert!(entire.entire_cache);
            let step = entire.resolve(0x10d0_d8b4, &regs);
            assert_eq!(step.source_value, 0x10d0_d000);
            assert_eq!(step.effective_address, 0);

            for operation_field in 0..=3 {
                let instruction =
                    DcciInstruction::decode(architecture, 0x402c_8000 | operation_field).unwrap();
                assert_eq!(instruction.operation_field, operation_field as u8);
            }
            assert!(DcciInstruction::decode(architecture, 0x4020_000a).is_none());
            assert!(DcciInstruction::decode(architecture, 0x422c_8000).is_none());
            assert!(DcciInstruction::decode(architecture, 0x602c_8000).is_none());
        }
    }

    #[test]
    fn dsb_decodes_the_scope_field() {
        let step = DsbStep::decode(0x1131_2170, 0x41c1_0000).unwrap();
        assert_eq!(step.scope_field, 1);
        for scope in 0..=7 {
            let word = 0x41c0_0000 | (scope << 16);
            assert_eq!(
                DsbStep::decode(0x1000, word).unwrap().scope_field,
                scope as u8
            );
        }
        assert!(DsbStep::decode(0x1000, 0x402c_8000).is_none());
        assert!(DsbStep::decode(0x1000, 0x61c1_0000).is_none());
    }

    #[test]
    fn barriers_decode_only_the_supported_scope_words() {
        assert_eq!(
            PipelineBarrierStep::decode(Architecture::Dav2201, 0x1131_2638, 0x40e0_0400),
            Some(PipelineBarrierStep {
                pc: 0x1131_2638,
                word: 0x40e0_0400,
                scope: PipelineBarrierScope::Vector,
            })
        );
        assert_eq!(
            PipelineBarrierStep::decode(Architecture::Dav2201, 0x1131_2090, 0x40e0_1800)
                .unwrap()
                .scope,
            PipelineBarrierScope::All
        );
        assert_eq!(
            PipelineBarrierStep::decode(Architecture::Dav3510, 0x10d0_d090, 0x40e0_1800)
                .unwrap()
                .scope,
            PipelineBarrierScope::All
        );
        for word in [0x40e0_0401, 0x40e0_0800, 0x40e0_1801] {
            assert!(PipelineBarrierStep::decode(Architecture::Dav2201, 0x1000, word).is_none());
        }
        for word in [0x40e0_0400, 0x40e0_1801] {
            assert!(PipelineBarrierStep::decode(Architecture::Dav3510, 0x1000, word).is_none());
        }
    }

    #[test]
    fn flag_ids_use_the_live_encoded_register_on_both_architectures() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let instruction = FlagInstruction::decode(architecture, 0x40a2_0630).unwrap();
            assert_eq!(instruction.operation, FlagOperation::Set);
            assert_eq!(instruction.source_pipe_code, 1);
            assert_eq!(instruction.trigger_pipe_code, 4);
            assert_eq!(instruction.id_source, FlagIdSource::Register(12));
            let mut xregs = [0_u64; 32];
            assert_eq!(instruction.resolve(0x1131_26f8, &xregs).flag_id, 0);
            xregs[12] = 1;
            let step = instruction.resolve(0x1131_2704, &xregs);
            assert_eq!(step.flag_id, 1);
            assert_eq!(step.source_value, Some(1));

            let output_set = FlagInstruction::decode(architecture, 0x40a2_06b8).unwrap();
            assert_eq!(output_set.id_source, FlagIdSource::Register(14));
            assert_eq!(
                (output_set.source_pipe_code, output_set.trigger_pipe_code),
                (1, 5)
            );
            let output_wait = FlagInstruction::decode(architecture, 0x40c2_06b4).unwrap();
            assert_eq!(output_wait.operation, FlagOperation::Wait);
            assert_eq!(output_wait.id_source, FlagIdSource::Register(13));
            assert_eq!(
                (output_wait.source_pipe_code, output_wait.trigger_pipe_code),
                (1, 5)
            );

            let immediate = FlagInstruction::decode(architecture, 0x40a0_1000).unwrap();
            assert_eq!(immediate.id_source, FlagIdSource::Immediate(0));
            assert_eq!(immediate.resolve(0x1000, &xregs).source_value, None);
            assert!(FlagInstruction::decode(architecture, 0x42a2_0630).is_none());
            assert!(FlagInstruction::decode(architecture, 0x40e0_1800).is_none());
        }
    }

    #[test]
    fn c310_buffer_get_and_release_resolve_live_ids_and_pipe_codes() {
        let mut xregs = [0_u64; 32];
        xregs[0] = 1;
        xregs[1] = 0x21;
        xregs[25] = 7;
        for (word, operation, register, id, mode) in [
            (0x4202_1004, BufferOperation::Get, 1, 1, 0),
            (0x4222_1004, BufferOperation::Release, 1, 1, 0),
            (0x4202_1000, BufferOperation::Get, 0, 1, 0),
            (0x4202_1005, BufferOperation::Get, 1, 1, 1),
            (0x4202_1065, BufferOperation::Get, 25, 7, 1),
        ] {
            let instruction = C310BufferInstruction::decode(Architecture::Dav3510, word).unwrap();
            assert_eq!(instruction.operation, operation);
            assert_eq!(instruction.encoding, BufferEncoding::FlowControl);
            assert_eq!(instruction.pipe_code, 4);
            assert_eq!(instruction.id_source, BufferIdSource::Register(register));
            assert_eq!(instruction.mode_field, mode);
            let resolved = instruction.resolve(0x10d0_d690, &xregs);
            assert_eq!(resolved.buffer_id, id);
            assert_eq!(resolved.source_value, Some(xregs[usize::from(register)]));
        }
        let immediate = C310BufferInstruction::decode(Architecture::Dav3510, 0x4200_100c).unwrap();
        assert_eq!(immediate.id_source, BufferIdSource::Immediate(3));
        assert_eq!(immediate.resolve(0x1000, &xregs).buffer_id, 3);
        assert_eq!(immediate.resolve(0x1000, &xregs).source_value, None);
        for (word, operation) in [
            (0x4202_1400, BufferOperation::Get),
            (0x4222_1400, BufferOperation::Release),
        ] {
            let instruction = C310BufferInstruction::decode(Architecture::Dav3510, word).unwrap();
            assert_eq!(instruction.operation, operation);
            assert_eq!(instruction.pipe_code, 5);
            assert_eq!(instruction.id_source, BufferIdSource::Register(0));
        }
        for word in [0x15c0_8001, 0x40a2_0630, 0x4202_1004] {
            assert!(C310BufferInstruction::decode(Architecture::Dav2201, word).is_none());
        }
        assert!(C310BufferInstruction::decode(Architecture::Dav3510, 0x15c0_8021).is_none());
    }

    #[test]
    fn c310_push_queue_buffer_forms_resolve_register_and_immediate_ids() {
        let mut xregs = [0_u64; 32];
        xregs[0] = 0x41;
        for (word, operation, mode) in [
            (0x15c0_8001, BufferOperation::Get, 0),
            (0x15c0_0001, BufferOperation::Release, 0),
            (0x15c0_c001, BufferOperation::Get, 1),
        ] {
            let instruction = C310BufferInstruction::decode(Architecture::Dav3510, word).unwrap();
            assert_eq!(instruction.encoding, BufferEncoding::PushQueue);
            assert_eq!(instruction.operation, operation);
            assert_eq!(instruction.pipe_code, 1);
            assert_eq!(instruction.id_source, BufferIdSource::Register(0));
            assert_eq!(instruction.mode_field, mode);
            let resolved = instruction.resolve(0x10d0_d768, &xregs);
            assert_eq!(resolved.buffer_id, 1);
            assert_eq!(resolved.source_value, Some(0x41));
        }
        for (word, operation) in [
            (0x15e3_8001, BufferOperation::Get),
            (0x15e3_0001, BufferOperation::Release),
        ] {
            let instruction = C310BufferInstruction::decode(Architecture::Dav3510, word).unwrap();
            assert_eq!(instruction.operation, operation);
            assert_eq!(instruction.id_source, BufferIdSource::Immediate(3));
            let resolved = instruction.resolve(0x10d0_d768, &xregs);
            assert_eq!(resolved.buffer_id, 3);
            assert_eq!(resolved.source_value, None);
        }
        xregs[13] = 0x3f;
        for (word, operation) in [
            (0x15cd_8001, BufferOperation::Get),
            (0x15cd_0001, BufferOperation::Release),
        ] {
            let instruction = C310BufferInstruction::decode(Architecture::Dav3510, word).unwrap();
            assert_eq!(instruction.operation, operation);
            assert_eq!(instruction.id_source, BufferIdSource::Register(13));
            assert_eq!(instruction.resolve(0x10d0_d768, &xregs).buffer_id, 31);
        }
    }

    #[test]
    fn flow_nop_advances_one_word_on_both_architectures() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let nop = FlowNop::decode(architecture, 0x11312160, 0x4140_0000).unwrap();
            assert_eq!(nop.target_pc, 0x11312164);
            assert!(FlowNop::decode(architecture, 0x11312164, 0x4140_0000).is_some());
            assert!(FlowNop::decode(architecture, 0x11312160, 0x4000_0000).is_none());
            assert!(FlowNop::decode(architecture, 0x11312160, 0x4940_0000).is_none());
        }
    }

    #[test]
    fn relative_immediate_jump_uses_signed_word_displacement() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let regs = [0; 32];
            let forward = UnconditionalJump::decode(architecture, 0x4000_03e0).unwrap();
            assert_eq!(forward.resolve(0x112f_5348, &regs).target_pc, 0x112f_62c8);
            let backward = UnconditionalJump::decode(architecture, 0x4000_fff9).unwrap();
            let target = backward.resolve(0x126e_cfb8, &regs);
            assert_eq!(target.effective_offset_words, -7);
            assert_eq!(target.target_pc, 0x126e_cf9c);
        }
    }

    #[test]
    fn register_jump_sign_extends_bit_45_and_rejects_other_flow_routes() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut regs = [0; 32];
            regs[3] = 0x3fff_ffff_fff9;
            let jump = UnconditionalJump::decode(architecture, 0x4002_3000).unwrap();
            assert_eq!(jump.offset_source, JumpOffsetSource::Register { index: 3 });
            let target = jump.resolve(0x1000, &regs);
            assert_eq!(target.source_value, Some(0x3fff_ffff_fff9));
            assert_eq!(target.effective_offset_words, -7);
            assert_eq!(target.target_pc, 0x0fe4);
            assert!(UnconditionalJump::decode(architecture, 0x4020_0002).is_none());
            assert!(UnconditionalJump::decode(architecture, 0x4884_1521).is_none());
            assert!(UnconditionalJump::decode(architecture, 0x0000_0000).is_none());
        }
        assert!(UnconditionalJump::decode(Architecture::Dav3510, 0x4200_0000).is_none());
    }

    #[test]
    fn conditional_jump_selects_taken_or_fallthrough_target() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let regs = [0; 32];
            let jump = ConditionalJump::decode(architecture, 0x4020_0002).unwrap();
            let not_taken = jump.resolve(0x126e_cfb4, &regs, 0);
            assert!(!not_taken.branch_taken);
            assert_eq!(not_taken.target_pc, 0x126e_cfb8);
            assert_eq!(not_taken.taken_target.target_pc, 0x126e_cfbc);
            let taken = jump.resolve(0x126e_cfb4, &regs, 1);
            assert!(taken.branch_taken);
            assert_eq!(taken.target_pc, 0x126e_cfbc);

            let backward = ConditionalJump::decode(architecture, 0x4020_fff9).unwrap();
            assert_eq!(
                backward.resolve(0x126e_d00c, &regs, 1).target_pc,
                0x126e_cff0
            );
            assert!(ConditionalJump::decode(architecture, 0x4000_0002).is_none());
            assert!(ConditionalJump::decode(architecture, 0x4024_0002).is_none());
        }
    }

    #[test]
    fn jump_compare_decodes_and_evaluates_integer_operands() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut regs = [0; 32];
            regs[8] = 1;
            let immediate = JumpCompare::decode(architecture, 0x4884_1521).unwrap();
            assert_eq!(immediate.first_source_register, 8);
            assert_eq!(immediate.condition_field, 1);
            assert_eq!(
                immediate.second_operand,
                JumpCompareOperand::Immediate { encoded: 1 }
            );
            assert_eq!(
                immediate.offset_source,
                JumpCompareOffset::Immediate {
                    encoded_words: 0xa9
                }
            );
            let result = immediate.evaluate(0x112f_50b8, &regs).unwrap();
            assert!(!result.branch_taken);
            assert_eq!(result.target_pc, 0x112f_50bc);
            assert_eq!(result.spr11_value, 0);

            regs[1] = 0x209;
            regs[2] = 0x301;
            let register = JumpCompare::decode(architecture, 0x4a09_83a2).unwrap();
            assert_eq!(register.first_source_register, 1);
            assert_eq!(
                register.second_operand,
                JumpCompareOperand::Register { index: 2 }
            );
            let result = register.evaluate(0x126e_d14c, &regs).unwrap();
            assert!(result.branch_taken);
            assert_eq!(result.target_pc, 0x126e_d1c0);
            assert_eq!(result.spr11_value, 1);
        }
    }

    #[test]
    fn jump_compare_sign_extends_immediate_and_backwards_displacement() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut regs = [0; 32];
            regs[8] = 1;
            let word = (0x4884_1521 & !0x7fff) | (0x3ff << 5) | 0x1f;
            let jump = JumpCompare::decode(architecture, word).unwrap();
            let result = jump.evaluate(0x1000, &regs).unwrap();
            assert_eq!(result.second_source_value, u64::MAX);
            assert_eq!(result.taken_target.effective_offset_words, -1);
            assert_eq!(result.target_pc, 0xffc);

            let signed_less =
                JumpCompare::decode(architecture, (word & !(7 << 18)) | (2 << 18)).unwrap();
            let unsigned_less =
                JumpCompare::decode(architecture, signed_less.word | (1 << 25)).unwrap();
            assert!(!signed_less.evaluate(0x1000, &regs).unwrap().branch_taken);
            assert!(unsigned_less.evaluate(0x1000, &regs).unwrap().branch_taken);

            let unsupported = JumpCompare::decode(architecture, word | (3 << 25)).unwrap();
            assert_eq!(
                unsupported.evaluate(0x1000, &regs),
                Err(JumpCompareError::UnsupportedDtype(3))
            );
            let invalid_condition = JumpCompare::decode(architecture, word | (6 << 18)).unwrap();
            assert_eq!(
                invalid_condition.evaluate(0x1000, &regs),
                Err(JumpCompareError::UnsupportedCondition(7))
            );
        }
    }

    #[test]
    fn jump_compare_register_offset_uses_signed_46_bit_words() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let word = (0x4884_1521 & !0x7fe0) | 0x2_0000 | (3 << 5);
            let jump = JumpCompare::decode(architecture, word).unwrap();
            assert_eq!(jump.offset_source, JumpCompareOffset::Register { index: 3 });
            let mut regs = [0; 32];
            regs[3] = 0x3fff_ffff_fff9;
            regs[8] = 2;
            let result = jump.evaluate(0x1000, &regs).unwrap();
            assert!(result.branch_taken);
            assert_eq!(result.taken_target.effective_offset_words, -7);
            assert_eq!(result.target_pc, 0xfe4);
        }
    }

    #[test]
    fn float_compare_uses_low_word_bits_and_ordered_nan_behavior() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut regs = [0; 32];
            let equal_word = (0x4884_1521 & !(7 << 18)) | (2 << 25);
            let equal = JumpCompare::decode(architecture, equal_word).unwrap();
            regs[8] = 1;
            assert!(equal.evaluate(0x1000, &regs).unwrap().branch_taken);
            regs[8] = (1_u64 << 32) | 1;
            assert!(equal.evaluate(0x1000, &regs).unwrap().branch_taken);
            regs[8] = (-0.0_f32).to_bits().into();
            let equal_zero = JumpCompare::decode(architecture, equal_word & !0x1f).unwrap();
            assert!(equal_zero.evaluate(0x1000, &regs).unwrap().branch_taken);

            let not_equal = JumpCompare::decode(architecture, equal_word | (1 << 18)).unwrap();
            regs[8] = 0x7fc0_0000;
            assert!(!not_equal.evaluate(0x1000, &regs).unwrap().branch_taken);
            let nan_immediate = JumpCompare::decode(architecture, not_equal.word | 0x1f).unwrap();
            regs[8] = 0;
            assert!(!nan_immediate.evaluate(0x1000, &regs).unwrap().branch_taken);
            regs[8] = 0x7f80_0000;
            assert!(not_equal.evaluate(0x1000, &regs).unwrap().branch_taken);

            let register_word = (0x4a09_83a2 & !(3 << 25) & !(7 << 18)) | (2 << 25) | (3 << 18);
            let greater = JumpCompare::decode(architecture, register_word).unwrap();
            regs[1] = 0x7f80_0000;
            regs[2] = 1.0_f32.to_bits().into();
            let outcome = greater.evaluate(0x2000, &regs).unwrap();
            assert!(outcome.branch_taken);
            assert_eq!(outcome.spr11_value, 1);
            regs[2] = 0x7fc0_0000;
            assert!(!greater.evaluate(0x2000, &regs).unwrap().branch_taken);
        }
    }
}
