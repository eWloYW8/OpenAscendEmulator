use crate::isa::class::AicClass;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C310BufferOperation {
    Get,
    Release,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C310BufferEncoding {
    FlowControl,
    PushQueue,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C310BufferIdSource {
    Immediate(u8),
    Register(u8),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C310BufferInstruction {
    pub word: u32,
    pub encoding: C310BufferEncoding,
    pub operation: C310BufferOperation,
    pub pipe_code: u8,
    pub id_source: C310BufferIdSource,
    pub mode_field: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C310BufferStep {
    pub pc: u64,
    pub instruction: C310BufferInstruction,
    pub buffer_id: u8,
    pub source_value: Option<u64>,
}

impl C310BufferInstruction {
    pub const fn decode(word: u32) -> Option<Self> {
        if word & 0xffc0_3fff == 0x15c0_0001 {
            let operation = if word & 0x8000 != 0 {
                C310BufferOperation::Get
            } else {
                C310BufferOperation::Release
            };
            let index = ((word >> 16) & 0x1f) as u8;
            let id_source = if word & 0x0020_0000 != 0 {
                C310BufferIdSource::Immediate(index)
            } else {
                C310BufferIdSource::Register(index)
            };
            return Some(Self {
                word,
                encoding: C310BufferEncoding::PushQueue,
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
            0x4200_0000 => C310BufferOperation::Get,
            0x4220_0000 => C310BufferOperation::Release,
            _ => return None,
        };
        let index = ((word >> 2) & 0x1f) as u8;
        let id_source = if word & 0x0002_0000 != 0 {
            C310BufferIdSource::Register(index)
        } else {
            C310BufferIdSource::Immediate(index)
        };
        Some(Self {
            word,
            encoding: C310BufferEncoding::FlowControl,
            operation,
            pipe_code: ((word >> 10) & 0xf) as u8,
            id_source,
            mode_field: (word & 1) as u8,
        })
    }

    pub fn resolve(self, pc: u64, xregs: &[u64; 32]) -> C310BufferStep {
        let (buffer_id, source_value) = match self.id_source {
            C310BufferIdSource::Immediate(id) => (id, None),
            C310BufferIdSource::Register(index) => {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flow_forms_resolve_live_ids_and_pipe_codes() {
        let mut xregs = [0_u64; 32];
        xregs[0] = 1;
        xregs[1] = 0x21;
        xregs[25] = 7;
        for (word, operation, register, id, mode) in [
            (0x4202_1004, C310BufferOperation::Get, 1, 1, 0),
            (0x4222_1004, C310BufferOperation::Release, 1, 1, 0),
            (0x4202_1000, C310BufferOperation::Get, 0, 1, 0),
            (0x4202_1005, C310BufferOperation::Get, 1, 1, 1),
            (0x4202_1065, C310BufferOperation::Get, 25, 7, 1),
        ] {
            let instruction = C310BufferInstruction::decode(word).unwrap();
            assert_eq!(instruction.operation, operation);
            assert_eq!(instruction.encoding, C310BufferEncoding::FlowControl);
            assert_eq!(instruction.pipe_code, 4);
            assert_eq!(
                instruction.id_source,
                C310BufferIdSource::Register(register)
            );
            assert_eq!(instruction.mode_field, mode);
            let resolved = instruction.resolve(0x10d0_d690, &xregs);
            assert_eq!(resolved.buffer_id, id);
            assert_eq!(resolved.source_value, Some(xregs[usize::from(register)]));
        }
        let immediate = C310BufferInstruction::decode(0x4200_100c).unwrap();
        assert_eq!(immediate.id_source, C310BufferIdSource::Immediate(3));
        assert_eq!(immediate.resolve(0x1000, &xregs).buffer_id, 3);
        assert_eq!(immediate.resolve(0x1000, &xregs).source_value, None);
        for (word, operation) in [
            (0x4202_1400, C310BufferOperation::Get),
            (0x4222_1400, C310BufferOperation::Release),
        ] {
            let instruction = C310BufferInstruction::decode(word).unwrap();
            assert_eq!(instruction.operation, operation);
            assert_eq!(instruction.pipe_code, 5);
            assert_eq!(instruction.id_source, C310BufferIdSource::Register(0));
        }
        assert!(C310BufferInstruction::decode(0x15c0_8021).is_none());
        assert!(C310BufferInstruction::decode(0x40a2_0630).is_none());
    }

    #[test]
    fn push_queue_forms_resolve_register_and_immediate_ids() {
        let mut xregs = [0_u64; 32];
        xregs[0] = 0x41;
        for (word, operation, mode) in [
            (0x15c0_8001, C310BufferOperation::Get, 0),
            (0x15c0_0001, C310BufferOperation::Release, 0),
            (0x15c0_c001, C310BufferOperation::Get, 1),
        ] {
            let instruction = C310BufferInstruction::decode(word).unwrap();
            assert_eq!(instruction.encoding, C310BufferEncoding::PushQueue);
            assert_eq!(instruction.operation, operation);
            assert_eq!(instruction.pipe_code, 1);
            assert_eq!(instruction.id_source, C310BufferIdSource::Register(0));
            assert_eq!(instruction.mode_field, mode);
            let resolved = instruction.resolve(0x10d0_d768, &xregs);
            assert_eq!(resolved.buffer_id, 1);
            assert_eq!(resolved.source_value, Some(0x41));
        }
        for (word, operation) in [
            (0x15e3_8001, C310BufferOperation::Get),
            (0x15e3_0001, C310BufferOperation::Release),
        ] {
            let instruction = C310BufferInstruction::decode(word).unwrap();
            assert_eq!(instruction.operation, operation);
            assert_eq!(instruction.id_source, C310BufferIdSource::Immediate(3));
            let resolved = instruction.resolve(0x10d0_d768, &xregs);
            assert_eq!(resolved.buffer_id, 3);
            assert_eq!(resolved.source_value, None);
        }
        xregs[13] = 0x3f;
        for (word, operation) in [
            (0x15cd_8001, C310BufferOperation::Get),
            (0x15cd_0001, C310BufferOperation::Release),
        ] {
            let instruction = C310BufferInstruction::decode(word).unwrap();
            assert_eq!(instruction.operation, operation);
            assert_eq!(instruction.id_source, C310BufferIdSource::Register(13));
            assert_eq!(instruction.resolve(0x10d0_d768, &xregs).buffer_id, 31);
        }
    }
}
