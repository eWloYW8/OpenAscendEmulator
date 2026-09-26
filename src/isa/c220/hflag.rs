use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220HardwareFlagOperation {
    Set,
    Wait,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220HardwareEventSource {
    Immediate(u8),
    Register(u8),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220HardwareFlagSourcePipe {
    Cube,
    Mte1,
    Fix,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220MatrixMemory {
    L0a,
    L0b,
    L0c,
    BiasTable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220HardwareFlagInstruction {
    pub word: u32,
    pub operation: C220HardwareFlagOperation,
    pub event_source: C220HardwareEventSource,
    pub source_pipe: C220HardwareFlagSourcePipe,
    pub destination_pipe_code: u8,
    pub memory: C220MatrixMemory,
    pub trigger: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220HardwareFlagStep {
    pub pc: u64,
    pub instruction: C220HardwareFlagInstruction,
    pub event_id: u32,
    pub source_value: Option<u64>,
}

impl C220HardwareFlagInstruction {
    pub const fn execution_pipe_code(self) -> u8 {
        match self.operation {
            C220HardwareFlagOperation::Wait => self.destination_pipe_code,
            C220HardwareFlagOperation::Set => match self.source_pipe {
                C220HardwareFlagSourcePipe::Cube => 2,
                C220HardwareFlagSourcePipe::Mte1 => 3,
                C220HardwareFlagSourcePipe::Fix => 10,
            },
        }
    }

    pub const fn decode(word: u32) -> Option<Self> {
        if !matches!(super::control::special_flow_form(word), Some(0 | 2 | 6 | 7)) {
            return None;
        }
        let form = ((word >> 5) & 3) as u8;
        let operation = if form & 1 == 0 {
            C220HardwareFlagOperation::Set
        } else {
            C220HardwareFlagOperation::Wait
        };
        let event_source = if form & 2 == 0 {
            C220HardwareEventSource::Immediate((word & 3) as u8)
        } else {
            C220HardwareEventSource::Register((word & 0x1f) as u8)
        };
        let source_pipe = match (word >> 10) & 0xf {
            2 => C220HardwareFlagSourcePipe::Cube,
            3 => C220HardwareFlagSourcePipe::Mte1,
            10 => C220HardwareFlagSourcePipe::Fix,
            _ => return None,
        };
        let destination_pipe_code = (((word >> 7) & 7) | (((word >> 14) & 1) << 3)) as u8;
        let memory = match (word >> 15) & 7 {
            1 => C220MatrixMemory::L0a,
            2 => C220MatrixMemory::L0b,
            3 => C220MatrixMemory::L0c,
            5 => C220MatrixMemory::BiasTable,
            _ => return None,
        };
        if !matches!(
            (source_pipe, destination_pipe_code, memory),
            (
                C220HardwareFlagSourcePipe::Mte1,
                2,
                C220MatrixMemory::L0a | C220MatrixMemory::L0b | C220MatrixMemory::BiasTable
            ) | (C220HardwareFlagSourcePipe::Fix, 2, C220MatrixMemory::L0c)
                | (
                    C220HardwareFlagSourcePipe::Cube,
                    3,
                    C220MatrixMemory::L0a | C220MatrixMemory::L0b | C220MatrixMemory::BiasTable
                )
                | (C220HardwareFlagSourcePipe::Cube, 10, C220MatrixMemory::L0c)
        ) {
            return None;
        }
        Some(Self {
            word,
            operation,
            event_source,
            source_pipe,
            destination_pipe_code,
            memory,
            trigger: word & (1 << 19) != 0,
        })
    }

    pub fn resolve(
        self,
        pc: u64,
        xregs: &[u64; 32],
    ) -> Result<C220HardwareFlagStep, C220HardwareFlagError> {
        let (event_id, source_value) = match self.event_source {
            C220HardwareEventSource::Immediate(event_id) => (u32::from(event_id), None),
            C220HardwareEventSource::Register(index) => {
                let value = xregs[usize::from(index)];
                let event_id = u32::try_from(value)
                    .map_err(|_| C220HardwareFlagError::InvalidEventId { event_id: value })?;
                (event_id, Some(value))
            }
        };
        if event_id > 3 {
            return Err(C220HardwareFlagError::InvalidEventId {
                event_id: u64::from(event_id),
            });
        }
        Ok(C220HardwareFlagStep {
            pc,
            instruction: self,
            event_id,
            source_value,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum C220HardwareFlagError {
    #[error("C220 hardware event id {event_id} is outside 0..=3")]
    InvalidEventId { event_id: u64 },
}
