use crate::isa::c220::control::C220SetCrossCoreInstruction;

/// Synchronization payload. The enclosing event supplies time and core identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220DeviceSync {
    pub value: u64,
    pub mode: u8,
    pub flag_id: u8,
}

impl C220DeviceSync {
    pub const fn from_value(value: u64) -> Self {
        Self {
            value,
            mode: ((value >> 4) & 3) as u8,
            flag_id: ((value >> 8) & 15) as u8,
        }
    }
}

/// A cross-core notification received and retired by its selected pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220CrossCoreReception {
    pub instruction_id: u64,
    pub pc: u64,
    pub tick: u64,
    pub instruction: C220SetCrossCoreInstruction,
    pub payload: C220DeviceSync,
}
