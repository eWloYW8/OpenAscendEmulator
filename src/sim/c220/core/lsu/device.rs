use super::{C220CoreLoadCompletion, C220CoreLsuCompletion};

/// Device synchronization payload emitted when a device load or store retires.
/// The enclosing completion supplies the instruction identity and retirement time.
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

impl C220CoreLoadCompletion {
    pub fn device_sync(&self) -> Option<C220DeviceSync> {
        self.retirement
            .device_sync_value
            .map(C220DeviceSync::from_value)
    }
}

impl C220CoreLsuCompletion {
    pub fn device_sync(&self) -> C220DeviceSync {
        C220DeviceSync::from_value(self.issue.operands.source_value)
    }
}
