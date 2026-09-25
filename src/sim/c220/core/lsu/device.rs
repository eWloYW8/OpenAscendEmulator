use super::{C220CoreLoadCompletion, C220CoreLsuCompletion};
use crate::sim::c220::sync::C220DeviceSync;

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
