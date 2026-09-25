use std::collections::BTreeMap;

/// Per-core notification credits, independent of local pipeline hardware flags.
#[derive(Debug, Clone, Default)]
pub struct C220DeviceFlagState {
    counters: BTreeMap<u8, u32>,
    blocked: bool,
    waiting_for: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220DeviceFlagDelivery {
    pub flag_id: u8,
    pub count: u32,
    pub overflow: bool,
    pub dispatch_unblocked: bool,
}

impl C220DeviceFlagState {
    pub fn counters(&self) -> &BTreeMap<u8, u32> {
        &self.counters
    }

    pub fn count(&self, flag_id: u32) -> u32 {
        u8::try_from(flag_id)
            .ok()
            .and_then(|id| self.counters.get(&id).copied())
            .unwrap_or(0)
    }

    pub fn is_blocked(&self) -> bool {
        self.blocked
    }

    /// A failed wait remains pending while delivery permits a retry.
    pub fn waiting_for(&self) -> Option<u32> {
        self.waiting_for
    }

    /// Call at the fabric's delivery boundary, not at the sender's issue time.
    /// Even an unrelated flag can permit the waiting instruction to retry.
    pub fn receive(&mut self, flag_id: u8) -> C220DeviceFlagDelivery {
        let count = self.counters.entry(flag_id).or_default();
        *count = count.wrapping_add(1);
        let overflow = *count > 15;
        let dispatch_unblocked = self.blocked && !overflow;
        if !overflow {
            self.blocked = false;
        }
        C220DeviceFlagDelivery {
            flag_id,
            count: *count,
            overflow,
            dispatch_unblocked,
        }
    }

    pub(crate) fn try_wait(&mut self, flag_id: u32) -> bool {
        if let Ok(id) = u8::try_from(flag_id)
            && let Some(count) = self.counters.get_mut(&id)
            && *count != 0
        {
            *count -= 1;
            self.blocked = false;
            self.waiting_for = None;
            return true;
        }
        self.blocked = true;
        self.waiting_for = Some(flag_id);
        false
    }

    /// Clear delivered credits without changing the dispatch hazard latch.
    pub fn reset_counters(&mut self) {
        self.counters.clear();
    }

    pub fn all_consumed(&self) -> bool {
        self.counters.values().all(|&count| count == 0)
    }
}
