#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum C220MteL1ReadPort {
    Port0 = 0,
    Port1 = 1,
    Port2 = 2,
}

impl C220MteL1ReadPort {
    const ALL: [Self; 3] = [Self::Port0, Self::Port1, Self::Port2];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220MteL1ReadDestination {
    External,
    L0a,
    L0b,
    L0c,
    Smask,
    Fb,
    Bt,
    Sp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MteL1ReadHead {
    pub ready_tick: u64,
    pub destination: C220MteL1ReadDestination,
}

impl C220MteL1ReadHead {
    pub const fn eligible(self, tick: u64, output_fragments: usize) -> bool {
        self.ready_tick <= tick
            && match self.destination {
                C220MteL1ReadDestination::Sp => true,
                C220MteL1ReadDestination::Bt => output_fragments == 0,
                _ => output_fragments <= 1,
            }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MteL1ReadDecision {
    pub eligible: [bool; 3],
    pub selected: Option<C220MteL1ReadPort>,
}

/// Arbitrates the three MTE source queues feeding the single L1 read transport.
/// Selection advances priority even if the transport subsequently rejects it.
/// Call once per send callback; callers retain all unaccepted queue heads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MteL1ReadArbiter {
    last_selected: C220MteL1ReadPort,
}

impl Default for C220MteL1ReadArbiter {
    fn default() -> Self {
        Self {
            last_selected: C220MteL1ReadPort::Port2,
        }
    }
}

impl C220MteL1ReadArbiter {
    pub const fn last_selected(&self) -> C220MteL1ReadPort {
        self.last_selected
    }

    /// Output occupancy counts expanded fragments, not pending acknowledgments.
    pub fn preview(
        &self,
        tick: u64,
        heads: [Option<C220MteL1ReadHead>; 3],
        output_fragments: usize,
    ) -> C220MteL1ReadDecision {
        let eligible = heads.map(|head| head.is_some_and(|h| h.eligible(tick, output_fragments)));
        let selected = (1..=3)
            .map(|offset| C220MteL1ReadPort::ALL[(self.last_selected as usize + offset) % 3])
            .find(|port| eligible[*port as usize]);
        C220MteL1ReadDecision { eligible, selected }
    }

    pub fn arbitrate(
        &mut self,
        tick: u64,
        heads: [Option<C220MteL1ReadHead>; 3],
        output_fragments: usize,
    ) -> C220MteL1ReadDecision {
        let decision = self.preview(tick, heads, output_fragments);
        if let Some(port) = decision.selected {
            self.last_selected = port;
        }
        decision
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotates_on_selection_and_gates_by_expanded_output_occupancy() {
        use C220MteL1ReadDestination::{Bt, L0a, Sp};
        use C220MteL1ReadPort::{Port0, Port1, Port2};
        let mut arbiter = C220MteL1ReadArbiter::default();
        let heads = [Bt, L0a, Sp].map(|destination| {
            Some(C220MteL1ReadHead {
                ready_tick: 4,
                destination,
            })
        });
        assert_eq!(arbiter.arbitrate(3, heads, 0).selected, None);
        assert_eq!(arbiter.last_selected(), Port2);
        // Keep all heads offered: selection rotates even without dequeueing.
        for expected in [Port0, Port1, Port2, Port0] {
            assert_eq!(arbiter.arbitrate(4, heads, 0).selected, Some(expected));
        }
        let decision = arbiter.arbitrate(5, heads, 1);
        assert_eq!(decision.eligible, [false, true, true]);
        assert_eq!(decision.selected, Some(Port1));
        let decision = arbiter.arbitrate(6, heads, 2);
        assert_eq!(decision.eligible, [false, false, true]);
        assert_eq!(decision.selected, Some(Port2));
        assert_eq!(
            arbiter.arbitrate(7, [heads[0], heads[1], None], 2).selected,
            None
        );
        assert_eq!(arbiter.last_selected(), Port2);
    }
}
