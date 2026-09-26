use super::C220Core;

/// Outstanding execution work, excluding completed results and ready event tokens.
/// This is an occupancy snapshot, not an END instruction latency prediction.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct C220CoreActivity {
    pub scalar: bool,
    pub lsu: bool,
    pub vector: bool,
    pub cube: bool,
    pub mte1: bool,
    pub mte2: bool,
    pub mte3: bool,
    pub fixp: bool,
    pub memory: bool,
    pub deferred_events: bool,
}

impl C220CoreActivity {
    pub fn is_idle(self) -> bool {
        self == Self::default()
    }
}

impl C220Core {
    pub fn activity(&self) -> C220CoreActivity {
        C220CoreActivity {
            scalar: self.scalar_timing.pending_drain_tick().is_some(),
            lsu: self.lsu.as_ref().is_some_and(|lsu| !lsu.is_idle()),
            vector: !self.vector.is_idle(),
            cube: self.cube.pipeline.pending_drain_tick().is_some()
                || !self.cube_frontend.commands.is_empty()
                || self.cube_frontend.active.is_some()
                || !self.cube_frontend.barriers.is_empty(),
            mte1: self.mte1.pending_commands().next().is_some()
                || !self.mte1_frontend.issued.is_empty()
                || !self.mte1_frontend.commands.is_empty()
                || !self.mte1_frontend.barriers.is_empty(),
            mte2: self.mte2.is_busy() || !self.mte2_frontend.is_idle(),
            mte3: self.mte3.pending_commands().next().is_some()
                || !self.mte3.native_commands.is_empty()
                || !self.mte3_issue_queue.is_idle(),
            fixp: !self.fixp_frontend.is_idle()
                || self
                    .fixp
                    .as_ref()
                    .is_some_and(|fixp| !fixp.engine.is_idle())
                || self
                    .external_fixp
                    .as_ref()
                    .is_some_and(|fixp| !fixp.engine.is_idle()),
            memory: self
                .mte_pipeline
                .as_ref()
                .is_some_and(|pipeline| !pipeline.is_idle()),
            deferred_events: self.pipeline_events.pending().next().is_some(),
        }
    }
}
