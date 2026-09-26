use super::C220Core;
use crate::sim::c220::schedule::C220StallCause;

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
    /// First outstanding execution domain blocking BAR.ALL. Independent memory
    /// transport and already-published event tokens do not block this barrier.
    pub fn barrier_all_blocker(&self) -> Option<C220StallCause> {
        let activity = self.activity();
        let fixp_pending = self.fixp_engine().is_some_and(|engine| {
            !engine.commands().is_empty()
                || !engine.factor_commands().is_empty()
                || !engine.control_commands().is_empty()
                || !engine.cross_core_commands().is_empty()
        }) || !self.fixp_frontend.is_idle();
        [
            (activity.scalar, C220StallCause::ScalarDependency),
            (
                self.lsu
                    .as_ref()
                    .is_some_and(|lsu| lsu.instructions_pending()),
                C220StallCause::LsuDependency,
            ),
            (activity.vector, C220StallCause::VectorDependency),
            (activity.cube, C220StallCause::CubeDependency),
            (activity.mte1, C220StallCause::Mte1Dependency),
            (activity.mte2, C220StallCause::Mte2Dependency),
            (activity.mte3, C220StallCause::Mte3Dependency),
            (fixp_pending, C220StallCause::FixpDependency),
            (
                activity.deferred_events,
                C220StallCause::PipelineEventDependency,
            ),
        ]
        .into_iter()
        .find_map(|(pending, cause)| pending.then_some(cause))
    }

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
