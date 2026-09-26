use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220StallCause {
    InstructionRate,
    ScalarDependency,
    LsuDependency,
    Mte1IssueRate,
    Mte1IssueQueueFull,
    Mte1CommandQueueFull,
    Mte1OutstandingLimit,
    Mte1Dependency,
    Mte1Barrier,
    Mte2Barrier,
    MtePhysicalDependency,
    HardwareFlagDependency,
    PipelineEventDependency,
    DeviceFlagDependency,
    Mte2IssueRate,
    Mte2IssueQueueFull,
    Mte2CommandQueueFull,
    Mte2OutstandingLimit,
    Mte2Dependency,
    Mte3IssueRate,
    Mte3QueueFull,
    Mte3IssueQueueFull,
    Mte3OutstandingLimit,
    Mte3Barrier,
    Mte3Dependency,
    VectorDependency,
    VectorIssueQueueFull,
    VectorReceptionQueueFull,
    VectorOutstandingLimit,
    CubeDependency,
    CubeIssueQueueFull,
    CubeOutstandingLimit,
    CubeBarrier,
    FixpDependency,
    FixpIssueQueueFull,
    FixpCommandQueueFull,
    FixpOutstandingLimit,
    FixpBarrier,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Stall {
    pub tick: u64,
    pub pc: u64,
    pub resume_tick: u64,
    pub cause: C220StallCause,
}

#[derive(Debug, Error)]
pub enum C220ScheduleError {
    #[error("tick {requested} precedes the previously observed tick {previous}")]
    TimeReversed { requested: u64, previous: u64 },
    #[error("instruction clock overflowed")]
    TimeOverflow,
}

#[derive(Default)]
pub(crate) struct C220IssueClock {
    last_tick: Option<u64>,
    next_instruction_tick: u64,
}

impl C220IssueClock {
    pub(crate) fn observe(
        &mut self,
        tick: u64,
        pc: u64,
    ) -> Result<Option<C220Stall>, C220ScheduleError> {
        if let Some(previous) = self.last_tick
            && tick < previous
        {
            return Err(C220ScheduleError::TimeReversed {
                requested: tick,
                previous,
            });
        }
        self.last_tick = Some(tick);
        Ok((tick < self.next_instruction_tick).then_some(C220Stall {
            tick,
            pc,
            resume_tick: self.next_instruction_tick,
            cause: C220StallCause::InstructionRate,
        }))
    }

    pub(crate) fn finish(&mut self, tick: u64) -> Result<(), C220ScheduleError> {
        self.next_instruction_tick = tick.checked_add(1).ok_or(C220ScheduleError::TimeOverflow)?;
        Ok(())
    }
}
