use crate::isa::c310::buffer::{C310BufferOperation, C310BufferStep};
use crate::isa::flow::{DcciStep, DsbStep, PipelineBarrierStep};
use crate::sim::c310::buffer::{
    C310BufferAdmissionState, C310BufferCounterError, C310BufferDisposition, C310GetBufAdmission,
    C310GetBufDispatch, C310ReleaseAdmission,
};
use crate::sim::c310::predicate_buffer::{C310PushPbDisposition, C310PushPbStep};
use crate::sim::c310::scalar::C310ScalarBus;
use crate::sim::c310::vector_queue::{C310VfQueueDisposition, C310VfQueueStep};
use crate::sim::common::scalar::ScalarMemoryBus;
use thiserror::Error;

pub trait C310BufferPipeSink {
    fn try_enqueue(&mut self, pipe_code: u8, step: C310BufferStep) -> bool;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C310BufferAdmissionRoute {
    DirectGet(C310GetBufDispatch),
    QueuedGet(C310GetBufDispatch),
    QueuedRelease,
    NoPipeRelease,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C310BufferAdmissionRecord {
    pub step: C310BufferStep,
    pub route: C310BufferAdmissionRoute,
}

#[derive(Debug, Error)]
pub enum C310BufferBusError<E: std::error::Error + 'static> {
    #[error(transparent)]
    Buffer(#[from] C310BufferCounterError),
    #[error("wrapped scalar memory bus: {0}")]
    Inner(#[source] E),
}

#[derive(Debug)]
pub struct C310BufferAdmissionBus<B, P> {
    inner: B,
    pipes: P,
    admission: C310BufferAdmissionState,
    accepted: Vec<C310BufferAdmissionRecord>,
}

impl<B, P> C310BufferAdmissionBus<B, P> {
    pub const fn new(inner: B, pipes: P, admission: C310BufferAdmissionState) -> Self {
        Self {
            inner,
            pipes,
            admission,
            accepted: Vec::new(),
        }
    }

    pub const fn inner(&self) -> &B {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut B {
        &mut self.inner
    }

    pub const fn pipes(&self) -> &P {
        &self.pipes
    }

    pub fn pipes_mut(&mut self) -> &mut P {
        &mut self.pipes
    }

    pub const fn admission(&self) -> &C310BufferAdmissionState {
        &self.admission
    }

    pub fn admission_mut(&mut self) -> &mut C310BufferAdmissionState {
        &mut self.admission
    }

    pub fn accepted(&self) -> &[C310BufferAdmissionRecord] {
        &self.accepted
    }

    pub fn into_parts(
        self,
    ) -> (
        B,
        P,
        C310BufferAdmissionState,
        Vec<C310BufferAdmissionRecord>,
    ) {
        (self.inner, self.pipes, self.admission, self.accepted)
    }
}

impl<B: ScalarMemoryBus, P: C310BufferPipeSink> ScalarMemoryBus for C310BufferAdmissionBus<B, P> {
    type Error = C310BufferBusError<B::Error>;

    fn read(&mut self, address: u64, destination: &mut [u8]) -> Result<(), Self::Error> {
        self.inner
            .read(address, destination)
            .map_err(C310BufferBusError::Inner)
    }

    fn write(&mut self, address: u64, source: &[u8]) -> Result<(), Self::Error> {
        self.inner
            .write(address, source)
            .map_err(C310BufferBusError::Inner)
    }

    fn maintain_data_cache(&mut self, step: DcciStep) -> Result<bool, Self::Error> {
        self.inner
            .maintain_data_cache(step)
            .map_err(C310BufferBusError::Inner)
    }

    fn synchronize_pipeline(&mut self, step: DsbStep) -> Result<bool, Self::Error> {
        self.inner
            .synchronize_pipeline(step)
            .map_err(C310BufferBusError::Inner)
    }

    fn synchronize_barrier(&mut self, step: PipelineBarrierStep) -> Result<bool, Self::Error> {
        self.inner
            .synchronize_barrier(step)
            .map_err(C310BufferBusError::Inner)
    }
}

impl<B: C310ScalarBus, P: C310BufferPipeSink> C310ScalarBus for C310BufferAdmissionBus<B, P> {
    fn execute_buffer(
        &mut self,
        step: C310BufferStep,
    ) -> Result<C310BufferDisposition, Self::Error> {
        let id = step.buffer_id;
        let pipe_code = step.instruction.pipe_code;
        let route = match step.instruction.operation {
            C310BufferOperation::Get => match self.admission.admit_get(
                id,
                pipe_code,
                step.instruction.mode_field,
                || self.pipes.try_enqueue(pipe_code, step),
            )? {
                C310GetBufAdmission::RingFull | C310GetBufAdmission::PipeStalled => {
                    return Ok(C310BufferDisposition::Stalled);
                }
                C310GetBufAdmission::Direct(dispatch) => {
                    C310BufferAdmissionRoute::DirectGet(dispatch)
                }
                C310GetBufAdmission::Queued(dispatch) => {
                    C310BufferAdmissionRoute::QueuedGet(dispatch)
                }
            },
            C310BufferOperation::Release => {
                match self
                    .admission
                    .admit_release(id, pipe_code, || self.pipes.try_enqueue(pipe_code, step))?
                {
                    C310ReleaseAdmission::NoPipe => C310BufferAdmissionRoute::NoPipeRelease,
                    C310ReleaseAdmission::PipeStalled => return Ok(C310BufferDisposition::Stalled),
                    C310ReleaseAdmission::Queued => C310BufferAdmissionRoute::QueuedRelease,
                }
            }
        };
        self.accepted
            .push(C310BufferAdmissionRecord { step, route });
        Ok(C310BufferDisposition::Accepted)
    }

    fn execute_push_pb(
        &mut self,
        step: C310PushPbStep,
    ) -> Result<C310PushPbDisposition, Self::Error> {
        self.inner
            .execute_push_pb(step)
            .map_err(C310BufferBusError::Inner)
    }

    fn enqueue_vf(&mut self, step: C310VfQueueStep) -> Result<C310VfQueueDisposition, Self::Error> {
        self.inner
            .enqueue_vf(step)
            .map_err(C310BufferBusError::Inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::architecture::Architecture;
    use crate::sim::c310::scalar::{
        C310ScalarExecutionError, C310ScalarInstructionStep, C310ScalarStepper,
    };
    use crate::sim::common::scalar::ScalarMachine;
    use std::collections::VecDeque;
    use std::convert::Infallible;

    struct UnusedMemory;

    impl ScalarMemoryBus for UnusedMemory {
        type Error = Infallible;

        fn read(&mut self, _address: u64, _destination: &mut [u8]) -> Result<(), Self::Error> {
            unreachable!("test instructions do not read memory")
        }

        fn write(&mut self, _address: u64, _source: &[u8]) -> Result<(), Self::Error> {
            unreachable!("test instructions do not write memory")
        }
    }

    impl C310ScalarBus for UnusedMemory {}

    #[derive(Debug)]
    struct OneEntryPipe {
        entries: VecDeque<C310BufferStep>,
        attempts: usize,
    }

    impl C310BufferPipeSink for OneEntryPipe {
        fn try_enqueue(&mut self, pipe_code: u8, step: C310BufferStep) -> bool {
            assert_eq!(pipe_code, 4);
            self.attempts += 1;
            if !self.entries.is_empty() {
                return false;
            }
            self.entries.push_back(step);
            true
        }
    }

    fn bus(
        ring_size: u32,
        direct_release_enabled: bool,
    ) -> C310BufferAdmissionBus<UnusedMemory, OneEntryPipe> {
        C310BufferAdmissionBus::new(
            UnusedMemory,
            OneEntryPipe {
                entries: VecDeque::new(),
                attempts: 0,
            },
            C310BufferAdmissionState::new(1, ring_size, direct_release_enabled).unwrap(),
        )
    }

    fn core() -> C310ScalarStepper {
        C310ScalarStepper::new(
            ScalarMachine::from_pem_initial_state(Architecture::Dav3510),
            0x1000,
        )
    }

    #[test]
    fn core_stepper_retries_pipe_stall_without_advancing_or_duplicating_admission() {
        let mut core = core();
        let mut bus = bus(4, false);
        let get = 0x4200_1000;
        let first = core.step_word(get, &mut bus).unwrap();
        assert!(matches!(
            first.instruction,
            C310ScalarInstructionStep::Buffer(_)
        ));
        assert_eq!(core.pc(), 0x1004);
        assert!(matches!(
            core.step_word(get, &mut bus),
            Err(C310ScalarExecutionError::BufferStalled { pc: 0x1004, .. })
        ));
        assert_eq!(core.pc(), 0x1004);
        assert_eq!(bus.accepted().len(), 1);
        assert_eq!(
            bus.admission()
                .counters()
                .counter(0)
                .unwrap()
                .dispatch_count,
            1
        );
        assert_eq!(bus.pipes().attempts, 2);

        bus.pipes_mut().entries.pop_front().unwrap();
        let second = core.step_word(get, &mut bus).unwrap();
        assert_eq!(second.pc, 0x1004);
        assert_eq!(core.pc(), 0x1008);
        assert_eq!(bus.accepted().len(), 2);
        assert!(matches!(
            bus.accepted()[1].route,
            C310BufferAdmissionRoute::QueuedGet(C310GetBufDispatch {
                expected_release_count: 1,
                assigned_dispatch_count: 2
            })
        ));
    }

    #[test]
    fn ring_stall_precedes_pipe_attempt_and_preserves_pc() {
        let mut core = core();
        let mut bus = bus(2, false);
        core.step_word(0x4200_1000, &mut bus).unwrap();
        assert_eq!(bus.pipes().attempts, 1);
        assert!(matches!(
            core.step_word(0x4200_1000, &mut bus),
            Err(C310ScalarExecutionError::BufferStalled { .. })
        ));
        assert_eq!(bus.pipes().attempts, 1);
        assert_eq!(core.pc(), 0x1004);
    }

    #[test]
    fn matching_release_enables_direct_get_without_pipe_capacity() {
        let mut core = core();
        let mut bus = bus(4, true);
        core.step_word(0x4220_1000, &mut bus).unwrap();
        assert_eq!(bus.pipes().entries.len(), 1);
        let get = core.step_word(0x4200_1000, &mut bus).unwrap();
        assert_eq!(get.pc, 0x1004);
        assert_eq!(bus.pipes().attempts, 1);
        assert_eq!(bus.accepted().len(), 2);
        assert_eq!(
            bus.accepted()[0].route,
            C310BufferAdmissionRoute::QueuedRelease
        );
        assert!(matches!(
            bus.accepted()[1].route,
            C310BufferAdmissionRoute::DirectGet(_)
        ));
        assert_eq!(
            bus.admission()
                .counters()
                .counter(0)
                .unwrap()
                .outstanding_get_bufs,
            0
        );
    }

    #[test]
    fn zero_pipe_release_bypasses_sink_and_invalid_pipe_preserves_state() {
        let mut core = core();
        let mut bus = bus(4, false);
        core.step_word(0x4220_0000, &mut bus).unwrap();
        assert_eq!(
            bus.accepted()[0].route,
            C310BufferAdmissionRoute::NoPipeRelease
        );
        assert_eq!(bus.pipes().attempts, 0);
        assert!(matches!(
            core.step_word(0x4200_1800, &mut bus),
            Err(C310ScalarExecutionError::BufferBackend(_))
        ));
        assert_eq!(core.pc(), 0x1004);
        assert_eq!(bus.accepted().len(), 1);
        assert_eq!(bus.pipes().attempts, 0);
    }
}
