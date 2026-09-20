use crate::buffer_c310::C310BufferDisposition;
use crate::flow::{C310BufferStep, DcciStep, DsbStep, PipelineBarrierStep};
use crate::machine::ScalarMemoryBus;
use crate::predicate_buffer_c310::{
    C310PredicateBuffer, C310PredicateBufferError, C310PredicateBufferVfIssue,
    C310PredicateBufferWrite, C310PushPbDisposition, C310PushPbStep,
};
use crate::vec_queue_c310::{C310VfQueueDisposition, C310VfQueueStep};
use serde::Serialize;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum C310PredicateBufferBusError<E: std::error::Error + 'static> {
    #[error(transparent)]
    PredicateBuffer(#[from] C310PredicateBufferError),
    #[error("wrapped scalar memory bus: {0}")]
    Inner(#[source] E),
    #[error("predicate-buffer slot {slot_id} has no outstanding vector issue")]
    NoOutstandingVfIssue { slot_id: u16 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct C310PredicateBufferQueuedVfIssue {
    pub queue: C310VfQueueStep,
    pub predicate_buffer: C310PredicateBufferVfIssue,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct C310PredicateBufferCompletedVfIssue {
    pub issue: C310PredicateBufferQueuedVfIssue,
}

#[derive(Debug)]
pub struct C310PredicateBufferBus<B> {
    inner: B,
    predicate_buffer: C310PredicateBuffer,
    writes: Vec<C310PredicateBufferWrite>,
    vf_issues: Vec<C310PredicateBufferQueuedVfIssue>,
    outstanding_vf_issues: Vec<C310PredicateBufferQueuedVfIssue>,
    completed_vf_issues: Vec<C310PredicateBufferCompletedVfIssue>,
}

impl<B> C310PredicateBufferBus<B> {
    pub const fn new(inner: B, predicate_buffer: C310PredicateBuffer) -> Self {
        Self {
            inner,
            predicate_buffer,
            writes: Vec::new(),
            vf_issues: Vec::new(),
            outstanding_vf_issues: Vec::new(),
            completed_vf_issues: Vec::new(),
        }
    }

    pub const fn inner(&self) -> &B {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut B {
        &mut self.inner
    }

    pub const fn predicate_buffer(&self) -> &C310PredicateBuffer {
        &self.predicate_buffer
    }

    pub fn predicate_buffer_mut(&mut self) -> &mut C310PredicateBuffer {
        &mut self.predicate_buffer
    }

    pub fn writes(&self) -> &[C310PredicateBufferWrite] {
        &self.writes
    }

    pub fn vf_issues(&self) -> &[C310PredicateBufferQueuedVfIssue] {
        &self.vf_issues
    }

    pub fn outstanding_vf_issues(&self) -> &[C310PredicateBufferQueuedVfIssue] {
        &self.outstanding_vf_issues
    }

    pub fn completed_vf_issues(&self) -> &[C310PredicateBufferCompletedVfIssue] {
        &self.completed_vf_issues
    }

    pub fn complete_vf_issue(
        &mut self,
        slot_id: u16,
    ) -> Result<C310PredicateBufferCompletedVfIssue, C310PredicateBufferBusError<B::Error>>
    where
        B: ScalarMemoryBus,
    {
        let position = self
            .outstanding_vf_issues
            .iter()
            .position(|issue| issue.predicate_buffer.slot_id == slot_id)
            .ok_or(C310PredicateBufferBusError::NoOutstandingVfIssue { slot_id })?;
        self.predicate_buffer.recycle_slot(slot_id)?;
        let completion = C310PredicateBufferCompletedVfIssue {
            issue: self.outstanding_vf_issues.remove(position),
        };
        self.completed_vf_issues.push(completion);
        Ok(completion)
    }

    pub fn into_parts(self) -> (B, C310PredicateBuffer, Vec<C310PredicateBufferWrite>) {
        (self.inner, self.predicate_buffer, self.writes)
    }
}

impl<B: ScalarMemoryBus> ScalarMemoryBus for C310PredicateBufferBus<B> {
    type Error = C310PredicateBufferBusError<B::Error>;

    fn read(&mut self, address: u64, destination: &mut [u8]) -> Result<(), Self::Error> {
        self.inner
            .read(address, destination)
            .map_err(C310PredicateBufferBusError::Inner)
    }

    fn write(&mut self, address: u64, source: &[u8]) -> Result<(), Self::Error> {
        self.inner
            .write(address, source)
            .map_err(C310PredicateBufferBusError::Inner)
    }

    fn maintain_data_cache(&mut self, step: DcciStep) -> Result<bool, Self::Error> {
        self.inner
            .maintain_data_cache(step)
            .map_err(C310PredicateBufferBusError::Inner)
    }

    fn synchronize_pipeline(&mut self, step: DsbStep) -> Result<bool, Self::Error> {
        self.inner
            .synchronize_pipeline(step)
            .map_err(C310PredicateBufferBusError::Inner)
    }

    fn synchronize_barrier(&mut self, step: PipelineBarrierStep) -> Result<bool, Self::Error> {
        self.inner
            .synchronize_barrier(step)
            .map_err(C310PredicateBufferBusError::Inner)
    }

    fn execute_c310_buffer(
        &mut self,
        step: C310BufferStep,
    ) -> Result<C310BufferDisposition, Self::Error> {
        self.inner
            .execute_c310_buffer(step)
            .map_err(C310PredicateBufferBusError::Inner)
    }

    fn execute_c310_push_pb(
        &mut self,
        step: C310PushPbStep,
    ) -> Result<C310PushPbDisposition, Self::Error> {
        let Some(write) = self.predicate_buffer.push(step)? else {
            return Ok(C310PushPbDisposition::Stalled);
        };
        self.writes.push(write);
        Ok(C310PushPbDisposition::Accepted)
    }

    fn enqueue_c310_vf(
        &mut self,
        step: C310VfQueueStep,
    ) -> Result<C310VfQueueDisposition, Self::Error> {
        if self.predicate_buffer.active_slot().is_none() {
            return Err(C310PredicateBufferError::NoActiveSlot.into());
        }
        let disposition = self
            .inner
            .enqueue_c310_vf(step)
            .map_err(C310PredicateBufferBusError::Inner)?;
        if disposition == C310VfQueueDisposition::Accepted {
            let issue = C310PredicateBufferQueuedVfIssue {
                queue: step,
                predicate_buffer: self.predicate_buffer.advance_for_vf_issue()?,
            };
            self.vf_issues.push(issue);
            self.outstanding_vf_issues.push(issue);
        }
        Ok(disposition)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::architecture::Architecture;
    use crate::machine::{ScalarInstructionError, ScalarInstructionStep, ScalarMachine};
    use crate::rvec_pb_c310::project_c310_pb_rvec_scalar_init;
    use crate::stepper::ScalarStepper;
    use std::convert::Infallible;

    struct UnusedMemory;

    impl ScalarMemoryBus for UnusedMemory {
        type Error = Infallible;

        fn read(&mut self, _address: u64, _destination: &mut [u8]) -> Result<(), Self::Error> {
            unreachable!("the test instruction does not access memory")
        }

        fn write(&mut self, _address: u64, _source: &[u8]) -> Result<(), Self::Error> {
            unreachable!("the test instruction does not access memory")
        }
    }

    struct VfQueueMemory {
        disposition: C310VfQueueDisposition,
        steps: Vec<C310VfQueueStep>,
    }

    impl ScalarMemoryBus for VfQueueMemory {
        type Error = Infallible;

        fn read(&mut self, _address: u64, _destination: &mut [u8]) -> Result<(), Self::Error> {
            unreachable!("the test instruction does not access memory")
        }

        fn write(&mut self, _address: u64, _source: &[u8]) -> Result<(), Self::Error> {
            unreachable!("the test instruction does not access memory")
        }

        fn enqueue_c310_vf(
            &mut self,
            step: C310VfQueueStep,
        ) -> Result<C310VfQueueDisposition, Self::Error> {
            self.steps.push(step);
            Ok(self.disposition)
        }
    }

    fn mul_machine() -> ScalarMachine {
        let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav3510);
        machine.set_xreg(12, 0x0000_0100_0000_003e).unwrap();
        machine.set_xreg(23, 1).unwrap();
        machine.set_xreg(2, 0x0000_0100_0000_0080).unwrap();
        machine
    }

    #[test]
    fn scalar_push_commits_live_register_values_to_a_slot() {
        let mut stepper = ScalarStepper::new(mul_machine(), 0x10d0_d6fc);
        let mut bus =
            C310PredicateBufferBus::new(UnusedMemory, C310PredicateBuffer::with_default_slots());
        let result = stepper.step_word(0x4319_7108, &mut bus).unwrap();
        assert!(matches!(
            result.instruction,
            ScalarInstructionStep::PushPb(_)
        ));
        assert_eq!(stepper.pc(), 0x10d0_d700);
        assert_eq!(bus.writes().len(), 1);
        let slot = bus.predicate_buffer().read_slot(0).unwrap();
        assert_eq!(slot[..8], 0x0000_0100_0000_003e_u64.to_le_bytes());
        assert_eq!(slot[8..16], 1_u64.to_le_bytes());
        assert_eq!(slot[16..24], 0x0000_0100_0000_0080_u64.to_le_bytes());
        assert_eq!(slot[24..32], 0x0000_0100_0000_0080_u64.to_le_bytes());
        let projection = project_c310_pb_rvec_scalar_init(&slot);
        assert_eq!(projection.big_flags, 0x3e);
        assert_eq!(projection.consumed_payload_words, 5);
        assert_eq!(projection.writes.len(), 24);
    }

    #[test]
    fn a_full_buffer_stalls_without_advancing_the_scalar_machine() {
        let mut stepper = ScalarStepper::new(mul_machine(), 0x10d0_d6fc);
        let mut bus =
            C310PredicateBufferBus::new(UnusedMemory, C310PredicateBuffer::new(1).unwrap());
        stepper.step_word(0x4319_7108, &mut bus).unwrap();
        let before = stepper.clone();
        assert!(matches!(
            stepper.step_word(0x4319_7108, &mut bus),
            Err(ScalarInstructionError::PushPbStalled {
                pc: 0x10d0_d700,
                word: 0x4319_7108,
            })
        ));
        assert_eq!(stepper, before);
        assert_eq!(bus.writes().len(), 1);
    }

    #[test]
    fn accepted_vf_issue_advances_then_completion_recycles_its_slot() {
        let inner = VfQueueMemory {
            disposition: C310VfQueueDisposition::Accepted,
            steps: Vec::new(),
        };
        let mut bus = C310PredicateBufferBus::new(inner, C310PredicateBuffer::new(3).unwrap());
        let push = C310PushPbStep {
            pc: 0x10d0_d6fc,
            instruction: crate::predicate_buffer_c310::C310PushPbInstruction::decode(
                Architecture::Dav3510,
                0x4319_7108,
            )
            .unwrap(),
            source_values: [1, 2, 3, 4],
            bytes: [0x5a; crate::predicate_buffer_c310::C310_PB_PUSH_BYTES],
        };
        assert_eq!(
            bus.execute_c310_push_pb(push).unwrap(),
            C310PushPbDisposition::Accepted
        );
        let queue = C310VfQueueStep {
            pc: 0x10d0_d718,
            instruction: crate::vec_queue_c310::C310VfQueueInstruction::decode(
                Architecture::Dav3510,
                0x1542_0000,
                0x15e0_0125,
            )
            .unwrap(),
            vector_pc: 0x10d0_d900,
        };
        assert_eq!(
            bus.enqueue_c310_vf(queue).unwrap(),
            C310VfQueueDisposition::Accepted
        );
        assert_eq!(bus.inner().steps, [queue]);
        assert_eq!(bus.vf_issues().len(), 1);
        assert_eq!(bus.vf_issues()[0].queue, queue);
        assert_eq!(bus.vf_issues()[0].predicate_buffer.slot_id, 0);
        assert_eq!(
            bus.vf_issues()[0].predicate_buffer.next_available_slot,
            Some(1)
        );
        assert_eq!(bus.predicate_buffer().halfword_cursor(), 0);
        assert_eq!(bus.outstanding_vf_issues(), bus.vf_issues());

        let completion = bus.complete_vf_issue(0).unwrap();
        assert_eq!(completion.issue, bus.vf_issues()[0]);
        assert!(bus.outstanding_vf_issues().is_empty());
        assert_eq!(bus.completed_vf_issues(), [completion]);
        assert_eq!(bus.predicate_buffer().occupied_slots(), 0);
        assert!(matches!(
            bus.complete_vf_issue(0),
            Err(C310PredicateBufferBusError::NoOutstandingVfIssue { slot_id: 0 })
        ));
    }

    #[test]
    fn rejected_vf_issue_does_not_advance_predicate_buffer() {
        for disposition in [
            C310VfQueueDisposition::Stalled,
            C310VfQueueDisposition::Unsupported,
        ] {
            let inner = VfQueueMemory {
                disposition,
                steps: Vec::new(),
            };
            let mut buffer = C310PredicateBuffer::new(2).unwrap();
            let instruction = crate::predicate_buffer_c310::C310PushPbInstruction::decode(
                Architecture::Dav3510,
                0x4319_7108,
            )
            .unwrap();
            buffer
                .push(instruction.from_source_values(0x1000, [1, 2, 3, 4]))
                .unwrap();
            let before = buffer.clone();
            let mut bus = C310PredicateBufferBus::new(inner, buffer);
            let queue = C310VfQueueStep {
                pc: 0x1010,
                instruction: crate::vec_queue_c310::C310VfQueueInstruction::decode(
                    Architecture::Dav3510,
                    0x1542_0000,
                    0x15e0_0125,
                )
                .unwrap(),
                vector_pc: 0x2000,
            };
            assert_eq!(bus.enqueue_c310_vf(queue).unwrap(), disposition);
            assert_eq!(bus.predicate_buffer(), &before);
            assert!(bus.vf_issues().is_empty());
            assert!(bus.outstanding_vf_issues().is_empty());
        }
    }

    #[test]
    fn vf_issue_without_an_active_slot_never_reaches_inner_queue() {
        let inner = VfQueueMemory {
            disposition: C310VfQueueDisposition::Accepted,
            steps: Vec::new(),
        };
        let mut bus = C310PredicateBufferBus::new(inner, C310PredicateBuffer::new(2).unwrap());
        let queue = C310VfQueueStep {
            pc: 0x1010,
            instruction: crate::vec_queue_c310::C310VfQueueInstruction::decode(
                Architecture::Dav3510,
                0x1542_0000,
                0x15e0_0125,
            )
            .unwrap(),
            vector_pc: 0x2000,
        };
        assert!(matches!(
            bus.enqueue_c310_vf(queue),
            Err(C310PredicateBufferBusError::PredicateBuffer(
                C310PredicateBufferError::NoActiveSlot
            ))
        ));
        assert!(bus.inner().steps.is_empty());
    }
}
