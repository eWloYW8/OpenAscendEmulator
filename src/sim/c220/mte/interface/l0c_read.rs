use std::collections::VecDeque;

use crate::sim::c220::memory::{C220L0c, C220L0cError, C220L0cReadBlock, C220L0cReadRequest};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MteL0cReadOperation {
    /// Capture a 1024-byte functional snapshot when this read is accepted.
    pub begins_unit: bool,
    pub instruction_id: u64,
    pub uop_id: u32,
    pub conversion_mode: u32,
    pub last_in_instruction: bool,
    pub request: C220L0cReadRequest,
    /// Transfer size is independent of the request's unit-flag span.
    pub data_bytes: u32,
    pub destination_address: u64,
    pub output_bytes: u32,
    pub last_in_uop: bool,
    pub end_of_burst: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MteL0cReadAcknowledgment {
    pub operation: C220MteL0cReadOperation,
    pub accepted_tick: u64,
    pub data_ready_tick: u64,
    pub queue_ready_tick: u64,
    pub retry_ready_tick: Option<u64>,
}

impl C220MteL0cReadAcknowledgment {
    pub const fn ready_tick(self) -> u64 {
        let ready = if self.data_ready_tick > self.queue_ready_tick {
            self.data_ready_tick
        } else {
            self.queue_ready_tick
        };
        match self.retry_ready_tick {
            Some(retry) if retry > ready => retry,
            _ => ready,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220MteL0cReadDelivery {
    Accept,
    Retry,
    DeferUntil(u64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220MteL0cReadSend {
    Idle,
    InputLatency { ready_tick: u64 },
    Sent,
    PendingLimit,
    AcknowledgmentLimit,
    TransportFull,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MteL0cReadEntry {
    pub operation: C220MteL0cReadOperation,
    pub ready_tick: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220MteL0cReadResponse {
    Idle,
    Blocked(C220L0cReadBlock),
    Accepted(C220MteL0cReadAcknowledgment),
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum C220MteL0cReadError {
    #[error("L0C read interface time reversed from {previous} to {requested}")]
    TimeReversed { previous: u64, requested: u64 },
    #[error("L0C read interface {phase} callback already ran at tick {tick}")]
    RepeatedCallback { phase: &'static str, tick: u64 },
    #[error("L0C read request ID {0} is already pending")]
    DuplicateRequest(u32),
    #[error(transparent)]
    Memory(#[from] C220L0cError),
}

/// Input staging, egress and return path for MTE L0C read micro-operations.
/// The owner schedules admission, send, response, and FIXP delivery independently.
/// Direct `send` is available for operations whose input delay was modeled by
/// the caller; otherwise use `enqueue` followed by `send_queued`.
/// Response acceptance exposes the functional-read point; delivery only
/// removes an acknowledgment when the downstream FIXP accepts it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220MteL0cReadInterface {
    bank_count: u8,
    data_latency: u32,
    input: VecDeque<C220MteL0cReadEntry>,
    pending: VecDeque<C220MteL0cReadOperation>,
    acknowledgments: VecDeque<C220MteL0cReadAcknowledgment>,
    observed_tick: Option<u64>,
    callbacks: [Option<u64>; 4],
}

impl C220MteL0cReadInterface {
    pub fn new(bank_count: u8, data_latency: u32) -> Result<Self, C220MteL0cReadError> {
        if !(1..=32).contains(&bank_count) {
            return Err(C220L0cError::InvalidReadBankCount(bank_count).into());
        }
        Ok(Self {
            bank_count,
            data_latency,
            input: VecDeque::new(),
            pending: VecDeque::new(),
            acknowledgments: VecDeque::new(),
            observed_tick: None,
            callbacks: [None; 4],
        })
    }

    pub fn pending(&self) -> &VecDeque<C220MteL0cReadOperation> {
        &self.pending
    }

    pub fn acknowledgments(&self) -> &VecDeque<C220MteL0cReadAcknowledgment> {
        &self.acknowledgments
    }

    pub fn is_idle(&self) -> bool {
        self.input.is_empty() && self.pending.is_empty() && self.acknowledgments.is_empty()
    }

    pub fn input(&self) -> &VecDeque<C220MteL0cReadEntry> {
        &self.input
    }

    pub fn can_enqueue(&self) -> bool {
        self.input.len() < 5
    }

    /// Admission keeps credit charged for both delayed and ready input entries.
    pub fn enqueue(
        &mut self,
        tick: u64,
        operation: C220MteL0cReadOperation,
    ) -> Result<bool, C220MteL0cReadError> {
        self.begin_callback(tick, 3, "enqueue")?;
        if !self.can_enqueue() {
            return Ok(false);
        }
        if self
            .input
            .iter()
            .any(|entry| entry.operation.request.id == operation.request.id)
            || self
                .pending
                .iter()
                .any(|entry| entry.request.id == operation.request.id)
            || self
                .acknowledgments
                .iter()
                .any(|entry| entry.operation.request.id == operation.request.id)
        {
            return Err(C220MteL0cReadError::DuplicateRequest(operation.request.id));
        }
        let ready_tick = tick.checked_add(4).ok_or(C220L0cError::TimeOverflow)?;
        self.input.push_back(C220MteL0cReadEntry {
            operation,
            ready_tick,
        });
        Ok(true)
    }

    pub fn send_queued(
        &mut self,
        tick: u64,
        memory: &mut C220L0c,
    ) -> Result<C220MteL0cReadSend, C220MteL0cReadError> {
        let Some(entry) = self.input.front().copied() else {
            self.begin_callback(tick, 0, "send")?;
            return Ok(C220MteL0cReadSend::Idle);
        };
        if tick < entry.ready_tick {
            self.begin_callback(tick, 0, "send")?;
            return Ok(C220MteL0cReadSend::InputLatency {
                ready_tick: entry.ready_tick,
            });
        }
        let sent = self.send(tick, entry.operation, memory)?;
        if sent == C220MteL0cReadSend::Sent {
            self.input.pop_front();
        }
        Ok(sent)
    }

    pub fn send(
        &mut self,
        tick: u64,
        operation: C220MteL0cReadOperation,
        memory: &mut C220L0c,
    ) -> Result<C220MteL0cReadSend, C220MteL0cReadError> {
        self.begin_callback(tick, 0, "send")?;
        if self.pending.len() > 2 {
            return Ok(C220MteL0cReadSend::PendingLimit);
        }
        if self.acknowledgments.len() > self.data_latency as usize {
            return Ok(C220MteL0cReadSend::AcknowledgmentLimit);
        }
        if self
            .pending
            .iter()
            .any(|entry| entry.request.id == operation.request.id)
        {
            return Err(C220MteL0cReadError::DuplicateRequest(operation.request.id));
        }
        if !memory
            .read_port_mut()
            .send_request(tick, operation.request)?
        {
            return Ok(C220MteL0cReadSend::TransportFull);
        }
        self.pending.push_back(operation);
        Ok(C220MteL0cReadSend::Sent)
    }

    pub fn receive(
        &mut self,
        tick: u64,
        memory: &mut C220L0c,
    ) -> Result<C220MteL0cReadResponse, C220MteL0cReadError> {
        self.begin_callback(tick, 1, "receive")?;
        let Some(&operation) = self.pending.front() else {
            return Ok(C220MteL0cReadResponse::Idle);
        };
        let queue_ready_tick = tick.checked_add(1).ok_or(C220L0cError::TimeOverflow)?;
        let data_ready_tick = match memory.poll_read(
            tick,
            operation.request.id,
            self.bank_count,
            self.data_latency,
        )? {
            Ok(ready) => ready,
            Err(block) => return Ok(C220MteL0cReadResponse::Blocked(block)),
        };
        let acknowledgment = C220MteL0cReadAcknowledgment {
            operation,
            accepted_tick: tick,
            data_ready_tick,
            queue_ready_tick,
            retry_ready_tick: None,
        };
        self.pending.pop_front();
        self.acknowledgments.push_back(acknowledgment);
        Ok(C220MteL0cReadResponse::Accepted(acknowledgment))
    }

    pub fn deliver(
        &mut self,
        tick: u64,
        downstream_ready: bool,
    ) -> Result<Option<C220MteL0cReadAcknowledgment>, C220MteL0cReadError> {
        self.deliver_with::<C220MteL0cReadError>(tick, |_| {
            Ok(if downstream_ready {
                C220MteL0cReadDelivery::Accept
            } else {
                C220MteL0cReadDelivery::Retry
            })
        })
    }

    /// Invoke the receiver only for a ready head. Deferred retries retain
    /// their queue credit until the receiver accepts them.
    pub fn deliver_with<E: From<C220MteL0cReadError>>(
        &mut self,
        tick: u64,
        receiver: impl FnOnce(&C220MteL0cReadAcknowledgment) -> Result<C220MteL0cReadDelivery, E>,
    ) -> Result<Option<C220MteL0cReadAcknowledgment>, E> {
        self.begin_callback(tick, 2, "deliver")?;
        let Some(head) = self.acknowledgments.front_mut() else {
            return Ok(None);
        };
        if head.ready_tick() > tick {
            return Ok(None);
        }
        match receiver(head)? {
            C220MteL0cReadDelivery::Accept => Ok(self.acknowledgments.pop_front()),
            C220MteL0cReadDelivery::Retry => Ok(None),
            C220MteL0cReadDelivery::DeferUntil(ready) => {
                head.retry_ready_tick = Some(ready);
                Ok(None)
            }
        }
    }

    fn begin_callback(
        &mut self,
        tick: u64,
        index: usize,
        phase: &'static str,
    ) -> Result<(), C220MteL0cReadError> {
        if let Some(previous) = self.observed_tick
            && tick < previous
        {
            return Err(C220MteL0cReadError::TimeReversed {
                previous,
                requested: tick,
            });
        }
        if self.callbacks[index] == Some(tick) {
            return Err(C220MteL0cReadError::RepeatedCallback { phase, tick });
        }
        self.observed_tick = Some(tick);
        self.callbacks[index] = Some(tick);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::c220::cube::C220CubeL0cAccess;
    use crate::sim::c220::memory::C220L0cFragmentRequest;

    fn operation(id: u32, check_unit_flags: bool) -> C220MteL0cReadOperation {
        C220MteL0cReadOperation {
            begins_unit: true,
            instruction_id: 10,
            uop_id: id,
            conversion_mode: 0,
            last_in_instruction: true,
            data_bytes: 128,
            destination_address: 0,
            output_bytes: 128,
            last_in_uop: true,
            end_of_burst: true,
            request: C220L0cReadRequest {
                id,
                data_type: 0,
                half_accumulator: false,
                fragments: C220L0cFragmentRequest {
                    address: 0,
                    bytes: 1024,
                    access: C220CubeL0cAccess::Read,
                    check_unit_flags,
                    update_unit_flags: check_unit_flags,
                },
            },
        }
    }

    #[test]
    fn staged_input_keeps_credit_until_transport_acceptance() {
        let mut memory = C220L0c::new(131072, 12).unwrap();
        let mut interface = C220MteL0cReadInterface::new(32, 0).unwrap();
        for tick in 0..5 {
            assert!(
                interface
                    .enqueue(tick, operation(tick as u32, false))
                    .unwrap()
            );
        }
        assert!(!interface.can_enqueue());
        assert!(!interface.enqueue(5, operation(5, false)).unwrap());
        assert_eq!(
            interface.send_queued(5, &mut memory).unwrap(),
            C220MteL0cReadSend::Sent
        );
        assert!(interface.can_enqueue());
        interface.receive(6, &mut memory).unwrap();
        assert_eq!(
            interface.send_queued(6, &mut memory).unwrap(),
            C220MteL0cReadSend::AcknowledgmentLimit
        );
        assert_eq!(interface.input()[0].operation.request.id, 1);
        assert_eq!(interface.input().len(), 4);
        interface.deliver(7, true).unwrap();
        assert_eq!(
            interface.send_queued(7, &mut memory).unwrap(),
            C220MteL0cReadSend::Sent
        );
        let mut delayed = C220MteL0cReadInterface::new(32, 0).unwrap();
        delayed.enqueue(10, operation(10, false)).unwrap();
        assert_eq!(
            delayed.send_queued(13, &mut memory).unwrap(),
            C220MteL0cReadSend::InputLatency { ready_tick: 14 }
        );
        assert!(!delayed.is_idle());
    }

    #[test]
    fn acknowledgment_delay_and_downstream_backpressure_gate_sends() {
        let mut memory = C220L0c::new(131072, 12).unwrap();
        let mut interface = C220MteL0cReadInterface::new(32, 0).unwrap();
        assert_eq!(
            interface.send(0, operation(1, false), &mut memory).unwrap(),
            C220MteL0cReadSend::Sent
        );
        assert!(matches!(
            interface.receive(0, &mut memory).unwrap(),
            C220MteL0cReadResponse::Blocked(C220L0cReadBlock::AwaitingBankAdmission)
        ));
        let C220MteL0cReadResponse::Accepted(ack) = interface.receive(1, &mut memory).unwrap()
        else {
            panic!("read should be accepted")
        };
        assert_eq!((ack.data_ready_tick, ack.queue_ready_tick), (1, 2));
        assert_eq!(ack.operation.data_bytes, 128);
        assert_eq!(ack.operation.request.fragments.bytes, 1024);
        assert_eq!(interface.deliver(1, true).unwrap(), None);
        assert_eq!(
            interface.send(1, operation(2, false), &mut memory).unwrap(),
            C220MteL0cReadSend::AcknowledgmentLimit
        );
        assert_eq!(interface.deliver(2, false).unwrap(), None);
        assert_eq!(
            interface.send(2, operation(2, false), &mut memory).unwrap(),
            C220MteL0cReadSend::AcknowledgmentLimit
        );
        assert_eq!(interface.deliver(3, true).unwrap(), Some(ack));
        assert!(interface.is_idle());
        assert_eq!(
            interface.send(3, operation(2, false), &mut memory).unwrap(),
            C220MteL0cReadSend::Sent
        );
    }

    #[test]
    fn receiver_can_defer_a_rejected_ack_without_releasing_credit() {
        let mut memory = C220L0c::new(131072, 12).unwrap();
        let mut interface = C220MteL0cReadInterface::new(32, 0).unwrap();
        interface.send(0, operation(1, false), &mut memory).unwrap();
        interface.receive(1, &mut memory).unwrap();
        assert_eq!(
            interface
                .deliver_with::<C220MteL0cReadError>(2, |_| Ok(C220MteL0cReadDelivery::DeferUntil(
                    6
                )))
                .unwrap(),
            None
        );
        assert_eq!(interface.acknowledgments()[0].ready_tick(), 6);
        assert_eq!(interface.acknowledgments()[0].data_ready_tick, 1);
        assert_eq!(
            interface.send(3, operation(2, false), &mut memory).unwrap(),
            C220MteL0cReadSend::AcknowledgmentLimit
        );
        assert_eq!(
            interface
                .deliver_with::<C220MteL0cReadError>(5, |_| panic!(
                    "receiver called before retry deadline"
                ))
                .unwrap(),
            None
        );
        assert!(interface.deliver(6, true).unwrap().is_some());
        assert!(interface.is_idle());
    }

    #[test]
    fn unit_flag_stalls_keep_pending_credits_until_response() {
        let mut memory = C220L0c::new(131072, 12).unwrap();
        let mut interface = C220MteL0cReadInterface::new(32, 5).unwrap();
        for tick in 0..3 {
            if tick != 0 {
                assert!(matches!(
                    interface.receive(tick, &mut memory).unwrap(),
                    C220MteL0cReadResponse::Blocked(C220L0cReadBlock::UnitFlag(_))
                ));
            }
            assert_eq!(
                interface
                    .send(tick, operation(tick as u32 + 1, true), &mut memory)
                    .unwrap(),
                C220MteL0cReadSend::Sent
            );
        }
        assert_eq!(
            interface.send(3, operation(4, true), &mut memory).unwrap(),
            C220MteL0cReadSend::PendingLimit
        );
        assert_eq!(interface.pending().len(), 3);
        assert!(interface.acknowledgments().is_empty());
    }
}
