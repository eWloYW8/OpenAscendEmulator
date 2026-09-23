use super::{
    C220FixpL1Output, C220FixpL1WriteInterface, C220FixpWritePipeline, C220FixpWritePipelineError,
    C220FixpWriteProgress,
};
use crate::sim::c220::mte::C220MteReadPayload;
use crate::sim::c220::mte::factor::C220FactorReadPacket;
use crate::sim::c220::mte::interface::{
    C220MteL1Interface, C220MteL1ReadOperation, C220MteL1ReadPort, C220MteOutputFragment,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220FixpDispatchPacket {
    Write(C220MteOutputFragment),
    FactorBatch {
        port: C220MteL1ReadPort,
        cursor: crate::sim::c220::mte::factor::C220FactorRequestCursor,
    },
    FactorRead {
        port: C220MteL1ReadPort,
        operation: C220MteL1ReadOperation<C220FactorReadPacket>,
    },
}

/// Factor reads and converted writes compete for one ordered dispatch queue.
pub type C220FixpDispatchPipeline = C220FixpWritePipeline<C220FixpDispatchPacket>;

impl C220FixpDispatchPipeline {
    pub fn enqueue_factor_batch(
        &mut self,
        tick: u64,
        port: C220MteL1ReadPort,
        cursor: crate::sim::c220::mte::factor::C220FactorRequestCursor,
    ) -> Result<bool, C220FixpWritePipelineError> {
        if cursor.remaining() == 0 {
            return Ok(false);
        }
        self.enqueue(tick, C220FixpDispatchPacket::FactorBatch { port, cursor })?;
        Ok(true)
    }

    pub fn generate_shared(
        &mut self,
        tick: u64,
    ) -> Result<C220FixpWriteProgress<C220FixpDispatchPacket>, C220FixpWritePipelineError> {
        self.generate_with(tick, |packet| match packet {
            C220FixpDispatchPacket::FactorBatch { port, mut cursor } => {
                let operation = cursor
                    .next()
                    .ok_or(C220FixpWritePipelineError::EmptyBatch)?;
                let remainder = (cursor.remaining() != 0)
                    .then_some(C220FixpDispatchPacket::FactorBatch { port, cursor });
                Ok((
                    C220FixpDispatchPacket::FactorRead { port, operation },
                    remainder,
                ))
            }
            packet => Ok((packet, None)),
        })
    }
    pub fn packetize_output(
        &mut self,
        tick: u64,
        output: &mut C220FixpL1Output,
    ) -> Result<Option<C220MteOutputFragment>, C220FixpWritePipelineError> {
        self.begin(tick, 0, "packetize")?;
        tick.checked_add(1)
            .ok_or(C220FixpWritePipelineError::Overflow)?;
        let packet = output.take_write(tick, true)?;
        if let Some(packet) = packet {
            self.enqueue(tick, C220FixpDispatchPacket::Write(packet))?;
        }
        Ok(packet)
    }

    pub fn send_shared(
        &mut self,
        tick: u64,
        writer: &mut C220FixpL1WriteInterface,
        reader: &mut C220MteL1Interface<C220MteReadPayload>,
    ) -> Result<C220FixpWriteProgress<C220FixpDispatchPacket>, C220FixpWritePipelineError> {
        self.send_with(tick, |packet| match packet {
            C220FixpDispatchPacket::FactorBatch { .. } => {
                Err(C220FixpWritePipelineError::UnexpandedBatch)
            }
            C220FixpDispatchPacket::Write(fragment) => {
                writer.enqueue(tick, fragment)?;
                Ok(true)
            }
            C220FixpDispatchPacket::FactorRead { port, operation } => Ok(reader
                .push(
                    tick,
                    port,
                    operation.map_payload(C220MteReadPayload::Factor),
                )?
                .is_some()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::c220::mte::factor::{C220FactorDescriptor, C220FactorLoad, C220FactorSource};
    use crate::sim::c220::mte::factor::c220_factor_l1_requests;
    use std::num::NonZeroU32;

    #[test]
    fn factor_batch_expansion_preserves_readiness_and_queue_credit() {
        let width = NonZeroU32::new(32).unwrap();
        let cursor = c220_factor_l1_requests(
            C220FactorLoad {
                source: C220FactorSource::L1,
                source_address: 0,
                destination_address: 0,
                descriptor: C220FactorDescriptor((2 << 16) | (1 << 4)),
            },
            7,
            width,
            width,
        );
        let port = C220MteL1ReadPort::Port2;
        let mut queue = C220FixpDispatchPipeline::default();
        assert!(queue.enqueue_factor_batch(0, port, cursor).unwrap());
        for tick in 1..=6 {
            let C220FixpWriteProgress::Advanced(C220FixpDispatchPacket::FactorRead {
                operation,
                ..
            }) = queue.generate_shared(tick).unwrap()
            else {
                panic!("expected factor read")
            };
            assert_eq!(operation.access.address, (tick - 1) * 32);
            assert!(!operation.last_in_instruction);
        }
        assert_eq!(queue.packets().len(), 1);
        let pending = queue.packets().front().copied().unwrap();
        assert_eq!(pending.ready_tick, 1);
        assert_eq!(
            queue.generate_shared(7).unwrap(),
            C220FixpWriteProgress::QueueFull
        );
        assert_eq!(queue.packets().front(), Some(&pending));
        let mut reader = C220MteL1Interface::default();
        let mut writer = C220FixpL1WriteInterface::default();
        for tick in 8..=9 {
            queue.send_shared(tick, &mut writer, &mut reader).unwrap();
            let C220FixpWriteProgress::Advanced(C220FixpDispatchPacket::FactorRead {
                operation,
                ..
            }) = queue.generate_shared(tick).unwrap()
            else {
                panic!("expected factor read")
            };
            assert_eq!(operation.access.address, (tick - 2) * 32);
            assert_eq!(operation.last_in_instruction, tick == 9);
        }
        assert!(queue.packets().is_empty());
    }

    #[test]
    fn blocked_factor_head_holds_later_write_in_shared_fifo() {
        let width = NonZeroU32::new(128).unwrap();
        let operation = c220_factor_l1_requests(
            C220FactorLoad {
                source: C220FactorSource::L1,
                source_address: 0,
                destination_address: 0,
                descriptor: C220FactorDescriptor((1 << 16) | (1 << 4)),
            },
            7,
            width,
            width,
        )
        .next()
        .unwrap();
        let port = C220MteL1ReadPort::Port2;
        let mut reader = C220MteL1Interface::default();
        for _ in 0..5 {
            assert!(
                reader
                    .push(0, port, operation.map_payload(C220MteReadPayload::Factor))
                    .unwrap()
                    .is_some()
            );
        }
        let mut writer = C220FixpL1WriteInterface::default();
        let mut queue = C220FixpDispatchPipeline::default();
        queue
            .enqueue(0, C220FixpDispatchPacket::FactorRead { port, operation })
            .unwrap();
        queue
            .enqueue(
                0,
                C220FixpDispatchPacket::Write(C220MteOutputFragment {
                    instruction_id: 8,
                    request_id: 10,
                    destination_address: 0,
                    bytes: 32,
                    last_in_uop: true,
                    last_in_instruction: true,
                }),
            )
            .unwrap();
        assert!(matches!(
            queue.generate(0).unwrap(),
            C220FixpWriteProgress::Delayed { ready_tick: 1 }
        ));
        queue.generate(1).unwrap();
        queue.generate(2).unwrap();
        assert_eq!(
            queue.send_shared(2, &mut writer, &mut reader).unwrap(),
            C220FixpWriteProgress::QueueFull
        );
        assert!(writer.is_idle());
        assert_eq!(queue.dispatch_queue().len(), 2);
        assert!(reader.send_request(4, true).unwrap().sent.is_some());
        assert!(matches!(
            queue.send_shared(4, &mut writer, &mut reader).unwrap(),
            C220FixpWriteProgress::Advanced(C220FixpDispatchPacket::FactorRead { .. })
        ));
        assert!(matches!(
            queue.send_shared(5, &mut writer, &mut reader).unwrap(),
            C220FixpWriteProgress::Advanced(C220FixpDispatchPacket::Write(_))
        ));
        assert!(queue.is_idle());
        assert!(!writer.is_idle());
    }
}
