use super::{
    C220FixpBiuWrite, C220FixpExternalOutput, C220FixpL1Output, C220FixpL1WriteInterface,
    C220FixpStoreBuffer, C220FixpWritePipeline, C220FixpWritePipelineError, C220FixpWriteProgress,
};
use crate::sim::c220::mte::C220MteReadPayload;
use crate::sim::c220::mte::factor::C220FactorReadPacket;
use crate::sim::c220::mte::interface::{
    C220MteL1Interface, C220MteL1ReadOperation, C220MteL1ReadPort, C220MteOutputFragment,
};
use crate::sim::c220::mte::uop::C220DmaUopMode;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220FixpDispatchPacket {
    Write(C220MteOutputFragment),
    External(super::C220FixpBiuWrite),
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
    pub fn packetize_l1_source(
        &mut self,
        tick: u64,
        output: &mut C220FixpExternalOutput,
        stores: &mut C220FixpStoreBuffer,
        mode: C220DmaUopMode,
    ) -> Result<Option<C220FixpBiuWrite>, C220FixpWritePipelineError> {
        self.begin(tick, 0, "packetize")?;
        tick.checked_add(1)
            .ok_or(C220FixpWritePipelineError::Overflow)?;
        let packet = output
            .take_l1_source_write(tick, true, stores)?
            .map(|write| C220FixpBiuWrite { write, mode });
        if let Some(packet) = packet {
            self.enqueue(tick, C220FixpDispatchPacket::External(packet))?;
        }
        Ok(packet)
    }

    pub fn packetize_l1_shared(
        &mut self,
        tick: u64,
        output: &mut C220FixpExternalOutput,
    ) -> Result<Option<C220MteOutputFragment>, C220FixpWritePipelineError> {
        self.begin(tick, 0, "packetize")?;
        tick.checked_add(1)
            .ok_or(C220FixpWritePipelineError::Overflow)?;
        let packet = output.take_l1_write(tick, true)?;
        if let Some(packet) = packet {
            self.enqueue(tick, C220FixpDispatchPacket::Write(packet))?;
        }
        Ok(packet)
    }

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
        biu: Option<
            &mut crate::sim::c220::mte::interface::biu_write::command::C220BiuWriteCommands,
        >,
    ) -> Result<C220FixpWriteProgress<C220FixpDispatchPacket>, C220FixpWritePipelineError> {
        let mut biu = biu;
        self.send_with(tick, |packet| match packet {
            C220FixpDispatchPacket::External(packet) => {
                use crate::sim::c220::mte::interface::{
                    biu_read::C220BiuSubcore, biu_write::command::C220BiuWriteInput,
                };
                let commands = biu
                    .as_mut()
                    .ok_or(C220FixpWritePipelineError::BiuDisconnected)?;
                if !commands.can_push(C220BiuSubcore::Cube) {
                    return Ok(false);
                }
                Ok(commands.push(
                    tick,
                    C220BiuWriteInput::from_fixp(packet.write, packet.mode, tick),
                )?)
            }
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

impl C220FixpDispatchPipeline {
    /// Enqueue the optional second channel before the primary packet, with
    /// the same queue-ready tick. The returned receipt identifies the primary.
    pub fn packetize_external(
        &mut self,
        tick: u64,
        output: &mut C220FixpExternalOutput,
        stores: &mut C220FixpStoreBuffer,
        mode: C220DmaUopMode,
    ) -> Result<Option<C220FixpBiuWrite>, C220FixpWritePipelineError> {
        self.begin(tick, 0, "packetize")?;
        tick.checked_add(1)
            .ok_or(C220FixpWritePipelineError::Overflow)?;
        let batch = output.take_writes(tick, true, stores)?;
        if let Some(write) = batch.and_then(|batch| batch.second_channel) {
            self.enqueue(
                tick,
                C220FixpDispatchPacket::External(C220FixpBiuWrite { write, mode }),
            )?;
        }
        let fragment = batch.map(|batch| C220FixpBiuWrite {
            write: batch.primary,
            mode,
        });
        if let Some(fragment) = fragment {
            self.enqueue(tick, C220FixpDispatchPacket::External(fragment))?;
        }
        Ok(fragment)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::c220::mte::factor::{C220FactorDescriptor, C220FactorLoad, C220FactorSource};
    use crate::sim::c220::mte::factor::c220_factor_l1_requests;
    use std::num::NonZeroU32;

    #[test]
    fn external_backpressure_blocks_younger_l1_write() {
        use crate::sim::c220::mte::interface::biu_write::command::{
            C220BiuWriteCommands, C220BiuWriteConfig, C220BiuWriteInput,
        };
        let mut stores = super::super::C220FixpStoreBuffer::default();
        let fragment = C220MteOutputFragment {
            instruction_id: 1,
            request_id: 1,
            destination_address: 4096,
            bytes: 32,
            last_in_uop: true,
            last_in_instruction: true,
        };
        let external = super::super::C220FixpBiuWrite {
            write: stores.publish(fragment),
            mode: crate::sim::c220::mte::uop::C220DmaUopMode::Wide512,
        };
        let mut commands = C220BiuWriteCommands::new(C220BiuWriteConfig {
            outstanding: NonZeroU32::new(1).unwrap(),
            weights: [1; 3],
            source_bandwidth: NonZeroU32::new(32).unwrap(),
        });
        for _ in 0..4 {
            assert!(
                commands
                    .push(
                        0,
                        C220BiuWriteInput::from_fixp(external.write, external.mode, 0)
                    )
                    .unwrap()
            );
        }
        let mut queue = C220FixpDispatchPipeline::default();
        queue
            .enqueue(0, C220FixpDispatchPacket::External(external))
            .unwrap();
        queue
            .enqueue(
                0,
                C220FixpDispatchPacket::Write(C220MteOutputFragment {
                    instruction_id: 2,
                    ..fragment
                }),
            )
            .unwrap();
        queue.generate_shared(1).unwrap();
        queue.generate_shared(2).unwrap();
        let mut writer = C220FixpL1WriteInterface::default();
        let mut reader = C220MteL1Interface::default();
        assert!(matches!(
            queue.send_shared(2, &mut writer, &mut reader, None),
            Err(C220FixpWritePipelineError::BiuDisconnected)
        ));
        assert_eq!(
            queue
                .send_shared(3, &mut writer, &mut reader, Some(&mut commands))
                .unwrap(),
            C220FixpWriteProgress::QueueFull
        );
        assert!(writer.is_idle());
        assert_eq!(queue.dispatch_queue().len(), 2);
        commands.advance(4).unwrap();
        assert_eq!(
            queue
                .send_shared(4, &mut writer, &mut reader, Some(&mut commands))
                .unwrap(),
            C220FixpWriteProgress::Advanced(C220FixpDispatchPacket::External(external))
        );
        assert!(matches!(
            queue
                .send_shared(5, &mut writer, &mut reader, Some(&mut commands))
                .unwrap(),
            C220FixpWriteProgress::Advanced(C220FixpDispatchPacket::Write(_))
        ));
        assert!(queue.is_idle());
        assert!(!writer.is_idle());
        assert_eq!(stores.len(), 1);
    }

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
            queue
                .send_shared(tick, &mut writer, &mut reader, None)
                .unwrap();
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
            queue
                .send_shared(2, &mut writer, &mut reader, None)
                .unwrap(),
            C220FixpWriteProgress::QueueFull
        );
        assert!(writer.is_idle());
        assert_eq!(queue.dispatch_queue().len(), 2);
        assert!(reader.send_request(4, true).unwrap().sent.is_some());
        assert!(matches!(
            queue
                .send_shared(4, &mut writer, &mut reader, None)
                .unwrap(),
            C220FixpWriteProgress::Advanced(C220FixpDispatchPacket::FactorRead { .. })
        ));
        assert!(matches!(
            queue
                .send_shared(5, &mut writer, &mut reader, None)
                .unwrap(),
            C220FixpWriteProgress::Advanced(C220FixpDispatchPacket::Write(_))
        ));
        assert!(queue.is_idle());
        assert!(!writer.is_idle());
    }
}
