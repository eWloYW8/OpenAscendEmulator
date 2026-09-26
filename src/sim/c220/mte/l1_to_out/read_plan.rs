use std::num::NonZeroU32;

use crate::isa::c220::mte::l1_to_out::C220MovL1ToOutTransfer;
use crate::sim::c220::mte::uop::C220DmaUopMode;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220L1OutputRoute {
    PerBurst,
    Contiguous,
    Gather,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220L1OutputRead {
    pub source_address: u64,
    pub bytes: u32,
    pub destination_address: u64,
    pub destination_bytes: u32,
    pub last_in_transaction: bool,
    pub last_in_instruction: bool,
}

impl C220L1OutputRead {
    pub fn l1_operation(
        self,
        instruction_id: u64,
    ) -> crate::sim::c220::mte::interface::C220MteL1ReadOperation<Self> {
        use crate::sim::c220::memory::l1::C220L1Access;
        use crate::sim::c220::mte::interface::{
            C220MteL1OutputDestination, C220MteL1ReadOperation,
        };
        C220MteL1ReadOperation {
            instruction_id,
            access: C220L1Access {
                address: self.source_address,
                bytes: self.bytes,
            },
            destination: C220MteL1OutputDestination::External,
            output_address: self.destination_address,
            output_bytes: self.destination_bytes,
            output_bandwidth: NonZeroU32::MAX,
            completes_logical_uop: self.last_in_transaction,
            last_in_instruction: self.last_in_instruction,
            payload: self,
        }
    }

    pub fn output_fragment(
        self,
        instruction_id: u64,
        request_id: u64,
    ) -> crate::sim::c220::mte::interface::C220MteOutputFragment {
        crate::sim::c220::mte::interface::C220MteOutputFragment {
            instruction_id,
            request_id,
            destination_address: self.destination_address,
            bytes: self.destination_bytes,
            last_in_uop: self.last_in_transaction,
            last_in_instruction: self.last_in_instruction,
        }
    }
}

/// Lazy L1 reads grouped by their external write transaction.
#[derive(Debug, Clone)]
pub struct C220L1OutputReadPlan {
    transfer: C220MovL1ToOutTransfer,
    mode: C220DmaUopMode,
    bandwidth: NonZeroU32,
    route: C220L1OutputRoute,
    groups: u32,
    group_bytes: u32,
    source_stride: u32,
    destination_stride: u64,
    group: u32,
    offset: u32,
    transaction_bytes: u32,
    read_offset: u32,
    gather_source_offset: u32,
}

impl C220L1OutputReadPlan {
    pub fn new(
        transfer: C220MovL1ToOutTransfer,
        mode: C220DmaUopMode,
        bandwidth: NonZeroU32,
    ) -> Self {
        let count = ((transfer.xm >> 4) & 4095) as u32;
        let length = (transfer.xm >> 16) as u16;
        let source_gap = (transfer.xm >> 32) as u16;
        let destination_gap = (transfer.xm >> 48) as u16;
        let route = if source_gap == 0 && destination_gap == 0 {
            C220L1OutputRoute::Contiguous
        } else if transfer.source_address.is_multiple_of(32)
            && transfer.destination_address.is_multiple_of(64)
            && source_gap != 0
            && destination_gap == 0
            && count > 1
            && length == 2
        {
            C220L1OutputRoute::Gather
        } else {
            C220L1OutputRoute::PerBurst
        };
        let flatten = route != C220L1OutputRoute::PerBurst;
        Self {
            transfer,
            mode,
            bandwidth,
            route,
            groups: if count == 0 || length == 0 {
                0
            } else if flatten {
                1
            } else {
                count
            },
            group_bytes: (u32::from(length) * 32).wrapping_mul(if flatten { count } else { 1 }),
            source_stride: (u32::from(length) + u32::from(source_gap)) * 32,
            destination_stride: (u64::from(length) + u64::from(destination_gap)) * 32,
            group: 0,
            offset: 0,
            transaction_bytes: 0,
            read_offset: 0,
            gather_source_offset: 0,
        }
    }

    pub const fn route(&self) -> C220L1OutputRoute {
        self.route
    }
}

impl Iterator for C220L1OutputReadPlan {
    type Item = C220L1OutputRead;

    fn next(&mut self) -> Option<Self::Item> {
        if self.group >= self.groups || self.group_bytes == 0 {
            return None;
        }
        let destination_address = self
            .transfer
            .destination_address
            .wrapping_add(u64::from(self.group) * self.destination_stride)
            .wrapping_add(u64::from(self.offset));
        if self.transaction_bytes == 0 {
            self.transaction_bytes = self
                .mode
                .split_bytes(destination_address, self.group_bytes - self.offset);
        }
        let (source_offset, bytes) = if self.route == C220L1OutputRoute::Gather {
            let source_offset = self.gather_source_offset;
            self.gather_source_offset = self.gather_source_offset.wrapping_add(self.source_stride);
            (u64::from(source_offset), 64)
        } else {
            (
                u64::from(self.group.wrapping_mul(self.source_stride))
                    + u64::from(self.offset)
                    + u64::from(self.read_offset),
                (self.transaction_bytes - self.read_offset).min(self.bandwidth.get()),
            )
        };
        self.read_offset += bytes;
        let last_in_transaction = self.read_offset == self.transaction_bytes;
        let read = C220L1OutputRead {
            source_address: self.transfer.source_address.wrapping_add(source_offset),
            bytes,
            destination_address,
            destination_bytes: self.transaction_bytes,
            last_in_transaction,
            last_in_instruction: last_in_transaction
                && self.offset + self.transaction_bytes == self.group_bytes
                && self.group + 1 == self.groups,
        };
        if last_in_transaction {
            self.offset += self.transaction_bytes;
            self.transaction_bytes = 0;
            self.read_offset = 0;
            if self.offset == self.group_bytes {
                self.group += 1;
                self.offset = 0;
            }
        }
        Some(read)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::c220::mte::l1_to_out::C220MovL1ToOutInstruction;

    #[test]
    fn l1_responses_forward_whole_transactions_with_shared_head_backpressure() {
        use super::super::{C220L1OutputCommand, C220L1OutputEngine, C220L1OutputEngineConfig};
        use crate::sim::c220::memory::l1::{C220L1Geometry, C220L1Port, C220L1Transport};
        use crate::sim::c220::mte::fixp::{
            C220FixpAdmission, C220FixpReadProgress, C220FixpStoreBuffer,
        };
        use crate::sim::c220::mte::interface::{
            C220MteL1Interface, C220MteL1OutputCredits,
            biu_write::command::{C220BiuWriteCommands, C220BiuWriteConfig},
            biu_write::cube::C220BiuCubeWriteSource,
            biu_write::data::C220BiuWriteDataPort,
        };

        let instruction =
            C220MovL1ToOutInstruction::decode((3 << 29) | (2 << 27) | (4 << 23) | (2 << 3))
                .unwrap();
        let transfer = C220MovL1ToOutTransfer {
            instruction,
            source_address: 0,
            destination_address: 4096,
            xm: (1 << 4) | (16 << 16),
        };
        let config = C220L1OutputEngineConfig {
            read_bandwidth: NonZeroU32::new(32).unwrap(),
            instruction_fifo_depth: 1,
            write_outstanding_limit: 1,
        };
        let command = C220L1OutputCommand {
            transfer,
            control: 0,
            mode: C220DmaUopMode::Wide512,
        };
        let mut engine = C220L1OutputEngine::new(config);
        assert_eq!(
            engine.admit(0, 7, command).unwrap(),
            C220FixpAdmission::Active
        );
        assert_eq!(
            engine.admit(0, 8, command).unwrap(),
            C220FixpAdmission::ReadGenerationBusy
        );
        let mut independent = C220L1OutputEngine::new(config);
        assert_eq!(
            independent.admit(0, 8, command).unwrap(),
            C220FixpAdmission::Active
        );
        let mut stores = C220FixpStoreBuffer::default();
        let mut biu = C220BiuWriteCommands::new(C220BiuWriteConfig {
            outstanding: NonZeroU32::new(1).unwrap(),
            weights: [1; 3],
            source_bandwidth: config.read_bandwidth,
        });
        let mut interface = C220MteL1Interface::default();
        let mut memory = C220L1Transport::new(C220L1Geometry::new(32, 1, 1, 0).unwrap());
        let mut responses = 0;
        let mut forwarded = Vec::new();
        let mut blocked = false;
        for tick in 0..200 {
            let generated = engine.generate_read(tick).unwrap();
            if tick < 2 {
                assert_eq!(generated, C220FixpReadProgress::Delayed { ready_tick: 2 });
            }
            engine
                .send_read(tick, &stores, &mut interface, |read| read)
                .unwrap();
            memory.advance(tick).unwrap();
            let sent = interface
                .send_request(tick, memory.request_ready(C220L1Port::MteRead))
                .unwrap();
            if let Some(request) = sent.sent {
                assert!(
                    memory
                        .send_request(tick, C220L1Port::MteRead, request.l1_request())
                        .unwrap()
                );
            }
            if let Some(response) = memory.receive_response(tick, C220L1Port::MteRead).unwrap() {
                let request = interface
                    .receive_response(tick, Some(response.request.id))
                    .unwrap()
                    .unwrap();
                responses += 1;
                if !request.operation.completes_logical_uop {
                    assert_eq!(interface.queue_state().output.acknowledged, 0);
                } else {
                    assert!(interface.external_head(tick).is_none());
                    assert_eq!(interface.acknowledgment_ready_tick(), Some(tick + 1));
                }
            }
            let accepted = if let Some(head) = interface.external_head(tick) {
                blocked |= tick < 100;
                tick >= 100 && engine.receive_source(tick, &stores, head).unwrap()
            } else {
                false
            };
            let sent = interface
                .send_output(
                    tick,
                    C220MteL1OutputCredits {
                        external: accepted,
                        ..Default::default()
                    },
                )
                .unwrap();
            if let Some(transfer) = sent.sent {
                forwarded.push(transfer.fragment);
            }
            assert_eq!(interface.queue_state().output.output_fragments, 0);
            assert!(interface.retire(tick).unwrap().is_none());
            engine.packetize(tick, &mut stores).unwrap();
            engine.generate_write(tick).unwrap();
            engine.send_write(tick, &mut biu).unwrap();
            biu.advance(tick).unwrap();
        }
        assert!(blocked);
        assert_eq!(responses, 16);
        assert!(memory.is_idle());
        assert!(interface.is_idle());
        assert!(engine.read_pipeline().is_idle());
        assert!(engine.write_pipeline().is_idle());
        assert!(engine.output().bursts().is_empty());
        assert!(!engine.is_idle());
        assert!(engine.instruction_fifo().is_empty());
        assert_eq!(engine.commands()[&7].source_completed_tick, Some(100));
        assert_eq!(engine.commands()[&7].write_dispatched_tick, Some(103));
        assert!(engine.retire(7).is_err());
        assert_eq!(forwarded.len(), 1);
        assert_eq!(forwarded[0].bytes, 512);
        assert!(forwarded[0].last_in_instruction);
        let request = biu.take_request(200).unwrap().unwrap().command;
        let tag = request.tag;
        let mut source = C220BiuCubeWriteSource::default();
        source
            .register(request.source_request(), request.input.store_token.unwrap())
            .unwrap();
        biu.mark_dbid(tag);
        source.receive_dbid(200, tag).unwrap();
        source
            .ingress(201, |tag| {
                biu.begin_source(tag);
                true
            })
            .unwrap();
        let mut port = C220BiuWriteDataPort::default();
        for tick in 202..=205 {
            let ready = source.egress(tick, &mut stores).unwrap();
            assert_eq!(ready.is_some(), tick == 205);
        }
        assert!(stores.is_empty());
        let ready = source.take_data_ready(206).unwrap().unwrap();
        port.send(206, [Some(ready), None, None]).unwrap();
        port.take_request(207).unwrap().unwrap();
        let response = port.receive_response(208, tag).unwrap();
        source.release_response(208, tag).unwrap();
        biu.release_tag(tag).unwrap();
        assert_eq!(engine.complete_response(response).unwrap(), Some(7));
        assert_eq!(engine.retire(7).unwrap().response_tick, Some(208));
        assert!(engine.is_idle());
    }

    #[test]
    fn transaction_splits_read_bandwidth_and_gather_rows() {
        let instruction =
            C220MovL1ToOutInstruction::decode((3 << 29) | (2 << 27) | (4 << 23) | (2 << 3))
                .unwrap();
        let mut transfer = C220MovL1ToOutTransfer {
            instruction,
            source_address: 0,
            destination_address: 0x1000,
            xm: (3 << 4) | (2 << 16) | (1 << 32),
        };
        let width = NonZeroU32::new(32).unwrap();
        let plan = C220L1OutputReadPlan::new(transfer, C220DmaUopMode::Wide512, width);
        assert_eq!(plan.route(), C220L1OutputRoute::Gather);
        let reads: Vec<_> = plan.collect();
        assert_eq!(
            reads
                .iter()
                .map(|r| (
                    r.source_address,
                    r.bytes,
                    r.destination_bytes,
                    r.last_in_transaction
                ))
                .collect::<Vec<_>>(),
            [
                (0, 64, 128, false),
                (96, 64, 128, true),
                (192, 64, 64, true)
            ]
        );
        assert!(reads.last().unwrap().last_in_instruction);
        transfer.xm = (2 << 4) | (8 << 16);
        transfer.destination_address = 0x1070;
        let plan = C220L1OutputReadPlan::new(transfer, C220DmaUopMode::Wide512, width);
        assert_eq!(plan.route(), C220L1OutputRoute::Contiguous);
        let reads: Vec<_> = plan.collect();
        assert_eq!(reads[0].bytes, 16);
        assert_eq!(reads[1].source_address, 16);
        assert_eq!(reads[1].destination_bytes, 128);
        assert_eq!(reads.iter().map(|r| r.bytes).sum::<u32>(), 512);
        assert_eq!(reads.iter().filter(|r| r.last_in_instruction).count(), 1);
        {
            use crate::sim::c220::mte::fixp::{
                C220FixpDispatchPipeline, C220FixpExternalOutput, C220FixpStoreBuffer,
            };
            let mut output = C220FixpExternalOutput::default();
            let mut dispatch = C220FixpDispatchPipeline::default();
            let mut stores = C220FixpStoreBuffer::default();
            let transaction = C220MovL1ToOutTransfer {
                xm: (1 << 4) | (32 << 16),
                destination_address: 0x1000,
                ..transfer
            };
            let plan = C220L1OutputReadPlan::new(transaction, C220DmaUopMode::Unbounded, width);
            for (index, read) in plan.enumerate() {
                let tick = index as u64;
                let fragment = read.output_fragment(9, tick);
                assert!(
                    !output
                        .receive_l1_source(tick, tick, fragment, false)
                        .unwrap()
                );
                assert!(
                    !output
                        .receive_l1_source(tick, tick + 1, fragment, true)
                        .unwrap()
                );
                assert!(
                    output
                        .receive_l1_source(tick, tick, fragment, true)
                        .unwrap()
                );
                assert_eq!(output.bursts().len(), usize::from(read.last_in_transaction));
            }
            assert!(
                dispatch
                    .packetize_l1_source(31, &mut output, &mut stores, C220DmaUopMode::Unbounded)
                    .unwrap()
                    .is_none()
            );
            let packet = dispatch
                .packetize_l1_source(32, &mut output, &mut stores, C220DmaUopMode::Unbounded)
                .unwrap()
                .unwrap();
            assert_eq!(packet.write.fragment.bytes, 1024);
            assert_eq!(packet.write.fragment.instruction_id, 9);
            assert!(packet.write.fragment.last_in_instruction);
            assert_eq!(stores.len(), 1);
            assert!(output.bursts().is_empty());
            assert_eq!(dispatch.packets().len(), 1);
        }
        transfer.xm |= 1 << 48;
        assert_eq!(
            C220L1OutputReadPlan::new(transfer, C220DmaUopMode::Wide512, width).route(),
            C220L1OutputRoute::PerBurst
        );
        transfer.xm = 0;
        assert!(
            C220L1OutputReadPlan::new(transfer, C220DmaUopMode::Wide512, width)
                .next()
                .is_none()
        );
    }
}
