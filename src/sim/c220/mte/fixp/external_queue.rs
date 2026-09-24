use std::collections::VecDeque;
#[cfg(test)]
use std::num::NonZeroU32;

use super::nz2nd::{C220FixpNz2ndStaging, C220FixpNz2ndStagingEntry, C220FixpNz2ndStagingError};
use super::{
    C220FixpConversionEntry, C220FixpConversionError, C220FixpConversionPipeline,
    C220FixpExternalOutputPolicy, C220FixpStoreBuffer, C220FixpStoreWrite,
};
use crate::sim::c220::mte::interface::C220MteOutputFragment;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_columns_publish_two_ordered_packets_except_partial_tail() {
        use crate::isa::c220::mte::fixp::C220FixpDescriptor;
        use crate::sim::c220::{
            memory::C220L0c,
            mte::{
                fixp::{
                    C220FixpCommand, C220FixpDispatchPipeline, C220FixpReadGenerator,
                    C220FixpSourceFormat,
                },
                interface::C220MteL0cReadInterface,
                uop::C220DmaUopMode,
            },
        };
        for columns in [16u16, 24, 32] {
            let command = C220FixpCommand {
                descriptor: C220FixpDescriptor {
                    xt: (33 << 32) | (1 << 16) | (u64::from(columns) << 4),
                    xm: (1 << 42) | 16,
                    nd: 0,
                },
                source_format: C220FixpSourceFormat::Fp32,
                source_address: 0,
                destination_address: 4096,
                control: 0,
                scalar_slope: 0,
                scalar_dequant: 0,
                slope_base_block: 0,
                dequant_base_block: 0,
            };
            let mut memory = C220L0c::new(131072, 12).unwrap();
            let mut input = C220MteL0cReadInterface::new(32, 0).unwrap();
            let mut conversion = C220FixpConversionPipeline::default();
            let mut output = C220FixpExternalOutput::default();
            let mut stores = C220FixpStoreBuffer::default();
            let mut writes = C220FixpDispatchPipeline::default();
            for (index, read) in C220FixpReadGenerator::new(command, 9, 1, 256)
                .unwrap()
                .enumerate()
            {
                let tick = index as u64 * 10;
                input.send(tick, read.operation, &mut memory).unwrap();
                input.receive(tick + 1, &mut memory).unwrap();
                conversion.receive(tick + 2, &mut input, false).unwrap();
                assert!(
                    output
                        .receive_columns(
                            tick + 6,
                            &mut conversion,
                            C220FixpExternalOutputPolicy::new(0, 0)
                        )
                        .unwrap()
                        .is_some()
                );
                assert!(
                    writes
                        .packetize_external(
                            tick + 6,
                            &mut output,
                            &mut stores,
                            C220DmaUopMode::Wide512
                        )
                        .unwrap()
                        .is_none()
                );
                writes
                    .packetize_external(tick + 7, &mut output, &mut stores, C220DmaUopMode::Wide512)
                    .unwrap()
                    .unwrap();
            }
            let packets: Vec<_> = writes
                .packets()
                .iter()
                .map(|entry| match entry.fragment {
                    super::super::C220FixpDispatchPacket::External(packet) => packet.write,
                    _ => panic!("expected external write"),
                })
                .collect();
            let addresses: Vec<_> = packets
                .iter()
                .map(|write| write.fragment.destination_address)
                .collect();
            assert_eq!(
                addresses,
                match columns {
                    16 => vec![5120, 4096],
                    24 => vec![5120, 4096, 6208],
                    _ => vec![5120, 4096, 7232, 6208],
                }
            );
            assert_eq!(stores.len(), packets.len());
            for (index, write) in packets.iter().enumerate() {
                assert_eq!(write.token.get(), index as u32 + 1);
                assert_eq!(write.fragment.bytes, 32);
                assert_eq!(
                    write.fragment.last_in_instruction,
                    index + 1 == packets.len()
                );
            }
            assert!(output.bursts().is_empty());
        }
    }

    #[test]
    fn cube_packets_drain_through_shared_pipeline_after_dbid_and_store_beats() {
        use crate::sim::c220::memory::l1::C220L1Geometry;
        use crate::sim::c220::mte::{
            interface::biu_read::C220BiuSubcore,
            interface::biu_write::command::C220BiuWriteConfig,
            mte1::frontend::C220Mte1ReadBandwidths,
            pipeline::{C220MtePipeline, C220MtePipelineConfig, C220MtePipelineEvent},
            set2d::C220Set2dBandwidths,
            uop::C220DmaUopMode,
        };
        let width = NonZeroU32::new(32).unwrap();
        let mut pipeline = C220MtePipeline::new(
            0,
            C220MtePipelineConfig {
                core_kind: crate::sim::c220::device::C220CoreKind::Cube,
                l1: C220L1Geometry::new(32, 1, 1, 0).unwrap(),
                read_width: width,
                output_bandwidths: C220Mte1ReadBandwidths {
                    l0a: width,
                    l0b: width,
                    bt: width,
                },
                set2d_bandwidths: C220Set2dBandwidths {
                    l0a: width,
                    l0b: width,
                    l1: width,
                },
            },
        );
        pipeline
            .connect_fixp_biu(C220BiuWriteConfig {
                outstanding: NonZeroU32::new(1).unwrap(),
                weights: [1; 3],
                source_bandwidth: width,
            })
            .unwrap();
        let mut output = C220FixpExternalOutput::default();
        output.bursts.push_back(C220FixpExternalBurst {
            instruction_id: 91,
            request_id: 7,
            address: 512,
            bytes: 512,
            closed: true,
            last_in_instruction: true,
            gather: false,
            row_bytes: 512,
            row_offset: 0,
            second_channel_offset: None,
            ready_tick: 0,
            policy: C220FixpExternalOutputPolicy::new(0, 512),
        });
        let mut writes = crate::sim::c220::mte::fixp::C220FixpDispatchPipeline::default();
        let write = pipeline
            .packetize_fixp_biu_output(&mut output, &mut writes, C220DmaUopMode::Wide512)
            .unwrap()
            .unwrap();
        let mut tag = None;
        let mut completed = false;
        for tick in 0..24 {
            pipeline.advance(tick).unwrap();
            writes.generate(tick).unwrap();
            let sent = pipeline.send_fixp_biu_output(&mut writes).unwrap();
            assert_eq!(
                matches!(
                    sent,
                    crate::sim::c220::mte::fixp::C220FixpWriteProgress::Advanced(_)
                ),
                tick == 2
            );
            assert!(
                !pipeline
                    .last_events()
                    .iter()
                    .any(|event| matches!(event, C220MtePipelineEvent::UbReadSent(..)))
            );
            if let Some(command) = pipeline.take_biu_write_command().unwrap() {
                assert_eq!(tick, 7);
                assert_eq!(command.command.input.store_token, Some(write.write.token));
                assert!(tag.replace(command.command.tag).is_none());
            }
            if tick == 10 {
                assert!(
                    pipeline
                        .receive_biu_write_dbid(C220BiuSubcore::Vector0, tag.unwrap())
                        .is_err()
                );
                pipeline
                    .receive_biu_write_dbid(C220BiuSubcore::Cube, tag.unwrap())
                    .unwrap();
                assert!(pipeline.receive_biu_write_response(tag.unwrap()).is_err());
            }
            assert_eq!(pipeline.fixp_store_buffer().len(), usize::from(tick < 15));
            if let Some(data) = pipeline.take_biu_write_data().unwrap() {
                assert_eq!(tick, 17);
                assert_eq!(data.subcore, C220BiuSubcore::Cube);
                assert!(!pipeline.is_idle());
                let response = pipeline.receive_biu_write_response(tag.unwrap()).unwrap();
                assert_eq!(response.retired_instruction(), Some(91));
                assert_eq!(pipeline.fixp_completions(), [91]);
                assert!(pipeline.is_idle());
                assert!(writes.is_idle());
                completed = true;
                break;
            }
        }
        assert!(completed);
    }

    #[test]
    fn row_packets_preserve_holes_and_gather_respects_alignment_and_control() {
        use crate::sim::c220::mte::uop::C220DmaUopMode;

        for control in [0, 1, 3, 5, 7] {
            let native = C220FixpExternalOutputPolicy::new(control, 1024);
            let downstream = C220DmaUopMode::from_mode_word(control);
            for address in [0, 16, 64, 128, 256, 384, 496, 512] {
                for bytes in [1, 16, 127, 128, 129, 256, 384, 512, 700] {
                    let packet = native.row_packet(address, bytes);
                    assert_eq!(downstream.split_bytes(address, packet), packet);
                }
            }
        }

        let policy = C220FixpExternalOutputPolicy {
            burst_sizes: [512, 256, 32].map(|n| NonZeroU32::new(n).unwrap()),
            burst_control: 0,
            row_stride_bytes: 1024,
        };
        let mut output = C220FixpExternalOutput::default();
        let mut stores = C220FixpStoreBuffer::default();
        output.bursts.push_back(C220FixpExternalBurst {
            instruction_id: 3,
            request_id: 7,
            address: 16,
            bytes: 80,
            closed: true,
            last_in_instruction: true,
            gather: false,
            row_bytes: 40,
            row_offset: 0,
            second_channel_offset: None,
            policy,
            ready_tick: 1,
        });
        assert!(output.take_writes(0, true, &mut stores).unwrap().is_none());
        assert!(output.take_writes(1, false, &mut stores).unwrap().is_none());
        assert!(stores.is_empty());
        for (index, (address, bytes)) in [(16, 16), (32, 24), (1040, 16), (1056, 24)]
            .into_iter()
            .enumerate()
        {
            let write = output
                .take_writes(index as u64 + 2, true, &mut stores)
                .unwrap()
                .unwrap();
            assert_eq!(write.primary.token.get(), index as u32 + 1);
            let fragment = write.primary.fragment;
            assert_eq!(
                (fragment.destination_address, fragment.bytes),
                (address, bytes)
            );
            assert_eq!(fragment.last_in_instruction, index == 3);
        }
        assert!(output.bursts().is_empty());
        assert_eq!(policy.gathered_packet(0, 128, false), None);
        assert_eq!(policy.gathered_packet(0, 128, true), Some(32));
        assert_eq!(policy.gathered_packet(16, 8, true), Some(8));
        assert_eq!(policy.gathered_packet(0, 512, false), Some(512));
        for (control, expected) in [(0, 512), (3, 256), (5, 32), (7, 700)] {
            let restricted = C220FixpExternalOutputPolicy {
                burst_control: control,
                ..policy
            };
            assert_eq!(restricted.row_packet(0, 700), expected);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FixpExternalBurst {
    pub instruction_id: u64,
    pub request_id: u64,
    pub address: u64,
    pub bytes: u32,
    pub closed: bool,
    pub last_in_instruction: bool,
    pub gather: bool,
    pub row_bytes: u32,
    pub row_offset: u32,
    pub second_channel_offset: Option<u64>,
    pub policy: C220FixpExternalOutputPolicy,
    pub ready_tick: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum C220FixpExternalOutputError {
    #[error(transparent)]
    Conversion(#[from] C220FixpConversionError),
    #[error(transparent)]
    Staging(#[from] C220FixpNz2ndStagingError),
    #[error("FIX external output time reversed from {previous} to {requested}")]
    TimeReversed { previous: u64, requested: u64 },
    #[error("FIX external output packet callback repeated at tick {0}")]
    RepeatedPacket(u64),
    #[error("FIX external output time overflow")]
    TimeOverflow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FixpExternalWriteBatch {
    pub second_channel: Option<C220FixpStoreWrite>,
    pub primary: C220FixpStoreWrite,
}

/// Shared L1/external packetization for column and NZ2ND output. Address
/// holes between rows are retained; downstream transaction credit and write
/// acknowledgments remain the responsibility of the external interface.
#[derive(Debug, Clone, Default)]
pub struct C220FixpExternalOutput {
    bursts: VecDeque<C220FixpExternalBurst>,
    observed_tick: Option<u64>,
    packet_tick: Option<u64>,
}

impl C220FixpExternalOutput {
    pub fn take_l1_write(
        &mut self,
        tick: u64,
        destination_ready: bool,
    ) -> Result<Option<C220MteOutputFragment>, C220FixpExternalOutputError> {
        self.observe(tick)?;
        if self.packet_tick == Some(tick) {
            return Err(C220FixpExternalOutputError::RepeatedPacket(tick));
        }
        self.packet_tick = Some(tick);
        let Some(head) = self.bursts.front_mut() else {
            return Ok(None);
        };
        if !destination_ready || head.ready_tick > tick || (!head.closed && head.bytes < 256) {
            return Ok(None);
        }
        let bytes = head.bytes.min(256);
        let finished = head.closed && head.bytes == bytes;
        let fragment = C220MteOutputFragment {
            instruction_id: head.instruction_id,
            request_id: head.request_id,
            destination_address: head.address,
            bytes,
            last_in_uop: true,
            last_in_instruction: finished && head.last_in_instruction,
        };
        head.bytes -= bytes;
        head.address = head.address.wrapping_add(u64::from(bytes));
        if finished {
            self.bursts.pop_front();
        }
        Ok(Some(fragment))
    }

    pub fn receive_columns(
        &mut self,
        tick: u64,
        conversion: &mut C220FixpConversionPipeline,
        policy: C220FixpExternalOutputPolicy,
    ) -> Result<Option<C220FixpConversionEntry>, C220FixpExternalOutputError> {
        self.observe(tick)?;
        let ready_tick = tick
            .checked_add(1)
            .ok_or(C220FixpExternalOutputError::TimeOverflow)?;
        let entry = conversion.take_ready(tick, self.can_receive())?;
        if let Some(entry) = entry {
            let op = entry.acknowledgment.operation;
            if op.last_in_uop {
                self.append(C220FixpExternalBurst {
                    instruction_id: op.instruction_id,
                    request_id: u64::from(op.request.id),
                    address: op.destination_address,
                    bytes: op.output_bytes,
                    closed: op.end_of_burst,
                    last_in_instruction: op.last_in_instruction,
                    gather: true,
                    row_bytes: 0,
                    row_offset: 0,
                    second_channel_offset: op.second_channel_offset,
                    policy,
                    ready_tick,
                });
            }
        }
        Ok(entry)
    }

    fn append(&mut self, burst: C220FixpExternalBurst) {
        if let Some(tail) = self.bursts.back_mut().filter(|tail| !tail.closed) {
            tail.bytes = tail.bytes.wrapping_add(burst.bytes);
            tail.closed = burst.closed;
            tail.last_in_instruction = burst.last_in_instruction;
        } else {
            self.bursts.push_back(burst);
        }
    }
    pub fn bursts(&self) -> &VecDeque<C220FixpExternalBurst> {
        &self.bursts
    }
    pub fn can_receive(&self) -> bool {
        self.bursts.len() <= 2
    }

    pub fn receive(
        &mut self,
        tick: u64,
        staging: &mut C220FixpNz2ndStaging,
        policy: C220FixpExternalOutputPolicy,
    ) -> Result<Option<C220FixpNz2ndStagingEntry>, C220FixpExternalOutputError> {
        self.observe(tick)?;
        let ready_tick = tick
            .checked_add(1)
            .ok_or(C220FixpExternalOutputError::TimeOverflow)?;
        let entry = staging.take_ready(tick, self.can_receive())?;
        if let Some(entry) = entry {
            let op = entry.operation;
            let d = op.descriptor;
            if d.last_in_uop {
                self.append(C220FixpExternalBurst {
                    instruction_id: op.instruction_id,
                    request_id: u64::from(op.request_id),
                    address: d.destination_address,
                    bytes: d.bytes,
                    closed: d.end_of_burst,
                    last_in_instruction: op.last_in_instruction,
                    gather: d.gather,
                    row_bytes: d.burst_bytes,
                    row_offset: 0,
                    second_channel_offset: None,
                    policy,
                    ready_tick,
                });
            }
        }
        Ok(entry)
    }

    pub fn take_writes(
        &mut self,
        tick: u64,
        destination_ready: bool,
        stores: &mut C220FixpStoreBuffer,
    ) -> Result<Option<C220FixpExternalWriteBatch>, C220FixpExternalOutputError> {
        self.observe(tick)?;
        if self.packet_tick == Some(tick) {
            return Err(C220FixpExternalOutputError::RepeatedPacket(tick));
        }
        self.packet_tick = Some(tick);
        let Some(head) = self.bursts.front_mut() else {
            return Ok(None);
        };
        if !destination_ready || head.ready_tick > tick {
            return Ok(None);
        }
        let address = if head.gather {
            head.address
        } else {
            head.address.wrapping_add(u64::from(head.row_offset))
        };
        let bytes = if head.gather {
            let Some(bytes) = head
                .policy
                .gathered_packet(address, head.bytes, head.closed)
            else {
                return Ok(None);
            };
            head.address = head.address.wrapping_add(u64::from(bytes));
            bytes
        } else {
            if head.row_offset >= head.row_bytes {
                return Ok(None);
            }
            let bytes = head
                .policy
                .row_packet(address, head.row_bytes - head.row_offset);
            head.row_offset += bytes;
            if head.row_offset == head.row_bytes {
                head.row_offset = 0;
                head.address = head
                    .address
                    .wrapping_add(u64::from(head.policy.row_stride_bytes));
            }
            bytes
        };
        head.bytes = head.bytes.wrapping_sub(bytes);
        let finished = head.closed && head.bytes == 0;
        let fragment = C220MteOutputFragment {
            instruction_id: head.instruction_id,
            request_id: head.request_id,
            destination_address: address,
            bytes,
            last_in_uop: true,
            last_in_instruction: finished && head.last_in_instruction,
        };
        let second_channel = head.second_channel_offset.map(|offset| {
            stores.publish(C220MteOutputFragment {
                destination_address: fragment.destination_address.wrapping_add(offset),
                last_in_instruction: false,
                ..fragment
            })
        });
        if finished {
            self.bursts.pop_front();
        }
        Ok(Some(C220FixpExternalWriteBatch {
            second_channel,
            primary: stores.publish(fragment),
        }))
    }

    fn observe(&mut self, tick: u64) -> Result<(), C220FixpExternalOutputError> {
        if let Some(previous) = self.observed_tick
            && tick < previous
        {
            return Err(C220FixpExternalOutputError::TimeReversed {
                previous,
                requested: tick,
            });
        }
        self.observed_tick = Some(tick);
        Ok(())
    }
}
