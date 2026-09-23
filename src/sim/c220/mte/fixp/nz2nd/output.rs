use std::collections::VecDeque;
#[cfg(test)]
use std::num::NonZeroU32;

use super::super::{C220FixpExternalOutputPolicy, C220FixpStoreBuffer, C220FixpStoreWrite};
use super::{C220FixpNz2ndStaging, C220FixpNz2ndStagingEntry, C220FixpNz2ndStagingError};
use crate::sim::c220::mte::interface::C220MteOutputFragment;

#[cfg(test)]
mod tests {
    use super::*;

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
        let mut output = C220FixpNz2ndOutput::default();
        output.bursts.push_back(C220FixpNz2ndBurst {
            instruction_id: 91,
            request_id: 7,
            address: 512,
            bytes: 512,
            closed: true,
            last_in_instruction: true,
            gather: false,
            row_bytes: 512,
            row_offset: 0,
            ready_tick: 0,
            policy: C220FixpExternalOutputPolicy::new(0, 512),
        });
        let mut writes = crate::sim::c220::mte::fixp::C220FixpBiuWritePipeline::default();
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
        let mut output = C220FixpNz2ndOutput::default();
        let mut stores = C220FixpStoreBuffer::default();
        output.bursts.push_back(C220FixpNz2ndBurst {
            instruction_id: 3,
            request_id: 7,
            address: 16,
            bytes: 80,
            closed: true,
            last_in_instruction: true,
            gather: false,
            row_bytes: 40,
            row_offset: 0,
            policy,
            ready_tick: 1,
        });
        assert!(output.take_write(0, true, &mut stores).unwrap().is_none());
        assert!(output.take_write(1, false, &mut stores).unwrap().is_none());
        assert!(stores.is_empty());
        for (index, (address, bytes)) in [(16, 16), (32, 24), (1040, 16), (1056, 24)]
            .into_iter()
            .enumerate()
        {
            let write = output
                .take_write(index as u64 + 2, true, &mut stores)
                .unwrap()
                .unwrap();
            assert_eq!(write.token.get(), index as u32 + 1);
            let fragment = write.fragment;
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
pub struct C220FixpNz2ndBurst {
    pub instruction_id: u64,
    pub request_id: u64,
    pub address: u64,
    pub bytes: u32,
    pub closed: bool,
    pub last_in_instruction: bool,
    pub gather: bool,
    pub row_bytes: u32,
    pub row_offset: u32,
    pub policy: C220FixpExternalOutputPolicy,
    pub ready_tick: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum C220FixpNz2ndOutputError {
    #[error(transparent)]
    Staging(#[from] C220FixpNz2ndStagingError),
    #[error("NZ2ND output time reversed from {previous} to {requested}")]
    TimeReversed { previous: u64, requested: u64 },
    #[error("NZ2ND output packet callback repeated at tick {0}")]
    RepeatedPacket(u64),
    #[error("NZ2ND output time overflow")]
    TimeOverflow,
}

/// External-memory packetization following NZ2ND alignment staging. Address
/// holes between rows are retained; downstream transaction credit and write
/// acknowledgments remain the responsibility of the external interface.
#[derive(Debug, Clone, Default)]
pub struct C220FixpNz2ndOutput {
    bursts: VecDeque<C220FixpNz2ndBurst>,
    observed_tick: Option<u64>,
    packet_tick: Option<u64>,
}

impl C220FixpNz2ndOutput {
    pub fn bursts(&self) -> &VecDeque<C220FixpNz2ndBurst> {
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
    ) -> Result<Option<C220FixpNz2ndStagingEntry>, C220FixpNz2ndOutputError> {
        self.observe(tick)?;
        let ready_tick = tick
            .checked_add(1)
            .ok_or(C220FixpNz2ndOutputError::TimeOverflow)?;
        let entry = staging.take_ready(tick, self.can_receive())?;
        if let Some(entry) = entry {
            let op = entry.operation;
            let d = op.descriptor;
            if d.last_in_uop {
                if let Some(tail) = self.bursts.back_mut().filter(|tail| !tail.closed) {
                    tail.bytes = tail.bytes.wrapping_add(d.bytes);
                    tail.closed = d.end_of_burst;
                    tail.last_in_instruction = op.last_in_instruction;
                } else {
                    self.bursts.push_back(C220FixpNz2ndBurst {
                        instruction_id: op.instruction_id,
                        request_id: u64::from(op.request_id),
                        address: d.destination_address,
                        bytes: d.bytes,
                        closed: d.end_of_burst,
                        last_in_instruction: op.last_in_instruction,
                        gather: d.gather,
                        row_bytes: d.burst_bytes,
                        row_offset: 0,
                        policy,
                        ready_tick,
                    });
                }
            }
        }
        Ok(entry)
    }

    pub fn take_write(
        &mut self,
        tick: u64,
        destination_ready: bool,
        stores: &mut C220FixpStoreBuffer,
    ) -> Result<Option<C220FixpStoreWrite>, C220FixpNz2ndOutputError> {
        self.observe(tick)?;
        if self.packet_tick == Some(tick) {
            return Err(C220FixpNz2ndOutputError::RepeatedPacket(tick));
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
        if finished {
            self.bursts.pop_front();
        }
        Ok(Some(stores.publish(fragment)))
    }

    fn observe(&mut self, tick: u64) -> Result<(), C220FixpNz2ndOutputError> {
        if let Some(previous) = self.observed_tick
            && tick < previous
        {
            return Err(C220FixpNz2ndOutputError::TimeReversed {
                previous,
                requested: tick,
            });
        }
        self.observed_tick = Some(tick);
        Ok(())
    }
}
