use super::*;
use crate::isa::c220::mte::fixp::C220FixpDescriptor;
use crate::memory::{region::MemoryRegion, sparse::SparseMemory};
use crate::sim::c220::memory::l1::C220L1Geometry;
use crate::sim::c220::mte::{
    interface::biu_write::command::C220BiuWriteConfig, mte1::frontend::C220Mte1ReadBandwidths,
    pipeline::C220MtePipelineConfig, set2d::C220Set2dBandwidths,
};
use std::num::NonZeroU32;

#[test]
fn nz2nd_engine_executes_rows_and_waits_for_external_response() {
    for conversion in [0, 1, 8, 9, 10, 11, 12, 13, 21, 22, 23, 24, 25, 26] {
        let int4 = matches!(conversion, 21 | 22 | 25 | 26);
        use C220FixpNz2ndStage::*;
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
        let mut engine = C220FixpNz2ndEngine::new(
            C220FixpEngineConfig {
                instruction_fifo_depth: 1,
                read_bandwidth: 128,
                read_bank_count: 32,
                read_data_latency: 2,
                l0c_capacity: 131072,
            },
            8,
            16,
        )
        .unwrap();
        let mut l0c = C220L0c::new(131072, 12).unwrap();
        let input = if matches!(conversion, 8..=13 | 21 | 22) {
            2_i32.to_le_bytes()
        } else {
            1_f32.to_le_bytes()
        };
        l0c.buffer_mut()
            .write_known_linear(0, &input.repeat(1024))
            .unwrap();
        let mut slopes = C220LocalBuffer::new(2176);
        if matches!(conversion, 8 | 10 | 12 | 21 | 23 | 25) {
            let factor = if matches!(conversion, 23 | 25) {
                0x3f80_0000_u64
            } else if matches!(conversion, 8 | 10 | 21) {
                0x3f00_0000_u64
            } else {
                0
            };
            slopes
                .write_known_linear(0, &factor.to_le_bytes().repeat(32))
                .unwrap();
            slopes.write_known_linear(2048, &[0; 128]).unwrap();
        }
        let mut memory = MappedMemory::bind(
            SparseMemory::new(
                vec![MemoryRegion::new(256, vec![0; 256]).unwrap()],
                4096,
                4096,
            ),
            &[4096],
        )
        .unwrap();
        let operands = C220FixpExternalCommand {
            command: C220FixpCommand {
                descriptor: C220FixpDescriptor {
                    xt: (32 << 32) | (2 << 16) | ((if int4 { 19 } else { 17 }) << 4),
                    xm: (1 << 43) | (conversion << 34) | 2,
                    nd: 1,
                },
                source_format: if matches!(conversion, 8..=13 | 21 | 22) {
                    C220FixpSourceFormat::Int32
                } else {
                    C220FixpSourceFormat::Fp32
                },
                source_address: 0,
                destination_address: 4096,
                control: 0,
                scalar_slope: 0,
                slope_base_block: 0,
                dequant_base_block: 0,
                scalar_dequant: if matches!(conversion, 24 | 26) {
                    0x3f80_0000
                } else if matches!(conversion, 9 | 11 | 22) {
                    0x3f00_0000
                } else {
                    0
                },
            },
            biu_mode_word: 0,
            output_mode_word: 0,
        };
        assert_eq!(
            engine.admit(0, 7, 0, 0, operands).unwrap(),
            C220FixpAdmission::Active
        );
        pipeline
            .bind_nz2nd_stages(&[
                GenerateRead,
                SendRead,
                SendL0c,
                ReceiveL0c,
                Convert,
                Slice,
                Transpose,
                Align,
                Packetize,
                GenerateWrite,
                SendWrite,
            ])
            .unwrap();
        assert!(matches!(
            pipeline.advance(0),
            Err(C220MtePipelineError::FixpContextMismatch)
        ));
        let mut dbids = VecDeque::new();
        let mut responses = VecDeque::new();
        let mut retired = None;
        let mut executed = None;
        let mut sent_bytes = 0;
        for tick in 0..512 {
            pipeline
                .advance_nz2nd(
                    tick,
                    &mut engine,
                    C220FixpNz2ndMemory {
                        l0c: &mut l0c,
                        slopes: &slopes,
                        external: &mut memory,
                        atomics: C220FixpAtomicConfig::default(),
                    },
                    false,
                )
                .unwrap();
            for event in pipeline.last_events() {
                if let crate::sim::c220::mte::pipeline::C220MtePipelineEvent::Nz2nd(
                    C220FixpNz2ndEvent::Read(C220FixpEvent::ReceivedL0c {
                        functional: Some(event),
                        ..
                    }),
                ) = event
                    && event.executed
                {
                    assert!(executed.replace(tick).is_none());
                }
            }
            if let Some(command) = pipeline.take_biu_write_command().unwrap() {
                dbids.push_back((tick + 5, command.command.tag));
            }
            if let Some((_, tag)) = dbids.pop_front_if(|(ready, _)| *ready <= tick) {
                pipeline
                    .receive_biu_write_dbid(C220BiuSubcore::Cube, tag)
                    .unwrap();
            }
            if let Some(data) = pipeline.take_biu_write_data().unwrap() {
                sent_bytes += data.source.request.bytes;
                responses.push_back((tick + 10, data.source.request.tag));
                assert!(!engine.is_idle());
            }
            if let Some((_, tag)) = responses.pop_front_if(|(ready, _)| *ready <= tick) {
                let response = pipeline.receive_biu_write_response(tag).unwrap();
                if let Some(state) = engine.retire_response(response).unwrap() {
                    retired = Some((tick, state));
                    break;
                }
            }
        }
        let (tick, state) = retired.expect("external instruction must retire");
        assert_eq!(state.lifecycle.executed_tick, executed);
        assert!(executed.unwrap() < state.lifecycle.write_dispatched_tick.unwrap());
        assert!(state.lifecycle.write_dispatched_tick.unwrap() < tick);
        assert!(engine.is_idle() && pipeline.is_idle());
        assert_eq!(pipeline.next_nz2nd_event_tick(&engine), None);
        if int4 {
            assert_eq!(sent_bytes, 2);
            for row in 0..2 {
                assert_eq!(memory.read_known_at(4096 + row * 16, 9).unwrap(), [0x11; 9]);
                assert_eq!(memory.read_known_at(4105 + row * 16, 7).unwrap(), [0; 7]);
            }
            continue;
        }
        let lane = if conversion == 0 {
            1_f32.to_le_bytes().to_vec()
        } else if matches!(conversion, 8 | 9 | 23 | 24) {
            vec![1]
        } else if matches!(conversion, 12 | 13) {
            1_i16.to_le_bytes().to_vec()
        } else {
            0x3c00u16.to_le_bytes().to_vec()
        };
        assert_eq!(
            sent_bytes as usize,
            if matches!(conversion, 8 | 9 | 23 | 24) {
                2
            } else {
                34 * lane.len()
            }
        );
        for row in 0..2 {
            let address = 4096 + row * 32 * lane.len() as u64;
            assert_eq!(
                memory.read_known_at(address, 17 * lane.len()).unwrap(),
                lane.repeat(17)
            );
            assert_eq!(
                memory
                    .read_known_at(address + 17 * lane.len() as u64, 15 * lane.len())
                    .unwrap(),
                vec![0; 15 * lane.len()]
            );
        }
    }
}
