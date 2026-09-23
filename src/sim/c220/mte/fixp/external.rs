use super::atomic::{C220FixpAtomicConfig, combine_atomic};
use crate::isa::c220::mte::fixp::C220FixpDestination;
use crate::memory::mapped::MappedMemory;
use crate::sim::c220::memory::C220LocalBuffer;
use crate::sim::c220::mte::uop::C220DmaUopMode;

use super::{
    C220FixpCommand, C220FixpExecutionError, C220FixpNz2ndInstructionPlan,
    C220FixpNz2ndOutputPolicy, C220FixpNz2ndPlanError, C220FixpOutputFormat, C220FixpSliceResult,
};

/// External-destination FIX operands and independent output/BIU controls.
/// Capturing a command does not issue it or publish destination data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FixpExternalCommand {
    pub command: C220FixpCommand,
    pub biu_mode_word: u64,
    pub output_mode_word: u64,
}

impl C220FixpExternalCommand {
    /// Publishes functional results without advancing timing or retiring BIU
    /// requests. The observer receives conversion data and actual stored bytes
    /// separately, since atomic operations can change the latter.
    pub fn execute_to_external(
        self,
        l0c: &C220LocalBuffer,
        slopes: &C220LocalBuffer,
        memory: &mut MappedMemory,
        atomics: C220FixpAtomicConfig,
        mut observe: impl FnMut(&C220FixpSliceResult, &[u8]),
    ) -> Result<(), C220FixpExecutionError> {
        let operation = ((self.command.control >> 9) & 3) as u8;
        let data_type = ((self.command.control >> 6) & 7) as u8;
        for result in self.command.evaluate(l0c, slopes)? {
            let result = result?;
            let address = result.coordinate.destination_address;
            let mut bytes = result.conversion.bytes.clone();
            if atomics.enabled && operation != 3 {
                let previous = memory.read_known_at(address, bytes.len())?;
                combine_atomic(
                    &mut bytes,
                    &previous,
                    data_type,
                    operation,
                    self.command.control,
                    atomics.fp16_rounding,
                );
            }
            memory.write_known_at(address, &bytes)?;
            observe(&result, &bytes);
        }
        Ok(())
    }

    pub fn capture(
        word: u32,
        control: u64,
        isa_instance_index: u32,
        read_gpr: impl FnMut(u8) -> Option<u64>,
        mut read_spr: impl FnMut(u8) -> Option<u64>,
    ) -> Result<Self, C220FixpExecutionError> {
        let command = C220FixpCommand::capture_destination(
            word,
            C220FixpDestination::External,
            control,
            read_gpr,
            &mut read_spr,
        )?;
        let (biu_mode_word, output_mode_word) = if isa_instance_index == 0 {
            (0, 0)
        } else {
            (
                read_spr(93).ok_or(C220FixpExecutionError::MissingSpr(93))?,
                read_spr(94).ok_or(C220FixpExecutionError::MissingSpr(94))?,
            )
        };
        Ok(Self {
            command,
            biu_mode_word,
            output_mode_word,
        })
    }

    pub const fn biu_mode(self) -> C220DmaUopMode {
        C220DmaUopMode::from_mode_word(self.biu_mode_word)
    }

    pub fn nz2nd_output_policy(self) -> Result<C220FixpNz2ndOutputPolicy, C220FixpExecutionError> {
        self.command.layout()?;
        let format = C220FixpOutputFormat::from_conversion_mode(
            self.command.source_format,
            self.command.descriptor.conversion_mode(),
        )
        .expect("validated format");
        Ok(C220FixpNz2ndOutputPolicy::new(
            self.output_mode_word,
            self.command
                .descriptor
                .destination_stride()
                .wrapping_mul(format.lane_bytes()),
        ))
    }

    pub fn plan_nz2nd(
        self,
        instruction_id: u64,
        first_read_id: u32,
        first_write_id: u32,
        bandwidth: u32,
        main_slots: u32,
    ) -> Result<C220FixpNz2ndInstructionPlan, C220FixpNz2ndPlanError> {
        C220FixpNz2ndInstructionPlan::new(
            self.command,
            instruction_id,
            first_read_id,
            first_write_id,
            bandwidth,
            main_slots,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::c220::numeric::fp16::C220Fp16AddRounding;
    const ENABLED: C220FixpAtomicConfig = C220FixpAtomicConfig {
        enabled: true,
        fp16_rounding: C220Fp16AddRounding::NearestEven,
    };
    const DISABLED: C220FixpAtomicConfig = C220FixpAtomicConfig {
        enabled: false,
        ..ENABLED
    };
    use crate::isa::c220::mte::fixp::C220FixpDescriptor;
    use crate::memory::{region::MemoryRegion, sparse::SparseMemory};
    use crate::sim::c220::mte::fixp::C220FixpSourceFormat;

    #[test]
    fn half_width_atomics_respect_saturation_nan_zero_and_rounding() {
        for (data_type, operation, first, second, saturating, nonsaturating) in [
            (6, 0, 0x3f80u16, 0x3b80u16, 0x3f80u16, 0x3f80u16),
            (6, 0, 0x3f81, 0x3b80, 0x3f82, 0x3f82),
            (6, 0, 0x7f7f, 0x7f7f, 0x7f7f, 0x7f80),
            (6, 0, 0x7f80, 0xff80, 0, 0x7fff),
            (6, 0, 1, 1, 2, 2),
            (6, 0, 0x8000, 0x8000, 0x8000, 0x8000),
            (6, 1, 0x7f80, 0x3f80, 0x7f7f, 0x7f80),
            (6, 2, 0xff80, 0x3f80, 0xff7f, 0xff80),
            (6, 1, 0x7fc1, 0x3f80, 0, 0x7fff),
            (6, 1, 0x8000, 0, 0, 0),
            (6, 2, 0x8000, 0, 0x8000, 0x8000),
            (2, 1, 0x7c00, 0x3c00, 0x7bff, 0x7c00),
            (2, 2, 0xfc00, 0x3c00, 0xfbff, 0xfc00),
            (2, 1, 0x7e01, 0x3c00, 0, 0x7fff),
            (2, 2, 0x8000, 0, 0x8000, 0x8000),
            (2, 0, 0x3c01, 0x1000, 0x3c02, 0x3c02),
            (2, 0, 0x7c00, 0xfc00, 0, 0x7fff),
            (2, 0, 0x7bff, 0x4c00, 0x7bff, 0x7c00),
        ] {
            for (control, expected) in [(0, saturating), (1 << 48, nonsaturating)] {
                let mut bytes = first.to_le_bytes();
                combine_atomic(
                    &mut bytes,
                    &second.to_le_bytes(),
                    data_type,
                    operation,
                    control,
                    C220Fp16AddRounding::NearestEven,
                );
                assert_eq!(u16::from_le_bytes(bytes), expected);
            }
        }
    }

    #[test]
    fn fp32_atomic_writes_canonicalize_nan_and_select_signed_zero() {
        let source = [0x8000_0000u32, 0x7f80_0000, 0x7fc0_1234, 0x3f80_0000];
        let old = [0u32, 0xff80_0000, 0x3f80_0000, 0x3380_0000];
        let mut l0c = C220LocalBuffer::new(64);
        let source: Vec<u8> = source.into_iter().flat_map(u32::to_le_bytes).collect();
        l0c.write_known_linear(0, &source.repeat(4)).unwrap();
        let previous: Vec<u8> = old.into_iter().flat_map(u32::to_le_bytes).collect();
        let slopes = C220LocalBuffer::new(0);
        for (operation, expected) in [
            (0, [0u32, 0x7fff_ffff, 0x7fff_ffff, 0x3f80_0000]),
            (1, [0u32, 0x7f80_0000, 0x7fff_ffff, 0x3f80_0000]),
            (2, [0x8000_0000u32, 0xff80_0000, 0x7fff_ffff, 0x3380_0000]),
        ] {
            let command = C220FixpExternalCommand {
                command: C220FixpCommand {
                    descriptor: C220FixpDescriptor {
                        xt: (16 << 32) | (1 << 16) | (4 << 4),
                        xm: 1 << 43,
                        nd: 1,
                    },
                    source_format: C220FixpSourceFormat::Fp32,
                    source_address: 0,
                    destination_address: 4096,
                    control: (operation << 9) | (1 << 6),
                    scalar_slope: 0,
                    slope_base_block: 0,
                    dequant_base_block: 0,
                    scalar_dequant: 0,
                },
                biu_mode_word: 0,
                output_mode_word: 0,
            };
            let mut memory = MappedMemory::bind(
                SparseMemory::new(
                    vec![MemoryRegion::new(16, previous.clone()).unwrap()],
                    64,
                    64,
                ),
                &[4096],
            )
            .unwrap();
            command
                .execute_to_external(&l0c, &slopes, &mut memory, ENABLED, |_, _| {})
                .unwrap();
            let expected: Vec<u8> = expected.into_iter().flat_map(u32::to_le_bytes).collect();
            assert_eq!(memory.read_known_at(4096, 16).unwrap(), expected);
        }
    }

    #[test]
    fn external_writes_preserve_row_gaps_and_apply_signed_atomic_controls() {
        let command = C220FixpExternalCommand {
            command: C220FixpCommand {
                descriptor: C220FixpDescriptor {
                    xt: (32 << 32) | (2 << 16) | (17 << 4),
                    xm: (1 << 43) | 2,
                    nd: 1,
                },
                source_format: C220FixpSourceFormat::Int32,
                source_address: 0,
                destination_address: 4096,
                control: 4 << 6,
                scalar_slope: 0,
                slope_base_block: 0,
                dequant_base_block: 0,
                scalar_dequant: 0,
            },
            biu_mode_word: 0,
            output_mode_word: 0,
        };
        let mut l0c = C220LocalBuffer::new(4096);
        l0c.write_known_linear(0, &(-2i32).to_le_bytes().repeat(1024))
            .unwrap();
        let slopes = C220LocalBuffer::new(0);
        let mut memory = MappedMemory::bind(
            SparseMemory::new(
                vec![MemoryRegion::new(256, 1i32.to_le_bytes().repeat(64)).unwrap()],
                4096,
                4096,
            ),
            &[4096],
        )
        .unwrap();
        let mut observed = 0;
        command
            .execute_to_external(&l0c, &slopes, &mut memory, ENABLED, |slice, stored| {
                assert_eq!(
                    slice.conversion.bytes,
                    (-2i32).to_le_bytes().repeat(stored.len() / 4)
                );
                assert_eq!(stored, (-1i32).to_le_bytes().repeat(stored.len() / 4));
                observed += 1;
            })
            .unwrap();
        assert_eq!(observed, 4);
        for row in 0..2 {
            assert_eq!(
                memory.read_known_at(4096 + row * 128, 68).unwrap(),
                (-1i32).to_le_bytes().repeat(17)
            );
            assert_eq!(
                memory.read_known_at(4096 + row * 128 + 68, 60).unwrap(),
                1i32.to_le_bytes().repeat(15)
            );
        }
        command
            .execute_to_external(&l0c, &slopes, &mut memory, DISABLED, |_, stored| {
                assert_eq!(stored, (-2i32).to_le_bytes().repeat(stored.len() / 4));
            })
            .unwrap();

        let mut bytes = [127, 128, 255];
        combine_atomic(
            &mut bytes,
            &[1, 255, 1],
            5,
            0,
            0,
            C220Fp16AddRounding::NearestEven,
        );
        assert_eq!(bytes, [128, 127, 0]);
        combine_atomic(
            &mut bytes,
            &[1, 255, 1],
            5,
            1,
            0,
            C220Fp16AddRounding::NearestEven,
        );
        assert_eq!(bytes, [1, 127, 1]);
        combine_atomic(
            &mut bytes,
            &[255, 128, 0],
            5,
            2,
            0,
            C220Fp16AddRounding::NearestEven,
        );
        assert_eq!(bytes, [255, 128, 0]);
    }

    #[test]
    fn external_capture_keeps_controls_independent_and_feeds_nz2nd_plan() {
        let word = (6 << 29) | (2 << 24) | (1 << 17) | (2 << 12) | (3 << 7) | (4 << 2);
        let gpr = |r| match r {
            1 => Some(4096),
            2 => Some(0),
            3 => Some((32 << 32) | (2 << 16) | (17 << 4)),
            4 => Some((1 << 43) | (1 << 34) | 2),
            _ => None,
        };
        let spr = |r| match r {
            61 | 64 => Some(0),
            97 => Some(1),
            93 => Some(5),
            94 => Some(3),
            _ => None,
        };
        let captured = C220FixpExternalCommand::capture(word, 1 << 48, 1, gpr, spr).unwrap();
        assert_eq!(captured.command.control, 1 << 48);
        assert_eq!(captured.biu_mode(), C220DmaUopMode::Fixed128);
        let policy = captured.nz2nd_output_policy().unwrap();
        assert_eq!((policy.burst_control, policy.row_stride_bytes), (3, 64));
        let mut plan = captured.plan_nz2nd(71, 8, 16, 128, 8).unwrap();
        assert!(plan.reads.next().is_some());
        assert!(plan.writes.next().is_some());
        let zero = C220FixpExternalCommand::capture(word, 0, 0, gpr, |r| {
            assert!(!matches!(r, 93 | 94));
            spr(r)
        })
        .unwrap();
        assert_eq!((zero.biu_mode_word, zero.output_mode_word), (0, 0));
        assert!(matches!(
            C220FixpExternalCommand::capture(word, 0, 1, gpr, |r| if r == 94 {
                None
            } else {
                spr(r)
            }),
            Err(C220FixpExecutionError::MissingSpr(94))
        ));
        assert!(matches!(
            C220FixpExternalCommand::capture(word | (1 << 24), 0, 0, gpr, spr),
            Err(C220FixpExecutionError::Destination(C220FixpDestination::L1))
        ));
    }
}
