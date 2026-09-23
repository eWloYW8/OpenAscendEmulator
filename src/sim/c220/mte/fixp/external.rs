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
        atomics_enabled: bool,
        mut observe: impl FnMut(&C220FixpSliceResult, &[u8]),
    ) -> Result<(), C220FixpExecutionError> {
        let operation = ((self.command.control >> 9) & 3) as u8;
        let data_type = ((self.command.control >> 6) & 7) as u8;
        for result in self.command.evaluate(l0c, slopes)? {
            let result = result?;
            let address = result.coordinate.destination_address;
            let mut bytes = result.conversion.bytes.clone();
            if atomics_enabled && operation != 3 {
                if matches!(data_type, 1 | 2 | 6) {
                    return Err(C220FixpExecutionError::AtomicDataType(data_type));
                }
                let previous = memory.read_known_at(address, bytes.len())?;
                combine_integer_atomic(&mut bytes, &previous, data_type, operation);
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

fn combine_integer_atomic(bytes: &mut [u8], previous: &[u8], data_type: u8, operation: u8) {
    macro_rules! combine {
        ($ty:ty, $width:expr) => {
            for (next, old) in bytes
                .chunks_exact_mut($width)
                .zip(previous.chunks_exact($width))
            {
                let next_value = <$ty>::from_le_bytes(next.try_into().expect("lane width"));
                let old_value = <$ty>::from_le_bytes(old.try_into().expect("lane width"));
                let value = match operation {
                    0 => next_value.wrapping_add(old_value),
                    1 => next_value.max(old_value),
                    2 => next_value.min(old_value),
                    _ => unreachable!("atomic operation is decoded before execution"),
                };
                next.copy_from_slice(&value.to_le_bytes());
            }
        };
    }
    match data_type {
        3 => combine!(i16, 2),
        4 => combine!(i32, 4),
        5 => combine!(i8, 1),
        0 | 7 => {}
        _ => unreachable!("floating-point atomics are checked before execution"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::c220::mte::fixp::C220FixpDescriptor;
    use crate::memory::{region::MemoryRegion, sparse::SparseMemory};
    use crate::sim::c220::mte::fixp::C220FixpSourceFormat;

    #[test]
    fn external_writes_preserve_row_gaps_and_apply_signed_atomic_controls() {
        let mut command = C220FixpExternalCommand {
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
            .execute_to_external(&l0c, &slopes, &mut memory, true, |slice, stored| {
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
        command.command.control = 1 << 6;
        assert!(matches!(
            command.execute_to_external(&l0c, &slopes, &mut memory, true, |_, _| {}),
            Err(C220FixpExecutionError::AtomicDataType(1))
        ));
        command
            .execute_to_external(&l0c, &slopes, &mut memory, false, |_, stored| {
                assert_eq!(stored, (-2i32).to_le_bytes().repeat(stored.len() / 4));
            })
            .unwrap();

        let mut bytes = [127, 128, 255];
        combine_integer_atomic(&mut bytes, &[1, 255, 1], 5, 0);
        assert_eq!(bytes, [128, 127, 0]);
        combine_integer_atomic(&mut bytes, &[1, 255, 1], 5, 1);
        assert_eq!(bytes, [1, 127, 1]);
        combine_integer_atomic(&mut bytes, &[255, 128, 0], 5, 2);
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
