use super::super::{C220FixpCommand, C220FixpLayoutError, C220FixpOutputFormat};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FixpNz2ndWriteDescriptor {
    pub destination_address: u64,
    pub bytes: u32,
    pub last_in_uop: bool,
    pub end_of_burst: bool,
    pub gather: bool,
    pub burst_bytes: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::c220::mte::fixp::C220FixpDescriptor;
    use crate::sim::c220::mte::fixp::C220FixpSourceFormat;

    #[test]
    fn plans_partial_rows_gather_and_aligned_suppression() {
        let mut command = C220FixpCommand {
            descriptor: C220FixpDescriptor {
                xt: (96 << 32) | (1 << 16) | (65 << 4),
                xm: (1 << 43) | (1 << 34),
                nd: 1,
            },
            source_format: C220FixpSourceFormat::Fp32,
            source_address: 0,
            destination_address: 0,
            control: 0,
            scalar_slope: 0,
            slope_base_block: 0,
            dequant_base_block: 0,
            scalar_dequant: 0,
        };
        let plan = C220FixpNz2ndWritePlanner::new(command, 8)
            .unwrap()
            .plan_row(4, true, 4096);
        assert_eq!(plan.leading.unwrap().bytes, 128);
        assert!(!plan.leading.unwrap().last_in_uop);
        assert_eq!(plan.tail.bytes, 130);
        assert!(plan.tail.end_of_burst && plan.advances_column);
        command.descriptor.xt = (65 << 32) | (1 << 16) | (65 << 4);
        let plan = C220FixpNz2ndWritePlanner::new(command, 8)
            .unwrap()
            .plan_row(4, true, 4096);
        assert!(plan.tail.gather);
        assert!(!plan.tail.end_of_burst);
        assert!(!plan.advances_column);
        command.descriptor.xt = (512 << 32) | (1 << 16) | (256 << 4);
        let planner = C220FixpNz2ndWritePlanner::new(command, 8).unwrap();
        let suppressed = planner.plan_row(7, false, 4352);
        assert_eq!(
            (suppressed.tail.destination_address, suppressed.tail.bytes),
            (4096, 512)
        );
        assert!(!suppressed.tail.last_in_uop);
        assert!(planner.plan_row(15, true, 4352).tail.last_in_uop);
        command.descriptor.xm &= !(31 << 34);
        command.descriptor.xt = (32 << 32) | (1 << 16) | (9 << 4);
        assert_eq!(
            C220FixpNz2ndWritePlanner::new(command, 8)
                .unwrap()
                .plan_row(1, true, 0)
                .tail
                .bytes,
            36
        );
        command.descriptor.xt = (1024 << 32) | (1 << 16) | (1024 << 4);
        for mode in [19, 20, 21, 22] {
            command.descriptor.xm = (1 << 43) | (mode << 34);
            assert_eq!(
                C220FixpNz2ndWritePlanner::new(command, 1).unwrap().gather(),
                mode <= 20
            );
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FixpNz2ndWriteBatch {
    pub leading: Option<C220FixpNz2ndWriteDescriptor>,
    pub tail: C220FixpNz2ndWriteDescriptor,
    pub advances_column: bool,
}

impl C220FixpNz2ndWriteBatch {
    pub fn descriptors(self) -> impl Iterator<Item = C220FixpNz2ndWriteDescriptor> {
        self.leading.into_iter().chain([self.tail])
    }
}

#[derive(Debug, thiserror::Error)]
pub enum C220FixpNz2ndWriteError {
    #[error(transparent)]
    Layout(#[from] C220FixpLayoutError),
    #[error("NZ2ND write planning requires an NZ2ND command")]
    NotNzToNd,
    #[error("NZ2ND main transpose buffer size must be nonzero")]
    BufferSize,
}

/// Plans one row's write descriptors when a conversion group closes. These
/// descriptors consume transpose credits; they are not memory write packets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FixpNz2ndWritePlanner {
    command: C220FixpCommand,
    format: C220FixpOutputFormat,
    main_slots: u32,
    gather: bool,
    aligned_512: bool,
}

impl C220FixpNz2ndWritePlanner {
    pub fn new(command: C220FixpCommand, main_slots: u32) -> Result<Self, C220FixpNz2ndWriteError> {
        command.layout()?;
        let d = command.descriptor;
        if !d.nz_to_nd() {
            return Err(C220FixpNz2ndWriteError::NotNzToNd);
        }
        if main_slots == 0 {
            return Err(C220FixpNz2ndWriteError::BufferSize);
        }
        let format =
            C220FixpOutputFormat::from_conversion_mode(command.source_format, d.conversion_mode())
                .expect("validated layout");
        let column_bytes = if d.conversion_mode() == 0 && d.channel_split() {
            32
        } else if matches!(d.conversion_mode(), 19 | 20) {
            0
        } else {
            format.storage_bytes(16)
        };
        let gather = column_bytes * u32::from(d.columns()).div_ceil(16)
            <= main_slots.wrapping_mul(32)
            && u32::from(d.columns()) == d.destination_stride();
        let aligned_512 = format
            .storage_bytes(u32::from(d.columns()))
            .is_multiple_of(512)
            && command.destination_address.is_multiple_of(512)
            && format
                .storage_bytes(d.destination_stride())
                .is_multiple_of(512);
        Ok(Self {
            command,
            format,
            main_slots,
            gather,
            aligned_512,
        })
    }

    pub fn gather(&self) -> bool {
        self.gather
    }

    pub fn plan_row(
        &self,
        group_index: u32,
        end_of_group: bool,
        destination_address: u64,
    ) -> C220FixpNz2ndWriteBatch {
        let d = self.command.descriptor;
        let remainder = u32::from(d.columns() % 16);
        let parts = if d.conversion_mode() == 0 {
            if end_of_group && remainder != 0 {
                remainder.div_ceil(8)
            } else {
                2
            }
        } else {
            1
        };
        let tail_lanes = u32::from(d.columns()) % (16 / parts);
        let bytes = if !end_of_group {
            self.main_slots.wrapping_mul(32)
        } else if tail_lanes != 0 {
            32 * (group_index & 7) + self.format.storage_bytes(tail_lanes)
        } else {
            32 * ((group_index & 7) + 1)
        };
        let leading = (group_index & 7 > 3).then_some(C220FixpNz2ndWriteDescriptor {
            destination_address,
            bytes: 128,
            last_in_uop: false,
            end_of_burst: false,
            gather: false,
            burst_bytes: 0,
        });
        let mut tail = C220FixpNz2ndWriteDescriptor {
            destination_address,
            bytes,
            last_in_uop: true,
            end_of_burst: !self.gather,
            gather: self.gather,
            burst_bytes: if self.gather { 0 } else { bytes },
        };
        if leading.is_some() && !self.gather && self.aligned_512 {
            tail.destination_address &= !511;
            tail.bytes = 512;
            tail.burst_bytes = 512;
            tail.last_in_uop = group_index & 15 == 15;
            tail.end_of_burst = tail.last_in_uop;
        }
        C220FixpNz2ndWriteBatch {
            leading,
            tail,
            advances_column: !self.gather,
        }
    }
}
