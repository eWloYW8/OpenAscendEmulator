use super::{C220FixpCommand, C220FixpLayoutError};
use crate::sim::c220::cube::C220CubeL0cAccess;
use crate::sim::c220::memory::{C220L0cFragmentRequest, C220L0cReadRequest};
use crate::sim::c220::mte::interface::C220MteL0cReadOperation;

#[derive(Debug, Clone)]
pub enum C220FixpReadStream {
    Columns(C220FixpReadGenerator),
    Nz2nd(super::C220FixpNz2ndReadGenerator),
}

impl From<C220FixpReadGenerator> for C220FixpReadStream {
    fn from(generator: C220FixpReadGenerator) -> Self {
        Self::Columns(generator)
    }
}

impl From<super::C220FixpNz2ndReadGenerator> for C220FixpReadStream {
    fn from(generator: super::C220FixpNz2ndReadGenerator) -> Self {
        Self::Nz2nd(generator)
    }
}

impl Iterator for C220FixpReadStream {
    type Item = C220FixpReadUop;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Columns(generator) => generator.next(),
            Self::Nz2nd(generator) => generator.next(),
        }
    }
}

impl std::iter::FusedIterator for C220FixpReadStream {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FixpReadUop {
    pub operation: C220MteL0cReadOperation,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum C220FixpReadGeneratorError {
    #[error(transparent)]
    Layout(#[from] C220FixpLayoutError),
    #[error("FIX effective read bandwidth must be nonzero")]
    ZeroBandwidth,
    #[error("NZ-to-ND requires its own FIX read generator")]
    NzToNd,
}

/// Read packets are column-major, unlike functional slices. Channel splitting
/// reads full source columns but accounts for one half-column at the output.
/// Byte output combines column pairs: only the group's last column publishes
/// output credits, with separate accounting for a singleton tail.
/// Bandwidth is an explicit model input, not an assumed device constant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220FixpReadGenerator {
    command: C220FixpCommand,
    instruction_id: u64,
    next_id: u32,
    bandwidth: u32,
    column_block: u32,
    column_blocks: u32,
    source_offset: u32,
    destination_offset: u32,
}

impl C220FixpReadGenerator {
    pub fn new(
        command: C220FixpCommand,
        instruction_id: u64,
        first_id: u32,
        bandwidth: u32,
    ) -> Result<Self, C220FixpReadGeneratorError> {
        command.layout()?;
        let bandwidth = if command.descriptor.conversion_mode() == 0 {
            bandwidth / 2
        } else {
            bandwidth
        };
        if bandwidth == 0 {
            return Err(C220FixpReadGeneratorError::ZeroBandwidth);
        }
        if command.descriptor.nz_to_nd() {
            return Err(C220FixpReadGeneratorError::NzToNd);
        }
        Ok(Self {
            command,
            instruction_id,
            next_id: first_id,
            bandwidth,
            column_block: 0,
            column_blocks: if command.descriptor.is_disabled() {
                0
            } else {
                u32::from(command.descriptor.columns()).div_ceil(16)
            },
            source_offset: 0,
            destination_offset: 0,
        })
    }
}

impl Iterator for C220FixpReadGenerator {
    type Item = C220FixpReadUop;

    fn next(&mut self) -> Option<Self::Item> {
        if self.column_block == self.column_blocks {
            return None;
        }
        let d = self.command.descriptor;
        let split = d.conversion_mode() == 0 && d.channel_split();
        let int4 = matches!(d.conversion_mode(), 21 | 22 | 25 | 26);
        let group_columns = if int4 { 4 } else { 2 };
        let merge = matches!(d.conversion_mode(), 8 | 9 | 23 | 24)
            || (int4 && d.columns().is_multiple_of(64));
        let group_start = self.column_block / group_columns * group_columns;
        let singleton = merge && group_start + 1 == self.column_blocks;
        let partial_singleton = singleton && !d.columns().is_multiple_of(16);
        let source_row_bytes = 16 * self.command.source_format.lane_bytes();
        let column_bytes = u32::from(d.rows())
            * if partial_singleton {
                32
            } else {
                source_row_bytes
            };
        let address = self
            .command
            .source_address
            .wrapping_add(u64::from(
                self.column_block
                    .wrapping_mul(u32::from(d.source_stride()))
                    .wrapping_mul(source_row_bytes),
            ))
            .wrapping_add(u64::from(self.source_offset));
        let boundary = self.bandwidth - (address as u32 % self.bandwidth);
        let data_bytes = (column_bytes - self.source_offset).min(boundary);
        let output_bytes = if singleton {
            data_bytes / 4
        } else if split {
            data_bytes / 2
        } else if d.conversion_mode() == 0 {
            data_bytes
        } else if int4 && !merge {
            data_bytes.wrapping_mul(8) / 64
        } else {
            data_bytes.wrapping_mul(32) / source_row_bytes
        };
        let begins_unit = address.is_multiple_of(1024) || self.source_offset == 0;
        let end_of_burst = self.source_offset + data_bytes == column_bytes;
        let operation = C220MteL0cReadOperation {
            begins_unit,
            instruction_id: self.instruction_id,
            uop_id: self.next_id,
            conversion_mode: u32::from(d.conversion_mode()),
            last_in_instruction: end_of_burst && self.column_block + 1 == self.column_blocks,
            request: C220L0cReadRequest {
                id: self.next_id,
                fragments: C220L0cFragmentRequest {
                    address,
                    bytes: 1024,
                    access: C220CubeL0cAccess::Read,
                    check_unit_flags: begins_unit && matches!(d.unit_flag_mode(), 2 | 3),
                    update_unit_flags: begins_unit && d.unit_flag_mode() == 3,
                },
                data_type: 0,
                half_accumulator: false,
            },
            data_bytes,
            destination_address: if merge {
                self.command.destination_address.wrapping_add(
                    u64::from(group_start / group_columns)
                        * u64::from(d.destination_stride().wrapping_mul(32)),
                )
            } else {
                self.command
                    .destination_address
                    .wrapping_add(
                        u64::from(self.column_block)
                            * u64::from(d.destination_stride().wrapping_mul(32))
                            * if split { 2 } else { 1 },
                    )
                    .wrapping_add(u64::from(self.destination_offset))
            },
            output_bytes,
            second_channel_offset: (split
                && (d.columns().is_multiple_of(16) || self.column_block + 1 != self.column_blocks))
                .then_some(u64::from(d.destination_stride().wrapping_mul(32)) & !511),
            last_in_uop: !merge
                || singleton
                || self.column_block % group_columns == group_columns - 1,
            end_of_burst,
        };
        self.next_id = self.next_id.wrapping_add(1);
        if end_of_burst {
            self.column_block += 1;
            self.source_offset = 0;
            self.destination_offset = 0;
        } else {
            self.source_offset += data_bytes;
            self.destination_offset += output_bytes;
        }
        Some(C220FixpReadUop { operation })
    }
}

impl std::iter::FusedIterator for C220FixpReadGenerator {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::c220::mte::fixp::C220FixpDescriptor;

    #[test]
    fn unaligned_reads_split_columns_and_mark_only_final_instruction_packet() {
        let command = C220FixpCommand {
            source_format: crate::sim::c220::mte::fixp::C220FixpSourceFormat::Fp32,
            descriptor: C220FixpDescriptor {
                xt: (64 << 32) | (20 << 16) | (17 << 4),
                xm: (1 << 34) | (3 << 32) | 32,
                nd: 0,
            },
            source_address: 128,
            destination_address: 4096,
            control: 0,
            scalar_slope: 0,
            slope_base_block: 0,
            dequant_base_block: 0,
            scalar_dequant: 0,
        };
        let packets: Vec<_> = C220FixpReadGenerator::new(command, 7, 10, 256)
            .unwrap()
            .collect();
        assert_eq!(packets.len(), 12);
        assert_eq!(
            packets
                .iter()
                .map(|p| p.operation.data_bytes)
                .collect::<Vec<_>>(),
            [128, 256, 256, 256, 256, 128, 128, 256, 256, 256, 256, 128]
        );
        assert_eq!(
            packets.iter().filter(|p| p.operation.begins_unit).count(),
            4
        );
        assert_eq!(
            packets
                .iter()
                .filter(|p| p.operation.last_in_instruction)
                .count(),
            1
        );
        assert!(packets[11].operation.last_in_instruction);
        assert_eq!(packets[6].operation.request.fragments.address, 2176);
        assert_eq!(packets[6].operation.destination_address, 6144);
        assert_eq!(packets[5].operation.output_bytes, 64);
        assert!(packets[5].operation.end_of_burst);
        assert!(!packets[1].operation.request.fragments.check_unit_flags);

        let mut fp32 = command;
        fp32.descriptor.xm &= !(31 << 34);
        let packets: Vec<_> = C220FixpReadGenerator::new(fp32, 8, 30, 256)
            .unwrap()
            .collect();
        assert_eq!(packets.len(), 20);
        assert!(packets.iter().all(|p| p.operation.data_bytes == 128
            && p.operation.output_bytes == 128
            && p.operation.conversion_mode == 0));
        assert_eq!(packets[9].operation.destination_address, 4096 + 9 * 128);
        assert_eq!(packets[10].operation.destination_address, 6144);
        let mut split = fp32;
        split.descriptor.xm |= 1 << 42;
        let packets: Vec<_> = C220FixpReadGenerator::new(split, 9, 50, 256)
            .unwrap()
            .collect();
        assert_eq!(packets.len(), 20);
        assert!(
            packets
                .iter()
                .all(|p| p.operation.data_bytes == 128 && p.operation.output_bytes == 64)
        );
        assert_eq!(packets[9].operation.destination_address, 4096 + 9 * 64);
        assert_eq!(packets[10].operation.destination_address, 8192);
        assert_eq!(packets[10].operation.request.fragments.address, 2176);
        assert!(packets[9].operation.end_of_burst);
        assert!(!packets[9].operation.last_in_instruction);
        assert!(packets[19].operation.last_in_instruction);
        assert!(matches!(
            C220FixpReadGenerator::new(fp32, 8, 30, 1),
            Err(C220FixpReadGeneratorError::ZeroBandwidth)
        ));
    }
}
