use super::{C220FixpFp16Command, C220FixpLayoutError};
use crate::sim::c220::cube::C220CubeL0cAccess;
use crate::sim::c220::memory::{C220L0cFragmentRequest, C220L0cReadRequest};
use crate::sim::c220::mte::interface::C220MteL0cReadOperation;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FixpReadUop {
    pub operation: C220MteL0cReadOperation,
    pub begins_unit: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum C220FixpReadGeneratorError {
    #[error(transparent)]
    Layout(#[from] C220FixpLayoutError),
    #[error("FIX read bandwidth must be nonzero")]
    ZeroBandwidth,
    #[error("NZ-to-ND requires its own FIX read generator")]
    NzToNd,
}

/// Ordinary mode-1 read packets are column-major, unlike functional slices.
/// Bandwidth is an explicit model input, not an assumed device constant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220FixpReadGenerator {
    command: C220FixpFp16Command,
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
        command: C220FixpFp16Command,
        instruction_id: u64,
        first_id: u32,
        bandwidth: u32,
    ) -> Result<Self, C220FixpReadGeneratorError> {
        command.layout()?;
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
        let column_bytes = u32::from(d.rows()) * 64;
        let address = self
            .command
            .source_address
            .wrapping_add(u64::from(
                self.column_block
                    .wrapping_mul(u32::from(d.source_stride()))
                    .wrapping_mul(64),
            ))
            .wrapping_add(u64::from(self.source_offset));
        let boundary = self.bandwidth - (address as u32 % self.bandwidth);
        let data_bytes = (column_bytes - self.source_offset).min(boundary);
        let output_bytes = data_bytes.wrapping_mul(32) / 64;
        let begins_unit = address.is_multiple_of(1024) || self.source_offset == 0;
        let end_of_burst = self.source_offset + data_bytes == column_bytes;
        let operation = C220MteL0cReadOperation {
            instruction_id: self.instruction_id,
            uop_id: self.next_id,
            conversion_mode: 1,
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
            destination_address: self
                .command
                .destination_address
                .wrapping_add(
                    u64::from(self.column_block)
                        * u64::from(d.destination_stride().wrapping_mul(32)),
                )
                .wrapping_add(u64::from(self.destination_offset)),
            output_bytes,
            last_in_uop: true,
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
        Some(C220FixpReadUop {
            operation,
            begins_unit,
        })
    }
}

impl std::iter::FusedIterator for C220FixpReadGenerator {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::c220::mte::fixp::C220FixpDescriptor;

    #[test]
    fn unaligned_reads_split_columns_and_mark_only_final_instruction_packet() {
        let command = C220FixpFp16Command {
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
        assert_eq!(packets.iter().filter(|p| p.begins_unit).count(), 4);
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
    }
}
